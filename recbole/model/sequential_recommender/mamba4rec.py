# -*- coding: utf-8 -*-
"""
Mamba4Rec-MB: AMPL/RecBole-compatible Mamba baseline for
multi-task multi-behavior sequential recommendation.

Adapted for:
- item_id_list + item_type_list benchmark files
- target behavior item_type
- masked prediction anchor
- full-ranking evaluation
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None


def _config_get(config, key, default=None):
    try:
        v = config[key]
    except Exception:
        return default
    return default if v is None else v


def _as_int(v, default):
    if v is None:
        return int(default)
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) else default
    return int(v)


def _as_float(v, default):
    if v is None:
        return float(default)
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) else default
    return float(v)


def _as_bool(v, default=False):
    if v is None:
        return bool(default)
    if isinstance(v, str):
        return v.lower() in {"true", "1", "yes", "y"}
    return bool(v)


def _as_str(v, default):
    if v is None:
        return str(default)
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) else default
    return str(v)


def _has_inter_field(interaction, field: str) -> bool:
    try:
        _ = interaction[field]
        return True
    except Exception:
        return False


class FeedForward(nn.Module):
    def __init__(self, d_model: int, inner_size: int, dropout: float = 0.1, eps: float = 1e-12):
        super().__init__()
        self.fc1 = nn.Linear(d_model, inner_size)
        self.fc2 = nn.Linear(inner_size, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        residual = x
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return self.norm(x + residual)


class MambaLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        inner_size: Optional[int] = None,
        layer_norm_eps: float = 1e-12,
        residual: bool = True,
        fallback_to_gru: bool = False,
    ):
        super().__init__()
        self.residual = residual
        if Mamba is None:
            if not fallback_to_gru:
                raise ImportError(
                    "mamba_ssm is not installed. Install it or set fallback_to_gru: True for debugging only."
                )
            self.mamba = nn.GRU(d_model, d_model, batch_first=True)
            self.is_gru = True
        else:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.is_gru = False

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = FeedForward(d_model, inner_size or d_model * 4, dropout, layer_norm_eps)

    def forward(self, x, valid_mask=None):
        if self.is_gru:
            y, _ = self.mamba(x)
        else:
            y = self.mamba(x)
        x = self.norm1(self.dropout(y) + x) if self.residual else self.norm1(self.dropout(y))
        x = self.ffn(x)
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)
        return x


class Mamba4Rec(SequentialRecommender):
    """Behavior-aware Mamba baseline for AMPL-style multi-behavior tasks."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.hidden_size = _as_int(_config_get(config, "hidden_size", 64), 64)
        self.loss_type = _as_str(_config_get(config, "loss_type", "CE"), "CE").upper()
        self.num_layers = _as_int(_config_get(config, "num_layers", 2), 2)
        self.dropout_prob = _as_float(_config_get(config, "dropout_prob", 0.3), 0.3)
        self.hidden_dropout_prob = _as_float(_config_get(config, "hidden_dropout_prob", self.dropout_prob), self.dropout_prob)
        self.layer_norm_eps = _as_float(_config_get(config, "layer_norm_eps", 1e-12), 1e-12)
        self.initializer_range = _as_float(_config_get(config, "initializer_range", 0.02), 0.02)

        self.d_state = _as_int(_config_get(config, "d_state", 16), 16)
        self.d_conv = _as_int(_config_get(config, "d_conv", 4), 4)
        self.expand = _as_int(_config_get(config, "expand", 2), 2)
        self.inner_size = _as_int(_config_get(config, "inner_size", self.hidden_size * 4), self.hidden_size * 4)
        self.fallback_to_gru = _as_bool(_config_get(config, "fallback_to_gru", False), False)

        self.max_len = _as_int(_config_get(config, "MAX_ITEM_LIST_LENGTH", 50), 50)
        self.model_seq_len = self.max_len + 1
        self.num_neg = _as_int(_config_get(config, "num_neg", 1), 1)

        self.use_behavior_embedding = _as_bool(_config_get(config, "use_behavior_embedding", True), True)
        self.use_position_embedding = _as_bool(_config_get(config, "use_position_embedding", True), True)
        self.use_interaction_target_behavior = _as_bool(_config_get(config, "use_interaction_target_behavior", True), True)
        self.mask_behavior_as_target = _as_bool(_config_get(config, "mask_behavior_as_target", True), True)

        self.type_seq_field = _as_str(_config_get(config, "ITEM_TYPE_SEQ_FIELD", "item_type_list"), "item_type_list")
        self.target_type_field = _as_str(_config_get(config, "ITEM_TYPE_FIELD", "item_type"), "item_type")
        self.type_vocab_size = self._infer_type_vocab_size(config, dataset)
        self.default_target_behavior = self._infer_default_target_behavior(config)

        self.mask_token = self.n_items
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.type_embedding = nn.Embedding(self.type_vocab_size, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.model_seq_len + 1, self.hidden_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                inner_size=self.inner_size,
                layer_norm_eps=self.layer_norm_eps,
                residual=True,
                fallback_to_gru=self.fallback_to_gru,
            )
            for _ in range(self.num_layers)
        ])

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "BCE":
            self.loss_fct = nn.BCEWithLogitsLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("loss_type must be one of ['BPR', 'BCE', 'CE'].")

        self.apply(self._init_weights)
        with torch.no_grad():
            nn.init.normal_(self.item_embedding.weight[self.mask_token], mean=0.0, std=self.initializer_range)

    def _infer_type_vocab_size(self, config, dataset):
        candidates = [_as_int(_config_get(config, "num_behaviors", 4), 4) + 1]
        for field in ("item_type_list", "item_type"):
            try:
                mapping = dataset.field2token_id[field]
                if mapping:
                    vals = []
                    for v in mapping.values():
                        try:
                            vals.append(int(v))
                        except Exception:
                            pass
                    candidates.append(max(vals) + 1 if vals else len(mapping) + 1)
            except Exception:
                pass
        return max(candidates)

    def _infer_default_target_behavior(self, config):
        explicit = _config_get(config, "target_behavior_token", None)
        if explicit is not None:
            return int(explicit)
        return min(_as_int(_config_get(config, "num_behaviors", 4), 4), self.type_vocab_size - 1)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _get_type_seq(self, interaction, item_seq):
        if _has_inter_field(interaction, self.type_seq_field):
            return interaction[self.type_seq_field].long().clamp(0, self.type_vocab_size - 1)
        if _has_inter_field(interaction, "item_type_list"):
            return interaction["item_type_list"].long().clamp(0, self.type_vocab_size - 1)
        return torch.zeros_like(item_seq)

    def _get_target_behavior(self, interaction, batch_size, device):
        if self.use_interaction_target_behavior and _has_inter_field(interaction, self.target_type_field):
            return interaction[self.target_type_field].long().clamp(0, self.type_vocab_size - 1)
        if self.use_interaction_target_behavior and _has_inter_field(interaction, "item_type"):
            return interaction["item_type"].long().clamp(0, self.type_vocab_size - 1)
        return torch.full((batch_size,), self.default_target_behavior, dtype=torch.long, device=device)

    def _append_anchor(self, item_seq, type_seq, target_behavior, target_item=None, item_seq_len=None):
        B, T = item_seq.shape
        device = item_seq.device
        if item_seq_len is None:
            lengths = torch.count_nonzero(item_seq, dim=1).clamp(max=T)
        else:
            lengths = item_seq_len.long().clamp(min=0, max=T)

        zero = torch.zeros(B, 1, dtype=item_seq.dtype, device=device)
        item_ext = torch.cat([item_seq, zero], dim=1)
        type_ext = torch.cat([type_seq, zero], dim=1)

        row = torch.arange(B, device=device)
        item_ext[row, lengths] = self.mask_token
        type_ext[row, lengths] = target_behavior if self.mask_behavior_as_target else 0

        targets = None if target_item is None else target_item.long().clamp(0, self.n_items - 1)
        return (
            item_ext[:, : self.model_seq_len],
            type_ext[:, : self.model_seq_len].clamp(0, self.type_vocab_size - 1),
            lengths.clamp(max=self.model_seq_len - 1),
            targets,
        )

    def forward(self, item_seq, type_seq, anchor_pos):
        B, T = item_seq.shape
        item_seq = item_seq[:, : self.model_seq_len]
        type_seq = type_seq[:, : self.model_seq_len]
        T = item_seq.size(1)

        valid_mask = (item_seq != 0).float()
        pos_ids = torch.arange(T, device=item_seq.device).unsqueeze(0).expand(B, T)

        x = self.item_embedding(item_seq.clamp(0, self.n_items))
        if self.use_behavior_embedding:
            x = x + self.type_embedding(type_seq.clamp(0, self.type_vocab_size - 1))
        if self.use_position_embedding:
            x = x + self.position_embedding(pos_ids)

        x = self.LayerNorm(self.dropout(x))
        x = x * valid_mask.unsqueeze(-1)

        for layer in self.mamba_layers:
            x = layer(x, valid_mask)

        return self.gather_indexes(x, anchor_pos)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        type_seq = self._get_type_seq(interaction, item_seq)
        target_item = interaction[self.POS_ITEM_ID] if _has_inter_field(interaction, self.POS_ITEM_ID) else interaction[self.ITEM_ID]
        target_behavior = self._get_target_behavior(interaction, item_seq.size(0), item_seq.device)

        input_items, input_types, anchor_pos, targets = self._append_anchor(
            item_seq, type_seq, target_behavior, target_item=target_item
        )
        seq_output = self.forward(input_items, input_types, anchor_pos)
        item_table = self.item_embedding.weight[: self.n_items]

        if self.loss_type == "BPR":
            if _has_inter_field(interaction, self.NEG_ITEM_ID):
                neg_items = interaction[self.NEG_ITEM_ID].long().clamp(1, self.n_items - 1)
            else:
                neg_items = torch.randint(1, self.n_items, targets.shape, device=targets.device)
            pos_score = torch.sum(seq_output * item_table[targets], dim=-1)
            neg_score = torch.sum(seq_output * item_table[neg_items], dim=-1)
            return self.loss_fct(pos_score, neg_score)

        if self.loss_type == "BCE":
            pos_score = torch.sum(seq_output * item_table[targets], dim=-1)
            neg_items = torch.randint(1, self.n_items, (targets.size(0), self.num_neg), device=targets.device)
            neg_items = torch.where(neg_items.eq(targets.unsqueeze(1)), (neg_items % (self.n_items - 1)) + 1, neg_items)
            neg_score = torch.einsum("bd,bkd->bk", seq_output, item_table[neg_items])
            scores = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
            labels = torch.zeros_like(scores)
            labels[:, 0] = 1.0
            return self.loss_fct(scores, labels)

        logits = torch.matmul(seq_output, item_table.transpose(0, 1))
        return self.loss_fct(logits, targets)

    def predict(self, interaction):
        scores = self.full_sort_predict(interaction)
        return scores.gather(1, interaction[self.ITEM_ID].view(-1, 1)).squeeze(1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        type_seq = self._get_type_seq(interaction, item_seq)
        item_seq_len = interaction[self.ITEM_SEQ_LEN] if _has_inter_field(interaction, self.ITEM_SEQ_LEN) else torch.count_nonzero(item_seq, dim=1)
        target_behavior = self._get_target_behavior(interaction, item_seq.size(0), item_seq.device)

        input_items, input_types, anchor_pos, _ = self._append_anchor(
            item_seq, type_seq, target_behavior, target_item=None, item_seq_len=item_seq_len
        )
        seq_output = self.forward(input_items, input_types, anchor_pos)

        item_table = self.item_embedding.weight[: self.n_items]
        scores = torch.matmul(seq_output, item_table.transpose(0, 1))
        scores[:, 0] = -1e9
        return scores
