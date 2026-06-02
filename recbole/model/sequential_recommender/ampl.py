# -*- coding: utf-8 -*-
"""
AMPL — RecBole plug-in version for AMPL-style multi-task multi-behavior
sequential recommendation.

Paper:
    Multi-Task Multi-Behavior Sequential Recommendation, WWW 2026.

This implementation is designed for the AMPL processed JD/UB datasets converted
by convert_ampl_to_recbole_seq.py. It follows the main AMPL ideas:
    1) IPL: integrated preference learning with fused-behavior decomposition.
    2) SIPL: sequence-independent behavior preference aggregation.
    3) Shared-specific gating between IPL and SIPL.
    4) BIPL: behavior-independent pure-item sequential modeling.

It is intentionally compact and RecBole-friendly:
    - full_sort_predict() supports full-ranking evaluation.
    - calculate_loss() supports BPR / BCE / CE.
    - no config.get() is used, for old RecBole Config compatibility.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender


def _config_get(config, key, default=None):
    """Config-safe getter: old RecBole Config may not support .get()."""
    try:
        return config[key]
    except Exception:
        return default


def _has_inter_field(interaction, field: str) -> bool:
    try:
        _ = interaction[field]
        return True
    except Exception:
        return False


class _CausalSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=1, inner_size=None, dropout=0.5,
                 attn_dropout=0.5, layer_norm_eps=1e-12, hidden_act="relu"):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        inner_size = inner_size or d_model

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        act = nn.GELU() if str(hidden_act).lower() == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, inner_size),
            act,
            nn.Dropout(dropout),
            nn.Linear(inner_size, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, valid_mask):
        # x: (B,L,D), valid_mask: (B,L) bool
        B, L, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,L,L)

        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~causal.view(1, 1, L, L), -1e4)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask.view(B, 1, 1, L), -1e4)

        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.drop(self.proj(out))
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)

        x = x + self.ffn(self.norm2(x))
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)
        return x


class _TargetBehaviorAttention(nn.Module):
    """Target-behavior-aware causal attention used in AMPL IPL."""

    def __init__(self, d_model, n_heads=1, inner_size=None, dropout=0.5,
                 attn_dropout=0.5, layer_norm_eps=1e-12, hidden_act="relu"):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        inner_size = inner_size or d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        act = nn.GELU() if str(hidden_act).lower() == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, inner_size),
            act,
            nn.Dropout(dropout),
            nn.Linear(inner_size, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, key_value_seq, query_seq, valid_mask):
        # key_value_seq/query_seq: (B,L,D)
        B, L, D = key_value_seq.shape

        q = self.q_proj(query_seq)
        k = self.k_proj(key_value_seq)
        v = self.v_proj(key_value_seq)

        def split_heads(t):
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=key_value_seq.device))
        scores = scores.masked_fill(~causal.view(1, 1, L, L), -1e4)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask.view(B, 1, 1, L), -1e4)

        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)

        out = self.norm(out + key_value_seq)
        out = out + self.ffn(out)
        if valid_mask is not None:
            out = out * valid_mask.unsqueeze(-1).to(out.dtype)
        return out


class _SequenceIndependentAggregator(nn.Module):
    """Target-behavior-aware feature aggregator for SIPL."""

    def __init__(self, d_model, dropout=0.5):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.scale = d_model ** -0.5

    def forward(self, target_query, behavior_summaries):
        # target_query: (B,L,D), behavior_summaries: (B,K,D)
        q = self.q(target_query).unsqueeze(2)       # (B,L,1,D)
        k = self.k(behavior_summaries).unsqueeze(1) # (B,1,K,D)
        v = self.v(behavior_summaries).unsqueeze(1) # (B,1,K,D)
        scores = (q * k).sum(-1) * self.scale       # (B,L,K)
        weight = torch.softmax(scores.float(), dim=-1).to(target_query.dtype)
        weight = self.drop(weight)
        out = (weight.unsqueeze(-1) * v).sum(dim=2) # (B,L,D)
        return out


class AMPL(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.hidden_size = int(_config_get(config, "hidden_size", 50))
        self.n_layers = int(_config_get(config, "num_layers", 1))
        self.n_heads = int(_config_get(config, "num_heads", 1))
        self.inner_size = int(_config_get(config, "inner_size", self.hidden_size))
        self.dropout_prob = float(_config_get(config, "dropout_prob",
                                  _config_get(config, "hidden_dropout_prob", 0.5)))
        self.attn_dropout_prob = float(_config_get(config, "attn_dropout_prob", self.dropout_prob))
        self.hidden_act = _config_get(config, "hidden_act", "relu")
        self.layer_norm_eps = float(_config_get(config, "layer_norm_eps", 1e-12))
        self.loss_type = str(_config_get(config, "loss_type", "BPR")).upper()
        self.reg_weight = float(_config_get(config, "reg_weight", 0.0))

        self.type_seq_field = _config_get(config, "ITEM_TYPE_SEQ_FIELD", "item_type_list")
        self.target_type_field = _config_get(config, "ITEM_TYPE_FIELD", "item_type")
        self.target_behavior_token = int(_config_get(config, "target_behavior_token", 3))

        # Behavior token vocabulary. In RecBole atomic token fields, real behavior ids
        # are usually re-indexed to 1..K with 0 reserved for padding.
        cfg_beh = int(_config_get(config, "num_behaviors", 4))
        n_type_tokens = cfg_beh + 1
        try:
            n_type_tokens = max(n_type_tokens, int(dataset.num(self.target_type_field)))
        except Exception:
            pass
        try:
            n_type_tokens = max(n_type_tokens, int(dataset.num(self.type_seq_field)))
        except Exception:
            pass
        self.n_type_tokens = n_type_tokens
        self.behavior_ids = list(range(1, self.n_type_tokens))  # exclude padding 0
        self.num_tasks = len(self.behavior_ids)

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.beh_embedding = nn.Embedding(self.n_type_tokens, self.hidden_size, padding_idx=0)
        self.item_pos_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)
        self.beh_pos_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)

        self.emb_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.emb_dropout = nn.Dropout(self.dropout_prob)

        # IPL: one target-behavior-aware attention network per behavior.
        self.ipl_attn = nn.ModuleList([
            _TargetBehaviorAttention(
                self.hidden_size, self.n_heads, self.inner_size, self.dropout_prob,
                self.attn_dropout_prob, self.layer_norm_eps, self.hidden_act
            )
            for _ in self.behavior_ids
        ])

        # SIPL
        self.sipl_agg = _SequenceIndependentAggregator(self.hidden_size, self.dropout_prob)

        # Shared-specific gating: operate on [Z_it || Z_si].
        self.shared_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.specific_gates = nn.ModuleList([
            nn.Linear(self.hidden_size * 2, self.hidden_size)
            for _ in self.behavior_ids
        ])

        # BIPL: pure-item SASRec-style causal sequence model.
        self.bipl_blocks = nn.ModuleList([
            _CausalSelfAttentionBlock(
                self.hidden_size, self.n_heads, self.inner_size, self.dropout_prob,
                self.attn_dropout_prob, self.layer_norm_eps, self.hidden_act
            )
            for _ in range(max(1, self.n_layers))
        ])

        self.final_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _get_type_seq(self, interaction, item_seq):
        if _has_inter_field(interaction, self.type_seq_field):
            type_seq = interaction[self.type_seq_field].long()
            return type_seq.clamp(0, self.n_type_tokens - 1)
        return torch.zeros_like(item_seq)

    def _get_target_type(self, interaction, item_seq):
        if _has_inter_field(interaction, self.target_type_field):
            tgt = interaction[self.target_type_field].long()
            return tgt.clamp(0, self.n_type_tokens - 1)
        return torch.full((item_seq.size(0),), self.target_behavior_token,
                          dtype=torch.long, device=item_seq.device).clamp(0, self.n_type_tokens - 1)

    def _avg_other_behavior_emb(self, beh_id: int):
        ids = [b for b in self.behavior_ids if b != beh_id]
        if not ids:
            ids = [beh_id]
        idx = torch.tensor(ids, dtype=torch.long, device=self.beh_embedding.weight.device)
        return self.beh_embedding(idx).mean(dim=0)  # (D,)

    def forward(self, item_seq, item_seq_len, type_seq=None, target_type=None):
        B, L = item_seq.shape
        device = item_seq.device
        valid_mask = item_seq.ne(0)

        if type_seq is None:
            type_seq = torch.zeros_like(item_seq)
        type_seq = type_seq.long().clamp(0, self.n_type_tokens - 1)
        if target_type is None:
            target_type = torch.full((B,), self.target_behavior_token,
                                     dtype=torch.long, device=device)
        target_type = target_type.long().clamp(0, self.n_type_tokens - 1)

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        item_emb = self.item_embedding(item_seq)
        type_emb = self.beh_embedding(type_seq)

        # ── IPL: fused-behavior decomposition + target-behavior-aware attention.
        target_emb = self.beh_embedding(target_type)  # (B,D)
        z_it = item_emb.new_zeros(B, L, self.hidden_size)
        for mod_idx, beh_id in enumerate(self.behavior_ids):
            avg_other = self._avg_other_behavior_emb(beh_id).view(1, 1, self.hidden_size)

            own_mask = type_seq.eq(beh_id).unsqueeze(-1)
            fused_type_seq = torch.where(own_mask, type_emb, avg_other.expand(B, L, -1))

            # Fused target behavior query for this decomposed behavior view.
            target_is_own = target_type.eq(beh_id).float().view(B, 1, 1)
            avg_target = avg_other.expand(B, L, -1)
            own_target = target_emb.view(B, 1, self.hidden_size).expand(B, L, -1)
            fused_target_seq = target_is_own * own_target + (1.0 - target_is_own) * avg_target

            x_beh = item_emb + fused_type_seq + self.beh_pos_embedding(pos)
            z_it = z_it + self.ipl_attn[mod_idx](x_beh, fused_target_seq, valid_mask)

        # ── SIPL: behavior-wise sequence-independent aggregation.
        summaries = []
        for beh_id in self.behavior_ids:
            m = (type_seq.eq(beh_id) & valid_mask).float().unsqueeze(-1)
            denom = m.sum(dim=1).clamp(min=1.0)
            summaries.append((item_emb * m).sum(dim=1) / denom)
        behavior_summaries = torch.stack(summaries, dim=1)  # (B,K,D)
        target_query = target_emb.view(B, 1, self.hidden_size).expand(B, L, -1)
        z_si = self.sipl_agg(target_query, behavior_summaries)

        # ── Shared-specific gate.
        cat = torch.cat([z_it, z_si], dim=-1)
        shared = torch.sigmoid(self.shared_gate(cat))  # (B,L,D)
        spec_all = torch.stack(
            [torch.sigmoid(g(cat)) for g in self.specific_gates], dim=2
        )  # (B,L,K,D)

        # map behavior token 1..K to index 0..K-1
        task_idx = (target_type - 1).clamp(0, self.num_tasks - 1)
        gather_idx = task_idx.view(B, 1, 1, 1).expand(B, L, 1, self.hidden_size)
        specific = spec_all.gather(dim=2, index=gather_idx).squeeze(2)
        gate = torch.sigmoid(shared + specific)
        z_fused = gate * z_it + (1.0 - gate) * z_si

        # ── BIPL: pure-item sequential dependencies.
        z_bi = item_emb + self.item_pos_embedding(pos)
        z_bi = self.emb_norm(self.emb_dropout(z_bi))
        z_bi = z_bi * valid_mask.unsqueeze(-1).to(z_bi.dtype)
        for blk in self.bipl_blocks:
            z_bi = blk(z_bi, valid_mask)

        out = self.final_norm(z_fused + z_bi)
        out = out * valid_mask.unsqueeze(-1).to(out.dtype)
        return out

    def _user_repr_from_interaction(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        type_seq = self._get_type_seq(interaction, item_seq)
        target_type = self._get_target_type(interaction, item_seq)
        seq_out = self.forward(item_seq, item_seq_len, type_seq=type_seq, target_type=target_type)
        last_idx = item_seq_len.long().clamp(min=1, max=item_seq.size(1)) - 1
        row = torch.arange(item_seq.size(0), device=item_seq.device)
        return seq_out[row, last_idx]

    def _sample_neg_items(self, pos_items):
        neg = torch.randint(1, self.n_items, pos_items.shape, device=pos_items.device)
        # Avoid exact positive collisions in a cheap way.
        neg = torch.where(neg.eq(pos_items), (neg % (self.n_items - 1)) + 1, neg)
        return neg

    def calculate_loss(self, interaction):
        user_repr = self._user_repr_from_interaction(interaction)
        pos_items = interaction[self.ITEM_ID].long()
        pos_emb = self.item_embedding(pos_items)

        if self.loss_type == "CE":
            logits = torch.matmul(user_repr, self.item_embedding.weight.transpose(0, 1))
            loss = F.cross_entropy(logits, pos_items)
        else:
            neg_items = self._sample_neg_items(pos_items)
            neg_emb = self.item_embedding(neg_items)
            pos_score = (user_repr * pos_emb).sum(dim=-1)
            neg_score = (user_repr * neg_emb).sum(dim=-1)
            if self.loss_type == "BCE":
                scores = torch.cat([pos_score, neg_score], dim=0)
                labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)
                loss = F.binary_cross_entropy_with_logits(scores, labels)
            else:  # BPR
                loss = -F.logsigmoid(pos_score - neg_score).mean()

        if self.reg_weight > 0:
            reg = (self.item_embedding.weight.pow(2).mean() +
                   self.beh_embedding.weight.pow(2).mean())
            loss = loss + self.reg_weight * reg
        return loss

    def predict(self, interaction):
        user_repr = self._user_repr_from_interaction(interaction)
        item = interaction[self.ITEM_ID].long()
        return (user_repr * self.item_embedding(item)).sum(dim=-1)

    def full_sort_predict(self, interaction):
        user_repr = self._user_repr_from_interaction(interaction)
        return torch.matmul(user_repr, self.item_embedding.weight.transpose(0, 1))
