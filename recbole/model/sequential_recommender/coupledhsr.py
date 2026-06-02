# -*- coding: utf-8 -*-
"""
CoupledHSRA6 — Masked Coupled Hamiltonian Transformer for multi-behavior
sequential recommendation.

A6 experiment:
    A3 = lightweight masked Transformer encoder
    A6 = A3 + frequency-adaptive causal Hamiltonian coupling residual

This file is designed as an easy drop-in RecBole model for the MBHT-style
retail_beh / tmall_beh / ijcai_beh datasets, but with a clean full-ranking
protocol through full_sort_predict().

Main differences from the previous CoupledHSR:
  1. Uses MBHT-style masked item reconstruction objective.
  2. Uses MBHT-style mask-token test-time readout.
  3. Keeps full-ranking evaluation through full_sort_predict().
  4. Replaces MBHT's expensive dynamic hypergraph branch with a small
     frequency-adaptive causal Hamiltonian residual.
  5. Uses a first-order causal transfer approximation by default instead of
     per-frequency complex matrix solves, improving memory and latency.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender


def _config_get(config, key, default=None):
    return config[key] if key in config else default


class SafeSelfAttentionBlock(nn.Module):
    """Small BERT-style bidirectional self-attention block.

    We avoid torch.nn.MultiheadAttention so the code remains robust across
    CUDA / PyTorch versions and sequence lengths.
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        inner_size: int,
        dropout: float,
        attn_dropout: float,
        layer_norm_eps: float = 1e-12,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must divide n_heads={n_heads}")
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        act = nn.GELU() if hidden_act.lower() == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            act,
            nn.Dropout(dropout),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """x: [B,T,D], valid_mask: [B,T] with 1 for valid tokens."""
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B,T,H,d]
        q = q.transpose(1, 2)        # [B,H,T,d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T,T]
        key_mask = valid_mask[:, None, None, :].to(dtype=torch.bool)
        attn = attn.masked_fill(~key_mask, -1e4)
        attn = F.softmax(attn.float(), dim=-1).to(dtype=x.dtype)
        attn = self.attn_dropout(attn)

        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_dropout(self.out_proj(ctx))
        x = x * valid_mask.unsqueeze(-1)

        x = x + self.ffn(self.norm2(x))
        x = x * valid_mask.unsqueeze(-1)
        return x


class FrequencyAdaptiveCausalHamiltonian(nn.Module):
    """Frequency-adaptive causal behavior coupling residual.

    Physical view:
        Each behavior is a damped oscillator mode with mass m, damping c and
        stiffness k. Historical item embeddings excite behavior-specific forces.
        Causal cross-behavior coupling transfers source modes into downstream
        target modes in the frequency domain.

    Efficient approximation:
        The default solver is a first-order transfer approximation:
            q_tgt(ω) = base_tgt(ω) + Σ_src C[tgt,src,ω] * base_src(ω)
        This avoids constructing and solving [B_behavior x B_behavior] complex
        systems for every batch/frequency/channel.
    """

    def __init__(
        self,
        hidden_size: int,
        max_len: int,
        num_behavior_tokens: int,
        kernel_size: int = 3,
        n_coup_bands: int = 8,
        dropout: float = 0.1,
        coupling_mode: str = "causal",
        funnel_order: Optional[list] = None,
        coupling_scale: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.B = num_behavior_tokens
        self.coupling_mode = coupling_mode
        self.n_coup_bands = max(2, int(n_coup_bands))
        self.coupling_scale = float(coupling_scale)

        n_freq = max_len // 2 + 1
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.in_proj = nn.Linear(hidden_size, hidden_size * 3)

        # Per-behavior, per-channel physical parameters.
        self.m_raw = nn.Parameter(torch.zeros(self.B, hidden_size))
        self.c_raw = nn.Parameter(torch.zeros(self.B, hidden_size))
        self.k_raw = nn.Parameter(torch.zeros(self.B, hidden_size))

        # Complex Green numerator. Small init is more stable for residual use.
        self.psi_re = nn.Parameter(torch.full((self.B, hidden_size, n_freq), 0.05))
        self.psi_im = nn.Parameter(torch.zeros(self.B, hidden_size, n_freq))

        # Frequency-adaptive causal transfer: [target_behavior, source_behavior, channel, band].
        self.coupling = nn.Parameter(
            torch.zeros(self.B, self.B, hidden_size, self.n_coup_bands)
        )

        cmask = self._build_causal_mask(funnel_order)
        self.register_buffer("cmask", cmask)

        omega = 2.0 * math.pi * torch.arange(n_freq).float() / max_len
        self.register_buffer("omega", omega)

        # Local impulse response, same spirit as HSR but used as residual.
        pad = kernel_size - 1
        self.kernel_size = kernel_size
        self.impulse_conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=False
        )
        self.local_pad = pad

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _build_causal_mask(self, funnel_order: Optional[list]) -> torch.Tensor:
        cmask = torch.zeros(self.B, self.B)
        if self.coupling_mode == "none":
            return cmask
        if self.coupling_mode == "symmetric":
            cmask = torch.ones(self.B, self.B) - torch.eye(self.B)
            cmask[0, :] = 0.0
            cmask[:, 0] = 0.0
            return cmask

        # causal: source earlier in funnel can influence target later in funnel.
        if not funnel_order:
            # Generic order: 1 -> 2 -> ...; 0 is pad/mask and does not transfer.
            funnel_order = list(range(1, self.B))
        for tgt_pos, tgt in enumerate(funnel_order):
            for src in funnel_order[:tgt_pos]:
                if 0 <= int(tgt) < self.B and 0 <= int(src) < self.B:
                    cmask[int(tgt), int(src)] = 1.0
        return cmask

    def _interp_coupling(self, n_freq: int) -> torch.Tensor:
        # [B,B,D,bands] -> [B,B,D,n_freq]
        raw = torch.tanh(self.coupling) * self.cmask[:, :, None, None]
        full = F.interpolate(
            raw.reshape(self.B * self.B * self.hidden_size, 1, self.n_coup_bands),
            size=n_freq,
            mode="linear",
            align_corners=True,
        ).reshape(self.B, self.B, self.hidden_size, n_freq)
        return self.coupling_scale * full

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        behavior_ids: torch.Tensor,
    ) -> torch.Tensor:
        Bsz, T, D = x.shape
        h = self.norm(x)
        force, local_u, gate = self.in_proj(h).chunk(3, dim=-1)
        force = F.gelu(force)
        local_u = F.gelu(local_u)
        gate = torch.sigmoid(gate)

        behavior_ids = behavior_ids.clamp(0, self.B - 1)
        onehot = F.one_hot(behavior_ids.long(), num_classes=self.B).to(x.dtype)
        force_b = onehot.permute(0, 2, 1).unsqueeze(-1) * force.unsqueeze(1)
        # [batch, behavior, D, freq]
        F_hat = torch.fft.rfft(force_b.permute(0, 1, 3, 2), n=self.max_len, dim=-1)
        n_freq = F_hat.shape[-1]

        w = self.omega[:n_freq].view(1, 1, 1, n_freq)
        m = (F.softplus(self.m_raw) + 1e-2).view(1, self.B, D, 1)
        c = (F.softplus(self.c_raw) + 1e-2).view(1, self.B, D, 1)
        k = (F.softplus(self.k_raw) + 1e-2).view(1, self.B, D, 1)

        denom = torch.complex(k - m * w.pow(2), c * w)
        psi = torch.complex(
            self.psi_re[:, :, :n_freq],
            self.psi_im[:, :, :n_freq],
        ).unsqueeze(0)  # [1, behavior, D, freq]

        base = psi * F_hat / (denom + 1e-6)

        if self.coupling_mode != "none":
            # ``base`` is complex after rFFT/Green filtering. The coupling kernel is
            # a real-valued behavior-transfer amplitude, but it must be promoted to
            # the same complex dtype before multiplying the complex spectrum.
            C = self._interp_coupling(n_freq).to(device=base.device, dtype=base.dtype)
            # C[tgt,src,d,f] * base[batch,src,d,f] -> transfer[batch,tgt,d,f]
            transfer = torch.einsum("tsdf,bsdf->btdf", C, base)
            q_hat = base + transfer
        else:
            q_hat = base

        # Back to [batch, behavior, time, D], then select each token's behavior.
        q_time = torch.fft.irfft(q_hat, n=self.max_len, dim=-1)[..., :T]
        q_time = q_time.permute(0, 1, 3, 2).contiguous()
        q_seq = torch.einsum("btv,bvtd->btd", onehot, q_time)

        local = F.pad(local_u.transpose(1, 2), (self.local_pad, 0))
        local = self.impulse_conv(local).transpose(1, 2)[..., :D]

        delta = self.out_proj((q_seq + local) * gate)
        delta = self.dropout(torch.nan_to_num(delta, nan=0.0, posinf=10.0, neginf=-10.0))
        return delta * valid_mask.unsqueeze(-1)


class CoupledHSR(SequentialRecommender):
    """A6: masked Transformer + frequency-adaptive causal HNN residual."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.hidden_size = int(config["hidden_size"])
        self.n_layers = int(_config_get(config, "n_layers", _config_get(config, "num_layers", 1)))
        self.n_heads = int(_config_get(config, "n_heads", 2))
        self.inner_size = int(_config_get(config, "inner_size", self.hidden_size * 4))
        self.dropout_prob = float(_config_get(config, "hidden_dropout_prob", _config_get(config, "dropout_prob", 0.1)))
        self.attn_dropout_prob = float(_config_get(config, "attn_dropout_prob", self.dropout_prob))
        self.layer_norm_eps = float(_config_get(config, "layer_norm_eps", 1e-12))
        self.hidden_act = _config_get(config, "hidden_act", "gelu")

        self.mask_ratio = float(_config_get(config, "mask_ratio", 0.2))
        self.kernel_size = int(_config_get(config, "kernel_size", 3))
        self.n_coup_bands = int(_config_get(config, "n_coup_bands", 8))
        self.coupling_mode = _config_get(config, "coupling_mode", "causal")
        self.coupling_scale = float(_config_get(config, "coupling_scale", 0.1))
        self.hnn_residual_init = float(_config_get(config, "hnn_residual_init", -5.0))
        self.use_hnn = bool(_config_get(config, "use_hnn", True))
        self.use_transformer = bool(_config_get(config, "use_transformer", True))
        self.mask_behavior_as_target = bool(_config_get(config, "mask_behavior_as_target", False))
        self.target_behavior_token = _config_get(config, "target_behavior_token", None)

        self.max_len = int(config["MAX_ITEM_LIST_LENGTH"])
        # We append one mask-token position during train/test.
        self.model_seq_len = self.max_len + 1

        self.initializer_range = float(_config_get(config, "initializer_range", 0.02))
        self.dataset_name = config["dataset"]

        # MBHT-style target behavior: usually the buy behavior token.
        self.buy_type = self._infer_buy_type(config, dataset)
        self.target_behavior_id = int(self.target_behavior_token) if self.target_behavior_token is not None else int(self.buy_type)

        self.type_vocab_size = self._infer_type_vocab_size(config, dataset)
        # item mask token is n_items; valid real items are [0, n_items-1].
        self.mask_token = self.n_items
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.type_embedding = nn.Embedding(self.type_vocab_size, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.model_seq_len + 1, self.hidden_size)

        self.emb_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.emb_dropout = nn.Dropout(self.dropout_prob)

        self.transformer_blocks = nn.ModuleList()
        if self.use_transformer:
            self.transformer_blocks = nn.ModuleList([
                SafeSelfAttentionBlock(
                    hidden_size=self.hidden_size,
                    n_heads=self.n_heads,
                    inner_size=self.inner_size,
                    dropout=self.dropout_prob,
                    attn_dropout=self.attn_dropout_prob,
                    layer_norm_eps=self.layer_norm_eps,
                    hidden_act=self.hidden_act,
                )
                for _ in range(self.n_layers)
            ])

        funnel_order = _config_get(config, "funnel_order", None)
        self.hnn_blocks = nn.ModuleList()
        if self.use_hnn:
            hnn_layers = int(_config_get(config, "n_hnn_layers", self.n_layers))
            self.hnn_blocks = nn.ModuleList([
                FrequencyAdaptiveCausalHamiltonian(
                    hidden_size=self.hidden_size,
                    max_len=self.model_seq_len,
                    num_behavior_tokens=self.type_vocab_size,
                    kernel_size=self.kernel_size,
                    n_coup_bands=self.n_coup_bands,
                    dropout=self.dropout_prob,
                    coupling_mode=self.coupling_mode,
                    funnel_order=funnel_order,
                    coupling_scale=self.coupling_scale,
                    layer_norm_eps=self.layer_norm_eps,
                )
                for _ in range(hnn_layers)
            ])
            self.hnn_alpha = nn.Parameter(torch.full((hnn_layers,), self.hnn_residual_init))
        else:
            self.hnn_alpha = None

        self.final_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

        # Keep mask token initialized but not zero.
        with torch.no_grad():
            nn.init.normal_(self.item_embedding.weight[self.mask_token], mean=0.0, std=self.initializer_range)

    def _infer_type_vocab_size(self, config, dataset) -> int:
        candidates = []
        for field in ("item_type_list", "item_type"):
            if hasattr(dataset, "field2token_id") and field in dataset.field2token_id:
                mapping = dataset.field2token_id[field]
                if mapping:
                    try:
                        candidates.append(max(int(v) for v in mapping.values()) + 1)
                    except Exception:
                        candidates.append(len(mapping) + 1)
        candidates.append(int(_config_get(config, "num_behaviors", 4)) + 1)
        # Need room for token 0 pad/mask and all behavior IDs.
        return max(candidates)

    def _infer_buy_type(self, config, dataset) -> int:
        # MBHT uses dataset.field2token_id["item_type_list"]['0'] as buy_type.
        for field in ("item_type_list", "item_type"):
            try:
                mapping = dataset.field2token_id[field]
                if "0" in mapping:
                    return int(mapping["0"])
                if 0 in mapping:
                    return int(mapping[0])
            except Exception:
                pass
        return int(_config_get(config, "buy_type", _config_get(config, "target_behavior_token", 1)))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _append_target_and_mask(
        self,
        item_seq: torch.Tensor,
        type_seq: torch.Tensor,
        target_item: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized MBHT-style masked training construction.

        Returns:
            masked_items: [B,T+1]
            masked_types: [B,T+1]
            mask_positions: [B,T+1] bool
            targets: [num_masked_positions]
        """
        B, T = item_seq.shape
        device = item_seq.device
        lengths = torch.count_nonzero(item_seq, dim=1).clamp(max=T)

        zero_col = torch.zeros(B, 1, dtype=item_seq.dtype, device=device)
        item_ext = torch.cat([item_seq, zero_col], dim=1)
        type_ext = torch.cat([type_seq, zero_col], dim=1)

        row = torch.arange(B, device=device)
        item_ext[row, lengths] = target_item
        type_ext[row, lengths] = int(self.target_behavior_id)

        L = item_ext.shape[1]
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        valid = pos <= lengths.unsqueeze(1)
        hist_pos = pos < lengths.unsqueeze(1)
        target_pos = pos == lengths.unsqueeze(1)

        rand_mask = (torch.rand(B, L, device=device) < self.mask_ratio) & hist_pos & (item_ext != 0)
        mask_positions = rand_mask | target_pos

        targets = item_ext[mask_positions]
        masked_items = item_ext.clone()
        masked_types = type_ext.clone()
        masked_items[mask_positions] = self.mask_token
        if self.mask_behavior_as_target:
            masked_types[mask_positions] = int(self.target_behavior_id)
        else:
            # MBHT masks the behavior type at the masked item positions.
            masked_types[mask_positions] = 0

        masked_items = masked_items[:, : self.model_seq_len]
        masked_types = masked_types[:, : self.model_seq_len]
        mask_positions = mask_positions[:, : self.model_seq_len]

        return masked_items, masked_types, mask_positions, targets

    def _append_test_mask(
        self,
        item_seq: torch.Tensor,
        item_seq_len: torch.Tensor,
        type_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = item_seq.shape
        device = item_seq.device
        lengths = item_seq_len.long().clamp(min=0, max=T)
        zero_col = torch.zeros(B, 1, dtype=item_seq.dtype, device=device)
        item_ext = torch.cat([item_seq, zero_col], dim=1)
        type_ext = torch.cat([type_seq, zero_col], dim=1)
        row = torch.arange(B, device=device)
        item_ext[row, lengths] = self.mask_token
        if self.mask_behavior_as_target:
            type_ext[row, lengths] = int(self.target_behavior_id)
        else:
            type_ext[row, lengths] = 0
        return item_ext[:, : self.model_seq_len], type_ext[:, : self.model_seq_len], lengths

    def forward(self, item_seq: torch.Tensor, type_seq: torch.Tensor) -> torch.Tensor:
        B, T = item_seq.shape
        T = min(T, self.model_seq_len)
        item_seq = item_seq[:, :T]
        type_seq = type_seq[:, :T].clamp(0, self.type_vocab_size - 1)

        positions = torch.arange(T, device=item_seq.device).unsqueeze(0).expand(B, T)
        valid_mask = (item_seq != 0).float()

        x = (
            self.item_embedding(item_seq.clamp(0, self.n_items))
            + self.type_embedding(type_seq)
            + self.position_embedding(positions)
        )
        x = self.emb_norm(self.emb_dropout(x))
        x = x * valid_mask.unsqueeze(-1)

        n_steps = max(len(self.transformer_blocks), len(self.hnn_blocks), 1)
        for i in range(n_steps):
            if self.use_transformer and i < len(self.transformer_blocks):
                x = self.transformer_blocks[i](x, valid_mask)
            if self.use_hnn and i < len(self.hnn_blocks):
                delta = self.hnn_blocks[i](x, valid_mask, type_seq)
                alpha = torch.sigmoid(self.hnn_alpha[i])
                x = x + alpha * delta
                x = x * valid_mask.unsqueeze(-1)

        return self.final_norm(x)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        type_seq = interaction["item_type_list"]
        target_item = interaction[self.POS_ITEM_ID]

        masked_items, masked_types, mask_positions, targets = self._append_target_and_mask(
            item_seq, type_seq, target_item
        )
        seq_output = self.forward(masked_items, masked_types)
        masked_output = seq_output[mask_positions]

        # score only real item IDs; mask token is not a candidate label.
        item_table = self.item_embedding.weight[: self.n_items]
        logits = torch.matmul(masked_output, item_table.transpose(0, 1))

        # CE includes label 0 if it appears; targets should be nonzero for masked positions.
        targets = targets.clamp(0, self.n_items - 1)
        loss = self.loss_fct(logits, targets)

        # A tiny regularizer keeps the Hamiltonian residual from becoming the whole model too early.
        if self.use_hnn and self.hnn_alpha is not None:
            loss = loss + float(getattr(self, "alpha_reg", 0.0)) * torch.sigmoid(self.hnn_alpha).mean()
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        type_seq = interaction["item_type_list"]
        if self.ITEM_SEQ_LEN in interaction:
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
        else:
            item_seq_len = torch.count_nonzero(item_seq, dim=1)

        masked_items, masked_types, mask_pos = self._append_test_mask(item_seq, item_seq_len, type_seq)
        seq_output = self.forward(masked_items, masked_types)
        row = torch.arange(seq_output.size(0), device=seq_output.device)
        q = seq_output[row, mask_pos.clamp(max=seq_output.size(1) - 1)]

        item_table = self.item_embedding.weight[: self.n_items]
        scores = torch.matmul(q, item_table.transpose(0, 1))
        scores[:, 0] = -1e9
        return scores

    def predict(self, interaction):
        # Used by point-wise evaluation if needed.
        scores = self.full_sort_predict(interaction)
        return scores.gather(1, interaction[self.ITEM_ID].view(-1, 1)).squeeze(1)


# Backward-compatible alias: copy this file over your existing coupledhsr.py
# and keep running `--model CoupledHSR`.
class CoupledHSR(CoupledHSR):
    pass
