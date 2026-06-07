# -*- coding: utf-8 -*-
"""
TacoRec — Target-conditioned Frequency-Adaptive Causal Coupling
for AMPL-style multi-task multi-behavior sequential recommendation.

Compared with v4/A6, v5 adds three practical changes:

1) Target-conditioned behavior memory
   A lightweight AMPL-inspired behavior summary module. It pools historical
   representations by behavior and lets the target behavior query the behavior
   memory. This stabilizes sparse target behaviors without AMPL's heavy
   multi-view decomposition.

2) Target-conditioned gated residual fusion
   The Hamiltonian/spectral residual is no longer injected with a global scalar
   only. Its gate is conditioned on the target behavior, so click/cart/favorite/
   purchase tasks can use different coupling strengths.

3) Residual warm-up and safer losses
   The HNN residual can be warmed up after the Transformer anchor becomes stable.
   CE/BCE/BPR are supported for quick protocol comparison.

Drop-in usage:
    cp TacoRec.py recbole/model/sequential_recommender/coupledhsr.py
    python run_ampl_seq_full.py --model TacoRec --dataset ampl_jd_purchase \
        --config_files configs/model/TacoRec.yaml -g 0
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender


def _config_get(config, key, default=None):
    """Old RecBole Config may not support .get()."""
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


class SafeSelfAttentionBlock(nn.Module):
    """Small safe Transformer block without torch.nn.MultiheadAttention."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        inner_size: int,
        dropout: float,
        attn_dropout: float,
        layer_norm_eps: float = 1e-12,
        hidden_act: str = "gelu",
        causal: bool = True,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must divide n_heads={n_heads}")
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        act = nn.GELU() if str(hidden_act).lower() == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            act,
            nn.Dropout(dropout),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        key_mask = valid_mask[:, None, None, :].to(dtype=torch.bool)
        attn = attn.masked_fill(~key_mask, -1e4)

        if self.causal:
            causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            attn = attn.masked_fill(~causal_mask.view(1, 1, T, T), -1e4)

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

    Efficient first-order spectral transfer:
        base_b(ω) = Ψ_b(ω) F_b(ω) / D_b(ω)
        q_t(ω) = base_t(ω) + Σ_s C[t,s,ω] base_s(ω)

    v5 keeps the efficient approximation but exposes:
        - causal/symmetric/none coupling
        - frequency-band interpolation
        - local depth-wise impulse
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
        self.coupling_mode = str(coupling_mode)
        self.n_coup_bands = max(1, int(n_coup_bands))
        self.coupling_scale = float(coupling_scale)

        n_freq = max_len // 2 + 1
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.in_proj = nn.Linear(hidden_size, hidden_size * 3)

        self.m_raw = nn.Parameter(torch.zeros(self.B, hidden_size))
        self.c_raw = nn.Parameter(torch.zeros(self.B, hidden_size))
        self.k_raw = nn.Parameter(torch.zeros(self.B, hidden_size))

        self.psi_re = nn.Parameter(torch.full((self.B, hidden_size, n_freq), 0.05))
        self.psi_im = nn.Parameter(torch.zeros(self.B, hidden_size, n_freq))

        self.coupling = nn.Parameter(
            torch.zeros(self.B, self.B, hidden_size, self.n_coup_bands)
        )

        cmask = self._build_causal_mask(funnel_order)
        self.register_buffer("cmask", cmask)

        omega = 2.0 * math.pi * torch.arange(n_freq).float() / max_len
        self.register_buffer("omega", omega)

        self.kernel_size = int(kernel_size)
        self.local_pad = self.kernel_size - 1
        self.impulse_conv = nn.Conv1d(
            hidden_size, hidden_size, self.kernel_size, groups=hidden_size, bias=False
        )

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

        # Default causal order: 1 -> 2 -> ...; 0 is pad/mask.
        if not funnel_order:
            funnel_order = list(range(1, self.B))
        for tgt_pos, tgt in enumerate(funnel_order):
            for src in funnel_order[:tgt_pos]:
                tgt_i, src_i = int(tgt), int(src)
                if 0 <= tgt_i < self.B and 0 <= src_i < self.B:
                    cmask[tgt_i, src_i] = 1.0
        return cmask

    def _interp_coupling(self, n_freq: int) -> torch.Tensor:
        raw = torch.tanh(self.coupling) * self.cmask[:, :, None, None]
        if self.n_coup_bands == n_freq:
            full = raw
        else:
            full = F.interpolate(
                raw.reshape(self.B * self.B * self.hidden_size, 1, self.n_coup_bands),
                size=n_freq,
                mode="linear",
                align_corners=True if self.n_coup_bands > 1 else None,
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

        # [batch, behavior, time, dim]
        force_b = onehot.permute(0, 2, 1).unsqueeze(-1) * force.unsqueeze(1)

        # [batch, behavior, dim, freq]
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
        ).unsqueeze(0)

        base = psi * F_hat / (denom + 1e-6)

        if self.coupling_mode != "none":
            C = self._interp_coupling(n_freq).to(device=base.device, dtype=base.dtype)
            transfer = torch.einsum("tsdf,bsdf->btdf", C, base)
            q_hat = base + transfer
        else:
            q_hat = base

        q_time = torch.fft.irfft(q_hat, n=self.max_len, dim=-1)[..., :T]
        q_time = q_time.permute(0, 1, 3, 2).contiguous()
        q_seq = torch.einsum("btv,bvtd->btd", onehot, q_time)

        local = F.pad(local_u.transpose(1, 2), (self.local_pad, 0))
        local = self.impulse_conv(local).transpose(1, 2)

        delta = self.out_proj((q_seq + local) * gate)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=10.0, neginf=-10.0)
        return self.dropout(delta) * valid_mask.unsqueeze(-1)


class TargetConditionedBehaviorMemory(nn.Module):
    """AMPL-inspired lightweight behavior memory.

    Given sequence states H and behavior ids, this module creates one summary
    vector per behavior, then uses the target behavior embedding as a query to
    retrieve target-specific behavior context.

    This is intentionally much lighter than AMPL's IPL/SIPL/BIPL multi-view
    decomposition, but it supplies the missing target conditioning that v4 lacks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_behavior_tokens: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.B = num_behavior_tokens
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def summarize(
        self,
        x: torch.Tensor,
        behavior_ids: torch.Tensor,
        memory_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B,T,D], behavior_ids: [B,T], memory_valid_mask: [B,T]
        Bsz, T, D = x.shape
        behavior_ids = behavior_ids.clamp(0, self.B - 1)
        onehot = F.one_hot(behavior_ids.long(), num_classes=self.B).to(x.dtype)
        onehot = onehot * memory_valid_mask.unsqueeze(-1).to(x.dtype)
        denom = onehot.sum(dim=1).clamp(min=1.0).unsqueeze(-1)       # [B,Bh,1]
        summaries = torch.einsum("btv,btd->bvd", onehot, x) / denom # [B,Bh,D]
        return summaries

    def query(
        self,
        x_pos: torch.Tensor,
        summaries: torch.Tensor,
        target_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_pos: [N,D], summaries: [N,Bh,D], target_emb: [N,D]
        q = self.q_proj(target_emb).unsqueeze(1)    # [N,1,D]
        k = self.k_proj(summaries)                  # [N,Bh,D]
        v = self.v_proj(summaries)                  # [N,Bh,D]
        score = (q * k).sum(-1) / math.sqrt(self.hidden_size)
        # Do not attend to padding behavior 0 when possible.
        if summaries.size(1) > 1:
            score[:, 0] = -1e4
        attn = F.softmax(score.float(), dim=-1).to(x_pos.dtype)
        ctx = torch.sum(attn.unsqueeze(-1) * v, dim=1)
        ctx = self.dropout(self.out_proj(ctx))
        g = torch.sigmoid(self.gate(torch.cat([x_pos, ctx], dim=-1)))
        return x_pos + g * ctx, attn

    def forward(
        self,
        x: torch.Tensor,
        behavior_ids: torch.Tensor,
        memory_valid_mask: torch.Tensor,
        target_behavior: torch.Tensor,
        type_embedding: nn.Embedding,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        summaries = self.summarize(self.norm(x), behavior_ids, memory_valid_mask)
        target_behavior = target_behavior.clamp(0, self.B - 1)
        target_emb = type_embedding(target_behavior)
        return self.query(x, summaries, target_emb)


class TacoRec(SequentialRecommender):
    """v5 model."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.hidden_size = int(_config_get(config, "hidden_size", 50))
        self.n_layers = int(_config_get(config, "num_layers", _config_get(config, "n_layers", 1)))
        self.n_heads = int(_config_get(config, "num_heads", _config_get(config, "n_heads", 1)))
        self.inner_size = int(_config_get(config, "inner_size", self.hidden_size * 2))
        self.dropout_prob = float(_config_get(config, "hidden_dropout_prob", _config_get(config, "dropout_prob", 0.5)))
        self.attn_dropout_prob = float(_config_get(config, "attn_dropout_prob", self.dropout_prob))
        self.layer_norm_eps = float(_config_get(config, "layer_norm_eps", 1e-12))
        self.hidden_act = str(_config_get(config, "hidden_act", "gelu"))
        self.causal_anchor = bool(_config_get(config, "causal_anchor", True))

        self.mask_ratio = float(_config_get(config, "mask_ratio", 0.0))
        self.kernel_size = int(_config_get(config, "kernel_size", 3))
        self.n_coup_bands = int(_config_get(config, "n_coupling_bands", _config_get(config, "n_coup_bands", 8)))
        self.coupling_mode = str(_config_get(config, "coupling_mode", "causal"))
        self.coupling_scale = float(_config_get(config, "coupling_scale", 0.1))
        self.hnn_residual_init = float(_config_get(config, "hnn_residual_init", -6.0))

        self.use_hnn = bool(_config_get(config, "use_hnn", True))
        self.use_transformer = bool(_config_get(config, "use_transformer", True))
        self.use_behavior_memory = bool(_config_get(config, "use_behavior_memory", True))
        self.use_target_residual_gate = bool(_config_get(config, "use_target_residual_gate", True))
        self.mask_behavior_as_target = bool(_config_get(config, "mask_behavior_as_target", True))
        self.use_interaction_target_behavior = bool(_config_get(config, "use_interaction_target_behavior", True))

        self.residual_warmup_steps = int(_config_get(config, "residual_warmup_steps", 0))
        self.max_residual_scale = float(_config_get(config, "max_residual_scale", 1.0))
        self.memory_scale_init = float(_config_get(config, "memory_scale_init", -2.0))

        self.loss_type = str(_config_get(config, "loss_type", "CE")).upper()
        self.num_neg = int(_config_get(config, "num_neg", 1))
        self.alpha_reg = float(_config_get(config, "alpha_reg", 0.0))

        self.max_len = int(_config_get(config, "MAX_ITEM_LIST_LENGTH", 50))
        self.model_seq_len = self.max_len + 1
        self.initializer_range = float(_config_get(config, "initializer_range", 0.02))
        self.dataset_name = _config_get(config, "dataset", "")

        self.type_seq_field = str(_config_get(config, "ITEM_TYPE_SEQ_FIELD", "item_type_list"))
        self.target_type_field = str(_config_get(config, "ITEM_TYPE_FIELD", "item_type"))

        self.type_vocab_size = self._infer_type_vocab_size(config, dataset)
        self.default_target_behavior = self._infer_default_target_behavior(config, dataset)

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
                    causal=self.causal_anchor,
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
            if self.use_target_residual_gate:
                self.target_residual_gate = nn.Linear(self.hidden_size, hnn_layers)
            else:
                self.target_residual_gate = None
        else:
            self.hnn_alpha = None
            self.target_residual_gate = None

        if self.use_behavior_memory:
            self.behavior_memory = TargetConditionedBehaviorMemory(
                hidden_size=self.hidden_size,
                num_behavior_tokens=self.type_vocab_size,
                dropout=self.dropout_prob,
                layer_norm_eps=self.layer_norm_eps,
            )
            self.memory_scale = nn.Parameter(torch.tensor(self.memory_scale_init))
        else:
            self.behavior_memory = None
            self.memory_scale = None

        self.final_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self.apply(self._init_weights)

        with torch.no_grad():
            nn.init.normal_(self.item_embedding.weight[self.mask_token], mean=0.0, std=self.initializer_range)

    def _infer_type_vocab_size(self, config, dataset) -> int:
        candidates = [int(_config_get(config, "num_behaviors", 4)) + 1]
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
                    candidates.append((max(vals) + 1) if vals else (len(mapping) + 1))
            except Exception:
                pass
        return max(candidates)

    def _infer_default_target_behavior(self, config, dataset) -> int:
        explicit = _config_get(config, "target_behavior_token", None)
        if explicit is not None:
            return int(explicit)
        # Fallback: purchase is usually the last non-padding behavior.
        return min(int(_config_get(config, "num_behaviors", 4)), self.type_vocab_size - 1)

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
        return torch.full((batch_size,), int(self.default_target_behavior), dtype=torch.long, device=device)

    def _append_target_and_mask(
        self,
        item_seq: torch.Tensor,
        type_seq: torch.Tensor,
        target_item: torch.Tensor,
        target_behavior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = item_seq.shape
        device = item_seq.device
        lengths = torch.count_nonzero(item_seq, dim=1).clamp(max=T)

        zero_col = torch.zeros(B, 1, dtype=item_seq.dtype, device=device)
        item_ext = torch.cat([item_seq, zero_col], dim=1)
        type_ext = torch.cat([type_seq, zero_col], dim=1)

        row = torch.arange(B, device=device)
        item_ext[row, lengths] = target_item
        type_ext[row, lengths] = target_behavior

        L = item_ext.shape[1]
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        hist_pos = pos < lengths.unsqueeze(1)
        target_pos = pos == lengths.unsqueeze(1)

        # v5 default: mask only appended target unless mask_ratio > 0.
        rand_mask = (torch.rand(B, L, device=device) < self.mask_ratio) & hist_pos & (item_ext != 0)
        mask_positions = rand_mask | target_pos

        targets = item_ext[mask_positions].long()
        masked_target_behavior = type_ext[mask_positions].long().clamp(0, self.type_vocab_size - 1)

        masked_items = item_ext.clone()
        masked_types = type_ext.clone()
        masked_items[mask_positions] = self.mask_token
        if self.mask_behavior_as_target:
            masked_types[mask_positions] = type_ext[mask_positions]
        else:
            masked_types[mask_positions] = 0

        return (
            masked_items[:, : self.model_seq_len],
            masked_types[:, : self.model_seq_len].clamp(0, self.type_vocab_size - 1),
            mask_positions[:, : self.model_seq_len],
            targets.clamp(0, self.n_items - 1),
            masked_target_behavior,
        )

    def _append_test_mask(
        self,
        item_seq: torch.Tensor,
        item_seq_len: torch.Tensor,
        type_seq: torch.Tensor,
        target_behavior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = item_seq.shape
        device = item_seq.device
        lengths = item_seq_len.long().clamp(min=0, max=T)
        zero_col = torch.zeros(B, 1, dtype=item_seq.dtype, device=device)
        item_ext = torch.cat([item_seq, zero_col], dim=1)
        type_ext = torch.cat([type_seq, zero_col], dim=1)
        row = torch.arange(B, device=device)
        item_ext[row, lengths] = self.mask_token
        type_ext[row, lengths] = target_behavior if self.mask_behavior_as_target else 0
        return (
            item_ext[:, : self.model_seq_len],
            type_ext[:, : self.model_seq_len].clamp(0, self.type_vocab_size - 1),
            lengths.clamp(max=self.model_seq_len - 1),
        )

    def _warmup_scale(self) -> float:
        if self.residual_warmup_steps <= 0:
            return 1.0
        step = int(self.global_step.item())
        return min(1.0, float(step) / float(max(1, self.residual_warmup_steps)))

    def encode(
        self,
        item_seq: torch.Tensor,
        type_seq: torch.Tensor,
        target_behavior: torch.Tensor,
    ) -> torch.Tensor:
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
        warm = self._warmup_scale()
        target_emb = self.type_embedding(target_behavior.clamp(0, self.type_vocab_size - 1))

        for i in range(n_steps):
            if self.use_transformer and i < len(self.transformer_blocks):
                x = self.transformer_blocks[i](x, valid_mask)

            if self.use_hnn and i < len(self.hnn_blocks):
                delta = self.hnn_blocks[i](x, valid_mask, type_seq)
                base_alpha = self.hnn_alpha[i]
                if self.use_target_residual_gate and self.target_residual_gate is not None:
                    target_alpha = self.target_residual_gate(target_emb)[:, i].view(B, 1, 1)
                    alpha = torch.sigmoid(base_alpha + target_alpha)
                else:
                    alpha = torch.sigmoid(base_alpha).view(1, 1, 1)
                alpha = alpha * self.max_residual_scale * warm
                x = x + alpha * delta
                x = x * valid_mask.unsqueeze(-1)

        return self.final_norm(x)

    def enhance_readout(
        self,
        seq_output: torch.Tensor,
        item_seq: torch.Tensor,
        type_seq: torch.Tensor,
        positions: torch.Tensor,
        target_behavior: torch.Tensor,
    ) -> torch.Tensor:
        row = torch.arange(seq_output.size(0), device=seq_output.device)
        q = seq_output[row, positions.clamp(max=seq_output.size(1) - 1)]

        if not self.use_behavior_memory or self.behavior_memory is None:
            return q

        memory_valid = (item_seq != 0) & (item_seq != self.mask_token)
        summaries = self.behavior_memory.summarize(seq_output, type_seq, memory_valid)
        # Reuse query logic with a global scale. target_behavior is [B].
        target_emb = self.type_embedding(target_behavior.clamp(0, self.type_vocab_size - 1))
        q2, _ = self.behavior_memory.query(q, summaries, target_emb)
        scale = torch.sigmoid(self.memory_scale)
        return q + scale * (q2 - q)

    def calculate_loss(self, interaction):
        if self.training:
            self.global_step += 1

        item_seq = interaction[self.ITEM_SEQ]
        type_seq = self._get_type_seq(interaction, item_seq)
        target_item = interaction[self.POS_ITEM_ID] if _has_inter_field(interaction, self.POS_ITEM_ID) else interaction[self.ITEM_ID]
        target_behavior = self._get_target_behavior(interaction, item_seq.size(0), item_seq.device)

        masked_items, masked_types, mask_positions, targets, masked_target_behavior = self._append_target_and_mask(
            item_seq, type_seq, target_item, target_behavior
        )

        seq_output = self.encode(masked_items, masked_types, target_behavior)

        # For masked-history positions, target behaviors may differ. Build enhanced q for all masked positions.
        masked_output = seq_output[mask_positions]
        if self.use_behavior_memory and self.behavior_memory is not None:
            memory_valid = (masked_items != 0) & (masked_items != self.mask_token)
            summaries = self.behavior_memory.summarize(seq_output, masked_types, memory_valid)
            batch_idx = mask_positions.nonzero(as_tuple=False)[:, 0]
            q_summaries = summaries[batch_idx]
            q_target_emb = self.type_embedding(masked_target_behavior.clamp(0, self.type_vocab_size - 1))
            q2, _ = self.behavior_memory.query(masked_output, q_summaries, q_target_emb)
            scale = torch.sigmoid(self.memory_scale)
            masked_output = masked_output + scale * (q2 - masked_output)

        item_table = self.item_embedding.weight[: self.n_items]

        if self.loss_type == "BCE":
            pos_score = torch.sum(masked_output * item_table[targets], dim=-1)
            neg_items = torch.randint(1, self.n_items, (targets.size(0), self.num_neg), device=targets.device)
            neg_emb = item_table[neg_items]
            neg_score = torch.einsum("nd,nkd->nk", masked_output, neg_emb)
            scores = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
            labels = torch.zeros_like(scores)
            labels[:, 0] = 1.0
            loss = self.bce_loss(scores, labels)
        elif self.loss_type == "BPR":
            neg_items = torch.randint(1, self.n_items, targets.shape, device=targets.device)
            neg_items = torch.where(neg_items.eq(targets), (neg_items % (self.n_items - 1)) + 1, neg_items)
            pos_score = torch.sum(masked_output * item_table[targets], dim=-1)
            neg_score = torch.sum(masked_output * item_table[neg_items], dim=-1)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
        else:
            logits = torch.matmul(masked_output, item_table.transpose(0, 1))
            loss = self.ce_loss(logits, targets)

        if self.use_hnn and self.hnn_alpha is not None and self.alpha_reg > 0:
            loss = loss + self.alpha_reg * torch.sigmoid(self.hnn_alpha).mean()
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        type_seq = self._get_type_seq(interaction, item_seq)
        if _has_inter_field(interaction, self.ITEM_SEQ_LEN):
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
        else:
            item_seq_len = torch.count_nonzero(item_seq, dim=1)

        target_behavior = self._get_target_behavior(interaction, item_seq.size(0), item_seq.device)
        masked_items, masked_types, mask_pos = self._append_test_mask(
            item_seq, item_seq_len, type_seq, target_behavior
        )
        seq_output = self.encode(masked_items, masked_types, target_behavior)
        q = self.enhance_readout(seq_output, masked_items, masked_types, mask_pos, target_behavior)

        item_table = self.item_embedding.weight[: self.n_items]
        scores = torch.matmul(q, item_table.transpose(0, 1))
        scores[:, 0] = -1e9
        return scores

    def predict(self, interaction):
        scores = self.full_sort_predict(interaction)
        return scores.gather(1, interaction[self.ITEM_ID].view(-1, 1)).squeeze(1)


# RecBole looks for class name equal to --model TacoRec.
class TacoRec(TacoRec):
    pass
