"""Inkling attention: hybrid local/global, per-head q/k RMSNorm, relative-position
logits bias, optional log-scaling, and short-convolution on k/v.

Mirrors ``InklingAttention`` + ``InklingRelativeLogits`` from transformers PR #47347.
This implementation is prefill-oriented (full-sequence, no KV cache); an incremental
cache (including the 4 per-layer conv states) can be layered on top later.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .common import RMSNorm, ShortConvolution
from .config import TextConfig

NEG_INF = -1e30


class RelativeLogits(nn.Module):
    """Hidden-state-conditioned relative position bias.

    ``proj`` is a bank of bias-vs-distance profiles ``[d_rel, rel_extent]``. Each
    query's ``d_rel`` relative-state vector mixes them into one bias value per
    backward distance; the bias is zero outside ``0 <= distance < rel_extent``.
    """

    def __init__(self, d_rel: int, rel_extent: int):
        super().__init__()
        self.rel_extent = rel_extent
        self.proj = mx.zeros((d_rel, rel_extent))

    def __call__(self, relative_states, q_pos, kv_pos):
        # relative_states: [B, Lq, heads, d_rel]
        # rel_logits: [B, Lq, heads, rel_extent] -> [B, heads, Lq, rel_extent]
        rel_logits = mx.swapaxes(relative_states @ self.proj, 1, 2)
        B, H, Lq, _ = rel_logits.shape
        distance = q_pos[:, None] - kv_pos[None, :]          # [Lq, Lkv]
        gather = mx.clip(distance, 0, self.rel_extent - 1)   # [Lq, Lkv]
        gather = mx.broadcast_to(gather[None, None], (B, H, Lq, gather.shape[-1]))
        bias = mx.take_along_axis(rel_logits, gather, axis=-1)  # [B, H, Lq, Lkv]
        valid = (distance >= 0) & (distance < self.rel_extent)  # [Lq, Lkv]
        return mx.where(valid[None, None], bias, 0.0)


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"

        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.num_heads = config.swa_num_attention_heads if self.is_sliding else config.num_attention_heads
        self.num_kv_heads = config.swa_num_key_value_heads if self.is_sliding else config.num_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.sliding_window = config.sliding_window_size if self.is_sliding else None
        self.rel_extent = config.sliding_window_size if self.is_sliding else config.rel_extent
        self.d_rel = config.d_rel
        # q/k are per-head RMS-normalized, hence 1/d rather than 1/sqrt(d)
        self.scaling = 1.0 / self.head_dim

        h = config.hidden_size
        self.wq_du = nn.Linear(h, self.num_heads * self.head_dim, bias=False)
        self.wk_dv = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.wv_dv = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.wr_du = nn.Linear(h, self.num_heads * self.d_rel, bias=False)
        self.wo_ud = nn.Linear(self.num_heads * self.head_dim, h, bias=False)

        self.k_sconv = ShortConvolution(self.num_kv_heads * self.head_dim, config.sconv_kernel_size)
        self.v_sconv = ShortConvolution(self.num_kv_heads * self.head_dim, config.sconv_kernel_size)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rel_logits_proj = RelativeLogits(self.d_rel, self.rel_extent)

    def __call__(self, hidden_states, start_pos=0, kv_cache=None,
                 k_conv=None, v_conv=None, conv_mask=None):
        B, L, _ = hidden_states.shape

        q = self.wq_du(hidden_states)
        k = self.k_sconv(self.wk_dv(hidden_states), mask=conv_mask, cache=k_conv)
        v = self.v_sconv(self.wv_dv(hidden_states), mask=conv_mask, cache=v_conv)
        rel = self.wr_du(hidden_states)

        q = self.q_norm(q.reshape(B, L, self.num_heads, self.head_dim))
        k = self.k_norm(k.reshape(B, L, self.num_kv_heads, self.head_dim))
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)

        # -> [B, heads, L, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q_pos = mx.arange(L) + start_pos
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)      # full history
        kv_pos = mx.arange(k.shape[2])

        rel = rel.reshape(B, L, self.num_heads, self.d_rel)
        position_bias = self.rel_logits_proj(rel, q_pos, kv_pos)  # [B, heads, Lq, Lkv]

        # log-scaling (global layers only; no-op for context <= n_floor)
        if not self.is_sliding and self.config.log_scaling_n_floor is not None:
            n_floor = self.config.log_scaling_n_floor
            eff_n = (q_pos + 1).astype(mx.float32)
            tau = 1.0 + self.config.log_scaling_alpha * mx.log(
                mx.maximum(eff_n / n_floor, 1.0)
            )
            tau_q = tau.reshape(1, 1, -1, 1)
            q = (q.astype(mx.float32) * tau_q).astype(q.dtype)
            position_bias = (position_bias.astype(mx.float32) * tau_q).astype(position_bias.dtype)

        # Fused SDPA: O = softmax(scale·QKᵀ + mask)·V, softmax in fp32, native GQA
        # (k/v passed un-tiled). mask = relative-position bias + causal/sliding, additive.
        mask = position_bias + self._causal_mask(q_pos, kv_pos)   # [B, heads, Lq, Lkv]
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scaling, mask=mask.astype(q.dtype)
        )                                                        # [B, heads, Lq, head_dim]

        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        return self.wo_ud(out)

    def _causal_mask(self, q_pos, kv_pos):
        distance = q_pos[:, None] - kv_pos[None, :]  # [Lq, Lkv]
        allowed = distance >= 0
        if self.sliding_window is not None:
            allowed = allowed & (distance < self.sliding_window)
        mask = mx.where(allowed, 0.0, NEG_INF)
        return mask[None, None].astype(mx.float32)
