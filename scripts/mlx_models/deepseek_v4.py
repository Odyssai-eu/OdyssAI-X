# Copyright © 2026 OdyssAI
#
# DeepSeek-V4-Flash — `model_type: deepseek_v4`, arch `DeepseekV4ForCausalLM`.
# Full rewrite (2026-06-07) per recon w2tumujwa vs ref-deepseek-v4/ + the real
# 3513-key DeepSeek-V4-Flash-8bit checkpoint. The earlier V4-Pro draft (which
# grafted deepseek_v32 attention/MoE) was wrong on every axis and is replaced.
#
# Flash ≠ Pro ≠ v32. Per-layer it is: MLA (combined wkv latent, K=V) + attn_sink
# + grouped o-LoRA (wo_a.0..7 + wo_b) + an optional Compressor (long-context KV
# compression) + an optional Indexer (sparse top-k), with a sliding window (128).
# All 43 layers are MoE; routing is hash (tid2eid) on layers 0-2 and noaux_tc
# (bias-shifted top-k) on 3-42; scoring is sqrtsoftplus. Residuals flow through
# HyperConnection (hc_mult=4 copies). embed/head/experts/attn-projs are 8-bit
# affine (group_size 64); hc_*, attn_sink, ape, gate.bias/tid2eid stay raw.
#
# STATUS (2026-06-07): P0 structural PASS (1629/1629 keys) + forward smoke PASS +
# COMPONENT NUMERICAL VALIDATION PASS (vs pure-numpy oracles of the reference math,
# ~1e-7 float32; /tmp/numtest_dsv4.py). Double-review (Claude re-derivation + Codex)
# + a 5-agent reference-mapping workflow underpin this — see PLAN-REVIEW-LOG-dsv4.md.
# VALIDATED: rope (FlashRoPE — interleaved pairs, YaRN inv_freq, inverse on output),
# attn_sink, grouped o-LoRA, noaux_tc gate, hash routing, the Compressor (overlap
# joint-2*ratio softmax + plain pool + strided group-start rope), the combined
# window-128 + compressed-block dense mask, the combined-KV sink-softmax, and the
# SwiGLU swiglu_limit=10 clamp. The PREFILL forward is numerically faithful for
# prompts up to ~2048 tokens. Two Codex review rounds converged (REVISE, no blocker).
# DEFERRED (P3, documented drifts — empirical validation on the real model, NOT
# short-context correctness blockers): Indexer top-k pruning (only bites for prompts
# > index_topk*ratio ≈ 2048; we attend all visible blocks, a graceful superset); kv
# non-rope act-quant FP8 sim (no MLX fp8 dtype; QAT-robust → safe to skip); the
# incremental-decode KV cache (full-recompute on decode). No custom Metal kernel;
# dense-mask emulation throughout.
#
# REAL-MODEL STATUS (updated 2026-06-08 after empirical A/B on real V4-Flash 3-node):
# the real 282GB DeepSeek-V4-Flash-8bit LOADS and runs a CORRECT single-node forward
# (".../France is" -> " Paris", 2.6s). The earlier "MLX-distributed collective
# deadlock" framing was a MISDIAGNOSIS — corrected here:
#   (1) Q8 CACHE CRASH (THIS FIX): with kv_q8 (prod default), the Hy3 cache read hit
#       QuantizedKVCache's tuple return -> rank 0 crashed mid-forward, surviving ranks
#       busy-waited -> looked like a deadlock. Fixed below (dequantize-on-read).
#   (2) rank-0 COMPUTE HANG = the wired-memory leak (blocker B): gone on clean memory.
#   (3) RESIDUAL: on clean memory the LAST rank still wedges materializing its shard
#       (some op) — under diagnosis (per-layer-eval trace), not yet root-caused.
# Blocker B (wired leak survives process exit) is a macOS Metal-driver issue, not this
# module; mitigated by reboot / load-once. See SESSION-2026-06-08 + PLAN-REVIEW-LOG-dsv4.md.
# The Hy3-pattern cache + flat-3D HC activation are kept (correct + pipeline-friendly).

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    # MoE
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    n_shared_experts: int = 1
    moe_intermediate_size: int = 2048
    routed_scaling_factor: float = 1.5
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    norm_topk_prob: bool = True
    # Indexer (sparse attention)
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # Long-context compression
    compress_ratios: list = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    sliding_window: int = 128
    # HyperConnection
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    # rope / norm
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 1  # MTP — NOT shipped in the Flash ckpt; skipped.

    def __post_init__(self):
        if not self.compress_ratios:
            # default cadence [0,0,4,128,4,128,...] up to num_hidden_layers
            cr = [0, 0]
            while len(cr) < self.num_hidden_layers:
                cr += [4, 128]
            self.compress_ratios = cr[: self.num_hidden_layers]


# ── HyperConnection (confirmed correct vs reference) ────────────────────────

def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps):
    """mixes [...,(2+hc)*hc] -> (pre[...,hc], post[...,hc], comb[...,hc,hc]).
    pre/post = sigmoid affines ; comb = Sinkhorn doubly-stochastic
    (reference_kernel.py:391-425)."""
    hc = hc_mult
    pre = mx.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * mx.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = comb.reshape(*comb.shape[:-1], hc, hc)
    comb = mx.softmax(comb.astype(mx.float32), axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _rsqrt_norm(xf, eps):
    return mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + eps)


def _flash_mask(s, n_real, offset, window, ratio, n_comp, dtype):
    # Additive attention mask over the combined key axis [window(n_real) ++ compressed(n_comp)],
    # the dense-mask emulation of sparse_attn's topk gather (ref get_window_topk_idxs
    # 255-265 + get_compress_topk_idxs 268-276). Offset-aware for decode: query local index
    # i sits at absolute position offset+i; window keys are absolute positions 0..n_real-1.
    #   window region : query attends real j iff  (offset+i)-window < j <= (offset+i)  (sliding-128)
    #   compressed reg: query attends slot c iff  c < (offset+i+1)//ratio              (fully-past blocks)
    # attn_sink is NOT a column here — it stays in the softmax denominator.
    # P3: for ratio==4 with n_comp > index_topk (prompt > ~2048), the real Indexer prunes the
    # compressed slots to the top-512; here we attend ALL visible blocks (graceful superset).
    i = mx.arange(s).reshape(s, 1) + offset                                            # absolute query pos
    j = mx.arange(n_real).reshape(1, n_real)                                           # window key abs pos
    win_ok = (j <= i) & (j > i - window)
    mask = mx.where(win_ok, mx.array(0.0, dtype=dtype), mx.array(-1e9, dtype=dtype))   # [s,n_real]
    if n_comp > 0 and ratio > 0:                                                        # ratio>0 guaranteed when n_comp>0
        c = mx.arange(n_comp).reshape(1, n_comp)
        cutoff = (i + 1) // ratio                                                       # [s,1]
        comp_ok = c < cutoff
        comp_mask = mx.where(comp_ok, mx.array(0.0, dtype=dtype), mx.array(-1e9, dtype=dtype))
        mask = mx.concatenate([mask, comp_mask], axis=-1)                               # [s, n_real+n_comp]
    return mask


# ── RoPE (per-layer: YaRN on compressed layers, base on sliding-only) ───────
# Interleaved-pair rope matching reference apply_rotary_emb (view_as_complex on
# consecutive pairs == nn.RoPE traditional=True, NOT the half-split variant) with
# YaRN NTK-by-parts freq interpolation (precompute_freqs_cis, ref:200-228).

def _yarn_inv_freq(dim, base, factor, original_seq_len, beta_fast, beta_slow):
    """inv_freq [dim//2]; YaRN linear-ramp interpolation when original_seq_len>0."""
    freqs = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    if original_seq_len and original_seq_len > 0:
        def corr_dim(num_rot):
            return dim * math.log(original_seq_len / (num_rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corr_dim(beta_fast)), 0)
        high = min(math.ceil(corr_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = mx.clip((mx.arange(dim // 2, dtype=mx.float32) - low) / (high - low), 0.0, 1.0)
        smooth = 1.0 - ramp                      # NTK-by-parts: 1 outside [low,high]
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


class FlashRoPE:
    """Plain class (NOT nn.Module) → contributes no parameters, so the P0 key-diff
    stays exact. Rotates consecutive interleaved pairs; forward on q/kv, inverse
    (complex conjugate, ref:236-237) on the attention output."""

    def __init__(self, args: ModelArgs, compressed: bool):
        dim = args.qk_rope_head_dim
        if compressed:
            rs = args.rope_scaling or {}
            base = args.compress_rope_theta
            factor = float(rs.get("factor", 1.0))
            osl = int(rs.get("original_max_position_embeddings", 0))
            bf = float(rs.get("beta_fast", 32))
            bs = float(rs.get("beta_slow", 1))
        else:
            base, factor, osl, bf, bs = args.rope_theta, 1.0, 0, 0.0, 0.0
        self.inv_freq = _yarn_inv_freq(dim, base, factor, osl, bf, bs)  # [dim//2]

    def __call__(self, x, inverse=False, offset=0, positions=None):
        # x: [..., seq(axis 1), ..., rd]; rotate the last (rope) dim in pairs.
        # positions: explicit per-token positions [seq] (e.g. strided c*ratio for
        # compressed KV, ref:364); defaults to contiguous arange(offset, offset+seq).
        *lead, rd = x.shape
        seq = x.shape[1]
        half = rd // 2
        pos = positions if positions is not None else mx.arange(offset, offset + seq, dtype=mx.float32)
        ang = pos[:, None] * self.inv_freq[None, :]           # [seq, half]
        cos, sin = mx.cos(ang), mx.sin(ang)
        x2 = x.reshape(*lead, half, 2).astype(mx.float32)
        xr, xi = x2[..., 0], x2[..., 1]                        # [..., half]
        bshape = [1] * xr.ndim
        bshape[1], bshape[-1] = seq, half                     # broadcast over batch/heads
        cos, sin = cos.reshape(bshape), sin.reshape(bshape)
        if inverse:                                           # conjugate: rotate by -theta
            yr, yi = xr * cos + xi * sin, -xr * sin + xi * cos
        else:
            yr, yi = xr * cos - xi * sin, xr * sin + xi * cos
        return mx.stack([yr, yi], axis=-1).reshape(*lead, rd).astype(x.dtype)


# ── Compressor (gated softmax mean-pool over `ratio` tokens) ────────────────

class FlashCompressor(nn.Module):
    """Long-context KV compression (reference_model.py:279-377), prefill path.
    ratio==4 uses overlap (coff=2): each output group's softmax mixes the PREVIOUS
    group's first-half contributors with the CURRENT group's second-half — ONE joint
    softmax over 2*ratio (ref overlap_transform 307-342), NOT two independent
    softmaxes summed. ratio==128 is a plain gated mean-pool (coff=1). The compressed
    KV is RMSNorm'd then rope'd at the group-start absolute positions g*ratio
    (ref:362-367). Remainder (s % ratio) tokens are dropped — always within the
    sliding window, so output is unaffected. Decode/incremental state, act_quant and
    the indexer's Hadamard rotate are deferred (# P3)."""

    def __init__(self, args: ModelArgs, ratio: int, head_dim: int, hidden: int):
        super().__init__()
        self.ratio = ratio
        self.head_dim = head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.coff = 2 if ratio == 4 else 1
        out = self.coff * head_dim
        self.wkv = nn.Linear(hidden, out, bias=False)
        self.wgate = nn.Linear(hidden, out, bias=False)
        self.norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.ape = mx.zeros((ratio, out))
        self.rope = FlashRoPE(args, compressed=True)   # plain class → adds no params

    def __call__(self, x):
        b, s, _ = x.shape
        r, d = self.ratio, self.head_dim
        G = s // r                                     # full groups; drop remainder
        if G == 0:
            return mx.zeros((b, 0, d), dtype=x.dtype)
        xt = x[:, : G * r, :].astype(mx.float32)       # compression runs in fp32 (ref:322)
        kv = self.wkv(xt).reshape(b, G, r, self.coff * d)
        score = self.wgate(xt).reshape(b, G, r, self.coff * d) + self.ape   # ape on in-window axis
        if self.coff == 1:                             # plain ratio==128
            w = mx.softmax(score, axis=2)
            comp = mx.sum(w * kv, axis=2)              # [b,G,d]
        else:                                          # overlap ratio==4 — JOINT 2*ratio softmax
            kv_prev, kv_cur = kv[..., :d], kv[..., d:]
            sc_prev, sc_cur = score[..., :d], score[..., d:]
            zpad = mx.zeros((b, 1, r, d), dtype=kv.dtype)
            ninf = mx.full((b, 1, r, d), -1e9, dtype=score.dtype)   # group 0 has no prev
            kv8 = mx.concatenate([mx.concatenate([zpad, kv_prev[:, :-1]], axis=1), kv_cur], axis=2)
            sc8 = mx.concatenate([mx.concatenate([ninf, sc_prev[:, :-1]], axis=1), sc_cur], axis=2)
            w8 = mx.softmax(sc8, axis=2)               # one softmax over the 2*ratio contributors
            comp = mx.sum(kv8 * w8, axis=2)            # [b,G,d]
        comp = self.norm(comp.astype(x.dtype))
        a, rp = comp[..., : -self.rope_dim], comp[..., -self.rope_dim:]
        rp = self.rope(rp, positions=mx.arange(G, dtype=mx.float32) * r)   # group-start positions
        return mx.concatenate([a, rp], axis=-1)        # [b,G,d]


# ── Indexer (sparse top-k over compressed positions, ratio==4 layers) ───────

class FlashIndexer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.topk = args.index_topk
        self.softmax_scale = self.head_dim ** -0.5
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.compressor = FlashCompressor(args, ratio=4, head_dim=self.head_dim, hidden=args.hidden_size)

    def __call__(self, x, qr):
        # P3: returns the top-k compressed-KV indices to gather; faithful scoring
        # but dense (no Metal). Numerics validated separately.
        b, s, _ = x.shape
        q = self.wq_b(qr).reshape(b, s, self.n_heads, self.head_dim)
        idx_kv = self.compressor(x)  # [b, t, head_dim]
        if idx_kv.shape[1] == 0:
            return None
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        scores = mx.einsum("bshd,btd->bsht", q, idx_kv)
        scores = mx.maximum(scores, 0.0) * weights[..., None]
        scores = scores.sum(axis=2)  # [b, s, t]
        k = min(self.topk, idx_kv.shape[1])
        return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]


# ── Attention (MLA + sink + grouped o-LoRA + compressor + indexer) ──────────

class DeepseekV4FlashAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.scale = args.head_dim ** -0.5
        self.window = args.sliding_window
        ratio = args.compress_ratios[layer_idx]
        self.ratio = ratio

        self.wq_a = nn.Linear(args.hidden_size, args.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.attn_sink = mx.zeros((self.n_heads,))
        # grouped o-LoRA: o_groups separate down-projs + one up-proj
        group_in = (self.n_heads * self.head_dim) // args.o_groups
        self.wo_a = [nn.Linear(group_in, args.o_lora_rank, bias=False) for _ in range(args.o_groups)]
        self.wo_b = nn.Linear(args.o_groups * args.o_lora_rank, args.hidden_size, bias=False)
        self.o_groups = args.o_groups

        self.compressor = FlashCompressor(args, ratio, self.head_dim, args.hidden_size) if ratio else None
        self.indexer = FlashIndexer(args) if ratio == 4 else None
        self.rope = FlashRoPE(args, compressed=bool(ratio))

    def _rope(self, x, inverse=False, offset=0):
        # rope only on the last rope_dim dims (ref:499/505; inverse on output ref:534).
        # offset = cache.offset for decode-position-aware rope.
        a, b = x[..., : -self.rope_dim], x[..., -self.rope_dim:]
        return mx.concatenate([a, self.rope(b, inverse=inverse, offset=offset)], axis=-1)

    def __call__(self, x, mask=None, cache=None):
        # Prefill (start_pos==0) dense-mask emulation of sparse_attn: query attends a
        # sliding-window-128 of real keys + the fully-past compressed-KV blocks, one
        # joint softmax with attn_sink in the denominator. K==V==the combined latent.
        # The `mask` arg is ignored — we build the window+compressed mask per layer.
        # P3: window/compressed KV cache + incremental decode + Indexer top-k pruning
        # (only bites for prompts > ~2048) are deferred; full-recompute on decode.
        b, s, _ = x.shape
        off = cache.offset if cache is not None else 0
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(b, s, self.n_heads, self.head_dim)
        q = q * _rsqrt_norm(q.astype(mx.float32), self.args.rms_norm_eps).astype(q.dtype)  # ref:498 unweighted
        q = self._rope(q, offset=off)

        kv = self._rope(self.kv_norm(self.wkv(x)), offset=off)   # [b,s,d] window latent, K==V (ref:502-504)
        if cache is not None:
            # Hy3 pattern: store the REAL window latent and USE the returned cached kv in
            # attention (ref hy_v3.py:114-120). This makes cache.keys an UPSTREAM input of
            # the layer output/send (not a detached orphan that the pipeline's
            # mx.depends(cache.keys, send) retroactively couples downstream → the distributed
            # deadlock). Bonus: correct decode for contexts <= window (cached window covers
            # all real keys; the mask limits the active window to `self.window`).
            kvf, _ = cache.update_and_fetch(kv[:, None, :, :], kv[:, None, :, :])  # [b,1,total,d] or quantized tuple
            if isinstance(kvf, tuple):
                # kv_q8: QuantizedKVCache.update_and_fetch returns (data, scales,
                # biases) tuples, not arrays. Our Hy3-latent attention needs the
                # dense array (it concatenates compressed KV + builds masks), so
                # dequantize on read. Q8 is the prod default for big-MoE; this path
                # is never hit single-node with a plain/None cache, which is why
                # validation missed it and rank 0 crashed mid-forward distributed
                # (TypeError: tuple indices ...), masquerading as a deadlock.
                kvf = mx.dequantize(*kvf, group_size=cache.group_size, bits=cache.bits)
            kv = kvf[:, 0]                                                          # [b,total,d]
        n_real = kv.shape[1]                          # cached window length (= s at prefill)
        if self.compressor is not None:
            kv_comp = self.compressor(x)             # [b,n_comp,d] already RMSNorm'd + roped
            if kv_comp.shape[1] > 0:
                kv = mx.concatenate([kv, kv_comp], axis=1)   # append compressed AFTER window (ref:526)
        n_comp = kv.shape[1] - n_real
        kvh = kv[:, None, :, :]                       # [b,1,n_kv,d] broadcast over heads
        qh = q.transpose(0, 2, 1, 3)                  # [b,h,s,d]

        scores = (qh * self.scale) @ kvh.transpose(0, 1, 3, 2)  # [b,h,s,n_kv]
        scores = scores + _flash_mask(s, n_real, off, self.window, self.ratio, n_comp, scores.dtype)
        # attn_sink: virtual zero-value sink in the softmax denominator (kernel:345)
        m = mx.max(scores, axis=-1, keepdims=True)
        ex = mx.exp(scores - m)
        sink = mx.exp(self.attn_sink[None, :, None, None] - m)
        denom = ex.sum(axis=-1, keepdims=True) + sink
        attn = ex / denom
        o = attn @ kvh                               # [b,h,s,d]  V == COMBINED kv latent
        o = o.transpose(0, 2, 1, 3)                  # [b,s,h,d]
        o = self._rope(o, inverse=True, offset=off)  # inverse-rope on output rope dims (ref:534)
        o = o.reshape(b, s, self.n_heads * self.head_dim)
        # grouped o-LoRA
        og = o.reshape(b, s, self.o_groups, -1)
        downs = [self.wo_a[g](og[:, :, g, :]) for g in range(self.o_groups)]  # each [b,s,o_lora_rank]
        return self.wo_b(mx.concatenate(downs, axis=-1))


# ── MoE (dual-mode gate + SwitchGLU + shared expert) ────────────────────────

class _ClampedSwiGLU(nn.Module):
    """DeepSeek swiglu_limit clamp (reference Expert.forward:600-602): up is clamped
    to [-limit, limit], gate is max-clamped to limit, then silu(gate)*up. Bounds
    activation growth across the 43 MoE layers. SwitchGLU calls activation(x_up, x_gate)."""

    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x_up, x_gate):
        x_up = mx.clip(x_up, -self.limit, self.limit)
        x_gate = mx.minimum(x_gate, self.limit)
        return nn.silu(x_gate) * x_up


class FlashMoEGate(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.route_scale = args.routed_scaling_factor
        self.scoring_func = args.scoring_func
        self.hash = layer_idx < args.num_hash_layers
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        if self.hash:
            self.tid2eid = mx.zeros((args.vocab_size, self.top_k), dtype=mx.int32)
        else:
            self.bias = mx.zeros((args.n_routed_experts,))

    def __call__(self, x, input_ids):
        scores = x.astype(mx.float32) @ self.weight.T
        if self.scoring_func == "sqrtsoftplus":
            scores = mx.sqrt(nn.softplus(scores))
        elif self.scoring_func == "sigmoid":
            scores = mx.sigmoid(scores)
        else:
            scores = mx.softmax(scores, axis=-1)
        if self.hash:
            inds = self.tid2eid[input_ids.reshape(-1)].reshape(*x.shape[:-1], self.top_k)
        else:
            s2 = scores + self.bias
            inds = mx.argpartition(-s2, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        w = mx.take_along_axis(scores, inds, axis=-1)
        if self.scoring_func != "softmax":
            w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
        w = w * self.route_scale
        return inds, w.astype(x.dtype)


class FlashMoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.n_routed_experts,
                                    activation=_ClampedSwiGLU(args.swiglu_limit))
        self.swiglu_limit = args.swiglu_limit
        self.gate = FlashMoEGate(args, layer_idx)
        self.shared_experts = nn.Module()
        self.shared_experts.gate_proj = nn.Linear(args.hidden_size, args.moe_intermediate_size, bias=False)
        self.shared_experts.up_proj = nn.Linear(args.hidden_size, args.moe_intermediate_size, bias=False)
        self.shared_experts.down_proj = nn.Linear(args.moe_intermediate_size, args.hidden_size, bias=False)

    def _shared(self, x):
        se = self.shared_experts
        L = self.swiglu_limit
        gate = mx.minimum(se.gate_proj(x), L)
        up = mx.clip(se.up_proj(x), -L, L)
        return se.down_proj(nn.silu(gate) * up)

    def __call__(self, x, input_ids):
        inds, w = self.gate(x, input_ids)
        y = self.switch_mlp(x, inds)
        y = (y * w[..., None]).sum(axis=-2)
        return y + self._shared(x)


# ── HyperConnection layer wrapper (confirmed correct) ───────────────────────

class HCLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = DeepseekV4FlashAttention(args, layer_idx)
        self.mlp = FlashMoE(args, layer_idx)
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        d = args.hidden_size
        hc = args.hc_mult
        self.hc_mult = hc
        self.hc_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.norm_eps = args.rms_norm_eps
        mix_hc = (2 + hc) * hc
        hc_dim = hc * d
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim))
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim))
        self.hc_attn_base = mx.zeros((mix_hc,))
        self.hc_ffn_base = mx.zeros((mix_hc,))
        self.hc_attn_scale = mx.zeros((3,))
        self.hc_ffn_scale = mx.zeros((3,))

    def _hc_pre(self, x, fn, scale, base):
        b, s, hc, d = x.shape
        xf = x.reshape(b, s, hc * d).astype(mx.float32)
        mixes = (xf @ fn.T) * _rsqrt_norm(xf, self.norm_eps)
        pre, post, comb = hc_split_sinkhorn(mixes, scale, base, self.hc_mult, self.hc_iters, self.hc_eps)
        y = mx.sum(pre[..., None] * x.astype(mx.float32), axis=2)
        return y.astype(x.dtype), post, comb

    def _hc_post(self, x, residual, post, comb):
        term1 = post[..., None] * x[:, :, None, :]
        term2 = mx.sum(comb[..., None] * residual[:, :, :, None, :], axis=2)
        return (term1 + term2).astype(x.dtype)

    def __call__(self, x, input_ids, mask=None, cache=None):
        # The inter-layer activation is the FLAT 3D HC state [b, s, hc*d] (so the pipeline's
        # send/recv/all_gather see a standard 3D tensor like Hy3, not a 4D one). We unflatten
        # to [b, s, hc, d] only inside the layer; the math is unchanged.
        bsz, slen, hd = x.shape
        x = x.reshape(bsz, slen, self.hc_mult, hd // self.hc_mult)
        residual = x
        h, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.self_attn(self.attn_norm(h), mask, cache)
        x = self._hc_post(h, residual, post, comb)

        residual = x
        h, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.mlp(self.ffn_norm(h), input_ids)
        x = self._hc_post(h, residual, post, comb)
        return x.reshape(bsz, slen, hd)


class DeepseekV4Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hc_mult = args.hc_mult
        self.hc_eps = args.hc_eps
        self.norm_eps = args.rms_norm_eps
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [HCLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        hc = args.hc_mult
        self.hc_head_fn = mx.zeros((hc, hc * args.hidden_size))
        self.hc_head_base = mx.zeros((hc,))
        self.hc_head_scale = mx.zeros((1,))

    def _hc_head(self, x):
        b, s, hc, d = x.shape
        xf = x.reshape(b, s, hc * d).astype(mx.float32)
        mixes = (xf @ self.hc_head_fn.T) * _rsqrt_norm(xf, self.norm_eps)
        pre = mx.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        return mx.sum(pre[..., None] * x.astype(mx.float32), axis=2).astype(x.dtype)

    def __call__(self, input_ids, cache=None):
        h = self.embed_tokens(input_ids)               # [b,s,d]
        b, s, d = h.shape
        # Replicate into hc streams but keep them FLAT as [b, s, hc*d] across layers, so the
        # pipeline send/recv/all_gather move a standard 3D tensor (not the 4D HC state).
        h = mx.concatenate([h] * self.hc_mult, axis=-1)   # [b,s,hc*d], each d-block a copy
        if cache is None:
            cache = [None] * len(self.layers)
        mask = None  # the attention builds its own offset-aware _flash_mask; create_attention_mask
        # is unused (kept None to avoid threading a non-recv-derived tensor through every layer).
        for layer, c in zip(self.layers, cache):
            h = layer(h, input_ids, mask, c)            # 3D [b,s,hc*d] in/out
        h = self._hc_head(h.reshape(b, s, self.hc_mult, d))
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV4Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        return self.lm_head(self.model(inputs, cache))

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        # Flat checkpoint keys (no `model.` prefix) -> our MLX module tree.
        out = {}
        for k, v in weights.items():
            nk = k
            if k.startswith("embed."):
                nk = "model.embed_tokens." + k[len("embed."):]
            elif k.startswith("head."):
                nk = "lm_head." + k[len("head."):]
            elif k == "norm.weight":
                nk = "model.norm.weight"
            elif k.startswith("hc_head_"):
                nk = "model." + k
            elif k.startswith("layers."):
                rest = k[len("layers."):]
                ln, sub = rest.split(".", 1)
                if sub.startswith("attn."):
                    sub = "self_attn." + sub[len("attn."):]
                elif sub.startswith("ffn.experts.w1"):
                    sub = "mlp.switch_mlp.gate_proj" + sub[len("ffn.experts.w1"):]
                elif sub.startswith("ffn.experts.w3"):
                    sub = "mlp.switch_mlp.up_proj" + sub[len("ffn.experts.w3"):]
                elif sub.startswith("ffn.experts.w2"):
                    sub = "mlp.switch_mlp.down_proj" + sub[len("ffn.experts.w2"):]
                elif sub.startswith("ffn.shared_experts.w1"):
                    sub = "mlp.shared_experts.gate_proj" + sub[len("ffn.shared_experts.w1"):]
                elif sub.startswith("ffn.shared_experts.w3"):
                    sub = "mlp.shared_experts.up_proj" + sub[len("ffn.shared_experts.w3"):]
                elif sub.startswith("ffn.shared_experts.w2"):
                    sub = "mlp.shared_experts.down_proj" + sub[len("ffn.shared_experts.w2"):]
                elif sub.startswith("ffn.gate."):
                    sub = "mlp.gate." + sub[len("ffn.gate."):]
                # attn_norm / ffn_norm / hc_attn_* / hc_ffn_* pass through unchanged
                nk = f"model.layers.{ln}.{sub}"
            # drop any MTP/nextn keys (not built)
            if "nextn" in nk or "mtp" in nk.lower():
                continue
            out[nk] = v
        return out
