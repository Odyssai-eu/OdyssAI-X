# Port MLX de Qwen3.8-Flash-Next (HF model_type: qwen4_exp)
# Composants inedits vs qwen3_next : QSA sparse attention, gated residual
# (hyper-connections), n-gram / PLE embedding shardé, projections deltanet splittées.
#
# VENDORISÉ — source : PR ml-explore/mlx-lm#1788 (auteur eauchs), extraite du
# diff le 2026-08-26, ~1h après ouverture — NON REVIEWÉE upstream à cette date.
# Garder byte-identique au diff sauf ce bloc ET le drop mtp.* dans sanitize
# (delta marque "DELTA vs PR#1788" — la PR ne droppe pas la tete MTP du
# checkpoint officiel, load strict refuse). Si la PR merge avec des
# changements, re-vendoriser depuis la version mergée et re-verifier ce point.
# Cible : Qwen/Qwen3.8-Flash-Next (bf16 officiel). Le sanitize strippe le
# wrapper language_model., droppe la vision, transpose les conv1d.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import ArraysCache, KVCache, _BaseCache
from .gated_delta import gated_delta_update
from .switch_layers import SwitchGLU


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    layer_types: list = field(default_factory=list)
    full_attention_interval: int = 4
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    # gated deltanet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"
    # hyper-connections
    hc_count: int = 4
    hc_lowrank: int = 320
    # QSA
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    # n-gram / PLE
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    ple_embed_dim: int = 2560
    ple_layer_ids: list = field(default_factory=lambda: [2])
    ple_conv_kernel_size: int = 4
    seed: int = 0
    eos_token_id: Any = 248044
    partial_rotary_factor: float = 0.25
    rope_parameters: dict = field(default_factory=dict)
    rope_theta: float = 10_000_000.0
    tie_word_embeddings: bool = False


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp"
    text_config: dict = field(default_factory=dict)
    vision_config: dict = field(default_factory=dict)
    quantization: Any = None

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)
        rp = self.text.rope_parameters or {}
        self.text.rope_theta = float(rp.get("rope_theta", self.text.rope_theta))
        self.text.partial_rotary_factor = float(
            rp.get("partial_rotary_factor", self.text.partial_rotary_factor)
        )
        if not self.text.layer_types:
            n, k = self.text.num_hidden_layers, self.text.full_attention_interval
            self.text.layer_types = [
                "full_attention" if (i + 1) % k == 0 else "linear_attention"
                for i in range(n)
            ]


# --------------------------------------------------------------------------- norms


class RMSNorm(nn.Module):
    """RMSNorm, avec normalisation par groupes quand group_size est donné.

    Les hyper-connections normalisent chacun des hc_count flux séparément, d'où
    le reshape : un poids de taille hc_count*hidden, mais une statistique par flux.
    """

    # DELTA vs PR#1788 (4/4) — LE bug du charabia : les poids RMSNorm du
    # checkpoint qwen4_exp sont CENTRES SUR ZERO (reference equipe Qwen,
    # mlx-vlm PR#2028 Qwen4ExpRMSNorm : « checkpoint weights are centered at
    # zero », application y * (1.0 + weight)). La PR#1788 appliquait
    # y * weight -> activations multipliees par ~0 des la premiere couche.
    # La variante Gated (deltanet) reste CLASSIQUE, conforme a la reference.
    def __init__(self, dim: int, group_size: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros(dim)
        self.eps = eps
        self.group_size = group_size
        if group_size is not None and dim % group_size:
            raise ValueError(f"dim {dim} non divisible par group_size {group_size}")

    def __call__(self, x: mx.array) -> mx.array:
        if self.group_size is None:
            return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.group_size)
        x = mx.fast.rms_norm(x, None, self.eps).reshape(shape)
        return x * (1.0 + self.weight)


class RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, activation: str = "sigmoid"):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.activation = activation

    def __call__(self, x: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        out = mx.fast.rms_norm(x, self.weight, self.eps)
        if gate is None:
            return out.astype(x.dtype)
        act = mx.sigmoid if self.activation == "sigmoid" else nn.silu
        g = act(gate.astype(mx.float32))
        return (g * out.astype(mx.float32)).astype(x.dtype)


# ------------------------------------------------------------------- rope / helpers


def _rope_partial(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Applique le rope sur les `rotary_dim` premières dimensions seulement."""
    d = cos.shape[-1]
    # cos/sin sont calcules en float32 : sans ce cast ils promeuvent x et toute
    # l'attention repasse en float32.
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
    xr, xp = x[..., :d], x[..., d:]
    half = d // 2
    x1, x2 = xr[..., :half], xr[..., half:]
    rot = mx.concatenate([-x2, x1], axis=-1)
    xr = xr * cos + rot * sin
    return mx.concatenate([xr, xp], axis=-1) if xp.shape[-1] else xr


class RotaryEmbedding:
    def __init__(self, dim: int, base: float):
        self.dim = dim
        self.inv_freq = base ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)

    def __call__(self, positions: mx.array):
        # positions: (B, T) -> cos/sin (B, T, dim)
        freqs = positions.astype(mx.float32)[..., None] * self.inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


# ------------------------------------------------------------------------ QSA


class QSAIndexer(nn.Module):
    """Sélectionne, par requête, un budget de blocs de clés compressées.

    L'implémentation PyTorch de référence boucle sur (batch, query) ; ici tout est
    vectorisé : les clés poolées ne dépendent pas de la requête, donc on les
    calcule une fois puis on fait un top-k par ligne.
    """

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.kv_heads = args.indexer_kv_heads
        self.head_dim = args.indexer_head_dim
        self.token_budget = args.indexer_budget
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            args.hidden_size, (self.n_heads + self.kv_heads) * self.head_dim, bias=False
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def __call__(self, x, rope, cache, offset: int) -> Optional[mx.array]:
        B, S, _ = x.shape
        qk = self.index_qk_proj(x)
        split = self.n_heads * self.head_dim
        q = qk[..., :split].reshape(B, S, self.n_heads, self.head_dim)
        raw_k = qk[..., split:].reshape(B, S, self.head_dim)

        if cache is not None:
            raw_k = cache.update(raw_k)
        kv_len = raw_k.shape[1]

        # Sans sparsification possible : tous les tokens visibles tiennent dans le
        # budget, le top-k retiendrait tout. Le masque causal habituel suffit.
        if kv_len <= self.token_budget:
            return None

        n_blocks = kv_len // self.compress_ratio
        pooled = raw_k[:, : n_blocks * self.compress_ratio].reshape(
            B, n_blocks, self.compress_ratio, self.head_dim
        )
        pooled = self.k_layernorm(
            pooled.astype(mx.float32).mean(axis=2).astype(raw_k.dtype)
        )

        block_starts = mx.arange(n_blocks) * self.compress_ratio
        cos_k, sin_k = rope(block_starts[None, :])
        pooled = _rope_partial(pooled, cos_k, sin_k)

        q_pos = mx.arange(offset, offset + S)
        cos_q, sin_q = rope(q_pos[None, :])
        q = self.q_layernorm(q)
        q = _rope_partial(q, cos_q[:, :, None, :], sin_q[:, :, None, :])

        # scores: somme sur les têtes de relu(q.k), par bloc
        scores = mx.einsum(
            "bshd,bnd->bsnh", q.astype(mx.float32), pooled.astype(mx.float32)
        )
        scores = mx.maximum(scores, 0).sum(axis=-1) / math.sqrt(self.head_dim)

        # un bloc n'est candidat que s'il est entièrement dans le passé de la requête
        block_end = block_starts + self.compress_ratio - 1
        visible = block_end[None, None, :] <= q_pos[None, :, None]
        scores = mx.where(visible, scores, -mx.inf)

        k = min(self.block_topk, n_blocks)
        top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]  # (B, S, k)

        keep_block = mx.zeros((B, S, n_blocks + 1), dtype=mx.bool_)
        top = mx.where(mx.take_along_axis(visible, top, axis=-1), top, n_blocks)
        keep_block = mx.put_along_axis(keep_block, top, mx.array(True), axis=-1)[
            ..., :n_blocks
        ]

        # remap bloc -> tokens, puis la queue incomplète est toujours visible
        keep = mx.repeat(keep_block, self.compress_ratio, axis=-1)
        tail = kv_len - n_blocks * self.compress_ratio
        if tail:
            keep = mx.concatenate(
                [keep, mx.ones((B, S, tail), dtype=mx.bool_)], axis=-1
            )
        return keep[:, None]  # (B, 1, S, kv_len)


class Attention(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        d = args.hidden_size
        # q_proj porte aussi le gate de sortie : n_heads * head_dim * 2
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(args)

    def __call__(self, x, rope, mask, cache, idx_cache) -> mx.array:
        B, S, _ = x.shape
        offset = cache.offset if cache is not None else 0

        sparse = self.indexer(x, rope, idx_cache, offset)

        q, gate = mx.split(self.q_proj(x).reshape(B, S, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, S, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        cos, sin = rope(mx.arange(offset, offset + S)[None])
        cos, sin = cos[:, None], sin[:, None]
        q, k = _rope_partial(q, cos, sin), _rope_partial(k, cos, sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if sparse is not None:
            neg = mx.finfo(q.dtype).min if hasattr(mx, "finfo") else -1e9
            add = mx.where(sparse, mx.array(0, q.dtype), mx.array(neg, q.dtype))
            mask = (
                add
                if mask is None
                else (mask + add if not isinstance(mask, str) else add)
            )

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))


# ------------------------------------------------------------------- gated deltanet


class GatedDeltaNet(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_v = args.linear_num_value_heads
        self.n_k = args.linear_num_key_heads
        self.dk = args.linear_key_head_dim
        self.dv = args.linear_value_head_dim
        self.key_dim = self.dk * self.n_k
        self.value_dim = self.dv * self.n_v
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        d = args.hidden_size

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )
        # contrairement a qwen3-next, les projections sont splittees
        self.in_proj_qkv = nn.Linear(d, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(d, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(d, self.n_v, bias=False)
        self.in_proj_a = nn.Linear(d, self.n_v, bias=False)
        self.dt_bias = mx.ones(self.n_v)
        self.A_log = mx.zeros(self.n_v)
        self.norm = RMSNormGated(
            self.dv, eps=args.rms_norm_eps, activation=args.output_gate_type
        )
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)

    def __call__(self, x, mask, cache) -> mx.array:
        B, S, _ = x.shape
        mixed_qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, S, self.n_v, self.dv)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        conv_state = (
            cache[0]
            if (cache is not None and cache[0] is not None)
            else mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        )
        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(B, S, self.n_k, self.dk)
        k = k.reshape(B, S, self.n_k, self.dk)
        v = v.reshape(B, S, self.n_v, self.dv)

        inv_scale = self.dk**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        state = cache[1] if cache is not None else None
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        return self.out_proj(self.norm(out, z).reshape(B, S, -1))


# ------------------------------------------------------------------------- MoE


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(-logits, self.top_k - 1, axis=-1)[..., : self.top_k]
        w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
        out = (self.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        return out + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ------------------------------------------------------ hyper-connections (residual)


class GatedResidual(nn.Module):
    def __init__(self, args: TextArgs, use_combine: bool = True):
        super().__init__()
        self.hc = args.hc_count
        self.d = args.hidden_size
        hc_dim = self.hc * self.d
        self.hc_norm = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_dim, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_dim, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_dim, self.hc, bias=False) if use_combine else None
        )

    def __call__(self, hyper: mx.array):
        normed = self.hc_norm(hyper)
        w = nn.silu(self.input_mix_weight_down(normed) / self.hc)
        w = mx.sigmoid(self.input_mix_weight_up(w))
        w = w.reshape(*w.shape[:-1], self.hc, self.d)
        mixed = (w * normed.reshape(*normed.shape[:-1], self.hc, self.d)).mean(axis=-2)
        if self.block_inject_weight is None:
            return mixed
        inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.hc)
        return mixed, hyper, inject


# -------------------------------------------------------------- n-gram / PLE


_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1, _M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(v: int) -> int:
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


class NGramEmbedding(nn.Module):
    """Table de hachage n-gram, shardée en `split_ngram_parts` morceaux.

    ~51 Md de paramètres : on ne fait jamais de lookup dense. Les indices sont
    triés par shard côté hôte, comme dans l'implémentation llama.cpp.
    """

    def __init__(self, args: TextArgs, embed_dim: int, ple_layer_index: int = 0):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.eos_token_id = (
            args.eos_token_id[0]
            if isinstance(args.eos_token_id, list)
            else args.eos_token_id
        )
        head_dim = embed_dim // self.ngram_heads

        sizes, offsets, total = [], [], 0
        for h in range(self.ngram_heads):
            g = ple_layer_index * self.ngram_heads + h
            s = _nth_prime_after(args.ngram_vocab_size_base - 1, g + 1)
            sizes.append(s)
            offsets.append(total)
            total += s
        self.head_vocab_sizes = sizes

        div = args.make_ngram_vocab_size_divisible_by
        padded = math.ceil(total / div) * div
        self.n_shards = args.split_ngram_parts
        self.rows_per_shard = math.ceil(padded / self.n_shards)
        self.ngram_embedding = _ShardedEmbedding(
            self.n_shards, self.rows_per_shard, head_dim
        )

        # buffers repris tels quels depuis le checkpoint
        mults = []
        max_long = (1 << 63) - 1
        half = max(1, (max_long // max(args.vocab_size, 1)) // 2)
        base_seed = args.seed + _PRIME_1 * ple_layer_index
        for i in range(self.ngram_size):
            mults.append(
                2 * (_splitmix64((base_seed + _GAMMA * (i + 1)) & _MASK64) % half) + 1
            )
        # Attributs publics : uniquement pour absorber les tenseurs du checkpoint.
        # Ils sont dans parameters(), donc un astype(float16) les detruirait ; les
        # valeurs reellement utilisees vivent dans les copies prefixees `_`, hors
        # parameters() et reconstruites a l'identique depuis la config.
        self.layer_multipliers = mx.array(mults, dtype=mx.int64)
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)
        self._mults = mx.array(mults, dtype=mx.int64)
        self._sizes = mx.array(sizes, dtype=mx.int64)
        self._offsets = mx.array(offsets, dtype=mx.int64)

    def _shift_right(self, ids: mx.array, shift: int) -> mx.array:
        """Décale de `shift`, sans franchir une frontière d'EOS."""
        if shift == 0:
            return ids
        B, T = ids.shape
        pos = mx.arange(T)
        eos_pos = mx.where(ids == self.eos_token_id, pos, -1)
        prev_incl = mx.cummax(eos_pos, axis=1)
        prev = mx.concatenate(
            [mx.full((B, 1), -1, dtype=prev_incl.dtype), prev_incl[:, :-1]], axis=1
        )
        in_segment = pos[None] - (prev + 1)
        src = pos - shift
        gathered = mx.take_along_axis(
            ids, mx.broadcast_to(mx.maximum(src, 0)[None], (B, T)), axis=1
        )
        ok = (in_segment >= shift) & (src[None] >= 0)
        return mx.where(ok, gathered, self.eos_token_id)

    def __call__(self, ids: mx.array, prev_context: mx.array) -> mx.array:
        n_new = ids.shape[1]
        history = mx.concatenate([prev_context, ids], axis=1).astype(mx.int64)
        shifted = [self._shift_right(history, s) for s in range(self.ngram_size)]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            lo = (ngram - 2) * self.heads_per_ngram
            hi = lo + self.heads_per_ngram
            mixed = shifted[0] * self._mults[0]
            for p in range(1, ngram):
                mixed = mx.bitwise_xor(mixed, shifted[p] * self._mults[p])
            gid = mixed[..., None] % self._sizes[lo:hi].reshape(1, 1, -1)
            blocks.append(gid + self._offsets[lo:hi].reshape(1, 1, -1))

        gid = mx.concatenate(blocks, axis=-1)[:, -n_new:]
        return self.ngram_embedding(gid).reshape(*gid.shape[:2], -1)


class _ShardedEmbedding(nn.Module):
    """N tables d'embedding concaténées logiquement, adressées par index global."""

    def __init__(self, n_shards: int, rows: int, dim: int):
        super().__init__()
        self.n_shards = n_shards
        self.rows = rows
        self.dim = dim
        for i in range(n_shards):
            setattr(self, f"shard_{i}", nn.Embedding(rows, dim))

    def __call__(self, gid: mx.array) -> mx.array:
        flat = gid.reshape(-1)
        shard_of = flat // self.rows
        row_of = flat % self.rows

        # quels shards sont réellement touchés : décidé côté hôte, comme llama.cpp
        touched = np.unique(np.array(shard_of, copy=False))
        out = mx.zeros((flat.size, self.dim), dtype=mx.float32)
        for s in touched.tolist():
            sel = mx.array(np.nonzero(np.array(shard_of, copy=False) == s)[0])
            emb = getattr(self, f"shard_{s}")(mx.take(row_of, sel))
            out = mx.put_along_axis(out, sel[:, None], emb.astype(mx.float32), axis=0)
        return out.reshape(*gid.shape, self.dim)


class PLELayer(nn.Module):
    def __init__(self, args: TextArgs, ple_layer_index: int):
        super().__init__()
        self.d = args.hidden_size
        self.hc = args.hc_count
        hc_dim = self.d * self.hc
        self.ple_embedding = NGramEmbedding(args, args.ple_embed_dim, ple_layer_index)
        k = args.ple_conv_kernel_size
        self.dilation = args.ngram_size
        self.short_conv_state_len = (k - 1) * self.dilation
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_dim, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, self.d, bias=False)
        self.norm_key = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.norm_query = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.norm_conv = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            hc_dim,
            hc_dim,
            kernel_size=k,
            groups=hc_dim,
            dilation=self.dilation,
            bias=False,
        )

    def _short_conv(self, x: mx.array, cache) -> mx.array:
        S = x.shape[1]
        n = self.short_conv_state_len
        state = (
            cache[2]
            if (cache is not None and cache[2] is not None)
            else mx.zeros((x.shape[0], n, x.shape[-1]), dtype=x.dtype)
        )
        full = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(full[:, -n:, :])
        return nn.silu(self.conv1d(full[:, -(n + S) :, :]))

    def __call__(
        self, hidden: mx.array, ids: mx.array, prev_ctx: mx.array, cache
    ) -> mx.array:
        emb = self.ple_embedding(ids, prev_ctx).astype(hidden.dtype)
        key = self.norm_key(self.key_proj(emb))
        key = key.reshape(*key.shape[:-1], self.hc, self.d)
        value = self.value_proj(emb)
        query = self.norm_query(hidden)
        query = query.reshape(*query.shape[:-1], self.hc, self.d)

        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(self.d)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(*gated.shape[:-2], -1)
        return gated + self._short_conv(self.norm_conv(gated), cache)


# ------------------------------------------------------------------- decoder / model


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.mlp = SparseMoeBlock(args)
        ple_idx = (
            args.ple_layer_ids.index(layer_idx + 1)
            if (layer_idx + 1) in args.ple_layer_ids
            else None
        )
        self.ple = PLELayer(args, ple_idx) if ple_idx is not None else None
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    def __call__(self, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx):
        if self.ple is not None:
            h = h + self.ple(h, ids, prev_ctx, cache)

        x, hyper, inject = self.attn_hyper_connection(h)
        if self.layer_type == "linear_attention":
            x = self.linear_attn(x, conv_mask, cache)
        else:
            x = self.self_attn(x, rope, mask, cache, idx_cache)
        h = hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)

        x, hyper, inject = self.mlp_hyper_connection(h)
        x = self.mlp(x)
        return hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)


class Qwen4ExpModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.hc = args.hc_count
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        # pas de `norm` finale dans ce modèle : c'est ce mixer qui la porte
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope = RotaryEmbedding(rotary_dim, args.rope_theta)
        self.ple_layers = [
            i for i in range(args.num_hidden_layers) if (i + 1) in args.ple_layer_ids
        ]

    def __call__(self, ids: mx.array, cache=None, input_embeddings=None):
        h = self.embed_tokens(ids) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)

        full_idx = [
            i for i, l in enumerate(self.layers) if l.layer_type == "full_attention"
        ]
        attn_cache = cache[full_idx[0]] if full_idx else None
        mask = create_attention_mask(
            h, [attn_cache] if attn_cache is not None else None
        )
        conv_mask = None

        prev_ctx = None
        if self.ple_layers:
            ctx_len = self.args.ngram_size - 1
            eos = self.args.eos_token_id
            eos = eos[0] if isinstance(eos, list) else eos
            pc = cache[self.ple_layers[0]]
            prev = pc[3] if pc is not None else None
            prev_ctx = (
                prev
                if prev is not None
                else mx.full((ids.shape[0], ctx_len), eos, ids.dtype)
            )
            if pc is not None:
                tail = mx.concatenate([prev_ctx, ids], axis=1)[:, -ctx_len:]
                pc[3] = tail

        h = mx.tile(h, (1, 1, self.hc))
        for layer, c in zip(self.layers, cache):
            idx_c = c.indexer if (c is not None and hasattr(c, "indexer")) else None
            h = layer(h, self.rope, mask, conv_mask, c, idx_c, ids, prev_ctx)
        return self.hyper_connection_mixer(h)


class _IndexerCache(_BaseCache):
    """Garde les clés brutes de l'indexeur (une par token, non poolées)."""

    def __init__(self):
        self.keys = None

    def update(self, k: mx.array) -> mx.array:
        self.keys = k if self.keys is None else mx.concatenate([self.keys, k], axis=1)
        return self.keys

    @property
    def state(self):
        return self.keys

    @state.setter
    def state(self, v):
        self.keys = v


class _AttnCache(KVCache):
    def __init__(self):
        super().__init__()
        self.indexer = _IndexerCache()


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args.text)
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.text.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for i, t in enumerate(self.args.text.layer_types):
            if t == "full_attention":
                caches.append(_AttnCache())
            else:
                # 0: conv deltanet, 1: état ssm, 2: conv PLE, 3: contexte n-gram
                caches.append(ArraysCache(4))
        return caches

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.startswith("language_model."):
                k = k[len("language_model.") :]
            # DELTA vs PR#1788 (2/2) : le checkpoint officiel imbrique le
            # wrapper SOUS `model.` — `model.language_model.layers...` — que la
            # regle ci-dessus ne strippe pas. On normalise vers `model.layers...`.
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model.") :]
            if k.startswith("vision_tower.") or k.startswith("model.visual."):
                continue  # text-only pour l'instant
            # DELTA vs PR#1788 : le checkpoint officiel Qwen/Qwen3.8-Flash-Next
            # embarque une tete MTP (mtp.layers.0.* — indexer sparse inclus) que
            # la PR ne droppe pas ; le load strict la refuse. Meme traitement
            # que laguna.py / le dispatcher qwen3_5_moe : on la droppe.
            if k.startswith("mtp.") or k.startswith("model.mtp."):
                continue
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                # (C, 1, K) torch -> (C, K, 1) mlx
                if v.shape[1] == 1:
                    v = v.transpose(0, 2, 1)
            # DELTA vs PR#1788 (3/3) : le checkpoint officiel stocke les experts
            # FUSIONNES et stackes (`mlp.experts.gate_up_proj` [E, 2*mi, H],
            # `mlp.experts.down_proj` [E, H, mi]) ; SwitchGLU attend
            # switch_mlp.{gate,up,down}_proj.weight separes. Split sur dim -2,
            # gate = premiere moitie — convention Qwen (identique a qwen3_5,
            # validee sur les quants Q6h16 de la famille).
            if k.endswith(".mlp.experts.gate_up_proj"):
                base = k[: -len(".experts.gate_up_proj")]
                mid = v.shape[-2] // 2
                out[f"{base}.switch_mlp.gate_proj.weight"] = v[..., :mid, :]
                out[f"{base}.switch_mlp.up_proj.weight"] = v[..., mid:, :]
                continue
            if k.endswith(".mlp.experts.down_proj"):
                base = k[: -len(".experts.down_proj")]
                out[f"{base}.switch_mlp.down_proj.weight"] = v
                continue
            out[k] = v
        return out

    @property
    def quant_predicate(self):
        def fn(path, module, _):
            # seul le routeur MoE reste en pleine precision (les norms et conv1d
            # ne sont de toute facon jamais quantifies)
            return not path.endswith("mlp.gate")

        return fn
