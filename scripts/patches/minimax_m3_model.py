# Copyright © 2026 odyssai.eu
#
# MiniMax-M3 — text tower of minimax_m3_vl, vendored for mlx-lm (nuit du
# 2026-06-12, mission « casser le mur »). Reference math ported VERBATIM from
# transformers main: models/minimax_m3_vl/modeling_minimax_m3_vl.py
# (eager path: MSA sparse attention implemented as an additive block mask).
#
# Architecture (428B A23B):
#   - 60 layers; layers with layer_types[i] == "minimax_m3_sparse" carry the
#     MSA lightning indexer (57/60); the first 3 are full attention.
#   - GQA 64q/4kv, head_dim 128, per-head gemma-RMSNorm on q/k AFTER the head
#     reshape, partial NeoX RoPE (rotary_dim = head_dim/2 = 64, theta 5e6).
#   - MSA: 4 index heads × 128 dims vs ONE index key head; fp32 scores;
#     block max-pool (block 128) over keys then max over heads; +inf boost on
#     the query's own block (local_blocks=1); top-16 blocks; the selection is
#     expanded into an additive 0/-inf mask over the standard attention.
#   - MoE on mlp_layer_types[i] == "sparse" (57/60): 128 experts top-4,
#     sigmoid scoring + e_score_correction_bias (selection only), gathered
#     sigmoid weights normalised to sum 1, routed_scaling 2.0, plus shared
#     expert (DenseMLP, shared_intermediate 3072). Dense layers use
#     dense_intermediate 12288.
#   - Activation swigluoai (gpt-oss family, NON-interleaved):
#       gate = clamp(gate, max=limit); up = clamp(up, ±limit)
#       out  = (up + 1) * gate * sigmoid(alpha * gate)
#   - lm_head untied. Final gemma-RMSNorm.
#
# Checkpoint mapping (HF hub naming -> this module) happens in sanitize():
#   language_model. prefix stripped; vision tower / projector / patch_merge
#   dropped; block_sparse_moe.experts.N.{w1,w2,w3} stacked into
#   switch_mlp.{gate,down,up}_proj (the M2/Mixtral convention, confirmed in
#   mlx-lm minimax.py).

from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "minimax_m3"
    vocab_size: int = 200064
    hidden_size: int = 6144
    num_hidden_layers: int = 60
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5e6
    partial_rotary_factor: float = 0.5
    max_position_embeddings: int = 1048576
    tie_word_embeddings: bool = False
    # mlp
    dense_intermediate_size: int = 12288
    intermediate_size: int = 3072
    shared_intermediate_size: int = 3072
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    routed_scaling_factor: float = 2.0
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    # sparse attention (flattened from sparse_attention_config by the
    # converter; legacy nested dict also accepted below)
    index_n_heads: int = 4
    index_head_dim: int = 128
    index_block_size: int = 128
    index_topk_blocks: int = 16
    index_local_blocks: int = 1
    # per-layer dispatch; derived from the freq arrays when absent
    layer_types: Optional[list] = None
    mlp_layer_types: Optional[list] = None
    moe_layer_freq: Optional[list] = None
    sparse_attention_config: Optional[dict] = None

    def __post_init__(self):
        sc = self.sparse_attention_config or {}
        legacy = {
            "index_n_heads": "sparse_num_index_heads",
            "index_head_dim": "sparse_index_dim",
            "index_block_size": "sparse_block_size",
            "index_topk_blocks": "sparse_topk_blocks",
            "index_local_blocks": "sparse_local_block",
        }
        for flat, old in legacy.items():
            if old in sc:
                setattr(self, flat, sc[old])
        freq = sc.get("sparse_attention_freq")
        if self.layer_types is None:
            if freq is None:
                freq = [0, 0, 0] + [1] * (self.num_hidden_layers - 3)
            self.layer_types = [
                "minimax_m3_sparse" if f else "full_attention" for f in freq
            ]
        if self.mlp_layer_types is None:
            mfreq = self.moe_layer_freq
            if mfreq is None:
                mfreq = [0, 0, 0] + [1] * (self.num_hidden_layers - 3)
            self.mlp_layer_types = ["sparse" if f else "dense" for f in mfreq]


class GemmaRMSNorm(nn.Module):
    """fp32 RMS norm scaled by (1 + w) — the M3/Gemma convention."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class SwiGLUOAI:
    """gpt-oss style clamped swiglu, non-interleaved (SwitchGLU activation)."""

    def __init__(self, alpha: float, limit: float):
        self.alpha = alpha
        self.limit = limit

    def __call__(self, x_up, x_gate):
        gate = mx.clip(x_gate, a_min=None, a_max=self.limit)
        up = mx.clip(x_up, a_min=-self.limit, a_max=self.limit)
        glu = gate * mx.sigmoid(gate * self.alpha)
        return (up + 1.0) * glu


class DenseMLP(nn.Module):
    """Dense MLP with the same swigluoai recipe, separate gate/up weights
    (the checkpoint ships them separate; transformers fuses at load)."""

    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)
        self.alpha = args.swiglu_alpha
        self.limit = args.swiglu_limit

    def __call__(self, x):
        gate = mx.clip(self.gate_proj(x), a_min=None, a_max=self.limit)
        up = mx.clip(self.up_proj(x), a_min=-self.limit, a_max=self.limit)
        glu = gate * mx.sigmoid(gate * self.alpha)
        return self.down_proj((up + 1.0) * glu)


class MoEGate(nn.Module):
    """Sigmoid router with selection-only correction bias (dsv3 family)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.weight = mx.zeros((args.num_local_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.num_local_experts,))

    def __call__(self, x):
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        choice = scores + self.e_score_correction_bias.astype(mx.float32)
        k = self.top_k
        inds = mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(scores, inds, axis=-1)
        weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        return inds, weights


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = MoEGate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.intermediate_size,
            args.num_local_experts,
            activation=SwiGLUOAI(args.swiglu_alpha, args.swiglu_limit),
            bias=False,
        )
        self.shared_experts = DenseMLP(args, args.shared_intermediate_size)
        self.routed_scaling_factor = args.routed_scaling_factor

    def __call__(self, x):
        shared = self.shared_experts(x)
        inds, weights = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None].astype(y.dtype)).sum(axis=-2)
        return y * self.routed_scaling_factor + shared


class M3CacheLayer(KVCache):
    """KV cache + the indexer's key history (sparse layers only).

    `idx_keys` mirrors the reference SparseCacheLayer.update_index: plain
    append, no eviction — selection happens at score time, not storage time.
    """

    def __init__(self):
        super().__init__()
        self.idx_keys = None

    def update_index(self, idx_k):
        if self.idx_keys is None:
            self.idx_keys = idx_k
        else:
            self.idx_keys = mx.concatenate([self.idx_keys, idx_k], axis=-2)
        return self.idx_keys


class Indexer(nn.Module):
    """MSA lightning indexer — selection branch only (no value path)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.block_size = args.index_block_size
        self.topk_blocks = args.index_topk_blocks
        self.local_blocks = args.index_local_blocks
        self.rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope_theta = args.rope_theta
        self.index_q_proj = nn.Linear(
            args.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.index_k_proj = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.index_q_norm = GemmaRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def __call__(self, h, offset: int, cache: Optional[M3CacheLayer]):
        B, S, _ = h.shape
        q = self.index_q_proj(h).reshape(B, S, self.num_heads, self.head_dim)
        q = self.index_q_norm(q).transpose(0, 2, 1, 3)  # [B, Hi, S, D]
        k = self.index_k_proj(h).reshape(B, S, 1, self.head_dim)
        k = self.index_k_norm(k).transpose(0, 2, 1, 3)  # [B, 1, S, D]

        # Partial NeoX rope on the first rotary_dim dims — the reference
        # applies the MAIN rope's cos/sin (width 64) to the index heads.
        q = mx.fast.rope(
            q, self.rotary_dim, traditional=False,
            base=self.rope_theta, scale=1.0, offset=offset,
        )
        k = mx.fast.rope(
            k, self.rotary_dim, traditional=False,
            base=self.rope_theta, scale=1.0, offset=offset,
        )

        if cache is not None:
            k = cache.update_index(k)
        k_len = k.shape[2]

        num_blocks = (k_len + self.block_size - 1) // self.block_size
        pad = num_blocks * self.block_size - k_len

        scores = q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)
        # token-level causal mask: key position > query position -> -inf
        q_pos = offset + mx.arange(S)  # [S]
        k_pos = mx.arange(k_len)  # [k_len]
        future = k_pos[None, None, None, :] > q_pos[None, None, :, None]
        neg = mx.array(-mx.inf, dtype=mx.float32)
        scores = mx.where(future, neg, scores)
        if pad:
            scores = mx.pad(
                scores, [(0, 0), (0, 0), (0, 0), (0, pad)], constant_values=-mx.inf
            )
        scores = scores.reshape(B, self.num_heads, S, num_blocks, self.block_size)
        block_scores = scores.max(axis=-1).max(axis=1)  # [B, S, num_blocks]

        # local boost: the query's own block (and local_blocks-1 preceding)
        # always wins a slot.
        q_block = q_pos // self.block_size  # [S]
        local = mx.arange(self.local_blocks)
        local_idx = mx.maximum(
            q_block[None, :, None] - local[None, None, :], 0
        )  # [1, S, local]
        local_idx = mx.broadcast_to(local_idx, (B, S, self.local_blocks))
        block_scores = mx.put_along_axis(
            block_scores, local_idx, mx.array(mx.inf, dtype=mx.float32), axis=-1
        )

        topk = min(self.topk_blocks, num_blocks)
        neg_scores = -block_scores
        if topk < num_blocks:
            inds = mx.argpartition(neg_scores, kth=topk - 1, axis=-1)[..., :topk]
        else:
            inds = mx.broadcast_to(
                mx.arange(num_blocks)[None, None, :], (B, S, num_blocks)
            )
        vals = mx.take_along_axis(block_scores, inds, axis=-1)
        # invalid (future/empty) blocks keep -inf scores -> flag with -1
        inds = inds.astype(mx.int32)
        inds = mx.where(vals == -mx.inf, mx.array(-1, dtype=mx.int32), inds)
        return inds, num_blocks, k_len

    def build_block_mask(self, inds, num_blocks, k_len, offset, S, dtype):
        """Expand selected block indices into an additive [B,1,S,k_len] mask
        composed with the token-level causal mask."""
        B = inds.shape[0]
        safe = mx.where(inds < 0, num_blocks, inds)
        bias = mx.full((B, S, num_blocks + 1), False)
        bias = mx.put_along_axis(
            bias, safe, mx.array(True), axis=-1
        )[..., :num_blocks]  # [B, S, nb] keep flags
        keep = mx.repeat(bias, self.block_size, axis=-1)[..., :k_len]
        q_pos = offset + mx.arange(S)
        k_pos = mx.arange(k_len)
        causal = k_pos[None, None, :] <= q_pos[None, :, None]
        keep = keep & causal
        mask = mx.where(
            keep[:, None, :, :],
            mx.array(0.0, dtype=dtype),
            mx.array(mx.finfo(mx.float32).min, dtype=dtype),
        )
        return mask


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope_theta = args.rope_theta
        d = args.hidden_size
        self.q_proj = nn.Linear(d, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d, bias=False)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.is_sparse = args.layer_types[layer_idx] == "minimax_m3_sparse"
        self.indexer = Indexer(args) if self.is_sparse else None

    def __call__(self, x, mask, cache: Optional[M3CacheLayer]):
        B, S, _ = x.shape
        offset = cache.offset if cache is not None else 0

        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = mx.fast.rope(
            q, self.rotary_dim, traditional=False,
            base=self.rope_theta, scale=1.0, offset=offset,
        )
        k = mx.fast.rope(
            k, self.rotary_dim, traditional=False,
            base=self.rope_theta, scale=1.0, offset=offset,
        )

        sparse_mask = None
        if self.indexer is not None:
            inds, num_blocks, k_len_idx = self.indexer(x, offset, cache)
            sparse_mask = self.indexer.build_block_mask(
                inds, num_blocks, k_len_idx, offset, S, q.dtype
            )

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if sparse_mask is not None:
            attn_mask = sparse_mask
        else:
            attn_mask = mask

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=attn_mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        if args.mlp_layer_types[layer_idx] == "sparse":
            self.block_sparse_moe = SparseMoeBlock(args)
            self._moe = True
        else:
            self.mlp = DenseMLP(args, args.dense_intermediate_size)
            self._moe = False
        self.input_layernorm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask, cache):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        r = self.block_sparse_moe(self.post_attention_layernorm(h)) if self._moe \
            else self.mlp(self.post_attention_layernorm(h))
        return h + r


class MiniMaxM3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx
        self.pipeline_rank = 0
        self.pipeline_size = 1

    def pipeline(self, group):
        # Reverse split — rank 0 serves the LAST layers (dsv32 convention).
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        layers_per_rank = len(self.layers) // self.pipeline_size
        extra = len(self.layers) - layers_per_rank * self.pipeline_size
        if self.pipeline_rank < extra:
            layers_per_rank += 1
        self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * layers_per_rank
        self.end_idx = self.start_idx + layers_per_rank
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx
        self.num_layers = len(self.layers) - self.start_idx

    def __call__(self, x, cache=None):
        h = self.embed_tokens(x)

        if cache is None:
            cache = [None] * self.num_layers

        # Full-attention layers consume the standard causal mask; sparse
        # layers build their own. The mask must cover offset (cache) prefill.
        offset = 0
        for c in cache:
            if c is not None:
                offset = c.offset
                break
        S = h.shape[1]
        if S > 1:
            q_pos = offset + mx.arange(S)
            k_pos = mx.arange(offset + S)
            causal = k_pos[None, :] <= q_pos[:, None]
            mask = mx.where(
                causal,
                mx.array(0.0, dtype=h.dtype),
                mx.array(mx.finfo(mx.float32).min, dtype=h.dtype),
            )[None, None]
        else:
            mask = None

        if self.pipeline_rank < self.pipeline_size - 1:
            h = mx.distributed.recv_like(h, (self.pipeline_rank + 1))

        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[i])

        if self.pipeline_rank != 0:
            h = mx.distributed.send(h, (self.pipeline_rank - 1) % self.pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniMaxM3Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def make_cache(self):
        return [M3CacheLayer() for _ in range(self.model.num_layers)]

    def sanitize(self, weights):
        # 1) text tower only, prefix stripped
        out = {}
        for k, v in weights.items():
            if k.startswith(("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")):
                continue
            if k.startswith("language_model."):
                k = k[len("language_model."):]
            # indexer weights live flat under self_attn in the checkpoint
            # (self_attn.index_q_proj) but in a submodule here.
            k = k.replace(".self_attn.index_", ".self_attn.indexer.index_")
            k = k.replace(
                ".block_sparse_moe.e_score_correction_bias",
                ".block_sparse_moe.gate.e_score_correction_bias",
            )
            out[k] = v
        weights = out

        # 2) stack per-expert w1/w2/w3 into switch_mlp (M2/Mixtral mapping)
        if any("block_sparse_moe.experts.0.w1.weight" in k for k in weights):
            mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            n_experts = self.args.num_local_experts
            new = {}
            for k, v in weights.items():
                if ".block_sparse_moe.experts." in k:
                    continue
                new[k] = v
            for l in range(self.args.num_hidden_layers):
                prefix = f"model.layers.{l}.block_sparse_moe"
                if f"{prefix}.experts.0.w1.weight" not in weights:
                    continue
                for w, name in mapping.items():
                    stack = mx.stack(
                        [
                            weights[f"{prefix}.experts.{e}.{w}.weight"]
                            for e in range(n_experts)
                        ]
                    )
                    new[f"{prefix}.switch_mlp.{name}.weight"] = stack
            weights = new

        # 3) router weight name: checkpoint `gate.weight` IS our MoEGate.weight,
        #    e_score_correction_bias rides along — names already match.
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(path):
            return "e_score_correction_bias" not in path
        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, module, config):
            if path.endswith("block_sparse_moe.gate"):
                return False
            return True
        return predicate
