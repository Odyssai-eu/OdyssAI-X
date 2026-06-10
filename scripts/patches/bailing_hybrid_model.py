"""bailing_hybrid — Ring-2.5 / Ling-2.6 1T hybrid (MLA + GLA) for mlx-lm 0.31.3.

Vendored port (#43 WU-B). Upstream mlx-lm has no bailing_hybrid support
(ml-explore/mlx-lm#1233); the previous hand remap bailing_hybrid ->
bailing_moe_linear silently DROPPED the 10 MLA attention layers' weights
(fused-QKV module vs MLA weight names) — the model could never emit a sane
token. This module assembles the correct architecture from parts that
already exist in mlx-lm 0.31.3, verified against the converted checkpoint's
weight index and the official reference implementation
(inclusionAI/Ring-V2.5 models/modeling_bailing_moe_v2_5.py):

  * full-attention layers ((idx+1) % layer_group_size == 0): DeepSeek-style
    MLA with q-lora + absorbed projections — forward copied from
    mlx_lm.models.deepseek_v3.DeepseekV3Attention (same weight layout the
    homemade Q6 conversion produced: q_a_proj/q_a_layernorm/q_b_proj/
    kv_a_proj_with_mqa/kv_a_layernorm/embed_q/unembed_out), out proj named
    `dense` to match the checkpoint;
  * linear-attention layers: bailing_moe_linear.LinearAttention reused as-is
    (weight names query_key_value/g_proj/g_norm/query_layernorm/
    key_layernorm/dense match the checkpoint exactly);
  * MoE/MLP/gate: bailing_moe_linear blocks reused as-is;
  * attn_idx/gla_idx computed DYNAMICALLY from the layer list (kimi_linear
    pattern) so pipeline slicing cannot desynchronize them (the #43 crash
    class) — auto_parallel re-derives them per shard anyway.

Reference rope config (converted config.json): rope_theta=6e6,
qk_rope_head_dim=64 (= rotary_dim, partial_rotary_factor 0.5),
rope_interleave=True -> traditional rope, rope_scaling None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import bailing_moe_linear as _bml
from mlx_lm.models.base import (
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class ModelArgs(_bml.ModelArgs):
    # MLA additions over the bailing_moe_linear args (values from the Ring-2.5
    # converted config.json; BaseModelArgs.from_dict drops unknown keys).
    q_lora_rank: Optional[int] = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: Optional[int] = None
    rope_interleave: bool = True
    num_kv_heads_for_linear_attn: Optional[int] = None
    linear_silu: bool = False

    @classmethod
    def from_dict(cls, params):
        # A required-no-default parent field can't be re-defaulted in a
        # dataclass subclass (ordering violation). Some conversions
        # (inferencerlabs 3.7bit) omit norm_topk_prob — family convention
        # (DeepSeek-style noaux_tc sigmoid routing) normalizes top-k probs.
        params = dict(params)
        params.setdefault("norm_topk_prob", True)
        return super().from_dict(params)


class MLAAttention(nn.Module):
    """DeepSeek-style MLA (q-lora + absorbed embed_q/unembed_out).

    Forward logic mirrors mlx_lm.models.deepseek_v3.DeepseekV3Attention
    (mlx-lm 0.31.3) — the conversion produced weights in that exact layout.
    Only the output projection is named `dense` (checkpoint name) instead of
    `o_proj`.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim or args.head_dim or args.qk_nope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.q_lora_rank = args.q_lora_rank
        self.scale = self.q_head_dim**-0.5

        hidden = args.hidden_size
        bias = args.use_qkv_bias or args.use_bias

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                hidden, self.num_heads * self.q_head_dim, bias=bias
            )
        else:
            self.q_a_proj = nn.Linear(hidden, self.q_lora_rank, bias=bias)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.dense = nn.Linear(
            self.num_heads * self.v_head_dim, hidden, bias=args.use_bias
        )

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=bool(args.rope_interleave),
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)
            # QuantizedKVCache (runner kv_q8) returns (data, scales, biases)
            # tuples; the manual pe_scores matmul and the absorbed embed_q/
            # unembed_out projections need dense arrays — dequantize on fetch.
            # (The sdpa call below handles quantized tuples by itself, but the
            # MLA pe path does not.)
            if isinstance(kv_latent, (tuple, list)):
                kv_latent = mx.dequantize(
                    *kv_latent, group_size=cache.group_size, bits=cache.bits
                )
            if isinstance(k_pe, (tuple, list)):
                k_pe = mx.dequantize(
                    *k_pe, group_size=cache.group_size, bits=cache.bits
                )

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        # cache=None: sdpa dispatches the quantized path on hasattr(cache,
        # "bits") — we already dequantized k/v above, force the dense path.
        output = scaled_dot_product_attention(
            q_nope, k, v, cache=None, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


class DecoderLayer(nn.Module):
    """bailing_moe_linear.DecoderLayer with MLA on the full-attention layers."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_global = (
            (layer_idx + 1) % args.layer_group_size == 0
            or layer_idx
            >= args.num_hidden_layers // args.layer_group_size * args.layer_group_size
        )

        if self.is_global:
            self.attention = MLAAttention(args)
        else:
            self.attention = _bml.LinearAttention(args, layer_idx=layer_idx)

        self.mlp = (
            _bml.SparseMoeBlock(args)
            if (
                args.num_experts is not None and layer_idx >= args.first_k_dense_replace
            )
            else _bml.MLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        offset: int = 0,
    ) -> mx.array:
        if self.is_global:
            r = self.attention(self.input_layernorm(x), mask, cache)
        else:
            r = self.attention(self.input_layernorm(x), mask, cache, offset=offset)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LanguageModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Dynamic (kimi_linear pattern) — NOT the layer_group_size-1 hardcode
        # that made bailing_moe_linear unshardable (#43): pick the first layer
        # of each kind so a sliced layer list stays self-consistent.
        self.attn_idx = next(
            (i for i, l in enumerate(self.layers) if l.is_global), 0
        )
        self.gla_idx = next(
            (i for i, l in enumerate(self.layers) if not l.is_global), 0
        )

    def _live_indices(self) -> tuple[int, int]:
        # Recomputed from the CURRENT layer list on every forward: pipeline
        # slicing replaces self.layers (and wraps the ends in layers that
        # delegate attribute reads), so init-time indices go stale — the exact
        # #43 crash class. Scanning <=80 flags per call is negligible.
        attn = next(
            (i for i, l in enumerate(self.layers) if getattr(l, "is_global", False)),
            None,
        )
        gla = next(
            (
                i
                for i, l in enumerate(self.layers)
                if not getattr(l, "is_global", True)
            ),
            None,
        )
        if attn is None or gla is None:
            missing = "full-attention" if attn is None else "linear-attention"
            raise ValueError(
                f"bailing_hybrid shard has no {missing} layer — re-cut the "
                "pipeline so every shard spans at least one full layer group."
            )
        return attn, gla

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.word_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        attn_idx, gla_idx = self._live_indices()
        offset = 0
        # return_array=True: the MLA path applies the mask via mx.where on the
        # pe_scores (deepseek_v3:342 / kimi_linear:449 do the same) — the
        # "causal" string fast-path would crash it on prefill.
        attn_mask = create_attention_mask(h, cache[attn_idx], return_array=True)
        gla_mask = create_ssm_mask(h, cache[gla_idx])
        if cache[attn_idx] is not None:
            offset = cache[attn_idx].offset

        for layer, c in zip(self.layers, cache):
            if getattr(layer, "is_global", False):
                h = layer(h, attn_mask, c)
            else:
                h = layer(h, gla_mask, c, offset=offset)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.norm_head = args.norm_head
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.word_embeddings.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    # Same checkpoint conventions as bailing_moe_linear (the conversion uses
    # switch_mlp/gate.gate_proj naming already, but keep the expert-stacking
    # path for raw HF-converted checkpoints).
    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        if self.norm_head:
            w = weights["lm_head.weight"]
            dtype = w.dtype
            weight_norm = (
                mx.linalg.norm(w.astype(mx.float32), axis=0, keepdims=True) + 1e-7
            )
            weights["lm_head.weight"] = (w / weight_norm).astype(dtype)

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            if l >= self.args.first_k_dense_replace:
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                            to_join = [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                                for e in range(self.args.num_experts)
                            ]
                            weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(
                                to_join
                            )
                if f"{prefix}.mlp.gate.weight" in weights:
                    weights[f"{prefix}.mlp.gate.gate_proj.weight"] = weights.pop(
                        f"{prefix}.mlp.gate.weight"
                    )
                if f"{prefix}.mlp.gate.bias" in weights:
                    weights[f"{prefix}.mlp.gate.gate_proj.bias"] = weights.pop(
                        f"{prefix}.mlp.gate.bias"
                    )

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate.gate_proj"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for l in self.layers:
            if l.is_global:
                caches.append(KVCache())
            else:
                caches.append(ArraysCache(size=1))
        return caches
