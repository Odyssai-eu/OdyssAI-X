# MLX text-only implementation of Mistral-Large-4-1T-A52B (bailing of consolidated FP8 ckpt).
#
# Spec: transformers `models/mistral_large4` (HF main, 29/10/2026). Differences vs mistral4.py:
#   - ssmax query scaling  q *= s[h] * log(pos + 1) + b[h]   (per head, per layer)
#   - LeRoPE: yarn inv_freq (factor 2048, orig 512, beta 32/1, theta 1e4) * exp(alpha)
#     folded at load time into `model.rotary.freqs` (mx.fast.rope "freqs" convention:
#     freqs = 1/inv_freq, interleaved/traditional=True).
#   - Per-layer no-rope (every 4th layer): qk_rope=0, qk_nope=128, kv_a emits 512 only.
#   - Cross-layer KV sharing: layers 39,43,...,71 read layer 35's latent cache
#     (fetch-only; Model.__call__ re-points their cache to the source layer's entry).
#   - lm_head softcapping 30 * tanh(logits / 30).
#   - DSA indexer, vision tower/projector and MTP (mtp_dspark) are dropped by sanitize
#     (text-only build, full attention: exact for seq <= index_topk = 2048).
# Pipeline (OdyssAI-X, 2026-10-30): 858 GB at Q6h16 needs Argo's 5 nodes, and
# tensor is out (64 KV heads, 5 does not divide it; 4 nodes is 214 GB a rank).
# The sharing layers 39..71 read layer 35's latent, and 35..71 (444 GB) does not
# fit one rank, so the latent crosses ranks: the rank that runs layer 35 sends
# its new-token latent along with h, every later rank appends it to a local
# mirror of cache[35] (same offset, same contents) before its own layers.
# Source is the native Mistral consolidated layout (layers.N.attention.wq + qscale_weight
# FP8 e4m3 with 128x128 F32 block scales); sanitize remaps, dequantizes and fuses.

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .deepseek_v3 import DeepseekV3MLP
from .mla import MultiLinear
from .mistral4 import mistral4_expert_select
from .pipeline import PipelineMixin
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 128
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    norm_topk_prob: bool = True
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    topk_group: int = 1
    num_experts_per_tok: int = 4
    first_k_dense_replace: int = 1
    n_group: int = 1
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 64
    v_head_dim: int = 128
    no_rope_layers: Optional[List[int]] = None
    kv_source_layers: Optional[List[int]] = None
    rope_parameters: Optional[Dict] = None
    head_dim: Optional[int] = None
    qk_head_dim: Optional[int] = None
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    final_logit_softcapping: float = 30.0

    def __post_init__(self, **kwargs):
        if self.no_rope_layers is None:
            self.no_rope_layers = [
                int((i + 1) % 4 != 0) for i in range(self.num_hidden_layers)
            ]
        if self.kv_source_layers is None:
            self.kv_source_layers = list(range(self.num_hidden_layers))
        if self.qk_head_dim is None:
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.head_dim is None:
            self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.rope_parameters is not None:
            self.rope_theta = self.rope_parameters.get("rope_theta", 10000.0)


class LeRoPE(nn.Module):
    """Interleaved RoPE with externally supplied per-pair frequencies.

    `freqs` follows mx.fast.rope's convention (freqs = 1/inv_freq). Sanitize
    stores yarn_interpolated * exp(-alpha) so that angle = pos / freqs equals
    transformers' pos * yarn_inv_freq * exp(alpha).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.freqs = mx.ones(dim // 2, dtype=mx.float32)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        return mx.fast.rope(
            x,
            self.dim,
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self.freqs,
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.layer_idx = layer_idx

        self.use_rope = bool(args.no_rope_layers[layer_idx])
        self.is_shared = args.kv_source_layers[layer_idx] != layer_idx
        # On no-rope layers the rope budget moves into the nope part.
        self.rope_dim = args.qk_rope_head_dim if self.use_rope else 0
        self.nope_dim = (
            args.qk_nope_head_dim
            if self.use_rope
            else args.qk_nope_head_dim + args.qk_rope_head_dim
        )
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.v_head_dim = args.v_head_dim

        self.scale = self.q_head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.q_head_dim, bias=False
        )
        if not self.is_shared:
            self.kv_a_proj_with_mqa = nn.Linear(
                self.hidden_size,
                self.kv_lora_rank + self.rope_dim,
                bias=args.attention_bias,
            )
            self.kv_a_layernorm = nn.RMSNorm(
                self.kv_lora_rank, eps=args.rms_norm_eps
            )
        # kv_b_proj absorbed: embed_q maps q_nope into latent space,
        # unembed_out maps the attended latent to v.
        self.embed_q = MultiLinear(self.nope_dim, self.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

        # ssmax per-head scaling (applied to the whole query, both halves).
        self.ssmax_scale = mx.zeros(self.num_heads)
        self.ssmax_bias = mx.zeros(self.num_heads)

    def __call__(
        self,
        x: mx.array,
        attn_factor: Optional[mx.array],
        rope: LeRoPE,
        offset: int,
        mask: Optional[Any],
        cache: Optional[Any],
    ) -> mx.array:
        B, L, D = x.shape

        q = self.q_proj(x)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_rope = mx.split(q, [self.nope_dim], axis=-1)

        if self.is_shared:
            # Fetch-only: the source layer already stored the latent KV.
            kv_latent, k_rope = cache.keys, cache.values
            if kv_latent.shape[2] > cache.offset:
                # Ring/growing caches may hold padding beyond the offset.
                kv_latent = kv_latent[:, :, : cache.offset, :]
                if k_rope is not None and k_rope.shape[2] > cache.offset:
                    k_rope = k_rope[:, :, : cache.offset, :]
        else:
            compressed_kv = self.kv_a_proj_with_mqa(x)
            k_latent, k_rt = mx.split(
                compressed_kv, [self.kv_lora_rank], axis=-1
            )
            kv_latent = mx.expand_dims(self.kv_a_layernorm(k_latent), axis=1)
            k_rope = (
                k_rt.reshape(B, L, 1, self.rope_dim).transpose(0, 2, 1, 3)
                if self.use_rope
                else None
            )
            if self.use_rope:
                k_rope = rope(k_rope, offset)
            if cache is not None:
                if self.use_rope:
                    kv_latent, k_rope = cache.update_and_fetch(kv_latent, k_rope)
                else:
                    dummy = mx.zeros((B, 1, L, 0), dtype=kv_latent.dtype)
                    kv_latent, _ = cache.update_and_fetch(kv_latent, dummy)
                k_rope = k_rope if self.use_rope else None

        if self.use_rope:
            q_rope = rope(q_rope, offset)

        # ssmax scales the entire query (nope and rope halves alike).
        if attn_factor is not None:
            q_nope = q_nope * attn_factor
            if self.use_rope:
                q_rope = q_rope * attn_factor

        if L == 1:
            # Decode: attend to the latent directly.
            q_abs = self.embed_q(q_nope)
            if self.use_rope:
                pe_scores = (q_rope * self.scale) @ k_rope.swapaxes(-1, -2)
                if mask is not None:
                    pe_scores = mx.where(
                        mask,
                        pe_scores,
                        mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
                    )
            else:
                pe_scores = None
            output = scaled_dot_product_attention(
                q_abs,
                kv_latent,
                kv_latent,
                cache=cache,
                scale=self.scale,
                mask=pe_scores,
            )
            output = self.unembed_out(output)
        else:
            k_nope = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)
            if self.use_rope:
                k_rope = mx.broadcast_to(
                    k_rope, [B, self.num_heads, k_rope.shape[2], self.rope_dim]
                )
                k = mx.concatenate([k_nope, k_rope], axis=-1)
                q_full = mx.concatenate([q_nope, q_rope], axis=-1)
            else:
                k = k_nope
                q_full = q_nope
            output = scaled_dot_product_attention(
                q_full, k, v, cache=cache, scale=self.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts
        )
        self.gate = nn.Linear(args.hidden_size, args.n_routed_experts, bias=False)
        self.shared_experts = DeepseekV3MLP(
            args, intermediate_size=args.moe_intermediate_size * args.n_shared_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = mistral4_expert_select(
            self.gate(x),
            self.num_experts_per_tok,
            self.args.n_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.norm_topk_prob,
        )
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        y = y + self.shared_experts(x)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = (
            MoE(args)
            if layer_idx >= args.first_k_dense_replace
            else DeepseekV3MLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        attn_factor: Optional[mx.array],
        rope: LeRoPE,
        offset: int,
        mask: Optional[Any],
        cache: Optional[Any],
    ) -> mx.array:
        r = self.self_attn(
            self.input_layernorm(x), attn_factor, rope, offset, mask, cache
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


def _env_layer_split(num_layers: int, size: int) -> Optional[List[int]]:
    """Layers per rank from the orchestrator's split env (same variables and
    precedence as runner.shard_pipeline): RUNNER_LAYER_BOUNDS, else
    RUNNER_RAM_WEIGHTS (proportional), else None (PipelineMixin's even split).
    Only the counts are taken: PipelineMixin gives rank 0 the LAST layers."""
    bounds = None
    raw = os.environ.get("RUNNER_LAYER_BOUNDS", "")
    if raw:
        b = [int(x) for x in raw.split(",")]
        if len(b) != size + 1 or b[0] != 0 or b[-1] != num_layers:
            raise ValueError(f"RUNNER_LAYER_BOUNDS={raw!r} invalid for size={size}, "
                             f"num_layers={num_layers}")
        bounds = b
    else:
        raw = os.environ.get("RUNNER_RAM_WEIGHTS", "")
        try:
            weights = [int(x) for x in raw.split(",")] if raw else []
        except ValueError:
            weights = []
        if len(weights) == size and all(w >= 0 for w in weights) and sum(weights) > 0:
            from auto_parallel import compute_proportional_bounds

            bounds = compute_proportional_bounds(num_layers, weights)
    if bounds is None:
        return None
    return [bounds[r + 1] - bounds[r] for r in range(size)]


class TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.rotary = LeRoPE(args.qk_rope_head_dim)

    def make_cache(self):
        from .cache import ConcatenateKVCache

        caches = [ConcatenateKVCache() for _ in self.layers]
        # Sharing layers reuse their source's cache entry (fetch-only). A
        # pipeline rank holds layers [0, end_idx) only (None below its start).
        for l, src in enumerate(self.args.kv_source_layers[: len(caches)]):
            if src != l:
                caches[l] = caches[src]
        return caches

    def pipeline(self, group, split=None):
        if split is None and group.size() > 1:
            split = _env_layer_split(len(self.layers), group.size())
        super().pipeline(group, split)

    def _carried_sources(self, before: int) -> List[int]:
        """KV source layers below `before` that a layer at or after `before`
        reads: their latent has to cross the rank boundary at `before`."""
        src = self.args.kv_source_layers
        return sorted({src[l] for l in range(before, self.args.num_hidden_layers)
                       if src[l] < before})

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(x)
        B, L, D = h.shape
        rank, size = self.pipeline_rank, self.pipeline_size
        start = self.start_idx
        end = start + len(self.pipeline_layers)
        local = range(start, end)
        src = self.args.kv_source_layers
        R = self.args.kv_lora_rank

        if cache is None:
            cache = [None] * len(self.layers)
        # Offset and mask from a layer this rank owns and writes itself: on a
        # pipeline rank cache[0] is never updated, and a sharing layer's entry
        # is a mirror updated below.
        ref = cache[next(l for l in local if src[l] == l)]
        offset = ref.offset if ref is not None else 0
        mask = create_attention_mask(h, ref)

        # Receive h, plus the new-token latent of every source layer computed
        # upstream that this rank or a later one reads.
        recv_src = self._carried_sources(start) if rank < size - 1 else []
        if rank < size - 1:
            if recv_src:
                got = mx.distributed.recv_like(
                    mx.zeros((B, L, D + R * len(recv_src)), dtype=h.dtype), rank + 1)
                h = got[..., :D]
                for i, s in enumerate(recv_src):
                    lat = got[..., D + i * R: D + (i + 1) * R]
                    if cache[s] is not None:
                        cache[s].update_and_fetch(
                            mx.expand_dims(lat, axis=1),
                            mx.zeros((B, 1, L, 0), dtype=h.dtype))
            else:
                h = mx.distributed.recv_like(h, rank + 1)

        L = h.shape[1]
        log_pos = mx.log(
            mx.arange(offset, offset + L, dtype=mx.float32) + 1.0
        )

        for l in local:
            layer = self.layers[l]
            attn = layer.self_attn
            if cache[l] is not None:
                f = attn.ssmax_scale.astype(mx.float32)[
                    None, :, None, None
                ] * log_pos[None, None, :, None] + attn.ssmax_bias.astype(
                    mx.float32
                )[None, :, None, None]
                attn_factor = f.astype(h.dtype)
            else:
                attn_factor = None
            c = cache[l]
            if c is not None and attn.is_shared:
                c = cache[src[l]]
                if c is None:
                    raise ValueError(
                        f"layer {l} shares KV with layer {src[l]} but its cache is None"
                    )
            h = layer(h, attn_factor, self.rotary, offset, mask, c)

        if rank != 0:
            send_src = self._carried_sources(end)
            if send_src and any(cache[s] is None for s in send_src):
                raise ValueError("pipeline KV sharing needs a cache on every rank")
            out = mx.concatenate(
                [h] + [cache[s].keys[:, 0, offset: offset + L, :].astype(h.dtype)
                       for s in send_src], axis=-1) if send_src else h
            h = mx.distributed.send(out, (rank - 1) % size)[..., :D]
            own = cache[next(l for l in reversed(local) if src[l] == l)]
            if own is not None:
                own.keys = mx.depends(own.keys, h)

        # Broadcast h while keeping it in the graph
        if size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = TextModel(args)
        self.softcap = args.final_logit_softcapping
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        if self.softcap:
            out = (mx.tanh(out.astype(mx.float32) / self.softcap) * self.softcap).astype(
                out.dtype
            )
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return self.model.make_cache()

    def quant_predicate(self, path, module):
        if path in (
            "model.embed_tokens",
            "lm_head",
            "model.norm",
        ) or path.endswith(
            (
                "input_layernorm",
                "post_attention_layernorm",
                "kv_a_layernorm",
                "gate",
                "embed_q",
                "unembed_out",
            )
        ):
            return False
        if ".ssmax_" in path or path.endswith("rotary"):
            return False
        if path.endswith("kv_a_proj_with_mqa"):
            # Sensitive shared-latent path (bf16 in the source); keep bf16.
            return False
        return True

    # ---- loading: consolidated -> module layout, fp8 dequant, kv_b fusion ----

    def _yarn_base_powers(self) -> mx.array:
        """Yarn-interpolated frequencies in mx.fast.rope convention (= 1/inv_freq).

        Same math as transformers _compute_yarn_parameters (truncate=True) and
        mlx_lm rope_utils.YarnRoPE.
        """
        p = self.args.rope_parameters or {}
        dims = self.args.qk_rope_head_dim
        base = p.get("rope_theta", 10000.0)
        factor = p.get("factor", 1.0)
        orig = p.get("original_max_position_embeddings", 8192)
        beta_fast = p.get("beta_fast", 32)
        beta_slow = p.get("beta_slow", 1)

        def corr_dim(rot):
            return (
                dims * math.log(orig / (rot * 2 * math.pi))
            ) / (2 * math.log(base))

        low = max(math.floor(corr_dim(beta_fast)), 0)
        high = min(math.ceil(corr_dim(beta_slow)), dims - 1)

        f = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        if factor == 1.0:
            return f
        ramp = mx.clip(
            (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low), 0, 1
        )
        mask = 1.0 - ramp  # 1 at high-frequency (low index) dims
        fi = factor * f
        return (fi * f) / (fi * mask + f * (1 - mask))

    def sanitize(self, weights):
        BS = 128

        def dequant(w, s):
            w = mx.from_fp8(w, dtype=mx.float32)
            if s.ndim == 0:
                return (w * s).astype(mx.bfloat16)
            m, n = w.shape
            pad_m, pad_n = (-m) % BS, (-n) % BS
            w = mx.pad(w, [(0, pad_m), (0, pad_n)])
            w = w.reshape((m + pad_m) // BS, BS, (n + pad_n) // BS, BS)
            w = (w * s[:, None, :, None].astype(mx.float32)).reshape(
                m + pad_m, n + pad_n
            )
            return w[:m, :n].astype(mx.bfloat16)

        # Pass 1: drop unused (vision / mtp / indexer / activation scales).
        drop_prefixes = (
            "vision_encoder",
            "patch_merger",
            "vision_language_adapter",
            "pre_mm_projector_norm",
            "mtp_dspark",
        )
        new = {}
        for k, v in weights.items():
            if k.startswith(drop_prefixes):
                continue
            if ".attention.indexer." in k:
                continue
            if k.endswith("activation_scale"):
                continue
            new[k] = v
        weights = new

        # Pass 2: rename + dequant fp8 pairs.
        ATT = {
            "attention.wq": "self_attn.q_proj",
            "attention.wo": "self_attn.o_proj",
            "attention.wkv_a_with_mqa": "self_attn.kv_a_proj_with_mqa",
            "attention.kv_a_norm": "self_attn.kv_a_layernorm",
            "attention.wkv_b": "self_attn.kv_b_proj",
            "attention.ssmax_scaling.s": "self_attn.ssmax_scale",
            "attention.ssmax_scaling.b": "self_attn.ssmax_bias",
            "attention_norm": "input_layernorm",
            "ffn_norm": "post_attention_layernorm",
            "feed_forward.w1": "mlp.gate_proj",
            "feed_forward.w3": "mlp.up_proj",
            "feed_forward.w2": "mlp.down_proj",
            "gate": "mlp.gate",
            "shared_experts.w1": "mlp.shared_experts.gate_proj",
            "shared_experts.w3": "mlp.shared_experts.up_proj",
            "shared_experts.w2": "mlp.shared_experts.down_proj",
        }
        W = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}

        def rename(k):
            if k == "tok_embeddings.weight":
                return "model.embed_tokens.weight"
            if k == "norm.weight":
                return "model.norm.weight"
            if k == "output.weight":
                return "lm_head.weight"
            if k == "lerope.alpha":
                return k
            if not k.startswith("layers."):
                return k
            parts = k.split(".")
            l = parts[1]
            rest = ".".join(parts[2:])
            scale = False
            if rest.endswith(".qscale_weight"):
                rest = rest[: -len(".qscale_weight")]
                scale = True
            has_weight = rest.endswith(".weight")
            body = rest[: -len(".weight")] if has_weight else rest
            if body in ATT:
                dst = ATT[body]
                if (has_weight or scale) and not dst.endswith(
                    ("ssmax_scale", "ssmax_bias")
                ):
                    dst += ".weight"
                return f"model.layers.{l}.{dst}" + ("._scale_inv" if scale else "")
            if body.startswith("experts."):
                ep = body.split(".")
                if len(ep) == 3 and ep[2] in W:
                    return (
                        f"model.layers.{l}.mlp.experts.{ep[1]}.{W[ep[2]]}.weight"
                        + ("._scale_inv" if scale else "")
                    )
            return None

        new = {}
        n_layers = self.args.num_hidden_layers
        for k, v in weights.items():
            if k == "lerope.alpha":
                new[k] = v
                continue
            rk = rename(k)
            if rk is None:
                continue
            # Skip layers beyond the configured depth (slice pilots) ...
            if rk.startswith("model.layers."):
                li = int(rk.split(".")[2])
                if li >= n_layers:
                    continue
            if rk.endswith("._scale_inv"):
                raw = k[: -len(".qscale_weight")] + ".weight"
                if raw not in weights:
                    continue  # ... or pairs split across shards
                wk = rk[: -len("._scale_inv")]
                new[wk] = dequant(weights[raw], v)
            elif rk not in new:
                new[rk] = v
        weights = new

        # Pass 3: LeRoPE folding.
        alpha = weights.pop("lerope.alpha", None)
        base_powers = self._yarn_base_powers()
        if alpha is not None:
            # transformers: inv_freq * exp(alpha)  <=>  freqs (1/inv) / exp(alpha)
            base_powers = base_powers / mx.exp(alpha.astype(mx.float32))
        weights["model.rotary.freqs"] = base_powers

        args = self.args
        # Pass 4: stack routed experts + fuse kv_b_proj.
        for l in range(args.num_hidden_layers):
            prefix = f"model.layers.{l}"

            for m in ("gate_proj", "up_proj", "down_proj"):
                first = f"{prefix}.mlp.experts.0.{m}.weight"
                if first in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{m}.weight")
                        for e in range(args.n_routed_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{m}.weight"] = mx.stack(
                        to_join
                    )

            attn = f"{prefix}.self_attn"
            if args.kv_source_layers[l] != l:
                # Shared layers have no kv_a module; drop stray keys.
                weights.pop(f"{attn}.kv_a_proj_with_mqa.weight", None)
                weights.pop(f"{attn}.kv_a_layernorm.weight", None)
            kb = weights.pop(f"{attn}.kv_b_proj.weight", None)
            if kb is not None:
                layer = self.model.layers[l].self_attn
                w = kb.reshape(
                    args.num_attention_heads,
                    layer.nope_dim + args.v_head_dim,
                    args.kv_lora_rank,
                )
                weights[f"{attn}.embed_q.weight"] = mx.contiguous(
                    w[:, : layer.nope_dim, :].swapaxes(-1, -2)
                )
                weights[f"{attn}.unembed_out.weight"] = mx.contiguous(
                    w[:, layer.nope_dim :, :]
                )

        return weights
