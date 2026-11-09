from functools import partial
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from ..rope_utils import initialize_rope
from .config import TextConfig

ACTIVATIONS = {"silu": nn.silu, "gelu": nn.gelu, "gelu_pytorch_tanh": nn.gelu_approx}


@partial(mx.compile, shapeless=True)
def logit_softcap(softcap, mult, x):
    # T * tanh(logits * mult / T)  (Gemma-style tanh softcap, MuseGlimmer pre-scale)
    return mx.tanh(x * mult / softcap) * softcap


class RMSNormNoScale(nn.Module):
    """RMSNorm without learnable scale (embed_norm / qk_norm / perception_emb_norm)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACTIVATIONS[config.hidden_activation]

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    """Gated attention with unscaled QK-RMSNorm, per-layer theta (0 = NoPE on
    full-attention layers), sigmoid output gate before o_proj."""

    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"

        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.qk_scale = config.qk_scale_factor
        self.qk_norm = RMSNormNoScale(eps=config.rms_norm_eps)

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=config.attention_bias)
        self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)

        # theta == 0 → NoPE layer (full-attention layers receive no position embeddings)
        theta = config.layer_rope_theta[layer_idx]
        if theta:
            self.rope = initialize_rope(
                dims=self.head_dim,
                traditional=False,
                base=float(theta),
                scaling_config=config.rope_parameters,
                max_position_embeddings=config.max_position_embeddings,
            )
        else:
            self.rope = None

    def __call__(self, x: mx.array, mask: Optional[mx.array], cache: Optional[Any]) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.qk_norm(queries) * self.qk_scale
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        keys = self.qk_norm(keys)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        if self.rope is not None:
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        # headwise sigmoid gate computed from the layer input, before o_proj
        output = output * nn.sigmoid(self.gate_proj(x))
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        # Gemma-style sandwich: the two "post" norms run with a tighter eps (1e-8)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.post_norm_eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.post_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array], cache: Optional[Any]) -> mx.array:
        residual = x
        h = self.self_attn(self.input_layernorm(x), mask, cache)
        h = self.post_attention_layernorm(h)
        x = residual + h

        residual = x
        h = self.mlp(self.pre_feedforward_layernorm(x))
        h = self.post_feedforward_layernorm(h)
        return residual + h


class GlimmerTextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.window_size = config.sliding_window
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = RMSNormNoScale(eps=config.rms_norm_eps)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _make_masks(self, h, cache):
        mask = {}
        masks = []
        for l, c in zip(self.layers, cache):
            if l.layer_type not in mask:
                if l.layer_type == "full_attention":
                    mask["full_attention"] = create_attention_mask(h, c)
                else:
                    mask["sliding_attention"] = create_attention_mask(
                        h, c, window_size=self.window_size
                    )
            masks.append(mask[l.layer_type])
        return masks

    def __call__(
        self,
        inputs: mx.array = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        capture_layer_ids: Optional[List[int]] = None,
        hidden_sink: Optional[List[mx.array]] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            h = self.embed_norm(self.embed_tokens(inputs))
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)
        else:
            cache = cache + [None] * (len(self.layers) - len(cache))

        if mask is None:
            masks = self._make_masks(h, cache)
        else:
            masks = [mask] * len(self.layers)

        # DFlash/EAGLE3 drafters read the target's hidden states at fixed layers
        # (Muse-Glimmer assistant: 1, 13, 25, 37, 49), outputs before the final norm.
        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for idx, (layer, c, m) in enumerate(zip(self.layers, cache, masks)):
            h = layer(h, m, c)
            if hidden_sink is not None and idx in capture_set:
                hidden_sink.append(h)

        return self.norm(h)


class LanguageModel(nn.Module):
    # Sliding layers use RotatingKVCache: a batched rollback cannot drop a ragged
    # per-row tail, so accepts are clamped to the batch minimum (mlx-vlm #1962).
    requires_uniform_batch_acceptance = True

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = GlimmerTextModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.final_logit_softcapping = config.final_logit_softcapping
        self.output_multiplier = config.output_multiplier

    def logits_from_hidden(self, hidden: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden)
        else:
            logits = self.lm_head(hidden)
        if self.final_logit_softcapping is not None:
            logits = logit_softcap(
                self.final_logit_softcapping, self.output_multiplier, logits
            )
        return logits

    def __call__(
        self,
        inputs: mx.array = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        if inputs is None:
            inputs = kwargs.get("input_ids")
        capture_layer_ids = kwargs.pop("capture_layer_ids", None)
        hidden_sink = [] if capture_layer_ids is not None else None
        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
        )
        return LanguageModelOutput(
            logits=self.logits_from_hidden(out), hidden_states=hidden_sink
        )

    def chunked_prefill_policy(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        prompt_cache=None,
        draft_model=None,
        draft_kind=None,
        prefill_kwargs=None,
    ) -> bool:
        del input_ids, inputs_embeds, prompt_cache
        if draft_model is None:
            return True
        prefill_kwargs = prefill_kwargs or {}
        if draft_kind in ("dflash", "eagle3"):
            return prefill_kwargs.get("capture_layer_ids") is not None
        return False

    def rollback_speculative_cache(
        self,
        caches: List[Any],
        gdn_states: Any,
        accepted: Any,
        block_size: int,
    ) -> int:
        """Rewind the target KV caches to the accepted speculative prefix."""
        del gdn_states
        if isinstance(accepted, int):
            accepted = mx.array([accepted])
        if isinstance(accepted, (list, tuple)):
            accepted = mx.array(accepted, dtype=mx.int32)

        max_accepted = int(accepted.max().item())
        retained = max_accepted + 1
        trim = block_size - retained
        is_batch = accepted.size > 1
        valid_ends = accepted + 1

        for cache in caches:
            if cache is None:
                continue
            if trim > 0 and hasattr(cache, "trim"):
                cache.trim(trim)
            if (
                is_batch
                and hasattr(cache, "_idx")
                and cache.keys is not None
                and max_accepted > 0
            ):
                cache_length = cache._idx
                verify_start = cache_length - retained
                if any(
                    verify_start + int(v) < cache_length for v in valid_ends.tolist()
                ):
                    raise RuntimeError(
                        "Muse Glimmer batched speculative rollback requires uniform "
                        f"per-row acceptance; got ragged accepts {accepted.tolist()}."
                    )
        return max_accepted

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    def sanitize(self, weights):
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb" not in k
        }

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, m):
            if not hasattr(m, "to_quantized"):
                return False
            return path.startswith("model.layers.")

        return predicate

    def make_cache(self):
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.config.sliding_window, keep=0))
        return caches
