# Copyright © 2026 OdyssAI
#
# mlx_lm adapter for Inkling (Thinking Machines) — `model_type: inkling_mm_model`,
# checkpoint arch `InklingForConditionalGeneration` (natively multimodal MoE).
#
# TEXT-ONLY single-node v1. This is a THIN WRAPPER around pipenetwork's bundled
# `inkling_mlx` package (vendored side-by-side as `inkling_mlx/`), NOT a
# re-implementation: we reuse `inkling_mlx.text.TextModel` verbatim (its
# `embed_tokens` / `backbone` / `logits`). The runner loads model code via
# `mlx_lm.load_model`, which imports `mlx_lm.models.inkling_mm_model` and expects
# a `ModelArgs` + `Model` pair.
#
# Why text-only: the vision (HMLP) and audio (dMel) towers are dead weight for
# text serving AND drag in construction-time deps (scipy `linear_sum_assignment`
# for the vision patch-merge, PIL, transformers.audio_utils). We build ONLY the
# `model.llm.*` backbone and `sanitize` drops `model.visual.*`, `model.audio.*`
# and the inference-irrelevant `model.mtp.*` — so strict load_weights sees exactly
# the llm params. Multimodal is a later revision (build InnerModel, add scipy).
#
# Quantization: the checkpoint is standard mlx affine (weight U32 + scales + biases,
# g64/b8, recipe "uniform" → every quant target under model.llm.* has `.scales`).
# mlx_lm's default load_model class_predicate quantizes exactly the modules whose
# `.scales` are present in the (sanitized) weights, so NO custom predicate here.
#
# Position handling: inkling's backbone takes an explicit absolute `start_pos`
# (it does not read it from the cache). mlx_lm's generate calls `model(y, cache=)`
# WITHOUT start_pos, so we derive it from the per-layer KV cache offset — which,
# by construction (KVCache keeps full history from pos 0), equals the absolute
# position of the first token in this call. See inkling_mlx/generate.py.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .inkling_mlx.cache import LayerCache
from .inkling_mlx.config import InklingConfig
from .inkling_mlx.text import TextModel


class _MlxLmLayerCache(LayerCache):
    """inkling's LayerCache (KVCache + 4 short-conv ConvCaches) with the mlx_lm
    cache interface bolted on. Upstream inkling-mlx has NO mlx_lm integration
    (custom greedy_generate + custom cache), so `mlx_lm.generate` — which the
    runner uses — needs `.offset` / `.state` / `.is_trimmable()` / `.nbytes`
    (see mlx_lm/generate.py: `mx.eval([c.state for c in prompt_cache])`, the
    quantized-KV `c.offset` gate, and the memory accounting). We expose them
    over the composite (KV + 4 conv) state without forking the vendored source.

    `is_trimmable()` is False: the short-conv state has no simple truncation, so
    prompt-cache trimming / reuse is disabled (single-stream v1). Batched merge is
    likewise unsupported — the runner forces inkling to the single-stream loop."""

    # The 6 arrays that constitute this layer's full incremental state, in a
    # fixed order shared by the getter and setter so save/restore round-trips.
    def _arrays(self):
        return (
            self.kv.keys, self.kv.values,
            self.k_conv.state, self.v_conv.state,
            self.attn_conv.state, self.mlp_conv.state,
        )

    @property
    def offset(self) -> int:
        return self.kv.offset

    @property
    def state(self):
        return self._arrays()

    @state.setter
    def state(self, v):
        (self.kv.keys, self.kv.values,
         self.k_conv.state, self.v_conv.state,
         self.attn_conv.state, self.mlp_conv.state) = v

    def is_trimmable(self) -> bool:
        return False

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self._arrays() if a is not None)


@dataclass
class ModelArgs:
    # Keep the raw checkpoint config dict — InklingConfig.from_dict parses the
    # nested text_config/vision_config/audio_config layout itself.
    config_dict: Dict[str, Any]
    model_type: str = "inkling_mm_model"

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelArgs":
        return cls(
            config_dict=dict(params),
            model_type=params.get("model_type", "inkling_mm_model"),
        )


class _TextInner(nn.Module):
    """The `model.` level, text-only: holds just the `llm` backbone so parameter
    paths are `model.llm.*` — exactly the checkpoint's text keys."""

    def __init__(self, config: InklingConfig):
        super().__init__()
        self.llm = TextModel(config.text)


class Model(nn.Module):
    """mlx_lm-facing text-only Inkling. Exposes the hooks the runner/mlx_lm rely
    on: `__call__(inputs, cache=)`, `make_cache()`, `sanitize()`, `layers`."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.config = InklingConfig.from_dict(args.config_dict)
        self.model = _TextInner(self.config)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        # Absolute start position from the KV cache: empty on prefill (0), else the
        # count of tokens already processed == position of inputs[0]. Matches
        # inkling_mlx/generate.py's start_pos bookkeeping.
        start_pos = 0
        if cache is not None and len(cache) and cache[0] is not None:
            start_pos = cache[0].kv.offset
        llm = self.model.llm
        h = llm.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        h = llm.backbone(h, caches=cache, start_pos=start_pos)
        # Full-sequence logits [B, L, V]; the runner's generate slices [:, -1].
        return llm.logits(h)

    def make_cache(self):
        # One mlx_lm-compatible LayerCache (KVCache + 4 ConvCaches) per text
        # decoder layer. Same structure as inkling_mlx.cache.make_cache, but the
        # subclass carries the mlx_lm cache interface the runner's generate needs.
        return [_MlxLmLayerCache() for _ in range(len(self.model.llm.layers))]

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # Text-only: keep just the llm backbone. Drops model.visual.* / model.audio.*
        # (towers not built) and model.mtp.* (inference-irrelevant).
        return {k: v for k, v in weights.items() if k.startswith("model.llm.")}

    @property
    def layers(self):
        return self.model.llm.layers
