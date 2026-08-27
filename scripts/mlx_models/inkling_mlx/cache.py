"""Incremental caches for Inkling generation.

Two kinds of per-layer state must persist across decode steps:

* ``KVCache``   — the appended key/value tensors for attention.
* ``ConvCache`` — the last ``kernel-1`` inputs of each depthwise short-convolution
  (there are 4 per layer: k, v, post-attn, post-mlp).

A ``LayerCache`` bundles one KVCache + the 4 ConvCaches; ``make_cache`` builds one
per decoder layer. Absolute key positions are always ``arange(kv_len)`` because the
cache holds every key from position 0 (KVCache keeps the full history — correct for
both global and sliding layers, since the sliding-window constraint is enforced by
the attention mask).
"""

from __future__ import annotations

import mlx.core as mx


class ConvCache:
    """Holds the last ``kernel-1`` inputs of a short convolution."""

    __slots__ = ("state",)

    def __init__(self):
        self.state = None  # [B, kernel-1, C] or None


class KVCache:
    """Appends keys/values along the sequence axis (full history)."""

    __slots__ = ("keys", "values")

    def __init__(self):
        self.keys = None    # [B, heads, T, d]
        self.values = None

    @property
    def offset(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    def update(self, k: mx.array, v: mx.array):
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = mx.concatenate([self.keys, k], axis=2)
            self.values = mx.concatenate([self.values, v], axis=2)
        return self.keys, self.values


class LayerCache:
    __slots__ = ("kv", "k_conv", "v_conv", "attn_conv", "mlp_conv")

    def __init__(self):
        self.kv = KVCache()
        self.k_conv = ConvCache()
        self.v_conv = ConvCache()
        self.attn_conv = ConvCache()
        self.mlp_conv = ConvCache()


def make_cache(model) -> list[LayerCache]:
    """One LayerCache per text decoder layer."""
    n = len(model.model.llm.layers) if hasattr(model, "model") else len(model.layers)
    return [LayerCache() for _ in range(n)]
