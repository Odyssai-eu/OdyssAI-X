"""Inkling decoder layer: attention + MLP, each wrapped by a pre-norm and a
trailing short-convolution, with residual adds. Mirrors ``InklingDecoderLayer``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention
from .common import RMSNorm, ShortConvolution
from .config import TextConfig
from .moe import DenseMLP, MoE


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.attn = Attention(config, layer_idx)
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MoE(config)
        else:
            self.mlp = DenseMLP(config)
        self.attn_sconv = ShortConvolution(config.hidden_size, config.sconv_kernel_size)
        self.mlp_sconv = ShortConvolution(config.hidden_size, config.sconv_kernel_size)

    def __call__(self, x, start_pos=0, cache=None, conv_mask=None):
        kv = cache.kv if cache is not None else None
        residual = x
        h = self.attn_norm(x)
        h = self.attn(
            h, start_pos=start_pos, kv_cache=kv,
            k_conv=cache.k_conv if cache is not None else None,
            v_conv=cache.v_conv if cache is not None else None,
            conv_mask=conv_mask,
        )
        h = self.attn_sconv(h, mask=conv_mask, cache=cache.attn_conv if cache is not None else None)
        x = residual + h

        residual = x
        h = self.mlp_norm(x)
        h = self.mlp(h)
        h = self.mlp_sconv(h, mask=conv_mask, cache=cache.mlp_conv if cache is not None else None)
        x = residual + h
        return x
