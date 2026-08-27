"""Shared low-level modules for the Inkling MLX port."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """Llama-style RMSNorm (compute in fp32, weight is a gain).

    Matches ``LlamaRMSNorm``: ``x_fp32 * rsqrt(mean(x^2) + eps) * weight``.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class ShortConvolution(nn.Module):
    """Depthwise causal 1-D convolution with a residual add, computed in fp32.

    Mirrors ``InklingShortConvolution``: a per-channel (groups == channels) causal
    conv1d of ``kernel_size`` taps, no bias, no activation, then ``out + input``.
    The reference keeps this module in fp32 regardless of the model dtype
    (``_keep_in_fp32_modules_strict``), so we upcast here too.

    Weight layout (MLX ``conv1d``): ``[channels, kernel_size, 1]``.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        # [C_out, K, C_in // groups] with groups == channels -> [C, K, 1]
        self.weight = mx.zeros((channels, kernel_size, 1))

    def __call__(self, x: mx.array, mask: mx.array | None = None, cache=None) -> mx.array:
        # x: [batch, seq, channels]
        in_dtype = x.dtype
        xf = x.astype(mx.float32)
        residual = xf
        if mask is not None:
            xf = xf * mask.astype(mx.float32)
        k = self.kernel_size
        B, seq, C = xf.shape
        w = self.weight.astype(mx.float32)
        if cache is not None:
            # left-context = cached last (k-1) inputs (zeros on the first call);
            # a "valid" conv over [left, xf] yields exactly `seq` causal outputs.
            left = cache.state if cache.state is not None else mx.zeros((B, k - 1, C), dtype=mx.float32)
            x_in = mx.concatenate([left, xf], axis=1)
            out = mx.conv1d(x_in, w, padding=0, groups=self.channels)
            cache.state = x_in[:, -(k - 1):, :]
        else:
            # causal: left-pad by (k-1), keep first `seq` outputs (== zero left-context)
            out = mx.conv1d(xf, w, padding=k - 1, groups=self.channels)[:, :seq, :]
        out = out + residual
        return out.astype(in_dtype)
