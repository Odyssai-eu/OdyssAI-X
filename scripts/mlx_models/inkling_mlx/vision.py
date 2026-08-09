"""Inkling vision tower: an HMLP (hierarchical MLP) patch encoder.

No attention — each layer folds space/time into the channel dim then projects
(Linear -> RMSNorm -> GELU), progressively growing channels up to the text hidden
size. Mirrors ``InklingVisionModel`` / ``InklingVisionEncoderLayer`` /
``plan_out_scales``. Checkpoint keys are flat: ``visual.layers.linear_{i}`` and
``visual.layers.norm_{i}`` plus ``visual.final_norm``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .common import RMSNorm
from .config import VisionConfig


def _prime_factors(n: int) -> list[int]:
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    p = 3
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 2
    if n > 1:
        factors.append(n)
    return factors


def plan_out_scales(temporal_patch_size: int, patch_size: int, n_layers: int, n_channels: int):
    """Port of the reference ``plan_out_scales`` (returns an ``(n_layers+1, 4)``
    array of (t, h, w, c) grid sizes). Uses numpy + scipy for the assignment."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    h = np.cumprod(np.array(_prime_factors(patch_size)[::-1]))
    t = np.cumprod(np.array(_prime_factors(temporal_patch_size)[::-1]))

    h_ch = np.ceil(h**2 * n_channels / 64).astype(np.int64) * 64
    t_ch = (np.ceil(h[-1] ** 2 * n_channels * t)).astype(np.int64) * 64

    base = np.array([[1, 1, 1, n_channels]], dtype=np.int64)
    spatial = np.stack([np.ones_like(h), h, h, h_ch], axis=1)
    temporal = np.stack([t, np.full_like(t, h[-1]), np.full_like(t, h[-1]), t_ch], axis=1)
    scales = np.concatenate([base, spatial, temporal], axis=0).astype(np.int64)

    size_reduction = np.prod(scales[:, :-1], axis=1).astype(np.float64)
    total_elements = patch_size * patch_size * temporal_patch_size * n_channels
    log_ideal = np.linspace(0.0, math.log(total_elements), n_layers + 1)
    cost = np.abs(log_ideal[:, None] - np.log(size_reduction)[None, :])

    if n_layers >= scales.shape[0]:
        idxs = np.argmin(cost, axis=1)
    else:
        _, idxs = linear_sum_assignment(cost)
    idxs = np.array(idxs)
    idxs[0] = 0
    idxs[-1] = scales.shape[0] - 1
    return scales[idxs]


def _fold_timespace_to_depth(x, t_fold, hw_fold):
    # x: [B, T, H, W, C] -> [B, T//t, H//hw, W//hw, C*t*hw*hw]
    B, T, H, W, C = x.shape
    t_new, h_new, w_new = T // t_fold, H // hw_fold, W // hw_fold
    x = x.reshape(B, t_new, t_fold, h_new, hw_fold, w_new, hw_fold, C)
    x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)
    x = x.reshape(B, t_new, h_new, w_new, t_fold * hw_fold * hw_fold * C)
    return x


class _VisionLayers(nn.Module):
    """Holds ``linear_{i}`` / ``norm_{i}`` to match checkpoint keys."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        scales = plan_out_scales(
            config.temporal_patch_size, config.patch_size, config.n_layers, config.num_channels
        )
        self.n_layers = config.n_layers
        self.folds = []  # (t_fold, hw_fold, add_norm)
        for i in range(config.n_layers):
            start, end = scales[i], scales[i + 1]
            shuffle = (
                (end[0] // start[0]) * (end[1] // start[1]) * (end[2] // start[2])
            )
            hw_fold = int(end[1] // start[1])
            t_fold = int(end[0] // start[0])
            in_dim = int(start[3]) * int(shuffle)
            add_norm = i != config.n_layers - 1
            out_dim = config.text_hidden_size if i == config.n_layers - 1 else int(end[3])
            setattr(self, f"linear_{i}", nn.Linear(in_dim, out_dim, bias=False))
            if add_norm:
                setattr(self, f"norm_{i}", RMSNorm(out_dim, eps=config.rms_norm_eps))
            self.folds.append((t_fold, hw_fold, add_norm))

    def __call__(self, x):
        for i, (t_fold, hw_fold, add_norm) in enumerate(self.folds):
            if hw_fold > 1 or t_fold > 1:
                x = _fold_timespace_to_depth(x, t_fold, hw_fold)
            x = getattr(self, f"linear_{i}")(x)
            if add_norm:
                x = getattr(self, f"norm_{i}")(x)
                x = nn.gelu(x)
        return x


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.layers = _VisionLayers(config)
        self.final_norm = RMSNorm(config.text_hidden_size, eps=config.rms_norm_eps)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        num_patches = pixel_values.shape[0]
        h = self.layers(pixel_values)
        h = self.final_norm(h)
        return h.reshape(num_patches, -1)
