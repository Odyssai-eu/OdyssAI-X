from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


class PatchEmbedder(nn.Module):
    """Linear patch projection + learned (side×side) position table, bilinearly
    resampled to each image's (h, w) patch grid (align_corners=False, zeros
    padding — same as F.grid_sample in the reference)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Linear(
            config.patch_temporal * config.in_channels * config.patch_size**2,
            config.hidden_size,
            bias=False,
        )
        self.position_embedding_table = nn.Embedding(
            config.pos_emb_height * config.pos_emb_width, config.hidden_size
        )
        self.side = config.pos_emb_height

    @staticmethod
    def _axis_taps(n: int, side: int):
        """Returns (taps, weights) of shape (n, 2) for one axis."""
        i = np.arange(n, dtype=np.float64)
        src = (i + 0.5) * side / n - 0.5
        floor = np.floor(src)
        raw = np.stack([floor, floor + 1.0], axis=-1)  # (n, 2)
        taps = np.clip(raw, 0, side - 1).astype(np.int64)
        frac = src - floor
        wgt = np.stack([1.0 - frac, frac], axis=-1)
        wgt = wgt * ((raw >= 0) & (raw <= side - 1))
        return taps, wgt

    def _pos_embeds(self, grid_thw: mx.array) -> mx.array:
        side = self.side
        rows = []
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            ht, hw = self._axis_taps(h, side)
            wt, ww = self._axis_taps(w, side)
            idx = (
                ht[:, None, :, None] * side + wt[None, :, None, :]
            )  # (h, w, 2, 2)
            wgt = hw[:, None, :, None] * ww[None, :, None, :]
            idx = mx.array(idx.reshape(h * w, 4))
            wgt = mx.array(wgt.reshape(h * w, 4), dtype=mx.float32)
            table = self.position_embedding_table.weight
            pos = (table[idx] * wgt[:, :, None]).sum(1)  # (h*w, hidden)
            rows.append(mx.tile(pos[None], (t, 1, 1)).reshape(t * h * w, -1))
        return mx.concatenate(rows, axis=0)

    def __call__(self, pixel_values: mx.array, grid_thw: mx.array) -> mx.array:
        embeddings = self.patch_embedding(pixel_values.astype(self.patch_embedding.weight.dtype))
        pos = self._pos_embeds(grid_thw)
        return embeddings + pos.astype(embeddings.dtype)


class VisionRotaryEmbedding:
    """Axial 2D rope: head_dim//4 frequencies per axis, layout [h|w|h|w]
    (duplicated halves — compatible with rotate_half). Plain class — inv_freq is
    derived, not a checkpoint parameter."""

    def __init__(self, config: VisionConfig):
        self.head_dim = config.head_dim
        spatial = self.head_dim // 2
        base = config.rope_parameters.get("rope_theta", 10000.0)
        self.inv_freq = 1.0 / (
            mx.power(mx.array(float(base)), mx.arange(0, spatial, 2, dtype=mx.float32) / spatial)
        )  # (spatial//2,)

    def __call__(self, pos_ids: mx.array) -> tuple:
        """pos_ids: (N, 2) — col 0 = w coords, col 1 = h coords (already swapped+1)."""
        freqs = pos_ids.astype(mx.float32)[:, :, None] * self.inv_freq  # (N, 2, F)
        f_h, f_w = freqs[:, 0], freqs[:, 1]  # (N, F) each
        ang = mx.concatenate([f_h, f_w], axis=-1)  # (N, head_dim//2)
        ang = mx.concatenate([ang, ang], axis=-1)  # (N, head_dim)
        return mx.cos(ang), mx.sin(ang)


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array, cu_seqlens, cos, sin) -> mx.array:
        N = x.shape[0]
        q = self.q_proj(x).reshape(N, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(N, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(N, self.num_heads, self.head_dim)

        cos = cos[:, None, :]  # (N, 1, D) broadcast sur les tetes
        sin = sin[:, None, :]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        q = q.transpose(1, 0, 2)[None]  # (1, H, N, D)
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]

        split_at = [int(i) for i in cu_seqlens[1:-1].tolist()]
        splits = [mx.split(t, split_at, axis=2) for t in (q, k, v)]
        outs = [
            mx.fast.scaled_dot_product_attention(qq, kk, vv, scale=self.scale, mask=None)
            for qq, kk, vv in zip(*splits)
        ]
        out = mx.concatenate(outs, axis=2)[0].transpose(1, 0, 2).reshape(N, -1)
        return self.proj(out)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def __call__(self, x, cu_seqlens, cos, sin) -> mx.array:
        x = x + self.attn(self.norm1(x), cu_seqlens, cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type != "muse_glimmer_vision":
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.patch_embedder = PatchEmbedder(config)
        self.rotary_emb = VisionRotaryEmbedding(config)
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [VisionBlock(config) for _ in range(config.num_hidden_layers)]
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.merge_size = config.merge_size
        self.patch_size = config.patch_size
        # window/positions run un-merged (spatial_merge 1); merge is deferred to pixel_shuffle
        self.window_size_patches = config.window_size // config.patch_size

    def _cu_seqlens(self, grid_thw: mx.array):
        seqlens = []
        for t, h, w in grid_thw.tolist():
            seqlens.extend([int(h) * int(w)] * int(t))
        cu = np.cumsum([0] + seqlens)
        return mx.array(cu, dtype=mx.int32)

    def _position_ids(self, grid_thw: mx.array) -> mx.array:
        """(N, 2), cols [w, h] + 1 — the reference builds [h, w] then flips the
        last axis (swapping the axes) and adds a +1 offset."""
        rows = []
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            hpos = mx.repeat(mx.arange(h)[:, None], w, axis=1).reshape(-1)  # row-major
            wpos = mx.tile(mx.arange(w), (h,))
            stacked = mx.stack([wpos, hpos], axis=-1) + 1  # flip → [w, h]
            rows.append(mx.tile(stacked[None], (t, 1, 1)).reshape(t * h * w, 2))
        return mx.concatenate(rows, axis=0)

    def _window_index(self, grid_thw: mx.array):
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        win = self.window_size_patches
        for grid_t, grid_h, grid_w in grid_thw.tolist():
            grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
            index = np.arange(grid_t * grid_h * grid_w).reshape(grid_t, grid_h, grid_w)
            pad_h = win - grid_h % win
            pad_w = win - grid_w % win
            num_h = (grid_h + pad_h) // win
            num_w = (grid_w + pad_w) // win
            padded = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
            padded = padded.reshape(grid_t, num_h, win, num_w, win)
            padded = padded.transpose(0, 1, 3, 2, 4).reshape(grid_t, num_h * num_w, win, win)
            seqlens = (padded != -100).sum(axis=(2, 3)).reshape(-1)
            padded = padded.reshape(-1)
            keep = np.where(padded != -100)[0]
            index_new = mx.array(padded[keep])
            window_index.append(index_new + window_index_id)
            cu_window_seqlens.extend(
                (np.cumsum(seqlens) + cu_window_seqlens[-1]).tolist()
            )
            window_index_id += grid_t * grid_h * grid_w
        window_index = mx.concatenate(window_index, axis=0)
        cu = np.unique(np.array(cu_window_seqlens))
        return window_index, mx.array(cu, dtype=mx.int32)

    @staticmethod
    def _pixel_shuffle_index(grid_thw: mx.array, merge: int) -> mx.array:
        indices = []
        offset = 0
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            perm = np.arange(h * w).reshape(h // merge, merge, w // merge, merge)
            perm = perm.transpose(0, 2, 1, 3).reshape(-1)
            if t > 1:
                offs = (np.arange(t) * h * w)[:, None]
                perm = (perm[None, :] + offs).reshape(-1)
            indices.append(mx.array(perm) + offset)
            offset += t * h * w
        return mx.concatenate(indices, axis=0)

    def pixel_shuffle(self, x: mx.array, grid_thw: mx.array) -> mx.array:
        f = self.merge_size
        dim = x.shape[-1]
        idx = self._pixel_shuffle_index(grid_thw, f)
        x = x[idx]
        return x.reshape(-1, f * f, dim).transpose(0, 2, 1).reshape(-1, dim * f * f)

    def __call__(
        self, pixel_values: mx.array, grid_thw: mx.array, output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        cu_seqlens = self._cu_seqlens(grid_thw)
        window_index, cu_window_seqlens = self._window_index(grid_thw)

        x = self.patch_embedder(pixel_values, grid_thw)
        x = self.ln_pre(x)
        x = x[window_index, :]

        pos = self._position_ids(grid_thw)[window_index, :]
        cos, sin = self.rotary_emb(pos)

        for i, block in enumerate(self.layers):
            cu = cu_seqlens if self.config.layer_types[i] == "full_attention" else cu_window_seqlens
            x = block(x, cu, cos, sin)

        x = x[mx.argsort(window_index), :]
        x = self.ln_post(x)
        return self.pixel_shuffle(x, grid_thw)

    def sanitize(self, weights):
        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}
