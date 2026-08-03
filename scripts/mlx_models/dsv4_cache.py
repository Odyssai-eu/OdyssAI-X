"""PoolingCache extrait du fork ivan-mlx-lm (branche ds4) pour DeepSeek V4.

Stock mlx-lm 0.31.3 ne l embarque pas; le reste de cache.py est identique
au stock, donc on n importe que la base privee.
"""
import mlx.core as mx

from .cache import _BaseCache


class PoolingCache(_BaseCache):
    """Cache for pooled (compressed) KV tokens with a remainder buffer.

    Stores two things:
      1. A growing pool of compressed tokens (step-allocated).
      2. A small remainder buffer of tokens not yet forming a full window.
    """

    def __init__(self, ratio: int):
        self.ratio = ratio

        self.buf_kv = None
        self.buf_gate = None
        self.remainder = 0

        self.pooled = None

    @property
    def offset(self):
        return 0 if self.pooled is None else self.pooled.shape[1]

    def accumulate_windows(self, kv: mx.array, gate: mx.array, offset):
        B, L, D1 = kv.shape
        _, _, D2 = gate.shape

        if self.buf_kv is None:
            self.buf_kv = mx.zeros((B, self.ratio, D1), dtype=kv.dtype)
            self.buf_gate = mx.zeros((B, self.ratio, D2), dtype=gate.dtype)

        # Prompt mode
        if L > 1:
            total = L + self.remainder
            usable = (total // self.ratio) * self.ratio
            new_remainder = total % self.ratio

            if usable > 0:
                r_kv = mx.concatenate(
                    [
                        self.buf_kv[:, : self.remainder],
                        kv[:, : (usable - self.remainder)],
                    ],
                    axis=1,
                )
                r_gate = mx.concatenate(
                    [
                        self.buf_gate[:, : self.remainder],
                        gate[:, : (usable - self.remainder)],
                    ],
                    axis=1,
                )
                r_base = offset - self.remainder
                self.remainder = 0
            else:
                r_kv = mx.zeros((B, 0, D1), dtype=kv.dtype)
                r_gate = mx.zeros((B, 0, D2), dtype=gate.dtype)
                r_base = 0

            if new_remainder > 0:
                self.buf_kv[:, self.remainder : new_remainder] = kv[:, -new_remainder:]
                self.buf_gate[:, self.remainder : new_remainder] = gate[
                    :, -new_remainder:
                ]
            self.remainder = new_remainder

            return r_kv, r_gate, r_base

        # Decode mode
        else:
            self.buf_kv[:, self.remainder : self.remainder + 1] = kv
            self.buf_gate[:, self.remainder : self.remainder + 1] = gate
            self.remainder = (self.remainder + 1) % self.ratio

            if self.remainder == 0:
                r_kv = self.buf_kv
                r_gate = self.buf_gate
                r_base = offset - self.ratio + 1
            else:
                r_kv = mx.zeros((B, 0, D1), dtype=kv.dtype)
                r_gate = mx.zeros((B, 0, D2), dtype=gate.dtype)
                r_base = 0

            return r_kv, r_gate, r_base

    def update_and_fetch(self, px: mx.array):
        if px.shape[1] == 0:
            if self.pooled is None:
                return mx.zeros((px.shape[0], 0, px.shape[-1]), dtype=px.dtype)
            return self.pooled

        if self.pooled is None:
            self.pooled = px
        else:
            self.pooled = mx.concatenate([self.pooled, px], axis=1)
        return self.pooled

    def make_mask(self, L: int = 1, offset: int = 0):
        """Build a causal validity mask for pooled positions.

        Query at absolute position ``offset + j`` can attend to pooled token
        ``i`` iff ``i < (offset + j) // ratio``.

        Returns ``(N, P)`` bool mask, or ``None`` when every pooled position
        is visible to every query (common during decode).
        """
        if self.pooled is None or L == 1:
            return None

        pool_idx = mx.arange(self.pooled.shape[1])
        query_idx = mx.arange(offset + 1, offset + L + 1)
        return pool_idx < query_idx[:, None] // self.ratio

    @property
    def state(self):
        buf_kv = self.buf_kv[:, : self.remainder] if self.remainder > 0 else None
        buf_gate = self.buf_gate[:, : self.remainder] if self.remainder > 0 else None
        return (buf_kv, buf_gate, self.pooled)

    @state.setter
    def state(self, v):
        buf_kv, buf_gate, pooled = v
        self.remainder = 0
        self.buf_kv = self.buf_gate = None
        if buf_kv is not None:
            self.accumulate_windows(buf_kv, buf_gate, 0)
        self.pooled = pooled

    @property
    def meta_state(self):
        return self.ratio

    @meta_state.setter
    def meta_state(self, v):
        self.ratio = v

    def is_trimmable(self):
        return self.pooled is None

    def trim(self, n):
        n = min(self.remainder, n)
        self.remainder -= n
        return n

    def size(self):
        return 0 if self.pooled is None else self.pooled.shape[1]

    def empty(self):
        return self.pooled is None and self.remainder == 0

    @property
    def nbytes(self):
        total = 0
        if self.buf_kv is not None:
            total += self.buf_kv.nbytes + self.buf_gate.nbytes
        if self.pooled is not None:
            total += self.pooled.nbytes
        return total

    @classmethod
    def merge(cls, caches):
        return BatchPoolingCache.merge(caches)


