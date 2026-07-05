"""Unit tests for the native-MTP speculative loop (plan §E1.3).

Self-contained (no pytest): `python3 test_mtp_spec.py` exits 0 with a PASS
line per test, non-zero on the first failure. Needs mlx + mlx_lm only —
runs on any cluster node venv, no cluster, no big model.

The toy trunk uses REAL single-head attention over mlx_lm KVCache so the
trim/offset mechanics under test are the production ones, not mocks. The
toy MTP block reuses the REAL NativeMTPModule class with a tiny layer.

The load-bearing property (test 1): for ANY depth D, greedy native-MTP
emits the EXACT same token sequence as plain AR greedy — per position, not
just a final hash (review finding F-31).
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_module import NativeMTPModule            # noqa: E402
from mtp_spec import native_mtp_stream_generate   # noqa: E402

V, H, LAYERS = 97, 32, 2
mx.random.seed(7)


class TinyAttnLayer(nn.Module):
    """Single-head causal attention + MLP, driven by a real KVCache."""

    def __init__(self, dims: int):
        super().__init__()
        self.wq = nn.Linear(dims, dims, bias=False)
        self.wk = nn.Linear(dims, dims, bias=False)
        self.wv = nn.Linear(dims, dims, bias=False)
        self.wo = nn.Linear(dims, dims, bias=False)
        self.mlp = nn.Sequential(nn.Linear(dims, dims * 2), nn.GELU(),
                                 nn.Linear(dims * 2, dims))
        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)

    def __call__(self, x, mask=None, cache=None):
        y = self.norm1(x)
        B, S, D = y.shape
        q = self.wq(y)[:, None]                  # [B, 1 head, S, D]
        k = self.wk(y)[:, None]
        v = self.wv(y)[:, None]
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=D ** -0.5, mask=mask)
        x = x + self.wo(out[:, 0])
        return x + self.mlp(self.norm2(x))


class TinyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, H)
        self.layers = [TinyAttnLayer(H) for _ in range(LAYERS)]
        self.norm = nn.RMSNorm(H)

    def __call__(self, x, cache=None):
        from mlx_lm.models.base import create_attention_mask
        h = self.embed_tokens(x)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0], return_array=True)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


class TinyTrunk(nn.Module):
    """Duck-types the mlx-lm Model surface the loop touches."""

    def __init__(self):
        super().__init__()
        self.model = TinyInner()
        self.lm_head = nn.Linear(H, V, bias=False)

    def __call__(self, x, cache=None):
        return self.lm_head(self.model(x, cache))

    def make_cache(self):
        from mlx_lm.models.cache import KVCache
        return [KVCache() for _ in self.model.layers]


class TinyMTPLayer(nn.Module):
    """layer_cls stand-in for NativeMTPModule: (args, layer_idx) ctor."""

    def __init__(self, args, layer_idx: int):
        super().__init__()
        self.attn = TinyAttnLayer(int(args.hidden_size))

    def __call__(self, x, mask=None, cache=None):
        return self.attn(x, mask=mask, cache=cache)


class TinyArgs:
    hidden_size = H
    rms_norm_eps = 1e-5


def make_trunk_cache(trunk: TinyTrunk):
    from mlx_lm.models.cache import KVCache
    return [KVCache() for _ in trunk.model.layers]


def make_mtp(trunk: TinyTrunk) -> NativeMTPModule:
    mtp = NativeMTPModule(TinyArgs(), TinyMTPLayer, layer_idx=LAYERS)
    mtp.bind_trunk(trunk)

    # The tiny MTP block is a single TinyAttnLayer, not a CacheList consumer:
    # give it a plain KVCache and index-compatible wrapper.
    from mlx_lm.models.cache import KVCache

    def _factory():
        return [KVCache()]
    mtp._cache_factory = _factory
    mx.eval(mtp.parameters())
    return mtp


class OracleMTP:
    """Draft oracle: runs the SAME trunk weights over the same token stream,
    so its drafts equal the AR continuation → 100% acceptance. Exercises the
    n=D full-accept path (emit D+1/round, zero trunk trim, seed at index D).

    The mtp interface receives the stream shifted by one (pairs are
    (token_{p+1}, hidden_p)) — the oracle re-prepends token 0 on its first
    call so its internal cache matches the trunk's exactly.
    `corrupt_every=k` deterministically breaks every k-th draft step
    (partial-acceptance path: trims 0<D-n<D)."""

    def __init__(self, trunk: TinyTrunk, first_token: int, corrupt_every: int = 0):
        self.trunk = trunk
        self.first = first_token
        self.started = False
        self.corrupt_every = corrupt_every
        self.calls = 0

    def make_cache(self):
        return make_trunk_cache(self.trunk)

    def draft_step(self, token_ids, prev_hidden, cache):
        toks = token_ids
        if not self.started:
            toks = mx.concatenate(
                [mx.array([[self.first]], dtype=mx.uint32), toks], axis=1)
            self.started = True
        logits = self.trunk(toks, cache)
        logits = logits[:, -token_ids.shape[1]:, :]
        self.calls += 1
        if self.corrupt_every and self.calls % self.corrupt_every == 0:
            logits = mx.roll(logits, 1, axis=-1)   # shifts the argmax → reject
        return logits, prev_hidden  # hidden unused by the oracle chain


def ar_greedy(trunk: TinyTrunk, prompt: list[int], max_tokens: int) -> list[int]:
    cache = make_trunk_cache(trunk)
    logits = trunk(mx.array([prompt], dtype=mx.uint32), cache)
    out = [int(mx.argmax(logits[:, -1, :]).item())]
    while len(out) < max_tokens:
        logits = trunk(mx.array([[out[-1]]], dtype=mx.uint32), cache)
        out.append(int(mx.argmax(logits[:, -1, :]).item()))
    return out


class FakeDetok:
    def reset(self):
        self.last_segment = ""

    def add_token(self, tok):
        self.last_segment = f"<{tok}>"


class FakeTokenizer:
    detokenizer = FakeDetok()


def run_mtp(trunk, mtp, prompt, max_tokens, depth, **kw) -> tuple[list[int], list]:
    canary = []
    toks = [r.token for r in native_mtp_stream_generate(
        trunk, FakeTokenizer(), prompt, mtp=mtp, depth=depth,
        max_tokens=max_tokens, canary_cb=lambda *a: canary.append(a), **kw)]
    return toks, canary


def main() -> None:
    trunk = TinyTrunk()
    mx.eval(trunk.parameters())
    mtp = make_mtp(trunk)
    prompt = [3, 14, 15, 92, 65, 35, 89, 79, 32, 38, 46, 26, 43]
    N = 40

    ref = ar_greedy(trunk, prompt, N)

    # 1 — per-position parity vs AR greedy, every depth.
    for depth in (1, 2, 3):
        got, canary = run_mtp(trunk, mtp, prompt, N, depth)
        assert len(got) == len(ref), f"D{depth}: len {len(got)} != {len(ref)}"
        for p, (a, b) in enumerate(zip(got, ref)):
            assert a == b, f"D{depth}: divergence at position {p}: {a} != {b}"
        final = canary[-1]
        assert final[0] == -1 and final[2] <= final[1], f"canary totals {final}"
        print(f"PASS parity D{depth} (accept {final[2]}/{final[1]} drafted)")

    # 1b — ORACLE mtp (same weights): 100% acceptance, full n=D path,
    # still per-position AR-exact. The load-bearing rollback/seed test.
    for depth in (1, 2, 3):
        got, canary = run_mtp(trunk, OracleMTP(trunk, prompt[0]), prompt, N, depth)
        assert got == ref, f"oracle D{depth} diverged from AR"
        final = canary[-1]
        assert final[2] == final[1], f"oracle D{depth}: acceptance {final[2]}/{final[1]} != 100%"
        rounds = [c for c in canary if c[0] > 0]
        assert all(c[2] == c[1] for c in rounds), "oracle round with n<D"
        print(f"PASS oracle full-accept D{depth} ({len(rounds)} rounds for {N} toks)")

    # 1c — corrupted oracle: partial acceptance (0<n<D paths + mixed trims).
    got, canary = run_mtp(trunk, OracleMTP(trunk, prompt[0], corrupt_every=3),
                          prompt, N, 3)
    assert got == ref, "partial-accept run diverged from AR"
    final = canary[-1]
    assert 0 < final[2] < final[1], f"expected partial acceptance, got {final[2]}/{final[1]}"
    print(f"PASS partial accept (accept {final[2]}/{final[1]})")

    # 2 — warm session-cache resume (invariant F-5) still AR-exact.
    warm = make_trunk_cache(trunk)
    trunk(mx.array([prompt], dtype=mx.uint32), warm)   # cache holds FULL prompt
    got, _ = run_mtp(trunk, mtp, prompt, N, 3,
                     prompt_cache=warm, prefix_len=len(prompt))
    assert got == ref, "warm-cache resume diverged from AR"
    print("PASS warm-cache resume")

    # 3 — stop_ids: cut at the first occurrence, finish_reason == "stop".
    stop_tok = ref[7]
    rs = list(native_mtp_stream_generate(
        trunk, FakeTokenizer(), prompt, mtp=make_mtp(trunk), depth=3,
        max_tokens=N, stop_ids={stop_tok}))
    cut = next(i for i, t in enumerate(ref) if t == stop_tok)
    assert [r.token for r in rs] == ref[:cut + 1], "stop path emitted wrong prefix"
    assert rs[-1].finish_reason == "stop", "missing finish_reason=stop"
    print("PASS stop_ids early exit")

    # 4 — pre_norm capture path runs and restores the real norm after.
    inner_norm_before = trunk.model.norm
    got, _ = run_mtp(trunk, make_mtp(trunk), prompt, 16, 2,
                     hidden_source="pre_norm")
    assert trunk.model.norm is inner_norm_before, "norm not restored"
    assert len(got) == 16, "pre_norm path wrong length"
    print("PASS pre_norm capture + restore")

    # 5 — max_tokens=1: just the prefill bonus, finish=length.
    rs = list(native_mtp_stream_generate(
        trunk, FakeTokenizer(), prompt, mtp=make_mtp(trunk), depth=3,
        max_tokens=1))
    assert len(rs) == 1 and rs[0].finish_reason == "length" and rs[0].token == ref[0]
    print("PASS max_tokens=1 edge")

    print("ALL PASS")


if __name__ == "__main__":
    main()
