#!/usr/bin/env python3
"""The runner must leave mlx-lm's BatchGenerator internals alone (2026-10-23).

Two local layers used to sit on mlx-lm's BatchGenerator:
  * `patches/opt_batch_gen.py` (exo, April 2026) replaced `GenerationBatch._step`
    and stored `_next_logprobs` as an mx.array; mlx-lm 0.32.x keeps a list, so
    upstream `extend` died with `'array' object has no attribute 'extend'`;
  * runner.py's shim over `_extend_cache` and `GenerationBatch.extend` (May 2026)
    hid exactly that, until mlx-lm#1830 (0.32.x) renamed the attributes the shim
    touched (`state_machines` -> `stop_sequences`, `_matcher_states` ->
    `_matchers`) and every batched request died at the first prompt -> generation
    hand-off (`0 toks . finish=error`, amigos / Qwen3-Coder-Next, 2026-10-23).
Both are gone; mlx-lm's own code runs.

This drives a real BatchGenerator with a tiny fake model through that exact path:
a second sequence is inserted while the first one decodes, so GenerationBatch.extend
runs once on an empty batch and once on a live one, with a KV cache and an
ArraysCache (the qwen3_next hybrid layout) merged, extended and filtered along the
way. No model download, a few zero tensors only. Needs a venv with mlx + mlx-lm
(a node's ~/mlx-cluster/.venv works):

    .venv/bin/python scripts/test_batch_generation_extend.py
"""
import os
import sys

sys.argv = ["runner.py"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner  # noqa: E402  -- whatever it does to mlx_lm happens here

import mlx.core as mx  # noqa: E402
from mlx_lm.generate import BatchGenerator, GenerationBatch  # noqa: E402
from mlx_lm.models.cache import ArraysCache, KVCache  # noqa: E402

_gen_mod = sys.modules["mlx_lm.generate"]
FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


# --- 1. the runner did not replace upstream code -----------------------------
check("GenerationBatch.extend is upstream's",
      GenerationBatch.extend.__module__, "mlx_lm.generate")
check("_extend_cache is upstream's",
      _gen_mod._extend_cache.__module__, "mlx_lm.generate")
check("GenerationBatch._step is upstream's (no exo opt_batch_gen mirror)",
      GenerationBatch._step.__module__, "mlx_lm.generate")
check("runner still imports BatchGenerator", runner._BATCH_AVAILABLE, True)


# --- 2. two sequences through a real BatchGenerator ---------------------------
class FakeModel:
    """Two 'layers': one attention (KVCache) and one linear (ArraysCache, the
    qwen3_next layout). Writes both caches like a real model so merge / extend /
    filter / extract run on populated state. Zero logits -> greedy picks token 0."""
    vocab = 16

    def __init__(self):
        self.layers = [object(), object()]

    def make_cache(self):
        return [KVCache(), ArraysCache(size=2)]

    def __call__(self, x, cache=None):
        B, S = x.shape
        if cache:
            kv = mx.zeros((B, 1, S, 4), dtype=mx.float16)
            cache[0].update_and_fetch(kv, kv)
            ac = cache[1]
            ac[0] = mx.zeros((B, 3, 8), dtype=mx.float16)      # conv state
            ac[1] = mx.zeros((B, 2, 4, 4), dtype=mx.float16)   # recurrent state
            if hasattr(ac, "advance"):
                ac.advance(S)
        return mx.zeros((B, S, self.vocab))


bg = BatchGenerator(FakeModel(), max_tokens=3, completion_batch_size=4,
                    prefill_batch_size=2, prefill_step_size=8)
ntoks = {}
finish = {}
errors = []


def tick():
    try:
        out = bg.next()
    except Exception as e:  # the 2026-10-23 failure mode
        errors.append(f"{type(e).__name__}: {e}")
        return
    gen = out[1] if isinstance(out, tuple) else out
    for r in gen:
        ntoks[r.uid] = ntoks.get(r.uid, 0) + 1
        if r.finish_reason:
            finish[r.uid] = r.finish_reason


(uid_a,) = bg.insert(prompts=[[1, 2, 3, 4]], max_tokens=[3])
tick()                                       # A: prompt processed
(uid_b,) = bg.insert(prompts=[[5, 6, 7]], max_tokens=[3])
for _ in range(12):
    tick()
    if errors or {uid_a, uid_b} <= set(finish):
        break

check("bg.next() never raised", errors, [])
check("A finished by length", finish.get(uid_a), "length")
check("B finished by length (inserted while A decoded)", finish.get(uid_b), "length")
check("A produced max_tokens tokens", ntoks.get(uid_a), 3)
check("B produced max_tokens tokens", ntoks.get(uid_b), 3)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("all checks passed")
