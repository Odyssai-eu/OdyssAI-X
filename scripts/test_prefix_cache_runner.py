#!/usr/bin/env python3
"""The runner's session store and what a snapshot keeps.

2026-11-05: a snapshot shared what generation overwrites, so a hit on a hybrid
model started from the state AFTER the previous reply. Needs mlx-lm 0.32 (a node's
venv), no model is loaded.

    ~/mlx-cluster/.venv/bin/python test_prefix_cache_runner.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache, _BaseCache  # noqa: E402
import runner  # noqa: E402

FAILS = []


def check(label, ok, got=""):
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": {got}"))
    if not ok:
        FAILS.append(label)


# ── the session store ───────────────────────────────────────────────────
def filled_kv(n):
    cache = KVCache()
    cache.update_and_fetch(mx.zeros((1, 2, n, 4)), mx.zeros((1, 2, n, 4)))
    return [cache]


def store(sid, tokens, cache):
    runner._session_store[sid] = {"cache": cache, "tokens": tokens, "model_id": "m",
                                  "last_used": 0, "bytes": 0}


prompt = list(range(100))
store("s", prompt + [500, 501], filled_kv(102))
cache, suffix, kind = runner._session_lookup("s", "m", prompt + [7, 8])
check("hit: rewinds to the shared prefix", cache[0].offset == 100 and suffix == [7, 8],
      (cache[0].offset, suffix))
check("hit: the request owns the entry", "s" not in runner._session_store)
check("hit: a second request starts cold",
      runner._session_lookup("s", "m", prompt)[2] == "cold")

store("s", prompt + [500], filled_kv(101))
cache, suffix, kind = runner._session_lookup("s", "m", prompt)
check("replay: reuses all but the last token",
      cache is not None and suffix == [99] and cache[0].offset == 99, (kind, suffix))

store("s", prompt, filled_kv(100))
cache, suffix, kind = runner._session_lookup("s", "m", prompt[:50])
check("shorter prompt: still a hit", cache is not None and suffix == [49], (kind, suffix))

store("s", prompt, filled_kv(100))
check("divergent: no reuse", runner._session_lookup("s", "m", [9] * 100)[0] is None)


# ── what a snapshot keeps ───────────────────────────────────────────────
def as_list(a):
    return a.tolist()


arrays = ArraysCache(size=1)
arrays[0] = mx.array([1.0])
snap = runner._snap_cache([arrays])
arrays[0] = mx.array([2.0])
restored = runner._restore_from_snap([arrays], snap)
check("ArraysCache: snapshot keeps the state before the reply",
      as_list(restored[0][0]) == [1.0], as_list(restored[0][0]))
restored[0][0] = mx.array([3.0])
again = runner._restore_from_snap([arrays], snap)
check("ArraysCache: snapshot survives its first restore", as_list(again[0][0]) == [1.0])

rot = RotatingKVCache(max_size=8)
rot.update_and_fetch(mx.ones((1, 1, 4, 2)), mx.ones((1, 1, 4, 2)))
mx.eval(rot.keys, rot.values)
snap = runner._snap_cache([rot])
rot.keys[..., 0:4, :] = mx.zeros((1, 1, 4, 2))
restored = runner._restore_from_snap([rot], snap)
check("RotatingKVCache: overwriting the window leaves the snapshot alone",
      mx.all(restored[0].keys[..., :4, :] == 1).item())


class Inner(_BaseCache):
    def __init__(self):
        self.keys = None


class Outer(KVCache):
    def __init__(self):
        super().__init__()
        self.indexer = Inner()


outer = Outer()
outer.update_and_fetch(mx.zeros((1, 1, 2, 2)), mx.zeros((1, 1, 2, 2)))
outer.indexer.keys = mx.array([1.0])
snap = runner._snap_cache([outer])
outer.indexer.keys = mx.array([1.0, 2.0])
restored = runner._restore_from_snap([outer], snap)
check("nested cache: snapshot keeps its own", as_list(restored[0].indexer.keys) == [1.0],
      as_list(restored[0].indexer.keys))

kv = filled_kv(4)[0]
state = runner._own_state(kv, vars(kv))
check("KVCache: the big buffers stay shared", state["keys"] is kv.keys)


print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
