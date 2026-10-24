#!/usr/bin/env python3
"""The routed-expert tensor split keeps the per-expert NVFP4 global scales whole.

MiniMax-M3.1-Flash (mlx-vlm PR #2580) stores one global scale per expert,
weight_global_scale [E], next to the expert bank weight [E, out, in]. mlx's
"all-to-sharded" / "sharded-to-all" predicates slice it on the expert axis
(2026-10-24), so a rank holds E/n scales for E experts. vlm_runner's
_expert_sharding replicates it. This emulates a 2-rank split of a float expert
bank with the predicate and checks the recombined output against the unsplit
one, and checks that remote_vlm_cmd runs the runner from the checkpoint's
mlx-vlm venv. No node is reached.

    ~/mlx-cluster/.venv/bin/python scripts/test_vlm_expert_shard.py
"""
import os
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


try:
    import mlx.core as mx
    from mlx.nn.layers.distributed import _all_to_sharded
    import vlm_runner
except ImportError as e:
    mx = None
    print(f"SKIP predicate checks (no mlx: {e})")

if mx is not None:
    N, E, OUT, IN = 2, 8, 12, 16

    def split(params, predicate, r):
        out = {}
        for path, w in params.items():
            s = predicate(path, w)
            out[path] = w if s is None else mx.split(w, N, axis=s[0])[r]
        return out

    mx.random.seed(0)
    up = {"weight": mx.random.normal((E, OUT, IN)),
          "weight_global_scale": mx.random.uniform(0.5, 2.0, (E,))}
    down = {"weight": mx.random.normal((E, IN, OUT)),
            "weight_global_scale": mx.random.uniform(0.5, 2.0, (E,))}
    x = mx.random.normal((IN,))
    e = 5

    def proj(p, v, e):
        return (p["weight"][e] @ v) / p["weight_global_scale"][e]

    full = proj(down, proj(up, x, e), e)
    a2s, s2a = vlm_runner._expert_sharding("all-to-sharded"), vlm_runner._expert_sharding("sharded-to-all")
    ranks = [(split(up, a2s, r), split(down, s2a, r)) for r in range(N)]
    check("global scale replicated (all-to-sharded)", ranks[1][0]["weight_global_scale"].shape, (E,))
    check("global scale replicated (sharded-to-all)", ranks[1][1]["weight_global_scale"].shape, (E,))
    check("up weight split on the out dim", ranks[1][0]["weight"].shape, (E, OUT // N, IN))
    check("down weight split on the in dim", ranks[1][1]["weight"].shape, (E, IN, OUT // N))
    partial = sum(proj(d, proj(u, x, e), e) for u, d in ranks)
    check("2-rank partial sum equals the unsplit expert",
          bool(mx.allclose(partial, full, rtol=1e-5, atol=1e-5).item()), True)
    check("bias of all-to-sharded split on the last axis", a2s("bias", mx.zeros((E, OUT))), (-1, 1))
    check("bias of sharded-to-all replicated", s2a("bias", mx.zeros((E, IN))), None)
    check("weight_scale_2 replicated", a2s("weight_scale_2", mx.ones((E,))), None)
    try:
        a2s("pre_quant_scale", mx.ones((IN,)))
        check("pre_quant_scale refused", False, True)
    except ValueError:
        check("pre_quant_scale refused", True, True)
    # The bug being fixed: mlx's own predicate cuts the [E] scale.
    check("mlx all-to-sharded slices the [E] scale (the bug)",
          _all_to_sharded(1)("weight_global_scale", up["weight_global_scale"])[0], 0)

    # The replicated indexer hands each rank its own index heads: 4 heads on
    # 2 ranks -> rank 1 gets heads 2 and 3, so 32 local query heads map 16 per
    # index head as on a single node.
    class _FakeIndexer:
        block_size = 128
        def __call__(self, x):
            return mx.arange(4)[None, :, None, None] * mx.ones((1, 4, 3, 16), dtype=mx.int32), 4096, x
    ri = vlm_runner._RankIndexer(_FakeIndexer(), 4, 1, 2)
    bi, total, qpos = ri("qpos")
    check("rank 1 of 2 keeps index heads 2 and 3", (bi.shape, bi[0, :, 0, 0].tolist()), ((1, 2, 3, 16), [2, 3]))
    check("total_len and q_positions pass through", (total, qpos), (4096, "qpos"))
    check("attributes come from the wrapped indexer", ri.block_size, 128)
    check("None selection (dense context) passes through",
          vlm_runner._RankIndexer(lambda x: (None, 100, x), 4, 0, 2)("q")[0], None)
    try:
        vlm_runner._RankIndexer(_FakeIndexer(), 4, 0, 3)
        check("3 ranks refused for 4 index heads", False, True)
    except ValueError:
        check("3 ranks refused for 4 index heads", True, True)

try:
    import api
except ImportError as e:
    api = None
    print(f"SKIP remote_vlm_cmd checks (api imports need the repo venv: {e})")

if api is not None:
    nodes = [{"rank": 0, "ssh": "admin@10.0.0.1"}, {"rank": 1, "ssh": "admin@10.0.0.2"}]
    cmd = api.remote_vlm_cmd(nodes[0], nodes, "MiniMaxAI/MiniMax-M3.1-Flash", 5555, devices_json="[]")
    check("M3.1 runner from mlx-vlm-m31", "$HOME/.venvs/mlx-vlm-m31/bin/python " in cmd, True)
    check("M3.1 runner without MLX_METAL_FAST_SYNCH", "MLX_METAL_FAST_SYNCH" in cmd, False)
    cmd = api.remote_vlm_cmd(nodes[0], nodes, "odyssai/MiniMax-M3-VL-mlx-6bit-headbf16", 5555, devices_json="[]")
    check("other VL model keeps VLM_PYTHON_REMOTE", f" {api.VLM_PYTHON_REMOTE} " in cmd, True)
    check("other VL model keeps MLX_METAL_FAST_SYNCH=1", "MLX_METAL_FAST_SYNCH=1 " in cmd, True)

print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAIL'}")
sys.exit(1 if FAILS else 0)
