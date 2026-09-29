#!/usr/bin/env python3
"""Auto-reload must never rebuild a VLM pool as a RunnerPool text pool.

A distributed VL pool is persisted with `is_vlm_dist` only; the guard of
_auto_reload_purged missed that key, so MiMo-V2.6-Pro (VLMDistPool) was reloaded
through the text runner, its ranks died, and the leak recovery rebooted the Argo
nodes in a loop (2026-09-29/30). No node is reached: RunnerPool, ssh and the
leak-reboot helper are stubbed.

    python3 scripts/test_auto_reload_vlm.py
"""
import asyncio
import json
import os
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({"argo": {"name": "A", "kind": "mlx-distributed", "backend": "jaccl",
                    "nodes": [{"host": f"n{i}", "ssh": f"admin@198.51.100.{i + 1}", "master": i == 0}
                              for i in range(4)]}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


built = []


class FakeRunnerPool:
    def __init__(self, **kw):
        built.append(kw.get("alias"))

    async def start(self):
        raise RuntimeError("would spawn text runners")


async def no_reboot(*a, **k):
    raise AssertionError("leak-recovery reboot reached")


async def reachable(_ssh):
    return True

api.RunnerPool = FakeRunnerPool
api._reboot_leaked_pool_nodes = no_reboot
api._node_reachable = reachable
api._cluster_is_degraded = lambda cid: False


def run_with_state(entries):
    built.clear()
    api._AUTO_RELOAD_PENDING.clear()
    api._AUTO_RELOAD_RETRIES.clear()
    api.load_cluster_state_v2 = lambda cid: entries
    try:
        asyncio.run(api._auto_reload_purged("argo", [e["alias"] for e in entries]))
    except AssertionError as e:
        return f"reboot: {e}"
    return list(built)


# the exact entry persisted on .39 for MiMo-V2.6-Pro (state-main.json, 2026-09-29 22:10)
mimo = {"alias": "mimo-v2-6-pro-rl-q9h16", "model": "/Volumes/models/odysseus/odyssai/MiMo-V2.6-Pro-RL-Q9h16",
        "is_vlm_dist": True, "node_indices": [0, 1, 2, 3], "backend": "jaccl"}
check("distributed VL pool (is_vlm_dist) is not rebuilt as a text pool", run_with_state([mimo]), [])
check("… and leaves the pending queue", ("argo", mimo["alias"]) in api._AUTO_RELOAD_PENDING, False)
check("single-node VL pool (is_vlm) skipped too",
      run_with_state([{"alias": "v", "model": "/m", "is_vlm": True, "node_indices": [0]}]), [])
text = {"alias": "glm", "model": "/m/GLM", "mode": "pipeline", "node_indices": [0], "backend": "ring"}
check("a text pool is still auto-reloaded", run_with_state([text]), ["glm"])

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
