#!/usr/bin/env python3
"""A watchdog / preventive / reset reload of a distributed VL pool must reach the
VL loader, not the text runner.

_pool_reload_request() snapshots a pool into an ArgoLoadRequest with force=True (skip the
size preflight and the degraded gate). force=True also bypassed the vision auto-detect in
admin_cluster_load, so the keepalive recovery ladder reloaded MiMo-V2.6-Pro through
RunnerPool: its four ranks died on "The model does not support tensor parallelism" and the
cluster was marked degraded again (2026-09-30, after the 4-node reboot). No node is reached:
every remote helper is stubbed.

    python3 scripts/test_reload_vlm_dist.py
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
                    "models_dir": "/Volumes/models/odysseus",
                    "nodes": [{"host": f"n{i}", "ssh": f"admin@198.51.100.{i + 1}", "master": i == 0}
                              for i in range(4)]}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402
from fastapi import HTTPException  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


class FakeVLMDist:
    made = []

    def __init__(self, **kw):
        FakeVLMDist.made.append(kw)
        self.load_s = 1.0
        self.model, self.alias = kw["model"], kw["alias"]
        self.backend = kw.get("backend")

    async def start(self):
        return None

    async def stop(self):
        return None


class TextRunnerReached(Exception):
    pass


class BoomRunnerPool:
    def __init__(self, **kw):
        raise TextRunnerReached(kw.get("model"))


async def arch_mimo(ssh, path):
    return {"is_vision": True, "model_type": "mimo_v2"}


async def no_purge(cid):
    return []


async def preflight_ok(cid, model, draft=None, venv=None):
    return {"ok": True}


def _pending(*a, **k):
    raise AssertionError("a node was contacted")

api.VLMDistPool = FakeVLMDist
api.RunnerPool = BoomRunnerPool
api.get_model_arch_meta = arch_mimo
api._purge_dead_pools = no_purge
api._gather_preflight = preflight_ok


async def size_stub(ssh, path):
    return 566_317_924_352          # 527 GB, the real du of MiMo-V2.6-Pro-RL-Q9h16

api.get_model_size_bytes = size_stub
api.VLM_DISTRIBUTED_ENABLED = True
api.save_cluster_state_v2 = lambda cid: None
api._vlm_resolve_model_path = lambda m, d: m
api._ssh_capture = _pending
MODEL = "/Volumes/models/odysseus/odyssai/MiMo-V2.6-Pro-RL-Q9h16"


def dead_pool():
    """The pool object the keepalive recovery snapshots (a VLMDistPool, alive 0)."""
    p = FakeVLMDist(model=MODEL, cluster="argo", alias="mimo", node_indices=[0, 1, 2, 3],
                    shard_mode="", backend="jaccl")
    p.mode, p.use_ap, p.nodes_count, p.kv_q8 = "tensor", False, 4, False
    p.draft_model, p.num_draft_tokens, p.node_indices, p.backend = None, 4, [0, 1, 2, 3], "jaccl"
    FakeVLMDist.made.clear()
    return p


def run_load(req):
    for a, _ in list(api.list_pools("argo")):     # each case starts from an empty cluster
        api.del_pool("argo", a)
    api._cluster_degraded.pop("argo", None)
    api._mark_cluster_degraded("argo", "keepalive timeout — peer unresponsive", {})
    try:
        return asyncio.run(api.admin_cluster_load("argo", req))
    except TextRunnerReached as e:
        return f"TEXT RUNNER reached for {e}"
    except HTTPException as e:
        return f"HTTP {e.status_code}"


req = api._pool_reload_request(dead_pool())
check("reload request skips the size preflight and degraded gate (force=True)", req.force, True)
check("reload request is marked as an internal reload", req.internal_reload, True)
check("reload request keeps the pool's mode and nodes", (req.mode, req.nodes, req.node_indices), ("tensor", 4, [0, 1, 2, 3]))

res = run_load(req)
check("recovery reload of a VL-dist pool is not refused as degraded and does not reach the text runner",
      isinstance(res, dict) and res.get("dispatched"), "vlm-dist-pool")
check("… it built the distributed VL pool on the same nodes and model",
      [(m["model"], m["node_indices"]) for m in FakeVLMDist.made], [(MODEL, [0, 1, 2, 3])])
check("… in the pool's own shard mode (tensor)", FakeVLMDist.made[-1]["shard_mode"], "tensor")

# An operator's force=true keeps its documented meaning: bypass the VL detection.
FakeVLMDist.made.clear()
manual_force = api.ArgoLoadRequest(model=MODEL, nodes=4, node_indices=[0, 1, 2, 3], alias="mimo", force=True)
check("a manual force=true still bypasses the VL detection (text runner, as documented)",
      str(run_load(manual_force)).startswith("TEXT RUNNER"), True)

# A manual load into a degraded cluster is still refused.
plain = api.ArgoLoadRequest(model=MODEL, nodes=4, node_indices=[0, 1, 2, 3], alias="mimo")
check("a manual load into a degraded cluster is still refused (409)", run_load(plain), "HTTP 409")

# Source guard: the vision branch reads internal_reload, and only _pool_reload_request sets it.
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.py")).read()
check("only _pool_reload_request builds an internal reload", src.count("internal_reload=True"), 1)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
