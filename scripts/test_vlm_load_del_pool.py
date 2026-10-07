#!/usr/bin/env python3
"""A FAILED distributed VL load must leave the alias ABSENT, not registered
as None.

Bug 4 (2026-09-30 incident, root proven, errors still in the 05/10 logs): the
vision branch of admin_cluster_load stopped the old pool then did
set_pool(cluster_id, alias, None) before building the new one. If the new
VLMDistPool.start() failed, the alias stayed registered as None: list_pools()
does not filter None, so every status poll crashed in _pool_view on
'NoneType' has no attribute 'started_at' and the [jaccl-stability] loop
degraded the cluster. The fix registers nothing: del_pool on the old entry,
set_pool only after a successful start. No node is reached: every remote
helper is stubbed.

    .venv/bin/python scripts/test_vlm_load_del_pool.py
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


class OldPool:
    """The pool already registered under the alias before the reload."""
    stopped = 0

    def alive_count(self):
        return 1  # alive: _purge_dead_pools must keep it

    async def stop(self):
        OldPool.stopped += 1


class FailingVLMDist:
    made = []

    def __init__(self, **kw):
        FailingVLMDist.made.append(kw)
        self.model, self.alias = kw["model"], kw["alias"]

    async def start(self):
        raise RuntimeError("rank 1 died during load: ValueError: [convert] Only length-1 arrays")

    async def stop(self):
        return None


async def arch_mimo(ssh, path):
    return {"is_vision": True, "model_type": "mimo_v2"}


async def preflight_ok(cid, model, draft=None, venv=None):
    return {"ok": True}


def _pending(*a, **k):
    raise AssertionError("a node was contacted")


api.VLMDistPool = FailingVLMDist
api.get_model_arch_meta = arch_mimo
api._gather_preflight = preflight_ok
api.save_cluster_state_v2 = lambda cid: None
api._vlm_resolve_model_path = lambda m, d: m
api._ssh_capture = _pending
api.VLM_DISTRIBUTED_ENABLED = True
MODEL = "/Volumes/models/odysseus/odyssai/MiMo-V2.6-Pro-RL-Q9h16"

old = OldPool()
old.mode, old.use_ap, old.nodes_count, old.kv_q8 = "tensor", False, 4, False
old.draft_model, old.num_draft_tokens, old.node_indices, old.backend = None, 4, [0, 1, 2, 3], "jaccl"
old.cluster, old.model, old.alias = "argo", MODEL, "mimo"
api.set_pool("argo", "mimo", old)

req = api._pool_reload_request(old)
try:
    asyncio.run(api.admin_cluster_load("argo", req))
    outcome = "NO ERROR"
except HTTPException as e:
    outcome = f"HTTP {e.status_code}"
except Exception as e:  # the bug path let raw errors escape in some builds
    outcome = f"RAW {type(e).__name__}"

check("the failed VL-dist load surfaces an HTTP 500", outcome, "HTTP 500")
check("the old pool was stopped before the new one started", OldPool.stopped, 1)
check("the alias is ABSENT from the registry (not registered as None)",
      "mimo" in api._pools.get("argo", {}), False)
check("nothing that iterates the pools sees a None entry",
      [p for _, _, p in api.list_all_pools() if p is None], [])
check("a later reload is not blocked by a phantom pool",
      api.get_pool("argo", "mimo") is None, True)

# Source guard: the vision branch never stores None again.
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.py")).read()
check("no set_pool(..., None) remains in api.py", "set_pool(cluster_id, alias, None)" in src, False)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
