#!/usr/bin/env python3
"""Replica loads batch by default (#107): resolve_replica_batch is the single
source of truth — absent/null defaults to True on kind=replica (a forgotten
checkbox served one request at a time per replica: 164 vs 431-625 tok/s on
2026-10-07), explicit false opts out, null never leaks to persistence or
status. The dashboard now always sends an explicit bool, and a state entry
saved without the key restores batched like a fresh load. No node is reached.

    .venv/bin/python scripts/test_batch_default_replica.py
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
json.dump({"repl": {"name": "R", "kind": "replica", "backend": "ring",
                    "models_dir": "/m",
                    "nodes": [{"host": f"n{i}", "ssh": f"admin@198.51.100.{i + 1}", "master": i == 0}
                              for i in range(3)]}},
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


# --- resolver unit cases (the 3-amigos contract) ---
logs = []
res = lambda raw, kind: api.resolve_replica_batch(raw, kind, "c", log=logs.append)

logs.clear()
check("absent + replica → True (the #107 default)", res(None, "replica"), True)
check("… and the defaulting leaves a trace", len(logs), 1)
logs.clear()
check("null (JSON) + replica → True, null ≡ absent", res(None, "replica"), True)
check("absent + non-replica kind → False", res(None, "mlx-distributed"), False)
logs.clear()
check("explicit true wins, no log", (res(True, "replica"), len(logs)), (True, 0))
logs.clear()
check("explicit false opts out, no log", (res(False, "replica"), len(logs)), (False, 0))
check("the result is always a strict bool", all(isinstance(res(r, "replica"), bool)
      for r in (None, True, False)), True)


# --- integration: the replica load branch resolves, never passes None ---
class ReplicaRecorder:
    made = []

    def __init__(self, **kw):
        ReplicaRecorder.made.append(kw)
        self.model, self.alias = kw["model"], kw["alias"]
        self.batch = kw["batch"]
        self.load_s = 1.0

    def alive_count(self):
        return 1

    def mark_capacity_fatal(self, cb):
        pass

    def replica_stats(self):
        return []

    async def start(self):
        return None

    async def stop(self):
        return None


async def no_arch(ssh, path):
    return {"is_vision": False}          # text model: text ReplicaPool path


async def preflight_ok(cid, model, draft=None, venv=None):
    return {"ok": True}


async def size_small(ssh, path):
    return 8 * 1024 ** 3                 # 8 GB: fits any node


api.ReplicaPool = ReplicaRecorder
api.get_model_arch_meta = no_arch
api._gather_preflight = preflight_ok
api.get_model_size_bytes = size_small
api.save_cluster_state_v2 = lambda cid: None
api._ssh_capture = lambda *a, **k: (_ for _ in ()).throw(AssertionError("a node was contacted"))
MODEL = "/m/some-model-q6"


def run_load(**overrides):
    for a, _ in list(api.list_pools("repl")):
        api.del_pool("repl", a)
    ReplicaRecorder.made.clear()
    req = api.ArgoLoadRequest(model=MODEL, nodes=3, **overrides)
    try:
        return asyncio.run(api.admin_cluster_load("repl", req))
    except HTTPException as e:
        return f"HTTP {e.status_code}"


run_load()
check("a replica load WITHOUT the flag builds a batched pool",
      ReplicaRecorder.made[-1]["batch"], True)
run_load(batch=True)
check("batch=true explicit → True", ReplicaRecorder.made[-1]["batch"], True)
run_load(batch=False)
check("batch=false explicit → opt-out honored", ReplicaRecorder.made[-1]["batch"], False)

# --- restore: explicit value wins, missing key gets the new default ---
check("restore resolver: saved false stays false",
      api.resolve_replica_batch(False, "replica", "c", log=lambda _m: None), False)
check("restore resolver: legacy entry WITHOUT the key → batched (arbitrage #107)",
      api.resolve_replica_batch(None, "replica", "c", log=lambda _m: None), True)

# --- source guards ---
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.py")).read()
check("the request field is Optional (tri-state), never a bare bool default",
      "batch: Optional[bool] = None" in src, True)
check("no raw req.batch reaches ReplicaPool outside the resolver",
      "batch=bool(req.batch)" in src, False)
check("no raw entry batch default leaks outside the resolver",
      'entry.get("batch", False)' in src, False)
dash = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")).read()
check("dashboard default box is checked", "default_batch: true," in dash, True)
check("dashboard ALWAYS sends an explicit bool (unchecked = false, not omitted)",
      "body.batch = !!f.default_batch;" in dash, True)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
