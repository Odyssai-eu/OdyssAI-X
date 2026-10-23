#!/usr/bin/env python3
"""A replaced ReplicaPool must never keep resurrecting runners (2026-10-23).

Timeline of the amigos leak: reboot-all -> the pool's watchdog starts healing its
rebooted replicas -> the dead-pool purge skips a healing pool -> the reboot-reload
restore builds a SECOND ReplicaPool under the same alias and set_pool() overwrites
the registry entry -> the first object is never stop()ped; its watchdog restarts
its three replicas after every later sweep (three orphan 65 GB runners next to
the serving ones, six ssh sessions in the container).

Two guards, both exercised here without touching a node:
  1. _restore_cluster_pools skips an alias that is still registered;
  2. once a ReplicaPool has been registered (set_pool arms it), its watchdog
     tears its replicas down and exits as soon as the registry no longer holds
     the object (identity check, not alias lookup). A registered pool and a
     pool that was never registered are left alone.

    .venv/bin/python scripts/test_replica_zombie_guard.py
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

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


class FakeChild:
    """Stands in for the per-replica single-node RunnerPool: alive, idle."""
    stopped: list = []

    def __init__(self, **kw):
        self.alias = kw.get("alias")
        self.nodes = [{"host": "n", "ssh": None}]
        self.runners = [object()]           # non-empty -> _child_alive() asks alive_count
        self.busy_count = 0

    def alive_count(self):
        return 1

    async def start(self):
        return None

    async def stop(self):
        FakeChild.stopped.append(self.alias)


api.RunnerPool = FakeChild
api.REPLICA_WATCHDOG_S = 0.01
api._ssh_capture = lambda *a, **k: (_ for _ in ()).throw(AssertionError("a node was contacted"))
R = sorted(f"qwen#r{i}" for i in range(3))


async def run_watchdog(pool):
    """Let the watchdog tick a few times, then cancel it like stop() would.
    Returns True when it was still running at that point (a healthy watchdog),
    False when it had already exited on its own (the zombie guard fired)."""
    task = asyncio.create_task(pool._watchdog())
    await asyncio.sleep(0.1)
    was_running = not task.done()
    if was_running:
        task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    return was_running


def new_pool(alias="qwen"):
    p = api.ReplicaPool(model="/m/x", cluster="repl", alias=alias,
                        node_indices=[0, 1, 2], batch=True)
    p._live = {0, 1, 2}
    return p


# --- 2a. never registered (start() done, set_pool() not yet): left alone ------
FakeChild.stopped.clear()
fresh = new_pool()
running = asyncio.run(run_watchdog(fresh))
check("never registered: no replica stopped", FakeChild.stopped, [])
check("never registered: still healing-capable", fresh._stopping, False)
check("never registered: watchdog kept running", running, True)

# --- 2b. registered: left alone ---------------------------------------------
FakeChild.stopped.clear()
live = new_pool()
api.set_pool("repl", "qwen", live)
check("set_pool arms the guard", live._was_registered, True)
running = asyncio.run(run_watchdog(live))
check("registered: no replica stopped", FakeChild.stopped, [])
check("registered: still healing-capable", live._stopping, False)
check("registered: watchdog kept running", running, True)

# --- 2c. replaced under the SAME alias (the amigos leak): old object exits ----
FakeChild.stopped.clear()
newer = new_pool()
api.set_pool("repl", "qwen", newer)          # overwrites `live`, as reboot-reload did
running = asyncio.run(run_watchdog(live))
check("replaced: old object stops its replicas", sorted(FakeChild.stopped), R)
check("replaced: old object marked stopping", live._stopping, True)
check("replaced: old object has no live replica", live._live, set())
check("replaced: watchdog exited by itself", running, False)
check("replaced: new object untouched", newer._stopping, False)

# --- 2d. dropped from the registry without stop(): same outcome ----------------
FakeChild.stopped.clear()
api.del_pool("repl", "qwen")                 # `newer` is now unreferenced by the registry
running = asyncio.run(run_watchdog(newer))
check("dropped: object stops its replicas", sorted(FakeChild.stopped), R)
check("dropped: marked stopping", newer._stopping, True)
check("dropped: watchdog exited by itself", running, False)


# --- 1. the restore never duplicates a registered alias ------------------------
class ReplicaRecorder:
    made: list = []

    def __init__(self, **kw):
        ReplicaRecorder.made.append(kw)
        self.alias = kw["alias"]

    async def start(self):
        return None


api.ReplicaPool = ReplicaRecorder
api.load_cluster_state_v2 = lambda cid: [
    {"alias": "qwen", "is_replica": True, "model": "/m/x", "node_indices": [0, 1, 2],
     "batch": True}]

ReplicaRecorder.made.clear()
holder = object()
api.set_pool("repl", "qwen", holder)         # something is registered under the alias
restored = asyncio.run(api._restore_cluster_pools("repl"))
check("restore: registered alias skipped", restored, [])
check("restore: no second pool built", ReplicaRecorder.made, [])
check("restore: registry still holds the original object",
      api.get_pool("repl", "qwen") is holder, True)

api.del_pool("repl", "qwen")
ReplicaRecorder.made.clear()
restored = asyncio.run(api._restore_cluster_pools("repl"))
check("restore: free alias restored", restored, ["qwen"])
check("restore: one pool built, batched as saved",
      (len(ReplicaRecorder.made), ReplicaRecorder.made[0]["batch"] if ReplicaRecorder.made else None),
      (1, True))
api.del_pool("repl", "qwen")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("all checks passed")
