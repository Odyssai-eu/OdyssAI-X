#!/usr/bin/env python3
"""A wedged replica stays a replica problem (2026-11-03).

Twice that evening one LFM3 node stopped producing tokens; the no-progress
watchdog of its child RunnerPool marked the whole cluster degraded and ran
_cluster_reset, which stopped the three healthy replicas too and left the pool
down until a manual reset.

Exercised on the real RunnerPool.submit generator, without touching a node:
  1. a wedged child is quarantined, the request moves on, nothing cluster-wide;
  2. the same after tokens were already emitted;
  3. a runner exit stays per replica;
  4. the watchdog does not re-admit a quarantined replica, it restarts it;
  5. reports are idempotent, and our own restart is not a new failure;
  6. retry reaches the last untried replica;
  7. a distributed pool keeps its cluster-wide handling.

    .venv/bin/python scripts/test_replica_wedge_scope.py
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
json.dump({"repl": {"name": "R", "kind": "replica", "backend": "ring", "models_dir": "/m",
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


class Proc:
    stdin = None

    def poll(self):
        return None


class FakeRunner:
    """A connected rank 0. `script` decides what it answers to a generation."""

    def __init__(self, pool, script):
        self.pool, self.script = pool, script
        self.node = {"rank": 0, "ssh": None}
        self.proc = Proc()

    def stderr_tail(self, _n):
        return ""

    def send(self, req):
        if req.get("cmd") != "gen":
            return
        q = self.pool._listeners[req["id"]]
        for ev in self.script:
            q.put_nowait(dict(ev))


TOKEN = {"event": "token", "text": "x"}
DONE = {"event": "done"}
WEDGED = []                              # sends nothing, ever
EXIT = [{"event": "runner_exit", "rc": 1}]

marked, resets = [], []


async def fake_reset(cluster):
    resets.append(cluster)


api._mark_cluster_degraded = lambda c, reason=None, details=None: marked.append(reason)
api._cluster_reset = fake_reset
api._GEN_PREFILL_DEADLINE_S = 0.05


def new_pool(*scripts):
    marked.clear()
    resets.clear()
    pool = api.ReplicaPool(model="/m/x", cluster="repl", alias="lfm",
                           node_indices=list(range(len(scripts))), batch=True)
    pool._live = set(range(len(scripts)))
    for child, script in zip(pool._replicas, scripts):
        child.runners = [FakeRunner(child, script)]
        child._decode_deadline_s = 0.05
    return pool


async def collect(pool):
    out = []
    try:
        async for ev in pool.submit(prompt="p", max_tokens=4, enable_thinking=False):
            out.append(ev.get("event"))
    except RuntimeError as e:
        out.append(f"raised: {str(e).split(':')[0]}")
    return out


# --- 1. wedged before the first token: the request moves on --------------------
pool = new_pool(WEDGED, [TOKEN, DONE], [TOKEN, DONE], [TOKEN, DONE])
got = asyncio.run(collect(pool))
check("wedge: request served by another replica", got, ["token", "done"])
check("wedge: replica out of the routing", pool._live, {1, 2, 3})
check("wedge: replica quarantined", pool._heal[0].get("quarantined"), True)
check("wedge: cluster not degraded", marked, [])
check("wedge: no cluster reset", resets, [])
check("wedge: healthy child untouched", pool._heal[1].get("quarantined"), None)
check("wedge: child itself not degraded", pool._replicas[0].degraded, False)

# --- 2. wedged after tokens: the client sees the error, routing is still fixed --
pool = new_pool([TOKEN], [TOKEN, DONE])
got = asyncio.run(collect(pool))
check("mid-stream: error reaches the caller", got, ["token", "raised: runner wedged mid-generation"])
check("mid-stream: replica out of the routing", pool._live, {1})
check("mid-stream: nothing cluster-wide", (marked, resets), ([], []))

# --- 3. a runner exit stays per replica ----------------------------------------
pool = new_pool(EXIT, [TOKEN, DONE])
got = asyncio.run(collect(pool))
check("exit: request served by the other replica", got, ["token", "done"])
check("exit: replica quarantined", pool._heal[0].get("quarantined"), True)
check("exit: cluster not degraded", marked, [])

# --- 4. the watchdog restarts a quarantined replica, it does not re-admit it ----
pool = new_pool(WEDGED, [TOKEN, DONE])
asyncio.run(collect(pool))
child = pool._replicas[0]
stops = []


async def fake_stop():
    stops.append("stop")


async def fake_start():
    return None


async def node_ready(_i):
    return True


child.stop, child.start, pool._node_ready = fake_stop, fake_start, node_ready
asyncio.run(pool._heal_tick())
check("heal: backoff respected, not re-admitted", (pool._live, stops), ({1}, []))
pool._heal[0]["next_at"] = 0.0
asyncio.run(pool._heal_tick())
check("heal: live runner is stopped and restarted", stops, ["stop"])
check("heal: back in the pool", pool._live, {0, 1})
check("heal: quarantine lifted", pool._heal[0]["quarantined"], False)
check("heal: restart counted", pool._heal[0]["restarts"], 1)

# --- 5. idempotent, and our own restart is not a new failure --------------------
pool = new_pool(WEDGED, [TOKEN, DONE])
child = pool._replicas[0]
pool._on_child_failed(child, "first")
first = dict(pool._heal[0])
pool._on_child_failed(child, "second")
check("idempotent: first reason kept", pool._heal[0]["last_error"], first["last_error"])
pool = new_pool(WEDGED, [TOKEN, DONE])
pool._heal[0]["restarting"] = True
pool._on_child_failed(pool._replicas[0], "exit from our own stop")
check("restart: own stop ignored", (pool._live, pool._heal[0].get("quarantined")), ({0, 1}, None))
pool = new_pool(WEDGED, [TOKEN, DONE])
pool._stopping = True
pool._on_child_failed(pool._replicas[0], "exit from a pool stop")
check("stop: pool stop ignored", pool._live, {0, 1})

# --- 6. retry reaches the last untried replica ----------------------------------
pool = new_pool(WEDGED, WEDGED, [TOKEN, DONE])
got = asyncio.run(collect(pool))
check("retry: third replica serves after two wedges", got, ["token", "done"])
pool = new_pool(WEDGED, WEDGED)
got = asyncio.run(collect(pool))
check("retry: all wedged raises", got, ["raised: runner wedged mid-generation"])
check("retry: all quarantined, still nothing cluster-wide", (pool._live, marked, resets), (set(), [], []))


# --- 7. a distributed pool keeps its cluster-wide handling ----------------------
async def distributed():
    marked.clear()
    resets.clear()
    rp = api.RunnerPool(model="/m/x", mode="pipeline", use_ap=False, nodes_count=1, cluster="repl",
                        kv_q8=False, draft_model=None, alias="dist", node_indices=[0], mtp=None, batch=False)
    rp.runners = [FakeRunner(rp, WEDGED)]
    try:
        async for _ in rp.submit(prompt="p", max_tokens=4, enable_thinking=False):
            pass
    except RuntimeError as e:
        await asyncio.sleep(0)
        return rp.degraded, str(e)
    return rp.degraded, ""


degraded, message = asyncio.run(distributed())
check("distributed: pool degraded", degraded, True)
check("distributed: cluster degraded", marked, ["gen no-progress watchdog"])
check("distributed: recovery ladder fired", resets, ["repl"])
check("distributed: message unchanged", "recovery triggered" in message, True)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("all checks passed")
