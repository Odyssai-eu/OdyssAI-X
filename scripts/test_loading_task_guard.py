#!/usr/bin/env python3
"""The loading flag is never stale-cleared while its load task is alive.

2026-10-06, GLM-5.3-Flash Q6h16: estimate 21 s for 66 GB, real load past
2 min (cold weights). The time-only rule in _loading_snapshot (clear past
max(5x est, 120 s)) fired mid-load: the dashboard's loading state vanished
while the runners were still working. The fix: _begin_loading records a
weakref to the load request task; _loading_snapshot only applies the time
rule once that task is gone (or was never recorded — pre-deploy states). No
node is reached.

    .venv/bin/python scripts/test_loading_task_guard.py
"""
import asyncio
import json
import os
import sys
import tempfile
import time

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({"argo": {"name": "A", "kind": "mlx-distributed", "backend": "ring",
                    "models_dir": "/m",
                    "nodes": [{"host": "n0", "ssh": "admin@198.51.100.1", "master": True}]}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


async def scenario():
    state = api._loading_state_for("argo")
    api._begin_loading(state, "GLM-5.3-Flash-Q6h16", 2, 66 * 1024 ** 3, 21.0)
    # The 2026-10-06 situation: past every time bound, but the load task
    # (this coroutine) is still alive.
    state["started_at"] = time.time() - 9999.0
    snap = api._loading_snapshot(state)
    check("a live load task keeps the flag (snapshot still shows the load)",
          snap is not None and state["in_progress"], True)
    check("the live load still reports progress (not stuck at 0)",
          isinstance(snap.get("loaded_bytes"), (int, float)) or "progress" in json.dumps(snap), True)
    return state


state = asyncio.run(scenario())
# The load task is done now (asyncio.run returned): the same stale flag is
# clearable again — this is the leaked-flag path the rule exists for.
snap = api._loading_snapshot(state)
check("once the load task is gone, a stale flag is cleared again",
      snap is None and not state["in_progress"], True)

# A state with no task recorded (created before the fix, or by a sync caller)
# keeps the old behaviour: stale → cleared.
legacy = {"in_progress": True, "model": "m", "nodes": 1, "size_bytes": 1,
          "estimated_s": 1.0, "started_at": time.time() - 9999.0}
check("a legacy state without a task ref still self-heals",
      api._loading_snapshot(legacy) is None and not legacy["in_progress"], True)

# A young flag is never cleared, task or not.
fresh = {"in_progress": True, "model": "m", "nodes": 1, "size_bytes": 1,
         "estimated_s": 30.0, "started_at": time.time()}
check("a young flag is never cleared", api._loading_snapshot(fresh) is not None, True)

# _end_loading drops the task handle with everything else.
state2 = api._loading_state_for("argo2")
api._begin_loading(state2, "m", 1, 1, 1.0)
api._end_loading(state2)
check("_end_loading clears the task handle too", "_task_ref" in state2, False)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
