#!/usr/bin/env python3
"""/admin/nodes/telemetry never makes the dashboard wait on a dead node once warm.

2026-11-03: four minis off the network, each ssh probe hit the 4 s connect
timeout, so every probe round outlasted the 4 s TTL. Every dashboard poll missed
the cache and started its own round; the rounds piled up on the thread pool and
the page took 20 s to show. Now one round runs at a time, and a stale snapshot
is served at once while the next round runs. No node is reached.

    python3 scripts/test_telemetry_cache.py
"""
import asyncio
import json
import os
import sys
import tempfile
import time

os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({}, open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []
PROBE_S = 0.3
probes = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


async def slow_probe(host):
    probes.append(host["id"])
    await asyncio.sleep(PROBE_S)
    return {"id": host["id"], "round": len(probes)}


api.HOSTS_REGISTRY = [{"id": "a", "ssh": "admin@198.51.100.1"}]
api._probe_telemetry = slow_probe


async def main():
    # Cold cache: five concurrent polls share one probe round.
    res = await asyncio.gather(*(api.admin_nodes_telemetry() for _ in range(5)))
    check("cold: one probe round for five concurrent polls", len(probes), 1)
    check("cold: every poll gets the snapshot", all(r["hosts"] for r in res), True)

    # Warm and fresh: served from the cache, no probe.
    await api.admin_nodes_telemetry()
    check("warm: no new probe inside the TTL", len(probes), 1)

    # Stale: answered at once with the old snapshot, one refresh behind it.
    api._telemetry_cache["ts"] -= api._TELEMETRY_TTL_S + 1
    t0 = time.perf_counter()
    res = await asyncio.gather(*(api.admin_nodes_telemetry() for _ in range(5)))
    check("stale: answered without waiting for the probe", time.perf_counter() - t0 < PROBE_S / 3, True)
    check("stale: the old snapshot is served", res[0]["hosts"][0]["round"], 1)
    await asyncio.sleep(PROBE_S * 2)
    check("stale: a single refresh ran", len(probes), 2)
    res = await api.admin_nodes_telemetry()
    check("after the refresh: the new snapshot is served", res["hosts"][0]["round"], 2)

    # ?fresh=1 waits for a probe, and shares one already running.
    api._telemetry_cache["ts"] -= api._TELEMETRY_TTL_S + 1
    await api.admin_nodes_telemetry()          # starts a refresh
    res = await api.admin_nodes_telemetry(fresh=True)
    check("fresh joins the running refresh", len(probes), 3)
    check("fresh returns the new snapshot", res["hosts"][0]["round"], 3)


asyncio.run(main())
print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
