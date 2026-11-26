#!/usr/bin/env python3
"""A single-node pool served by an external server dies with its node and comes back.

VLMPool, DecisionPool, DFlashPool and InklingPool reported alive_count() == 1
forever, so after a node reboot the pool stayed "loaded" behind a 502 until a
manual reload (max-64 and ultra-96c, 2026-11-26). The sweeper now probes their
port: closed on two consecutive ticks marks the pool dead, the purge drops it
and auto-reload restores it through _restore_cluster_pools. Real sockets on
127.0.0.1, no node reached (stop, ssh reachability and the restore are stubbed).

    python3 scripts/test_external_pool_liveness.py
"""
import asyncio
import json
import os
import socket
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({"dm": {"name": "D", "kind": "decision", "backend": "ring",
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


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def decision_pool(port):
    p = api.DecisionPool(model_path="/m/d3", cluster="dm", alias="d3", node_indices=[0],
                         upstream=f"http://127.0.0.1:{port}", port=port,
                         ssh_target="admin@198.51.100.1", host="n0")
    async def no_stop():
        pass
    p.stop = no_stop
    return p


async def main():
    # A listening server: never marked dead, whatever the number of ticks.
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(); up_port = srv.getsockname()[1]
    api.set_pool("dm", "d3", decision_pool(up_port))
    for _ in range(3):
        await api._probe_external_pools()
    check("listening server: alive", api.get_pool("dm", "d3").alive_count(), 1)
    srv.close()
    api.del_pool("dm", "d3")

    # A closed port: alive after one tick, dead after the second.
    pool = decision_pool(free_port())
    api.set_pool("dm", "d3", pool)
    await api._probe_external_pools()
    check("closed port, one tick: still alive", pool.alive_count(), 1)
    await api._probe_external_pools()
    check("closed port, two ticks: dead", pool.alive_count(), 0)

    # The purge drops it, auto-reload restores it through the startup path.
    entry = {"alias": "d3", "model": "/m/d3", "is_decision": True, "node_indices": [0],
             "port": pool.port, "upstream": pool.upstream}
    api.save_cluster_state_v2 = lambda cid, **k: None
    api.load_cluster_state_v2 = lambda cid: [entry]
    api.load_saved_pools = lambda cid: {"saved_at": None, "pools": [entry]}
    api._cluster_is_degraded = lambda cid: False
    api._autostart = lambda cid: True

    async def reachable(_ssh):
        return True
    api._node_reachable = reachable
    restored = []

    async def fake_restore(cid, leaked_hosts=None, restore_down=True, entries=None):
        for e in entries:
            restored.append(e["alias"])
            api.set_pool(cid, e["alias"], decision_pool(free_port()))
        return [e["alias"] for e in entries]
    api._restore_cluster_pools = fake_restore

    class NoRunner:
        def __init__(self, **k):
            raise AssertionError("a decision pool was rebuilt as a text RunnerPool")
    api.RunnerPool = NoRunner

    purged = await api._purge_dead_pools("dm")
    check("dead pool purged", purged, ["d3"])
    await api._auto_reload_purged("dm", purged)
    check("restored through _restore_cluster_pools", restored, ["d3"])
    check("registered again", api.get_pool("dm", "d3") is not None, True)
    check("nothing left pending", ("dm", "d3") in api._AUTO_RELOAD_PENDING, False)


asyncio.run(main())
print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
