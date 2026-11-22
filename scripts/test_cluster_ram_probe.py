#!/usr/bin/env python3
"""The read-only cluster RAM lookup never blocks on a node it cannot reach.

`_cluster_total_ram_bytes` falls back to a synchronous ssh probe for a node
with no telemetry. `admin_cluster_get` (polled by the dashboard) passes
`probe=False`, so a powered-off node must not reach `_ssh_exec`. The load
paths keep `probe=True` and still probe. No node is reached: `_ssh_exec` is
replaced by a recorder.

    .venv/bin/python scripts/test_cluster_ram_probe.py
"""
import os
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


TOPO = [{"host": "dead-node", "ssh": "admin@192.0.2.1"}]
FAKE_OUT = (0, "196608 274877906944", "")  # wired_limit_mb, hw.memsize (256 GiB)

calls: list = []
real_ssh_exec = api._ssh_exec
real_cache = api._telemetry_cache


def recorder_refusing(ssh_target, cmd, timeout=12):
    calls.append((ssh_target, cmd))
    raise RuntimeError("probe=False must not reach _ssh_exec")


def recorder_fake(ssh_target, cmd, timeout=12):
    calls.append((ssh_target, cmd))
    return FAKE_OUT


try:
    # Empty telemetry: the node has never been probed.
    api._telemetry_cache = {"ts": 0.0, "data": None}

    # probe=False: no ssh, the unknown node counts as 0 RAM.
    api._ssh_exec = recorder_refusing
    total, per_node = api._cluster_total_ram_bytes("x", 1, topo=TOPO, probe=False)
    check("probe=False makes no ssh call", len(calls), 0)
    check("probe=False total is 0", total, 0)
    check("probe=False node ram is 0", per_node[0]["ram_bytes"], 0)
    check("probe=False wired limit is 0", per_node[0]["wired_limit_bytes"], 0)
    check("probe=False host kept", per_node[0]["host"], "dead-node")

    # probe=True: exactly one ssh probe, values taken from its output.
    calls.clear()
    api._ssh_exec = recorder_fake
    total, per_node = api._cluster_total_ram_bytes("x", 1, topo=TOPO, probe=True)
    check("probe=True calls ssh once", len(calls), 1)
    check("probe=True ssh target", calls[0][0] if calls else None, "admin@192.0.2.1")
    check("probe=True ram from hw.memsize", per_node[0]["ram_bytes"], 274877906944)
    check("probe=True wired limit from iogpu (MiB to bytes)",
          per_node[0]["wired_limit_bytes"], 196608 * 1024 * 1024)
    check("probe=True total", total, 274877906944)
finally:
    api._ssh_exec = real_ssh_exec
    api._telemetry_cache = real_cache

print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAIL'}")
sys.exit(1 if FAILS else 0)
