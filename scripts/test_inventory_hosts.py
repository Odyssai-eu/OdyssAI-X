#!/usr/bin/env python3
"""Hosts of dashboard-added clusters are part of the host inventory.

The "+ Add node" list read KNOWN_HOSTS, built from topology.yaml only: the 96 GB Macs,
declared only in dashboard-added clusters (cluster-config.json), never appeared in it,
so no cluster or replica could be composed with them (2026-09-30). No node is reached.

    python3 scripts/test_inventory_hosts.py
"""
import asyncio
import json
import os
import sys
import tempfile

os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
os.environ.setdefault("ODYSSAI_X_STATE_DIR", tempfile.mkdtemp())
json.dump({"solo-96c": {"name": "solo-96c", "kind": "mlx-distributed", "backend": "ring",
                        "max_nodes": 1,
                        "nodes": [{"host": "ultra-96c", "ssh": "admin@198.51.100.62", "master": True}]}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


inv = asyncio.run(api.admin_inventory())["hosts"]
ids = [h["id"] for h in inv]
check("a host of a dashboard-added cluster is in the inventory", "ultra-96c" in ids, True)
check("... with its ssh target", next(h["ssh"] for h in inv if h["id"] == "ultra-96c"), "admin@198.51.100.62")
check("the topology hosts are still listed first", ids[:len(api.KNOWN_HOSTS)], [h["id"] for h in api.KNOWN_HOSTS])
check("no host is listed twice", len(ids), len(set(ids)))

# The dashboard sends the ssh taken from the inventory; an API caller may send the host id
# only, and a single-node ring cluster then gets its ssh from the same inventory.
nodes = [{"host": "ultra-96c", "master": True}]
err = api.validate_cluster_def("solo", {"name": "solo", "kind": "mlx-distributed", "backend": "ring", "nodes": nodes})
check("a node given by id only validates", err, None)
check("... and gets its ssh filled", nodes[0].get("ssh"), "admin@198.51.100.62")
check("an unknown host is still refused", api.validate_cluster_def("x",
    {"name": "x", "kind": "mlx-distributed", "backend": "ring", "nodes": [{"host": "nope", "master": True}]}),
    "unknown host: nope")

# The recovery ladder resolves hosts added after the engine started.
cc = json.load(open(os.environ["CLUSTER_CONFIG_FILE"]))
cc["solo-96d"] = {"name": "solo-96d", "kind": "mlx-distributed", "backend": "ring", "max_nodes": 1,
                  "nodes": [{"host": "ultra-96d", "ssh": "admin@198.51.100.63", "master": True}]}
api._save_cluster_config(cc)   # what a dashboard edit does (refreshes the config cache)
check("a host added after start is resolved for reboot", (api._resolve_host("ultra-96d") or {}).get("ssh"),
      "admin@198.51.100.63")

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
