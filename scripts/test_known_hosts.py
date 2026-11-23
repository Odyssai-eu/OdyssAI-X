#!/usr/bin/env python3
"""A host removed from its last cluster stays in the "+ Add node" inventory.

The inventory was the nodes of the current clusters only. Saving the cluster
config now records every host a cluster has had in `known_hosts`, and
`_inventory_hosts` lists them.

    .venv/bin/python scripts/test_known_hosts.py
"""
import os
import sys
import tempfile

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


def inventory_ssh(host_id):
    return next((h["ssh"] for h in api._inventory_hosts() if h["id"] == host_id), None)


A = {"host": "node-a", "ssh": "admin@192.0.2.1", "master": True}
B = {"host": "node-b", "ssh": "admin@192.0.2.2", "master": False}

api._save_cluster_config({"c1": {"kind": "replica", "nodes": [A, B]},
                          "settings": {"x": 1}, "crew": [{"id": "t"}]})
check("a member host is in the inventory", inventory_ssh("node-b"), "admin@192.0.2.2")

# Drop node-b from its only cluster, without known_hosts in the saved dict:
# the on-disk entry is the only place that still names it.
api._save_cluster_config({"c1": {"kind": "replica", "nodes": [A]}})
check("a removed host stays in the inventory", inventory_ssh("node-b"), "admin@192.0.2.2")

with api._cluster_config_txn() as cfg:
    cfg["c1"] = {"_removed": True}
check("hosts of a deleted cluster stay", inventory_ssh("node-a"), "admin@192.0.2.1")

with api._cluster_config_txn() as cfg:
    cfg["c2"] = {"kind": "replica", "nodes": [dict(B, ssh="admin@192.0.2.9", master=True)]}
check("a current member's ssh wins over the remembered one",
      inventory_ssh("node-b"), "admin@192.0.2.9")
check("non-cluster keys are not read as hosts",
      sorted(api._load_cluster_config()["known_hosts"]), ["node-a", "node-b"])

# A node named by id only gets its ssh from the inventory, for every kind:
# replica and decision returned before the fill and persisted ssh=None.
for kind in ("replica", "decision", "mlx-distributed"):
    d = {"kind": kind, "backend": "ring",
         "nodes": [{"host": "node-a", "master": True}]}
    err = api.validate_cluster_def("c9", d)
    check(f"{kind}: id-only node validates", err, None)
    check(f"{kind}: id-only node gets its ssh", d["nodes"][0].get("ssh"), "admin@192.0.2.1")
check("unknown id-only host is refused",
      api.validate_cluster_def("c9", {"kind": "replica", "backend": "ring",
                                      "nodes": [{"host": "nope", "master": True}]}),
      "unknown host: nope")

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
