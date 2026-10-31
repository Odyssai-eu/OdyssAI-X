#!/usr/bin/env python3
"""Saved pools + autostart (1.65.0): Save snapshots the live pools, Load and every
automatic load (restart, reboot-all, crash self-heal) use that snapshot, and
autostart off loads nothing on its own. Also the node-versions mismatch flag.
No node is reached: RunnerPool, ssh and the registry are stubbed.

    python3 scripts/test_saved_pools.py
"""
import asyncio
import json
import os
import sys
import tempfile

os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({"argo": {"name": "A", "kind": "mlx-distributed", "backend": "jaccl",
                    "nodes": [{"host": f"n{i}", "ssh": f"admin@198.51.100.{i + 1}", "master": i == 0}
                              for i in range(3)]}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []
CID = "argo"


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


built = []


class FakeRunnerPool:
    def __init__(self, **kw):
        self.alias = kw.get("alias")

    async def start(self):
        built.append(self.alias)


async def reachable(_ssh):
    return True

api.RunnerPool = FakeRunnerPool
api._node_reachable = reachable
api._cluster_is_degraded = lambda cid: False
api.set_pool = lambda cid, alias, pool: None
api.save_cluster_state_v2 = lambda cid, **kw: None

desired = [
    {"alias": "default", "model": "/m/big", "mode": "pipeline", "node_indices": [0, 1, 2]},
    {"alias": "old", "model": "/m/old", "mode": "pipeline", "node_indices": [0], "down": True},
]
api.load_cluster_state_v2 = lambda cid: [dict(e) for e in desired]
live = {"default"}
api.list_pools = lambda cid: [(a, object()) for a in sorted(live)]

# Seeding: no file yet -> the pools live at the last state-save, never saved.
sp = api.load_saved_pools(CID)
check("no file: seeded from the non-down desired entries",
      [p["alias"] for p in sp["pools"]], ["default"])
check("seeded set has no saved_at", sp["saved_at"], None)
check("seed is written once", api.saved_pools_file_for(CID).exists(), True)

# Autostart inherits reload_on_restart until set.
check("autostart defaults on", api._autostart(CID), True)
api.save_cluster_def(CID, {"reload_on_restart": False})
check("autostart follows an old reload_on_restart=false", api._autostart(CID), False)
api.save_cluster_def(CID, {"autostart": True})
check("an explicit autostart wins over reload_on_restart", api._autostart(CID), True)

# Save: only live aliases, `down` stripped.
desired.append({"alias": "coder", "model": "/m/coder", "mode": "pipeline", "node_indices": [1]})
live = {"default", "coder"}
res = asyncio.run(api.admin_saved_pools_save(CID))
check("save keeps the live pools only", sorted(p["alias"] for p in res["pools"]), ["coder", "default"])
check("save stamps saved_at", isinstance(res["saved_at"], float), True)
check("saved entries carry no down marker",
      any("down" in p for p in api.load_saved_pools(CID)["pools"]), False)

# Restore with entries= uses them, not the desired state.
api.get_pool = lambda cid, alias=None: None
built.clear()
asyncio.run(api._restore_cluster_pools(CID, entries=api.load_saved_pools(CID)["pools"]))
check("restore loads the saved set", sorted(built), ["coder", "default"])

# Crash self-heal: autostart on -> saved aliases only; off -> nothing, pending cleared.
live = set()
built.clear()
api._AUTO_RELOAD_PENDING.clear(); api._AUTO_RELOAD_RETRIES.clear()
asyncio.run(api._auto_reload_purged(CID, ["default", "old"]))
check("self-heal reloads a saved alias, not an unsaved one", built, ["default"])
api.save_cluster_def(CID, {"autostart": False})
built.clear()
api._AUTO_RELOAD_PENDING.clear()
asyncio.run(api._auto_reload_purged(CID, ["default"]))
check("autostart off: no self-heal load", built, [])
check("autostart off: nothing left pending",
      [k for k in api._AUTO_RELOAD_PENDING if k[0] == CID], [])

# Node versions: parsing + mismatch.
outs = {
    "admin@198.51.100.1": "macos=26.5\nmlx=0.32.2\nmlx-vlm=0.4.1\n",
    "admin@198.51.100.2": "macos=26.5\nmlx=0.32.2\nmlx-vlm=\n",
    "admin@198.51.100.3": "macos=26.4.1\nmlx=0.32.2\nmlx-vlm=0.4.1\n",
}
api._ssh_exec = lambda ssh, cmd, timeout=12: (0, outs[ssh], "")
v = asyncio.run(api.admin_cluster_node_versions(CID, fresh=True))
check("versions parsed", [(n["macos"], n["mlx"], n["mlx_vlm"]) for n in v["nodes"]],
      [("26.5", "0.32.2", "0.4.1"), ("26.5", "0.32.2", None), ("26.4.1", "0.32.2", "0.4.1")])
check("mismatch: macOS and a missing mlx-vlm, not mlx", v["mismatch"], ["macos", "mlx_vlm"])

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
