#!/usr/bin/env python3
"""Dashboard-created (overlay) clusters restore their pools at startup (#70).

The 2026-08-27 symptom: restart restored only topology.yaml clusters — the
boot loop iterated the topology seed, not the merged view. The fix routes the
startup restore through active_cluster_ids() (union of topology + overlay,
non-cluster sections and tombstones filtered). This test pins the three
properties of that function; the live proof is the 2026-10-07/08 restarts
restoring `amigos` — a dashboard-created cluster absent from topology.yaml —
with no intervention. No node is reached.

    .venv/bin/python scripts/test_overlay_cluster_restore.py
"""
import json
import os
import sys
import tempfile

CFG = os.path.join(tempfile.mkdtemp(), "cc.json")
os.environ["CLUSTER_CONFIG_FILE"] = CFG
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
# The topology seed itself comes from topology.py defaults; the overlay below
# adds a dashboard cluster, a NON-cluster section, and a tombstoned one.
json.dump({
    "amigos": {"kind": "replica", "nodes": [{"host": "n0", "ssh": "admin@198.51.100.1"}]},
    "settings": {"enable_thinking_default": True},          # not a cluster
    "crew": ["a", "b"],                                     # not a cluster
    "dead-cluster": {"kind": "replica", "_removed": True},  # tombstoned
}, open(CFG, "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


ids = api.active_cluster_ids()

check("un cluster overlay (dashboard) est dans la liste de restauration",
      "amigos" in ids, True)
check("les sections non-cluster (settings, crew) ne sont PAS des clusters",
      ("settings" in ids, "crew" in ids), (False, False))
check("les tombstones restent exclues", "dead-cluster" in ids, False)
check("les clusters topology restent inclus (union)",
      any(c in ids for c in api.DEFAULT_CLUSTER_DEFS), True)
check("_reload_on_restart lit la définition FUSIONNÉE (l'overlay compte)",
      api._reload_on_restart("amigos"), True)

# The boot loop's contract: restore iterates the union — pinned by source
# guard so a revert to a topology-only iteration fails loudly.
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.py")).read()
check("le restore au démarrage itère active_cluster_ids() (pas la seule topology)",
      "for cid in active_cluster_ids():" in src, True)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
