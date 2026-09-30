#!/usr/bin/env python3
"""The Load form must never use a node count the selected cluster does not offer.

f.default_nodes / f.default_nodes_user_picked live in ONE page-wide form. A pick of 4
on Argo survived switching to a 1-node cluster and the Default load POSTed nodes=4:
HTTP 400 "unsupported nodes=4 (available: [1], max=1)" (2026-09-30). resolveDefaultNodes()
now drops a pick outside the selected cluster's range. The function is extracted from
dashboard.html and run under node with a stubbed clusterMaxNodes; no server is touched.

    python3 scripts/test_dashboard_default_nodes.py
"""
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "dashboard.html"), encoding="utf-8", errors="replace").read()

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


node = shutil.which("node")
if not node:
    print("SKIP node not found: the dashboard logic is not run")
    sys.exit(0)

m = re.search(r"function resolveDefaultNodes\(f, cluster, optNodes\) \{.*?\n\}\n", SRC, re.S)
check("resolveDefaultNodes is defined in dashboard.html", bool(m), True)
if not m:
    sys.exit(1)

JS = """
const MAX = {one: 1, two: 2, argo: 5};
function clusterMaxNodes(c) { return Math.max(MAX[c] || 0, 1); }
%s
const cases = JSON.parse(process.argv[1]);
console.log(JSON.stringify(cases.map(c => {
  const f = {...c.f};
  const out = resolveDefaultNodes(f, c.cluster, c.opt);
  return {out, nodes: f.default_nodes, picked: !!f.default_nodes_user_picked};
})));
""" % m.group(0)

# (label, form, cluster, optimal nodes or None, expected result, expected user_picked)
CASES = [
    ("a pick of 4 made on Argo is dropped on a 1-node cluster",
     {"default_nodes": 4, "default_nodes_user_picked": True}, "one", 1, 1, False),
    ("... also while the capacity data of that cluster has not arrived (no optimal yet)",
     {"default_nodes": 4, "default_nodes_user_picked": True}, "one", None, 1, False),
    ("a fresh form (3) on a 1-node cluster resolves to 1",
     {"default_nodes": 3, "default_nodes_user_picked": False}, "one", None, 1, False),
    ("a valid manual pick is kept",
     {"default_nodes": 4, "default_nodes_user_picked": True}, "argo", 2, 4, True),
    ("no manual pick: the smallest topology that fits wins",
     {"default_nodes": 3, "default_nodes_user_picked": False}, "argo", 2, 2, False),
    ("no manual pick and no capacity data: the form value stays on a wide cluster",
     {"default_nodes": 3, "default_nodes_user_picked": False}, "argo", None, 3, False),
    ("no capacity data on a 2-node cluster: 3 is out of range, so 2",
     {"default_nodes": 3, "default_nodes_user_picked": False}, "two", None, 2, False),
    ("an unusable pick (parseInt gave NaN) is out of range",
     {"default_nodes": None, "default_nodes_user_picked": True}, "argo", 2, 2, False),
]

run = subprocess.run([node, "-e", JS, json.dumps([
    {"f": f, "cluster": cl, "opt": opt} for _, f, cl, opt, _, _ in CASES])],
    capture_output=True, text=True, timeout=30)
if run.returncode != 0:
    print("FAIL node error:", run.stderr.strip()[:400])
    sys.exit(1)
for (label, _, _, _, want, want_picked), got in zip(CASES, json.loads(run.stdout)):
    check(label, (got["out"], got["nodes"], got["picked"]), (want, want, want_picked))

# The form value is what the Default load POSTs, so it is the contract this protects.
check("renderLoadConfig resolves the node count through the helper",
      "const selectedNodes = resolveDefaultNodes(f, state.selected_cluster, optNodes);" in SRC, True)
check("the Default load still POSTs f.default_nodes",
      "nodes: f.default_nodes, kv_q8: f.default_kv_q8" in SRC, True)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
