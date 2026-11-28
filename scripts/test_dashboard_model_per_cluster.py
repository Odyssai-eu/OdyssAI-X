#!/usr/bin/env python3
"""The Load form must never send a model picked on another cluster.

f.default_model lives in ONE page-wide form. A model picked on the decider cluster
(mini-16c) survived switching to Melenas: the picker, finding no matching option,
showed Melenas' first model (Nanbeige4.3-33B-Base) while the Default load POSTed
the mini-16c one: "REFUSE - ce modele n'est pas sur ce cluster (absent de ultra-96c) ;
il est sur : mini-16c" (2026-11-28). pickCluster() now drops the choice on a cluster
change. The function is extracted from dashboard.html and run under node with a
stubbed page; no server is touched.

    python3 scripts/test_dashboard_model_per_cluster.py
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

m = re.search(r"function pickCluster\(id\) \{.*?\n\}\n", SRC, re.S)
check("pickCluster is defined in dashboard.html", bool(m), True)
if not m:
    sys.exit(1)

JS = """
const localStorage = {setItem() {}};
const _logsES = null;
function _startLogs() {}
function render() {}
const state = {
  selected_cluster: "decider",
  cluster_models: {decider: [{id: "/m/Qwen3Guard-Gen-8B"}],
                   melenas: [{id: "/m/Nanbeige4.3-33B-Base"}, {id: "/m/Nanbeige4.3-33B-FP8"}]},
  default_models: [],
  pending_form: {default_model: "/m/Qwen3Guard-Gen-8B", default_draft: "/m/drafter", default_dflash: true},
};
%s
// What the Default load POSTs (same expression as the default-load action).
const sent = () => state.pending_form.default_model || (state.default_models[0] && state.default_models[0].id);
const out = {};
pickCluster("decider");
out.same = sent();
pickCluster("melenas");
out.switched = sent();
out.draft = state.pending_form.default_draft;
out.dflash = state.pending_form.default_dflash;
state.pending_form.default_model = "/m/Nanbeige4.3-33B-FP8";
pickCluster("melenas");
out.kept = sent();
console.log(JSON.stringify(out));
""" % m.group(0)

run = subprocess.run([node, "-e", JS], capture_output=True, text=True, timeout=30)
if run.returncode != 0:
    print("FAIL node error:", run.stderr.strip()[:400])
    sys.exit(1)
got = json.loads(run.stdout)
check("re-picking the same cluster keeps the model choice", got["same"], "/m/Qwen3Guard-Gen-8B")
check("switching cluster sends that cluster's first model, the one the picker shows",
      got["switched"], "/m/Nanbeige4.3-33B-Base")
check("the drafter of the previous model does not ride along", (got["draft"], got["dflash"]), ("", False))
check("a model picked on the new cluster is kept", got["kept"], "/m/Nanbeige4.3-33B-FP8")

# The expression above is the contract: keep the test honest if the action changes.
check("the Default load still sends f.default_model, else the first listed model",
      "const model = f.default_model || (state.default_models[0] && state.default_models[0].id);" in SRC, True)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
