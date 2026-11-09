#!/usr/bin/env python3
"""VL pool speculative drafter (Muse-Glimmer + its DFlash assistant, 1.65.12):
`draft` on a VL load reaches mlx_vlm.server as --draft-model, the pool keeps it,
the cluster state persists it and the restore relaunches with it. No node is
reached.

    .venv/bin/python scripts/test_vlm_drafter.py
"""
import json
import os
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({"c": {"name": "C", "models_dir": "/m",
                 "nodes": [{"host": "n0", "ssh": "admin@198.51.100.1", "master": True}]},
           "model_drafters": {"/m/glimmer": {"draft": "/m/glimmer-assistant"}}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


cmd = api._vlm_launch_cmd("t", "/v", "/m/glimmer", 8080, draft_model="/m/glimmer assistant")
check("drafter → --draft-model quoted, before the redirect",
      "--draft-model '/m/glimmer assistant' > " in cmd, True)
check("no --draft-kind: mlx_vlm detects it from the drafter's model_type",
      "--draft-kind" in cmd, False)
check("no drafter → no flag",
      "--draft-model" in api._vlm_launch_cmd("t", "/v", "/m/x", 8080), False)

pool = api.VLMPool(model_path="/m/glimmer", cluster="c", alias="g", node_indices=[0],
                   upstream="http://198.51.100.1:8080", port=8080,
                   ssh_target="admin@198.51.100.1", host="n0", draft_model="/m/glimmer-assistant")
check("VLMPool keeps its drafter", pool.draft_model, "/m/glimmer-assistant")
check("VLMPool without drafter", api.VLMPool(
    model_path="/m/x", cluster="c", alias="x", node_indices=[0], upstream="http://h:1",
    port=1, ssh_target="admin@198.51.100.1", host="n0").draft_model, None)

check("model_drafters entry without dflash is found",
      api._drafter_for_model("/m/glimmer"), {"draft": "/m/glimmer-assistant"})

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.py")).read()
check("the state persists the drafter",
      '"draft_model": getattr(pool, "draft_model", None),' in src, True)
check("the restore relaunches with it",
      src.count('draft_model=entry.get("draft_model")') >= 2, True)
check("the native muse_glimmer_server branch is gone",
      "MuseGlimmerPool" in src or "_launch_muse_glimmer_server" in src, False)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
