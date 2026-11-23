#!/usr/bin/env python3
"""The VLM replica pool carries its speculative drafter to every node.

`--draft-model` is appended by `_vlm_launch_cmd` only when a drafter is given.
A VLMReplicaPool built with draft_model must hand it to each child launch
(_start_child, the watchdog relaunch path) and to each child VLMPool, and the
replica load path must pass the resolved drafter to the pool. Source-level
checks, no node is reached.

    .venv/bin/python scripts/test_vlm_replica_draft.py
"""
import inspect
import os
import re
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


def call_texts(src: str, name: str) -> list[str]:
    """Full text of every `name(...)` call in src, parentheses matched."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth = 0
        for j in range(m.end() - 1, len(src)):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    out.append(src[m.start():j + 1])
                    break
    return out


SRC = open(api.__file__, encoding="utf-8").read()

# (a) _vlm_launch_cmd: --draft-model only when a drafter is given.
DRAFT = "/models/odyssai/drafter-dflash"
with_draft = api._vlm_launch_cmd("vid", "/venv", "/models/m", 8000, draft_model=DRAFT)
no_draft = api._vlm_launch_cmd("vid", "/venv", "/models/m", 8000)
check("launch cmd has --draft-model when given", "--draft-model" in with_draft, True)
check("launch cmd carries the drafter path", f"--draft-model {DRAFT}" in with_draft, True)
check("launch cmd has no --draft-model when not given", "--draft-model" in no_draft, False)

# VLMReplicaPool.__init__ accepts draft_model.
sig = inspect.signature(api.VLMReplicaPool.__init__)
check("VLMReplicaPool accepts draft_model", "draft_model" in sig.parameters, True)
check("VLMReplicaPool draft_model defaults to None",
      sig.parameters["draft_model"].default if "draft_model" in sig.parameters else "missing",
      None)

# (b) every _launch_vlm_server( call inside the VLMReplicaPool class body
# passes draft_model=self.draft_model.
cls_start = SRC.index("\nclass VLMReplicaPool:")
cls_end = SRC.index("\nclass ", cls_start + 1)
cls_body = SRC[cls_start:cls_end]
launches = call_texts(cls_body, "_launch_vlm_server")
check("VLMReplicaPool has _launch_vlm_server call sites", len(launches) >= 1, True)
for i, text in enumerate(launches):
    check(f"VLMReplicaPool launch call {i + 1} passes draft_model=self.draft_model",
          "draft_model=self.draft_model" in text, True)

# Each child VLMPool built by VLMReplicaPool gets the drafter too.
children = call_texts(cls_body, "VLMPool")
children = [c for c in children if "model_path=" in c]
check("VLMReplicaPool builds child VLMPool", len(children) >= 1, True)
for i, text in enumerate(children):
    check(f"child VLMPool {i + 1} passes draft_model=draft_model",
          "draft_model=draft_model" in text, True)

# (c) _load_vlm_replica passes draft_model= to VLMReplicaPool( and returns it.
load_src = inspect.getsource(api._load_vlm_replica)
rep_calls = call_texts(load_src, "VLMReplicaPool")
check("_load_vlm_replica builds VLMReplicaPool", len(rep_calls), 1)
check("_load_vlm_replica passes draft_model= to VLMReplicaPool",
      "draft_model=" in rep_calls[0] if rep_calls else False, True)
check("_load_vlm_replica returns draft_model", '"draft_model"' in load_src, True)

# (d) persisted state and startup restore carry the drafter.
check("save block writes draft_model for vlm replica",
      '"draft_model": getattr(pool, "draft_model", None),' in SRC, True)
restore_idx = SRC.index('if entry.get("is_vlm_replica"):')
restore_block = SRC[restore_idx:restore_idx + 800]
check("restore passes draft_model=entry.get(\"draft_model\")",
      'draft_model=entry.get("draft_model")' in restore_block, True)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: " + "; ".join(FAILS))
    sys.exit(1)
print("all OK")
