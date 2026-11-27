#!/usr/bin/env python3
"""Ling-3.2-air: the template prefills <think>, the reasoning goes to reasoning_content.

2026-11-27, pool `ling-3-2-air-q8h16-gs32`: with thinking on the engine returned
"17 * 23 = ... = 391.</think>391" in `content` and an empty reasoning_content
(TMB witness failed). Only "ling-3.1" / "ling-3-1" were known. No node is reached.

    .venv/bin/python scripts/test_ling32_think.py
"""
import json
import os
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({}, open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    if got != want:
        FAILS.append(f"{label}: got {got!r}, want {want!r}")


for model in ("ling-3-2-air-q8h16-gs32", "/Volumes/models/odyssai/Ling-3.2-air-Q8h16-gs32"):
    check(f"{model} auto-opens think", api._model_auto_opens_think(model), True)
    check(f"{model} filter, thinking on", api._should_filter_think(model, True), True)
    check(f"{model} filter, thinking unset", api._should_filter_think(model, None), True)
    # The flag is honoured: thinking off emits no block, so no filter (else the answer ghosts).
    check(f"{model} filter, thinking off", api._should_filter_think(model, False), False)
    check(f"{model} seeds inside the block", api._seed_in_think(model, True), True)
    check(f"{model} markers", api._model_think_markers(model), ("<think>", "</think>"))

# The stream measured on the pool, cut mid-tag: reasoning and answer split, no tag left.
model = "ling-3-2-air-q8h16-gs32"
state = {"in_think": api._seed_in_think(model, True), "carry": ""}
reasoning, visible = [], []
for chunk in ("17 * 23 = 17*20 + 17*3 = 340 + 51 = 391.</th", "ink>391"):
    v, r = api._split_think_stream(chunk, state)
    visible.append(v)
    reasoning.append(r)
v, r = api._flush_think_stream(state)
visible.append(v)
reasoning.append(r)
check("reasoning", "".join(reasoning), "17 * 23 = 17*20 + 17*3 = 340 + 51 = 391.")
check("content", "".join(visible), "391")

# A sibling that must NOT match: Inkling writes its own open tag.
check("inkling-2-mini keeps its own markers", api._model_think_markers("inkling-2-mini")[0], "<|content_thinking|>")

if FAILS:
    print("\n".join(FAILS))
    sys.exit(1)
print("ok")
