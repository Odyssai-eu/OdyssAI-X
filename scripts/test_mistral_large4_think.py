#!/usr/bin/env python3
"""Mistral-Large-4's [THINK]…[/THINK] goes to reasoning_content, not content.

2026-10-30, Argo pipeline-5: "[THINK]17 × 23 = 391.[/THINK]17 × 23 = 391." came
back whole in `content` — the engine only knew <think>…</think>. The model
writes its own open tag (the template prefills nothing) and has no
enable_thinking switch; its template raises on any reasoning_effort outside
low/medium/high. No node is reached.

    .venv/bin/python scripts/test_mistral_large4_think.py
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
MODEL = "/Volumes/models/odysseus/odyssai/Mistral-Large-4-1T-A52B-textonly-Q6h16-gs32"
OUT = "[THINK]17 × 23 = 391.[/THINK]17 × 23 = 391."


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


def split(text, chunk, enable_thinking):
    """The streaming path's filter, fed `chunk` characters at a time."""
    if not api._should_filter_think(MODEL, enable_thinking):
        return text, ""
    open_m, close_m = api._model_think_markers(MODEL)
    st = {"in_think": api._seed_in_think(MODEL, enable_thinking), "carry": "",
          "open": open_m, "close": close_m, "strip": api._model_think_strips(MODEL)}
    vis, rea = "", ""
    for i in range(0, len(text), chunk):
        v, r = api._split_think_stream(text[i:i + chunk], st)
        vis, rea = vis + v, rea + r
    v, r = api._flush_think_stream(st)
    return vis + v, rea + r


for et in (None, True, False):
    for chunk in (1, 3, len(OUT)):
        check(f"enable_thinking={et} chunk={chunk}: split",
              split(OUT, chunk, et), ("17 × 23 = 391.", "17 × 23 = 391."))
check("an answer with no [THINK] block stays whole in content",
      split("Bonjour.", 2, None), ("Bonjour.", ""))
for given, want in (("minimal", "low"), ("medium", "medium"), ("xhigh", "high"),
                    ("max", "high"), ("high", "high")):
    check(f"reasoning_effort {given} -> {want}", api._map_reasoning_effort(MODEL, given), want)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
