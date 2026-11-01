#!/usr/bin/env python3
"""K2-Horizon: <ifm|think…> goes to reasoning_content, and a multi-turn history
renders instead of killing the rank.

2026-11-01, Argo: the template prefills <ifm|think> (effort high), the model
wrote "…</ifm|think>391" and all of it landed in `content`. A history with an
assistant turn and no thinking field raised in the template and killed rank 0.
medium/low use <ifm|think_fast> / <ifm|think_faster>. No node is reached.

    .venv/bin/python scripts/test_k2_horizon_think.py
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

# runner.py imports mlx at module level: lift the one helper out of its source.
import ast  # noqa: E402
import types  # noqa: E402
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")).read()
_keep = [n for n in ast.parse(_src).body
         if (isinstance(n, ast.FunctionDef) and n.name == "_pad_thinking_fields")
         or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_THINKING_FIELDS"
                                               for t in n.targets))]
runner = types.SimpleNamespace()
exec(compile(ast.Module(body=_keep, type_ignores=[]), "runner.py", "exec"), runner.__dict__)

FAILS = []
MODEL = "/Volumes/models/odysseus/odyssai/K2-Horizon-375B-A23B-Q6h16-gs32"


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


def split(text, chunk, effort):
    if not api._should_filter_think(MODEL, None):
        return text, ""
    open_m, close_m = api._model_think_markers(MODEL, effort)
    st = {"in_think": api._seed_in_think(MODEL, None), "carry": "",
          "open": open_m, "close": close_m, "strip": api._model_think_strips(MODEL)}
    vis, rea = "", ""
    for i in range(0, len(text), chunk):
        v, r = api._split_think_stream(text[i:i + chunk], st)
        vis, rea = vis + v, rea + r
    v, r = api._flush_think_stream(st)
    return vis + v, rea + r


for effort, tag in ((None, "ifm|think"), ("high", "ifm|think"), ("xhigh", "ifm|think"),
                    ("medium", "ifm|think_fast"), ("low", "ifm|think_faster"),
                    ("minimal", "ifm|think_faster")):
    for chunk in (1, 4, 1000):
        check(f"effort={effort} chunk={chunk}",
              split(f"17*23 = 391.\n</{tag}>391", chunk, effort), ("391", "17*23 = 391.\n"))
for given, want in (("minimal", "low"), ("medium", "medium"), ("max", "high")):
    check(f"reasoning_effort {given} -> {want}", api._map_reasoning_effort(MODEL, given), want)


class Tok:
    chat_template = '{{ raise_exception("Assistant message is missing a thinking field.") }}'


hist = [{"role": "user", "content": "Salut"}, {"role": "assistant", "content": "Bonjour !"},
        {"role": "assistant", "content": "x", "reasoning": "r"}, {"role": "user", "content": "2+2"}]
out = runner._pad_thinking_fields(Tok(), hist)
check("assistant turn without thinking gets an empty reasoning_content",
      out[1].get("reasoning_content"), "")
check("an existing thinking field is kept as is", "reasoning_content" in out[2], False)
check("user turns untouched", out[0], hist[0])
Tok.chat_template = "{{ messages }}"
check("other templates: messages untouched", runner._pad_thinking_fields(Tok(), hist), hist)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
