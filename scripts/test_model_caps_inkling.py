#!/usr/bin/env python3
"""/v1/models capabilities for Inkling-2-Mini (model_type inkling2_loop).

It was listed with supports_tools false and context_length null: its config
has no max_position_embeddings (the window is text_config.model_max_length)
and its chat template lives in chat_template.jinja, not config.json.

    .venv/bin/python scripts/test_model_caps_inkling.py
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
    ok = got == want
    print(("PASS " if ok else "FAIL ") + label + ("" if ok else f"\n   got:  {got!r}\n   want: {want!r}"))
    if not ok:
        FAILS.append(label)


def caps_for(cfg):
    caps = {}
    api._enrich_caps_from_config(caps, cfg)
    return caps


# The fields of the released Inkling-2-Mini config.json that the code reads.
inkling = caps_for({
    "model_type": "inkling2_loop",
    "text_config": {"model_type": "inkling2_loop_text", "model_max_length": 1048576,
                    "sliding_window_size": 1024},
    "vision_config": {"model_type": "inkling2_vision"},
})
check("inkling: context_length", inkling.get("context_length"), 1048576)
check("inkling: supports_tools", inkling.get("supports_tools"), True)
check("inkling: supports_vision", inkling.get("supports_vision"), True)

# max_position_embeddings still wins when both are present.
both = caps_for({"model_type": "qwen3_5_moe", "max_position_embeddings": 262144,
                 "model_max_length": 999})
check("max_position_embeddings wins", both.get("context_length"), 262144)

# An unknown family with neither field is unchanged: no window, no tools.
other = caps_for({"model_type": "some_unknown_arch"})
check("unknown: no context_length", other.get("context_length"), None)
check("unknown: supports_tools false", other.get("supports_tools"), False)

print()
if FAILS:
    print(f"{len(FAILS)} FAIL")
    sys.exit(1)
print("all PASS")
