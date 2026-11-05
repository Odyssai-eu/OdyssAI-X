#!/usr/bin/env python3
"""The key of a client that sends no id, and the panel's numbers.

2026-11-05: an agent resends its whole history at every step and every step paid
the whole prefill, because only a client-sent id turned the prefix cache on.

    python3 scripts/test_prefix_session.py
"""
import json
import os
import sys
import tempfile

os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({}, open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, ok, got=""):
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": {got}"))
    if not ok:
        FAILS.append(label)


# ── the conversation key ────────────────────────────────────────────────
SYSTEM = {"role": "system", "content": "tu es un agent"}
TOOLS = [{"type": "function", "function": {"name": "read"}}]
step1 = [SYSTEM, {"role": "user", "content": "tache A"}]
step2 = step1 + [{"role": "assistant", "content": "ok"}, {"role": "tool", "content": "x"}]
key = api._conversation_key("m", step1, TOOLS)
check("key: same at every step", key == api._conversation_key("m", step2, TOOLS))
check("key: another task, another key",
      key != api._conversation_key("m", [SYSTEM, {"role": "user", "content": "tache B"}], TOOLS))
check("key: other tools, other key", key != api._conversation_key("m", step1, None))
check("key: other model, other key", key != api._conversation_key("n", step1, TOOLS))
check("key: no user message, no key", api._conversation_key("m", [SYSTEM], TOOLS) is None)


# ── the panel's numbers ─────────────────────────────────────────────────
api._metrics.clear()
api.record_metric("c", 100, 110.0, 100.0, 400, "m", prompt_tokens=900)
row = api._metrics[0]
check("metric: speed leaves the prefill out", row["tps"] == 10.0, row["tps"])
check("metric: real prompt tokens kept", row["prompt_tokens"] == 900)
api.record_metric("c", 10, 5.0, None, 40, "m")
check("metric: no first token time, whole duration", api._metrics[0]["tps"] == 2.0)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
