#!/usr/bin/env python3
"""Past-turn reasoning survives the gateway (Inkling-2-Mini via .39).

ChatMessage used to omit reasoning_content / reasoning, so pydantic dropped
them and the local chat template rendered no thinking for prior assistant
turns. Now the fields are declared; cloud and telemak relays strip them.

    .venv/bin/python scripts/test_reasoning_passthrough.py
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


# 1. A past assistant turn keeps its reasoning through pydantic.
m = api.ChatMessage(role="assistant", content="ok", reasoning_content="step 1")
check("reasoning_content kept by model_dump",
      m.model_dump(exclude_none=True).get("reasoning_content"), "step 1")
m2 = api.ChatMessage(role="assistant", content="ok", reasoning="step 2")
check("reasoning kept by model_dump",
      m2.model_dump(exclude_none=True).get("reasoning"), "step 2")

# 2. A message without reasoning dumps exactly as before (no new key).
plain = api.ChatMessage(role="assistant", content="ok").model_dump(exclude_none=True)
check("no reasoning -> no new keys", plain, {"role": "assistant", "content": "ok"})

# 3. Cloud / telemak relays strip the fields; other messages untouched.
body = {"model": "x", "messages": [
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "a", "reasoning_content": "r", "reasoning": "r2",
     "tool_calls": [{"id": "1"}]},
    {"role": "tool", "content": "t", "tool_call_id": "1"},
]}
out = api._strip_local_message_fields(body)
check("cloud strip removes reasoning fields", out["messages"][1],
      {"role": "assistant", "content": "a", "tool_calls": [{"id": "1"}]})
check("cloud strip keeps other messages", out["messages"][0], {"role": "user", "content": "q"})
check("cloud strip keeps tool message", out["messages"][2],
      {"role": "tool", "content": "t", "tool_call_id": "1"})

# 4. Local chat path: the reasoning reaches the runner's message dicts.
req_msgs = [api.ChatMessage(role="user", content="q"),
            api.ChatMessage(role="assistant", content="a", reasoning_content="long")]
dumped = [m.model_dump(exclude_none=True) for m in req_msgs]
check("local path keeps reasoning_content in message dicts",
      dumped[1].get("reasoning_content"), "long")

print()
if FAILS:
    print(f"{len(FAILS)} FAIL")
    sys.exit(1)
print("all PASS")
