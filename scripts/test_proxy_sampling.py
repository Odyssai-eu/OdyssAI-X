#!/usr/bin/env python3
"""Unit tests for proxied-upstream sampling (2026-09-24,
docs/PLAN-2026-09-24-proxy-sampling.md). Covers:
  1. client sampling survives ChatCompletionRequest.model_dump(exclude_none=True)
     (it was silently dropped by pydantic before 1.52.1)
  2. no sampling sent -> no sampling key in the dump, max_tokens default intact
  3. a MiMo upstream with nothing sent gets the authors' 1.0 / 0.95
  4. an explicit client value always wins over the default
  5. an explicit 0.0 is not overwritten
  6. a model without an entry is left untouched
  7. no model id -> nothing injected
  8. thinking / effort alias spellings fold into enable_thinking /
     reasoning_effort on chat; the canonical field wins
  9. /v1/messages keeps thinking, top_p, top_k, reasoning_effort
Run: ODYSSAI_X_STATE_DIR=$(mktemp -d) python3 scripts/test_proxy_sampling.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    if got != want:
        FAILS.append(f"{label}: got {got!r}, want {want!r}")
    else:
        print(f"OK  {label}")


MSG = [{"role": "user", "content": "x"}]
MIMO = "/Volumes/models/odysseus/scratch/MiMo-V2.6-Flash-RL-Q9-fixed"
QWEN = "/Volumes/models/odysseus/odyssai/Qwen3.8-Flash-Next-Q6"
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                 "presence_penalty", "frequency_penalty", "seed")

d = api.ChatCompletionRequest(model="m", messages=MSG, temperature=0.3, top_p=0.8,
                              top_k=20, seed=7).model_dump(exclude_none=True)
check("1 dump keeps client sampling",
      {k: d.get(k) for k in ("temperature", "top_p", "top_k", "seed")},
      {"temperature": 0.3, "top_p": 0.8, "top_k": 20, "seed": 7})

d = api.ChatCompletionRequest(model="m", messages=MSG).model_dump(exclude_none=True)
check("2 nothing sent -> no sampling key", [k for k in SAMPLING_KEYS if k in d], [])
check("2 max_tokens default intact", d.get("max_tokens"), 512)

b = {}
check("3 mimo default returned", api._apply_default_sampling(MIMO, b), {"temperature": 1.0, "top_p": 0.95})
check("3 mimo default in body", b, {"temperature": 1.0, "top_p": 0.95})

b = {"temperature": 0.2}
check("4 client temperature wins (applied)", api._apply_default_sampling(MIMO, b), {"top_p": 0.95})
check("4 client temperature kept", b["temperature"], 0.2)

b = {"temperature": 0.0, "top_p": 1.0}
check("5 explicit 0.0 not overwritten (applied)", api._apply_default_sampling(MIMO, b), {})
check("5 explicit 0.0 body unchanged", b, {"temperature": 0.0, "top_p": 1.0})

b = {}
check("6 no entry -> nothing applied", api._apply_default_sampling(QWEN, b), {})
check("6 no entry -> body untouched", b, {})

b = {"temperature": None}
check("3b explicit None filled", api._apply_default_sampling(MIMO, b), {"temperature": 1.0, "top_p": 0.95})

check("7 no model id", api._apply_default_sampling(None, {}), {})

def chat(**kw):
    r = api.ChatCompletionRequest(model="m", messages=MSG, **kw)
    return r.enable_thinking, r.reasoning_effort

check("8 chat_template_kwargs.enable_thinking=false", chat(chat_template_kwargs={"enable_thinking": False}), (False, None))
check("8 chat_template_kwargs.thinking=true", chat(chat_template_kwargs={"thinking": True}), (True, None))
check("8 thinking bool", chat(thinking=False), (False, None))
check("8 thinking anthropic dict", chat(thinking={"type": "disabled"}), (False, None))
check("8 reasoning.effort", chat(reasoning={"effort": "low"}), (None, "low"))
check("8 chat_template_kwargs.reasoning_effort", chat(chat_template_kwargs={"reasoning_effort": "high"}), (None, "high"))
check("8 canonical wins", chat(enable_thinking=True, chat_template_kwargs={"enable_thinking": False},
                               reasoning_effort="high", reasoning={"effort": "low"}), (True, "high"))
check("8 nothing sent", chat(), (None, None))
d = api.ChatCompletionRequest(model="m", messages=MSG, chat_template_kwargs={"enable_thinking": False},
                              reasoning={"effort": "low"}).model_dump(exclude_none=True)
check("8 aliases not forwarded raw", [k for k in ("chat_template_kwargs", "reasoning", "thinking") if k in d], [])

a = api.AnthropicMessagesRequest(model="m", max_tokens=10, messages=MSG, top_p=0.9, top_k=40,
                                 thinking={"type": "enabled", "budget_tokens": 2000},
                                 reasoning_effort="low").model_dump(exclude_none=True)
check("9 messages keeps thinking/top_p/top_k/effort",
      {k: a.get(k) for k in ("thinking", "top_p", "top_k", "reasoning_effort")},
      {"thinking": {"type": "enabled", "budget_tokens": 2000}, "top_p": 0.9, "top_k": 40, "reasoning_effort": "low"})
check("9 anthropic thinking -> bool", api._thinking_from_aliases({"thinking": {"type": "disabled", "budget_tokens": 0}}), False)
check("9 no thinking -> None", api._thinking_from_aliases({"thinking": None}), None)

if FAILS:
    print("\n".join("FAIL " + f for f in FAILS))
    sys.exit(1)
print("all OK")
