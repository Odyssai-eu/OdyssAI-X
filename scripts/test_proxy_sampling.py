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

if FAILS:
    print("\n".join("FAIL " + f for f in FAILS))
    sys.exit(1)
print("all OK")
