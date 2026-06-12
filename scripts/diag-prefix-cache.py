#!/usr/bin/env python3
"""Diagnostic: prefix cache effectiveness on any OdyssAI-X cluster.

Sends two requests with the same `session_id`:
- Probe 1 (cold): only system + user. Should be a miss → full prefill paid.
- Probe 2 (warm): same conversation + a short assistant reply + new user
  turn. With prefix cache, prefill cost should be reduced to ~the new
  tokens only.

Reports TTFT, prompt_tokens (when available), and the cache hit status
inferred from the runner log.

Usage:
  ODYSSAI_X_URL=http://<your-engine-host>:8000 \
  python3 scripts/diag-prefix-cache.py <model-id>

The cluster must be loaded (curl $ODYSSAI_X_URL/admin/clusters/<id> →
nodes.loaded:true).
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid

import httpx


URL = os.environ.get("ODYSSAI_X_URL") or os.environ.get("ODYSSEUS_URL")
if not URL:
    sys.exit("set ODYSSAI_X_URL env var (e.g. http://<your-host>:8000) before running")


SYSTEM_PROMPT = (
    "You are a helpful assistant. Keep replies short (under 30 words)."
)
USER_TURN_1 = "Say hi in 5 words exactly."
USER_TURN_2 = "Now say goodbye in 5 words exactly."


def _probe(model: str, session_id: str, messages: list[dict]) -> dict:
    """Send a non-stream request and measure TTFT + total time."""
    body = {
        "model": model,
        "messages": messages,
        "session_id": session_id,
        "max_tokens": 40,
        "stream": False,
    }
    t0 = time.monotonic()
    with httpx.Client(timeout=300.0) as client:
        r = client.post(f"{URL}/v1/chat/completions", json=body)
    elapsed = time.monotonic() - t0
    if r.status_code >= 400:
        return {"ok": False, "status": r.status_code, "text": r.text[:300],
                "elapsed_s": round(elapsed, 2)}
    try:
        payload = r.json()
    except Exception:
        return {"ok": False, "status": r.status_code, "text": r.text[:300],
                "elapsed_s": round(elapsed, 2)}
    choice = (payload.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = payload.get("usage") or {}
    return {
        "ok": True,
        "elapsed_s": round(elapsed, 2),
        "content": (msg.get("content") or "")[:120],
        "prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "completion_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def main():
    if len(sys.argv) < 2:
        print("usage: diag-prefix-cache.py <model_id>", file=sys.stderr)
        sys.exit(2)
    model = sys.argv[1]
    session_id = f"diag-{uuid.uuid4().hex[:8]}"
    print(f"== Prefix cache diagnostic ==")
    print(f"  url        : {URL}")
    print(f"  model      : {model}")
    print(f"  session_id : {session_id}")
    print()

    # Probe 1: cold (miss expected).
    messages_1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TURN_1},
    ]
    print("→ Probe 1 (cold, expect fp16·miss)...")
    r1 = _probe(model, session_id, messages_1)
    if not r1["ok"]:
        print(f"  ❌ {r1['status']}: {r1['text']}")
        sys.exit(1)
    print(f"  elapsed={r1['elapsed_s']}s  prompt_toks={r1.get('prompt_tokens')}"
          f"  completion={r1.get('completion_tokens')}")
    print(f"  reply: {r1['content']!r}")

    # Probe 2: warm (hit expected on prefix = system + first user + assistant).
    messages_2 = messages_1 + [
        {"role": "assistant", "content": r1["content"]},
        {"role": "user",      "content": USER_TURN_2},
    ]
    print()
    print("→ Probe 2 (warm, expect session-HIT)...")
    r2 = _probe(model, session_id, messages_2)
    if not r2["ok"]:
        print(f"  ❌ {r2['status']}: {r2['text']}")
        sys.exit(1)
    print(f"  elapsed={r2['elapsed_s']}s  prompt_toks={r2.get('prompt_tokens')}"
          f"  completion={r2.get('completion_tokens')}")
    print(f"  reply: {r2['content']!r}")
    print()

    # Interpretation.
    speedup = (r1["elapsed_s"] / r2["elapsed_s"]) if r2["elapsed_s"] > 0 else 0
    print(f"== Interpretation ==")
    print(f"  Probe 2 / Probe 1  : {r2['elapsed_s']}s / {r1['elapsed_s']}s "
          f"= {speedup:.2f}× speedup")
    if speedup > 1.5:
        print(f"  ✅ Prefix cache likely working (probe 2 reused warm cache).")
    elif speedup > 1.1:
        print(f"  🟡 Borderline. Could be cache hit + a few extra prefill tokens, "
              f"or could be a partial miss. Check runner logs for `session-HIT`.")
    else:
        print(f"  ❌ Probe 2 is not faster than probe 1. Cache miss or"
              f" divergence. Look for `fp16·divergent` in runner logs.")
    print()
    print("To verify, tail the OdyssAI-X container logs and grep for the "
          "session id:")
    print(f"  ssh <odysseus-host> '/usr/local/bin/docker logs "
          f"mlx-odyss-eu 2>&1' | grep '{session_id[:6]}'")


if __name__ == "__main__":
    main()
