#!/usr/bin/env python3
"""Un refus Anthropic (stop_reason « refusal ») doit se voir (1.62.13).

Avant, tout stop_reason hors table devenait « stop » : un refus de Fable 5.1
sur TMB (A/a08, R2/r08, 2026-10-26) ressortait comme une réponse vide normale.
Hors réseau : on traduit des réponses Anthropic fabriquées.

    .venv/bin/python scripts/test_anthropic_refusal.py
"""
import asyncio
import json
import os
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402


def response(stop_reason):
    return {"id": "msg_x", "content": [], "stop_reason": stop_reason,
            "usage": {"input_tokens": 217, "output_tokens": 0}}


def stream(stop_reason):
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 217, "output_tokens": 0}}},
        {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 0}},
        {"type": "message_stop"},
    ]

    async def gen():
        for ev in events:
            yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode()
    return gen()


async def stream_finish(stop_reason):
    finish = None
    async for chunk in api._translate_anthropic_sse_to_openai(stream(stop_reason), "m"):
        for line in chunk.decode().splitlines():
            if line.startswith("data: {"):
                for c in json.loads(line[6:]).get("choices") or []:
                    finish = c.get("finish_reason") or finish
    return finish


def main():
    ok = True
    for sr, want in (("refusal", "content_filter"), ("end_turn", "stop"),
                     ("max_tokens", "length"), ("pause_turn", "stop")):
        choice = api._anthropic_to_openai_response(response(sr), "m")["choices"][0]
        got = (choice["finish_reason"], choice["stop_reason"])
        s = asyncio.run(stream_finish(sr))
        good = got == (want, sr) and s == want
        ok &= good
        print(("OK  " if good else "FAIL"), sr, "->", got, "stream:", s)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
