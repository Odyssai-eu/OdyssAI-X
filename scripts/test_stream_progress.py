#!/usr/bin/env python3
"""#112: a tool call the runner holds back must not leave the stream silent.

The runner sends `progress` events (tokens advanced, text withheld by
ToolMarkupGate). The OpenAI stream relays them as empty-delta chunks and the
Anthropic stream as pings, at most one per STREAM_PROGRESS_CHUNK_S; the text
never carries the call markup. A fake pool stands in for the runner, so no node
is touched (no lifespan, no sweep). The runner side (`_gated_event`) is in
test_tool_markup_gate.py (needs mlx).

    ODYSSAI_X_STATE_DIR=$(mktemp -d) CLUSTER_CONFIG_FILE=$(mktemp -d)/cc.json \\
      uv run --no-project --python 3.12 --with-requirements requirements.txt \\
      python scripts/test_stream_progress.py
"""
import asyncio
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("CLUSTER_CONFIG_FILE", os.path.join(tempfile.mkdtemp(), "cc.json"))
os.environ.setdefault("ODYSSAI_X_STATE_DIR", tempfile.mkdtemp())
os.environ["ODYSSAI_X_STREAM_PROGRESS_CHUNK_S"] = "0.2"
os.environ.setdefault("ODO_FILE", os.path.join(tempfile.mkdtemp(), "odometer.json"))  # not the repo
json.dump({}, open(os.environ["CLUSTER_CONFIG_FILE"], "w"))

import httpx  # noqa: E402
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


# --- engine side: progress events through the two streams ---------------------------------
CALL = ("<tool_call>write_file<arg_key>path</arg_key><arg_value>a.txt</arg_value>"
        "<arg_key>content</arg_key><arg_value>x</arg_value></tool_call>")


class FakePool:
    """Yields what a runner with the gate sends for one long tool call: a little
    text, then only progress events for ~1.2 s, then done with the parsed call."""
    cluster = "fake"
    alias = "fake"
    model = "/Volumes/models/odysseus/test/Fake-GLM"
    is_vlm = False
    is_decision = False
    nodes_count = 1

    def alive_count(self):
        return 1

    async def submit(self, *a, **kw):
        yield {"event": "token", "text": "Je l'écris. "}
        for i in range(12):
            await asyncio.sleep(0.1)
            yield {"event": "progress", "phase": "decode", "ntoks": 10 * (i + 1)}
        yield {"event": "done", "ntoks": 130, "elapsed_s": 1.2, "markup_withheld": True,
               "tool_calls": [{"id": "call_1", "type": "function",
                               "function": {"name": "write_file",
                                            "arguments": json.dumps({"path": "a.txt", "content": "x"})}}]}


api._route_pool = lambda model: FakePool()
TOOLS = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}}]


async def main():
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine", timeout=30) as c:
        # OpenAI stream
        times, chunks = [], []
        t0 = time.monotonic()
        async with c.stream("POST", "/v1/chat/completions", json={
                "model": "fake", "stream": True, "tools": TOOLS,
                "messages": [{"role": "user", "content": "écris a.txt"}]}) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    times.append(time.monotonic() - t0)
                    chunks.append(json.loads(line[6:]))
        empties = [ch for ch in chunks if ch.get("choices") and ch["choices"][0].get("delta") == {}
                   and ch["choices"][0].get("finish_reason") is None]
        check("openai: progress relayed as empty-delta chunks", len(empties) >= 3, True)
        check("openai: at most one per STREAM_PROGRESS_CHUNK_S (12 events → ≤ 7 chunks)", len(empties) <= 7, True)
        # (httpx's ASGITransport delivers the body at once: arrival gaps are measured
        # on the real engine, not here — the pacing is checked by the counts.)
        content = "".join((ch["choices"][0].get("delta") or {}).get("content") or ""
                          for ch in chunks if ch.get("choices"))
        check("openai: no markup in content", "<tool_call" in content, False)
        last = chunks[-1]["choices"][0]
        check("openai: tool_calls intact in the final chunk",
              (last.get("finish_reason"), last["delta"]["tool_calls"][0]["function"]["name"]),
              ("tool_calls", "write_file"))

        # Anthropic stream
        events = []
        async with c.stream("POST", "/v1/messages", json={
                "model": "fake", "stream": True, "max_tokens": 256,
                "tools": [{"name": "write_file", "input_schema": {"type": "object"}}],
                "messages": [{"role": "user", "content": "écris a.txt"}]}) as r:
            async for line in r.aiter_lines():
                if line.startswith("event: "):
                    events.append(line[7:])
        pings = events.count("ping")
        check("anthropic: progress relayed as pings", 3 <= pings <= 7, True)
        check("anthropic: tool_use block still emitted", "content_block_start" in events
              and events.count("content_block_start") >= 2, True)


asyncio.run(main())
if FAILS:
    print(f"{len(FAILS)} FAILED")
    sys.exit(1)
print("all OK")
