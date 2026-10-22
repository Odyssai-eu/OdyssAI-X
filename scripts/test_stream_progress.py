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

    seen_max_tokens = []

    async def submit(self, prompt, max_tokens, *a, **kw):
        FakePool.seen_max_tokens.append(max_tokens)
        yield {"event": "token", "text": "Je l'écris. "}
        for i in range(12):
            await asyncio.sleep(0.1)
            yield {"event": "progress", "phase": "decode", "ntoks": 10 * (i + 1),
                   "prompt_tokens": 16443}
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
        prog = [ch.get("x_odyssai_progress") for ch in empties]
        check("openai: progress chunks carry the token counts (#113)",
              all(p and p.get("prompt_tokens") == 16443 and p.get("completion_tokens") > 0 for p in prog), True)
        # (httpx's ASGITransport delivers the body at once: arrival gaps are measured
        # on the real engine, not here — the pacing is checked by the counts.)
        content = "".join((ch["choices"][0].get("delta") or {}).get("content") or ""
                          for ch in chunks if ch.get("choices"))
        check("openai: no markup in content", "<tool_call" in content, False)
        last = chunks[-1]["choices"][0]
        check("openai: tool_calls intact in the final chunk",
              (last.get("finish_reason"), last["delta"]["tool_calls"][0]["function"]["name"]),
              ("tool_calls", "write_file"))

        check("no max_tokens from the client → no engine cap (None to the runner)",
              FakePool.seen_max_tokens[-1], None)

        # VL proxy: no client cap → the model's context window, not mlx_vlm's 2048
        async def fake_cfg(ssh, path, timeout=6.0):
            return {"text_config": {"max_position_embeddings": 262144}}
        _orig = api._read_model_config
        api._read_model_config = fake_cfg
        class VL:
            ssh_target, model = "admin@x", "/Volumes/models/odysseus/x/MiMo"
        check("VL proxy: window read from config.json", await api._pool_context_window(VL()), 262144)
        api._read_model_config = _orig

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

        # A second VL pool on a node: the default port is held by our gemma pool,
        # the next free one is taken (skipping our other pools and any listener).
        class P:
            def __init__(self, ssh, port): self.ssh_target, self.port = ssh, port
        _lap, _pl = api.list_all_pools, api._port_listening
        api.list_all_pools = lambda: [("fith", "gemma", P("admin@192.168.86.33", 8080)),
                                      ("fith", "other", P("admin@192.168.86.33", 8082)),
                                      ("x", "y", P("admin@192.168.86.49", 8083))]
        async def listening(ip, port, timeout=1.5): return port == 8081
        api._port_listening = listening
        check("vlm port: owner of 8080 on .33 is fith/gemma",
              api._pool_on_node_port("admin@192.168.86.33", 8080), ("fith", "gemma"))
        check("vlm port: nobody owns 8080 on .49", api._pool_on_node_port("admin@192.168.86.49", 8080), None)
        check("vlm port: skips a listener (8081) and our pool (8082) → 8083",
              await api._free_vlm_port("192.168.86.33", "admin@192.168.86.33", 8081), 8083)
        api.list_all_pools, api._port_listening = _lap, _pl

        # tool_choice "none": Muse-Spark's template never sees the tools; other
        # models keep them (Bench A sends them with "none" on purpose).
        muse = "/Volumes/models/odysseus/odyssai/Muse-Spark-Open-816B-A42B-Q6a8h16"
        check("muse: tool_choice none → no tools in the template",
              api._tools_for_template(muse, TOOLS, "none"), None)
        check("muse: anthropic {type: none} → no tools",
              api._tools_for_template(muse, TOOLS, {"type": "none"}), None)
        check("muse: tool_choice auto keeps the tools", api._tools_for_template(muse, TOOLS, "auto"), TOOLS)
        check("glm: tool_choice none keeps the tools",
              api._tools_for_template("/Volumes/models/odysseus/odyssai/GLM-5.3-Flash-Q8h16", TOOLS, "none"), TOOLS)

        # #136: /v1/messages without max_tokens → None to the runner, like OpenAI
        r = await c.post("/v1/messages", json={
            "model": "fake", "messages": [{"role": "user", "content": "bonjour"}]})
        check("anthropic: max_tokens optional (no 422)", r.status_code, 200)
        check("anthropic: no max_tokens → None to the runner", FakePool.seen_max_tokens[-1], None)

        # #135: the runner refuses a prompt that fills the window → 400
        api._route_pool = lambda model: OverflowPool()
        r = await c.post("/v1/chat/completions", json={
            "model": "fake", "messages": [{"role": "user", "content": "trop long"}]})
        check("openai: context overflow → 400", r.status_code, 400)
        check("openai: code context_length_exceeded", "context_length_exceeded" in r.text, True)
        r = await c.post("/v1/messages", json={
            "model": "fake", "messages": [{"role": "user", "content": "trop long"}]})
        check("anthropic: context overflow → 400 invalid_request_error",
              (r.status_code, "invalid_request_error" in r.text), (400, True))
        chunks = []
        async with c.stream("POST", "/v1/chat/completions", json={
                "model": "fake", "stream": True,
                "messages": [{"role": "user", "content": "trop long"}]}) as rs:
            async for line in rs.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(json.loads(line[6:]))
        check("openai stream: error chunk with code context_length_exceeded",
              (chunks[-1].get("error") or {}).get("code"), "context_length_exceeded")


class OverflowPool(FakePool):
    """What RunnerPool.submit raises when the runner's done event refuses the
    prompt for filling the context window (#135)."""

    async def submit(self, prompt, max_tokens, *a, **kw):
        raise api.RunnerRequestError(
            "context_length_exceeded: the prompt is 40000 tokens and the model's "
            "context window is 32768 tokens")
        yield  # pragma: no cover — makes this an async generator


asyncio.run(main())
if FAILS:
    print(f"{len(FAILS)} FAILED")
    sys.exit(1)
print("all OK")
