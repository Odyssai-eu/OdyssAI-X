#!/usr/bin/env python3
"""Tests for VLM replica pools (#76, 2026-09-24) against two fake
mlx_vlm.server instances on 127.0.0.1. No node, no model. Covers:
  1. concurrent requests spread over the two replicas (least in-flight)
  2. session affinity: the same session_id stays on its replica
  3. a replica whose port is closed is skipped and marked down
  4. the response `model` is the parent alias, not the child
  5. streaming: in-flight counted until the stream is consumed
  6. adoption only when the port serves THIS model
  7. state: is_vlm_replica is saved and restored BEFORE is_replica
  8. an image in the latest user message to a text pool -> 400;
     an image in an older turn does not block
Run: ODYSSAI_X_STATE_DIR=$(mktemp -d) CLUSTER_CONFIG_FILE=$(mktemp -d)/cc.json python3 scripts/test_vlm_replica.py
"""
import asyncio
import json
import os
import socket
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    if got != want:
        FAILS.append(f"{label}: got {got!r}, want {want!r}")
    else:
        print(f"OK  {label}")


MODEL = "/Volumes/models/odysseus/test/Fake-VL-Model"
HITS = {}


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def fake_server(name, port, served=MODEL, delay=0.4):
    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": served}]}

    @app.post("/v1/chat/completions")
    async def cc(r: Request):
        b = await r.json()
        HITS.setdefault(name, []).append(b)
        await asyncio.sleep(delay)
        if b.get("stream"):
            async def gen():
                for w in ("hel", "lo"):
                    yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': w}}]})}\n\n".encode()
                    await asyncio.sleep(0.2)
                yield b"data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return {"id": "x", "object": "chat.completion", "model": b["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": name},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}
    t = threading.Thread(target=lambda: uvicorn.run(app, host="127.0.0.1", port=port, log_level="error"),
                         daemon=True)
    t.start()


PA, PB, PDEAD, PFOREIGN = free_port(), free_port(), free_port(), free_port()
fake_server("A", PA)
fake_server("B", PB)
fake_server("F", PFOREIGN, served="/Volumes/models/other/Another-Model")
time.sleep(1.5)


def make_pool(ports):
    pool = api.VLMReplicaPool(model_path=MODEL, cluster="testc", alias="vl-rep",
                              node_indices=list(range(len(ports))), port=ports[0], venv="/tmp/venv")
    for i, port in enumerate(ports):
        c = pool.children[i]
        c.ssh_target = "admin@127.0.0.1"
        c.host = f"n{i}"
        c.port = port
        c.upstream = f"http://127.0.0.1:{port}"
    pool._live = set(range(len(ports)))
    return pool


BODY = {"model": "vl-rep", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}


async def main():
    # 1 + 4. two concurrent unary requests -> two replicas; label = parent alias
    pool = make_pool([PA, PB])
    HITS.clear()
    r1, r2 = await asyncio.gather(pool.dispatch(dict(BODY), False), pool.dispatch(dict(BODY), False))
    check("1 concurrent requests spread", sorted([r1["choices"][0]["message"]["content"],
                                                   r2["choices"][0]["message"]["content"]]), ["A", "B"])
    check("4 response model is the parent alias", (r1["model"], r2["model"]), ("vl-rep", "vl-rep"))
    check("1 in-flight back to zero", pool._inflight, {0: 0, 1: 0})

    # 2. affinity
    HITS.clear()
    first = (await pool.dispatch(dict(BODY), False, session_id="s1"))["choices"][0]["message"]["content"]
    again = [(await pool.dispatch(dict(BODY), False, session_id="s1"))["choices"][0]["message"]["content"]
             for _ in range(3)]
    check("2 same session stays on its replica", again, [first] * 3)

    # 3. closed port skipped and marked down
    pool3 = make_pool([PDEAD, PB])
    r = await pool3.dispatch(dict(BODY), False)
    check("3 dead replica skipped", r["choices"][0]["message"]["content"], "B")
    check("3 dead replica marked down", sorted(pool3._live), [1])
    check("3 stats show it", [x["live"] for x in pool3.replica_stats()], [False, True])

    # 5. streaming in-flight until consumed
    pool5 = make_pool([PA])
    resp = await pool5.dispatch(dict(BODY, stream=True), True)
    check("5 stream in-flight while open", pool5._inflight[0], 1)
    chunks = [c async for c in resp.body_iterator]
    text = b"".join(c if isinstance(c, bytes) else c.encode() for c in chunks)
    check("5 stream content passed through", b'"hel"' in text and b'"lo"' in text, True)
    check("5 stream in-flight released", pool5._inflight[0], 0)

    # 6. adoption: same model adopted, foreign model refused (never launched)
    pool6 = make_pool([PA, PFOREIGN])
    pool6._live = set()
    await pool6._start_child(0)
    check("6 same model adopted (no launch)", pool6.children[0].pid, None)
    try:
        await pool6._start_child(1)
        check("6 foreign model refused", "no error", "RuntimeError")
    except RuntimeError as e:
        check("6 foreign model refused", "another model" in str(e), True)

    # 3b. all replicas dead -> 503
    pool7 = make_pool([PDEAD])
    try:
        await pool7.dispatch(dict(BODY), False)
        check("3b no live replica -> 503", "no error", 503)
    except api.HTTPException as e:
        check("3b no live replica -> 503", e.status_code, 503)
    check("3b alive_count keeps the sentinel", pool7.alive_count(), 1)


asyncio.run(main())

# 7. state order: the VL replica key is handled before the text replica key
src = open(os.path.join(REPO, "scripts", "api.py")).read()
save = src[src.index("def save_cluster_state_v2"):]
check("7 save: is_vlm_replica before is_replica",
      save.index('getattr(pool, "is_vlm_replica"') < save.index('getattr(pool, "is_replica"'), True)
rest = src[src.index('if entry.get("is_vlm_replica"):'):]
check("7 restore: is_vlm_replica before is_replica",
      rest.index('if entry.get("is_vlm_replica"):') < rest.index('if entry.get("is_replica"):'), True)

# 8. image refusal
class TextPool:
    is_vlm = False
class VisionPool:
    is_vlm = True
IMG = [{"role": "user", "content": [{"type": "text", "text": "what is it"},
                                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}]
try:
    api._refuse_image_on_text_pool("qwen-text", TextPool(), IMG)
    check("8 image to text pool -> 400", "no error", 400)
except api.HTTPException as e:
    check("8 image to text pool -> 400", e.status_code, 400)
api._refuse_image_on_text_pool("vl", VisionPool(), IMG)
check("8 image to vision pool passes", True, True)
OLD = IMG + [{"role": "assistant", "content": "a cat"}, {"role": "user", "content": "thanks"}]
api._refuse_image_on_text_pool("qwen-text", TextPool(), OLD)
check("8 image in an older turn does not block", True, True)
ANT = [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "AAA"}}]}]
check("8 anthropic image block detected", api._last_user_has_image(ANT), True)
check("8 plain text not an image", api._last_user_has_image([{"role": "user", "content": "hi"}]), False)
check("same model path by basename", api._same_model_path("/a/b/Fake-VL-Model/", MODEL), True)

if FAILS:
    print("\n".join("FAIL " + f for f in FAILS))
    sys.exit(1)
print("all OK")
