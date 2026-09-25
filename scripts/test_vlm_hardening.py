#!/usr/bin/env python3
"""Tests for the 2026-09-25 VL hardening (no node, no model):
  1. text anti-loop detector: real prose / tables / code do NOT trip it;
     a short loop and a multi-paragraph loop DO
  2. VL proxy stream against a fake mlx_vlm.server that loops forever:
     the stream ends with finish_reason=stop + x_odyssai.anti_loop and the
     upstream connection is closed; anti_loop:false lets it run to the end
  3. #77 sampling for distributed VL pools only (text pools untouched)
  4. #78 supported distributed VL types
Run: ODYSSAI_X_STATE_DIR=$(mktemp -d) CLUSTER_CONFIG_FILE=$(mktemp -d)/cc.json python3 scripts/test_vlm_hardening.py
"""
import asyncio, json, os, socket, sys, threading, time, types

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


# 1. detector
PROSE = ("La mer couvre plus de soixante-dix pour cent de la surface du globe. Ses courants "
         "redistribuent la chaleur entre l'équateur et les pôles, et ses profondeurs abritent "
         "des écosystèmes que l'on commence à peine à cartographier. ") * 1 + \
        "Chaque marée rappelle que la Lune et le Soleil tirent sur cette masse d'eau immense. " \
        "Les récifs coralliens, eux, tiennent de fragiles équilibres chimiques et thermiques."
TABLE = "| id | name | value |\n|---|---|---|\n" + "".join(f"| {i} | item-{i} | {i*7 % 13} |\n" for i in range(60))
CODE = "".join(f"    x{i} = compute(a[{i}], b[{i}])\n" for i in range(80))
SHORT_LOOP = "Let me think. " + "I need to check the answer again. " * 30
BLOCK = ("Actually, let me reconsider the whole approach from the start. The function must return "
         "the sorted list, and the edge case with an empty input needs a guard. So the code becomes:\n"
         "```python\ndef f(xs):\n    if not xs:\n        return []\n    return sorted(xs)\n```\n"
         "Wait, is that right? Let me verify each step carefully before answering.\n\n")
LARGE_LOOP = "Intro text.\n" + BLOCK * 6
check("1 prose does not trip", api._detect_text_loop(PROSE), None)
check("1 table with varying cells does not trip", api._detect_text_loop(TABLE), None)
check("1 similar code lines do not trip", api._detect_text_loop(CODE), None)
check("1 short loop trips", bool(api._detect_text_loop(SHORT_LOOP)), True)
hit = api._detect_text_loop(LARGE_LOOP)
check("1 multi-paragraph loop trips (period > 64)", bool(hit) and hit[0] > 64, True)
check("1 three repeats of a block is NOT a loop", api._detect_text_loop("Intro.\n" + BLOCK * 3), None)


# 2. proxy against a looping fake upstream
def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


STATE = {"sent": 0, "closed": False}
PORT = free_port()
fake = FastAPI()


@fake.get("/v1/models")
def models():
    return {"data": [{"id": "/models/Loop-VL"}]}


@fake.post("/v1/chat/completions")
async def cc(r: Request):
    b = await r.json()
    STATE["sent"] = 0
    STATE["closed"] = False
    limit = 400

    async def gen():
        try:
            for i in range(limit):
                piece = "I need to check the answer again. "
                yield ("data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": piece}}]}) + "\n\n").encode()
                STATE["sent"] += 1
                await asyncio.sleep(0.002)
            yield b"data: [DONE]\n\n"
        finally:
            STATE["closed"] = STATE["sent"] < limit
    return StreamingResponse(gen(), media_type="text/event-stream")


threading.Thread(target=lambda: uvicorn.run(fake, host="127.0.0.1", port=PORT, log_level="error"), daemon=True).start()
time.sleep(1.2)
pool = types.SimpleNamespace(upstream=f"http://127.0.0.1:{PORT}", model="/models/Loop-VL", alias="loop-vl",
                             cluster="testc", last_used_at=0)


async def run(anti_loop):
    body = {"model": "loop-vl", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100000, "stream": True}
    if anti_loop is not None:
        body["anti_loop"] = anti_loop
    resp = await api._vlm_pool_proxy_chat_completion(pool, body, True)
    chunks = [c async for c in resp.body_iterator]
    raw = b"".join(c if isinstance(c, bytes) else c.encode() for c in chunks).decode()
    await asyncio.sleep(0.3)
    return raw


raw = asyncio.run(run(None))
check("2 loop cut: anti_loop marker in stream", '"anti_loop"' in raw, True)
check("2 loop cut: stream ends with [DONE]", raw.rstrip().endswith("data: [DONE]"), True)
check("2 loop cut early (far fewer than 400 chunks sent)", STATE["sent"] < 100, True)
check("2 upstream connection closed", STATE["closed"], True)
raw = asyncio.run(run(False))
check("2 anti_loop:false runs to the end", STATE["sent"], 400)
check("2 anti_loop:false has no marker", '"anti_loop"' in raw, False)

# 3. #77
req = api.ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}], temperature=0.4, top_k=20)
dist = types.SimpleNamespace(is_vlm_dist=True, cluster="c", alias="a", model="/m/MiniMax-M3-VL")
text = types.SimpleNamespace(is_vlm_dist=False, cluster="c", alias="t", model="/m/Qwen")
check("3 dist pool gets client sampling", api._vlmdist_sampling_kw(dist, req), {"sampling": {"temperature": 0.4, "top_k": 20}})
check("3 text pool gets nothing", api._vlmdist_sampling_kw(text, req), {})
mimo = types.SimpleNamespace(is_vlm_dist=True, cluster="c", alias="m", model="/m/MiMo-V2.6-VL")
req0 = api.ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}])
check("3 dist pool with no client value gets the per-model default",
      api._vlmdist_sampling_kw(mimo, req0), {"sampling": {"temperature": 1.0, "top_p": 0.95}})
import inspect
check("3 RunnerPool.submit accepts sampling", "sampling" in inspect.signature(api.RunnerPool.submit).parameters, True)

# 4. #78
check("4 supported distributed VL types", api.VLM_DIST_SUPPORTED, {"minimax_m3_vl": "tensor", "qwen3_5_moe": "pipeline"})
src = open(os.path.join(REPO, "scripts", "vlm_runner.py")).read()
check("4 vlm_runner backstop present", "has no distributed split" in src, True)

if FAILS:
    print("\n".join("FAIL " + f for f in FAILS))
    sys.exit(1)
print("all OK")
