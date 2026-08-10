#!/usr/bin/env python3
# Copyright © 2026 OdyssAI
#
# Native single-node OpenAI-compatible server for Inkling (thinkingmachines),
# using pipenetwork's bundled `inkling_mlx` package DIRECTLY — the way its HF
# model card prescribes (`inkling_mlx.load` + a native decode loop), NOT stock
# mlx_lm. The mlx_lm-adapter path ran <0.2 tok/s (short-conv + custom LayerCache
# forced through mlx_lm.generate); the native path is what Inferencer runs at
# ~11 tok/s. This server is engine-managed exactly like mlx_vlm.server /
# mlx-dspark serve: launched over ssh, health = GET /v1/models non-empty,
# chat proxied by OdyssAI-X to POST /v1/chat/completions. Full multimodal:
# images (and audio) flow through InklingProcessor → the vision/audio towers.
#
# Run: python inkling_server.py --model <dir> --host 0.0.0.0 --port 8080
# The model dir must contain the `inkling_mlx/` package (it does on HF).

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx


# ── model + processor (loaded once at startup) ────────────────────────────────
class _Engine:
    def __init__(self, model_path: str, wired_limit_gb: float):
        # The bundled inkling_mlx lives inside the model dir — make it importable
        # exactly like running from the repo (the HF card's assumption).
        sys.path.insert(0, str(Path(model_path).resolve()))
        try:
            mx.set_wired_limit(int(wired_limit_gb * 1e9))
        except Exception as e:
            sys.stderr.write(f"[inkling] set_wired_limit: {e}\n")

        from inkling_mlx.load import load
        from inkling_mlx.cache import make_cache
        from inkling_mlx.processing import InklingProcessor
        from transformers import AutoTokenizer

        t0 = time.time()
        sys.stderr.write(f"[inkling] loading {model_path} (eager, wired)…\n")
        self.model, self.config = load(model_path, lazy=False)
        sys.stderr.write(f"[inkling] model ready in {time.time()-t0:.0f}s\n")
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        chat_template = getattr(self.tok, "chat_template", None) or ""
        self.processor = InklingProcessor(self.tok, chat_template)
        self._make_cache = make_cache
        self.eos_id = int(getattr(self.config, "eos_token_id", 200006))
        self.model_path = model_path

    # -- OpenAI message → InklingProcessor parts (image_url → PIL) --------------
    def _to_processor_messages(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content")
            if isinstance(content, str) or content is None:
                out.append({"role": role, "content": content or ""})
                continue
            parts = []
            for p in content:
                t = p.get("type")
                if t == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif t in ("image_url", "image"):
                    parts.append({"type": "image", "image": _load_image(p)})
                elif t in ("input_audio", "audio"):
                    parts.append({"type": "audio",
                                  "audio": _load_audio(p),
                                  "sampling_rate": 16000})
            out.append({"role": role, "content": parts})
        return out

    def prepare(self, messages: list[dict], reasoning_effort: str = "none"):
        inp = self.processor.apply(self._to_processor_messages(messages),
                                   reasoning_effort=reasoning_effort)
        return inp

    # -- sampled, streaming decode (native path; yields text pieces) ------------
    def stream(self, inp: dict, max_tokens: int, temperature: float,
               top_p: float, stop_flag):
        ids = list(inp["input_ids"])
        pixel_values = inp.get("pixel_values")
        audio_input_ids = inp.get("audio_input_ids")
        caches = self._make_cache(self.model)
        prompt_len = len(ids)

        # Prefill the whole prompt (with media) in one native forward.
        logits = self.model(mx.array([ids]), caches=caches, start_pos=0,
                            last_logit_only=True, pixel_values=pixel_values,
                            audio_input_ids=audio_input_ids)
        next_id = _sample(logits[0, -1], temperature, top_p)
        mx.eval(next_id)
        pos = prompt_len
        n_out = 0
        detok_prev = ""
        produced: list[int] = []

        while n_out < max_tokens:
            tid = int(next_id.item())
            if tid == self.eos_id:
                break
            produced.append(tid)
            n_out += 1
            # incremental detokenize (decode the running id list, emit the delta)
            text = self.tok.decode(produced)
            if len(text) > len(detok_prev):
                yield text[len(detok_prev):]
                detok_prev = text
            if stop_flag and stop_flag():
                break
            logits = self.model(mx.array([[tid]]), caches=caches, start_pos=pos,
                                last_logit_only=True)
            next_id = _sample(logits[0, -1], temperature, top_p)
            mx.eval(next_id)
            pos += 1

        # finish_reason for the caller
        self._last = {"prompt_tokens": prompt_len, "completion_tokens": n_out,
                      "finish": "stop" if n_out < max_tokens else "length"}


def _sample(logits: mx.array, temperature: float, top_p: float) -> mx.array:
    if temperature <= 0.0:
        return mx.argmax(logits)
    logits = logits * (1.0 / temperature)
    if 0.0 < top_p < 1.0:
        probs = mx.softmax(logits, axis=-1)
        order = mx.argsort(-probs)
        sp = mx.take(probs, order)
        csum = mx.cumsum(sp)
        # keep the smallest prefix whose cumulative prob >= top_p
        mask = csum - sp < top_p
        sp = mx.where(mask, sp, 0.0)
        sp = sp / mx.sum(sp)
        choice = mx.random.categorical(mx.log(sp + 1e-12))
        return order[choice]
    return mx.random.categorical(logits)


def _load_image(part: dict):
    from PIL import Image
    url = None
    if part.get("type") == "image_url":
        u = part.get("image_url")
        url = u.get("url") if isinstance(u, dict) else u
    else:
        img = part.get("image")
        if img is not None and hasattr(img, "convert"):
            return img
        url = img
    if isinstance(url, str) and url.startswith("data:"):
        b64 = url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(url, timeout=20) as r:
            return Image.open(io.BytesIO(r.read()))
    raise ValueError("unsupported image part")


def _load_audio(part: dict):
    import numpy as np
    data = part.get("input_audio") or part.get("audio")
    if isinstance(data, dict):
        data = data.get("data")
    if isinstance(data, str) and data.startswith("data:"):
        raw = base64.b64decode(data.split(",", 1)[1])
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    raise ValueError("unsupported audio part")


# ── FastAPI app ───────────────────────────────────────────────────────────────
def build_app(engine: _Engine):
    from fastapi import Body, FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()
    model_id = engine.model_path

    @app.get("/v1/models")
    async def models():
        return {"object": "list",
                "data": [{"id": model_id, "object": "model", "owned_by": "odyssai"}]}

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat(body: dict = Body(...)):
        messages = body.get("messages") or []
        max_tokens = int(body.get("max_tokens") or 512)
        temperature = float(body.get("temperature") if body.get("temperature") is not None else 0.0)
        top_p = float(body.get("top_p") if body.get("top_p") is not None else 1.0)
        stream = bool(body.get("stream"))
        eff = body.get("reasoning_effort") or "none"
        eff = eff if eff in ("none", "minimal", "low", "medium", "high", "max") else "none"
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        inp = engine.prepare(messages, reasoning_effort=eff)

        if stream:
            def sse():
                head = {"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
                yield f"data: {json.dumps(head)}\n\n"
                for piece in engine.stream(inp, max_tokens, temperature, top_p, None):
                    ch = {"id": cid, "object": "chat.completion.chunk", "created": created,
                          "model": model_id,
                          "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                    yield f"data: {json.dumps(ch)}\n\n"
                fin = getattr(engine, "_last", {})
                tail = {"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": fin.get("finish", "stop")}],
                        "usage": {"prompt_tokens": fin.get("prompt_tokens", 0),
                                  "completion_tokens": fin.get("completion_tokens", 0),
                                  "total_tokens": fin.get("prompt_tokens", 0) + fin.get("completion_tokens", 0)}}
                yield f"data: {json.dumps(tail)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        text = "".join(engine.stream(inp, max_tokens, temperature, top_p, None))
        fin = getattr(engine, "_last", {})
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": fin.get("finish", "stop")}],
            "usage": {"prompt_tokens": fin.get("prompt_tokens", 0),
                      "completion_tokens": fin.get("completion_tokens", 0),
                      "total_tokens": fin.get("prompt_tokens", 0) + fin.get("completion_tokens", 0)},
        })

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--wired-limit-gb", type=float, default=460.0)
    args = ap.parse_args()

    import uvicorn
    engine = _Engine(args.model, args.wired_limit_gb)
    app = build_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
