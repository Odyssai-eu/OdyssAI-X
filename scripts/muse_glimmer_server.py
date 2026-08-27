#!/usr/bin/env python3
# Copyright © 2026 OdyssAI
#
# Native single-node OpenAI-compatible server for Muse-Glimmer-30B
# (pipenetwork / meta-models), using PipeNetwork's `muse_glimmer_mlx` package
# DIRECTLY — the runtime its HF card prescribes. No released mlx-vlm or mlx-lm
# can load `muse_glimmer` (neither carries the module), so this server IS the
# runtime, exactly like inkling_server.py wraps `inkling_mlx`. Engine-managed
# like mlx_vlm.server / inkling_server: launched over ssh, health = GET
# /v1/models non-empty, chat proxied by OdyssAI-X to POST /v1/chat/completions.
#
# Muse-Glimmer is a harmony/"ATEM"-format model (gpt-oss family): the assistant
# emits a `to=self` reasoning channel, then a final channel, and tool calls as
# `to=<tool>` ATEM XML blocks. This server is channel-aware — reasoning →
# `reasoning_content`, final → `content`, ATEM → OpenAI `tool_calls`. Full
# multimodal: `<|patch|>` image sentinels are expanded to N placeholder rows
# (muse_glimmer_mlx.image.preprocess + image_placeholder_tokens) and the vision
# tower splices features into the text stream.
#
# Run: python muse_glimmer_server.py --model <dir> --host 0.0.0.0 --port 8081
# The `muse_glimmer_mlx` package must be importable in the venv (pip install).

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import numpy as np


# ── harmony control tokens (from the Muse-Glimmer tokenizer) ──────────────────
TOK_START = 200022    # <|start|>
TOK_MESSAGE = 200023  # <|message|>
TOK_EOM = 200007      # <|eom|>  (end of message → next channel, NOT end of turn)
TOK_EOT = 200008      # <|eot|>  (end of turn → stop)
TOK_EOS = 200001      # <|end_of_text|>
DEFAULT_EOS_IDS = {TOK_EOT, TOK_EOS}

# ATEM tool-call block: <atem:function_calls> … </atem:function_calls>
_ATEM_BLOCK = re.compile(r"<atem:function_calls>(.*?)</atem:function_calls>", re.DOTALL)
_ATEM_INVOKE = re.compile(r'<atem:invoke name="([^"]+)">(.*?)</atem:invoke>', re.DOTALL)
_ATEM_PARAM = re.compile(r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>', re.DOTALL)


def _parse_atem(text: str) -> list[dict]:
    """Parse ATEM (Anthropic-XML) tool-call blocks → OpenAI tool_calls.

    A parameter value is kept as a string unless it parses as JSON (so objects,
    arrays, numbers and booleans round-trip), matching how the chat template
    serialises arguments back the other way."""
    calls: list[dict] = []
    for block in _ATEM_BLOCK.findall(text):
        for name, body in _ATEM_INVOKE.findall(block):
            args: dict[str, Any] = {}
            for pname, pval in _ATEM_PARAM.findall(body):
                pval = pval.strip()
                try:
                    args[pname] = json.loads(pval)
                except Exception:
                    args[pname] = pval
            calls.append({
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            })
    return calls


# ── model + tokenizer (loaded once at startup) ────────────────────────────────
class _Engine:
    def __init__(self, model_path: str, wired_limit_gb: float,
                 cache_limit_gb: float = 8.0):
        # Right-size residency for a ~33GB model: wired_limit holds the weights
        # (+ KV + activations) resident without paging, and cache_limit bounds
        # MLX's freed-buffer cache so it RETURNS memory instead of hoarding up
        # to the wired ceiling. Without the cache bound, a long bench run climbs
        # to fill the wired allowance (191GB wired for a 33GB model, 2026-08-13).
        try:
            mx.set_wired_limit(int(wired_limit_gb * 1e9))
        except Exception as e:
            sys.stderr.write(f"[muse] set_wired_limit: {e}\n")
        try:
            mx.set_cache_limit(int(cache_limit_gb * 1e9))
        except Exception as e:
            sys.stderr.write(f"[muse] set_cache_limit: {e}\n")

        from muse_glimmer_mlx.load import load
        from muse_glimmer_mlx.image import preprocess, image_placeholder_tokens
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        from transformers import AutoTokenizer

        self._preprocess = preprocess
        self._placeholder_tokens = image_placeholder_tokens
        self._KVCache = KVCache
        self._RotatingKVCache = RotatingKVCache

        t0 = time.time()
        sys.stderr.write(f"[muse] loading {model_path} (eager, wired)…\n")
        self.model = load(model_path)
        mx.eval(self.model.parameters())
        sys.stderr.write(f"[muse] model ready in {time.time()-t0:.0f}s\n")

        self.cfg = self.model.config.text_config
        self.image_token_id = int(self.model.config.image_token_id)
        self.model_path = model_path

        self.tok = AutoTokenizer.from_pretrained(model_path)
        ct_path = Path(model_path) / "chat_template.jinja"
        self.chat_template = ct_path.read_text() if ct_path.exists() else None

        # eos ids from generation_config.json (falls back to {eot, eos})
        self.eos_ids = set(DEFAULT_EOS_IDS)
        try:
            gc = json.loads((Path(model_path) / "generation_config.json").read_text())
            eid = gc.get("eos_token_id")
            if isinstance(eid, int):
                self.eos_ids = {eid}
            elif isinstance(eid, (list, tuple)) and eid:
                self.eos_ids = {int(x) for x in eid}
        except Exception:
            pass

    # -- OpenAI messages → template messages, extracting images in order --------
    def _to_template_messages(self, messages: list[dict]):
        images: list = []
        out = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content")
            entry: dict[str, Any] = {"role": role}
            # passthrough of agent fields the harmony template understands
            for k in ("reasoning_content", "tool_call_id", "name", "recipient"):
                if m.get(k) is not None:
                    entry[k] = m[k]
            # The ATEM template requires tool_call.function.arguments to be a
            # dict (it raises on a JSON string — "cannot be parsed in the HF
            # jinja sandbox"). Callers/harnesses send OpenAI-standard string
            # arguments in prior-turn tool_calls, so normalise string → dict.
            if m.get("tool_calls") is not None:
                entry["tool_calls"] = _normalize_tool_calls(m["tool_calls"])
            if isinstance(content, str) or content is None:
                entry["content"] = content or ""
                out.append(entry)
                continue
            parts = []
            for p in content:
                t = p.get("type")
                if t == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif t in ("image_url", "image"):
                    images.append(_load_image(p))
                    parts.append({"type": "image"})
            entry["content"] = parts
            out.append(entry)
        return out, images

    def prepare(self, messages: list[dict], reasoning_strength: str = "high",
                tools: Optional[list] = None) -> dict:
        tmpl_messages, images = self._to_template_messages(messages)
        kwargs: dict[str, Any] = dict(add_generation_prompt=True, return_dict=True,
                                      reasoning_strength=reasoning_strength)
        if tools:
            kwargs["tools"] = tools
        if self.chat_template is not None:
            kwargs["chat_template"] = self.chat_template
        enc = self.tok.apply_chat_template(tmpl_messages, **kwargs)
        ids = list(enc["input_ids"])
        if ids and isinstance(ids[0], list):
            ids = ids[0]

        pixel_values = None
        grid_thw = None
        if images:
            all_pixels, grids, counts = [], [], []
            for img in images:
                px, gr = self._preprocess(img)
                all_pixels.append(px)
                grids.append(gr)
                counts.append(int(self._placeholder_tokens(gr)[0]))
            pixel_values = mx.array(np.concatenate(all_pixels, axis=0).astype(np.float32))
            grid_thw = np.concatenate(grids, axis=0)
            ids = _expand_image_placeholders(ids, self.image_token_id, counts)

        return {"input_ids": ids, "pixel_values": pixel_values, "grid_thw": grid_thw}

    def _make_cache(self):
        return [self._RotatingKVCache(max_size=self.cfg.sliding_window, keep=0)
                if self.cfg.is_sliding(i) else self._KVCache()
                for i in range(self.cfg.num_hidden_layers)]

    # -- channel-aware harmony decode: yields ("reasoning"|"content", piece) ----
    #    tool-channel text is buffered and surfaced via self._last["tool_calls"].
    def stream(self, inp: dict, max_tokens: int, temperature: float,
               top_p: float, stop_flag):
        ids = list(inp["input_ids"])
        pixel_values = inp.get("pixel_values")
        grid_thw = inp.get("grid_thw")
        cache = self._make_cache()
        prompt_len = len(ids)

        # Prefill (with media on the first forward) then step one token at a time.
        if pixel_values is not None:
            logits = self.model(mx.array([ids]), pixel_values=pixel_values,
                                grid_thw=grid_thw, cache=cache)
        else:
            logits = self.model(mx.array([ids]), cache=cache)
        next_id = _sample(logits[0, -1], temperature, top_p)
        mx.eval(next_id)

        # We enter mid-header: the prompt ended on `<|start|>assistant`, so the
        # model's first tokens complete the header (e.g. " to=self") up to
        # <|message|>. State machine over the harmony structural tokens.
        state = "header"          # 'header' | 'body'
        channel = None            # 'reasoning' | 'final' | 'tool'
        header_ids: list[int] = []
        body_ids: list[int] = []
        detok_prev = ""
        tool_text_parts: list[str] = []
        n_out = 0

        def classify(header: str) -> str:
            h = header.strip()
            if "to=self" in h:
                return "reasoning"
            m = re.search(r"to=(\S+)", h)
            if m and m.group(1) not in ("user", "assistant"):
                return "tool"
            return "final"

        while n_out < max_tokens:
            tid = int(next_id.item())
            n_out += 1

            if tid in self.eos_ids:
                break

            if tid == TOK_START:
                # close any open body, restart a header
                if channel == "tool":
                    tool_text_parts.append(self.tok.decode(body_ids))
                state, channel = "header", None
                header_ids, body_ids, detok_prev = [], [], ""
            elif tid == TOK_EOM:
                if channel == "tool":
                    tool_text_parts.append(self.tok.decode(body_ids))
                state, channel = "await_start", None
                header_ids, body_ids, detok_prev = [], [], ""
            elif tid == TOK_MESSAGE:
                channel = classify(self.tok.decode(header_ids))
                state = "body"
                body_ids, detok_prev = [], ""
            else:
                if state in ("header", "await_start"):
                    # tokens before <|message|> are header text (or stray after
                    # <|eom|> before the next <|start|> — fold into header)
                    header_ids.append(tid)
                elif state == "body":
                    body_ids.append(tid)
                    if channel in ("reasoning", "final"):
                        text = self.tok.decode(body_ids)
                        if len(text) > len(detok_prev):
                            yield (channel, text[len(detok_prev):])
                            detok_prev = text

            if stop_flag and stop_flag():
                break
            logits = self.model(mx.array([[tid]], dtype=mx.int32), cache=cache)
            next_id = _sample(logits[0, -1], temperature, top_p)
            mx.eval(next_id)

        # flush a trailing open tool body (no terminating <|eom|>/<|eot|>)
        if channel == "tool" and body_ids:
            tool_text_parts.append(self.tok.decode(body_ids))

        tool_calls = _parse_atem("".join(tool_text_parts)) if tool_text_parts else []
        finish = "tool_calls" if tool_calls else ("stop" if n_out < max_tokens else "length")
        self._last = {"prompt_tokens": prompt_len, "completion_tokens": n_out,
                      "finish": finish, "tool_calls": tool_calls}


def _normalize_tool_calls(tool_calls: list) -> list:
    """Return tool_calls with each function.arguments as a dict (parsing a JSON
    string when needed). The ATEM chat template raises on string arguments."""
    out = []
    for tc in tool_calls or []:
        fn = dict((tc.get("function") or {}))
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                fn["arguments"] = json.loads(args) if args.strip() else {}
            except Exception:
                fn["arguments"] = {}
        elif args is None:
            fn["arguments"] = {}
        ntc = dict(tc)
        ntc["function"] = fn
        out.append(ntc)
    return out


def _expand_image_placeholders(ids: list[int], image_token_id: int,
                               counts: list[int]) -> list[int]:
    """Each image renders as ONE `<|patch|>` (image_token_id) in the template;
    the vision tower produces `counts[k]` features for image k. Expand the k-th
    sentinel into counts[k] copies so placeholder rows == vision features."""
    out: list[int] = []
    k = 0
    for t in ids:
        if t == image_token_id:
            n = counts[k] if k < len(counts) else 1
            out.extend([image_token_id] * n)
            k += 1
        else:
            out.append(t)
    return out


def _sample(logits: mx.array, temperature: float, top_p: float) -> mx.array:
    if temperature <= 0.0:
        return mx.argmax(logits)
    logits = logits * (1.0 / temperature)
    if 0.0 < top_p < 1.0:
        probs = mx.softmax(logits, axis=-1)
        order = mx.argsort(-probs)
        sp = mx.take(probs, order)
        csum = mx.cumsum(sp)
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


# ── FastAPI app ───────────────────────────────────────────────────────────────
def build_app(engine: _Engine):
    from fastapi import Body, FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()
    model_id = engine.model_path

    _EFFORT_TO_STRENGTH = {"none": "low", "minimal": "low", "low": "low",
                           "medium": "medium", "high": "high", "max": "high"}

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
        max_tokens = int(body.get("max_tokens") or 1024)
        temperature = float(body.get("temperature") if body.get("temperature") is not None else 0.0)
        top_p = float(body.get("top_p") if body.get("top_p") is not None else 1.0)
        stream = bool(body.get("stream"))
        eff = (body.get("reasoning_effort") or "high")
        strength = _EFFORT_TO_STRENGTH.get(eff, "high")
        tools = body.get("tools") or None
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        inp = engine.prepare(messages, reasoning_strength=strength, tools=tools)

        if stream:
            def sse():
                head = {"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
                yield f"data: {json.dumps(head)}\n\n"
                for kind, piece in engine.stream(inp, max_tokens, temperature, top_p, None):
                    field = "reasoning_content" if kind == "reasoning" else "content"
                    ch = {"id": cid, "object": "chat.completion.chunk", "created": created,
                          "model": model_id,
                          "choices": [{"index": 0, "delta": {field: piece}, "finish_reason": None}]}
                    yield f"data: {json.dumps(ch)}\n\n"
                fin = getattr(engine, "_last", {})
                delta: dict[str, Any] = {}
                if fin.get("tool_calls"):
                    delta["tool_calls"] = fin["tool_calls"]
                tail = {"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": delta,
                                     "finish_reason": fin.get("finish", "stop")}],
                        "usage": {"prompt_tokens": fin.get("prompt_tokens", 0),
                                  "completion_tokens": fin.get("completion_tokens", 0),
                                  "total_tokens": fin.get("prompt_tokens", 0) + fin.get("completion_tokens", 0)}}
                yield f"data: {json.dumps(tail)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        reasoning_parts, content_parts = [], []
        for kind, piece in engine.stream(inp, max_tokens, temperature, top_p, None):
            (reasoning_parts if kind == "reasoning" else content_parts).append(piece)
        fin = getattr(engine, "_last", {})
        message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if fin.get("tool_calls"):
            message["tool_calls"] = fin["tool_calls"]
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": model_id,
            "choices": [{"index": 0, "message": message,
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
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--wired-limit-gb", type=float, default=64.0)
    ap.add_argument("--cache-limit-gb", type=float, default=8.0)
    args = ap.parse_args()

    import uvicorn
    engine = _Engine(args.model, args.wired_limit_gb, args.cache_limit_gb)
    app = build_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
