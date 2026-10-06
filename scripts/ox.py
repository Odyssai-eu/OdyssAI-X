#!/usr/bin/env python3
"""ox — command-line client for an OdyssAI-X server, modelled on MiniMax's mmx.

Any agent with a shell (Claude Code, Codex, opencode, a script) can call an
OdyssAI-X model the way it calls `mmx text chat`: same flags, same Anthropic
Messages body on stdout with --output json, tool calls returned as `tool_use`
blocks. The caller runs the tools and sends `tool_result` blocks back, so the
model reaches exactly what the caller gives it.

Every alias works (local pools, or:*, MI:*, ...): ox always talks to
POST /v1/chat/completions and converts Anthropic <-> OpenAI on both sides.

  ox models [--output json]
  ox text chat --model or:kimi --message "Hello"
  ox text chat --model glm-5-3-flash-q6h16 --system "..." \
      --messages-file history.json --tool tools/read_file.json --output json

Server: --base-url, else $ODYSSAI_BASE_URL, else ~/.config/ox/config.json
{"base_url": ...}, else http://localhost:8000. Token: --api-key, else
$ODYSSAI_API_KEY (sent as a Bearer token), else none.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid

VERSION = "0.1.0"
CONFIG_PATH = os.path.expanduser("~/.config/ox/config.json")


# ---------------------------------------------------------------- config / http

def _config() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _base_url(arg):
    url = arg or os.environ.get("ODYSSAI_BASE_URL") or _config().get("base_url") \
        or "http://localhost:8000"
    return url.rstrip("/")


def _api_key(arg):
    return arg or os.environ.get("ODYSSAI_API_KEY") or _config().get("api_key")


def _request(method, url, key, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    return urllib.request.urlopen(req, timeout=timeout)


def _http_error(e):
    try:
        detail = json.loads(e.read().decode() or "{}")
    except ValueError:
        detail = {}
    msg = detail.get("detail") or detail.get("error") or e.reason
    if isinstance(msg, dict):
        msg = msg.get("message") or json.dumps(msg)
    return {"type": "error", "error": {"type": f"http_{e.code}", "message": str(msg)}}


# ------------------------------------------------- Anthropic -> OpenAI (request)

def _text_of(content):
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if isinstance(b, str):
            parts.append(b)
        elif b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts)


def _to_openai_messages(messages, system):
    out = [{"role": "system", "content": system}] if system else []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
            calls = [{
                "id": b["id"], "type": "function",
                "function": {"name": b["name"],
                             "arguments": json.dumps(b.get("input", {}))},
            } for b in content if b.get("type") == "tool_use"]
            msg = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        # user turn: tool_result blocks become role=tool messages, text stays user
        text_parts = []
        for b in content:
            if b.get("type") == "tool_result":
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                            "content": _text_of(b.get("content", ""))})
            elif b.get("type") == "text":
                text_parts.append(b.get("text", ""))
        if text_parts:
            out.append({"role": "user", "content": "\n".join(text_parts)})
    return out


def _to_openai_tools(tools):
    return [{"type": "function", "function": {
        "name": t["name"],
        "description": t.get("description", ""),
        "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
    }} for t in tools]


# ------------------------------------------------ OpenAI -> Anthropic (response)

_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
         "function_call": "tool_use", "content_filter": "end_turn"}


def _to_anthropic(resp, model):
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    blocks = []
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning})
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {"_raw_arguments": fn.get("arguments")}
        blocks.append({"type": "tool_use", "id": c.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                       "name": fn.get("name"), "input": args})
    stop = _STOP.get(choice.get("finish_reason"), "end_turn")
    if any(b["type"] == "tool_use" for b in blocks):
        stop = "tool_use"
    usage = resp.get("usage") or {}
    return {
        "id": resp.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message", "role": "assistant",
        "model": resp.get("model") or model,
        "content": blocks, "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                  "output_tokens": usage.get("completion_tokens", 0)},
    }


# ----------------------------------------------------------------------- input

def _load_json_arg(value):
    """--tool accepts inline JSON or a file path, like mmx."""
    v = value.strip()
    if v.startswith("{") or v.startswith("["):
        return json.loads(v)
    with open(os.path.expanduser(v)) as f:
        return json.load(f)


def _gather_messages(a):
    messages = []
    if a.messages_file:
        src = sys.stdin if a.messages_file == "-" else open(a.messages_file)
        data = json.load(src)
        messages = data.get("messages", data) if isinstance(data, dict) else data
    for m in a.message or []:
        role, text = "user", m
        head, sep, rest = m.partition(":")
        if sep and head in ("user", "assistant", "system"):
            role, text = head, rest
        if role == "system":
            a.system = (a.system + "\n" if a.system else "") + text
        else:
            messages.append({"role": role, "content": text})
    return messages


# -------------------------------------------------------------------- commands

def cmd_chat(a):
    messages = _gather_messages(a)
    if not messages:
        sys.exit("ox: no messages — pass --message or --messages-file")
    tools = []
    for t in a.tool or []:
        loaded = _load_json_arg(t)
        tools.extend(loaded if isinstance(loaded, list) else [loaded])
    body = {"model": a.model, "messages": _to_openai_messages(messages, a.system),
            "max_tokens": a.max_tokens, "stream": bool(a.stream and not tools
                                                       and a.output != "json")}
    if tools:
        body["tools"] = _to_openai_tools(tools)
    if a.temperature is not None:
        body["temperature"] = a.temperature
    if a.top_p is not None:
        body["top_p"] = a.top_p
    url = f"{_base_url(a.base_url)}/v1/chat/completions"
    if a.verbose:
        sys.stderr.write(f"> POST {url} model={a.model} tools={len(tools)}\n")
    try:
        r = _request("POST", url, _api_key(a.api_key), body, timeout=a.timeout)
    except urllib.error.HTTPError as e:
        err = _http_error(e)
        print(json.dumps(err) if a.output == "json" else f"ox: {err['error']['message']}",
              file=sys.stdout if a.output == "json" else sys.stderr)
        return 1
    except urllib.error.URLError as e:
        sys.exit(f"ox: cannot reach {url}: {e.reason}")

    if body["stream"]:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                delta = json.loads(chunk)["choices"][0].get("delta", {})
            except (ValueError, KeyError, IndexError):
                continue
            if delta.get("content"):
                sys.stdout.write(delta["content"])
                sys.stdout.flush()
        sys.stdout.write("\n")
        return 0

    resp = json.loads(r.read().decode())
    if a.verbose:
        sys.stderr.write(f"< {r.status} finish={(resp.get('choices') or [{}])[0].get('finish_reason')}\n")
    out = _to_anthropic(resp, a.model)
    if a.output == "json":
        print(json.dumps(out, ensure_ascii=False, indent=None if a.quiet else 2))
        return 0
    for b in out["content"]:
        if b["type"] == "text":
            print(b["text"])
        elif b["type"] == "tool_use":
            print(f"[tool_use {b['name']}] {json.dumps(b['input'], ensure_ascii=False)}")
    return 0


def cmd_models(a):
    url = f"{_base_url(a.base_url)}/v1/models"
    try:
        data = json.loads(_request("GET", url, _api_key(a.api_key), timeout=30).read().decode())
    except urllib.error.HTTPError as e:
        sys.exit(f"ox: {_http_error(e)['error']['message']}")
    except urllib.error.URLError as e:
        sys.exit(f"ox: cannot reach {url}: {e.reason}")
    ids = [m.get("id") for m in data.get("data", [])]
    print(json.dumps(ids, indent=2) if a.output == "json" else "\n".join(ids))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="ox", description="OdyssAI-X CLI (mmx-compatible)")
    p.add_argument("--version", action="version", version=f"ox {VERSION}")
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("--base-url")
    g.add_argument("--api-key")
    g.add_argument("--output", choices=["text", "json"], default="text")
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", parents=[g], help="list the server's model aliases") \
        .set_defaults(func=cmd_models)

    text = sub.add_parser("text", help="text generation")
    tsub = text.add_subparsers(dest="tcmd", required=True)
    c = tsub.add_parser("chat", parents=[g], help="chat completion (Anthropic Messages body)")
    c.add_argument("--model", required=True)
    c.add_argument("--message", action="append",
                   help="message text (repeatable, prefix role: to set the role)")
    c.add_argument("--messages-file", help="JSON messages array, or - for stdin")
    c.add_argument("--system")
    c.add_argument("--max-tokens", type=int, default=4096)
    c.add_argument("--temperature", type=float)
    c.add_argument("--top-p", type=float)
    c.add_argument("--stream", action="store_true", help="stream text (ignored with tools or --output json)")
    c.add_argument("--tool", action="append", help="tool definition, JSON or file path (repeatable)")
    c.add_argument("--timeout", type=int, default=900, help="seconds")
    c.set_defaults(func=cmd_chat)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
