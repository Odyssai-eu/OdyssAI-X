#!/usr/bin/env python3
"""emit — dispatch one relay envelope to N local models in parallel.

The second half of the relay circuit: `relay` defines the envelope, this tool
sends it. One envelope, several emitters, one structured summary — plus an
optional smoke-execution of the python blocks each model returned (the 07/10
finding: local emitters never run their own code, so the harness does).

    python3 scripts/emit.py --envelope RELAY.md --models qwen3-coder-next,mimo-v2-6-flash-rl-q9-fixed
    python3 scripts/emit.py --envelope RELAY.md --models qwen3-coder-next --warmup --run-blocks

Output: <envelope-stem>.<model>.md (the reply) + <envelope-stem>.summary.json.
Never treats a model reply as validated: --run-blocks reports, the reviewer
(a stronger model or a human) decides.

Config: same chain as ox — $ODYSSAI_BASE_URL, else ~/.config/ox/config.json
{"base_url"}, else http://localhost:8000. No API key needed on the LAN.
"""
import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

OX_CONFIG_PATH = os.path.expanduser("~/.config/ox/config.json")


def _base_url():
    if os.environ.get("ODYSSAI_BASE_URL"):
        return os.environ["ODYSSAI_BASE_URL"]
    try:
        with open(OX_CONFIG_PATH) as f:
            u = (json.load(f) or {}).get("base_url")
            if u:
                return u
    except Exception:
        pass
    return "http://localhost:8000"


def _chat(base_url, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return data, time.time() - t0


def _text_of(reply):
    msg = (reply.get("choices") or [{}])[0].get("message") or {}
    c = msg.get("content")
    if isinstance(c, list):
        return "\n".join(b.get("text", "") for b in c if b.get("type") == "text")
    return c or ""


def extract_python_blocks(text):
    """The fenced ```python blocks, in order. Marks unclosed fences as errors
    rather than silently returning partial code."""
    blocks, cur = [], None
    for line in text.splitlines():
        if line.strip().startswith("```python"):
            cur = []
        elif line.strip() == "```" and cur is not None:
            blocks.append("\n".join(cur))
            cur = None
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        blocks.append("\n".join(cur))
        blocks[-1] = None          # unclosed fence: code possibly truncated
    return blocks


def run_block(i, code, python_bin):
    r = subprocess.run([python_bin, "-c", code], capture_output=True,
                       text=True, timeout=120)
    return {"block": i, "rc": r.returncode,
            "stderr_tail": "\n".join(r.stderr.strip().splitlines()[-3:])}


def emit_one(base_url, model, prompt, max_tokens, timeout, warmup):
    if warmup:
        try:
            _chat(base_url, model, "Say ok.", 10, 60)
        except Exception:
            pass                    # warm-up is best effort, never fatal
    reply, wall = _chat(base_url, model, prompt, max_tokens, timeout)
    return model, reply, wall


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--envelope", required=True, help="RELAY-*.md file")
    ap.add_argument("--models", required=True,
                    help="comma-separated model aliases")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--timeout", type=int, default=480)
    ap.add_argument("--warmup", action="store_true",
                    help="one tiny request first (cold pools measure 2x slow)")
    ap.add_argument("--run-blocks", action="store_true",
                    help="smoke-execute returned ```python blocks")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for --run-blocks (default: this one)")
    args = ap.parse_args()

    base_url = _base_url()
    envelope = open(args.envelope).read()
    stem = re.sub(r"\.md$", "", args.envelope)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    summary = {"envelope": args.envelope, "base_url": base_url,
               "emitted_at": time.strftime("%Y-%m-%d %H:%M:%S"), "models": []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as ex:
        futs = {ex.submit(emit_one, base_url, m, envelope,
                          args.max_tokens, args.timeout, args.warmup): m
                for m in models}
        results = {}
        for f in concurrent.futures.as_completed(futs):
            m = futs[f]
            try:
                results[m] = f.result()
            except Exception as e:
                results[m] = (m, {"error": str(e)}, 0.0)

    for m in models:
        _, reply, wall = results[m]
        entry = {"model": m, "wall_s": round(wall, 1)}
        if "error" in reply:
            entry["error"] = reply["error"]
            summary["models"].append(entry)
            continue
        text = _text_of(reply)
        open(f"{stem}.{m}.md", "w").write(text)
        entry.update({
            "reply_file": f"{stem}.{m}.md",
            "output_tokens": (reply.get("usage") or {}).get(
                "completion_tokens", (reply.get("usage") or {}).get("output_tokens")),
            "finish_reason": (reply.get("choices") or [{}])[0].get("finish_reason"),
        })
        if args.run_blocks:
            blocks = extract_python_blocks(text)
            entry["blocks"] = []
            for i, code in enumerate(blocks):
                if code is None:
                    entry["blocks"].append({"block": i, "rc": None,
                                            "stderr_tail": "UNCLOSED FENCE"})
                else:
                    try:
                        entry["blocks"].append(run_block(i, code, args.python))
                    except subprocess.TimeoutExpired:
                        entry["blocks"].append({"block": i, "rc": None,
                                                "stderr_tail": "TIMEOUT 120s"})
        summary["models"].append(entry)

    out = f"{stem}.summary.json"
    json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)
    for e in summary["models"]:
        if "error" in e:
            print(f"  {e['model']}: ERROR {e['error']}")
            continue
        line = (f"  {e['model']}: {e['output_tokens']} tok in {e['wall_s']}s "
                f"({e['finish_reason']}) -> {e['reply_file']}")
        if args.run_blocks:
            line += " | blocks: " + ", ".join(
                f"#{b['block']} rc={b['rc']}" for b in e["blocks"])
        print(line)
    print(f"summary: {out}")
    # Exit code: 0 if every model ANSWERED (their code quality is the
    # reviewer's business, not the harness'). A model error is a harness
    # failure: 1.
    sys.exit(1 if any("error" in e for e in summary["models"]) else 0)


if __name__ == "__main__":
    main()
