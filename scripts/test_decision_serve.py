#!/usr/bin/env python3
"""Tests for scripts/decision/decision_serve.py without MLX: a fake decider stands in for
MLXDecider; the vendored decision_core (pure Python) builds the options exactly as in prod.

    python3 scripts/test_decision_serve.py
"""
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

os.environ["PROMPT_STYLE"] = "semif"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "decision"))

import decision_core  # noqa: E402
import decision_serve as ds  # noqa: E402

FAILS = []
N = [0]


def check(name, cond, detail=""):
    N[0] += 1
    print(("OK  " if cond else "FAIL") + f" {N[0]} {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class FakeDecider:
    """First option gets 0.7, the rest share 0.3; records the calling thread and the path used."""

    def __init__(self):
        self.threads, self.calls = set(), []

    def _p(self, opts):
        n = len(opts)
        rest = 0.3 / (n - 1)
        return {lab: (0.7 if i == 0 else rest) for i, (lab, _) in enumerate(opts)}

    def dist(self, state, q, opts):
        self.threads.add(threading.current_thread().name)
        self.calls.append("dist")
        return self._p(opts), 10

    def dist_many_cached(self, state, items):
        self.threads.add(threading.current_thread().name)
        self.calls.append("many")
        return [(self._p(o), 10) for _, o in items]


def start(sym=False):
    fake = FakeDecider()
    w = ds.Worker("/nonexistent", sym, factory=lambda: (fake, decision_core))
    w.start()
    assert w.ready.wait(5)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ds.make_handler(w, "eikos-test", {"prompt_version": "letter-v1-semif", "readout": "letter-logit"}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return fake, w, srv, f"http://127.0.0.1:{srv.server_address[1]}"


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


# --- config gate ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as d:
    json.dump({"prompt_version": "letter-v1-semif", "readout": "letter-logit"}, open(os.path.join(d, "decision_config.json"), "w"))
    check("known config accepted", ds.read_decision_config(d)["readout"] == "letter-logit")
    json.dump({"prompt_version": "letter-v9-x", "readout": "letter-logit"}, open(os.path.join(d, "decision_config.json"), "w"))
    try:
        ds.read_decision_config(d)
        check("unknown prompt_version refused", False)
    except SystemExit:
        check("unknown prompt_version refused", True)
with tempfile.TemporaryDirectory() as d:
    try:
        ds.read_decision_config(d)
        check("folder without decision_config refused", False)
    except SystemExit:
        check("folder without decision_config refused", True)

# --- answer format (same as upstream serve.py) ----------------------------------------------
a = ds.format_answer({"type": "noul"}, {"yes": 0.8, "no": 0.2})
check("noul format", a["value"] is True and a["probability"] == 0.8 and a["confidence"] == 0.8)
a = ds.format_answer({"type": "score"}, {"0": 0.1, "1": 0.6, "2": 0.3})
check("score format", a["score"] == 1 and abs(a["expected"] - 1.2) < 1e-9)
a = ds.format_answer({"type": "choice"}, {"a": 0.2, "b": 0.8})
check("choice format", a["choice"] == "b" and a["confidence"] == 0.8)

# --- HTTP + worker thread -------------------------------------------------------------------
fake, w, srv, base = start()
code, h = get(base + "/health")
check("health 200 when loaded", code == 200 and h["ok"] and h["kind"] == "decision" and h["loaded"] == ["eikos-test"])
code, m = get(base + "/v1/models")
check("/v1/models lists the model path once loaded", code == 200 and m["data"][0]["id"] == "/nonexistent")
code, r = post(base + "/v1/systemone", {"state": "Order 1,500 XYZ, equity 48k, rule 4.2 caps at 50%.",
                                       "questions": {"allowed": {"type": "noul", "instructions": "Allowed?",
                                                                 "criteria": {"true": "ok", "false": "breach"}}}})
check("single question → 200", code == 200 and r["answers"]["allowed"]["value"] is True, r)
check("single question uses dist", fake.calls[-1] == "dist")
code, r = post(base + "/v1/systemone", {"state": "s", "questions": {
    "allowed": {"type": "noul", "instructions": "?", "criteria": {"true": "y", "false": "n"}},
    "action": {"type": "choice", "instructions": "?", "criteria": {"execute": "go", "reject": "no"}},
    "risk": {"type": "score", "instructions": "?", "criteria": ["low", "moderate", "high"]}}})
check("three questions → 200", code == 200 and set(r["answers"]) == {"allowed", "action", "risk"}, r)
check("several questions batched in one pass", fake.calls[-1] == "many")
check("choice answer key", r["answers"]["action"]["choice"] == "execute")
check("score answer", r["answers"]["risk"]["score"] == 0)
check("usage counts input tokens", r["usage"]["input_tokens"] == 30 and r["usage"]["output_tokens"] == 0)
check("model name reported", r["model"] == "eikos-test")
check("all MLX calls on the one worker thread", fake.threads == {"mlx-decider"}, fake.threads)
code, r = post(base + "/v1/systemone", {"state": "s", "questions": {"x": {"type": "choice", "instructions": "?", "criteria": {"only": "one"}}}})
check("one option → 422", code == 422)
code, r = post(base + "/v1/systemone", {"state": "s", "questions": {}})
check("empty questions → 422", code == 422)
code, r = post(base + "/v1/evaluate", {"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}})
check("/v1/evaluate alias", code == 200)
code, r = post(base + "/v1/chat/completions", {"messages": []})
check("chat path → 404", code == 404)
srv.shutdown()

# --- sym mode averages the reversed order ------------------------------------------------------
fake, w, srv, base = start(sym=True)
code, r = post(base + "/v1/systemone", {"state": "s", "questions": {"q": {"type": "choice", "instructions": "?",
                                                                          "criteria": {"a": "A", "b": "B"}}}})
p = r["answers"]["q"]["probabilities"]
check("sym averages both orders", code == 200 and abs(p["a"] - 0.5) < 1e-9 and abs(p["b"] - 0.5) < 1e-9, p)
srv.shutdown()

# --- load failure → 503, never a fake answer ------------------------------------------------------
def boom():
    raise RuntimeError("weights missing")
w = ds.Worker("/nonexistent", False, factory=boom)
w.start()
w.ready.wait(5)
srv = ThreadingHTTPServer(("127.0.0.1", 0), ds.make_handler(w, "x", {}))
threading.Thread(target=srv.serve_forever, daemon=True).start()
b = f"http://127.0.0.1:{srv.server_address[1]}"
code, h = get(b + "/health")
check("health 503 after load failure", code == 503 and not h["ok"] and "weights missing" in (h["error"] or ""))
code, m = get(b + "/v1/models")
check("/v1/models empty after load failure (probe stays not-ready)", code == 200 and m["data"] == [])
code, r = post(b + "/v1/systemone", {"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}})
check("decision 503 after load failure", code == 503)
srv.shutdown()

# --- abandoned jobs are dropped, the queue is bounded -------------------------------------
import socket as _socket
import time as _time


class SlowDecider(FakeDecider):
    def dist(self, state, q, opts):
        _time.sleep(1.0)
        return super().dist(state, q, opts)


slow = SlowDecider()
ws = ds.Worker("/nonexistent", False, factory=lambda: (slow, decision_core))
ws.start(); ws.ready.wait(5)
srvs = ThreadingHTTPServer(("127.0.0.1", 0), ds.make_handler(ws, "slow", {}))
threading.Thread(target=srvs.serve_forever, daemon=True).start()
bs = f"http://127.0.0.1:{srvs.server_address[1]}"
Q1 = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}}


def post_and_drop(url_port, body):
    """Send a request, then close the socket without reading the answer."""
    raw = json.dumps(body).encode()
    s = _socket.create_connection(("127.0.0.1", url_port))
    s.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              + f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw)
    _time.sleep(0.2)
    s.close()


port_s = srvs.server_address[1]
threading.Thread(target=post, args=(bs + "/v1/systemone", Q1), daemon=True).start()   # occupies the worker
_time.sleep(0.2)
for _ in range(3):
    post_and_drop(port_s, Q1)                                                             # 3 abandoned
_time.sleep(4.5)
check("abandoned requests are never computed", (slow.calls.count("dist") if hasattr(slow, "calls") else None, ws.skipped) == (None, 3) or ws.skipped == 3, ws.skipped)
check("served only the live one", ws.served == 1, (ws.served, ws.skipped))

ds.MAX_QUEUE = 2
threading.Thread(target=post, args=(bs + "/v1/systemone", Q1), daemon=True).start()
_time.sleep(0.2)
for _ in range(2):
    threading.Thread(target=post, args=(bs + "/v1/systemone", Q1), daemon=True).start()
_time.sleep(0.3)
code, r = post(bs + "/v1/systemone", Q1)
check("queue full → 503 at once", code == 503 and "busy" in r.get("error", ""), (code, r))
ds.MAX_QUEUE = 16
_time.sleep(3.5)
srvs.shutdown()

# --- Julia backend (fake engine, real julia.data.sequence encoding) -----------------------
try:
    import torch  # noqa: F401  (julia.data imports torch)
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

if HAVE_TORCH:
    class FakeTok:
        mask_token, mask_token_id, cls_token_id, sep_token_id, pad_token_id = "[MASK]", 4, 1, 2, 0

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [10 + (ord(c) % 50) for c in text]}

    class FakeCollate:
        max_length, head_length = 256, 64

    class FakeJuliaEngine:
        """Option 1 wins (logit 2.0, others 0.0); records the rows it scored."""
        def __init__(self):
            self.tokenizer, self.collate, self.rows = FakeTok(), FakeCollate(), []

        def logits(self, rows):
            self.rows = rows
            return [[0.0] + [2.0] + [0.0] * (len(r["options"]) - 2) for r in rows]

    fj_engine = FakeJuliaEngine()
    wj = ds.Worker("/nonexistent", False, cfg={"backend": "julia"},
                   factory=lambda: ds.JuliaBackend(fj_engine, decision_core, "cpu"))
    wj.start(); wj.ready.wait(5)
    srvj = ThreadingHTTPServer(("127.0.0.1", 0), ds.make_handler(wj, "julia-test", {"backend": "julia"}))
    threading.Thread(target=srvj.serve_forever, daemon=True).start()
    bj = f"http://127.0.0.1:{srvj.server_address[1]}"
    code, r = post(bj + "/v1/systemone", {"state": "I was charged twice.", "questions": {
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Billing", "shipping": "Shipping", "access": "Access"}},
        "ok": {"type": "noul", "instructions": "Refund?", "criteria": {"true": "refund", "false": "no refund"}},
        "bare": {"type": "noul", "instructions": "Urgent?"},
        "sev": {"type": "score", "instructions": "Severity?", "criteria": ["low", "mid", "high"]}}})
    check("julia: 200", code == 200, r)
    a = r.get("answers", {})
    check("julia: choice picks the highest logit (option 2 = shipping)", a.get("team", {}).get("choice") == "shipping")
    check("julia: noul options sent as [false, true]", fj_engine.rows[1]["options"] == ["no refund", "refund"])
    check("julia: noul true = second logit", a.get("ok", {}).get("value") is True)
    check("julia: noul without criteria uses literal false/true", fj_engine.rows[2]["options"] == ["false", "true"])
    check("julia: score index", a.get("sev", {}).get("score") == 1)
    check("julia: one batch for all questions", len(fj_engine.rows) == 4)
    check("julia: input tokens counted", r["usage"]["input_tokens"] > 0)
    code, r = post(bj + "/v1/systemone", {"state": "x" * 2000, "questions": {
        "q": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}}}})
    check("julia: state beyond max_length flagged truncated", r["answers"]["q"].get("truncated") is True, r)
    code, r = post(bj + "/v1/systemone", {"state": "s", "questions": {
        "q": {"type": "choice", "instructions": "?", "criteria": {f"o{i}": f"O{i}" for i in range(21)}}}})
    check("julia: more than 20 options → 422", code == 422)
    srvj.shutdown()
    with tempfile.TemporaryDirectory() as d:
        json.dump({"format_version": 1, "architecture": "JuliaDecisionModel"}, open(os.path.join(d, "julia_config.json"), "w"))
        json.dump({"max_length": 8192, "head_length": 512, "weights_sha256": "ab"}, open(os.path.join(d, "inference-policy.json"), "w"))
        c = ds.read_decision_config(d)
        check("julia config detected", (c["backend"], c["prompt_version"], c["readout"], c["max_length"]) == ("julia", "julia-v1", "julia-markers", 8192))
else:
    print("SKIP julia backend tests (no torch)")

# letter readout on a long state: prefill in blocks, KV-bounded question batches
try:
    import mlx.core as mx
except ImportError:
    mx = None
if mx is not None:
    class Cell:
        def __init__(self):
            self.state = mx.zeros((1,))

    calls = []

    def inner(ids, cache=None):
        calls.append(ids.shape[1])
        return ids.astype(mx.float32)[..., None]

    cp = ds.ChunkedPrefill(inner, step=4)
    h = cp(mx.arange(10)[None], [Cell()])
    check("long input goes through the cache in blocks", calls == [4, 4, 2], calls)
    check("blocks concatenate back to the one-shot output", h[0, :, 0].tolist() == list(range(10)), h.shape)
    calls.clear()
    cp(mx.arange(10)[None], None)
    check("no cache → one call, unchanged", calls == [10], calls)
    lb = ds.LetterBackend.__new__(ds.LetterBackend)
    lb._kvpt = 65536                      # Eikos-27B: 16 attention layers x 4 KV heads x 256 x 2 x bf16
    check("short state keeps upstream's batch of 32", lb.batch_size(3000) == 32, lb.batch_size(3000))
    check("33k-token state: batch sized to the KV budget", 1 <= lb.batch_size(33000) <= 4, lb.batch_size(33000))
else:
    print("SKIP chunked prefill tests (no mlx)")

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
