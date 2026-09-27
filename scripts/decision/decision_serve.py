#!/usr/bin/env python3
"""decision_serve — typed-decision server for OdyssAI-X decision pools (MLX, one node).

Serves a *decision model* (weights that answer typed questions by reading the next-token
logits over option letters, e.g. Eikos-27B) behind the TypeSafe / Jev wire protocol that
OdyssAI-X already routes: `POST /v1/systemone` (and `/v1/evaluate`), `GET /health`.

- Readout: the vendored, byte-identical `decision_core.py` + `mlx_decide.py` (MIT, see
  UPSTREAM.md). Nothing is imported from the model folder: it only provides the weights,
  `decision_config.json` and `calib.json`. An unknown `prompt_version` / `readout` is
  refused at start instead of being answered with a format the weights were not trained on.
- Threads: MLX keeps a default stream per thread, and a graph built on one thread fails
  when evaluated on another ("There is no Stream(gpu, N) in current thread", mlx-vlm#2352).
  The model is therefore loaded and run on ONE worker thread; HTTP handler threads only
  enqueue jobs and wait for them.
- Response shape: the same as upstream `serve.py` v1.2 (`answers`, `usage`, `latency_s`).

usage: python decision_serve.py --model <dir> [--port 8095] [--host 0.0.0.0] [--name alias] [--sym]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights are local, never download

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # the vendored readout, never the model folder

VERSION = "odyssai-decision-1.0"
KNOWN_PROMPT_VERSIONS = {"letter-v1-semif", "letter-v1-ours"}
KNOWN_READOUTS = {"letter-logit"}
MAX_BODY = 8 * 1024 * 1024
JOB_TIMEOUT_S = float(os.environ.get("DECISION_JOB_TIMEOUT_S", "600"))


def log(msg: str) -> None:
    print(f"[decision_serve] {msg}", flush=True)


def read_decision_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, "decision_config.json")
    try:
        cfg = json.load(open(path))
    except FileNotFoundError:
        raise SystemExit(f"not a decision model: {path} is missing")
    pv, ro = cfg.get("prompt_version"), cfg.get("readout")
    if pv not in KNOWN_PROMPT_VERSIONS or ro not in KNOWN_READOUTS:
        raise SystemExit(f"unsupported decision model: prompt_version={pv!r} readout={ro!r} "
                         f"(known: {sorted(KNOWN_PROMPT_VERSIONS)} / {sorted(KNOWN_READOUTS)})")
    return cfg


def format_answer(q: dict, probs: dict) -> dict:
    """Identical to upstream serve.py Decider._format."""
    t = q.get("type")
    top = max(probs, key=probs.get)
    if t in ("noul", "boolean"):
        return {"type": t, "noul": probs["yes"], "probability": probs["yes"], "value": probs["yes"] >= 0.5,
                "confidence": max(probs.values())}
    if t == "score":
        return {"type": "score", "probabilities": probs, "score": int(top),
                "expected": sum(float(k) * v for k, v in probs.items()), "confidence": probs[top]}
    return {"type": "choice", "choice": top, "probabilities": probs, "confidence": probs[top]}


class Worker(threading.Thread):
    """Owns MLX: loads the model and runs every decision on this one thread."""

    def __init__(self, model_dir: str, sym: bool, factory=None):
        super().__init__(daemon=True, name="mlx-decider")
        self.model_dir, self.sym = model_dir, sym
        self._factory = factory  # tests: returns (decider, decision_core module) without MLX
        self.jobs: "queue.Queue[tuple]" = queue.Queue()
        self.ready = threading.Event()
        self.error: str | None = None
        self.busy = False
        self.served = 0

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            if self._factory:
                self.decider, self.dc = self._factory()
            else:
                from mlx_decide import MLXDecider  # sets PROMPT_STYLE, then imports decision_core
                self.decider = MLXDecider(self.model_dir)
                import decision_core
                self.dc = decision_core
            log(f"model loaded in {time.perf_counter() - t0:.1f} s ({self.model_dir}), "
                f"prompt={self.dc.PROMPT_VERSION}, one pass up to {self.dc.MAX_ONE_PASS} options")
            self._decide_all("warm-up", {"w": {"type": "noul", "instructions": "Is this a warm-up?",
                                               "criteria": {"true": "yes", "false": "no"}}})
        except BaseException as e:  # noqa: BLE001 — report and let the server answer 503
            self.error = f"{type(e).__name__}: {e}"
            log(f"LOAD FAILED: {self.error}")
            self.ready.set()
            return
        self.ready.set()
        while True:
            state, questions, box, done = self.jobs.get()
            self.busy = True
            try:
                box["result"] = self._decide_all(state, questions)
            except ValueError as e:
                box["error"] = (422, str(e))
            except Exception as e:  # noqa: BLE001
                box["error"] = (500, f"{type(e).__name__}: {e}")
            finally:
                self.busy = False
                self.served += 1
                done.set()

    def _decide_all(self, state, questions: dict) -> dict:
        """Same batching as upstream serve.py decide_all: every question of the request in one
        pass over a shared prefix (the state is processed once)."""
        if not isinstance(questions, dict) or not questions:
            raise ValueError("'questions' must be a non-empty object")
        items, idx = [], []
        for name, q in questions.items():
            if not isinstance(q, dict):
                raise ValueError(f"{name}: question must be an object")
            opts = self.dc.options_of(q)
            if len(opts) < 2:
                raise ValueError(f"{name}: at least 2 options are required")
            idx.append((name, len(items), opts))
            items.append((q, opts))
            if self.sym:
                items.append((q, list(reversed(opts))))
        if len(items) == 1:
            q, opts = items[0]
            res = [self.decider.dist(state, q, opts)]
        else:
            res = self.decider.dist_many_cached(state, items)
        out = {}
        for name, k, _opts in idx:
            probs, n = res[k]
            if self.sym:
                p2, n2 = res[k + 1]
                probs = {x: 0.5 * (probs[x] + p2[x]) for x in probs}
                n += n2
            out[name] = (format_answer(questions[name], probs), n)
        return out

    def submit(self, state, questions: dict) -> dict:
        box: dict = {}
        done = threading.Event()
        self.jobs.put((state, questions, box, done))
        if not done.wait(JOB_TIMEOUT_S):
            raise TimeoutError(f"decision not finished after {JOB_TIMEOUT_S:.0f} s")
        if "error" in box:
            code, msg = box["error"]
            raise (ValueError(msg) if code == 422 else RuntimeError(msg))
        return box["result"]


def make_handler(worker: Worker, name: str, cfg: dict):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code: int, obj) -> None:
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            ok = worker.ready.is_set() and worker.error is None
            if self.path == "/v1/models":
                # Same readiness/identity probe the engine already uses for sidecar servers
                # (_vlm_probe_ready / _vlm_served_model): the list is empty until the model is loaded.
                return self._send(200, {"object": "list", "data": [
                    {"id": worker.model_dir, "object": "model", "owned_by": "decision"}] if ok else []})
            if self.path not in ("/health", "/v1/health"):
                return self._send(404, {"error": "not found"})
            self._send(200 if ok else 503, {
                "ok": ok, "model": name, "loaded": [name] if ok else [], "device": "mlx",
                "version": VERSION, "kind": "decision", "prompt_version": cfg.get("prompt_version"),
                "readout": cfg.get("readout"), "loading": not worker.ready.is_set(),
                "error": worker.error, "busy": worker.busy, "queued": worker.jobs.qsize(),
                "served": worker.served})

        def do_POST(self):
            if self.path not in ("/v1/systemone", "/v1/evaluate"):
                return self._send(404, {"error": "not found"})
            if not worker.ready.is_set():
                return self._send(503, {"error": "model still loading"})
            if worker.error:
                return self._send(503, {"error": f"model failed to load: {worker.error}"})
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n > MAX_BODY:
                return self._send(413, {"error": f"body larger than {MAX_BODY} bytes"})
            t0 = time.perf_counter()
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
                res = worker.submit(body.get("state", ""), body.get("questions") or {})
            except ValueError as e:
                return self._send(422, {"error": str(e)})
            except TimeoutError as e:
                return self._send(504, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            self._send(200, {"model": name, "answers": {k: v[0] for k, v in res.items()},
                             "usage": {"input_tokens": sum(v[1] for v in res.values()), "output_tokens": 0},
                             "latency_s": time.perf_counter() - t0})
    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--name", default=None, help="model id reported in responses (default: folder name)")
    ap.add_argument("--sym", action="store_true", help="average with the reversed option order (2 passes)")
    a = ap.parse_args()
    model_dir = os.path.abspath(a.model)
    cfg = read_decision_config(model_dir)
    name = a.name or os.path.basename(model_dir.rstrip("/"))
    worker = Worker(model_dir, a.sym)
    worker.start()
    ThreadingHTTPServer.request_queue_size = 256
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(worker, name, cfg))
    log(f"{VERSION} listening on http://{a.host}:{a.port} (model loading in background), name={name}, sym={a.sym}")
    threading.Thread(target=lambda: (worker.ready.wait(),
                                     log("ready" if worker.error is None else "NOT ready (load failed)")),
                     daemon=True).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
