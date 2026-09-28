#!/usr/bin/env python3
"""decision_serve — typed-decision server for OdyssAI-X decision pools (one node).

Serves a *decision model* behind the TypeSafe / Jev wire protocol that OdyssAI-X already
routes: `POST /v1/systemone` (and `/v1/evaluate`), `GET /health`. Two model formats:

- `letter` (MLX): next-token logits over option letters, e.g. Eikos-27B. Readout = the
  vendored, byte-identical `decision_core.py` + `mlx_decide.py` (MIT). The folder ships
  `decision_config.json` + `calib.json`.
- `julia` (PyTorch, MPS or CPU): Supersonic Labs' encoder + marker head, e.g. Julia-1.
  Runtime = the vendored, byte-identical `julia/` package (Apache-2.0), pure PyTorch path
  only. The folder ships `julia_config.json` + `inference-policy.json`; the weights are
  checked against its `weights_sha256` before loading.

Nothing is imported from the model folder (see UPSTREAM.md). An unknown format is refused
at start instead of being answered with a format the weights were not trained on.
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
# Backpressure: beyond this many waiting requests the server answers 503 at once
# (Retry-After) instead of queueing work that would be served long after its client
# gave up — 254 abandoned requests piled up behind a bench judge on 2026-09-28.
MAX_QUEUE = int(os.environ.get("DECISION_MAX_QUEUE", "16"))


class QueueFull(Exception):
    pass


def log(msg: str) -> None:
    print(f"[decision_serve] {msg}", flush=True)


JULIA_PROMPT_VERSION, JULIA_READOUT = "julia-v1", "julia-markers"
CLM_PROMPT_VERSION, CLM_READOUT = "clm-v0.1", "clm-contrastive"


def read_decision_config(model_dir: str) -> dict:
    """The model's decision format. `backend` is 'letter' (decision_config.json) or 'julia'
    (julia_config.json, architecture JuliaDecisionModel, format_version 1)."""
    path = os.path.join(model_dir, "decision_config.json")
    if os.path.exists(path):
        cfg = json.load(open(path))
        pv, ro = cfg.get("prompt_version"), cfg.get("readout")
        if pv not in KNOWN_PROMPT_VERSIONS or ro not in KNOWN_READOUTS:
            raise SystemExit(f"unsupported decision model: prompt_version={pv!r} readout={ro!r} "
                             f"(known: {sorted(KNOWN_PROMPT_VERSIONS)} / {sorted(KNOWN_READOUTS)})")
        return {**cfg, "backend": "letter"}
    jpath = os.path.join(model_dir, "julia_config.json")
    if os.path.exists(jpath):
        jc = json.load(open(jpath))
        if jc.get("architecture") != "JuliaDecisionModel" or jc.get("format_version") != 1:
            raise SystemExit(f"unsupported Julia checkpoint: architecture={jc.get('architecture')!r} "
                             f"format_version={jc.get('format_version')!r}")
        pol_path = os.path.join(model_dir, "inference-policy.json")
        pol = json.load(open(pol_path)) if os.path.exists(pol_path) else {}
        return {"backend": "julia", "prompt_version": JULIA_PROMPT_VERSION, "readout": JULIA_READOUT,
                "max_one_pass": 20, "max_length": pol.get("max_length"),
                "head_length": pol.get("head_length") or 512,
                "weights_sha256": pol.get("weights_sha256")}
    cpath = os.path.join(model_dir, "config.json")
    if os.path.exists(cpath):
        cc = json.load(open(cpath))
        if cc.get("model_type") == "clm":
            ck = (cc.get("checkpoints") or [None])[0]
            if not ck or cc.get("encoder_pooling") != "last-token":
                raise SystemExit(f"unsupported CLM checkpoint: checkpoints={cc.get('checkpoints')!r} "
                                 f"pooling={cc.get('encoder_pooling')!r}")
            return {"backend": "clm", "prompt_version": CLM_PROMPT_VERSION, "readout": CLM_READOUT,
                    "base_model": cc.get("base_model"), "checkpoint": ck,
                    "embedding_dim": cc.get("embedding_dim"), "max_tokens": 2048}
    raise SystemExit(f"not a decision model: no decision_config.json, julia_config.json nor "
                     f"CLM config.json in {model_dir}")


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


PREFILL_STEP = int(os.environ.get("DECISION_PREFILL_STEP", "2048"))
BATCH_KV_GB = float(os.environ.get("DECISION_BATCH_KV_GB", "8"))


class ChunkedPrefill:
    """Wraps the decider's inner model so a long input goes through the cache in blocks of
    `step` tokens, as mlx_lm prefills before generating. The vendored readout feeds the whole
    state in one call; its activations grow ~1.8 GB per 1k tokens on Eikos-27B (measured on
    .42: 13k tokens → 78 GB peak with the 54 GB model), so a 30k-token state outgrew the
    node's RAM and the request thrashed for minutes. Same computation (causal attention,
    recurrent state carried by the cache), bounded memory. Calls without a cache or no
    longer than `step` pass through unchanged."""

    def __init__(self, inner, step: int = PREFILL_STEP):
        self._inner, self._step = inner, max(1, int(step))

    def __call__(self, ids, cache=None, *args, **kwargs):
        n = ids.shape[1]
        if cache is None or n <= self._step:
            return self._inner(ids, cache, *args, **kwargs)
        import mlx.core as mx
        outs = []
        for s in range(0, n, self._step):
            h = self._inner(ids[:, s:s + self._step], cache, *args, **kwargs)
            mx.eval(h, [c.state for c in cache])
            outs.append(h)
            mx.clear_cache()
        return mx.concatenate(outs, axis=1)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class LetterBackend:
    """Letter-logit readout (MLX): every question of a request in one pass over the shared
    prefix, exactly as upstream serve.py decide_all."""
    device = "mlx"

    def __init__(self, decider, dc):
        self.decider, self.dc = decider, dc
        self._kvpt = None
        if hasattr(decider, "inner"):
            decider.inner = ChunkedPrefill(decider.inner)

    def _kv_bytes_per_token(self) -> int:
        """KV bytes one cached token costs across the attention layers (bf16 keys + values).
        Linear-attention layers keep a fixed-size state and are not counted."""
        if self._kvpt is None:
            from mlx_lm.models.cache import KVCache
            lm = self.decider.lm
            args = getattr(lm, "args", None)
            n_kv = sum(isinstance(c, KVCache) for c in lm.make_cache())
            heads, hd = getattr(args, "num_key_value_heads", None), getattr(args, "head_dim", None)
            self._kvpt = n_kv * heads * hd * 2 * 2 if heads and hd else 128 * 1024
        return self._kvpt

    def batch_size(self, n_tok: int) -> int:
        """Questions read together: the readout copies the state's KV cache once per question
        of a batch (Eikos-27B, 33k tokens: ~2 GB a copy), so the batch is sized to
        BATCH_KV_GB instead of upstream's fixed 32."""
        return max(1, min(32, int(BATCH_KV_GB * 1e9 // max(1, n_tok * self._kv_bytes_per_token()))))

    def dist_items(self, state, items):
        if len(items) == 1:
            q, opts = items[0]
            p, n = self.decider.dist(state, q, opts)
            return [(p, n, {})]
        if not hasattr(self.decider, "lm"):  # test doubles
            return [(p, n, {}) for p, n in self.decider.dist_many_cached(state, items)]
        n_tok = len(self.decider._ids(state, *items[0]))
        chunk = self.batch_size(n_tok)
        return [(p, n, {}) for p, n in self.decider.dist_many_cached(state, items, chunk=chunk)]


class JuliaBackend:
    """Julia marker-head readout (PyTorch): one batch for all questions of a request. Rows
    are encoded once with the model's own `sequence()` so a state cut at max_length is
    reported (`truncated`) instead of silently scored on a prefix."""

    def __init__(self, engine, dc, device: str):
        self.engine, self.dc, self.device = engine, dc, device

    def dist_items(self, state, items):
        import math
        from julia.data import sequence
        rows, orders = [], []
        for q, opts in items:
            if not 2 <= len(opts) <= 20:
                raise ValueError(f"Julia reads 2-20 options per question (got {len(opts)})")
            if q.get("type") in ("noul", "boolean"):
                order = ["no", "yes"]                        # Julia reads [false, true]
                descs = dict(opts)
                # No criteria → Julia's literal "false"/"true" (its trained form), not the
                # "yes"/"no" placeholders options_of fills in for the letter readout.
                options = ([descs["no"], descs["yes"]] if q.get("criteria")
                           else ["false", "true"])
                qtype = "noul"
            else:
                order = [lab for lab, _ in opts]
                options = [desc or lab for lab, desc in opts]
                qtype = "score" if q.get("type") == "score" else "choice"
            row = {"state": state, "question": str(q.get("instructions") or ""),
                   "type": qtype, "options": options}
            row["_encoded"] = sequence(self.engine.tokenizer, row, self.engine.collate.max_length,
                                       self.engine.collate.head_length)
            rows.append(row)
            orders.append(order)
        out = []
        for row, order, z in zip(rows, orders, self.engine.logits(rows)):
            m = max(z)
            e = [math.exp(x - m) for x in z]
            t = sum(e)
            probs = {lab: v / t for lab, v in zip(order, e)}
            enc = row["_encoded"]
            out.append((probs, len(enc["ids"]), {"truncated": True} if enc.get("truncated") else {}))
        return out


class CLMBackend:
    """Contrastive LM (CLM): a frozen Qwen3-8B encoder (last-token pooling, L2-normalised,
    last 2,048 tokens kept — the vLLM `--runner pooling` setup the heads were trained on),
    here on MLX, plus the vendored state/action projection heads (torch, CPU). Each
    candidate is scored by the scaled cosine between projected state and projected
    candidate; the question text and candidate texts come from the vendored schema."""
    device = "mlx+cpu"

    def __init__(self, model, tokenizer, head, dc, max_tokens: int = 2048):
        from collections import OrderedDict
        self.model, self.tokenizer, self.head, self.dc = model, tokenizer, head, dc
        self.max_tokens = max_tokens
        self.cache: "OrderedDict[str, object]" = OrderedDict()

    def _embed(self, texts: list[str]):
        import numpy as np
        import mlx.core as mx
        out, spent = [], 0
        for t in texts:
            v = self.cache.get(t)
            if v is None:
                ids = self.tokenizer.encode(t)[-self.max_tokens:]
                spent += len(ids)
                h = self.model.model(mx.array([ids]))[0, -1].astype(mx.float32)
                v = np.array(h)
                v = v / (np.linalg.norm(v) + 1e-12)
                self.cache[t] = v
                while len(self.cache) > 4096:
                    self.cache.popitem(last=False)
            else:
                self.cache.move_to_end(t)
            out.append(v)
        return np.stack(out).astype(np.float32), spent

    def dist_items(self, state, items):
        import math
        from clm.schema import build_pairs
        out = []
        for q, _opts in items:
            (st, keys, texts), = build_pairs(state, {"q": q}).values()
            zs_in, n1 = self._embed([st])
            zc_in, n2 = self._embed(texts)
            zs, zc = self.head.project(zs_in, zc_in)
            logits = [self.head.scale * float(x) for x in (zc @ zs[0])]
            m = max(logits)
            e = [math.exp(v - m) for v in logits]
            t = sum(e)
            p = {k: v / t for k, v in zip(keys, e)}
            if q.get("type") in ("noul", "boolean"):
                p = {"yes": p["true"], "no": p["false"]}
            out.append((p, n1 + n2, {}))
        return out


def _load_clm(model_dir: str, cfg: dict):
    import torch
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    if (major, minor) < (2, 6):
        # heads.py calls torch.load() without weights_only; only >= 2.6 defaults it to True.
        raise RuntimeError(f"torch {torch.__version__} < 2.6: refusing to unpickle the CLM checkpoint")
    enc = os.environ.get("DECISION_CLM_ENCODER") or os.path.join(
        os.path.dirname(os.path.dirname(model_dir.rstrip("/"))), cfg.get("base_model") or "")
    if not os.path.exists(os.path.join(enc, "config.json")):
        raise RuntimeError(f"CLM encoder {cfg.get('base_model')} not found at {enc} "
                           f"(set DECISION_CLM_ENCODER)")
    from mlx_lm import load as mlx_load
    from clm.heads import HeadPair
    import decision_core
    model, tok = mlx_load(enc)
    head = HeadPair("clm", os.path.join(model_dir, cfg["checkpoint"]), device="cpu").ensure()
    return CLMBackend(model, tok, head, decision_core, int(cfg.get("max_tokens") or 2048))


def _load_julia(model_dir: str, cfg: dict):
    import hashlib
    import torch
    want = cfg.get("weights_sha256")
    if want:
        h = hashlib.sha256()
        with open(os.path.join(model_dir, "model.safetensors"), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            raise RuntimeError(f"model.safetensors sha256 {h.hexdigest()} != inference-policy {want}")
    device = os.environ.get("DECISION_DEVICE") or (
        "mps" if torch.backends.mps.is_available() else "cpu")
    from julia.inference import TransformerEngine
    engine = TransformerEngine(model_dir, device=device, max_length=cfg.get("max_length"),
                               head_length=int(cfg.get("head_length") or 512))
    import decision_core
    return JuliaBackend(engine, decision_core, device)


class Worker(threading.Thread):
    """Owns the model: loads it and runs every decision on this one thread (MLX keeps a
    stream per thread; PyTorch MPS likes a single owner too)."""

    def __init__(self, model_dir: str, sym: bool, factory=None, cfg: dict | None = None):
        super().__init__(daemon=True, name="mlx-decider")
        self.model_dir, self.sym = model_dir, sym
        self.cfg = dict(cfg or {"backend": "letter"})
        # tests: returns (decider, decision_core) for a letter backend, or a backend object
        self._factory = factory
        self.jobs: "queue.Queue[tuple]" = queue.Queue()
        self.ready = threading.Event()
        self.error: str | None = None
        self.busy = False
        self.served = 0
        self.skipped = 0
        self.wired: dict = {}

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            if self._factory:
                made = self._factory()
                self.backend = made if hasattr(made, "dist_items") else LetterBackend(*made)
            elif self.cfg.get("backend") == "julia":
                self.backend = _load_julia(self.model_dir, self.cfg)
            elif self.cfg.get("backend") == "clm":
                self.backend = _load_clm(self.model_dir, self.cfg)
                self._pin_weights(self.backend.model)
            else:
                from mlx_decide import MLXDecider  # sets PROMPT_STYLE, then imports decision_core
                import decision_core
                self.backend = LetterBackend(MLXDecider(self.model_dir), decision_core)
                self.decider = self.backend.decider
                self._pin_weights(self.decider.model)
            self.dc = self.backend.dc
            log(f"model loaded in {time.perf_counter() - t0:.1f} s ({self.model_dir}), "
                f"backend={self.cfg.get('backend')} device={getattr(self.backend, 'device', '?')}")
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
            alive = box.get("alive")
            if box.get("cancelled") or (alive is not None and not alive()):
                # Its client is gone (disconnected or timed out): never compute it.
                box["cancelled"] = True
                self.skipped += 1
                done.set()
                continue
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

    def _pin_weights(self, model) -> None:
        """Wire the weights for the life of the process, as mlx_lm does around every
        generation (`wired_limit`). The readout calls the model directly, so without
        this macOS evicts the Metal buffers of a model close to the node's working
        set: seen 2026-09-27 on ultra-96b with Eikos-27B bf16 (55 GB) — GPU at 97 %
        while its resident memory swung 30 → 6.6 GB and each request took minutes."""
        import mlx.core as mx
        from mlx.utils import tree_reduce
        if not mx.metal.is_available():
            return
        rec = mx.device_info()["max_recommended_working_set_size"]
        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0)
        mx.set_wired_limit(rec)
        self.wired = {"model_gb": round(model_bytes / 1e9, 1), "wired_limit_gb": round(rec / 1e9, 1)}
        log(f"weights wired: model {self.wired['model_gb']} GB, wired limit {self.wired['wired_limit_gb']} GB"
            + (" — close to the limit, expect pressure" if model_bytes > 0.9 * rec else ""))

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
        res = self.backend.dist_items(state, items)
        out = {}
        for name, k, _opts in idx:
            probs, n, extra = res[k]
            if self.sym:
                p2, n2, _ = res[k + 1]
                probs = {x: 0.5 * (probs[x] + p2[x]) for x in probs}
                n += n2
            out[name] = ({**format_answer(questions[name], probs), **extra}, n)
        return out

    def submit(self, state, questions: dict, alive=None) -> dict:
        """Queue a job and wait for it. `alive()` (the HTTP handler's view of its client)
        is polled while waiting: a job whose client left is marked cancelled and dropped
        by the worker instead of being computed for nobody."""
        if self.jobs.qsize() >= MAX_QUEUE:
            raise QueueFull(f"{self.jobs.qsize()} decisions already waiting (max {MAX_QUEUE})")
        box: dict = {"alive": alive}
        done = threading.Event()
        self.jobs.put((state, questions, box, done))
        deadline = time.monotonic() + JOB_TIMEOUT_S
        while not done.wait(0.5):
            if alive is not None and not alive():
                box["cancelled"] = True
                raise ConnectionAbortedError("client disconnected")
            if time.monotonic() > deadline:
                box["cancelled"] = True
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

        def _client_alive(self) -> bool:
            """False once the client closed its connection (readable + zero-byte peek)."""
            import select
            import socket
            try:
                r, _, _ = select.select([self.connection], [], [], 0)
                if not r:
                    return True
                return self.connection.recv(1, socket.MSG_PEEK) != b""
            except OSError:
                return False

        def _send(self, code: int, obj, retry_after: int | None = None) -> None:
            b = json.dumps(obj).encode()
            self.send_response(code)
            if retry_after:
                self.send_header("Retry-After", str(retry_after))
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
                "ok": ok, "model": name, "loaded": [name] if ok else [],
                "device": getattr(getattr(worker, "backend", None), "device", None),
                "backend": worker.cfg.get("backend"),
                "version": VERSION, "kind": "decision", "prompt_version": cfg.get("prompt_version"),
                "readout": cfg.get("readout"), "loading": not worker.ready.is_set(),
                "error": worker.error, "busy": worker.busy, "queued": worker.jobs.qsize(),
                "served": worker.served, "skipped": worker.skipped, "max_queue": MAX_QUEUE,
                "wired": worker.wired})

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
                res = worker.submit(body.get("state", ""), body.get("questions") or {},
                                    alive=self._client_alive)
            except QueueFull as e:
                return self._send(503, {"error": f"busy: {e}"}, retry_after=5)
            except ConnectionAbortedError:
                return  # nobody to answer
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
    worker = Worker(model_dir, a.sym, cfg=cfg)
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
