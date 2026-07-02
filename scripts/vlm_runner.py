#!/usr/bin/env python3
"""vlm_runner.py — distributed (tensor-parallel) mlx-vlm runner for OdyssAI-X.

Slim sibling of runner.py for VISION models served through mlx-vlm. Same
process contract so the engine's RunnerProc machinery drives it unchanged:

  stdin  (every rank, one JSON per line):
      {"cmd":"gen","id":...,"messages":[...],"max_tokens":...,
       "temperature":...,"top_p":...,"repetition_penalty":...}
      {"cmd":"cancel","id":...}     intercepted by the reader thread
      {"cmd":"keepalive","id":...}  tiny all_sum on all ranks, rank 0 acks
      {"cmd":"stop"}                graceful exit
  stdout (rank 0 ONLY):
      {"event":"ready","rank":0,"size":N,"load_s":...,"is_vlm":true}
      {"event":"token","id":...,"text":"..."}
      {"event":"done","id":...,"ntoks":...,"prompt_tokens":...,
       "cached_tokens":0,"elapsed_s":...,"tps":...[,"finish_reason":...]}
      {"event":"bye"}
  stderr (all ranks): logs, hostname-prefixed; load-phase lines match the
      engine's _PHASE_MARKERS needles where applicable.

Coordination model (identical to runner.py multi-rank): the ENGINE fans the
same JSONL line out to every rank's stdin; every rank recomputes the request
deterministically (template, image decode, vision tower, forward); collectives
inside the sharded language model stay aligned because inputs are identical;
emit() gates stdout to rank 0. Images arrive INSIDE `messages` as data URIs or
local paths — ranks never fetch over the network (the engine resolves URLs).

v0 hard cuts (each is a deliberate divergence-source removal, not an omission):
no session/prefix cache, no prewarm, no radix, no disk cache, no speculative,
no batching, no kv-q8. Fresh cache per request, single-stream.

Config via env (no argparse, same contract as runner.py):
  RUNNER_MODEL       model dir (local path)
  RUNNER_BACKEND     ring | jaccl        (default ring — frozen: ring first)
  MLX_WORLD_SIZE     "1" => no distributed init (single-node bypass)
  RUNNER_EMIT_BATCH  tokens coalesced per stdout token event (default 10)
  RUNNER_MAX_IMAGE_MB  per-image decoded cap, safety valve (default 64)
"""

import base64
import binascii
import json
import os
import queue
import re
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx

# ── stdout/stderr contract ────────────────────────────────────────────────────

def emit(rank: int, obj: dict) -> None:
    if rank == 0:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def log(msg: str) -> None:
    sys.stderr.write(f"[{socket.gethostname()}] {msg}\n")
    sys.stderr.flush()


def _active_gb() -> float:
    try:
        if hasattr(mx, "get_active_memory"):
            return mx.get_active_memory() / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def free_metal(reason: str = "") -> None:
    before = _active_gb()
    try:
        mx.clear_cache()
    except Exception as e:
        log(f"free_metal: clear_cache failed ({e})")
        return
    log(f"free_metal{f' ({reason})' if reason else ''}: "
        f"active {before:.1f} GB -> {_active_gb():.1f} GB")


import atexit as _atexit  # noqa: E402  (after free_metal exists)

_atexit.register(lambda: free_metal("atexit"))

# ── hard-cancel registry (populated by reader thread, drained between tokens) ─

_cancelled_ids: set[str] = set()
_cancelled_lock = threading.Lock()


def _mark_cancelled(req_id) -> None:
    if req_id:
        with _cancelled_lock:
            _cancelled_ids.add(req_id)


def _is_cancelled(req_id: str) -> bool:
    with _cancelled_lock:
        return req_id in _cancelled_ids


def _clear_cancelled(req_id: str) -> None:
    with _cancelled_lock:
        _cancelled_ids.discard(req_id)


# ── model load: in-repo replication of mlx_vlm.utils.sharded_load @ecc457b ───
# Four deliberate fixes vs upstream:
#   1. calls a shard function directly — the top-level minimax_m3_vl.Model has
#      no shard() delegator, so upstream's hasattr(model, "shard") gate raises
#      ValueError;
#   2. no bare print() to stdout (upstream prints "Materializing", which would
#      corrupt rank 0's JSONL event stream);
#   3. materializes the replicated vision tower/projector at load — upstream
#      only evals language_model params, so vision weights would otherwise
#      materialize lazily on all ranks during the first image request,
#      mid-collective (indistinguishable from a hang);
#   4. REPLICATES the MSA indexer instead of sharding it (see below) — the
#      upstream LanguageModel.shard() produces deterministic garbage under TP.


def _shard_lm_replicated_indexer(lm, group):
    """Tensor-shard the MiniMax-M3 language model, REPLICATING the MSA indexer.

    Upstream LanguageModel.shard() shards index_q_proj and divides index_heads
    by the group size. But the block selector aggregates scores ACROSS index
    heads (mx.max(block_scores, axis=1) in _build_sparse_causal_mask_compiled),
    so each rank selects DIFFERENT key blocks from its local head subset ->
    o_proj all_sum mixes incoherent attentions -> deterministic garbage
    (identical on all ranks, wrong vs baseline — measured live 2026-07-02,
    Gate-1 A/B: upstream shard sha 9ec8d76e vs baseline d73be672).
    Keeping the tiny indexer (~190M params over 60 layers) replicated makes
    block selection bit-identical to single-node on every rank. Everything
    else mirrors upstream shard(): q/k/v all-to-sharded, o_proj
    sharded-to-all, head counts divided, MoE switch_mlp shard_inplace +
    sharding_group for the block all_sum.
    """
    from mlx_vlm.models.minimax_m3_vl import language as _lang
    shard_linear = _lang.shard_linear
    shard_inplace = _lang.shard_inplace
    n = group.size()
    for layer in lm.layers:
        sa = layer.self_attn
        sa.q_proj = shard_linear(sa.q_proj, "all-to-sharded", group=group)
        sa.k_proj = shard_linear(sa.k_proj, "all-to-sharded", group=group)
        sa.v_proj = shard_linear(sa.v_proj, "all-to-sharded", group=group)
        sa.o_proj = shard_linear(sa.o_proj, "sharded-to-all", group=group)
        sa.num_attention_heads //= n
        sa.num_key_value_heads //= n
        # index_q_proj / index_k_proj / index_heads deliberately NOT sharded.
        if not layer.is_moe_layer:
            continue
        moe = layer.block_sparse_moe
        if moe.pack_shared_expert:
            # SECOND upstream TP bug (micro-proven 2026-07-02): the packed
            # variant fuses [gate|up] on the out dim and the forward splits at
            # the midpoint (mx.split(gate_up, 2, axis=-1)). Contiguous
            # all-to-sharded slicing hands rank0 all-gate rows and rank1
            # all-up rows -> activation(gate)*up is scrambled -> deterministic
            # garbage identical on all ranks. Slice each half separately so
            # the local midpoint split stays correctly paired (exact to
            # quantization tolerance: relmax 0.0033 on layer-1 forward).
            _shard_fused_gate_up_inplace(moe.switch_mlp.gate_up_proj, group)
        else:
            shard_inplace(moe.switch_mlp.gate_proj, "all-to-sharded",
                          group=group)
            shard_inplace(moe.switch_mlp.up_proj, "all-to-sharded",
                          group=group)
        shard_inplace(moe.switch_mlp.down_proj, "sharded-to-all", group=group)
        moe.sharding_group = group


def _shard_fused_gate_up_inplace(sl, group):
    """Per-half out-dim slicing for a fused [gate|up] SwitchLinear (quantized
    or not): rank r keeps gate[r*I/n:(r+1)*I/n] ++ up[same range]. The out dim
    is never the packed axis, so plain row slicing of weight/scales/biases is
    exact for any quant bit-width."""
    n, r = group.size(), group.rank()

    def slice_fused(t):
        out2 = t.shape[1]
        half_i = out2 // 2
        h = half_i // n
        return mx.concatenate(
            [t[:, r * h:(r + 1) * h],
             t[:, half_i + r * h: half_i + (r + 1) * h]], axis=1)

    for name in ("weight", "scales", "biases"):
        if hasattr(sl, name):
            setattr(sl, name, slice_fused(getattr(sl, name)))


def sharded_vlm_load(path: str, group):
    from mlx_vlm.utils import (get_model_path, load_image_processor,
                               load_model, load_processor)
    shard_mode = os.environ.get("RUNNER_SHARD_MODE", "replicated-indexer")
    model_path = get_model_path(path)
    log(f"loading model (lazy=True) from {model_path}")
    model = load_model(model_path, lazy=True, strict=False)
    config = model.config.to_dict()
    processor = load_processor(
        model_path, True, eos_token_ids=config.get("eos_token_id", None))
    image_processor = load_image_processor(model_path)
    if image_processor is not None:
        processor.image_processor = image_processor
    if group is not None and group.size() > 1:
        log(f"rank {group.rank()} sharding language_model "
            f"(tensor, {shard_mode})")
        if shard_mode == "upstream":
            model.language_model.shard(group)
        else:
            _shard_lm_replicated_indexer(model.language_model, group)
    mx.eval(model.language_model.parameters())
    log("materializing vision tower + projector (replicated)")
    mx.eval(model.parameters())
    model.eval()
    return model, processor, config


# ── image transport: pull image sources out of OpenAI-shape messages ─────────
# The engine forwards `messages` verbatim. Multimodal content is a list of
# parts; we keep text parts in the message (joined) and collect image sources
# in encounter order. Supported sources: data URIs (decoded to a temp file so
# every rank feeds prepare_inputs identical bytes) and local paths. http(s) is
# REFUSED — resolving URLs is the engine's job, ranks must never fetch.

_DATA_URI_RE = re.compile(r"^data:image/[\w.+-]+;base64,", re.IGNORECASE)


class ImageExtractionError(Exception):
    pass


def _extract_images(messages: list[dict], req_id: str,
                    max_image_mb: int) -> tuple[list[dict], list[str], list[str]]:
    """Returns (template_messages, image_paths, temp_paths_to_cleanup)."""
    out_msgs: list[dict] = []
    images: list[str] = []
    temps: list[str] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out_msgs.append(m)
            continue
        texts: list[str] = []
        for p in content:
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype == "text":
                texts.append(p.get("text") or "")
                continue
            if ptype not in ("image_url", "input_image", "image"):
                continue
            url = p.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            url = url or p.get("url") or p.get("image")
            if not isinstance(url, str) or not url:
                raise ImageExtractionError("image part with no usable url")
            if _DATA_URI_RE.match(url):
                b64 = url.split(",", 1)[1]
                try:
                    raw = base64.b64decode(b64, validate=True)
                except (binascii.Error, ValueError) as e:
                    raise ImageExtractionError(f"bad base64 image: {e}")
                if len(raw) > max_image_mb * 1024 * 1024:
                    raise ImageExtractionError(
                        f"image {len(raw)//(1024*1024)}MB > cap {max_image_mb}MB")
                tmp = f"/tmp/vlmr_{req_id}_{len(images)}.img"
                with open(tmp, "wb") as f:
                    f.write(raw)
                temps.append(tmp)
                images.append(tmp)
            elif url.startswith(("http://", "https://")):
                raise ImageExtractionError(
                    "http(s) image URL reached the runner — the engine must "
                    "resolve URLs; ranks never fetch")
            else:
                if not Path(url).exists():
                    raise ImageExtractionError(f"image path not found: {url}")
                images.append(url)
        nm = dict(m)
        nm["content"] = "\n".join(t for t in texts if t)
        out_msgs.append(nm)
    return out_msgs, images, temps


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    repo = os.environ.get("RUNNER_MODEL", "")
    backend = os.environ.get("RUNNER_BACKEND", "ring").strip().lower()
    world_size_env = int(os.environ.get("MLX_WORLD_SIZE", "0") or "0")
    emit_batch_n = int(os.environ.get("RUNNER_EMIT_BATCH", "10"))
    max_image_mb = int(os.environ.get("RUNNER_MAX_IMAGE_MB", "64"))
    if not repo:
        log("RUNNER_MODEL is required")
        sys.exit(2)

    stop_requested = {"flag": False}

    def handle_sig(signum, _frame):
        log(f"signal {signum} received, marking stop")
        stop_requested["flag"] = True

    signal.signal(signal.SIGTERM, handle_sig)
    signal.signal(signal.SIGINT, handle_sig)

    t0 = time.time()
    if world_size_env == 1:
        group, rank, size = None, 0, 1
        log(f"init single-node mode (no distributed), model={repo}, vlm")
    else:
        log(f"init {backend} backend, model={repo}, vlm")
        group = mx.distributed.init(backend=backend, strict=True)
        rank, size = group.rank(), group.size()
        log(f"rank {rank}/{size} group ready in {time.time()-t0:.2f}s")

    t1 = time.time()
    model, processor, config = sharded_vlm_load(repo, group)

    if size > 1:
        log(f"rank {rank} barrier before ready")
        mx.eval(mx.distributed.all_sum(mx.array([1.0]), group=group))

    load_s = time.time() - t1
    log(f"rank {rank} model loaded in {load_s:.1f}s "
        f"(active {_active_gb():.1f} GB)")
    emit(rank, {"event": "ready", "rank": rank, "size": size,
                "load_s": load_s, "is_vlm": True})

    from mlx_vlm.generate import stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    # Reader thread: parse every stdin line, intercept cancel immediately
    # (between stream_generate yields), queue everything else in order.
    in_q: queue.Queue = queue.Queue()
    EOF = object()

    def reader() -> None:
        while not stop_requested["flag"]:
            line = sys.stdin.readline()
            if not line:
                in_q.put(EOF)
                return
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                log(f"bad json: {e}")
                continue
            if msg.get("cmd") == "cancel":
                _mark_cancelled(msg.get("id"))
                continue
            in_q.put(msg)

    threading.Thread(target=reader, daemon=True,
                     name=f"vlm-runner-stdin-r{rank}").start()

    while not stop_requested["flag"]:
        try:
            req = in_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if req is EOF:
            log("stdin closed, exiting loop")
            break

        cmd = req.get("cmd")
        if cmd == "stop":
            log("stop cmd received")
            break
        if cmd == "keepalive":
            if group is not None and size > 1:
                try:
                    mx.eval(mx.distributed.all_sum(mx.array([1.0]), group=group))
                except Exception as e:
                    log(f"keepalive all_sum error: {e}")
                    emit(rank, {"event": "keepalive_ok",
                                "id": req.get("id", ""), "ok": False,
                                "error": str(e)})
                    continue
            emit(rank, {"event": "keepalive_ok", "id": req.get("id", ""),
                        "ok": True})
            continue
        if cmd == "session_clear":
            # v0 has no session cache — ack silently so the engine's generic
            # plumbing doesn't error against a VL pool.
            log("session_clear: no-op (vlm v0 has no session cache)")
            continue
        if cmd == "prewarm":
            emit(rank, {"event": "prewarm", "id": req.get("id", ""),
                        "ok": False,
                        "result": {"note": "vlm v0: no prefix cache"}})
            continue
        if cmd != "gen":
            log(f"unknown cmd: {cmd}")
            continue

        req_id = req.get("id", "")
        messages = req.get("messages")
        prompt = req.get("prompt")
        max_tokens = int(req.get("max_tokens", 512))
        if messages is None:
            if not prompt:
                log("gen with neither messages nor prompt — skipping")
                continue
            messages = [{"role": "user", "content": prompt}]

        temps: list[str] = []
        try:
            tmpl_messages, images, temps = _extract_images(
                messages, req_id or "noid", max_image_mb)

            chat_kwargs: dict = {"num_images": len(images)}
            enable_thinking = req.get("enable_thinking", None)
            if enable_thinking is not None:
                chat_kwargs["enable_thinking"] = enable_thinking
                if "minimax-m3" in repo.lower():
                    chat_kwargs["thinking_mode"] = (
                        "enabled" if enable_thinking else "disabled")
            reasoning_effort = req.get("reasoning_effort", None)
            if reasoning_effort:
                chat_kwargs["reasoning_effort"] = reasoning_effort
            formatted = apply_chat_template(
                processor, config, tmpl_messages, **chat_kwargs)

            gen_kwargs: dict = {"max_tokens": max_tokens}
            for k in ("temperature", "top_p", "top_k", "min_p",
                      "repetition_penalty", "seed"):
                if req.get(k) is not None:
                    gen_kwargs[k] = req[k]

            if rank == 0:
                log(f"req {req_id}: images={len(images)} "
                    f"max_tokens={max_tokens} "
                    f"sampling={ {k: v for k, v in gen_kwargs.items() if k != 'max_tokens'} }")

            ntoks = 0
            t_gen = time.time()
            buf: list[str] = []
            finish_reason = None
            cancelled_mid_gen = False
            last = None
            for res in stream_generate(model, processor, formatted,
                                       image=images or None, **gen_kwargs):
                buf.append(res.text)
                ntoks += 1
                last = res
                if len(buf) >= emit_batch_n:
                    emit(rank, {"event": "token", "id": req_id,
                                "text": "".join(buf)})
                    buf.clear()
                if _is_cancelled(req_id):
                    cancelled_mid_gen = True
                    break
                if stop_requested["flag"]:
                    break
            if buf:
                emit(rank, {"event": "token", "id": req_id,
                            "text": "".join(buf)})
            elapsed = time.time() - t_gen
            tps = ntoks / elapsed if elapsed > 0 else 0.0
            done_event = {
                "event": "done",
                "id": req_id,
                "ntoks": ntoks,
                "prompt_tokens": int(getattr(last, "prompt_tokens", 0) or 0),
                "cached_tokens": 0,
                "elapsed_s": elapsed,
                "tps": tps,
            }
            if cancelled_mid_gen:
                done_event["finish_reason"] = "cancelled"
            elif getattr(last, "finish_reason", None):
                done_event["finish_reason"] = last.finish_reason
            emit(rank, done_event)
            log(f"req {req_id}: {ntoks} toks in {elapsed:.1f}s = {tps:.2f} tok/s"
                + (" · CANCELLED" if cancelled_mid_gen else ""))
        except ImageExtractionError as e:
            log(f"req {req_id}: image extraction failed: {e}")
            emit(rank, {"event": "done", "id": req_id, "ntoks": 0,
                        "prompt_tokens": 0, "cached_tokens": 0,
                        "elapsed_s": 0.0, "tps": 0.0,
                        "finish_reason": "error", "error": str(e)})
        finally:
            _clear_cancelled(req_id)
            for t in temps:
                try:
                    os.unlink(t)
                except OSError:
                    pass

    # Teardown: drop refs, clear Metal cache, let destructors close transports.
    try:
        model = None
        processor = None
        free_metal("shutdown")
    except Exception as e:
        log(f"shutdown cleanup error: {e}")
    emit(rank, {"event": "bye"})
    log("exiting cleanly")


if __name__ == "__main__":
    main()
