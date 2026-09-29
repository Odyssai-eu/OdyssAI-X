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
  RUNNER_BACKEND     ring | jaccl        (default ring; the engine sends jaccl)
  RUNNER_SHARD_MODE  "" (auto: the model type's own split) | tensor | pipeline
                     | upstream (model.language_model.shard(group))
  RUNNER_LAYER_BOUNDS / RUNNER_RAM_WEIGHTS  pipeline split, same contract as
                     runner.py (manual bounds CSV > per-rank weights > even)
  MLX_WORLD_SIZE     "1" => no distributed init (single-node bypass)
  RUNNER_EMIT_BATCH  tokens coalesced per stdout token event (default 10)
  RUNNER_MAX_IMAGE_MB  per-image / per-audio decoded cap, safety valve (default 64)
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
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

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


class _AllSumAfter(nn.Module):
    """Sum a tensor-sharded block's partial output across ranks.

    mlx-vlm's MiMo MoE.__call__ has no sharding_group hook (mlx-lm's MoE
    classes all_sum inside the block). The routed experts are sharded on the
    intermediate dim, so each rank returns a partial sum over its slice; the
    weighted sum over the top-k experts is linear, so one all_sum of the block
    output restores the full result."""

    def __init__(self, inner, group):
        super().__init__()
        self.inner = inner
        self.sharding_group = group

    def __call__(self, x):
        return mx.distributed.all_sum(self.inner(x), group=self.sharding_group)


def _shard_lm_mimo_tensor(lm, group):
    """Tensor-shard the MiMo-V2 language model (mlx-vlm mimo_v2, MiMo-V2.6).

    mlx-vlm's MiMo LanguageModel has no shard(). This follows the shard() of
    the mlx-lm mimo_v2_flash port (ml-explore/mlx-lm PR #1219): q/k/v
    all-to-sharded, o_proj sharded-to-all, head counts divided, the per-head
    attention sinks sliced to this rank's heads, dense MLP gate/up
    all-to-sharded + down sharded-to-all, routed experts sharded in place on
    the intermediate dim with one all_sum per MoE block (_AllSumAfter). The
    router stays replicated, so every rank picks the same experts.

    Rank r keeps q heads [r*H/n, (r+1)*H/n) and kv heads [r*K/n, (r+1)*K/n):
    with contiguous GQA grouping (q head h reads kv head h // (H/K)) the two
    slices pair up exactly. MXFP4 experts: the down_proj input dim (2048)
    splits into whole 32-value scale groups for n <= 4 (2048/32/4 = 16)."""
    n, r = group.size(), group.rank()
    for layer in lm.layers:
        sa = layer.self_attn
        if sa.n_heads % n or sa.n_kv_heads % n:
            raise ValueError(
                f"mimo tensor split: {sa.n_heads} q / {sa.n_kv_heads} kv heads "
                f"do not divide by {n} ranks")
        sa.q_proj = shard_linear(sa.q_proj, "all-to-sharded", group=group)
        sa.k_proj = shard_linear(sa.k_proj, "all-to-sharded", group=group)
        sa.v_proj = shard_linear(sa.v_proj, "all-to-sharded", group=group)
        sa.o_proj = shard_linear(sa.o_proj, "sharded-to-all", group=group)
        sa.n_heads //= n
        sa.n_kv_heads //= n
        if sa.attention_sink_bias is not None:
            h = sa.n_heads
            sa.attention_sink_bias = sa.attention_sink_bias[r * h:(r + 1) * h]
        mlp = layer.mlp
        if hasattr(mlp, "switch_mlp"):
            shard_inplace(mlp.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(mlp.switch_mlp.up_proj, "all-to-sharded", group=group)
            shard_inplace(mlp.switch_mlp.down_proj, "sharded-to-all", group=group)
            shared = getattr(mlp, "shared_experts", None)
            if shared is not None:
                # Partial like the routed experts: covered by the same all_sum.
                shard_inplace(shared.gate_proj, "all-to-sharded", group=group)
                shard_inplace(shared.up_proj, "all-to-sharded", group=group)
                shard_inplace(shared.down_proj, "sharded-to-all", group=group)
            layer.mlp = _AllSumAfter(mlp, group)
        else:
            mlp.gate_proj = shard_linear(mlp.gate_proj, "all-to-sharded", group=group)
            mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
            mlp.down_proj = shard_linear(mlp.down_proj, "sharded-to-all", group=group)


# Distributed split per model type. The first mode listed is the type's
# default (RUNNER_SHARD_MODE unset); api.py's VLM_DIST_SUPPORTED mirrors this
# table — keep them in sync.
#   tensor   : the type's in-repo sharder below (forward all_sums over JACCL).
#   pipeline : layer split through the exo machinery (auto_parallel) — needs
#              neither head divisibility nor forward collectives. qwen3.5-MoE
#              only knows pipeline: a HYBRID (self_attn + GatedDeltaNet) with
#              num_kv_heads=2, which won't divide 3/4 nodes.
_TENSOR_SHARDERS = {
    "minimax_m3_vl": _shard_lm_replicated_indexer,
    "mimo_v2": _shard_lm_mimo_tensor,
}
_DIST_MODES = {
    "minimax_m3_vl": ("tensor",),
    "qwen3_5_moe": ("pipeline",),
    "mimo_v2": ("tensor", "pipeline"),
}


def _pipeline_bounds(num_layers: int, size: int) -> tuple[list[int], str]:
    """Cumulative layer bounds, same precedence as runner.shard_pipeline:
    RUNNER_LAYER_BOUNDS (manual) > RUNNER_RAM_WEIGHTS (per-rank bytes, sent by
    the engine from wired limits) > even split. An even split on .29 (460 GiB
    wired) + a 256 GB node (200 GiB) OOMs the small node."""
    from auto_parallel import compute_proportional_bounds
    bounds_env = os.environ.get("RUNNER_LAYER_BOUNDS", "")
    if bounds_env:
        b = [int(x) for x in bounds_env.split(",")]
        if len(b) != size + 1 or b[0] != 0 or b[-1] != num_layers:
            raise ValueError(
                f"RUNNER_LAYER_BOUNDS={bounds_env!r} invalid for size={size}, "
                f"num_layers={num_layers}")
        return b, "manual"
    weights_env = os.environ.get("RUNNER_RAM_WEIGHTS", "")
    if weights_env:
        try:
            w = [int(x) for x in weights_env.split(",")]
        except ValueError:
            w = []
        if len(w) == size and all(x >= 0 for x in w) and sum(w) > 0:
            return compute_proportional_bounds(num_layers, w), f"proportional({w})"
        log(f"RUNNER_RAM_WEIGHTS={weights_env!r} invalid for size={size}; even split")
    per = num_layers // size
    return [0] + [per * i for i in range(1, size)] + [num_layers], "even"


def _shard_lm_pipeline(model, group, num_layers: int):
    """Pipeline-shard a VLM's language model across ranks (layer split).

    Slices inner.layers to this rank's [start,end), wraps the ends with
    Pipeline{First,Last}Layer (recv/send), materialises only local layers.
    auto_parallel recomputes fa_idx/ssm_idx for the stock mlx-lm hybrid class
    but NOT for the mlx-vlm Qwen3_5MoeModel — do it here (shard-LOCAL indices,
    else cache[fa_idx] points at the wrong local layer's cache). MiMo's
    swa_idx/ga_idx are recomputed inside auto_parallel for both the mlx-lm
    and the mlx-vlm class."""
    from auto_parallel import pipeline_auto_parallel
    from exo_stubs import PipelineShardMetadata
    rank, size = group.rank(), group.size()
    b, split_source = _pipeline_bounds(num_layers, size)
    start, end = b[rank], b[rank + 1]
    log(f"rank {rank} pipeline shard layers [{start}, {end}) of {num_layers} "
        f"split={split_source}")
    # Pass the LanguageModel WRAPPER: get_inner_model() resolves .model ->
    # Qwen3_5MoeModel (the one carrying .layers) and mutates it in place.
    meta = PipelineShardMetadata(device_rank=rank, world_size=size,
                                 start_layer=start, end_layer=end)
    gen = pipeline_auto_parallel(model.language_model, group, meta)
    while True:
        try:
            next(gen)
        except StopIteration:
            break
    # LOCAL hybrid indices (auto_parallel's isinstance misses the mlx-vlm class).
    inner = model.language_model.model            # sliced+wrapped in place
    if not hasattr(inner, "fa_idx"):              # not a qwen3.5 hybrid
        return
    fa = [i for i, l in enumerate(inner.layers)
          if not getattr(l, "is_linear", True)]
    ssm = [i for i, l in enumerate(inner.layers)
           if getattr(l, "is_linear", False)]
    inner.fa_idx = fa[0] if fa else 0
    inner.ssm_idx = ssm[0] if ssm else 0
    log(f"rank {rank} local fa_idx={inner.fa_idx} ssm_idx={inner.ssm_idx} "
        f"({len(inner.layers)} local layers)")


def sharded_vlm_load(path: str, group):
    from mlx_vlm.utils import (get_model_path, load_image_processor,
                               load_model, load_processor)
    shard_mode = os.environ.get("RUNNER_SHARD_MODE", "").strip().lower()
    if shard_mode == "replicated-indexer":     # pre-2026-09-29 name
        shard_mode = "tensor"
    model_path = get_model_path(path)
    log(f"loading model (lazy=True) from {model_path}")
    model = load_model(model_path, lazy=True, strict=False)
    config = model.config.to_dict()
    processor = load_processor(
        model_path, True, eos_token_ids=config.get("eos_token_id", None))
    image_processor = load_image_processor(model_path)
    if image_processor is not None:
        processor.image_processor = image_processor
    model_type = config.get("model_type") or ""
    if group is not None and group.size() > 1:
        modes = _DIST_MODES.get(model_type, ())
        mode = shard_mode or (modes[0] if modes else "")
        # #78 — each sharder is written for one layout; applied to another
        # type it crashes or corrupts silently. The engine refuses such loads
        # (VLM_DIST_SUPPORTED in api.py); this is the backstop for a forced
        # load. "upstream" is the explicit escape hatch to the model's own
        # shard().
        if mode != "upstream" and mode not in modes:
            raise RuntimeError(
                f"vlm_runner: model_type {model_type!r} has no {mode or 'distributed'} "
                f"split (known: {_DIST_MODES}); serve it on one node or on a "
                f"replica cluster")
        log(f"rank {group.rank()} sharding language_model ({mode}, {model_type})")
        if mode == "pipeline":
            n_layers = len(model.language_model.model.layers)
            _shard_lm_pipeline(model, group, n_layers)
        elif mode == "upstream":
            model.language_model.shard(group)
        else:
            _TENSOR_SHARDERS[model_type](model.language_model, group)
    mx.eval(model.language_model.parameters())
    log("materializing vision tower + projector (replicated)")
    mx.eval(model.parameters())
    model.eval()
    return model, processor, config


# ── media transport: pull image/audio sources out of OpenAI-shape messages ───
# The engine forwards `messages` verbatim. Multimodal content is a list of
# parts; we keep text parts in the message (joined) and collect image and
# audio sources in encounter order. Supported sources: data URIs / bare base64
# (decoded to a temp file so every rank feeds prepare_inputs identical bytes)
# and local paths. http(s) is REFUSED — resolving URLs is the engine's job,
# ranks must never fetch. Audio follows mlx_vlm.server's OpenAI shape:
# {"type": "input_audio", "input_audio": {"data": <base64|data:audio/..>,
# "format": "wav"}}. Any other non-text part (video, ...) is refused: a
# silently dropped part makes the model answer blind.

_DATA_URI_RE = re.compile(r"^data:image/[\w.+-]+;base64,", re.IGNORECASE)
_AUDIO_DATA_URI_RE = re.compile(r"^data:audio/([\w.+-]+);base64,", re.IGNORECASE)
_TEXTLIKE_PARTS = ("text", "input_text")


class ImageExtractionError(Exception):
    """A media part the runner cannot hand to the model (image or audio)."""


def _decode_audio_part(part: dict, req_id: str, idx: int, max_mb: int) -> str:
    ia = part.get("input_audio")
    if not isinstance(ia, dict) or not isinstance(ia.get("data"), str) or not ia["data"]:
        raise ImageExtractionError("input_audio part with no data")
    data = ia["data"].strip()
    fmt = (ia.get("format") or "wav").lower()
    m = _AUDIO_DATA_URI_RE.match(data)
    if m:
        fmt = ia.get("format") or m.group(1)
        data = data.split(",", 1)[1]
    elif data.startswith(("http://", "https://")):
        raise ImageExtractionError(
            "http(s) audio URL reached the runner — the engine must resolve "
            "URLs; ranks never fetch")
    elif Path(data).suffix and Path(data).exists():
        return data
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageExtractionError(f"bad base64 audio: {e}")
    if len(raw) > max_mb * 1024 * 1024:
        raise ImageExtractionError(f"audio {len(raw)//(1024*1024)}MB > cap {max_mb}MB")
    fmt = re.sub(r"[^a-z0-9]", "", fmt.lower()) or "wav"
    tmp = f"/tmp/vlmr_{req_id}_a{idx}.{fmt}"
    with open(tmp, "wb") as f:
        f.write(raw)
    return tmp


def _extract_media(messages: list[dict], req_id: str, max_image_mb: int
                   ) -> tuple[list[dict], list[str], list[str], list[str]]:
    """Returns (template_messages, image_paths, audio_paths, temp_paths)."""
    out_msgs: list[dict] = []
    images: list[str] = []
    audios: list[str] = []
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
            if ptype in _TEXTLIKE_PARTS:
                texts.append(p.get("text") or "")
                continue
            if ptype == "input_audio":
                path = _decode_audio_part(p, req_id, len(audios), max_image_mb)
                if path.startswith("/tmp/vlmr_"):
                    temps.append(path)
                audios.append(path)
                continue
            if ptype not in ("image_url", "input_image", "image"):
                raise ImageExtractionError(
                    f"unsupported content part {ptype!r} (this runner reads "
                    f"text, image and input_audio)")
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
    return out_msgs, images, audios, temps


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
            tmpl_messages, images, audios, temps = _extract_media(
                messages, req_id or "noid", max_image_mb)

            chat_kwargs: dict = {"num_images": len(images)}
            if audios:
                chat_kwargs["num_audios"] = len(audios)
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
                      "repetition_penalty"):
                if req.get(k) is not None:
                    gen_kwargs[k] = req[k]
            # One RNG stream on every rank. mlx-vlm's stream_generate ignores a
            # `seed` kwarg (only its CLI seeds), so each rank sampled from its
            # own process RNG: at temperature > 0 the ranks could draw
            # different tokens, and one reaching EOS first leaves the others
            # blocked in the next collective. Without a client seed, rank 0
            # draws one and the all_sum hands it to everyone (a 24-bit value
            # stays exact in float32).
            seed = req.get("seed")
            if seed is None and size > 1:
                local = (int.from_bytes(os.urandom(3), "little")
                         if rank == 0 else 0)
                shared = mx.distributed.all_sum(
                    mx.array([float(local)], dtype=mx.float32), group=group)
                seed = int(shared.item())
            if seed is not None:
                mx.random.seed(int(seed))

            if rank == 0:
                log(f"req {req_id}: images={len(images)} audios={len(audios)} "
                    f"max_tokens={max_tokens} seed={seed} "
                    f"sampling={ {k: v for k, v in gen_kwargs.items() if k != 'max_tokens'} }")

            ntoks = 0
            t_gen = time.time()
            buf: list[str] = []
            finish_reason = None
            cancelled_mid_gen = False
            last = None
            for res in stream_generate(model, processor, formatted,
                                       image=images or None,
                                       audio=audios or None, **gen_kwargs):
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
            log(f"req {req_id}: media extraction failed: {e}")
            emit(rank, {"event": "done", "id": req_id, "ntoks": 0,
                        "prompt_tokens": 0, "cached_tokens": 0,
                        "elapsed_s": 0.0, "tps": 0.0,
                        "finish_reason": "error", "error": str(e)})
        except Exception as e:
            import traceback as _tb
            _t = _tb.format_exc()
            try:
                with open(f"/tmp/vlm_err_rank{rank}.log", "a") as _f:
                    _f.write(f"=== req {req_id} ===\n{_t}\n")
            except Exception:
                pass
            log(f"req {req_id}: GEN ERROR: {e}")
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
