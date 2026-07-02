"""GATE-1 caption probe — distributed mlx-vlm TP correctness vs single-node baseline.

Env: PROBE_MODEL, PROBE_IMAGE, PROBE_PROMPT, PROBE_MAX_TOKENS,
     PROBE_WORLD (1 = no distributed init; N = ring TP via MLX_HOSTFILE/MLX_RANK),
     PROBE_OUT (json result path).
All logging goes to stderr; stdout stays clean (JSONL hygiene rehearsal for the
future vlm_runner). Greedy only — divergence must be visible, never sampled away.
"""
import atexit
import hashlib
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

MODEL = os.environ.get(
    "PROBE_MODEL",
    "/Volumes/models/odysseus/odyssai/MiniMax-M3-VL-mlx-6bit-headbf16")
IMAGE = os.environ.get("PROBE_IMAGE", "/tmp/probe_image.png")
PROMPT = os.environ.get(
    "PROBE_PROMPT", "Describe this image: name every shape and its color.")
_pf = os.environ.get("PROBE_PROMPT_FILE")
if _pf:
    with open(_pf) as _f:
        PROMPT = _f.read()
MAXTOK = int(os.environ.get("PROBE_MAX_TOKENS", "64"))
WORLD = int(os.environ.get("PROBE_WORLD", "1"))
OUT = os.environ.get("PROBE_OUT", "")


def sha_arr(a):
    return hashlib.sha256(
        np.ascontiguousarray(np.array(a.astype(mx.float32)))).hexdigest()[:16]


if WORLD == 1:
    group, rank, size = None, 0, 1
else:
    group = mx.distributed.init(backend="ring", strict=True)
    rank, size = group.rank(), group.size()


def log(msg):
    print(f"PROBE r{rank}/{size} {msg}", file=sys.stderr, flush=True)


atexit.register(mx.clear_cache)

log(f"start model={MODEL} image={IMAGE} maxtok={MAXTOK}")

from mlx_vlm.utils import (get_model_path, load_image_processor, load_model,
                           load_processor)

SHARD_MODE = os.environ.get("PROBE_SHARD_MODE", "replicated-indexer")


def _shard_lm_replicated_indexer(lm, group):
    """Tensor-shard the MiniMax-M3 language model, REPLICATING the MSA indexer.

    Upstream LanguageModel.shard() shards index_q_proj and divides index_heads
    by the group size. But the block selector aggregates scores ACROSS index
    heads (mx.max(block_scores, axis=1) in _build_sparse_causal_mask_compiled),
    so each rank selects DIFFERENT key blocks from its local head subset ->
    o_proj all_sum mixes incoherent attentions -> deterministic garbage
    (identical on all ranks, wrong vs baseline — observed live 2026-07-02).
    Keeping the tiny indexer replicated makes block selection bit-identical to
    single-node on every rank. Everything else mirrors upstream shard().
    """
    from mlx_vlm.models.minimax_m3_vl import language as _lang
    shard_linear = _lang.shard_linear
    shard_inplace = _lang.shard_inplace
    n = group.size()
    for layer in lm.layers:
        sa = layer.self_attn
        if SHARD_MODE != "moe-only":
            sa.q_proj = shard_linear(sa.q_proj, "all-to-sharded", group=group)
            sa.k_proj = shard_linear(sa.k_proj, "all-to-sharded", group=group)
            sa.v_proj = shard_linear(sa.v_proj, "all-to-sharded", group=group)
            sa.o_proj = shard_linear(sa.o_proj, "sharded-to-all", group=group)
            sa.num_attention_heads //= n
            sa.num_key_value_heads //= n
        # index_q_proj / index_heads deliberately NOT sharded (replicated).
        if not layer.is_moe_layer:
            continue
        moe = layer.block_sparse_moe
        if moe.pack_shared_expert:
            # Fused [gate|up] projection: contiguous all-to-sharded slicing
            # gives rank0 all-gate rows and rank1 all-up rows (garbage).
            # Slice each half separately so the forward's midpoint split
            # yields correctly paired local gate/up (micro-proven exact).
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
    or not): rank r keeps gate[r*I/n:(r+1)*I/n] ++ up[same range]."""
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


def sharded_vlm_load(path, group):
    """In-repo replication of mlx_vlm.utils.sharded_load @ecc457b with 3 fixes:
    (1) calls model.language_model.shard(group) directly (top-level Model has no
    shard delegator upstream -> hasattr gate fails); (2) no bare print to stdout;
    (3) materializes the replicated vision tower/projector at load instead of
    lazily during the first image request mid-collective."""
    t0 = time.time()
    path = get_model_path(path)
    model = load_model(path, lazy=True, strict=False)
    config = model.config.to_dict()
    processor = load_processor(
        path, True, eos_token_ids=config.get("eos_token_id", None))
    image_processor = load_image_processor(path)
    if image_processor is not None:
        processor.image_processor = image_processor
    if group is not None and group.size() > 1:
        if SHARD_MODE == "upstream":
            model.language_model.shard(group)
            log("sharded language_model (upstream shard: MSA indexer sharded)")
        else:
            _shard_lm_replicated_indexer(model.language_model, group)
            log("sharded language_model (replicated-indexer variant)")
    log("materializing language model (this rank's slice)")
    mx.eval(model.language_model.parameters())
    log("materializing vision tower + projector (replicated)")
    mx.eval(model.parameters())
    model.eval()
    if group is not None and group.size() > 1:
        mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    return model, processor, config, time.time() - t0


model, processor, config, load_s = sharded_vlm_load(MODEL, group)
try:
    peak_gb = mx.get_peak_memory() / 1024**3
except Exception:
    peak_gb = -1.0
log(f"LOADED load_s={load_s:.1f} peak_gb={peak_gb:.1f}")

# Capture the vision-consistency evidence (#1 unknown): hash what enters and
# leaves get_input_embeddings on every rank.
hashes = {}
_orig_gie = model.get_input_embeddings


def _wrapped_gie(input_ids, pixel_values=None, **kw):
    out = _orig_gie(input_ids, pixel_values, **kw)
    if pixel_values is not None and "pv" not in hashes:
        hashes["ids"] = sha_arr(input_ids)
        hashes["pv"] = sha_arr(pixel_values)
        try:
            emb = out.inputs_embeds if hasattr(out, "inputs_embeds") else out
            hashes["emb"] = sha_arr(emb)
        except Exception as e:  # keep probing even if the shape surprises us
            hashes["emb"] = f"ERR:{e}"
    return out


model.get_input_embeddings = _wrapped_gie

from mlx_vlm.generate import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template

formatted = apply_chat_template(processor, config, PROMPT, num_images=1)
log(f"prompt formatted ({len(str(formatted))} chars)")

tokens, text_parts = [], []
t_gen0 = time.time()
ttft = None
last = None
ck = hashlib.sha256()
for chunk in stream_generate(model, processor, formatted, image=[IMAGE],
                             max_tokens=MAXTOK, temperature=0.0):
    if ttft is None:
        ttft = time.time() - t_gen0
    if chunk.token is not None:
        tokens.append(int(chunk.token))
        ck.update(int(chunk.token).to_bytes(4, "little", signed=False))
        if len(tokens) % 16 == 0:
            log(f"CK n={len(tokens)} sha={ck.hexdigest()[:12]}")
    text_parts.append(chunk.text)
    last = chunk

text = "".join(text_parts)
result = {
    "world": size,
    "rank": rank,
    "hash_ids": hashes.get("ids"),
    "hash_pv": hashes.get("pv"),
    "hash_emb": hashes.get("emb"),
    "n_tokens": len(tokens),
    "tokens_sha": ck.hexdigest()[:16],
    "tokens": tokens,
    "text": text,
    "ttft_s": round(ttft if ttft is not None else -1.0, 2),
    "gen_tps": round(getattr(last, "generation_tps", -1.0) or -1.0, 2),
    "prompt_tokens": getattr(last, "prompt_tokens", -1),
    "finish_reason": getattr(last, "finish_reason", None),
    "load_s": round(load_s, 1),
    "peak_gb": round(peak_gb, 1),
}
log("HASH ids={hash_ids} pv={hash_pv} emb={hash_emb}".format(**result))
log("TOKENS n={n_tokens} sha={tokens_sha}".format(**result))
log("PERF ttft={ttft_s}s gen_tps={gen_tps} prompt_tokens={prompt_tokens} "
    "finish={finish_reason}".format(**result))
if rank == 0:
    log("TEXT " + json.dumps(text)[:600])
if OUT:
    with open(OUT, "w") as f:
        json.dump(result, f)
log("done")
