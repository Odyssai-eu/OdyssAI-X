"""Two-pass Q6 of Qwen3.8-Flash-Next with the mlx_lm 0.31.3 built-in qwen4_exp port
(zero-centred norms, batch-aware, no bundled model_file).
Pass 1: 6-bit gs64 everywhere except the MoE router (bf16, port rule) and the n-gram
        embedding shards (160 cols, not divisible by 64 -> left bf16 in this pass).
Pass 2: the n-gram shards at 6-bit gs32 (pipenetwork layout) -> ~138 GB total.
Saving is streamed shard by shard with ONE tensor evaluated per step: the stock
save_model evaluates a whole ~5 GB shard in one go, which tripped the Metal GPU
watchdog (kIOGPUCommandBufferCallbackErrorTimeout) on ultra-512 while the GLM
runner shares the GPU.

Usage (on the converter node, mlx-cluster venv, source bf16 present locally):
    python scripts/qwen4_exp_convert_q6.py
Edit SRC / P1 / DST below. Recipe and measurements: docs/FEATURE-2026-09-06-replica-continuous-batching.md §6.3.
Result on 2026-09-06: 137 GB, 6.65 bpw, 37.5 tok/s single-stream on a Mac Studio 256 GB
(same as the pipenetwork 6-bit), greedy parity, batch-aware (mlx_lm 0.31.3 built-in port).
"""
import glob, json, shutil, time
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_map_with_path
from mlx_lm.utils import load, quantize_model, save_config, make_shards, get_total_parameters

SRC = Path("/Volumes/models/odysseus/Qwen/Qwen3.8-Flash-Next")
P1 = Path("/Volumes/models/odysseus/odyssai/Qwen3.8-Flash-Next-Q6.pass1")
DST = Path("/Volumes/models/odysseus/odyssai/Qwen3.8-Flash-Next-Q6")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def stream_save(dst: Path, model, tokenizer, config, extras_src: Path):
    dst.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.parameters()))
    shards = make_shards(weights)
    n = len(shards)
    fmt = "model-{:05d}-of-{:05d}.safetensors" if n > 1 else "model.safetensors"
    index = {"metadata": {"total_size": sum(v.nbytes for v in weights.values()),
                          "total_parameters": get_total_parameters(model)},
             "weight_map": {}}
    model.update(tree_map(lambda _: mx.array([]), model.parameters()))   # donate
    weights.clear(); del weights
    for i in range(n):
        shard = shards[i]; shards[i] = None
        for v in shard.values():
            mx.eval(v)                       # small command buffers, no GPU watchdog
        name = fmt.format(i + 1, n)
        mx.save_safetensors(str(dst / name), shard, metadata={"format": "mlx"})
        for k in shard:
            index["weight_map"][k] = name
        del shard
        mx.clear_cache()
        log(f"  shard {i + 1}/{n} written")
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=4)
    save_config(config, config_path=dst / "config.json")
    tokenizer.save_pretrained(dst)
    for p in ("generation_config.json",):
        for f in glob.glob(str(extras_src / p)):
            shutil.copy(f, dst)

# Small-output projections stay bf16: a 6-bit matmul with N=1..48 rows is far slower
# than the bf16 one (v1 of this Q6 quantized them and decoded at 21 tk/s vs 37-43 for
# the pipenetwork 6-bit, which leaves exactly these in bf16). Same checkpoint, same
# code otherwise — measured on ultra-256d, 2026-09-06.
SMALL_OUT = ("linear_attn.in_proj_a", "linear_attn.in_proj_b", "block_inject_weight",
             "mlp.shared_expert_gate", "indexer.index_qk_proj")
_excluded = []
def pred1(path, module, _=None):
    if path.endswith("mlp.gate"):
        return False                      # router stays bf16 (port's own rule)
    if "ngram_embedding.shard_" in path:
        return False                      # pass 2, gs32
    w = getattr(module, "weight", None)
    if path.endswith(SMALL_OUT) or (w is not None and w.ndim == 2 and w.shape[0] < 64):
        _excluded.append(path)
        return False
    return True

def pred2(path, module, _=None):
    return "ngram_embedding.shard_" in path

if P1.exists():
    shutil.rmtree(P1)
if DST.exists():
    raise SystemExit(f"{DST} already exists — refusing to overwrite")

t0 = time.time()
log(f"pass1: {SRC} -> {P1}")
model, tokenizer, config = load(str(SRC), return_config=True, lazy=True,
                                tokenizer_config={"trust_remote_code": False})
dtype_name = config.get("torch_dtype") or (config.get("text_config") or {}).get("dtype") or "bfloat16"
dtype = getattr(mx, dtype_name)
cast_predicate = getattr(model, "cast_predicate", lambda _: True)
def set_dtype(k, v):
    return v.astype(dtype) if (cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating)) else v
model.update(tree_map_with_path(set_dtype, model.parameters()))
model, config = quantize_model(model, config, 64, 6, mode="affine", quant_predicate=pred1)
import collections, re
fam = collections.Counter(re.sub(r"layers\.\d+", "layers.N", x) for x in _excluded)
log(f"kept bf16 (small-output rule): {dict(fam)}")
stream_save(P1, model, tokenizer, config, SRC)
log(f"pass1 done in {time.time() - t0:.0f}s")
del model; mx.clear_cache()

t1 = time.time()
log(f"pass2: n-gram shards 6-bit gs32 -> {DST}")
model, tokenizer, config = load(str(P1), return_config=True, lazy=True)
model, config = quantize_model(model, config, 32, 6, mode="affine", quant_predicate=pred2)
stream_save(DST, model, tokenizer, config, P1)
log(f"pass2 done in {time.time() - t1:.0f}s; removing {P1}")
del model; mx.clear_cache()
shutil.rmtree(P1)
log("ALL DONE")
