"""LongCat-2.0 raw bf16 -> MLX 3-bit headbf16 — STREAMING converter.

Why not mlx_lm.convert: mlx#3803 — Metal watchdog kills GPU command buffers
whose kernels stall on mmap page-faults from external volumes (/Volumes/models
is a TB external). convert fuses disk loads into GPU quantize buffers -> GPU
Timeout at the ngram embedders regardless of MLX_MAX_OPS_PER_BUFFER.

Fix (zcbenz's workaround, adapted to stream since 3.3 TB >> 512 GB RAM):
  1. default device = CPU  -> every sanitize node (dequant/stack/split) is CPU
  2. per tensor: mx.eval on CPU (fs read to RAM, no watchdog on CPU)
  3. mx.quantize(..., stream=mx.gpu) on the RAM-resident buffer (0.6 s/16.9 GB,
     far under the watchdog), probe-validated
  4. flush shards of ~5 GB, free, clear_cache

Quant policy (mirrors the aborted driver):
  - 3-bit g64 affine default, INCLUDING the 16 ngram embedders (the capacity
    win: 272.8 GB bf16 -> ~60 GB)
  - lm_head bf16 (headbf16 house policy)
  - DSA indexer bf16 (mirror MiniMax-M3 policy)
  - MoE router *.classifier at 8-bit g64 (upstream model.quant_predicate)
  - per-path overrides recorded in config["quantization"] for load parity
"""
import glob
import json
import os
import shutil
import time

import mlx.core as mx

RAW = "/Volumes/models/mlx/raw/meituan-longcat/LongCat-2.0"
OUT = "/Volumes/models/odysseus/odyssai/LongCat-2.0-mlx-3bit-headbf16"
SHARD_BYTES = 5 * 1024**3

mx.set_default_device(mx.cpu)   # BEFORE sanitize: all its lazy nodes -> CPU

from mlx_lm.utils import _get_classes  # noqa: E402

config = json.load(open(os.path.join(RAW, "config.json")))
model_class, model_args_class = _get_classes(config)
model = model_class(model_args_class.from_dict(config))  # lazy init, no eval

print("loading raw shards (lazy)...", flush=True)
weights = {}
for wf in sorted(glob.glob(os.path.join(RAW, "model*.safetensors"))):
    weights.update(mx.load(wf))
print(f"raw keys: {len(weights)}", flush=True)

weights = model.sanitize(weights)   # lazy CPU nodes (renames/stack/dequant)
print(f"sanitized keys: {len(weights)}", flush=True)
del model   # structure served its purpose; its random-init lazies die here


def decide(key, w):
    """None = passthrough; else (group_size, bits)."""
    if not key.endswith(".weight"):
        return None
    if w.ndim < 2 or w.shape[-1] % 64 != 0:
        return None
    path = key[: -len(".weight")]
    if path.endswith("lm_head"):
        return None                       # headbf16
    if ".indexer." in path or path.endswith("indexer"):
        return None                       # DSA indexer bf16
    if path.endswith("classifier"):
        return (64, 8)                    # MoE router (upstream policy)
    return (64, 3)


os.makedirs(OUT, exist_ok=True)
for f in glob.glob(os.path.join(OUT, "*")):
    os.remove(f)

qcfg = {"group_size": 64, "bits": 3, "mode": "affine"}
weight_map = {}
shard, shard_size, shard_i, total = {}, 0, 0, 0
n_q = n_pass = 0
t0 = time.time()
keys = list(weights.keys())


def flush():
    global shard, shard_size, shard_i
    if not shard:
        return
    shard_i += 1
    name = f"model-{shard_i:05d}.safetensors"
    mx.save_safetensors(os.path.join(OUT, name), shard,
                        metadata={"format": "mlx"})
    for k in shard:
        weight_map[k] = name
    shard, shard_size = {}, 0
    mx.clear_cache()


for i, k in enumerate(keys):
    w = weights.pop(k)
    mx.eval(w)                            # CPU: fs read (+stack/dequant) -> RAM
    d = decide(k, w)
    if d is None:
        shard[k] = w
        shard_size += w.nbytes
        n_pass += 1
    else:
        gs, bits = d
        q, s, b = mx.quantize(w, group_size=gs, bits=bits, stream=mx.gpu)
        mx.eval(q, s, b)
        del w
        path = k[: -len(".weight")]
        if (gs, bits) != (64, 3):
            qcfg[path] = {"group_size": gs, "bits": bits}
        shard[k] = q
        shard[path + ".scales"] = s
        shard[path + ".biases"] = b
        shard_size += q.nbytes + s.nbytes + b.nbytes
        n_q += 1
    if shard_size >= SHARD_BYTES:
        flush()
    if i % 2000 == 0:
        print(f"  [{i}/{len(keys)}] shards={shard_i} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
flush()

total_bytes = 0
for f in glob.glob(os.path.join(OUT, "model-*.safetensors")):
    total_bytes += os.path.getsize(f)
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as fid:
    json.dump({"metadata": {"total_size": total_bytes},
               "weight_map": weight_map}, fid, indent=1)

config["quantization"] = qcfg
config["quantization_config"] = qcfg
with open(os.path.join(OUT, "config.json"), "w") as fid:
    json.dump(config, fid, indent=2)

for pat in ("*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken", "*.py"):
    for f in glob.glob(os.path.join(RAW, pat)):
        base = os.path.basename(f)
        if base in ("config.json", "model.safetensors.index.json"):
            continue
        shutil.copy2(f, os.path.join(OUT, base))

print(f"CONVERSION DONE: {shard_i} shards, {total_bytes/1e9:.0f} GB, "
      f"{n_q} quantized / {n_pass} passthrough, "
      f"{(time.time()-t0)/60:.0f} min", flush=True)
