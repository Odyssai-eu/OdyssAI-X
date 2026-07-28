"""Attempted single-node capacity probe for one Kimi K3 rank — INCONCLUSIVE.

Kept for the record of what was tried and what it taught. Three runs on .33
(245 GiB wired limit, 227 GiB slice of the real converted checkpoint):

  1. one-shot mx.eval of the slice        -> process killed, no verdict
  2. per-layer eval + mx.set_wired_limit  -> GPU Timeout at ~layer 12: a 15 GiB
     layer materialising under I/O starvation demand-pages inside one Metal
     command buffer, which has a timeout; CPU reads do not
  3. + CPU page-prefetch per layer        -> prefetch cured the GPU timeout,
     then jetsam SIGKILLed the process (file-cache pressure, RSS small)

Root limitation: mx.eval of mmap-loaded weights is zero-copy — wired stayed at
~4.6 GiB in every run. The weights only get wired at generation time (residency
set + set_wired_limit inside the serving process). A harness cannot reproduce
that lifecycle without BEING the runner, so the 95%-utilisation question is
only answerable by the real load, with a human watching vm_stat and the Q3
fallback one command away.

What transfers to the real load: prefetching a layer's pages on the CPU before
the GPU touches them removes the GPU-timeout failure mode, and the last rank
of a 256 GiB node enters the starved regime for roughly its final third.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import mlx.core as mx
import mlx.nn as nn

GIB = 1024**3


def vm_stat():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = 16384
    vals = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().rstrip(".")
        if v.isdigit():
            vals[k.strip()] = int(v) * page
    return {
        "free": vals.get("Pages free", 0),
        "wired": vals.get("Pages wired down", 0),
        "compressor": vals.get("Pages occupied by compressor", 0),
        "swapins": vals.get("Swapins", 0),
        "swapouts": vals.get("Swapouts", 0),
    }


def show(tag, s):
    print(
        f"  {tag:<22} free {s['free']/GIB:7.1f}  wired {s['wired']/GIB:7.1f}  "
        f"compressor {s['compressor']/GIB:6.2f}  swapouts {s['swapouts']/GIB:.2f}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target-gib", type=float, default=233.0)
    ap.add_argument("--ctx", type=int, default=2048)
    a = ap.parse_args()

    limit = int(
        subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True
        ).stdout.strip()
        or 0
    )
    print(f"node wired limit: {limit} MB ({limit/1024:.0f} GiB)")
    print(f"target: {a.target_gib:.0f} GiB of weights (one rank's share)\n")

    cfg = json.load(open(f"{a.model}/config.json"))
    index = json.load(open(f"{a.model}/model.safetensors.index.json"))
    wmap = index["weight_map"]

    # Take whole shards, in order, until the target is reached. Shards map 1:1
    # to layers, so this is exactly what a rank-0 pipeline shard looks like.
    shards, total = [], 0
    for shard in sorted(set(wmap.values())):
        p = os.path.join(a.model, shard)
        size = os.path.getsize(p)
        if total + size > a.target_gib * GIB and shards:
            break
        shards.append(shard)
        total += size

    layers = set()
    for k, shard in wmap.items():
        if shard in shards and k.startswith("model.layers."):
            layers.add(int(k.split(".")[2]))
    n_layers = max(layers) + 1 if layers else 0
    print(f"{len(shards)} shards, {total/GIB:.1f} GiB, layers 0..{n_layers-1}")

    # Wire MLX buffers the way generation does (mlx_lm.generate:257 /
    # runner via stream_generate): without this the weights stay file-backed,
    # nothing gets wired, and the first GPU op demand-pages 227 GiB through
    # the disk — observed as a GPU Timeout crash, not a measurement.
    info = mx.device_info()
    mx.set_wired_limit(info["max_recommended_working_set_size"])
    print(f"mlx wired limit: {info['max_recommended_working_set_size']/GIB:.0f} GiB")

    before = vm_stat()
    show("avant", before)

    from mlx_lm.models.kimi_k3 import Model, ModelArgs

    cfg = dict(cfg)
    cfg["num_hidden_layers"] = n_layers
    args = ModelArgs.from_dict(cfg)
    model = Model(args)

    weights = {}
    for shard in shards:
        weights.update(mx.load(os.path.join(a.model, shard)))
    weights = model.sanitize(weights)

    q = cfg["quantization"]

    def class_predicate(p, m):
        if p in q:
            return q[p]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{p}.scales" in weights

    nn.quantize(
        model, group_size=q["group_size"], bits=q["bits"],
        mode=q.get("mode", "affine"), class_predicate=class_predicate,
    )

    # Only the layers this slice carries; the tail tensors live elsewhere.
    have = {k: v for k, v in weights.items() if k.startswith("model.layers.")}
    model.model.layers = model.model.layers[:n_layers]
    model.load_weights(list(have.items()), strict=False)
    # Materialise layer by layer, the way pipeline_auto_parallel does. A single
    # eval of 227 GiB forces the kernel to evict that much file cache at once
    # and got the process jetsam-killed on a node fresh out of a 1.4 TB sync.
    # Map each layer to its shard so its pages can be prefetched on the CPU
    # right before the GPU materialises it. Without this, once free RAM is
    # exhausted the eval demand-pages a 15 GiB layer from disk INSIDE a Metal
    # command buffer and dies on kIOGPUCommandBufferCallbackErrorTimeout —
    # observed twice at layer ~12. Sequential CPU reads have no timeout.
    layer_shard = {}
    for k, sh in wmap.items():
        if k.startswith("model.layers.") and sh in shards:
            layer_shard.setdefault(int(k.split(".")[2]), set()).add(sh)

    def prefetch(paths):
        buf = bytearray(64 * 1024 * 1024)
        for path in paths:
            with open(path, "rb", buffering=0) as f:
                while f.readinto(buf):
                    pass

    for i, layer in enumerate(model.model.layers):
        prefetch(os.path.join(a.model, sh) for sh in layer_shard.get(i, ()))
        mx.eval(layer.parameters())
        mx.clear_cache()
        if i % 4 == 0:
            show(f"couche {i}", vm_stat())
    loaded = vm_stat()
    show("apres chargement", loaded)

    print(f"\nforward, contexte {a.ctx}")
    h = mx.random.normal((1, a.ctx, args.hidden_size)).astype(mx.bfloat16)
    br = mx.zeros((a.ctx, 0, args.hidden_size), dtype=h.dtype)
    t0 = time.time()
    for layer in model.model.layers:
        out = layer(h, mask=None, cache=None, block_residual=br)
        h, br = out if isinstance(out, tuple) else (out, br)
    mx.eval(h)
    dt = time.time() - t0
    after = vm_stat()
    show("apres forward", after)
    print(f"  {n_layers} couches en {dt:.1f}s ({dt/n_layers*1000:.0f} ms/couche)")

    grew = (after["compressor"] - before["compressor"]) / GIB
    swapped = (after["swapouts"] - before["swapouts"]) / GIB
    free_left = after["free"] / GIB
    print(
        f"\ncompressor +{grew:.2f} GiB   swapouts +{swapped:.2f} GiB   "
        f"free restant {free_left:.1f} GiB"
    )

    paging = grew > 2.0 or swapped > 0.5 or free_left < 2.0
    print("\nVERDICT: " + ("PAGING" if paging else "CLEAN"))
    if not paging:
        print(
            f"  {total/GIB:.0f} GiB tiennent sous {limit/1024:.0f} GiB sans pagination ; "
            f"extrapole a 5 rangs -> le split complet tient."
        )
    else:
        print("  le split ne tient pas — repli experts Q3 (--experts q3).")
    sys.exit(1 if paging else 0)


if __name__ == "__main__":
    main()
