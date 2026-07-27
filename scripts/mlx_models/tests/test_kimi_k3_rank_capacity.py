"""Does one rank's share of Kimi K3 actually fit under its wired limit?

The load is planned at ~95% of the wired budget, which is further than this
cluster has ever gone (previous high: Ling-1T at 74%). The failure mode at that
level is not an error — it is PAGING: the allocation succeeds, macOS starts
compressing and swapping, and throughput collapses. That has to be measured,
not argued about.

This loads a real slice of the converted checkpoint sized to one rank's share
on a single node, runs a forward, and reports what the memory system did. No
cluster, no distributed init, no risk to anything else.

    test_kimi_k3_rank_capacity.py --model <converted dir> --target-gib 233
                                  [--ctx 2048]

Reads `--target-gib` worth of consecutive layers starting at layer 0. Verdict:

  CLEAN   allocation stayed wired, compressor flat -> the real load will hold
  PAGING  free collapsed / compressor grew -> the split does NOT fit, fall back
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
    mx.eval(model.parameters())
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
