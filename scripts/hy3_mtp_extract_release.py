"""hy3_mtp_extract_release.py — Extract the MoE-MTP head (model.layers.80) from
the LOCAL tencent/Hy3 release (bf16, 2026-07) into the CANONICAL sidecar format
consumed by scripts/mtp_module.py (the Python-distributed MTP framework).

Canonical contract (verified against mtp_module.detect_native_mtp/_load_source_
weights and the working GLM-5.2 sidecar):
  - keys kept RAW, `model.layers.80.*` verbatim, bf16 — the module does its own
    strip/rename/expert-stacking at load;
  - file name `mtp-sidecar.safetensors`, living in `<model_dir>/mtp-sidecar/`
    (auto-discovered) — copy it into each served Hy3-mlx-* variant dir;
  - `module-q6.safetensors` (pre-quantized fast-path) is built later by
    mtp_module.build_prequantized_sidecar, not here.

NOTE: hy_v3 is not yet in mtp_module._DEEPSEEK_FAMILY — the head is GQA
(q/k_norm, no MLA kv_b) + final_layernorm, a new family binding to write when
the MTP dossier reopens. This sidecar just puts the weights where the framework
will look for them.
"""
import json
import os
import sys

import mlx.core as mx

SRC = "/Volumes/models/odysseus/tencent/Hy3"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hy3-mtp-sidecar-staging"
LAYER = 80
PFX = f"model.layers.{LAYER}."


def log(*a):
    print(*a, flush=True)


def main():
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    shards = sorted(set(v for k, v in wm.items() if k.startswith(PFX)))
    log(f"[src] layers.{LAYER} spans {len(shards)} shard(s)")
    w = {}
    for s in shards:
        for k, v in mx.load(os.path.join(SRC, s)).items():
            if k.startswith(PFX):
                w[k] = v          # keys kept verbatim — canonical raw format
    log(f"[src] {len(w)} tensors")
    for probe in ("eh_proj", "enorm", "hnorm"):
        if f"{PFX}{probe}.weight" not in w:
            sys.exit(f"missing {probe} — not an MTP layer?")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "mtp-sidecar.safetensors")
    mx.save_safetensors(path, w)
    log(f"[save] {path} ({os.path.getsize(path)/1e9:.1f} GB, raw bf16)")
    with open(os.path.join(OUT, "sidecar-index.json"), "w") as f:
        json.dump({"source": "tencent/Hy3 (release 2026-07)",
                   "layer": LAYER, "tensors": sorted(w)}, f, indent=1)
    log("DONE — canonical sidecar at " + OUT)


if __name__ == "__main__":
    main()
