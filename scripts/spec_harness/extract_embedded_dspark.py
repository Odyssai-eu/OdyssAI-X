#!/Users/admin/mlx-cluster/.venv/bin/python
"""extract_embedded_dspark — B: pull the embedded DSpark/MTP drafter out of
DeepSeek-V4-Flash-DSpark into a standalone checkpoint.

The full checkpoint refuses to load in dspartha by design (embedded format);
the drafter tensors live under `mtp.*` in exactly 3 of 48 shards. This reads
ONLY those shards (selectively downloaded), extracts every `mtp.*` tensor,
strips the `mtp.<i>.` prefix, and writes:

  out/model.safetensors          — drafter weights (prefix-stripped)
  out/config.json                — synthesized standalone config: the dspark_*
                                   fields + the target-arch fields the drafter
                                   block shares (MLA dims, experts, vocab)
  out/EXTRACT-REPORT.json        — key inventory: every pattern, shapes,
                                   dtypes, per-tensor bytes (the map the V4
                                   drafter model-code will be written against)

Lossless: tensors are copied bit-for-bit (no dequant, no cast).
"""
import argparse
import json
import os
import re
from collections import defaultdict

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with the 3 shards + index + config")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    idx = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    spark_keys = {k: v for k, v in wm.items() if k.startswith("mtp.")}
    shards = sorted(set(spark_keys.values()))
    missing = [s for s in shards if not os.path.exists(os.path.join(args.src, s))]
    if missing:
        raise SystemExit(f"shards manquants: {missing}")
    print(f"{len(spark_keys)} clés mtp.* dans {len(shards)} shards")

    os.makedirs(args.out, exist_ok=True)
    tensors = {}
    for shard in shards:
        data = mx.load(os.path.join(args.src, shard))
        for k, v in data.items():
            if k.startswith("mtp."):
                # mtp.<block>.rest -> keep block index only if >1 block
                nk = re.sub(r"^mtp\.0\.", "", k) if all(
                    kk.startswith("mtp.0.") for kk in spark_keys
                ) else k[len("mtp."):]
                tensors[nk] = v
    print(f"{len(tensors)} tenseurs extraits")

    # Inventory report — the ground truth the drafter model-code is written
    # against (patterns, shapes, dtypes).
    pats = defaultdict(list)
    total = 0
    for k, v in tensors.items():
        pat = re.sub(r"\.\d+\.", ".N.", k)
        pats[pat].append((k, list(v.shape), str(v.dtype)))
        total += v.nbytes
    report = {
        "n_tensors": len(tensors),
        "total_gb": round(total / 1e9, 2),
        "patterns": {
            p: {"count": len(v), "example": v[0][0],
                "shape": v[0][1], "dtype": v[0][2]}
            for p, v in sorted(pats.items())
        },
    }
    json.dump(report, open(os.path.join(args.out, "EXTRACT-REPORT.json"), "w"),
              indent=1)

    # Synthesized standalone config: dspark_* + the shared arch fields.
    full = json.load(open(os.path.join(args.src, "config.json")))
    keep = [
        "vocab_size", "hidden_size", "num_attention_heads",
        "num_key_value_heads", "q_lora_rank", "kv_lora_rank",
        "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
        "n_routed_experts", "num_experts_per_tok", "n_shared_experts",
        "moe_intermediate_size", "intermediate_size", "rms_norm_eps",
        "rope_theta", "num_nextn_predict_layers",
        "dspark_block_size", "dspark_noise_token_id",
        "dspark_target_layer_ids", "dspark_markov_rank",
        "quantization", "quantization_config", "torch_dtype",
    ]
    cfg = {k: full[k] for k in keep if k in full}
    cfg["model_type"] = "deepseek_v4_dspark_drafter"
    cfg["extracted_from"] = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
    cfg["num_target_layers"] = full.get("num_hidden_layers")
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)

    mx.save_safetensors(os.path.join(args.out, "model.safetensors"), tensors)
    print(f"écrit: {args.out} ({report['total_gb']} GB)")
    print("top patterns:")
    for p in list(sorted(pats))[:12]:
        e = report["patterns"][p]
        print(f"  {p:55s} x{e['count']:3d} {e['shape']} {e['dtype']}")


if __name__ == "__main__":
    main()
