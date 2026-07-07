"""hy3_mtp_extract_release.py — Extract the MoE-MTP head (model.layers.80) from
the LOCAL tencent/Hy3 release (bf16, 2026-07) into a sidecar for the DISTRIBUTED
native-MTP framework (RUNNER_MTP_SIDECAR — the pre-quantized fast-path that
fixed the multi-rank OOM).

Variant of hy3_mtp_extract.py (which curl'd the 2 preview shards): the release
is already on disk, so the shards holding layers.80 are derived from the local
index. Key naming is identical to the preview original (verified: the release's
layer-80 orphans are exactly eh_proj/enorm/hnorm/final_layernorm + a full MoE
decoder layer). Same transforms:
  - strip `model.layers.80.` ; head keys at root ; decoder layer under layers.0.
  - mlp.expert_bias -> layers.0.mlp.router.expert_bias
  - stack 192 experts -> layers.0.mlp.switch_mlp.{gate,up,down}_proj [E,out,in]
  - quantize 2D+ weights (HEAD_BITS/HEAD_GROUP, default Q4 g32) — drafter
    precision only affects acceptance, never output (trunk verifies).
"""
import json
import os
import sys

import mlx.core as mx

SRC = "/Volumes/models/odysseus/tencent/Hy3"
OUT = "/Volumes/models/odysseus/odyssai/Hy3-MTP-head"
LAYER = 80
N_EXPERTS = 192
PFX = f"model.layers.{LAYER}."
QUANT_BITS = int(os.environ.get("HEAD_BITS", "4"))
QUANT_GROUP = int(os.environ.get("HEAD_GROUP", "32"))


def log(*a):
    print(*a, flush=True)


def load_layer():
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    shards = sorted(set(v for k, v in wm.items() if k.startswith(PFX)))
    log(f"[src] layers.{LAYER} spans {len(shards)} shard(s): {shards}")
    w = {}
    for s in shards:
        for k, v in mx.load(os.path.join(SRC, s)).items():
            if k.startswith(PFX):
                w[k[len(PFX):]] = v
    log(f"[src] {len(w)} tensors for the head")
    return w


def transform(w):
    out = {}
    consumed = set()
    for k in ["eh_proj.weight", "enorm.weight", "hnorm.weight",
              "final_layernorm.weight"]:
        out[k] = w[k]
        consumed.add(k)
    if "mlp.expert_bias" in w:
        out["layers.0.mlp.router.expert_bias"] = w["mlp.expert_bias"]
        consumed.add("mlp.expert_bias")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        stack = [w[f"mlp.experts.{e}.{proj}.weight"] for e in range(N_EXPERTS)]
        out[f"layers.0.mlp.switch_mlp.{proj}.weight"] = mx.stack(stack)
    for k in w:
        if k in consumed or k.startswith("mlp.experts."):
            continue
        out[f"layers.0.{k}"] = w[k]
    log(f"[transform] {len(out)} tensors (experts stacked, head keys at root)")
    return out


def quantize_head(out):
    q = {}
    nq = 0
    for k, w in out.items():
        if k.endswith(".weight") and w.ndim >= 2 and w.shape[-1] % QUANT_GROUP == 0:
            wq, sc, bi = mx.quantize(w, group_size=QUANT_GROUP, bits=QUANT_BITS)
            base = k[: -len(".weight")]
            q[k] = wq
            q[base + ".scales"] = sc
            q[base + ".biases"] = bi
            nq += 1
        else:
            q[k] = w
    log(f"[quant] {nq} matrices -> {QUANT_BITS}-bit g{QUANT_GROUP}; "
        f"{len(out) - nq} kept full (norms + expert_bias)")
    return q


def main():
    w = load_layer()
    out = transform(w)
    missing = [k for k in ("eh_proj.weight", "enorm.weight", "hnorm.weight")
               if k not in out]
    if missing:
        sys.exit(f"missing head keys: {missing}")
    out = quantize_head(out)
    os.makedirs(OUT, exist_ok=True)
    mx.save_safetensors(os.path.join(OUT, "mtp.safetensors"), out)
    log(f"[save] mtp.safetensors "
        f"({os.path.getsize(os.path.join(OUT, 'mtp.safetensors'))/1e9:.1f} GB)")

    t = json.load(open(os.path.join(SRC, "config.json")))
    cfg = {
        "model_type": "hy_v3_mtp",
        "architectures": ["HYV3MTPForCausalLM"],
        "num_mtp_layers": 1,
        "quantization": {"bits": QUANT_BITS, "group_size": QUANT_GROUP,
                         "mode": "affine"},
        "quantization_config": {"bits": QUANT_BITS, "group_size": QUANT_GROUP,
                                "mode": "affine"},
        "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
        "_trunk": "odyssai/Hy3-mlx-bf16 (tencent/Hy3 release 2026-07)",
        "_note": "embed_tokens + lm_head SHARED from the trunk at load.",
    }
    for k in ("hidden_size", "intermediate_size", "moe_intermediate_size",
              "num_attention_heads", "num_key_value_heads", "head_dim",
              "num_experts", "num_experts_per_tok", "num_shared_experts",
              "rms_norm_eps", "vocab_size", "max_position_embeddings",
              "tie_word_embeddings"):
        if k in t:
            cfg[k] = t[k]
    for k in ("router_scaling_factor", "route_norm", "qk_norm",
              "rope_parameters", "rope_theta", "rope_scaling"):
        if t.get(k) is not None:
            cfg[k] = t[k]
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
    log("[save] config.json (model_type hy_v3_mtp)")
    log("DONE — sidecar at " + OUT)


if __name__ == "__main__":
    main()
