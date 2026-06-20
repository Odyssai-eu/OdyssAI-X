#!/usr/bin/env python3
"""
hy3_mtp_extract.py — Extract Hy3's embedded MoE-MTP head (model.layers.80) from
the full-precision original `tencent/Hy3-preview` into a sidecar `mtp.safetensors`
drafter, paired at load time with the existing 9-bit trunk.

Why: mlx_lm convert strips the MTP head, so the 9-bit trunk on .29 has 0 MTP
tensors. The head DOES exist in the original at `model.layers.80` (verified) — a
full Hy3 MoE decoder layer plus the MTP-specific `eh_proj`/`enorm`/`hnorm`/
`final_layernorm`. It lives in just 2 of the 112 shards (111, 112).

Norm contract (resolved from vLLM hy_v3_mtp.py, the canonical impl):
  PRE-norm trunk hidden; concat embedding-FIRST into eh_proj; the head has its
  OWN final_layernorm before the SHARED lm_head; embed+lm_head shared with trunk.

We ship the head in **bf16** (no quant): simpler, more accurate (better acceptance),
and ~7GB is negligible next to the 309GB trunk. mlx-swift loads it as plain
Linear/SwitchGLU since the head config carries no `quantization` block.

Key transforms vs the raw checkpoint (strip the `model.layers.80.` prefix):
  - stack mlp.experts.{0..191}.{gate,up,down}_proj  ->  mlp.switch_mlp.{...}  (3D [E,out,in])
  - rename  mlp.expert_bias  ->  mlp.router.expert_bias   (match the trunk's HYV3MoE/HYV3Router)
  - keep eh_proj / enorm / hnorm / final_layernorm / input_layernorm /
    post_attention_layernorm / self_attn.* flat at the head root.

Output dir (on the model host): <models>/odyssai/Hy3-preview-MTP-head/
  mtp.safetensors + config.json (model_type "hy_v3_mtp") + tokenizer symlinks.
"""
import json, os, sys, glob, subprocess

import mlx.core as mx

REPO = "tencent/Hy3-preview"
SHARDS = ["model-00111-of-00112.safetensors", "model-00112-of-00112.safetensors"]
LAYER = 80
N_EXPERTS = 192

TRUNK = "/Volumes/models/odysseus/inferencerlabs/Hy3-preview-MLX-9bit"
DL_DIR = "/Volumes/models/.cache/hy3-mtp-src"          # where we curl the 2 shards
OUT = "/Volumes/models/odysseus/odyssai/Hy3-preview-MTP-head"

PFX = f"model.layers.{LAYER}."


def log(*a):
    print(*a, flush=True)


def download():
    os.makedirs(DL_DIR, exist_ok=True)
    for s in SHARDS:
        dst = os.path.join(DL_DIR, s)
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000_000:
            log(f"[dl] {s} present ({os.path.getsize(dst)/1e9:.1f} GB), skip")
            continue
        url = f"https://huggingface.co/{REPO}/resolve/main/{s}"
        log(f"[dl] {s} <- {url}")
        # -C - resumes; -L follows the CDN redirect; fail loudly on HTTP error.
        r = subprocess.run(["curl", "-fL", "-C", "-", "-o", dst, url])
        if r.returncode != 0:
            sys.exit(f"download failed for {s} (curl rc={r.returncode})")
        log(f"[dl] {s} done ({os.path.getsize(dst)/1e9:.1f} GB)")


def load_layer80():
    """Load all tensors from the 2 shards, keep only model.layers.80.*"""
    w = {}
    for s in SHARDS:
        path = os.path.join(DL_DIR, s)
        log(f"[load] {s}")
        t = mx.load(path)
        for k, v in t.items():
            if k.startswith(PFX):
                w[k[len(PFX):]] = v          # strip the model.layers.80. prefix
        del t
        mx.clear_cache()
    log(f"[load] kept {len(w)} layer-{LAYER} tensors")
    return w


def transform(w):
    """Head-level keys (eh_proj/enorm/hnorm/final_layernorm) at root; the single
    MTP decoder layer under `layers.0.` — mirrors Qwen35MTPDraftModel's
    {pre_fc_norm*, fc, norm} + layers:[MTPDecoderLayer] structure so the iterator
    generalization treats both drafts uniformly."""
    out = {}
    # head root (the MTP-specific projections/norms) — kept as-is
    head_keys = ["eh_proj.weight", "enorm.weight", "hnorm.weight", "final_layernorm.weight"]
    for k in head_keys:
        if k not in w:
            sys.exit(f"MISSING head key: {k}")
        out[k] = w[k]

    # the decoder layer -> prefix "layers.0."
    layer_exact = [
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "mlp.router.gate.weight",
        "mlp.shared_mlp.gate_proj.weight", "mlp.shared_mlp.up_proj.weight",
        "mlp.shared_mlp.down_proj.weight",
    ]
    for k in layer_exact:
        if k not in w:
            sys.exit(f"MISSING layer key: {k}")
        out[f"layers.0.{k}"] = w[k]

    # expert_bias: mlp.expert_bias -> layers.0.mlp.router.expert_bias (match HYV3Router)
    if "mlp.expert_bias" not in w:
        sys.exit("MISSING mlp.expert_bias")
    out["layers.0.mlp.router.expert_bias"] = w["mlp.expert_bias"]

    # stack the 192 per-expert tensors -> layers.0.mlp.switch_mlp.{proj} as [E, out, in]
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        stack = []
        for e in range(N_EXPERTS):
            ek = f"mlp.experts.{e}.{proj}.weight"
            if ek not in w:
                sys.exit(f"MISSING expert tensor: {ek}")
            stack.append(w[ek])
        key = f"layers.0.mlp.switch_mlp.{proj}.weight"
        out[key] = mx.stack(stack, axis=0)
        log(f"[stack] {key} -> {out[key].shape}")

    # sanity: every input expert tensor consumed, nothing odd left
    consumed = set(head_keys) | set(layer_exact) | {"mlp.expert_bias"}
    leftover = [k for k in w if k not in consumed and not k.startswith("mlp.experts.")]
    if leftover:
        log(f"[warn] unconsumed keys: {leftover}")
    return out


def write_outputs(out):
    os.makedirs(OUT, exist_ok=True)
    mtp_path = os.path.join(OUT, "mtp.safetensors")
    mx.save_safetensors(mtp_path, out)
    log(f"[save] {mtp_path}  ({len(out)} tensors, {os.path.getsize(mtp_path)/1e9:.1f} GB)")

    trunk_cfg = json.load(open(os.path.join(TRUNK, "config.json")))
    cfg = {
        "model_type": "hy_v3_mtp",
        "architectures": ["HYV3MTPForCausalLM"],
        "hidden_size": trunk_cfg["hidden_size"],
        "intermediate_size": trunk_cfg["intermediate_size"],
        "moe_intermediate_size": trunk_cfg["moe_intermediate_size"],
        "num_attention_heads": trunk_cfg["num_attention_heads"],
        "num_key_value_heads": trunk_cfg["num_key_value_heads"],
        "head_dim": trunk_cfg["head_dim"],
        "num_experts": trunk_cfg["num_experts"],
        "num_experts_per_tok": trunk_cfg["num_experts_per_tok"],
        "num_shared_experts": trunk_cfg["num_shared_experts"],
        "router_scaling_factor": trunk_cfg.get("router_scaling_factor", 1.0),
        "route_norm": trunk_cfg.get("route_norm", True),
        "qk_norm": trunk_cfg.get("qk_norm", True),
        "rms_norm_eps": trunk_cfg["rms_norm_eps"],
        "vocab_size": trunk_cfg["vocab_size"],
        "max_position_embeddings": trunk_cfg["max_position_embeddings"],
        "rope_parameters": trunk_cfg.get("rope_parameters"),
        "tie_word_embeddings": trunk_cfg.get("tie_word_embeddings", False),
        "num_mtp_layers": 1,
        # head ships bf16; no `quantization` block -> mlx-swift loads plain Linear/SwitchGLU.
        "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
        "_trunk": "inferencerlabs/Hy3-preview-MLX-9bit",
        "_note": "embed_tokens + lm_head are SHARED from the trunk at load (not in this file).",
    }
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
    log(f"[save] config.json (model_type hy_v3_mtp)")

    # symlink tokenizer from the trunk (the draft borrows it)
    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "generation_config.json"]:
        src = os.path.join(TRUNK, f)
        dst = os.path.join(OUT, f)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    log("[save] tokenizer symlinked from trunk")


def verify(out):
    # cross-check the stacked expert + a couple of shapes vs the trunk's switch_mlp
    import re
    log("=== verify ===")
    sm = out["layers.0.mlp.switch_mlp.gate_proj.weight"]
    log(f"  switch_mlp.gate_proj: {sm.shape} (expect [{N_EXPERTS}, moe_inter, hidden])")
    log(f"  eh_proj: {out['eh_proj.weight'].shape} (expect [hidden, 2*hidden])")
    log(f"  router.gate: {out['layers.0.mlp.router.gate.weight'].shape} (expect [num_experts, hidden])")
    log(f"  router.expert_bias: {out['layers.0.mlp.router.expert_bias'].shape} dtype={out['layers.0.mlp.router.expert_bias'].dtype}")
    log(f"  enorm/hnorm/final_layernorm dims: "
        f"{out['enorm.weight'].shape}/{out['hnorm.weight'].shape}/{out['final_layernorm.weight'].shape}")
    nbytes = sum(v.nbytes for v in out.values())
    log(f"  total head size: {nbytes/1e9:.2f} GB, {len(out)} tensors")


if __name__ == "__main__":
    download()
    w = load_layer80()
    out = transform(w)
    verify(out)
    write_outputs(out)
    log("DONE — sidecar at " + OUT)
