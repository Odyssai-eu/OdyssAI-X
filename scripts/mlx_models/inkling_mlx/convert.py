"""Streaming HF -> MLX conversion + quantization for Inkling.

The model is far too large (~1.9 TB bf16) to instantiate in RAM, so we convert
tensor-by-tensor: read each source shard (mmap), remap the name, apply the layout
transform, optionally affine-quantize the weight, and write output shards. Affine
quantization has no cross-tensor dependency, so per-tensor streaming is exactly
equivalent to ``nn.quantize(model)``.

Name/layout transforms vs. the checkpoint:
  * ``*_sconv.weight``  [C,1,K] -> [C,K,1]            (MLX conv1d layout)
  * ``mlp.w13_dn``      [2I,H]  -> gate_proj/up_proj   (split dense fused gate+up)
  * ``experts.w13_weight``        [E,2I,H] -> gate_proj/up_proj (split)
  * ``experts.w2_weight``         [E,H,I]  -> down_proj          (identity)
  * ``shared_experts.shared_w13`` [2,2I,H] -> gate_proj/up_proj  (split)
  * ``model.mtp.*``               dropped (inference-irrelevant)
  * everything else: identity
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil

import mlx.core as mx

_LAYER_RE = re.compile(r"model\.llm\.layers\.(\d+)\.")
_N_SHARED = 2   # Inkling: 2 shared experts (router rows after the routed ones)


def prune_moe_tensor(name: str, w: mx.array, keep, dense_mlp_idx: int) -> mx.array:
    """Subset a MoE tensor to the kept experts (REAP). ``keep`` maps sparse-layer
    index -> array of kept routed-expert indices. Applied BEFORE map_name/transform.
    Routed experts + the router (gate) are subset; shared experts are untouched."""
    m = _LAYER_RE.search(name)
    if m is None:
        return w
    L = int(m.group(1))
    if L < dense_mlp_idx:                        # dense layer — no routed experts
        return w
    kidx = mx.array(keep[L - dense_mlp_idx])     # [K]
    if name.endswith(("experts.w13_weight", "experts.w2_weight")) and "shared" not in name:
        return w[kidx]                           # [E, ...] -> [K, ...]
    if name.endswith("mlp.gate.weight"):         # [n_routed + n_shared, hidden]
        n_routed = w.shape[0] - _N_SHARED        # rows: [routed... , shared...]
        return mx.concatenate([w[kidx], w[n_routed:]], axis=0)   # [K + n_shared, hidden]
    if name.endswith("mlp.gate.bias"):           # [n_routed] correction bias
        return w[kidx]
    return w                                     # shared_experts.*, gate.global_scale, etc.


def map_name(name: str):
    """HF checkpoint tensor name -> list of (out_name, kind) for the MLX model."""
    if name.startswith("model.mtp."):
        return []  # drop MTP head

    if name.endswith(("k_sconv.weight", "v_sconv.weight", "attn_sconv.weight", "mlp_sconv.weight")):
        return [(name, "sconv")]

    # dense MLP fused gate+up / down
    if name.endswith("mlp.w13_dn.weight"):
        base = name[: -len("w13_dn.weight")]
        return [(base + "gate_proj.weight", "w13_gate"), (base + "up_proj.weight", "w13_up")]
    if name.endswith("mlp.w2_md.weight"):
        return [(name[: -len("w2_md.weight")] + "down_proj.weight", "identity")]

    # routed experts fused
    if name.endswith("experts.w13_weight"):
        base = name[: -len("w13_weight")]
        return [(base + "gate_proj.weight", "w13_gate"), (base + "up_proj.weight", "w13_up")]
    if name.endswith("experts.w2_weight"):
        return [(name[: -len("w2_weight")] + "down_proj.weight", "identity")]

    # shared experts fused
    if name.endswith("shared_experts.shared_w13_weight"):
        base = name[: -len("shared_w13_weight")]
        return [(base + "gate_proj.weight", "w13_gate"), (base + "up_proj.weight", "w13_up")]
    if name.endswith("shared_experts.shared_w2_weight"):
        return [(name[: -len("shared_w2_weight")] + "down_proj.weight", "identity")]

    return [(name, "identity")]


def transform(w: mx.array, kind: str) -> mx.array:
    if kind == "identity":
        return w
    if kind == "sconv":
        # [C, 1, K] -> [C, K, 1]
        return mx.swapaxes(w, 1, 2)
    if kind in ("w13_gate", "w13_up"):
        # The checkpoint stores gate/up INTERLEAVED row-wise: [g0, u0, g1, u1, ...]
        # (SGLang `deinterleave_w13`). De-interleave: gate = rows 0::2, up = rows 1::2.
        # A contiguous [:half]/[half:] split scrambles gate<->up in every MLP.
        n = w.shape[-2] // 2
        g = w.reshape(*w.shape[:-2], n, 2, w.shape[-1])
        return g[..., 0, :] if kind == "w13_gate" else g[..., 1, :]
    raise ValueError(kind)


# ---- quantization target predicate (must be identical in convert and load) ----

# Quant "recipes" — which module leaves get affine-quantized.
#   uniform       : everything (attention, MLP/experts, embed/unembed, audio, vision)
#   experts_only  : ONLY the MLP/expert matmuls (+ audio/vision); attention and
#                   embed/unembed stay bf16. Inkling attention dominates 4-bit error
#                   (~58% per layer vs ~15% for experts), so this keeps a 4-bit-sized
#                   build coherent while the ~927 B experts still fit in 512 GB.
_RECIPES = {
    "uniform": {"wq_du", "wk_dv", "wv_dv", "wr_du", "wo_ud",
                "gate_proj", "up_proj", "down_proj", "embed", "unembed", "encoder"},
    "experts_only": {"gate_proj", "up_proj", "down_proj", "encoder"},
}


def is_quant_target(out_name: str, quant_axis_size: int, group_size: int, recipe: str = "uniform") -> bool:
    """Whether ``out_name`` (a converted param path) should be affine-quantized."""
    if not out_name.endswith(".weight"):
        return False
    leaf = out_name[: -len(".weight")].rsplit(".", 1)[-1]
    leaves = _RECIPES[recipe]
    # vision projection layers (linear_0 .. linear_3) — quantized in both recipes
    is_vision_linear = leaf.startswith("linear_") and ".visual." in out_name
    if leaf not in leaves and not is_vision_linear:
        return False
    # router gate stays fp (leaf == "gate", excluded above); norms/sconv excluded by leaf
    # can only group-quantize when the input dim is a multiple of group_size
    return quant_axis_size % group_size == 0


# ------------------------------ streaming driver ------------------------------

_SHARD_CAP_BYTES = 5_000_000_000  # ~5 GB per output shard


def _process_tensor(name, w, bits, group_size, out_dtype, recipe="uniform", keep=None, dmi=2):
    """Yield (out_name, array) pairs for one source tensor."""
    if keep is not None:
        w = prune_moe_tensor(name, w, keep, dmi)   # REAP: subset to kept experts
    for out_name, kind in map_name(name):
        wt = transform(w, kind)
        quantize = bits is not None and is_quant_target(out_name, wt.shape[-1], group_size, recipe)
        if quantize:
            qw, scales, biases = mx.quantize(wt, group_size=group_size, bits=bits)
            base = out_name[: -len(".weight")]
            yield out_name, qw
            yield base + ".scales", scales
            yield base + ".biases", biases
        else:
            # keep norms/router/sconv/rel-proj in fp32-safe dtype; matmul weights in out_dtype
            keep_hi = wt.dtype == mx.float32 and (".global_scale" in out_name or ".bias" in out_name
                                                  or out_name.endswith(("_norm.weight", "norm.weight")))
            yield out_name, wt.astype(mx.float32 if keep_hi else out_dtype)


def convert_model(src: str, dst: str, bits=None, group_size: int = 64, out_dtype=mx.bfloat16,
                  recipe: str = "uniform", keep_path=None):
    """Stream-convert an Inkling checkpoint from ``src`` to ``dst``.

    ``bits=None`` -> plain dtype cast (bf16). ``bits in {4,6,8}`` -> affine quant.
    ``recipe`` selects which modules are quantized (see ``_RECIPES``).
    ``keep_path`` (REAP): npz with ``keep`` [n_sparse_layers, K] + ``dense_mlp_idx`` ->
    prune each MoE layer to its K kept experts and set ``n_routed_experts=K`` in config.
    Processes one source shard at a time; never holds the whole model in RAM.
    """
    os.makedirs(dst, exist_ok=True)
    keep = dmi = new_ne = None
    if keep_path is not None:
        import numpy as np
        kd = np.load(keep_path)
        keep = kd["keep"]; dmi = int(kd["dense_mlp_idx"]); new_ne = int(kd["K"])
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    weight_map = index["weight_map"]

    shard_to_names: dict[str, list[str]] = {}
    for n, s in weight_map.items():
        shard_to_names.setdefault(s, []).append(n)

    out_index: dict[str, str] = {}
    buffer: dict[str, mx.array] = {}
    buffer_bytes = 0
    out_shard_id = 0
    total_out_shards_placeholder = "{:05d}"

    def flush(final=False):
        nonlocal buffer, buffer_bytes, out_shard_id
        if not buffer:
            return
        out_shard_id += 1
        fname = f"model-{total_out_shards_placeholder.format(out_shard_id)}.safetensors"
        mx.save_safetensors(os.path.join(dst, fname), buffer, metadata={"format": "mlx"})
        for k in buffer:
            out_index[k] = fname
        buffer = {}
        buffer_bytes = 0

    for shard in sorted(shard_to_names):
        path = os.path.join(src, shard)
        tensors = mx.load(path)  # mmap
        for name in shard_to_names[shard]:
            w = tensors[name]
            for out_name, arr in _process_tensor(name, w, bits, group_size, out_dtype, recipe, keep, dmi):
                mx.eval(arr)
                buffer[out_name] = arr
                buffer_bytes += arr.nbytes
                if buffer_bytes >= _SHARD_CAP_BYTES:
                    flush()
        del tensors
    flush(final=True)

    # rename shards with correct total, build index.json
    _finalize_index(dst, out_index, out_shard_id)
    _write_config(src, dst, bits, group_size, recipe, new_ne)
    _copy_aux(src, dst)
    return dst


def _finalize_index(dst, out_index, n_shards):
    # rewrite shard filenames to model-XXXXX-of-YYYYY.safetensors
    remap = {}
    for i in range(1, n_shards + 1):
        old = f"model-{i:05d}.safetensors"
        new = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        if old != new and os.path.exists(os.path.join(dst, old)):
            os.rename(os.path.join(dst, old), os.path.join(dst, new))
        remap[old] = new
    weight_map = {k: remap[v] for k, v in out_index.items()}
    total = sum(os.path.getsize(os.path.join(dst, f)) for f in set(weight_map.values()))
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)


def _write_config(src, dst, bits, group_size, recipe="uniform", new_ne=None):
    cfg = json.load(open(os.path.join(src, "config.json")))
    if bits is not None:
        cfg["quantization"] = {"group_size": group_size, "bits": bits, "recipe": recipe}
    if new_ne is not None:                       # REAP: fewer routed experts
        cfg["text_config"]["n_routed_experts"] = new_ne
        cfg.setdefault("reap", {})["kept_experts"] = new_ne
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def _copy_aux(src, dst):
    for pat in ("tokenizer*", "special_tokens_map.json", "*.tiktoken", "tiktoken",
                "chat_template.jinja", "processor_config.json"):
        for p in glob.glob(os.path.join(src, pat)):
            base = os.path.basename(p)
            target = os.path.join(dst, base)
            if os.path.isdir(p):
                shutil.copytree(p, target, dirs_exist_ok=True)
            else:
                shutil.copy2(p, target)
