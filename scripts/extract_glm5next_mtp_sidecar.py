"""Extract the GLM-5.3-Flash native MTP head (checkpoint layer index ==
num_hidden_layers) into a module-layout sidecar the NativeMTPModule (glm5_next
family) loads STRICT.

GLM-5.3-Flash ships a DeepSeek-V3-lineage MTP head at layer 45
(`model.language_model.layers.45.*`): enorm / hnorm / eh_proj + one
sparse-DSA decoder block (NoPE MLA + Glm5NextIndexer + DeepseekV32MoE, 288
experts) + shared_head.norm. Our Q8h16/Q6h16 trunk conversions DROP it
(deepseek_v32.sanitize trims layers >= num_hidden_layers), so the head must be
recovered from the original bf16 checkpoint.

Absorption is NOT hand-rolled: the raw checkpoint ships the FUSED `kv_b_proj`
and per-expert MLPs, but Glm5NextSparseAttention consumes the ABSORBED
embed_q/unembed_out (MultiLinear) and DeepseekV32MoE the STACKED switch_mlp.
We reuse glm5_next's own `Model.sanitize` (which delegates to
deepseek_v32.Model.sanitize for MLA absorption + expert stacking) by renaming
the layer-45 keys to a 1-layer model (layer 0, sparse) and running sanitize,
then remap the sanitized `model.layers.0.*` onto the module tree:

  enorm.* / hnorm.* / eh_proj.*                     (top-level scaffold)
  shared_head_norm.*                                (<- shared_head.norm.*)
  mtp_block.self_attn.*  (embed_q/unembed_out absorbed, indexer passthrough)
  mtp_block.mlp.*        (experts stacked into switch_mlp)
  mtp_block.input_layernorm.* / post_attention_layernorm.*

`shared_head.head.*` and `embed_tokens.*` are NOT present at layer 45 (shared
with the trunk lm_head / embed) — nothing to strip, but guarded anyway.

Output (REAL files, no symlinks) at each trunk's:
  <trunk>/mtp-sidecar/mtp-sidecar.safetensors

Usage (on a node with the shared /Volumes/models and mlx_lm.models.glm5_next):
  python3 extract_glm5next_mtp_sidecar.py
    [--bf16 <checkpoint>] [--out <trunk_dir> ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

DEFAULT_BF16 = "/Volumes/models/mlx/safe/zai-org/GLM-5.3-Flash-BF16"
DEFAULT_OUTS = [
    "/Volumes/models/odysseus/odyssai/GLM-5.3-Flash-Q8h16",
    "/Volumes/models/odysseus/odyssai/GLM-5.3-Flash-Q6h16",
]


def _log(m: str) -> None:
    sys.stderr.write(f"[extract-glm5next] {m}\n")
    sys.stderr.flush()


def extract(bf16_dir: Path) -> dict[str, mx.array]:
    config = json.load(open(bf16_dir / "config.json"))
    tc = config.get("text_config", config)
    n = int(tc["num_hidden_layers"])                 # 45 -> MTP head at layer n
    src_prefix = f"model.language_model.layers.{n}."
    _log(f"MTP head at checkpoint layer {n} (prefix {src_prefix})")

    # 1. Pull ONLY the layer-n tensors (lazy-load: filtering keys avoids
    #    materializing the rest of each shard).
    idx = json.load(open(bf16_dir / "model.safetensors.index.json"))["weight_map"]
    shards = sorted({idx[k] for k in idx if k.startswith(src_prefix)})
    raw: dict[str, mx.array] = {}
    for sh in shards:
        for k, v in mx.load(str(bf16_dir / sh)).items():
            if k.startswith(src_prefix):
                raw[k] = v
    if not raw:
        raise RuntimeError(f"no {src_prefix}* tensors found under {bf16_dir}")
    _log(f"loaded {len(raw)} raw layer-{n} tensors from {len(shards)} shard(s)")

    # 2. Rename layer-n -> layer-0 (a 1-layer glm5_next model): this makes the
    #    head survive sanitize's `layers >= num_hidden_layers` trim and lets the
    #    absorption/stacking loops (which iterate range(num_hidden_layers)=[0])
    #    process it. Strip the VLM `language_model.` container while we are here.
    renamed = {"model.layers.0." + k[len(src_prefix):]: v for k, v in raw.items()}

    # 3. Build a minimal 1-layer glm5_next Model whose layer 0 is a sparse-DSA
    #    MoE layer, and run its own sanitize (== deepseek_v32.sanitize for the
    #    MLA absorption + expert stacking, plus the fp8 dequant which is a no-op
    #    on this bf16 checkpoint).
    from mlx_lm.models.glm5_next import Model, ModelArgs
    mini = dict(tc)
    mini["num_hidden_layers"] = 1
    mini["layer_types"] = ["full_attention"]
    mini["mlp_layer_types"] = ["sparse"]
    mini["first_k_dense_replace"] = 0
    mini["model_type"] = config.get("model_type", "glm5_next")
    args = ModelArgs.from_dict(mini)
    model = Model(args)
    _log("built 1-layer glm5_next stub; running Model.sanitize (absorb+stack)")
    sanitized = model.sanitize(renamed)

    # 4. Remap sanitized `model.layers.0.*` onto the NativeMTPModule tree.
    P = "model.layers.0."
    out: dict[str, mx.array] = {}
    skipped = []
    for k, v in sanitized.items():
        if not k.startswith(P):
            skipped.append(k)                        # model.norm/embed/lm_head — not from our input
            continue
        s = k[len(P):]
        if s.startswith("shared_head.head.") or s.startswith("embed_tokens."):
            skipped.append(k)                        # shared with the trunk
            continue
        if s.startswith("shared_head.norm."):
            out["shared_head_norm." + s[len("shared_head.norm."):]] = v
        elif s.split(".", 1)[0] in ("enorm", "hnorm", "eh_proj"):
            out[s] = v
        else:
            out["mtp_block." + s] = v
    if skipped:
        _log(f"skipped {len(skipped)} shared/foreign key(s): {sorted(skipped)[:4]}")

    # No `experts.N` must survive (would mean stacking failed).
    leftover_experts = [k for k in out if ".experts." in k]
    if leftover_experts:
        raise RuntimeError(f"expert stacking failed — leftover: {leftover_experts[:3]}")
    if "mtp_block.self_attn.embed_q.weight" not in out:
        raise RuntimeError("kv_b absorption failed — no embed_q.weight")

    mx.eval(list(out.values()))
    _log(f"produced {len(out)} module-tree tensors")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", default=DEFAULT_BF16)
    ap.add_argument("--out", action="append", default=None,
                    help="trunk dir(s) to write <dir>/mtp-sidecar/. Repeatable.")
    a = ap.parse_args()
    outs = a.out or DEFAULT_OUTS

    weights = extract(Path(a.bf16))

    first_file: Path | None = None
    for d in outs:
        od = Path(d) / "mtp-sidecar"
        od.mkdir(parents=True, exist_ok=True)
        dst = od / "mtp-sidecar.safetensors"
        if first_file is None:
            mx.save_safetensors(str(dst), weights)
            first_file = dst
        else:
            shutil.copyfile(first_file, dst)          # REAL file copy, never a symlink
        gb = sum(v.nbytes for v in weights.values()) / 1e9
        _log(f"wrote {dst}  ({len(weights)} tensors, {gb:.2f} GB bf16)")

    # Report the tree so a human can eyeball the shapes.
    _log("sample keys:")
    for k in sorted(weights)[:6]:
        _log(f"    {k}  {tuple(weights[k].shape)}")
    _log("... switch_mlp / indexer:")
    for k in sorted(weights):
        if "switch_mlp" in k or "indexer" in k or k.endswith("embed_q.weight") \
                or k.endswith("unembed_out.weight") or k == "eh_proj.weight":
            _log(f"    {k}  {tuple(weights[k].shape)}")


if __name__ == "__main__":
    main()
