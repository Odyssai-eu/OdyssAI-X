"""vLLM-SR d3 package (Qwen3.5 VL backbone + 255-way code readout) -> MLX folder for decision_serve (backend d3).

The checkpoint is a bare Qwen3_5Model: keys are ``language_model.*`` / ``visual.*`` (no ``model.``
prefix, which mlx-vlm's sanitize_key expects) and there is no lm_head (tie_word_embeddings false);
decisions come from readout.safetensors applied to the last hidden state. So the keys get their
``model.`` prefix back, the lm_head module mlx-vlm builds for untied configs is dropped, and the
load stays strict: any other missing or unexpected key aborts the conversion.

The (1 + w) RMSNorm scales (input / post-attention / q / k / final norms) are stored exact in FP32, computed from
the source: mlx-vlm adds the 1 in bf16, which rounds the scale (up to 1/64 on values in [2, 4)) and moved d3-nano's
probabilities 2.4x further from the FP32 reference than upstream's own bf16 runtime. d3_mlx applies them as
transformers does (FP32 scale, output cast back to the input dtype).

usage: python convert_d3.py <source package> <out dir> [q8 | bf16 | norms]   (norms: rewrite the norms of an existing out dir)
"""
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
mx.set_default_device(mx.cpu)   # CPU: Metal GPU-timeout on lazy save (seen 2026-10-04)
import mlx.nn as nn
from mlx_vlm import convert
from mlx_vlm.models.qwen3_5 import qwen3_5 as Q

SRC = sys.argv[1]
OUT = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "q8"
assert MODE in ("q8", "bf16", "norms"), f"unknown mode {MODE}"
NORMS = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".self_attn.q_norm.weight",
         ".self_attn.k_norm.weight", "language_model.model.norm.weight")

_sanitize = Q.Model.sanitize


def sanitize(self, weights):
    # Without a model.safetensors.index.json mlx-vlm globs *.safetensors, readout.safetensors included: its
    # single "weight" is the decision readout, kept as its own file, not a backbone parameter.
    readout = weights.pop("weight", None)
    if readout is not None:
        assert readout.shape == (255, self.config.text_config.hidden_size), f"unexpected weight {readout.shape}"
    weights = {
        ("model." + k if k.startswith(("language_model.", "visual.")) else k): v
        for k, v in weights.items()
    }
    return _sanitize(self, weights)


def load_weights(self, file_or_weights, strict=True):
    self.language_model.pop("lm_head", None)
    return nn.Module.load_weights(self, file_or_weights, strict=True)


Q.Model.sanitize = sanitize
Q.Model.load_weights = load_weights


def pred(path, module):
    p = path.lower()
    if any(t in p for t in ("visual", "vision", "vl_connector", "vision_tower")):
        return False   # vision bf16
    return {"group_size": 64, "bits": 8}   # body + embed 8-bit gs64


if MODE == "q8":
    convert(hf_path=SRC, mlx_path=OUT, quantize=True, q_bits=8, q_group_size=64, quant_predicate=pred)
elif MODE == "bf16":
    convert(hf_path=SRC, mlx_path=OUT)

src, out = Path(SRC), Path(OUT)
for name in ("readout.safetensors", "README.md", "LICENSE", "NOTICE", "chat_template.jinja", "merges.txt"):
    if (src / name).exists() and MODE != "norms":
        shutil.copy2(src / name, out / name)


def weight_map(root):
    index = root / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    return {k: f.name for f in root.glob("model*.safetensors") for k in mx.load(str(f))}


si, oi = weight_map(src), weight_map(out)
assert not any("lm_head" in k for k in oi), "lm_head saved"
assert any(k.startswith("vision_tower.") for k in oi), "vision tower missing"


def tensor(root, index, key):
    return mx.load(str(root / index[key]))[key]


def source_key(key):
    return key.replace("language_model.model.", "language_model.", 1)


# Exact (1 + w) norm scales in FP32, shard by shard; mlx-vlm's bf16 value must be their bf16 rounding (or, on a
# rerun, already exact).
rewritten = 0
for shard in sorted(set(oi.values())):
    weights, meta = mx.load(str(out / shard), return_metadata=True)
    keys = [k for k in weights if k.endswith(NORMS)]
    if not keys:
        continue
    for k in keys:
        exact = 1 + tensor(src, si, source_key(k)).astype(mx.float32)
        old = weights[k]
        assert mx.array_equal(old, exact.astype(mx.bfloat16) if old.dtype == mx.bfloat16 else exact).item(), \
            f"{k}: stored scale is not 1 + source weight"
        weights[k] = exact
    mx.save_safetensors(str(out / shard), weights, metadata=meta or {"format": "mlx"})
    rewritten += len(keys)
assert rewritten, "no norm scale found"

# Values against the source: norms exact (1 + w, FP32) and embed rows (8-bit or exact).
for k in (k for k in oi if k.endswith(NORMS)):
    assert mx.array_equal(tensor(out, oi, k), 1 + tensor(src, si, source_key(k)).astype(mx.float32)).item(), k
QUANTIZE = any(k.endswith("embed_tokens.scales") for k in oi)
e = tensor(src, si, "language_model.embed_tokens.weight")[:1024].astype(mx.float32)
p = "language_model.model.embed_tokens."
if QUANTIZE:
    w = mx.dequantize(
        *(tensor(out, oi, p + n)[:1024] for n in ("weight", "scales", "biases")), group_size=64, bits=8
    ).astype(mx.float32)
else:
    w = tensor(out, oi, p + "weight")[:1024].astype(mx.float32)
rel = (mx.linalg.norm(w - e) / mx.linalg.norm(e)).item()
assert rel < (5e-2 if QUANTIZE else 1e-6), f"embed rel err {rel}"   # 8-bit gs64 noise is ~1e-2; a key mismatch is ~1
print(f"values OK: {len(oi)} tensors, {rewritten} norm scales exact FP32, embed rel err {rel:.2e}", flush=True)
print("D3 CONVERT DONE", flush=True)
