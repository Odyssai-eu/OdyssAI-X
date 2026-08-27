"""Do the converter's keys match what the loader expects?

Builds the real model structure (truncated to the 2 layers the probe shards
contain), runs the loader's sanitize over the converted weights, and diffs the
key sets. A mismatch here is what a 1.4 TB load would discover the hard way.
"""
import json
import sys

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models.kimi_k3 import Model, ModelArgs

DST = sys.argv[1]

cfg = json.load(open(f"{DST}/config.json"))
cfg["num_hidden_layers"] = 2  # shards 1-2 hold layers 0 and 1
args = ModelArgs.from_dict(cfg)
model = Model(args)

weights = {}
for shard in ("model-00001-of-000096.safetensors", "model-00002-of-000096.safetensors"):
    weights.update(mx.load(f"{DST}/{shard}"))
print(f"converted keys loaded: {len(weights)}")

weights = model.sanitize(weights)

# Quantize exactly the way mlx_lm.utils.load_model does, so the parameter names
# include .scales/.biases where the checkpoint has them.
import mlx.nn as nn

q = cfg["quantization"]


def class_predicate(p, m):
    if p in q:
        return q[p]
    if not hasattr(m, "to_quantized"):
        return False
    return f"{p}.scales" in weights


nn.quantize(
    model,
    group_size=q["group_size"],
    bits=q["bits"],
    mode=q.get("mode", "affine"),
    class_predicate=class_predicate,
)

expected = {k for k, _ in tree_flatten(model.parameters())}
got = set(weights)

# Compare only the decoder layers these shards actually carry. The model-level
# tensors (embed_tokens, lm_head, model.norm, output_attn_res_*) live in the
# tail shards, which are not downloaded yet — checked separately once they are.
PREFIXES = ("model.layers.0.", "model.layers.1.")
scope = {k for k in expected if k.startswith(PREFIXES)}
got_scope = {k for k in got if k.startswith(PREFIXES)}
print("hors perimetre (tenseurs de queue, pas encore telecharges):",
      sorted(k for k in expected if not k.startswith(PREFIXES)))

missing = sorted(scope - got_scope)
extra = sorted(got_scope - scope)

print(f"\nmodel expects (layers 0-1 + embed): {len(scope)}")
print(f"checkpoint provides:                {len(got_scope)}")
print(f"\nMANQUANTS ({len(missing)}):")
for k in missing[:25]:
    print(f"  {k}  {tuple(dict(tree_flatten(model.parameters()))[k].shape)}")
print(f"\nEN TROP ({len(extra)}):")
for k in extra[:25]:
    print(f"  {k}  {tuple(weights[k].shape)}")

# Shapes must agree too, not just names.
params = dict(tree_flatten(model.parameters()))
bad = [
    (k, tuple(params[k].shape), tuple(weights[k].shape))
    for k in sorted(scope & got_scope)
    if tuple(params[k].shape) != tuple(weights[k].shape)
]
print(f"\nFORMES DIVERGENTES ({len(bad)}):")
for k, want, have in bad[:25]:
    print(f"  {k}: modele {want} != checkpoint {have}")

ok = not missing and not extra and not bad
print("\n" + ("NOMS ET FORMES: OK" if ok else "NOMS ET FORMES: ECHEC"))
sys.exit(0 if ok else 1)
