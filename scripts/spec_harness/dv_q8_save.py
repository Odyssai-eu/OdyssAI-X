#!/Users/admin/mlx-cluster/.venv/bin/python
"""Materialise le drafter DSpark en checkpoint MLX quantise.

Sans ca, chaque run paie : (a) un pic de ~208 GB (la dequant FP8/FP4 -> bf16
cree de gros intermediaires), (b) le temps de dequant, (c) la dependance au
repo pipenetwork sur le node. Apres, le drafter se charge par un mx.load
direct de ~40 GB (Q8) sans aucune de ces trois choses."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/ivan-mlx-lm"))
sys.path.insert(0, os.path.expanduser("~/deepseek-v4-mlx"))
sys.path.insert(0, os.path.expanduser("~"))
import mlx.core as mx                                    # noqa: E402
import mlx.nn as nn                                      # noqa: E402
from mlx.utils import tree_flatten                       # noqa: E402
import mlx_lm.models.deepseek_v4 as V4                   # noqa: E402
from dv_d_drafter import V4DSparkDrafter, load_drafter_weights   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--drafter", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--bits", type=int, default=8)
args = ap.parse_args()

a = V4.ModelArgs.from_dict(json.load(open(os.path.join(args.target, "config.json"))))
dcfg = json.load(open(os.path.join(args.drafter, "config.json")))
d = V4DSparkDrafter(a, dcfg)
d.load_weights(list(load_drafter_weights(args.drafter, o_groups=a.o_groups).items()), strict=False)
mx.eval(d.parameters())
print(f"bf16 charge, peak {mx.get_peak_memory()/1e9:.1f} GB", flush=True)

nn.quantize(d, group_size=64, bits=args.bits)
mx.eval(d.parameters())
w = dict(tree_flatten(d.parameters()))
os.makedirs(args.out, exist_ok=True)
mx.save_safetensors(os.path.join(args.out, "model.safetensors"), w)
cfg = dict(dcfg)
cfg["quantization"] = {"group_size": 64, "bits": args.bits}
cfg["prequantized"] = True
json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
gb = sum(v.nbytes for v in w.values()) / 1e9
print(f"ecrit {args.out}: {len(w)} tenseurs, {gb:.1f} GB (Q{args.bits})")
