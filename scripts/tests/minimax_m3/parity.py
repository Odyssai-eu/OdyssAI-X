#!/usr/bin/env python3
"""Parité MLX ↔ référence torch sur le tiny M3 (nommage hub de bout en bout)."""
import json
import sys

import numpy as np

sys.path.insert(0, "/tmp/m3-night")
import mlx.core as mx
import minimax_m3
mx.set_default_device(mx.cpu)  # le matmul Metal fp32 a un plancher ~1e-3 ; l'exactitude se juge sur CPU

OUT = "/tmp/m3-night/tiny-hub"
cfg = json.load(open(f"{OUT}/config.json"))
args = minimax_m3.ModelArgs.from_dict(cfg)
print("layer_types:", args.layer_types)
print("mlp_layer_types:", args.mlp_layer_types)

model = minimax_m3.Model(args)
weights = mx.load(f"{OUT}/model.safetensors")
weights = model.sanitize(weights)
model.load_weights(list(weights.items()), strict=True)
mx.eval(model.parameters())

ref = np.load("/tmp/m3-night/ref_outputs.npz")
S = 24
tokens = mx.arange(S)[None] % cfg["vocab_size"]

def cmp(name, a_mx, b_np, atol=2e-4):
    a = np.array(a_mx.astype(mx.float32))
    d = np.abs(a - b_np).max()
    c = np.corrcoef(a.ravel(), b_np.ravel())[0, 1]
    flag = "OK " if d < atol else "FAIL"
    print(f"[{flag}] {name:24s} max|Δ|={d:.3e} corr={c:.8f}")
    return d < atol

ok = True

# — prefill complet —
logits = model(tokens)
mx.eval(logits)
ok &= cmp("logits prefill", logits, ref["logits_prefill"], atol=5e-4)

# — décode caché : prefill 20 + 4 pas —
cache = model.make_cache()
pre = model(tokens[:, :20], cache=cache)
mx.eval(pre)
steps = [np.array(pre[:, -1].astype(mx.float32))]
for i in range(20, 24):
    stp = model(tokens[:, i:i + 1], cache=cache)
    mx.eval(stp)
    steps.append(np.array(stp[:, -1].astype(mx.float32)))
dec = np.concatenate(steps, axis=0)
d = np.abs(dec - ref["logits_decode"]).max()
c = np.corrcoef(dec.ravel(), ref["logits_decode"].ravel())[0, 1]
flag = "OK " if d < 5e-4 else "FAIL"
print(f"[{flag}] logits décode (cache)    max|Δ|={d:.3e} corr={c:.8f}")
ok &= d < 5e-4

# — cohérence prefill-vs-décode interne (le pas 20..23 doit égaler le prefill) —
internal = np.abs(dec[1:] - np.array(logits[0, 20:24].astype(mx.float32))).max()
print(f"[{'OK ' if internal < 5e-4 else 'FAIL'}] décode≡prefill interne   max|Δ|={internal:.3e}")

print("\nVERDICT:", "PARITÉ" if ok else "DIVERGENCE")
sys.exit(0 if ok else 1)
