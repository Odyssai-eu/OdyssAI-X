#!/usr/bin/env python3
"""Edge checks for the decode block-gather threshold (OdyssAI-X#53 phase 1).

Confirms, on the real tiny model:
  1. At k_len == threshold (topk*block) the gather path does NOT fire (dense),
     and at k_len == threshold+1 it DOES fire — the strict '> 2048' boundary.
  2. Across the boundary the gather output equals the dense output < 1e-6
     (so even the first gather step right above the threshold is exact).
Judged on CPU fp32.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/tmp/m3-night")
import mlx.core as mx

mx.set_default_device(mx.cpu)
import minimax_m3

TOL = 1e-6

cfg = json.load(open("/tmp/m3-night/tiny-hub/config.json"))
args = minimax_m3.ModelArgs.from_dict(cfg)
model = minimax_m3.Model(args)
w = mx.load("/tmp/m3-night/tiny-hub/model.safetensors")
w = model.sanitize(w)
model.load_weights(list(w.items()), strict=True)
mx.eval(model.parameters())

V = cfg["vocab_size"]
BLK = cfg["index_block_size"]
TOPK = cfg["index_topk_blocks"]
thresh = BLK * TOPK  # 8

# Wrap gather_decode on every sparse layer to record the k_len it sees.
fired_klens = []
for layer in model.model.layers:
    if layer is None or layer.self_attn.indexer is None:
        continue
    idxr = layer.self_attn.indexer
    orig = idxr.gather_decode

    def make(o):
        def wrapped(inds, num_blocks, k_len, offset, k, v, n_repeat, dtype):
            fired_klens.append(k_len)
            return o(inds, num_blocks, k_len, offset, k, v, n_repeat, dtype)
        return wrapped
    idxr.gather_decode = make(orig)


def decode_seq(gather_on):
    minimax_m3.DECODE_BLOCK_GATHER = gather_on
    fired_klens.clear()
    cache = model.make_cache()
    # prefill exactly `thresh` tokens -> first decode step has k_len = thresh+1.
    # but we want to also OBSERVE a step at k_len == thresh: prefill thresh-1,
    # decode steps then run k_len = thresh, thresh+1, ...
    S = thresh + 6
    n_prefill = thresh - 1  # 7 -> first decode at k_len=8 (==thresh, dense)
    tokens = (mx.arange(S)[None] * 5 + 1) % V
    out = model(tokens[:, :n_prefill], cache=cache)
    mx.eval(out)
    steps = []
    for i in range(n_prefill, S):
        stp = model(tokens[:, i:i + 1], cache=cache)
        mx.eval(stp)
        steps.append(np.array(stp[:, -1].astype(mx.float32)))
    return np.concatenate(steps, axis=0), list(fired_klens)


dense, _ = decode_seq(False)
gather, klens = decode_seq(True)

d = float(np.abs(dense - gather).max())
# k_len the gather fired at (per sparse layer, deduped & sorted)
unique_klens = sorted(set(klens))
fired_at_thresh = thresh in unique_klens
fired_above = (thresh + 1) in unique_klens

print(f"threshold topk*block = {thresh}")
print(f"gather fired at k_len values: {unique_klens}")
print(f"  k_len == {thresh} (should be DENSE, gather NOT fired): "
      f"{'FAIL — fired!' if fired_at_thresh else 'OK — not fired'}")
print(f"  k_len == {thresh + 1} (should be GATHER, fired): "
      f"{'OK — fired' if fired_above else 'FAIL — not fired'}")
print(f"max|Δ| gather-vs-dense across boundary = {d:.3e} (tol {TOL:.0e}): "
      f"{'OK' if d < TOL else 'FAIL'}")

minimax_m3.DECODE_BLOCK_GATHER = True
ok = (not fired_at_thresh) and fired_above and (d < TOL)
print("\nVERDICT:", "EDGES PASS" if ok else "EDGES FAIL")
sys.exit(0 if ok else 1)
