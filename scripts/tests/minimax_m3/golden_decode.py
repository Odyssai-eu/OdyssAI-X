#!/usr/bin/env python3
"""Golden test — DECODE block-gather (voie A exact) vs the dense-mask reference.

OdyssAI-X#53 phase 1. Proves the new decode block-gather path produces output
IDENTICAL (< 1e-6 on CPU fp32) to the legacy dense-mask path, token by token,
in the SPARSE regime (k_len > topk*block).

How the sparse regime is exercised
----------------------------------
The tiny config has index_block_size=4, index_topk_blocks=2, so the useful
regime threshold is topk*block = 8 keys (the prod analogue is 128*16 = 2048).
We prefill a short prompt then decode well past k_len=8, so the gather path is
actually engaged (k_len_idx > 8) and the indexer returns a strict SUBSET of
blocks. We assert at runtime that the gather path fired.

Method (exact apples-to-apples)
-------------------------------
Two model instances built from the SAME weights, fed the SAME token sequence
with independent caches:
  - GATHER : minimax_m3.DECODE_BLOCK_GATHER = True  (production path)
  - DENSE  : minimax_m3.DECODE_BLOCK_GATHER = False (legacy reference)
Any per-token logit difference is therefore purely gather-vs-dense math.
Judged on CPU (mx.cpu) because the Metal fp32 matmul has a ~1e-3 floor.

Run with both the fp32 tiny (tiny-hub) and the Q6 tiny (tiny-q6).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/tmp/m3-night")
import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)  # exactitude jugée sur CPU (plancher Metal fp32 ~1e-3)

import minimax_m3

TOL = 1e-6


def load_fp32(path):
    cfg = json.load(open(f"{path}/config.json"))
    args = minimax_m3.ModelArgs.from_dict(cfg)
    model = minimax_m3.Model(args)
    weights = mx.load(f"{path}/model.safetensors")
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model, cfg


def load_q6(path):
    cfg = json.load(open(f"{path}/config.json"))
    qc = cfg["quantization"]
    idx = json.load(open(f"{path}/model.safetensors.index.json"))
    weights = {}
    for shard in sorted(set(idx["weight_map"].values())):
        weights.update(mx.load(f"{path}/{shard}"))
    args = minimax_m3.ModelArgs.from_dict(cfg)
    model = minimax_m3.Model(args)
    wq = model.sanitize(weights)
    nn.quantize(
        model, group_size=qc["group_size"], bits=qc["bits"],
        class_predicate=lambda p, m: hasattr(m, "to_quantized") and f"{p}.scales" in wq,
    )
    model.load_weights(list(wq.items()), strict=True)
    mx.eval(model.parameters())
    return model, cfg


def instrument_gather_fired(model):
    """Wrap each sparse Attention.gather_decode (via the indexer) to count how
    many times the gather path actually fires. Returns a mutable [count] list."""
    counter = [0]
    for layer in model.model.layers:
        if layer is None:
            continue
        idxr = layer.self_attn.indexer
        if idxr is None:
            continue
        orig = idxr.gather_decode

        def make(orig_fn):
            def wrapped(*a, **kw):
                counter[0] += 1
                return orig_fn(*a, **kw)
            return wrapped

        idxr.gather_decode = make(orig)
    return counter


def run_variant(model, tokens, n_prefill, gather_on):
    """Prefill n_prefill tokens, then decode the rest one at a time. Returns the
    stacked per-step last-position logits [n_decode_steps, vocab] as fp32 numpy."""
    minimax_m3.DECODE_BLOCK_GATHER = gather_on
    cache = model.make_cache()
    out = model(tokens[:, :n_prefill], cache=cache)
    mx.eval(out)
    steps = []
    S = tokens.shape[1]
    for i in range(n_prefill, S):
        stp = model(tokens[:, i:i + 1], cache=cache)
        mx.eval(stp)
        steps.append(np.array(stp[:, -1].astype(mx.float32)))
    return np.concatenate(steps, axis=0)


def golden(name, model, cfg):
    V = cfg["vocab_size"]
    BLK = cfg["index_block_size"]
    TOPK = cfg["index_topk_blocks"]
    thresh = BLK * TOPK
    # prompt long enough that decode runs well past the threshold. With
    # n_prefill=6 (<= thresh so the first decode steps are still dense) and
    # decode out to S=40, k_len climbs 7..40 — crossing the threshold=8 and
    # spending most steps strictly in the sparse regime.
    S = 40
    n_prefill = 6
    tokens = (mx.arange(S)[None] * 7 + 3) % V  # deterministic, non-trivial

    counter = instrument_gather_fired(model)

    dense = run_variant(model, tokens, n_prefill, gather_on=False)
    n_dense_fired = counter[0]
    counter[0] = 0
    gather = run_variant(model, tokens, n_prefill, gather_on=True)
    n_gather_fired = counter[0]

    d = float(np.abs(dense - gather).max())
    n_sparse_steps = S - n_prefill - max(0, (thresh + 1) - (n_prefill + 1))
    ok = d < TOL
    flag = "OK " if ok else "FAIL"
    print(f"[{flag}] {name:14s} max|Δ| gather-vs-dense = {d:.3e}  (tol {TOL:.0e})")
    print(f"        threshold k_len>{thresh}; decode steps={S - n_prefill}; "
          f"gather fired {n_gather_fired}x (dense run fired {n_dense_fired}x)")
    # The gather must actually have engaged in the sparse layers, else the test
    # is vacuous. Sparse layers = 3 (layers 1,2,3). Steps with k_len>thresh.
    if n_gather_fired == 0:
        print("        !! gather NEVER fired — test vacuous, FAILING")
        ok = False
    if n_dense_fired != 0:
        print("        !! gather fired during the DENSE reference run — flag leak")
        ok = False
    return ok, d


if __name__ == "__main__":
    results = []
    print("=== golden: decode block-gather vs dense-mask (CPU fp32) ===\n")

    model, cfg = load_fp32("/tmp/m3-night/tiny-hub")
    ok_fp32, d_fp32 = golden("tiny-fp32", model, cfg)
    results.append(("tiny-fp32", ok_fp32))
    print()

    model, cfg = load_q6("/tmp/m3-night/tiny-q6")
    ok_q6, d_q6 = golden("tiny-q6", model, cfg)
    results.append(("tiny-q6", ok_q6))
    print()

    # restore default for any later import
    minimax_m3.DECODE_BLOCK_GATHER = True

    all_ok = all(r[1] for r in results)
    print("SUMMARY")
    for n, r in results:
        print(f"  {n:14s} {'PASS' if r else 'FAIL'}")
    print(f"\nGOLDEN fp32 max|Δ| = {d_fp32:.3e}")
    print("VERDICT:", "GOLDEN PASS (decode gather == dense, <1e-6)" if all_ok else "GOLDEN FAIL")
    sys.exit(0 if all_ok else 1)
