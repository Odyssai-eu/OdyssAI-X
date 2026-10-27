#!/usr/bin/env python3
"""The step5_vl tensor split reproduces the single-node forward (vlm_runner).

A tiny step5_vl (the mlx-vlm PR #2539 test case: 4 q heads over 2 KV groups,
one MoE layer, one sparse full-attention layer) is built with the same seed on
every rank. Each rank runs the unsplit model, then splits a copy with
vlm_runner._shard_lm_step5_tensor and runs it again: prefill and three decode
steps, with prompts long enough that the sparse indexer selects regions
(region 2, topk 2). The logits must match, quantized (8 bits) and not. The
region selection is exercised directly too: the mask a rank uses must be the
slice of the single-node mask for its heads.

Needs an mlx-vlm venv with step3p7 (0.7.3) and two local ranks:

    ~/.venvs/mlx-vlm/bin/mlx.launch --backend ring -n 2 \\
        scripts/test_vlm_step5_tensor.py
"""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx_vlm.models as _models  # noqa: E402

_models.__path__.append(os.path.join(HERE, "mlx_vlm_models"))
from mlx_vlm.models.step5_vl import Model, ModelConfig  # noqa: E402

import vlm_runner  # noqa: E402

group = mx.distributed.init()
N, RANK = group.size(), group.rank()
FAILS = []


def check(label, ok, detail=""):
    if RANK == 0:
        print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": {detail}"))
    if not ok:
        FAILS.append(label)


CONFIG = {
    "model_type": "step5_vl",
    "text_config": {
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_attention_groups": 2, "head_dim": 16,
        "vocab_size": 256, "moe_num_experts": 4, "moe_top_k": 2,
        "moe_intermediate_size": 64, "share_expert_dim": 64, "moe_layer_list": [1],
        "layer_types": ["sliding_attention", "full_attention"], "sliding_window": 4,
        "sparse_config": {"enabled": True, "proxy_dim": 16, "sparse_indexer_num_heads": 4,
                          "sparse_indexer_rope_dim": 8, "region_block_size": 2, "topk": 2},
    },
    "vision_config": {"width": 32, "layers": 1, "heads": 4, "image_size": 28,
                      "patch_size": 14, "mlp_ratio": 2},
}


def build(quantize):
    mx.random.seed(7)
    m = Model(ModelConfig.from_dict(copy.deepcopy(CONFIG)))
    # Non-trivial router bias and norms, so a wrong slice cannot hide.
    for layer in m.language_model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.router_bias = mx.random.normal((4,)) * 0.1
    if quantize:
        nn.quantize(m.language_model, group_size=32, bits=8,
                    class_predicate=lambda p, mod: hasattr(mod, "to_quantized")
                    and not p.endswith(("gate.gate", "sparse_indexer_w", "g_proj"))
                    and mod.weight.shape[-1] % 32 == 0)
    mx.eval(m.parameters())
    return m


def run(lm, tokens, steps=3):
    cache = lm.make_cache()
    out = [lm(tokens[None], cache=cache).logits[0, -1]]
    nxt = mx.argmax(out[-1])[None]
    for _ in range(steps):
        out.append(lm(nxt[None], cache=cache).logits[0, -1])
        nxt = mx.argmax(out[-1])[None]
    return mx.stack(out)


tokens = mx.random.randint(0, 256, (21,), key=mx.random.key(3))
for quantize in (False, True):
    tag = "q8" if quantize else "bf"
    ref_model = build(quantize)
    ref = run(ref_model.language_model, tokens)
    tp_model = build(quantize)
    vlm_runner._shard_lm_step5_tensor(tp_model.language_model, group)
    mx.eval(tp_model.parameters())
    got = run(tp_model.language_model, tokens)
    err = mx.max(mx.abs(got - ref)).item()
    scale = mx.max(mx.abs(ref)).item()
    check(f"{tag}: prefill + 3 decode logits match one node (max err {err:.2e} of {scale:.2f})",
          err <= 2e-2 * max(scale, 1.0), f"err {err} scale {scale}")
    check(f"{tag}: same greedy tokens", bool(mx.all(mx.argmax(got, -1) == mx.argmax(ref, -1)).item()))

# The rank's region mask is the single-node mask cut to its heads, and the
# selection is genuinely sparse at this length.
ref_model = build(False)
tp_model = build(False)
vlm_runner._shard_lm_step5_tensor(tp_model.language_model, group)
la_ref = ref_model.language_model.model.layers[1].self_attn
la_tp = tp_model.language_model.model.layers[1].self_attn
x = mx.random.normal((1, 21, 64), key=mx.random.key(5)).astype(mx.bfloat16)
full = la_ref._sparse_mask(x, None, 0)
mine = la_tp._sparse_mask(x, None, 0)
per = full.shape[1] // N
check("region selection is sparse at 21 tokens", full.ndim == 4 and not bool(mx.all(full == mx.tril(mx.ones(full.shape[-2:], dtype=mx.bool_))).item()))
check("rank mask = single-node mask for its heads",
      mine.shape[1] == per and bool(mx.all(mine == full[:, RANK * per:(RANK + 1) * per]).item()))
check("local head counts restored after the mask", (la_tp.n_heads, la_tp.n_kv_heads) == (4 // N, 2 // N))

if RANK == 0:
    print()
    print(f"{len(FAILS)} FAILED: {FAILS}" if FAILS else "all checks passed")
sys.exit(1 if FAILS else 0)
