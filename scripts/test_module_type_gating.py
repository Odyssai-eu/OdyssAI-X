#!/usr/bin/env python3
"""Tensor / pipeline gates key on the mlx-lm MODULE, not the config model_type.

Ling-3.1-flash says model_type bailing_hybrid like Ring-2.5 / Ling-2.6, but
mlx-lm loads it with bailing_moe_v3 (architecture BailingMoeV3ForCausalLM),
whose overlay shard() splits it 4 ways. Ring-2.5 stays on bailing_hybrid, which
has no shard(). bailing_moe_v3 has no pipeline path. No node is reached.

    .venv/bin/python scripts/test_module_type_gating.py
"""
import asyncio
import json
import os
import sys
import tempfile
import types

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


LING31 = {"model_type": "bailing_hybrid", "architectures": ["BailingMoeV3ForCausalLM"],
          "num_hidden_layers": 72, "num_key_value_heads": 64}
RING25 = {"model_type": "bailing_hybrid", "architectures": ["BailingMoeV2_5ForCausalLM"],
          "num_hidden_layers": 80, "num_key_value_heads": 8}
QWEN3 = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
         "num_hidden_layers": 48, "num_key_value_heads": 8}

check("Ling-3.1 -> bailing_moe_v3", api._mlx_module_type(LING31), "bailing_moe_v3")
check("Ring-2.5 keeps bailing_hybrid", api._mlx_module_type(RING25), "bailing_hybrid")
check("qwen3 keeps qwen3", api._mlx_module_type(QWEN3), "qwen3")
check("architectures as a string", api._mlx_module_type(
    {"model_type": "bailing_hybrid", "architectures": "BailingMoeV3ForCausalLM"}), "bailing_moe_v3")
check("no architectures", api._mlx_module_type({"model_type": "bailing_hybrid"}), "bailing_hybrid")


def arch_of(cfg):
    real_run = api.subprocess.run
    api.subprocess.run = lambda *a, **k: types.SimpleNamespace(stdout=json.dumps(cfg))
    try:
        return asyncio.run(api.get_model_arch_meta("admin@198.51.100.1", "/m/x"))
    finally:
        api.subprocess.run = real_run


ling, ring = arch_of(LING31), arch_of(RING25)
check("arch meta: Ling-3.1 module_type", ling["module_type"], "bailing_moe_v3")
check("arch meta: Ling-3.1 model_type unchanged", ling["model_type"], "bailing_hybrid")
check("arch meta: Ring-2.5 module_type", ring["module_type"], "bailing_hybrid")


def mode(arch, n, m):
    return api._validate_load_mode(arch["module_type"], arch["num_hidden_layers"],
                                   arch["num_key_value_heads"], n, m)[0]


check("Ling-3.1 tensor x4 allowed", mode(ling, 4, "tensor"), True)
check("Ling-3.1 tensor x2 allowed", mode(ling, 2, "tensor"), True)
check("Ling-3.1 pipeline x3 refused", mode(ling, 3, "pipeline"), False)
check("Ling-3.1 single node allowed", mode(ling, 1, "pipeline"), True)
check("Ring-2.5 tensor refused", mode(ring, 4, "tensor"), False)
check("Ring-2.5 pipeline allowed", mode(ring, 4, "pipeline"), True)
check("bailing_moe_v3 forces use_ap=False", "bailing_moe_v3" in api.FORCE_NO_AP_MODEL_TYPES, True)
check("bailing_hybrid not tensor-capable", "bailing_hybrid" in api.TENSOR_CAPABLE_MODEL_TYPES, False)

print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAIL'}")
sys.exit(1 if FAILS else 0)
