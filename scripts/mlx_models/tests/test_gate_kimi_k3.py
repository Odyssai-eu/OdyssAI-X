"""Gate check: does _validate_load_fits accept the real K3 numbers?

Exercises the arch-aware path against the flat-factor path, and the
wired-limits-fell-back-to-default case (the one a reboot produces).
"""
import sys
import types
from unittest import mock

sys.path.insert(0, "scripts")

# api.py pulls in the whole FastAPI app at import; stub what it needs from the
# environment so we can import it for a pure-function test.
import os
os.environ.setdefault("ODYSSEUS_SKIP_BOOT", "1")

import api  # noqa: E402

GIB = 1024**3
K3_SIZE = int(1390 * GIB)

ARCH_K3 = {
    "model_type": "kimi_k3",
    "num_hidden_layers": 93,
    "num_key_value_heads": 96,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "linear_attn_config": {
        "num_heads": 96,
        "head_dim": 128,
        "full_attn_layers": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52,
                             56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93],
    },
}


def fake_cluster(wired_gib):
    ram = [512, 256, 256, 256, 256]
    per_node = [
        {"host": f"n{i}", "ssh": "", "ram_bytes": int(r * GIB),
         "wired_limit_bytes": int(w * GIB), "from_telemetry": True}
        for i, (r, w) in enumerate(zip(ram, wired_gib))
    ]
    return sum(n["ram_bytes"] for n in per_node), per_node


def run(label, wired_gib, arch, expect_ok):
    with mock.patch.object(api, "_cluster_total_ram_bytes",
                           return_value=fake_cluster(wired_gib)):
        ok, reason, detail = api._validate_load_fits(
            K3_SIZE, "main", 5, mode="pipeline", arch=arch
        )
    flag = "OK " if ok == expect_ok else "FAIL"
    print(f"  [{flag}] {label}: fits={ok} (attendu {expect_ok})")
    print(f"         {reason}")
    if detail.get("arch_overhead"):
        print(f"         overhead: {detail['arch_overhead']}")
        print(f"         par rang: +{detail['arch_overhead_rank_gb']} GiB, "
              f"requis/rang {detail['per_rank_required_gb']} GB, "
              f"budget/rang {detail['per_node_budget_gb']} GB")
    return ok == expect_ok


print("K3 = 1390 GiB, 5 nodes, mode pipeline")
results = [
    run("wired poses (480 + 4x245), gate arch-aware", [480, 245, 245, 245, 245],
        ARCH_K3, True),
    run("wired retombes au defaut apres reboot (200)", [480, 200, 200, 200, 200],
        ARCH_K3, False),
    run("meme modele SANS arch (forfaits 1.15/1.10)", [480, 245, 245, 245, 245],
        None, False),
]
print()
print("TOUS OK" if all(results) else "ECHEC")
sys.exit(0 if all(results) else 1)
