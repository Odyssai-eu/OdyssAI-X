#!/usr/bin/env python3
"""A refused load says the real cause: the node count, or where the model is.

2026-11-04: Ling-3.1 on a cluster without it answered "taille 0, à synchroniser"
while it sat on four other nodes, and on Argo (3 x 256) a budget gap with no word
of the 4 nodes it takes. No node is reached.

    python3 scripts/test_preflight_refusal_reason.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preflight  # noqa: E402

GB = 1024 ** 3
FAILS = []
LING = {"model_type": "bailing_hybrid", "num_key_value_heads": 64, "num_hidden_layers": 72}


def check(label, ok, got=""):
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": {got}"))
    if not ok:
        FAILS.append(label)


def cap(nodes, per_rank_gb):
    return {str(n): {"max_loadable_bytes": int(n * per_rank_gb * GB),
                     "max_loadable_pipeline_bytes": int(n * per_rank_gb * GB)}
            for n in range(1, nodes + 1)}


def verdict(config, size_gb, nodes, per_rank_gb, **kw):
    return preflight.evaluate(config=config, size_bytes=int(size_gb * GB),
                              capacity_by_nodes=cap(nodes, per_rank_gb),
                              max_nodes=nodes, **kw)


# Argo 3 x 256: says 4 nodes, the head rule and the memory gap.
s = verdict(LING, 592.3, 3, 181.8)["summary"]
check("3 nodes: names the 4 it takes", "il faut 4 nodes" in s and "en a 3" in s, s)
check("3 nodes: names the head rule", "doit diviser 64" in s, s)
check("3 nodes: names the memory gap", "592.3 GB pour 545.4 GB" in s, s)

# Fits in memory, only the split is wrong.
s = verdict(LING, 400, 3, 181.8)["summary"]
check("fits in memory: 4 nodes, no budget line", "il faut 4 nodes" in s and "de budget" not in s, s)

# Pipeline-capable: memory alone sets the count.
s = verdict({"model_type": "glm4_moe", "num_key_value_heads": 64}, 592.3, 3, 100)["summary"]
check("pipeline: 6 nodes, no head rule", "il faut 6 nodes" in s and "têtes" not in s, s)

# No count works: 7 heads, memory needs 2 nodes or more on 7 only.
s = verdict({"model_type": "qwen2", "num_key_value_heads": 7}, 5000, 3, 100)["summary"]
check("no count works: says so", "aucun nombre de nodes" in s, s)

# Absent from this cluster, present elsewhere: where it is, one line.
r = verdict({}, 0, 1, 100, target_node="ultra-256d", present_on=["ultra-256a", "ultra-512"])
s = r["summary"]
check("absent: says where it is", "absent de ultra-256d" in s and "ultra-256a, ultra-512" in s, s)
check("absent: one cause, one line", len(r["blockers"]) == 1, r["blockers"])
check("absent: no old wording", "taille 0" not in s and "synchroniser" not in s, s)

# On that node per the matrix, yet unreadable: the volume, not a sync.
s = verdict({}, 0, 1, 100, target_node="a", present_on=["a"])["summary"]
check("present but unreadable: volume", "volume" in s and "illisible" in s, s)

# Nowhere, and matrix unavailable.
s = verdict({}, 0, 1, 100, target_node="a", present_on=[])["summary"]
check("nowhere: to download", "aucun node" in s, s)
s = verdict({}, 0, 1, 100, target_node="a", present_on=None)["summary"]
check("matrix unknown: both causes", "absent, ou volume" in s, s)

# A model that fits still plans as before.
r = verdict(LING, 592.3, 4, 181.8)
check("4 nodes: OK", r["ok"] and r["plan"]["nodes"] == 4, r["summary"])

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
