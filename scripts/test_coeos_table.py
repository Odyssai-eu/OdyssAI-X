#!/usr/bin/env python3
"""Unit tests for the CoeOS self-ranking resolver (2026-07-14, Sophie: "il
prend la Data, pas le settings ; resolver dans CoeOS"). Pure config-dict
functions — no live pools, no topology, no state dir needed. Covers:
  1. operator pin always wins over the score_table proposal
  2. unpinned axis + score_table -> best-scoring model proposed
  3. tie on score -> cheaper model wins (None cost sorts last)
  4. role=reference rows (our benchmark etalons) are never proposed
  5. a table model absent from the operator's registry is ignored
  6. no score_table imported -> identical to the pre-change behaviour
     (unpinned axes stay unbound, exactly as before)
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    if got != want:
        FAILS.append(f"{label}: got {got!r}, want {want!r}")
    else:
        print(f"OK  {label}")


TABLE = {
    "format": "tmb-score-table/1",
    "models": {
        "aion3 OR": {
            "role": "contender", "or_id": "aion-labs/aion-3.0",
            "cost_per_test": 0.05,
            "axes": {"reasoning": {"score": 90.0, "n": 1, "verified": True},
                     "calc": {"score": 100.0, "n": 1, "verified": True}},
        },
        "nemotron3 super OR": {
            "role": "contender", "or_id": "nvidia/nemotron-3-ultra-550b-a55b",
            "cost_per_test": 0.0,
            "axes": {"reasoning": {"score": 90.0, "n": 1, "verified": True}},
        },
        "Fusion-REF": {
            "role": "reference", "or_id": "anthropic/claude-opus-4.8",
            "cost_per_test": None,
            "axes": {"reasoning": {"score": 100.0, "n": 1, "verified": True}},
        },
        "unmapped-model": {
            "role": "contender", "or_id": "some/unmapped",
            "cost_per_test": 0.001,
            "axes": {"reasoning": {"score": 99.9, "n": 1, "verified": True}},
        },
    },
}

REGISTRY = {
    "aion3": {"name": "aion3", "endpoint": "aion3-endpoint"},
    "nemotron3": {"name": "nemotron3", "endpoint": "nemotron3-endpoint"},
    # NB: no registry entry maps to "unmapped-model" — check 5.
}


def cfg_with(axes, score_table=None):
    return {"axes": axes, "models": REGISTRY,
            **({"score_table": score_table} if score_table is not None else {})}


# 1) pin wins over table, even though the table would score higher elsewhere.
c = cfg_with([{"key": "reasoning", "model": "pinned-model"}], TABLE)
check("1 pin wins", api._coeos_axis_models(c), {"reasoning": "pinned-model"})

# 2) unpinned axis + table -> best score among registry-joinable rows.
#    aion3 (90.0) and nemotron3 (90.0) tie in TABLE for "reasoning"; add a
#    non-tied axis to isolate the "pick the best" behaviour first.
TABLE2 = {"models": {
    "aion3 OR": {"role": "contender", "cost_per_test": 0.05,
                 "axes": {"calc": {"score": 80.0, "n": 1, "verified": True}}},
    "nemotron3 super OR": {"role": "contender", "cost_per_test": 0.0,
                           "axes": {"calc": {"score": 95.0, "n": 1, "verified": True}}},
}}
lut2 = api._coeos_table_lookup(cfg_with([], TABLE2))
check("2a table lookup joins by row name",
      sorted(lut2.keys()), sorted(["aion3 or", "nemotron3 super or"]))
c2 = cfg_with([{"key": "calc"}], TABLE2)
# join happens via the registry's LOGICAL NAME matching a table row name —
# here neither "aion3" nor "nemotron3" (registry keys) match "aion3 OR" /
# "nemotron3 super OR" (table row names) exactly, so nothing resolves.
# Re-key the registry to the table's own row names to exercise the real join.
c2b = {"axes": [{"key": "calc"}],
       "models": {"aion3 OR": {"name": "aion3", "endpoint": "e1"},
                  "nemotron3 super OR": {"name": "nemotron3", "endpoint": "e2"}},
       "score_table": TABLE2}
logical, row, score = api._coeos_propose(c2b, "calc")
check("2b best score proposed", (logical, score), ("nemotron3 super OR", 95.0))
check("2c axis_models fills unpinned axis via table",
      api._coeos_axis_models(c2b), {"calc": "nemotron3 super OR"})

# 3) tie on score -> cheaper wins (aion3=0.05 vs nemotron3=0.0 tie on "reasoning").
c3 = {"axes": [{"key": "reasoning"}],
      "models": {"aion3 OR": {"name": "aion3", "endpoint": "e1"},
                 "nemotron3 super OR": {"name": "nemotron3", "endpoint": "e2"}},
      "score_table": TABLE}
logical, row, score = api._coeos_propose(c3, "reasoning")
check("3 tie -> cheaper (free) wins", logical, "nemotron3 super OR")

# 4) reference-role rows are never proposed, even with the top score.
c4 = {"axes": [{"key": "reasoning"}],
      "models": {"aion3 OR": {"name": "aion3", "endpoint": "e1"},
                 "Fusion-REF": {"name": "fusion", "endpoint": "e2"}},
      "score_table": TABLE}
logical, row, score = api._coeos_propose(c4, "reasoning")
check("4 REF never proposed (aion3 wins over Fusion-REF's 100.0)", logical, "aion3 OR")

# 4b) the REAL-WORLD case (caught live on coeos-se .21:4600, 2026-07-14,
#     ported here): a settings generator SLUGIFIES display names into
#     registry keys ("aion3 OR" -> "aion3-or"), so the key essentially
#     never equals a table row's own name — only the registry entry's OWN
#     `endpoint` id does (mirroring the table's `or_id`). A join that only
#     tries the key (the original bug — see the 2a/2b workaround above,
#     which sidestepped this instead of catching it) resolves nothing.
c4b = {"axes": [{"key": "reasoning"}],
       "models": {"aion3-or": {"name": "aion3 OR", "endpoint": "aion-labs/aion-3.0"}},
       "score_table": TABLE}
logical, row, score = api._coeos_propose(c4b, "reasoning")
check("4b matches via registry endpoint when key is slugified", logical, "aion3-or")

# 5) a table model with no registry entry is ignored entirely.
c5 = {"axes": [{"key": "reasoning"}],
      "models": {},  # empty registry: nothing is "this operator's fleet"
      "score_table": TABLE}
logical, row, score = api._coeos_propose(c5, "reasoning")
check("5 empty registry -> nothing proposed", (logical, row, score), (None, None, None))
check("5b axis_models: unpinned axis stays unbound", api._coeos_axis_models(c5), {})

# 6) no score_table at all -> identical to pre-change behaviour: unpinned
#    axes are simply absent from the output dict (never a KeyError downstream
#    since coeos_resolve does axis_models.get(axis)).
c6 = {"axes": [{"key": "reasoning"}, {"key": "calc", "model": "pinned"}],
      "models": REGISTRY}
check("6 no table: unpinned dropped, pin kept (pre-change behaviour)",
      api._coeos_axis_models(c6), {"calc": "pinned"})

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print(" ", f)
    sys.exit(1)
print("All CoeOS self-ranking tests passed.")
