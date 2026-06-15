#!/usr/bin/env python3
"""Canari EOS-guard — découplage stop / répétition (rapport §9.4).

Prouve que `make_eos_guard_processors` (runner.py) épingle chaque token de stop
à l'écart EXACT qu'il avait sous le leader de contenu AVANT pénalité — donc la
`repetition_penalty` ne peut plus promouvoir l'EOS « par effet de bord » (le bug
Bruit-Blanc : « fréquence » pénalisé sous le token de stop -> bailout mid-phrase
qui coupe l'image finale), sans jamais étouffer un VRAI stop (pas de runaway).

Le bug que ce canari garde : avant le fix, le seul rempart anti-bailout était la
valeur magique `repetition_penalty=1.05`, « calibrée à fréquence ». Le fix la
remplace par un mécanisme SANS seuil — l'écart est mesuré, pas choisi. Ce test
échoue (exit 1) si quelqu'un retire le guard ou casse l'invariant d'écart.

Test UNITAIRE pur : logits synthétiques [1,V], pas de modèle, pas de réseau, pas
de poids prod. On exécute les VRAIS processeurs shippés (snapshot sur les logits
bruts, une pénalité simulée, puis clamp) et on assert l'argmax post-chaîne.
Déterministe. CI : `python eos_guard_decoupling.py` -> exit 0 (vert) / 1 (rouge).
"""
import ast
import pathlib
import sys

import mlx.core as mx

# Test the SHIPPED source of `make_eos_guard_processors` without importing all of
# runner.py: the stock pip mlx_lm on a dev Mac lacks the cluster's patched symbols
# (sharded_load, …), so a full import fails off-cluster. The function only needs
# `mx`, so we extract its exact source via AST and exec it — still the real code
# (re-read each run, no drift), just isolated from the cluster-only deps.
RUNNER = pathlib.Path(__file__).resolve().parents[2] / "runner.py"
_src = RUNNER.read_text()
_func_src = None
for _node in ast.parse(_src).body:
    if isinstance(_node, ast.FunctionDef) and _node.name == "make_eos_guard_processors":
        _func_src = ast.get_source_segment(_src, _node)
        break
if _func_src is None:
    print("FAIL: make_eos_guard_processors introuvable dans runner.py")
    sys.exit(1)
_ns: dict = {"mx": mx}
exec(_func_src, _ns)
make_eos_guard_processors = _ns["make_eos_guard_processors"]

NEG = float("-inf")
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def chain(raw_row, pen_row, stop_ids):
    """snapshot(raw) -> [simulated penalty/bans] -> clamp(penalized). Returns the
    post-chain logits row as a Python list. The penalty step is modelled by simply
    handing `clamp` the already-penalized logits — exactly what the real chain does
    (rep-penalty + ngram + cjk all run between snapshot and clamp)."""
    snap, clmp = make_eos_guard_processors(stop_ids)
    raw = mx.array([raw_row], dtype=mx.float32)
    snap(None, raw)                                   # records gap0 from RAW logits
    pen = mx.array([pen_row], dtype=mx.float32)
    out = clmp(None, pen)                             # pins the stop tokens
    return [float(x) for x in out[0].tolist()]


def argmax(row):
    return max(range(len(row)), key=lambda i: row[i])


print("EOS-guard — découplage stop/répétition (rapport §9.4)\n")

# ── A. La promotion EOS induite par la pénalité est effacée ──────────────────
# raw: id3 ("fréquence") mène à 5.0 ; stop id7 = 3.0 (sous le contenu).
# pénalité: id3 chute à 1.0 (pénalisé sous le stop). SANS guard -> argmax = stop
# (bailout = le bug). AVEC guard -> le stop est rabattu, le contenu gagne.
raw = [0, 2, 0, 5, 0, 0, 0, 3]
pen = [0, 2, 0, 1, 0, 0, 0, 3]
print("A. promotion induite par la pénalité (le bug 'fréquence')")
check("sans guard, le stop gagnait (reproduit le bug)", argmax(pen) == 7,
      f"argmax_sans_guard=id{argmax(pen)}")
outA = chain(raw, pen, {7})
check("avec guard, le stop ne gagne plus", argmax(outA) != 7, f"argmax=id{argmax(outA)}")
check("avec guard, le leader de contenu gagne (id1)", argmax(outA) == 1)
# invariant: le stop garde >= son écart pré-pénalité (2.0) sous le leader (2.0).
leader1A = max(outA[i] for i in range(8) if i != 7)
check("invariant d'écart: leader1 - stop >= gap0 (2.0)",
      leader1A - outA[7] >= 2.0 - 1e-4, f"leader1-stop={leader1A - outA[7]:.3f}")

# ── B. Un VRAI stop est préservé (pas de runaway) ────────────────────────────
# raw: stop id7 = 5.0 mène naturellement (le modèle VEUT finir). pénalité baisse
# même le contenu. AVEC guard -> le stop gagne toujours.
print("\nB. vrai stop préservé (le modèle veut finir)")
outB = chain([0, 0, 0, 2, 0, 0, 0, 5], [0, 0, 0, 1, 0, 0, 0, 5], {7})
check("le vrai stop gagne toujours (no runaway)", argmax(outB) == 7, f"argmax=id{argmax(outB)}")

# ── C. No-op quand le stop reste bas et n'est pas promu ──────────────────────
# raw: id3=5 mène, stop id7=1 (bas). pénalité touche id1 (PAS le leader). Le
# guard ne doit RIEN changer : id3 reste argmax, le stop reste à 1.0.
print("\nC. no-op quand la pénalité ne promeut pas le stop")
outC = chain([0, 4, 0, 5, 0, 0, 0, 1], [0, 0, 0, 5, 0, 0, 0, 1], {7})
check("le leader de contenu reste argmax (id3)", argmax(outC) == 3)
check("le logit de stop est inchangé (1.0)", abs(outC[7] - 1.0) < 1e-4, f"stop={outC[7]:.3f}")

# ── D. Garde 'tout le contenu banni' (pas de deadlock) ───────────────────────
# pénalité/ban met TOUT le non-stop à -inf (ngram a tout banni). Le guard doit
# laisser passer le stop (sinon plus rien n'est sélectionnable).
print("\nD. garde 'tout le contenu banni' (pas de deadlock)")
outD = chain([0, 0, 0, 5, 0, 0, 0, 2], [NEG, NEG, NEG, NEG, NEG, NEG, NEG, 2], {7})
check("le stop reste fini/sélectionnable", outD[7] == 2.0 and argmax(outD) == 7,
      f"stop={outD[7]}")

# ── E. Multi-stop : les deux ids sont épinglés (vectorisation) ───────────────
# stop = {6, 7}. id3 mène à 5.0, pénalisé à 0.5. Les DEUX stops doivent être
# rabattus sous le leader -> le contenu gagne.
print("\nE. multi-stop (vectorisation des deux ids de stop)")
outE = chain([0, 0, 0, 5, 0, 0, 3, 1], [0, 0, 0, 0.5, 0, 0, 3, 1], {6, 7})
check("aucun des deux stops ne gagne", argmax(outE) not in (6, 7), f"argmax=id{argmax(outE)}")
check("le leader de contenu gagne (id3)", argmax(outE) == 3)

# ── Verdict ──────────────────────────────────────────────────────────────────
print()
if FAILURES:
    print(f"ROUGE — {len(FAILURES)} assertion(s) en échec: {', '.join(FAILURES)}")
    sys.exit(1)
print("VERT — le guard EOS découple stop et répétition (sans seuil, sans runaway).")
sys.exit(0)
