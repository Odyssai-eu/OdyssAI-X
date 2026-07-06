# EVIDENCE — 2026-07-06 — MTP distribué, gate G2

> Nuit d'exécution du plan `docs/PLAN-distributed-mtp.md` (E0→E3).
> Résultat **split** : alignement PROUVÉ, perf BLOQUÉE (diagnostiquée, non
> fondamentale). Décision : STOP au gate (règle cheval-mort) + reco.

## Ce qui est prouvé (banké)

### G2 — ALIGNEMENT : PASS (le risque novateur du chantier)
- **77 rounds comparés sur les 3 rangs (.30/.31/.32, GLM-5.2-Q6 pipeline
  ring, MTP natif D3) → 0 divergence.** Chaque round, les 3 rangs
  s'accordent sur `(drafted, accepted, sha256(tokens))` à l'identique.
- Confirme empiriquement l'insight central D1/D3 du plan : heads MTP
  **répliqués par rang** + verify **greedy déterministe** → accept-count
  identique **par construction**, aucun desync. C'était l'inconnue #1.
- Acceptance **0,55** sur 77 rounds (au-dessus du plancher 0,4). Le head
  MTP GLM-5.2 draft juste.

### E1 — boucle + module : PASS
- `mtp_spec.py` + `mtp_module.py`, 11/11 tests unitaires (parité par
  position vs AR greedy D1-3, oracle 100%-accept, accepts partiels,
  warm-resume, stop, pre_norm). Commits `473d2e2`, `8985eaf`.

### Intégration engine : livrée (non déployée)
- `api.py` : knob pool `mtp:{enabled,depth}`, activation par-requête
  JSONL, `CanaryAggregator` (barrier par round + trip auto-disable), badge
  dashboard, v1.14.0. Commit `4d337c6`. **Pas déployée sur .39** (G2 perf
  non franchi — rien à pousser en prod tant que le MTP n'accélère pas).

## Ce qui bloque

### G2 — PERF : FAIL
- **MTP ≈ 0,61 tok/s vs AR 12,3 tok/s → 0,05× (≈20× plus LENT).**
  Seuil G2 = ≥1,25×. Échec net.

### Diagnostic — PAS le mur MoE fondamental
- Le smoke standalone (.31) mesurait **`draft_step` = 9,6 ms** chaud.
  3 drafts/round = ~29 ms. Le draft n'est donc PAS le goulot.
- Les ~3 s/round viennent de la **boucle verify distribuée non-pipelinée** :
  `native_mtp_stream_generate` fait `mx.eval` à CHAQUE round pour décider
  l'acceptance, en série, sans overlap. Chaque round paie la latence ring
  3-hops complète, bloquante. En AR, la pipeline auto_parallel garde les
  rangs occupés en flux ; le spec loop les sérialise round par round.
- **Ce n'est pas cheval-mort au sens fondamental** : le plafond idéal si la
  latence verify était masquée ≈ AR × (accept+1)/round ≈ **~1,8×** à
  acceptance 0,55-0,69. Le prix est réel ; le blocage est structurel-loop,
  pas physique.

## Bugs racine corrigés en route (nuit)
1. **Cache combiné draft+trunk** : le path draft-model + prompt_cache
   crashait le prefill draft (préexistant prod single-rank). `8985eaf`.
2. **Deadlock #23 driver** : stdout rank>0 non-DEVNULL → back-pressure ring.
3. **OOM chargement MTP multi-rang** : `_stack_moe_experts` = `mx.stack`
   des 768 experts bf16 = pic ~20 GB sur les rangs MoE-lourds → SIGKILL
   silencieux (rank0 dense survivait, rank1/2 mouraient). **Fix racine :
   sidecar pré-quantizé Q6 module-layout (8,09 GB), buildé offline une
   fois, chargé par rang sans stack ni rewrite.** `build_prequantized_sidecar`.
4. **Poids MTP strippés** par les convert MLX (GLM-5.2-Q6 layer 78 absente) :
   récupérés par range-HTTP du repo source (`sidecar_fetch.py`, 19,9 GB).

## Écarté (testé, ne pas re-proposer)
- **Draft-model séparé multi-rang** (harness E0) : le path spéculatif
  mlx-lm 0.31.3 WEDGE en multi-rang TP-ring (0% CPU, bloqué Metal ;
  reproduit .29 puis .30 propre). Jamais le path produit.
- **Charger le sidecar bf16 par rang** : OOM (cf fix #3).

## Reco / next
Le chantier n'est PAS mort — il est **alignement-résolu, perf-bloquée sur
un point diagnostiqué**. Un fix ciblé de la boucle verify (masquer la
latence ring : ne pas `eval` bloquant chaque round ; pipeliner le verify /
batcher les rounds indépendants) vise ~1,8×. C'est un **changement
d'archi de la boucle** (décision conséquente) → GO Sophie avant de
l'attaquer, pas en solo. Alternative : parquer, MTP reste off en prod.
