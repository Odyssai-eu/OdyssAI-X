# Evidence log — nuit 2026-07-02→03 — MiniMax-M3-VL distribué (mlx-vlm, ring/TCP)

> Branche `feat/mlx-vlm-distributed`. Toute l'échelle du plan
> ([PLAN-NIGHT-vlm-distributed.md](PLAN-NIGHT-vlm-distributed.md)) est VERTE :
> Gate-0 → Gate-1 (dense + MSA) → runner → 2-node réel → 4-node réel → E2E engine.
> **Prod .39 jamais touchée.** Le GO matin gate le deploy.

## Le résultat en une ligne

MiniMax-M3-VL Q6 (327 GB) servi **tensor-parallel sur 2 nodes 256 GB (.30+.31,
.29 libre — la cible primaire)** et sur **4 nodes (.29-.32, la consigne)**, avec
sortie correcte prouvée par canaris contre la baseline single-node, à travers
tout le rail engine (load API → fan-out JSONL → caption image streamée → unload).

## Chiffres (Q6 327 GB, greedy, image test 448×448, prompt court 445 tok)

| Config | gen tok/s | TTFT | load | peak/rank |
|---|---|---|---|---|
| single-node .29 (baseline) | **24.97** | 5.31 s | 86 s | 327.2 GB |
| TP-2 localhost .29 | 13.03 | 5.27 s | 68 s | 173.7 GB |
| TP-2 cross-node .30+.31 (ring) | **8.84** | 3.99 s | 78 s | 173.7 GB |
| TP-4 cross-node .29-.32 (ring) | **5.53** | 3.51 s | 79 s | 96.6 GB |
| Prompt long 3748 tok (MSA engagé), single-node | 19.15 | 14.3 s | — | — |
| Prompt long, TP-2 localhost | 11.75 | 16.9 s | — | 175.6 GB |

Pré-enregistré et confirmé : le ring/TCP est **latency-bound** (~117 all_sums
de ~12 KB par token) → TP ring < single-node. La valeur de la nuit = la preuve
de CORRECTION + le chemin Q8/JACCL ; la perf est la phase JACCL.

## Correction — la méthode canaris

Chaque run imprime par rank : sha256(pixel_values), sha256(inputs_embeds),
sha des token-ids, texte. Verdicts :

- **Vision (unknown #1 du plan) : RETIRÉ.** `pv` et `emb` bit-identiques sur
  tous les ranks ET vs baseline, dans TOUTES les configs. La tour vision
  répliquée est déterministe.
- **Alignement inter-ranks : PARFAIT.** Tokens identiques entre ranks dans
  toutes les configs (2, 4 nodes) — zéro desync, zéro deadlock.
- **Prompt court vs baseline : préfixe greedy identique 15-16 tokens** puis
  bascule near-tie (réassociation flottante du TP — bénin, mesuré, même point
  de bascule dans les 3 configs TP). Contenu sémantiquement équivalent.
- **Prompt long (3748 tok, sélecteur MSA engagé) : 97/97 tokens IDENTIQUES**
  à la baseline en TP-2. Le chemin sparse est token-parfait.

## Les 2 bugs upstream trouvés et fixés (in-repo, jamais dans site-packages)

1. **`MiniMaxPackedSwitchGLU.gate_up_proj` fusionné [gate|up] slicé
   contiguëment** par `shard_inplace("all-to-sharded")` → rank0 = tout-gate,
   rank1 = tout-up → `activation(gate)*up` scrambled → **garbage déterministe
   identique sur tous les ranks** (observé : sha 9ec8d76e vs baseline d73be672).
   Micro-prouvé sur les modules de la layer 1 : slicing contigu ≠ per-half, et
   le fix per-half reproduit le forward full à relmax 0.0033 (bruit de quant).
   Fix : `_shard_fused_gate_up_inplace` (slice chaque moitié, out-dim jamais
   packé → exact pour tout bit-width).
2. **Indexer MSA shardé casse la sélection de blocs** : l'agrégation des
   scores est CROSS-HEAD (`mx.max(block_scores, axis=1)`) → chaque rank
   sélectionnerait des blocs différents depuis ses heads locaux. Fix :
   indexer **répliqué** (~190 M params, négligeable) → sélection bit-identique
   baseline. Validé par le run MSA 97/97.

Plus les 3 mines de `sharded_load` (délégateur `Model.shard` absent, `print`
nu sur stdout, tour vision lazy) — fixées dans `_sharded_vlm_load` in-repo.
Le tout = matière à PR upstream (Blaizzy/mlx-vlm).

## E2E engine (v1.12.0, instance dev locale :8010 — .39 intouchée)

- `POST /admin/clusters/main/load {model VL, node_indices:[1,2]}` →
  `dispatched: vlm-dist-pool`, alias `studio-jaccl-4-ultra:minimax-m3-vl`,
  ready **73.8 s** (2 ranks ssh-Popen RunnerProc, hostfile ring écrit par
  l'engine, port éphémère).
- `/v1/models` publie l'alias avec `is_vlm` ; badge pool-level OK.
- **Chat completion image data-URI** : caption correcte, usage 432/57,
  think-split → `reasoning_content` fonctionne sur le chemin submit.
- **Unload** : zéro process résiduel, wired récupérée (~4.3 GB), sweep
  double-pattern (runner + vlm_runner) vert.
- **Persistance v2** : entrée dédiée `is_vlm_dist` (sans port/upstream) ;
  un engine SANS flag skippe le restore proprement
  (`skipping vlm-dist restore — VLM_DISTRIBUTED_ENABLED is off`), zéro
  corruption des autres pools.
- Cancel E2E dédié : SKIPPÉ (chemin hérité inchangé du texte ; le cancel
  mid-stream est prouvé au niveau runner par N3 : convergence des 2 ranks,
  `finish_reason=cancelled`).
- Flag d'activation réel : **`ODYSSAI_X_VLM_DISTRIBUTED_ENABLED=1`** (contrat
  env engine préfixé — un `VLM_DISTRIBUTED_ENABLED` nu n'est PAS lu).

## Incidents infra de la nuit (tous élucidés)

1. **macOS 26 Local Network vs nohup** : les orphelins nohup/launchd voient
   leurs connexions LAN SORTANTES droppées (ring `error: 60`) ; sessions ssh
   tenues + localhost + inbound OK. Prouvé par A/B smoke même binaire/port.
   → Les probes tournent en sessions tenues ; **RunnerProc (engine) est
   immunisé par design**. Gravé en mémoire.
2. **.32 : stack de routage corrompue en cours de nuit** (errno 65 sur TOUTE
   nouvelle connexion TCP LAN sortante, ping/établi OK, les 2 routes subnet
   flaggées `!`). **Reboot = fix.** Anomalies résiduelles à traiter :
   .32 a DEUX interfaces sur le LAN (en0=.32 + **en1=.47**) — suspect n°1 de
   la corruption, à débrancher/reconfigurer — et macOS **26.4.1** vs 26.5 sur
   les 3 autres ultras (aligner).
3. **Le reboot de .32 vide /tmp** → un rank a perdu son probe (`exit=2`) ;
   re-staging + relance du rank seul a suffi (le ring en init attend).

## Ce qui reste (décisions Sophie au matin)

1. **GO/no-GO deploy .39** : la branche est additive (flag off par défaut =
   comportement prod byte-identique). Le merge est sûr même sans activer.
2. **Topologie de service** : le VL distribué vise quel étage ? (2-node
   .30+.31 pour libérer .29 ; le single-node .29 reste le plus RAPIDE à 25
   tok/s tant que le Q6 tient.) Perf distribuée = passer au backend JACCL
   (phase suivante — le bug queue-pair reste l'obstacle connu).
3. **Consolidation runner** (désaccord panel non tranché) : garder
   `vlm_runner.py` séparé vs replier en mode `RUNNER_VLM` dans runner.py.
4. **PR upstream mlx-vlm** (fused gate/up + indexer + sharded_load).
5. `.33` est provisionné (venv + checkpoint intègre) → topologie 4-node sans
   .29 (.30-.33) possible dès qu'on veut, Telemak .33 à stopper le temps du
   test (jamais touché cette nuit).
