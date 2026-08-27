# spec_harness — harnais de test spec-decode V4 (DSpark)

## Quoi

Harnais de test du speculative decoding DeepSeek-V4 distribué (drafter
DSpark réel + target V4 distribué, pipeline 5-rangs Argo). Il ne fait
partie d'aucun chemin de service (pas de registration, pas d'import
depuis `api.py`/`runner.py`) : ce sont des scripts de recherche/validation
lancés à la main sur les nœuds du cluster, hors du runtime de prod.

Référence : le plan approuvé qui en dépend intégralement est
`docs/PLAN-2026-08-04-v4-pro-dspark-fast-prefill.md`.

Scripts :

| Script | Rôle |
|---|---|
| `dv_g.py` | Run principal — drafter DSpark + target V4 distribué (ds4 pipeline), speculative decoding bout en bout. |
| `dv_d_drafter.py` | Le drafter DSpark en MLX (module + chargement/dequant des poids). |
| `dv_q8_save.py` | Matérialise le drafter en checkpoint MLX quantisé (évite la dequant FP8/FP4 à chaque run). |
| `extract_embedded_dspark.py` | Extrait le drafter DSpark embarqué (`mtp.*`) d'un checkpoint DeepSeek-V4-Flash-DSpark complet vers un checkpoint standalone. |
| `dv_pro_draft_debug.py` | Debug de shape du forward du drafter Pro, single-node, contexte factice. |

## Env requis par script

| Variable | Sens | Exemple |
|---|---|---|
| `MLX_RANK` | Rang du process dans le groupe distribué. | `0` |
| `MLX_WORLD_SIZE` | Taille du groupe distribué. | `5` |
| `MLX_JACCL_COORDINATOR` | Host:port du coordinateur JACCL (rang 0). Construit à partir de `SPEC_COORD_HOST`. | `$SPEC_COORD_HOST:<port>` |
| `MLX_IBV_DEVICES` | Chemin du fichier de mapping des devices RDMA. | `/tmp/mlx_jaccl_devices.json` |
| `MLX_METAL_FAST_SYNCH` | Active la synchro Metal rapide. | `1` |
| `A0_BACKEND` | Backend distribué MLX. | `jaccl` |
| `SPEC_COORD_HOST` | Host de coordination (ex-rang0) — requis, pas de défaut en dur. | (IP ou nom du nœud rang0) |

Ces variables sont lues par `mlx.distributed.init()` / l'environnement
d'exécution, pas hardcodées dans les scripts — c'est le scrub qui les a
externalisées (règle repo : jamais d'IP/host en dur, même en défaut).

## Prédicats d'état propre avant run

(recopiés depuis `docs/RUNBOOK-argo-v4.md`, section « Prédicats d'état
propre » — à tenir synchronisés si le runbook évolue)

- `uptime < 1 min` par nœud (vérifier l'uptime RÉEL — un `sudo reboot`
  peut ne pas partir ; `sudo shutdown -r now` est plus fiable, utilisé
  avec succès 3×).
- `wired < 20 G` par nœud (`vm_stat`).
- `apipa == 4` par nœud (4 interfaces `inet 169.254.250.x` — le GID que
  JACCL route, absent ~10 s au boot).

Ne pas lancer un script du harnais si un de ces trois prédicats n'est
pas vérifié sur chaque nœud impliqué.

## Avertissement

Jamais de `pkill` sur un run en collectif natif (JACCL/A0) : la fuite
mémoire wired qui en résulte n'est récupérable que par reboot du nœud
(pages Metal reboot-only). Arrêter proprement (laisser le run terminer,
ou couper au niveau applicatif via le protocole `OP_STOP` du script,
jamais au niveau process).
