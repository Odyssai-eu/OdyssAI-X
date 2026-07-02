# Session 2026-07-02→03 (nuit) — MiniMax-M3-VL distribué sur mlx-vlm

> Nuit full-autonome sur le /goal de Sophie : « minimax-vl opérationnel sur
> [les] nodes 256go, mlx-vlm distributed installé sur les 4 nodes ». L'échelle
> entière passe au vert — deux bugs upstream jamais vus (le chemin TP de
> mlx-vlm n'avait tourné nulle part), un runner distribué au contrat runner.py,
> l'intégration engine, le deploy prod v1.12.0. Puis la dernière heure rappelle
> la loi des restarts : l'état persisté rejoue un pool Ornith mort, un reboot
> mal placé le dégrade, « tout plante », « stop ». La nuit livre le distribué ;
> le matin hérite d'un reset à faire et d'un vrai fix shutdown/restore.

---

## TL;DR — Avant / Après

| Aspect | Avant | Après |
|---|---|---|
| VL multi-node | Impossible — `sharded_load` upstream sans caller, `Model.shard` absent, jamais exécuté | **TP-2 et TP-4 validés par canaris** (sha pixel_values/inputs_embeds/tokens vs baseline), MSA engagé 97/97 tokens identiques |
| Bugs upstream mlx-vlm | Inconnus (chemin mort) | **2 trouvés + fixés in-repo** : gate_up_proj fusionné [gate\|up] slicé contiguëment (garbage déterministe), indexer MSA shardé casse la sélection cross-head |
| Runner VL | `mlx_vlm.server` nohup single-node only | `scripts/vlm_runner.py` — contrat runner.py exact (fan-out JSONL, emit rank-0, keepalive all_sum, stdin-EOF, free_metal), v0 sans caches |
| Engine | v1.11.0, VL = clamp 1 node | **v1.12.0** : `VLMDistPool(RunnerPool)`, hostfile ring écrit par l'engine, persistance `is_vlm_dist`, orphan sweep double pattern, flag `ODYSSAI_X_VLM_DISTRIBUTED_ENABLED` (off = byte-identique) |
| Prod .39 | v1.11.0 | **v1.12.0 + flag actif**, load prod `argo:minimax-m3-vl` sur .30+.31 réussi (76 s), chat image E2E OK |
| .29 (ultra-512) | Seul node VL possible | **Libéré** — le VL tient sur 2×256 Go (167 Go/rank, marge 33 Go) |
| .33 | Ni venv ni checkpoint | Provisionné (py3.12 + mlx-vlm 0.6.3 + Q6 intègre) |
| Perf (greedy, image, Q6) | single 24.97 tok/s | TP-2 cross 8.84 · TP-4 5.53 — ring latency-bound pré-enregistré, la perf = phase JACCL |
| État à la fermeture | — | Cluster `main` **degraded** (cascade restart, cf. Phase 8), VL non chargé, reset à faire au réveil |

Versions de sortie : OdyssAI-X **v1.11.0 → v1.12.0** (7 commits, ~1 355 lignes).

---

## Phase 0 — Le plan (recon 4-lecteurs + panel à 3)

Reprise du handoff `vlm-distributed`. Workflow : 4 lecteurs parallèles
(`runner.py`, machinerie `api.py`, sources mlx-vlm @ecc457b installées sur .29,
état live des 4 ultras) puis panel direct/alternative/sceptique à convergence.
Architecture unanime : **fan-out JSONL côté engine, aucun HTTP sur les ranks**
(le pattern prod de runner.py), images en base64 dans les `messages`, nouveau
`vlm_runner.py` séparé (blast-radius), `_sharded_vlm_load` in-repo, ring/TCP
d'abord. Échelle N0→N7 avec gates empiriques + critères d'abort, 18 points.
Plan écrit ([PLAN-NIGHT-vlm-distributed.md](PLAN-NIGHT-vlm-distributed.md)),
commit `5307657`.

Sophie : *« crée un branch pour ne pas endommager la version actuelle de
OdyssAI-X. fait le plan et ensuite on y va en full autonome pour la nuit.
ne code rien avant mon GO »* — puis : *« GO. /goal minimax-vl opérationel sur
3 nodes 256go. , mlx-vlm distributed installé sur les 4 nodes »*. Le « 3 nodes »
est structurellement impossible (4 KV heads, TP ∈ {2,4}) — recadré en TP-2 sur
.30+.31 (.29 libéré = l'intent), accepté : *« OK, minimax sur 2 nodes est
accepté »*.

## Phase 1 — Gates 0 et 1 : le bug qui a failli passer pour de la magie

Ring smoke OK (localhost + cross .30↔.31). Baseline single-node : 65 tokens
greedy, caption exacte, 24.97 tok/s. Puis TP-2 avec le shard() upstream :
**garbage déterministe** (« RZ ANANWell MANny… »), identique sur les deux ranks,
hashes vision parfaits — le forward est FAUX, pas désynchronisé.

Fausse piste n°1 (l'indexer MSA) disqualifiée proprement par A/B : sha
byte-identique avec ou sans le fix → l'indexer ne s'exécutait même pas
(prompt < block×topk). Discriminateur MoE-only → garbage aussi → le MoE est
coupable. Micro-test au niveau module (layer 1, sans charger les 327 Go) :
**`MiniMaxPackedSwitchGLU.gate_up_proj` est fusionné [gate|up] sur l'out-dim,
et `shard_inplace("all-to-sharded")` slice contiguëment → rank0 = tout-gate,
rank1 = tout-up**. Le fix per-half reproduit le forward full à relmax 0.0033.
Le fix indexer est GARDÉ (l'agrégation `mx.max(block_scores, axis=1)` est
cross-head — le sharder casserait la sélection dès que le prompt dépasse
1536 tokens) : indexer répliqué (~190 M params).

Gate-1 re-run : caption correcte, préfixe greedy 15 tokens puis bascule
near-tie (réassociation fp, bénigne, mesurée). Gate-1b prompt long 3748 tokens
(sélecteur MSA engagé) : **97/97 tokens identiques à la baseline**. Commit
`d8bc4e7`.

## Phase 2 — vlm_runner.py + le driver

`scripts/vlm_runner.py` (517 lignes) : contrat miroir de runner.py — env vars,
reader thread + cancel set, emit gaté rank 0, phase markers stderr, barrier,
keepalive all_sum, stdin-EOF kill-switch, free_metal ; extraction d'images
data-URI/path (les ranks ne fetchent jamais) ; v0 sans session-cache/prewarm/
radix/spec/batch (chaque coupe = une source de divergence en moins). Test via
driver mini-RunnerProc : caption data-URI, texte-only, **cancel mid-stream
convergé sur les 2 ranks**, keepalive, stop propre, wired récupérée. Commit
`c42317c`.

## Phase 3 — 2-node réel, et le gotcha macOS de la nuit

Premier essai cross-node : `[ring] Couldn't connect (error: 60)`. A/B décisif :
même binaire, même port — session ssh TENUE = OK, orphelin nohup = SYN droppés.
**macOS 26 bloque le LAN sortant des orphelins launchd** (Local Network) ;
localhost et l'inbound passent — d'où `mlx_vlm.server` nohup qui sert très bien
en prod. Conséquence architecturale : **RunnerProc (ssh tenu) est immunisé par
design**. Gravé en mémoire.

N4 en sessions tenues : **PASS** — 8.84 tok/s, TTFT 3.99 s, 173.7 Go/rank,
même point de divergence bénigne que localhost. La cible primaire (VL sur les
256 Go, .29 libre) est prouvée.

## Phase 4 — 4-node : .32 se saborde, reboot, PASS

Sophie : *« minimax q6 est sur les 4 nodes ,29 .30 ,31 , .32 — test avec ca »*.
Premier TP=4 : rank3 (.32) err60 même en session tenue. Diagnostic factuel :
**la stack de routage de .32 s'est corrompue en cours de nuit** — errno 65 sur
TOUTE nouvelle connexion TCP LAN sortante, ping et connexions établies OK, les
deux routes subnet flaggées `!`. Anomalies de fond : .32 a DEUX pattes LAN
(en0=.32 + en1=.47) et macOS 26.4.1 vs 26.5 ailleurs. Reboot = fix (+ /tmp
vidé → re-staging). TP=4 : **PASS** — 5.53 tok/s, 96.6 Go/rank, 4 ranks
alignés, caption correcte.

## Phase 5 — Intégration engine (v1.12.0)

`api.py` : `remote_vlm_cmd` (hostfile ring `[["ip:port"],…]` en echo-prefix,
venv vlm), `VLMDistPool(RunnerPool)` (submit/cancel/stop hérités ;
`vlm_proxy=False` → chemin submit, pas le proxy http), `RunnerProc.match_pattern`
(pkill ciblé vlm_runner, jamais runner.py ni les services mlx_vlm prod),
persistance v2 `is_vlm_dist` dédiée (avant `is_vlm` — shape VLMPool aurait
corrompu le save), orphan sweep double pattern, branche is_vision gated
`len(node_indices)>1` + flag. Commit `bff3849`. E2E sur engine dev local :
load 73.8 s → chat image → unload zéro résidu → garde-fou flag-off au restore.
Gotcha découvert : le flag réel est **`ODYSSAI_X_VLM_DISTRIBUTED_ENABLED`**
(contrat env préfixé — un nom nu n'est jamais lu).

## Phase 6 — Deploy prod

Sophie : *« ok, on a la transparence ml-vlm sur les nodes comme sur .29 ? »* →
oui (même endpoint, même nommage, même badge) avec les nuances v0 — puis
*« go »*. Merge `fec2c42`, push forge+internal, hot-patch .39 + flag dans le
compose (backup fait), **v1.12.0 vérifiée**. Découverte au passage : le clone
git de .39 est mort (189 behind, 16 commits locaux, dirty) — la prod vit par
overlays docker cp ; rebuild interdit, follow-up hygiène. Load prod :
**`argo:minimax-m3-vl` TP-2 sur .30+.31, ready 76.4 s, chat image OK.**
Fix cosmétique dans la foulée (`/v1/models` publiait http-proxy/1 node pour le
pool dist) : commit `9252e11`, docker cp, restart.

## Phase 7 — La cascade du restart (« tout plante »)

Ce restart-là déclenche la leçon de la nuit :

1. Les ranks du pool chargé **survivent à la mort du container** (bloqués en
   collectif ring, SIGTERM ignoré ; le graceful du lifespan n'a pas le temps
   sous la grace docker).
2. Le boot-sweep les SIGKILL après 12 s → **wired leak ~182 Go sur .31**.
3. Le restore respawne aveuglément sur le node leaké → OOM rank1, restore failed.
4. Et surtout : **le state du volume rejouait AUSSI un pool `ornith` d'hier
   soir** (le gotcha connu « restart rejoue l'état persisté ») — l'engine a
   respawné Ornith sur .30/.31/.32 pendant mes opérations, puis mon reboot de
   .31 (pour le leak) a tué un de ses ranks → keepalive 2/3 → **cluster
   degraded**, reload VL refusé.

Sophie : *« tu fais quoi ? »* — *« tout plante »* — *« stop »*. Arrêt immédiat,
autopsie livrée, aucune commande de plus. *« c'est ok »*.

**Mes deux fautes, nommées** : restart du container sans vérifier ce que le
state du volume allait rejouer (gotcha documenté, en mémoire depuis juin) ;
reboot d'un node sans re-vérifier qui l'utilisait après le restart.

## État à la fermeture (~4 h 30)

- Prod .39 : **v1.12.0 + flag**, healthy, proxies telemak/cloud intacts.
- Cluster `main` : **degraded** (ornith fantôme), VL non chargé, orphelins
  Ornith possibles sur .30/.32, .31 propre (rebooté).
- Le VL distribué : prouvé de bout en bout, y compris un load prod réussi.
- `.33` provisionné. `.32` réparé mais en1/.47 + 26.4.1 à traiter.

## Phase 9 — Matin : nommage nu + le faux « degraded »

Sophie voit le dashboard : `argo:minimax-m3-vl` **VLM** à côté de `ornith` nu.
*« les nom des pools n'est pas OK. le pool 1 = argo:modele, pool 2 = modele.
pourquoi ? je veux le meme. pool 1 = modele, pool 2 = modele »*.

**Pourquoi l'incohérence** : le contrat #64 que j'avais shippé — alias
**auto-dérivé** prenait le préfixe `argo:` (`_derive_pool_alias` =
`cluster:model`), alias **explicite** (`ornith`, tapé hier) restait verbatim.
Deux chemins de création → deux styles. Rien à voir avec le modèle.

Faux pas dans la foulée : j'ai proposé un reset en annonçant le cluster
`main` **degraded** — *« Le cluster main est degraded ??? on a reboot 5x
depuis hier »*. Elle avait raison : mon « degraded » datait de la cascade
d'hier soir, **périmé**. Vérif live : `degraded: None`, health idle, deux
pools chargés. Retiré, excuse faite. Leçon renforcée : ne jamais ressortir un
état sans le re-sonder.

**Le fix** (`361ad78`, **v1.12.1**) : `_derive_pool_alias` publie désormais le
**slug modèle nu** par défaut (`minimax-m3-vl`). Le préfixe `<cluster>:` ne
revient qu'en **garde anti-collision** — appliqué seulement si le même slug est
déjà chargé sur un AUTRE cluster (jamais le cas : un seul cluster distribué).
Supersede le schéma #64 toujours-préfixé (le préfixe achetait peu, lisait mal à
côté des pools nommés).

Rename des pools vivants sans downtime : édition de l'alias dans
`/app/state-main.json` (`argo:minimax-m3-vl` → `minimax-m3-vl`, backup fait)
puis restart — le serveur VL `.29:8080` (nohup) **survit et est ré-adopté** sous
le nouveau nom sans recharger les 327 Go ; ornith se recharge (~90 s, enfants
ssh). Vérifié : `minimax-m3-vl` + `ornith`, tous deux nus, loaded.

## Numbers de la nuit

- **Commits** : 9 sur main (`5307657` → `361ad78`), branche
  `feat/mlx-vlm-distributed` mergée, push forge + internal.
- **Version** : v1.11.0 → **v1.12.1**.
- **Diff** : ~1 355 lignes ajoutées (dont `vlm_runner.py` 517,
  `vlm_caption_probe.py` 232).
- **Perf Q6 greedy** : single 24.97 · TP-2 local 13.03 · TP-2 cross 8.84 ·
  TP-4 5.53 tok/s ; mémoire/rank 327 → 174 → 97 Go.
- **Canaris** : vision bit-identique partout ; MSA long-prompt 97/97.

## TODO direct (par ordre)

1. ~~Reset + assainissement~~ — **MOOT** : les reboots de Sophie avaient déjà
   nettoyé, cluster sain au matin (mon « degraded » était périmé). Nommage nu
   réglé (v1.12.1). Règle d'or confirmée : **unload avant tout restart engine**,
   et toujours lire (`cat state-main.json`) ce que le restart rejouera.
2. **Fix root shutdown/restore** : stop effectif des pools sous la grace
   docker (ou stop-before-restart outillé), wired-guard au restore (le sweep
   retourne déjà `wired_bytes` — skip un node leaké), grace SIGTERM adaptative
   aux gros modèles.
3. `.32` : retirer/reconfigurer la 2e patte LAN (en1=.47), aligner macOS 26.5.
4. VL v0 follow-ups : thinking control (le template VL ignore
   `enable_thinking`), tool-calls, prefix-cache.
5. Phase perf : backend JACCL pour le VL distribué + PR upstream mlx-vlm
   (fused gate/up, indexer, sharded_load).

## Lessons learned

- **Un chemin upstream jamais exécuté ment deux fois** : le premier bug (fused
  gate/up) masquait le second (indexer) — sans l'A/B qui a disqualifié ma
  première « racine », j'empilais un fix inutile sur un vrai bug.
- **Les canaris (hashes par étage + comparaison greedy vs baseline) sont
  l'outil de la nuit** : chaque hypothèse tranchée en un run, zéro narration.
- **La règle des restarts est absolue** : l'état persisté EST un acteur. Tout
  restart d'engine = d'abord lire ce qu'il va rejouer.
