# PLAN NUIT — MiniMax-M3-VL distribué (mlx-vlm) sur les 4 ultras

> Branche `feat/mlx-vlm-distributed` (off main v1.11.0). Plan issu d'une recon
> 4-lecteurs (runner.py, api.py, upstream mlx-vlm @ecc457b sur .29, état live des
> 4 nodes) + panel à 3 mandats (direct / alternative / sceptique) à convergence.
> Exécution full-autonome la nuit du 2026-07-02→03 **après GO Sophie**.
> Prod .39 INTOUCHÉE cette nuit — E2E sur instance engine dev locale.

## Architecture convergée (unanime, ancrée sur faits vérifiés)

**Coordination = fan-out JSONL côté engine, AUCUN HTTP sur les ranks.**
Le seul pattern multi-rank prouvé en prod (runner.py) est repris tel quel :
l'engine écrit le MÊME JSONL sur le stdin de chaque rank ; chaque rank recompute
déterministiquement ; `emit()` gate stdout au rank 0 ; keepalive = petit
`all_sum` ; stdin-EOF = kill-switch orphelin. Le shape « mlx_vlm.server sur
rank 0 » est rejeté par les 3 mandats : pas de primitive broadcast
variable-shape dans mx.distributed, `ResponseGenerator` = continuous-batching
(interdit multi-rank, même règle qui gate BatchGenerator à size==1 dans
runner.py), `sharded_load` a zéro caller dans `server/` (grep vérifié).

**Images** : base64/data-URI DANS les `messages` du JSONL fan-out. L'engine
résout les URLs UNE fois ; les ranks ne fetchent jamais. Chaque rank décode →
`prepare_inputs` → `get_input_embeddings` sur sa vision tower répliquée
(décision gelée, ~1.7 GB). Vérifié dans `ar.py` : pixel_values consommés
UNIQUEMENT au prefill ; le decode = token-ids + KV cache = pattern texte.

**Runner** : nouveau `scripts/vlm_runner.py`, sibling mince de runner.py au
contrat IDENTIQUE (env vars pas d'argparse, reader thread + cancel set, emit
rank-0, phase markers stderr pour RunnerProc, barrier all_sum post-load,
`{"event":"ready"}`, keepalive, stdin-EOF exit, free_metal). Lancé avec le venv
épinglé `/Users/admin/.venvs/mlx-vlm` (JAMAIS `mlx-vlm-main`).

**Le gap `Model.shard` + les mines de `sharded_load`** : fix in-repo
`_sharded_vlm_load()` (~30 lignes) répliquant `mlx_vlm.utils.sharded_load` mais
appelant `model.language_model.shard(group)` en direct. Tue 3 mines vérifiées
d'un coup : (1) le délégateur manquant (`hasattr(model,"shard")` → ValueError) ;
(2) `print("Materializing")` nu sur stdout (corromprait le flux JSONL du
rank 0 → « engine never sees ready ») ; (3) vision tower lazy (`sharded_load`
n'eval que `language_model.parameters()` → 1re requête image matérialise les
poids vision en plein collectif = pseudo-hang). On n'édite JAMAIS les
site-packages sur les nodes (drift de config).

**Backend** : ring/TCP d'abord (gelé), via `RUNNER_BACKEND` env (défaut ring).
Plomberie NET-NEW assumée : runner.py hardcode `backend="jaccl"` (l.1815) et
`remote_cmd` n'a aucune machinerie hostfile. L'engine écrira un
`/tmp/mlx_ring_hostfile_<port>.json` par node (format `[["ip:port"],...]`
vérifié sur les relics des nodes), pattern echo-prefix identique au
`/tmp/mlx_jaccl_devices.json` existant. Hosts/ports dérivés de
`build_topology_from_indices` + `random_ephemeral_port` — zéro URL en dur.

**Intégration engine (rung stretch)** : la branche is_vision (api.py:10282)
gagne un chemin `len(node_indices)>1` derrière un flag explicite → spawn d'un
**RunnerPool** (mode tensor, runner_script=vlm_runner.py, python du venv vlm),
PAS un VLMPool : il coule dans le chemin `pool.submit()` JSONL→SSE existant
(submit forwarde `messages` tel quel, api.py:2094-2098). Le chemin single-node
nohup reste byte-identique. Nouvelle shape d'entrée de persistance (ne pas
muter le dict is_vlm — une entrée malformée corrompt le restore de TOUS les
pools). Nouveau `VLM_RUNNER_MATCH_PATTERN = "mlx-cluster/vlm_runner.py"` dans
l'orphan sweep — disjoint de `mlx_vlm.server` (services prod intouchables).

**Coupes v0 dans le runner (unanime)** : pas de session/prefix cache (l'égalité
de préfixe par token-ids est cassée par les image embeds = classe de corruption
silencieuse), pas de prewarm/radix/spec-decode/batch/kv-q8/disk-cache.
Single-stream, greedy-first.

## Vision-consistency (#1 unknown) — mesurer, pas supposer

Probe offline : chaque rank imprime sha256(pixel_values), sha256(inputs_embeds)
+ checksum token-ids tous les N tokens ; caption greedy comparée token-à-token
à la baseline single-node. Si divergence : UN fallback scripté (rank-0
autoritaire : `all_sum(embeds if rank==0 else zeros)` au prefill seulement —
collectif fixed-shape légitime, pas un protocole inventé à 3h). Si les deux
échouent : STOP l'échelle, write-up. Interdit de tuner le sampler pour masquer
(anti-pattern GLM-DSA, détecteur de rustine = stop dur).

## Cible nodes — arbitrage

Le panel avait coupé à 2 nodes ; Sophie demande explicitement les 4
(.29/.30/.31/.32). Arbitrage retenu : **échelle 2-node → 4-node DANS la même
nuit**. Les rsync vers .30/.31/.32 partent tous en Step 0 (parallèles,
.30 prioritaire) ; le 4-node est un rung APRÈS le green 2-node, pas un
préalable. .32 : 708 Gi libres pour 327 G → passe, mais le plus serré (81%) —
flaggé. kv_heads=4 vérifié → TP=2 et TP=4 légaux (sous réserve Step 0b, cf.
désaccord #1).

## L'échelle de la nuit (gates empiriques + critères d'abort)

| # | Step | Gate | Abort | Pts |
|---|---|---|---|---|
| 0 | **Preflights + long pole.** (a) rsync `-a --partial` du Q6 .29→.30/.31/.32 lancés immédiatement en background ; (b) relire config.json Q6 sur .29 → trancher la contradiction index_heads (désaccord #1) + divisibilité TP2/TP4 ; (c) versions mlx_vlm/mlx du venv `mlx-vlm` identiques sur les 4 nodes ; (d) `sysctl iogpu.wired_limit_mb` partout ; (e) vérifier que `chat_completions` passe le contenu multimodal non-aplati à `pool.submit` | rsyncs partis ; config tranchée ; venvs alignés ; wired limits attendus (471040/204800) | index_heads non divisible → STOP nuit (bloqué structurellement, pas de workaround). Skew venv non alignable → stop avant tout multi-rank. wired_limit reset → fix sysctl avant tout load | 1 |
| 1 | **GATE-0 ring smoke, zéro risque modèle.** Script ~10 lignes : hostfile 2 entrées, `init(backend="ring")` + `all_sum(1.0)`. Localhost .29 d'abord, puis .29+.30 cross-LAN. Chaque essai sous timeout wall-clock + pkill | 2 ranks rank/size corrects, all_sum=2.0, localhost ET cross-node | 3 échecs bornés → STOP nuit + write-up. PAS de fallback JACCL cette nuit, pas d'empilement de hacks env | 1 |
| 2 | **GATE-1 — le test qui décide la nuit** (sans code engine, sans attendre la copie). `caption_probe.py` standalone : TP-2 sur .29 SEUL en ring localhost (~350 GB < 460 GB wired), `_sharded_vlm_load` in-repo, 1 image fixe + prompt fixe, greedy 64 tokens, hashes par rank, comparaison token-à-token vs baseline world_size=1 même checkpoint. Tourne PENDANT les rsyncs | hashes identiques par rank ET caption greedy == baseline token-à-token. Retire d'un coup : le fix shard, TP2 sur ce checkpoint, ring sous vrai forward (~117 collectifs/token), sharded_load jamais exécuté, le #1 unknown vision | divergence → fallback scripté rank-0 all_sum embeds, UNE re-run. Toujours divergent ou hang inexpliqué → STOP échelle + write-up matin. Interdit sampler-masking | 3 |
| 3 | **Écrire `vlm_runner.py` complet** (miroir contrat) + rejouer Gate-1 à travers stdin JSONL piped à la main sur .29 localhost : 1 caption image, 1 requête texte-only, 1 cancel mid-stream, 1 stop. Commit rung vert | caption == baseline Gate-1 ; cancel converge sur les 2 ranks en 1 cycle emit ; teardown propre, wired mémoire récupérée (vm_stat) ; events ready/token/done parsables RunnerProc | wired non récupérée → node marqué needs-reboot, on ne boucle pas dessus. Divergence introduite par la plomberie → fix racine ou stop (2e workaround empilé = stop dur) | 3 |
| 4 | **2-node réel .29+.30** (après rsync .30 VÉRIFIÉ : count + tailles par fichier vs source). Même test offline cross-node : caption greedy vs baseline + UNE mesure perf enregistrée (attendu : ring 2-node < baseline 25 tok/s single-node — pré-enregistré, ~117 all_sums 12 KB latency-bound par token) | caption cross-node == baseline ; 1 chiffre tok/s + TTFT loggés ; teardown propre 2 nodes | copie non vérifiée = le step ne démarre pas (on continue localhost + code step 5-6). Divergence SEULEMENT cross-node → suspecter les poids copiés d'abord (re-checksums) ; poids sains + divergence persiste → stop + write-up. AUCUNE optim de collectifs cette nuit (au-delà de MLX_METAL_FAST_SYNCH) | 2 |
| 5 | **4-node réel .29+.30+.31+.32** (TP=4, après green step 4 + rsyncs .31/.32 vérifiés). Même probe offline : caption greedy vs baseline + le chiffre perf 4-node | caption == baseline ; tok/s+TTFT loggés ; teardown propre 4 nodes | idem step 4. Si .32 pose problème (disque/état) : livrer 2-node + 3 n'existe pas (TP%4) → 4-node reporté au matin, PAS un échec de nuit | 2 |
| 6 | **STRETCH — intégration engine sur instance dev LOCALE** (uvicorn local contre topology.yaml ; .39 intouchée). Branche is_vision `len(node_indices)>1` + flag → RunnerPool(vlm_runner, tensor, venv vlm, ring hostfile écrit par l'engine) ; VLM_RUNNER_MATCH_PATTERN dans l'orphan sweep ; nouvelle entrée persistance ; single-node byte-identique. E2E : load `{model: VL, node_indices:[0,1]}` → ready → chat completion image data-URI → caption streamée ; cancel + teardown | caption E2E streamée via l'engine ; cancel/teardown = zéro process résiduel + wired récupérée ; roundtrip save/restore ne corrompt PAS l'état des pools texte ; le load VL single-node marche inchangé | risque de persistance partagée malformée → pool éphémère (skip persistance, follow-up matin) plutôt que hacker le schéma v2 de nuit. Tout changement débordant de la branche gated → stop. Règle absolue : pas de deploy .39, pas de restart container prod | 5 |
| 7 | **Package du matin.** Ladder de commits finalisée sur la branche (probe / runner / engine, Conventional+HEREDOC+Co-Authored-By) ; evidence log : captions vs baselines, hashes par rank, chiffres perf 2-node et 4-node, états nodes, chaque abort/timeout/flag reboot, issue des 2 désaccords | branche = dernier rung vert, démoable ; log suffisant pour décider GO/no-GO .39 sans rien re-runner | n/a — si l'échelle s'est arrêtée, le write-up du cheval mort EST le livrable | 1 |

**Total : 18 points.** Protocole d'autonomie : chaque tentative sous timeout
wall-clock + pkill (pattern vlm_runner) sur tous les nodes participants + check
récupération wired ; node à wired coincée → needs-reboot, droppé, jamais bouclé ;
`sysctl iogpu.wired_limit_mb` pre-flight avant chaque load ; commit par rung
vert ; détecteur de rustine : 2e workaround empilé → stop + write-up.

## Coupé de cette nuit (ne pas re-proposer)

- JACCL/RDMA pour le VL (gelé — ring d'abord ; la perf, c'est la phase JACCL).
- HTTP rank-0 / chirurgie de mlx_vlm.server (rejeté par les 3 mandats).
- Conversion Q8 ~454 GB (le checkpoint n'existe pas ; justification long-terme).
- Download proactif du 4-bit fallback (présent sur AUCUN node — fetch seulement
  si le Q6 bloque dur, en décision délibérée).
- Session/prefix cache, prewarm, radix, disk cache, spec decode, batch, kv-q8,
  tool-calls VL (v0 cuts — chacun ré-activable en issue de suivi).
- Toute optim perf au-delà de MLX_METAL_FAST_SYNCH + chiffres enregistrés.
- Deploy prod .39 / restart container (exception d'une nuit, loggée, à
  « serveur = main » ; le GO matin gate le push).
- Consolidation runner.py (cf. désaccord #2 — décision Sophie, pas de nuit).

## Les 2 désaccords du panel (non maquillés)

1. **Contradiction factuelle config.json** (les 2 mandats citent une lecture
   live du MÊME fichier) : ALTERNATIVE dit `index_heads: None` (pas de sparse
   index) ; SCEPTIQUE dit `sparse_num_index_heads=4`. Les conclusions
   concordent (TP2/TP4 légaux) mais un des deux a mal vérifié. Tranché
   empiriquement en **Step 0b** (relecture obligatoire) — d'ici là la
   divisibilité est traitée comme NON CONFIRMÉE.
2. **Propriété long-terme du code runner** : ALTERNATIVE veut à terme replier
   le VLM dans runner.py en mode `RUNNER_VLM` env-gated (mlx_lm 0.31.3
   importable dans le venv vlm — vérifié ; le miroir de contrat VA driftter,
   chaque fix futur atterrit deux fois). DIRECT+SCEPTIQUE : fichier séparé
   (blast radius : runner.py prod rsyncé identique partout reste intouché ;
   les patches exo/mlx-lm à l'import de runner.py jamais testés dans le venv
   vlm). **Cette nuit = fichier séparé** (fait de blast-radius). Le end-state
   → décision Sophie après que la nuit ait prouvé le chemin.

## État live vérifié (recon 2026-07-02 ~20h)

- 4 ultras fresh-boot (~33 min), idle, zéro process d'inférence, engine .39
  v1.11.0 sans pool Argo chargé → nodes libres pour la nuit.
- Q6 VL 327 G présent SEULEMENT sur .29. `.30`/.31 : 1.4-1.5 Ti libres ;
  .32 : 708 Gi (81%, le plus serré — passe pour 327 G).
- venv `/Users/admin/.venvs/mlx-vlm` présent sur les 4 ; .29 a en plus
  `mlx-vlm-main` (mlx_vlm 0.6.3 @ecc457b, mlx 0.31.2) — on épingle `mlx-vlm`,
  versions à confirmer alignées en Step 0c.
- wired limits : .29 = 471040 (~460 G) ; .30/.31/.32 = 204800 (200 G) —
  non persistants au reboot, pre-flight à chaque load.
