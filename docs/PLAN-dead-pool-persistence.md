# PLAN — Dead-pool persistence bug (handoff Fable → Opus)

> Branche `main`. Prod `.39` en **v1.13.0** (déployée, saine, 0 pool Argo chargé
> actuellement). Fable a shippé la moitié du chantier « dead-pool » (v1.13.0) et
> se fait bloquer par les safeguards sur le diagnostic. **Opus reprend et finit.**
> Contexte docker `.39` = `desktop-linux`. Après tout `compose up`/recreate,
> re-cp `api.py` ET `dashboard.html` (gotcha [[procedures_deploy_commit]]).

## Le Goal

**Un pool ne doit JAMAIS disparaître silencieusement de la persistance quand ses
ranks meurent.** Aujourd'hui : ranks morts → purge → state réécrit sans le pool
(ou fichier state supprimé) → un restart ne ramène rien. Une mort transitoire
(reboot node, OOM, crash JACCL) efface définitivement l'intention de servir.

## Root cause — CONFIRMÉE

Deux défauts qui composent, tous deux dans `scripts/api.py` :

1. **`_purge_dead_pools` (l.9666-9690) traite « ranks morts » comme « unload
   opérateur ».** Quand `alive_count()==0` il `del_pool()` PUIS
   `save_cluster_state_v2(cluster_id)` (l.9687). Le state persisté reflète alors
   le registre VIVANT — le pool mort en sort → **le restart suivant ne le
   restaure pas.** Or mort ≠ unload : l'unload = intention opérateur d'arrêter
   (doit persister le retrait) ; la mort = panne (doit persister le pool comme
   « voulu, actuellement down »).

2. **`save_cluster_state_v2` (l.2669) `sf.unlink()` le fichier quand
   `pools_payload` est vide (l.2727-2729).** Si la purge vide le registre (dernier
   pool mort), le fichier state est **supprimé** → au boot, restore lit du vide →
   **zéro pool ramené, définitivement.** C'est ce qui a effacé l'état ce matin
   (state ABSENT après le restart v1.13.0, 0 pool Argo dans `/v1/models`).

**Chronologie prouvée (logs `docker logs -t`, 2026-07-02 soir) :**
- 22:20:06 ornith keepalive 2/2 miss → WU3 recovery FIRED → reboot .30/.31/.32.
- 22:20:36 `[purge] main[ornith]: every rank had exited` (le reboot WU3 a tué les
  ranks) → `save_cluster_state_v2` réécrit le state SANS ornith.
- 22:21:43 WU3 recovery reload ornith → re-persiste. **Course** entre le
  dead-pool-sweeper (30s) et la recovery ladder : le sweeper purge+sauve pendant
  la fenêtre où les ranks sont down avant que la ladder ne recharge.

## Ce qui MANQUE — le diagnostic LIVE à faire d'abord (BLOQUANT #1)

Après le restart v1.13.0 de ce matin, **NI le VL NI ornith n'ont été restaurés**
et le fichier state est ABSENT. Il faut reproduire et isoler la cause exacte
AVANT de coder le fix, sur l'**engine dev local `:8010`** (jamais sur `.39` pour
le diagnostic) :

```bash
SC=<scratchpad>   # topology-dev.yaml + dev-state/ déjà là
cd "scripts" && ODYSSAI_X_TOPOLOGY=$SC/topology-dev.yaml ODYSSAI_X_STATE_DIR=$SC/dev-state \
  ODYSSAI_X_VLM_DISTRIBUTED_ENABLED=1 python3 api.py --host 127.0.0.1 --port 8010
```

Repro : charger 2 pools (un VL single-node + un texte) → `kill` un rank à la main
→ observer si (a) le dead-pool-sweeper purge+unlink, (b) le wired-guard v1.13.0
(l.~3540, `_leaked_hosts`) SKIP faussement au restore, (c) le shutdown concurrent
(l.~3630) tue le serveur VL nohup de sorte que `_restore_vlm_pool` (l.12463) ne
le ré-adopte pas. **Isoler laquelle des trois** — chaque hypothèse a un fix
différent, ne pas empiler.

Hypothèse la plus probable (à confirmer, pas à supposer) : le shutdown v1.13.0
appelle `pool.stop()` sur le VL → tue le `mlx_vlm.server` nohup ; au restore,
`_restore_vlm_pool` probe `.29:8080` DOWN → tente `_launch_vlm_server` → si ça
échoue (timeout/venv/OOM), le pool n'entre pas au registre → au 1er tick du
dead-pool-sweeper, registre `main` vide → `save_cluster_state_v2` **unlink**.

## Le fix — DESIGN (à valider par le diagnostic ci-dessus)

Séparer **INTENT (desired state)** de **LIVENESS (registre vivant)**.

- **F1 — `save_cluster_state_v2` ne DOIT JAMAIS unlink sur registre vide sauf
  unload explicite.** Paramétrer : `save_cluster_state_v2(cluster_id, allow_empty_delete=False)`
  par défaut. Seul le chemin unload opérateur (l.11960, l.12037) passe
  `allow_empty_delete=True`. Purge (l.9687), auto-unload TTL, dead-pool-sweeper →
  `False` : ils réécrivent l'état des pools VIVANTS **sans supprimer le fichier**,
  et surtout **sans retirer un pool mais désiré** (voir F2).

- **F2 — `_purge_dead_pools` ne persiste PAS le retrait.** Il `del_pool()` du
  registre vivant (pour libérer les node indices + le routing) MAIS n'appelle
  PAS `save_cluster_state_v2` — le desired-state sur disque reste intact, donc un
  restart re-tente le pool. Option robuste : tombstone `"down": true` dans
  l'entrée state au lieu de la retirer, pour que le dashboard montre « down,
  needs-attention » plutôt que « disparu ».

- **F3 — restore résilient (l.3591-3610 + les branches VLMDist/VLM/text).** Un
  pool qui échoue à `start()` au boot GARDE son entrée desired-state (log
  `needs-attention`), ne l'efface pas. Optionnel : une boucle de retry bornée
  (N tentatives espacées) branchée sur la recovery ladder existante `#40`.

- **F4 — tuer la course sweeper↔recovery.** Le dead-pool-sweeper ne doit PAS
  purger un pool dont le cluster est `_cluster_is_degraded` OU sous
  `_WATCHDOG_RECOVERY_BY_CLUSTER` (la ladder est en train de le récupérer). Skip
  ces pools dans `_purge_dead_pools` / `_dead_pool_sweeper` (l.3650).

- **F5 (lié à J1 déjà partiellement fait) — le VL nohup ne devrait pas être tué
  par un simple restart.** Si le diagnostic confirme que le shutdown concurrent
  tue le serveur VL, EXCLURE les `VLMPool` (nohup, survit au container) du
  `pool.stop()` de shutdown — ils doivent être ré-adoptés en place au boot, pas
  relancés. (Les `RunnerPool`/`VLMDistPool` = enfants ssh, eux DOIVENT être
  stoppés.)

## Vérif (empirique, sur dev `:8010` d'abord, `.39` seulement après GO Sophie)

1. Charger 2 pools → `kill` un rank → le sweeper purge du registre MAIS le
   fichier state garde l'entrée (tombstone `down`) → restart → le pool est
   re-tenté (pas disparu).
2. Charger 1 seul pool → tuer son rank → registre vide → **le fichier state
   existe toujours** (pas d'unlink) → restart → re-tenté.
3. Restart avec VL single-node chargé → le `mlx_vlm.server` survit → ré-adopté
   sous le même alias, **zéro reload du modèle**.
4. Un node en fuite wired (>seuil) → le pool sur ce node n'est PAS respawné mais
   son entrée desired-state est CONSERVÉE (`needs-attention`), pas effacée.
5. Non-régression : unload opérateur explicite → le fichier state EST bien
   supprimé/vidé (l'unlink légitime marche encore).

## Déjà FAIT dans v1.13.0 (ne pas refaire)

- **#40 WU1 débloqué** : `_pool_reload_request` pose `force_hot_swap=True` → le
  409-spam préventif (695 échecs/17h) est STOPPÉ (vérifié : 0 en prod sur 3 min).
  Backoff 30 min sur reload échoué. **Ce volet est clos.**
- **#40 WU4** : override backend par pool (`ArgoLoadRequest.backend` "ring"|"jaccl"),
  `remote_cmd` émet le hostfile ring, `runner.py` lit `RUNNER_BACKEND`, persisté.
  Ring pools skippés par la stability loop. Déployé (runner.py sur les 5 nodes).
- **#61 groundwork** : le proxy VL enregistre/tick/finalise les runs → activity
  par pool voit les générations VL. (Le rendu dashboard stage+tokens reste à
  finir, cf J2.)
- **#66 shutdown/restore v1** : shutdown concurrent borné + wired-guard au restore
  + grace SIGTERM tunable. **MAIS** c'est précisément ce lot qui a introduit/révélé
  le bug dead-pool ci-dessus → F1-F5 le corrigent proprement.
- Issue FJ **#66** ouverte (le shutdown/restore) — y rattacher le fix dead-pool
  ou ouvrir une issue dédiée « persistence: desired-state vs live registry ».

## Backlog Goal-de-nuit restant (après le dead-pool)

- **J2 — #61 finir le rendu** : dans `renderArgoPoolsList` (dashboard.html ~l.4100),
  afficher par pool le stage (profile/generating/answering) + tokens live, façon
  cartes Telemak. Le backend expose déjà les runs par `pool_alias`
  (`/admin/runs`) ; source du stage Telemak = `_maybe_update_phase`/`_PHASE_MARKERS`
  côté runner-log. `_runs_tick` approxime les tokens (len/4) — exposer ntoks exact
  si runner.py le publie.
- **J3 — #40 reste** : (1) valider le PIVOT empirique (unload+load contrôlé reset
  le RDMA SANS reboot → 0 errno 16/96/2) — si FAUX, WU1 devient reboot préventif ;
  (2) documenter `backend:"ring"` par pool comme option long-run ; (3) critère
  zéro-orphelin = le fix dead-pool ci-dessus. Endurance >48h = observationnelle.
- **J4 — #29 cleanup (Difficulty 3)** : archiver `inference.py`/`inference_pipe.py`
  (init jaccl top-level, 0 callers) ; `__import__("re")`→`re.compile` (api.py
  ~5279) ; borner `_config_json_cache` (OrderedDict 200 + TTL 1h) ; ttl-sweeper
  collecter les cids PUIS unload (pas de mutation mid-iteration) ; master.py
  `with open` + un seul stop + queue vs busy-poll. Close #29.
- **J5 — #28 split api.py (EN DERNIER, sur code stabilisé)** : 6 seams dans
  l'ordre de l'issue (state.py → runner_pool.py → cluster_config.py →
  cloud_proxy.py → router.py → admin_api.py), api.py = wiring <1.5k lignes, ZÉRO
  changement de comportement, smoke après chaque seam.

## Invariants (ne pas rouvrir)

- Prod `.39` desktop-linux, deploy hot-patch scp+docker cp (+restart si api.py),
  bump `APP_VERSION`, verify `/admin/version` bloquante.
- Nommage pool = slug nu (garde anti-collision), badge VLM pool-level.
- Ring d'abord pour le VL distribué ; JACCL = perf, la phase suivante.
- Diagnostic dead-pool sur dev `:8010` d'abord — `.39` seulement sur GO Sophie.
