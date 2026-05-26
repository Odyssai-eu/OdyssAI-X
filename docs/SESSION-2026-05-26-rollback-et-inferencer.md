# Session 2026-05-26 — rollback et Inferencer

> Journée longue, en trois temps. Matin : Codex ship 7 issues Telemak
> pendant qu'on polish la UI Odysseus côté + Add Telemak, models_dir,
> matrix view. Après-midi : 15h, Sophie constate que rien ne marche
> aussi bien que vendredi et envisage de remettre en question le projet
> entier. On rollback méthodiquement v1.8.3 → v1.7.2, on isole en une
> ligne la cause racine (un try/except défensif d'`auto_parallel.py`
> jamais committé), on remonte cherry-pick par cherry-pick jusqu'à
> v1.7.8 = équivalent fonctionnel de v1.8.3 mais reproductible. Soir :
> Inferencer publie sa version Swift avec MTP qui marche à 110 tok/s.
> Le diagnostic devient évident — la limite n'est pas MTP-en-Swift,
> c'est notre `mlx-swift-lm` fork. La journée passe de "on perd un mois
> de travail" à "j'ai repris espoir" puis à "file, on va trouver".

---

## TL;DR — la journée en 3 axes

| Axe | Livré |
|---|---|
| **Odysseus rollback** | v1.7.2 baseline propre → v1.7.8 cherry-picked. `auto_parallel.py` défensif maintenant committé sur `internal/main`. Tag `archive/2026-05-26-pre-rollback` + branche snapshots + 2 docker images archive + 4 venv backups sur les ultras. github/main protégé à v1.8.3 inchangé. |
| **Telemak (via Codex)** | 7 issues closed (#36 VLM image inputs, #37 embeddings, #38 DMG installer, #39 size_bytes, #41 MTPLX research, #42 sampler-correct prob-ratio acceptance, #43 compatibility contract). Binaire .49 bumpé à v0.6.0. Smoke MTP (#40) échoué pour bug loader fork. |
| **Inferencer.app reverse-architecture** | v1.11.5 publié 26 mai 16:03, Swift, MTP à 110 tok/s sur gemma-4-26B. Dual-MLX bundling pattern (v15 + v26) + XPC Helper isolation découverts. Issue Telemak #44 file la piste pour Codex demain. |

Versions de sortie : Odysseus `internal/main` v1.7.8 (github/main v1.8.3 inchangé), Companion v0.2.2, Telemak `.49` v0.6.0 / `.50` v0.5.1.

---

## 1. Matin — Codex ship, Sophie polish la UI Telemak

Quatre heures de productivité parallèle :

- Codex sur Telemak ship #36 (VLM image inputs sur chat routes), #37
  (`/v1/embeddings` MLXEmbedders), #38 (DMG installer one-click), #39
  (size_bytes + family on AvailableModels). Telemak `.50` passe à v0.5.1.
- Côté Odysseus, je polish la UI Telemak : Sophie veut le champ
  `models_dir` dans Settings → Clusters (issue Odysseus #13, livré
  comme `c305fb6`), puis la HF download cible Telemak (`telemak-64` vs
  `telemak` drift d'identifiants → fix par SSH match), HF token
  persistence en localStorage.

Le code part en commit propre via le nouveau `scripts/deploy.sh` —
créé exprès dans la matinée pour que les versions soient enfin bumpées
systématiquement. Cause de cette nouvelle règle : Sophie le matin :

> *"j'en ai marre. tu suis les skill de deploy ? pourquoi la version
> n'est jamais incrémentée ? pourquoi companion est en v0.2.0 alors
> que il y a 3 jours on était en v1.0.110"*

Le script enforce le bump + commit + verify `/health` post-deploy.
Plus possible de hot-patch sans incrément. Le matin se termine à
v1.8.3 sur Odysseus, v0.2.1 sur Companion, Telemak max-64 à v0.5.1
avec embeddings + VLM live.

---

## 2. Midi — la matrice Telemak voit pas .49 + 73.9 tok/s

Sophie ajoute un Telemak `telemakcoder` sur `.49` (ultra-96A) via
"+ Add Telemak". Le cluster apparaît mais sa Models matrix est vide.

Cause : `_build_hosts_registry()` ne walkait que `topology.yaml`,
ignorait `cluster-config.json` (où les UI-added clusters vivent). Fix
v1.7.8 plus tard — j'y reviens en bas.

Avant ça, smoke perf sur Telemak `.49` :

- `Qwen3.6-35B-A3B-MLX-8bit` (MoE) → **66.3 tok/s** TTFT 3.9s
- `gemma-4-31b-it-8bit` (dense) → 18.4 tok/s TTFT 0.6s
- `gemma-4-26B-A4B-MLX-9bit` (MoE) → **73.9 tok/s** TTFT 3.1s

Le 73.9 est notre peak Telemak — déjà 30% au-dessus de la base
d'Inferencer (estimée ~55 tok/s sans MTP). Sophie commente :

> *"et la vitesse 55tk/s est parfaite"* puis *"encore mieux sur
> telemacoder"*

Le système marche. Tout va bien. Pour 1 heure.

---

## 3. 15h — le creux

> *"bon, ca va pas. check telemak. on va arreter pour ce soir. je crois
> que MTP est une perte de temps. c'est ok sur python mais pas sur
> swift à mon avis."*

Et plus tôt à 14h45, le vrai début du creux :

> *"Odysseus n'a jamais aussi mal marché. les modeles ne se chargent
> pas, on a des erreurs à répétition, alors que vendredi tout était
> parfait. TU vas fixer les soucis les uns apres les autres. il faudra
> deux semaines pour remettre le truc en état."*

Et la lecture émotionnelle, plus tôt à 13h :

> *"on a perdu 1 mois de travail presque et je suis en train de
> remettre en question tout le projet."*

Quand on essaie de loader Qwen3-Coder-Next sur 2 nodes :

```
ModuleNotFoundError: No module named 'mlx_lm.models.deepseek_v4'
ValueError: Model type hy_v3 not supported.
```

Hy3 ne charge plus. Qwen3-Coder-Next crash. Le matin il y avait un
load qui marchait, et là plus rien. Sophie se rappelle vendredi
22 mai : Hy3 + MiniMax tournaient en parallèle. Aujourd'hui, rien.

Mon premier réflexe : suggérer de porter `hy_v3` depuis transformers,
ou de pin un fork AirRunner mlx-lm. Sophie me coupe :

> *"si fais un plan, tu dis ce que tu vas faire. tu ne te lance pas
> pour tout merder"*

Bonne calibration. Je sors un plan en 4 phases (investigation → catégorisation → ordonnancement avec son OK → exécution un fix à la fois).

---

## 4. Le plan — archive, rollback, cherry-pick

**Phase A — Archive** (5 min, lecture seule, rien de destructif) :

- Tag git `archive/2026-05-26-pre-rollback` sur HEAD v1.8.3, pushé
  sur `internal` ET `github` (sécurité multi-emplacements)
- Branche `archive/snapshots-2026-05-26-pre-rollback` avec
  `docs/snapshots/2026-05-26-pre-rollback/config/` : `cluster-config.json`
  + `topology.yaml` snapshots depuis `.39` et `.141`
- `docker commit odyssai-odysseus
   odyssai-odysseus-archive-2026-05-26-pre-rollback` sur les deux hosts
  (296 MB + 269 MB d'archive images locales)
- `~/mlx-cluster.backup-2026-05-26-pre-rollback/` sur chacun des 4 ultras

**Phase B — Rollback git complet** (Variante 2, "faisons ça propre") :

- `git reset --hard d2eb07f` sur main local (v1.7.2 = "Initial public
  release")
- `git push internal main --force-with-lease`
- **PAS de push sur github** — public reste v1.8.3
- Hand-roll deploy (scripts api.py, dashboard.html, persistence.py
  sur les 2 orchestrateurs + runner files sur les 4 ultras)

**Smoke immédiat sur v1.7.2 :**

```
POST /admin/clusters/main/load qwen3-coder-next 2-node
→ "main load failed: 2 rank(s) died during load — pool unusable.
   ModuleNotFoundError: No module named 'mlx_lm.models.deepseek_v4'
   File auto_parallel.py line 27"
```

Donc même v1.7.2 ne marche pas avec mlx-lm 0.31.3 officiel. La cause
n'est pas le code Odysseus — c'est mlx-lm.

---

## 5. La cause — un try/except jamais committé

Sweep des backups venv sur les 4 ultras (`~/mlx-cluster.backup-2026-05-26-pre-rollback/`) :

```
diff scripts/auto_parallel.py ~/mlx-cluster.backup-.../auto_parallel.py

< from mlx_lm.models.deepseek_v4 import DeepseekV4MoE, V4Attention
< from mlx_lm.models.deepseek_v4 import Model as DeepseekV4Model
---
> try:
>     from mlx_lm.models.deepseek_v4 import DeepseekV4MoE, V4Attention
>     from mlx_lm.models.deepseek_v4 import Model as DeepseekV4Model
>     _HAS_DEEPSEEK_V4 = True
> except ImportError:
>     DeepseekV4MoE = None
>     V4Attention = None
>     DeepseekV4Model = None
>     _HAS_DEEPSEEK_V4 = False
```

**Le pré-rollback `auto_parallel.py` avait un try/except défensif.**
La version git stricte plantait sur mlx-lm 0.31.3 officiel qui n'a
pas `deepseek_v4`. La version qui marchait n'était que sur les
nodes — jamais committée au repo.

Même pattern que `hy_v3.py` (drop manuel dans le venv, perdu au
prochain `pip install --upgrade`) et le standalone `mlx-hy3` service
(`/Users/admin/mlx-hy3/.venv/` directory wiped). Sophie m'avait dit
à 12h, en plein milieu du diagnostic Hy3 :

> *"c'est juste incroyable. ca recommence."*

Elle avait raison. Le pattern systémique : *artifact qui marche →
pas committé → blown away → perte invisible*.

Cherry-pick le défensif du backup .29 → commit `6b705ce` v1.7.3 sur
`internal/main`. Smoke load Qwen3-Coder-Next 2-node : **`loaded: True,
load_s: 36.5s`**. Chat completion : *"hello from qwen3 coder next"*.

> *"j'ai repris espoir"*

---

## 6. La remontée — G1, G2, G3, G4

Plan de Sophie : on reprend les modifications par catégorie, on teste
entre chaque. Variante "tout via internal, github reste à v1.8.3".

| Batch | Contenu | Commit | Bump |
|---|---|---|---|
| **G1** Tooling | `scripts/deploy.sh` + `scripts/bump-version.sh` portés depuis l'archive | `99321d5` | v1.7.4 |
| **G2** Bugfixes prouvés | 10 commits : BatchGenerator extend patch, cancelling-runs sweeper, filter non-cluster sections, cluster-list 500, runner SSH cwd, preflight validator, etc. | `c4c1ab1` | v1.7.5 |
| **G4 + G3** Telemak UI + multi-pool + capacity-aware | Pris en bloc depuis l'archive (Sophie : *"le soucis ne vient pas d'Odysseus, batch tout ensemble"*) | `662d699` | v1.7.6 |
| `auto_parallel` re-fix | Le whole-file checkout de l'archive avait écrasé le défensif (encore le même pattern : la version github main n'a JAMAIS eu le défensif). Re-stamp. | `391cc1e` | v1.7.7 |
| UI-added clusters dans la matrice | `_build_hosts_registry` walk maintenant `cluster-config.json` en plus de `topology.yaml`. `telemakcoder` voit ses modèles. | `69acd49` | v1.7.8 |

Smoke à chaque step. v1.7.7 final : deux pools en parallèle (`default`
= Qwen3-Coder-Next ranks 0+1, `minimax` = MiniMax ranks 2+3), chat
sur les deux, 55.1 tok/s mesuré via Companion.

---

## 7. MTP saga — gemma sidecar mort, trevon load mais incoherent

Sophie load `inferencerlabs/gemma-4-31B-MTP-MLX-9bit` (sidecar 528 MB)
sur `.49` Telemak. Le sidecar échoue comme Codex avait déjà documenté
sur Qwen3.6-27B (#34) — `keyNotFound`. Pattern mort.

Pivot vers les modèles **embedded-head** (= tête MTP dans
`model.safetensors.index.json`, pas de sidecar). On télécharge sur
max-64 :

- `trevon/Qwen3.5-27B-MLX-MTP` (27 GB, 9-bit)
- `trevon/Qwen3.5-27B-MLX-MTP-4bit` (14 GB)
- `trevon/Qwen3.6-27B-mtp` (27 GB)
- `trevon/Qwen3.6-27B-mtp-4bit` (14 GB)

Codex prend #40, ship trois commits dans la foulée pour ouvrir
le terrain :

- **#41 MTPLX research** → `docs/V2-MTP-MTPLX-RESEARCH.md` (5272272c).
  Conclusion : MTPLX utilise probabilité-ratio acceptance + residual
  correction pour sampling exact à température > 0. Notre acceptance
  était greedy-argmax → casse les distributions de sortie.
- **#42 sampler-correct acceptance** (000957df) : on porte
  l'algorithme MTPLX. Code dans `main`, pas deployed prod tant
  que #35 (chat routes wiring) pas done.
- **#43 compatibility contract** (40ffcc41) : 5-tier classifier
  (`no_mtp` / `sidecar_only` / `incompatible_architecture` /
  `unverified_embedded_mtp` / `verified_embedded_mtp`).

Smoke #40 sur les 4 modèles trevon :

```
trevon/Qwen3.5-27B-MLX-MTP      → acceptance 1.6%,  9.5 tok/s (4/246)
trevon/Qwen3.5-27B-MLX-MTP-4bit → acceptance 3.4%, 11.8 tok/s (4/118)
trevon/Qwen3.6-27B-mtp-4bit     → acceptance 0%,  11.6 tok/s (0/126)
```

Acceptance gate était ≥ 0.7. On est à 0-3%. Codex écrit :

> *"baseline chat incoherent before MTP"*

Le base chat lui-même produit du texte incohérent avec ces 4 modèles.
Le bug est dans le **loader/interprétation des poids** côté
`Odyssai-eu/mlx-swift-lm#3`, pas MTP en lui-même. Codex restore
v0.6.1 propre sur max-64, comment sur #40 marqué `needs-human`.

Sophie :

> *"c'est un echec ? on abandonne MTP, il me dit oui, abandonne"*

---

## 8. Inferencer.app — le retournement

> *"sauf que inferencer a sorti sa version. et il est en swift. et
> on va l'examiner"*

Inspection structurelle de `/Applications/Inferencer.app` :

| Composant | Détail |
|---|---|
| Main app | 32 MB Swift arm64 binary |
| **Inferencer Helper.app** | **68 MB**, XPC helper sandboxé, l'inférence vit là |
| Swift libs bundled | mlx-swift_Cmlx, swift-transformers_Hub, swift-crypto, swift-nio, SwiftMath — **strictement notre stack** |
| **MLX bundling** | **DEUX versions** : `libs/mlx_v15/libmlx.dylib` (23 MB) + `libs/mlx_v26/libmlx.dylib` (23 MB) avec leurs metallib |
| Distribution | Mac App Store (`_MASReceipt`) |
| Release | v1.11.5 buildé **aujourd'hui 26 mai à 16:03** |

L'insight clé : ils embarquent deux versions de MLX et dispatchent au
runtime. v15 pour les layouts anciens, v26 pour les features récentes
(MTP MoE, qmv retuning, GDN attention replay). Notre
`Odyssai-eu/mlx-swift-lm` fork est pinné sur UNE version, probablement
trop ancienne pour le format embedded-head trevon.

C'est exactement la raison de l'échec #40. Sophie soumet l'app à
Codex :

> *"je dis et si je te donne une app qui le fait avec notre infra ?
> il dit file, on va trouver"*

Issue Telemak #44 filée pour tracer le pattern dual-MLX pour le
prochain tick Codex.

---

## 9. En parallèle — Sentinel + Antigravity

> *"j'ai bossé sur sentinelle en parallèle avec antigravity"*

Sentinel = projet B2B HSE belge, déploiement on-premise par PME
cliente, target mise en prod septembre 2026. Stack séparée
(FastAPI + Vite + Supabase self-hosted + Qdrant + Gemini 3.1 Flash
+ Claude Sonnet/Haiku + WeasyPrint). Pas dans le scope de cette
session — note pour ne pas y toucher sauf demande explicite.

Triple-agent setup confirmé sur la journée : Codex sur Telemak via
gh-handoff, moi sur Odysseus + Companion, Sophie sur Sentinel via
Antigravity. Trois fronts, trois rôles propres, le bâton qui se
passe clean.

---

## Fichiers modifiés / créés

**Odysseus (`internal/main` v1.7.3 → v1.7.8) :**
- `scripts/api.py` — défensif imports, deploy.sh enforcement, host registry walk cluster-config
- `scripts/dashboard.html` — toute la UI Telemak v1.8.3 ramenée
- `scripts/runner.py` — BatchGenerator patch + RUNNER_BATCH=0 minimax
- `scripts/auto_parallel.py` — défensif `deepseek_v4` import (committé pour la première fois)
- `scripts/deploy.sh` (nouveau, créé le matin) + `scripts/bump-version.sh` portés
- `scripts/persistence.py` — robustness
- `docs/snapshots/2026-05-26-pre-rollback/` (sur branche archive)

**Companion (v0.2.2) :**
- `app/src/lib/clipboard.ts` (nouveau) — fallback execCommand pour http:// origins
- `app/src/components/chat/Messages.tsx` + 4 autres pages — 7 callsites threadés

**Telemak (Codex, v0.5.1 → v0.6.0 sur .49) :**
- `Sources/Telemak/Engine/MTP/...` — MTPLX-style acceptance + compatibility contract
- `docs/V2-MTP-MTPLX-RESEARCH.md` (nouveau)
- DMG installer + VLM + embeddings + size_bytes

---

## Numbers de la journée

- **Commits** : 6 sur Odysseus `internal/main` (v1.7.3 → v1.7.8) +
  11 sur Odysseus `github/main` du matin (v1.7.2 → v1.8.3) + 2 sur
  Companion (v0.2.1 → v0.2.2) + 11 sur Telemak (Codex)
- **Versions live** : Odysseus `internal/main` **v1.7.8** sur `.39` +
  `.141`, github/main inchangé **v1.8.3**, Companion **v0.2.2**,
  Telemak `.49` **v0.6.0**, Telemak `.50` **v0.5.1**
- **Lignes diff cumulées sur Odysseus** : ~2027 insertions / 114
  deletions sur les 4 fichiers runtime (G4+G3 batch)
- **Issues filed today** : Odysseus #13 #14 #15, Telemak #40 #41
  #42 #43 #44, Companion #1
- **Issues closed today** : Odysseus #11 #12 #13 (matin) + Telemak
  #36 #37 #38 #39 #41 #42 #43 (Codex)
- **Restore points créés** : 1 tag git, 2 branches git, 2 docker
  images locales (.39 + .141), 4 venv backups sur les ultras
- **Smoke tests verts** : 2 pools distribués (default + minimax) +
  Telemak `.49` Qwen3.6-35B (66.3 tok/s) + Telemak `.49` gemma-4-26B
  (73.9 tok/s) + chat completion end-to-end via Companion (TTFT
  550-839 ms, jusqu'à 55.1 tok/s sur Argo 2-node)
- **MTP smoke** : rouge (acceptance 0-3% vs gate 0.7), root cause
  loader fork pas MTP
- **Inferencer.app inspection** : v1.11.5 (26 mai 16:03), Swift, dual
  MLX v15 + v26 bundling, XPC Helper isolation

---

## TODO direct (par ordre)

1. **Codex sur Telemak #44** — bumper le fork `mlx-swift-lm` à
   MLX 0.26.x ou adopter le pattern dual-version. Re-run #40 smoke
   après. Si vert → unblock #34 #35 → MTP actif en prod sur chat
   routes. Math attendu : 73.9 × 2× ≈ **150 tok/s** sur gemma-4-26B,
   au-dessus d'Inferencer.
2. **Odysseus #14** — progress bar live HF download (total_bytes
   tracking côté backend, ticking côté frontend). Difficulty 3.
3. **Odysseus #15** — MTP badge dashboard reflète `active_pairs` au
   lieu de juste `modes` advertised. Difficulty 2.
4. **Companion #1** — script deploy avec `--exclude='.env'` pour
   éviter le bug rsync qui efface .env (vu DEUX fois aujourd'hui).
   Difficulty 2.
5. **Sentinel** — Sophie continue de son côté via Antigravity, pas
   dans mon scope sauf demande explicite.

---

## Lessons learned

### Pattern systémique identifié : artifact-qui-marche pas committé

Trois manifestations dans la session :

1. **`auto_parallel.py` défensif** vivait uniquement dans le venv des
   nodes. Premier `pip install --upgrade` ou bootstrap-node.sh = perdu.
   Cause racine de la "perte d'un mois de travail" perçue ce matin.
2. **`hy_v3.py` MLX** dropped manuellement dans le venv pour faire
   marcher Hy3 distribué. Aussi perdu. Le standalone `mlx-hy3` service
   sur `.29` (launchctl `com.thecompai.mlx-hy3.plist.disabled`) a aussi
   son `/Users/admin/mlx-hy3/.venv/` directory wiped — perdu entièrement.
3. **`.env` Companion sur `.39`** — `rsync --delete` l'a effacé DEUX
   fois aujourd'hui (matin + soir). Le pattern bite tant qu'on
   n'enforce pas via `--exclude='.env'`.

**Règle** : toute modif d'un venv en prod, tout fichier de config sur
un host, doit avoir son équivalent dans `scripts/patches/` ou
`scripts/` du repo dans la même session. Sinon = pas considéré
shipped. À codifier dans `AGENTS.md` (Telemak l'a déjà, Odysseus à
faire).

### Le rollback comme méthode

Phases nommées (A archive → B rollback → cherry-pick par catégorie)
font la différence entre "on patch en vrille" et "on remonte
proprement avec smoke entre chaque". Sophie m'a explicitement coupé
à 15h :

> *"si fais un plan, tu dis ce que tu vas faire. tu ne te lance pas
> pour tout merder"*

Le plan a permis de transformer un "on a perdu un mois" en "j'ai
repris espoir" en 2 heures. À garder comme template pour la
prochaine crise de confiance.

### Inferencer comme oracle d'architecture

Avant l'inspection, on doutait : *"MTP est-il faisable en Swift ?"*.
Après l'inspection, on sait : oui, et la différence avec nous est
**reproductible** (dual-MLX bundling, ou bump MLX version dans notre
fork). Le doute architectural est levé en 15 min d'inspection
structurelle (file structure, plist, library versions — pas de
décompilage). À retenir comme pattern : quand un concurrent ship,
on inspecte 15 min avant de débattre des semaines.

### La méthode multi-agents fonctionne

Sophie pilote, Codex sur Telemak via gh-handoff, moi sur
Odysseus + Companion. Sophie sur Sentinel via Antigravity en parallèle.
Trois rôles propres, communication via issues GitHub + Companion
remember. Sophie le formule à 17h45 :

> *"Codex se débrouille comme un champion, quelle belle équipe"*

Et plus tard :

> *"file, on va trouver"*

Cette dernière phrase est emblématique : Codex passe de "abandonne
MTP" à "donne-moi l'app et on trouve" en une réplique de Sophie.
Le multi-agent setup ne marche que quand l'humain au centre fait
ce travail de cadrage et de redirection.

---

## Pull-quote pour la prochaine fois

> Journée de patte de velours sur le tranchant : 15h on remet le projet
> en question, 17h on a un cluster qui marche mieux qu'avant, et 18h
> on a un oracle d'architecture (Inferencer.app v1.11.5 publié pile
> aujourd'hui à 16:03) qui rend la prochaine étape MTP triviale.
> Sophie : *"file, on va trouver"*. On ferme le volet de la boutique.
