# PLAN — MTP distribué (spéculatif natif multi-rang)

> Établi 2026-07-05 (session Fable de nuit, mandat = plan, pas code).
> **Rev. 3 après 2 rounds de review adversariale MiniMax-M3** — round 1
> REVISE (~15 findings arbitrés, F-4/F-18 réfutés puis RETIRÉS par le
> reviewer sur preuve source), round 2 : intégrations validées + 7 nits
> résiduels tous appliqués ici. Log : `PLAN-distributed-mtp-review.md`.
> Déclencheur : LongCat-2.0 (~1.78T params, MTP natif 3-step) exigera les
> 5 nodes — son MTP est inutilisable dans le path actuel. La solution vise
> AUSSI GLM-5.x et DeepSeek-V3.2 (mêmes heads natifs), pas que LongCat.
> Exécutant cible : Opus ou autre agent — chaque étape porte ses anchors,
> ses critères de done et ses canaris. Aucun fait ci-dessous n'est supposé :
> tout est vérifié source/poids en date du 2026-07-05.

---

## 0. But, portée, critère de succès

**But** : le décodage spéculatif par heads MTP **natifs** (dans le tronc, pas
de 2ᵉ modèle) fonctionne en serving **multi-rang** (tensor ET pipeline
parallel) dans le runner OdyssAI-X, en **greedy d'abord**, avec un gain
mesuré vs l'AR de base.

**Critère de succès final (G3)** : LongCat-2.0 Q4 sur 5 nodes, MTP D1-D3,
sortie greedy **token-for-token identique** à l'AR greedy, et
**tok/s ≥ 1.4× l'AR 5-node** du même modèle. Critère intermédiaire (G2) :
même invariant sur GLM-5.2 ou DeepSeek-V3.2 multi-rang, gain ≥ 1.25×.

**Hors scope v0** : sampling température>0 (phase 2, §E6) ; BatchGenerator
multi-rang ; MTP sur les pools VLM ; revival du path Swift Hy3 MoE-MTP.

**Décisions gelées héritées (NE PAS rouvrir)** :
- Base LongCat distribué D'ABORD — le MTP est un chantier séparé, follow-on
  du port. (Mais voir §3 : les phases E0-E3 n'ont AUCUNE dépendance LongCat
  et peuvent démarrer avant/pendant le port.)
- Gate empirique perf avant engagement : doit battre l'AR, sinon stop.
- Greedy/déterministe d'abord.
- Méthode canaris (nuit VLM) pour l'alignement.
- Panel-à-3 sur CE plan avant d'écrire du code.

---

## 1. Faits vérifiés (recon 2026-07-05 — anchors)

### 1.1 Runner OdyssAI-X (`scripts/runner.py`)

| Fait | Anchor |
|---|---|
| Draft-model spec gaté single-rank : chargé ssi `draft_repo and size == 1` | `runner.py:1890` |
| Multi-rang : « draft_model requested but size>1 … ignoring » | `runner.py:1902-1904` |
| BatchGenerator gaté `size == 1` (+ draft None + pas minimax-m3) | `runner.py:1931-1933` |
| Multi-rang = boucle legacy single-stream, `group` passé | `runner.py:1942-1945`, `_run_legacy_main:1969` |
| Point de gen : `stream_generate(model, tokenizer, gen_input, **gen_kwargs)` | `runner.py:2193` |
| Sampler par modèle : `_build_sampling_for` → `gen_kwargs["sampler"]` | `runner.py:2181-2185` |
| Draft passthrough existant : `gen_kwargs["draft_model"]=…, num_draft_tokens` | `runner.py:2189-2191` |
| Backend transport par pool : `RUNNER_BACKEND` jaccl\|ring | `runner.py:1818-1820` |
| Coordination multi-rang = fan-out JSONL engine→stdin de chaque rang, chaque rang recompute à l'identique, `emit()` gate stdout au rang 0 | design global runner/vlm_runner |

### 1.2 mlx-lm 0.31.3 (venv cluster `.29:~/mlx-cluster/.venv`, source vérifiée)

| Fait | Anchor (`site-packages/mlx_lm/`) |
|---|---|
| Boucle spéculative de référence : `speculative_generate_step(prompts, model, draft_model, num_draft_tokens=2, …)` | `generate.py:473` |
| Rollback : `_rewind_cache` → `trim_prompt_cache(model_cache, num_draft - num_accept)` + `trim_prompt_cache(draft_cache, max(num_draft-num_accept-1, 0))` | `generate.py:589-591` |
| Verify = UN forward du tronc sur `[y, draft_tokens]`, `_step(model, model_cache, y, num_draft+1)` | `generate.py:617-618` |
| Acceptance = exact-match : accepte tant que `token_target[n] == draft_token[n]` (déterministe en greedy) | `generate.py:623-651` |
| Exige cache trimmable : `can_trim_prompt_cache` sinon raise | `generate.py:529-532` |
| `KVCache.trim` / `QuantizedKVCache.trim` existent | `models/cache.py:378,309` |
| **Pipeline parallel : split INVERSE — rank 0 = dernières layers** | `models/deepseek_v32.py:426-439` |
| Flux PP : `recv_like(h, rank+1)` → slice → `send(h, rank-1)` | `deepseek_v32.py:460,467` |
| **`h = mx.distributed.all_gather(h)[: h.shape[0]]` en FIN de forward → TOUS les rangs ont le hidden final (celui du rank 0 = dernières layers)** | `deepseek_v32.py:473` |
| Chaque rang garde `embed_tokens` + norm + `lm_head` (seul `self.layers` est élagué) | `deepseek_v32.py:437-439,446` |
| Tensor parallel : logits identiques par rang après `all_sum` | `deepseek_v32.py:374` |

**Conséquence architecturale centrale** : en PP mlx-lm, *après chaque
forward*, tous les rangs détiennent le MÊME hidden final et calculent les
MÊMES logits localement. C'est l'invariant qui aligne déjà le sampling
multi-rang aujourd'hui. Donc **des heads MTP répliqués par rang draftent
localement à l'identique, sans AUCUN collectif nouveau**.

### 1.3 LongCat-2.0 — structure MTP réelle (index safetensors, pas le config)

Leçon mémoire qwen3_next appliquée : *l'index des poids fait foi*.
Index récupéré et parsé (88 780 clés) ; headers shards range-fetchés
(shapes exactes) — scratchpad `longcat-index.json`.

- **UN SEUL module MTP** : `model.mtp.layers.0.*` uniquement, malgré
  `mtp_num_layers: 3`. Les « 3 steps » = 3 applications chaînées du même
  module (`mtp_replicate_modules: true`). Confirmé par le README HF :
  « all 3 MTP draft steps share a single [indexing] pass ».
- **Le bloc MTP est DENSE** : `transformer_layer.mlp.{gate,up,down}_proj`
  [12288,8192] — AUCUN expert dans les clés mtp. **Le mur MoE-gather Hy3
  ne s'applique pas au draft LongCat.**
- Le module contient : MLA complète (q_a/q_b, kv_a 576, kv_b, o_proj) +
  **son propre indexeur DSA** (wk, wq_b, weights_proj, k_norm) +
  `embed_tokens` propre [163840,8192] + `eh_proj` [8192,16384] +
  `enorm`/`hnorm` (convention DeepSeek-V3) + `norm` final propre.
  **Pas de `mtp.lm_head`** → head de sortie partagé avec le tronc.
- **Tailles** : module complet = 136.94B params, MAIS 135B = les 16 tables
  n-gram `oe_embed_tokens0-15` ([~16.48M, 512] chacune). La partie
  *compute* : embed 1.34B + eh_proj 0.134B + attn ~0.13B + MLP 0.30B ≈
  **1.9B params**. Budget mémoire par rang SELON la politique de quantif
  de l'embed (à fixer dans la recette de conversion, pas à laisser
  flotter) : embed Q4 + reste Q4 ≈ **1.1 GB** ; embed bf16 (convention
  head-bf16) + reste Q4 ≈ **3.2 GB** ; tout bf16 ≈ 3.9 GB. Dans tous les
  cas : réplicable par rang sans douleur (les nodes portent ~200 GB de
  shard trunc). `mtp_disable_over_tokenizer: true` suggère que
  l'inférence MTP N'UTILISE PAS les tables oe → on ne les convertit/charge
  pas (à CONFIRMER source officielle, cf §7 V1).
- Trunk : 38 layers × 768 experts (top-12) + 128 zero-experts identity +
  dual-blocks (`input_layernorm.0/.1`, design ScMoE) + tables oe propres
  au trunk (135B) + `lm_head` [163840,8192]. Total ≈ 1.77T params — recolle
  avec les 3.55TB bf16.

### 1.4 Briques existantes réutilisables

- **MTPLX** (fork `Odyssai-eu/MTPLX`, upstream `youssofal/MTPLX`,
  Apache-2.0 — compatible one-way AGPL avec attribution NOTICE) :
  runtime MTP natif Python/MLX single-node, **2.24× @ temp 0.6** sur
  Qwen3.6-27B. Assets clés :
  `mtplx/backends/{deepseek_mtp,glm_mtp,mimo_mtp,qwen3_next}.py` (builders
  de module par famille), `deepseek_mtp_patch.py` (`inject_deepseek_mtp_support`
  l.294, `_make_deepseek_mtp_module` l.246, `_rewrite_kv_b_projection` l.118,
  quantize-on-load l.217), `generation.py` (loop + acceptance par depth +
  capture du verify-hidden `_verify_hidden_mode` l.522), rejection sampling
  exact (`acceptance_probability`) pour la phase 2.
- **Telemak** (`Sources/Telemak/Engine/MTP/MTPSpeculativeIterator.swift`,
  342 l.) : le design d'itérateur PROUVÉ (Qwen3.5 dense 1.71×). Contrat :
  `forwardWithHidden`, `targetVerify(verifyInput=[bonus, draft0..k-1])`
  → `(logits, hidden, rollbackBuffer)`, `rollbackSpeculativeCache(acceptedCount)`,
  greedy fast-path + probability-ratio (l.38-46, 192-221). Invariant à
  porter : le verify inclut TOUJOURS le bonus token ; `accepted` compte
  bonus + préfixe accepté.
- **vlm_runner.py / la nuit VLM** : fan-out JSONL, emit rang-0, **canaris
  sha par étage comparés vs baseline single-node** — la méthode de preuve.
- **runner.py `_run_legacy_main`** : reader-thread + cancel, session
  prefix-cache, stop-ids étendus — la boucle porteuse, à NE PAS réécrire.

### 1.5 Le mur Hy3 30× — requalifié

Le repro `/tmp/moegatherbench` a été purgé (tmp). Diagnostic consigné
(mémoires `hy3_moe_mtp_perf_wall`, session minimax-review) : le mur était
côté **Swift** — draft head MoE avec gather d'experts **bf16 non quantifiés**
(192 experts) + suspicion réutilisation KV trunk entre rounds. En Python
mlx-lm, le gather d'experts quantifiés (`gather_qmm`) est l'op que le trunk
exécute déjà 38-61×/token en prod — 1 layer MoE de plus par draft step est
marginal. **Et le draft LongCat est dense.** Le risque perf résiduel se
déplace sur : (a) le coût du verify multi-tokens sur trunc MoE (union des
experts routés), (b) les rebuilds de graphe Metal si les shapes varient.
Les deux sont mesurés aux gates G1/G2 (§5) — plus besoin de re-bencher le
path Swift.

---

## 2. Décisions d'architecture (D1-D8)

Chaque décision : choix → pourquoi → alternative écartée.

**D1 — Les heads MTP sont RÉPLIQUÉS sur tous les rangs.**
Fondement : `deepseek_v32.py:473` — le hidden final est all_gather'é sur
tous les rangs à chaque forward ; chaque rang a déjà embed/norm/lm_head.
Le module MTP (~1.1 GB Q4 pour LongCat) se charge partout ; le drafting
est un calcul **local, identique, sans collectif**.
*Écarté* : heads sur le dernier rang PP + broadcast des drafts — ajoute un
collectif par step, laisse les autres rangs idle, économise ~1 GB/rang.
Aucun avantage au vu des tailles.

**D2 — Verify = UN forward trunc standard sur `[bonus, d0..dk-1]` (k+1 tokens).**
Le pass pipeline/tensor existant porte déjà des seq_len arbitraires (le
prefill le fait). Aucune modification des collectifs : send/recv/all_gather
sont dimensionnés par la shape, identique sur tous les rangs. Le verify
retourne logits ET hidden (nécessaire pour seeder le round de draft
suivant) → on appelle `model.model(x, cache)` + norm/lm_head manuellement
(pas `stream_generate`).
*Écarté* : re-forward token-par-token des candidats (retombe au coût AR).

**D3 — Acceptance v0 = exact-match greedy (mlx-lm `generate.py:623`), déterministe.**
`argmax` de logits identiques → accept-count identique sur chaque rang
**par construction**. Pas de communication d'accord nécessaire. Le canari
(§6) le VÉRIFIE au lieu de le supposer.
*Phase 2 (E6)* : probability-ratio (MTPLX `acceptance_probability`) avec
graine RNG explicite par requête, broadcastée dans le JSONL fan-out —
protocole séparé, pas dans v0.

**D4 — Rollback = `trim_prompt_cache(model_cache, k - n)` par rang + trim du cache propre du module MTP.**
Sémantique vérifiée : trim retire n tokens de la FIN (`cache.py:378`).
n identique partout (D3) → trims cohérents. Le module MTP tient son
propre `prompt_cache` (MLA 1 layer — ~1.2 KB/token) trimé de
`max(k-n-1, 0)` (convention `generate.py:591`).
*Garde-fou* : `can_trim_prompt_cache` DOIT passer sur le cache du modèle
cible ; pour les caches custom du port LongCat (DSA/sparse), l'exigence
« trimmable » entre dans le contrat du port (§9). Fallback documenté si
un cache ne sait pas trim : rollback-buffer à la Telemak (capture/restore,
`MTPSpeculativeIterator.swift:41-46`) — mais on ne l'implémente QUE si
nécessaire.

**D5 — Le module MTP vit dans un fichier neuf `scripts/mtp_module.py`, buildé depuis MTPLX.**
`NativeMTPModule` : chargement des poids `model.mtp.*` depuis le dossier
modèle (filtre sur l'index safetensors), quantize-on-load aligné sur la
quantif du trunk (pattern `deepseek_mtp_patch.py:217`), chaînage
enorm/hnorm → concat → eh_proj → bloc transformer → norm → lm_head
PARTAGÉ (référence au `model.lm_head` du trunc, pas une copie).
API : `draft_step(tokens, prev_hidden, cache) -> (next_token_logits, hidden)`,
appliqué récursivement D fois (module unique, `mtp_replicate_modules`).
*Écarté* : monkeypatcher le modèle mlx-lm chargé (pattern inject de MTPLX)
— trop invasif pour le runner ; on garde le module À CÔTÉ du modèle,
le runner orchestre.

**D6 — La boucle vit dans `scripts/mtp_spec.py` : `native_mtp_stream_generate(...)`, générateur compatible `stream_generate`.**
Modelée sur `speculative_generate_step` (generate.py:473-654) : mêmes
conventions de rewind, mêmes invariants bonus-token que Telemak. Elle
yield des objets portant `.token`, `.text`, `.finish_reason`,
`.generation_tps`, `.prompt_tps` + `from_draft` — la boucle legacy du
runner (consommateur l.2193-2260) reste INCHANGÉE hors sélection du
générateur. Champs additionnels : `accept_rate`, `round_idx` pour les
métriques #61.

**D7 — Intégration runner : env `RUNNER_MTP` (off|native) pour le CHARGEMENT, champ par-REQUÊTE pour l'ACTIVATION.**
Dans `main()` : si `RUNNER_MTP=native`, charger `NativeMTPModule` après le
modèle (même device), **sur tous les rangs** (pas de gate size==1 — c'est
LE point du chantier). Mais l'env ne se change pas à chaud (lu au start du
process) → **le on/off effectif + depth voyagent PAR REQUÊTE dans le JSONL
fan-out** (`"mtp": {"on": true, "depth": 3}`, défaut = config pool). Le
fan-out étant identique sur tous les rangs, la décision est alignée par
construction. L'**auto-disable sur trip canari = l'engine cesse d'envoyer
`mtp.on`** aux requêtes suivantes du pool — zéro message de contrôle,
zéro restart, le pool reste up.
Le gate draft-model séparé (l.1890, 1902-1904) RESTE en place pour les
draft-models externes, avec une exception harness :
`RUNNER_SPEC_MULTIRANK=1` l'autorise multi-rang (E0, validation
d'alignement uniquement — jamais un défaut de prod).
Côté engine (`scripts/api.py`) : knob par pool `mtp: {enabled, depth}` dans
la config cluster (même plomberie que `RUNNER_BACKEND` #40 WU4), badge
dashboard + acceptance-rate dans l'activity pill #61 (shippée v1.13.2).

**D8 — Source du hidden pour le module MTP : flag `mtp_hidden_source: pre_norm|post_norm`, résolu depuis la source officielle, validé par canari d'acceptance.**
C'est LE piège connu (mémoire Mistral-EAGLE : acceptance ~0 sur suspicion
hidden PRE vs POST norm). Le module a son propre `hnorm` → l'entrée est
vraisemblablement le hidden PRÉ-norm final du trunc, mais on ne suppose
pas : lire l'impl SGLang/vLLM LongCat (V1, §7) et trancher. Le symptôme
d'une erreur est non-ambigu : acceptance ≈ 0 au lieu de ≥ 0.4.

---

## 3. Le déblocage de séquencement : E0-E3 sans LongCat

Le port LongCat (chantier séparé, PRIORITAIRE) et le MTP distribué ne se
bloquent PAS mutuellement :

- **E0** (harness d'alignement) tourne avec des modèles existants
  (Qwen3-8B + draft 0.6B) sur 2 nodes ring — zéro dépendance.
- **E1-E2** (module natif + boucle, single-node) tournent sur un modèle
  MTP-natif déjà sur disque (GLM-5.2 sur Argo, ou Qwen3.6-27B validé
  MTPLX) — zéro dépendance.
- **E3** (natif multi-rang) tourne sur GLM-5.2 (PipelineMixin, poids sur
  Argo, servi en prod régulièrement) ou DeepSeek-V3.2 — zéro dépendance.
- **E4-E5** (binding LongCat + gate final) attendent le port. Le port doit
  exposer le contrat §9 — à donner à l'agent du port AVANT qu'il fige ses
  interfaces.

Ordre recommandé : E0 → E1 → E2 → E3 (gates G0-G2), pendant/avant le port
LongCat ; puis E4 → E5 (G3) ; E6/E7 ensuite.

---

## 4. Phases détaillées

### E0 — Harness d'alignement multi-rang, modèle-agnostique — 3 pts

**But** : prouver l'invariant « accept-count identique par rang » et le
protocole canaris SANS module MTP, en réutilisant le spec draft-model de
mlx-lm tel quel.

1. `runner.py` : ajouter `RUNNER_SPEC_MULTIRANK` (défaut `0`). Modifier le
   gate l.1890 : `if draft_repo and (size == 1 or os.environ.get("RUNNER_SPEC_MULTIRANK") == "1")`.
   Le draft se charge alors sur CHAQUE rang (il est petit : 0.6B), log
   explicite `speculative MULTIRANK harness mode`.
2. Câbler les canaris (§6) dans le path draft existant : on ne peut pas
   instrumenter l'intérieur de mlx-lm → canari par TOKEN émis, pas par
   round, pour E0. **Point d'émission : la boucle consommatrice
   `for res in stream_generate(...)` (`runner.py:2193`)** — sha256
   cumulatif des token-ids, émis sur stderr toutes les 32 positions + à
   la fin.
3. Topologie test — **échelle graduée (pattern GATE-0 de la nuit VLM)** :
   (a) d'abord **2 rangs localhost ring sur .29** (2 process, un node —
   isole la logique de la variable réseau) ; (b) puis 2 nodes (.30/.31),
   backend `ring` (stabilité, pas de RDMA à réinitialiser). Target
   Qwen3-8B **tensor parallel**, draft Qwen3-0.6B répliqué.
   **Inventaire préalable** : lister les paires candidates PRÉSENTES sur
   les disques cluster (`/Volumes/models/odysseus`) et vérifier
   `tokenizer.json` sha IDENTIQUE target/draft (le chat template est
   indifférent — le draft reçoit les mêmes token-ids — l'identité du
   tokenizer ne l'est pas). Sinon paire de repli Qwen3-14B/0.6B.
   Puis répéter en **pipeline** si un petit modèle PipelineMixin est
   dispo (sinon TP suffit pour G0 — le PP est re-prouvé en E3).
4. Protocole : 3 prompts × 500 tokens greedy (temp=0), 2 répétitions.
   
**Done/G0** : (a) sha finaux identiques rank0/rank1 sur 6/6 runs ;
(b) sortie identique token-for-token au même modèle en AR greedy
single-rank (le spec exact-match greedy NE CHANGE PAS la distribution) ;
(c) zéro rank death, unload propre ;
(d) tok/s noté (informative — le gain avec un draft externe n'est pas le
gate, l'ALIGNEMENT l'est).

### E1 — `mtp_module.py` + `mtp_spec.py` (le cœur) — 5 pts

**But** : le module MTP natif générique + la boucle spéculative native,
testables single-node (size==1 est un cas particulier du même code).

1. `scripts/mtp_module.py` :
   - `detect_native_mtp(model_dir) -> MTPSpec | None` : parse
     `model.safetensors.index.json`, cherche les familles de clés connues
     (`model.mtp.*` LongCat ; `model.layers.<N>.eh_proj/…` + nextn
     GLM/DSv3 — mapping par famille repris de
     `mtplx/backends/{deepseek_mtp,glm_mtp}.py`).
   - `load_native_mtp(model, model_dir, spec, quantize_like_trunk=True) -> NativeMTPModule`.
     Poids chargés directement des shards (mx.load des fichiers listés
     par l'index pour les clés mtp), rewrite kv_b si MLA
     (`deepseek_mtp_patch.py:118`), quantif alignée trunk (l.217),
     **skip des clés `oe_*`** (cf V1). lm_head/embed : si le spec dit
     « partagé », référencer les modules du trunk chargé (pas de copie).
   - `NativeMTPModule.draft_step(token_ids, prev_hidden, cache) -> (logits, hidden)`
     et `.make_cache()` / cache trimmable.
2. `scripts/mtp_spec.py` : `native_mtp_stream_generate(model, tokenizer,
   prompt, *, mtp, depth, max_tokens, sampler, prompt_cache, group, rank,
   canary_cb)` :
   - Prefill trunc (chunké, réutilise le prefill legacy/session-cache du
     runner) → hidden final h0 + bonus token t0 (argmax).
   - Round : (i) draft D tokens en chaînant `draft_step` (chaque appel
     APPEND **UNE** position au cache mtp) ; (ii) verify :
     `[t_bonus, d0..dD-1]` (D+1 positions) en UN forward trunc + norm/head ;
     (iii) accept exact-match greedy (conventions `generate.py:623-651`) ;
     (iv) `trim_prompt_cache(model_cache, D-n)` + `trim mtp_cache
     max(D-n-1,0)` ; (v) yield les n+1 tokens ; (vi) canary_cb(round, D,
     n, sha_cum) ; (vii) seed du round suivant.
   - **Invariants sémantiques à graver dans le code (source de vérité :
     `generate.py:589-654`, mal-lisibles donc explicites ici)** :
     * Le **bonus = le token pendant** : sampled au round précédent, son
       K/V n'est PAS encore dans le cache trunc. Le verify l'écrit
       (`generate.py:617-618` : `y = concat([y, draft_tokens])` puis
       `_step(model, cache, y, num_draft+1)` — l'input verify EST
       `[bonus, drafts]`). Avance cache par round = **+D+1, puis trim
       (D-n)** ; asymétrie draft : trim `max(D-n-1, 0)` car le cache mtp
       a consommé le bonus en seed.
     * **Seed hidden du round suivant = `verify_hidden[n]`** (position de
       la DERNIÈRE acceptée, bonus = index 0). Jamais le hidden d'une
       position rejetée — le vérifier par assert en mode debug.
     * **Resume sur session prefix-cache chaud** : le cache ne stocke pas
       de hidden → convention « toujours laisser 1 token à forwarder » :
       trimmer le cache à `prompt_len-1` et re-forwarder le dernier token
       du prompt pour produire (hidden seed, bonus). 1 forward de coût,
       non négociable.
     * **Assert de fin de gen (rank-local, toujours ON)** :
       `cache.offset == prompt_len + tokens_émis` — attrape toute dérive
       d'arithmétique de trim avant qu'elle ne corrompe la session
       (interaction session-cache, cf R6).
   - **Capture du hidden pré-norm sans forker chaque modèle** : wrapper de
     capture posé sur `model.model.norm` (enregistre son INPUT = hidden
     pré-norm, retourne la sortie normée inchangée). Un seul mécanisme,
     toutes familles mlx-lm ; le flag D8 choisit input (pre) ou output
     (post) du wrapper.
   - Shapes STABLES : D fixe par requête (pas d'adaptatif v0) → pas de
     recompiles Metal par round. Précondition déterminisme (D3) : shapes
     constantes + ordre de réduction fixe (ring) ⇒ mêmes kernels par rang ;
     le canari est l'alarme, pas une preuve a priori.
3. Unit tests sans cluster (`scripts/test_mtp_spec.py`) : modèle jouet +
   module jouet en mlx pur —
   * **test de parité PAR POSITION** (pas un sha final) :
     `for p: assert mtp_tokens[p] == ar_tokens[p]` sur les mêmes prompts,
     pour D ∈ {1,2,3} — c'est LA revendication empirique du plan ;
   * les trims laissent `cache.offset` cohérent, **assert périodique
     toutes les 64 positions** (pas seulement en fin de gen — une dérive
     à mi-course doit péter à mi-course) ;
   * le bonus-token est compté juste (les 3 pièges classiques) ;
   * **assert d'aliasing** : le `lm_head`/embed partagés référencent les
     objets du trunc et ne sont jamais re-swappés pendant la vie du
     runner (adapter/hot-swap = hors contrat v0).

**Done** : tests verts + revue du diff par un 2ᵉ agent (règle double-review
high-stakes).

### E2 — Natif single-node : la preuve de non-régression perf — 3 pts

**But** : notre boucle E1 atteint ~ les chiffres MTPLX sur un modèle
vérifié, AVANT d'aller multi-rang (isole « notre loop est lente » de
« le multi-rang est lent »).

1. Modèle : Qwen3.6-27B (MTPLX-verified, 2.24× @ D3 temp 0.6 publié —
   en greedy attendre ≥ 2×) OU GLM-5.2 s'il est déjà sur le disque du
   node de test. Single-node sur un ultra (via pool `RUNNER_MTP=native`,
   world_size=1).
2. Bench : AR baseline vs D1/D2/D3, 3 prompts × 512 tokens, greedy.
   **Warmup obligatoire avant CHAQUE mesure** (une gen de ~200 tokens
   jetée — la 1ʳᵉ passe paie les compiles Metal ; sans warmup G1 sous-lit,
   leçon des timings « lazy » Hy3). Sortie D* == AR token-for-token
   (canari). **Profiler le wrapper de capture pré-norm** (avec/sans) :
   s'il coûte > 3 % (fusion de kernels cassée), passer à une capture par
   attribut/slice.
3. **Résoudre D8 ici (pas en E4)** : A/B `mtp_hidden_source=pre_norm` vs
   `post_norm` sur GLM-5.2 (même convention hnorm que la famille DSv3) —
   l'acceptance tranche empiriquement (bon ≈ ≥0.5, faux ≈ 0) ; la lecture
   source V2 devient une confirmation, plus un pari.
4. **G1 — BLOQUANT DUR** : D-best ≥ **1.5×** AR ET sortie exacte. Si
   < 1.5× : profiler (verify re-forward ? eval par round ? cache offset ?).
   **Deux tentatives de fix max** (règle R7/cheval mort) ; toujours
   < 1.5× → le chantier S'ARRÊTE ICI — une boucle qui ne gagne pas
   single-node ne gagnera jamais multi-rang (les hops s'ajoutent, ils ne
   se retranchent pas). Ne PAS aller en E3 « pour voir ».

### E3 — Natif multi-rang (le cœur du chantier) — 5 pts

**But** : G2 — le MTP natif multi-rang aligné et plus rapide que l'AR
multi-rang.

1. Modèle : **GLM-5.2 sur Argo** (PipelineMixin + poids présents + servi
   en prod → baseline AR connue) ; fallback DeepSeek-V3.2-Exp.
   Topologie : 2 nodes ring d'abord (.30/.31), puis 4 nodes (.29-.32),
   backend ring puis jaccl.
2. `runner.py` : path `RUNNER_MTP=native` multi-rang — le module se
   charge sur CHAQUE rang après le barrier l.1879-1882 ; la boucle legacy
   sélectionne `native_mtp_stream_generate` au lieu de `stream_generate`
   (l.2193) quand le module est chargé. `emit()` rang-0 inchangé.
3. Engine `api.py` : passer `RUNNER_MTP/RUNNER_MTP_DEPTH` dans l'env du
   `remote_cmd` (même veine que `RUNNER_BACKEND`), knob pool
   `mtp:{enabled,depth}`, badge dashboard.
4. Protocole canaris complet (§6) actif ; comparaison vs le même modèle
   en AR multi-rang ET vs single-node E2 si le modèle tient sur 1 node.
5. Matrice : {2 nodes, 4 nodes} × {ring, jaccl} × {D1, D2, D3} ×
   3 prompts × 512 tokens greedy.

**G2 (gate cheval-mort)** : sur la meilleure config — (a) alignement :
zéro trip canari sur toute la matrice ; (b) exactitude : token-for-token
vs AR greedy ; (c) **perf : ≥ 1.25× l'AR multi-rang même topologie**.
Si (c) < 1.1× après tuning D → **STOP chantier, rapport, on garde l'AR**
(l'infra E0-E2 reste : elle sert au binding LongCat si son profil diffère,
et le rapport documente pourquoi).

### E4 — Binding LongCat — 5 pts (APRÈS le port de base)

**But** : brancher le path natif sur LongCat-2.0 porté.
**Condition d'entrée** : le port de base passe son propre gate (gen AR
5-node saine). Tant que non → E4 PARQUÉ, rien à forcer (E0-E3 + G2 ont
déjà livré la valeur générique GLM/DSv3 ; pas de date, des gates).

1. Pré-requis contractuels sur le PORT (à donner à l'agent du port,
   cf §9) : hidden final accessible pré-norm, lm_head/embed référençables,
   caches trimmables, indexeur DSA factorisé réutilisable.
2. `mtp_module.py` : builder famille `longcat` — MLA (kv_lora 512,
   q_lora 1536, rope 64) + indexeur LSA du module (réutilise la classe
   indexer du port ; `dsa_mtp_cli: true` → UNE passe d'index partagée
   par les 3 draft steps, `cli_factor` 2 côté trunk) + eh_proj
   [8192,16384] (ordre de concat À CONFIRMER source, V1) + embed propre
   [163840,8192] + norm propre + lm_head partagé. Clés `model.mtp.oe_*`
   SKIPPÉES (V1).
   Draft v0 SANS indexeur si contexte < seuil dense-fallback (même règle
   que le trunk DSA — dense ≤ 2048/`index_local_tokens`) : le smoke court
   n'a pas besoin du path sparse.
3. Conversion : la recette Q4 head-bf16 du port DOIT inclure les clés
   `model.mtp.*` compute (~1.9B — négligeable dans le budget) et EXCLURE
   les `oe_*` du mtp si V1 confirme. (Trancher V1 AVANT de lancer la
   conversion 3.55TB — sinon on re-convertit.)
4. Smoke 5-node : caption… non — gen texte : 3 prompts × 256 tokens
   greedy, canaris, acceptance ≥ 0.4 attendue (sinon suspicion D8
   hidden-source → basculer le flag, re-smoke).

### E5 — Gate final LongCat + tune — 2 pts

Matrice D1/D2/D3 × {ring, jaccl} sur 5 nodes, 512 tokens.
**G3** : ≥ 1.4× vs AR 5-node, acceptance ≥ 0.5, zéro trip canari,
sortie exacte vs AR greedy. Consigner dans docs/EVIDENCE + VELOCITY.

### E6 — Phase 2 : sampling (SÉPARÉ, après ship v0) — 5 pts

Probability-ratio acceptance (MTPLX `acceptance_probability` +
residual correction) en multi-rang : graine RNG par requête dans le
JSONL fan-out, `mx.random.seed(seed)` par rang avant chaque requête,
canaris obligatoires (le non-déterminisme fp est LE risque — mesurer,
pas supposer). Gate dédié : distribution non dégradée (perplexité sur
un set fixe) + alignement tenu sur 10k tokens.

### E7 — Prod hardening — 3 pts

Auto-disable MTP sur trip canari (pool reste up, MTP off, alerte
dashboard) ; métriques #61 (acceptance, rounds, from_draft ratio) ;
doc RUNBOOK ; défauts prod : `mtp.enabled=false` partout tant que G3
n'est pas passé.

---

## 5. Récap gates

| Gate | Où | Critère chiffré | Échec → |
|---|---|---|---|
| G0 | E0, 2-node draft harness | sha identiques 6/6, sortie == AR greedy | debug alignement AVANT tout code MTP |
| G1 | E2, single-node natif | ≥ 1.5× AR, sortie exacte | profiler la boucle, ne PAS aller en E3 |
| G2 | E3, multi-rang natif | ≥ 1.25× AR multi-rang, zéro trip, exact | < 1.1× après tuning → STOP chantier (cheval mort), rapport |
| G3 | E5, LongCat 5-node | ≥ 1.4× AR 5-node, acceptance ≥ 0.5 | retune D / revisiter verify ; si mur structurel → MTP off par défaut, chantier documenté |

Modèle de coût honnête derrière G2/G3 (à valider, pas à croire) : en PP,
un round D3 = 1 pass trunc de 4 tokens ≈ lectures d'experts ~identiques à
4 tokens AR (top-12/token, union ≤ 48) MAIS attn+dense+embed+head lus UNE
fois au lieu de 4, et 1 seul aller-retour pipeline au lieu de 4. À
acceptance 0.6, ~2.8 tokens/round pour un coût round ~1.5-2.2× le
token AR → gain attendu 1.3-1.9×. Si la réalité mesurée sort de cette
fourchette vers le bas, le profil (bandwidth experts vs latence hops)
dira lequel des deux termes ment.

## 6. Protocole canaris (méthode nuit VLM, adaptée au spec)

Par rang, sur stderr (JSONL, préfixe `[canary]` — l'engine collecte déjà
les stderr par rang via `RunnerProc._drain_stderr`, mais aujourd'hui en
simple tail-buffer : **le parseur/agrégateur est un deliverable E3 à part
entière, pas un détail**) :
- Par round : `{"round": i, "drafted": D, "accepted": n, "sha": sha256_cum_tokens[:16]}`.
- Fin de gen : `{"final": true, "ntokens": N, "sha": …, "accept_rate": r}`.
Engine (`api.py`) : `CanaryAggregator` par run — buffer par rang **keyé
par round** ; comparaison UNIQUEMENT quand tous les rangs ont émis le
round i (les rangs n'écrivent pas en cadence — comparer des rounds
différents = faux positif garanti) ; timeout par round (~30 s) → warning
« rang à la traîne », pas trip. **Trip** = deux rangs ont émis le MÊME
round avec (accepted, sha) différents → abort de la gen,
`mtp.enabled=false` sur le pool (auto-disable D7, pool up), event
dashboard + log.
Validation d'exactitude (hors prod) : dump des token-ids complets rang 0
vs baseline AR greedy — diff vide exigé (G1/G2/G3).
Micro-coût : un sha256 de liste d'ints par round est négligeable devant
un forward — resté ON en prod (c'est l'alarme, pas un mode debug).

## 7. Vérifications-source AVANT E4 (V1-V3) — jamais supposer

- **V1 — Path oe dans le module MTP** : lire l'impl d'inférence officielle
  (GitHub `meituan-longcat/LongCat-2.0` → guides SGLang/vLLM ; le modeling
  SGLang `longcat_v2`/MTP). Confirmer que `mtp_disable_over_tokenizer:
  true` ⇒ tables `mtp.oe_*` inutilisées à l'inférence. Décide 77 GB de
  conversion et le budget mémoire réplication. **Bloquant pour la recette
  de conversion du port.**
- **V2 — D8 hidden pre/post-norm + ordre de concat eh_proj**
  (`[enorm(emb) ; hnorm(h)]` vs inverse) : même source. Symptôme si faux :
  acceptance ≈ 0 (leçon Mistral-EAGLE). NB : D8 est résolu EMPIRIQUEMENT
  dès E2 sur GLM-5.2 ; V2 confirme pour LongCat. Vérifier AUSSI que le
  layout MLA du module mtp LongCat suit la même convention kv_b que
  DeepSeek (le rewrite `deepseek_mtp_patch.py:118` s'applique tel quel ou
  pas — sinon path de chargement dédié).
- **V3 — Indexeur en mode MTP** (`dsa_mtp_cli`, `index_init_tokens: 16`,
  `index_local_tokens: 1024`) : sémantique exacte de la passe partagée.
  Nécessaire pour le path long-contexte seulement (v0 = dense fallback).

## 8. Risques & signaux cheval-mort

| # | Risque | Signal | Réponse |
|---|---|---|---|
| R1 | Divergence accept-count inter-rang (fp non-associatif dans un collectif) | trip canari round N | v0 greedy le rend improbable (argmax de logits identiques) ; si trip : dumper les logits de la position fautive par rang, comparer — si fp : arrondi/tie-break déterministe avant argmax |
| R2 | Verify multi-token trop cher sur trunc MoE (union experts) | G2 < 1.1× avec profil dominé par gather_qmm | réduire D ; si D1 insuffisant → STOP (cheval mort assumé) |
| R3 | ABI hidden faux (D8) | acceptance ≈ 0 | basculer `mtp_hidden_source`, re-smoke ; lire la source V2 |
| R4 | Cache du port LongCat non-trimmable (DSA custom) | `can_trim_prompt_cache` False | trim = EXIGENCE du contrat port (§9). Pas de promesse de fallback : le buffer capture/restore à la Telemak est un chantier en soi — si le port ne livre pas le trim, E4 est BLOQUÉ, point |
| R5 | Recompiles Metal par variation de shapes | tok/s qui s'effondre après round 1, temps « lazy » anormaux | D fixe par requête ; shapes de verify constantes ; pas d'adaptatif v0 |
| R6 | Interaction session prefix-cache × trims | 2ᵉ tour de chat corrompu | test dédié E3 : conversation 3 tours avec MTP on, diff vs AR |
| R7 | Empilement de rustines sur la boucle E1 pour sauver G1 | 2 patches sans amélioration mesurée | règle de base : stop, re-design, remonter |

## 9. Contrat de binding pour le PORT LongCat (à transmettre à l'agent du port)

Le port (chantier séparé, prioritaire) doit exposer — sinon E4 devra
patcher après coup :
1. `Model.model(x, cache)` retourne le hidden final **pré-norm** accessible
   (ou un hook équivalent) ; norm et lm_head appelables séparément.
2. `make_prompt_cache(model)` → caches **trimmables** (`is_trimmable()`
   True), y compris le cache de l'indexeur DSA s'il en tient un.
3. La classe MLA + la classe indexer importables et paramétrables
   hors-liste-de-layers (le module MTP les instancie avec ses propres
   poids).
4. Le mapping de conversion INCLUT `model.mtp.*` (hors `oe_*` si V1
   confirme) dans la recette Q4 head-bf16 — décision V1 AVANT de lancer
   les ~30h de conversion.
5. PipelineMixin conforme au pattern deepseek_v32 (split inverse,
   all_gather final l.473) — c'est l'invariant D1.

## 10. Écarté (déjà tranché — ne pas re-proposer)

- Draft-model séparé multi-rang comme path PRODUIT (reste un mode harness
  E0 + un plus opportuniste single-rank) — le natif in-trunk est la voie.
- BatchGenerator multi-rang (alignement collectifs, `runner.py:1931`).
- Revival du path Swift Hy3 MoE-MTP (30×, requalifié §1.5 — le path
  Python n'a pas cette pathologie ; Telemak reste single-node dense MTP).
- Heads MTP sur le dernier rang uniquement (D1).
- Supposer l'alignement sans canaris (piège VLM — 2 bugs vus aux hashes).
- Adaptatif de depth par round en v0 (shapes instables, R5).

## 11. Points & séquencement

| Phase | Pts | Dépendance |
|---|---|---|
| E0 harness | 3 | aucune |
| E1 module+loop | 5 | aucune |
| E2 single-node | 3 | E1 |
| E3 multi-rang | 5 | E0+E2 |
| E4 binding LongCat | 5 | port LongCat + V1-V3 |
| E5 gate final | 2 | E4 |
| E6 sampling ph.2 | 5 | ship v0 |
| E7 hardening | 3 | E3 |
| **v0 (E0-E3)** | **16** | — |

## 12. Addendum exécution (grill Sophie, 2026-07-05 soir — GO donné)

Panel-à-3 rempli par : plan ancré source (direct) + grill Sophie (intent)
+ MiniMax-M3 cross-model (sceptique, 3 rounds). Subagents indisponibles
cette session (gotcha Fable) — forme praticable consignée.

Décisions opérationnelles verrouillées au grill :
1. **Nodes** : tout Argo libre (.29-.32) — unload des pools chauds
   autorisé, RESTAURATION en fin de nuit. Prod engine .39 : deploy
   AUTORISÉ (cf 2). Goal = « MTP distribué fonctionnel » = **E0→E3, G2
   vert sur le modèle E3, greedy, canaris propres**. E4-E5 LongCat hors
   goal (port inexistant).
2. **Deploy .39 OK avec filet** : AVANT tout code — tag git de l'état
   courant + copie de sauvegarde des `api.py`/`dashboard.html` déployés
   (rollback 1-geste). `mtp.enabled=false` par défaut PARTOUT au deploy.
3. **Règle STOP confirmée** : G1/G2 ratés après 2 tentatives de fix →
   rapport-STOP documenté, fin de nuit. Pas de rustines empilées.

Matériel vérifié sur disque (zéro download pour E0) :
- E0 : target `mlx-community/Qwen3-Coder-30B-A3B-Instruct-6bit`
  (Qwen3Moe, 4 KV heads → TP-2/4, trimmable) + draft
  `mlx-community/Qwen3-0.6B-8bit` (5 nodes) — tokenizer.json sha
  IDENTIQUES (`aeb13307…`).
- E2/E3 : `kernelpool/GLM-5.2-Q6` — **vérifier D'ABORD que la conversion
  a gardé les poids mtp/nextn** (les convert MLX les strippent souvent —
  leçon qwen3_next). Si strippés : sidecar des shards mtp originaux
  (bf16, ~10-20 GB) téléchargés depuis le repo source + quantize-on-load
  (le loader D5 lit l'index séparément du trunc — prévu pour).
- Référence G1 : `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` (+
  `mlx-community/Qwen3.6-27B-MTP-4bit/bf16`) via MTPLX tel quel.
