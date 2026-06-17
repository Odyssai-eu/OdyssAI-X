# Plan Review Log: Mistral 3.5 intégration + vitesse (OdyssAI-X #54)

Act 1 (grill-with-docs) complete — plan verrouillé (`PLAN-mistral.md`), `CONTEXT.md` enrichi (section Serving & vitesse), `docs/adr/0003-mistral-speed-q6-eagle-mtplx.md` créé.

Décisions verrouillées en Act 1 :
- #54 = tout le chantier vitesse ; done = ≥20 tok/s (pas juste Q4).
- Q6 (pas Q4) ; EAGLE port complet engagé (pas de spike-gate) ; single-node d'abord, TP ensuite si < 20.
- Multilingue (pas de cjk_lock par défaut) ; texte-only.

MAX_ROUNDS=5. PLAN_FILE=PLAN-mistral.md. Reviewer Act 2 = MiniMax-M3 **cloud** (mmx) — Argo NON utilisé (occupé, tests code).

---

## Round 1 — MiniMax-M3 (cloud) — VERDICT: REVISE

A exploré le repo (cité `api.py:1276-1278/733/807`, `auto_parallel.py:57/575-582/719`, `CONTEXT.md`, `m3_convert.py:55-57`, le lock pinné). 14 findings :
1. MTPLX = repo externe non pinné. 2. `mlx_lm.models.mistral3` "n'existe pas" (seul `ministral3`), confond Mistral Medium 3.5 / Ministral 3. 3. **Le runner câble DÉJÀ `RUNNER_DRAFT_MODEL` (api.py:1276) = spéculatif natif mlx-lm** — le plan réimplémente EAGLE dans un repo tiers au lieu de ce chemin. 4. Argo = 3 nodes (CONTEXT) ; KV=8 % 3 ≠ 0 → TP3 invalide. 5. La sharding TP (auto_parallel.py:719) fait `n_kv_heads //= N` sans check divisibilité → `_validate_load_mode` (api.py:807) refusera le load. 6. **Maths : 1.71× (MTPLX prouvé) × 6.8 (Q6 base) = 11.6 tok/s single-node, loin des 20 ; 20 demande 2.9×/Q6 ou 4×/Q8 — au-delà d'un EAGLE 1-couche.** 7. Le draft EAGLE 3.5-128B non vérifiable dans le repo. 8. Pas de converter ; m3_convert attend `index.json`, pas le `consolidated`/`params.json` Mistral. 9. "Pas de cjk_lock" sans métrique de leak. 10. Validation greedy-vs-bf16 = charger bf16 ET Q6, ressources non allouées. 11. WU3 "TP si <20" sans critère d'arrêt. 12. `MODEL_SAMPLING_DEFAULTS` scopé M3 — pas de profil Mistral. 13. TP2 sur 3-node = 3ᵉ idle, casse alignement JACCL/pipeline. 14. Aucune observabilité (acceptance, tok/s réel, dashboard).

### Claude's response (arbitre)

**Incorporé (vrais trous) :**
- **#6 — le catch central.** J'avais sur-vendu "20 single-node". Spéculatif prouvé = 1.71× → Q6 single-node ≈ **12-17, PAS 20**. → **TP sur le chemin critique, plus en filet.** ADR-0003 + PLAN à corriger.
- **#3** — évaluer le chemin draft-model NATIF (`RUNNER_DRAFT_MODEL` déjà câblé) AVANT/contre le port MTPLX. NB : EAGLE (tête réutilisant les hidden states) ≠ draft-model standalone → le natif voudrait un petit Mistral standalone (acceptance moindre, mais ~5 lignes vs un 8 cross-repo). À peser en WU2.
- **#5 + #11** — gate divisibilité KV + échelle TP (2→4, JAMAIS 3) + critère d'arrêt (≥20 ou TP4 épuisé).
- **#12** — ajouter une entrée `mistral` à `MODEL_SAMPLING_DEFAULTS` (temp/top_p, pas de cjk_lock).
- **#14** — WU observabilité (log acceptance + tok/s réel + dashboard).
- **#2 (partiel)** — vérifier le wiring de `mistral3` dans les listes engine (`api.py:733`, `auto_parallel.py:57` n'ont que `ministral3`).
- **#4 (doc)** — CONTEXT.md PÉRIMÉ : Argo = **5 nodes** (topology 2026-06-10), pas 3 → à corriger. (Le point divisibilité est juste ; mon plan 2/4 l'évitait déjà.)

**Rejeté (limite du sandbox repo-only de M3 — il ne voit pas HF) :**
- **#7 — FAUX** : le draft EAGLE existe, vérifié sur HF `mistralai/Mistral-Medium-3.5-128B-EAGLE` (consolidated 3.07 GB, params.json, README vLLM/SGLang `method:eagle`). → ajouter le lien HF au plan pour l'ancrer.
- **#2 (l'autre moitié)** — `mistral3.py` EXISTE dans mlx-lm local + config.json HF dit `model_type: mistral3` (≠ ministral3). M3 a lu le lock pinné, pas HF. Mais la vigilance Mistral/Ministral est réelle.

**Nuancé :**
- **#8** — le modèle PRINCIPAL est HF-sharded (convertible normalement) ; seul le DRAFT est en consolidated/params.json. À distinguer.
- **#10** — validation SÉQUENTIELLE (bf16 puis Q6) ou Q6-vs-Q8 (tiennent tous deux), pas en parallèle.
- **#13** — vraie topo 5 nodes → TP2 = 2 des 5 ; mais le point "config pool TP vs pipeline + rank-count" est valide.

**Bilan du test :** MiniMax-M3 cloud = reviewer adversarial réel et code-ancré. Angle mort = tout ce qui est HORS repo (HF, topo courante) → le seeder avec les faits externes.

### Claude's response (round 1 → révision, 2026-06-16)

`PLAN-mistral.md` rév. 2 — deux changements majeurs :
1. **Pivot architecture (décision Sophie, hors-grill) :** chemin primaire = **Telemak Swift single-node** (JACCL-free, `Mistral3Text.swift` déjà porté dans le fork) ; Odysseus Python multi-node TP → **fallback**. Un 128B dense tient single-node, le splitter est le mauvais trade.
2. **Findings round 1 intégrés :**
   - **#6 (catch central, accepté)** : 20 single-node PAS garanti (1,71× × 6,7 ≈ 11,6) → cible honnête **12-17**, 20+ = fallback. Plus de sur-promesse.
   - **#5/#11** : échelle TP 2/4 (jamais 3, 8∤3) + critère d'arrêt → dans le fallback.
   - **#12** : profil sampling Mistral (WU3). **#14** : observabilité acceptance + tok/s (WU3).
   - **#2** : `mistral3` à câbler côté engine Odysseus (fallback) ; côté Telemak `Mistral3Text` existe → concern levé pour le primaire.
   - **#7 (rejeté)** : le draft EAGLE existe (`mistralai/Mistral-Medium-3.5-128B-EAGLE`, HF) — ajouté au plan ; cécité repo-only de M3.
   - **#3** : `RUNNER_DRAFT_MODEL` natif pertinent côté fallback Odysseus ; primaire Swift via Telemak-MTP/MTPLX.

Round 2 : MiniMax ne voit pas le fork Swift → seedé avec les faits Telemak.

---

## Round 2 — MiniMax-M3 (cloud) — VERDICT: REVISE

Sur le plan rév. 2 (Telemak Swift single-node primaire). 10 findings, dont 3 NEUFS et tranchants :

1. **Swift ≥ Python sur dense = PARI NON DÉMONTRÉ (le catch central).** « dense → pas de pénalité Swift » est une *inférence*, pas une mesure. Swift a ses coûts (Metal command-encoding, kernel launches, pas de fusion) ; aucun ratio Swift/Python publié sur un dense 128B. Le « 12-17 » exige Swift ≥ ~2× Python sur Q6 — non établi. **C'est le make-or-break du pivot.**
2. Intervalle 12-17 non justifié chiffré (6,8 base × 1,76-2,5×) — cohérent SEULEMENT si Swift=Python.
3. « 1,71× EAGLE prouvé » jamais sourcé (c'est notre MTPLX interne) ; et c'est CONSERVATEUR (EAGLE-2 ~3× Llama-70B). Plan cite un chiffre bas comme référence.
4. Draft EAGLE : existence invérifiable depuis le repo + format vLLM/SGLang → convertisseur MLX/Swift requis, non listé.
6. **Format de poids** : `Mistral3Text.swift` attend-il le naming HF (`model.embed_tokens`) ou Mistral natif (`tok_embeddings`) ? La conv Q6 (HF-style) pourrait ne pas charger. WU0 doit trancher le sanitize Swift.
7. **Le fallback #54 est BLOQUÉ** : mlx-lm 0.31.3 pinné ne supporte que `ministral3` (8B), pas `mistral3` (le 128B). Câbler les listes api.py est cosmétique — sans `mlx_lm.models.mistral3`, l'import plante. → MAJ dépendance ou **vendoriser** (chantier non chiffré, comme deepseek_v4/hy_v3).
8/9. Observabilité tok/s sans mécanisme (ni Telemak well-known, ni instrumentation runner.py) ; chaîne conv→charge→format Q6 non clarifiée.
10. Pas de **bascule routable** Telemak→Odysseus si le Mac Telemak tombe (le « fallback » reste manuel).

### Claude's response (arbitre, round 2)

**Accepté — incorporé en WU0 comme GATE de mesure :**
- **#1/#2/#3 (le cœur) :** le pivot repose sur un ratio Swift/Python dense NON MESURÉ. **WU0/WU1 deviennent un gate empirique : mesurer le ratio Swift/Python sur Q6 dense AVANT de s'engager.** Si Swift traîne aussi sur le dense (overhead général, pas que le gather MoE), le 12-17 tombe et le pivot s'affaiblit. 1,71× re-sourcé (MTPLX interne, conservateur ; public EAGLE 2-5×) → à mesurer sur Q6, pas à présumer.
- **#6/#9 :** WU0 doit vérifier la compat naming conv-MLX ↔ clés attendues par `Mistral3Text` + clarifier qui convertit/charge/dans quel format.
- **#7 (vrai bloqueur fallback) :** le fallback Odysseus suppose mlx-lm `mistral3`, **non garanti sur la version pinnée** → vérifier, sinon vendoriser un `scripts/mlx_models/mistral3.py` (chantier à chiffrer). Le fallback n'est PAS gratuit.
- **#10 :** la bascule Telemak→Odysseus doit être routable (dashboard/router), pas manuelle.
- **#5/#8 :** profil sampling Mistral côté Odysseus (fallback) + mécanisme de log tok/s.

**Maintenu :** le draft EAGLE existe (HF, URL donnée par Sophie) — mais le port format est réel (WU2).

### Convergence
Le grill a fait son travail sur DEUX rounds : il a pincé mes deux angles morts (round 1 : 20-single-node sur-vendu ; round 2 : Swift=Python pris pour acquis). La STRUCTURE est saine (Telemak Swift single-node, JACCL-free, architecturalement juste). Les risques restants sont **empiriques** (ratio Swift/Python, format poids, mlx-lm fallback) → résolubles seulement par **WU0**, pas par plus de planification. Le grill converge sur « la structure tient, va MESURER ».
