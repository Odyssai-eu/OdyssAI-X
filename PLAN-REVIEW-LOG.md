# Plan Review Log : attention block-sparse (MSA) M3 — phase 1

Act 1 (grill-with-docs) complete — plan locked, `CONTEXT-msa.md` créé + ADR-0002
écrit. 6 décisions résolues : cible 128K (TTFT+decode) / exactitude A défaut,
B prefill-only avec porte qualité / Python single-node / gather pur-MLX (Metal
= phase 2) / attention principale seule (indexer = phase 2) / critère relatif.
MAX_ROUNDS=5.

## Round 1 — Codex (VERDICT: REVISE)

16 findings. Les marquants :
- **BLOCKER indexer dense quadratique** : à 128K prefill, `q@kᵀ` de l'indexer `[B,4,S,K]` reste O(S·K) → le prefill ne devient pas vraiment rapide même avec l'attention principale sparse.
- **BLOCKER union prefill quasi-dense** : tuile 128 req × top16 à 128K peut couvrir presque tout le contexte.
- `mx.take(axis=2)` applique les mêmes indices à tout le batch ; MSA renvoie `[B,S,topk]` → `take_along_axis` broadcasté.
- SDPA MLX = masque broadcast, pas ragged → `T_kv` fixe par tuile + masque additif `[B,1,T_q,T_kv]`.
- `-1` (blocs invalides) dangereux au gather → clamp index sûr + masque `-inf`.
- « bit-identical » trop fort (ordre de réduction softmax) → équivalence math + tol CPU, indices triés par position.
- Masque causal intra-bloc = condition d'exactitude (le bloc local contient des futurs pour les 1res requêtes).
- Seuil : dense si `k_len <= 2048`, sparse strictement `> 2048`.
- BatchGenerator casse l'hypothèse B=1 (single-node ≠ single-request).
- Mémoire 128K sous-estimée : KV fp16 + `idx_keys` en plus des 355 GB poids ; `_cache_size_bytes` ignore `idx_keys`.
- Baseline dense 128K peut OOM (masque `[B,1,S,K]`) → cap + extrapolation.
- 3 couches full-attention restent O(n²) → plancher prefill à profiler.
- Voie B contredit « sans changer la sortie » → hors critère de livraison, flag expérimental.
- Observabilité : logger union size, gathered keys, fallback, timings indexer/gather/SDPA, mémoire max.
- **Alternative plus simple : decode-only d'abord** (sans craquer l'indexer ; prefill exact exige l'indexer tiled).

### Claude's response

**Recadrage majeur accepté : phase 1 = DECODE-ONLY.** La synthèse de Codex (findings indexer-dense + union-blow-up + decode-only) converge : le prefill exact-sparse exige de craquer l'indexer (déjà scopé phase 2) ET souffre du blow-up d'union. Le decode, lui, gagne énormément (≤2048 vs k_len, ~62× à 128K) **sans** toucher l'indexer (S_q=1 → indexer trivial), exact, bas risque. C'est aussi le coût que Companion paie par token. Donc : **phase 1 = decode block-gather exact ; prefill block-sparse + indexer tiled + kernel Metal = phase 2.** (Revoit la réponse Q1 « les deux phases » → à faire bénir par Sophie au sign-off.)

**Corrections techniques intégrées au plan** : `take_along_axis` broadcasté (pas `take`) ; `T_kv` fixe + masque additif `[B,1,T_q,T_kv]` (pas de ragged) ; `-1` clampé + masqué `-inf` ; exactitude reformulée « équivalence math, < 1e-6 CPU, indices triés » (pas bit-identical) ; composition `selected ∧ causal ∧ valid` explicite ; seuil strict `> 2048` ; ordre indexer→update KV→gather épinglé ; chemin legacy single-slot confirmé (M3 tourne `_run_legacy_main`, B=1) + B>1 hors scope ; budget mémoire = poids + KV fp16 + `idx_keys` (calculé) ; baseline 8k/32k réel + cap OOM sur 128k dense ; observabilité ajoutée.

**Rejeté/différé** : voie B sort du critère d'acceptation (flag expérimental, pas livrable). Sparsification indexer + prefill = phase 2 (pas un rejet, un séquencement).

## Round 2 — Codex (VERDICT: REVISE)

Blockers du round 1 confirmés traités. 5 nouveaux points fins :
- Bench 128K decode pas exécutable tel quel (cache 128K à constituer ; prefill dense OOM) → méthode explicite (injection synthétique harness + prefill-court-puis-génère en prod).
- « Forcer legacy » trop vague → `RUNNER_BATCH=0` explicite (détection `model_type==minimax_m3`).
- « ~62× » surestimé → ignore l'indexer decode (scanne tout, O(K)/token) + 3 full layers ; reformuler en réduction-K principale 62× mais gain réel ~10-13× borné par l'indexer.
- Tolérance golden trop absolue → deux niveaux : fp32 CPU < 1e-6, dtype prod tolérance réaliste + non-régression logits/tokens.
- Caches session/prewarm désactivés seulement « au bench » → aussi en prod long-contexte tant que `idx_keys` n'est pas compté.

### Claude's response

**Les 5 acceptés et intégrés** (aucun rejet). Le plus important : le « 62× »
était trompeur — à S_q=1 l'indexer reste O(K)/token et devient le plancher, donc
le gain decode phase-1 est ~10-13× (mesuré), et craquer l'indexer (phase 2)
débloque le reste. Goal reformulé honnêtement. Cache 128K : injection synthétique
au harness + prefill-court-puis-génère en prod, méthode figée identique
baseline/sparse. `RUNNER_BATCH=0` épinglé. Golden à deux niveaux (fp32 CPU +
dtype prod non-régression). Caches off en prod long-contexte aussi.

## Round 3 — Codex (VERDICT: APPROVED)

Les 5 points intégrés. Plan cohérent pour une phase 1 decode-only ; blockers
prefill/indexer/union hors scope, hypothèses MLX cadrées, `RUNNER_BATCH=0`
explicite, méthode bench 128K viable. Deux nits non-bloquants : (1) « 3 full
layers O(n²) » vrai au prefill, O(K) au decode — déjà neutralisé par la phrase
suivante ; (2) golden dtype prod doit comparer les tenseurs réellement produits
par le modèle chargé (poids Q6 + activations) — **foldé** dans le plan.

**Convergé en 3 rounds.** Plan verrouillé.
