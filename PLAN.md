# Plan : attention block-sparse (MSA) pour MiniMax-M3 en MLX — phase 1 (DECODE)
_Verrouillé via grill-with-docs — Claude + Sophie, 2026-06-13. Révisé après Codex round 1. Termes par `CONTEXT-msa.md`._

## Goal

Rendre le **decode** M3 plus rapide au long contexte (128K) en réalisant
*physiquement* la **MSA** au decode : chaque token généré n'attend que les
**blocs sélectionnés** par le **lightning indexer** (≤16 blocs de 128 = ≤2048
clés) au lieu du **masque-dense** qui calcule tout le `q·kᵀ`. Sans changer la
sortie. Servi **single-node Python** (`nodes=1`, ADR-0002).

**Re-scope post-Codex** : phase 1 = **decode seul**. L'attention principale par
token passe de `64 têtes × K` à `64 × 2048` (réduction ~62× du K *de l'attention
principale* à 128K). **MAIS le gain decode réel est borné** : l'indexer scanne
encore tout le contexte (`4 têtes × K`/token) et 3 couches restent full-attention
→ attendre **~10-13× mesuré** à 128K, pas 62× (l'indexer devient le plancher).
**Phase 1 POSSÈDE l'indexer exact** : au decode il est O(K) (scan de tous les
blocs pour le top-16 *exact* — irréductible sans approximation), déjà ~minimal
(4 têtes, pas de softmax) ; phase 1 le profile et garantit qu'il est efficace,
c'est lui qui fixe le plafond ~13-20×. **L'indexer hiérarchique APPROXIMATIF**
(scorer des super-blocs pour *dépasser* O(K), au prix d'une sélection approchée)
= phase 2, avec porte qualité. Le **prefill** block-sparse (union adaptative,
indexer tiled `[S,K]`) est aussi phase 2.

## Approach

0. **Baseline (cluster, nœud 512 GB).** Decode tok/s du M3 single-node
   **masque-dense** à 8k / 32k, et 128k **avec garde OOM/timeout** (le masque
   dense `[B,1,S,K]` peut OOM en prefill — mesurer ce qui passe, extrapoler le
   reste, instrumenté). Précédé d'un reboot `.29` (purge leak wired).
   **Constitution du cache long-contexte** (le prefill dense 128K peut OOM, donc
   on ne peut pas y arriver par un seul gros prompt) : injection synthétique du
   cache dans le harness (clés/valeurs aléatoires de longueur K) pour mesurer le
   decode isolément ; en prod, prefill court + génération jusqu'à K. Méthode
   figée avant le bench, identique baseline vs sparse pour une comparaison juste.
1. **Block gather decode (local, harness tiny).** À S_q=1, par couche sparse :
   - indexer → sélection `[B,1,topk]` (ordre : indexer **avant** `update_and_fetch`) ;
   - **clamp les `-1`** (blocs invalides) vers un index sûr, marquer ces slots ;
   - gather des blocs k/v depuis les K/V **retournés par `update_and_fetch`** via
     **`mx.take_along_axis`** (indices broadcastés par batch — PAS `mx.take`) ;
   - `T_kv` **fixe** = `topk × block` ; masque additif `[B,1,1,T_kv]` composant
     **bloc-sélectionné ∧ causal-token ∧ valide** (les `-1` → `-inf`) ;
   - `scaled_dot_product_attention` sur le sous-ensemble.
   **Golden test à deux niveaux** : (a) **fp32 CPU** — équivalence math vs
   masque-dense `< 1e-6` (indices triés par position pour minimiser la dérive de
   réduction softmax) ; (b) **modèle prod chargé** — comparer les tenseurs
   réellement produits par le modèle Q6 tel que chargé (poids quantifiés +
   activations), tolérance réaliste + non-régression logits/tokens (greedy
   identique sur N tokens). Pas « bit-identical ». À k_len court (< 2048,
   sparse=dense) ET long.
2. **Wire + deploy.** Seuil **strict** : `k_len <= 2048` → chemin masque-dense
   actuel inchangé ; `k_len > 2048` → block-gather decode. **Forcer le chemin
   legacy single-slot** : `RUNNER_BATCH=0` au load M3 sparse (le runner part en
   BatchGenerator si `RUNNER_BATCH=1`), idéalement déclenché par détection
   `model_type == minimax_m3` côté orchestrateur. **B>1 hors scope.** Charger `nodes=1`.
3. **Bench final (cluster).** Decode tok/s à 8k/32k/128k, A vs baseline ; smoke
   thinking ON/OFF (non-régression) ; vérifier le budget mémoire réel.

## Key decisions & tradeoffs

- **Phase 1 = decode-only, exact (voie A).** Prefill + indexer + kernel Metal =
  phase 2. (Re-scope post-Codex de la réponse Q1 « les deux phases » — à bénir
  au sign-off : decode est le coût par-token de Companion, le gros gain.)
- **Mécanisme : block gather pur-MLX** (`take_along_axis` + SDPA sur sous-ensemble,
  `T_kv` fixe). Pas de kernel, pas de ragged.
- **Exactitude** : équivalence mathématique au masque-dense, `< 1e-6` jugée sur
  CPU (le matmul Metal fp32 a un plancher ~1e-3), indices triés. Golden-bloquant.
- **Voie B (approché) : HORS critère de livraison.** Flag expérimental
  non-production seulement ; si jamais activée (prefill phase 2), porte qualité
  (T01 + éval factuelle). Le decode ne dérive jamais.
- **Plateforme : Python single-node** (ADR-0002 ; supprime bubble + JACCL).
- **Budget mémoire explicite** : poids 355 GB + KV fp16 @128K (~16 GB, 4 kv-heads
  ×128 ×60 ×128K ×2×2) + `idx_keys` fp (~2 GB) ≈ 373 GB < 460 budget. `idx_keys`
  ajouté au calcul de `_cache_size_bytes`. **Caches session/prewarm désactivés
  (ou plafonnés) pour M3 long-contexte EN PROD aussi** — pas seulement au bench —
  tant que la comptabilité mémoire n'est pas prouvée (ils recréent la pression
  à 128K).
- **Observabilité (requise avant bench)** : logger gathered-keys/token, count de
  fallback, timings indexer/gather/SDPA, mémoire max, offset cache + longueur `idx_keys`.
- **Critère d'acceptation** : (1) golden decode deux niveaux — fp32 CPU `< 1e-6`
  + dtype prod non-régression logits/tokens — court ET long ; (2) decode tok/s à
  ≥32k **nettement** > baseline (gain borné par l'indexer, attendu ~10-13× à
  128K, mesuré) ; (3) zéro régression < 2k + parité/smoke verts ; (4) déployé +
  smoké single-node, budget mémoire tenu.

## Risks / open questions

- **3 couches full-attention** (0-2, non sparse) restent O(n²) — plancher au
  prefill, négligeable au decode (S_q=1). À profiler séparément.
- **Overhead du gather** (`take_along_axis` + assemblage) vs calcul économisé :
  doit gagner à ≥32k ; vérifier pas de régression autour de 2k-8k (le seuil 2048
  protège en deçà).
- **`idx_keys` au decode** : déjà offset-indexé (fix de cette nuit) ; le gather
  lit les K/V principaux post-`update_and_fetch`, pas `idx_keys` — découplage à
  re-vérifier sous trim.
- **M3 Q8 single-node** (~470 GB) au-dessus du budget wired 460 → Q8 reste
  2-node ou attend la levée du budget. Q6 = cas nominal.

## Out of scope (phase 1)

- **Prefill block-sparse** (union adaptative plafonnée + indexer tiled) → phase 2.
- **Indexer hiérarchique APPROXIMATIF** (dépasser O(K) au decode) → phase 2.
  (L'indexer EXACT O(K) reste DANS la phase 1 — il est nécessaire à la sélection.)
- **Kernel Metal custom** (`mx.fast.metal_kernel`) → phase 2.
- Port Telemak/Swift de M3 ; MTP / têtes EAGLE ; bench 1M ; tensor-parallel.
