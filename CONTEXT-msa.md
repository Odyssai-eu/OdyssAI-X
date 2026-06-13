# CONTEXT — MSA / attention block-sparse M3 (glossaire)

Glossaire du domaine « attention sparse MiniMax-M3 en MLX ». Termes uniquement,
pas de détails d'implémentation. Second contexte du repo (le premier,
`CONTEXT.md`, couvre JACCL/RDMA — domaines distincts).

- **MSA (MiniMax Sparse Attention)** — l'attention block-sparse de M3 : chaque
  requête n'attend qu'un sous-ensemble de blocs de clés choisis dynamiquement,
  au lieu de toutes les clés. Réf. kernel CUDA SM100 `github.com/MiniMax-AI/MSA`
  (non portable Metal — on réimplémente l'algorithme, pas le kernel).
- **Lightning indexer** — la branche de *sélection* (sans projection de valeur) :
  4 têtes d'index × 128 dims scorent chaque requête contre chaque clé, max-pool
  par bloc, et gardent le top-k de blocs. Ne produit aucune sortie résiduelle.
- **Block (bloc)** — unité de sélection : `index_block_size` = 128 clés
  contiguës. La sélection se fait à la granularité du bloc, jamais de la clé.
- **Sélection par-requête** — l'indexer renvoie `[B, S_q, topk]` : chaque token
  de requête choisit ses propres ≤ `index_topk_blocks` (=16) blocs. Slots
  invalides (futurs/vides) = `-1`. Bloc local (celui de la requête) toujours gardé.
- **Exact (voie A)** vs **approché (voie B)** — A : union-load des blocs d'une
  tuile de requêtes + masque par-requête → sortie identique au masque-dense
  (golden-testable). B : sélection représentative partagée par tuile → plus
  rapide, mais change la sortie. **B n'a PAS de filet de vérification** (≠ MTP,
  où le draft approché est corrigé par le verify) → B exige une porte qualité.
- **Masque-dense (référence actuelle)** — l'implémentation de cette nuit :
  `build_block_mask` matérialise un masque additif `[B,1,S,k_len]` et SDPA
  calcule tout le `q@kᵀ`. Fonctionnellement correct, mais quadratique : la
  sparsité est *logique* (bons tokens), pas *physique* (calcule tout). C'est la
  référence d'or pour l'exactitude, et la cible à dépasser pour la vitesse.
- **Régime utile** — la sparsité ne gagne qu'au-delà de `topk×block` = 2048
  tokens de contexte. En deçà, tous les blocs sont sélectionnés → sparse = dense.
- **Block gather** — le mécanisme retenu (phase 1) : rassembler (`mx.take` sur
  l'axe des clés) les blocs k/v sélectionnés et appeler `scaled_dot_product_attention`
  sur ce seul sous-ensemble. Pur MLX, pas de kernel. Le **kernel Metal custom**
  (`mx.fast.metal_kernel`) est la phase 2, une fois le gather validé.
