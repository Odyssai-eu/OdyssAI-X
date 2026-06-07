# Plan : porter DeepSeek-V4-Flash-8bit dans OdyssAI-X (model_type `deepseek_v4`)
_Recon : workflow w2tumujwa (4 agents, ancrés file:line vs `ref-deepseek-v4/` + le checkpoint réel 3513 clés sur .29). Pour review adverse Codex AVANT d'écrire le modèle._

## La réalisation
Le draft de vendredi (`scripts/mlx_models/deepseek_v4.py`) est calé sur V4-**Pro** et réutilise l'attention/MoE de `deepseek_v32`. Le modèle staged sur .29/.30/.31 est **DeepSeek-V4-Flash-8bit** (mlx-community, **282 GB, 65 shards, 3513 tensors, 8-bit affine g64, clés FLAT sans préfixe `model.`**). Flash ≠ Pro ≠ v32 — attention et router diffèrent sur tous les axes. **Le draft couvre ~20% ; il ne chargerait pas.** C'est un module modèle from-scratch, ~13 pts.

## Architecture Flash (43 layers, TOUS MoE, hidden 4096, vocab 129280)

### A. Attention — REWRITE (Diff 13) → `DeepseekV4FlashAttention`
- **MLA** : `wq_a`→`q_norm`→`wq_b` (q 64h×512) ; `wkv`→`kv_norm` (latent COMBINÉ 512, K=V, num_kv_heads=1) ; rope sur les 64 derniers dims ; q-RMS par-tête **sans poids** (ref:498).
- **attn_sink** [64] f32 = token-sink virtuel au dénominateur du softmax (kernel:345).
- **o-LoRA groupé** : `wo_a.0..7` (8×[1024,4096] down par groupe) + `wo_b` [4096,8192] up ; **inverse-rope** sur la sortie d'abord (ref:534-542).
- **Compressor par-layer** (compress_ratios) : gated-softmax mean-pool sur `ratio` tokens + `ape` + `norm` + rope ; overlap (ratio4→coff2) vs plain (ratio128) — compression KV longue-contexte.
- **Indexer** (layers pairs ratio-4) : son propre Compressor (rotated) + `wq_b` + `weights_proj` → top-512 positions compressées.
- **Sliding window 128** + gather sparse. Rope par-layer : couches compressées = YaRN(orig 65536, θ 160000, factor 16) ; couches sliding-only = θ base, pas de YaRN.
- Variante par layer (compress_ratios `[0,0,4,128,...]`) : **L0/L1/L9 sliding-only** (pas de compressor) ; **pairs≥2 compressor+indexer** ; **impairs≥3 compressor seul**.
- **Premier atterrissage : attention dense-masquée émulant `sparse_attn` (correctness > perf)** ; différer le kernel Metal sparse + l'état incremental-decode du Compressor.

### B. MoE / router — REWRITE le gate, REUSE SwitchGLU (Diff 5) → `FlashMoE`
- **Gate** : `scores = sqrt(softplus(x @ gate.weightᵀ))` (scoring_func=sqrtsoftplus) ; **dual-mode** : layers 0-2 **hash** (`indices = tid2eid[input_ids]`, pas de scoring) ; layers 3-42 **noaux_tc** (`scores+bias` → top-6) ; poids lus sur les scores **NON biaisés**, renorm, ×`routed_scaling_factor`=1.5.
- **Experts** : SwitchGLU (reuse), pré-stackés `w1`=gate/`w2`=down/`w3`=up [256,…], **clamp swiglu_limit=10**. Shared expert toujours actif (n_shared=1).
- **Threader `input_ids`** model→layer→gate (le draft ne l'a pas). Toutes les 43 layers MoE (pas de préfixe dense).

### C. HyperConnection + embed/head + sanitize — (Diff 5)
- HC **par-layer** (`hc_attn_fn`/`hc_ffn_fn` [24,16384], base[24], scale[3]) + **top-level** `hc_head` ([4,16384]) — **le design wrapper-par-layer du draft est CONFIRMÉ correct**. embed→hc = **réplication** (broadcast_to ok). `hc_split_sinkhorn` matche la réf. **Pas de MTP** (le checkpoint l'a droppé).
- embed/head = QuantizedEmbedding/Linear 8-bit (auto via `class_predicate`). norm plain.
- **sanitize rewrite (le cœur)** : clés FLAT → ajouter préfixe `model.` ; `embed`→`model.embed_tokens`, `head`→`lm_head`, `norm`→`model.norm` ; `ffn.experts.w1/w2/w3` → `switch_mlp.gate_proj/down_proj/up_proj` (**RENAME pas stack** — déjà stackés, contrairement à v32) ; `gate.weight` reste ; `tid2eid`/`bias`/`attn_sink`/`ape`/`hc_*` gardés **bruts** (PAS quantizés). **Aucun dequant FP8** (affine natif).

### D. Load / quant / deploy — REUSE l'infra (Diff 8)
- Le load/quant mlx-lm est model-agnostic + **key-driven** (`class_predicate` quantize là où `.scales` existe). Le chemin `pipeline_auto_parallel` = celui de Hy3. `deepseek_v4` déjà dans `TENSOR_CAPABLE` (api.py:698) mais Flash est **pipeline-only** (kv_heads=1) → tensor jamais déclenché.
- **Deploy** : copier `deepseek_v4.py` dans `mlx_lm/models/` de chaque node (manuel, comme hy_v3 ; `bootstrap-node.sh` ne sync pas `mlx_models/`).
- **Le forward longue-contexte = LE risque** : Compressor + sliding-window-128 + Indexer-topk-512 + attn_sink + cache par-layer (taille = window + max_seq//ratio, buffers compressé+window séparés) — **PAS un KVCache mlx-lm standard** ; un cache naïf serait numériquement faux.

## Exécution phasée
- **P0 — validation structurelle (sans load)** : instancier depuis le vrai `config.json`, diff des clés params vs les 3513 clés checkpoint (0 manquante, 0 inattendue après sanitize). Cheap, attrape le module-tree + le sanitize. (Comme hy_v3.)
- **P1 — le module modèle** : écrire FlashAttention (émulation dense-mask) + FlashMoE (dual-gate + input_ids) + garder HC + rewrite sanitize + embed/head/norm. Forward py sur une tiny config.
- **P2 — load distribué** : deploy nodes, décharger Hy3, charger Flash pipeline 3-node (.29+.30+.31, RAM ~19/12/12 layers). Vérifier que **ça CHARGE** (quant auto, clés matchent, strict=False).
- **P3 — correctness forward** : cohérence greedy EN/code/FR (le cache longue-contexte + l'attention sparse doivent être justes) — la barre d'acceptation hy_v3. **LA** validation dure.

## Key decisions
- Module from-scratch, **NE PAS greffer v32** attn/MoE. Réutiliser seulement SwitchGLU + nn.Linear/Embedding/RMSNorm + l'infra load/quant/pipeline.
- Attention correctness-first : dense-mask émulant sparse_attn ; différer le kernel Metal + l'état incremental-decode.
- Nommer les sous-modules MLX pour minimiser le sanitize, mais le rename des experts + le préfixe `model.` + embed/head sont inévitables.
- Threading `input_ids` = structurel (model→layer→gate), à faire proprement.

## Risks / open questions
- **Correctness du forward longue-contexte** (Compressor + cache custom + attention sparse) = le vrai risque, pas le load.
- Plomberie quant embed/head/experts : vérifier que `class_predicate` construit QuantizedEmbedding/Linear/SwitchLinear depuis les `.scales/.biases` affine g64 (pas de chemin FP8).
- Mapping `gate.bias` → `e_score_correction_bias` vs le noaux_tc+sqrtsoftplus de Flash (peut différer du sigmoid+group de v32).
- Orientation des tenseurs experts (confirmer [E,out,in], pas de swapaxes).
- Exactitude du clamp swiglu_limit=10 (subclasser l'activation ou accepter le drift).
- 282 GB : .29 (512 GB) peut peut-être le tenir solo pour un forward rapide ; sinon pipeline 3-node.

## Out of scope (cette passe)
- Le kernel Metal `sparse_attn` custom (émulation dense d'abord).
- L'état streaming du Compressor en incremental-decode (prefill/full-recompute d'abord).
- La tête MTP/nextn (absente du checkpoint).
- DeepSeek-V4-Pro (780 GB) — Flash d'abord.
- Rewrite de `DeepseekV4ShardingStrategy` d'auto_parallel (Flash est pipeline-only).
