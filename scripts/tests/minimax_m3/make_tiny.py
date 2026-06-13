#!/usr/bin/env python3
"""Tiny M3 : config réduite + poids aléatoires, sauvés en NOMMAGE HUB
(language_model.model.*, block_sparse_moe.experts.N.w1/w2/w3, self_attn.index_*)
— le même contrat que le vrai checkpoint, pour tester le shim torch ET le
sanitize MLX, pas seulement la math."""
import json
import numpy as np
import torch
from safetensors.torch import save_file

OUT = "/tmp/m3-night/tiny-hub"
import os
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)

H = 64           # hidden
NH, NKV, HD = 4, 2, 16
IDX_H, IDX_D, BLK, TOPK, LOCAL = 2, 16, 4, 2, 1
NL = 4           # layers: 0 = full+dense, 1-3 = sparse+moe
NE, TOPE, I_MOE, I_SH, I_DENSE = 8, 2, 32, 32, 96
VOCAB = 96

cfg = {
    "model_type": "minimax_m3",
    "vocab_size": VOCAB,
    "hidden_size": H,
    "num_hidden_layers": NL,
    "num_attention_heads": NH,
    "num_key_value_heads": NKV,
    "head_dim": HD,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5e6,
    "partial_rotary_factor": 0.5,
    "rope_parameters": {"rope_theta": 5e6, "rope_type": "default",
                         "partial_rotary_factor": 0.5},
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
    "dense_intermediate_size": I_DENSE,
    "intermediate_size": I_MOE,
    "shared_intermediate_size": I_SH,
    "num_local_experts": NE,
    "num_experts_per_tok": TOPE,
    "routed_scaling_factor": 2.0,
    "swiglu_alpha": 1.702,
    "swiglu_limit": 7.0,
    "moe_layer_freq": [0, 1, 1, 1],
    "layer_types": ["full_attention", "minimax_m3_sparse",
                     "minimax_m3_sparse", "minimax_m3_sparse"],
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "index_n_heads": IDX_H,
    "index_head_dim": IDX_D,
    "index_block_size": BLK,
    "index_topk_blocks": TOPK,
    "index_local_blocks": LOCAL,
    "sparse_attention_config": {
        "use_sparse_attention": True,
        "sparse_attention_freq": [0, 1, 1, 1],
        "sparse_num_index_heads": IDX_H,
        "sparse_index_dim": IDX_D,
        "sparse_block_size": BLK,
        "sparse_topk_blocks": TOPK,
        "sparse_local_block": LOCAL,
        "sparse_init_block": 0,
        "sparse_score_type": "max",
    },
}
json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1)

def t(*shape):
    return (torch.randn(*shape, dtype=torch.float32) * 0.05)

w = {}
P = "language_model.model"
w[f"{P}.embed_tokens.weight"] = t(VOCAB, H)
w[f"{P}.norm.weight"] = t(H) * 0.1
w["language_model.lm_head.weight"] = t(VOCAB, H)

for l in range(NL):
    L = f"{P}.layers.{l}"
    w[f"{L}.input_layernorm.weight"] = t(H) * 0.1
    w[f"{L}.post_attention_layernorm.weight"] = t(H) * 0.1
    A = f"{L}.self_attn"
    w[f"{A}.q_proj.weight"] = t(NH * HD, H)
    w[f"{A}.k_proj.weight"] = t(NKV * HD, H)
    w[f"{A}.v_proj.weight"] = t(NKV * HD, H)
    w[f"{A}.o_proj.weight"] = t(H, NH * HD)
    w[f"{A}.q_norm.weight"] = t(HD) * 0.1
    w[f"{A}.k_norm.weight"] = t(HD) * 0.1
    if l >= 1:  # sparse layers
        w[f"{A}.index_q_proj.weight"] = t(IDX_H * IDX_D, H) * 8  # séparer les blocs
        w[f"{A}.index_k_proj.weight"] = t(IDX_D, H) * 8
        w[f"{A}.index_q_norm.weight"] = t(IDX_D) * 0.1
        w[f"{A}.index_k_norm.weight"] = t(IDX_D) * 0.1
        M = f"{L}.block_sparse_moe"
        w[f"{M}.gate.weight"] = t(NE, H) * 20  # séparer les experts (anti-tie)
        w[f"{M}.e_score_correction_bias"] = t(NE) * 0.01
        w[f"{M}.shared_experts.gate_proj.weight"] = t(I_SH, H)
        w[f"{M}.shared_experts.up_proj.weight"] = t(I_SH, H)
        w[f"{M}.shared_experts.down_proj.weight"] = t(H, I_SH)
        for e in range(NE):
            w[f"{M}.experts.{e}.w1.weight"] = t(I_MOE, H)
            w[f"{M}.experts.{e}.w2.weight"] = t(H, I_MOE)
            w[f"{M}.experts.{e}.w3.weight"] = t(I_MOE, H)
    else:  # dense mlp
        w[f"{L}.mlp.gate_proj.weight"] = t(I_DENSE, H)
        w[f"{L}.mlp.up_proj.weight"] = t(I_DENSE, H)
        w[f"{L}.mlp.down_proj.weight"] = t(H, I_DENSE)

save_file(w, f"{OUT}/model.safetensors")
idx = {"metadata": {"total_size": sum(t.numel() * 4 for t in w.values())},
       "weight_map": {k: "model.safetensors" for k in w}}
json.dump(idx, open(f"{OUT}/model.safetensors.index.json", "w"))
print(f"tiny hub checkpoint: {len(w)} tensors -> {OUT}")
