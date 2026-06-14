#!/usr/bin/env python3
"""Canari MSA — selection-vs-full + garde du piège __post_init__.

OdyssAI-X#53. CE test comble le trou de validation qui a coûté des semaines :
toute la tier golden (golden_decode / challenge2 / challenge3) prouve
`gather == dense-mask` — la MÊME politique de sélection appliquée à deux
implémentations — mais JAMAIS `selection == full attention`. Le bug
`__post_init__` (nested sparse_attention_config qui écrasait les flat index_*)
est passé inaperçu précisément là : tout tournait en 1/16 pendant que les
goldens restaient verts, car leur référence était elle-même la sélection
mal configurée.

Ce canari (le test « écrit-exécuté-puis-jeté » qui a produit le 6,6%->2,1%,
enfin committé) garde DEUX invariants que les goldens n'ont pas :

  PARTIE 1 — __post_init__ : les champs flat index_* explicites GAGNENT sur le
             nested sparse_attention_config. Si le nested ré-écrit silencieusement
             le flat, on retombe dans le bug 1/16. Test unitaire pur (pas de
             forward), donc rapide et imparable.

  PARTIE 2 — selection-vs-full (forward déterministe, teacher-forcé sur le tiny) :
             (a) sous le seuil (k_len <= topk*block), sparse == full EXACT
                 (tous les blocs sélectionnés) ;
             (b) au-dessus du seuil, sparse(topk=2,local=1) DIVERGE de full
                 (topk=tous) — la sélection a un effet réel end-to-end. Si la
                 divergence est nulle, soit la sélection n'agit pas, soit la
                 config est silencieusement écrasée -> FAIL ;
             (c) élargir la couverture RAPPROCHE du plein :
                 div(topk=3,local=2) <= div(topk=2,local=1). C'est la direction
                 du fix prod (16/1 -> 24/8). Si élargir n'améliore rien, la
                 config ne prend pas effet -> FAIL.

Jugé sur CPU (plancher Metal fp32 ~1e-3). Déterministe (tokens fixes, seeds fixes).
Aucune dépendance réseau / poids prod : le tiny-hub suffit (construit à la volée
s'il manque). À mettre en CI : `python canary_selection_vs_full.py` -> exit 0/1.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Source de vérité CI = le module vendored du repo ; fallback = staging déployé.
sys.path.insert(0, "/tmp/m3-night")
sys.path.insert(0, os.path.join(HERE, "..", "..", "patches"))

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)  # exactitude jugée sur CPU (plancher Metal fp32 ~1e-3)

try:
    import minimax_m3_model as minimax_m3  # repo: scripts/patches/minimax_m3_model.py
except ImportError:
    import minimax_m3  # staging /tmp/m3-night

TINY = os.path.join(HERE, "_tiny_canary")


def ensure_tiny():
    """Construit un tiny-hub déterministe en numpy+mlx (PAS de torch — portable CI).
    Même contrat de nommage HUB que make_tiny.py (language_model.model.*,
    block_sparse_moe.experts.N.w1/w2/w3, self_attn.index_*) pour exercer sanitize."""
    if os.path.exists(f"{TINY}/config.json"):
        return
    os.makedirs(TINY, exist_ok=True)
    H, NH, NKV, HD = 64, 4, 2, 16
    IDX_H, IDX_D, BLK, TOPK, LOCAL = 2, 16, 4, 2, 1
    NL, NE, TOPE, I_MOE, I_SH, I_DENSE, VOCAB = 4, 8, 2, 32, 32, 96, 96
    cfg = {
        "model_type": "minimax_m3", "vocab_size": VOCAB, "hidden_size": H,
        "num_hidden_layers": NL, "num_attention_heads": NH, "num_key_value_heads": NKV,
        "head_dim": HD, "rms_norm_eps": 1e-6, "rope_theta": 5e6,
        "partial_rotary_factor": 0.5,
        "rope_parameters": {"rope_theta": 5e6, "rope_type": "default",
                            "partial_rotary_factor": 0.5},
        "max_position_embeddings": 4096, "tie_word_embeddings": False,
        "dense_intermediate_size": I_DENSE, "intermediate_size": I_MOE,
        "shared_intermediate_size": I_SH, "num_local_experts": NE,
        "num_experts_per_tok": TOPE, "routed_scaling_factor": 2.0,
        "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "moe_layer_freq": [0, 1, 1, 1],
        "layer_types": ["full_attention", "minimax_m3_sparse",
                        "minimax_m3_sparse", "minimax_m3_sparse"],
        "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
        "index_n_heads": IDX_H, "index_head_dim": IDX_D, "index_block_size": BLK,
        "index_topk_blocks": TOPK, "index_local_blocks": LOCAL,
        "sparse_attention_config": {
            "use_sparse_attention": True, "sparse_attention_freq": [0, 1, 1, 1],
            "sparse_num_index_heads": IDX_H, "sparse_index_dim": IDX_D,
            "sparse_block_size": BLK, "sparse_topk_blocks": TOPK,
            "sparse_local_block": LOCAL, "sparse_init_block": 0, "sparse_score_type": "max"},
    }
    json.dump(cfg, open(f"{TINY}/config.json", "w"), indent=1)

    rng = np.random.default_rng(42)
    def t(*shape, s=0.05):
        return mx.array((rng.standard_normal(shape) * s).astype(np.float32))
    w = {}
    P = "language_model.model"
    w[f"{P}.embed_tokens.weight"] = t(VOCAB, H)
    w[f"{P}.norm.weight"] = t(H, s=0.1)
    w["language_model.lm_head.weight"] = t(VOCAB, H)
    for l in range(NL):
        L = f"{P}.layers.{l}"; A = f"{L}.self_attn"
        w[f"{L}.input_layernorm.weight"] = t(H, s=0.1)
        w[f"{L}.post_attention_layernorm.weight"] = t(H, s=0.1)
        w[f"{A}.q_proj.weight"] = t(NH * HD, H); w[f"{A}.k_proj.weight"] = t(NKV * HD, H)
        w[f"{A}.v_proj.weight"] = t(NKV * HD, H); w[f"{A}.o_proj.weight"] = t(H, NH * HD)
        w[f"{A}.q_norm.weight"] = t(HD, s=0.1); w[f"{A}.k_norm.weight"] = t(HD, s=0.1)
        if l >= 1:
            w[f"{A}.index_q_proj.weight"] = t(IDX_H * IDX_D, H, s=0.4)  # *8 pour séparer les blocs
            w[f"{A}.index_k_proj.weight"] = t(IDX_D, H, s=0.4)
            w[f"{A}.index_q_norm.weight"] = t(IDX_D, s=0.1)
            w[f"{A}.index_k_norm.weight"] = t(IDX_D, s=0.1)
            M = f"{L}.block_sparse_moe"
            w[f"{M}.gate.weight"] = t(NE, H, s=1.0)  # séparer les experts (anti-tie)
            w[f"{M}.e_score_correction_bias"] = t(NE, s=0.01)
            w[f"{M}.shared_experts.gate_proj.weight"] = t(I_SH, H)
            w[f"{M}.shared_experts.up_proj.weight"] = t(I_SH, H)
            w[f"{M}.shared_experts.down_proj.weight"] = t(H, I_SH)
            for e in range(NE):
                w[f"{M}.experts.{e}.w1.weight"] = t(I_MOE, H)
                w[f"{M}.experts.{e}.w2.weight"] = t(H, I_MOE)
                w[f"{M}.experts.{e}.w3.weight"] = t(I_MOE, H)
        else:
            w[f"{L}.mlp.gate_proj.weight"] = t(I_DENSE, H)
            w[f"{L}.mlp.up_proj.weight"] = t(I_DENSE, H)
            w[f"{L}.mlp.down_proj.weight"] = t(H, I_DENSE)
    mx.save_safetensors(f"{TINY}/model.safetensors", w)


def base_cfg():
    return json.load(open(f"{TINY}/config.json"))


# ── PARTIE 1 — le garde __post_init__ (unitaire, pas de forward) ────────────

def test_post_init_flat_wins():
    fails = []

    # (1a) flat explicite (24/8) + nested defaults (16/1) -> flat DOIT gagner.
    cfg = base_cfg()
    cfg["index_topk_blocks"] = 24
    cfg["index_local_blocks"] = 8
    cfg["sparse_attention_config"] = dict(cfg.get("sparse_attention_config", {}),
                                          sparse_topk_blocks=16, sparse_local_block=1)
    a = minimax_m3.ModelArgs.from_dict(cfg)
    if a.index_topk_blocks != 24:
        fails.append(f"flat index_topk_blocks écrasé : {a.index_topk_blocks} != 24 "
                     "(le nested sparse_topk_blocks=16 a clobbered le flat — BUG __post_init__)")
    if a.index_local_blocks != 8:
        fails.append(f"flat index_local_blocks écrasé : {a.index_local_blocks} != 8 "
                     "(le nested sparse_local_block=1 a clobbered le flat — BUG __post_init__)")

    # (1b) flat au défaut (16/1) + nested non-défaut (4/3) -> nested DOIT s'appliquer
    #      (la réconciliation legacy reste fonctionnelle quand c'est l'intention).
    cfg2 = base_cfg()
    cfg2.pop("index_topk_blocks", None)   # absent -> défaut dataclass (16)
    cfg2.pop("index_local_blocks", None)  # absent -> défaut dataclass (1)
    cfg2["sparse_attention_config"] = dict(cfg2.get("sparse_attention_config", {}),
                                           sparse_topk_blocks=4, sparse_local_block=3)
    b = minimax_m3.ModelArgs.from_dict(cfg2)
    if b.index_topk_blocks != 4:
        fails.append(f"nested non appliqué quand flat=défaut : index_topk_blocks={b.index_topk_blocks} != 4")
    if b.index_local_blocks != 3:
        fails.append(f"nested non appliqué quand flat=défaut : index_local_blocks={b.index_local_blocks} != 3")

    return fails


# ── PARTIE 2 — selection-vs-full (forward déterministe) ─────────────────────

def build_model(topk, local, weights):
    cfg = base_cfg()
    cfg["index_topk_blocks"] = topk
    cfg["index_local_blocks"] = local
    # garde le nested au défaut source (16/1) pour PROUVER que le flat gagne
    # dans le chemin forward, pas seulement en unitaire.
    cfg["sparse_attention_config"] = dict(cfg.get("sparse_attention_config", {}),
                                          sparse_topk_blocks=16, sparse_local_block=1)
    args = minimax_m3.ModelArgs.from_dict(cfg)
    # introspection défensive : la leçon « introspecter le ModelArgs chargé ».
    assert args.index_topk_blocks == topk and args.index_local_blocks == local, \
        f"ModelArgs chargé != voulu (topk={args.index_topk_blocks}/local={args.index_local_blocks})"
    model = minimax_m3.Model(args)
    model.load_weights(list(model.sanitize(dict(weights)).items()), strict=True)
    mx.eval(model.parameters())
    return model


def argmax_seq(model, tokens):
    """Prefill une séquence -> argmax par position [S]. build_block_mask applique
    la sélection au prefill (S>1) : varier topk/local change donc la sortie ici."""
    logits = model(mx.array(tokens)[None])  # [1, S, V]
    return np.array(mx.argmax(logits[0], axis=-1))


def test_selection_vs_full():
    cfg = base_cfg()
    block = cfg["index_block_size"]; topk = cfg["index_topk_blocks"]
    threshold = block * topk  # 4*2 = 8 (analogue prod 128*16 = 2048)
    V = cfg["vocab_size"]
    weights = mx.load(f"{TINY}/model.safetensors")

    S = 28
    tokens = ((np.arange(S) * 7 + 3) % V).astype(np.int32)

    full = argmax_seq(build_model(999, 1, weights), tokens)        # topk=999 -> tous les blocs
    sp21 = argmax_seq(build_model(2, 1, weights), tokens)          # source 1/16-analogue
    sp32 = argmax_seq(build_model(3, 2, weights), tokens)          # fix-analogue (plus large)

    # k_len(position i) = i+1 ; régime sparse = k_len > threshold (num_blocks > topk).
    below = [i for i in range(S) if (i + 1) <= threshold]
    above = [i for i in range(S) if (i + 1) > threshold]

    def div(a, b, idxs):
        return sum(int(a[i] != b[i]) for i in idxs) / max(1, len(idxs))

    d_below = div(full, sp21, below)
    d21 = div(full, sp21, above)
    d32 = div(full, sp32, above)

    print(f"  seuil topk*block = {threshold} | positions sous={len(below)} au-dessus={len(above)}")
    print(f"  (a) sous le seuil   : div(full, sparse2/1) = {d_below:.1%}  (attendu 0%)")
    print(f"  (b) au-dessus seuil : div(full, sparse2/1) = {d21:.1%}  (attendu > 0)")
    print(f"  (c) au-dessus seuil : div(full, sparse3/2) = {d32:.1%}  (attendu <= b)")

    fails = []
    if d_below != 0.0:
        fails.append(f"(a) sparse != full SOUS le seuil ({d_below:.1%}) — la sélection "
                     "devrait sélectionner tous les blocs là (exactitude rompue)")
    if d21 <= 0.0:
        fails.append("(b) sparse(2/1) == full au-dessus du seuil — la sélection N'AGIT PAS "
                     "end-to-end (config silencieusement écrasée, ou sélection inopérante : "
                     "exactement le mode du bug __post_init__)")
    if d32 > d21:
        fails.append(f"(c) élargir EMPIRE ({d32:.1%} > {d21:.1%}) — topk=3/local=2 devrait "
                     "rapprocher du plein ; la config ne prend pas effet ou la direction du fix est cassée")
    return fails


def main():
    ensure_tiny()
    all_fails = []

    print("PARTIE 1 — __post_init__ (flat index_* gagne sur le nested)")
    f1 = test_post_init_flat_wins()
    print("  PASS" if not f1 else "  FAIL")
    for m in f1:
        print("   -", m)
    all_fails += f1

    print("PARTIE 2 — selection-vs-full (forward déterministe)")
    f2 = test_selection_vs_full()
    print("  PASS" if not f2 else "  FAIL")
    for m in f2:
        print("   -", m)
    all_fails += f2

    if all_fails:
        print(f"\nCANARI FAIL — {len(all_fails)} invariant(s) rompu(s)")
        sys.exit(1)
    print("\nCANARI PASS — __post_init__ tient ET la sélection a un effet réel "
          "(sous le seuil exact, au-dessus elle diverge, élargir rapproche du plein)")


if __name__ == "__main__":
    main()
