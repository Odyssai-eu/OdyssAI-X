# Copyright © 2026 OdyssAI
#
# Tencent Hy4-preview — `model_type: hy_v4`, arch `HYV4ForCausalLM`.
#
# 780 B (49 B actifs). Lignée deepseek_v32 : MLA + DSA à indexer PARTAGE
# (indexer_types full/shared, comme GLM-5.3 FULL / glm_moe_dsa) + iHC 4 flux
# résiduels (variante « indépendante » : PAS de sinkhorn, fp32 forcé,
# hc_magnitude) + MoE 256/8 + shared expert + gated MLA (linear_gate sigmoid)
# + learnable attention sink + lm_head fp32. MTP droppé.
#
# Port mlx-lm dérivé de la référence transformers PR#48473 ("Add h4",
# modeling_hy_v4.py) et composé de nos modules cousins prouvés :
#   - attention/indexer/topk-partagé : lignée deepseek_v32 (cf. glm_moe_dsa)
#   - MoE (router e_score + switch_mlp + shared_experts) : DeepseekV32MoE
#   - iHC multi-flux + hc_head : réécrits ici (variante hy, pas la dsv4 sinkhorn)
#
# Deltas vérité-terrain checkpoint (outer format, experts DEJA fusionnés) :
#   attn gate = `linear_gate` ; sinks = `learnable_sink_param` ;
#   HC/couche = `hc_attn_layer.hc_pre` + `hc_mlp_layer.hc_pre`
#   {hc_fn[8,H*hc],hc_base[8],hc_scale[2]} ; tête = `hc_head.hc_head_{fn,base,scale}` ;
#   experts = `mlp.experts.gate_up_proj[256,4096,H]` fusionné -> à SPLITTER en
#   switch_mlp.{gate_proj,up_proj} + down_proj (deepseek_v32 stacke du per-index,
#   pas du pré-fusionné, d'où le split maison dans sanitize()).
#
# Déploiement : ce fichier va dans `mlx_lm/models/hy_v4.py` de chaque venv
# mlx-cluster. Sharding distribué = PipelineMixin (slice `.layers`), l'état
# inter-noeud est le tenseur multi-flux [B, S, hc_mult, D].

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask
from .cache import CacheList, KVCache
from .switch_layers import SwiGLU

# transformers < 5.16 ne connait pas `model_type: hy_v4` : AutoTokenizer tombe
# alors sur AutoConfig qui leve KeyError. On enregistre un config no-op des
# l'import de ce module (le runner importe hy_v4 via load_model AVANT de charger
# le tokenizer) pour que le tokenizer se charge sans transformers 5.16.
try:
    from transformers import AutoConfig as _AutoConfig, PretrainedConfig as _PC

    _AutoConfig.register(
        "hy_v4", type("HYV4AutoConfig", (_PC,), {"model_type": "hy_v4"})
    )
except Exception:
    pass  # deja enregistre (transformers 5.16) ou collision -> no-op


def _limited_swiglu(gate, up, limit):
    if limit and limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class LimitedSwiGLU(nn.Module):
    # Clamp swiglu (gpt_oss/hy) : SANS lui, les activations d'experts entrainees
    # (> swiglu_limit) explosent -> flux residuel sature -> collapse. Invisible a
    # l'oracle poids-random (petites activations), fatal en poids reels.
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):  # SwitchGLU appelle activation(x_up, x_gate)
        return _limited_swiglu(gate, x, self.limit)
from .deepseek_v32 import DeepseekV32MLP, DeepseekV32MoE


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "hy_v4"
    vocab_size: int = 120832
    hidden_size: int = 6144
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    rms_norm_eps: float = 1e-5
    # MLA
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    # MoE
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.827
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    # DSA indexer
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    indexer_types: Optional[List[str]] = None
    # iHC
    hc_mult: int = 4
    hc_magnitude: float = 2.0
    hc_eps: float = 1e-6
    # attention sink / swiglu clamp
    learnable_sink_init: float = 0.0
    swiglu_limit: float = 10.0
    # layer patterns
    mlp_layer_types: Optional[List[str]] = None
    layer_types: Optional[List[str]] = None
    # rope
    max_position_embeddings: int = 1048576
    rope_parameters: Optional[Dict] = None
    rope_theta: float = 1e7
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    sliding_window: Optional[int] = None

    def __post_init__(self):
        if self.rope_parameters is not None:
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        if self.indexer_types is None:
            # défaut réf : couche 0 full, puis full toutes les 4 (indices 1,5,9,...)
            self.indexer_types = [
                "full" if (i == 0 or (i - 1) % 4 == 0) else "shared"
                for i in range(self.num_hidden_layers)
            ]


class HYV4RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


def _rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # RoPE non-interleave (rotate_half), appliquée sur la tranche rope.
    # x: [..., D] avec D pair ; cos/sin: [L, D] broadcastés.
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rot = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rot * sin


class HYV4RotaryEmbedding(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        dim = config.qk_rope_head_dim
        inv_freq = 1.0 / (
            config.rope_theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        )
        self._inv_freq = inv_freq  # [dim/2]

    def __call__(self, positions: mx.array):
        # positions: [L] -> cos/sin: [L, dim]
        freqs = positions[:, None].astype(mx.float32) * self._inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


class HYV4Indexer(nn.Module):
    """DSA lightning-indexer (deepseek_v32 style) + k_norm LayerNorm (delta hy)."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim ** -0.5
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.rms_norm_eps)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)

    def __call__(self, x, q_resid, cos, sin, mask, cache=None):
        B, L, _ = x.shape
        q = self.wq_b(q_resid).reshape(B, L, self.n_heads, self.head_dim)
        # rope flippée : pass=[0:head_dim-rope], rot=[head_dim-rope:]
        split = self.head_dim - self.rope_dim
        q_pass, q_rot = q[..., :split], q[..., split:]
        k = self.k_norm(self.wk(x)).reshape(B, L, 1, self.head_dim)
        k_pass, k_rot = k[..., :split], k[..., split:]
        cse = cos[None, :, None, :self.rope_dim]
        sne = sin[None, :, None, :self.rope_dim]
        q_rot = _rope(q_rot, cse, sne)
        k_rot = _rope(k_rot, cse, sne)
        q = mx.concatenate([q_pass, q_rot], axis=-1).transpose(0, 2, 1, 3)  # [B,Hh,L,D]
        k = mx.concatenate([k_pass, k_rot], axis=-1).transpose(0, 2, 1, 3)  # [B,1,L,D]
        # cache des clés indexer -> en decode on score q contre TOUTES les clés
        # cachées (sinon topk de longueur 1 -> restriction a 1 position -> collapse).
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((B, 1, k.shape[2], 0), k.dtype))
        T = k.shape[2]
        # Pas de restriction quand seq <= topk (cas du vrai modele : 158 <= 2048).
        # None -> l'attention attend tout (causal). C'est la garde deepseek_v32.
        if T <= self.index_topk:
            return None
        scores = mx.matmul(q.astype(mx.float32), k.astype(mx.float32).swapaxes(-1, -2))  # [B,Hh,L,T]
        scores = nn.relu(scores)
        w = (
            self.weights_proj(x).astype(mx.float32)
            * (self.n_heads ** -0.5)
            * self.softmax_scale
        )  # [B,L,Hh]
        w = w.transpose(0, 2, 1)[..., None]        # [B,Hh,L,1]
        index_scores = (scores * w).sum(axis=1)    # [B,L,T]
        if mask is not None:
            index_scores = index_scores + mask
        topk = min(self.index_topk, T)
        return mx.argpartition(-index_scores, topk - 1, axis=-1)[..., :topk].astype(mx.int32)


def _sdpa_with_sink(q, k, v, scale, mask, sinks):
    # q,k,v: [B,H,L,D] / [B,H,T,Dk|Dv]. sinks: [H]. mask: [B,1|H,L,T] additif ou None.
    scores = (q.astype(mx.float32) * scale) @ k.astype(mx.float32).swapaxes(-1, -2)
    if mask is not None:
        scores = scores + mask.astype(mx.float32)
    s = mx.broadcast_to(
        sinks.astype(mx.float32).reshape(1, -1, 1, 1),
        (scores.shape[0], scores.shape[1], scores.shape[2], 1),
    )
    combined = mx.concatenate([scores, s], axis=-1)
    combined = combined - mx.max(combined, axis=-1, keepdims=True)
    probs = mx.softmax(combined, axis=-1, precise=True)
    probs = probs[..., :-1]  # drop le puits
    return probs.astype(v.dtype) @ v


class HYV4Attention(nn.Module):
    """MLA (deepseek) + DSA topk (indexer partagé) + gated MLA + learnable sink."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.scale = self.qk_head_dim ** -0.5
        self.index_topk = config.index_topk

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = HYV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=config.attention_bias
        )
        self.kv_a_layernorm = HYV4RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=config.attention_bias)
        self.linear_gate = nn.Linear(config.hidden_size, self.num_heads * self.v_head_dim, bias=False)
        self.learnable_sink_param = mx.zeros((self.num_heads,))

        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        self.indexer = None if self.skip_topk else HYV4Indexer(config)

    def __call__(self, x, cos, sin, mask=None, cache=None, prev_topk_indices=None):
        B, L, _ = x.shape

        # create_attention_mask renvoie un masque BOOLEEN (True=attend). On le
        # convertit en ADDITIF (0 / -1e9) : sans ça, `scores + bool` n'est pas
        # causal -> le modele voit le futur -> sortie scramblee. (L'indexer garde
        # le masque additif ; il ne recoit ce masque que pour le scoring topk.)
        if mask is not None and mask.dtype == mx.bool_:
            mask = mx.where(mask, mx.array(0.0, mx.float32), mx.array(-1e9, mx.float32))

        gate = self.linear_gate(x).reshape(B, L, self.num_heads, self.v_head_dim)

        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_resid).reshape(B, L, self.num_heads, self.qk_head_dim)
        q_nope, q_rot = q[..., : self.qk_nope_head_dim], q[..., self.qk_nope_head_dim :]

        ckv = self.kv_a_proj_with_mqa(x)
        kv_pass, k_rot = ckv[..., : self.kv_lora_rank], ckv[..., self.kv_lora_rank :]
        kv_latent = self.kv_a_layernorm(kv_pass)

        # expand_kv : latent -> par-tête k_nope + v
        kv = self.kv_b_proj(kv_latent).reshape(
            B, L, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, value = kv[..., : self.qk_nope_head_dim], kv[..., self.qk_nope_head_dim :]

        cse = cos[None, :, None, :]
        sne = sin[None, :, None, :]
        q_rot = _rope(q_rot, cse, sne)                                   # [B,L,H,rope]
        k_rot = _rope(k_rot.reshape(B, L, 1, self.qk_rope_head_dim), cse, sne)

        query = mx.concatenate([q_nope, q_rot], axis=-1).transpose(0, 2, 1, 3)  # [B,H,L,qk]
        k_rot = mx.broadcast_to(k_rot, (B, L, self.num_heads, self.qk_rope_head_dim))
        key = mx.concatenate([k_nope, k_rot], axis=-1).transpose(0, 2, 1, 3)    # [B,H,L,qk]
        value = value.transpose(0, 2, 1, 3)                                     # [B,H,L,v]

        if cache is not None:
            key, value = cache[0].update_and_fetch(key, value)

        # DSA : topk de cette couche (full) ou réutilisation (shared)
        if self.indexer is not None:
            idx_cache = cache[1] if cache is not None else None
            topk_indices = self.indexer(x, q_resid, cos, sin, mask, cache=idx_cache)
        else:
            topk_indices = prev_topk_indices

        attn_mask = mask
        if topk_indices is not None and key.shape[2] >= topk_indices.shape[-1]:
            T = key.shape[2]
            sparse = mx.zeros((B, L, T), dtype=mx.bool_)
            sparse = mx.put_along_axis(sparse, topk_indices, mx.array(True), axis=-1)
            add = mx.where(sparse, mx.array(0.0, mx.float32), mx.array(-1e9, mx.float32))
            attn_mask = add[:, None] if mask is None else (mask + add[:, None])

        out = _sdpa_with_sink(query, key, value, self.scale, attn_mask, self.learnable_sink_param)
        out = out.transpose(0, 2, 1, 3)  # [B,L,H,v]
        out = out * mx.sigmoid(gate)
        out = out.reshape(B, L, -1)
        return self.o_proj(out), topk_indices


class _UnweightedRMS(nn.Module):
    """Renvoie le facteur rsqrt(mean(x²)+eps) (norme comme résidu, pas x*...)."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        return mx.rsqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + self.eps)


class HYV4HyperConnection(nn.Module):
    """iHC indépendante (variante hy) : pre collapse les flux, post les ré-expand.
    Pas de sinkhorn/comb. fp32 forcé. Noms checkpoint : hc_fn/hc_base/hc_scale."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_magnitude = config.hc_magnitude
        mix = 2 * self.hc_mult
        self.hc_fn = mx.zeros((mix, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.hc_base = mx.zeros((mix,), dtype=mx.float32)
        self.hc_scale = mx.zeros((2,), dtype=mx.float32)
        self.input_norm = _UnweightedRMS(config.rms_norm_eps)

    def __call__(self, streams):
        # streams: [B,S,hc_mult,D]
        B, S, H, D = streams.shape
        flat = streams.reshape(B, S, H * D).astype(mx.float32)
        mixes = (flat @ self.hc_fn.T) * self.input_norm(flat)  # [B,S,mix]
        pre_logits = mixes[..., : self.hc_mult]
        post_logits = mixes[..., self.hc_mult :]
        pre_b = self.hc_base[: self.hc_mult]
        post_b = self.hc_base[self.hc_mult :]
        pre_scale, post_scale = self.hc_scale[0], self.hc_scale[1]
        pre = mx.sigmoid(pre_logits * pre_scale + pre_b) + self.hc_eps          # [B,S,hc]
        post = self.hc_magnitude * mx.sigmoid(post_logits * post_scale + post_b) + self.hc_eps
        out = mx.sum(pre[..., None] * streams.astype(mx.float32), axis=2)       # [B,S,D]
        return post, out.astype(streams.dtype)


class _HCWrap(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hc_pre = HYV4HyperConnection(config)


class HYV4HyperHead(nn.Module):
    """Collapse final des hc_mult flux -> 1. Noms : hc_head_fn/base/scale."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_head_fn = mx.zeros((self.hc_mult, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.hc_head_base = mx.zeros((self.hc_mult,), dtype=mx.float32)
        self.hc_head_scale = mx.zeros((1,), dtype=mx.float32)
        self.input_norm = _UnweightedRMS(config.rms_norm_eps)

    def __call__(self, streams):
        B, S, H, D = streams.shape
        flat = streams.reshape(B, S, H * D).astype(mx.float32)
        mixes = (flat @ self.hc_head_fn.T) * self.input_norm(flat)  # [B,S,hc]
        pre = mx.sigmoid(mixes * self.hc_head_scale[0] + self.hc_head_base) + self.hc_eps
        out = mx.sum(pre[..., None] * streams.astype(mx.float32), axis=2)
        return out.astype(streams.dtype)


class HYV4DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = HYV4Attention(config, layer_idx)
        self.input_layernorm = HYV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HYV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_attn_layer = _HCWrap(config)
        self.hc_mlp_layer = _HCWrap(config)
        is_sparse = config.mlp_layer_types[layer_idx] == "sparse"
        self.mlp = DeepseekV32MoE(config) if is_sparse else DeepseekV32MLP(config)
        if is_sparse and config.swiglu_limit and config.swiglu_limit > 0:
            # clamp swiglu sur les experts ROUTES uniquement (pas le shared)
            self.mlp.switch_mlp.activation = LimitedSwiGLU(config.swiglu_limit)

    def __call__(self, streams, cos, sin, mask=None, cache=None, prev_topk_indices=None):
        residual = streams
        post, h = self.hc_attn_layer.hc_pre(streams)
        h = self.input_layernorm(h)
        h, topk = self.self_attn(h, cos, sin, mask, cache, prev_topk_indices)
        streams = (
            post[..., None].astype(mx.float32) * h[:, :, None, :].astype(mx.float32)
            + residual.astype(mx.float32)
        ).astype(residual.dtype)

        residual = streams
        post, h = self.hc_mlp_layer.hc_pre(streams)
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        streams = (
            post[..., None].astype(mx.float32) * h[:, :, None, :].astype(mx.float32)
            + residual.astype(mx.float32)
        ).astype(residual.dtype)
        return streams, topk


class HYV4Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [HYV4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = HYV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = HYV4HyperHead(config)
        self.rotary_emb = HYV4RotaryEmbedding(config)

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        B, L, D = h.shape
        streams = mx.contiguous(
            mx.broadcast_to(h[:, :, None, :], (B, L, self.args.hc_mult, D))
        )

        if cache is None:
            cache = [None] * len(self.layers)
        mask_cache = cache[0][0] if isinstance(cache[0], CacheList) else cache[0]
        mask = create_attention_mask(h, mask_cache, return_array=True)

        offset = 0
        if mask_cache is not None:
            offset = mask_cache.offset
        positions = mx.arange(offset, offset + L)
        cos, sin = self.rotary_emb(positions)

        prev_topk = None
        for layer, c in zip(self.layers, cache):
            streams, prev_topk = layer(streams, cos, sin, mask, c, prev_topk)

        return self.norm(self.hc_head(streams))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = HYV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        return self.lm_head(self.model(inputs, cache))

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # full : KV attention + KV indexer (indispensable au decode : sans lui
        # l'indexer ne voit que le token courant -> topk=1 -> collapse) ;
        # shared : KV attention seul (pas d'indexer).
        caches = []
        for layer in self.layers:
            if getattr(layer.self_attn, "skip_topk", False):
                caches.append(CacheList(KVCache()))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches

    @property
    def cast_predicate(self):
        # garder en fp32 : biais routeur, params iHC, sink, k_norm.
        def predicate(k):
            return not any(
                s in k
                for s in (
                    "e_score_correction_bias",
                    "hc_pre.",
                    "hc_head.",
                    "learnable_sink_param",
                    "indexer.k_norm",
                )
            )

        return predicate

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        n_layers = self.args.num_hidden_layers
        out = {}
        for k, v in weights.items():
            # drop MTP / nextn / inv_freq
            if "mtp_layers" in k or "nextn" in k or "rotary_emb.inv_freq" in k:
                continue
            parts = k.split(".")
            if "layers" in parts:
                try:
                    if int(parts[parts.index("layers") + 1]) >= n_layers:
                        continue
                except (ValueError, IndexError):
                    pass

            # split experts.gate_up_proj -> switch_mlp.{gate_proj,up_proj}
            if k.endswith("mlp.experts.gate_up_proj"):
                prefix = k[: -len("experts.gate_up_proj")]
                half = v.shape[1] // 2
                out[prefix + "switch_mlp.gate_proj.weight"] = v[:, :half, :]
                out[prefix + "switch_mlp.up_proj.weight"] = v[:, half:, :]
                continue
            if k.endswith("mlp.experts.down_proj"):
                prefix = k[: -len("experts.down_proj")]
                out[prefix + "switch_mlp.down_proj.weight"] = v
                continue

            out[k] = v
        return out
