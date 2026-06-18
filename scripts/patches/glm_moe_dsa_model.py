"""Patch glm_moe_dsa (GLM-5.2) — fix DSA full/shared, OPTION A (dense fallback).

BUG : mlx-lm `deepseek_v32` instancie un `Indexer` par couche et IGNORE `indexer_types`.
Les couches `shared` (sans poids indexer dans le checkpoint) tournent donc un indexer
aléatoire/non-initialisé. ≤ `index_topk` (2048) l'indexer renvoie None (dense) → OK.
Au-dessus, ces faux indexers sélectionnent des top-k garbage → corruption d'attention qui
compounde → collapse (`0.0.0`). Confirmé EN+FR, Q4+Q8.

FIX (Option A — minimal, cf PLAN.md) : on RUN l'indexer sur TOUTES les couches (donc
`cache[1]`, le KVCache de l'indexer, reste consistant pour BatchGenerator — comme le
non-patché), mais pour une couche `shared` on **IGNORE son top-k** (poids aléatoires) →
`topk_indices = None` → attention DENSE (superset de la sélection sparse → correct ; perd
juste l'optim sparse sur ces couches). On ne touche QUE `Attention.__call__` : UNE seule
ligne ajoutée vs le non-patché (`if skip_topk: topk_indices = None`). Retour `o_proj(output)`
(array simple). `DecoderLayer.__call__`, `DeepseekV32Model.__call__` et le bloc pipeline
(recv/send/all_gather) restent INCHANGÉS (hérités) — c'est ce qui a tué la v1 quand on les a
réécrits. (Sauter l'indexer pour les shared, 1re tentative Option A, laissait `cache[1]` vide
→ offsets incohérents → 0-token via BatchGenerator.)

Option B (reuse exact du top-k de la `full` amont = schéma officiel IndexCache) = optim perf,
plus tard, une fois A validé 5000 mots.

Référence (vérité) : `transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py`.
Corps de `__call__` recopié verbatim du port mlx-lm `deepseek_v32` (validé line-accurate).
"""

import sys
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx

from mlx_lm.models import deepseek_v32 as _dsv32
from mlx_lm.models import glm_moe_dsa as _glm


@dataclass
class ModelArgs(_glm.ModelArgs):
    # deepseek_v32/glm_moe_dsa ModelArgs ne déclarent PAS ces champs → BaseModelArgs.
    # from_dict les DROP. Sans ça le patch lirait indexer_types=None partout.
    indexer_types: Optional[list] = None
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 0

    def __post_init__(self):
        super().__post_init__()
        # Synthèse par défaut (officiel) si absent : couche 0 full, puis une tous les
        # index_topk_freq, le reste shared.
        if self.indexer_types is None:
            freq = max(1, self.index_topk_freq)
            self.indexer_types = [
                "full" if (max(i - 1, 0) % freq) == 0 else "shared"
                for i in range(self.num_hidden_layers)
            ]


class Attention(_dsv32.DeepseekV32Attention):
    """DeepseekV32Attention + connaissance de son `layer_idx` / `skip_topk`. Corps de
    `__call__` recopié verbatim du port mlx-lm, avec UNIQUEMENT la branche indexer
    (shared → dense) et le guard du `mx.depends` modifiés."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config)
        self.layer_idx = layer_idx
        types = getattr(config, "indexer_types", None)
        self.skip_topk = bool(types) and types[layer_idx] == "shared"

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2

        # === SEUL changement vs deepseek_v32 (Option A) ===
        # On RUN l'indexer sur TOUTES les couches → `cache[1]` (KVCache de l'indexer) reste
        # consistant pour BatchGenerator (qui lit les offsets de cache) — c'est EXACTEMENT ce
        # que fait le modèle non-patché, qui sert correctement. Mais pour une couche `shared`
        # on IGNORE le top-k retourné (poids indexer aléatoires) → topk=None → attention DENSE
        # (superset correct). Sauter l'indexer cassait cache[1] → 0-token (le bug v1).
        topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        if self.skip_topk:
            topk_indices = None
        # ==================================================
        if topk_indices is not None:
            if L == 1:
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)
            else:
                shape = list(topk_indices.shape)
                shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, topk_indices, mx.array(True), axis=-1
                )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask
        # Garder l'indexer-cache évalué (toutes les couches appellent l'indexer maintenant
        # → cache[1] peuplé partout, comme le non-patché).
        if cache is not None and cache[0] is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = _dsv32.scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DecoderLayer(_dsv32.DeepseekV32DecoderLayer):
    """Identique à DeepseekV32DecoderLayer mais avec l'Attention patchée (qui connaît son
    layer_idx). `__call__` NON surchargé (hérité = original)."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Attention(config, layer_idx)


class _GlmModel(_dsv32.DeepseekV32Model):
    """Reconstruit les couches avec notre DecoderLayer. `__call__` NON surchargé →
    hérite le forward d'origine (mask + boucle + pipeline recv/send/all_gather intacts)."""

    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.layers = [
            DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]


class Model(_glm.Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.model = _GlmModel(config)


def apply_glm_dsa() -> None:
    """Enregistre ce module patché comme `mlx_lm.models.glm_moe_dsa` AVANT que le loader
    (`_get_classes`) ne l'importe. Même mécanique que apply_bailing_hybrid /
    apply_minimax_m3 : remplace l'entrée sys.modules."""
    sys.modules["mlx_lm.models.glm_moe_dsa"] = sys.modules[__name__]
