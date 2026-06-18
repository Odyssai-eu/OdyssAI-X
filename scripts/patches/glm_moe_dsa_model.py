"""Patch glm_moe_dsa (GLM-5.2) — partage d'indices DSA full/shared (IndexCache).

BUG : mlx-lm `deepseek_v32` instancie un `Indexer` par couche et IGNORE le champ
`indexer_types`. Les couches `shared` (qui n'ont PAS de poids indexer dans le
checkpoint GLM-5.2) tournent donc un indexer aléatoire/non-initialisé. En dessous
de `index_topk` (2048) l'indexer renvoie None (chemin dense) → OK. Au-dessus, ces
faux indexers sélectionnent des top-k garbage → corruption du prefill dès le 1er
token décodé. (Court / greedy OK car l'indexer ne tourne jamais.)

FIX : miroir du modeling officiel transformers `GlmMoeDsaAttention`
(`skip_topk` + `prev_topk_indices`). Chaque couche reçoit/renvoie ses
`topk_indices` ; une couche `full` les CALCULE et les passe ; une couche `shared`
RÉUTILISE ceux de la couche précédente (chaînés depuis la `full` la plus proche en
amont). On propage l'OUTPUT de l'indexer — qui peut être `None` (= dense, ≤ seuil)
→ une `shared` hérite alors du None.

Référence : `transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py`
(GlmMoeDsaAttention l.372-446 : `skip_topk = indexer_types[layer_idx]=="shared"`,
`topk_indices = prev_topk_indices` si skip).

PIPELINE (follow-up, cf PLAN.md) : le threading est intra-rank — valide single-node
(gate 1). Si une coupe pipeline sépare une `full` de ses `shared` aval, il faut
communiquer les indices entre rangs (non couvert ici).
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
        # glm_moe_dsa.__post_init__ mappe la rope (rope_scaling/theta) — on le garde.
        super().__post_init__()
        # Synthèse par défaut (officiel) : couche 0 full, puis une tous les
        # index_topk_freq, le reste shared.
        if self.indexer_types is None:
            freq = max(1, self.index_topk_freq)
            self.indexer_types = [
                "full" if (max(i - 1, 0) % freq) == 0 else "shared"
                for i in range(self.num_hidden_layers)
            ]


class Attention(_dsv32.DeepseekV32Attention):
    """Comme DeepseekV32Attention mais : connaît son layer_idx + skip_topk, et
    accepte/renvoie prev_topk pour le partage full/shared. Corps recopié verbatim
    du port mlx-lm (deepseek_v32) avec les seules lignes du topk modifiées."""

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
        prev_topk: Optional[mx.array] = None,
    ):
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

        # === SEUL changement vs deepseek_v32 : full calcule, shared réutilise ===
        if self.skip_topk and prev_topk is not None:
            topk_indices = prev_topk            # réutilise la couche full amont
        else:
            topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        topk_used = topk_indices                # propagé à la couche suivante
        # =======================================================================
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
        # Garder l'indexer-cache évalué — UNIQUEMENT pour les couches full (les
        # shared n'ont pas appelé l'indexer, cache[1] n'est pas mis à jour).
        if (not self.skip_topk) and cache is not None and cache[0] is not None:
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
        return self.o_proj(output), topk_used


class DecoderLayer(_dsv32.DeepseekV32DecoderLayer):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config, layer_idx)
        # remplace l'attention par la nôtre (avec layer_idx + skip_topk)
        self.self_attn = Attention(config, layer_idx)

    def __call__(self, x, mask=None, cache=None, prev_topk=None):
        r, topk_used = self.self_attn(self.input_layernorm(x), mask, cache, prev_topk)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, topk_used


class _GlmModel(_dsv32.DeepseekV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        # reconstruit les couches avec notre DecoderLayer
        self.layers = [
            DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx

    def __call__(self, x, cache=None):
        h = self.embed_tokens(x)
        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        if cache is None:
            cache = [None] * self.num_layers
        mask = _dsv32.create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))
        # threading des topk_indices full→shared (réinitialisé à chaque forward)
        prev_topk = None
        for i in range(self.num_layers):
            h, prev_topk = self.layers[self.start_idx + i](
                h, mask, cache[i], prev_topk
            )
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                h = mx.distributed.recv_like(h, (pipeline_rank - 1) % pipeline_size)
        return self.norm(h) if pipeline_rank == 0 else h


class Model(_glm.Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        # remplace le modèle interne par celui qui thread les topk
        self.model = _GlmModel(config)


def apply_glm_dsa() -> None:
    """Enregistre ce module patché comme `mlx_lm.models.glm_moe_dsa` AVANT que le
    loader (`_get_classes`) ne l'importe. Même mécanique que apply_bailing_hybrid /
    apply_minimax_m3 : on remplace l'entrée sys.modules."""
    sys.modules["mlx_lm.models.glm_moe_dsa"] = sys.modules[__name__]
