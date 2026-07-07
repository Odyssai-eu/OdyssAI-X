# Copyright © 2026 OdyssAI
#
# HunYuan-3 (Tencent) — `model_type: hy_v3`, arch `HYV3ForCausalLM`.
#
# Port mlx-lm pour OdyssAI-X. HYV3 = MoE DeepSeek-V3 (router sigmoid + biais
# d'experts + expert partagé + first_k_dense_replace) GREFFÉ sur une attention
# GQA standard avec qk_norm (style Qwen3) — PAS de MLA. Les poids du quant
# InferencerLabs sont déjà au layout mlx-lm (switch_mlp fused, router.gate,
# router.expert_bias, shared_mlp), donc aucun sanitize/stacking n'est requis.
#
# Déploiement : ce fichier va dans `mlx_lm/models/hy_v3.py` de chaque venv
# mlx-cluster (synchro identique entre nodes). Le sharding distribué passe par
# le `pipeline_auto_parallel` générique d'OdyssAI-X (slice `.layers`) — d'où une
# boucle forward simple ici plutôt que le PipelineMixin de mlx-lm.

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "hy_v3"
    hidden_size: int = 4096
    intermediate_size: int = 13312          # dense MLP (layer 0)
    moe_intermediate_size: int = 1536       # par-expert (routed + shared)
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 120832
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 262144
    rope_theta: float = 11158840.0
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    qk_norm: bool = True
    # MoE
    num_experts: int = 192
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    router_scaling_factor: float = 2.826
    route_norm: bool = True                 # normalise les poids top-k

    def __post_init__(self):
        # rope_theta peut être imbriqué dans rope_parameters (transformers 5.6).
        if self.rope_parameters:
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)


@mx.compile
def expert_select(gates, expert_bias, top_k, routed_scaling_factor, norm_topk_prob):
    """Routing DeepSeek-V3 sans groupes : sigmoid → +biais → top-k → poids
    originaux → (norm) → ×scaling. La sélection utilise les scores biaisés ;
    la pondération utilise les scores sigmoïde non biaisés."""
    scores = mx.sigmoid(gates.astype(mx.float32))
    orig = scores
    scores = scores + expert_bias
    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(orig, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        weights = weights / weights.sum(axis=-1, keepdims=True)
    weights = weights * routed_scaling_factor
    return inds, weights


class Attention(nn.Module):
    """GQA + qk_norm (RMSNorm par head sur q et k), style Qwen3. Pas de MLA."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = self.q_norm(q.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU dense — layer-0 et expert partagé."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Router(nn.Module):
    """gate quantisé (nn.Linear) + biais de correction d'experts (f32)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.route_norm
        self.routed_scaling_factor = args.router_scaling_factor
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros((args.num_experts,), dtype=mx.float32)

    def __call__(self, x):
        return expert_select(
            self.gate(x),
            self.expert_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.router = Router(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        shared_hidden = args.moe_intermediate_size * args.num_shared_experts
        self.shared_mlp = MLP(args.hidden_size, shared_hidden)

    def __call__(self, x):
        inds, weights = self.router(x)
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        return y + self.shared_mlp(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)
        if layer_idx >= args.first_k_dense_replace:
            self.mlp = MoE(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class HYV3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = HYV3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        # Deux layouts possibles en entrée :
        #  - quant InferencerLabs : déjà au layout final (switch_mlp fused,
        #    router.gate, router.expert_bias, shared_mlp) → rien à mapper ;
        #  - release HF brut tencent/Hy3 (2026-07) : experts PAR INDEX
        #    (mlp.experts.E.{gate,up,down}_proj) + expert_bias directement sous
        #    mlp → il faut stacker les 192 experts en switch_mlp (pattern
        #    longcat2) et déplacer expert_bias sous router. Le reste
        #    (router.gate, shared_mlp, attention GQA) matche déjà.
        def _keep(k):
            if "rotary_emb.inv_freq" in k or ".mtp" in k or "nextn" in k:
                return False
            if k.startswith("model.mtp"):
                return False
            # Release HF 2026-07 : la tête MTP est nommée model.layers.80.*
            # (eh_proj/enorm/hnorm…) au-delà du trunk 0..79 — on la droppe
            # comme longcat2 droppe ses layers hors num_layers.
            parts = k.split(".")
            if (len(parts) >= 3 and parts[1] == "layers" and parts[2].isdigit()
                    and int(parts[2]) >= self.args.num_hidden_layers):
                return False
            return True

        weights = {k: v for k, v in weights.items() if _keep(k)}

        if not any(".mlp.experts." in k for k in weights):
            return weights  # layout final (InferencerLabs) — inchangé

        n_experts = self.args.num_experts
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.mlp"
            if f"{prefix}.experts.0.gate_proj.weight" not in weights:
                continue  # layer dense (layer 0)
            for m in ("gate_proj", "up_proj", "down_proj"):
                to_join = [
                    weights.pop(f"{prefix}.experts.{e}.{m}.weight")
                    for e in range(n_experts)
                ]
                weights[f"{prefix}.switch_mlp.{m}.weight"] = mx.stack(to_join)
            bias_key = f"{prefix}.expert_bias"
            if bias_key in weights:
                weights[f"{prefix}.router.expert_bias"] = weights.pop(bias_key)
        return weights

    @property
    def cast_predicate(self):
        # Garder le biais d'experts en float32 (pas de cast vers le dtype modèle).
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def layers(self):
        return self.model.layers

    def shard(self, group: Optional[mx.distributed.Group] = None):
        """Tensor-parallel optionnel (non requis pour le pipeline OdyssAI-X).
        Shard les têtes d'attention + les projections MLP/MoE par node."""
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.model.layers:
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, "all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, "all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.n_heads //= N
            layer.self_attn.n_kv_heads //= N
