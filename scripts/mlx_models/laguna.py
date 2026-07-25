# Copyright © 2026 PipeNetwork
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# VENDORISÉ — ce fichier n'est PAS de nous. Copie verbatim de
#   https://huggingface.co/pipenetwork/Laguna-S-2.1-MLX-8bit/blob/main/laguna.py
#   (source : https://github.com/PipeNetwork/laguna-mlx, récupéré 2026-07-25)
# Garder byte-identique à l'upstream sauf ce bloc, pour que le diff d'une
# future mise à jour reste lisible.
#
# Pourquoi vendoriser : mlx-lm n'a AUCUN support de `laguna` — issue
# ml-explore/mlx-lm #1378, ouverte depuis 2026-06-08, et 0.31.3 (la version
# épinglée dans requirements-node.txt) est déjà la dernière publiée. Sans ce
# module, tout chargement de Laguna meurt sur "Model type laguna not supported".
#
# Attention au layout des poids : ce module attend les quants pipenetwork.
# Le quant Q9 d'inferencerlabs est INCOMPATIBLE — préfixe `language_model.`,
# routeur en `mlp.gate.proj.*`, et `switch_mlp.gate_up_proj` fusionné là où
# SwitchGLU veut `gate_proj`/`up_proj` séparés. Ne pas perdre de temps dessus.
#
# MLX port of poolside/Laguna-S-2.1 (118B-A8B sparse MoE).
#
# Laguna is architecturally close to Qwen3-MoE with these additions, all
# handled here:
#   * softplus attention output gating (per-head ``g_proj``)
#   * per-head Q/K RMSNorm (before RoPE)
#   * interleaved full / sliding-window (512) attention, one mask per type
#   * two RoPEs: full-attention layers use partial-rotary (0.5) YaRN
#     (theta 5e5, factor 128); sliding layers use plain RoPE (theta 1e4, full)
#   * sigmoid router with an aux-loss-free correction bias (DeepSeek-V3 style)
#   * an always-on shared expert added to the routed output
#   * layer 0 is a dense MLP (``mlp_only_layers=[0]``)
#
# The file plugs into stock ``mlx-lm``: register it as ``mlx_lm/models/laguna.py``
# (see ``scripts/install_model.sh``) and use ``mlx_lm convert`` / ``mlx_lm generate``.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    # MoE
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    decoder_sparse_step: int
    norm_topk_prob: bool
    moe_routed_scaling_factor: float = 1.0
    moe_router_logit_softcapping: float = 0.0
    mlp_only_layers: List[int] = field(default_factory=lambda: [0])
    # attention
    num_attention_heads_per_layer: Optional[List[int]] = None  # e.g. 48 (full) / 72 (sliding)
    gating: Any = "per-head"  # True / "per-element" / "per-head" / False
    sliding_window: Optional[int] = None
    layer_types: Optional[List[str]] = None
    # rope: config.json nests rope by attention type
    rope_parameters: Optional[Dict[str, Any]] = None
    swa_rope_parameters: Optional[Dict[str, Any]] = None
    partial_rotary_factor: Optional[float] = None
    max_position_embeddings: int = 1048576
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        # Derive the SWA rope from the nested rope_parameters when present.
        rp = self.rope_parameters
        if self.swa_rope_parameters is None and isinstance(rp, dict):
            self.swa_rope_parameters = rp.get("sliding_attention")


def _rope_for(cfg: dict, head_dim: int, max_pos: int) -> nn.Module:
    """Build the RoPE for one attention type from its rope sub-config.

    Handles partial rotary (only ``head_dim * partial_rotary_factor`` dims are
    rotated, the rest pass through — matching HF ``apply_rotary_pos_emb``) and
    YaRN. For YaRN, mlx-lm's ``YarnRoPE`` reproduces HF's mscale from ``factor``
    alone (``0.1·ln(factor)+1``), which equals Laguna's stored ``attention_factor``.
    """
    cfg = dict(cfg or {})
    theta = float(cfg.get("rope_theta", 10000.0))
    partial = float(cfg.get("partial_rotary_factor", 1.0))
    dims = int(head_dim * partial)
    rope_type = cfg.get("rope_type", "default")
    if rope_type in ("default", "linear"):
        return initialize_rope(dims, base=theta, traditional=False)
    scaling_config = {
        "rope_type": rope_type,
        "factor": float(cfg.get("factor", 1.0)),
    }
    for k in ("original_max_position_embeddings", "beta_fast", "beta_slow",
              "mscale", "mscale_all_dim"):
        if cfg.get(k) is not None:
            scaling_config[k] = cfg[k]
    return initialize_rope(
        dims, base=theta, traditional=False,
        scaling_config=scaling_config, max_position_embeddings=max_pos,
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        # Laguna varies the query-head count by layer type (e.g. 48 full / 72
        # sliding); KV heads stay constant. q/o/g projections size off this.
        per_layer = args.num_attention_heads_per_layer
        self.n_heads = per_layer[layer_idx] if per_layer is not None else args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        # Softplus output gating (Laguna-specific). "per-head" -> one gate per
        # head broadcast over head_dim; "per-element" -> one per (head, dim).
        self.gating = bool(args.gating)
        self.gate_per_head = args.gating == "per-head"
        if self.gating:
            g_out = self.n_heads if self.gate_per_head else self.n_heads * head_dim
            self.g_proj = nn.Linear(dim, g_out, bias=False)

        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = args.sliding_window if self.is_sliding else None
        rope_cfg = args.swa_rope_parameters if self.is_sliding else (
            (args.rope_parameters or {}).get("full_attention", args.rope_parameters)
        )
        self.rope = _rope_for(rope_cfg, head_dim, args.max_position_embeddings)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.head_dim)

        # Softplus gate, applied before o_proj (computed in fp32 like the reference).
        if self.gating:
            g = mx.logaddexp(self.g_proj(x).astype(mx.float32), mx.array(0.0))
            g = g.astype(out.dtype)
            if self.gate_per_head:
                out = (out.reshape(B, L, self.n_heads, self.head_dim) * g[..., None])
                out = out.reshape(B, L, -1)
            else:
                out = out * g

        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LagunaSparseMoeBlock(nn.Module):
    """Sigmoid router + aux-loss-free correction bias + shared expert.

    Selection uses ``sigmoid(logits) + e_score_correction_bias``; the combining
    weights are the *unbiased* sigmoid scores (optionally sum-normalised), scaled
    by ``moe_routed_scaling_factor``. A dense shared expert runs on every token
    and is added to the routed output.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.moe_routed_scaling_factor
        self.softcap = args.moe_router_logit_softcapping

        self.gate = nn.Linear(args.hidden_size, self.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((self.num_experts,))
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, self.num_experts
        )
        self.shared_expert = MLP(
            args.hidden_size, args.shared_expert_intermediate_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, L, H = x.shape
        xf = x.reshape(-1, H)

        # Router is precision-sensitive (near-tied scores flip expert choice and
        # compound over 47 MoE layers), so do sigmoid/top-k in fp32.
        logits = self.gate(xf).astype(mx.float32)
        if self.softcap and self.softcap > 0.0:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        scores = mx.sigmoid(logits)
        scores_for_choice = scores + self.e_score_correction_bias.astype(mx.float32)

        k = self.top_k
        inds = mx.argpartition(-scores_for_choice, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        weights = (weights * self.routed_scaling_factor).astype(x.dtype)

        y = self.switch_mlp(xf, inds)                 # [T, k, H]
        y = (y * weights[..., None]).sum(axis=-2)
        y = y + self.shared_expert(xf)                # always-on shared expert
        return y.reshape(B, L, H)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        is_moe = (layer_idx not in args.mlp_only_layers) and (
            args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        )
        if is_moe:
            self.mlp = LagunaSparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class LagunaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        types = args.layer_types
        self._first_full = next((i for i, t in enumerate(types) if t == "full_attention"), 0)
        self._first_swa = next((i for i, t in enumerate(types) if t == "sliding_attention"), 0)
        self._has_swa = "sliding_attention" in types

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(h, cache[self._first_full])
        if self._has_swa:
            swa_mask = create_attention_mask(
                h, cache[self._first_swa], window_size=self.args.sliding_window
            )
        else:
            swa_mask = full_mask

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.self_attn.is_sliding else full_mask
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LagunaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.mlp"
            # aux-loss-free bias ships under experts.* in the checkpoint
            bias = weights.pop(f"{prefix}.experts.e_score_correction_bias", None)
            if bias is not None:
                weights[f"{prefix}.e_score_correction_bias"] = bias
            # stack per-expert projections into SwitchGLU's 3D tensors
            if f"{prefix}.experts.0.gate_proj.weight" in weights:
                for n in ("gate_proj", "up_proj", "down_proj"):
                    weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack([
                        weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ])
        return weights

    def make_cache(self):
        caches = []
        for t in self.args.layer_types:
            if t == "sliding_attention":
                caches.append(RotatingKVCache(max_size=self.args.sliding_window, keep=0))
            else:
                caches.append(KVCache())
        return caches

    @property
    def quant_predicate(self):
        # Keep the tiny, precision-sensitive router at 8-bit.
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True
        return predicate

    @property
    def layers(self):
        return self.model.layers
