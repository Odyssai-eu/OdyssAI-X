# Kimi K3 (moonshotai/Kimi-K3) — text tower, MLX.
#
# Vendored architecture module: copy into mlx_lm/models/ on every node via
# scripts/install-model-modules.sh. Written against the CONVERTED checkpoint
# produced by scripts/convert_k3_mxfp4.py (mlx naming, routed experts kept in
# their native MXFP4, everything else affine-quantized).
#
# Built on mlx_lm.models.kimi_linear (Kimi-Linear 48B): KDA + gated MLA + the
# grouped-topk router are imported from it rather than duplicated. What K3 adds
# on top, and which this module implements:
#
#   * SiTU-GLU activation          hidden_act "situ", both halves transformed
#   * Attention residuals          softmax attention over the DEPTH axis
#   * Latent MoE                   routed experts live in a reduced space
#   * Full-rank KDA output gate    g_proj instead of the 48B's g_a/g_b pair
#   * Bounded KDA decay gate       gate_lower_bound changes the gate formula
#   * q-LoRA + gated MLA output    q_a/q_b projections, sigmoid output gate
#
# Vision (MoonViT-V2) is deliberately out of scope: the converter drops those
# weights, so nothing here references them.

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_kernel, gated_delta_ops
from .kimi_linear import ShortConv1d, _group_expert_select
from .mla import MultiLinear
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rms_norm_eps: float
    linear_attn_config: Dict[str, Any]
    num_experts: int
    moe_intermediate_size: int
    kv_lora_rank: int
    head_dim: Optional[int] = None
    q_lora_rank: Optional[int] = None
    max_position_embeddings: int = 1048576
    tie_word_embeddings: bool = False
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mla_use_nope: bool = True
    mla_use_output_gate: bool = False
    num_experts_per_token: int = 1
    num_shared_experts: int = 0
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    num_expert_group: int = 1
    topk_group: int = 1
    # K3 additions
    hidden_act: str = "situ"
    activation_situ_beta: float = 1.0
    activation_situ_linear_beta: Optional[float] = None
    attn_res_block_size: Optional[int] = None
    routed_expert_hidden_size: Optional[int] = None
    latent_moe_use_norm: bool = False

    @classmethod
    def from_dict(cls, params):
        # Multimodal-wrapper configs (KimiK3ForConditionalGeneration — the REAP
        # variants ship one) nest the text params under `text_config`; the
        # checkpoint itself is text-only (REAP prunes the vision weights, only a
        # vestigial vision_config remains). Flatten so the text ModelArgs gets its
        # fields. Original single-config K3 has no text_config → falls through.
        # `quantization` stays top-level (mlx_lm reads it there for nn.quantize).
        src = params.get("text_config") or params
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in src.items() if k in names}
        kw.setdefault("model_type", params.get("model_type", "kimi_k3"))
        return cls(**kw)


# ── SiTU-GLU ─────────────────────────────────────────────────────────────────
# beta * tanh(gate/beta) * sigmoid(gate), times linear_beta * tanh(up/linear_beta)
# when linear_beta is set. The reference computes this in fp32 regardless of the
# activation dtype (modeling_kimi_linear.py::SituAndMul) — keep that, the tanh
# saturates differently in bf16.


@mx.compile
def _situ(gate: mx.array, up: mx.array, beta: float, linear_beta: float) -> mx.array:
    g = gate.astype(mx.float32)
    u = up.astype(mx.float32)
    a = beta * mx.tanh(g / beta) * mx.sigmoid(g)
    if linear_beta > 0.0:
        u = linear_beta * mx.tanh(u / linear_beta)
    return a * u


class SituGLU(nn.Module):
    """Activation object for SwitchGLU / dense MLPs.

    NOTE the argument order: SwitchGLU calls `activation(x_up, x_gate)`, so the
    UP tensor arrives first. Getting this backwards is silent — both halves have
    the same shape — so the order is asserted by the micro-test.
    """

    def __init__(self, beta: float, linear_beta: Optional[float]):
        super().__init__()
        self.beta = float(beta)
        self.linear_beta = float(linear_beta) if linear_beta else 0.0

    def __call__(self, up: mx.array, gate: mx.array) -> mx.array:
        return _situ(gate, up, self.beta, self.linear_beta)


class KimiK3MLP(nn.Module):
    """Dense MLP — layer 0, and the shared experts of every MoE layer."""

    def __init__(
        self,
        args: ModelArgs,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        dim = hidden_size or args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.act = SituGLU(args.activation_situ_beta, args.activation_situ_linear_beta)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.act(self.up_proj(x), self.gate_proj(x))
        return self.down_proj(out.astype(x.dtype))


class KimiK3SparseMoE(nn.Module):
    """Latent MoE: the routed experts run in `routed_expert_hidden_size` space.

    Flow (modeling_kimi_linear.py::KimiSparseMoeBlock.forward): route on the
    ORIGINAL hidden, project down to the latent, run the top-k experts there,
    normalise, project back up, then add the shared experts computed on the
    ORIGINAL hidden — not on the latent.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        hidden = args.hidden_size
        experts = args.num_experts

        self.use_latent = args.routed_expert_hidden_size is not None
        expert_dim = args.routed_expert_hidden_size if self.use_latent else hidden

        self.gate = nn.Linear(hidden, experts, bias=False)
        self.e_score_correction_bias = mx.zeros((experts,), dtype=mx.float32)
        self.switch_mlp = SwitchGLU(
            expert_dim,
            args.moe_intermediate_size,
            experts,
            activation=SituGLU(
                args.activation_situ_beta, args.activation_situ_linear_beta
            ),
        )

        if self.use_latent:
            self.routed_expert_down_proj = nn.Linear(hidden, expert_dim, bias=False)
            self.routed_expert_up_proj = nn.Linear(expert_dim, hidden, bias=False)
            self.routed_expert_norm = (
                nn.RMSNorm(expert_dim, eps=args.rms_norm_eps)
                if args.latent_moe_use_norm
                else None
            )

        if args.num_shared_experts:
            self.shared_experts = KimiK3MLP(
                args,
                intermediate_size=args.moe_intermediate_size * args.num_shared_experts,
            )
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        inds, weights = _group_expert_select(
            self.gate(x),
            self.e_score_correction_bias,
            self.args.num_experts_per_token,
            self.args.num_expert_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.moe_renormalize,
            self.args.moe_router_activation_func,
        )

        h = self.routed_expert_down_proj(x) if self.use_latent else x
        out = self.switch_mlp(h, inds)
        out = (out * weights[..., None]).sum(axis=-2).astype(x.dtype)

        if self.use_latent:
            if self.routed_expert_norm is not None:
                out = self.routed_expert_norm(out)
            out = self.routed_expert_up_proj(out)

        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class KimiK3MLAAttention(nn.Module):
    """MLA with q-LoRA and a sigmoid output gate, NoPE.

    Mirrors kimi_linear.KimiMLAAttention (absorbed embed_q/unembed_out form)
    with two K3 changes: the query goes through a LoRA pair, and the attention
    output is gated before o_proj.

    NoPE: `mla_use_nope` is true and the reference NEVER applies a rotary — the
    q_pe/k_pe split is kept but carries no rotation. All positional information
    comes from the KDA layers. Do not "fix" this by adding RoPE.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        if not args.mla_use_nope:
            raise ValueError("kimi_k3 expects mla_use_nope=true (no rotary in MLA)")

        self.args = args
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim or args.head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim or 0
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim or args.head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.q_lora_rank = args.q_lora_rank
        self.scale = self.q_head_dim**-0.5

        hidden = args.hidden_size
        if self.q_lora_rank:
            self.q_a_proj = nn.Linear(hidden, self.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(
                hidden, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, args.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, args.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)

        self.use_output_gate = args.mla_use_output_gate
        if self.use_output_gate:
            self.g_proj = nn.Linear(
                hidden, self.num_heads * self.v_head_dim, bias=False
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.q_lora_rank:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = mx.expand_dims(self.kv_a_layernorm(compressed_kv), axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        # The un-rotated "pe" halves contribute an additive score term, which is
        # exactly q·k over the concatenated head_dim in the reference.
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

        out = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            out = self.unembed_out(out)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.use_output_gate:
            out = out * mx.sigmoid(self.g_proj(x))
        return self.o_proj(out)


@mx.compile
def _kda_decay_bounded(
    A_log: mx.array, a: mx.array, dt_bias: mx.array, lower_bound: float
) -> mx.array:
    """Decay factor for the bounded gate (fla `safe_gate=True`).

    fla/ops/kda: with a lower bound the gate activation switches from
    `-exp(A_log) * softplus(a + dt_bias)` to
    `lower_bound * sigmoid(exp(A_log) * (a + dt_bias))`, which clamps the
    log-decay into [lower_bound, 0). This returns the decay itself (exp of
    that), matching the contract of gated_delta.compute_g.
    """
    g_log = lower_bound * mx.sigmoid(
        mx.exp(A_log.astype(mx.float32)) * (a.astype(mx.float32) + dt_bias)
    )
    return mx.exp(g_log)


@mx.compile
def _kda_decay_softplus(
    A_log: mx.array, a: mx.array, dt_bias: mx.array
) -> mx.array:
    return mx.exp(
        -mx.exp(A_log.astype(mx.float32))
        * nn.softplus(a.astype(mx.float32) + dt_bias)
    )


class KimiK3DeltaAttention(nn.Module):
    """Kimi Delta Attention, K3 flavour.

    Differences from the 48B (kimi_linear.KimiDeltaAttention):
      * output gate is FULL RANK (`g_proj`) instead of the g_a/g_b pair. Note
        f_a_proj/f_b_proj still exist and are a different gate — they feed the
        DECAY, not the output. Do not merge the two.
      * `A_log` is per head_dim, not per head. The published reference declares
        it `[num_heads]` but every checkpoint tensor is `[head_dim]` (verified
        on layers 1-2), and dt_bias `[num_heads*head_dim]` fixes the rest of the
        broadcast. It therefore broadcasts along the LAST axis, shared across
        heads.
      * `gate_lower_bound` selects the bounded gate formula above.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        cfg = args.linear_attn_config

        self.layer_idx = layer_idx
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.conv_kernel = cfg.get("short_conv_kernel_size", 4)
        self.gate_lower_bound = cfg.get("gate_lower_bound", None)
        self.use_full_rank_gate = cfg.get("use_full_rank_gate", False)

        self.projection_dim = self.num_heads * self.head_dim
        hidden = args.hidden_size
        self.scale = float(self.head_dim) ** -0.5

        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)

        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.projection_dim, self.conv_kernel)

        # Decay gate (low rank) and per-head beta.
        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)

        # Output gate.
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        else:
            self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)

        self.A_log = mx.zeros((self.head_dim,), dtype=mx.float32)
        self.dt_bias = mx.zeros((self.projection_dim,), dtype=mx.float32)

        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state, k_state, v_state, ssm_state = cache
            lengths = cache.lengths
        else:
            q_state = k_state = v_state = ssm_state = None
            lengths = None

        if q_state is None:
            s = mx.zeros((B, self.conv_kernel - 1, self.projection_dim), dtype=dtype)
            q_state = k_state = v_state = s

        q_conv, q_state = self.q_conv(self.q_proj(x), q_state, mask, lengths)
        k_conv, k_state = self.k_conv(self.k_proj(x), k_state, mask, lengths)
        v_conv, v_state = self.v_conv(self.v_proj(x), v_state, mask, lengths)

        if cache is not None:
            cache[0], cache[1], cache[2] = q_state, k_state, v_state

        shape = (B, T, self.num_heads, self.head_dim)
        q = q_conv.reshape(shape)
        k = k_conv.reshape(shape)
        v = v_conv.reshape(shape)

        q = (self.scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = self.scale * mx.fast.rms_norm(k, None, 1e-6)

        a_logits = self.f_b_proj(self.f_a_proj(x)).reshape(shape)
        beta = mx.sigmoid(self.b_proj(x).reshape(B, T, self.num_heads))

        A_log = self.A_log.reshape(1, self.head_dim)
        dt_bias = self.dt_bias.reshape(self.num_heads, self.head_dim)
        if self.gate_lower_bound is not None:
            g = _kda_decay_bounded(A_log, a_logits, dt_bias, self.gate_lower_bound)
        else:
            g = _kda_decay_softplus(A_log, a_logits, dt_bias)

        if ssm_state is None:
            ssm_state = mx.zeros(
                (B, self.num_heads, self.head_dim, self.head_dim), dtype=mx.float32
            )

        use_kernel = (
            mx.default_device() == mx.gpu and mx.metal.is_available()
        ) and not self.training
        delta = gated_delta_kernel if use_kernel else gated_delta_ops
        out, ssm_state = delta(q, k, v, g, beta, ssm_state, mask)

        if cache is not None:
            cache[3] = ssm_state
            cache.advance(T)

        if self.use_full_rank_gate:
            gate = self.g_proj(x)
        else:
            gate = self.g_b_proj(self.g_a_proj(x))
        out = (self.o_norm(out.reshape(shape)) * mx.sigmoid(gate.reshape(shape)))
        return self.o_proj(out.reshape(B, T, -1).astype(dtype))


# ── Attention residuals ──────────────────────────────────────────────────────


def _apply_attn_res(
    prefix_sum: mx.array,
    block_residual: mx.array,
    proj: nn.Linear,
    norm: nn.RMSNorm,
    eps: float,
) -> mx.array:
    """Softmax attention over the depth axis.

    prefix_sum:     (tokens, hidden)
    block_residual: (tokens, blocks, hidden)

    Mirrors modeling_kimi_linear.py::_apply_attn_res — fp32 throughout, the
    scoring vector is the elementwise product of the norm weight and the rank-1
    projection, and the output is the softmax-weighted mean of the blocks plus
    the current prefix sum.
    """
    v = mx.concatenate([block_residual, mx.expand_dims(prefix_sum, 1)], axis=1)
    v32 = v.astype(mx.float32)
    k = v32 * mx.rsqrt(mx.mean(mx.square(v32), axis=-1, keepdims=True) + eps)
    score_weight = norm.weight.astype(mx.float32) * proj.weight.reshape(-1).astype(
        mx.float32
    )
    scores = mx.sum(k * score_weight, axis=-1)
    probs = mx.expand_dims(mx.softmax(scores, axis=-1, precise=True), 1)
    return mx.squeeze(probs @ v32, 1).astype(v.dtype)


class KimiK3DecoderLayer(nn.Module):
    """One decoder layer.

    With attention residuals enabled the layer takes and returns a
    `block_residual` alongside the hidden state — the checkpoint of the running
    prefix sum every `attn_res_block_size` layers. The tuple shape is what the
    pipeline wrappers in scripts/auto_parallel.py serialise across ranks.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.eps = args.rms_norm_eps
        kda_layers = args.linear_attn_config["kda_layers"]
        self.is_linear = (layer_idx + 1) in kda_layers

        if self.is_linear:
            self.self_attn = KimiK3DeltaAttention(args, layer_idx)
        else:
            self.self_attn = KimiK3MLAAttention(args)

        if (
            args.num_experts
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        ):
            self.mlp = KimiK3SparseMoE(args)
        else:
            self.mlp = KimiK3MLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        self.use_attn_residuals = args.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.attn_res_block_size = args.attn_res_block_size
            self.self_attention_res_norm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.mlp_res_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.self_attention_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
            self.mlp_res_proj = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        block_residual: Optional[mx.array] = None,
    ):
        if not self.use_attn_residuals:
            h = x + self.self_attn(self.input_layernorm(x), mask, cache)
            return h + self.mlp(self.post_attention_layernorm(h))

        B, S, H = x.shape
        prefix_sum = x

        if block_residual is not None and block_residual.shape[1] > 0:
            x = _apply_attn_res(
                prefix_sum.reshape(-1, H),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.eps,
            ).reshape(B, S, H)

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = mx.concatenate(
                [block_residual, mx.expand_dims(prefix_sum.reshape(-1, H), 1)], axis=1
            )
            prefix_sum = None

        y = self.self_attn(self.input_layernorm(x), mask, cache)
        prefix_sum = y if prefix_sum is None else prefix_sum + y

        x = _apply_attn_res(
            prefix_sum.reshape(-1, H),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            self.eps,
        ).reshape(B, S, H)

        z = self.mlp(self.post_attention_layernorm(x))
        return prefix_sum + z, block_residual


class KimiK3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.eps = args.rms_norm_eps
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            KimiK3DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        kda_layers = args.linear_attn_config["kda_layers"]
        self.ssm_idx = kda_layers[0] - 1
        self.attn_idx = 0
        for i in range(len(self.layers)):
            if (i + 1) not in kda_layers:
                self.attn_idx = i
                break

        self.use_attn_residuals = args.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.output_attn_res_norm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.output_attn_res_proj = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        attn_mask = create_attention_mask(h, cache[self.attn_idx], return_array=True)

        block_residual = None
        if self.use_attn_residuals:
            B, S, H = h.shape
            block_residual = mx.zeros((B * S, 0, H), dtype=h.dtype)

        for layer, layer_cache in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else attn_mask
            if self.use_attn_residuals:
                h, block_residual = layer(
                    h, mask=mask, cache=layer_cache, block_residual=block_residual
                )
            else:
                h = layer(h, mask=mask, cache=layer_cache)

        if self.use_attn_residuals:
            B, S, H = h.shape
            h = _apply_attn_res(
                h.reshape(-1, H),
                block_residual,
                self.output_attn_res_proj,
                self.output_attn_res_norm,
                self.eps,
            ).reshape(B, S, H)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = KimiK3Model(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            caches.append(ArraysCache(size=4) if layer.is_linear else KVCache())
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Split kv_b_proj into the absorbed embed_q / unembed_out pair.

        The converter leaves kv_b_proj alone (quantized like the rest of the
        attention), so the split happens here — same mechanic as
        kimi_linear.Model.sanitize, including the quantized path.
        """
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx, layer in enumerate(self.layers):
            attn_prefix = f"model.layers.{layer_idx}.self_attn"
            kv_b_key = f"{attn_prefix}.kv_b_proj.weight"
            if kv_b_key not in weights:
                continue

            qk_nope = self.args.qk_nope_head_dim or self.args.head_dim
            v_head = self.args.v_head_dim or self.args.head_dim
            head_dim = qk_nope + v_head
            num_heads = self.args.num_attention_heads

            quantized = f"{attn_prefix}.kv_b_proj.scales" in weights
            v = weights.pop(kv_b_key)

            if quantized:
                dims = self.args.kv_lora_rank
                scales = weights.pop(f"{attn_prefix}.kv_b_proj.scales")
                biases = weights.pop(f"{attn_prefix}.kv_b_proj.biases")
                bits = (v.shape[-1] * 32) // dims
                group_size = dims // scales.shape[-1]
                v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)

            v = v.reshape(num_heads, head_dim, -1)
            wk = mx.contiguous(v[:, :qk_nope, :].swapaxes(-1, -2))
            wv = mx.contiguous(v[:, qk_nope:, :])

            if quantized:
                wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                weights[f"{attn_prefix}.embed_q.scales"] = wk_s
                weights[f"{attn_prefix}.embed_q.biases"] = wk_b
                weights[f"{attn_prefix}.unembed_out.scales"] = wv_s
                weights[f"{attn_prefix}.unembed_out.biases"] = wv_b

            weights[f"{attn_prefix}.embed_q.weight"] = wk
            weights[f"{attn_prefix}.unembed_out.weight"] = wv

        return weights

    @property
    def cast_predicate(self):
        def predicate(path: str):
            keep_fp32 = (
                "e_score_correction_bias",
                "A_log",
                "dt_bias",
                "o_norm",
            )
            return not any(path.endswith(suffix) for suffix in keep_fp32)

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            # Rank-1 attention-residual scorers and the 1-wide projections have
            # no group to quantize over.
            if path.endswith("_res_proj"):
                return False
            return True

        return predicate
