"""Inkling MLP variants: dense SwiGLU (with a learned output scale) and the
sparse MoE (sigmoid router with correction bias, softmax-over-selected weights,
route/global scaling, and 2 always-on shared experts forming a routing "sink").

Mirrors ``InklingMLP`` / ``InklingTopkRouter`` / ``InklingExperts`` /
``InklingSharedExperts`` / ``InklingMoE`` from transformers PR #47347.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.switch_layers import SwitchGLU

from .config import TextConfig


class DenseMLP(nn.Module):
    """SwiGLU MLP with a learned scalar output gain (``global_scale``).

    The checkpoint fuses gate+up into ``w13_dn``; the converter splits it into
    ``gate_proj``/``up_proj`` so the standard MLX quantizer sees plain ``nn.Linear``s.
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        h = config.hidden_size
        inter = config.dense_intermediate_size
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)
        self.global_scale = mx.ones((1,))

    def __call__(self, x: mx.array) -> mx.array:
        y = self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        return y * self.global_scale


class Router(nn.Module):
    """Sigmoid top-k router with a correction bias and a shared-expert sink.

    Kept in full precision (tiny). Returns per-token routed weights/indices plus
    the two shared-expert gammas produced by the same softmax (the "sink").
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.n_total = self.num_experts + self.n_shared
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.hidden = config.hidden_size
        self.weight = mx.zeros((self.n_total, config.hidden_size))
        self.bias = mx.zeros((self.num_experts,))              # e_score_correction_bias
        self.global_scale = mx.ones((1,))

    def __call__(self, x: mx.array):
        # Routing (esp. the top-k selection) is precision-sensitive: in bf16 the
        # rounding of near-tied expert scores flips which experts fire, and a wrong
        # choice compounds over 64 MoE layers into incoherent output. Compute the
        # whole router in fp32.
        flat = x.reshape(-1, self.hidden).astype(mx.float32)
        router_logits = flat @ self.weight.T.astype(mx.float32)  # [T, n_total]
        scores = mx.sigmoid(router_logits)
        routed_scores = scores[:, : self.num_experts]
        scores_for_choice = routed_scores + self.bias

        # top-k experts (order within the top-k is irrelevant downstream)
        topk_idx = mx.argpartition(-scores_for_choice, kth=self.top_k - 1, axis=-1)[:, : self.top_k]

        routed_logits = router_logits[:, : self.num_experts]
        shared_logits = router_logits[:, self.num_experts :]  # [T, n_shared]
        gathered = mx.take_along_axis(routed_logits, topk_idx, axis=-1)  # [T, top_k]
        topk_logits = mx.concatenate([gathered, shared_logits], axis=-1)  # [T, top_k+n_shared]

        # softmax over the selected (+shared) logits, computed in the log domain
        log_probs = -mx.logaddexp(mx.zeros_like(topk_logits), -topk_logits)  # logsigmoid
        weights = mx.softmax(log_probs, axis=-1)
        weights = weights * self.route_scale * self.global_scale

        shared_gammas = weights[:, self.top_k :].astype(x.dtype)   # [T, n_shared]
        topk_weights = weights[:, : self.top_k].astype(x.dtype)    # [T, top_k]
        return topk_weights, topk_idx, shared_gammas


class MoE(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.n_shared = config.n_shared_experts
        self.gate = Router(config)
        self.experts = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, config.n_routed_experts, bias=False
        )
        self.shared_experts = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, config.n_shared_experts, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, L, H = x.shape
        topk_weights, topk_idx, shared_gammas = self.gate(x)
        xf = x.reshape(-1, H)                                  # [T, H]
        T = xf.shape[0]

        routed = self.experts(xf, topk_idx)                   # [T, top_k, H]
        routed = (routed * topk_weights[..., None]).sum(axis=1)

        shared_idx = mx.broadcast_to(
            mx.arange(self.n_shared)[None], (T, self.n_shared)
        )
        shared = self.shared_experts(xf, shared_idx)          # [T, n_shared, H]
        shared = (shared.astype(mx.float32) * shared_gammas[..., None].astype(mx.float32)).sum(axis=1)
        shared = shared.astype(routed.dtype)

        return (routed + shared).reshape(B, L, H)
