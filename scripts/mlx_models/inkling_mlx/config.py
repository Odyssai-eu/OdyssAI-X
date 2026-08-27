"""Configuration for the Inkling multimodal model (MLX port).

Mirrors ``thinkingmachines/Inkling`` ``config.json`` and the transformers PR #47347
reference (``InklingConfig`` / ``InklingTextConfig`` / ``InklingVisionConfig`` /
``InklingAudioConfig``). We parse the *checkpoint* config layout (top-level
``text_config`` / ``vision_config`` / ``audio_config`` / ``mtp_config``), not the
flattened transformers layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _get(d: dict, *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


@dataclass
class TextConfig:
    hidden_size: int = 6144
    num_hidden_layers: int = 66
    vocab_size: int = 201024
    unpadded_vocab_size: int | None = 200058

    # global (full) attention
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    # sliding-window attention
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 16
    swa_head_dim: int = 128
    sliding_window_size: int = 512

    # relative-position logits
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: int | None = 128000
    log_scaling_alpha: float = 0.1

    rms_norm_eps: float = 1e-6
    use_embed_norm: bool = True

    # short convolution
    sconv_kernel_size: int = 4

    # dense vs MoE MLP
    dense_mlp_idx: int = 2
    dense_intermediate_size: int = 24576  # dense MLP intermediate
    moe_intermediate_size: int = 3072     # per-expert intermediate

    # MoE routing
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    n_shared_experts: int = 2
    shared_expert_sink: bool = True
    route_scale: float = 8.0
    use_gate_bias: bool = True
    norm_after_topk: bool = True
    use_global_scale: bool = True

    logits_mup_width_multiplier: float = 24.0
    hidden_act: str = "silu"

    max_position_embeddings: int = 1048576

    # which layer indices use sliding-window ("local") attention
    local_layer_ids: list[int] = field(default_factory=list)

    # MTP head (dropped for inference)
    num_mtp_layers: int | None = None

    @property
    def layer_types(self) -> list[str]:
        local = set(self.local_layer_ids)
        return [
            "hybrid_sliding" if i in local else "hybrid"
            for i in range(self.num_hidden_layers)
        ]

    @property
    def mlp_layer_types(self) -> list[str]:
        return [
            "dense" if i < self.dense_mlp_idx else "sparse"
            for i in range(self.num_hidden_layers)
        ]

    @classmethod
    def from_dict(cls, tc: dict) -> "TextConfig":
        return cls(
            hidden_size=_get(tc, "hidden_size", default=6144),
            num_hidden_layers=_get(tc, "num_hidden_layers", default=66),
            vocab_size=_get(tc, "vocab_size", default=201024),
            unpadded_vocab_size=_get(tc, "unpadded_vocab_size"),
            num_attention_heads=_get(tc, "num_attention_heads", default=64),
            num_key_value_heads=_get(tc, "num_key_value_heads", default=8),
            head_dim=_get(tc, "head_dim", default=128),
            swa_num_attention_heads=_get(tc, "swa_num_attention_heads", default=64),
            swa_num_key_value_heads=_get(tc, "swa_num_key_value_heads", default=16),
            swa_head_dim=_get(tc, "swa_head_dim", default=128),
            sliding_window_size=_get(tc, "sliding_window_size", default=512),
            d_rel=_get(tc, "d_rel", default=16),
            rel_extent=_get(tc, "rel_extent", default=1024),
            log_scaling_n_floor=_get(tc, "log_scaling_n_floor"),
            log_scaling_alpha=_get(tc, "log_scaling_alpha", default=0.1),
            rms_norm_eps=_get(tc, "rms_norm_eps", default=1e-6),
            use_embed_norm=_get(tc, "use_embed_norm", default=True),
            sconv_kernel_size=_get(tc, "sconv_kernel_size", default=4),
            dense_mlp_idx=_get(tc, "dense_mlp_idx", default=2),
            dense_intermediate_size=_get(tc, "dense_intermediate_size", default=24576),
            # checkpoint labels the *MoE* intermediate as `intermediate_size`
            moe_intermediate_size=_get(tc, "intermediate_size", default=3072),
            n_routed_experts=_get(tc, "n_routed_experts", default=256),
            num_experts_per_tok=_get(tc, "num_experts_per_tok", default=6),
            n_shared_experts=_get(tc, "n_shared_experts", default=2),
            shared_expert_sink=_get(tc, "shared_expert_sink", default=True),
            route_scale=_get(tc, "route_scale", default=8.0),
            use_gate_bias=_get(tc, "use_gate_bias", default=True),
            norm_after_topk=_get(tc, "norm_after_topk", default=True),
            use_global_scale=_get(tc, "use_global_scale", default=True),
            logits_mup_width_multiplier=_get(tc, "logits_mup_width_multiplier", default=24.0),
            max_position_embeddings=_get(tc, "model_max_length", "max_position_embeddings", default=1048576),
            local_layer_ids=list(_get(tc, "local_layer_ids", default=[]) or []),
        )


@dataclass
class VisionConfig:
    text_hidden_size: int = 6144
    patch_size: int = 40
    temporal_patch_size: int = 2
    num_channels: int = 3
    n_layers: int = 4
    rms_norm_eps: float = 1e-6
    use_vision_norm: bool = True

    @classmethod
    def from_dict(cls, vc: dict, text_hidden: int) -> "VisionConfig":
        return cls(
            text_hidden_size=text_hidden,
            patch_size=_get(vc, "patch_size", default=40),
            temporal_patch_size=_get(vc, "temporal_patch_size", default=2),
            num_channels=_get(vc, "n_channels", "num_channels", default=3),
            n_layers=_get(vc, "n_layers", "num_hidden_layers", default=4),
            rms_norm_eps=_get(vc, "rms_norm_eps", default=1e-6),
            use_vision_norm=_get(vc, "use_vision_norm", default=True),
        )


@dataclass
class AudioConfig:
    text_hidden_size: int = 6144
    n_mel_bins: int = 80
    mel_vocab_size: int = 16
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, ac: dict, text_hidden: int) -> "AudioConfig":
        return cls(
            text_hidden_size=text_hidden,
            n_mel_bins=_get(ac, "n_mel_bins", default=80),
            mel_vocab_size=_get(ac, "mel_vocab_size", default=16),
            rms_norm_eps=_get(ac, "rms_norm_eps", default=1e-6),
        )


@dataclass
class InklingConfig:
    text: TextConfig
    vision: VisionConfig
    audio: AudioConfig
    image_token_id: int = 200054
    audio_token_id: int = 200053
    image_bos_token_id: int = 200005
    audio_bos_token_id: int = 200020
    eos_token_id: int = 200006
    model_type: str = "inkling_mm_model"

    @classmethod
    def from_dict(cls, cfg: dict) -> "InklingConfig":
        tc = dict(cfg.get("text_config", {}))
        mtp = cfg.get("mtp_config") or {}
        if mtp.get("num_nextn_predict_layers") is not None:
            tc.setdefault("num_mtp_layers", mtp.get("num_nextn_predict_layers"))
        text = TextConfig.from_dict(tc)
        vision = VisionConfig.from_dict(cfg.get("vision_config", {}) or {}, text.hidden_size)
        audio = AudioConfig.from_dict(cfg.get("audio_config", {}) or {}, text.hidden_size)
        return cls(
            text=text,
            vision=vision,
            audio=audio,
            image_token_id=_get(cfg, "image_token_id", default=200054),
            audio_token_id=_get(cfg, "audio_token_id", default=200053),
            image_bos_token_id=_get(cfg, "image_bos_token_id", default=200005),
            audio_bos_token_id=_get(cfg, "audio_bos_token_id", default=200020),
            eos_token_id=_get(cfg, "eos_token_id", default=200006),
            model_type=_get(cfg, "model_type", default="inkling_mm_model"),
        )

    @property
    def raw(self) -> dict[str, Any]:
        return {"model_type": self.model_type}
