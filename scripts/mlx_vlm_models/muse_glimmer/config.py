from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "muse_glimmer_text"
    hidden_size: int = 6656
    num_hidden_layers: int = 52
    intermediate_size: int = 19968
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    vocab_size: int = 202048
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    attention_dropout: float = 0.0
    attention_bias: bool = False
    hidden_activation: str = "silu"
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    sliding_window: int = 2048
    layer_types: Optional[List[str]] = None
    layer_rope_theta: Optional[List[float]] = None
    qk_scale_factor: float = 3.87
    output_multiplier: float = 1.0
    final_logit_softcapping: Optional[float] = None
    tie_word_embeddings: bool = False
    rope_parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.layer_rope_theta is None:
            self.layer_rope_theta = [self.rope_theta] * self.num_hidden_layers
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": self.rope_theta, "rope_type": "default"}


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "muse_glimmer_vision"
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 50
    num_attention_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    patch_temporal: int = 2
    pos_emb_height: int = 32
    pos_emb_width: int = 32
    merge_size: int = 2
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    layer_types: Optional[List[str]] = None
    rope_parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 10000.0, "rope_type": "axial"}

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def window_size(self):
        """Window size in pixels: pos grid side × patch size (32×14 = 448)."""
        return self.pos_emb_height * self.patch_size


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "muse_glimmer"
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    out_hidden_size: int = 6144
    projector_hidden_size: int = 4096
    projector_hidden_act: str = "gelu"
    image_token_id: int = 200092
    video_token_id: int = 200091
    eos_token_id: Optional[List[int]] = None
    tie_word_embeddings: bool = False
