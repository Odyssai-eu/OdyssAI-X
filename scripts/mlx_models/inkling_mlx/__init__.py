"""MLX port of thinkingmachines/Inkling (975B MoE, natively multimodal)."""

from .config import AudioConfig, InklingConfig, TextConfig, VisionConfig
from .model import InklingForConditionalGeneration
from .text import TextModel

__all__ = [
    "InklingConfig",
    "TextConfig",
    "VisionConfig",
    "AudioConfig",
    "InklingForConditionalGeneration",
    "TextModel",
]
