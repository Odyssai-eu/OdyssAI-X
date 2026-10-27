from ..base import install_auto_processor_patch
from ..step3p7.processing_step3p7 import Step3VLProcessor as Step5VLProcessor
from ..step3p7.vision import VisionModel
from .config import ModelConfig, TextConfig, VisionConfig
from .language import LanguageModel
from .step5_vl import Model

install_auto_processor_patch("step5_vl", Step5VLProcessor)

# OdyssAI-X: PR #2539 registers this format in mlx_vlm/prompt_utils.py; done here
# so the vendored package needs no edit of the installed mlx-vlm.
from ...prompt_utils import MODEL_CONFIG, MessageFormat  # noqa: E402

MODEL_CONFIG.setdefault("step5_vl", MessageFormat.IMAGE_PATCH_TOKEN)
