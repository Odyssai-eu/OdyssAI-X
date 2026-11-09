import re
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig, TextConfig, VisionConfig
from .language import LanguageModel, RMSNormNoScale
from .vision import VisionModel


def masked_scatter(input_tensor, mask, source):
    mask_flat = mask.flatten().astype(mx.int32)
    indices = mx.cumsum(mask_flat) - 1
    aligned = source.flatten()[indices % source.size]
    return mx.where(mask_flat, aligned, input_tensor.flatten()).reshape(
        input_tensor.shape
    )


class VisionAdapter(nn.Module):
    """out_hidden → projector_hidden → projector_hidden, GELU after each (no bias)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.out_hidden_size, config.projector_hidden_size, bias=False)
        self.fc2 = nn.Linear(config.projector_hidden_size, config.projector_hidden_size, bias=False)
        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.act(self.fc2(self.act(self.fc1(x))))


# language_model.model.layers.N.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight
_CENTERED_NORM = re.compile(
    r"^language_model\.model\.layers\.\d+\."
    r"(?:input_layernorm|post_attention_layernorm|pre_feedforward_layernorm|"
    r"post_feedforward_layernorm)\.weight$"
)


def _remap_key(k: str) -> str:
    if k.startswith("model.language_model."):
        return "language_model.model." + k[len("model.language_model."):]
    if k.startswith("model.vision_tower."):
        return "vision_tower." + k[len("model.vision_tower."):]
    if k.startswith("model.vision_adapter."):
        return "vision_adapter." + k[len("model.vision_adapter."):]
    if k.startswith("model.vision_projection."):
        return "vision_projection." + k[len("model.vision_projection."):]
    if k == "lm_head.weight":
        return "language_model.lm_head.weight"
    return k


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = VisionModel(config.vision_config)
        self.vision_adapter = VisionAdapter(config)
        self.vision_projection = nn.Linear(
            config.projector_hidden_size, config.text_config.hidden_size, bias=False
        )
        self.perception_emb_norm = RMSNormNoScale(eps=config.text_config.rms_norm_eps)

    def _embed_text(self, input_ids: mx.array) -> mx.array:
        return self.language_model.model.embed_norm(
            self.language_model.model.embed_tokens(input_ids)
        )

    def encode_image(self, pixel_values: mx.array, grid_thw: mx.array) -> mx.array:
        feats = self.vision_tower(pixel_values, grid_thw)
        feats = self.vision_adapter(feats)
        feats = self.vision_projection(feats)
        return self.perception_emb_norm(feats)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values_videos", None)

        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None or grid_thw is None:
            return InputEmbeddingsFeatures(inputs_embeds=self._embed_text(input_ids))

        inputs_embeds = self._embed_text(input_ids)

        vision_cache = kwargs.get("vision_cache", None)
        cached = kwargs.get("cached_image_features", None)
        if cached is None and vision_cache is not None:
            cached = vision_cache.get(kwargs.get("_image_key"))
        if cached is not None:
            features = cached.astype(inputs_embeds.dtype)
        else:
            features = self.encode_image(pixel_values, grid_thw).astype(inputs_embeds.dtype)
            if vision_cache is not None and kwargs.get("_image_key") is not None:
                mx.eval(features)
                vision_cache.put(kwargs["_image_key"], features)

        special_mask = (input_ids == self.config.image_token_id) | (
            input_ids == self.config.video_token_id
        )
        n_tokens = int(special_mask.sum())
        if n_tokens != features.shape[0]:
            raise ValueError(
                f"Image features and image tokens do not match: tokens {n_tokens}, "
                f"features {features.shape[0]}"
            )
        mask = mx.broadcast_to(special_mask[..., None], inputs_embeds.shape)
        inputs_embeds = masked_scatter(inputs_embeds, mask, features)
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        **kwargs,
    ):
        f = self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **kwargs,
        )
        return self.language_model(
            inputs=None,
            inputs_embeds=f.inputs_embeds,
            mask=mask,
            cache=cache,
            **kwargs,
        )

    def sanitize(self, weights):
        # Convention HF MuseGlimmer : les 4 norms "centered" stockent w (~0),
        # le runtime applique (1+w). Decision de shift GLOBALE (checkpoint reel :
        # moyennes ~0.3-0.45 ; artefact MLX deja shifte : ~1.3+).
        # NB: cast f32 obligatoire — mx.mean en bf16 accumule avec ~8 bits de
        # mantisse et renvoie des valeurs grossierement fausses (0.08 vs 1.2),
        # ce qui re-shifterait un artefact deja shifte au chargement.
        centered = [
            v.astype(mx.float32)
            for k, v in weights.items()
            if _CENTERED_NORM.match(_remap_key(k)) and v.ndim == 1
        ]
        shift = bool(centered) and float(sum(mx.abs(v).mean() for v in centered) / len(centered)) < 0.5

        sanitized = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k:
                continue
            new_k = _remap_key(k)
            if shift and _CENTERED_NORM.match(new_k) and v.ndim == 1:
                v = v + 1.0
            sanitized[new_k] = v
        return sanitized

    @property
    def layers(self):
        return self.language_model.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, m):
            if not hasattr(m, "to_quantized"):
                return False
            # house Q9-style default: only the text body linears
            return path.startswith("language_model.model.layers.")

        return predicate

    def make_cache(self):
        return self.language_model.make_cache()
