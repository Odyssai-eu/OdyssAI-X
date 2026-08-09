"""Top-level Inkling multimodal model.

Checkpoint layout: ``model.llm.*`` (text backbone + untied unembed), ``model.visual.*``
(HMLP vision tower), ``model.audio.*`` (dMel audio tower). Image/audio features are
scattered into the token-embedding stream at their placeholder-token positions, then
the text backbone runs and the untied unembed head produces (muP-scaled) logits.
The MTP head (``model.mtp.*``) is intentionally not loaded (inference-irrelevant).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .audio import AudioModel
from .config import InklingConfig
from .text import TextModel
from .vision import VisionModel


def _scatter_features(embeds, input_ids, token_id, features):
    """Replace ``embeds`` rows where ``input_ids == token_id`` with ``features``
    (in sequence order). ``input_ids`` is host-known so we resolve positions on CPU."""
    B, L, H = embeds.shape
    ids = np.array(input_ids).reshape(-1)
    pos = np.nonzero(ids == token_id)[0]
    if pos.size == 0:
        return embeds
    flat = embeds.reshape(B * L, H)
    flat[mx.array(pos)] = features.astype(flat.dtype)
    return flat.reshape(B, L, H)


class InnerModel(nn.Module):
    """The ``model.`` level holding the three towers."""

    def __init__(self, config: InklingConfig):
        super().__init__()
        self.llm = TextModel(config.text)
        self.visual = VisionModel(config.vision)
        self.audio = AudioModel(config.audio)


class InklingForConditionalGeneration(nn.Module):
    def __init__(self, config: InklingConfig):
        super().__init__()
        self.config = config
        self.model = InnerModel(config)

    # --- convenience accessors ---
    @property
    def llm(self) -> TextModel:
        return self.model.llm

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        audio_input_ids: mx.array | None = None,
        conv_mask=None,
        caches=None,
        start_pos: int = 0,
        last_logit_only: bool = False,
    ) -> mx.array:
        embeds = self.model.llm.embed_tokens(input_ids)

        if pixel_values is not None:
            img = self.model.visual(pixel_values)
            embeds = _scatter_features(embeds, input_ids, self.config.image_token_id, img)

        if audio_input_ids is not None:
            aud = self.model.audio(audio_input_ids)
            embeds = _scatter_features(embeds, input_ids, self.config.audio_token_id, aud)

        hidden = self.model.llm.backbone(embeds, conv_mask=conv_mask, caches=caches, start_pos=start_pos)
        if last_logit_only:
            hidden = hidden[:, -1:, :]
        return self.model.llm.logits(hidden)
