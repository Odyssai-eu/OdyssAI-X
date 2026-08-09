"""Inkling text backbone (``model.llm.*``): token embedding + embed-norm,
66 decoder layers, final norm, and the (untied) unembed head.

Mirrors ``InklingTextModel`` + the unembed / muP-logit scaling from
``InklingForConditionalGeneration``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .common import RMSNorm
from .config import TextConfig
from .layers import DecoderLayer


class TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.unembed = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        return self.embed_norm(self.embed(input_ids))

    def backbone(self, inputs_embeds: mx.array, conv_mask=None, caches=None, start_pos=0) -> mx.array:
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos=start_pos,
                      cache=caches[i] if caches is not None else None,
                      conv_mask=conv_mask)
        return self.norm(h)

    def logits(self, hidden: mx.array) -> mx.array:
        hidden = hidden / self.config.logits_mup_width_multiplier
        logits = self.unembed(hidden)
        uv = self.config.unpadded_vocab_size
        if uv is not None and uv < logits.shape[-1]:
            logits = logits[..., :uv]
        return logits

    def __call__(self, input_ids: mx.array, conv_mask=None) -> mx.array:
        h = self.embed_tokens(input_ids)
        h = self.backbone(h, conv_mask=conv_mask)
        return self.logits(h)
