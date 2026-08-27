"""Inkling audio tower: discrete dMel-token embedding + norm.

Each audio frame is ``n_mel_bins`` discretized bins (values in ``[0, mel_vocab_size)``);
each bin is embedded from its own slice of a shared table (offset ``bin * mel_vocab_size``)
and the per-bin embeddings are summed. Mirrors ``InklingAudioModel`` /
``InklingAudioModelEmbeddings``. Checkpoint keys: ``audio.encoder.weight`` (the
``[n_mel_bins*mel_vocab_size, hidden]`` table) and ``audio.final_norm.weight``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .common import RMSNorm
from .config import AudioConfig


class AudioModel(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Embedding(
            config.n_mel_bins * config.mel_vocab_size, config.text_hidden_size
        )
        self.final_norm = RMSNorm(config.text_hidden_size, eps=config.rms_norm_eps)
        # non-persistent: arange(n_mel_bins) * mel_vocab_size
        self._offsets = mx.arange(config.n_mel_bins) * config.mel_vocab_size

    def __call__(self, audio_input_ids: mx.array) -> mx.array:
        # audio_input_ids: [..., n_mel_bins] with values in [0, mel_vocab_size)
        embeds = self.encoder(audio_input_ids + self._offsets)  # [..., n_mel_bins, hidden]
        embeds = embeds.sum(axis=-2)                            # [..., hidden]
        return self.final_norm(embeds)
