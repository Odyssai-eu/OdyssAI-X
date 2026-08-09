"""Image + audio preprocessing for Inkling (MLX), ported from the reference
`InklingImageProcessor` / `InklingFeatureExtractor` / `InklingProcessor`.

  * image  -> pixel_values [num_patches, T=2, 40, 40, 3]  (feeds `VisionModel`)
  * audio  -> audio_input_ids [num_frames, 80] dMel bins  (feeds `AudioModel`)

`InklingProcessor.apply` builds the full multimodal input (input_ids + features)
from a chat message list, inserting the right number of placeholder soft-tokens.
Uses numpy/PIL + transformers' mel filterbank; no torch needed at inference.
"""

from __future__ import annotations

import math

import numpy as np

# CLIP normalization (OPENAI_CLIP_MEAN / STD), per processor_config.json
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

PATCH = 40           # image patch size (== vision patch_size)
TEMPORAL = 2         # temporal_patch_size (images duplicated across 2 frames)

# audio (processor_config.json / feature_extraction_inkling.py)
SR = 16000
HOP = 800            # audio_token_duration_s (0.05) * SR
WIN = 1600           # * window_size_multiplier (2.0)
N_FFT = 1600
N_MEL = 80
DMEL_BINS = 16
DMEL_MIN, DMEL_MAX = -7.0, 2.0

# special tokens
IMAGE_TOKEN_ID = 200054      # <|unused_200054|>  (soft-token slot)
AUDIO_TOKEN_ID = 200053      # <|unused_200053|>
IMAGE_BOS = "<|content_image|>"
AUDIO_BOS = "<|content_audio_input|>"


# ------------------------------- image -------------------------------

def preprocess_image(image, max_long_edge: int | None = None) -> tuple[np.ndarray, int]:
    """PIL.Image or HxWx3 uint8 array -> (pixel_values [N,2,40,40,3] float32, N).

    ``max_long_edge`` (optional) downscales the image (LANCZOS, aspect preserved) so its
    long edge is at most this many pixels *before* patchify. Each 40px patch is one
    vision soft-token, so this directly cuts the prompt length / prefill cost for big
    images (e.g. cap 512 -> ~130 patches vs ~450 at full 960px). ``None`` keeps native
    resolution (the reference default)."""
    from PIL import Image
    if not hasattr(image, "convert"):
        image = Image.fromarray(np.asarray(image).astype(np.uint8))
    image = image.convert("RGB")
    if max_long_edge and max(image.size) > max_long_edge:
        r = max_long_edge / max(image.size)
        image = image.resize((max(1, round(image.width * r)), max(1, round(image.height * r))), Image.LANCZOS)
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    img = image[..., :3].astype(np.float32).transpose(2, 0, 1)  # -> [C, H, W]
    C, H, W = img.shape

    num_rows = (H + PATCH - 1) // PATCH
    num_cols = W // PATCH + 1                       # reference: W//P + 1
    patches = []
    for i in range(num_rows):
        for j in range(num_cols):
            p = img[:, i * PATCH:(i + 1) * PATCH, j * PATCH:(j + 1) * PATCH]  # may be < 40
            padded = np.full((C, PATCH, PATCH), -1.0, dtype=np.float32)       # pad value -1.0
            padded[:, : p.shape[1], : p.shape[2]] = p
            patches.append(padded)
    patches = np.stack(patches, axis=0)             # [N, C, 40, 40]

    # rescale (1/255) + CLIP normalize per channel
    patches = patches / 255.0
    patches = (patches - CLIP_MEAN[None, :, None, None]) / CLIP_STD[None, :, None, None]

    # add temporal dim, duplicate x2, then -> [N, T, H, W, C]
    patches = np.repeat(patches[..., None], TEMPORAL, axis=-1)   # [N, C, 40, 40, 2]
    pixel_values = patches.transpose(0, 4, 2, 3, 1)              # [N, 2, 40, 40, C]
    return pixel_values.astype(np.float32), pixel_values.shape[0]


# ------------------------------- audio -------------------------------

_mel_fb = None
def _mel_filters() -> np.ndarray:
    global _mel_fb
    if _mel_fb is None:
        from transformers.audio_utils import mel_filter_bank
        fb = mel_filter_bank(num_frequency_bins=N_FFT // 2 + 1, num_mel_filters=N_MEL,
                             min_frequency=0.0, max_frequency=SR / 2.0, sampling_rate=SR,
                             norm="slaney", mel_scale="slaney")          # [801, 80]
        _mel_fb = np.ascontiguousarray(fb.T, dtype=np.float32)           # [80, 801]
    return _mel_fb


def _log_mel(waveform: np.ndarray) -> np.ndarray:
    """raw mono waveform -> log10-mel spectrogram [num_frames, 80]."""
    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    right = math.ceil(wav.shape[0] / HOP) * HOP - wav.shape[0]
    left = max(N_FFT - HOP, 0)
    wav = np.pad(wav, (left, right))
    window = np.hanning(WIN + 1)[:-1].astype(np.float32)                 # periodic Hann
    n_frames = 1 + (wav.shape[0] - N_FFT) // HOP                          # center=False
    frames = np.stack([wav[i * HOP: i * HOP + N_FFT] * window for i in range(n_frames)])  # [T, N_FFT]
    mag = np.abs(np.fft.rfft(frames, n=N_FFT, axis=-1))                   # [T, 801]
    mag = np.maximum(mag, 1e-10)
    mel = _mel_filters() @ mag.T                                         # [80, T]
    mel = np.log10(np.maximum(mel, 1e-10))
    return mel.T                                                         # [T, 80]


def preprocess_audio(waveform: np.ndarray, sampling_rate: int = SR) -> np.ndarray:
    """raw 16 kHz mono waveform -> dMel bin ids [num_frames, 80] (int32, 0..15)."""
    if sampling_rate != SR:
        raise ValueError(f"Inkling audio expects {SR} Hz, got {sampling_rate}")
    mel = _log_mel(waveform)                                             # [T, 80] log10
    n_valid = math.ceil(len(np.asarray(waveform).reshape(-1)) / HOP)
    mel = mel[:n_valid]                                                  # drop trailing pad frames
    centers = np.linspace(DMEL_MIN, DMEL_MAX, DMEL_BINS)                 # 16 bin centers
    clamped = np.clip(mel.astype(np.float64), DMEL_MIN, DMEL_MAX)
    bins = np.abs(clamped[..., None] - centers).argmin(-1)              # nearest center
    return bins.astype(np.int32)                                        # [T, 80]


# --------------------------- prompt assembly ---------------------------

class InklingProcessor:
    """Assembles multimodal model inputs from chat messages with image/audio parts.

    Content parts: {"type":"text","text":...}, {"type":"image","image":PIL/array},
    {"type":"audio","audio":waveform, "sampling_rate":16000}.
    """

    def __init__(self, tokenizer, chat_template: str):
        self.tok = tokenizer
        self.chat_template = chat_template
        self.image_bos_id = tokenizer.encode(IMAGE_BOS, add_special_tokens=False)[0]
        self.audio_bos_id = tokenizer.encode(AUDIO_BOS, add_special_tokens=False)[0]

    def apply(self, messages, reasoning_effort: str = "none", max_long_edge: int | None = None):
        import mlx.core as mx
        pixel_values, audio_ids = [], []
        # Render text via the chat template with placeholders stripped to a sentinel,
        # then splice media spans in. We build ids directly for robustness.
        ids: list[int] = []

        def emit_text(s):
            ids.extend(self.tok.encode(s, add_special_tokens=False))

        # header: thinking-effort system message (matches chat_template)
        eff = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "max": 0.99}[reasoning_effort]
        emit_text(f"<|message_system|><|content_text|>Thinking effort level: {0 if eff == 0 else eff}<|end_message|>")

        for msg in messages:
            role = {"user": "<|message_user|>", "assistant": "<|message_model|>",
                    "system": "<|message_system|>"}[msg["role"]]
            content = msg["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            for part in content:
                t = part.get("type", "text")
                if t == "text":
                    emit_text(role + "<|content_text|>" + part["text"] + "<|end_message|>")
                elif t == "image":
                    pv, n = preprocess_image(part["image"], max_long_edge=max_long_edge)
                    pixel_values.append(pv)
                    ids.append(self.tok.encode(role, add_special_tokens=False)[0])
                    ids.append(self.image_bos_id)
                    ids.extend([IMAGE_TOKEN_ID] * n)
                    ids.extend(self.tok.encode("<|end_message|>", add_special_tokens=False))
                elif t == "audio":
                    aid = preprocess_audio(part["audio"], part.get("sampling_rate", SR))
                    audio_ids.append(aid)
                    ids.append(self.tok.encode(role, add_special_tokens=False)[0])
                    ids.append(self.audio_bos_id)
                    ids.extend([AUDIO_TOKEN_ID] * aid.shape[0])
                    ids.extend(self.tok.encode("<|end_message|>", add_special_tokens=False))
        emit_text("<|message_model|>")   # generation prompt

        out = {"input_ids": ids}
        if pixel_values:
            out["pixel_values"] = mx.array(np.concatenate(pixel_values, axis=0))
        if audio_ids:
            out["audio_input_ids"] = mx.array(np.concatenate(audio_ids, axis=0))
        return out
