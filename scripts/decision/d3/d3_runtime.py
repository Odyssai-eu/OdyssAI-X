"""d3 runtime: System One typed decisions over a code-readout v1 checkpoint.

One question is decided per forward pass. The prompt lists every option under a single-token answer
code; a 255-way readout scores those codes at the last prompt position and the answer is a
probability for every option. No text is generated and no input is truncated.

    from d3_runtime import D3
    model = D3.from_pretrained("<package dir or Hub id>", device="cuda:0")
    model.system_one(state="...", questions={"route": {"type": "choice", ...}})
    # {"model": ..., "answers": {"route": {"type": "choice", "choice": ..., "probabilities": {...},
    #                                     "confidence": ...}}, "usage": {"input_tokens": n, "output_tokens": 0}}

    model.system_one(state="...", questions={...}, images=["photo.png", "label.jpg"])
    model.system_one(state="...", questions={...}, videos=["clip.mp4"])

The checkpoint directory holds ``config.json`` + ``model*.safetensors`` (a transformers
``Qwen3_5Model``, or a ``Qwen3VLModel`` when ``config.json`` says ``model_type: qwen3_vl``),
``readout.safetensors`` (``{"weight": [255, hidden]}``), ``decision_config.json``
(prompt family, answer codes, attention mode, pooling, temperature, input limit) and the tokenizer.
``d3_format.py`` next to this file is the prompt and answer-code contract of the model.

Images: a request takes any number of images (PIL images, local paths, http(s) URLs or base64
``data:image/...`` URLs), shared by all of its questions. They go before the text in the user turn,
one image placeholder per image in list order, and the checkpoint's own processor (``AutoProcessor``,
torchvision backend) resizes each to at most 1,638,400 pixels (1.6 MP) and at least 65,536, keeping
the aspect ratio; each 32 x 32 pixel patch is one input token. The prompt is the text prompt with the
images in front; a request without images takes exactly the text path. Image inputs need a checkpoint
with the vision tower (``visual.*`` weights).

Videos: a request also takes any number of videos (local paths, http(s) URLs, base64 ``data:video/...``
URLs, or frame arrays: a uint8 ``[frames, height, width, 3]`` RGB array or a list of PIL images, read as
frames sampled at 2 per second), shared by all of its questions and placed after the images. Files are
decoded with OpenCV (FFmpeg): 2 frames per second, at least 4 and at most 32 frames spread evenly over the
whole video, each read by the checkpoint's own video processor at up to 200,704 pixels (0.2 MP); every two
frames are one group of input tokens (one token per 32 x 32 pixels) after a timestamp. All videos of one
request take at most 16,384 input tokens. A request without videos takes exactly the text or image path.

Temperature: the code logits are divided by ``decision_config`` ``temperature`` before the softmax. An optional
``temperature_by_questions`` table (``temperature_table``) sets it per request from the request's number of choice,
yes / no and score questions instead; Box and track answers keep ``temperature``.

Numerics: BF16 backbone with SDPA attention, FP32 readout and softmax (unless the checkpoint says
otherwise). A request's questions run in request order, ``batch_size`` per forward pass, each batch
left-padded to its longest prompt. ``noncausal_full_attention`` (Qwen3.5 backbones only) lets the
full-attention layers see the whole prompt while the Gated DeltaNet layers stay causal. ``prefix_bidir``
(Qwen3.5 backbones only) splits each prompt into a state block and a question block. The state block is the
prompt's leading tokens that end at or before the end of the state text (``d3_format.state_token_count``): the
system turn, the request's images and videos, ``State:`` and the state. The question block is the rest, from the
blank line after the state. In the full-attention layers the state block attends to itself in both directions and
the question block attends to the state block and to itself in both directions; the state never attends to a
question, so its hidden states are the same for every question of a request. The Gated DeltaNet layers stay
causal over the whole prompt. The state block is computed once per request (``StateOnce``: its keys and values in
the full-attention layers, its recurrent and convolution states in the Gated DeltaNet layers), then the question
blocks run in batches on it, and ``usage.input_tokens`` counts the state block once plus each question block
(``D3_STATE_ONCE=0``: each question as its own row, counting its whole prompt). The Gated DeltaNet kernels
are the ones transformers binds at import: flash-linear-attention (and causal-conv1d) when installed, its
PyTorch reference implementation otherwise. Qwen3-VL backbones have no linear-attention
layers and run causal attention only.

``permutation_average=True`` (off by default) also scores every choice question with two or more options
with its options in reversed order, in the same forward passes as the original order, and answers with
the per-option mean of the two distributions. Noul and score questions are scored once. A pass carrying both
orders that would exceed ``MERGE_TOKENS`` padded tokens runs the reversed prompts in a pass of their own, so
peak memory stays that of the original order.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import io
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from .d3_format import (
        MAX_OPTIONS,
        SYSTEM_PROMPT,
        answer_codes,
        describe,
        normalized,
        options,
        render as render_d3,
        state_end_char,
        state_token_count,
        to_answer,
        user_prompt,
    )
except ImportError:
    from d3_format import (
        MAX_OPTIONS,
        SYSTEM_PROMPT,
        answer_codes,
        describe,
        normalized,
        options,
        render as render_d3,
        state_end_char,
        state_token_count,
        to_answer,
        user_prompt,
    )
try:
    from . import d3_fast
except ImportError:
    try:
        import d3_fast
    except ImportError:
        d3_fast = None

RUNTIME = "d3-runtime/1"
FORMAT_VERSION = 1
PROMPTS = ("d3",)
ATTENTION_MODES = ("causal", "noncausal_full_attention", "prefix_bidir")
# Modes whose text layers run through ``PromptEncoder`` (Qwen3.5 backbones only).
ENCODED_MODES = ("noncausal_full_attention", "prefix_bidir")
# The attention function name the full-attention layers use while ``PromptEncoder`` applies the block mask.
BLOCK_ATTENTION = "d3_prefix_bidir"
BACKBONES = ("qwen3_5", "qwen3_vl")
DEFAULT_BATCH_SIZE = 8
SCORE_LEVELS = (2, 10)
MANIFEST = "MODEL_MANIFEST.json"
MANIFEST_SCHEMA = "d3-package-manifest/1"
VERIFY_MODES = ("fast", "full", "release", "none")
# "fast" verification hashes every file up to this size and checks the size of larger ones.
FAST_HASH_BYTES = 64 << 20
# Files the runtime never reads: the card and other Markdown, its images, licence texts and packaging
# metadata. Loading ("fast", "full") checks every other file the manifest lists, so these cannot fail a
# load; "release" checks them too.
NOT_LOADED = ("LICENSE", "NOTICE", "requirements.txt", ".gitattributes")
NOT_LOADED_DIRS = ("assets/", "LICENSES/")
ERRORS = ("invalid_question", "max_length_exceeded", "invalid_model_output")
# The question types a request's temperature applies to (``temperature_by_questions``).
DECISION_KINDS = ("choice", "noul", "score")
# Each error's default reason (vllm-sr contract 3.1.0, per-question errors); a refusal may name a narrower one.
DEFAULT_REASONS = {
    "invalid_question": "question_malformed",
    "max_length_exceeded": "question_tokens",
    "invalid_model_output": "model_output",
}
# The documented long-context limit of a package that declares ``validated_length``: tokens per question.
LONG_CONTEXT_TOKENS = 262_144



def _d3_compose():
    """d3_compose, imported relatively: a ``trust_remote_code`` load copies only the files named by ``from .<module>
    import`` lines, so a plain ``import d3_compose`` fails there (it works only with the package dir on sys.path)."""
    try:
        from .d3_compose import check_references  # noqa: F401 - the line transformers copies d3_compose.py for
        from . import d3_compose
    except ImportError:  # a plain module (the server, tests with the package dir on sys.path)
        import d3_compose
    return d3_compose

def with_reasons(answers: dict[str, Any]) -> dict[str, Any]:
    """Every error answer with a ``reason``: its own, or its error's default."""
    for answer in answers.values():
        if isinstance(answer, dict) and "error" in answer and "reason" not in answer:
            answer["reason"] = DEFAULT_REASONS.get(answer["error"], "question_malformed")
    return answers


def mentions(value: Any, refs: Sequence[str]) -> bool:
    """Whether a string inside ``value`` (strings, lists and object values; never object keys) holds a reference."""
    if isinstance(value, str):
        return any(ref in value for ref in refs)
    if isinstance(value, Mapping):
        return any(mentions(v, refs) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(mentions(v, refs) for v in value)
    return False
IMAGE_MIN_PIXELS = 65_536
IMAGE_MAX_PIXELS = 1_638_400
IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
# Checked for every encoded image the server receives (strict loading); in-process inputs are only
# bounded by PIL's decompression-bomb guard.
MAX_IMAGE_BYTES = 8_000_000
MAX_IMAGE_SOURCE_PIXELS = 16_000_000
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP")
DOWNLOAD_TIMEOUT_SECONDS = 30
MAX_DOWNLOAD_BYTES = 64 << 20
MERGE_TOKENS = 8192
# What the processor returns for an image batch; the fast path runs exactly these through the fused layers.
IMAGE_INPUTS = (
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "pixel_values",
    "image_grid_thw",
)
VIDEO_FPS = 2.0
VIDEO_MIN_FRAMES = 4
VIDEO_MAX_FRAMES = 32
# Per frame; one input token per 32 x 32 pixels for every two frames.
VIDEO_MAX_PIXELS = 200_704
VIDEO_MIN_PIXELS = 4_096
# All videos of one request.
VIDEO_MAX_TOKENS = 16_384
# Checked for every video the server receives (strict loading).
MAX_VIDEO_BYTES = 32_000_000
MAX_VIDEO_SOURCE_PIXELS = 8_294_400
MAX_VIDEO_SECONDS = 300
VIDEO_FORMATS = ("mp4", "webm", "quicktime", "x-matroska")
# A source frame rate outside this range is treated as unknown.
VIDEO_SOURCE_FPS = (0.1, 1000.0)
VIDEO_UNKNOWN_FPS = 24.0


class QuestionRefused(ValueError):
    """A malformed question with a narrower reason than the default (vllm-sr contract 3.1.0, per-question errors)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class MaxLengthExceeded(ValueError):
    """A question prompt is longer than the checkpoint's input limit; nothing is truncated."""


# ---------------------------------------------------------------------------------------------
# Questions and prompts
# ---------------------------------------------------------------------------------------------


@dataclass
class Question:
    """One validated question: what the prompt renders and how the answer is reported."""

    kind: str  # choice | noul | score
    original: Mapping[str, Any]
    rendered: dict[str, Any]  # a choice or noul question in the d3_format contract
    keys: list[str]
    descriptions: list[Any]


def _json_value(value: Any) -> None:
    """Reject values that strict JSON cannot carry (NaN, infinities, non-JSON types)."""
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"not representable as JSON: {exc}") from exc


def normalize_question(question: Any) -> Question:
    """Validate one System One question; raises ValueError with the reason."""
    if not isinstance(question, Mapping):
        raise ValueError("a question is an object with a type")
    kind = question.get("type")
    instructions = question.get("instructions")
    if instructions is not None:
        _json_value(instructions)
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, Mapping) or not 1 <= len(criteria) <= MAX_OPTIONS:
            raise ValueError(
                f"choice criteria must map 1 to {MAX_OPTIONS} option keys to descriptions"
            )
        if any(not isinstance(k, str) or not k for k in criteria):
            raise ValueError("choice option keys must be nonempty strings")
        _json_value(dict(criteria))
        rendered = {
            "type": "choice",
            "instructions": instructions,
            "criteria": dict(criteria),
        }
        keys = list(criteria)
        return Question(kind, question, rendered, keys, [criteria[k] for k in keys])
    if kind == "noul":
        if criteria is not None:
            if not isinstance(criteria, Mapping) or set(criteria) - {"false", "true"}:
                raise ValueError('noul criteria may only describe "false" and "true"')
            _json_value(dict(criteria))
        rendered = {"type": "noul", "instructions": instructions}
        if criteria is not None:
            rendered["criteria"] = dict(criteria)
        keys, texts = options(rendered)
        return Question(kind, question, rendered, keys, texts)
    if kind == "score":
        low, high = SCORE_LEVELS
        if not isinstance(criteria, (list, tuple)) or not low <= len(criteria) <= high:
            raise ValueError(
                f"score criteria must be an ordered list of {low} to {high} levels"
            )
        _json_value(list(criteria))
        levels = [describe(level) for level in criteria]
        if any(not text for text in levels) or len(set(levels)) != len(levels):
            raise ValueError("score levels must be distinct and nonempty")
        # The ordered levels are the options, each shown by its text alone (code order = level order).
        rendered = {
            "type": "choice",
            "instructions": instructions,
            "criteria": {text: None for text in levels},
        }
        return Question(
            kind,
            question,
            rendered,
            [str(i) for i in range(len(levels))],
            list(criteria),
        )
    raise ValueError(f"unsupported question type {kind!r} (choice, noul or score)")


def render(
    tokenizer, prompt: str, state: Any, question: dict[str, Any], codes: Sequence[str],
    header: Sequence[dict[str, Any]] | None = None,
) -> str:
    if prompt == "d3":
        return render_d3(tokenizer, state, question, codes, header)
    raise ValueError(f"unknown prompt family {prompt!r}")


def image_messages(
    prompt: str,
    state: Any,
    question: dict[str, Any],
    codes: Sequence[str],
    n_images: int,
) -> list[dict[str, Any]]:
    """The prompt family's messages with one image placeholder per image before the user text."""
    if n_images < 1:
        raise ValueError(f"an image prompt takes at least 1 image, got {n_images}")
    images: list[dict[str, Any]] = [{"type": "image"} for _ in range(n_images)]
    if prompt == "d3":
        text = {"type": "text", "text": user_prompt(state, question, codes)}
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": images + [text]},
        ]
    raise ValueError(f"unknown prompt family {prompt!r}")


def video_messages(
    prompt: str,
    state: Any,
    question: dict[str, Any],
    codes: Sequence[str],
    n_images: int,
    n_videos: int,
) -> list[dict[str, Any]]:
    """The prompt family's messages with the image placeholders, then the video placeholders, before the user text."""
    if n_videos < 1:
        raise ValueError(f"a video prompt takes at least 1 video, got {n_videos}")
    media: list[dict[str, Any]] = [{"type": "image"} for _ in range(n_images)]
    media += [{"type": "video"} for _ in range(n_videos)]
    if prompt == "d3":
        text = {"type": "text", "text": user_prompt(state, question, codes)}
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": media + [text]},
        ]
    raise ValueError(f"unknown prompt family {prompt!r}")


def reversed_question(question: Question) -> dict[str, Any] | None:
    """The rendered choice question with its options in reverse order (None when there is nothing to permute)."""
    if question.kind != "choice" or len(question.keys) < 2:
        return None
    criteria = question.rendered["criteria"]
    return dict(
        question.rendered, criteria={key: criteria[key] for key in reversed(criteria)}
    )


def average_orders(forward: Sequence[float], backward: Sequence[float]) -> list[float]:
    """Per-option mean of the original-order and reversed-order distributions, in original option order."""
    return [(a + b) / 2 for a, b in zip(forward, reversed(backward))]


def _canonical(value: Any) -> str:
    """Stable text for a score level: strings verbatim, anything else as compact key-sorted JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _confidence(values: Sequence[float]) -> float:
    """One minus the normalized Shannon entropy of a distribution, in [0, 1]."""
    n = len(values)
    if n < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in values if p > 0)
    return max(0.0, min(1.0, 1 - entropy / math.log(n)))


def product_answer(
    question: Question, probabilities: Sequence[float]
) -> dict[str, Any]:
    """The public answer object of one question from its option probabilities (prompt order)."""
    if question.kind == "score":
        shares = normalized(probabilities, len(question.keys))
        return {
            "type": "score",
            "score": sum(level * p for level, p in enumerate(shares)),
            "probabilities": dict(zip(question.keys, shares)),
            "confidence": _confidence(shares),
            "legend": {
                key: _canonical(text)
                for key, text in zip(question.keys, question.descriptions)
            },
        }
    answer = to_answer(question.rendered, probabilities)
    if question.kind == "choice":
        answer["confidence"] = _confidence(list(answer["probabilities"].values()))
    return answer


# ---------------------------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------------------------


# The subtypes the server takes in ``data:image/<subtype>;base64,`` URLs.
DATA_URL_IMAGE_TYPES = ("png", "jpeg", "jpg", "webp")


def _split_data_url(value: str, media: str) -> tuple[str, str]:
    """``data:<media>/<rest>;base64,<text>`` -> ``(<rest>, <text>)``.

    Everything before the first comma is matched lower-cased and stripped of surrounding whitespace; ``<rest>``
    is the subtype with any parameters, ``<text>`` comes back exactly as given.
    """
    head, comma, text = value.partition(",")
    head = head.strip().lower()
    opening, closing = f"data:{media}/", ";base64"
    if not comma or not head.startswith(opening) or not head.endswith(closing):
        raise ValueError(
            f"expected a base64 data URL: data:{media}/<type>;base64,<data>"
        )
    return head[len(opening) : -len(closing)], text


def _decode_base64(text: str, limit: int | None = None) -> bytes:
    """Base64 text decoded with validation. With ``limit`` the text is refused unread when it is longer than
    ``limit`` bytes can encode to, and the decoded bytes may not exceed ``limit``."""
    if limit is not None and len(text) > 4 * math.ceil(limit / 3):
        raise ValueError(f"the encoded data exceeds the {limit:,} byte limit")
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:  # bad alphabet or padding, non-ASCII
        raise ValueError("the data URL does not hold valid base64") from exc
    if limit is not None and len(data) > limit:
        raise ValueError(
            f"the decoded data has {len(data):,} bytes, over the {limit:,} byte limit"
        )
    return data


def _image_from_bytes(data: bytes):
    """The RGB pixels of an encoded image; ValueError when PIL cannot identify or decode it."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - PIL raises several classes
        raise ValueError(
            f"the image could not be decoded ({type(exc).__name__}: {exc})"
        ) from exc


def _checked_image(value: str):
    """The server's reading of an image data URL (see ``load_image``)."""
    from PIL import Image

    subtype, text = _split_data_url(value, "image")
    if subtype not in DATA_URL_IMAGE_TYPES:
        raise ValueError(
            f"image data URLs take the types {', '.join(DATA_URL_IMAGE_TYPES)}, got {subtype!r}"
        )
    data = _decode_base64(text, MAX_IMAGE_BYTES)
    try:
        probe = Image.open(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - unidentified, decompression bomb, ...
        raise ValueError(
            f"the image could not be identified ({type(exc).__name__}: {exc})"
        ) from exc
    with probe:
        if probe.format not in IMAGE_FORMATS:
            raise ValueError(
                f"the image is {probe.format or 'of an unknown format'}; send PNG, JPEG or WebP"
            )
        width, height = probe.size
        if width * height > MAX_IMAGE_SOURCE_PIXELS:
            raise ValueError(
                f"the image has {width * height:,} pixels; at most {MAX_IMAGE_SOURCE_PIXELS:,} are accepted"
            )
        try:
            probe.verify()
        except Exception as exc:  # noqa: BLE001 - PIL raises several classes
            raise ValueError(
                f"the image failed the integrity check ({type(exc).__name__}: {exc})"
            ) from exc
    return _image_from_bytes(data)


def _download(url: str, what: str = "image") -> bytes:
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": RUNTIME})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise ValueError(
            f"the {what} at {url} is larger than {MAX_DOWNLOAD_BYTES:,} bytes"
        )
    return payload


def load_image(value: Any, *, strict: bool = False):
    """One image input as an RGB PIL image: pixels converted to RGB, no EXIF rotation and no resize.

    In process ``value`` is a PIL image, a local path (``~`` expanded), an http(s) URL or a
    ``data:image/<type>;base64,`` URL in any format PIL reads, bounded only by PIL's decompression-bomb guard.
    ``strict`` (the server) takes data URLs only: PNG, JPEG or WebP files of at most ``MAX_IMAGE_BYTES`` bytes and
    ``MAX_IMAGE_SOURCE_PIXELS`` pixels that pass PIL's integrity check. Anything else raises ValueError.
    """
    from PIL import Image

    if strict:
        if not isinstance(value, str) or not value.startswith("data:"):
            raise ValueError(
                "strict mode: images must be given as a data: URL string, not a file path, web address or PIL image"
            )
        return _checked_image(value)
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value:
        raise ValueError(
            "an image is a PIL image, a local path, an http(s) URL or a data:image/...;base64 URL"
        )
    if value.startswith("data:"):
        return _image_from_bytes(_decode_base64(_split_data_url(value, "image")[1]))
    try:
        if value.startswith(("http://", "https://")):
            data = _download(value)
        else:
            data = Path(value).expanduser().read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read the image {value} ({exc})") from exc
    return _image_from_bytes(data)


def load_processor(root: Path):
    """The checkpoint's ``AutoProcessor`` with left padding and the 1.6 MP image budget."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(root))
    processor.tokenizer.padding_side = "left"
    processor.image_processor.size = {
        "shortest_edge": IMAGE_MIN_PIXELS,
        "longest_edge": IMAGE_MAX_PIXELS,
    }
    return processor


def visual_tokens(image_processor, width: int, height: int) -> int:
    """Input tokens of one image after the processor's resize (raises ValueError on aspect ratio > 200)."""
    patches = image_processor.get_number_of_image_patches(height, width, {})
    return patches // image_processor.merge_size**2


def _linear_patch_embed_forward(self, hidden_states):
    import torch.nn.functional as F

    weight = self.proj.weight
    flat = hidden_states.reshape(-1, weight[0].numel()).to(weight.dtype)
    out = F.linear(flat, weight.reshape(weight.shape[0], -1), self.proj.bias)
    return out.view(-1, self.embed_dim)


def linearize_patch_embed(model) -> int:
    """Run the vision patch embedding as the matrix product it equals; returns how many were patched.

    The patch embedding is a Conv3d whose kernel equals its stride over inputs already cut into single
    patches, i.e. a linear map of each flattened patch (same weights, same result up to summation order).
    On ROCm, MIOpen searches a Conv3d kernel for every new patch count, so each new image size would stall
    for minutes; the evaluation engine runs the same matrix product.
    """
    import types

    import torch

    count = 0
    for module in model.modules():
        proj = getattr(module, "proj", None)
        if (
            isinstance(proj, torch.nn.Conv3d)
            and tuple(proj.kernel_size) == tuple(proj.stride)
            and hasattr(module, "embed_dim")
        ):
            module.forward = types.MethodType(_linear_patch_embed_forward, module)
            count += 1
    return count


# ---------------------------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------------------------


@dataclass
class Video:
    """A decoded video: the sampled RGB frames and where they come from in the source.

    ``frames`` is a uint8 array ``[frames, height, width, 3]`` holding an even number of frames (the last
    one is repeated when needed); ``indices`` are their source frame numbers and ``fps`` the source frame
    rate, which give each frame its timestamp.
    """

    frames: Any
    fps: float
    indices: list[int]
    total_frames: int

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the frames."""
        return int(self.frames.shape[2]), int(self.frames.shape[1])

    def metadata(self) -> dict[str, Any]:
        """The video processor's ``video_metadata`` for these frames."""
        width, height = self.size
        return {
            "total_num_frames": self.total_frames,
            "fps": self.fps,
            "width": width,
            "height": height,
            "duration": self.total_frames / self.fps,
            "frames_indices": list(self.indices),
        }


def video_sample_indices(total: int, fps: float) -> list[int]:
    """Source frames to read: 2 per second, at least 4 and at most 32, spread evenly over the whole video."""
    import numpy as np

    count = int(total / fps * VIDEO_FPS)
    count = min(max(count, VIDEO_MIN_FRAMES), VIDEO_MAX_FRAMES, total)
    return np.linspace(0, total - 1, count).round().astype(int).tolist()


def _video(frames: list[Any], indices: list[int], fps: float, total: int) -> Video:
    import numpy as np

    if len(frames) % 2:
        frames.append(frames[-1])
        indices.append(indices[-1])
    try:
        stacked = np.stack(frames)
    except ValueError as exc:
        raise ValueError("all frames of a video must have the same size") from exc
    return Video(np.ascontiguousarray(stacked), float(fps), list(indices), int(total))


def _video_data_url_payload(value: str, strict: bool) -> bytes:
    """The bytes of a ``data:video/<type>[;<parameters>];base64,`` URL; ``<type>`` ends at the first ``;``.

    ``strict`` (the server) takes only the ``VIDEO_FORMATS`` types and at most ``MAX_VIDEO_BYTES`` decoded bytes,
    refusing base64 text too long for that limit before decoding it.
    """
    rest, text = _split_data_url(value, "video")
    if not strict:
        return _decode_base64(text)
    subtype = rest.split(";", 1)[0]
    if subtype not in VIDEO_FORMATS:
        raise ValueError(
            f"video data URLs take the types {', '.join(VIDEO_FORMATS)}, got {subtype!r}"
        )
    return _decode_base64(text, MAX_VIDEO_BYTES)


def _decode_video(
    source: bytes | str, *, strict: bool, end_frame: int | None = None
) -> Video:
    """Decode a video file (path) or encoded video (bytes) with OpenCV and sample its frames."""
    try:
        import cv2
    except ImportError as exc:
        raise ValueError(
            "decoding video files needs OpenCV (pip install opencv-python-headless)"
        ) from exc
    import tempfile

    try:
        api = (
            cv2.CAP_FFMPEG
            if cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG)
            else cv2.CAP_ANY
        )
    except (AttributeError, cv2.error):
        api = cv2.CAP_ANY
    temporary = None
    if not isinstance(source, str):
        # Encoded bytes are read from a temporary file, so they decode exactly as the same file given by path.
        with tempfile.NamedTemporaryFile(prefix="d3-video-", delete=False) as handle:
            handle.write(source)
        source = temporary = handle.name

    def capture():
        return cv2.VideoCapture(source, api)

    frames, kept, reader = [], [], None
    try:
        reader = capture()
        if not reader.isOpened():
            raise ValueError("the video could not be opened")
        width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("the video has no frame size")
        if strict and width * height > MAX_VIDEO_SOURCE_PIXELS:
            raise ValueError(
                f"video frames must have at most {MAX_VIDEO_SOURCE_PIXELS:,} pixels"
            )
        fps = float(reader.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (
            math.isfinite(fps) and VIDEO_SOURCE_FPS[0] <= fps <= VIDEO_SOURCE_FPS[1]
        ):
            fps = VIDEO_UNKNOWN_FPS
        total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:  # the container does not say: count the frames, then read again
            limit = int(MAX_VIDEO_SECONDS * fps) if strict else None
            while reader.grab():
                total += 1
                if limit is not None and total > limit:
                    break
            reader.release()
            reader = capture()
        if end_frame is not None:
            total = min(total, int(end_frame))
        if total <= 0:
            raise ValueError("the video has no frames")
        if strict and total / fps > MAX_VIDEO_SECONDS:
            raise ValueError(
                f"each video must be at most {MAX_VIDEO_SECONDS} seconds long"
            )
        indices = video_sample_indices(total, fps)
        wanted = set(indices)
        for number in range(indices[-1] + 1):
            if not reader.grab():
                continue
            if number in wanted:
                ok, frame = reader.retrieve()
                if ok and frame is not None and frame.size:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    kept.append(number)
    except cv2.error as exc:
        raise ValueError(f"invalid video data ({exc.__class__.__name__})") from exc
    finally:
        if reader is not None:
            reader.release()
        if temporary is not None:
            os.unlink(temporary)
    if not frames:
        raise ValueError("no frame of the video could be decoded")
    return _video(frames, kept, fps, total)


def _decode_windows(source: bytes | str, rate: int) -> list[Video]:
    """A video decoded once, in one pass, into its track windows (``d3_compose.window_plan``); each window keeps
    its source frame numbers, so its timestamps are the clip's own."""
    import tempfile

    import cv2
    dc = _d3_compose()

    temporary = None
    if not isinstance(source, str):
        with tempfile.NamedTemporaryFile(prefix="d3-video-", delete=False) as handle:
            handle.write(source)
        source = temporary = handle.name
    reader, out = None, []
    try:
        reader = cv2.VideoCapture(source)
        if not reader.isOpened():
            raise ValueError("the video could not be opened")
        fps = float(reader.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (math.isfinite(fps) and VIDEO_SOURCE_FPS[0] <= fps <= VIDEO_SOURCE_FPS[1]):
            fps = VIDEO_UNKNOWN_FPS
        total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            raise ValueError("the video has no frames")
        if total / fps > MAX_VIDEO_SECONDS:
            raise ValueError(f"each video must be at most {MAX_VIDEO_SECONDS} seconds long")
        number = 0
        for window in dc.window_plan(total, fps, rate):
            frames: dict[int, Any] = {}
            while number <= window[-1]:
                if not reader.grab():
                    break
                if number in window:
                    ok, frame = reader.retrieve()
                    if ok and frame is not None and frame.size:
                        frames[number] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                number += 1
            kept = [i for i in window if i in frames]
            if kept:
                out.append(_video([frames[i] for i in kept], list(kept), fps, total))
    finally:
        if reader is not None:
            reader.release()
        if temporary is not None:
            os.unlink(temporary)
    if not out:
        raise ValueError("no frame of the video could be decoded")
    return out


def load_video_windows(value: Any, rate: int) -> list[Video]:
    """A path, http(s) URL or data:video URL decoded into its track windows at ``rate`` updates per second."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value:
        raise ValueError("a video Box needs a path, an http(s) URL or a data:video/...;base64 URL")
    if value.startswith("data:"):
        source: bytes | str = _video_data_url_payload(value, False)
    elif value.startswith(("http://", "https://")):
        source = _download(value, "video")
    else:
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"no video file at {value}")
        source = str(path)
    return _decode_windows(source, rate)


def _frames_video(value: Any, end_frame: int | None = None) -> Video:
    """A frame array as a video sampled at ``VIDEO_FPS``, thinned evenly to at most ``VIDEO_MAX_FRAMES``."""
    import numpy as np
    from PIL import Image

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("a video given as frames needs at least one frame")
        value = [
            (
                np.asarray(frame.convert("RGB"))
                if isinstance(frame, Image.Image)
                else np.asarray(frame)
            )
            for frame in value
        ]
        if len({frame.shape for frame in value}) != 1:
            raise ValueError("all frames of a video must have the same size")
        value = np.stack(value)
    array = np.asarray(value)
    if (
        array.ndim != 4
        or array.shape[-1] != 3
        or array.dtype != np.uint8
        or not len(array)
    ):
        raise ValueError(
            "a video given as frames is a uint8 [frames, height, width, 3] RGB array or a list of "
            "PIL images or [height, width, 3] arrays"
        )
    if end_frame is not None:
        array = array[: max(1, int(end_frame))]
    total = len(array)
    if total > VIDEO_MAX_FRAMES:
        indices = (
            np.linspace(0, total - 1, VIDEO_MAX_FRAMES).round().astype(int).tolist()
        )
    else:
        indices = list(range(total))
    return _video([array[i] for i in indices], indices, VIDEO_FPS, total)


def load_video(
    value: Any, *, strict: bool = False, end_frame: int | None = None
) -> Video:
    """One video input, decoded and sampled (see the module docstring).

    ``value`` is a local path, an http(s) URL, a ``data:video/<format>;base64,`` URL, a frame array (a uint8
    ``[frames, height, width, 3]`` RGB array or a list of PIL images or ``[height, width, 3]`` arrays, read
    as frames sampled at 2 per second) or a ``Video`` this function returned. ``end_frame`` reads only the
    frames before it. ``strict`` (the server) accepts only data URLs of MP4, WebM, QuickTime or Matroska
    videos of at most 32,000,000 bytes, 300 seconds and 8,294,400 pixels per frame.
    """
    if isinstance(value, Video) and not strict:
        return value
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        if strict:
            raise ValueError(
                "cannot load this video with strict=True; pass the clip inline as a data:video/... URL"
            )
        return _frames_video(value, end_frame)
    if not value:
        raise ValueError(
            "a video is a path, an http(s) URL, a data:video/...;base64 URL or a frame array"
        )
    if strict and not value.startswith("data:"):
        raise ValueError(
            "cannot load this video with strict=True; pass the clip inline as a data:video/... URL"
        )
    try:
        if value.startswith("data:"):
            source: bytes | str = _video_data_url_payload(value, strict)
        elif value.startswith(("http://", "https://")):
            source = _download(value, "video")
        else:
            path = Path(value).expanduser()
            if not path.is_file():
                raise ValueError(f"no video file at {value}")
            source = str(path)
    except OSError as exc:
        raise ValueError(f"video unreadable ({exc})") from exc
    return _decode_video(source, strict=strict, end_frame=end_frame)


def configure_video_processor(processor) -> None:
    """Read already-sampled frames at up to ``VIDEO_MAX_PIXELS`` each with the checkpoint's video processor."""
    video = processor.video_processor
    factor = video.patch_size * video.merge_size
    video.size = {
        "shortest_edge": VIDEO_MIN_PIXELS,
        "longest_edge": VIDEO_MAX_PIXELS * VIDEO_MAX_FRAMES,
    }
    video.cap_pixels_per_frame = True
    video.max_video_tokens = VIDEO_MAX_PIXELS // factor**2
    video.do_sample_frames = False


# ---------------------------------------------------------------------------------------------
# Package files
# ---------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_file_module(path: Path):
    """Import a flat package file by its path, once per file (two checkpoints can share a process)."""
    import importlib.util
    import sys

    # Not resolved: in a Hub snapshot the .py name is a link to a blob without a suffix, which has no loader.
    path = Path(path).absolute()
    name = f"{path.stem}_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"no loader for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def fast_module(root: Path):
    """``d3_fast.py`` of the checkpoint directory when it is not importable as a module.

    ``AutoModel.from_pretrained(..., trust_remote_code=True)`` copies only the modeling files and their
    ``from .x import y`` imports into its module cache; the package's own copy (verified against
    ``MODEL_MANIFEST.json``) is next to the weights.
    """
    path = Path(root) / "d3_fast.py"
    return load_file_module(path) if path.is_file() else None


def resolve_dir(
    name_or_path: str | os.PathLike, revision: str | None = None, **hub: Any
) -> Path:
    """A local checkpoint directory, or a Hub snapshot (one commit) in the Hugging Face cache."""
    path = Path(os.fspath(name_or_path)).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    options = {k: v for k, v in hub.items() if v is not None and v is not False}
    return Path(snapshot_download(str(name_or_path), revision=revision, **options))


def loaded_files(manifest: dict[str, Any]) -> list[str]:
    """The manifest's files a load checks: all but the card, Markdown, assets, licences and packaging metadata."""
    return sorted(
        name
        for name in manifest.get("files_sha256") or {}
        if name not in NOT_LOADED
        and not name.endswith(".md")
        and not name.startswith(NOT_LOADED_DIRS)
    )


def verify_package(root: Path, mode: str = "fast") -> dict[str, Any] | None:
    """Check the files against ``MODEL_MANIFEST.json``; returns the manifest (None without one).

    ``fast`` and ``full`` check the files the runtime loads (``loaded_files``). ``release`` checks every
    file the manifest lists, hashes all of them and reports files the manifest does not list.
    """
    if mode not in VERIFY_MODES:
        raise ValueError(f"verify must be one of {VERIFY_MODES}")
    path = root / MANIFEST
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{MANIFEST} is not a {MANIFEST_SCHEMA} manifest")
    if mode == "none":
        return manifest
    listed = manifest.get("files_sha256") or {}
    sizes = manifest.get("files_bytes") or {}
    problems = []
    for name in sorted(listed) if mode == "release" else loaded_files(manifest):
        target = root / name
        if not target.is_file():
            problems.append(f"missing {name}")
            continue
        size = target.stat().st_size
        if name in sizes and size != sizes[name]:
            problems.append(f"{name}: {size} bytes, manifest says {sizes[name]}")
            continue
        if mode != "fast" or size <= FAST_HASH_BYTES:
            if sha256_file(target) != listed[name]:
                problems.append(f"{name}: SHA-256 differs from {MANIFEST}")
    if mode == "release":
        present = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
        problems += [
            f"{p.as_posix()}: not listed in {MANIFEST}"
            for p in present
            if p.as_posix() not in listed
            and p.as_posix() != MANIFEST
            and not any(part.startswith(".") or part == "__pycache__" for part in p.parts)
        ]
    if problems:
        raise ValueError("package verification failed: " + "; ".join(problems[:8]))
    return manifest


def vision_weights_present(root: Path) -> bool:
    """True when the checkpoint stores the vision tower (``visual.*`` tensors)."""
    index = root / "model.safetensors.index.json"
    if index.is_file():
        names = list(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    else:
        from safetensors import safe_open

        names = []
        for path in sorted(root.glob("model*.safetensors")):
            with safe_open(str(path), framework="pt") as handle:
                names += list(handle.keys())
    return any(name.startswith(("visual.", "model.visual.")) for name in names)


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------


def backbone_type(root: Path) -> str:
    """``model_type`` of the checkpoint's ``config.json``: ``qwen3_5`` or ``qwen3_vl``."""
    kind = json.loads((Path(root) / "config.json").read_text(encoding="utf-8")).get(
        "model_type"
    )
    if kind not in BACKBONES:
        raise ValueError(f"unsupported backbone model_type {kind!r}")
    return kind


def backbone_class(kind: str):
    """The transformers backbone class of a ``model_type``."""
    if kind == "qwen3_vl":
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        return Qwen3VLModel
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    return Qwen3_5Model


def sdpa_backends(device):
    """CUDA builds skip the cuDNN SDPA backend; ROCm keeps the defaults.

    ``D3_CUDNN_SDPA=1`` keeps PyTorch's default backend choice on CUDA too.
    """
    import contextlib

    import torch

    if (
        torch.device(device).type != "cuda"
        or not torch.version.cuda
        or getattr(torch.version, "hip", None)
        or os.environ.get("D3_CUDNN_SDPA") == "1"
    ):
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    )


def kernel_report() -> dict[str, str]:
    """Which implementation transformers bound for the Gated DeltaNet ops (kernel package or PyTorch)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    def bound(fn) -> str:
        seen, stack = set(), [fn]
        while stack:
            f = stack.pop()
            if id(f) in seen or not callable(f):
                continue
            seen.add(id(f))
            module = getattr(f, "__module__", "") or ""
            if module.startswith(("fla", "causal_conv1d", "kernels")):
                return f"{module}.{getattr(f, '__name__', '?')}"
            for cell in getattr(f, "__closure__", None) or ():
                try:
                    stack.append(cell.cell_contents)
                except ValueError:
                    pass
        return "torch-reference"

    names = ("torch_chunk_gated_delta_rule", "causal_conv1d_fn")
    report = {name: bound(getattr(m, name)) for name in names if hasattr(m, name)}
    try:
        import fla

        report["fla"] = getattr(fla, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        report["fla"] = f"missing ({type(exc).__name__})"
    return report


class StateBlocks:
    """The ``prefix_bidir`` mask of a left-padded batch: per row, (first prompt token, end of the state block)."""

    def __init__(self, bounds: Sequence[tuple[int, int]]):
        self.bounds = [(int(first), int(end)) for first, end in bounds]

    @classmethod
    def of(cls, width: int, lengths: Sequence[int], states: Sequence[int]) -> StateBlocks:
        """Rows of ``lengths`` prompt tokens left-padded to ``width``, each with its state block's token count."""
        if len(lengths) != len(states):
            raise ValueError("one state block length per row is required")
        bounds = []
        for length, state in zip(lengths, states):
            if not 0 < state < length <= width:
                raise ValueError(
                    f"a {length}-token row padded to {width} cannot have a {state}-token state block"
                )
            bounds.append((width - length, width - length + state))
        return cls(bounds)


def block_attention(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
):
    """The full-attention layers' attention under ``prefix_bidir``: two unmasked SDPA calls per row.

    The state block's queries read the state block's keys; the question block's queries read every prompt key of
    their row, the state block's and their own. Neither call needs a mask, so together they equal the dense
    block-masked attention. Left padding is never read; its outputs are zero. Returns ``[B, L, heads, head_dim]``
    like transformers' SDPA function.
    """
    import torch

    if not isinstance(attention_mask, StateBlocks):
        raise TypeError("prefix_bidir attention needs the StateBlocks of its batch")
    if dropout:
        raise ValueError("prefix_bidir attention runs without dropout")
    groups = query.shape[1] // key.shape[1]
    sdpa = torch.nn.functional.scaled_dot_product_attention
    out = query.new_zeros(query.shape)
    for row, (first, end) in enumerate(attention_mask.bounds):
        q = query[row : row + 1, :, first:]
        k = key[row : row + 1, :, first:]
        v = value[row : row + 1, :, first:]
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        state = end - first
        out[row : row + 1, :, first:end] = sdpa(
            q[:, :, :state], k[:, :, :state], v[:, :, :state], scale=scaling
        )
        out[row : row + 1, :, end:] = sdpa(q[:, :, state:], k, v, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


class PromptEncoder:
    """The text layers of a Qwen3.5 backbone over a left-padded prompt batch, with no causal mask.

    The batch goes through the decoder layers in order as one forward pass without a cache. Every softmax
    attention layer gets the boolean key mask ``[B, 1, 1, L]`` (True on prompt tokens), so SDPA lets each query
    position read every prompt position of its own row, earlier or later, and none of the left padding. The
    Gated DeltaNet layers keep their causal scan and get transformers' recurrent padding mask for the batch
    (``create_recurrent_attention_mask``: the ``[B, L]`` mask, or None when no row is padded), as in the stock
    forward pass. RoPE positions are the stock ones: ``0..L-1`` on every row of a text batch, the multimodal
    positions of the backbone for image and video prompts. The result is the final-norm hidden states.

    With ``blocks`` (``prefix_bidir``) the softmax attention layers apply the block mask instead
    (``block_attention``); the Gated DeltaNet layers, the padding and the positions are as above.
    """

    def __init__(self, language_model: Any):
        from transformers.masking_utils import create_recurrent_attention_mask
        from transformers.modeling_utils import AttentionInterface

        config = language_model.config
        depth = config.num_hidden_layers
        self.language_model = language_model
        self.recurrent_mask = create_recurrent_attention_mask
        self.stack = [
            (layer, kind == "full_attention")
            for layer, kind in zip(
                language_model.layers[:depth], config.layer_types[:depth]
            )
        ]
        AttentionInterface.register(BLOCK_ATTENTION, block_attention)

    def __call__(
        self,
        embeds: Any,
        present: Any,
        mrope: Any | None = None,
        blocks: StateBlocks | None = None,
    ) -> Any:
        """``embeds`` [B, L, D]; ``present`` [B, L] (1 on prompt tokens, 0 on left padding); ``mrope`` the
        [3, B, L] multimodal positions of an image or video batch, None for text; ``blocks`` the rows' state
        blocks (``prefix_bidir``), None for the noncausal mask. Returns [B, L, D].
        """
        import torch

        model = self.language_model
        rows, width = present.shape
        if mrope is None:
            ramp = torch.arange(width, device=embeds.device).view(1, 1, -1)
            ramp = ramp.expand(4, rows, -1)
            text_positions, mrope = ramp[0], ramp[1:]
        else:
            text_positions = None
        rope = model.rotary_emb(embeds, mrope)
        keys = present.bool()[:, None, None, :] if blocks is None else blocks
        if blocks is not None and len(blocks.bounds) != rows:
            raise ValueError("one state block per row is required")
        scan = self.recurrent_mask(
            config=model.config,
            inputs_embeds=embeds,
            attention_mask=present,
            past_key_values=None,
            position_ids=text_positions,
        )
        config = model.config
        saved = config._attn_implementation
        if blocks is not None:
            config._attn_implementation = BLOCK_ATTENTION
        try:
            hidden = embeds
            for layer, softmax in self.stack:
                hidden = layer(
                    hidden,
                    position_embeddings=rope,
                    attention_mask=keys if softmax else scan,
                    position_ids=text_positions,
                    past_key_values=None,
                    use_cache=False,
                )
        finally:
            config._attn_implementation = saved
        return model.norm(hidden)


# A state-once question pass holds at most STATE_ONCE_ROWS question blocks, rows x (state + longest block) keys up to
# STATE_ONCE_KEYS and rows x longest block x keys attention scores per head up to STATE_ONCE_SCORES.
STATE_ONCE_ROWS = 64
STATE_ONCE_KEYS = 1 << 18
STATE_ONCE_SCORES = 1 << 27


def state_once_batches(lengths: Sequence[int], state: int) -> list[list[int]]:
    """Batches of question-block indices for the state-once question pass, shortest blocks first (less padding);
    a block too large for the bounds still gets a pass of its own."""
    batches: list[list[int]] = []
    current: list[int] = []
    for i in sorted(range(len(lengths)), key=lambda j: lengths[j]):
        rows, widest = len(current) + 1, lengths[i]
        if current and (
            rows > STATE_ONCE_ROWS
            or rows * (state + widest) > STATE_ONCE_KEYS
            or rows * widest * (state + widest) > STATE_ONCE_SCORES
        ):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


class StateOnce:
    """The text layers of a Qwen3.5 backbone for ``prefix_bidir`` requests, with the state block computed once.

    Under the block mask the state never attends to a question, so its hidden states are the same for every question
    of a request. The state pass runs the state block (images and videos included) through the decoder layers as one
    row: the full-attention layers attend within the state in both directions and keep the state's keys and values
    (RoPE applied); the Gated DeltaNet layers run their causal scan and keep the final recurrent state and the last
    ``conv_kernel - 1`` inputs of their short convolution. The question pass runs a batch of question blocks, right
    padded: a full-attention query reads the state's keys and values and every key of its own question block (both
    directions; padding excluded); a Gated DeltaNet layer continues the scan from the state's recurrent and
    convolution states. Layers, kernels (the ones transformers binds), RoPE and the final norm are the per-row
    path's; ``questions`` returns the final-norm hidden state of each question's last token.
    """

    def __init__(self, language_model: Any):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

        config = language_model.config
        depth = config.num_hidden_layers
        self.language_model = language_model
        self.qwen = qwen
        self.stack = [
            (layer, kind == "full_attention")
            for layer, kind in zip(language_model.layers[:depth], config.layer_types[:depth])
        ]

    def state(self, embeds: Any, rope: Any) -> list[Any]:
        """The state pass over ``embeds`` [1, S, D] with RoPE ``rope``: per layer, the full-attention layer's keys and
        values or the Gated DeltaNet layer's (convolution inputs, final recurrent state)."""
        cache = []
        hidden = embeds
        for layer, softmax in self.stack:
            hidden, entry = self._layer(layer, softmax, hidden, rope, None, None)
            cache.append(entry)
        return cache

    def questions(self, cache: list[Any], embeds: Any, lengths: Sequence[int], rope: Any) -> Any:
        """The question pass over ``embeds`` [B, Lq, D] (right padded, ``lengths`` real tokens per row) on the state's
        ``cache``. Returns the final-norm hidden state of each row's last token, [B, D]."""
        import torch

        width = embeds.shape[1]
        present = torch.arange(width, device=embeds.device)[None] < torch.as_tensor(
            list(lengths), device=embeds.device
        )[:, None]
        hidden = embeds * present[..., None].to(embeds.dtype)
        for (layer, softmax), entry in zip(self.stack, cache):
            hidden, _ = self._layer(layer, softmax, hidden, rope, entry, present)
        last = torch.as_tensor([n - 1 for n in lengths], device=hidden.device)
        return self.language_model.norm(hidden[torch.arange(len(lengths), device=hidden.device), last])

    def _layer(self, layer, softmax: bool, hidden, rope, entry, present):
        residual = hidden
        normed = layer.input_layernorm(hidden)
        if softmax:
            mixed, entry = self._attention(layer.self_attn, normed, rope, entry, present)
        else:
            mixed, entry = self._delta(layer.linear_attn, normed, entry, present)
        hidden = residual + mixed
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden)), entry

    def _attention(self, attn, hidden, rope, entry, present):
        """``Qwen3_5Attention.forward`` with the state's keys and values in front of a question block's."""
        import torch

        shape = hidden.shape[:-1]
        heads = (*shape, -1, attn.head_dim)
        query, gate = torch.chunk(attn.q_proj(hidden).view(*shape, -1, attn.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*shape, -1)
        query = attn.q_norm(query.view(heads)).transpose(1, 2)
        key = attn.k_norm(attn.k_proj(hidden).view(heads)).transpose(1, 2)
        value = attn.v_proj(hidden).view(heads).transpose(1, 2)
        cos, sin = rope
        query, key = self.qwen.apply_rotary_pos_emb(query, key, cos, sin)
        mask = None
        if entry is None:
            entry, keys, values = (key, value), key, value
        else:
            rows, length = shape
            keys = torch.cat([entry[0].expand(rows, -1, -1, -1), key], dim=2)
            values = torch.cat([entry[1].expand(rows, -1, -1, -1), value], dim=2)
            state = entry[0].shape[2]
            mask = torch.ones(rows, 1, length, state + length, dtype=torch.bool, device=hidden.device)
            mask[:, 0, :, state:] = present[:, None, :]
        groups = query.shape[1] // keys.shape[1]
        if groups > 1:
            keys = keys.repeat_interleave(groups, dim=1)
            values = values.repeat_interleave(groups, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(query, keys, values, attn_mask=mask, scale=attn.scaling)
        out = out.transpose(1, 2).reshape(*shape, -1)
        return attn.o_proj(out * torch.sigmoid(gate)), entry

    def _delta(self, delta, hidden, entry, present):
        """``Qwen3_5GatedDeltaNet.forward`` (no cache) continuing from the state's convolution and recurrent states."""
        import torch

        qwen = self.qwen
        rows, length, _ = hidden.shape
        mixed = delta.in_proj_qkv(hidden).transpose(1, 2)
        z = delta.in_proj_z(hidden).reshape(rows, length, -1, delta.head_v_dim)
        b = delta.in_proj_b(hidden)
        a = delta.in_proj_a(hidden)
        window = delta.conv_kernel_size - 1
        weight = delta.conv1d.weight.squeeze(1)
        if entry is None:
            conv = qwen.causal_conv1d_fn(mixed, weight, delta.conv1d.bias, activation=delta.activation)
            tail = mixed[:, :, -window:]
            if tail.shape[-1] < window:
                tail = torch.nn.functional.pad(tail, (window - tail.shape[-1], 0))
            initial = None
        else:
            tail, initial = entry
            joined = torch.cat([tail.expand(rows, -1, -1), mixed], dim=-1)
            conv = qwen.causal_conv1d_fn(joined, weight, delta.conv1d.bias, activation=delta.activation)[:, :, window:]
            initial = initial.expand(rows, -1, -1, -1).contiguous()
        query, key, value = torch.split(conv.transpose(1, 2), [delta.key_dim, delta.key_dim, delta.value_dim], dim=-1)
        query = query.reshape(rows, length, -1, delta.head_k_dim)
        key = key.reshape(rows, length, -1, delta.head_k_dim)
        value = value.reshape(rows, length, -1, delta.head_v_dim)
        beta = b.sigmoid()
        g = -delta.A_log.float().exp() * torch.nn.functional.softplus(a.float() + delta.dt_bias)
        if delta.num_v_heads // delta.num_k_heads > 1:
            query = query.repeat_interleave(delta.num_v_heads // delta.num_k_heads, dim=2)
            key = key.repeat_interleave(delta.num_v_heads // delta.num_k_heads, dim=2)
        out, final = qwen.torch_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta, initial_state=initial,
            output_final_state=entry is None, use_qk_l2norm_in_kernel=True,
        )
        out = delta.norm(out.reshape(-1, delta.head_v_dim), z.reshape(-1, delta.head_v_dim))
        out = delta.out_proj(out.reshape(rows, length, -1))
        return out, ((tail, final) if entry is None else entry)


# Question blocks up to this many tokens run the Gated DeltaNet scan with the recurrent kernel on the fused path.
RECURRENT_TOKENS = 256


def fast_kernels(fast: Any):
    """``d3_kernels`` next to the loaded ``d3_fast`` module (the package's own copy), else the importable one."""
    import importlib

    try:
        return fast.kernels_next_to_this_file()
    except (ImportError, OSError):
        return importlib.import_module("d3_kernels")


class FusedStateOnce(StateOnce):
    """``StateOnce`` through the fused decoder-layer kernels of ``d3_kernels.py`` (ROCm gfx942, BF16).

    The element-wise ops are d3's fast-path kernels, which compute what the eager BF16 ops compute (residual add
    and norm, SiLU product, the Gated DeltaNet's convolution / gates and output norm, the attention's head norms,
    RoPE and gate); GEMMs, attention and the delta-rule kernels are unchanged. A question block of at most
    ``RECURRENT_TOKENS`` tokens continues the state's scan with the recurrent delta-rule kernel (one launch per
    layer instead of the chunked kernel's several). Other dtypes take ``StateOnce``.
    """

    def __init__(self, language_model: Any, kernels: Any):
        import torch

        super().__init__(language_model)
        self.k = kernels
        self.recurrent = self.qwen.torch_recurrent_gated_delta_rule
        self.params = []
        with torch.no_grad():
            for layer, softmax in self.stack:
                p = {
                    "eps": layer.input_layernorm.eps,
                    "w1_in": (1.0 + layer.input_layernorm.weight.float()).contiguous(),
                    "w1_post": (1.0 + layer.post_attention_layernorm.weight.float()).contiguous(),
                }
                if softmax:
                    m = layer.self_attn
                    p.update(qw1=(1.0 + m.q_norm.weight.float()).contiguous(),
                             kw1=(1.0 + m.k_norm.weight.float()).contiguous())
                else:
                    m = layer.linear_attn
                    p.update(conv_w=m.conv1d.weight.squeeze(1).contiguous(), A_log=m.A_log.float().contiguous(),
                             dt_bias=m.dt_bias.float().contiguous())
                self.params.append(p)
            norm = language_model.norm
            self.final = ((1.0 + norm.weight.float()).contiguous(), norm.eps)

    def state(self, embeds: Any, rope: Any) -> list[Any]:
        import torch

        if embeds.dtype != torch.bfloat16:
            return super().state(embeds, rope)
        cache = []
        hidden, delta = embeds, None
        for (layer, softmax), p in zip(self.stack, self.params):
            hidden, x = self.k.add_rmsnorm(hidden, delta, p["w1_in"], p["eps"])
            if softmax:
                delta, entry = self._fused_attention(layer.self_attn, p, x, rope, None, None)
            else:
                delta, entry = self._fused_delta(layer.linear_attn, p, x, None)
            hidden, x = self.k.add_rmsnorm(hidden, delta, p["w1_post"], p["eps"])
            mlp = layer.mlp
            delta = mlp.down_proj(self.k.silu_mul(mlp.gate_proj(x), mlp.up_proj(x)))
            cache.append(entry)
        return cache

    def questions(self, cache: list[Any], embeds: Any, lengths: Sequence[int], rope: Any) -> Any:
        import torch

        if embeds.dtype != torch.bfloat16:
            return super().questions(cache, embeds, lengths, rope)
        rows, width = embeds.shape[:2]
        present = torch.arange(width, device=embeds.device)[None] < torch.as_tensor(
            list(lengths), device=embeds.device
        )[:, None]
        state = next(entry[0].shape[2] for entry, (_, softmax) in zip(cache, self.stack) if softmax)
        mask = torch.ones(rows, 1, width, state + width, dtype=torch.bool, device=embeds.device)
        mask[:, 0, :, state:] = present[:, None, :]
        hidden, delta = embeds * present[..., None].to(embeds.dtype), None
        for (layer, softmax), p, entry in zip(self.stack, self.params, cache):
            hidden, x = self.k.add_rmsnorm(hidden, delta, p["w1_in"], p["eps"])
            if softmax:
                delta, _ = self._fused_attention(layer.self_attn, p, x, rope, entry, mask)
            else:
                delta, _ = self._fused_delta(layer.linear_attn, p, x, entry)
            hidden, x = self.k.add_rmsnorm(hidden, delta, p["w1_post"], p["eps"])
            mlp = layer.mlp
            delta = mlp.down_proj(self.k.silu_mul(mlp.gate_proj(x), mlp.up_proj(x)))
        normed = self.k.add_rmsnorm(hidden, delta, self.final[0], self.final[1])[1]
        last = torch.as_tensor([n - 1 for n in lengths], device=normed.device)
        return normed[torch.arange(rows, device=normed.device), last]

    def _fused_attention(self, attn, p, x, rope, entry, mask):
        import torch

        rows, length, _ = x.shape
        hd = attn.head_dim
        qp, kp = attn.q_proj(x), attn.k_proj(x)
        heads, kv_heads = qp.shape[-1] // (2 * hd), kp.shape[-1] // hd
        cos, sin = rope
        query, key = self.k.attn_prep(qp, kp, p["qw1"], p["kw1"], cos, sin, heads, kv_heads, hd, attn.q_norm.eps)
        value = self.qwen.repeat_kv(attn.v_proj(x).view(rows, length, kv_heads, hd).transpose(1, 2), heads // kv_heads)
        if entry is None:
            entry, keys, values = (key, value), key, value
        else:
            keys = torch.cat([entry[0].expand(rows, -1, -1, -1), key], dim=2)
            values = torch.cat([entry[1].expand(rows, -1, -1, -1), value], dim=2)
        out = torch.nn.functional.scaled_dot_product_attention(query, keys, values, attn_mask=mask, scale=attn.scaling)
        gate = qp.view(rows, length, heads, 2 * hd)[..., hd:]
        return attn.o_proj(self.k.sigmoid_gate(out.transpose(1, 2), gate)), entry

    def _fused_delta(self, delta, p, x, entry):
        import torch

        rows, length, _ = x.shape
        mixed, b, a = delta.in_proj_qkv(x), delta.in_proj_b(x), delta.in_proj_a(x)
        window = delta.conv_kernel_size - 1
        prep = (p["conv_w"], p["A_log"], p["dt_bias"], delta.num_k_heads, delta.head_k_dim)
        if entry is None:
            q, k, v, g, beta = self.k.gdn_prep(mixed, b, a, *prep)
            tail = mixed[:, -window:]
            if tail.shape[1] < window:
                tail = torch.nn.functional.pad(tail, (0, 0, window - tail.shape[1], 0))
            core, final = self.chunk(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=True,
                                     use_qk_l2norm_in_kernel=True)
            entry = (tail.contiguous(), final)
        else:
            tail, final = entry
            pad = b.new_zeros(rows, window, b.shape[-1])
            q, k, v, g, beta = (
                t[:, window:].contiguous()
                for t in self.k.gdn_prep(
                    torch.cat([tail.expand(rows, -1, -1), mixed], dim=1).contiguous(),
                    torch.cat([pad, b], dim=1).contiguous(),
                    torch.cat([pad, a], dim=1).contiguous(),
                    *prep,
                )
            )
            scan = self.recurrent if length <= RECURRENT_TOKENS else self.chunk
            core, _ = scan(q, k, v, g=g, beta=beta, initial_state=final.expand(rows, -1, -1, -1).contiguous(),
                           output_final_state=False, use_qk_l2norm_in_kernel=True)
        out = self.k.gated_rmsnorm(core.reshape(-1, delta.head_v_dim), delta.in_proj_z(x).reshape(-1, delta.head_v_dim),
                                   delta.norm.weight, delta.norm.variance_epsilon)
        return delta.out_proj(out.reshape(rows, length, -1)), entry

    @property
    def chunk(self):
        return self.qwen.torch_chunk_gated_delta_rule


def temperature_table(value: Any, source: str) -> tuple[tuple[int, ...], tuple[float, ...]] | None:
    """``decision_config`` ``temperature_by_questions`` (None when absent): ``{"max_questions": [n_1, ..., n_k],
    "temperature": [t_1, ..., t_k, t_rest]}`` with strictly increasing positive counts and one more temperature than
    counts, each finite and positive. A request with n choice, yes / no and score questions takes the ``t_i`` of the
    first ``n <= n_i``, else ``t_rest``."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"max_questions", "temperature"}:
        raise ValueError(f"{source}: temperature_by_questions must hold max_questions and temperature only")
    bounds, temps = value["max_questions"], value["temperature"]
    if (
        not isinstance(bounds, list)
        or not isinstance(temps, list)
        or len(temps) != len(bounds) + 1
        or any(isinstance(b, bool) or not isinstance(b, int) or b < 1 for b in bounds)
        or any(a >= b for a, b in zip(bounds, bounds[1:]))
        or any(
            isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0
            for t in temps
        )
    ):
        raise ValueError(
            f"{source}: temperature_by_questions needs strictly increasing positive max_questions and one more "
            "finite, positive temperature than max_questions"
        )
    return tuple(bounds), tuple(float(t) for t in temps)


@dataclass
class Prepared:
    """A tokenized request: runnable questions in request order plus the per-question errors.

    A text request holds token ``sequences``; an image request holds the decoded ``images`` (shared by
    every question), the rendered prompt ``texts`` and the planned input ``lengths`` (text tokens plus
    image tokens), and the processor tokenizes it when it runs. A video request also holds the decoded
    ``videos``, their processed pixels (``media``, one copy per image and video) and each video's
    placeholder expansion (``video_texts``).
    """

    keys: list[str]
    questions: dict[str, Question] = field(default_factory=dict)
    sequences: dict[str, list[int]] = field(default_factory=dict)
    errors: dict[str, dict[str, Any]] = field(default_factory=dict)
    images: list[Any] = field(default_factory=list)
    texts: dict[str, str] = field(default_factory=dict)
    lengths: dict[str, int] = field(default_factory=dict)
    videos: list[Video] = field(default_factory=list)
    media: dict[str, Any] = field(default_factory=dict)
    video_texts: list[str] = field(default_factory=list)
    # Reversed-option prompts of the choice questions (permutation_average only).
    reversed_sequences: dict[str, list[int]] = field(default_factory=dict)
    reversed_texts: dict[str, str] = field(default_factory=dict)
    reversed_lengths: dict[str, int] = field(default_factory=dict)
    # State block token counts, images and videos included (prefix_bidir only).
    states: dict[str, int] = field(default_factory=dict)
    reversed_states: dict[str, int] = field(default_factory=dict)
    # The request's state (prefix_bidir: the state-once path splits each prompt at the end of its text).
    state: Any = None

    @property
    def runnable(self) -> list[str]:
        return [k for k in self.keys if k in self.sequences or k in self.texts]


class D3:
    """A loaded code-readout checkpoint answering System One requests."""

    def __init__(
        self,
        root: Path,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        manifest: dict[str, Any] | None = None,
        max_length: int | None = None,
        readout_dtype: str | None = None,
        model_name: str | None = None,
        permutation_average: bool = False,
    ):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        self.torch = torch
        self.root = Path(root)
        self.manifest = manifest
        started = time.perf_counter()
        config_file = self.root / "decision_config.json"
        config = json.loads(config_file.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"{config_file.name} must hold a JSON object")
        if config.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"{config_file.name} has format_version {config.get('format_version')!r}; "
                f"this runtime reads version {FORMAT_VERSION}"
            )
        self.config = config
        self.backbone_type = backbone_type(self.root)
        picked = {}
        for key, default, allowed in (
            ("prompt", "d3", PROMPTS),
            ("attention_mode", "causal", ATTENTION_MODES),
            ("pooling", "last", ("last",)),
        ):
            picked[key] = config.get(key, default)
            if picked[key] not in allowed:
                raise ValueError(
                    f"{config_file.name}: {key} must be one of {', '.join(allowed)}, got {picked[key]!r}"
                )
        self.prompt = picked["prompt"]
        self.prompt_layout = config.get("prompt_layout")
        if self.prompt_layout not in (None, "header-v1"):
            raise ValueError(f"{config_file.name}: unknown prompt_layout {self.prompt_layout!r}")
        self.attention_mode = picked["attention_mode"]
        self.blocked = self.attention_mode == "prefix_bidir"
        if (
            self.attention_mode in ENCODED_MODES
            and self.backbone_type != "qwen3_5"
        ):
            raise ValueError(
                f"{self.attention_mode} needs a qwen3_5 backbone; this checkpoint is {self.backbone_type}"
            )
        self.temperature = float(config.get("temperature", 1.0))
        if not (math.isfinite(self.temperature) and self.temperature > 0):
            raise ValueError(
                f"{config_file.name}: temperature must be finite and positive, got {self.temperature}"
            )
        self.temperature_by_questions = temperature_table(
            config.get("temperature_by_questions"), config_file.name
        )
        self.readout_dtype = readout_dtype or config.get("readout_dtype", "float32")
        if self.readout_dtype not in ("float32", "bfloat16"):
            raise ValueError(
                f"readout_dtype must be float32 or bfloat16, got {self.readout_dtype!r}"
            )
        limit = config.get("max_length") if max_length is None else max_length
        # ``context_limit``: the package's own cap (None: none). ``validated_length``: the per-question length the
        # model was validated at, the default cap when declared; ``long_context=True`` lifts it. ``max_length`` is
        # the cap the current request runs under.
        self.context_limit = int(limit) if limit else None
        validated = config.get("validated_length")
        self.validated_length = int(validated) if validated else None
        self.max_length = self.question_cap(False)
        # The card's System One features (decision_config ``systemone``): Box / track and images inside questions
        # are answered only where the size passed their bars; elsewhere the question gets a refusal with a reason.
        systemone = config.get("systemone") or {}
        self.question_types = ("choice", "noul", "score") + tuple(
            kind for kind in ("box", "track") if kind in (systemone.get("question_types") or ())
        )
        self.image_placements = ("state",) + tuple(
            p for p in ("instructions", "criteria") if p in (systemone.get("image_placements") or ())
        )
        self.box_calibration = systemone.get("box_calibration", config.get("box_calibration"))
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        self.model_name = (
            model_name or (manifest or {}).get("model_name") or self.root.name
        )
        self.permutation_average = bool(permutation_average)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.root))
        self.tokenizer.padding_side = "left"
        codes, token_ids = answer_codes(self.tokenizer)
        if config.get("codes") != codes or config.get("token_ids") != token_ids:
            raise ValueError(
                f"the answer codes or token ids in {config_file.name} differ from the ones this tokenizer gives"
            )
        self.codes, self.token_ids = codes, token_ids
        probe = "State:\nA"
        if (
            self.tokenizer(probe)["input_ids"]
            != self.tokenizer(probe, add_special_tokens=False)["input_ids"]
        ):
            raise ValueError(
                "tokenizer adds special tokens; prompts are tokenized as rendered"
            )
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("tokenizer has no pad token")

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.kernels = kernel_report() if self.backbone_type == "qwen3_5" else {}
        if self.device.type == "cpu" and any(
            v.startswith(("fla", "causal_conv1d"))
            for k, v in self.kernels.items()
            if k != "fla"
        ):
            raise RuntimeError(
                "flash-linear-attention / causal-conv1d kernels are GPU-only; run on a GPU or "
                "use an environment without them for CPU inference"
            )
        torch.manual_seed(20260919)
        self.backbone = backbone_class(self.backbone_type).from_pretrained(
            str(self.root),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": str(self.device)},
        )
        weight = load_file(str(self.root / "readout.safetensors"))["weight"]
        hidden = self.backbone.config.text_config.hidden_size
        if tuple(weight.shape) != (MAX_OPTIONS, hidden):
            raise ValueError(
                f"readout weight has shape {tuple(weight.shape)}, expected {(MAX_OPTIONS, hidden)}"
            )
        self.readout = weight.to(self.device, getattr(torch, self.readout_dtype))
        self.backbone.eval().requires_grad_(False)
        # The plain path's text layers for noncausal and prefix_bidir checkpoints; causal ones keep the stock forward.
        self.encoder = (
            PromptEncoder(self.backbone.language_model)
            if self.attention_mode in ENCODED_MODES
            else None
        )
        # prefix_bidir: the state block once per request, then the questions batched on it (D3_STATE_ONCE=0: one
        # row per question, the reference path).
        self.state_once = (
            StateOnce(self.backbone.language_model)
            if self.blocked and os.environ.get("D3_STATE_ONCE", "1") != "0"
            else None
        )
        self.processor = None
        self.image_unavailable: str | None = None
        if not vision_weights_present(self.root):
            self.image_unavailable = (
                "the checkpoint has no vision tower (visual.* weights)"
            )
        elif not linearize_patch_embed(self.backbone):
            self.image_unavailable = "the vision patch embedding was not found"
        else:
            try:
                self.processor = load_processor(self.root)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - text requests do not use the processor
                self.image_unavailable = (
                    f"the image processor failed to load ({type(exc).__name__}: {exc})"
                )
        self.video_unavailable: str | None = None
        if self.image_unavailable is not None:
            self.video_unavailable = self.image_unavailable
        elif self.backbone_type != "qwen3_5":
            self.video_unavailable = "video inputs need a Qwen3.5 backbone"
        elif getattr(self.processor, "video_processor", None) is None:
            self.video_unavailable = "the checkpoint has no video processor"
        else:
            try:
                configure_video_processor(self.processor)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - text and image requests do not use it
                self.video_unavailable = (
                    f"the video processor failed to load ({type(exc).__name__}: {exc})"
                )
        fast, reason = d3_fast, "d3_fast.py is not present"
        if fast is None:
            try:
                fast = fast_module(self.root)
            except Exception as exc:  # noqa: BLE001 - the plain path always loads
                reason = f"d3_fast.py failed to import ({type(exc).__name__}: {exc})"
        self.fast_skipped = None if fast else reason
        self.fast = fast.install(self) if fast else None
        # prefix_bidir on ROCm gfx942: the state-once path through the fast path's fused kernels.
        self.state_once_fused = None
        if self.state_once is not None and fast and self.device.type == "cuda" and getattr(torch.version, "hip", None):
            off = os.environ.get("D3_FUSED", "").strip().lower() in ("0", "false", "no", "off")
            self.state_once_fused = "D3_FUSED=0" if off else fast.fusable(self)
            if self.state_once_fused is None:
                self.state_once = FusedStateOnce(self.backbone.language_model, fast_kernels(fast))
                fast.cap_cpu_threads(torch)
        self.loaded_seconds = time.perf_counter() - started

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | os.PathLike,
        *,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        verify: str = "fast",
        token: str | bool | None = None,
        cache_dir: str | os.PathLike | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        max_length: int | None = None,
        readout_dtype: str | None = None,
        model_name: str | None = None,
        permutation_average: bool = False,
    ) -> D3:
        """Load a package directory or Hub repository.

        ``verify``: ``fast`` (default) checks the files of ``MODEL_MANIFEST.json`` the runtime may read
        (all but the card, other Markdown, assets, licences and requirements.txt), hashing those up to
        64 MiB and checking the size of the weight shards; ``full`` hashes all of them; ``release`` hashes
        every listed file, the card included; ``none`` skips the check. The card never fails a load. A
        checkpoint without a manifest (a plain code-readout export) loads unverified. ``permutation_average``: see the
        module docstring.
        """
        root = resolve_dir(
            name_or_path,
            revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            force_download=force_download,
        )
        manifest = verify_package(root, verify)
        return cls(
            root,
            device=device,
            batch_size=batch_size,
            manifest=manifest,
            max_length=max_length,
            readout_dtype=readout_dtype,
            model_name=model_name,
            permutation_average=permutation_average,
        )

    # ------------------------------------------------------------------ requests

    def text(self, state: Any, question: dict[str, Any], header: Sequence[dict[str, Any]] | None = None) -> str:
        return render(self.tokenizer, self.prompt, state, question, self.codes, header)

    def image_text(self, state: Any, question: dict[str, Any], n_images: int) -> str:
        return self.processor.apply_chat_template(
            image_messages(self.prompt, state, question, self.codes, n_images),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def video_text(
        self, state: Any, question: dict[str, Any], n_images: int, n_videos: int
    ) -> str:
        return self.processor.apply_chat_template(
            video_messages(
                self.prompt, state, question, self.codes, n_images, n_videos
            ),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def load_videos(
        self, videos: Sequence[Any], *, strict: bool = False
    ) -> list[Video]:
        """Decode and sample a request's videos, any number; raises ValueError for a video it cannot read."""
        if not videos:
            return []
        if self.video_unavailable is not None:
            raise ValueError(
                f"video inputs are not available: {self.video_unavailable}"
            )
        decoded = []
        for number, value in enumerate(videos):
            try:
                decoded.append(load_video(value, strict=strict))
            except ValueError as exc:
                raise ValueError(f"videos[{number}]: {exc}") from exc
        return decoded

    def load_images(self, images: Sequence[Any], *, strict: bool = False) -> list[Any]:
        """Decode a request's images (RGB PIL), any number; raises ValueError for an image it cannot read."""
        if not images:
            return []
        if self.image_unavailable is not None:
            raise ValueError(
                f"image inputs are not available: {self.image_unavailable}"
            )
        decoded = []
        for number, value in enumerate(images):
            try:
                decoded.append(load_image(value, strict=strict))
            except ValueError as exc:
                raise ValueError(f"images[{number}]: {exc}") from exc
        return decoded

    def prepare(
        self,
        state: Any,
        questions: Mapping[str, Any],
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> Prepared:
        """Validate and tokenize one request; malformed ``state``, ``questions``, ``images`` or ``videos`` raise
        ValueError.

        ``images`` (a list of any number of images shared by every question) select the image path, ``videos``
        (a list of any number of videos, with or without images) the video path. Without them the request takes
        the text path unchanged.
        """
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError(
                "questions must be a nonempty mapping of question IDs to questions"
            )
        if any(not isinstance(key, str) or not key for key in questions):
            raise ValueError("question IDs must be nonempty strings")
        _json_value(state)
        if images is not None and not isinstance(images, (list, tuple)):
            raise ValueError("images must be a list of images")
        if videos is not None and not isinstance(videos, (list, tuple)):
            raise ValueError("videos must be a list of videos")
        if videos:
            prepared = self._prepare_videos(
                state,
                questions,
                self.load_images(images or []),
                self.load_videos(videos),
            )
            prepared.state = state
            return prepared
        if images:
            prepared = self._prepare_images(state, questions, self.load_images(images))
            prepared.state = state
            return prepared
        prepared = Prepared(keys=list(questions), state=state)
        texts = []
        header = None
        if self.prompt_layout == "header-v1":
            header = []
            for question in questions.values():
                try:
                    header.append(normalize_question(question).rendered)
                except ValueError:
                    continue
        for key, question in questions.items():
            try:
                normalized = normalize_question(question)
                texts.append((key, self.text(state, normalized.rendered, header)))
            except ValueError as exc:
                kind = question.get("type") if isinstance(question, Mapping) else None
                prepared.errors[key] = {
                    "type": kind,
                    "error": "invalid_question",
                    "message": str(exc),
                }
                continue
            prepared.questions[key] = normalized
        if texts:
            encoded = self.tokenizer(
                [t for _, t in texts],
                add_special_tokens=False,
                return_offsets_mapping=self.blocked,
            )
            for number, ((key, text), sequence) in enumerate(
                zip(texts, encoded["input_ids"])
            ):
                if self.max_length is not None and len(sequence) > self.max_length:
                    prepared.errors[key] = {
                        "type": prepared.questions[key].kind,
                        "error": "max_length_exceeded",
                        "message": f"the question prompt has {len(sequence)} tokens, over the maximum context "
                        f"length of {self.max_length} tokens; nothing was truncated",
                    }
                    continue
                if self.blocked:
                    prepared.states[key] = self.state_block(
                        text, state, encoded["offset_mapping"][number], len(sequence)
                    )
                prepared.sequences[key] = sequence
        if self.permutation_average:
            self._prepare_reversed(state, prepared)
        return prepared

    def state_block(
        self, text: str, state: Any, offsets: Sequence[Sequence[int]], length: int, media: int = 0
    ) -> int:
        """The state block's token count of one ``prefix_bidir`` prompt of ``length`` input tokens.

        ``d3_format.state_token_count`` of the rendered ``text`` (its tokens' character ``offsets``), plus ``media``:
        the tokens the image and video placeholders, all inside the state block, add when they are expanded.
        """
        count = state_token_count(offsets, state_end_char(text, state)) + media
        if not 0 < count < length:
            raise RuntimeError(
                f"a {length}-token prompt cannot have a {count}-token state block"
            )
        return count

    def _prepare_reversed(
        self, state: Any, prepared: Prepared, visual: int = 0
    ) -> None:
        """Render and tokenize the reversed-option prompt of every runnable choice question.

        ``visual``: the image and video tokens of the request (image and video paths), counted like
        ``_prepare_images`` and ``_prepare_videos`` count them.
        """
        n_images, n_videos = len(prepared.images), len(prepared.videos)
        texts = []
        for key in prepared.runnable:
            flipped = reversed_question(prepared.questions[key])
            if flipped is not None:
                texts.append(
                    (
                        key,
                        (
                            self.video_text(state, flipped, n_images, n_videos)
                            if n_videos
                            else (
                                self.image_text(state, flipped, n_images)
                                if n_images
                                else self.text(state, flipped)
                            )
                        ),
                    )
                )
        if not texts:
            return
        tokenizer = self.processor.tokenizer if n_images or n_videos else self.tokenizer
        encoded = tokenizer(
            [t for _, t in texts],
            add_special_tokens=False,
            return_offsets_mapping=self.blocked,
        )
        for number, ((key, text), sequence) in enumerate(
            zip(texts, encoded["input_ids"])
        ):
            length = len(sequence) - n_images - n_videos + visual
            if self.max_length is not None and length > self.max_length:
                for planned in (
                    prepared.sequences,
                    prepared.texts,
                    prepared.lengths,
                    prepared.states,
                ):
                    planned.pop(key, None)
                prepared.errors[key] = {
                    "type": prepared.questions[key].kind,
                    "error": "max_length_exceeded",
                    "message": f"the reversed-option prompt has {length} tokens, over the maximum context "
                    f"length of {self.max_length} tokens; nothing was truncated",
                }
                continue
            if self.blocked:
                prepared.reversed_states[key] = self.state_block(
                    text,
                    state,
                    encoded["offset_mapping"][number],
                    length,
                    visual - n_images - n_videos,
                )
            if n_images or n_videos:
                prepared.reversed_texts[key] = text
                prepared.reversed_lengths[key] = length
            else:
                prepared.reversed_sequences[key] = sequence

    def _prepare_images(
        self, state: Any, questions: Mapping[str, Any], images: list[Any]
    ) -> Prepared:
        """Render every question with the images in front and plan its input tokens (images not run yet)."""
        try:
            visual = [
                visual_tokens(self.processor.image_processor, *image.size)
                for image in images
            ]
        except ValueError as exc:
            raise ValueError(f"image rejected by the processor: {exc}") from exc
        prepared = Prepared(keys=list(questions), images=images)
        texts = []
        for key, question in questions.items():
            try:
                normalized = normalize_question(question)
                text = self.image_text(state, normalized.rendered, len(images))
            except ValueError as exc:
                kind = question.get("type") if isinstance(question, Mapping) else None
                prepared.errors[key] = {
                    "type": kind,
                    "error": "invalid_question",
                    "message": str(exc),
                }
                continue
            if text.count(IMAGE_TOKEN) != len(images) or VIDEO_TOKEN in text:
                prepared.errors[key] = {
                    "type": normalized.kind,
                    "error": "invalid_question",
                    "message": "the state or question contains a literal image or video placeholder token",
                }
                continue
            prepared.questions[key] = normalized
            texts.append((key, text))
        if texts:
            encoded = self.processor.tokenizer(
                [t for _, t in texts],
                add_special_tokens=False,
                return_offsets_mapping=self.blocked,
            )
            for number, ((key, text), sequence) in enumerate(
                zip(texts, encoded["input_ids"])
            ):
                length = len(sequence) - len(images) + sum(visual)
                if self.max_length is not None and length > self.max_length:
                    prepared.errors[key] = {
                        "type": prepared.questions[key].kind,
                        "error": "max_length_exceeded",
                        "message": f"the question prompt has {length} tokens ({sum(visual)} for "
                        f"{len(images)} image(s)), over the maximum context length of {self.max_length} "
                        "tokens; nothing was truncated",
                    }
                    continue
                if self.blocked:
                    prepared.states[key] = self.state_block(
                        text,
                        state,
                        encoded["offset_mapping"][number],
                        length,
                        sum(visual) - len(images),
                    )
                prepared.texts[key] = text
                prepared.lengths[key] = length
        if self.permutation_average:
            self._prepare_reversed(state, prepared, sum(visual))
        return prepared

    def _prepare_videos(
        self,
        state: Any,
        questions: Mapping[str, Any],
        images: list[Any],
        videos: list[Video],
    ) -> Prepared:
        """Process the images and videos once, render every question with them in front and plan its tokens."""
        processor = self.processor
        try:
            visual = [
                visual_tokens(processor.image_processor, *image.size)
                for image in images
            ]
        except ValueError as exc:
            raise ValueError(f"image rejected by the processor: {exc}") from exc
        merge = processor.video_processor.merge_size**2
        try:
            planned = (
                sum(
                    processor.video_processor.get_num_of_video_patches(
                        len(video.frames), video.size[1], video.size[0]
                    )
                    for video in videos
                )
                // merge
            )
        except ValueError as exc:
            raise ValueError(f"video rejected by the processor: {exc}") from exc
        if planned > VIDEO_MAX_TOKENS:
            raise ValueError(
                f"the videos take {planned:,} input tokens, over the {VIDEO_MAX_TOKENS:,} tokens a "
                "request may spend on videos; send fewer or shorter videos"
            )
        try:
            media = dict(
                processor.video_processor(
                    videos=[video.frames for video in videos],
                    video_metadata=[video.metadata() for video in videos],
                    return_metadata=True,
                    return_tensors="pt",
                )
            )
        except ValueError as exc:
            raise ValueError(f"video rejected by the processor: {exc}") from exc
        video_texts = [
            processor.replace_video_token(media, video_idx=i)
            for i in range(len(videos))
        ]
        if int(media["video_grid_thw"].prod(-1).sum()) // merge != planned:
            raise RuntimeError("the video processor and its token count disagree")
        expanded = sum(
            len(ids)
            for ids in processor.tokenizer(video_texts, add_special_tokens=False)[
                "input_ids"
            ]
        )
        media = {
            "pixel_values_videos": media["pixel_values_videos"],
            "video_grid_thw": media["video_grid_thw"],
        }
        if images:
            media.update(processor.image_processor(images=images, return_tensors="pt"))
        prepared = Prepared(
            keys=list(questions),
            images=images,
            videos=videos,
            media=media,
            video_texts=video_texts,
        )
        texts = []
        for key, question in questions.items():
            try:
                normalized = normalize_question(question)
                text = self.video_text(
                    state, normalized.rendered, len(images), len(videos)
                )
            except ValueError as exc:
                kind = question.get("type") if isinstance(question, Mapping) else None
                prepared.errors[key] = {
                    "type": kind,
                    "error": "invalid_question",
                    "message": str(exc),
                }
                continue
            if text.count(IMAGE_TOKEN) != len(images) or text.count(VIDEO_TOKEN) != len(
                videos
            ):
                prepared.errors[key] = {
                    "type": normalized.kind,
                    "error": "invalid_question",
                    "message": "the state or question contains a literal image or video placeholder token",
                }
                continue
            prepared.questions[key] = normalized
            texts.append((key, text))
        if texts:
            encoded = processor.tokenizer(
                [t for _, t in texts],
                add_special_tokens=False,
                return_offsets_mapping=self.blocked,
            )
            for number, ((key, text), sequence) in enumerate(
                zip(texts, encoded["input_ids"])
            ):
                length = (
                    len(sequence) - len(images) - len(videos) + sum(visual) + expanded
                )
                if self.max_length is not None and length > self.max_length:
                    prepared.errors[key] = {
                        "type": prepared.questions[key].kind,
                        "error": "max_length_exceeded",
                        "message": f"the question prompt has {length} tokens ({sum(visual) + expanded} for "
                        f"{len(images)} image(s) and {len(videos)} video(s)), over the maximum context length "
                        f"of {self.max_length} tokens; nothing was truncated",
                    }
                    continue
                if self.blocked:
                    prepared.states[key] = self.state_block(
                        text,
                        state,
                        encoded["offset_mapping"][number],
                        length,
                        sum(visual) + expanded - len(images) - len(videos),
                    )
                prepared.texts[key] = text
                prepared.lengths[key] = length
        if self.permutation_average:
            self._prepare_reversed(state, prepared, sum(visual) + expanded)
        return prepared

    def blocks(
        self, width: int, lengths: Sequence[int], states: Sequence[int] | None
    ) -> StateBlocks | None:
        """The state blocks of a batch of rows padded to ``width`` (``prefix_bidir``; None in the other modes)."""
        if not self.blocked:
            return None
        if states is None:
            raise ValueError("prefix_bidir prompts need their state block token counts")
        return StateBlocks.of(width, lengths, states)

    def logits(
        self,
        sequences: Sequence[Sequence[int]],
        counts: Sequence[int],
        states: Sequence[int] | None = None,
    ):
        """Masked code logits (FP32, [B, 255]) for one left-padded batch; ``states``: the prompts' state block
        token counts (``prefix_bidir``)."""
        torch = self.torch
        width = max(len(s) for s in sequences)
        blocks = self.blocks(width, [len(s) for s in sequences], states)
        ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for i, sequence in enumerate(sequences):
            ids[i, width - len(sequence) :] = torch.as_tensor(
                sequence, dtype=torch.long
            )
            mask[i, width - len(sequence) :] = 1
        ids, mask = ids.to(self.device, non_blocking=True), mask.to(
            self.device, non_blocking=True
        )
        with torch.inference_mode(), sdpa_backends(self.device):
            if self.encoder is not None:
                embeds = self.backbone.get_input_embeddings()(ids)
                hidden = self.encoder(embeds, mask, blocks=blocks)[:, -1]
            else:
                hidden = self.backbone(
                    input_ids=ids, attention_mask=mask, use_cache=False
                ).last_hidden_state[:, -1]
            if self.readout_dtype == "float32":
                logits = hidden.float() @ self.readout.T
            else:
                logits = torch.nn.functional.linear(hidden, self.readout).float()
            limit = torch.as_tensor(list(counts), device=self.device)[:, None]
            invalid = torch.arange(MAX_OPTIONS, device=self.device)[None] >= limit
            return logits.masked_fill(invalid, float("-inf"))

    def probabilities(
        self,
        sequences: Sequence[Sequence[int]],
        counts: Sequence[int],
        states: Sequence[int] | None = None,
    ) -> list[list[float]]:
        """Option probabilities of one batch, each row cut to its question's option count."""
        if self.fast is not None:
            return self.fast.probabilities(sequences, counts)
        scaled = self.logits(sequences, counts, states) / self.temperature
        rows = self.torch.softmax(scaled, dim=-1).cpu().tolist()
        return [row[:count] for row, count in zip(rows, counts)]

    def image_embeddings(self, inputs: Mapping[str, Any]):
        """Input embeddings with the image features in place, and the multimodal RoPE positions, of an image
        batch: the steps the backbone's own forward takes before its text layers."""
        torch = self.torch
        unexpected = sorted(set(inputs) - set(IMAGE_INPUTS))
        if unexpected:
            raise RuntimeError(
                f"the processor returned inputs the runtime does not read: {unexpected}"
            )
        backbone = self.backbone
        ids = inputs["input_ids"]
        embeds = backbone.get_input_embeddings()(ids)
        pooled = backbone.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True
        ).pooler_output
        pooled = torch.cat(pooled, dim=0).to(embeds.device, embeds.dtype)
        slots, _ = backbone.get_placeholder_mask(
            ids, inputs_embeds=embeds, image_features=pooled
        )
        embeds = embeds.masked_scatter(slots, pooled)
        mrope = backbone.compute_3d_position_ids(
            input_ids=ids,
            image_grid_thw=inputs["image_grid_thw"],
            video_grid_thw=None,
            inputs_embeds=embeds,
            attention_mask=inputs["attention_mask"],
            past_key_values=None,
            mm_token_type_ids=inputs["mm_token_type_ids"],
        )
        return embeds, mrope

    def media_blocks(
        self, ids: Any, mask: Any, lengths: Sequence[int], states: Sequence[int] | None
    ) -> StateBlocks | None:
        """The state blocks of an expanded image or video batch (``prefix_bidir``; None in the other modes).

        Checks on the host tensors that every row has its planned token count and that no image or video token
        falls in a question block: the placeholders are all inside the state block.
        """
        if not self.blocked:
            return None
        real = mask.sum(-1).tolist()
        if real != list(lengths):
            raise RuntimeError(f"planned {list(lengths)} input tokens per row, the processor produced {real}")
        blocks = self.blocks(ids.shape[1], real, states)
        media = [
            self.processor.tokenizer.convert_tokens_to_ids(token)
            for token in (IMAGE_TOKEN, VIDEO_TOKEN)
        ]
        for row, (_, end) in enumerate(blocks.bounds):
            if self.torch.isin(ids[row, end:], self.torch.as_tensor(media)).any():
                raise RuntimeError("an image or video token falls outside the state block")
        return blocks

    def image_logits(
        self,
        texts: Sequence[str],
        images: Sequence[Any],
        counts: Sequence[int],
        width: int,
        lengths: Sequence[int] | None = None,
        states: Sequence[int] | None = None,
    ):
        """Masked code logits (FP32, [B, 255]) for one left-padded batch of prompts that share ``images``.

        The processor expands each prompt's image placeholders, resizes the images and pads the batch to
        ``width`` tokens (the longest planned prompt); the backbone gets every tensor it returns. ``lengths`` and
        ``states``: each prompt's planned token count and state block token count (``prefix_bidir``).
        """
        torch = self.torch
        encoded = self.processor(
            text=list(texts),
            images=[image for _ in texts for image in images],
            padding=True,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] != width:
            raise RuntimeError(
                f"planned {width} input tokens, the processor produced {encoded['input_ids'].shape[1]}"
            )
        blocks = self.media_blocks(
            encoded["input_ids"], encoded["attention_mask"], lengths or [], states
        )
        inputs = {name: value.to(self.device) for name, value in encoded.items()}
        fused = self.fast is not None and self.fast.fused is not None
        with torch.inference_mode(), sdpa_backends(self.device):
            if fused and set(inputs) == set(IMAGE_INPUTS):
                padded = not bool(encoded["attention_mask"].all())
                hidden = self.fast.image_hidden(inputs, padded)[:, -1]
            elif self.encoder is not None:
                embeds, mrope = self.image_embeddings(inputs)
                hidden = self.encoder(
                    embeds, inputs["attention_mask"], mrope, blocks=blocks
                )[:, -1]
            else:
                hidden = self.backbone(**inputs, use_cache=False).last_hidden_state[
                    :, -1
                ]
            if self.readout_dtype == "float32":
                logits = hidden.float() @ self.readout.T
            else:
                logits = torch.nn.functional.linear(hidden, self.readout).float()
            limit = torch.as_tensor(list(counts), device=self.device)[:, None]
            invalid = torch.arange(MAX_OPTIONS, device=self.device)[None] >= limit
            return logits.masked_fill(invalid, float("-inf"))

    def image_probabilities(
        self,
        texts: Sequence[str],
        images: Sequence[Any],
        counts: Sequence[int],
        width: int,
        lengths: Sequence[int] | None = None,
        states: Sequence[int] | None = None,
    ) -> list[list[float]]:
        probs = (
            (
                self.image_logits(texts, images, counts, width, lengths, states)
                / self.temperature
            )
            .softmax(-1)
            .cpu()
            .tolist()
        )
        return [p[:c] for p, c in zip(probs, counts)]

    def batch_states(
        self, prepared: Prepared, chunk: Sequence[str], extra: Sequence[str]
    ) -> list[int] | None:
        """The state block token counts of one pass, original-order prompts first (``prefix_bidir`` only)."""
        if not self.blocked:
            return None
        return [prepared.states[k] for k in chunk] + [
            prepared.reversed_states[k] for k in extra
        ]

    def _run_images(self, prepared: Prepared) -> tuple[dict[str, list[float]], int]:
        keys = prepared.runnable
        out: dict[str, list[float]] = {}
        for start in range(0, len(keys), self.batch_size):
            chunk = keys[start : start + self.batch_size]
            extra = [k for k in chunk if k in prepared.reversed_texts]
            texts = [prepared.texts[k] for k in chunk] + [
                prepared.reversed_texts[k] for k in extra
            ]
            counts = [len(prepared.questions[k].keys) for k in chunk + extra]
            widths = [prepared.lengths[k] for k in chunk] + [
                prepared.reversed_lengths[k] for k in extra
            ]
            states = self.batch_states(prepared, chunk, extra)
            n = len(chunk)
            if extra and len(widths) * max(widths) > MERGE_TOKENS:
                probs = self.image_probabilities(
                    texts[:n],
                    prepared.images,
                    counts[:n],
                    max(widths[:n]),
                    widths[:n],
                    states and states[:n],
                ) + self.image_probabilities(
                    texts[n:],
                    prepared.images,
                    counts[n:],
                    max(widths[n:]),
                    widths[n:],
                    states and states[n:],
                )
            else:
                probs = self.image_probabilities(
                    texts, prepared.images, counts, max(widths), widths, states
                )
            out.update(zip(chunk, probs))
            for key, backward in zip(extra, probs[n:]):
                out[key] = average_orders(out[key], backward)
        return out, sum(prepared.lengths[k] for k in keys) + sum(
            prepared.reversed_lengths.values()
        )

    def media_features(self, media: Mapping[str, Any]) -> dict[str, list[Any]]:
        """Vision-tower features of each image and video of a video request, computed once per request."""
        torch = self.torch
        backbone = self.backbone
        features: dict[str, list[Any]] = {}
        with torch.inference_mode(), sdpa_backends(self.device):
            if "pixel_values" in media:
                features["image"] = list(
                    backbone.get_image_features(
                        media["pixel_values"].to(self.device),
                        media["image_grid_thw"].to(self.device),
                        return_dict=True,
                    ).pooler_output
                )
            features["video"] = list(
                backbone.get_video_features(
                    media["pixel_values_videos"].to(self.device),
                    media["video_grid_thw"].to(self.device),
                    return_dict=True,
                ).pooler_output
            )
        return features

    def video_logits(
        self,
        texts: Sequence[str],
        prepared: Prepared,
        features: Mapping[str, list[Any]],
        counts: Sequence[int],
        width: int,
        lengths: Sequence[int] | None = None,
        states: Sequence[int] | None = None,
    ):
        """Masked code logits (FP32, [B, 255]) for one left-padded batch of prompts that share the request's videos.

        Each prompt's placeholders are expanded with the request's images and videos and the batch is padded to
        ``width`` tokens (the longest planned prompt); the vision features of the request go into every prompt,
        then the text layers run as for an image batch (the fused layers when the fast path is active).
        ``lengths`` and ``states``: each prompt's planned token count and state block token count (``prefix_bidir``).
        """
        torch = self.torch
        processor = self.processor
        backbone = self.backbone
        media = prepared.media
        image_texts = [
            processor.replace_image_token(media, image_idx=i)
            for i in range(len(prepared.images))
        ]
        expanded = [
            processor.get_text_with_replacements(
                [text], image_texts, prepared.video_texts
            )[0][0]
            for text in texts
        ]
        encoded = processor.tokenizer(
            expanded, padding=True, return_token_type_ids=False, return_tensors="pt"
        )
        if encoded["input_ids"].shape[1] != width:
            raise RuntimeError(
                f"planned {width} input tokens, the processor produced {encoded['input_ids'].shape[1]}"
            )
        blocks = self.media_blocks(
            encoded["input_ids"], encoded["attention_mask"], lengths or [], states
        )
        rows = len(texts)
        ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device)
        types = torch.as_tensor(
            processor.create_mm_token_type_ids(encoded["input_ids"]), device=self.device
        )
        padded = not bool(encoded["attention_mask"].all())
        with torch.inference_mode(), sdpa_backends(self.device):
            embeds = backbone.get_input_embeddings()(ids)
            image_grid = None
            if "image" in features:
                values = torch.cat(features["image"] * rows).to(
                    embeds.device, embeds.dtype
                )
                image_mask, _ = backbone.get_placeholder_mask(
                    ids, inputs_embeds=embeds, image_features=values
                )
                embeds = embeds.masked_scatter(image_mask, values)
                image_grid = media["image_grid_thw"].repeat(rows, 1).to(self.device)
            values = torch.cat(features["video"] * rows).to(embeds.device, embeds.dtype)
            _, video_mask = backbone.get_placeholder_mask(
                ids, inputs_embeds=embeds, video_features=values
            )
            embeds = embeds.masked_scatter(video_mask, values)
            positions = backbone.compute_3d_position_ids(
                input_ids=ids,
                image_grid_thw=image_grid,
                video_grid_thw=media["video_grid_thw"].repeat(rows, 1).to(self.device),
                inputs_embeds=embeds,
                attention_mask=mask,
                past_key_values=None,
                mm_token_type_ids=types,
            )
            fused = self.fast.fused if self.fast is not None else None
            if fused is not None:
                rotary = self.fast.lm.rotary_emb(embeds, positions)
                hidden = fused.forward(
                    embeds,
                    mask.bool()[:, None, None, :],
                    mask.reshape(-1) if padded else None,
                    rotary,
                )[:, -1]
            elif self.encoder is not None:
                hidden = self.encoder(embeds, mask, positions, blocks=blocks)[:, -1]
            else:
                hidden = backbone.language_model(
                    input_ids=None,
                    position_ids=positions,
                    attention_mask=mask,
                    past_key_values=None,
                    inputs_embeds=embeds,
                    use_cache=False,
                ).last_hidden_state[:, -1]
            if self.readout_dtype == "float32":
                logits = hidden.float() @ self.readout.T
            else:
                logits = torch.nn.functional.linear(hidden, self.readout).float()
            limit = torch.as_tensor(list(counts), device=self.device)[:, None]
            invalid = torch.arange(MAX_OPTIONS, device=self.device)[None] >= limit
            return logits.masked_fill(invalid, float("-inf"))

    def _run_videos(self, prepared: Prepared) -> tuple[dict[str, list[float]], int]:
        keys = prepared.runnable
        out: dict[str, list[float]] = {}
        if not keys:
            return out, 0
        features = self.media_features(prepared.media)

        def probabilities(texts, counts, widths, states):
            probs = (
                (
                    self.video_logits(
                        texts, prepared, features, counts, max(widths), widths, states
                    )
                    / self.temperature
                )
                .softmax(-1)
                .cpu()
                .tolist()
            )
            return [p[:c] for p, c in zip(probs, counts)]

        for start in range(0, len(keys), self.batch_size):
            chunk = keys[start : start + self.batch_size]
            extra = [k for k in chunk if k in prepared.reversed_texts]
            texts = [prepared.texts[k] for k in chunk] + [
                prepared.reversed_texts[k] for k in extra
            ]
            counts = [len(prepared.questions[k].keys) for k in chunk + extra]
            widths = [prepared.lengths[k] for k in chunk] + [
                prepared.reversed_lengths[k] for k in extra
            ]
            states = self.batch_states(prepared, chunk, extra)
            n = len(chunk)
            if extra and len(widths) * max(widths) > MERGE_TOKENS:
                probs = probabilities(
                    texts[:n], counts[:n], widths[:n], states and states[:n]
                ) + probabilities(
                    texts[n:], counts[n:], widths[n:], states and states[n:]
                )
            else:
                probs = probabilities(texts, counts, widths, states)
            out.update(zip(chunk, probs))
            for key, backward in zip(extra, probs[n:]):
                out[key] = average_orders(out[key], backward)
        return out, sum(prepared.lengths[k] for k in keys) + sum(
            prepared.reversed_lengths.values()
        )

    def _image_state(self, prepared: Prepared, text: str, state: int):
        """(expanded token ids of one image prompt, its state block's embeddings [1, S, D] and positions [3, 1, S])."""
        encoded = self.processor(text=[text], images=prepared.images, return_tensors="pt")
        embeds, mrope = self.image_embeddings({k: v.to(self.device) for k, v in encoded.items()})
        return encoded["input_ids"][0].tolist(), embeds[:, :state], mrope[:, :, :state]

    def _video_state(self, prepared: Prepared, text: str, state: int):
        """``_image_state`` for a video prompt: the request's images and videos expanded, their features in place."""
        torch = self.torch
        processor, backbone, media = self.processor, self.backbone, prepared.media
        image_texts = [processor.replace_image_token(media, image_idx=i) for i in range(len(prepared.images))]
        expanded = processor.get_text_with_replacements([text], image_texts, prepared.video_texts)[0][0]
        encoded = processor.tokenizer([expanded], return_token_type_ids=False, return_tensors="pt")
        ids = encoded["input_ids"].to(self.device)
        types = torch.as_tensor(processor.create_mm_token_type_ids(encoded["input_ids"]), device=self.device)
        features = self.media_features(media)
        embeds = backbone.get_input_embeddings()(ids)
        image_grid = None
        if "image" in features:
            values = torch.cat(features["image"]).to(embeds.device, embeds.dtype)
            image_mask, _ = backbone.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=values)
            embeds = embeds.masked_scatter(image_mask, values)
            image_grid = media["image_grid_thw"].to(self.device)
        values = torch.cat(features["video"]).to(embeds.device, embeds.dtype)
        _, video_mask = backbone.get_placeholder_mask(ids, inputs_embeds=embeds, video_features=values)
        embeds = embeds.masked_scatter(video_mask, values)
        positions = backbone.compute_3d_position_ids(
            input_ids=ids,
            image_grid_thw=image_grid,
            video_grid_thw=media["video_grid_thw"].to(self.device),
            inputs_embeds=embeds,
            attention_mask=encoded["attention_mask"].to(self.device),
            past_key_values=None,
            mm_token_type_ids=types,
        )
        return encoded["input_ids"][0].tolist(), embeds[:, :state], positions[:, :, :state]

    def _question_ids(self, prepared: Prepared, key: str, reverse: bool, state: int) -> list[int]:
        """The question block's token ids of an image or video prompt: the prompt text's tokens after its state
        block (``d3_format.state_token_count``; every image and video placeholder lies before it)."""
        text = prepared.reversed_texts[key] if reverse else prepared.texts[key]
        encoded = self.processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"][state_token_count(encoded["offset_mapping"], state_end_char(text, prepared.state)) :]
        planned = prepared.reversed_lengths[key] if reverse else prepared.lengths[key]
        if state + len(ids) != planned:
            raise RuntimeError(f"a {planned}-token prompt splits into a {state}-token state and {len(ids)} more tokens")
        return ids

    def _run_state_once(self, prepared: Prepared) -> tuple[dict[str, list[float]], int]:
        """``run`` for ``prefix_bidir``: the state block once (``StateOnce.state``), then the question blocks, the
        reversed-option prompts included, in batches of up to ``STATE_ONCE_ROWS`` rows. Usage counts the state block
        once plus each question block's tokens."""
        torch = self.torch
        media = bool(prepared.images or prepared.videos)
        reversed_ = prepared.reversed_texts if media else prepared.reversed_sequences
        blocks = [(k, False) for k in prepared.runnable]
        blocks += [(k, True) for k in prepared.runnable if k in reversed_]
        sizes = {prepared.reversed_states[k] if rev else prepared.states[k] for k, rev in blocks}
        if len(sizes) != 1:
            raise RuntimeError(f"the questions of a request have different state blocks: {sorted(sizes)}")
        state = sizes.pop()
        embed = self.backbone.get_input_embeddings()
        with torch.inference_mode(), sdpa_backends(self.device):
            if media:
                first = prepared.texts[blocks[0][0]]
                make = self._video_state if prepared.videos else self._image_state
                row, state_embeds, positions = make(prepared, first, state)
                pieces = [self._question_ids(prepared, k, rev, state) for k, rev in blocks]
                if row[state:] != pieces[0]:
                    raise RuntimeError("the expanded prompt does not end with its question block")
                media_ids = {self.processor.tokenizer.convert_tokens_to_ids(t) for t in (IMAGE_TOKEN, VIDEO_TOKEN)}
                if any(media_ids.intersection(p) for p in pieces):
                    raise RuntimeError("an image or video token falls outside the state block")
            else:
                first = list(prepared.sequences[blocks[0][0]][:state])
                pieces = []
                for k, rev in blocks:
                    sequence = list(prepared.reversed_sequences[k] if rev else prepared.sequences[k])
                    if sequence[:state] != first:
                        raise RuntimeError("the questions of a request do not share the state block's tokens")
                    pieces.append(sequence[state:])
                state_embeds = embed(torch.as_tensor([first], device=self.device))
                positions = torch.arange(state, device=self.device).view(1, 1, -1).expand(3, 1, -1)
            last = positions[:, 0, -1]
            if not bool((last == last[0]).all()):
                raise RuntimeError("the state block does not end with a text token")
            rotary = self.backbone.language_model.rotary_emb
            cache = self.state_once.state(state_embeds, rotary(state_embeds, positions))
            probs: list[list[float]] = [[] for _ in pieces]
            for batch in state_once_batches([len(p) for p in pieces], state):
                rows = [pieces[i] for i in batch]
                width = max(map(len, rows))
                ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
                for r, piece in enumerate(rows):
                    ids[r, : len(piece)] = torch.as_tensor(piece, dtype=torch.long)
                embeds = embed(ids.to(self.device))
                steps = (last[0] + 1 + torch.arange(width, device=self.device)).view(1, 1, -1)
                hidden = self.state_once.questions(
                    cache, embeds, [len(p) for p in rows], rotary(embeds, steps.expand(3, len(rows), -1))
                )
                if self.readout_dtype == "float32":
                    logits = hidden.float() @ self.readout.T
                else:
                    logits = torch.nn.functional.linear(hidden, self.readout).float()
                counts = [len(prepared.questions[blocks[i][0]].keys) for i in batch]
                limit = torch.as_tensor(counts, device=self.device)[:, None]
                logits = logits.masked_fill(torch.arange(MAX_OPTIONS, device=self.device)[None] >= limit, float("-inf"))
                values = (logits / self.temperature).softmax(-1).cpu().tolist()
                for i, p, c in zip(batch, values, counts):
                    probs[i] = p[:c]
        out: dict[str, list[float]] = {}
        for (key, rev), p in zip(blocks, probs):
            if not rev:
                out[key] = p
        for (key, rev), p in zip(blocks, probs):
            if rev:
                out[key] = average_orders(out[key], p)
        return out, state + sum(len(p) for p in pieces)

    def temperature_for(self, n_questions: int) -> float:
        """The temperature of a request with ``n_questions`` choice, yes / no and score questions: the
        ``temperature_by_questions`` entry (``temperature_table``), or ``temperature`` without that table."""
        if self.temperature_by_questions is None:
            return self.temperature
        bounds, temps = self.temperature_by_questions
        return next((t for b, t in zip(bounds, temps) if n_questions <= b), temps[-1])

    @contextlib.contextmanager
    def request_temperature(self, n_questions: int | None):
        """``temperature`` is ``temperature_for(n_questions)`` while one request's questions run (None: unchanged);
        requests run one at a time per loaded model."""
        base = self.temperature
        if n_questions is not None:
            self.temperature = self.temperature_for(n_questions)
        try:
            yield
        finally:
            self.temperature = base

    def run(self, prepared: Prepared) -> tuple[dict[str, list[float]], int]:
        """Probabilities per runnable question (request order, ``batch_size`` per pass) and the input tokens, at the
        request's temperature."""
        with self.request_temperature(len(prepared.questions)):
            return self._run(prepared)

    def _run(self, prepared: Prepared) -> tuple[dict[str, list[float]], int]:
        if self.state_once is not None and prepared.runnable:
            return self._run_state_once(prepared)
        if prepared.videos:
            return self._run_videos(prepared)
        if prepared.images:
            return self._run_images(prepared)
        keys = prepared.runnable
        out: dict[str, list[float]] = {}
        for start in range(0, len(keys), self.batch_size):
            chunk = keys[start : start + self.batch_size]
            extra = [k for k in chunk if k in prepared.reversed_sequences]
            sequences = [prepared.sequences[k] for k in chunk] + [
                prepared.reversed_sequences[k] for k in extra
            ]
            counts = [len(prepared.questions[k].keys) for k in chunk + extra]
            states = self.batch_states(prepared, chunk, extra)
            n = len(chunk)
            if extra and len(sequences) * max(map(len, sequences)) > MERGE_TOKENS:
                probs = self.probabilities(
                    sequences[:n], counts[:n], states and states[:n]
                ) + self.probabilities(sequences[n:], counts[n:], states and states[n:])
            else:
                probs = self.probabilities(sequences, counts, states)
            out.update(zip(chunk, probs))
            for key, backward in zip(extra, probs[n:]):
                out[key] = average_orders(out[key], backward)
        return out, sum(len(prepared.sequences[k]) for k in keys) + sum(
            len(s) for s in prepared.reversed_sequences.values()
        )

    def respond(
        self,
        prepared: Prepared,
        probabilities: Mapping[str, Sequence[float]],
        tokens: int,
    ) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        for key in prepared.keys:
            if key in prepared.errors:
                answers[key] = prepared.errors[key]
                continue
            question = prepared.questions[key]
            try:
                answers[key] = product_answer(question, probabilities[key])
            except ValueError as exc:
                answers[key] = {
                    "type": question.kind,
                    "error": "invalid_model_output",
                    "message": str(exc),
                }
        return {
            "model": self.model_name,
            "answers": with_reasons(answers),
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }

    def question_cap(self, long_context: bool) -> int | None:
        """The per-question token cap: the validated length by default, the package's own cap (or the documented
        long-context limit) with ``long_context``; a package without ``validated_length`` keeps its own cap."""
        if self.validated_length is None:
            return self.context_limit
        if long_context:
            return self.context_limit or LONG_CONTEXT_TOKENS
        return min(self.validated_length, self.context_limit or self.validated_length)

    def system_one(
        self,
        *,
        state: Any,
        questions: Mapping[str, Any],
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
        assets: Mapping[str, Any] | None = None,
        long_context: bool = False,
    ) -> dict[str, Any]:
        """Typed answers about one state (see ``_system_one``). ``long_context``: lift the per-question cap from
        the validated length (``decision_config`` ``validated_length``, 32,768 tokens for this package) to the package's
        long-context limit; slower and less accurate, as the model card says."""
        self.max_length = self.question_cap(long_context)
        try:
            return self._system_one(state, questions, images, videos, assets)
        finally:
            self.max_length = self.question_cap(False)

    def _system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
        assets: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Typed Choice / Noul / Score / Box / track answers about one state: ``{"model", "answers", "usage"}``.

        ``assets``: System One references (reference string -> image) used in the state (hoisted after ``images``)
        or inside a question's instructions or options (that question only). Box: ``{"type": "box",
        "instructions"}`` (``"at"``: seconds into the one video); track: ``{"type": "track", "instructions",
        "rate"?}`` over the whole video, 1 or 2 updates per second.

        ``images``: any number of images (PIL images, paths, http(s) or data URLs) that every question sees.
        ``videos``: any number of videos (paths, http(s) or data URLs, frame arrays) that every question sees.
        A question that cannot be answered gets ``{"type", "error", "reason", "message"}`` with ``error`` one of
        ``invalid_question``, ``max_length_exceeded`` (never truncated) or ``invalid_model_output`` and a fixed
        ``reason`` (``box_unsupported`` / ``track_unsupported`` / ``question_images_unsupported`` where the size
        did not pass that feature's bar); the other questions of the request are still answered.
        """
        if assets or any(
            isinstance(q, Mapping) and q.get("type") in ("box", "track") for q in questions.values()
        ):
            return self._system_one_composed(state, questions, list(images or []), list(videos or []), dict(assets or {}))
        prepared = self.prepare(state, questions, images, videos)
        probabilities, tokens = self.run(prepared)
        return self.respond(prepared, probabilities, tokens)

    # ------------------------------------------------------------------ composed images, Box and video Box

    def _system_one_composed(
        self,
        state: Any,
        questions: Mapping[str, Any],
        images: list[Any],
        videos: list[Any],
        assets: dict[str, Any],
    ) -> dict[str, Any]:
        """A request with ``assets`` or Box / track questions (``d3_compose``): every question as a row of its own
        under the block mask (state images and videos in its state block, its own images in its block), so its
        answer never depends on the other questions. A video Box reads the clip in windows (``d3_compose``).
        Usage counts each state once (the request's, and each window's) plus each question block."""
        dc = _d3_compose()

        if not self.blocked or self.encoder is None:
            raise ValueError("composed images and Box need a prefix_bidir package")
        refs = dc.check_references(list(assets))
        hoisted, uses = dc.hoist_state(state, refs, len(images))
        front = self.load_images(images + [assets[ref] for ref in uses]) if (images or uses) else []
        clips = [load_video(v) for v in videos] if videos else []
        answers: dict[str, Any] = {}
        states_counted: set[Any] = set()
        tokens = 0
        windows: dict[int, list[Any]] = {}
        decisions = sum(isinstance(q, Mapping) and q.get("type") in DECISION_KINDS for q in questions.values())
        for key, question in questions.items():
            kind = question.get("type") if isinstance(question, Mapping) else None
            refused = self.refusal(kind, question, refs)
            if refused is not None:
                answers[key] = refused
                continue
            try:
                if kind == "track" or (kind == "box" and question.get("at") is not None):
                    answer, used = self._video_box(key, question, hoisted, front, videos, refs, windows)
                    for state_key, state_tokens, block in used:
                        if state_key not in states_counted:
                            states_counted.add(state_key)
                            tokens += state_tokens
                        tokens += block
                    answers[key] = answer
                    continue
                with self.request_temperature(decisions if kind in DECISION_KINDS else None):
                    answer, state_tokens, block = self._composed_question(
                        question, hoisted, front, clips, refs, assets
                    )
                if "request" not in states_counted:
                    states_counted.add("request")
                    tokens += state_tokens
                tokens += block
                answers[key] = answer
            except MaxLengthExceeded as exc:
                answers[key] = {"type": kind, "error": "max_length_exceeded", "message": str(exc)}
            except ValueError as exc:
                answers[key] = {"type": kind, "error": "invalid_question", "message": str(exc)}
                if isinstance(exc, QuestionRefused):
                    answers[key]["reason"] = exc.reason
        return {"model": self.model_name, "answers": with_reasons(answers),
                "usage": {"input_tokens": tokens, "output_tokens": 0}}

    def refusal(self, kind: Any, question: Any, refs: Sequence[str]) -> dict[str, Any] | None:
        """The per-question refusal of a feature this size does not answer (its card's ``systemone`` block):
        Box / track questions, and images inside a question's instructions or options."""
        if kind in ("box", "track") and kind not in self.question_types:
            return {"type": kind, "error": "invalid_question", "reason": f"{kind}_unsupported",
                    "message": f"{self.model_name} does not answer {kind} questions"}
        if not isinstance(question, Mapping) or not refs:
            return None
        places = {"instructions": question.get("instructions"), "criteria": question.get("criteria")}
        for place, value in places.items():
            if place not in self.image_placements and mentions(value, refs):
                return {"type": kind, "error": "invalid_question", "reason": "question_images_unsupported",
                        "message": f"{self.model_name} reads images in the state only, not in a question's {place}"}
        return None

    def _composed_question(self, question, hoisted, front, clips, refs, assets):
        dc = _d3_compose()

        kind = question.get("type")
        if kind == "box":
            if set(question) - {"type", "instructions"} or not isinstance(question.get("instructions"), str):
                raise ValueError("a box question takes type and instructions")
            normalized = None
        else:
            normalized = normalize_question(question)
            if normalized.kind == "score":
                raise ValueError("score questions take no images here")
        parts = dc.user_parts(hoisted, question, self.codes, len(front), len(clips), refs)
        presence = False
        text = dc.render(self.processor, parts, dc.suffix(presence) if kind == "box" else "")
        own = [assets[item] for k, item in parts if k == dc.IMAGE and isinstance(item, str)]
        own_images = self.load_images(own) if own else []
        if kind == "box":
            offsets = dc.readout_offsets(self.processor.tokenizer, text, presence)
            counts = [dc.BINS] * len(offsets)
        else:
            offsets, counts = [0], [len(normalized.keys)]
        values, state_tokens, block = self._composed_row(
            text, hoisted, list(front) + own_images, len(front), clips, offsets, counts
        )
        if kind == "box":
            box = dc.decode(values)
            return {
                "type": "box",
                "box": box,
                "confidence": dc.calibrate(dc.confidence(values, box), self.box_calibration),
            }, state_tokens, block
        return product_answer(normalized, values[0]), state_tokens, block

    def _composed_row(self, text, hoisted, row_images, n_front, clips, offsets, counts):
        """Probabilities at the row's readouts (offsets back from its last token), with its state block's and its
        question block's token counts."""
        try:
            from .d3_format import state_end_char
        except ImportError:
            from d3_format import state_end_char

        torch = self.torch
        kwargs = {}
        if clips:
            kwargs = {"videos": [c.frames for c in clips], "video_metadata": [c.metadata() for c in clips]}
        encoded = self.processor(text=[text], images=list(row_images) or None, return_tensors="pt", **kwargs)
        ids = encoded["input_ids"]
        width = int(ids.shape[1])
        if self.max_length is not None and width > self.max_length:
            raise MaxLengthExceeded(
                f"the question prompt has {width} tokens, over the maximum context length of {self.max_length} "
                "tokens; nothing was truncated"
            )
        plain = self.processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        state = state_token_count(plain["offset_mapping"], state_end_char(text, hoisted))
        merge = self.processor.image_processor.merge_size ** 2
        grids = encoded.get("image_grid_thw")
        own = 0
        if grids is not None:
            own = sum(int(g.prod()) // merge - 1 for g in grids[n_front:])
        state += (width - len(plain["input_ids"])) - own
        inputs = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.inference_mode(), sdpa_backends(self.device):
            embeds, mrope = self._media_embeddings(inputs)
            blocks = self.blocks(width, [width], [state])
            hidden = self.encoder(embeds, inputs["attention_mask"], mrope, blocks=blocks)[0]
            picked = hidden[torch.as_tensor([width - 1 - o for o in offsets], device=hidden.device)]
            if self.readout_dtype == "float32":
                logits = picked.float() @ self.readout.T
            else:
                logits = torch.nn.functional.linear(picked, self.readout).float()
            limit = torch.as_tensor(counts, device=self.device)[:, None]
            logits = logits.masked_fill(torch.arange(MAX_OPTIONS, device=self.device)[None] >= limit, float("-inf"))
            values = (logits / self.temperature).softmax(-1).cpu().tolist()
        return [v[:c] for v, c in zip(values, counts)], state, width - state

    def _media_embeddings(self, inputs):
        """Input embeddings with every image's and video's features in place, and the 3-D rotary positions."""
        torch = self.torch
        backbone = self.backbone
        ids = inputs["input_ids"]
        embeds = backbone.get_input_embeddings()(ids)
        image_grid = inputs.get("image_grid_thw")
        video_grid = inputs.get("video_grid_thw")
        if inputs.get("pixel_values") is not None:
            pooled = torch.cat(
                backbone.get_image_features(inputs["pixel_values"], image_grid, return_dict=True).pooler_output
            ).to(embeds.device, embeds.dtype)
            slots, _ = backbone.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=pooled)
            embeds = embeds.masked_scatter(slots, pooled)
        if inputs.get("pixel_values_videos") is not None:
            pooled = torch.cat(
                backbone.get_video_features(inputs["pixel_values_videos"], video_grid, return_dict=True).pooler_output
            ).to(embeds.device, embeds.dtype)
            _, slots = backbone.get_placeholder_mask(ids, inputs_embeds=embeds, video_features=pooled)
            embeds = embeds.masked_scatter(slots, pooled)
        types = inputs.get("mm_token_type_ids")
        if types is None:
            types = torch.as_tensor(self.processor.create_mm_token_type_ids(ids.cpu()), device=ids.device)
        mrope = backbone.compute_3d_position_ids(
            input_ids=ids,
            image_grid_thw=image_grid,
            video_grid_thw=video_grid,
            inputs_embeds=embeds,
            attention_mask=inputs["attention_mask"],
            past_key_values=None,
            mm_token_type_ids=types,
        )
        return embeds, mrope

    def _video_box(self, key, question, hoisted, front, videos, refs, windows):
        """A box at a time (the frame group nearest ``at``) or a track (every frame group, ``rate`` per second)."""
        dc = _d3_compose()

        kind = question["type"]
        allowed = {"type", "instructions", "at"} if kind == "box" else {"type", "instructions", "rate"}
        if set(question) - allowed or not isinstance(question.get("instructions"), str):
            raise ValueError(f"a {kind} question takes {sorted(allowed)}")
        if not videos:
            raise QuestionRefused("video_needed", f"a {kind} question needs a video")
        if len(videos) > 1:
            raise QuestionRefused("video_target_ambiguous", f"a video {kind} needs exactly one video, got {len(videos)}")
        rate = int(question.get("rate", 1)) if kind == "track" else 1
        if rate not in (1, 2):
            raise ValueError("rate must be 1 or 2 updates per second")
        if rate not in windows:
            windows[rate] = load_video_windows(videos[0], rate)
        clips = windows[rate]
        moments = [(w, t) for w, clip in enumerate(clips) for t in dc.group_times(clip.indices, clip.fps)]
        if kind == "box":
            at = float(question["at"])
            if at < 0 or at > moments[-1][1] + 1.0:
                raise ValueError(f"at {at} s is outside the clip")
            moments = [min(moments, key=lambda m: abs(m[1] - at))]
        presence = kind == "track"
        entries, used = [], []
        for w, t in moments:
            moment = {"type": kind, "instructions": question["instructions"], "at": t}
            parts = dc.user_parts(hoisted, moment, self.codes, len(front), 1, refs)
            text = dc.render(self.processor, parts, dc.suffix(presence))
            if any(k == dc.IMAGE and isinstance(item, str) for k, item in parts):
                raise ValueError("a video Box reads the video; images go in the state")
            offsets = dc.readout_offsets(self.processor.tokenizer, text, presence)
            counts = [2 if name == dc.PRESENCE else dc.BINS for name in dc.markers(presence)]
            values, state_tokens, block = self._composed_row(text, hoisted, front, len(front), [clips[w]], offsets, counts)
            edges = values[-4:]
            box = dc.decode(edges)
            conf = dc.calibrate(dc.confidence(edges, box), self.box_calibration)
            used.append((("window", rate, w), state_tokens, block))
            entry = {"time": round(t, 3), "box": box, "confidence": conf}
            if presence:
                entry["visible"] = float(values[0][1] / sum(values[0]))
            entries.append(entry)
        if kind == "box":
            entry = entries[0]
            return {"type": "box", "box": entry["box"], "confidence": entry["confidence"], "time": entry["time"]}, used
        return {"type": "track", "track": sorted(entries, key=lambda e: e["time"])}, used

    def warmup(
        self,
        lengths: Sequence[int] = (37, 64, 320, 333, 1000, 1024),
        images: bool = True,
    ) -> float:
        """Compile and autotune the kernels for every batch size up to ``batch_size`` (twice that with
        ``permutation_average``) before serving.

        The Gated DeltaNet kernels take the batch size as a compile-time constant, and Triton specializes their
        length and chunk-count arguments on being 1 or a multiple of 16; these lengths cover every combination
        (chunks of 64 tokens). A fresh process otherwise pays several seconds on the first request of each new
        class. With ``images`` (and a vision tower) three image requests (one small image, one 1.6 MP image,
        four 1.6 MP images) also warm the vision tower. Answers are unchanged.
        """
        started = time.perf_counter()
        if self.fast is not None:
            self.fast.capture_all()
        widest = self.batch_size * (2 if self.permutation_average else 1)
        for size in range(1, widest + 1):
            for length in lengths:
                sequences = [
                    [
                        self.token_ids[(i + j) % len(self.token_ids)]
                        for j in range(max(8, length - 7 * i))
                    ]
                    for i in range(size)
                ]
                states = [len(s) // 2 for s in sequences] if self.blocked else None
                self.probabilities(sequences, [2] * size, states)
        if images and self.image_unavailable is None:
            from PIL import Image

            small = Image.new("RGB", (448, 336), (128, 128, 128))
            large = Image.new("RGB", (1280, 1280), (96, 160, 224))
            for batch in ([small], [large], [large] * 4):
                self.system_one(
                    state="warm-up", questions={"q": {"type": "noul"}}, images=batch
                )
        self.synchronize()
        return time.perf_counter() - started

    # ------------------------------------------------------------------ device and records

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def to(self, device: str) -> D3:
        target = self.torch.device(device)
        self.backbone.to(target)
        self.readout = self.readout.to(target)
        self.device = target
        if self.fast is not None:
            self.fast, self.fast_skipped = None, "moved after loading"
        return self

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters()) + self.readout.numel()

    def provenance(self) -> dict[str, Any]:
        files = {
            name: sha256_file(self.root / name)
            for name in (
                "decision_config.json",
                "readout.safetensors",
                "config.json",
                MANIFEST,
            )
            if (self.root / name).is_file()
        }
        identity = (self.manifest or {}).get("identity", {})
        record = {
            "kind": "d3-code-readout",
            "runtime": RUNTIME,
            "model_name": self.model_name,
            "repo_id": (self.manifest or {}).get("repo_id"),
            "model_sha256": identity.get("model_sha256"),
            "format_id": self.config.get("format_id"),
            "prompt": self.prompt,
            "attention_mode": self.attention_mode,
            "pooling": "last",
            "temperature": self.temperature,
            "max_length": self.max_length,
            "readout_dtype": self.readout_dtype,
            "backbone_dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "batch_size": self.batch_size,
            "files_sha256": files,
            "kernels": self.kernels,
            "policy": "One forward pass per question; options under single-token answer codes; last-token readout "
            "over the question's codes only; argmax choice; no truncation (over-limit questions are "
            "refused); no option filtering; one fixed prompt for every request.",
            "images": self.image_contract(),
            "videos": self.video_contract(),
        }
        if self.temperature_by_questions is not None:
            bounds, temps = self.temperature_by_questions
            record["temperature_by_questions"] = {"max_questions": list(bounds), "temperature": list(temps)}
        if self.permutation_average:
            record["permutation_average"] = True
            record["policy"] += (
                " Choice questions with two or more options are also scored with their options in reversed "
                "order, in the same forward passes, and answered with the per-option mean of both distributions."
            )
        return record

    def image_contract(self) -> dict[str, Any]:
        """How image inputs are read (or why they are not available)."""
        contract: dict[str, Any] = {
            "supported": self.image_unavailable is None,
            "min_pixels": IMAGE_MIN_PIXELS,
            "max_pixels": IMAGE_MAX_PIXELS,
            "placement": "before the text of the user turn, one placeholder per image, request order",
            "patch_embedding": "matrix product (equal to the Conv3d with kernel = stride)",
        }
        if self.processor is not None:
            contract["image_processor"] = type(self.processor.image_processor).__name__
            try:
                import torchvision

                contract["torchvision"] = torchvision.__version__
            except Exception:  # noqa: BLE001
                contract["torchvision"] = None
        if self.image_unavailable is not None:
            contract["unavailable"] = self.image_unavailable
        return contract

    def video_contract(self) -> dict[str, Any]:
        """How video inputs are read (or why they are not available)."""
        contract: dict[str, Any] = {
            "supported": self.video_unavailable is None,
            "fps": VIDEO_FPS,
            "min_frames": VIDEO_MIN_FRAMES,
            "max_frames": VIDEO_MAX_FRAMES,
            "max_pixels_per_frame": VIDEO_MAX_PIXELS,
            "max_tokens_per_request": VIDEO_MAX_TOKENS,
            "placement": "after the images, before the text of the user turn, one placeholder per video, "
            "request order; every two frames are one token group after their timestamp",
            "sampling": "source frames spread evenly over the whole video (frame arrays: frames at 2 per second)",
        }
        if self.video_unavailable is None:
            contract["video_processor"] = type(self.processor.video_processor).__name__
            try:
                import cv2

                contract["opencv"] = cv2.__version__
            except Exception:  # noqa: BLE001
                contract["opencv"] = None
        else:
            contract["unavailable"] = self.video_unavailable
        return contract

    def runtime_info(self) -> dict[str, Any]:
        import transformers

        torch = self.torch
        info = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(self.device),
            "cuda": torch.version.cuda,
            "hip": getattr(torch.version, "hip", None),
            "loaded_seconds": round(self.loaded_seconds, 2),
        }
        if self.device.type == "cuda":
            info["gpu"] = torch.cuda.get_device_name(self.device)
        info["fast_path"] = self.fast_report()
        return info

    def fast_report(self) -> dict[str, Any]:
        """What the ROCm fast path (``d3_fast.py``) does in this process, or why it is off."""
        if self.fast is None:
            return {"active": False, "reason": self.fast_skipped}
        return {"active": True, **self.fast.report()}
