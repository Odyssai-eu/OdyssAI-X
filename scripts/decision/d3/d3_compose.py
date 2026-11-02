"""Composed images (System One ``assets``), Box answers and video Box windows for the d3 package runtime.

The same prompt the reference renderer and the vllm-sr runtime build, byte for byte:

- State references are hoisted: each use becomes an image after the request's ``images`` (in reading order), and its
  text the label ``[image k]``. A reference in a question's instructions or options stays inline: its image sits in
  that question's block only. Only string values are searched (object keys are never replaced); a value equal to a
  reference is replaced whole, otherwise references are replaced left to right without overlap, the longest first
  at one position; a whole-value image inside a JSON value replaces the JSON string, quotes included.
- A Box question's block ends with four edge markers after the generation prompt (``top:``, ``bottom:``, ``left:``,
  ``right:``; a track moment adds ``visible:`` first); each marker's colon is read by the code readout, an edge over
  the first ``BINS`` codes (bin k's centre ``(k + 0.5) / BINS``), ``visible`` over the first two codes.
- A video Box reads the clip in windows of ``2 x rate`` frames per second, at most 32 frames each (16-s windows at
  2 fps, 8-s windows at 4 fps), each window a state of its own with the clip's own timestamps.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from .d3_format import NOUL_DESCRIPTIONS, QUESTION_MARK, STATE_MARK, SYSTEM_PROMPT, describe
except ImportError:  # a plain module (tests with the package dir on sys.path)
    from d3_format import NOUL_DESCRIPTIONS, QUESTION_MARK, STATE_MARK, SYSTEM_PROMPT, describe

TEXT, IMAGE, VIDEO = "text", "image", "video"
HOISTED_LABEL = "[image {k}]"
MAX_REFERENCE_BYTES = 1024
MAX_REFERENCES_BYTES = 65536
DEFAULT_INSTRUCTIONS = "Choose the best matching option."
REPLY = "Reply with only the code of the best option."
BINS = 128
EDGES = ("top", "bottom", "left", "right")
PRESENCE = "visible"
IOU_HIT = 0.5
WINDOW_FRAMES = 32
_OPEN, _CLOSE, _WHOLE = "\ue000", "\ue001", "\ue002"


def check_references(references: Sequence[str]) -> list[str]:
    refs, total = list(references), 0
    for ref in refs:
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError("an asset reference must be a non-blank string")
        size = len(ref.encode("utf-8"))
        if size > MAX_REFERENCE_BYTES:
            raise ValueError(f"an asset reference has {size} bytes, over {MAX_REFERENCE_BYTES}")
        if any(mark in ref for mark in (_OPEN, _CLOSE, _WHOLE)):
            raise ValueError("an asset reference contains a reserved character")
        total += size
    if total > MAX_REFERENCES_BYTES:
        raise ValueError(f"asset references have {total} bytes, over {MAX_REFERENCES_BYTES}")
    return sorted(refs, key=lambda ref: (-len(ref), ref))


def split_string(text: str, refs: Sequence[str]) -> list[tuple[str, Any]]:
    if text in refs:
        return [(IMAGE, text)]
    parts, at, start = [], 0, 0
    while at < len(text):
        hit = next((ref for ref in refs if text.startswith(ref, at)), None)
        if hit is None:
            at += 1
            continue
        if at > start:
            parts.append((TEXT, text[start:at]))
        parts.append((IMAGE, hit))
        at += len(hit)
        start = at
    if start < len(text):
        parts.append((TEXT, text[start:]))
    return parts


def _mark(value: Any, refs: Sequence[str], order: list[str]) -> Any:
    if isinstance(value, str):
        parts = split_string(value, refs)
        if not any(kind == IMAGE for kind, _ in parts):
            return value
        if len(parts) == 1:
            order.append(parts[0][1])
            return f"{_WHOLE}{len(order) - 1}{_WHOLE}"
        out = []
        for kind, item in parts:
            if kind == TEXT:
                out.append(item)
            else:
                order.append(item)
                out.append(f"{_OPEN}{len(order) - 1}{_CLOSE}")
        return "".join(out)
    if isinstance(value, Mapping):
        return {key: _mark(item, refs, order) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mark(item, refs, order) for item in value]
    return value


def _unmark(text: str, order: Sequence[str]) -> list[tuple[str, Any]]:
    parts, buffer, at = [], [], 0
    while at < len(text):
        char = text[at]
        if char in (_OPEN, _WHOLE):
            close = _CLOSE if char == _OPEN else _WHOLE
            end = text.index(close, at + 1)
            index = int(text[at + 1 : end])
            at = end + 1
            if char == _WHOLE and buffer and buffer[-1].endswith('"') and text.startswith('"', at):
                buffer[-1] = buffer[-1][:-1]
                at += 1
            if buffer:
                parts.append((TEXT, "".join(buffer)))
                buffer = []
            parts.append((IMAGE, order[index]))
            continue
        buffer.append(char)
        at += 1
    if buffer:
        parts.append((TEXT, "".join(buffer)))
    return parts


def describe_parts(value: Any, refs: Sequence[str]) -> list[tuple[str, Any]]:
    if isinstance(value, str):
        return split_string(value, refs)
    order: list[str] = []
    marked = _mark(value, refs, order)
    if not order:
        return [(TEXT, describe(value))]
    return _unmark(json.dumps(marked, ensure_ascii=False), order)


def hoist_state(state: Any, refs: Sequence[str], front: int) -> tuple[Any, list[str]]:
    hoisted: list[str] = []

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            parts = split_string(value, refs)
            if not any(kind == IMAGE for kind, _ in parts):
                return value
            out = []
            for kind, item in parts:
                if kind == TEXT:
                    out.append(item)
                else:
                    hoisted.append(item)
                    out.append(HOISTED_LABEL.format(k=front + len(hoisted)))
            return "".join(out)
        if isinstance(value, Mapping):
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [walk(item) for item in value]
        return value

    return walk(state), hoisted


def _merge(parts):
    out = []
    for kind, item in parts:
        if kind == TEXT:
            if not item:
                continue
            if out and out[-1][0] == TEXT:
                out[-1] = (TEXT, out[-1][1] + item)
                continue
        out.append((kind, item))
    return out


def format_seconds(value: float) -> str:
    return f"{float(value):.1f}"


def markers(presence: bool = False) -> tuple[str, ...]:
    return ((PRESENCE,) if presence else ()) + EDGES


def suffix(presence: bool = False) -> str:
    return "\n".join(f"{name}:" for name in markers(presence))


def tail_text(codes: Sequence[str], at: float | None = None, presence: bool = False) -> str:
    when = "" if at is None else f" in the video frame at {format_seconds(at)} seconds"
    lines = [
        "",
        "",
        "Box:",
        f"Find the target{when}. Give its top, bottom, left and right edges, each as a code on a {BINS}-step "
        f"scale across the image: {codes[0]} is the top or left side, {codes[BINS - 1]} the bottom or right side.",
    ]
    if presence:
        lines.append(f"Also say whether the target is visible then: {codes[0]} no, {codes[1]} yes.")
    lines += ["", "Reply with the edge codes."]
    return "\n".join(lines)


def user_parts(
    state: Any,
    question: Mapping[str, Any],
    codes: Sequence[str],
    n_front: int,
    n_videos: int,
    refs: Sequence[str],
) -> list[tuple[str, Any]]:
    """The user turn: front images, videos, ``State:``, the hoisted state, the question block (Box: its lines)."""
    body = describe(state) if state not in (None, "") else "(empty)"
    parts: list[tuple[str, Any]] = [(IMAGE, i) for i in range(n_front)] + [(VIDEO, i) for i in range(n_videos)]
    parts.append((TEXT, STATE_MARK + body))
    parts.append((TEXT, QUESTION_MARK))
    kind = question["type"]
    parts += describe_parts(question.get("instructions") or DEFAULT_INSTRUCTIONS, refs)
    if kind in ("box", "track"):
        presence = kind == "track"
        parts.append((TEXT, tail_text(codes, question.get("at"), presence)))
        return _merge(parts)
    if kind == "choice":
        options = [
            [(TEXT, key)] if value is None else [(TEXT, f"{key}: "), *describe_parts(value, refs)]
            for key, value in question["criteria"].items()
        ]
    else:
        given = question.get("criteria") or {}
        options = [
            describe_parts(given.get("false") or NOUL_DESCRIPTIONS[0], refs),
            describe_parts(given.get("true") or NOUL_DESCRIPTIONS[1], refs),
        ]
    parts.append((TEXT, "\n\nOptions:"))
    for code, option in zip(codes, options):
        parts.append((TEXT, f"\n{code}: "))
        parts += option
    parts.append((TEXT, "\n\n" + REPLY))
    return _merge(parts)


def render(processor, parts: Sequence[tuple[str, Any]], box_suffix: str = "") -> str:
    content = [{"type": "text", "text": item} if kind == TEXT else {"type": kind} for kind, item in parts]
    text = processor.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text + box_suffix


def readout_offsets(tokenizer, text: str, presence: bool) -> list[int]:
    """Offsets of the edge markers' colon tokens from the prompt's last token."""
    tail = suffix(presence)
    if not text.endswith(tail):
        raise ValueError("a Box prompt must end with its edge markers")
    base = len(text) - len(tail)
    ends, at = [], 0
    for name in markers(presence):
        at = tail.index(f"{name}:", at) + len(name) + 1
        ends.append(base + at)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    from_end = [stop for _, stop in encoded["offset_mapping"]][::-1]
    out = []
    for end in ends:
        if end not in from_end:
            raise ValueError("an edge marker's colon is not the end of a token")
        out.append(from_end.index(end))
    return out


def expected(probabilities: Sequence[float]) -> float:
    total, count = float(sum(probabilities)), len(probabilities)
    return sum(p * (k + 0.5) / count for k, p in enumerate(probabilities)) / total


def order_edges(a: float, b: float, bins: int = BINS) -> tuple[float, float]:
    if a < b:
        return a, b
    mid, half = (a + b) / 2, 0.5 / bins
    lo, hi = max(0.0, mid - half), min(1.0, mid + half)
    if lo >= hi:
        lo, hi = (max(0.0, hi - 2 * half), hi) if hi >= 1.0 else (lo, lo + 2 * half)
    return lo, hi


def decode(edges: Sequence[Sequence[float]]) -> dict[str, float]:
    top, bottom, left, right = (expected(p) for p in edges)
    top, bottom = order_edges(top, bottom)
    left, right = order_edges(left, right)
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _r4_points(n: int):
    phi = 1.0
    for _ in range(64):
        phi = (1 + phi) ** (1 / 5)
    alphas = [(1 / phi) ** (d + 1) % 1 for d in range(4)]
    return [tuple(((0.5 + alphas[d] * (i + 1)) % 1.0) for d in range(4)) for i in range(n)]


_POINTS = _r4_points(512)


def confidence(edges: Sequence[Sequence[float]], box: Mapping[str, float]) -> float:
    import numpy as np

    probabilities = np.asarray(edges, dtype=np.float64)
    cdf = np.cumsum(probabilities / probabilities.sum(axis=1, keepdims=True), axis=1)
    bins = probabilities.shape[1]
    points = np.asarray(_POINTS, dtype=np.float64)
    draws = np.empty_like(points)
    for d in range(4):
        index = np.minimum(np.searchsorted(cdf[d], points[:, d], side="left"), bins - 1)
        draws[:, d] = (index + 0.5) / bins
    top, bottom, left, right = draws.T
    valid = (top < bottom) & (left < right)
    inter_h = np.clip(np.minimum(bottom, box["bottom"]) - np.maximum(top, box["top"]), 0.0, None)
    inter_w = np.clip(np.minimum(right, box["right"]) - np.maximum(left, box["left"]), 0.0, None)
    inter = inter_h * inter_w
    area = np.clip(bottom - top, 0.0, None) * np.clip(right - left, 0.0, None)
    reported = max(0.0, box["bottom"] - box["top"]) * max(0.0, box["right"] - box["left"])
    union = area + reported - inter
    iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
    return float(np.count_nonzero(valid & (iou >= IOU_HIT))) / len(points)


def calibrate(value: float, table: Sequence[Sequence[float]] | None) -> float:
    if not table:
        return value
    xs, ys = [float(k[0]) for k in table], [float(k[1]) for k in table]
    if value <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if value <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (value - x0) / (x1 - x0)
    return ys[-1]


def window_plan(total: int, fps: float, rate: int) -> list[list[int]]:
    """Source frames of each track window (the vllm-sr runtime's ``videos.window_plan``)."""
    step = fps / (2.0 * rate)
    count = max(1, int(total / step))
    indices = [min(total - 1, int(round(k * step))) for k in range(count)]
    windows = [indices[i : i + WINDOW_FRAMES] for i in range(0, len(indices), WINDOW_FRAMES)]
    for window in windows:
        if len(window) % 2:
            window.append(window[-1])
        while len(window) < 4:
            window.append(window[-1])
    return windows


def group_times(indices: Sequence[int], fps: float) -> list[float]:
    """The processor's timestamp of each pair of frames: the mean of its first and last frame."""
    seconds = [i / fps for i in indices]
    return [(seconds[i] + seconds[i + 1]) / 2 for i in range(0, len(seconds), 2)]
