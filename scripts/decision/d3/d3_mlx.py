"""MLX forward for the vLLM Semantic Router d3 decision models (code-readout v1). Ours.

The vendored ``d3_runtime.D3`` (byte-identical upstream) keeps everything that is not a tensor: question
validation, prompt rendering (``d3_format``), tokenization, the state block of ``prefix_bidir``, batching,
temperature (``temperature_by_questions``), permutation average and the answers. ``D3MLX`` replaces only
what needs torch: loading (no torch import) and ``probabilities`` (the text layers on MLX).

The backbone is the checkpoint converted to MLX by mlx-vlm (a ``Qwen3_5Model`` without lm_head); the
255-way ``readout.safetensors`` stays FP32 as shipped. ``attention_mode`` is upstream's (``PromptEncoder``):
``noncausal_full_attention`` gives the full-attention layers no mask (each prompt token reads the whole
prompt), ``prefix_bidir`` the block mask (the state block reads itself, the question block reads
everything); the Gated DeltaNet layers keep their causal scan in every mode. One question per forward
pass, unpadded: the same values as upstream's left-padded batch. Images and videos are refused for now.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from . import d3_runtime as R

IMAGES_UNAVAILABLE = "images and videos are not served by the MLX runtime yet"


class ExactNorm(nn.Module):
    """transformers' Qwen3_5RMSNorm: the (1 + w) scale in FP32, the output cast back to the input dtype. The
    conversion stores that scale exact in FP32 (mlx-vlm's bf16 1 + w rounds it, and the rounding moved d3-nano's
    probabilities 2.4x further from the FP32 reference than upstream's bf16 runtime)."""

    def __init__(self, scale, eps):
        super().__init__()
        self.weight, self.eps = scale, eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps).astype(x.dtype)


def exact_norms(text_model) -> int:
    """Swap the (1 + w) RMSNorms of the text layers for ``ExactNorm``; their scales must be FP32 (convert_d3.py)."""
    swaps = [(layer, name) for layer in text_model.layers for name in ("input_layernorm", "post_attention_layernorm")]
    swaps += [(layer.self_attn, name) for layer in text_model.layers if not layer.is_linear for name in ("q_norm", "k_norm")]
    swaps.append((text_model, "norm"))
    for parent, name in swaps:
        norm = getattr(parent, name)
        if norm.weight.dtype != mx.float32:
            raise RuntimeError(f"norm scale {name} is {norm.weight.dtype}, expected the FP32 scales of convert_d3.py")
        setattr(parent, name, ExactNorm(norm.weight, norm.eps))
    return len(swaps)


def load_backbone(root: Path):
    """The converted backbone: every saved tensor loaded, no lm_head (d3 has none), nothing else missing."""
    from mlx_vlm.utils import load_model

    model = load_model(root, lazy=True, strict=False)
    model.language_model.pop("lm_head", None)
    index = root / "model.safetensors.index.json"
    saved = set(json.loads(index.read_text())["weight_map"]) if index.exists() else {
        k for f in root.glob("model*.safetensors") for k in mx.load(str(f))}
    params = {k for k, _ in tree_flatten(model.parameters())}
    if params != saved:
        raise RuntimeError(f"{root.name}: weights do not match the backbone "
                           f"(missing {sorted(params - saved)[:5]}, unexpected {sorted(saved - params)[:5]})")
    mx.eval(model.parameters())
    return model


class D3MLX(R.D3):
    """``D3`` with an MLX backbone: same requests, same answers, no torch."""

    def __init__(self, root: str | Path, *, batch_size: int = R.DEFAULT_BATCH_SIZE, model_name: str | None = None):
        from transformers import AutoTokenizer

        self.root = Path(root)
        self.manifest = None
        config = json.loads((self.root / "decision_config.json").read_text(encoding="utf-8"))
        if config.get("format_version") != R.FORMAT_VERSION:
            raise ValueError(f"decision_config.json format_version {config.get('format_version')!r}, "
                             f"this runtime reads {R.FORMAT_VERSION}")
        self.config = config
        self.backbone_type = R.backbone_type(self.root)
        if self.backbone_type != "qwen3_5":
            raise ValueError(f"the MLX runtime reads qwen3_5 backbones, this checkpoint is {self.backbone_type}")
        for key, default, allowed in (("prompt", "d3", R.PROMPTS), ("attention_mode", "causal", R.ATTENTION_MODES),
                                      ("pooling", "last", ("last",))):
            if config.get(key, default) not in allowed:
                raise ValueError(f"decision_config.json: {key} must be one of {allowed}, got {config.get(key)!r}")
        self.prompt = config.get("prompt", "d3")
        self.prompt_layout = config.get("prompt_layout")
        if self.prompt_layout not in (None, "header-v1"):
            raise ValueError(f"decision_config.json: unknown prompt_layout {self.prompt_layout!r}")
        self.attention_mode = config.get("attention_mode", "causal")
        self.blocked = self.attention_mode == "prefix_bidir"
        self.temperature = float(config.get("temperature", 1.0))
        if not (math.isfinite(self.temperature) and self.temperature > 0):
            raise ValueError(f"decision_config.json: temperature must be finite and positive, got {self.temperature}")
        self.temperature_by_questions = R.temperature_table(config.get("temperature_by_questions"),
                                                            "decision_config.json")
        self.readout_dtype = config.get("readout_dtype", "float32")
        if self.readout_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"readout_dtype must be float32 or bfloat16, got {self.readout_dtype!r}")
        limit = config.get("max_length")
        self.context_limit = int(limit) if limit else None
        validated = config.get("validated_length")
        self.validated_length = int(validated) if validated else None
        self.max_length = self.question_cap(False)
        self.question_types = ("choice", "noul", "score")
        self.image_placements = ("state",)
        self.box_calibration = None
        self.batch_size = int(batch_size)
        self.model_name = model_name or self.root.name
        self.permutation_average = False
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.root))
        self.tokenizer.padding_side = "left"
        codes, token_ids = R.answer_codes(self.tokenizer)
        if config.get("codes") != codes or config.get("token_ids") != token_ids:
            raise ValueError("the answer codes or token ids in decision_config.json differ from the tokenizer's")
        self.codes, self.token_ids = codes, token_ids
        probe = "State:\nA"
        if self.tokenizer(probe)["input_ids"] != self.tokenizer(probe, add_special_tokens=False)["input_ids"]:
            raise ValueError("tokenizer adds special tokens; prompts are tokenized as rendered")
        self.pad_id = self.tokenizer.pad_token_id
        # Upstream's torch-only paths stay off: no fast kernels, no state-once, no media.
        self.fast = self.state_once = self.state_once_fused = self.encoder = self.processor = None
        self.image_unavailable = self.video_unavailable = IMAGES_UNAVAILABLE
        self.backbone = load_backbone(self.root)
        self.text_model = self.backbone.language_model.model
        exact_norms(self.text_model)
        weight = mx.load(str(self.root / "readout.safetensors"))["weight"]
        hidden = self.backbone.config.text_config.hidden_size
        if tuple(weight.shape) != (R.MAX_OPTIONS, hidden):
            raise ValueError(f"readout weight has shape {tuple(weight.shape)}, expected {(R.MAX_OPTIONS, hidden)}")
        self.readout = weight.astype(mx.float32 if self.readout_dtype == "float32" else mx.bfloat16)

    def states(self, ids: Sequence[int], state: int | None = None, mode: str | None = None) -> mx.array:
        """Final-norm hidden states [L, D] of one prompt under ``mode`` (default: the package's ``attention_mode``):
        ``causal``, ``noncausal_full_attention`` or ``prefix_bidir`` with a ``state``-token state block."""
        mode = mode or self.attention_mode
        lm = self.text_model
        n = len(ids)
        h = lm.embed_tokens(mx.array([list(ids)]))
        positions = mx.tile(mx.arange(n)[None, None], (3, 1, 1))  # text: 0..L-1 on every mrope axis
        if mode == "causal":
            full = "causal"
        elif mode == "prefix_bidir":  # the state block reads itself; the question block reads the whole prompt
            row = mx.arange(n)[:, None] >= state
            full = (row | (mx.arange(n)[None, :] < state))[None, None]
        else:
            full = None
        for layer in lm.layers:
            h = layer(h, mask=None if layer.is_linear else full, position_ids=positions)
        return lm.norm(h)[0]

    def logits(self, sequences, counts, states=None, mode: str | None = None) -> np.ndarray:
        """Masked code logits (FP32, [B, 255]), as upstream's ``logits``; ``mode`` overrides ``attention_mode``."""
        rows = []
        for i, ids in enumerate(sequences):
            hidden = self.states(ids, states[i] if states else None, mode)[-1]
            z = hidden.astype(mx.float32) @ self.readout.T if self.readout_dtype == "float32" else (
                (hidden @ self.readout.T).astype(mx.float32))
            rows.append(z)
        out = np.array(mx.stack(rows), dtype=np.float32)
        out[np.arange(R.MAX_OPTIONS)[None] >= np.asarray(counts)[:, None]] = -np.inf
        return out

    def probabilities(self, sequences, counts, states=None, mode: str | None = None) -> list[list[float]]:
        z = self.logits(sequences, counts, states, mode).astype(np.float64) / self.temperature
        z = np.exp(z - z.max(axis=1, keepdims=True))
        p = z / z.sum(axis=1, keepdims=True)
        return [row[:c].tolist() for row, c in zip(p, counts)]


def load(root: str | Path, **kwargs: Any) -> D3MLX:
    return D3MLX(root, **kwargs)
