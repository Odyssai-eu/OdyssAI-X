#!/usr/bin/env python3
"""Suggest MLX models that fit a given cluster.

Used by INSTALL-CLUSTER.md step 7 (and by anyone planning capacity). Takes
the total cluster RAM (sum of unified-memory across ALL nodes) in GB and
prints the open-weights models that fit, keeping 20% margin for KV cache
and runtime overhead.

  Usage:
    scripts/suggest-models.py --ram-gb 96
    scripts/suggest-models.py --ram-gb 1024
    scripts/suggest-models.py --ram-gb 96 --json

What matters is cumulated RAM, not node count. 4 × 32 GB Macs total 128 GB
of usable unified memory — that doesn't unlock GLM-5.1 (1.49 TB). Compute
the sum by SSH-ing each node and reading `sysctl hw.memsize`.

The list is conservative — we use the published file size of the MLX
quantization and multiply by 1.20 to leave 20% for the KV cache and the
runner process overhead. Bigger contexts grow the KV cache; the headroom
factor here suits ~16k prompts. For long-context (100k+) work, halve the
working set.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Optional


# Headroom factor: model file size × this = working-set during inference.
# 20% margin covers the KV cache, activations, and runner process overhead
# for typical (~16k token) prompts.
KV_HEADROOM = 1.20


@dataclass
class ModelSuggestion:
    repo: str                # Hugging Face repo id
    size_gb: float           # On-disk size of the MLX weights
    role: str                # chat / code / vision / reasoner / autocomplete
    note: str = ""           # Short reason / context

    @property
    def working_set_gb(self) -> float:
        return self.size_gb * KV_HEADROOM


# Curated starter catalog — restricted to `mlx-community/*` repos, the
# official MLX team channel on Hugging Face. This is NOT exhaustive
# (third-party converters publish many more variants); it's the
# shortest list we're confident loads end-to-end with mlx-lm with no
# quantization surprises. For anything outside this set, operators
# point huggingface-cli at any HF repo they want — OdyssAI-X doesn't
# care who packaged it as long as it's a valid mlx-lm directory.
#
# Order: ascending size, so the agent picks the right tier per RAM.
CATALOG: list[ModelSuggestion] = [
    ModelSuggestion(
        repo="mlx-community/Qwen3.6-35B-A3B-8bit",
        size_gb=38.0,
        role="chat",
        note="Latest Qwen MoE. 3B active params → high tok/s. Vision variant available.",
    ),
    ModelSuggestion(
        repo="mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit",
        size_gb=85.0,
        role="chat",
        note="Long-context MoE. 3B active → fast despite 80B total.",
    ),
    ModelSuggestion(
        repo="mlx-community/Qwen3-Coder-Next-8bit",
        size_gb=85.0,
        role="code",
        note="Code-focused Qwen3-Next. Strong on agentic coding tasks.",
    ),
    ModelSuggestion(
        repo="mlx-community/Qwen3.5-122B-A10B-8bit",
        size_gb=131.0,
        role="chat",
        note="Qwen 3.5 mid-tier MoE. 10B active params, balanced speed/quality.",
    ),
    ModelSuggestion(
        repo="mlx-community/MiniMax-M2-8bit",
        size_gb=243.0,
        role="chat",
        note="MiniMax M2 8-bit. Fits a 256 GB Mac Studio or a 2× 256 GB pool.",
    ),
    ModelSuggestion(
        repo="mlx-community/Qwen3.5-397B-A17B-8bit",
        size_gb=422.0,
        role="reasoner",
        note="Qwen 3.5 frontier. Pipeline-parallel across ≥2 ultras.",
    ),
    ModelSuggestion(
        repo="mlx-community/Qwen3-Coder-480B-A35B-Instruct-8bit",
        size_gb=540.0,
        role="code",
        note="Frontier code model. 2× 512GB nodes or 3× 256GB nodes.",
    ),
    ModelSuggestion(
        repo="mlx-community/DeepSeek-V3.1-8bit",
        size_gb=713.0,
        role="reasoner",
        note="DeepSeek V3.1. 2× 512GB nodes or 3× 256GB nodes.",
    ),
    ModelSuggestion(
        repo="mlx-community/GLM-5-8bit-MXFP8",
        size_gb=767.0,
        role="chat",
        note="Zhipu GLM-5 in microscaling FP8. Frontier general chat.",
    ),
    ModelSuggestion(
        repo="mlx-community/GLM-5.1",
        size_gb=1490.0,
        role="reasoner",
        note="GLM-5.1, 1.49 TB. Top end.",
    ),
]


def suggest(total_ram_gb: float) -> list[ModelSuggestion]:
    """Return models whose working set fits `total_ram_gb` of cumulated unified memory."""
    return [m for m in CATALOG if m.working_set_gb <= total_ram_gb]


def format_table(models: list[ModelSuggestion]) -> str:
    if not models:
        return (
            "No suggested models fit. You may still load smaller community models "
            "manually; the catalog here is curated for the common cases.\n"
        )
    lines = [
        f"{'Model':<58} {'Size':>6}  {'Role':<11}  Note",
        "-" * 110,
    ]
    for m in models:
        lines.append(
            f"{m.repo:<58} {m.size_gb:>5.0f}G  {m.role:<11}  {m.note}"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="List MLX models that fit a given cluster.",
    )
    p.add_argument(
        "--ram-gb",
        type=float,
        required=True,
        help="Cumulated unified memory across ALL cluster nodes, in GB.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print the suggestions as JSON (suitable for piping to jq).",
    )
    args = p.parse_args(argv)

    suggestions = suggest(args.ram_gb)
    if args.json:
        print(
            json.dumps(
                {
                    "ram_gb": args.ram_gb,
                    "kv_headroom": KV_HEADROOM,
                    "suggestions": [asdict(m) for m in suggestions],
                },
                indent=2,
            )
        )
    else:
        print(format_table(suggestions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
