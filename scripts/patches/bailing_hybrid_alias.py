"""Register the vendored bailing_hybrid model (#43 WU-B) with mlx-lm.

Two actions at runner boot:
  1. seed sys.modules["mlx_lm.models.bailing_hybrid"] with our vendored module
     so mlx_lm.utils._get_classes' importlib.import_module() resolves it;
  2. point MODEL_REMAPPING["bailing_hybrid"] at it — OVERRIDING the obsolete
     hand-edit some fleet venvs carry at mlx_lm/utils.py:46 (the remap to
     bailing_moe_linear that silently dropped the MLA attention weights and
     could never produce sane output).

Drop this patch when upstream ships real bailing_hybrid support
(ml-explore/mlx-lm#1233).
"""

from __future__ import annotations

import sys

_applied = False


def apply_bailing_hybrid() -> None:
    global _applied
    if _applied:
        return
    try:
        from mlx_lm import utils as _mlx_utils
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"[patch] bailing_hybrid: mlx_lm.utils unavailable ({exc})\n")
        return
    try:
        from . import bailing_hybrid_model as _module
    except Exception as exc:
        sys.stderr.write(f"[patch] bailing_hybrid: vendored module failed ({exc})\n")
        return

    sys.modules["mlx_lm.models.bailing_hybrid"] = _module

    table = getattr(_mlx_utils, "MODEL_REMAPPING", None)
    if isinstance(table, dict):
        previous = table.get("bailing_hybrid")
        table["bailing_hybrid"] = "bailing_hybrid"
        if previous and previous != "bailing_hybrid":
            sys.stderr.write(
                f"[patch] bailing_hybrid: overriding obsolete remap -> {previous}\n"
            )
    sys.stderr.write(
        "[patch] bailing_hybrid: vendored MLA+GLA model registered (#43)\n"
    )
    _applied = True
