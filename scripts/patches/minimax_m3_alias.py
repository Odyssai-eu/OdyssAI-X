"""Register the vendored minimax_m3 text model (nuit du 2026-06-12) with mlx-lm.

Seeds sys.modules["mlx_lm.models.minimax_m3"] with our vendored module so
mlx_lm.utils._get_classes' importlib.import_module() resolves the model_type
written by m3_convert.py (text tower of MiniMaxAI/MiniMax-M3, minimax_m3_vl).

Parity stamp: golden-tested against transformers main (eager reference),
max|delta| < 1e-6 on prefill + cached decode (CPU stream — the Metal fp32
matmul carries a ~1e-3 noise floor, judge exactness on CPU).

Drop this patch when upstream ships minimax_m3 support.
"""

from __future__ import annotations

import sys

_applied = False


def apply_minimax_m3() -> None:
    global _applied
    if _applied:
        return
    try:
        from mlx_lm import utils as _mlx_utils  # noqa: F401
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"[patch] minimax_m3: mlx_lm unavailable ({exc})\n")
        return
    try:
        from . import minimax_m3_model as _module
    except Exception as exc:
        sys.stderr.write(f"[patch] minimax_m3: vendored module failed ({exc})\n")
        return

    sys.modules["mlx_lm.models.minimax_m3"] = _module
    sys.stderr.write("[patch] minimax_m3: vendored MSA+MoE text model registered\n")
    _applied = True
