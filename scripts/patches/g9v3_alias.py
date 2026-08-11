"""Register the vendored g9v3 model (ai9stars/G9v3-39A5B) with mlx-lm.

Seeds sys.modules["mlx_lm.models.g9v3"] with our vendored module so
mlx_lm.utils._get_classes' importlib.import_module() resolves the model_type
`g9v3`. G9v3 = DeepSeek-V3 MoE + GQA + gated attention — voir g9v3_model.py.

Drop ce patch quand upstream mlx-lm ajoute g9v3.
"""

from __future__ import annotations

import sys

_applied = False


def apply_g9v3() -> None:
    global _applied
    if _applied:
        return
    try:
        from mlx_lm import utils as _mlx_utils  # noqa: F401
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"[patch] g9v3: mlx_lm unavailable ({exc})\n")
        return
    try:
        from . import g9v3_model as _module
    except Exception as exc:
        sys.stderr.write(f"[patch] g9v3: vendored module failed ({exc})\n")
        return

    sys.modules["mlx_lm.models.g9v3"] = _module
    sys.stderr.write("[patch] g9v3: vendored DeepSeek-V3-MoE + gated-attn registered\n")
    _applied = True
