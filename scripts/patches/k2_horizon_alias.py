"""Register the vendored K2-Horizon (IFM, MoVA attention) model with mlx-lm.

mlx-lm 0.32.2 has no `k2_horizon` model type, so load_model fails on the
config of IFM/K2-Horizon-MoVA-36B-A4B and of the K2-Horizon-7B / 3.7B MLX
conversions (model_type "k2_horizon"). k2_horizon_model.py is the MLX port
shipped in the mlx-community K2-Horizon-7B-8bit directory, copied unchanged.

Seeds sys.modules["mlx_lm.models.k2_horizon"] with our vendored module so
mlx_lm.utils._get_classes' importlib.import_module() resolves the model_type.
Skipped as soon as the installed mlx-lm ships the module itself.

Checked on mlx-lm 0.32.2: a 2-layer build from the 3.7B config runs a forward
pass with finite logits (vocab 250624). Not checked against the reference
weights on this path.

Drop this patch when upstream mlx-lm ships k2_horizon support.
"""

from __future__ import annotations

import importlib.util
import sys

_applied = False


def apply_k2_horizon() -> None:
    global _applied
    if _applied:
        return
    try:
        if importlib.util.find_spec("mlx_lm.models.k2_horizon") is not None:
            _applied = True
            return
    except Exception:
        pass
    try:
        from . import k2_horizon_model as _module
    except Exception as exc:
        sys.stderr.write(f"[patch] k2_horizon: vendored module failed ({exc})\n")
        return

    sys.modules["mlx_lm.models.k2_horizon"] = _module
    sys.stderr.write("[patch] k2_horizon: vendored MoVA model registered\n")
    _applied = True
