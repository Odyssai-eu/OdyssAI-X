"""Register a model_type remap so mlx-lm can load `mimo_v2` models.

Upstream mlx-lm (0.31.3) ships `mimo_v2_flash.py` (the optimized variant)
but no `mimo_v2.py`. Models converted from upstream Xiaomi releases declare
`model_type: "mimo_v2"` in their config.json (no `_flash` suffix), so the
default importlib dispatch via `mlx_lm.utils._get_classes` fails with:

    ValueError: Model type mimo_v2 not supported.

`mimo_v2_flash.py` is implementation-compatible with the regular variant
(same MiMoV2ForCausalLM class name, no enum check on model_type). Adding
the remap in `MODEL_REMAPPING` lets the loader pick the right file.

If a real `mimo_v2.py` (non-flash) lands upstream later, drop this patch.
"""
from __future__ import annotations

import sys


_applied = False


def apply_mimo_v2_alias() -> None:
    global _applied
    if _applied:
        return
    try:
        from mlx_lm import utils as _mlx_utils
    except Exception as exc:
        sys.stderr.write(f"[patch] mimo_v2_alias: mlx_lm.utils unavailable ({exc})\n")
        return
    table = getattr(_mlx_utils, "MODEL_REMAPPING", None)
    if not isinstance(table, dict):
        sys.stderr.write("[patch] mimo_v2_alias: MODEL_REMAPPING not a dict, skipping\n")
        return
    if "mimo_v2" not in table:
        table["mimo_v2"] = "mimo_v2_flash"
        sys.stderr.write("[patch] mimo_v2_alias: mimo_v2 → mimo_v2_flash registered\n")
    _applied = True
