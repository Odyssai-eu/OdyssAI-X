#!/usr/bin/env python3
"""A VL checkpoint the default mlx-vlm venv cannot load gets its own venv.

MiniMax-M3.1-Flash fails on ~/.venvs/mlx-vlm (mlx-vlm 0.7.3, mlx 0.32.2):
"Received 741 parameters not in model" (2026-10-24). mlx-vlm PR #2580 in
~/.venvs/mlx-vlm-m31 serves it. The venv follows the checkpoint directory
name, an explicit request venv still wins, and every mlx_vlm.server launcher
plus the preflight venv probe resolve it the same way. No node is reached.

    .venv/bin/python scripts/test_vlm_model_venv.py
"""
import inspect
import os
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


M31 = "$HOME/.venvs/mlx-vlm-m31"
check("absolute source path", api._vlm_venv_for(
    "/volumes/models/odysseus/MiniMaxAI/MiniMax-M3.1-Flash"), M31)
check("relative id", api._vlm_venv_for("MiniMaxAI/MiniMax-M3.1-Flash"), M31)
check("trailing slash", api._vlm_venv_for("MiniMaxAI/MiniMax-M3.1-Flash/"), M31)
check("mlx-lm repack stays on the default venv",
      api._vlm_venv_for("odyssai/MiniMax-M3.1-Flash-nvfp4h16"), api.VLM_DEFAULT_VENV)
check("other VL model -> default venv",
      api._vlm_venv_for("odyssai/MiniMax-M3-VL-mlx-6bit-headbf16"), api.VLM_DEFAULT_VENV)
check("request venv wins", api._vlm_venv_for("MiniMaxAI/MiniMax-M3.1-Flash", " /opt/v "), "/opt/v")
check("blank request venv ignored", api._vlm_venv_for("MiniMaxAI/MiniMax-M3.1-Flash", "  "), M31)
check("empty model -> default venv", api._vlm_venv_for(""), api.VLM_DEFAULT_VENV)

# Every mlx_vlm.server launch site and the preflight probe go through the map.
for fn in (api._gather_preflight, api.admin_cluster_load, api._load_vlm_replica,
           api.admin_vlm_load):
    check(f"{fn.__name__} resolves the venv per checkpoint",
          "_vlm_venv_for(" in inspect.getsource(fn), True)

print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAIL'}")
sys.exit(1 if FAILS else 0)
