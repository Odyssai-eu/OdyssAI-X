#!/usr/bin/env bash
#
# install-mlx-vlm.sh — provision the single-node VLM serving venv on a node.
#
# Mirrors the manual steps that stood up mlx_vlm.server on .29:
#   1. create a dedicated python3.12 venv at /Users/admin/.venvs/mlx-vlm
#      (NOT the python3.11 cluster venv ~/mlx-cluster/.venv — never touched)
#   2. pip install mlx-vlm pinned to the merged VL commit + torch/torchvision
#   3. smoke-import mlx_vlm + the minimax_m3_vl model module
#
# Idempotent: re-running skips venv creation if it already imports cleanly, and
# pip install is a no-op when the pin is already satisfied.
#
# This is what the node installer calls so mlx-vlm ships with the cluster
# (provision-at-node-setup, rather than a hand-rolled venv per operator).
#
# Usage:
#   scripts/install-mlx-vlm.sh <ssh-target>
#   scripts/install-mlx-vlm.sh admin@192.168.86.30
#
# Env overrides:
#   VLM_VENV      target venv path      (default /Users/admin/.venvs/mlx-vlm)
#   MLX_VLM_REF   git ref of mlx-vlm    (default ecc457b)
#   PY312         python3.12 executable (default python3.12)
set -euo pipefail

SSH_TARGET="${1:-}"
if [[ -z "$SSH_TARGET" ]]; then
  echo "usage: $0 <ssh-target>   e.g. $0 admin@192.168.86.30" >&2
  exit 2
fi

VLM_VENV="${VLM_VENV:-/Users/admin/.venvs/mlx-vlm}"
MLX_VLM_REF="${MLX_VLM_REF:-ecc457b}"
PY312="${PY312:-python3.12}"
MLX_VLM_SPEC="git+https://github.com/Blaizzy/mlx-vlm.git@${MLX_VLM_REF}"

echo "[install-mlx-vlm] target=$SSH_TARGET venv=$VLM_VENV ref=$MLX_VLM_REF"

# The remote script runs under a clean env. We do the whole thing in one SSH
# round-trip so the venv activation persists across the pip + smoke steps.
# Single-quoted heredoc: expand our vars locally into a plain script string
# first, then send it (no remote var-expansion surprises).
REMOTE_SCRIPT=$(cat <<REMOTE
set -euo pipefail
export HOME=/Users/admin USER=admin TMPDIR=/tmp
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin

VENV="${VLM_VENV}"
SPEC="${MLX_VLM_SPEC}"

# Locate python3.12 (Homebrew installs it as python3.12; fall back to a probe).
PY="\$(command -v ${PY312} || true)"
if [[ -z "\$PY" ]]; then
  for cand in /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
    [[ -x "\$cand" ]] && PY="\$cand" && break
  done
fi
if [[ -z "\$PY" ]]; then
  echo "[install-mlx-vlm] ERROR: python3.12 not found on this node" >&2
  exit 1
fi
echo "[install-mlx-vlm] python3.12 = \$PY (\$(\$PY --version 2>&1))"

# 1. venv (create only if missing)
if [[ ! -x "\$VENV/bin/python" ]]; then
  echo "[install-mlx-vlm] creating venv at \$VENV"
  mkdir -p "\$(dirname "\$VENV")"
  "\$PY" -m venv "\$VENV"
else
  echo "[install-mlx-vlm] venv already exists at \$VENV"
fi

"\$VENV/bin/python" -m pip install --upgrade pip >/dev/null

# 2. install mlx-vlm (pinned) + torch/torchvision. pip is a no-op when the
#    pin is already satisfied, so re-runs are cheap.
echo "[install-mlx-vlm] pip install \$SPEC torch torchvision"
"\$VENV/bin/python" -m pip install "\$SPEC" torch torchvision

# 3. smoke import — fails loudly if the VL model module isn't present.
echo "[install-mlx-vlm] smoke import"
"\$VENV/bin/python" -c "import mlx_vlm; from mlx_vlm.models import minimax_m3_vl; print('[install-mlx-vlm] OK', mlx_vlm.__version__ if hasattr(mlx_vlm,'__version__') else '(no __version__)')"
REMOTE
)

ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_TARGET" "bash -s" <<<"$REMOTE_SCRIPT"

echo "[install-mlx-vlm] done on $SSH_TARGET"
