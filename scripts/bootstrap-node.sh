#!/usr/bin/env bash
# Bootstrap an Apple Silicon Mac as an Odysseus cluster node.
#
# The orchestrator (this repo, running in Docker) SSHes into each cluster
# node and expects to find:
#
#   ~/mlx-cluster/runner.py            # the MLX runner script
#   ~/mlx-cluster/.venv/bin/python     # a venv with mlx + mlx-lm installed
#   ~/mlx-cluster/auto_parallel.py     # helper for tensor parallel
#   ~/mlx-cluster/exo_stubs.py         # backport compatibility
#   ~/mlx-cluster/patches/             # per-model patches loaded at boot
#
# This script puts them in place on a remote node. Run it from the repo
# root for each node in your topology.
#
# Usage:
#   scripts/bootstrap-node.sh <ssh-target> [models-dir]
#
# Examples:
#   scripts/bootstrap-node.sh user@host-a.lan
#   scripts/bootstrap-node.sh user@host-b.lan '$HOME/mlx-models'
#   scripts/bootstrap-node.sh user@host-b.lan /Volumes/external/models
#
# Notes:
#   - The models dir is resolved on the REMOTE node, not locally. Pass
#     either an absolute path (`/Volumes/...`) or a shell-quoted string
#     like '$HOME/mlx-models' that expands on the remote.
#   - Python versions on the node are pinned via requirements-node.txt
#     in the repo root (currently mlx==0.31.2, mlx-lm==0.31.3).
#
# Prerequisites on the node:
#   - macOS with Apple Silicon (M-series)
#   - SSH key-based auth from this host
#   - Python 3.11+ (`brew install python@3.11` if missing)
#   - Enough disk for the venv + your models (~5 GB minimum)
#
# Idempotent: re-running on a node that's already bootstrapped just
# re-syncs the scripts and re-checks the venv.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <ssh-target> [models-dir]" >&2
  echo "Example: $0 user@host-a.lan" >&2
  echo "         $0 user@host-b.lan '\$HOME/mlx-models'" >&2
  exit 1
fi

NODE="$1"
# Default models-dir is a literal `$HOME/mlx-models` that resolves on
# the REMOTE node. Wrapping it in single quotes here keeps the dollar
# sign intact through this script and through SSH; the remote shell
# does the expansion.
MODELS_DIR="${2:-\$HOME/mlx-models}"
REMOTE_DIR="${ODYSSEUS_REMOTE_CLUSTER_DIR:-\$HOME/mlx-cluster}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "→ Bootstrap $NODE"
echo "  remote cluster dir : $REMOTE_DIR  (resolved on $NODE)"
echo "  models dir         : $MODELS_DIR  (resolved on $NODE)"
echo

# 1. Verify SSH + Python on the node
echo "[1/4] Checking SSH + Python on $NODE…"
# First-contact friendliness: accept-new pins the host key on first
# connection (the GUI runs us non-interactively — without this, a brand-new
# target fails "Host key verification failed" / exit 255). Known keys are
# still verified; only UNKNOWN hosts are added.
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"

ssh $SSH_OPTS "$NODE" "
  set -e
  hostname
  uname -m | grep -q arm64 || { echo 'ERROR: not Apple Silicon (arm64)'; exit 1; }
  # Non-interactive ssh misses /opt/homebrew/bin — extend PATH before probing.
  # Fail FAST when python3.11 is absent: the system python3 (3.9) used to fall
  # through silently, building a venv for which mlx>=0.30 has no wheels
  # (cryptic 'No matching distribution for mlx==0.31.2' much later).
  export PATH=\"/opt/homebrew/bin:/usr/local/bin:\$PATH\"
  python3.11 --version 2>/dev/null || { echo 'ERROR: python3.11 not found — brew install python@3.11 (system python3 is too old for current mlx wheels)'; exit 1; }
  mkdir -p $REMOTE_DIR
  mkdir -p $MODELS_DIR
"

# 2. Copy the scripts the orchestrator will spawn + the pinned requirements
echo "[2/4] Syncing runner + helpers + requirements to $NODE…"
scp $SSH_OPTS -q \
  "$REPO_ROOT/scripts/runner.py" \
  "$REPO_ROOT/scripts/auto_parallel.py" \
  "$REPO_ROOT/scripts/exo_stubs.py" \
  "$REPO_ROOT/scripts/inference.py" \
  "$REPO_ROOT/scripts/inference_pipe.py" \
  "$REPO_ROOT/scripts/persistence.py" \
  "$REPO_ROOT/requirements-node.txt" \
  "$NODE:$REMOTE_DIR/"

# Patches directory (per-model fixes loaded at runner boot)
ssh $SSH_OPTS "$NODE" "mkdir -p $REMOTE_DIR/patches"
scp $SSH_OPTS -q "$REPO_ROOT/scripts/patches/"*.py "$NODE:$REMOTE_DIR/patches/"

# 3. Create + populate the venv on the node, pinning via requirements-node.txt
echo "[3/4] Setting up Python venv on $NODE (pinned via requirements-node.txt)…"
ssh $SSH_OPTS "$NODE" "
  set -e
  cd $REMOTE_DIR
  export PATH=\"/opt/homebrew/bin:/usr/local/bin:\$PATH\"
  if [ ! -d .venv ]; then
    python3.11 -m venv .venv
  fi
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet -r requirements-node.txt
"

# 4. Smoke test
echo "[4/4] Smoke test on $NODE…"
ssh $SSH_OPTS "$NODE" "
  cd $REMOTE_DIR
  ./.venv/bin/python -c 'import mlx.core; import mlx_lm; print(\"OK\", mlx.core.__version__, mlx_lm.__version__)'
"

echo
echo "✓ $NODE bootstrapped."
echo "  In your topology.yaml, point this node at:"
echo "    ssh: $NODE"
echo "    models_dir: $MODELS_DIR  # resolves on the node"
