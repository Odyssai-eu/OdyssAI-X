#!/usr/bin/env bash
# Install the vendored/custom mlx-lm model modules onto cluster nodes.
#
# Some architectures have no upstream mlx-lm support (laguna → issue #1378) or
# needed a from-scratch port (deepseek_v4, hy_v3, inkling_mm). Those modules
# live in `scripts/mlx_models/` and must land in each node's
# `…/site-packages/mlx_lm/models/` for `_get_classes()` to resolve the
# model_type at load.
#
# Until now this was documented as manual ("copier dans mlx_lm/models/ de chaque
# node — bootstrap-node.sh ne sync pas mlx_models/", PLAN-dsv4flash.md). It
# drifted exactly as you would expect: on 2026-07-25 the Argo nodes carried
# THREE different states — hy_v3 stale on .30/.31 (pre-rebrand version, missing
# the raw-HF tencent/Hy3 layout support), absent on .32/.33, and .33 had none of
# the modules at all. A multi-rank load with divergent module versions across
# ranks is a silent-corruption class of bug. Hence this script.
#
# Usage:
#   scripts/install-model-modules.sh <ssh-target> [<ssh-target>…]
#   scripts/install-model-modules.sh --check <ssh-target> [<ssh-target>…]
#
# Examples:
#   scripts/install-model-modules.sh admin@192.168.86.29
#   scripts/install-model-modules.sh admin@192.168.86.{29,30,31,32,33}
#   scripts/install-model-modules.sh --check admin@192.168.86.30   # report only
#
# --check reports drift (new / stale / ok) and writes nothing. Exits non-zero
# when any node is out of sync, so it doubles as a preflight assertion.
#
# Idempotent: re-running on an in-sync node reports `ok` for every module and
# touches nothing. Modules already loaded into a RUNNING runner are not
# affected — the node picks up a new module at the next load, so reload the
# cluster after syncing.

set -euo pipefail

CHECK_ONLY=0
if [ "${1:-}" = "--check" ]; then
  CHECK_ONLY=1
  shift
fi

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 [--check] <ssh-target> [<ssh-target>…]" >&2
  echo "Example: $0 admin@192.168.86.29" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/scripts/mlx_models"

# Same override chain as bootstrap-node.sh. Single-quoted `$HOME` on purpose:
# it must survive this script and SSH, and expand on the REMOTE node.
REMOTE_DIR="${ODYSSAI_X_REMOTE_CLUSTER_DIR:-${ODYSSEUS_REMOTE_CLUSTER_DIR:-\$HOME/mlx-cluster}}"

# Same first-contact policy as bootstrap-node.sh: pin unknown host keys rather
# than failing "Host key verification failed" under a non-interactive caller.
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"

# Only top-level .py files. Reference dumps live in ref-*/ subdirs and are
# documentation, not importable modules — the glob leaves them alone.
shopt -s nullglob
MODULES=("$SRC_DIR"/*.py)
shopt -u nullglob

if [ "${#MODULES[@]}" -eq 0 ]; then
  echo "ERROR: no modules found in $SRC_DIR" >&2
  exit 1
fi

echo "→ ${#MODULES[@]} module(s) from scripts/mlx_models/"
for m in "${MODULES[@]}"; do echo "    $(basename "$m")"; done
echo

drift=0

for NODE in "$@"; do
  echo "── $NODE"

  # Resolve the venv's models dir on the node. Globbing python3.* keeps this
  # working when a node is rebuilt on a newer interpreter.
  DEST="$(ssh $SSH_OPTS "$NODE" \
    "ls -d $REMOTE_DIR/.venv/lib/python*/site-packages/mlx_lm/models 2>/dev/null | head -1")"
  if [ -z "$DEST" ]; then
    echo "   ERROR: no mlx_lm install found (run scripts/bootstrap-node.sh first)" >&2
    drift=1
    continue
  fi

  for src in "${MODULES[@]}"; do
    name="$(basename "$src")"
    local_md5="$(md5 -q "$src")"
    remote_md5="$(ssh $SSH_OPTS "$NODE" "md5 -q '$DEST/$name' 2>/dev/null || true")"

    if [ "$local_md5" = "$remote_md5" ]; then
      echo "   ok      $name"
      continue
    fi

    if [ -z "$remote_md5" ]; then state="new    "; else state="stale  "; fi
    drift=1

    if [ "$CHECK_ONLY" -eq 1 ]; then
      echo "   $state $name  (node=${remote_md5:0:8}  repo=${local_md5:0:8})"
      continue
    fi

    scp $SSH_OPTS -q "$src" "$NODE:$DEST/$name"
    # Syntax-check what we just wrote, so a truncated transfer surfaces here
    # and not as an ImportError three minutes into a 100+ GB load.
    ssh $SSH_OPTS "$NODE" \
      "$REMOTE_DIR/.venv/bin/python -c \"import ast; ast.parse(open('$DEST/$name').read())\""
    echo "   synced  $name  (${remote_md5:0:8} -> ${local_md5:0:8})"
  done
done

echo
if [ "$CHECK_ONLY" -eq 1 ]; then
  if [ "$drift" -ne 0 ]; then
    echo "✗ drift detected — run without --check to sync."
    exit 1
  fi
  echo "✓ all nodes in sync."
else
  echo "✓ modules installed. Reload the cluster for a running model to pick them up."
fi
