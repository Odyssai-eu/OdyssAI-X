#!/usr/bin/env bash
# Install the vendored/custom mlx-lm model modules AND the runtime patches onto
# cluster nodes, idempotently, with a drift report.
#
# Two sets of files, two destinations on every node:
#
#   scripts/mlx_models/*.py  →  …/site-packages/mlx_lm/models/   (venv, inside mlx-lm)
#   scripts/patches/*.py     →  ~/mlx-cluster/patches/           (next to runner.py)
#
# Model modules: some architectures have no upstream mlx-lm support (laguna →
# issue #1378) or needed a from-scratch port (deepseek_v4, hy_v3, inkling_mm,
# kimi_k3). They must land in each node's `mlx_lm/models/` for `_get_classes()`
# to resolve the model_type at load.
#
# Patches: `runner.py` does `from patches import apply_mlx_patches` at spawn and
# monkey-patches mlx-lm in place (pipeline split coverage, YaRN RoPE, batch gen,
# model aliases…). They are plain files next to the runner, NOT in the venv.
#
# Both drifted exactly as you would expect while they were copied by hand:
#   - 2026-07-25: the Argo nodes carried THREE different states of hy_v3
#     (stale on .30/.31, absent on .32/.33). Hence the mlx_models sync.
#   - 2026-09-18: a 5-node GLM-5.3 load died at the first request with
#     `IndexError` in glm_moe_dsa_model.py:103 on rank 4 (.33) — that node's
#     patches/ dated from June 21, the fix of 2026-08-29 had never reached it.
#     bootstrap-node.sh only scp'd patches/ on first contact, nothing re-synced
#     them afterwards. Hence the patches sync.
# A multi-rank load with divergent module or patch versions across ranks is a
# silent-corruption class of bug: compare md5s on every rank before hunting a
# model bug.
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
# --check reports drift (new / stale / ok, per file, for both sets) and writes
# nothing. Exits non-zero when any node is out of sync, so it doubles as a
# preflight assertion.
#
# Idempotent: re-running on an in-sync node reports `ok` for every file and
# touches nothing. Files already loaded into a RUNNING runner are not affected
# — a node picks up a new module or patch at the next load, so reload the
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
MODULES_SRC="$REPO_ROOT/scripts/mlx_models"
PATCHES_SRC="$REPO_ROOT/scripts/patches"

# Same override chain as bootstrap-node.sh. Single-quoted `$HOME` on purpose:
# it must survive this script and SSH, and expand on the REMOTE node.
REMOTE_DIR="${ODYSSAI_X_REMOTE_CLUSTER_DIR:-${ODYSSEUS_REMOTE_CLUSTER_DIR:-\$HOME/mlx-cluster}}"

# Same first-contact policy as bootstrap-node.sh: pin unknown host keys rather
# than failing "Host key verification failed" under a non-interactive caller.
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"

# Only top-level .py files, for both sets. In mlx_models/ the reference dumps
# live in ref-*/ subdirs and are documentation, not importable modules. In
# patches/ the .patch file (mlx_vlm_thinking_mode_disabled.patch) belongs to
# install-mlx-vlm.sh, not to the runner. The globs leave both alone.
shopt -s nullglob
MODULES=("$MODULES_SRC"/*.py)
PATCHES=("$PATCHES_SRC"/*.py)
shopt -u nullglob

if [ "${#MODULES[@]}" -eq 0 ]; then
  echo "ERROR: no modules found in $MODULES_SRC" >&2
  exit 1
fi
if [ "${#PATCHES[@]}" -eq 0 ]; then
  echo "ERROR: no patches found in $PATCHES_SRC" >&2
  exit 1
fi

echo "→ ${#MODULES[@]} module(s) from scripts/mlx_models/"
for m in "${MODULES[@]}"; do echo "    $(basename "$m")"; done
echo "→ ${#PATCHES[@]} patch(es) from scripts/patches/"
for p in "${PATCHES[@]}"; do echo "    $(basename "$p")"; done
echo

drift=0

# sync_set <node> <remote-dest-dir> <src-file>…
# Compares md5 per file; in --check mode only reports, otherwise scp's the
# stale/new ones and syntax-checks what was written with the node's venv
# python, so a truncated transfer surfaces here and not as an ImportError
# three minutes into a 100+ GB load. Sets drift=1 on any new/stale file.
sync_set() {
  local node="$1" dest="$2"
  shift 2
  local src name local_md5 remote_md5 state
  for src in "$@"; do
    name="$(basename "$src")"
    local_md5="$(md5 -q "$src")"
    remote_md5="$(ssh $SSH_OPTS "$node" "md5 -q '$dest/$name' 2>/dev/null || true")"

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

    scp $SSH_OPTS -q "$src" "$node:$dest/$name"
    ssh $SSH_OPTS "$node" \
      "$REMOTE_DIR/.venv/bin/python -c \"import ast; ast.parse(open('$dest/$name').read())\""
    echo "   synced  $name  (${remote_md5:0:8} -> ${local_md5:0:8})"
  done
}

for NODE in "$@"; do
  echo "── $NODE"

  # Resolve the venv's models dir on the node. Globbing python3.* keeps this
  # working when a node is rebuilt on a newer interpreter.
  MODELS_DEST="$(ssh $SSH_OPTS "$NODE" \
    "ls -d $REMOTE_DIR/.venv/lib/python*/site-packages/mlx_lm/models 2>/dev/null | head -1")"
  if [ -z "$MODELS_DEST" ]; then
    echo "   ERROR: no mlx_lm install found (run scripts/bootstrap-node.sh first)" >&2
    drift=1
    continue
  fi

  # Resolve patches/ to an absolute path too: scp (SFTP mode) does not expand
  # `$HOME` on the remote side, only the ssh shell does. Created on demand
  # when syncing; in --check mode a missing dir just reports every patch new.
  if [ "$CHECK_ONLY" -eq 0 ]; then
    ssh $SSH_OPTS "$NODE" "mkdir -p $REMOTE_DIR/patches"
  fi
  PATCHES_DEST="$(ssh $SSH_OPTS "$NODE" "eval echo \"$REMOTE_DIR/patches\"")"

  echo "   [mlx_lm/models]  $MODELS_DEST"
  sync_set "$NODE" "$MODELS_DEST" "${MODULES[@]}"
  echo "   [patches]        $PATCHES_DEST"
  sync_set "$NODE" "$PATCHES_DEST" "${PATCHES[@]}"
done

echo
if [ "$CHECK_ONLY" -eq 1 ]; then
  if [ "$drift" -ne 0 ]; then
    echo "✗ drift detected — run without --check to sync."
    exit 1
  fi
  echo "✓ all nodes in sync."
else
  echo "✓ modules and patches installed. Reload the cluster for a running model to pick them up."
fi
