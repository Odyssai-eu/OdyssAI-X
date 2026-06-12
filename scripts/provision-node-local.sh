#!/usr/bin/env bash
# provision-node-local.sh — installe le runtime node MLX 100 % en local,
# SANS dépendance externe : python embarqué (python-build-standalone) + wheels
# vendorisées dans le DMG. Aucun brew, aucun réseau, aucun terminal utilisateur
# (invoqué par « odyssai-configure node-setup base » depuis l'app).
#
# Entrées (env, posées par le CLI) :
#   ODYSSAI_PY_TARBALL   — cpython-*-aarch64-apple-darwin-install_only.tar.gz
#   ODYSSAI_WHEELS_DIR   — répertoire de wheels (pip --no-index --find-links)
#   ODYSSAI_SCRIPTS_DIR  — runner.py, auto_parallel.py, patches/…
#   ODYSSAI_REQS         — requirements-node.txt (pins mlx/mlx-lm)
#
# Idempotent et repassable. Sort 0 = node prêt.
set -euo pipefail

CLUSTER_DIR="${ODYSSAI_CLUSTER_DIR:-$HOME/mlx-cluster}"
PY_DIR="$CLUSTER_DIR/python"
VENV="$CLUSTER_DIR/.venv"

log() { echo "[node-base] $*" >&2; }

[ "$(id -u)" -eq 0 ] && {
  echo "ERROR: do not run as root — the venv would belong to root and" >&2
  echo "the orchestrator (user ssh) would never find it." >&2
  exit 1
}
[ "$(uname -m)" = "arm64" ] || { echo "ERROR: Apple Silicon required." >&2; exit 1; }

for v in ODYSSAI_PY_TARBALL ODYSSAI_WHEELS_DIR ODYSSAI_SCRIPTS_DIR ODYSSAI_REQS; do
  [ -n "${!v:-}" ] && [ -e "${!v}" ] || { echo "ERROR: $v missing (${!v:-unset})." >&2; exit 1; }
done

mkdir -p "$CLUSTER_DIR"

# 1. Python embarqué — extraction (marqueur de version pour l'idempotence).
PY_MARK="$PY_DIR/.odyssai-python-version"
WANT="$(basename "$ODYSSAI_PY_TARBALL")"
if [ ! -x "$PY_DIR/bin/python3.11" ] || [ "$(cat "$PY_MARK" 2>/dev/null)" != "$WANT" ]; then
  log "extracting the embedded python ($WANT)…"
  rm -rf "$PY_DIR"
  mkdir -p "$PY_DIR"
  # l'archive install_only contient un répertoire racine "python/"
  tar -xzf "$ODYSSAI_PY_TARBALL" -C "$CLUSTER_DIR"
  printf '%s' "$WANT" > "$PY_MARK"
else
  log "embedded python already in place."
fi
"$PY_DIR/bin/python3.11" --version >&2

# 2. venv + wheels vendorisées (zéro réseau).
if [ ! -x "$VENV/bin/python" ]; then
  log "creating the venv…"
  "$PY_DIR/bin/python3.11" -m venv "$VENV"
fi
log "installing dependencies from the embedded wheels…"
"$VENV/bin/pip" install --quiet --no-index \
  --find-links "$ODYSSAI_WHEELS_DIR" \
  --upgrade -r "$ODYSSAI_REQS"

# 3. Scripts du node (runner + helpers + patches).
log "copying node scripts…"
for f in runner.py auto_parallel.py exo_stubs.py inference.py inference_pipe.py persistence.py; do
  [ -f "$ODYSSAI_SCRIPTS_DIR/$f" ] && cp "$ODYSSAI_SCRIPTS_DIR/$f" "$CLUSTER_DIR/$f"
done
mkdir -p "$CLUSTER_DIR/patches"
cp "$ODYSSAI_SCRIPTS_DIR"/patches/*.py "$CLUSTER_DIR/patches/" 2>/dev/null || true
# requirements pour la traçabilité des pins
cp "$ODYSSAI_REQS" "$CLUSTER_DIR/requirements-node.txt"

# 4. Smoke.
log "smoke import…"
"$VENV/bin/python" - <<'PY' >&2
import mlx.core as mx
import mlx_lm
print(f"OK mlx={mx.__version__} mlx_lm={mlx_lm.__version__}")
PY

log "node ready — $CLUSTER_DIR (.venv + runner + patches)."
