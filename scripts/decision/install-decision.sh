#!/bin/zsh
# Install / update the decision server on a Mac node of a `kind: decision` cluster.
#   scripts/decision/install-decision.sh <admin@host> [backends]
#     backends: mlx (default, letter readout — Eikos), julia (PyTorch — Julia-1), or mlx,julia
# Creates ~/odyssai/decision/.venv (python 3.12, pinned deps, its own venv so a cluster-venv
# upgrade never changes the readout) and copies the server + the vendored readouts. The
# OdyssAI-X engine launches the server over ssh when a decision model is loaded on the
# cluster (DECISION_VENV / DECISION_SERVER_REMOTE in api.py): no launchd agent — a launchd
# agent cannot read the external models volume without a manual TCC grant per node, an ssh
# session can.
HOST=${1:?usage: install-decision.sh <admin@host> [mlx|julia|mlx,julia]}
BACKENDS=${2:-mlx}
MLX_PINS=${MLX_PINS:-"mlx==0.32.0 mlx-lm==0.31.3"}
JULIA_PINS=${JULIA_PINS:-"torch>=2.6 transformers>=5.0,<5.1 safetensors>=0.5 numpy>=1.26"}
DIR=$(cd "$(dirname "$0")" && pwd)
PKGS=""
[[ ",$BACKENDS," == *",mlx,"* ]] && PKGS="$PKGS $MLX_PINS"
[[ ",$BACKENDS," == *",julia,"* ]] && PKGS="$PKGS $JULIA_PINS"
[ -n "$PKGS" ] || { echo "unknown backends: $BACKENDS"; exit 2; }
QUOTED=$(for p in ${=PKGS}; do printf "'%s' " "$p"; done)
ssh $HOST "mkdir -p ~/odyssai/decision && cd ~/odyssai/decision && { [ -x .venv/bin/python ] || ~/.local/bin/uv venv -q --python 3.12 .venv; } && UV_CACHE_DIR=~/odyssai/decision/.uvcache ~/.local/bin/uv pip install -q --python .venv/bin/python $QUOTED" || exit 1
scp -q "$DIR/decision_serve.py" "$DIR/decision_core.py" "$DIR/mlx_decide.py" "$DIR/UPSTREAM.md" "$DIR/LICENSE-eikos" $HOST:odyssai/decision/ || exit 1
if [[ ",$BACKENDS," == *",julia,"* ]]; then
  ssh $HOST "rm -rf ~/odyssai/decision/julia && mkdir -p ~/odyssai/decision/julia" || exit 1
  scp -q "$DIR"/julia/*.py "$DIR"/julia/LICENSE-APACHE-2.0.txt $HOST:odyssai/decision/julia/ || exit 1
fi
# Smoke: each installed backend imports (no model loaded).
ssh $HOST "cd ~/odyssai/decision && .venv/bin/python - '$BACKENDS'" <<'PY' || exit 1
import sys, decision_core
b = sys.argv[1].split(",")
out = ["prompt " + decision_core.PROMPT_VERSION]
if "mlx" in b:
    import mlx.core as mx, mlx_lm
    out.append(f"mlx {mx.__version__} mlx_lm {mlx_lm.__version__}")
if "julia" in b:
    import torch, transformers, julia.inference  # noqa: F401
    out.append(f"torch {torch.__version__} (mps={torch.backends.mps.is_available()}) transformers {transformers.__version__}")
print("decision server installed:", "; ".join(out))
PY
