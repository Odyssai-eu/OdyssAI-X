#!/bin/zsh
# Install / update the decision server on a Mac node of a `kind: decision` cluster.
#   scripts/decision/install-decision.sh <admin@host> [backends]
#     backends: mlx (default, letter readout — Eikos), julia (PyTorch — Julia-1),
#               clm (MLX Qwen3-8B encoder + torch heads — CLM-v0.1-8B),
#               mlxvlm (mlx-vlm 0.7.x — LiquidAI d1, Cloudflare Clef), comma-separated
# Creates ~/odyssai/decision/.venv (python 3.12, pinned deps, its own venv so a cluster-venv
# upgrade never changes the readout) and copies the server + the vendored readouts. mlxvlm
# goes in ~/odyssai/decision/.venv-mlxvlm: mlx-vlm 0.7.x needs mlx and transformers versions
# the Julia pins exclude. Which venv runs which format is decision_serve.py's BACKEND_VENV;
# the engine reads it from `decision_serve.py --probe`. The OdyssAI-X engine launches the
# server over ssh when a decision model is loaded on the cluster (DECISION_VENV /
# DECISION_SERVER_REMOTE in api.py): no launchd agent — a launchd agent cannot read the
# external models volume without a manual TCC grant per node, an ssh session can.
HOST=${1:?usage: install-decision.sh <admin@host> [mlx|julia|clm|mlxvlm, comma-separated]}
BACKENDS=${2:-mlx}
MLX_PINS=${MLX_PINS:-"mlx==0.32.0 mlx-lm==0.31.3"}
JULIA_PINS=${JULIA_PINS:-"torch>=2.6 transformers>=5.0,<5.1 safetensors>=0.5 numpy>=1.26"}
MLXVLM_PINS=${MLXVLM_PINS:-"mlx==0.32.3 mlx-vlm==0.7.8 transformers==5.19.0"}
DIR=$(cd "$(dirname "$0")" && pwd)
PKGS=""
[[ ",$BACKENDS," == *",mlx,"* ]] && PKGS="$PKGS $MLX_PINS"
[[ ",$BACKENDS," == *",julia,"* ]] && PKGS="$PKGS $JULIA_PINS"
[[ ",$BACKENDS," == *",clm,"* ]] && PKGS="$PKGS $MLX_PINS torch>=2.6 numpy>=1.26"
VLM=""
[[ ",$BACKENDS," == *",mlxvlm,"* ]] && VLM=1
[ -n "$PKGS$VLM" ] || { echo "unknown backends: $BACKENDS"; exit 2; }
quoted() { for p in ${=1}; do printf "'%s' " "$p"; done }
UV="UV_CACHE_DIR=${UV_CACHE:-~/odyssai/decision/.uvcache} ~/.local/bin/uv"  # UV_CACHE: reuse another cache on the node
# The base venv always exists: the engine runs `decision_serve.py --probe` with its python
# (standard library only) before it knows which backend a model needs.
ssh $HOST "mkdir -p ~/odyssai/decision && cd ~/odyssai/decision && { [ -x .venv/bin/python ] || ~/.local/bin/uv venv -q --python 3.12 .venv; }" || exit 1
if [ -n "$PKGS" ]; then
  ssh $HOST "cd ~/odyssai/decision && $UV pip install -q --python .venv/bin/python $(quoted "$PKGS")" || exit 1
fi
if [ -n "$VLM" ]; then
  ssh $HOST "cd ~/odyssai/decision && { [ -x .venv-mlxvlm/bin/python ] || ~/.local/bin/uv venv -q --python 3.12 .venv-mlxvlm; } && $UV pip install -q --python .venv-mlxvlm/bin/python $(quoted "$MLXVLM_PINS")" || exit 1
fi
scp -q "$DIR/decision_serve.py" "$DIR/decision_core.py" "$DIR/mlx_decide.py" "$DIR/UPSTREAM.md" "$DIR/LICENSE-eikos" $HOST:odyssai/decision/ || exit 1
if [[ ",$BACKENDS," == *",julia,"* ]]; then
  ssh $HOST "rm -rf ~/odyssai/decision/julia && mkdir -p ~/odyssai/decision/julia" || exit 1
  scp -q "$DIR"/julia/*.py "$DIR"/julia/LICENSE-APACHE-2.0.txt $HOST:odyssai/decision/julia/ || exit 1
fi
if [[ ",$BACKENDS," == *",clm,"* ]]; then
  ssh $HOST "rm -rf ~/odyssai/decision/clm && mkdir -p ~/odyssai/decision/clm" || exit 1
  scp -q "$DIR"/clm/*.py "$DIR"/clm/LICENSE $HOST:odyssai/decision/clm/ || exit 1
fi
if [ -n "$VLM" ]; then
  ssh $HOST "rm -rf ~/odyssai/decision/clef && mkdir -p ~/odyssai/decision/clef" || exit 1
  scp -q "$DIR"/clef/*.py "$DIR"/clef/LICENSE $HOST:odyssai/decision/clef/ || exit 1
fi
# Smoke: each installed backend imports (no model loaded).
if [ -n "$PKGS" ]; then
ssh $HOST "cd ~/odyssai/decision && .venv/bin/python - '$BACKENDS'" <<'PY' || exit 1
import sys, decision_core
b = sys.argv[1].split(",")
out = ["prompt " + decision_core.PROMPT_VERSION]
if "mlx" in b:
    import mlx.core as mx, mlx_lm
    out.append(f"mlx {mx.__version__} mlx_lm {mlx_lm.__version__}")
if "clm" in b:
    import mlx.core as mx, torch, clm.heads, clm.schema  # noqa: F401
    out.append(f"clm: mlx {mx.__version__} torch {torch.__version__}")
if "julia" in b:
    import torch, transformers, julia.inference  # noqa: F401
    out.append(f"torch {torch.__version__} (mps={torch.backends.mps.is_available()}) transformers {transformers.__version__}")
print("decision server installed:", "; ".join(out))
PY
fi
if [ -n "$VLM" ]; then
ssh $HOST "cd ~/odyssai/decision && .venv-mlxvlm/bin/python -" <<'PY' || exit 1
import importlib.metadata as md
import mlx.core as mx
import mlx_vlm.models.d1.d1 as d1  # noqa: F401
from clef import clef_mlx  # noqa: F401
print(f"decision server installed (mlxvlm): mlx {mx.__version__} mlx-vlm {md.version('mlx-vlm')} "
      f"transformers {md.version('transformers')}; d1 + clef import")
PY
fi
