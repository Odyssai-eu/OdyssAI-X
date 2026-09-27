#!/bin/zsh
# Install / update the decision server on a Mac node of a `kind: decision` cluster.
#   scripts/decision/install-decision.sh <admin@host>
# Creates ~/odyssai/decision/.venv (python 3.12, pinned mlx + mlx-lm, its own venv so a
# cluster-venv upgrade never changes the readout) and copies the server + the vendored
# readout. The OdyssAI-X engine launches the server over ssh when a decision model is
# loaded on the cluster (DECISION_VENV / DECISION_SERVER_REMOTE in api.py): no launchd
# agent — a launchd agent cannot read the external models volume without a manual TCC
# grant per node, an ssh session can.
HOST=${1:?usage: install-decision.sh <admin@host>}
MLX_PIN=${MLX_PIN:-mlx==0.32.0}
MLX_LM_PIN=${MLX_LM_PIN:-mlx-lm==0.31.3}
DIR=$(cd "$(dirname "$0")" && pwd)
ssh $HOST "mkdir -p ~/odyssai/decision && cd ~/odyssai/decision && { [ -x .venv/bin/python ] || ~/.local/bin/uv venv -q --python 3.12 .venv; } && UV_CACHE_DIR=~/odyssai/decision/.uvcache ~/.local/bin/uv pip install -q --python .venv/bin/python '$MLX_PIN' '$MLX_LM_PIN'" || exit 1
scp -q "$DIR/decision_serve.py" "$DIR/decision_core.py" "$DIR/mlx_decide.py" "$DIR/UPSTREAM.md" "$DIR/LICENSE-eikos" $HOST:odyssai/decision/ || exit 1
# Smoke: the vendored readout imports and the MLX stack is there (no model loaded).
ssh $HOST 'cd ~/odyssai/decision && .venv/bin/python -c "import decision_core, mlx.core as mx, mlx_lm; print(\"decision server installed: mlx\", mx.__version__, \"mlx_lm\", mlx_lm.__version__, \"prompt\", decision_core.PROMPT_VERSION)"' || exit 1
