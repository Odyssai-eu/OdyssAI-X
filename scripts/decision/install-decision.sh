#!/bin/zsh
# Install / update the decision-model service on a Mac node.
#   scripts/decision/install-decision.sh <admin@host> [model_dir] [name]
# Creates ~/odyssai/decision/.venv (python 3.12, pinned mlx + mlx-lm, its own venv so a
# cluster-venv upgrade never changes the readout), copies the server and the vendored
# readout, installs the launchd agent, (re)starts it and waits for /health.
# Weights stay on the models volume; nothing is executed from the model folder.
# macOS privacy (TCC): a launchd agent cannot read an external volume until its python has
# been granted access once (a prompt on the node's screen, or Privacy & Security > Full Disk
# Access / Files and Folders). Without it the server blocks in open() on decision_config.json.
HOST=${1:?usage: install-decision.sh <admin@host> [model_dir] [name]}
MODEL=${2:-/Volumes/models/odysseus/caiovicentino1/Eikos-27B}
NAME=${3:-eikos-27b}
PORT=8095
MLX_PIN=${MLX_PIN:-mlx==0.32.0}
MLX_LM_PIN=${MLX_LM_PIN:-mlx-lm==0.31.3}
DIR=$(cd "$(dirname "$0")" && pwd)
ssh $HOST "test -f '$MODEL/decision_config.json'" || { echo "$MODEL/decision_config.json not found on $HOST"; exit 1; }
ssh $HOST "mkdir -p ~/odyssai/decision && cd ~/odyssai/decision && { [ -x .venv/bin/python ] || ~/.local/bin/uv venv -q --python 3.12 .venv; } && UV_CACHE_DIR=~/odyssai/decision/.uvcache ~/.local/bin/uv pip install -q --python .venv/bin/python '$MLX_PIN' '$MLX_LM_PIN'" || exit 1
scp -q "$DIR/decision_serve.py" "$DIR/decision_core.py" "$DIR/mlx_decide.py" "$DIR/UPSTREAM.md" "$DIR/LICENSE-eikos" $HOST:odyssai/decision/ || exit 1
sed -e "s|__MODEL__|$MODEL|" -e "s|__NAME__|$NAME|" "$DIR/eu.odyssai.decision.plist" \
  | ssh $HOST 'cat > ~/Library/LaunchAgents/eu.odyssai.decision.plist' || exit 1
# bootout returns before the job is gone; bootstrapping too early fails with
# "Bootstrap failed: 5" and leaves the service down (same fix as install-laya.sh).
ssh $HOST 'U=$(id -u); launchctl bootout gui/$U/eu.odyssai.decision 2>/dev/null; for i in $(seq 1 30); do launchctl print gui/$U/eu.odyssai.decision >/dev/null 2>&1 || break; sleep 1; done; launchctl bootstrap gui/$U ~/Library/LaunchAgents/eu.odyssai.decision.plist' || exit 1
H=${HOST#*@}
# A 27B bf16 load takes a few minutes: /health answers 503 while loading, 200 when ready.
for i in $(seq 1 180); do
  r=$(curl -s --max-time 2 -w ' %{http_code}' http://$H:$PORT/health) && [ "${r##* }" = 200 ] && { echo "OK -> ${r% *}"; exit 0; }
  sleep 5
done
echo "decision service not ready on $H:$PORT — see ~/odyssai/decision/decision.{out,err}.log"; exit 1
