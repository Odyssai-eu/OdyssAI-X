#!/bin/zsh
# Install / update the Laya decision service on a Mac node (default max-64).
#   scripts/laya/install-laya.sh [admin@host]
# Creates ~/odyssai/laya/.venv (python 3.12, laya[serve] + torch with MPS),
# copies the launcher and the launchd agent, (re)starts it, checks /health.
# Weights are expected on the node under /Volumes/models/odysseus/convaiinnovations.
HOST=${1:-admin@192.168.86.50}
DIR=$(cd "$(dirname "$0")" && pwd)
ssh $HOST 'mkdir -p ~/odyssai/laya && cd ~/odyssai/laya && { [ -x .venv/bin/python ] || ~/.local/bin/uv venv -q --python 3.12 .venv; } && UV_CACHE_DIR=~/odyssai/laya/.uvcache ~/.local/bin/uv pip install -q --python .venv/bin/python "laya[serve]"' || exit 1
scp -q "$DIR/laya_serve_local.py" $HOST:odyssai/laya/ || exit 1
scp -q "$DIR/eu.odyssai.laya.plist" $HOST:Library/LaunchAgents/ || exit 1
# bootout returns before the job is gone; bootstrapping too early fails with
# "Bootstrap failed: 5" and leaves the service DOWN (seen 2026-09-24).
ssh $HOST 'U=$(id -u); launchctl bootout gui/$U/eu.odyssai.laya 2>/dev/null; for i in $(seq 1 30); do launchctl print gui/$U/eu.odyssai.laya >/dev/null 2>&1 || break; sleep 1; done; launchctl bootstrap gui/$U ~/Library/LaunchAgents/eu.odyssai.laya.plist' || exit 1
H=${HOST#*@}
for i in $(seq 1 60); do r=$(curl -s --max-time 2 http://$H:8790/health) && [ -n "$r" ] && { echo "OK -> $r"; exit 0; }; sleep 2; done
echo "laya did not answer /health on $H:8790 — see ~/odyssai/laya/laya.err.log"; exit 1
