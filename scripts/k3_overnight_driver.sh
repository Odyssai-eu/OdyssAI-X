#!/usr/bin/env bash
# Wait for the Kimi K3 download to finish, then convert. Runs on the node that
# holds the download (.29), detached — the work is local disk I/O only, so it
# does not care about the macOS orphaned-process LAN behaviour that rules out
# nohup for anything multi-node.
#
# The fan-out is deliberately NOT here: it goes through the orchestrator's
# /admin/sync/rsync, which owns its own transfers.
#
#   k3_overnight_driver.sh <src-dir> <dst-dir> [expected-shards]
#
# Progress and result land in $DST.log; the last line is READY or FAILED.

set -uo pipefail

SRC="${1:?src dir}"
DST="${2:?dst dir}"
EXPECT="${3:-96}"
LOG="${DST}.log"
VENV="$HOME/mlx-cluster/.venv/bin/python"
CONVERTER="$HOME/mlx-cluster/convert_k3_mxfp4.py"

exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') driver start: $SRC -> $DST (expect $EXPECT shards)"

# ── wait for the download ───────────────────────────────────────────────────
stable=0
while :; do
  n=$(ls "$SRC"/model-*.safetensors 2>/dev/null | wc -l | tr -d ' ')
  partial=$(find "$SRC" -name '*.incomplete' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -ge "$EXPECT" ] && [ "$partial" -eq 0 ]; then
    # Two consecutive clean polls: the last file has to stop growing before
    # the converter reads it.
    stable=$((stable + 1))
    [ "$stable" -ge 2 ] && break
  else
    stable=0
  fi
  echo "$(date '+%T') waiting: $n/$EXPECT shards, $partial incomplete"
  sleep 120
done
echo "$(date '+%T') download complete: $n shards"

# ── do not compete with a bench or a live runner ────────────────────────────
while pgrep -f 'runner.py|bench_stress' >/dev/null 2>&1; do
  echo "$(date '+%T') waiting: a runner/bench is active on this node"
  sleep 120
done

# ── convert ─────────────────────────────────────────────────────────────────
echo "$(date '+%T') converting"
if caffeinate -i "$VENV" "$CONVERTER" --src "$SRC" --dst "$DST"; then
  echo "$(date '+%T') converted: $(du -sh "$DST" | cut -f1), $(ls "$DST"/*.safetensors | wc -l | tr -d ' ') shards"
  echo "READY"
else
  echo "$(date '+%T') conversion FAILED (exit $?)"
  echo "FAILED"
fi
