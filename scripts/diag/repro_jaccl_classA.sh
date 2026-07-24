#!/bin/bash
# Repro classe A — cycles init jaccl → all_sum → exit propre, 2 nodes (.29 rank0, .30 rank1)
# Edge mesh: .29:rdma_en5 ↔ .30:rdma_en5 (matrice wiring GLM 4-node, rangs 0-1)
# Usage: repro_classA.sh <n_cycles> [venv_path]   (venv défaut = ~/mlx-cluster/.venv)
set -u
N=${1:-10}
VENV=${2:-'$HOME/mlx-cluster/.venv'}
PORT=5061
DEV='[[null,"rdma_en5"],["rdma_en5",null]]'
PY_SNIPPET='
import os, sys, time
import mlx.core as mx
t0=time.time()
try:
    g = mx.distributed.init(backend="jaccl", strict=True)
    r = mx.distributed.all_sum(mx.array([1.0]), group=g)
    mx.eval(r)
    print(f"CYCLE_OK rank={g.rank()} sum={float(r):.0f} t={time.time()-t0:.2f}s", flush=True)
except Exception as e:
    print(f"CYCLE_FAIL rank_env={os.environ.get(chr(77)+chr(76)+chr(88)+chr(95)+chr(82)+chr(65)+chr(78)+chr(75))} err={e}", flush=True)
    sys.exit(1)
'
ok=0; fail=0
for i in $(seq 1 "$N"); do
  # les deux ranks en parallèle, ssh TENU (pattern immune macOS 26)
  ssh admin@192.168.86.29 "echo '$DEV' > /tmp/repro_devices.json; MLX_RANK=0 MLX_WORLD_SIZE=2 MLX_JACCL_COORDINATOR=192.168.86.29:$PORT MLX_IBV_DEVICES=/tmp/repro_devices.json $VENV/bin/python -c '$PY_SNIPPET'" > /tmp/repro_r0.$i.log 2>&1 &
  P0=$!
  ssh admin@192.168.86.30 "echo '$DEV' > /tmp/repro_devices.json; MLX_RANK=1 MLX_WORLD_SIZE=2 MLX_JACCL_COORDINATOR=192.168.86.29:$PORT MLX_IBV_DEVICES=/tmp/repro_devices.json $VENV/bin/python -c '$PY_SNIPPET'" > /tmp/repro_r1.$i.log 2>&1 &
  P1=$!
  # borne 60s par cycle
  (sleep 60 && kill $P0 $P1 2>/dev/null) & WD=$!
  wait $P0; RC0=$?
  wait $P1; RC1=$?
  kill $WD 2>/dev/null; wait $WD 2>/dev/null
  if [ $RC0 -eq 0 ] && [ $RC1 -eq 0 ]; then
    ok=$((ok+1)); echo "cycle $i: OK"
  else
    fail=$((fail+1)); echo "cycle $i: FAIL (rc0=$RC0 rc1=$RC1)"
    echo "--- r0 ---"; tail -3 /tmp/repro_r0.$i.log
    echo "--- r1 ---"; tail -3 /tmp/repro_r1.$i.log
  fi
  PORT=$((PORT+1))   # port frais par cycle (évite TIME_WAIT du coordinator)
done
echo "RESULT ok=$ok fail=$fail / $N"
