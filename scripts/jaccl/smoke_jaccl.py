"""One rank of a JACCL smoke group: all_sum loop with optional abrupt death.

Env (as the runner sets them): MLX_RANK, MLX_IBV_DEVICES (json matrix file),
MLX_JACCL_COORDINATOR (ip:port of rank 0). Optional: DIE_AT=<iter> to _exit()
without teardown (simulates a crashed rank), MSG_MB (default 16), ITER_SLEEP.

Usage: smoke_jaccl.py [iters]
Exit code 0 = all iterations OK; non-zero = an exception (the patched JACCL
raises on a dead peer instead of spinning forever).
"""
import os, sys, time
import mlx.core as mx

iters = int(sys.argv[1]) if len(sys.argv) > 1 else 100
die_at = int(os.environ.get("DIE_AT", "-1"))
size_mb = float(os.environ.get("MSG_MB", "16"))
sleep_s = float(os.environ.get("ITER_SLEEP", "0"))

t0 = time.time()
g = mx.distributed.init(backend="jaccl", strict=True)
print(f"rank {g.rank()}/{g.size()} init in {time.time()-t0:.2f}s", flush=True)
n = int(size_mb * 1024 * 1024 / 4)
x = mx.ones((n,), dtype=mx.float32) * (g.rank() + 1)
expect = g.size() * (g.size() + 1) / 2
for i in range(1, iters + 1):
    if i == die_at:
        print(f"rank {g.rank()} DIES at iter {i} (_exit, no teardown)", flush=True)
        os._exit(0)
    t = time.time()
    try:
        y = mx.distributed.all_sum(x, group=g)
        mx.eval(y)
    except Exception as e:
        print(f"rank {g.rank()} iter {i} RAISED after {time.time()-t:.2f}s: {e}", flush=True)
        sys.exit(3)
    dt = (time.time() - t) * 1000
    if i <= 2 or i % 10 == 0:
        ok = float(y[0]) == expect
        print(f"rank {g.rank()} iter {i} all_sum {size_mb:g}MB {dt:.1f}ms ok={ok}", flush=True)
    if sleep_s:
        time.sleep(sleep_s)
print(f"rank {g.rank()} done {iters} iters", flush=True)
