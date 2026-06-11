#!/usr/bin/env python3
"""Standalone pipeline-parallel JACCL deadlock repro for mlx issue #3149.

Reproduces the PREFILL collective-ordering hang that bites the 43-layer
DeepSeek-V4-Flash serve across 3+ nodes (blocker A). It is *pipeline-faithful*:
it mirrors exactly what scripts/auto_parallel.py's PipelineFirstLayer /
PipelineLastLayer do on the prefill path — a chain of
``recv_like`` (rank > 0) -> small GPU compute -> ``send`` (rank < N-1) — but
deliberately uses a DIFFERENT activation shape on every iteration, which is the
precise trigger described in #3149 ("consecutive send/recv with different shapes
produces wrong data or HANGS").

It does NOT import auto_parallel (kept light/fast, no mlx_lm) — it talks to
``mx.distributed`` directly so it launches in a second on a cold cluster.

The point of the harness:
  * ``--barrier none``  -> at scale (3+ ranks) this should HANG (stop printing).
  * ``--barrier per-step`` or ``--barrier stage`` -> this should COMPLETE and
    print ``RANK r OK`` on every rank.

The two barrier modes mirror the two count-symmetric designs implemented in
auto_parallel.py, so a green run here predicts a green run in prod:

  per-step : ONE full-group all_sum CPU barrier per micro-step, issued exactly
             once by EVERY rank (rank 0 included), before any P2P. Symmetric by
             construction: 1 barrier/rank/step.

  stage    : every rank issues a barrier at its shard ENTRY (where rank>0 would
             recv) and at its shard EXIT (where rank<N-1 would send) — 2
             barriers/rank/step, identical count on every rank. The P2P happens
             only on the active side; the barrier count does not depend on
             position, so no rank can run a collective the others skip.

Launch (4 ranks, JACCL/RDMA over TB5):

  mlx.launch --backend jaccl --hostfile hosts.json -- \
    python repro_pp_barrier.py --barrier none  --iters 200

  mlx.launch --backend jaccl --hostfile hosts.json -- \
    python repro_pp_barrier.py --barrier per-step --iters 200
  mlx.launch --backend jaccl --hostfile hosts.json -- \
    python repro_pp_barrier.py --barrier stage --iters 200 --decode

On a hang the harness simply stops emitting progress lines; the operator detects
the wedge by wall-clock (no new line for > timeout). On success every rank
prints a final ``RANK r OK iters=N``.
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx


def _ts() -> str:
    return time.strftime("%H:%M:%S", time.localtime()) + f".{int((time.time() % 1) * 1000):03d}"


def _log(rank: int, msg: str) -> None:
    # Unbuffered, flushed: on a hang the LAST flushed line is the diagnosis.
    sys.stderr.write(f"[{_ts()}] rank {rank}: {msg}\n")
    sys.stderr.flush()


def _barrier(group: mx.distributed.Group) -> None:
    """The maintainer-recommended stop-gap: a full-group all_sum on the CPU
    stream. Every rank that calls this MUST be matched by a call on every other
    rank in the group, or it deadlocks. Callers are responsible for symmetry."""
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu, group=group))


def _iter_shape(it: int, hidden: int) -> tuple[int, int, int]:
    """Different activation shape every iter — the #3149 trigger. We vary the
    sequence length deterministically (so every rank agrees on the shape for a
    given iter) across a spread that forces re-allocation of the RDMA staging
    buffer, which is what surfaces the consecutive-different-shape hang."""
    # 1 .. 257, never zero, monotone-ish sawtooth so consecutive iters differ.
    seq = 1 + ((it * 37 + 1) % 256)
    return (1, seq, hidden)


def run(
    rank: int,
    world: int,
    group: mx.distributed.Group,
    *,
    barrier: str,
    iters: int,
    hidden: int,
    decode: bool,
    log_every: int,
) -> None:
    is_first = rank == 0
    is_last = rank == world - 1

    # Small persistent GPU weight to occupy the GPU between collectives, like a
    # real decoder layer's matmuls. Keeps the Metal queue busy so the
    # collective-ordering race is realistic (matches the "small GPU compute"
    # between recv and send in PipelineFirstLayer/PipelineLastLayer).
    w = mx.random.normal((hidden, hidden)) * (1.0 / (hidden ** 0.5))
    mx.eval(w)

    _log(rank, f"start world={world} barrier={barrier} iters={iters} "
               f"hidden={hidden} decode={decode} (first={is_first} last={is_last})")

    for it in range(iters):
        shape = _iter_shape(it, hidden)

        # ---- per-step barrier: exactly ONE per rank per micro-step, up front,
        # before ANY point-to-point. Symmetric by construction. ----
        if barrier == "per-step":
            _barrier(group)

        # ===== shard ENTRY (mirror of PipelineFirstLayer.__call__) =====
        if barrier == "stage":
            # EVERY rank barriers at entry — count is position-independent.
            _barrier(group)
        if not is_first:
            x = mx.zeros(shape, dtype=mx.float32)
            mx.eval(x)  # keep recv on CPU stream, no GPU timeout
            x = mx.distributed.recv_like(x, rank - 1, group=group)
            mx.eval(x)
        else:
            # Rank 0 fabricates the activation (embedding stand-in).
            x = mx.ones(shape, dtype=mx.float32) * 0.01
            mx.eval(x)

        # ===== small GPU compute (stand-in for the decoder layers) =====
        # x is (1, seq, hidden); contract over hidden via the persistent weight.
        x = mx.tanh(x @ w)
        mx.eval(x)

        # ===== shard EXIT (mirror of PipelineLastLayer.__call__) =====
        if barrier == "stage":
            # EVERY rank barriers at exit — symmetric with the entry barrier.
            _barrier(group)
        if not is_last:
            sent = mx.distributed.send(x, rank + 1, group=group)
            mx.eval(sent)

        # ===== optional decode all_gather (mirror of the DECODE path) =====
        # In real decode every rank's PipelineLastLayer ends with an all_gather,
        # which itself acts as a full-group sync. We replicate it so --decode
        # exercises that path too. all_gather is symmetric on its own.
        if decode:
            g = mx.distributed.all_gather(x.reshape(1, -1)[:, :hidden], group=group)
            mx.eval(g)

        if (it % log_every) == 0 or it == iters - 1:
            _log(rank, f"iter {it}/{iters} seq={shape[1]} done")

    # Final clean full-group barrier so all ranks agree they finished (and so a
    # straggler can't make a peer's last send dangle). This is symmetric.
    _barrier(group)
    _log(rank, f"OK iters={iters} barrier={barrier}")
    print(f"RANK {rank} OK iters={iters} barrier={barrier}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--barrier", choices=["none", "per-step", "stage"], default="none",
                   help="symmetric barrier mode (default: none -> should hang at scale)")
    p.add_argument("--iters", type=int, default=200, help="micro-steps to run")
    p.add_argument("--hidden", type=int, default=1024, help="hidden dim of the GPU matmul")
    p.add_argument("--decode", action="store_true",
                   help="also run the per-step all_gather (decode-path sync)")
    p.add_argument("--log-every", type=int, default=10, help="progress print cadence")
    p.add_argument("--backend", default=None,
                   help="distributed backend override (else mlx.launch sets it)")
    args = p.parse_args()

    if args.backend:
        group = mx.distributed.init(backend=args.backend, strict=True)
    else:
        group = mx.distributed.init()

    rank = group.rank()
    world = group.size()

    if world < 2:
        _log(rank, "WARNING: world_size < 2 — the #3149 hang only reproduces at "
                   "scale (3+ ranks). Running a self-consistency pass only.")

    try:
        run(
            rank, world, group,
            barrier=args.barrier,
            iters=args.iters,
            hidden=args.hidden,
            decode=args.decode,
            log_every=max(1, args.log_every),
        )
    except Exception as e:  # noqa: BLE001 — repro tool, surface everything
        _log(rank, f"EXCEPTION {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
