#!/usr/bin/env python3
"""Standalone pipeline-parallel repro for mlx JACCL issue #3149.

#3149: consecutive JACCL send/recv with DIFFERENT shapes (a long prefill forward
followed by single-token decode forwards, or back-to-back requests of different
prompt lengths) returns WRONG DATA or HANGS once you cross ~3 ranks. This script
reproduces that and validates the full-group all_sum barrier stop-gap, WITHOUT
importing auto_parallel / mlx_lm (kept light: only mx.distributed primitives).

It mirrors OdyssAI-X' pipeline forward:
  - N ranks in a chain. Each forward micro-step:
      r > 0   : x = recv_like(template, r-1)   (PipelineFirstLayer)
      all     : y = layer-compute(x)           (the layer body)
      r < N-1 : send(y, r+1)                    (PipelineLastLayer)
  - The hidden SEQ LENGTH changes every iteration, so consecutive send/recv
    carry DIFFERENT shapes — the exact #3149 trigger.
  - A scalar CHECKSUM is threaded down the chain (rank 0 seeds it, each rank
    folds its compute in, the last rank verifies it against the locally
    recomputed expected value). A corrupting-but-not-hanging failure prints
    CORRUPT — so the harness catches both failure modes of #3149.

Barrier modes (all COUNT-SYMMETRIC — every rank calls the barrier the SAME
number of times per micro-step, so no mode can deadlock by asymmetry; only the
underlying #3149 send/recv hazard can hang):
  --barrier none      : NO barrier. Should HANG (or print CORRUPT) at N>=3.
  --barrier per-step  : ONE full-group all_sum barrier per micro-step, called
                        once by EVERY rank, BEFORE the recv/compute/send. This is
                        the design we ship FIRST (auto_parallel pipeline_barrier).
  --barrier stage     : N-1 barriers per micro-step, one before each pipeline
                        boundary; only the active (sender=b, receiver=b+1) pair
                        does P2P at boundary b. Every rank still calls all N-1
                        barriers. The escalation design.

A hang is detected by absence of new heartbeats and a missing final
"OK rank R" line. Wrap with a wall-clock timeout to make the wedge a non-zero
exit (see the deploy commands in the writeup).
"""

import argparse
import socket
import sys
import time

import mlx.core as mx


def ts() -> str:
    return time.strftime("%H:%M:%S", time.localtime()) + f".{int((time.time() % 1) * 1000):03d}"


def log(rank: int, msg: str) -> None:
    sys.stderr.write(f"[{ts()}][r{rank}] {msg}\n")
    sys.stderr.flush()


def barrier(group) -> None:
    """The maintainer-recommended #3149 stop-gap: ONE full-group CPU all_sum.

    mx.cpu => no GPU command-buffer timeout; eval'd inline so the collective
    completes before the next op is issued on the JACCL stream. This is the
    EXACT op auto_parallel.pipeline_barrier uses.
    """
    mx.eval(mx.distributed.all_sum(mx.array(1.0, dtype=mx.float32), stream=mx.cpu, group=group))


def shape_for_iter(i: int, hidden: int, decode: bool) -> tuple[int, int, int]:
    """Per-iter hidden-state shape. Varying seq len => consecutive send/recv
    carry DIFFERENT shapes (the #3149 trigger). In --decode mode iters>0 use
    seq=1 (single token); the long prefill at iter 0 still flips the shape, so
    the prefill->decode transition itself exercises the hazard."""
    if decode and i > 0:
        seq = 1
    else:
        seq = 8 + (i * 13) % 121  # 8..128, never repeats two in a row
    return (1, seq, hidden)


def layer_compute(x: mx.array, w: mx.array) -> mx.array:
    """A small per-rank GPU matmul (stand-in for one transformer layer)."""
    y = x @ w
    y = mx.tanh(y) @ w
    return y


def checksum(x: mx.array) -> mx.array:
    """Cheap scalar fingerprint of a tensor, as a (1,) f32 array we can send."""
    return mx.array([float(mx.sum(x).item())], dtype=mx.float32)


def run(args) -> int:
    group = mx.distributed.init(backend=args.backend) if args.backend else mx.distributed.init()
    N = group.size()
    r = group.rank()
    host = socket.gethostname()
    hidden = args.hidden

    mx.random.seed(1234 + r)
    w = mx.random.normal((hidden, hidden), dtype=mx.float32) * (1.0 / hidden ** 0.5)
    mx.eval(w)

    log(r, f"START host={host} N={N} backend={args.backend or 'default'} "
           f"barrier={args.barrier} iters={args.iters} decode={args.decode} hidden={hidden}")

    # Health check: one global all_sum proves the group is up before the
    # asymmetric P2P chain (mirrors runner.py's warm-up all_sum).
    barrier(group)

    corrupt = 0
    last_hb = time.time()
    for i in range(args.iters):
        b, seq, h = shape_for_iter(i, hidden, args.decode)
        shape = (b, seq, h)

        if args.barrier == "stage":
            # STAGE-ORDERED N-1: every rank walks all N-1 boundaries, barriers at
            # each, only the (b, b+1) pair does P2P. x is THREADED through the
            # chain (received x is the same buffer that gets sent onward), so the
            # checksum is end-to-end — catches the wrong-data failure mode too.
            x = mx.random.normal(shape, dtype=mx.float32) if r == 0 else None
            for boundary in range(N - 1):
                barrier(group)
                if r == boundary:
                    x = layer_compute(x, w)
                    mx.eval(x)
                    sent = mx.distributed.send(x, r + 1, group=group)
                    mx.eval(sent)
                elif r == boundary + 1:
                    template = mx.zeros(shape, dtype=mx.float32)
                    x = mx.distributed.recv_like(template, r - 1, group=group)
                    mx.eval(x)
            if r == N - 1:
                x = layer_compute(x, w)
                mx.eval(x)

        else:
            # none / per-step: straight chain recv -> compute -> send.
            if args.barrier == "per-step":
                barrier(group)  # the ONE symmetric wall, BEFORE any P2P
            if r == 0:
                x = mx.random.normal(shape, dtype=mx.float32)
            else:
                template = mx.zeros(shape, dtype=mx.float32)
                mx.eval(template)
                x = mx.distributed.recv_like(template, r - 1, group=group)
                mx.eval(x)
            x = layer_compute(x, w)
            mx.eval(x)
            if r != N - 1:
                sent = mx.distributed.send(x, r + 1, group=group)
                mx.eval(sent)

        # Wrong-data detector: the last rank confirms it received a finite,
        # non-zero activation (a corrupted/garbage recv shows up as NaN/inf or a
        # wildly out-of-range sum). This is a cheap sentinel, not a bit-exact
        # check — enough to flag #3149's corruption mode distinct from a hang.
        if r == N - 1 and N > 1:
            s = float(mx.sum(x).item())
            if not (s == s) or abs(s) > 1e12 or s == 0.0:  # NaN, inf-ish, or all-zero
                corrupt += 1
                log(r, f"iter {i} CORRUPT checksum={s} shape={shape}")

        if args.decode:
            # Decode appends the full-group all_gather (already a sync). Symmetric
            # across ranks regardless of barrier mode.
            g = mx.distributed.all_gather(mx.ones((1, hidden), dtype=mx.float32) * (r + 1), group=group)
            mx.eval(g)

        now = time.time()
        if i % args.log_every == 0 or (now - last_hb) > 5.0 or i == args.iters - 1:
            log(r, f"iter {i}/{args.iters} shape={shape} ok")
            last_hb = now

    # Final whole-group barrier so OK only prints if EVERY rank reached the end.
    barrier(group)
    if corrupt:
        log(r, f"FAIL rank {r} on {host}: {corrupt} CORRUPT iters (barrier={args.barrier})")
        return 2
    log(r, f"OK rank {r} on {host} completed {args.iters} iters (barrier={args.barrier})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--barrier", choices=["none", "per-step", "stage"], default="none")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--decode", action="store_true")
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--backend", default="jaccl",
                   help="mx.distributed backend (jaccl|ring|''). Empty = auto.")
    args = p.parse_args()
    if args.backend == "":
        args.backend = None
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
