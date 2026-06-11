#!/usr/bin/env python3
"""Standalone pipeline-parallel repro for JACCL #3149 (consecutive send/recv,
different shapes -> wrong data or HANG) and the PER-STEP CPU-barrier stop-gap.

This faithfully mimics Odysseus' pipeline forward WITHOUT importing
auto_parallel / mlx_lm (kept light/fast — only mx.distributed primitives):

  - N ranks in a chain. Each "forward micro-step":
      r > 0      : recv_like(x, r-1)        (PipelineFirstLayer)
      every rank : a small GPU matmul        (the layer compute)
      r < N-1    : send(out, r+1)            (PipelineLastLayer)
  - The hidden seq length CHANGES every iteration, so consecutive send/recv
    carry DIFFERENT shapes — the exact #3149 trigger.
  - Looped --iters times. Optional --decode adds the per-token all_gather that
    the real decode path runs (which already acts as a full-group sync).

Barrier modes:
  --barrier none      : NO barrier. Should HANG at scale (#3149).
  --barrier per-step  : ONE full-group all_sum CPU barrier per micro-step,
                        called once by EVERY rank, BEFORE the recv/compute/send.
                        This is implementer #1's design (mirrors patched_call).
  --barrier stage     : N-1 barriers per micro-step, one before each pipeline
                        boundary; only the active (sender,receiver) pair does
                        the P2P at that boundary. Every rank calls all N-1
                        barriers -> also count-symmetric. (For comparison.)

COUNT-SYMMETRY: in EVERY mode each rank calls the barrier the SAME number of
times per micro-step (per-step: 1; stage: N-1; none: 0). No deadlock by
construction — only the underlying #3149 send/recv hazard can hang.

Run:
  mlx.launch --backend jaccl --hostfile hosts.json -- \
      python repro_pp_perstep.py --barrier none     --iters 200
  mlx.launch --backend jaccl --hostfile hosts.json -- \
      python repro_pp_perstep.py --barrier per-step --iters 200
  mlx.launch --backend jaccl --hostfile hosts.json -- \
      python repro_pp_perstep.py --barrier per-step --iters 200 --decode

Detection: each rank prints a timestamped heartbeat every --log-every iters and
a final "OK rank R completed I iters". On a hang the process simply STOPS
printing — the operator (or a wall-clock wrapper) detects the wedge by absence
of new heartbeats and the missing final OK line.
"""

import argparse
import sys
import time

import mlx.core as mx


def ts() -> str:
    return time.strftime("%H:%M:%S", time.localtime()) + f".{int((time.time() % 1) * 1000):03d}"


def log(rank: int, msg: str) -> None:
    sys.stderr.write(f"[{ts()}][r{rank}] {msg}\n")
    sys.stderr.flush()


def barrier(group) -> None:
    """One full-group CPU all_sum barrier — the exact op the stop-gap uses."""
    mx.eval(
        mx.distributed.all_sum(mx.array(1.0, dtype=mx.float32), stream=mx.cpu, group=group)
    )


def shape_for_iter(i: int, hidden: int, decode: bool) -> tuple[int, int, int]:
    """Per-iter hidden-state shape. Varying seq len => consecutive send/recv
    carry DIFFERENT shapes, which is the #3149 trigger. In --decode mode the
    seq len is 1 (single token) but the prefill that precedes it (iter 0) is
    long, so the prefill->decode transition still changes shape."""
    if decode and i > 0:
        seq = 1
    else:
        # Prefill-like: a different non-trivial length every iter.
        seq = 8 + (i * 13) % 121  # 8..128, never repeats two in a row
    return (1, seq, hidden)


def gpu_busy(x: mx.array, w: mx.array) -> mx.array:
    """A small per-rank GPU matmul to occupy the GPU (stand-in for a layer)."""
    y = x @ w
    y = mx.tanh(y) @ w
    return y


def run(args) -> int:
    group = mx.distributed.init(backend=args.backend) if args.backend else mx.distributed.init()
    N = group.size()
    r = group.rank()
    hidden = args.hidden

    # Per-rank weight matrix (the "layer"). Fixed across iters.
    mx.random.seed(1234 + r)
    w = mx.random.normal((hidden, hidden), dtype=mx.float32) * (1.0 / hidden ** 0.5)
    mx.eval(w)

    log(r, f"START N={N} backend={args.backend or 'default'} barrier={args.barrier} "
           f"iters={args.iters} decode={args.decode} hidden={hidden}")

    last_hb = time.time()
    for i in range(args.iters):
        b, seq, h = shape_for_iter(i, hidden, args.decode)

        if args.barrier == "per-step":
            # ONE barrier for the whole micro-step, BEFORE any P2P. Symmetric:
            # every rank calls exactly once.
            barrier(group)

        if args.barrier == "stage":
            # Stage-ordered: walk the N-1 boundaries. EVERY rank calls a barrier
            # at every boundary; only the (boundary, boundary+1) pair does P2P.
            # Rank 0 is the source and always materializes x up front (also
            # covers the N==1 degenerate case where there are no boundaries).
            x = mx.random.normal((b, seq, h), dtype=mx.float32) if r == 0 else None
            for boundary in range(N - 1):
                barrier(group)
                if r == boundary:
                    x = gpu_busy(x, w)
                    mx.eval(x)
                    sent = mx.distributed.send(x, r + 1, group=group)
                    mx.eval(sent)
                elif r == boundary + 1:
                    template = mx.zeros((b, seq, h), dtype=mx.float32)
                    x = mx.distributed.recv_like(template, r - 1, group=group)
                    mx.eval(x)
            if r == N - 1 and x is not None:
                # The last rank runs its compute (it already received x above,
                # or — at N==1 — produced it as the source).
                x = gpu_busy(x, w)
                mx.eval(x)
        else:
            # none / per-step: straight chain (recv -> compute -> send).
            if r == 0:
                x = mx.random.normal((b, seq, h), dtype=mx.float32)
            else:
                template = mx.zeros((b, seq, h), dtype=mx.float32)
                mx.eval(template)
                x = mx.distributed.recv_like(template, r - 1, group=group)
                mx.eval(x)

            x = gpu_busy(x, w)
            mx.eval(x)

            if r != N - 1:
                sent = mx.distributed.send(x, r + 1, group=group)
                mx.eval(sent)

        if args.decode:
            # Decode adds the full-group all_gather (already a sync). Symmetric
            # across ranks regardless of barrier mode.
            g = mx.distributed.all_gather(x.reshape(-1)[:hidden], group=group)
            mx.eval(g)

        now = time.time()
        if i % args.log_every == 0 or (now - last_hb) > 5.0:
            log(r, f"iter {i}/{args.iters} shape=({b},{seq},{h}) ok")
            last_hb = now

    # Final whole-group barrier so ranks don't tear down mid-flight, then OK.
    barrier(group)
    log(r, f"OK rank {r} completed {args.iters} iters")
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
