#!/usr/bin/env python3
"""Oracle for vlm_runner's MiMo-V2 tensor split: sharded forward == one-node forward.

Runs the first K decoder layers (+ final norm + lm_head) of a converted MiMo-V2.6
checkpoint on a fixed token sequence longer than the 128-token sliding window,
then one cached decode step, and saves the last-position logits.

  single-node reference (one process, no distributed):
      ORACLE_MODE=single python tp_oracle_mimo.py <model_dir> <K> <out_dir>
  tensor or pipeline split (every rank, JACCL env as remote_vlm_cmd writes it;
  pipeline honours RUNNER_LAYER_BOUNDS / RUNNER_RAM_WEIGHTS like the runner):
      ORACLE_MODE=tensor|pipeline MLX_RANK=.. MLX_WORLD_SIZE=..
      MLX_JACCL_COORDINATOR=.. MLX_IBV_DEVICES=..
      python tp_oracle_mimo.py <model_dir> <K> <out_dir>

Compare the saved arrays with --compare <out_dir>: relative error of the
prefill and decode logits, top-1/top-5 agreement, and rank-to-rank equality
(all ranks must hold bit-identical logits after the all_sums).

K=8 covers layer 0 (dense MLP, full attention), 1-6 (MoE, sliding window with
sinks) and 7 (MoE, full attention). Imports vlm_runner from the directory the
script is launched from (--runner-dir, default ~/mlx-cluster).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def compare(out_dir: Path, expect: int = 0) -> int:
    ref = {k: np.load(out_dir / f"single_{k}.npy") for k in ("prefill", "decode")}
    split = "pipeline" if list(out_dir.glob("pipeline_r*_prefill.npy")) else "tensor"
    ranks = sorted(out_dir.glob(f"{split}_r*_prefill.npy"))
    if not ranks:
        print("no split outputs")
        return 1
    print(f"split: {split}")
    got = [int(p.name.split("_r")[1].split("_")[0]) for p in ranks]
    if got != list(range(len(got))) or (expect and len(got) != expect):
        print(f"ORACLE FAIL: rank outputs {got}, expected 0..{(expect or len(got)) - 1} "
              f"(a rank died before saving)")
        return 1
    ok = True
    for k in ("prefill", "decode"):
        outs = [np.load(out_dir / p.name.replace("prefill", k)) for p in ranks]
        same = all(np.array_equal(outs[0], o) for o in outs[1:])
        a, b = ref[k].astype(np.float64), outs[0].astype(np.float64)
        rel = np.linalg.norm(a - b) / np.linalg.norm(a)
        top1 = int(a.argmax()) == int(b.argmax())
        top5 = len(set(np.argsort(-a)[:5]) & set(np.argsort(-b)[:5]))
        print(f"{k}: rel_err={rel:.3e} top1_equal={top1} top5_overlap={top5}/5 "
              f"ranks_bit_identical={same} ({len(outs)} ranks)")
        ok &= rel < 2e-2 and top1 and same
    print("ORACLE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("k", nargs="?", type=int, default=8)
    ap.add_argument("out", nargs="?")
    ap.add_argument("--compare")
    ap.add_argument("--expect", type=int, default=0, help="number of ranks that must have saved")
    ap.add_argument("--runner-dir", default=os.path.expanduser("~/mlx-cluster"))
    ap.add_argument("--tokens", type=int, default=300)
    args = ap.parse_args()
    if args.compare:
        return compare(Path(args.compare), args.expect)

    import mlx.core as mx
    sys.path.insert(0, args.runner_dir)
    import vlm_runner as vr
    from mlx_vlm.utils import load_model

    mode = os.environ.get("ORACLE_MODE", "single")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    group, rank = None, 0
    if mode in ("tensor", "pipeline"):
        group = mx.distributed.init(backend="jaccl", strict=True)
        rank = group.rank()
    t0 = time.time()
    model = load_model(Path(args.model), lazy=True, strict=False)
    lm = model.language_model
    lm.model.layers = lm.model.layers[: args.k]
    if mode == "tensor":
        vr._shard_lm_mimo_tensor(lm, group)
        if os.environ.get("ORACLE_NEGATIVE_SINKS"):
            # Negative control: hand each rank another rank's sinks. The
            # oracle must then FAIL, proving it can see a mis-sliced head.
            import mlx.core as _mx
            for layer in lm.layers:
                sa = layer.self_attn
                if sa.attention_sink_bias is not None:
                    sa.attention_sink_bias = _mx.roll(sa.attention_sink_bias, 7)
    elif mode == "pipeline":
        vr._shard_lm_pipeline(model, group, args.k)
        if os.environ.get("ORACLE_NEGATIVE_GAIDX"):
            # Negative control: restore the GLOBAL swa/ga indices (the bug the
            # local recompute in auto_parallel fixes). Must FAIL on a rank
            # whose first local layer is not layer 0/1.
            cfg = lm.args
            lm.model.swa_idx = cfg.hybrid_layer_pattern.index(1)
            lm.model.ga_idx = cfg.hybrid_layer_pattern.index(0)
    mx.eval(lm.parameters())
    vr.log(f"rank {rank} {mode}: {args.k} layers ready in {time.time() - t0:.1f}s "
           f"active={mx.get_active_memory() / 1e9:.1f}GB")

    rng = np.random.RandomState(0)
    toks = mx.array(rng.randint(1000, 150000, size=(1, args.tokens)).astype(np.int32))
    cache = lm.make_cache()   # local layers only after a pipeline slice
    # Two prefill chunks: the second runs on a filled cache (offset > 0), the
    # path where a wrong ga_idx/swa_idx builds the mask from the wrong cache.
    cut = max(1, (2 * args.tokens) // 3)
    lm(toks[:, :cut], cache=cache)
    prefill = lm(toks[:, cut:], cache=cache).logits[0, -1].astype(mx.float32)
    nxt = mx.array([[int(prefill.argmax().item())]], dtype=mx.int32)
    decode = lm(nxt, cache=cache).logits[0, -1].astype(mx.float32)
    mx.eval(prefill, decode)
    tag = "single" if mode == "single" else f"{mode}_r{rank}"
    np.save(out / f"{tag}_prefill.npy", np.array(prefill))
    np.save(out / f"{tag}_decode.npy", np.array(decode))
    (out / f"{tag}.json").write_text(json.dumps(
        {"k": args.k, "tokens": args.tokens, "next": int(nxt.item()),
         "peak_gb": mx.get_peak_memory() / 1e9}))
    vr.log(f"rank {rank} {mode}: saved {tag} next={int(nxt.item())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
