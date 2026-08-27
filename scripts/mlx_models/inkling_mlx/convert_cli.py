"""CLI: convert/quantize an Inkling checkpoint to MLX.

Examples:
  python -m inkling_mlx.convert_cli --src /path/Inkling-src --dst out-bf16
  python -m inkling_mlx.convert_cli --src /path/Inkling-src --dst out-4bit --bits 4
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from .convert import convert_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Inkling bf16 source dir (HF layout)")
    ap.add_argument("--dst", required=True, help="output dir")
    ap.add_argument("--bits", type=int, default=None, choices=[2, 3, 4, 5, 6, 8],
                    help="quantization bits; omit for bf16 passthrough")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"],
                    help="cpu avoids the Metal GPU-timeout watchdog on huge tensors (slower, robust)")
    ap.add_argument("--recipe", default="uniform", choices=["uniform", "experts_only"],
                    help="experts_only keeps attention + embed/unembed at bf16 (coherent 4-bit-sized build)")
    ap.add_argument("--prune", default=None,
                    help="REAP keep-indices npz (from prune_experts.py) -> prune experts during convert")
    args = ap.parse_args()

    if args.device == "cpu":
        mx.set_default_device(mx.cpu)
        print("[convert] using CPU device (avoids Metal command-buffer timeout)")

    dtype = {"bfloat16": mx.bfloat16, "float16": mx.float16}[args.dtype]
    t0 = time.time()
    print(f"[convert] {args.src} -> {args.dst}  bits={args.bits} group_size={args.group_size} dtype={args.dtype} recipe={args.recipe} prune={args.prune}")
    convert_model(args.src, args.dst, bits=args.bits, group_size=args.group_size, out_dtype=dtype,
                  recipe=args.recipe, keep_path=args.prune)
    print(f"[convert] done in {time.time()-t0:.0f}s -> {args.dst}")


if __name__ == "__main__":
    main()
