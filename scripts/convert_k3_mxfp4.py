#!/usr/bin/env python3
"""Convert moonshotai/Kimi-K3 to an MLX checkpoint, keeping the experts native.

The routed experts ship as MXFP4 (4-bit E2M1 packed two per byte, one E8M0
uint8 scale per group of 32) and were quantization-aware-trained in that format.
MLX reads and computes that layout natively — verified on the real checkpoint:
the packed bytes viewed as uint32 ARE mlx's layout, the scales are already E8M0,
and `quantized_matmul`/`gather_qmm` with mode="mxfp4" match a dense dequantized
matmul exactly. So the experts are byte-copied, never dequantized, and the QAT
grid survives untouched.

Everything Moonshot left in bf16 (attention, shared experts, the dense layer-0
MLP, the latent projections, embeddings, lm_head) is quantized to Q6 affine —
that is the ~63 GiB that makes the model fit the cluster.

Vision (vision_tower, mm_projector) is dropped: v1 serves text only.

    convert_k3_mxfp4.py --src <hf dir> --dst <out dir> [--experts native|q3]
                        [--dry-run] [--limit-shards N]
"""

import argparse
import json
import os
import shutil
import struct
import sys
import time
from collections import defaultdict

import mlx.core as mx
import numpy as np

TEXT_PREFIX = "language_model."
DROP_PREFIXES = ("vision_tower.", "mm_projector.")

Q_BITS, Q_GROUP = 6, 64          # everything Moonshot left in bf16
GATE_BITS, GATE_GROUP = 8, 64    # the MoE router stays finer, as in kimi_linear
MXFP4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
Q3 = {"group_size": 32, "bits": 3, "mode": "affine"}

# Source config values the loader's maths depends on. A silent change upstream
# would produce a checkpoint that loads and generates garbage, so assert.
EXPECTED_CONFIG = {
    "num_expert_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "first_k_dense_replace": 1,
    "hidden_act": "situ",
    "mla_use_nope": True,
    "mla_use_output_gate": True,
}

EXPERT_TO_PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def keep_verbatim(key, ndim):
    """Weights that must not be quantized.

    Norms, the KDA decay parameters, the router bias, the rank-1 attention
    residual scorers, and anything that is not a 2-D matrix (the short convs
    are [C, 1, K]).
    """
    if ndim != 2:
        return True
    if key.endswith(("A_log", "dt_bias", "e_score_correction_bias")):
        return True
    if key.endswith("_res_proj.weight"):  # rank-1: nothing to group over
        return True
    parts = key.split(".")
    return len(parts) >= 2 and "norm" in parts[-2]


def rename(key):
    """Source key -> mlx key. Returns None for anything dropped."""
    if not key.startswith(TEXT_PREFIX):
        return None
    key = key[len(TEXT_PREFIX) :]
    if key.startswith(DROP_PREFIXES):
        return None

    # Short convolutions: torch (C, 1, K) -> mlx Conv1d (C, K, 1), and the
    # module is nested one level deeper.
    for c in ("q", "k", "v"):
        key = key.replace(f".self_attn.{c}_conv1d.weight", f".self_attn.{c}_conv.conv.weight")

    key = key.replace(".block_sparse_moe.gate.e_score_correction_bias",
                      ".mlp.e_score_correction_bias")
    key = key.replace(".block_sparse_moe.", ".mlp.")
    return key


def preflight_mxfp4(src, shard):
    """Re-prove on this machine that MLX consumes the vendor packing as-is."""
    header = read_header(os.path.join(src, shard))
    packed_key = next(
        (k for k in header if k.endswith("experts.0.w1.weight_packed")), None
    )
    if packed_key is None:
        log("preflight: no expert tensor in the probe shard, skipping")
        return
    scale_key = packed_key.replace("weight_packed", "weight_scale")

    with open(os.path.join(src, shard), "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        f.read(n)
        base = 8 + n

        def grab(k):
            m = header[k]
            f.seek(base + m["data_offsets"][0])
            raw = f.read(m["data_offsets"][1] - m["data_offsets"][0])
            return np.frombuffer(raw, dtype=np.uint8).reshape(m["shape"])

        packed, scales = grab(packed_key), grab(scale_key)

    e2m1 = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32
    )
    codes = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = packed >> 4
    ref = e2m1[codes].reshape(packed.shape[0], -1, 32)
    ref = (ref * (2.0 ** (scales.astype(np.int32) - 127))[..., None]).reshape(
        packed.shape[0], -1
    )

    w = mx.view(mx.array(packed), mx.uint32)
    got = np.array(
        mx.dequantize(w, mx.array(scales), group_size=32, bits=4, mode="mxfp4").astype(
            mx.float32
        )
    )
    if not np.array_equal(got, ref):
        raise SystemExit("preflight FAILED: MLX mxfp4 dequant != independent decode")

    x = mx.random.normal((1, packed.shape[1] * 2)).astype(mx.bfloat16)
    y = mx.quantized_matmul(
        x, w, scales=mx.array(scales), transpose=True, group_size=32, bits=4,
        mode="mxfp4",
    )
    mx.eval(y)
    log(f"preflight OK — mxfp4 dequant identical, quantized_matmul {tuple(y.shape)}")


def convert(args):
    src, dst = args.src, args.dst
    cfg_all = json.load(open(os.path.join(src, "config.json")))
    text_cfg = cfg_all["text_config"]

    for k, want in EXPECTED_CONFIG.items():
        got = text_cfg.get(k)
        if got != want:
            raise SystemExit(f"config assert FAILED: {k} = {got!r}, expected {want!r}")
    log(f"config asserts OK ({len(EXPECTED_CONFIG)} values)")

    shards = sorted(
        f for f in os.listdir(src) if f.endswith(".safetensors")
    )
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    log(f"{len(shards)} shard(s) to convert")

    preflight_mxfp4(src, next((s for s in shards if "00002" in s), shards[0]))

    expert_spec = MXFP4 if args.experts == "native" else Q3
    os.makedirs(dst, exist_ok=True)

    weight_map = {}
    per_path_quant = {}
    total_bytes = 0
    n_tensors = 0

    for shard in shards:
        t0 = time.time()
        src_path = os.path.join(src, shard)
        raw = mx.load(src_path)
        out = {}
        experts = defaultdict(dict)  # (layer, proj) -> {idx: (packed, scales)}

        for key, val in raw.items():
            if key.endswith(".weight_scale"):
                continue  # picked up alongside its weight_packed
            if key.endswith(".weight_packed"):
                stem = key[: -len(".weight_packed")]
                mlx_key = rename(stem)
                if mlx_key is None:
                    continue
                # …layers.N.mlp.experts.E.wX
                parts = mlx_key.split(".")
                proj = EXPERT_TO_PROJ[parts[-1]]
                idx = int(parts[-2])
                layer_prefix = ".".join(parts[:-3])
                experts[(layer_prefix, proj)][idx] = (
                    val,
                    raw[key.replace("weight_packed", "weight_scale")],
                )
                continue

            mlx_key = rename(key)
            if mlx_key is None:
                continue

            if mlx_key.endswith("_conv.conv.weight"):
                val = mx.swapaxes(val, 1, 2)  # (C,1,K) -> (C,K,1)

            if keep_verbatim(mlx_key, val.ndim):
                out[mlx_key] = val
                continue

            stem = mlx_key[: -len(".weight")]
            if stem.endswith("mlp.gate"):
                bits, group = GATE_BITS, GATE_GROUP
                per_path_quant[stem] = {"group_size": group, "bits": bits}
            else:
                bits, group = Q_BITS, Q_GROUP
            wq, sc, bi = mx.quantize(val, group_size=group, bits=bits)
            out[stem + ".weight"], out[stem + ".scales"], out[stem + ".biases"] = (
                wq, sc, bi,
            )

        for (layer_prefix, proj), members in experts.items():
            n = len(members)
            packed = mx.stack([members[i][0] for i in range(n)])
            scales = mx.stack([members[i][1] for i in range(n)])
            stem = f"{layer_prefix}.switch_mlp.{proj}"
            if args.experts == "native":
                out[f"{stem}.weight"] = mx.view(packed, mx.uint32)
                out[f"{stem}.scales"] = scales
            else:
                deq = mx.dequantize(
                    mx.view(packed, mx.uint32), scales, group_size=32, bits=4,
                    mode="mxfp4",
                ).astype(mx.bfloat16)
                wq, sc, bi = mx.quantize(
                    deq, group_size=Q3["group_size"], bits=Q3["bits"]
                )
                out[f"{stem}.weight"], out[f"{stem}.scales"], out[f"{stem}.biases"] = (
                    wq, sc, bi,
                )
            per_path_quant[stem] = dict(expert_spec)
            del packed, scales

        del raw, experts
        if not out:
            log(f"  {shard}: no text tensors, skipped")
            continue

        mx.eval(list(out.values()))
        shard_bytes = sum(v.nbytes for v in out.values())
        total_bytes += shard_bytes
        n_tensors += len(out)

        if not args.dry_run:
            mx.save_safetensors(os.path.join(dst, shard), out)
        for k in out:
            weight_map[k] = shard
        log(
            f"  {shard}: {len(out)} tensors, {shard_bytes / 1024**3:.1f} GiB "
            f"({time.time() - t0:.0f}s)"
        )
        del out
        mx.clear_cache()

    # ── config + index ──────────────────────────────────────────────────────
    cfg = {k: v for k, v in text_cfg.items() if not k.startswith("_")}
    cfg["model_type"] = "kimi_k3"
    cfg.pop("quantization_config", None)
    cfg.pop("architectures", None)
    cfg.pop("auto_map", None)
    cfg["quantization"] = {
        "group_size": Q_GROUP,
        "bits": Q_BITS,
        "mode": "affine",
        **per_path_quant,
    }
    cfg["eos_token_id"] = collect_eos(src, text_cfg)

    if not args.dry_run:
        json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
        json.dump(
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
            open(os.path.join(dst, "model.safetensors.index.json"), "w"),
        )
        # The tokenizer's dynamic-module chain hashes EVERY local source file it
        # references — omitting encoding_k3.py killed rank 3 at startup on the
        # first real load (FileNotFoundError inside transformers' module hash).
        for extra in (
            "tokenizer_config.json", "tokenization_kimi.py", "tiktoken.model",
            "generation_config.json", "chat_template.jinja", "tokenizer.json",
            "encoding_k3.py", "configuration_kimi_k3.py", "media_utils.py",
        ):
            p = os.path.join(src, extra)
            if os.path.exists(p):
                shutil.copy2(p, dst)

    stray = [k for k in weight_map if not k.startswith(("model.", "lm_head."))]
    if stray:
        raise SystemExit(f"index assert FAILED: {len(stray)} stray keys, e.g. {stray[:3]}")
    log(f"index assert OK — {len(weight_map)} keys, all under model.*/lm_head.*")

    report_fit(total_bytes)
    log(
        f"done: {n_tensors} tensors, {total_bytes / 1024**3:.1f} GiB"
        f"{' (dry run, nothing written)' if args.dry_run else ''}"
    )


def collect_eos(src, text_cfg):
    """Merge every stop id: generation_config plus the chat template's enders.

    A single eos_token_id is how Laguna ended up looping on '</assistant>'
    (c47120b) — the template can close a turn with a token the config never
    mentions.
    """
    ids = set()
    for v in (text_cfg.get("eos_token_id"),):
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, list):
            ids.update(v)
    gen = os.path.join(src, "generation_config.json")
    if os.path.exists(gen):
        v = json.load(open(gen)).get("eos_token_id")
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, list):
            ids.update(v)

    tok = os.path.join(src, "tokenizer_config.json")
    if os.path.exists(tok):
        tc = json.load(open(tok))
        added = tc.get("added_tokens_decoder", {}) or {}
        for tid, meta in added.items():
            content = meta.get("content", "") if isinstance(meta, dict) else str(meta)
            if any(
                marker in content
                for marker in ("<|im_end|>", "eot", "endoftext", "end_of_turn")
            ):
                ids.add(int(tid))
    out = sorted(ids)
    log(f"eos ids: {out}")
    return out


def report_fit(total_bytes):
    """Project the capacity-aware pipeline split and check every rank fits."""
    wired_gib = [480, 245, 245, 245, 245]  # .29 then the four 256 GiB nodes
    total_gib = total_bytes / 1024**3
    s = sum(wired_gib)
    log(f"fit projection for {total_gib:.0f} GiB over wired {wired_gib} GiB:")
    ok = True
    for i, w in enumerate(wired_gib):
        part = total_gib * w / s
        head = w - part
        flag = "" if head >= 6 else "   <-- TIGHT"
        ok &= head >= 6
        log(f"    rank {i}: {part:6.0f} GiB of {w} GiB  (headroom {head:5.1f} GiB){flag}")
    if not ok:
        log("WARNING: at least one rank has under 6 GiB of headroom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--experts", choices=("native", "q3"), default="native")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()
    if args.experts == "q3":
        log("WARNING: --experts q3 leaves the vendor's QAT grid. Fallback only.")
    convert(args)


if __name__ == "__main__":
    main()
