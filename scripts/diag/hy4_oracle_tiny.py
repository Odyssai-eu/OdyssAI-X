#!/usr/bin/env python3
"""Oracle for scripts/mlx_models/hy_v4.py against the transformers reference.

A tiny random HYV4 (6 layers: full, full, shared, shared, shared, full; 8 heads;
index_topk=8 so sparse selection is active past 8 tokens; 16 indexer heads: with
4, a key has a 1/16 chance of an exactly-zero relu score, and torch and MLX
break those ties differently — both valid, measured 2026-09-29) is built in torch
(transformers >= 5.17, eager), its weights are renamed to the checkpoint names
and loaded through our module's sanitize(). Both run in float32.

RoPE: the transformers reference (add-h4) rotates halves (rotate_half), but the
released checkpoint needs interleaved pairs, as kernelpool/mlx-lm add-hy4-preview
does (traditional=True). Our module followed the reference and generated
scrambled text on the real weights (2026-10-05). The reference is therefore
patched to interleaved RoPE by default; `--rope reference` keeps its own.

Checks, logits at every position against the reference's full causal forward:
  1. MLX full forward, no cache (prefill path, L > 1)
  2. MLX with the latent cache: prefill in two chunks (the second on a filled
     cache), then one-token decode steps (absorbed path + sparse gather)

Needs one venv with mlx, mlx-lm (0.31.x), torch and transformers:
    python hy4_oracle_tiny.py [--module scripts/mlx_models/hy_v4.py] [--tokens 30]
Tensor split (Model.shard), N local ranks, every rank checks the same logits:
    mlx.launch --backend ring -n 2 hy4_oracle_tiny.py --tp
Pipeline split (auto_parallel, the runner's path), bounds as RUNNER_LAYER_BOUNDS:
    mlx.launch --backend ring -n 2 hy4_oracle_tiny.py --pp 0,1,6
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _interleaved_rope(q, k, cos, sin, unsqueeze_dim=1):
    """apply_rotary_pos_emb on (even, odd) pairs instead of halves. cos/sin come
    from the reference's rotary embedding, cat(freqs, freqs): the first half
    holds one frequency per pair."""
    import torch

    d = cos.shape[-1]
    c = cos[..., : d // 2].repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    s = sin[..., : d // 2].repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)

    def rot(x):
        return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)

    return (q * c) + (rot(q) * s), (k * c) + (rot(k) * s)


def build_reference(n_tokens: int, seed: int, rope: str = "interleaved"):
    import torch
    import transformers.models.hy_v4.modeling_hy_v4 as hm
    from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config
    from transformers.models.hy_v4.modeling_hy_v4 import HYV4ForCausalLM

    if rope == "interleaved":
        hm.apply_rotary_pos_emb = _interleaved_rope

    cfg = HYV4Config(
        vocab_size=512, hidden_size=96, intermediate_size=192, moe_intermediate_size=48,
        num_hidden_layers=6, num_attention_heads=8, q_lora_rank=64, kv_lora_rank=32,
        qk_nope_head_dim=24, qk_rope_head_dim=8, v_head_dim=32, n_routed_experts=8,
        num_experts_per_tok=2, index_topk=8, index_head_dim=16, index_n_heads=16, hc_mult=4,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0},
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = HYV4ForCausalLM(cfg).eval().float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("layernorm.weight", "norm.weight", "k_norm.weight")):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif name.endswith("e_score_correction_bias"):
                p.copy_(0.01 * torch.randn_like(p))
            elif name.endswith("sinks"):
                p.copy_(torch.randn_like(p))
            else:
                p.copy_(0.08 * torch.randn_like(p))
    toks = torch.randint(3, cfg.vocab_size, (1, n_tokens), generator=torch.Generator().manual_seed(seed + 1))
    with torch.no_grad():
        logits = model(toks, use_cache=False).logits[0].float().numpy()
    sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
    return cfg, sd, toks[0].numpy(), logits


def to_checkpoint_names(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        n = k
        n = n.replace(".self_attn.sinks", ".self_attn.learnable_sink_param")
        n = n.replace(".self_attn.gate_proj.", ".self_attn.linear_gate.")
        for src, dst in ((".attn_hc.", ".hc_attn_layer.hc_pre.hc_"), (".ffn_hc.", ".hc_mlp_layer.hc_pre.hc_")):
            if src in n:
                n = n.replace(src, dst)
        n = n.replace("model.hc_head.hc_", "model.hc_head.hc_head_")
        out[n] = v
    return out


def load_module(path: Path):
    import mlx_lm.models  # noqa: F401  (package for the module's relative imports)
    # Registered under its production name: auto_parallel imports
    # mlx_lm.models.hy_v4.Model for its isinstance checks (_set_layers).
    spec = importlib.util.spec_from_file_location("mlx_lm.models.hy_v4", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=str(HERE.parent / "mlx_models" / "hy_v4.py"))
    ap.add_argument("--tokens", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tp", action="store_true", help="shard() over the launched group")
    ap.add_argument("--pp", default="", help="pipeline bounds over the launched group, e.g. 0,1,6")
    ap.add_argument("--rope", choices=("interleaved", "reference"), default="interleaved",
                    help="reference RoPE: interleaved pairs (the checkpoint's) or the "
                         "transformers reference's rotate_half")
    args = ap.parse_args()

    cfg, sd, toks, ref = build_reference(args.tokens, args.seed, args.rope)

    import mlx.core as mx
    # CPU: exact fp32. On M5-class GPUs MLX's fp32 matmul runs at reduced
    # precision (7.7e-4 relative on a plain 30x96x192 matmul, measured
    # 2026-09-29), which compounds to ~0.3 over 6 layers and hides real bugs.
    mx.set_default_device(mx.cpu)
    mod = load_module(Path(args.module))
    conf = {k: v for k, v in cfg.to_dict().items()}
    conf["model_type"] = "hy_v4"
    margs = mod.ModelArgs.from_dict(conf)
    model = mod.Model(margs)
    weights = model.sanitize({k: mx.array(v) for k, v in to_checkpoint_names(sd).items()})
    model.load_weights(list(weights.items()), strict=True)
    tag = ""
    if args.tp:
        group = mx.distributed.init()
        model.shard(group)
        tag = f"[tp rank {group.rank()}/{group.size()}] "
    if args.pp:
        sys.path.insert(0, str(HERE.parent))
        from auto_parallel import pipeline_auto_parallel, misaligned_pipeline_starts
        from exo_stubs import PipelineShardMetadata
        group = mx.distributed.init()
        b = [int(v) for v in args.pp.split(",")]
        bad = misaligned_pipeline_starts(b, margs.indexer_types)
        tag = f"[pp rank {group.rank()}/{group.size()} bounds {b}] "
        if bad:
            print(f"{tag}ranks start on shared layers {bad} (the runner refuses these bounds)")
        meta = PipelineShardMetadata(device_rank=group.rank(), world_size=group.size(),
                                     start_layer=b[group.rank()], end_layer=b[group.rank() + 1])
        gen = pipeline_auto_parallel(model, group, meta)
        for _ in gen:
            pass
    mx.eval(model.parameters())

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    ok = True
    x = mx.array(toks[None])
    full = np.array(model(x).astype(mx.float32))[0]
    r = rel(full, ref)
    top1 = float((full.argmax(-1) == ref.argmax(-1)).mean())
    print(f"{tag}1. full forward, no cache: rel_err={r:.2e} top1_agree={top1:.2f}")
    ok &= r < 1e-4

    cache = model.make_cache()
    c1, c2 = 12, 20
    rows = []
    model(x[:, :c1], cache=cache)
    rows.append(np.array(model(x[:, c1:c2], cache=cache).astype(mx.float32))[0])
    for t in range(c2, len(toks)):
        rows.append(np.array(model(x[:, t:t + 1], cache=cache).astype(mx.float32))[0])
    got = np.concatenate(rows, axis=0)
    want = ref[c1:]
    r2 = rel(got, want)
    per = [rel(got[i], want[i]) for i in range(len(want))]
    print(f"{tag}2. cached: chunk {c1}..{c2 - 1} + decode {c2}..{len(toks) - 1}: rel_err={r2:.2e} "
          f"worst_row={max(per):.2e} top1_agree={float((got.argmax(-1) == want.argmax(-1)).mean()):.2f}")
    ok &= r2 < 1e-4 and max(per) < 1e-4
    lat = cache[0][0].keys.shape
    print(f"   latent cache per layer: keys {tuple(lat)} values {tuple(cache[0][0].values.shape)}")
    print(f"{tag}ORACLE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
