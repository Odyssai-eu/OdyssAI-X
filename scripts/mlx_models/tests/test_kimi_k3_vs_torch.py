"""Cross-check the K3-specific components: MLX port vs the vendor's torch code.

The three things K3 adds that a reader can get wrong are checked against the
ACTUAL reference implementation (not a paraphrase), on REAL layer-1 weights
pulled out of the checkpoint:

  * SiTU-GLU               vs modeling_kimi_linear.SituAndMul
  * attention residuals    vs modeling_kimi_linear._apply_attn_res
  * MoE routing (noaux_tc) vs modeling_kimi_linear.KimiMoEGate.forward

The KDA kernels are NOT checked here — the reference needs fla-core/triton,
which does not run on Metal. That axis is covered by the Kimi-Linear 48B smoke.

Two stages, so neither side has to host the other's dependencies and the MLX
side runs on the node's real mlx-lm:

  stage 1 (Mac, torch):   --emit-ref  --ref <vendor file> --npz <probe> --out <bundle>
  stage 2 (node, mlx):    --check     --bundle <bundle>

Inputs are generated once in stage 1 and carried in the bundle, so both stages
see byte-identical inputs.
"""

import argparse
import importlib.util
import sys
import types

import numpy as np

FAILURES = []
P = "language_model.model.layers.1"
BETA, LINEAR_BETA = 4.0, 25.0
TOP_K = 16
EPS = 1e-5


def check(name, delta, tol):
    ok = delta <= tol
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  max|delta| = {delta:.3e}  (tol {tol:.0e})")
    if not ok:
        FAILURES.append(name)


# ── stage 1: torch reference ─────────────────────────────────────────────────


def load_reference(path):
    """Import the vendor module with its unsatisfiable deps stubbed out.

    modeling_kimi_linear.py hard-imports fla (triton kernels) and a relative
    config module. Neither is needed for the pure functions exercised here, so
    they are stubbed before execution. Nothing in the code paths under test is
    replaced.
    """
    for name in (
        "fla", "fla.modules", "fla.ops", "fla.ops.kda", "fla.ops.utils",
        "fla.ops.utils.index", "fla.utils",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["fla.modules"].FusedRMSNormGated = object
    sys.modules["fla.modules"].ShortConvolution = object
    sys.modules["fla.ops.kda"].chunk_kda = None
    sys.modules["fla.ops.kda"].fused_recurrent_kda = None
    sys.modules["fla.ops.utils.index"].prepare_cu_seqlens_from_mask = None
    sys.modules["fla.ops.utils.index"].prepare_lens_from_mask = None
    sys.modules["fla.utils"].tensor_cache = lambda f: f

    pkg = types.ModuleType("k3ref")
    pkg.__path__ = []
    sys.modules["k3ref"] = pkg
    cfg = types.ModuleType("k3ref.configuration_kimi_k3")
    cfg.KimiLinearConfig = object
    sys.modules["k3ref.configuration_kimi_k3"] = cfg

    spec = importlib.util.spec_from_file_location("k3ref.modeling_kimi_linear", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["k3ref.modeling_kimi_linear"] = mod
    spec.loader.exec_module(mod)
    return mod


def emit_ref(ref_path, npz_path, out_path):
    import torch

    ref = load_reference(ref_path)
    npz = np.load(npz_path)
    rng = np.random.default_rng(0)
    bundle = {}

    def bf16(key):
        raw = npz[key + "|bf16"]
        t = torch.from_numpy(raw.copy()).view(torch.bfloat16)
        return t.float().numpy(), t

    # 1. SiTU-GLU
    x = (rng.standard_normal((7, 512)) * 3.0).astype(np.float32)
    bundle["situ_in"] = x
    bundle["situ_ref"] = ref.SituAndMul(beta=BETA, linear_beta=LINEAR_BETA)(
        torch.from_numpy(x)
    ).numpy()

    # 2. attention residuals, real layer-1 scorers
    for stem in ("self_attention_res", "mlp_res"):
        nw, nw_t = bf16(f"{P}.{stem}_norm.weight")
        pw, pw_t = bf16(f"{P}.{stem}_proj.weight")
        H = nw.shape[0]
        prefix = (rng.standard_normal((5, H)) * 0.5).astype(np.float32)
        blk = (rng.standard_normal((5, 3, H)) * 0.5).astype(np.float32)

        norm_t = torch.nn.Module()
        norm_t.weight = torch.nn.Parameter(nw_t.float())
        norm_t.variance_epsilon = EPS
        proj_t = torch.nn.Module()
        proj_t.weight = torch.nn.Parameter(pw_t.float())

        bundle[f"{stem}_norm_w"] = nw
        bundle[f"{stem}_proj_w"] = pw
        bundle[f"{stem}_prefix"] = prefix
        bundle[f"{stem}_blocks"] = blk
        with torch.no_grad():
            bundle[f"{stem}_ref"] = ref._apply_attn_res(
                torch.from_numpy(prefix), torch.from_numpy(blk), proj_t, norm_t
            ).numpy()

    # 3. MoE routing, real layer-1 router
    gw, gw_t = bf16(f"{P}.block_sparse_moe.gate.weight")
    bias = npz[f"{P}.block_sparse_moe.gate.e_score_correction_bias|f32"]
    hidden = (rng.standard_normal((1, 6, gw.shape[1])) * 0.05).astype(np.float32)

    gate_t = ref.KimiMoEGate.__new__(ref.KimiMoEGate)
    torch.nn.Module.__init__(gate_t)
    gate_t.top_k, gate_t.num_experts = TOP_K, gw.shape[0]
    gate_t.routed_scaling_factor = 1.0
    gate_t.moe_router_activation_func = "sigmoid"
    gate_t.num_expert_group, gate_t.topk_group = 1, 1
    gate_t.moe_renormalize = True
    gate_t.training = False
    gate_t.weight = torch.nn.Parameter(gw_t.float())
    gate_t.e_score_correction_bias = torch.nn.Parameter(torch.from_numpy(bias))
    idx_t, w_t = gate_t(torch.from_numpy(hidden))

    bundle["moe_gate_w"] = gw
    bundle["moe_bias"] = bias
    bundle["moe_hidden"] = hidden
    bundle["moe_ref_idx"] = idx_t.numpy()
    bundle["moe_ref_w"] = w_t.detach().numpy()

    np.savez(out_path, **bundle)
    print(f"wrote reference bundle: {out_path} ({len(bundle)} arrays)")


# ── stage 2: mlx under test ──────────────────────────────────────────────────


def run_check(bundle_path):
    import mlx.core as mx
    from mlx_lm.models.kimi_k3 import SituGLU, _apply_attn_res, _group_expert_select

    b = np.load(bundle_path)

    print("1. SiTU-GLU (beta=4.0, linear_beta=25.0)")
    x = b["situ_in"]
    gate, up = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 :]
    got = np.array(SituGLU(BETA, LINEAR_BETA)(mx.array(up), mx.array(gate)).astype(mx.float32))
    check("SituAndMul", float(np.abs(got - b["situ_ref"]).max()), 1e-4)

    print("2. attention residuals (_apply_attn_res, real layer-1 scorers)")
    for stem, tag in (("self_attention_res", "self-attn"), ("mlp_res", "mlp")):
        norm_m = types.SimpleNamespace(weight=mx.array(b[f"{stem}_norm_w"]))
        proj_m = types.SimpleNamespace(weight=mx.array(b[f"{stem}_proj_w"]))
        got = np.array(
            _apply_attn_res(
                mx.array(b[f"{stem}_prefix"]),
                mx.array(b[f"{stem}_blocks"]),
                proj_m,
                norm_m,
                EPS,
            ).astype(mx.float32)
        )
        check(f"_apply_attn_res ({tag})", float(np.abs(got - b[f"{stem}_ref"]).max()), 1e-4)

    print("3. MoE routing noaux_tc (real layer-1 router, 896 experts, top-16)")
    logits = mx.array(b["moe_hidden"]) @ mx.array(b["moe_gate_w"]).T
    inds_m, w_m = _group_expert_select(
        logits, mx.array(b["moe_bias"]), TOP_K, 1, 1, 1.0, True, "sigmoid"
    )
    inds_m = np.array(inds_m).reshape(-1, TOP_K)
    w_m = np.array(w_m.astype(mx.float32)).reshape(-1, TOP_K)
    idx_np, w_np = b["moe_ref_idx"], b["moe_ref_w"]

    same = all(set(idx_np[i]) == set(inds_m[i]) for i in range(idx_np.shape[0]))
    print(f"  [{'OK ' if same else 'FAIL'}] selected expert SET identical (top-{TOP_K})")
    if not same:
        FAILURES.append("moe expert selection")
        for i in range(idx_np.shape[0]):
            diff = set(idx_np[i]) ^ set(inds_m[i])
            if diff:
                print(f"      token {i}: symmetric diff {sorted(diff)}")

    dw = 0.0
    for i in range(idx_np.shape[0]):
        ref_map = dict(zip(idx_np[i].tolist(), w_np[i].tolist()))
        for e, w in zip(inds_m[i].tolist(), w_m[i].tolist()):
            if e in ref_map:
                dw = max(dw, abs(ref_map[e] - w))
    check("routing weights (renormalised)", dw, 1e-5)

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} — {', '.join(FAILURES)}")
        sys.exit(1)
    print("ALL OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-ref", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ref")
    ap.add_argument("--npz")
    ap.add_argument("--out")
    ap.add_argument("--bundle")
    a = ap.parse_args()

    if a.emit_ref:
        emit_ref(a.ref, a.npz, a.out)
    elif a.check:
        run_check(a.bundle)
    else:
        ap.error("pass --emit-ref or --check")


if __name__ == "__main__":
    main()
