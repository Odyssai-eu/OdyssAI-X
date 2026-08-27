"""Micro-config smoke for the kimi_k3 architecture module.

Runs on a node (needs Metal for the gated-delta kernel). Random weights, tiny
dims — this checks the wiring, not the numerics against the vendor:

  1. SiTU-GLU argument order (SwitchGLU passes up first, gate second)
  2. prefill forward
  3. cached step-by-step decode == prefill  (the AttnRes / KDA state contract)
  4. deepcopy -> identical logits

Usage on a node:
    ~/mlx-cluster/.venv/bin/python ~/mlx-cluster/test_kimi_k3_micro.py
"""

import copy
import sys

import mlx.core as mx

from mlx_lm.models.kimi_k3 import ModelArgs, Model, SituGLU

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def micro_args():
    # kda_layers is 1-INDEXED (like the real config): layers 0,1,3 are KDA,
    # layer 2 is MLA. attn_res_block_size 2 -> checkpoints at layers 0 and 2.
    return ModelArgs(
        model_type="kimi_k3",
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256,
        rms_norm_eps=1e-5,
        linear_attn_config={
            "kda_layers": [1, 2, 4],
            "full_attn_layers": [3],
            "num_heads": 4,
            "head_dim": 64,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
        num_experts=8,
        moe_intermediate_size=64,
        kv_lora_rank=32,
        head_dim=32,
        q_lora_rank=24,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        mla_use_nope=True,
        mla_use_output_gate=True,
        num_experts_per_token=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        attn_res_block_size=2,
        routed_expert_hidden_size=64,
        latent_moe_use_norm=True,
    )


def test_situ_order():
    """SwitchGLU calls activation(x_up, x_gate) — assert we honour that."""
    act = SituGLU(4.0, 25.0)
    gate = mx.array([[2.0]])
    up = mx.array([[3.0]])
    got = act(up, gate)  # up first, as SwitchGLU does

    b, lb = 4.0, 25.0
    g = float(gate[0, 0])
    u = float(up[0, 0])
    want = (
        b * mx.tanh(mx.array(g / b)) * mx.sigmoid(mx.array(g)) * lb
        * mx.tanh(mx.array(u / lb))
    )
    check(
        "SiTU argument order + formula",
        abs(float(got[0, 0]) - float(want)) < 1e-5,
        f"got {float(got[0, 0]):.6f} want {float(want):.6f}",
    )
    # Swapping the arguments must change the result, else the test is vacuous.
    check("SiTU is order-sensitive", abs(float(act(gate, up)[0, 0]) - float(got[0, 0])) > 1e-3)


def main():
    mx.random.seed(0)
    args = micro_args()
    model = Model(args)
    model.eval()
    mx.eval(model.parameters())

    test_situ_order()

    T = 16
    tokens = mx.random.randint(0, args.vocab_size, (1, T))

    full = model(tokens)
    mx.eval(full)
    check("prefill forward", full.shape == (1, T, args.vocab_size), str(full.shape))
    check("prefill finite", bool(mx.all(mx.isfinite(full))))

    # Cached decode: prefill T-1, then feed the last token alone. The final
    # logits must match the full prefill's last position.
    cache = model.make_cache()
    _ = model(tokens[:, :-1], cache=cache)
    step = model(tokens[:, -1:], cache=cache)
    mx.eval(step)

    d = float(mx.max(mx.abs(step[0, -1] - full[0, -1])))
    check("decode step == prefill (last position)", d < 2e-2, f"max|delta| = {d:.2e}")

    # Multi-step: 4 tokens one at a time from a fresh cache must reproduce the
    # same tail of the prefill.
    cache2 = model.make_cache()
    _ = model(tokens[:, : T - 4], cache=cache2)
    worst = 0.0
    for i in range(T - 4, T):
        out = model(tokens[:, i : i + 1], cache=cache2)
        mx.eval(out)
        worst = max(worst, float(mx.max(mx.abs(out[0, -1] - full[0, i]))))
    check("4-step incremental decode == prefill", worst < 2e-2, f"max|delta| = {worst:.2e}")

    clone = copy.deepcopy(model)
    clone.eval()
    cloned = clone(tokens)
    mx.eval(cloned)
    dd = float(mx.max(mx.abs(cloned - full)))
    check("deepcopy -> identical logits", dd == 0.0, f"max|delta| = {dd:.2e}")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} — {', '.join(FAILURES)}")
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
