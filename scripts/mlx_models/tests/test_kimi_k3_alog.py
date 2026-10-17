"""KDA A_log is one value per head; the checkpoint pads it to head_dim (#141).

On moonshotai/Kimi-K3 the stored A_log is [128] with num_heads=96: the first 96
values are trained, the last 32 are exactly 0. The module must read only the
first num_heads entries, one per head, broadcast over head_dim. Random weights,
tiny dims (same micro config as test_kimi_k3_micro.py).

Usage on a node (needs Metal for the gated-delta kernel):
    ~/mlx-cluster/.venv/bin/python ~/mlx-cluster/test_kimi_k3_alog.py
"""

import sys

import mlx.core as mx

from mlx_lm.models.kimi_k3 import KimiK3DeltaAttention, Model

from test_kimi_k3_micro import micro_args

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main():
    mx.random.seed(0)
    model = Model(micro_args())
    mx.eval(model.parameters())
    kda = next(m for _, m in model.named_modules() if isinstance(m, KimiK3DeltaAttention))
    H, D = kda.num_heads, kda.head_dim
    check("A_log keeps the stored (padded) shape", tuple(kda.A_log.shape) == (D,), str(kda.A_log.shape))

    x = mx.random.normal((1, 5, model.args.hidden_size)).astype(mx.float32)
    heads = mx.array([0.3, -0.2, 0.9, 0.1], dtype=mx.float32)[:H]

    def run(padding_value, heads_values):
        kda.A_log = mx.concatenate([heads_values, mx.full((D - H,), padding_value, mx.float32)])
        out = kda(x)
        mx.eval(out)
        return out

    base = run(0.0, heads)
    same = run(99.0, heads)
    check("padding entries do not change the output",
          mx.allclose(base, same, atol=1e-6).item(),
          f"max diff {mx.abs(base - same).max().item():.2e}")
    bumped = heads.at[1].add(1.5)
    moved = run(0.0, bumped)
    check("a head's own value does change the output",
          not mx.allclose(base, moved, atol=1e-6).item())


if __name__ == "__main__":
    main()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all OK")
