"""3-rank pipeline check for mistral_large4's cross-layer KV sharing.

Runs the same micro-model twice -- once whole, once split across three ring
ranks on localhost -- and requires the logits to match, on the prefill and on
three decode steps. Layout mirrors the real checkpoint at 1/6 scale: 12 layers,
no-rope every 4th (3, 7, 11), layers 7 and 11 read layer 3's latent the way
39..71 read 35's. What it exercises:

  * the source latent crossing one boundary (split 4/4/4: layer 3 on the top
    rank, its readers on the two others) and two boundaries in a row, the
    middle rank forwarding from its mirror;
  * the source on the same rank as a reader (bounds 0,4,9,12 = 4/5/3 layers
    from rank 0: rank 2 runs 0..2, rank 1 runs 3..7, rank 0 runs 8..11);
  * offset and mask taken from a layer the rank owns (cache[0] is never
    written on a downstream rank).

Launched the way Odysseus launches ranks (MLX_RANK + MLX_HOSTFILE), one process
per rank on one node:

    echo '[["127.0.0.1:5200"], ["127.0.0.1:5201"], ["127.0.0.1:5202"]]' > /tmp/ring_ml4.json
    for b in 0,4,8,12 0,4,9,12; do
      for r in 0 1 2; do
        RUNNER_LAYER_BOUNDS=$b MLX_RANK=$r MLX_HOSTFILE=/tmp/ring_ml4.json \\
          ~/mlx-cluster/.venv/bin/python ~/mlx-cluster/test_mistral_large4_pp.py &
      # and again with EVAL_CACHE=0
      done; wait
    done
"""

import os
import sys

import mlx.core as mx

sys.path.insert(0, "/Users/admin/mlx-cluster")

from mlx_lm.models.mistral_large4 import Model, ModelArgs  # noqa: E402
# The runner patches PipelineMixin.pipeline at startup: run under the same
# patch (an earlier version of it dropped the `split` argument).
from patches.pipeline_split_coverage import apply_pipeline_split_fix  # noqa: E402

apply_pipeline_split_fix()

VOCAB, T, STEPS = 512, 10, 3
N = 12


def micro_args():
    src = list(range(N))
    src[7] = src[11] = 3
    return ModelArgs(
        model_type="mistral_large4",
        vocab_size=VOCAB,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=N,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=8,
        num_experts_per_tok=2,
        kv_lora_rank=32,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=32,
        kv_source_layers=src,
        rope_parameters={"rope_theta": 10000.0, "factor": 1.0},
    )


def build():
    mx.random.seed(0)
    m = Model(micro_args())
    # Non-zero ssmax so the per-position query scaling is exercised.
    for layer in m.model.layers:
        layer.self_attn.ssmax_scale = mx.random.normal((4,)) * 0.1
        layer.self_attn.ssmax_bias = 1.0 + mx.random.normal((4,)) * 0.1
    mx.eval(m.parameters())
    return m


def run(model):
    cache = model.make_cache()
    mx.random.seed(1)
    toks = mx.random.randint(0, VOCAB, (1, T))
    # One eval per forward, like the runner: a graph spanning several
    # send/recv rounds waits on peers inside one command buffer (GPU timeout).
    # EVAL_CACHE=0 evaluates the logits only, as mlx_lm's generate_step does.
    def ev(x):
        if os.environ.get("EVAL_CACHE", "1") == "1":
            mx.eval(x, [c.state for c in cache if c is not None and c.keys is not None])
        else:
            mx.eval(x)

    outs = [model(toks, cache=cache)[:, -1, :]]
    ev(outs[-1])
    for _ in range(STEPS):
        nxt = mx.argmax(outs[-1], axis=-1)[:, None]
        outs.append(model(nxt, cache=cache)[:, -1, :])
        ev(outs[-1])
    return outs


group = mx.distributed.init(backend="ring")
rank = group.rank()

ref = run(build())

pp = build()
pp.model.pipeline(group)
got = run(pp)

bounds = os.environ.get("RUNNER_LAYER_BOUNDS", "even")
span = (pp.model.start_idx, pp.model.start_idx + len(pp.model.pipeline_layers))
diffs = [float(mx.abs(a - b).max()) for a, b in zip(ref, got)]
ok = all(d < 1e-4 for d in diffs)
print(f"rank {rank} ref checksum={float(ref[0].sum()):.4f} bounds={bounds} layers [{span[0]}, {span[1]}) "
      f"max|diff| per step={['%.2e' % d for d in diffs]} {'OK' if ok else 'FAIL'}",
      flush=True)
sys.exit(0 if ok else 1)
