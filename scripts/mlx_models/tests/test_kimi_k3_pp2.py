"""2-rank pipeline check for the kimi_k3 attention-residual transport.

Runs the same micro-model twice — once whole, once split across two ring ranks
on localhost — and requires the logits to match. What this actually exercises:

  * the (hidden, block_residual) tuple crossing a rank boundary as one payload
  * the recv template built at the payload width (a plain recv_like truncates)
  * the shard-LOCAL recompute of ssm_idx / attn_idx

Layout: 6 layers, alternating KDA / MLA, attn_res_block_size 2. Rank 1 starts at
global layer 3, so it must receive 2 depth checkpoints — the payload is 3x the
hidden width. Both shards hold at least one layer of each kind, which the
pipeline branch requires.

Launched the way Odysseus launches ranks (api.py:1619-1637) rather than through
mlx.launch: MLX_RANK + MLX_HOSTFILE, one process per rank, both on localhost.

    echo '[["127.0.0.1:5100"], ["127.0.0.1:5101"]]' > /tmp/ring_pp2.json
    for r in 0 1; do
      MLX_RANK=$r MLX_HOSTFILE=/tmp/ring_pp2.json \\
        ~/mlx-cluster/.venv/bin/python ~/mlx-cluster/test_kimi_k3_pp2.py &
    done; wait
"""

import sys

import mlx.core as mx

sys.path.insert(0, "/Users/admin/mlx-cluster")

from auto_parallel import pipeline_auto_parallel  # noqa: E402
from exo_stubs import PipelineShardMetadata  # noqa: E402
from mlx_lm.models.kimi_k3 import Model, ModelArgs  # noqa: E402

HIDDEN, VOCAB, T = 128, 512, 12


def micro_args():
    # kda_layers 1-INDEXED: layers 0,2,4 are KDA, layers 1,3,5 are MLA.
    return ModelArgs(
        model_type="kimi_k3",
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256,
        rms_norm_eps=1e-5,
        linear_attn_config={
            "kda_layers": [1, 3, 5],
            "full_attn_layers": [2, 4, 6],
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


def main():
    group = mx.distributed.init(backend="ring")
    rank, size = group.rank(), group.size()
    if size != 2:
        print(f"rank {rank}: expected world size 2, got {size}")
        sys.exit(1)

    # Same seed on both ranks -> identical weights, so each rank can compute the
    # single-process reference for itself before the model is sharded in place.
    mx.random.seed(1234)
    args = micro_args()
    model = Model(args)
    model.eval()
    mx.eval(model.parameters())

    tokens = mx.random.randint(0, VOCAB, (1, T), key=mx.random.key(7))
    reference = model(tokens)
    mx.eval(reference)

    per_rank = args.num_hidden_layers // size
    meta = PipelineShardMetadata(
        device_rank=rank,
        world_size=size,
        start_layer=rank * per_rank,
        end_layer=(rank + 1) * per_rank,
    )
    gen = pipeline_auto_parallel(model, group, meta)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        model = stop.value

    inner = model.model
    print(
        f"rank {rank}: shard [{meta.start_layer},{meta.end_layer}) "
        f"ssm_idx={inner.ssm_idx} attn_idx={inner.attn_idx} "
        f"recv_blocks={getattr(model.layers[0], 'attn_res_blocks', 0)}",
        flush=True,
    )

    out = model(tokens)
    mx.eval(out)

    delta = float(mx.max(mx.abs(out - reference)))
    ok = delta < 1e-2
    print(f"rank {rank}: max|pipeline - single| = {delta:.3e}  ->  {'OK' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
