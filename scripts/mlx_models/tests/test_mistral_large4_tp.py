"""Tensor-parallel check for mistral_large4.shard().

Builds one micro-model (8 layers, no-rope on 3 and 7, layer 7 reading layer 3's
latent like 39..71 read 35, MoE with a shared expert, quantized Q4 gs32 with the
module's predicate), runs it whole, shards it, runs it again and requires the
same logits on the prefill and on four greedy decode steps.

    mlx.launch --backend ring -n 4 --hosts 127.0.0.1 \\
      --python ~/mlx-cluster/.venv/bin/python test_mistral_large4_tp.py

2026-10-31 on .29: max |ref - tp| 2.4e-6 (TP2) and 1.7e-6 (TP4), same argmax.
"""
from mlx_lm.models import mistral_large4 as M
from mlx_lm.models.cache import make_prompt_cache

group = mx.distributed.init()
r, N = group.rank(), group.size()
nl = 8
args = M.ModelArgs.from_dict(dict(
    model_type="mistral_large4", vocab_size=1000, hidden_size=512, intermediate_size=1024,
    moe_intermediate_size=256, num_hidden_layers=nl, num_attention_heads=8,
    n_routed_experts=8, num_experts_per_tok=2, kv_lora_rank=64, qk_rope_head_dim=16,
    qk_nope_head_dim=16, v_head_dim=32,
    kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 3]))
mx.random.seed(0)
model = M.Model(args)
for l in model.layers:
    a = l.self_attn
    a.ssmax_scale = mx.random.uniform(0, 0.2, a.ssmax_scale.shape)
    a.ssmax_bias = mx.random.uniform(0.8, 1.2, a.ssmax_bias.shape)
model.model.rotary.freqs = 10000 ** (mx.arange(0, 16, 2, dtype=mx.float32) / 16)
nn.quantize(model, group_size=32, bits=4,
            class_predicate=lambda p, m: hasattr(m, "to_quantized") and model.quant_predicate(p, m))
mx.eval(model.parameters())

tok = mx.array([[1, 17, 42, 99, 5, 300, 7]])
def run():
    cache = make_prompt_cache(model)
    outs = [model(tok, cache=cache)[:, -1]]
    nxt = mx.argmax(outs[-1], -1)[:, None]
    for _ in range(4):
        outs.append(model(nxt, cache=cache)[:, -1])
        nxt = mx.argmax(outs[-1], -1)[:, None]
    return mx.stack(outs).astype(mx.float32)

ref = run(); mx.eval(ref)
model.shard(group)
tp = run(); mx.eval(tp)
d = mx.abs(ref - tp).max().item()
print(f"rank {r}/{N} max|ref-tp|={d:.3e} ref_absmax={mx.abs(ref).max().item():.3f} "
      f"argmax_equal={bool(mx.all(mx.argmax(ref,-1)==mx.argmax(tp,-1)).item())}", flush=True)
