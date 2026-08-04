#!/Users/admin/mlx-cluster/.venv/bin/python
"""Debug shape du forward drafter Pro (_attn), single-node, contexte factice.
Le target Pro ne tient pas single-node -> on ne charge QUE le drafter et on
exerce _attn avec des tenseurs bidon (le bug est purement une shape)."""
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/ivan-mlx-lm"))
sys.path.insert(0, os.path.expanduser("~/deepseek-v4-mlx"))
sys.path.insert(0, os.path.expanduser("~"))
import mlx.core as mx
import mlx_lm.models.deepseek_v4 as V4
from dv_d_drafter import V4DSparkDrafter, load_drafter_weights

TGT="/Volumes/models/odysseus/odyssai/DSV4-Pro-q4kx-q8"
DR=os.environ.get("DRAFT_DIR","/Volumes/models/odysseus/odyssai/DSV4-Pro-DSpark-drafter")
a=V4.ModelArgs.from_dict(json.load(open(TGT+"/config.json")))
dcfg=json.load(open(DR+"/config.json"))
print(f"a: o_groups={a.o_groups} n_heads={a.num_attention_heads} head_dim={a.head_dim} o_lora={a.o_lora_rank} hidden={a.hidden_size}")
d=V4DSparkDrafter(a, dcfg)
if dcfg.get("prequantized"):
    import mlx.nn as _nn
    _nn.quantize(d, group_size=dcfg["quantization"]["group_size"],
                 bits=dcfg["quantization"]["bits"])
    d.load_weights(os.path.join(DR,"model.safetensors"), strict=False)
    mx.eval(d.parameters())
    print(f"drafter charge Q{dcfg['quantization']['bits']}, peak {mx.get_peak_memory()/1e9:.1f} GB")
else:
    d.load_weights(list(load_drafter_weights(DR, o_groups=a.o_groups).items()), strict=False)
    mx.eval(d.parameters())
    print(f"drafter charge bf16, peak {mx.get_peak_memory()/1e9:.1f} GB")

blk=d.layers[0]; attn=blk.attn
print(f"attn.o_groups={attn.o_groups} attn.n_heads={attn.n_heads} attn.head_dim={attn.head_dim}")
print(f"wo_a.weight.shape={attn.wo_a.weight.shape}  wo_b.weight.shape={attn.wo_b.weight.shape}")

# contexte factice : 1 token de ctx, bloc de L=5
B,L=1,5
x=mx.random.normal((B,L,a.hidden_size)).astype(mx.bfloat16)
off=1
# reproduit _attn de dv_g avec prints
q=attn.wq_b(attn.q_norm(attn.wq_a(x))).reshape(B,L,attn.n_heads,attn.head_dim)
q=mx.fast.rms_norm(q,None,a.rms_norm_eps).transpose(0,2,1,3); q=attn.rope(q,off)
kv=attn.kv_norm(attn.wkv(x)).reshape(B,1,L,attn.head_dim); kv=attn.rope(kv,off)
k=kv; v=k
from mlx_lm.models.base import scaled_dot_product_attention as sdpa
out=sdpa(q,k,v,cache=None,scale=attn.scale,mask=None,sinks=attn.attn_sink.astype(q.dtype))
mx.eval(out); print(f"sdpa out={out.shape}  (attendu B,n_heads,L,head_dim)")
out=attn.rope(out,off,inverse=True)
o1=out.reshape(B,attn.o_groups,-1,L,attn.head_dim); print(f"reshape(o_groups)={o1.shape}")
o2=o1.transpose(0,1,3,2,4).flatten(-2); print(f"transpose+flatten={o2.shape}  -> wo_a attend last={attn.wo_a.weight.shape}")
try:
    o3=attn.wo_a(o2); mx.eval(o3); print(f"wo_a OK -> {o3.shape}")
except Exception as e:
    print(f"wo_a CRASH: {e}")
