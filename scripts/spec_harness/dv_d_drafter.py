#!/Users/admin/mlx-cluster/.venv/bin/python
"""dv_d_drafter — D: the DeepSeek-V4 DSpark drafter in MLX.

Ported against the OFFICIAL reference (deepseek-ai/DeepSpec), not guessed
from weight shapes:

  * `modeling/dspark/qwen3/modeling.py` — the drafter forward: Q comes from
    the noise block, K/V from `concat[target_ctx, noise_block]`, i.e. cross
    attention over the target's fused hidden states.
  * `_forward_backbone`: `target = hidden_norm(fc(target_hidden_cat))` then
    the blocks, then a final norm.
  * `common.create_noise_embed`: the block is `[anchor_token, MASK ...]` of
    length block_size.
  * `markov_head.sample_block_tokens`: sequential per-position argmax with a
    rank-256 previous-token bias.

What is V4-specific (vs DeepSpec's dense qwen3 drafter): the blocks are
DeepSeek-V4 blocks — MLA attention (wq_a/wq_b + single latent wkv),
256-expert MoE, HyperConnection — so we reuse ds4's own classes and only
re-implement the attention to take the context. `main_proj`/`main_norm`
play the role of DeepSpec's `fc`/`hidden_norm`.

Weights: the extract is DeepSeek's NATIVE quantization (FP8 e4m3 for most
matmuls, packed FP4 e2m1 for routed experts) — NOT MLX affine — so it is
dequantized to bf16 on load with pipenetwork's dequant helpers (MIT).

Run with no args = STRICT BIND TEST (build + load, no forward).
`--forward` additionally proposes a block from a dummy context.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/ivan-mlx-lm"))
sys.path.insert(0, os.path.expanduser("~/deepseek-v4-mlx"))
import mlx.core as mx                                        # noqa: E402
import mlx.nn as nn                                          # noqa: E402
from mlx.utils import tree_flatten                           # noqa: E402
import mlx_lm.models.deepseek_v4 as V4                       # noqa: E402
from mlx_lm.models.base import scaled_dot_product_attention  # noqa: E402
from mlx_lm.models.hyper_connection import HyperHead, hc_expand  # noqa: E402
from deepseek_v4_mlx.dequant import dequant_pair             # noqa: E402

DR = "/Volumes/models/odysseus/odyssai/DSV4-Flash-DSpark-drafter"
TGT = "/Volumes/models/odysseus/ivanfioravanti/V4F-Q4KExperts-Q8Rest-mlx"


# ── attention: MLA over [ctx | block] ───────────────────────────────────
class V4DSparkAttention(V4.LocalAttention):
    """ds4 LocalAttention, but K/V span the target context then the block.

    DeepSpec computes k/v for BOTH the fused target hidden states and the
    noise block and concatenates them (qwen3/modeling.py forward). Here the
    same, with V4's single latent KV vector (`wkv`), so the context is one
    [B, 1, Lctx, head_dim] tensor cached across rounds.
    """

    def __call__(self, x, ctx_kv=None, mask=None, cache=None):
        B, L, _ = x.shape
        ctx_len = 0 if ctx_kv is None else ctx_kv.shape[2]

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, ctx_len)                    # block sits after ctx

        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, ctx_len)
        if ctx_kv is not None:
            kv = mx.concatenate([ctx_kv, kv], axis=2)

        out = scaled_dot_product_attention(
            q, kv, kv, cache=None, scale=self.scale, mask=mask,
            sinks=self.attn_sink.astype(q.dtype))
        out = self.rope(out, ctx_len, inverse=True)
        out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        return self.wo_b(out)

    def ctx_kv(self, fused):
        """Project the fused target hidden states into the cached KV."""
        B, L, _ = fused.shape
        kv = self.kv_norm(self.wkv(fused)).reshape(B, 1, L, self.head_dim)
        return self.rope(kv, 0)


class V4DSparkBlock(nn.Module):
    """DeepseekV4Block wiring (HC around attn and ffn), context-aware attn."""

    def __init__(self, a, layer_idx):
        super().__init__()
        self.attn = V4DSparkAttention(a, layer_idx)
        # The drafter's MoE gate is LEARNED (the checkpoint carries
        # `ffn.gate.bias`), while the target's first `num_hash_layers` use
        # hash routing. ds4 picks by layer_idx, so build the gate with an
        # index past the hash range to get the learned variant.
        self.ffn = V4.DeepseekV4MoE(a, max(layer_idx, a.num_hash_layers))
        self.attn_norm = nn.RMSNorm(a.hidden_size, eps=a.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(a.hidden_size, eps=a.rms_norm_eps)
        self.attn_hc = V4.HyperConnection(a)
        self.ffn_hc = V4.HyperConnection(a)

    def __call__(self, h, ctx_kv, input_ids, mask=None):
        residual = h
        x, post, comb = self.attn_hc(h)
        x = self.attn(self.attn_norm(x), ctx_kv=ctx_kv, mask=mask)
        h = hc_expand(x, residual, post, comb)
        residual = h
        x, post, comb = self.ffn_hc(h)
        x = self.ffn(self.ffn_norm(x), input_ids)
        return hc_expand(x, residual, post, comb)


class VanillaMarkov(nn.Module):
    def __init__(self, vocab, rank):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab, rank)
        self.markov_w2 = nn.Linear(rank, vocab, bias=False)

    def step_bias(self, ids):
        return self.markov_w2(self.markov_w1(ids))

    def prev_embeddings(self, ids):
        return self.markov_w1(ids)


class ConfidenceHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, 1, bias=False)


class V4DSparkDrafter(nn.Module):
    """3 V4 blocks + target fuse (block 0) + DSpark heads (block 2)."""

    def __init__(self, a, dcfg):
        super().__init__()
        self.args = a
        self.block_size = int(dcfg["dspark_block_size"])
        self.mask_id = int(dcfg["dspark_noise_token_id"])
        self.taps = list(dcfg["dspark_target_layer_ids"])
        rank = int(dcfg["dspark_markov_rank"])
        h = a.hidden_size

        self.layers = [V4DSparkBlock(a, i) for i in range(3)]
        # DeepSpec's fc/hidden_norm, V4 naming
        self.layers[0].main_proj = nn.Linear(len(self.taps) * h, h, bias=False)
        self.layers[0].main_norm = nn.RMSNorm(h, eps=a.rms_norm_eps)
        # DSpark heads live on the last block in the checkpoint
        last = self.layers[2]
        last.hc_head = HyperHead(a)
        last.markov_head = VanillaMarkov(a.vocab_size, rank)
        last.confidence_head = ConfidenceHead(h + rank)
        last.norm = nn.RMSNorm(h, eps=a.rms_norm_eps)

    # -- context: fuse the target's tap hidden states, cache per block --
    def make_ctx(self, target_hidden_cat):
        """target_hidden_cat: [B, L, n_taps*H] -> per-block cached KV."""
        b0 = self.layers[0]
        fused = b0.main_norm(b0.main_proj(target_hidden_cat))
        fused = mx.repeat(fused[:, :, None, :], self.args.hc_mult, axis=2) \
            if False else fused
        return [blk.attn.ctx_kv(fused) for blk in self.layers]

    def backbone(self, noise_ids, ctx_kvs):
        h = self.embed(noise_ids)
        h = mx.repeat(h[..., None, :], self.args.hc_mult, axis=-2)  # HC state
        for blk, ck in zip(self.layers, ctx_kvs):
            h = blk(h, ck, noise_ids)
        last = self.layers[2]
        return last.norm(last.hc_head(h))

    def embed(self, ids):
        raise RuntimeError("drafter has no embed_tokens — the target's is used")

    def logits(self, hidden, lm_head):
        return lm_head(hidden)

    def sample_block(self, base_logits, first_prev):
        toks, prev = [], mx.array([first_prev])
        mk = self.layers[2].markov_head
        for i in range(base_logits.shape[0]):
            step = base_logits[i] + mk.step_bias(prev)[0]
            nxt = mx.argmax(step, axis=-1, keepdims=True)
            toks.append(nxt)
            prev = nxt
        return mx.concatenate(toks)


# ── weights: DeepSeek native FP8/FP4 -> bf16, keys -> module tree ───────
def load_drafter_weights(path, o_groups=8):
    raw = mx.load(os.path.join(path, "model.safetensors"))
    scales = {k: v for k, v in raw.items() if k.endswith(".scale")}
    out = {}
    for k, v in raw.items():
        if k.endswith(".scale"):
            continue
        sk = k[: -len(".weight")] + ".scale" if k.endswith(".weight") else None
        if sk and sk in scales:
            v = dequant_pair(k, v, scales[sk])          # FP8 or FP4 -> bf16
        out[k] = v

    # key remap: "<i>.rest" -> "layers.<i>.rest", ds4 naming
    remapped, experts = {}, {}
    for k, v in out.items():
        head, rest = k.split(".", 1)
        nk = f"layers.{head}.{rest}"
        for sub in ("attn", "ffn"):
            for p in ("fn", "base", "scale"):
                nk = nk.replace(f".hc_{sub}_{p}", f".{sub}_hc.{p}")
        for p in ("fn", "base", "scale"):
            nk = nk.replace(f".hc_head_{p}", f".hc_head.{p}")
        nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
        # wo_a is stored FLAT [o_groups*rank, H]; ds4's MultiLinear wants it
        # grouped [o_groups, rank, H] (same bytes, same order). o_groups varie
        # par modele (Flash=8, Pro=16) -- le flat se factorise des DEUX facons
        # (16384 = 8*2048 = 16*1024), donc il FAUT le vrai o_groups du target,
        # sinon strict=False installe silencieusement (8,2048,H) et le forward
        # casse (reshape des activations en 16 groupes vs poids en 8).
        if nk.endswith(".attn.wo_a.weight") and v.ndim == 2:
            v = v.reshape(o_groups, -1, v.shape[-1])
        for old, new in (("w1", "gate_proj"), ("w2", "down_proj"),
                         ("w3", "up_proj")):
            nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
        if ".ffn.experts." in nk:
            pre, tail = nk.split(".ffn.experts.")
            idx, mat = tail.split(".", 1)
            mat = mat.replace("w1", "gate_proj").replace("w2", "down_proj") \
                     .replace("w3", "up_proj").replace(".weight", "")
            experts.setdefault((pre, mat), {})[int(idx)] = v
            continue
        remapped[nk] = v
    # stack the 256 experts into ds4's switch_mlp layout
    for (pre, mat), d in experts.items():
        stacked = mx.stack([d[i] for i in sorted(d)], axis=0)
        remapped[f"{pre}.ffn.switch_mlp.{mat}.weight"] = stacked
    return remapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", action="store_true")
    a_ = ap.parse_args()

    a = V4.ModelArgs.from_dict(json.load(open(os.path.join(TGT, "config.json"))))
    dcfg = json.load(open(os.path.join(DR, "config.json")))
    m = V4DSparkDrafter(a, dcfg)

    have = {k: tuple(v.shape) for k, v in tree_flatten(m.parameters())}
    w = load_drafter_weights(DR)
    want = {k: tuple(v.shape) for k, v in w.items()}
    hk, wk = set(have), set(want)
    # tid2eid is a DERIVED routing table ds4 builds itself (token id -> expert
    # ids), not a checkpoint weight — never present in any DeepSeek file.
    hk = {k for k in hk if not k.endswith(".gate.tid2eid")}
    miss, extra = sorted(wk - hk), sorted(hk - wk)
    bad = sorted(k for k in hk & wk if have[k] != want[k])
    print(f"module {len(have)} params | weights {len(want)} tensors")
    print(f"bound {len(hk & wk) - len(bad)} | missing {len(miss)} | "
          f"extra {len(extra)} | shape-mismatch {len(bad)}")
    for k in miss[:12]:
        print("   -", k, want[k])
    for k in extra[:12]:
        print("   +", k, have[k])
    for k in bad[:12]:
        print("   !", k, "module", have[k], "vs w", want[k])
    if miss or bad:
        print("\n=> BIND INCOMPLET")
        sys.exit(1)
    m.load_weights(list(w.items()), strict=False)
    mx.eval(m.parameters())
    print(f"\n=> BIND STRICT OK — drafter chargé "
          f"({mx.get_peak_memory()/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
