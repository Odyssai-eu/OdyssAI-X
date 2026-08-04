#!/Users/admin/mlx-cluster/.venv/bin/python
"""dv_e — GOAL: DeepSeek-V4 Flash + DSpark drafter, speculative, DISTRIBUTED.

The full thing: the real V4 drafter (accept 3.83 measured single-node)
proposes; the distributed V4 target (rank0+rank1, ds4 pipeline) verifies;
accepted prefix committed, rejected tail rolled back (snapshot/restore,
since V4 PoolingCache is not trimmable); the drafter's context grows
incrementally with the committed tokens' fused tap hidden (dspartha's
CtxCache convention).

Pipeline (ds4 convention): rank0 owns the LAST layers -> rank0 is the
DRIVER and the taps 40/41/42 are LOCAL to it, so no extra collective for
the tap capture. rank1 is a servant: it runs its shard on every broadcast
block ([op,S,ids...] via all_sum, A0 pattern) and mirrors the KV rollback.
The drafter lives entirely on rank0 (53.7 GB beside its 90 GB shard).

Gate: ids == plain distributed (dv_c1), same sharding (near-tie fp flips
excluded). Speedup vs the 33.5 tok/s plain-dist baseline.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/ivan-mlx-lm"))
sys.path.insert(0, os.path.expanduser("~/deepseek-v4-mlx"))
sys.path.insert(0, os.path.expanduser("~"))
import mlx.core as mx                                        # noqa: E402
from mlx_lm import load                                      # noqa: E402
from mlx_lm.models.base import create_attention_mask         # noqa: E402
from mlx_lm.models.cache import CacheList                    # noqa: E402
import mlx_lm.models.deepseek_v4 as V4                       # noqa: E402
from mlx_lm.models.pipeline import PipelineMixin              # noqa: E402


def _pipeline_fixed(self, group):
    """Correctif amont : sans lui des couches ne sont calculees par AUCUN
    rang des que num_layers % world_size != 0 (Pro 61L/5r perd 48..51,
    Flash 43L/2r perd la 21). Panne silencieuse : le texte reste fluide.
    Ce script importe mlx-lm en direct, il ne passe donc PAS par le
    scripts/patches/ du runner -- d'ou la copie ici."""
    self.pipeline_rank = group.rank()
    self.pipeline_size = group.size()
    n = len(self.layers)
    base, extra = divmod(n, self.pipeline_size)
    sizes = [base + (1 if r < extra else 0) for r in range(self.pipeline_size)]
    end = n - sum(sizes[: self.pipeline_rank])
    self.start_idx = end - sizes[self.pipeline_rank]
    self.end_idx = end
    self.layers = self.layers[: self.end_idx]
    self.layers[: self.start_idx] = [None] * self.start_idx


PipelineMixin.pipeline = _pipeline_fixed
# The drafter (and its FP8/FP4 dequant chain from deepseek_v4_mlx) lives
# ONLY on rank0. Importing it at top-level forced rank1 — a pure target
# servant — to carry the whole drafter dependency tree; guard it so the
# servant node needs nothing but ds4.
DR = "/Volumes/models/odysseus/odyssai/DSV4-Pro-DSpark-drafter-q8"
TGT = "/Volumes/models/odysseus/odyssai/DSV4-Pro-q4kx-q8"

MAXS, HDR = 2048, 3           # MAXS covers the prefill prompt, not just K
OP_RUN, OP_STOP = 0, 1


# ── drafter context cache (dspartha's CtxCache, append-only) ────────────
class CtxCache:
    __slots__ = ("k", "v")

    def __init__(self):
        self.k = self.v = None

    def append(self, k, v):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = mx.concatenate([self.k, k], axis=2)
            self.v = mx.concatenate([self.v, v], axis=2)

    @property
    def length(self):
        return 0 if self.k is None else self.k.shape[2]


def collapse_mean(t):                       # 4D HC state -> 2D
    return t.mean(axis=-2)


class DrafterRunner:
    """Wraps V4DSparkDrafter with the incremental cross-attention ctx that
    dspartha's loop needs: fuse committed target taps -> per-block CtxCache,
    then draft a block attending [ctx | block]."""

    def __init__(self, drafter, a):
        self.d = drafter
        self.a = a
        self.hc = a.hidden_size

    def make_ctx(self):
        return [CtxCache() for _ in self.d.layers]

    def update_ctx(self, target_hidden_cat, caches):
        """target_hidden_cat: [B, L, n_taps*H] committed positions -> append
        their projected K/V to each block's ctx cache (dspartha update_context)."""
        b0 = self.d.layers[0]
        fused = b0.main_norm(b0.main_proj(target_hidden_cat))     # [B,L,H]
        off = caches[0].length
        for blk, c in zip(self.d.layers, caches):
            attn = blk.attn
            B, L, _ = fused.shape
            kv = attn.kv_norm(attn.wkv(fused)).reshape(B, 1, L, attn.head_dim)
            kv = attn.rope(kv, off)
            c.append(kv, kv)

    def _attn(self, attn, x, ctx_c):
        B, L, _ = x.shape
        off = ctx_c.length
        q = attn.wq_b(attn.q_norm(attn.wq_a(x))).reshape(B, L, attn.n_heads, attn.head_dim)
        q = mx.fast.rms_norm(q, None, self.a.rms_norm_eps).transpose(0, 2, 1, 3)
        q = attn.rope(q, off)
        kv = attn.kv_norm(attn.wkv(x)).reshape(B, 1, L, attn.head_dim)
        kv = attn.rope(kv, off)
        k = mx.concatenate([ctx_c.k, kv], axis=2) if ctx_c.k is not None else kv
        v = k
        from mlx_lm.models.base import scaled_dot_product_attention as sdpa
        out = sdpa(q, k, v, cache=None, scale=attn.scale, mask=None,
                   sinks=attn.attn_sink.astype(q.dtype))
        out = attn.rope(out, off, inverse=True)
        out = out.reshape(B, attn.o_groups, -1, L, attn.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = attn.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        return attn.wo_b(out)

    def propose(self, pending, caches, embed, lm_head, K, MASK):
        from mlx_lm.models.hyper_connection import hc_expand
        noise = mx.array([[pending] + [MASK] * (K - 1)])
        h = embed(noise)
        h = mx.repeat(h[..., None, :], self.a.hc_mult, axis=-2)
        for blk, c in zip(self.d.layers, caches):
            residual = h
            x, post, comb = blk.attn_hc(h)
            x = self._attn(blk.attn, blk.attn_norm(x), c)
            h = hc_expand(x, residual, post, comb)
            residual = h
            x, post, comb = blk.ffn_hc(h)
            x = blk.ffn(blk.ffn_norm(x), noise)
            h = hc_expand(x, residual, post, comb)
        last = self.d.layers[2]
        base = lm_head(last.norm(last.hc_head(h)))[0]
        return [int(x) for x in self.d.sample_block(base, pending).tolist()]


# ── distributed target forward with tap capture (rank0 = driver) ────────
def make_forward(model, taps):
    mm = model.model

    def fwd(inputs, cache):
        h = mm.embed_tokens(inputs)
        h = mx.broadcast_to(h[:, :, None, :],
                            (h.shape[0], h.shape[1], mm.args.hc_mult, h.shape[2]))
        h = mx.contiguous(h)
        pr, ps = mm.pipeline_rank, mm.pipeline_size
        lock = ps > 2      # ps>2 : chaine de collectives lazy -> deadlock
        first = cache[0]
        mc = first[0] if isinstance(first, CacheList) else first
        mask = create_attention_mask(h[:, :, 0, :], mc,
                                     window_size=mm.args.sliding_window,
                                     return_array=True)
        if pr < ps - 1:
            h = mx.distributed.recv_like(h, pr + 1)
            if lock:
                mx.eval(h)
        caps = {}
        for j, (layer, lc) in enumerate(zip(mm.pipeline_layers, cache)):
            h = layer(h, mask, lc, inputs)
            gid = mm.start_idx + j
            if gid in taps:
                caps[gid] = collapse_mean(h)
        if lock:
            mx.eval(h, *caps.values())
        if pr != 0:
            h = mx.distributed.send(h, (pr - 1) % ps)
            ci = cache[-1]
            ci = ci[0] if isinstance(ci, CacheList) else ci
            if ci is not None:
                ci.keys = mx.depends(ci.keys, h)
            if lock:
                mx.eval(h)
        if ps > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]
            if lock:
                mx.eval(h)
        return model.lm_head(mm.norm(mm.hc_head(h))), caps
    return fwd


def snap(cache):
    out = []
    for c in cache:
        if isinstance(c, CacheList):
            out.append(("l", [dict(vars(x)) for x in c.caches]))
        else:
            out.append(("o", dict(vars(c))))
    return out


def restore(cache, s):
    for c, (k, st) in zip(cache, s):
        if k == "l":
            for sub, d in zip(c.caches, st):
                sub.__dict__.update(d)
        else:
            c.__dict__.update(st)


def bcast(world, rank, vec=None):
    if rank == 0:
        arr = mx.array(vec + [0] * (HDR + MAXS - len(vec)), dtype=mx.int32)
    else:
        arr = mx.zeros((HDR + MAXS,), dtype=mx.int32)
    out = mx.distributed.all_sum(arr, group=world)
    mx.eval(out)
    return [int(x) for x in out.tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Explain photosynthesis in 3 sentences.")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--drafter-bits", type=int, default=0,
                    help="0 = bf16 (E baseline), 8 or 4 = quantise le drafter")
    ap.add_argument("--drafter", default=None, help="override du dir drafter")
    ap.add_argument("--target", default=None, help="override du dir target")
    args = ap.parse_args()

    global DR, TGT
    if args.drafter:
        DR = args.drafter
    if args.target:
        TGT = args.target

    world = mx.distributed.init(backend=os.environ.get("A0_BACKEND", "any"))
    rank, size = world.rank(), world.size()
    assert size >= 2

    model, tok = load(TGT, lazy=True)
    model.model.pipeline(world)
    mx.eval(model.parameters())
    a = V4.ModelArgs.from_dict(json.load(open(os.path.join(TGT, "config.json"))))
    # Everything drafter-side (config, taps, K, MASK) is rank0-only. rank1 is
    # a pure target servant: its shard holds the LOW layers, so it never owns
    # a tap layer (40/41/42) and captures nothing — empty taps is correct and
    # keeps the drafter checkpoint dir off the servant node entirely.
    if rank == 0:
        dcfg = json.load(open(os.path.join(DR, "config.json")))
        taps = list(dcfg["dspark_target_layer_ids"])
        K, MASK = int(dcfg["dspark_block_size"]), int(dcfg["dspark_noise_token_id"])
    else:
        taps = []
    fwd = make_forward(model, set(taps))
    _mm = model.model
    print(f"r{rank}: couches {_mm.start_idx}..{_mm.end_idx - 1} "
          f"({len(_mm.pipeline_layers)}) | target peak "
          f"{mx.get_peak_memory()/1e9:.0f} GB", flush=True)

    # ── rank1: servant ──
    # Mirrors rank0's KV bookkeeping exactly: before a normal verify (flag 0)
    # it checkpoints; on a re-commit (flag 1, sent only after a rejected
    # verify) it first rolls back to that checkpoint, then runs the shorter
    # forward — so both ranks' caches stay identical round to round.
    if rank != 0:
        cache = model.make_cache()
        servant_snap = None
        while True:
            v = bcast(world, rank)
            op, S, flag = v[0], v[1], v[2]
            print(f"r{rank} recv op={op} S={S} flag={flag}",
                  file=sys.stderr, flush=True)
            if op == OP_STOP:
                break
            if flag:
                restore(cache, servant_snap)
            else:
                servant_snap = snap(cache)
            o, _ = fwd(mx.array([v[HDR:HDR + S]], dtype=mx.int32), cache)
            mx.eval(o)          # lockstep: run collectives NOW, once
        print(f"r{rank}: servant done, peak {mx.get_peak_memory()/1e9:.0f} GB")
        return

    # ── rank0: driver (drafter + loop) ──
    import mlx.nn as _nn                                    # noqa: E402
    from dv_d_drafter import V4DSparkDrafter                # rank0 only
    drafter = V4DSparkDrafter(a, dcfg)
    print("r0: ctor OK", flush=True)

    if dcfg.get("prequantized"):
        # Checkpoint deja quantise (dv_q8_save) : quantifier le SQUELETTE puis
        # charger directement. Evite la chaine de dequant FP8/FP4 -- pic ~208 GB
        # et dependance pipenetwork -- que le chemin brut paie a CHAQUE run.
        qb = int(dcfg["quantization"]["bits"])
        qg = int(dcfg["quantization"]["group_size"])
        _nn.quantize(drafter, group_size=qg, bits=qb)
        drafter.load_weights(os.path.join(DR, "model.safetensors"), strict=False)
        mx.eval(drafter.parameters())
        print(f"r0: drafter Q{qb} prequantise, peak "
              f"{mx.get_peak_memory()/1e9:.0f} GB", flush=True)
        vals = []
    else:
        from dv_d_drafter import load_drafter_weights
        w = load_drafter_weights(DR, o_groups=a.o_groups)
        print("r0: dequant graph OK", flush=True)
        drafter.load_weights(list(w.items()), strict=False)
        print("r0: bind OK, eval...", flush=True)
        vals = list(w.values())
    # Chunked eval: one giant eval of the whole dequant graph wedged Metal
    # (cv-wait, workers idle, 0 GPU) on repeat runs. Small submissions,
    # like mlx-lm's per-shard loads, keep the command buffers sane and
    # give visible progress.
    for i in range(0, len(vals), 8):
        mx.eval(*vals[i:i + 8])
        print(f"r0: eval {min(i + 8, len(vals))}/{len(vals)}", flush=True)
    mx.eval(drafter.parameters())
    if args.drafter_bits in (4, 8) and not dcfg.get("prequantized"):
        # Le propose est le poste dominant mesure sur Flash (108 ms/round
        # = 66 %) : block_size etapes markov SEQUENTIELLES, chacune streamant
        # les poids du drafter. Quantiser divise ce trafic.
        _nn.quantize(drafter, group_size=64, bits=args.drafter_bits)
        mx.eval(drafter.parameters())
        print(f"r0: drafter quantise Q{args.drafter_bits}, peak "
              f"{mx.get_peak_memory()/1e9:.0f} GB", flush=True)
    R = DrafterRunner(drafter, a)
    embed, lm_head = model.model.embed_tokens, model.lm_head
    print(f"r0: +drafter, peak {mx.get_peak_memory()/1e9:.0f} GB", flush=True)

    cache = model.make_cache()
    ctx = R.make_ctx()
    ids = list(tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                       add_generation_prompt=True, tokenize=True))
    eos = {tok.eos_token_id}

    # prefill (broadcast the whole prompt so rank1 runs it too)
    bcast(world, rank, [OP_RUN, len(ids), 0] + ids)
    logits, caps = fwd(mx.array([ids], dtype=mx.int32), cache)
    # Eval-isolation (gotcha Inkling): materialize logits AND caps in ONE
    # pass so no later lazy pull (propose's ctx chain) re-runs the
    # distributed collectives out of lockstep.
    mx.eval(logits, *caps.values())
    fused = mx.concatenate([caps[t] for t in taps], axis=-1)      # [1,L,3H]
    R.update_ctx(fused, ctx)                                      # commit prompt
    pending = int(mx.argmax(logits[:, -1, :]).item())
    out = [pending]

    rounds = accepts = extra = 0
    t_prop = t_ver = t_rec = 0.0
    t0 = time.time()
    while len(out) < args.max_tokens and out[-1] not in eos:
        _t = time.time()
        draft = R.propose(pending, ctx, embed, lm_head, K, MASK)
        t_prop += time.time() - _t
        block = [pending] + draft
        s = snap(cache)
        bcast(world, rank, [OP_RUN, len(block), 0] + block)
        _t = time.time()
        vlog, vcaps = fwd(mx.array([block], dtype=mx.int32), cache)
        mx.eval(vlog, *vcaps.values())
        t_ver += time.time() - _t
        tt = [int(x) for x in mx.argmax(vlog[0], axis=-1).tolist()]
        n = 0
        while n < len(draft) and draft[n] == tt[n]:
            n += 1
        committed = draft[:n] + [tt[n]]
        vfused = mx.concatenate([vcaps[t] for t in taps], axis=-1)  # [1,K,3H]
        if n < len(draft):                       # rollback + re-commit
            restore(cache, s)
            recommit = [block[0]] + committed[:-1]
            bcast(world, rank, [OP_RUN, len(recommit), 1] + recommit)
            _t = time.time()
            rlog, rcaps = fwd(mx.array([recommit], dtype=mx.int32), cache)
            mx.eval(rlog, *rcaps.values())   # rlog too: its all_gather must run
            t_rec += time.time() - _t
            vfused = mx.concatenate([rcaps[t] for t in taps], axis=-1)
            extra += 1
        R.update_ctx(vfused[:, : n + 1, :], ctx)   # commit [pending,accepted]
        rounds += 1
        accepts += len(committed)
        for t in committed:
            out.append(t)
            if t in eos:
                break
        pending = out[-1]
    dt = time.time() - t0
    bcast(world, rank, [OP_STOP, 0, 0])

    print(f"DIST-V4-SPEC ids: {out[:40]}")
    print(f"DIST-V4-SPEC {len(out)} tok in {dt:.1f}s = {len(out)/dt:.2f} tok/s | "
          f"rounds {rounds} | mean_accept {accepts/max(rounds,1):.2f} | "
          f"rollbacks {extra}/{rounds} | peak {mx.get_peak_memory()/1e9:.0f} GB")
    _r = max(rounds, 1)
    print(f"DIST-V4-SPEC breakdown/round: propose {t_prop/_r*1000:.0f} ms | "
          f"verify {t_ver/_r*1000:.0f} ms | recommit {t_rec/_r*1000:.0f} ms "
          f"({extra} rounds) | total {(t_prop+t_ver+t_rec)/_r*1000:.0f} ms")
    print("DIST-V4-SPEC text:", tok.decode(out)[:200].replace("\n", " "))


if __name__ == "__main__":
    main()
