"""DSpark speculative decoding — multi-rank, drop-in generator for the runner.

Porte la loop PROUVEE du harnais `dv_g` (goal E/Pro, lossless accept ~2.88,
ids == plain) dans le chemin de service. Contrat = celui de
`mtp_spec.native_mtp_stream_generate` : un generateur qui yield des
`MTPResponse` (token/text/finish_reason) consommes tels quels par la boucle
d'emission du runner. Aucune modif du protocole emit.

Modele multi-rang (SPMD : la requete arrive sur TOUS les rangs) :
  * rank0 possede le drafter DSpark (V4DSparkDrafter). Il prefill, propose un
    bloc de K tokens, le DIFFUSE (all_sum, pattern A0) aux servants, tous les
    rangs font le forward-verify du target EN LOCKSTEP, rank0 accepte le
    prefixe qui matche l'argmax target (lossless : le token committe EST
    toujours l'argmax du target), rollback+recommit sur rejet (snapshot/restore
    car PoolingCache V4 n'est pas trimmable), et yield les tokens acceptes.
  * les servants tournent la boucle de reception (recoivent le bloc, forward
    target) et NE YIELDENT RIEN — leur `for res in gen_iter` du runner ne voit
    donc aucun token (correct : les servants n'emettent pas).

Le drafter et sa chaine de dequant ne vivent QUE sur rank0 : les imports
correspondants sont gardes rank0 pour ne pas imposer l'arbre de deps au
servant (un pur servant target).
"""
import os
import sys
import time
from typing import Generator, Optional

# Les modeles V4 + le drafter vivent hors du venv mlx-cluster (fork Ivan +
# checkpoints). Memes chemins que le harnais dv_g, presents sur les nodes.
for _p in ("~/ivan-mlx-lm", "~/deepseek-v4-mlx", "~"):
    _ap = os.path.expanduser(_p)
    if _ap not in sys.path:
        sys.path.insert(0, _ap)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn    # noqa: E402
from mlx_lm.models.base import create_attention_mask  # noqa: E402
from mlx_lm.models.cache import CacheList              # noqa: E402

from mtp_spec import MTPResponse  # noqa: E402  (reuse le duck-type existant)

MAXS, HDR = 2048, 3          # MAXS couvre le prompt de prefill, pas juste K
OP_RUN, OP_STOP = 0, 1


# ── ctx-cache drafter (dspartha CtxCache, append-only) — copie dv_g ──────────
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


def _collapse_mean(t):        # 4D HC state -> 2D
    return t.mean(axis=-2)


# ── forward target distribue avec capture de taps (copie dv_g make_forward) ──
def _make_forward(model, taps):
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
                caps[gid] = _collapse_mean(h)
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


# ── snapshot / restore O(1) (copie dv_g) ─────────────────────────────────────
def _snap(cache) -> list:
    out = []
    for c in cache:
        if isinstance(c, CacheList):
            out.append(("l", [dict(vars(x)) for x in c.caches]))
        else:
            out.append(("o", dict(vars(c))))
    return out


def _restore(cache, s) -> None:
    for c, (k, st) in zip(cache, s):
        if k == "l":
            for sub, d in zip(c.caches, st):
                sub.__dict__.update(d)
        else:
            c.__dict__.update(st)


# ── broadcast rank0 -> servants via all_sum (pattern A0, copie dv_g) ─────────
def _bcast(group, rank, vec=None) -> list:
    if rank == 0:
        arr = mx.array(vec + [0] * (HDR + MAXS - len(vec)), dtype=mx.int32)
    else:
        arr = mx.zeros((HDR + MAXS,), dtype=mx.int32)
    out = mx.distributed.all_sum(arr, group=group)
    mx.eval(out)
    return [int(x) for x in out.tolist()]


# ── drafter runner (copie dv_g DrafterRunner) ────────────────────────────────
class _DrafterRunner:
    def __init__(self, drafter, a):
        self.d = drafter
        self.a = a
        self.hc = a.hidden_size

    def make_ctx(self):
        return [CtxCache() for _ in self.d.layers]

    def update_ctx(self, target_hidden_cat, caches):
        b0 = self.d.layers[0]
        fused = b0.main_norm(b0.main_proj(target_hidden_cat))
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


# ── chargement drafter (rank0 uniquement) — chemin prequantise de dv_g ───────
def load_dspark_drafter(drafter_dir: str, target_args, dcfg: dict):
    """Charge le drafter DSpark Q8-prequantise sur rank0. Quantifie le
    squelette puis load_weights (evite la chaine dequant FP8/FP4 -- pic RAM +
    dependance pipenetwork -- payee sinon a chaque load)."""
    from dv_d_drafter import V4DSparkDrafter  # rank0 only (arbre de deps)
    import json
    drafter = V4DSparkDrafter(target_args, dcfg)
    if dcfg.get("prequantized"):
        qb = int(dcfg["quantization"]["bits"])
        qg = int(dcfg["quantization"]["group_size"])
        nn.quantize(drafter, group_size=qg, bits=qb)
        drafter.load_weights(os.path.join(drafter_dir, "model.safetensors"), strict=False)
        mx.eval(drafter.parameters())
    else:
        from dv_d_drafter import load_drafter_weights
        w = load_drafter_weights(drafter_dir, o_groups=target_args.o_groups)
        drafter.load_weights(list(w.items()), strict=False)
        vals = list(w.values())
        for i in range(0, len(vals), 8):
            mx.eval(*vals[i:i + 8])
        mx.eval(drafter.parameters())
    return drafter


def load_target_args(target_dir: str):
    """ModelArgs V4 du target (pour le drafter + le forward). Rank0 only."""
    import json
    from deepseek_v4 import ModelArgs as V4Args  # fork Ivan / mlx_models
    return V4Args.from_dict(json.load(open(os.path.join(target_dir, "config.json"))))


# ── le generateur drop-in ────────────────────────────────────────────────────
def spec_dspark_stream_generate(
    model,
    tokenizer,
    prompt_ids,
    *,
    max_tokens: int,
    rank: int,
    size: int,
    group,
    drafter=None,          # rank0 : V4DSparkDrafter ; servants : None
    target_args=None,      # rank0 : ModelArgs V4
    dcfg=None,             # rank0 : config drafter (taps/block/noise)
    stop_ids=None,
) -> Generator[MTPResponse, None, None]:
    taps = list(dcfg["dspark_target_layer_ids"]) if (rank == 0 and dcfg) else []
    fwd = _make_forward(model, set(taps))
    cache = model.make_cache()
    eos = set(stop_ids or [])
    _e = getattr(tokenizer, "eos_token_id", None)
    if _e is not None:
        eos.add(int(_e))

    # ── servants : boucle de reception, aucun yield ──
    if rank != 0:
        servant_snap = None
        while True:
            v = _bcast(group, rank)
            op, S, flag = v[0], v[1], v[2]
            if op == OP_STOP:
                break
            if flag:
                _restore(cache, servant_snap)
            else:
                servant_snap = _snap(cache)
            o, _ = fwd(mx.array([v[HDR:HDR + S]], dtype=mx.int32), cache)
            mx.eval(o)
        return

    # ── rank0 : drive ──
    K = int(dcfg["dspark_block_size"])
    MASK = int(dcfg["dspark_noise_token_id"])
    R = _DrafterRunner(drafter, target_args)
    ctx = R.make_ctx()
    embed, lm_head = model.model.embed_tokens, model.lm_head
    ids = list(prompt_ids)

    detok = tokenizer.detokenizer
    detok.reset()
    t0 = time.time()
    emitted = 0
    rounds = 0
    accepted_total = 0
    drafted_total = 0

    def _mk(tok: int, finish=None, from_draft=False) -> MTPResponse:
        detok.add_token(tok)
        return MTPResponse(
            text=detok.last_segment, token=tok, finish_reason=finish,
            prompt_tokens=len(ids),
            generation_tokens=emitted,
            generation_tps=emitted / max(time.time() - t0, 1e-9),
            peak_memory=mx.get_peak_memory() / 1e9,
            from_draft=from_draft,
            accept_rate=(accepted_total / drafted_total) if drafted_total else 0.0,
            round_idx=rounds,
        )

    # try/finally : les servants sont bloques sur le PROCHAIN _bcast a tout
    # instant ; QUEL QUE SOIT le chemin de sortie (fin normale, break du
    # consommateur -> GeneratorExit, exception), on DOIT leur envoyer OP_STOP
    # sinon ils hangent et wedgent le pool. Exactement UN OP_STOP par sortie.
    finish = None
    try:
        # ── prefill : forward complet, capture taps -> ctx drafter ──
        _bcast(group, rank, [OP_RUN, len(ids), 0] + ids)
        logits, caps = fwd(mx.array([ids], dtype=mx.int32), cache)
        # eval-isolation : logits ET caps materialises ensemble, sinon un pull
        # lazy (chaine ctx du propose) rejoue les collectives hors lockstep.
        mx.eval(logits, *caps.values())
        fused = mx.concatenate([caps[t] for t in taps], axis=-1)
        R.update_ctx(fused, ctx)
        pending = int(mx.argmax(logits[:, -1, :]).item())

        # Le prefill argmax EST le 1er token genere (parite AR / mlx-lm).
        emitted += 1
        finish = "stop" if pending in eos else ("length" if emitted >= max_tokens else None)
        yield _mk(pending, finish=finish)

        # ── boucle spec : propose -> verify -> accept -> yield ──
        while finish is None:
            draft = R.propose(pending, ctx, embed, lm_head, K, MASK)
            drafted_total += len(draft)
            block = [pending] + draft
            s = _snap(cache)
            _bcast(group, rank, [OP_RUN, len(block), 0] + block)
            vlog, vcaps = fwd(mx.array([block], dtype=mx.int32), cache)
            mx.eval(vlog, *vcaps.values())
            tt = [int(x) for x in mx.argmax(vlog[0], axis=-1).tolist()]
            n = 0
            while n < len(draft) and draft[n] == tt[n]:
                n += 1
            committed = draft[:n] + [tt[n]]    # lossless : tt[n] = argmax target
            vfused = mx.concatenate([vcaps[t] for t in taps], axis=-1)
            if n < len(draft):                 # rollback + recommit
                _restore(cache, s)
                recommit = [block[0]] + committed[:-1]
                _bcast(group, rank, [OP_RUN, len(recommit), 1] + recommit)
                rlog, rcaps = fwd(mx.array([recommit], dtype=mx.int32), cache)
                mx.eval(rlog, *rcaps.values())
                vfused = mx.concatenate([rcaps[t] for t in taps], axis=-1)
            R.update_ctx(vfused[:, : n + 1, :], ctx)
            rounds += 1
            accepted_total += len(committed)

            for tok in committed:
                emitted += 1
                finish = "stop" if tok in eos else ("length" if emitted >= max_tokens else None)
                yield _mk(tok, finish=finish, from_draft=(tok != committed[-1]))
                if finish is not None:
                    break
            pending = committed[-1]
    finally:
        # collectif : les servants attendent sur all_sum -> les libere.
        try:
            _bcast(group, rank, [OP_STOP, 0, 0])
        except Exception:
            pass
