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
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Generator, Optional

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

# dv_d_drafter + le propose du drafter importent `mlx_lm.models.hyper_connection`,
# absent du mlx_lm du runner (venv) mais present dans le fork Ivan. Le runner a
# DEJA importe mlx_lm (venv) au demarrage, donc un simple sys.path ne suffit pas
# (le package est deja lie a son __path__). On ETEND le __path__ de
# mlx_lm.models vers les models du fork : les sous-modules MANQUANTS
# (hyper_connection) s'y resolvent, sans deloger deepseek_v4 deja patche+en
# cache dans sys.modules. Inoffensif sur les servants (ne l'importent jamais).
try:
    import mlx_lm.models as _mlxm  # noqa: E402
    _ivan_models = os.path.expanduser("~/ivan-mlx-lm/mlx_lm/models")
    if os.path.isdir(_ivan_models) and _ivan_models not in _mlxm.__path__:
        _mlxm.__path__.append(_ivan_models)
except Exception:
    pass

@dataclass
class SpecResponse:
    """Duck-type les champs de GenerationResponse que la boucle emit du runner
    lit (token/text/finish_reason/...). Local : pas de dep sur mtp_spec (absent
    du ~/mlx-cluster des nodes ; le runner ne l'importe qu'en lazy sous MTP)."""
    text: str
    token: int
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tokens: int = 0
    generation_tps: float = 0.0
    peak_memory: float = 0.0
    from_draft: bool = False
    logprobs: Any = None
    accept_rate: float = 0.0
    round_idx: int = 0


MAXS, HDR = 2048, 3          # MAXS couvre le prompt de prefill, pas juste K
OP_RUN, OP_STOP = 0, 1

# Seuil du confidence-gate P2 (0 = off) : sigmoid de la tete de confiance
# APPRISE du drafter (DeepSpec) — tronque le draft a la premiere position
# dont P(accept) < seuil. Prioritaire sur le margin-gate quand la tete existe.
_CONF_GATE = float(os.environ.get("SPEC_CONF_GATE", "0.7"))
# Seuil du margin-gate fallback (0 = off). Marge logit top1-top2 :
# en dessous, le draft est tronque (moins de rejets = moins de recommits).
_MARGIN_GATE = float(os.environ.get("SPEC_MARGIN_GATE", "1.2"))
# Cold-skip : apres N proposes consecutifs gates a vide (passage "froid",
# prose), on saute le propose (~20ms/round) et on ne re-sonde que tous les
# PROBE rounds — le round froid devient ~un step plain. Reset des qu'un
# draft survit au gate.
_COLD_AFTER = int(os.environ.get("SPEC_COLD_AFTER", "3"))
_COLD_PROBE = int(os.environ.get("SPEC_COLD_PROBE", "3"))


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
        hid = last.norm(last.hc_head(h))             # [1,K,H] pre-lm_head
        base = lm_head(hid)[0]
        drafts = [int(x) for x in self.d.sample_block(base, pending).tolist()]
        # Confidence-gate P2 (semantique DeepSpec, eval/dspark/draft_ops.py
        # `_confident_prefix_length`) : la tete APPRISE du checkpoint —
        # features = concat([hidden pre-lm_head, markov_w1(prev_ids)]) avec
        # prev_ids = [pending] + drafts[:-1] — un logit par position ;
        # sigmoid = P(step accepte) ; on tronque a la PREMIERE position sous
        # le seuil. Remplace le proxy margin (signal appris et calibre —
        # reliability diagrams DeepSpec — vs marge logit aveugle).
        if _CONF_GATE > 0 and getattr(last, "confidence_head", None) is not None:
            prev = mx.array([[pending] + drafts[:-1]], dtype=mx.int32)
            feats = mx.concatenate(
                [hid, last.markov_head.prev_embeddings(prev).astype(hid.dtype)],
                axis=-1)                              # [1,K,H+rank]
            p = mx.sigmoid(last.confidence_head.proj(feats))[0, :, 0]
            mx.eval(p)
            probs = [float(x) for x in p.tolist()]
            keep = 0
            for i in range(len(drafts)):
                if probs[i] < _CONF_GATE:
                    break
                keep += 1
            drafts = drafts[:keep]
        # Margin-gate (fallback si pas de tete de confiance) : position gardee
        # ssi le token choisi == argmax de base ET marge top1-top2 >= seuil.
        # Draft vide = step plain (1 forward, 1 token) -> plancher ~plain.
        elif _MARGIN_GATE > 0:
            top2 = mx.topk(base, k=2, axis=-1)          # [K, 2] les 2 plus grands
            top1_ids = mx.argmax(base, axis=-1)          # [K]
            margin = mx.abs(top2[:, 0] - top2[:, 1])     # |top1-top2|, ordre indiff.
            mx.eval(margin, top1_ids)
            m = [float(x) for x in margin.tolist()]
            t1 = [int(x) for x in top1_ids.tolist()]
            keep = 0
            for i, d in enumerate(drafts):
                if d == t1[i] and m[i] >= _MARGIN_GATE:
                    keep += 1
                else:
                    break
            drafts = drafts[:keep]
        return drafts


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
    """ModelArgs V4 du target (pour le drafter + le forward). Rank0 only.
    Meme import que le harnais dv_g : le module ds4 du fork Ivan."""
    import mlx_lm.models.deepseek_v4 as V4  # fork Ivan (PYTHONPATH node)
    return V4.ModelArgs.from_dict(json.load(open(os.path.join(target_dir, "config.json"))))


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
    drafter=None,
    target_args=None,
    dcfg=None,
    stop_ids=None,
) -> Generator["SpecResponse", None, None]:
    """Wrapper : ouvre wired_limit + generation_stream autour de la loop.

    LE fix perf (la nuit du 04->05/08, ~90x) : sans wired_limit, macOS
    compresse/pagine les poids froids entre les forwards (la bulle pipeline
    laisse le temps de re-compresser) -> chaque forward re-faulte ses experts
    -> coput fixe ~2,1s/rang/forward INDEPENDANT de S. stream_generate (8,1
    tok/s plain, memes nodes) enveloppe TOUTE la generation dans wired_limit ;
    mtp_spec.py documente le meme mur (~4,5s/forward, paging-bound). Le limit
    est global : il epingle aussi le drafter de rank0."""
    try:
        from mlx_lm.generate import wired_limit as _wl, generation_stream as _gs
    except Exception:
        import contextlib
        _gs = mx.default_stream(mx.default_device())

        def _wl(_m, _s):  # no-op fallback (meme pattern que mtp_spec)
            return contextlib.nullcontext()

    with _wl(model, [_gs]), mx.stream(_gs):
        yield from _spec_body(
            model, tokenizer, prompt_ids, max_tokens=max_tokens, rank=rank,
            size=size, group=group, drafter=drafter, target_args=target_args,
            dcfg=dcfg, stop_ids=stop_ids)


def _spec_body(
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
) -> Generator[SpecResponse, None, None]:
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
            elif S > 1:
                # S==1 flag=0 = step plain (draft vide) : pas de rollback
                # possible -> pas de snapshot (miroir du skip rank0). Un
                # recommit (flag=1) ne suit jamais qu'un verify S>1, donc
                # servant_snap est toujours frais quand il sert.
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
    rollbacks = 0

    def _mk(tok: int, finish=None, from_draft=False) -> SpecResponse:
        detok.add_token(tok)
        return SpecResponse(
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
        cold = 0          # proposes consecutifs gates a vide
        since_probe = 0   # rounds depuis le dernier propose en mode froid
        while finish is None:
            if cold >= _COLD_AFTER and since_probe < _COLD_PROBE:
                draft = []            # round froid : pas de propose du tout
                since_probe += 1
            else:
                draft = R.propose(pending, ctx, embed, lm_head, K, MASK)
                since_probe = 0
                if draft:
                    cold = 0
                else:
                    cold += 1
            drafted_total += len(draft)
            block = [pending] + draft
            # Draft vide (gate) = step plain : aucun rollback possible -> on
            # saute le snapshot (61 couches de copies vars(), pur overhead).
            # Les servants font pareil sur S==1 (voir boucle servant).
            s = _snap(cache) if draft else None
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
                rollbacks += 1
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
        try:
            _dt = max(time.time() - t0, 1e-9)
            sys.stderr.write(
                f"[spec] rounds={rounds} emitted={emitted} "
                f"accept/round={accepted_total / max(rounds, 1):.2f} "
                f"rollbacks={rollbacks}/{rounds} "
                f"drafted={drafted_total} conf_gate={_CONF_GATE} margin={_MARGIN_GATE} "
                f"tok/s={emitted / _dt:.2f}\n")
            sys.stderr.flush()
        except Exception:
            pass
