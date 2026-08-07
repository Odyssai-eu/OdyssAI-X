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


MAXS, HDR = 8192, 3          # MAXS couvre le prompt de prefill, pas juste K
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
# Drafter context window (2026-08-06 root-cause fix). The drafter's per-layer
# CtxCache used to grow UNBOUNDED (mx.concatenate every round over the whole
# generation), so its self-attention was O(gen_len): progressive slowdown then
# a hard stall/blowup past ~10k tokens (bench C freeze). Bounding it to the last
# W target-hidden entries caps the cost. LOSSLESS-SAFE: the drafter only
# PROPOSES — the target verify is authoritative — so windowing can only affect
# accept-rate/speed, never correctness. W>=typical response length => zero
# behavior change for normal chat; only pathological long gens get bounded.
# 0 disables the window (legacy unbounded behavior).
_DRAFTER_CTX_WINDOW = int(os.environ.get("SPEC_DRAFTER_CTX_WINDOW", "2048"))
# Diag mémoire opt-in : log [mem] (active/cache/peak) tous les 500 rounds.
# Sert à confirmer que le pooled reste borné (cf fix _cache_arrays dans la boucle
# de décode). OFF par défaut — n'affecte pas le serving.
_SPEC_MEM_DEBUG = os.environ.get("SPEC_MEM_DEBUG", "0") not in ("0", "", "false")
# Filet mx.clear_cache() périodique — OFF par défaut. La racine du crash
# metal::malloc à ~12k tokens était l'accumulation lazy de l'état pooled de
# PoolingCache (jamais matérialisé dans la boucle), pas le pool de buffers ;
# corrigée en évaluant *_cache_arrays(cache) chaque round. clear_cache reste
# dispo comme filet mais inutile avec le vrai fix.
_CLEAR_CACHE_EVERY = int(os.environ.get("SPEC_CLEAR_CACHE_EVERY", "0")) or 10**9
_COLD_PROBE = int(os.environ.get("SPEC_COLD_PROBE", "3"))


# ── ctx-cache drafter (dspartha CtxCache, append-only) — copie dv_g ──────────
class CtxCache:
    # `_abs` = absolute count of appended entries (the rope offset), tracked
    # separately from the (possibly windowed) stored k/v so rope stays correct
    # after the sliding-window drop — q@abs and k@abs keep true relative
    # positions. See _DRAFTER_CTX_WINDOW.
    __slots__ = ("k", "v", "_abs")

    def __init__(self):
        self.k = self.v = None
        self._abs = 0

    def append(self, k, v):
        self._abs += k.shape[2]
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = mx.concatenate([self.k, k], axis=2)
            self.v = mx.concatenate([self.v, v], axis=2)
        # Sliding window: keep only the last W entries. Bounds the drafter's
        # self-attention cost (root-cause fix for the long-gen stall). k/v are
        # already rope-baked at absolute positions, so dropping the oldest is a
        # correct sliding-window truncation. Lossless: drafter only proposes.
        W = _DRAFTER_CTX_WINDOW
        if W > 0 and self.k.shape[2] > W:
            self.k = self.k[:, :, -W:, :]
            self.v = self.v[:, :, -W:, :]

    @property
    def length(self):
        # ABSOLUTE position (rope offset), NOT the windowed store size.
        return self._abs


def _collapse_mean(t):        # 4D HC state -> 2D
    return t.mean(axis=-2)


def _cache_arrays(cache) -> list:
    """Aplatit tout mx.array vivant du prompt cache (eval-isolation, lecon
    1.24.0) : l'etat POOLE de PoolingCache n'est pas ancetre des logits, donc
    un eval des seuls logits le laisse lazy — snapshotte puis restaure, son
    graphe rejouerait les collectives HORS lockstep au tour suivant. On
    materialise tout avant de snapshotter."""
    out: list = []

    def _pull(c) -> None:
        got = False
        st = getattr(c, "state", None)
        if st is not None:
            stack = [st]
            while stack:
                x = stack.pop()
                if isinstance(x, mx.array):
                    out.append(x); got = True
                elif isinstance(x, (list, tuple)):
                    stack.extend(x)
                elif isinstance(x, dict):
                    stack.extend(x.values())
        if not got:
            try:
                for v in vars(c).values():
                    if isinstance(v, mx.array):
                        out.append(v)
            except TypeError:
                pass

    for c in cache:
        if isinstance(c, CacheList):
            for sub in c.caches:
                _pull(sub)
        else:
            _pull(c)
    return out


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
        # Per-instance gate thresholds (default = module globals for the
        # distributed path). The single-node trim path sets these to 0: with a
        # free trim rollback the gates only cost speculation depth, no benefit.
        self.conf_gate = _CONF_GATE
        self.margin_gate = _MARGIN_GATE

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
        if self.conf_gate > 0 and getattr(last, "confidence_head", None) is not None:
            prev = mx.array([[pending] + drafts[:-1]], dtype=mx.int32)
            feats = mx.concatenate(
                [hid, last.markov_head.prev_embeddings(prev).astype(hid.dtype)],
                axis=-1)                              # [1,K,H+rank]
            p = mx.sigmoid(last.confidence_head.proj(feats))[0, :, 0]
            mx.eval(p)
            probs = [float(x) for x in p.tolist()]
            keep = 0
            for i in range(len(drafts)):
                if probs[i] < self.conf_gate:
                    break
                keep += 1
            drafts = drafts[:keep]
        # Margin-gate (fallback si pas de tete de confiance) : position gardee
        # ssi le token choisi == argmax de base ET marge top1-top2 >= seuil.
        # Draft vide = step plain (1 forward, 1 token) -> plancher ~plain.
        elif self.margin_gate > 0:
            top2 = mx.topk(base, k=2, axis=-1)          # [K, 2] les 2 plus grands
            top1_ids = mx.argmax(base, axis=-1)          # [K]
            margin = mx.abs(top2[:, 0] - top2[:, 1])     # |top1-top2|, ordre indiff.
            mx.eval(margin, top1_ids)
            m = [float(x) for x in margin.tolist()]
            t1 = [int(x) for x in top1_ids.tolist()]
            keep = 0
            for i, d in enumerate(drafts):
                if d == t1[i] and m[i] >= self.margin_gate:
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
    session_id=None,
    model_id=None,
    session_get=None,
    session_put=None,
    restore_from_snap=None,
    stats_out=None,
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
            dcfg=dcfg, stop_ids=stop_ids,
            session_id=session_id, model_id=model_id,
            session_get=session_get, session_put=session_put,
            restore_from_snap=restore_from_snap, stats_out=stats_out)


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
    session_id=None,       # A (spec+snap) : cle session (None = pas de cache)
    model_id=None,
    session_get=None,      # hooks runner : store byte-budgete partage
    session_put=None,
    restore_from_snap=None,  # constructeurs frais prouves (chemin plain 1.24.0)
    stats_out=None,        # dict mutable : cached_tokens publie au done event
) -> Generator[SpecResponse, None, None]:
    taps = list(dcfg["dspark_target_layer_ids"]) if (rank == 0 and dcfg) else []
    fwd = _make_forward(model, set(taps))
    cache = model.make_cache()
    eos = set(stop_ids or [])
    _e = getattr(tokenizer, "eos_token_id", None)
    if _e is not None:
        eos.add(int(_e))
    ids = list(prompt_ids)

    # ── A (spec+snap) : lookup session par rang + CONSENSUS all_sum ──
    # Chaque rang decide LOCALEMENT (son store, ses tailles d'entrees — un
    # evicteur peut diverger entre rangs), puis une barriere all_sum 1-int
    # verifie l'unanimite : hit seulement si TOUS les rangs ont le hit
    # (sinon repli miss partout). C'est le design P6/A.2 du plan v2.3 :
    # tous calculent, la collective est la barriere de divergence.
    _entry = None
    _local_hit = 0
    if session_id and session_get is not None and restore_from_snap is not None:
        _entry = session_get(session_id, model_id)
        if _entry and _entry.get("snap") and _entry.get("snap_tokens"):
            _st = _entry["snap_tokens"]
            # suffixe >= 2 : le prefill en 2 temps (prompt[:-1] snapshotte +
            # dernier token forwarde) exige _pre non vide. Condition IDENTIQUE
            # sur tous les rangs (memes ids, meme _st) -> consensus preserve.
            if (len(_st) <= len(ids) - 2 and ids[:len(_st)] == list(_st)):
                _local_hit = 1
    _consensus = mx.distributed.all_sum(
        mx.array([_local_hit], dtype=mx.int32), group=group)
    mx.eval(_consensus)
    use_hit = int(_consensus.item()) == size
    if os.environ.get("SPEC_SESS_DEBUG"):
        _st_dbg = _entry["snap_tokens"] if (_entry and _entry.get("snap_tokens")) else None
        _snl = len(_st_dbg) if _st_dbg else -1
        _cpl = -1
        if _st_dbg:
            _cpl = 0
            for _a, _b in zip(ids, _st_dbg):
                if _a != _b:
                    break
                _cpl += 1
        sys.stderr.write(
            f"[spec-sess] r{rank} sid={str(session_id)[:8]} entry={_entry is not None} "
            f"has_snap={bool(_entry and _entry.get('snap'))} "
            f"local_hit={_local_hit} consensus={int(_consensus.item())}/{size} "
            f"use_hit={use_hit} snaplen={_snl} idslen={len(ids)} common_prefix={_cpl}\n")
        sys.stderr.flush()
    suffix = ids[len(_entry["snap_tokens"]):] if use_hit else ids
    cached_n = len(ids) - len(suffix) if use_hit else 0
    if use_hit:
        cache = restore_from_snap(_entry["cache"], _entry["snap"])
    if stats_out is not None:
        stats_out["cached_tokens"] = cached_n

    # ── servants : boucle de reception, aucun yield ──
    if rank != 0:
        servant_snap = None
        _prompt_snap = None
        _prompt_len = 0
        first_forward = True
        while True:
            v = _bcast(group, rank)
            op, S, flag = v[0], v[1], v[2]
            if op == OP_STOP:
                break
            if flag:
                _restore(cache, servant_snap)
            elif S > 1 and not first_forward:
                # S==1 flag=0 = step plain (draft vide) : pas de rollback
                # possible -> pas de snapshot (miroir du skip rank0). Un
                # recommit (flag=1) ne suit jamais qu'un verify S>1, donc
                # servant_snap est toujours frais quand il sert. Le PREMIER
                # forward (prefill) n'est jamais rollback -> pas de snap.
                servant_snap = _snap(cache)
            o, _ = fwd(mx.array([v[HDR:HDR + S]], dtype=mx.int32), cache)
            if first_forward:
                # Fin de prompt : eval de l'ETAT COMPLET (isolation cross-
                # requete, lecon 1.24.0) puis snapshot inerte pour la session.
                mx.eval(o, *_cache_arrays(cache))
                _prompt_snap = _snap(cache)
                _prompt_len = cached_n + S
                first_forward = False
            else:
                mx.eval(o)
        try:
            if session_id and session_put is not None and _prompt_snap is not None:
                session_put(session_id, model_id, cache, ids,
                            _prompt_snap, ids[:_prompt_len])
        except Exception as _e:
            sys.stderr.write(f"[spec-sess] r{rank} SERVANT STORE FAILED: {_e!r}\n")
            sys.stderr.flush()
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
    _prompt_snap = None
    _ctx_snap = None
    t_prop = t_ver = 0.0
    try:
        if len(suffix) > MAXS:
            raise ValueError(
                f"spec prefill: suffix {len(suffix)} tokens > MAXS {MAXS} — "
                f"prompt trop long pour le transport bcast")
        # ── A : restauration du ctx drafter sur hit (rank0 only) ──
        if use_hit and _entry.get("spec_ctx"):
            ctx = []
            for _t in _entry["spec_ctx"]:
                c = CtxCache()
                c.k, c.v = _t[0], _t[1]
                c._abs = _t[2] if len(_t) > 2 else (0 if c.k is None else c.k.shape[2])
                ctx.append(c)
        # ── prefill en DEUX temps (aligne sur le chemin plain) : le snapshot
        # est pris a prompt[:-1] et le DERNIER token du prompt est forwarde a
        # part. Raison (bug 05/08) : le dernier token (debut-assistant) N'EST
        # PAS a la meme position au tour suivant (common_prefix = len-1), donc
        # snap_tokens = ids[:-1] pour que le match tienne, et le cache
        # snapshotte doit correspondre = etat a prompt[:-1].
        _pre = suffix[:-1]
        _last = suffix[-1]
        if _pre:
            _bcast(group, rank, [OP_RUN, len(_pre), 0] + _pre)
            lp, cp = fwd(mx.array([_pre], dtype=mx.int32), cache)
            # eval-isolation : etat COMPLET materialise avant le snapshot
            # (lecon 1.24.0 : l'etat poole lazy rejouerait les collectives).
            mx.eval(lp, *cp.values(), *_cache_arrays(cache))
            R.update_ctx(mx.concatenate([cp[t] for t in taps], axis=-1), ctx)
        # Snapshot fin-de-prompt-moins-un (target + ctx drafter).
        _prompt_snap = _snap(cache)
        _ctx_snap = [(c.k, c.v, c._abs) for c in ctx]
        # Forward du DERNIER token du prompt -> premier pending.
        _bcast(group, rank, [OP_RUN, 1, 0, _last])
        ll, cl = fwd(mx.array([[_last]], dtype=mx.int32), cache)
        mx.eval(ll, *cl.values())
        R.update_ctx(mx.concatenate([cl[t] for t in taps], axis=-1), ctx)
        pending = int(mx.argmax(ll[:, -1, :]).item())

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
                _tp = time.time()
                draft = R.propose(pending, ctx, embed, lm_head, K, MASK)
                t_prop += time.time() - _tp
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
            _tv = time.time()
            _bcast(group, rank, [OP_RUN, len(block), 0] + block)
            vlog, vcaps = fwd(mx.array([block], dtype=mx.int32), cache)
            mx.eval(vlog, *vcaps.values())
            t_ver += time.time() - _tv
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
        # A : persiste la session rank0 (snap fin-de-prompt + ctx drafter).
        # Sur disconnect client, le snap reste valide (fin-de-prompt) ; le
        # cache vivant ne sert que de porteur de classes pour le restore.
        try:
            if session_id and session_put is not None and _prompt_snap is not None:
                # snap_tokens = ids[:-1] : le snapshot est a prompt[:-1] (le
                # dernier token est re-prefille au tour suivant, cf prefill).
                session_put(session_id, model_id, cache, ids,
                            _prompt_snap, ids[:-1], spec_ctx=_ctx_snap)
                if os.environ.get("SPEC_SESS_DEBUG"):
                    sys.stderr.write(
                        f"[spec-sess] r{rank} STORED sid={str(session_id)[:8]} "
                        f"snaptok={len(ids) - 1}\n"); sys.stderr.flush()
        except Exception as _e:
            sys.stderr.write(f"[spec-sess] r{rank} STORE FAILED: {_e!r}\n")
            sys.stderr.flush()
        try:
            _dt = max(time.time() - t0, 1e-9)
            _r = max(rounds, 1)
            sys.stderr.write(
                f"[spec] rounds={rounds} emitted={emitted} "
                f"accept/round={accepted_total / _r:.2f} "
                f"rollbacks={rollbacks}/{rounds} "
                f"drafted={drafted_total} conf_gate={_CONF_GATE} "
                f"cached={cached_n} "
                f"prop_ms={t_prop / _r * 1000:.0f} ver_ms={t_ver / _r * 1000:.0f} "
                f"tok/s={emitted / _dt:.2f}\n")
            sys.stderr.flush()
        except Exception:
            pass


# ── Single-node trimmable-pool spec (Inferencer-style, NO re-forward) ─────────
# world_size==1 path: drafter in-process, verify a block, accept the longest
# greedy prefix, and roll back by TRIMMING the rejected tail (trimmable pool
# un-fold + wrap-safe rotating slice) instead of restore+recommit. The re-forward
# was the whole tax (85% of rounds) that pinned the distributed/naive path to
# ~1.0x; trimming keeps the verify K/V, giving ~1.2-1.4x on code, lossless.
# Requires a 512G node (target 182G + drafter bf16 54G, both wired).
def _make_forward_single(model, taps):
    """Single-node forward: iterate ALL layers locally (no pipeline attrs).
    The distributed `_make_forward` needs mm.pipeline_layers/start_idx set by
    `model.model.pipeline(group)`, which single-node mode never calls. Taps are
    global ids and == local index j here (start_idx == 0). Mirrors dv_i.py."""
    mm = model.model

    def fwd(inputs, cache):
        h = mm.embed_tokens(inputs)
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], mm.args.hc_mult, h.shape[2]))
        h = mx.contiguous(h)
        fc = cache[0]
        mc = fc[0] if isinstance(fc, CacheList) else fc
        mask = create_attention_mask(
            h[:, :, 0, :], mc, window_size=mm.args.sliding_window,
            return_array=True)
        caps = {}
        for j, (layer, lc) in enumerate(zip(mm.layers, cache)):
            h = layer(h, mask, lc, inputs)
            if j in taps:
                caps[j] = _collapse_mean(h)
        return model.lm_head(mm.norm(mm.hc_head(h))), caps

    return fwd


def _trim_leaf(leaf, n):
    if type(leaf).__name__ == "RotatingKVCache":
        # Spec forwards are always S>1 -> _update_concat keeps keys temporally
        # ordered, so slicing the tail is wrap-safe (unlike stock offset-=n).
        if getattr(leaf, "keys", None) is None:
            return
        L = leaf.keys.shape[2]
        m = min(n, L)
        leaf.keys = leaf.keys[..., :L - m, :]
        leaf.values = leaf.values[..., :L - m, :]
        leaf.offset -= m
        leaf._idx = leaf.keys.shape[2]
    elif hasattr(leaf, "trim"):
        leaf.trim(n)          # PoolingCache un-fold (dsv4_cache), KVCache, ...


def _trim_all(cache, n):
    if n <= 0:
        return
    for c in cache:
        leaves = c.caches if hasattr(c, "caches") else [c]
        for leaf in leaves:
            _trim_leaf(leaf, n)


def single_node_spec_stream_generate(
    model, tokenizer, prompt_ids, *, max_tokens, drafter, target_args, dcfg,
    stop_ids=None, stats_out=None,
    session_id=None, model_id=None, session_get=None, session_put=None,
    restore_from_snap=None,
):
    try:
        from mlx_lm.generate import wired_limit as _wl, generation_stream as _gs
    except Exception:
        import contextlib
        _gs = mx.default_stream(mx.default_device())

        def _wl(_m, _s):
            return contextlib.nullcontext()

    taps = list(dcfg["dspark_target_layer_ids"])
    fwd = _make_forward_single(model, taps)
    K = int(dcfg["dspark_block_size"])
    MASK = int(dcfg["dspark_noise_token_id"])
    R = _DrafterRunner(drafter, target_args)
    # Gate ON by default (STABILITY). Gate-off wins ~+5% on PREDICTABLE code,
    # but on prose/unpredictable content the drafter proposes 5 tokens/round all
    # rejected (~99% rollback) → thrash at ~0.45× plain + runaway generations →
    # instability (bench crash 2026-08-05: 12795-tok gen, 99.5% rollback,
    # 12.75 tok/s). The gate truncates uncertain drafts → empty → cold-skip →
    # ~plain floor; the code gain survives. Set SPEC_SINGLE_CONF_GATE=0 to force
    # gate-off for pure-code workloads.
    R.conf_gate = float(os.environ.get("SPEC_SINGLE_CONF_GATE", "0.7"))
    R.margin_gate = float(os.environ.get("SPEC_SINGLE_MARGIN_GATE", "1.2"))
    embed, lm_head = model.model.embed_tokens, model.lm_head
    eos = set(stop_ids or [])
    _e = getattr(tokenizer, "eos_token_id", None)
    if _e is not None:
        eos.add(int(_e))
    ids = list(prompt_ids)
    detok = tokenizer.detokenizer

    # ── fast-prefill : lookup session (snap-cache), meme design que _spec_body
    # (single-rank -> pas de consensus all_sum). Hit ssi snap_tokens est un
    # prefixe de ids ET laisse >=2 tokens de suffixe (prefill en 2 temps).
    _entry = None
    use_hit = False
    if session_id and session_get is not None and restore_from_snap is not None:
        _entry = session_get(session_id, model_id)
        if _entry and _entry.get("snap") and _entry.get("snap_tokens"):
            _st = _entry["snap_tokens"]
            if len(_st) <= len(ids) - 2 and ids[:len(_st)] == list(_st):
                use_hit = True
    suffix = ids[len(_entry["snap_tokens"]):] if use_hit else ids
    cached_n = len(ids) - len(suffix) if use_hit else 0
    if stats_out is not None:
        stats_out["cached_tokens"] = cached_n

    t0 = time.time()
    emitted = 0
    rounds = accepted_total = drafted_total = rollbacks = 0

    def _mk(tok, finish=None, from_draft=False):
        detok.add_token(tok)
        return SpecResponse(
            text=detok.last_segment, token=tok, finish_reason=finish,
            prompt_tokens=len(ids), generation_tokens=emitted,
            generation_tps=emitted / max(time.time() - t0, 1e-9),
            peak_memory=mx.get_peak_memory() / 1e9, from_draft=from_draft,
            accept_rate=(accepted_total / drafted_total) if drafted_total else 0.0,
            round_idx=rounds)

    _prompt_snap = None
    _ctx_snap = None
    cache = None
    try:
        with _wl(model, [_gs]), mx.stream(_gs):
            cache = (restore_from_snap(_entry["cache"], _entry["snap"])
                     if use_hit else model.make_cache())
            ctx = R.make_ctx()
            if use_hit and _entry.get("spec_ctx"):
                ctx = []
                for _t in _entry["spec_ctx"]:
                    c = CtxCache()
                    c.k, c.v = _t[0], _t[1]
                    c._abs = _t[2] if len(_t) > 2 else (0 if c.k is None else c.k.shape[2])
                    ctx.append(c)
            # prefill en 2 temps : snapshot a suffix[:-1] (= ids[:-1] au total),
            # dernier token forwarde a part (sa position bouge au tour suivant).
            _pre = suffix[:-1]
            _last = suffix[-1]
            if _pre:
                lp, cp = fwd(mx.array([_pre], dtype=mx.int32), cache)
                # eval-isolation : etat POOLE complet materialise avant snapshot
                mx.eval(lp, *cp.values(), *_cache_arrays(cache))
                R.update_ctx(mx.concatenate([cp[t] for t in taps], axis=-1), ctx)
            _prompt_snap = _snap(cache)
            _ctx_snap = [(c.k, c.v, c._abs) for c in ctx]
            # Store EAGER (pas dans le finally) : le runner casse la boucle sur
            # finish sans fermer ce generateur single-node (pas de servants a
            # OP_STOP comme en distribue) -> le finally serait differe/saute et
            # la session ne persisterait jamais (store_keys vide). Le snap
            # fin-de-prompt est deja fige ici ; le cache stocke = porteur de
            # classes pour le restore. Idempotent, re-store en finally en filet.
            try:
                if session_id and session_put is not None:
                    session_put(session_id, model_id, cache, ids,
                                _prompt_snap, ids[:-1], spec_ctx=_ctx_snap)
            except Exception as _e:
                sys.stderr.write(f"[spec-single] eager store failed: {_e!r}\n")
                sys.stderr.flush()
            ll, cl = fwd(mx.array([[_last]], dtype=mx.int32), cache)
            mx.eval(ll, *cl.values())
            R.update_ctx(mx.concatenate([cl[t] for t in taps], axis=-1), ctx)
            pending = int(mx.argmax(ll[:, -1, :]).item())
            emitted += 1
            finish = "stop" if pending in eos else (
                "length" if emitted >= max_tokens else None)
            yield _mk(pending, finish=finish)

            cold = since_probe = 0
            while finish is None:
                if cold >= _COLD_AFTER and since_probe < _COLD_PROBE:
                    draft = []
                    since_probe += 1
                else:
                    draft = R.propose(pending, ctx, embed, lm_head, K, MASK)
                    since_probe = 0
                    cold = 0 if draft else cold + 1
                drafted_total += len(draft)
                block = [pending] + draft
                vlog, vcaps = fwd(mx.array([block], dtype=mx.int32), cache)
                # Fix 2026-08-06 (racine du crash metal::malloc à ~12k tokens) :
                # l'etat POOLE de PoolingCache n'est PAS ancetre des logits (cf
                # _cache_arrays, lecon 1.24.0). Sans le materialiser ici, chaque
                # round empile un mx.concatenate lazy sur pooled ; la chaine
                # grossit avec l'offset (active reste plat car lazy != alloue)
                # puis se materialise d'un coup a haut offset -> pic ~275G >
                # limite 499G -> runner mort. On force la materialisation de
                # tout l'etat cache CHAQUE round (borne la chaine a 1 niveau),
                # meme pattern qu'au snapshot fin-de-prompt.
                mx.eval(vlog, *vcaps.values(), *_cache_arrays(cache))
                tt = [int(x) for x in mx.argmax(vlog[0], axis=-1).tolist()]
                n = 0
                while n < len(draft) and draft[n] == tt[n]:
                    n += 1
                committed = draft[:n] + [tt[n]]   # lossless : tt[n]=argmax target
                vfused = mx.concatenate([vcaps[t] for t in taps], axis=-1)
                if n < len(draft):
                    rollbacks += 1
                    _trim_all(cache, len(draft) - n)  # keep verify K/V — NO re-forward
                R.update_ctx(vfused[:, : n + 1, :], ctx)
                rounds += 1
                accepted_total += len(committed)
                if rounds % _CLEAR_CACHE_EVERY == 0:
                    try:
                        mx.clear_cache()
                    except Exception:
                        pass
                if _SPEC_MEM_DEBUG and rounds % 500 == 0:  # diag mémoire opt-in
                    try:
                        _cm = getattr(mx, "get_cache_memory", lambda: 0)()
                        sys.stderr.write(
                            f"[mem] emit={emitted} "
                            f"active={mx.get_active_memory()/1e9:.1f}G "
                            f"cache={_cm/1e9:.1f}G "
                            f"peak={mx.get_peak_memory()/1e9:.1f}G\n")
                        sys.stderr.flush()
                        mx.reset_peak_memory()
                    except Exception:
                        pass
                for tok in committed:
                    emitted += 1
                    finish = "stop" if tok in eos else (
                        "length" if emitted >= max_tokens else None)
                    yield _mk(tok, finish=finish, from_draft=(tok != committed[-1]))
                    if finish is not None:
                        break
                pending = committed[-1]
    finally:
        # store la session (snap fin-de-prompt + ctx drafter) — vaut aussi sur
        # disconnect client (GeneratorExit) : le snap fin-de-prompt reste valide.
        try:
            if (session_id and session_put is not None
                    and _prompt_snap is not None and cache is not None):
                session_put(session_id, model_id, cache, ids,
                            _prompt_snap, ids[:-1], spec_ctx=_ctx_snap)
        except Exception as _e:
            sys.stderr.write(f"[spec-single] session store failed: {_e!r}\n")
            sys.stderr.flush()
        try:
            _dt = max(time.time() - t0, 1e-9)
            _r = max(rounds, 1)
            sys.stderr.write(
                f"[spec-single] rounds={rounds} emitted={emitted} "
                f"accept/round={accepted_total / _r:.2f} "
                f"rollbacks={rollbacks}/{rounds} drafted={drafted_total} "
                f"cached={cached_n} tok/s={emitted / _dt:.2f}\n")
            sys.stderr.flush()
        except Exception:
            pass
