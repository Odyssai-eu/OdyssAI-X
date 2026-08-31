"""Native-MTP speculative generation loop (greedy v0).

Drop-in generator for the runner's legacy loop: yields objects carrying the
same fields the `stream_generate` consumer reads (token/text/finish_reason/
tps), so `_run_legacy_main` swaps generators without touching its own logic.

Semantics contract (plan docs/PLAN-distributed-mtp.md §E1 — verified against
mlx-lm 0.31.3 generate.py:589-654 and the MTPLX GLM backend):

  * BONUS = the pending token. Sampled on the previous round, its K/V is NOT
    yet in the trunk cache. The verify forward processes [bonus, d0..dD-1]
    (D+1 positions) and writes them all; rollback trims (D - n).
  * Next round's draft seeds from verify_hidden[n] — the hidden of the LAST
    ACCEPTED position (bonus = index 0). Never a rejected position's hidden.
  * The MTP module keeps its own cache, advanced n+1 per round like the
    trunk: all D speculative entries are dropped (trim D) and the accepted
    positions re-forwarded in ONE batched mtp pass with TRUNK hiddens —
    speculative entries beyond step 1 were computed from mtp-approximated
    hiddens and must not pollute future drafts' attention. The mtp cache
    stays exactly ONE position behind the trunk's (pairs are shifted).
  * MTP prefill: the mtp block attends over the whole sequence, so after the
    trunk prefill each prompt position pair (token_{p+1}, trunk_hidden_p) is
    pushed through the mtp block chunk by chunk. One extra layer over the
    prompt — negligible vs the trunk's N layers.
  * Determinism v0 = greedy only (argmax). Multi-rank alignment relies on
    identical logits per rank (TP all_sum / PP all_gather, dsv32:374,473);
    accept counts are then identical BY CONSTRUCTION — and the canary
    callback lets the engine VERIFY that instead of assuming it.
  * Cache invariant (checked ~every ASSERT_EVERY tokens and at the end):
    trunk_cache_offset == prompt_len + emitted (pending bonus NOT in cache).

Greedy exactness property (unit-tested): for any depth D, the emitted token
sequence is IDENTICAL to plain AR greedy decoding of the same model.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional

import mlx.core as mx

PREFILL_CHUNK = 2048
ASSERT_EVERY = 64


@dataclass
class MTPResponse:
    """Duck-types the mlx_lm GenerationResponse fields the runner consumes."""
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
    # MTP extras (dashboard/#61 + canaries)
    accept_rate: float = 0.0
    round_idx: int = 0


def _cache_offset(cache: list) -> int:
    """Best-effort offset of the trunk cache (CacheList-aware); -1 = unknown."""
    for c in cache:
        probe = c
        try:
            probe = c[0]          # CacheList supports indexing (dsv32 usage)
        except Exception:
            pass
        off = getattr(probe, "offset", None)
        if isinstance(off, int):
            return off
    return -1


def _snapshot_hybrid_state(cache: list) -> list:
    """Round-start snapshot of a HYBRID trunk cache (#72 v1).

    Per entry: ArraysCache-like (rebound constant-size slots — deltanet conv,
    ssm state, PLE conv, n-gram tail) -> defensive copy of every slot;
    KVCache-like (offset-carrying; `_AttnCache.trim` also trims its indexer
    keys) -> the integer offset; CacheList -> recurse. The defensive copy
    guards against in-place slot mutation (review §2: the "refs suffice" bet
    is retired — the copy of constant-size states costs ~nothing).
    """
    snap: list = []
    for c in cache:
        if c is None:
            snap.append(None)
            continue
        children = getattr(c, "caches", None)
        if children is not None:                      # CacheList
            snap.append(("list", _snapshot_hybrid_state(children)))
            continue
        slots = getattr(c, "cache", None)
        if isinstance(slots, list):                   # ArraysCache family
            copies = [None if s is None else mx.array(s) for s in slots]
            lengths = getattr(c, "lengths", None)
            lp = getattr(c, "left_padding", None)
            snap.append(("arrays", copies,
                         None if lengths is None else mx.array(lengths),
                         None if lp is None else mx.array(lp)))
            continue
        off = getattr(c, "offset", None)
        if isinstance(off, int):                      # KVCache family
            snap.append(("kv", off))
            continue
        raise TypeError(
            f"hybrid snapshot: unsupported cache {type(c).__name__}")
    return snap


def _restore_hybrid_state(cache: list, snap: list) -> None:
    """Rebind ArraysCache slots to the snapshot and trim KV back to the
    round-start offset. Exact inverse of `_snapshot_hybrid_state`."""
    for c, s in zip(cache, snap):
        if s is None:
            continue
        kind = s[0]
        if kind == "list":
            _restore_hybrid_state(c.caches, s[1])
        elif kind == "arrays":
            _, copies, lengths, lp = s
            for i, v in enumerate(copies):
                c[i] = v
            c.lengths = lengths
            c.left_padding = lp
        else:                                          # "kv"
            delta = c.offset - s[1]
            if delta > 0:
                c.trim(delta)


def native_mtp_stream_generate(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    *,
    mtp: Any,                      # NativeMTPModule (bound to this trunk)
    depth: int = 3,
    max_tokens: int = 512,
    prompt_cache: Optional[list] = None,
    prefix_len: int = 0,           # positions already in prompt_cache
    hidden_source: str = "post_norm",   # D8 flag; "pre_norm" = capture wrapper
    stop_ids: Optional[set[int]] = None,
    canary_cb: Optional[Callable[[int, int, int, str], None]] = None,
) -> Generator[MTPResponse, None, None]:
    """Greedy native-MTP speculative decoding. See module docstring."""
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    if depth < 1:
        raise ValueError("depth must be >= 1")
    stop_ids = stop_ids or set()

    trunk_cache = prompt_cache if prompt_cache is not None else make_prompt_cache(model)
    mtp_cache = mtp.make_cache()

    # Family-aware trunk resolution: qwen3 wraps the text model one level down
    # under .language_model (VL config); deepseek/hy_v3 expose it at the top.
    # `inner(tokens, cache)` must return the POST-norm hidden; lm_head projects.
    if getattr(mtp, "family", "") == "qwen3":
        _lm = model.language_model
        inner = _lm.model
        lm_head = _lm.lm_head
    else:
        inner = model.model            # dsv32-style inner: returns POST-norm hidden
        lm_head = model.lm_head

    # A module family can impose its capture point (qwen4_exp: "hc_multi" —
    # the drafter consumes the PRE-final-mixer multi stream, vLLM scheme A).
    _hs_override = getattr(mtp, "hidden_source_override", None)
    if _hs_override:
        hidden_source = _hs_override
    # #72: hybrid linear-attn trunks roll back by snapshot/restore + commit
    # re-forward instead of trim (ArraysCache has no .trim).
    hybrid = bool(getattr(mtp, "hybrid_snapshot", False))

    # Capture wrappers: record the tapped tensor while returning the normal
    # output — one mechanism for every mlx-lm family.
    #  * "pre_norm" (D8 A/B): the final norm's INPUT.
    #  * "hc_multi" (qwen4_exp): the final hyper-connection mixer's INPUT,
    #    the [B, S, hc*H] multi stream.
    capture: dict[str, mx.array] = {}
    _unpatch: Optional[Callable[[], None]] = None
    if hidden_source == "pre_norm":
        real_norm = inner.norm
        def _capture_norm(x):
            capture["h"] = x
            return real_norm(x)
        inner.norm = _capture_norm
        def _unpatch_norm():
            inner.norm = real_norm
        _unpatch = _unpatch_norm
    elif hidden_source == "hc_multi":
        real_mixer = inner.hyper_connection_mixer
        def _capture_mixer(x):
            capture["h"] = x
            return real_mixer(x)
        inner.hyper_connection_mixer = _capture_mixer
        def _unpatch_mixer():
            inner.hyper_connection_mixer = real_mixer
        _unpatch = _unpatch_mixer

    # mlx-lm runs every generation forward inside a dedicated thread-local
    # stream (generate.py:226). Without it, the distributed pipeline's
    # send/recv collectives serialize against the default stream and every
    # forward pays a fixed ~4s stall (measured: constant 4.2s at any S,
    # vs 0.08s for AR which uses this stream). Same stream = same fast path.
    try:
        from mlx_lm.generate import generation_stream as _gen_stream
    except Exception:
        _gen_stream = mx.default_stream(mx.default_device())
    # mlx-lm holds a `wired_limit` context around the WHOLE generation
    # (generate.py:714) so the model's buffers (esp. the 768-expert MoE
    # weights, different experts routed per token) stay Metal-resident
    # instead of paging from disk each forward. Missing it = the ~4.5s
    # constant per forward we measured (paging-bound, S-independent).
    try:
        from mlx_lm.generate import wired_limit as _wired_limit
    except Exception:
        import contextlib
        def _wired_limit(_m, _s):  # no-op fallback
            return contextlib.nullcontext()

    def trunk_forward(tokens: mx.array) -> tuple[mx.array, mx.array]:
        """One trunk pass -> (logits, hidden) for all S positions."""
        with mx.stream(_gen_stream):
            h_post = inner(tokens, trunk_cache)
            h = (capture.pop("h")
                 if hidden_source in ("pre_norm", "hc_multi") else h_post)
            return lm_head(h_post), h

    # Enter the wired-limit context manually (avoids re-indenting the whole
    # generator body); exited in the finally below.
    _wl_ctx = _wired_limit(model, [_gen_stream])
    _wl_ctx.__enter__()
    # Initialized BEFORE the try: the finally's canary references them, and a
    # prefill-time crash must not shadow the real error with UnboundLocalError.
    accepted_total = 0
    drafted_total = 0
    sha = hashlib.sha256()
    _final_sent = False
    try:
        # ── Prefill (chunked): trunk + mtp pairs (token_{p+1}, hidden_p) ──
        t0 = time.time()
        todo = prompt_ids[prefix_len:]
        if not todo:
            if hybrid:
                # F-5's trim(1)+re-forward would double-advance the linear
                # states (no rollback on ArraysCache). Rebuild from scratch,
                # IN PLACE so the caller's session-cache list stays coherent —
                # correctness over the one-time prefill cost.
                trunk_cache.clear()
                trunk_cache.extend(make_prompt_cache(model))
                todo = prompt_ids
                prefix_len = 0
            else:
                # Warm session cache holds the WHOLE prompt: no hidden was
                # stored, so re-forward the last prompt token (invariant F-5).
                trim_prompt_cache(trunk_cache, 1)
                todo = prompt_ids[-1:]
                prefix_len = len(prompt_ids) - 1

        h_carry: Optional[mx.array] = None   # last hidden, spans chunk borders
        logits = h = None
        for i in range(0, len(todo), PREFILL_CHUNK):
            chunk = todo[i:i + PREFILL_CHUNK]
            toks = mx.array([chunk], dtype=mx.uint32)
            logits, h = trunk_forward(toks)
            # Same stream for the mtp pairs + eval (MLX_METAL_FAST_SYNCH:
            # cross-stream host reads return stale data, cf round loop).
            with mx.stream(_gen_stream):
                if h_carry is not None:
                    mtp_h = mx.concatenate([h_carry, h[:, :-1, :]], axis=1)
                    mtp_t = toks
                else:
                    # Very first token of the context has no predecessor pair.
                    mtp_h = h[:, :-1, :]
                    mtp_t = toks[:, 1:]
                if mtp_t.shape[1] > 0:
                    mtp.draft_step(mtp_t, mtp_h, mtp_cache)
                h_carry = h[:, -1:, :]
                mx.eval(logits)
        prompt_tps = len(todo) / max(time.time() - t0, 1e-9)

        with mx.stream(_gen_stream):
            bonus = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        seed_hidden = h[:, -1:, :]   # hidden of the position that produced bonus
        prompt_len_total = prefix_len + len(todo)

        detok = tokenizer.detokenizer
        detok.reset()

        emitted = 0
        round_idx = 0
        gen_t0 = time.time()
        finish: Optional[str] = None

        def _mk(tok: int, from_draft: bool) -> MTPResponse:
            detok.add_token(tok)
            return MTPResponse(
                text=detok.last_segment,
                token=tok,
                prompt_tokens=prompt_len_total,
                prompt_tps=prompt_tps,
                generation_tokens=emitted,
                generation_tps=emitted / max(time.time() - gen_t0, 1e-9),
                peak_memory=mx.get_peak_memory() / 1e9,
                from_draft=from_draft,
                accept_rate=(accepted_total / drafted_total) if drafted_total else 0.0,
                round_idx=round_idx,
            )

        # Final canary (round -1). Emitted BEFORE the terminal yield: the
        # runner's consumer breaks on finish_reason and never pulls again, so
        # anything after that yield only runs at generator GC — the engine's
        # accept-rate (dashboard pill) stayed null until the NEXT request.
        # Idempotent; the finally keeps a guarded copy for crash paths.
        def _final_canary():
            nonlocal _final_sent
            if not _final_sent and canary_cb is not None:
                _final_sent = True
                canary_cb(-1, drafted_total, accepted_total,
                          sha.hexdigest()[:16])

        # The prefill's argmax IS the first generated token (parity with AR /
        # mlx-lm, which yields it before any speculative round).
        emitted += 1
        sha.update(bonus.to_bytes(4, "little"))
        if bonus in stop_ids:
            finish = "stop"
        elif emitted >= max_tokens:
            finish = "length"
        first = _mk(bonus, from_draft=False)
        if finish:
            first.finish_reason = finish
            _final_canary()
        yield first

        # Instrumentation (TIMING_MTP=1): per-phase wall time, emitted in the
        # final canary. draft loop + verify both force eval (.item / mx.eval)
        # so these are REAL compute times, not lazy graph-build.
        _timing = os.environ.get("TIMING_MTP", "0") == "1"
        t_draft_tot = t_verify_tot = t_round_tot = 0.0
        t_snap_tot = t_restore_tot = t_commit_tot = 0.0
        # Hybrid round state (referenced by the final reconcile).
        round_snap: Optional[list] = None
        _rs_bonus = bonus
        _round_emitted = 0
        hybrid_flips = 0

        # One-time BISECT (collective — all ranks run each forward): isolate
        # what makes inner() 4.2s. lm_head is already known-fast (draft_step
        # runs it 3x in 68ms). Compare inner() on the POPULATED cache vs a
        # FRESH empty cache (pure 1-token, no context), and dump the cache
        # structure (entry count should match the LOCAL shard's layer count).
        if _timing and hybrid:
            # The bisect's populated-cache probe + trim(1) would desync the
            # linear states (no rollback) — KV-pur debug tool only.
            import sys as _sys
            _sys.stderr.write("[mtp-bisect] skipped (hybrid trunk)\n")
        if _timing and not hybrid:
            import sys as _sys
            from mlx_lm.models.cache import make_prompt_cache as _mkc
            _p = mx.array([[bonus]], dtype=mx.uint32)
            _sys.stderr.write(
                f"[mtp-bisect] cache_entries={len(trunk_cache)} "
                f"type={type(trunk_cache[0]).__name__} "
                f"offset={_cache_offset(trunk_cache)}\n")
            with mx.stream(_gen_stream):
                _t = time.time(); _h = inner(_p, trunk_cache); mx.eval(_h)
                _ti = time.time() - _t
                _t = time.time(); _l = lm_head(_h); mx.eval(_l)
                _tl = time.time() - _t
            trim_prompt_cache(trunk_cache, 1)
            _sys.stderr.write(
                f"[mtp-bisect] POPULATED S=1: inner={_ti*1000:.0f}ms "
                f"lm_head={_tl*1000:.0f}ms\n")
            _fc = _mkc(model)
            with mx.stream(_gen_stream):
                _t = time.time(); _h2 = inner(_p, _fc); mx.eval(_h2)
                _tf = time.time() - _t
            _sys.stderr.write(
                f"[mtp-bisect] FRESH S=1: inner={_tf*1000:.0f}ms "
                f"(entries={len(_fc)})\n")
            _sys.stderr.flush()

        while finish is None:
            round_idx += 1
            _tr = time.time()
            D = min(depth, max_tokens - emitted)  # never draft past the budget

            # (i) Draft chain: D sequential mtp steps, ONE position each.
            # SAME stream as the trunk: mixing the default stream with
            # _gen_stream corrupts host reads under MLX_METAL_FAST_SYNCH=1
            # (round-lagged argmax — the engine sets that env on every
            # runner; root-caused 2026-08-31, #72).
            _td = time.time()
            drafts: list[int] = []
            d_tok, d_hid = bonus, seed_hidden
            with mx.stream(_gen_stream):
                for _ in range(D):
                    d_logits, d_hid = mtp.draft_step(
                        mx.array([[d_tok]], dtype=mx.uint32), d_hid, mtp_cache)
                    d_tok = int(mx.argmax(d_logits[:, -1, :], axis=-1).item())
                    drafts.append(d_tok)
            drafted_total += D
            t_draft_tot += time.time() - _td

            # (i-bis) HYBRID: round-start snapshot BEFORE verify dirties the
            # linear states (the draft chain only touches mtp_cache).
            if hybrid:
                _rs_bonus = bonus
                _ts = time.time()
                with mx.stream(_gen_stream):
                    round_snap = _snapshot_hybrid_state(trunk_cache)
                t_snap_tot += time.time() - _ts

            # (ii) Verify: ONE trunk pass over [bonus, d0..dD-1] (D+1 pos).
            _tv = time.time()
            v_in = mx.array([[bonus] + drafts], dtype=mx.uint32)
            v_logits, v_hidden = trunk_forward(v_in)
            with mx.stream(_gen_stream):
                v_tokens_arr = mx.argmax(v_logits, axis=-1)
                mx.eval(v_tokens_arr)
            t_verify_tot += time.time() - _tv
            v_tokens = [int(t) for t in v_tokens_arr[0].tolist()]

            # (iii) Greedy exact-match acceptance.
            n = 0
            while n < D and v_tokens[n] == drafts[n]:
                n += 1
            new_bonus = v_tokens[n]
            accepted_total += n
            if os.environ.get("RUNNER_MTP_DEBUG_ROUNDS") and round_idx <= 4:
                import sys as _sys
                _vfin = bool(mx.isfinite(v_logits).all().item())
                _vmax = float(mx.max(v_logits).item())
                _hfin = bool(mx.isfinite(v_hidden.astype(mx.float32)).all().item())
                _sys.stderr.write(
                    f"[mtp-dbg] r{round_idx} bonus={bonus} drafts={drafts} "
                    f"v_tokens={v_tokens} n={n} "
                    f"v_finite={_vfin} v_max={_vmax:.2f} h_finite={_hfin} "
                    f"off={_cache_offset(trunk_cache)} "
                    f"seed_norm={float(mx.linalg.norm(seed_hidden.astype(mx.float32)).item()):.1f} "
                    f"inner={type(inner).__name__}\n")
                _sys.stderr.flush()

            # (iv) Rollback.
            if hybrid:
                # #72 v1 "verify on snapshot, commit by re-forward": restore
                # the linear slots + trim KV to ROUND START (not D-n), then
                # ONE batched re-forward of the accepted chunk rebuilds KV
                # and linear states exactly at the accepted position.
                _ts = time.time()
                with mx.stream(_gen_stream):
                    _restore_hybrid_state(trunk_cache, round_snap)
                t_restore_tot += time.time() - _ts
                _tc = time.time()
                c_in = mx.array([[bonus] + drafts[:n]], dtype=mx.uint32)
                c_logits, c_hidden = trunk_forward(c_in)
                with mx.stream(_gen_stream):
                    c_last = int(mx.argmax(c_logits[:, n, :], axis=-1).item())
                t_commit_tot += time.time() - _tc
                # Drift gate (review §1, AMENDED 2026-08-31): the linear-scan
                # kernels are S-dependent at bf16 noise level (verify S=D+1 vs
                # commit S=n+1 -> logit_maxdiff ~0.4 measured), so a TIED
                # top-2 can legitimately flip argmax (round-35 case:
                # top2_gap=0.0000, ground-truthed VERIFY-side correct). Rare
                # flips = kernel noise, tolerated (emission stays verify-truth
                # and the state is rebuilt from exact tokens each round — no
                # accumulation). SYSTEMATIC flips = a real restore bug (the
                # fast-synch race flipped 100% of rounds): trip on rate.
                if c_last != new_bonus:
                    hybrid_flips += 1
                    _dmax = float(mx.max(mx.abs(
                        c_logits[:, n, :].astype(mx.float32)
                        - v_logits[:, n, :].astype(mx.float32))).item())
                    import sys as _sys
                    _sys.stderr.write(
                        f"[mtp-hybrid] argmax flip {hybrid_flips} at round "
                        f"{round_idx} (n={n}, commit {c_last} vs verify "
                        f"{new_bonus}, logit_maxdiff={_dmax:.4f}) — "
                        f"emitting verify\n")
                    _sys.stderr.flush()
                    if hybrid_flips >= 8 and hybrid_flips > 0.10 * round_idx:
                        raise AssertionError(
                            f"hybrid commit drift SYSTEMATIC: "
                            f"{hybrid_flips} flips in {round_idx} rounds "
                            f"(last: commit {c_last} != verify {new_bonus}, "
                            f"n={n}, logit_maxdiff={_dmax:.4f}) — restore "
                            f"presumed broken, refusing to serve")
                upd_tokens = c_in
                upd_hidden = mx.concatenate(
                    [seed_hidden, c_hidden[:, :n, :]], axis=1)
            else:
                # Trunk keeps n+1 of D+1.
                if D - n:
                    trim_prompt_cache(trunk_cache, D - n)
                upd_tokens = mx.array([[bonus] + drafts[:n]], dtype=mx.uint32)
                upd_hidden = mx.concatenate(
                    [seed_hidden, v_hidden[:, :n, :]], axis=1)
            # MTP drops ALL D speculative entries, then re-forwards the n+1
            # accepted pairs with TRUNK hiddens (advance == trunk's n+1;
            # stays 1 behind).
            with mx.stream(_gen_stream):
                trim_prompt_cache(mtp_cache, D)
                mtp.draft_step(upd_tokens, upd_hidden, mtp_cache)

            # (v) Emit: the n accepted drafts + the new bonus (n+1 tokens).
            out_this_round = drafts[:n] + [new_bonus]
            _round_emitted = 0
            for j, tok in enumerate(out_this_round):
                emitted += 1
                _round_emitted += 1
                sha.update(tok.to_bytes(4, "little"))
                if tok in stop_ids:
                    finish = "stop"
                elif emitted >= max_tokens:
                    finish = "length"
                r = _mk(tok, from_draft=(j < n))
                if finish:
                    r.finish_reason = finish
                    _final_canary()
                yield r
                if finish:
                    break

            # (vi) Canary + periodic cache-drift assert (~every ASSERT_EVERY).
            if canary_cb is not None:
                canary_cb(round_idx, D, n, sha.hexdigest()[:16])
            if finish is None and emitted % ASSERT_EVERY < (n + 1):
                off = _cache_offset(trunk_cache)
                expect = prompt_len_total + emitted - 1  # new bonus is pending
                assert off in (-1, expect), (
                    f"cache drift: offset={off} expected={expect} "
                    f"(round {round_idx}, emitted {emitted})")

            # (vii) Seed next round (invariant F-1: LAST ACCEPTED position).
            # Hybrid: the COMMITTED hidden is canonical (review §1 —
            # v_hidden[n] would be a silent-drift channel the canary can't see).
            bonus = new_bonus
            seed_hidden = (c_hidden[:, n:n + 1, :] if hybrid
                           else v_hidden[:, n:n + 1, :])
            t_round_tot += time.time() - _tr
            if _timing and round_idx % 8 == 0:
                import sys as _sys
                _hyb = (f"snap={t_snap_tot/round_idx*1000:.0f}ms "
                        f"restore={t_restore_tot/round_idx*1000:.0f}ms "
                        f"commit={t_commit_tot/round_idx*1000:.0f}ms "
                        if hybrid else "")
                _sys.stderr.write(
                    f"[mtp-timing] round {round_idx}: draft={t_draft_tot/round_idx*1000:.0f}ms "
                    f"verify={t_verify_tot/round_idx*1000:.0f}ms "
                    f"{_hyb}"
                    f"round={t_round_tot/round_idx*1000:.0f}ms "
                    f"(other={((t_round_tot-t_draft_tot-t_verify_tot)/round_idx)*1000:.0f}ms)\n")
                _sys.stderr.flush()

        # End-of-gen invariant: offset = prompt + emitted - 1 (final bonus
        # pending). A mid-round "stop" leaves the round's remaining verified
        # positions in the cache — reconcile by trimming, THEN assert.
        off = _cache_offset(trunk_cache)
        expect = prompt_len_total + emitted - 1
        if off > expect:
            if hybrid and round_snap is not None:
                # Trimming would desync the linear states. Restore to round
                # start and re-forward EXACTLY the emitted chunk (the last
                # emitted token stays pending, like the bonus convention).
                _restore_hybrid_state(trunk_cache, round_snap)
                k = _round_emitted
                if k > 0:
                    trunk_forward(mx.array([[_rs_bonus] + drafts[:k - 1]],
                                           dtype=mx.uint32))
                trim_prompt_cache(mtp_cache, off - expect)
            else:
                trim_prompt_cache(trunk_cache, off - expect)
                trim_prompt_cache(mtp_cache, off - expect)
            off = _cache_offset(trunk_cache)
        assert off in (-1, expect), f"final cache drift: {off} != {expect}"
    finally:
        if _unpatch is not None:
            _unpatch()
        try:
            _wl_ctx.__exit__(None, None, None)
        except Exception:
            pass
        if canary_cb is not None and not _final_sent:
            canary_cb(-1, drafted_total, accepted_total, sha.hexdigest()[:16])
