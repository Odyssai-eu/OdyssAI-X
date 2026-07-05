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

    inner = model.model            # dsv32-style inner: returns POST-norm hidden
    lm_head = model.lm_head

    # D8 "pre_norm" A/B: wrap the final norm to record its INPUT while
    # returning the normal output — one mechanism for every mlx-lm family.
    capture: dict[str, mx.array] = {}
    real_norm = inner.norm
    if hidden_source == "pre_norm":
        def _capture_norm(x):
            capture["h"] = x
            return real_norm(x)
        inner.norm = _capture_norm

    def trunk_forward(tokens: mx.array) -> tuple[mx.array, mx.array]:
        """One trunk pass -> (logits, hidden) for all S positions."""
        h_post = inner(tokens, trunk_cache)
        h = capture.pop("h") if hidden_source == "pre_norm" else h_post
        return lm_head(h_post), h

    try:
        # ── Prefill (chunked): trunk + mtp pairs (token_{p+1}, hidden_p) ──
        t0 = time.time()
        todo = prompt_ids[prefix_len:]
        if not todo:
            # Warm session cache holds the WHOLE prompt: no hidden was stored,
            # so re-forward the last prompt token (invariant F-5, 1-token cost).
            trim_prompt_cache(trunk_cache, 1)
            todo = prompt_ids[-1:]
            prefix_len = len(prompt_ids) - 1

        h_carry: Optional[mx.array] = None   # last hidden, spans chunk borders
        logits = h = None
        for i in range(0, len(todo), PREFILL_CHUNK):
            chunk = todo[i:i + PREFILL_CHUNK]
            toks = mx.array([chunk], dtype=mx.uint32)
            logits, h = trunk_forward(toks)
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

        bonus = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        seed_hidden = h[:, -1:, :]   # hidden of the position that produced bonus
        prompt_len_total = prefix_len + len(todo)

        detok = tokenizer.detokenizer
        detok.reset()

        emitted = 0
        accepted_total = 0
        drafted_total = 0
        round_idx = 0
        sha = hashlib.sha256()
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
        yield first

        while finish is None:
            round_idx += 1
            D = min(depth, max_tokens - emitted)  # never draft past the budget

            # (i) Draft chain: D sequential mtp steps, ONE position each.
            drafts: list[int] = []
            d_tok, d_hid = bonus, seed_hidden
            for _ in range(D):
                d_logits, d_hid = mtp.draft_step(
                    mx.array([[d_tok]], dtype=mx.uint32), d_hid, mtp_cache)
                d_tok = int(mx.argmax(d_logits[:, -1, :], axis=-1).item())
                drafts.append(d_tok)
            drafted_total += D

            # (ii) Verify: ONE trunk pass over [bonus, d0..dD-1] (D+1 pos).
            v_in = mx.array([[bonus] + drafts], dtype=mx.uint32)
            v_logits, v_hidden = trunk_forward(v_in)
            v_tokens_arr = mx.argmax(v_logits, axis=-1)
            mx.eval(v_tokens_arr)
            v_tokens = [int(t) for t in v_tokens_arr[0].tolist()]

            # (iii) Greedy exact-match acceptance.
            n = 0
            while n < D and v_tokens[n] == drafts[n]:
                n += 1
            new_bonus = v_tokens[n]
            accepted_total += n

            # (iv) Rollback. Trunk keeps n+1 of D+1. MTP drops ALL D
            # speculative entries, then re-forwards the n+1 accepted pairs
            # with TRUNK hiddens (advance == trunk's n+1; stays 1 behind).
            if D - n:
                trim_prompt_cache(trunk_cache, D - n)
            trim_prompt_cache(mtp_cache, D)
            upd_tokens = mx.array([[bonus] + drafts[:n]], dtype=mx.uint32)
            upd_hidden = mx.concatenate([seed_hidden, v_hidden[:, :n, :]], axis=1)
            mtp.draft_step(upd_tokens, upd_hidden, mtp_cache)

            # (v) Emit: the n accepted drafts + the new bonus (n+1 tokens).
            out_this_round = drafts[:n] + [new_bonus]
            for j, tok in enumerate(out_this_round):
                emitted += 1
                sha.update(tok.to_bytes(4, "little"))
                if tok in stop_ids:
                    finish = "stop"
                elif emitted >= max_tokens:
                    finish = "length"
                r = _mk(tok, from_draft=(j < n))
                if finish:
                    r.finish_reason = finish
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
            bonus = new_bonus
            seed_hidden = v_hidden[:, n:n + 1, :]

        # End-of-gen invariant: offset = prompt + emitted - 1 (final bonus
        # pending). A mid-round "stop" leaves the round's remaining verified
        # positions in the cache — reconcile by trimming, THEN assert.
        off = _cache_offset(trunk_cache)
        expect = prompt_len_total + emitted - 1
        if off > expect:
            trim_prompt_cache(trunk_cache, off - expect)
            trim_prompt_cache(mtp_cache, off - expect)
            off = _cache_offset(trunk_cache)
        assert off in (-1, expect), f"final cache drift: {off} != {expect}"
    finally:
        inner.norm = real_norm
        if canary_cb is not None:
            canary_cb(-1, drafted_total, accepted_total, sha.hexdigest()[:16])
