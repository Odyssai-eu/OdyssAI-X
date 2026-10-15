"""Anti-loop: detect a degenerate repetition in a stream of generated token ids.

One source for every local generation path (#113): runner.py imports it, and
install-mlx-vlm.sh step 7 copies it into mlx_vlm (as mlx_vlm/odyssai_loop_detect.py)
for the vlm-replica servers. Standard library only.
"""
from __future__ import annotations

from typing import Iterable, Optional

# ── Anti-loop — detect-and-stop on degenerate repetition ──────────────────
# When the TAIL of the generated token-id stream is >= N consecutive
# repetitions of the same p-token block, generation ends cleanly
# (finish_reason="stop" + `loop_detected` on the done event) instead of
# burning tokens to the max_tokens cap.
#
# Deterministic — a pure function of the emitted ids — so on multi-rank
# pools every rank reaches the same verdict at the same token and breaks in
# lockstep, exactly like the _stop_ids break in the stream loop. Distinct
# from the sampler-level no_repeat_ngram profiles (MODEL_SAMPLING_DEFAULTS):
# those BEND the distribution to avoid repeats (and can garble); this never
# touches sampling, it only ends the turn once the loop is proven. Per
# request opt-out via the fan-out field `anti_loop: false`.
ANTI_LOOP_CHECK_EVERY = 16   # tokens between checks
ANTI_LOOP_MIN_SPAN = 48      # repeating tail must cover >= this many tokens
ANTI_LOOP_MAX_PERIOD = 64    # longest repeating block considered
ANTI_LOOP_MIN_REPEATS = 4    # and repeat at least this many times
ANTI_LOOP_WINDOW = 640       # ids scanned (>= MAX_PERIOD * (MIN_REPEATS + 1))


# Large-period companion (2026-08-09, MiMo c02): multi-paragraph self-doubt
# cycles ("Actually, let me reconsider…" + code block — ~450-token period
# repeated 378× VERBATIM) sit far above ANTI_LOOP_MAX_PERIOD=64, invisible to
# _detect_loop by construction. Same contract: pure function of token ids →
# identical verdict on every rank (multi-rank lockstep safe). The anchor trick
# keeps it cheap: a period-p loop implies ids[-1] == ids[-1-p], so we slice-
# verify only anchored candidates; checked every 64 tokens, not 16.
ANTI_LOOP_L_CHECK_EVERY = 64    # tokens between large-period checks
ANTI_LOOP_L_MAX_PERIOD = 1024   # longest repeating block considered
ANTI_LOOP_L_MIN_REPEATS = 4     # 4 EXACT reps of a 65+-token block = stuck
ANTI_LOOP_L_WINDOW = ANTI_LOOP_L_MAX_PERIOD * (ANTI_LOOP_L_MIN_REPEATS + 1)


def detect_loop_large(ids: list) -> Optional[tuple]:
    """(period, repeats) when the tail loops with period 65..1024 tokens —
    the multi-paragraph regime _detect_loop cannot see. 4+ exact repetitions
    of a 65+-token block does not happen in legitimate output (three identical
    consecutive code stubs is the worst realistic case; four is pathological)."""
    n = len(ids)
    if n < (ANTI_LOOP_MAX_PERIOD + 1) * ANTI_LOOP_L_MIN_REPEATS:
        return None
    tail = ids[-ANTI_LOOP_L_WINDOW:]
    n = len(tail)
    last = tail[-1]
    for p in range(ANTI_LOOP_MAX_PERIOD + 1, min(ANTI_LOOP_L_MAX_PERIOD, n // 2) + 1):
        if tail[-1 - p] != last:
            continue
        r = 1
        while (n - (r + 1) * p >= 0
               and tail[n - (r + 1) * p: n - r * p] == tail[n - r * p: n - (r - 1) * p]):
            r += 1
        if r >= ANTI_LOOP_L_MIN_REPEATS:
            return p, r
    return None


def detect_loop(ids: list) -> Optional[tuple]:
    """(period, repeats) when the tail of `ids` is a degenerate loop, else None.

    Thresholds are deliberately conservative: 4+ EXACT repetitions of the
    same token block covering 48+ tokens is vanishingly rare in legitimate
    output (tables vary cell by cell, code lines differ), while a stuck
    model produces hundreds. Short periods need proportionally more repeats
    (period 1 → 48 repeats) so a stylistic "!!!" never trips it."""
    n = len(ids)
    if n < ANTI_LOOP_MIN_SPAN:
        return None
    tail = ids[-ANTI_LOOP_WINDOW:]
    n = len(tail)
    for p in range(1, ANTI_LOOP_MAX_PERIOD + 1):
        if n < 2 * p:
            break
        r = 1
        while (n - (r + 1) * p >= 0
               and tail[n - (r + 1) * p: n - r * p] == tail[n - r * p: n - (r - 1) * p]):
            r += 1
        if r >= max(ANTI_LOOP_MIN_REPEATS, -(-ANTI_LOOP_MIN_SPAN // p)):
            return p, r
    return None


class LoopGuard:
    """Per-request state for a decode loop that gets one OR several token ids per
    step (mlx_vlm yields `token_ids` on multi-token steps): runs detect_loop every
    ANTI_LOOP_CHECK_EVERY ids and detect_loop_large every ANTI_LOOP_L_CHECK_EVERY,
    the same cadence as the runner's per-token checks."""

    def __init__(self):
        self.ids: list = []

    def feed(self, new_ids: Iterable[int]) -> Optional[tuple]:
        for t in new_ids:
            self.ids.append(int(t))
            n = len(self.ids)
            if n % ANTI_LOOP_CHECK_EVERY == 0:
                hit = detect_loop(self.ids)
                if hit is None and n % ANTI_LOOP_L_CHECK_EVERY == 0:
                    hit = detect_loop_large(self.ids)
                if hit:
                    return hit
        return None
