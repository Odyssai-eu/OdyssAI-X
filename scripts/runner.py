"""Long-lived JACCL runner with tensor + pipeline support.

Init MLX distributed once, load model once, loop on JSONL prompts. Tensor mode
uses exo-ported auto_parallel for models that mlx-lm's sharded_load doesn't
natively support (e.g. qwen3_next).

Stdin protocol (one JSON per line):
    {"cmd": "gen", "id": "...", "prompt": "...", "max_tokens": 200}
    {"cmd": "gen", "id": "...", "messages": [...], "tools": [...], "max_tokens": 200}
    {"cmd": "stop"}

Stdout events (rank 0 only, one JSON per line):
    {"event": "ready", ...}
    {"event": "token", "id": "...", "text": "..."}
    {"event": "done", "id": "...", "tool_calls": [...], ...}
    {"event": "bye"}
"""

import hashlib
import json
import os
import pickle
import queue
import re
import signal
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.utils import load_model, sharded_load, hf_repo_to_path
from mlx_lm import tokenizer_utils
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# Continuous-batching generator (mlx-lm built-in, used in single-rank mode for
# concurrent multi-request serving). Multi-rank stays on stream_generate so
# that JACCL collectives stay aligned across ranks.
try:
    from mlx_lm.generate import BatchGenerator  # type: ignore
    _BATCH_AVAILABLE = True
    # ── Monkey-patch _extend_cache (mlx-lm 0.31.3 bug) ────────────────
    # Upstream `_extend_cache(cache_a, cache_b)` does
    # `for ca, cb in zip(cache_a, cache_b): ca.extend(cb)` — but for
    # certain cache types (Q8 quantized, MLA, some fp16 variants on
    # newer model families), the cache object is backed by an mx.array
    # that doesn't expose `.extend`. Result: every 2nd request through
    # BatchGenerator dies with `'array' object has no attribute 'extend'`
    # and the BG's internal _prompt_batch.prompt_cache stays poisoned
    # until the BG is re-created.
    #
    # Workaround: replace cache_b's entries into cache_a in-place when
    # extend is unavailable. We lose cache reuse for that pair (the new
    # request starts cold for that layer), but the runner keeps serving.
    # When mlx-lm fixes _extend_cache upstream, this patch becomes a
    # no-op (ca.extend works again, the else branch never fires).
    try:
        import sys as _sys_for_bg
        # `import mlx_lm.generate as X` resolves to the `generate` FUNCTION
        # (because mlx_lm/__init__.py re-exports it from the submodule),
        # not the module itself. Reach the module via sys.modules instead.
        _bg_module = _sys_for_bg.modules["mlx_lm.generate"]

        # ── Bug #1: _extend_cache lacks fallback for non-list caches ──
        # `for ca, cb in zip(a, b): ca.extend(cb)` — some cache flavours
        # are mx.array-backed and have no .extend. Replace those entries
        # instead of crashing.
        def _safe_extend_cache(cache_a, cache_b):
            if not cache_a:
                return cache_b
            if not cache_b:
                return cache_a
            for i, (ca, cb) in enumerate(zip(cache_a, cache_b)):
                try:
                    ca.extend(cb)
                except AttributeError:
                    cache_a[i] = cb
            return cache_a

        _bg_module._extend_cache = _safe_extend_cache  # type: ignore[attr-defined]

        # ── Bug #2: GenerationBatch.extend assumes _current/_next_logprobs
        #            are always lists. They can become mx.array after the
        #            first extend (line 1304 assignment), then line 1309
        #            crashes with 'array' object has no attribute 'extend'.
        # Repro: every 2nd request through the BG dies, runner emits
        #        completion_tokens=0, Companion shows nothing.
        # Wrap the method to coerce array → list before extending.
        _GB = _bg_module.GenerationBatch  # type: ignore[attr-defined]
        _orig_gb_extend = _GB.extend

        def _safe_gb_extend(self, batch):
            # Mirror upstream logic but coerce array → list at the
            # extend points that crash. We do NOT touch the mx.concatenate
            # branches (those work fine).
            self.uids.extend(batch.uids)
            self.prompt_cache = _bg_module._extend_cache(
                self.prompt_cache, batch.prompt_cache
            )
            self.tokens.extend(batch.tokens)
            self.samplers.extend(batch.samplers)
            self.logits_processors.extend(batch.logits_processors)
            self.max_tokens.extend(batch.max_tokens)
            self.state_machines.extend(batch.state_machines)
            import mlx.core as _mx_core
            if self._current_tokens is None:
                self._current_tokens = batch._current_tokens
                self._current_logprobs = batch._current_logprobs
            elif batch._current_tokens is not None:
                self._current_tokens = _mx_core.concatenate(
                    [self._current_tokens, batch._current_tokens]
                )
                if isinstance(self._current_logprobs, _mx_core.array):
                    self._current_logprobs = list(self._current_logprobs)
                if isinstance(batch._current_logprobs, _mx_core.array):
                    self._current_logprobs.extend(list(batch._current_logprobs))
                else:
                    self._current_logprobs.extend(batch._current_logprobs)
            if self._next_tokens is None:
                self._next_tokens = batch._next_tokens
                self._next_logprobs = batch._next_logprobs
            elif batch._next_tokens is not None:
                self._next_tokens = _mx_core.concatenate(
                    [self._next_tokens, batch._next_tokens]
                )
                if isinstance(self._next_logprobs, _mx_core.array):
                    self._next_logprobs = list(self._next_logprobs)
                if isinstance(batch._next_logprobs, _mx_core.array):
                    self._next_logprobs.extend(list(batch._next_logprobs))
                else:
                    self._next_logprobs.extend(batch._next_logprobs)
            self._token_context.extend(batch._token_context)
            self._num_tokens.extend(batch._num_tokens)
            self._matcher_states.extend(batch._matcher_states)

        _GB.extend = _safe_gb_extend  # type: ignore[method-assign]
    except Exception:
        pass
except Exception:
    BatchGenerator = None  # type: ignore
    _BATCH_AVAILABLE = False

# KV cache helpers (Q8 quantized cache halves the per-token KV memory cost).
try:
    from mlx_lm.models.cache import make_prompt_cache, QuantizedKVCache  # type: ignore
    _CACHE_AVAILABLE = True
except Exception:
    make_prompt_cache = None  # type: ignore
    QuantizedKVCache = None  # type: ignore
    _CACHE_AVAILABLE = False
try:
    from mlx_lm.models.cache import CacheList as _CacheListCls  # type: ignore
except Exception:
    _CacheListCls = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Session-based prefix cache
#
# When a client passes the same `session_id` across multi-turn requests, we
# keep the populated `prompt_cache` in memory keyed by session. On the next
# request we tokenize the new templated prompt; if it begins with the stored
# tokens, we feed only the SUFFIX through `stream_generate(prompt_cache=…)` —
# the prefix's K/V are already in the cache. Big TTFT win for chat.
#
# Invariants:
#   - only one entry per session (overwrites on each turn)
#   - bound: TTL 1h, max 32 sessions, evicted LRU
#   - invalidated when the loaded model changes (model_id mismatch)
#   - dropped on prefix divergence (user backtracks / system prompt changes)
#
# The runner is single-threaded; the lock is only against the eviction sweep.
# ──────────────────────────────────────────────────────────────────────────────
import threading

# Prefix-cache par snapshot pour les caches NON-trimmables (V4 PoolingCache).
# Defaut ON : le tour-2 pendait parce que le snapshot capturait des arrays
# ENCORE LAZY (l'etat poole de PoolingCache n'est PAS ancetre des logits
# courants, donc mx.eval(logits) ne le materialisait pas) — au tour suivant,
# restaurer puis forward ces arrays re-executait le graphe de collectives
# distribues du tour precedent HORS lockstep -> deadlock. Corrige :
# _manual_prefill force desormais l'eval de l'ETAT COMPLET du cache (voir
# _cache_state_arrays), le snapshot est donc inerte. RUNNER_SNAP_CACHE=0 pour
# desactiver (repli au re-prefill integral).
_SNAP_CACHE_ENABLED = os.environ.get("RUNNER_SNAP_CACHE", "1") == "1"

_session_store: dict[str, dict] = {}
_session_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────────────────────
# Hard-cancel registry — populated by the reader thread when a "cancel" cmd
# arrives, drained by the gen loops between tokens (legacy) or in the
# request-drain step (batched). The API broadcasts cancel to every rank,
# so the set converges across the cluster within one token-emit cycle.
#
# Why per-req_id and not a global flag: an admin cancel must only affect
# the targeted request — multi-rank pools can have multiple sessions
# queued behind a long-running gen, and we don't want to drop them all.
# ──────────────────────────────────────────────────────────────────────────────
_cancelled_ids: set[str] = set()
_cancelled_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────────────────────
# Per-model sampling overrides.
#
# mlx-lm's stream_generate defaults to temp=0 (greedy) with no repetition
# penalty. Greedy on conversational prompts with chat-tuned models like
# MiniMax-M2 can trigger degenerate repetition loops — the model picks
# the locally-best next token, lands in an attractor, and emits the same
# 2-paragraph block ~30 times until max_tokens runs out (observed
# NEMO-20-5, 2026-05-20: 579 chunks, 142s, no EOS).
#
# This table lets us add targeted guardrails per model without changing
# defaults for everything else. Keyed by case-insensitive substring of
# the model repo name. Order matters: first match wins, so put more
# specific substrings before general ones.
#
# Each entry passes through to make_sampler / make_logits_processors:
#   - temp, top_p, top_k       → sampler shape
#   - repetition_penalty       → penalize already-generated tokens
#   - repetition_context_size  → how far back to look (default 20)
#
# If no entry matches, mlx-lm defaults apply — same behavior as before.
MODEL_SAMPLING_DEFAULTS: dict[str, dict] = {
    # MiniMax-M3 (bilingual prose engine). MUST come before the general
    # "minimax" entry — first match wins, and "minimax-m3" is a substring of
    # the repo path while "minimax" alone is the fallback for other MiniMax
    # checkpoints (e.g. M2.x coder/tool models, where no_repeat_ngram would
    # wrongly ban legitimate repeated syntax/JSON/indentation).
    "minimax-m3": {
        "temp": 0.7,
        "top_p": 0.9,
        # 1.05, not 1.1/1.15: ABOVE ~1.15 the penalty hits words that MUST
        # recur (names, "fréquence", articles) and pushes toward rarer
        # alternatives — including high-frequency CJK tokens (the zh-leak
        # 陪伴/可能). But even 1.1 was too strong at the thematically-critical
        # final beat: on a Bruit-Blanc Q8 run (2026-06-14) the model bailed to
        # EOS at "Elma a entendu la [fréquence]" because "fréquence" (used 4x)
        # was penalized below the stop token, cutting the killer final image.
        # 1.05 keeps a light touch on real loops (n=8 ngram ban does the heavy
        # lifting) without starving the recurring words the story is built on.
        "repetition_penalty": 1.05,
        # 512, not 128: M3's verbatim clause loops recur at 200-400 tokens'
        # distance ("équations fondamentales régissant fonctionnement système"
        # ~10x). mlx-lm's repetition_penalty only looks back
        # repetition_context_size tokens, so 128 never even sees the clause it
        # is repeating. 512 covers the multi-paragraph cycle.
        "repetition_context_size": 512,
        # Hard-bans repeating any N-gram (HF-style) to kill verbatim clause
        # recycling. 8, NOT 4: at n=4 the ban fires after only a 3-token prefix
        # match, which in narrative prose hits LEGITIMATE repeats constantly
        # (character actions "...sur son épaule" → bans "gauche"/"droite";
        # recurring names) and forces the model into garbled mid-word
        # substitutions ("épaucherche", "habituellle") — corruption that
        # compounds with context length. Verified on a 2800-word Bruit-Blanc Q8
        # run (2026-06-14): the typos clustered exactly where common phrases
        # recurred. n=8 still catches real loops (a looping clause of unit>=4
        # tokens repeats 8-grams across its boundary) but spares legitimate
        # short prose repeats. mlx-lm has no native support; see
        # make_no_repeat_ngram_processor. SCOPED to M3 — do NOT inherit into "minimax".
        "no_repeat_ngram_size": 8,
        # CJK language-lock: bans the ~55,600 CJK/Hangul/kana/fullwidth token
        # ids from the vocab. M3 is bilingual fr/zh, so the Chinese tokens carry
        # a small but non-zero mass everywhere. Empirically (2026-06-14): greedy
        # is clean (argmax is always French), and min_p does NOT cut the leak —
        # the zh synonyms that leak (改良/无处不在 after "Sonde audio"/"était")
        # are COMPETITIVE candidates (~5% rel. prob at temp 0.7), above any sane
        # min_p floor, so temp sampling draws one ~1 run in 2 over a long text.
        # No sampling lever removes it cleanly; only a deterministic ban does,
        # and a ban touches ZERO French logits (the model falls to the French
        # #1 candidate greedy already picked — verified, prose intact). Odysseus
        # serves M3 as a French writer and never needs zh output, so the ban is
        # default-on. SCOPED to M3 (the "minimax" fallback below does NOT lock).
        "cjk_lock": True,
        # EOS guard (report §9.4): pin every stop token to the gap it held below
        # the natural content leader BEFORE the repetition penalty, so the
        # penalty can no longer promote EOS by side effect (the "fréquence"
        # mid-sentence bailout). Parameter-free, monotonic-safe. SCOPED to M3 —
        # the "minimax" fallback below does NOT enable it.
        "eos_guard": True,
    },
    # Other MiniMax checkpoints (M2.x coder/tool). Conservative, unchanged from
    # the original guardrail — no no_repeat_ngram (would break code/tool output).
    "minimax": {
        "temp": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        # 128 catches multi-paragraph cycles (the NEMO-20-5 loop was a
        # ~150-token block repeating; 20 was way too short to detect).
        "repetition_context_size": 128,
    },
    # MiMo-V2.5-Pro (Xiaomi). Had no profile until 2026-07-09 → fell through to
    # mlx-lm greedy + no repetition penalty, the exact degenerate-loop recipe
    # this table exists to prevent (see the NEMO-20-5 comment above). Observed on
    # a 4-node pipeline: a long code generation (repetitive bash `${...}`)
    # collapsed into a 20802-token run of a single char with NO EOS — the
    # non-stream path buffered the whole response so Companion showed nothing
    # while the engine kept emitting. Authors' recommended sampling is
    # temp 1.0 / top_p 0.95. repetition_penalty 1.05 + context 512 breaks the
    # loop; NO no_repeat_ngram — mimo generates code, and the n-gram ban drops
    # operators/identifiers and corrupts syntax (same reason the "minimax" M2
    # entry above omits it).
    "mimo": {
        "temp": 1.0,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
        "repetition_context_size": 512,
    },
}


def make_no_repeat_ngram_processor(ngram_size: int):
    """A logits processor that hard-bans repeating any `ngram_size`-gram, the
    way HF transformers' `no_repeat_ngram_size` does. mlx-lm ships repetition /
    presence / frequency penalties but not this — and those penalties are the
    wrong tool for M3's failure mode: they soft-discount individual *tokens*
    seen in the last N positions, whereas M3 recycles whole *clauses* verbatim
    ("équations fondamentales régissant fonctionnement système" ~10x in one
    Bruit-Blanc run). Banning the n-gram kills the clause loop surgically
    without touching legitimate reuse of common short words. See OdyssAI-X#53.

    Signature matches mlx-lm processors: (tokens, logits) -> logits, where
    `tokens` is the full mx.array of prompt+generated ids (it grows by >=1 each
    call — verified against mlx-lm 0.31.3 generate.py, `mx.concat`) and `logits`
    is the [1, vocab] current-step logits. State is per-processor (rebuilt fresh
    per request by _build_sampling_for) and updated incrementally, so the cost
    is O(new tokens) per step — amortized O(1) — not O(context).
    """
    n = int(ngram_size)
    history: list[int] = []
    prefix_map: dict[tuple, set] = {}

    def processor(tokens, logits):
        # Sync the running history with the token array, indexing every newly
        # completed n-gram: its (n-1)-token prefix maps to the token that
        # followed. The first call delivers the whole prompt at once.
        cur_len = int(tokens.shape[0]) if hasattr(tokens, "shape") else len(tokens)
        if cur_len > len(history):
            tail = tokens[len(history):]
            new_ids = tail.tolist() if hasattr(tail, "tolist") else list(tail)
            for t in new_ids:
                history.append(int(t))
                if len(history) >= n:
                    prefix_map.setdefault(tuple(history[-n:-1]), set()).add(history[-1])
        if len(history) >= n - 1:
            banned = prefix_map.get(tuple(history[-(n - 1):]))
            if banned:
                logits[:, mx.array(list(banned))] = -float("inf")
        return logits

    return processor


_CJK_BANNED_CACHE: dict = {}


def _cjk_banned_ids(tokenizer):
    """Token ids whose decoded text contains a CJK / Hangul / kana / fullwidth
    char. Scanned once per tokenizer (~15-20s over the 200k vocab) then cached.
    Decode-based, not raw-vocab-string: byte-level BPE tokens for CJK chars are
    byte-encoded in the raw vocab (e.g. 'æ\\x94¹' for 改) and would be missed;
    decode([id]) reconstructs the real character. See cjk_lock in
    MODEL_SAMPLING_DEFAULTS for why M3 needs this.
    """
    key = id(tokenizer)
    cached = _CJK_BANNED_CACHE.get(key)
    if cached is not None:
        return cached

    def is_cjk(c):
        o = ord(c)
        return (0x3000 <= o <= 0x303F or 0x3040 <= o <= 0x30FF or
                0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or
                0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or
                0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF or
                0x20000 <= o <= 0x2FA1F)

    try:
        V = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer.get_vocab()))
    except Exception:
        V = len(tokenizer.get_vocab())
    banned = []
    for tid in range(V):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            continue
        if s and any(is_cjk(c) for c in s):
            banned.append(tid)
    arr = mx.array(banned)
    _CJK_BANNED_CACHE[key] = arr
    log(f"cjk_lock: banned {len(banned)}/{V} CJK/Hangul token ids")
    return arr


def make_cjk_lock_processor(tokenizer):
    """A logits processor that hard-bans every CJK/Hangul/kana token (-inf),
    deterministically suppressing the French->Chinese leak without touching any
    French logit. Cost per step is one scatter over the cached banned-id array,
    negligible against the MoE forward."""
    banned = _cjk_banned_ids(tokenizer)

    def processor(tokens, logits):
        logits[:, banned] = -float("inf")
        return logits

    return processor


def make_eos_guard_processors(stop_ids):
    """Decouple the stop decision from the repetition penalty — report §9.4.

    M3's `repetition_penalty` was doing EOS control by side effect: it discounts
    recently-seen content tokens, so when a *thematic* word (the Bruit-Blanc
    "fréquence", recurring 4x) is pushed below the stop token, the model emits
    EOS mid-sentence and the ending is cut. The historical fix was to hand-tune
    the penalty down to 1.05 — a magic number "calibré à fréquence" that couples
    two concerns which should be independent.

    This is the parameter-free decoupling. Returns (snapshot, clamp), two
    cooperating logits processors:
      • `snapshot` runs FIRST (raw logits) and records, per step, the gap each
        stop token sits below the natural content leader: gap0 = leader0 - stop0,
        where leader0 = max over NON-stop logits before any penalty.
      • `clamp` runs LAST (after rep-penalty + ngram + cjk) and enforces
        stop <= leader1 - gap0, where leader1 = max over surviving NON-stop
        logits.
    Effect: a stop token keeps EXACTLY its pre-penalty standing relative to the
    content leader — no more, no less. A stop that naturally led (gap0 < 0) gets
    a cap above the leader, so a genuine end is never suppressed (no runaway). A
    stop the model did not want (gap0 > 0) can never close that gap by penalty
    alone, so there is no penalty-induced premature EOS. The gap is *measured*,
    not chosen: no tunable threshold replaces the old 1.05. Cost is two
    vocab-wide maxes per step, negligible against the MoE forward (cf. cjk_lock).

    Order is load-bearing: snapshot must be inserted at the FRONT of the
    processor chain and clamp APPENDED at the end (see _build_sampling_for). The
    guard is monotonic — it only ever lowers a stop logit that the penalty would
    have promoted, never raises anything — so it cannot make a clean run worse.
    """
    ids = sorted({int(s) for s in stop_ids})
    if not ids:
        def _noop(tokens, logits):
            return logits
        return _noop, _noop
    stop_arr = mx.array(ids)
    NEG = -float("inf")
    state: dict = {"gap0": None, "neg_mask": None}

    def _neg_mask(logits):
        nm = state["neg_mask"]
        if nm is None or nm.shape != logits.shape:
            nm = mx.zeros(logits.shape, dtype=logits.dtype)
            nm[:, stop_arr] = NEG
            state["neg_mask"] = nm
        return state["neg_mask"]

    def snapshot(tokens, logits):
        leader0 = mx.max(logits + _neg_mask(logits), axis=-1, keepdims=True)
        state["gap0"] = leader0 - logits[:, stop_arr]
        return logits

    def clamp(tokens, logits):
        gap0 = state["gap0"]
        if gap0 is None:
            return logits
        leader1 = mx.max(logits + _neg_mask(logits), axis=-1, keepdims=True)
        if not (float(leader1.sum()) > NEG):
            return logits  # all non-stop logits banned — let the stop through.
        logits[:, stop_arr] = mx.minimum(logits[:, stop_arr], leader1 - gap0)
        return logits

    return snapshot, clamp


# ── Request-type detection (prose vs code) ─────────────────────────────────
# Per-request sampling profile (integration report §9.1). MODEL_SAMPLING_DEFAULTS
# is keyed per-MODEL, but M3 is BOTH a prose writer AND a coder — and the prose
# profile's no_repeat_ngram ban drops operators in code (T03/T04/C01 all failed
# ONLY on token-drops: missing comma / "=" / identifier). A light heuristic flips
# the profile per request: STRONG code signals only, so prose (Bruit Blanc, GDPR)
# stays prose. False positive = a prose loop slips through; false negative = code
# token-drops — we bias toward catching real code without flagging prose.
_CODE_EXT_RE = re.compile(
    r"\.(py|js|ts|tsx|jsx|swift|c|cc|cpp|h|hpp|rs|go|java|rb|sh|bash|sql|kt|"
    r"php|scala|lua|mm|cs|jl)\b", re.I)
_CODE_VERB_RE = re.compile(
    r"\b(écris|ecris|write|crée|cree|create|implémente|implemente|implement|"
    r"génère|genere|generate|coder?|debug|débug|debogue|refactor|corrige|fix)\b",
    re.I)
_CODE_NOUN_RE = re.compile(
    r"\b(script|fonction|function|classe|class|programme|program|CLI|module|"
    r"méthode|method|endpoint|regex|algorithme|algorithm|snippet)\b", re.I)


def _last_user_text(messages) -> str:
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(
                    p.get("text", "") for p in c
                    if isinstance(p, dict) and p.get("type") in (None, "text")
                )
    return ""


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

# ── Context-limit — hard cap on total context (prompt + generated) ──────────
# Server-wide guard, INDEPENDENT of the client's per-request max_tokens. When
# prompt + generated tokens reach this cap, the turn ends cleanly with a
# `context_limit` flag on the done event (finish_reason stays "length" for
# OpenAI-compat clients). Prevents a runaway generation from saturating the
# worker / OOMing before max_tokens. 0 = off. Per-request override via the
# `context_limit` field. Inspired by Inferencer 2.3.1's contextLimit; enforced
# per-step here, not just at enqueue (a long prefill can already exceed).
_CONTEXT_LIMIT = int(os.environ.get("RUNNER_CONTEXT_LIMIT", "0"))


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


def _detect_loop_large(ids: list) -> Optional[tuple]:
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


def _detect_loop(ids: list) -> Optional[tuple]:
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


def _looks_like_code_request(messages, tools=None) -> bool:
    """Conservative: True only on strong code signals — a ``` fence, a source
    file extension, a code verb + code noun IN PROXIMITY (same clause), or
    injected tools (= the auto-router judged this agentic/code intent).
    Proximity matters: a prose spec can carry an incidental "function" (e.g. the
    Bruit-Blanc JSON's `function_narrative` key) far from an "Écris" — that must
    NOT flip the profile on the very prompt the prose sampling was tuned for."""
    if tools:
        return True
    txt = _last_user_text(messages)
    if not txt:
        return False
    if "```" in txt:
        return True
    if _CODE_EXT_RE.search(txt):
        return True
    # code verb with a code noun within ~40 chars (either side) = real intent.
    for vm in _CODE_VERB_RE.finditer(txt):
        window = txt[max(0, vm.start() - 40): vm.end() + 40]
        if _CODE_NOUN_RE.search(window):
            return True
    return False


def _build_sampling_for(repo: str, tokenizer=None, is_code: bool = False):
    """Return (sampler, logits_processors) for this model, or (None, None)
    to use mlx-lm defaults. Match is case-insensitive substring on repo path.
    `is_code` flips M3 to a code-safe profile (drops the prose no_repeat_ngram).
    """
    needle = repo.lower()
    for key, params in MODEL_SAMPLING_DEFAULTS.items():
        if key in needle:
            sampler = make_sampler(
                temp=float(params.get("temp", 0.0)),
                top_p=float(params.get("top_p", 0.0)),
                min_p=float(params.get("min_p", 0.0)),
                top_k=int(params.get("top_k", 0)),
            )
            lp_kwargs = {}
            if "repetition_penalty" in params:
                lp_kwargs["repetition_penalty"] = float(params["repetition_penalty"])
            if "repetition_context_size" in params:
                lp_kwargs["repetition_context_size"] = int(
                    params["repetition_context_size"]
                )
            logits_processors = make_logits_processors(**lp_kwargs) if lp_kwargs else None
            # no_repeat_ngram_size is not an mlx-lm penalty kwarg — append our
            # own processor to the (possibly empty) list. SKIPPED for code
            # requests (§9.1): the ban is tuned for prose clause-loops; in code
            # it drops ultra-frequent n-grams (", ", "= None", " is ", "var.")
            # → missing comma / "=" / identifier = syntax errors. rep_penalty
            # (soft) + cjk_lock stay on, so the prose guardrails aren't lost.
            nrn = params.get("no_repeat_ngram_size")
            if nrn and int(nrn) >= 2:
                if is_code:
                    log(f"sampling[{key}]: code request → no_repeat_ngram (n={nrn}) disabled")
                else:
                    if logits_processors is None:
                        logits_processors = []
                    logits_processors.append(make_no_repeat_ngram_processor(int(nrn)))
            # CJK language-lock (M3 French-serving). Flag-gated, needs the
            # tokenizer to scan the vocab; appended last so it overrides any
            # other processor's score on a CJK id.
            if params.get("cjk_lock") and tokenizer is not None:
                if logits_processors is None:
                    logits_processors = []
                logits_processors.append(make_cjk_lock_processor(tokenizer))
            # EOS guard (report §9.4): decouple the stop decision from the
            # repetition penalty. snapshot goes to the FRONT of the chain (must
            # see the raw, pre-penalty logits); clamp is appended at the very END
            # (must see the fully penalized logits) — the order is load-bearing.
            if params.get("eos_guard") and tokenizer is not None:
                _stops = {s[0] for s in _resolve_eos_token_seqs(tokenizer) if s}
                if _stops:
                    _snap, _clamp = make_eos_guard_processors(_stops)
                    if logits_processors is None:
                        logits_processors = []
                    logits_processors.insert(0, _snap)
                    logits_processors.append(_clamp)
            return sampler, logits_processors
    return None, None



def _is_cancelled(req_id: Optional[str]) -> bool:
    """Cheap thread-safe check. Inlined into the gen loops between tokens."""
    if not req_id:
        return False
    with _cancelled_lock:
        return req_id in _cancelled_ids


def _mark_cancelled(req_id: Optional[str]) -> None:
    if not req_id:
        return
    with _cancelled_lock:
        _cancelled_ids.add(req_id)


def _clear_cancelled(req_id: Optional[str]) -> None:
    """Called when a request finishes (done, cancelled, or errored) so the
    set doesn't grow unbounded over the runner's lifetime."""
    if not req_id:
        return
    with _cancelled_lock:
        _cancelled_ids.discard(req_id)
_SESSION_TTL_S = 3600.0
_SESSION_MIN_PREFIX = 32  # don't bother caching shorter prefixes

# Byte budget for the session store. Big-model KV caches at 128k context can
# weigh 20+ GB each; bounding by count alone (the old behaviour) could OOM
# the runner. We size the budget from `RUNNER_CACHE_BUDGET_BYTES` env, falling
# back to a conservative default. Eviction runs when total bytes > budget.
_SESSION_BUDGET_BYTES = int(os.environ.get("RUNNER_CACHE_BUDGET_BYTES",
                                           str(40 * 1024 * 1024 * 1024)))  # 40 GB
_SESSION_MAX = int(os.environ.get("RUNNER_CACHE_MAX_SESSIONS", "64"))

# Prewarm cache: a single prefilled cache for a configured system prefix
# (system prompt + tools + wiki context, anything that's constant across
# conversations). Cloned into each new session on cold start, so the first
# turn of every conversation skips the prefill of the shared prefix. This is
# the biggest TTFT win in chat workloads where Companion sends the same
# wiki+system context to every conversation.
_prewarm: Optional[dict] = None  # {"cache": ..., "tokens": [...], "model_id": ...}
_prewarm_lock = threading.Lock()


def _cache_size_bytes(cache) -> int:
    """Estimate the wired-memory cost of a prompt cache by summing each
    layer's (offset × n_kv_heads × head_dim × bytes_per_elem). Handles plain
    KVCache (fp16 = 2 bytes/elem, fp16·2 for both K and V) and QuantizedKVCache
    (group-quantized to 8 bits, so ~1 byte/elem + small scales overhead).

    Rough — we don't introspect every cache backend, just the common ones.
    Conservative: when in doubt we round up so eviction errs on freeing more.
    """
    if cache is None:
        return 0
    total = 0
    for c in cache:
        try:
            offset = int(getattr(c, "offset", 0))
            if offset <= 0:
                continue
            # KVCache stores K and V each as (B=1, n_kv_heads, max_len, head_dim)
            keys = getattr(c, "keys", None)
            if keys is not None:
                # keys.shape is mlx.core shape — convert via tolist
                shape = list(keys.shape) if hasattr(keys, "shape") else None
                if shape and len(shape) == 4:
                    _, n_kv, _alloc, head_dim = shape
                    # bytes per element: 2 for fp16, ~1 for q8 (group_size=64,
                    # bits=8). We can't reliably tell without import, so use
                    # the class name as a hint.
                    bpe = 1 if "Quantized" in type(c).__name__ else 2
                    # Both K and V → ×2
                    total += offset * n_kv * head_dim * bpe * 2
                    continue
            # Fallback: assume 8 KB per token per layer (typical for 7B-ish
            # models). Big models will under-count here, but we still get a
            # rough signal.
            total += offset * 8192
        except Exception:
            continue
    return total


def _store_total_bytes_locked() -> int:
    """Sum the byte cost across the whole session store. Caller holds the lock."""
    return sum(int(e.get("bytes", 0)) for e in _session_store.values())


def _evict_sessions() -> None:
    """TTL + byte-budget + count-cap eviction.

    Order:
      1. Drop anything past TTL.
      2. If total bytes > budget, evict LRU until under.
      3. If count > max sessions, evict LRU until at cap.

    Caller may or may not hold _session_lock — we re-acquire.
    """
    now = time.time()
    with _session_lock:
        # 1. TTL
        for sid in list(_session_store.keys()):
            if now - _session_store[sid].get("last_used", 0) > _SESSION_TTL_S:
                del _session_store[sid]
        # 2. Byte budget — LRU eviction until total < budget
        total = _store_total_bytes_locked()
        if total > _SESSION_BUDGET_BYTES:
            ordered = sorted(_session_store.items(),
                             key=lambda kv: kv[1].get("last_used", 0))
            for sid, entry in ordered:
                if total <= _SESSION_BUDGET_BYTES:
                    break
                total -= int(entry.get("bytes", 0))
                del _session_store[sid]
                sys.stderr.write(
                    f"[cache] evicted session {sid[:8]} "
                    f"({int(entry.get('bytes', 0)) / 1024**3:.1f} GB) — byte budget\n"
                )
        # 3. Hard count cap (defensive against tiny entries that don't trigger budget)
        if len(_session_store) > _SESSION_MAX:
            ordered = sorted(_session_store.items(), key=lambda kv: kv[1].get("last_used", 0))
            for sid, _ in ordered[: len(_session_store) - _SESSION_MAX]:
                del _session_store[sid]


def _truncatable_cache(cache) -> bool:
    """Check whether every layer in `cache` is a plain `KVCache` /
    `QuantizedKVCache`. Hybrid models (Mamba SSM, sliding-window) carry
    state-machine caches that can't be safely rewound, so we skip prefix
    cache reuse for them."""
    if not _CACHE_AVAILABLE:
        return False
    try:
        from mlx_lm.models.cache import KVCache as _KV  # type: ignore
    except Exception:
        return False
    safe = (_KV, QuantizedKVCache) if QuantizedKVCache is not None else (_KV,)
    return all(isinstance(c, safe) for c in cache)


def _snap_cache(cache) -> list:
    """O(1) snapshot de l'etat d'un prompt cache — refs `vars()` par couche.

    Les arrays MLX sont immutables : toute mutation ulterieure (generation)
    REBIND les attributs des objets cache sur de nouveaux arrays, les refs
    capturees ici restent l'etat exact du moment. C'est le pattern prouve du
    rollback spec-decode (dv_c2, saga DSpark) applique au serving : il donne
    un prefix-cache aux modeles dont le cache n'est PAS trimmable (V4
    PoolingCache/CacheList) — au tour suivant on RESTAURE au lieu de
    rembobiner."""
    out = []
    for c in cache:
        if _CacheListCls is not None and isinstance(c, _CacheListCls):
            out.append(("l", [dict(vars(x)) for x in c.caches]))
        else:
            out.append(("o", dict(vars(c))))
    return out


def _restore_from_snap(template_cache, snap) -> list:
    """Reconstruit une liste de caches NEUFS depuis un snapshot.

    `template_cache` (le cache vivant stocke avec la session) fournit les
    CLASSES par couche ; le snapshot fournit l'etat. Les objets retournes
    sont frais : la generation les mute sans toucher ni au snapshot ni au
    cache stocke — le snapshot est donc reutilisable a chaque tour."""
    fresh = []
    for c, (kind, st) in zip(template_cache, snap):
        if kind == "l":
            subs = []
            for sub, d in zip(c.caches, st):
                obj = sub.__class__.__new__(sub.__class__)
                obj.__dict__.update(dict(d))
                subs.append(obj)
            cl = c.__class__.__new__(c.__class__)
            cl.__dict__.update(dict(vars(c)))
            cl.caches = subs
            fresh.append(cl)
        else:
            obj = c.__class__.__new__(c.__class__)
            obj.__dict__.update(dict(st))
            fresh.append(obj)
    return fresh


def _cache_state_arrays(cache) -> list:
    """Aplatit tout mx.array vivant d'un prompt cache, couche par couche et
    sous-cache de CacheList compris.

    But : forcer l'eval de l'ETAT COMPLET du cache apres un prefill, pas
    seulement les logits. Un array de cache laisse LAZY porte le graphe des
    collectives distribues (recv/send/all_gather) du forward courant ; si on le
    snapshotte puis qu'on le reutilise au tour suivant, son eval re-declenche
    ces collectives HORS lockstep -> deadlock. En particulier l'etat POOLE de
    PoolingCache (V4) n'est PAS ancetre des logits courants, donc mx.eval(logits)
    seul le laisse lazy — d'ou le hang du tour 2. On materialise tout ici."""
    import mlx.core as _mx  # local — module-level import may be aliased
    out: list = []

    def _pull(c) -> None:
        got = False
        st = getattr(c, "state", None)   # contrat mlx-lm : arrays (possiblement niches)
        if st is not None:
            stack = [st]
            while stack:
                x = stack.pop()
                if isinstance(x, _mx.array):
                    out.append(x); got = True
                elif isinstance(x, (list, tuple)):
                    stack.extend(x)
                elif isinstance(x, dict):
                    stack.extend(x.values())
        if not got:                      # repli : scan brut du __dict__
            try:
                for v in vars(c).values():
                    if isinstance(v, _mx.array):
                        out.append(v)
            except TypeError:
                pass

    for c in cache:
        if _CacheListCls is not None and isinstance(c, _CacheListCls):
            for sub in c.caches:
                _pull(sub)
        else:
            _pull(c)
    return out


def _manual_prefill(model, cache, tokens: list[int], chunk: int = 2048) -> None:
    """Prefill explicite de `tokens` dans `cache`, par chunks evalues.

    Reproduit ce que stream_generate fait en interne, mais SOUS NOTRE
    CONTROLE pour pouvoir snapshotter l'etat exactement en fin de prompt.
    L'eval par chunk borne la memoire ET sert de lockstep distribue (tous
    les rangs executent ce meme code sur les memes tokens).

    Eval-isolation (corrige le hang du tour-2, cf _cache_state_arrays) : on
    materialise logits ET l'etat complet du cache a chaque chunk — sinon l'etat
    poole reste lazy et son graphe de collectives est rejoue hors lockstep au
    tour suivant."""
    for i in range(0, len(tokens), chunk):
        seg = mx.array(tokens[i:i + chunk], dtype=mx.uint32)[None]
        logits = model(seg, cache=cache)
        mx.eval(logits, *_cache_state_arrays(cache))


def _truncate_cache_to(cache, target_offset: int) -> None:
    """Set every layer's offset to `target_offset`. New tokens fed via
    `stream_generate` will overwrite the K/V slots beyond that position."""
    for c in cache:
        try:
            c.offset = target_offset
        except Exception:
            # Defensive — caller should have called _truncatable_cache first.
            pass


def _clone_cache_truncated(cache, target_offset: int):
    """Deep-clone a prompt cache, truncated to `target_offset` tokens.

    Used to fork the prewarm cache into a fresh per-session cache: the new
    session must get its OWN K/V buffer (not a view into the donor) so its
    writes at offset+T don't corrupt the donor — which is reused for every
    new conversation.

    Implementation: rely on mlx_lm's own `update_and_fetch(k, v)`, which
    allocates `mx.zeros(shape, dtype)` for a fresh buffer and writes the
    incoming K/V via slice assignment. That guarantees independence from
    the donor's buffer (verified against KVCache source 2026-05-18: each
    update_and_fetch allocates with mx.zeros + slice-write, no aliasing).

    Returns None if any layer is non-truncatable (SSM / sliding-window /
    rotating cache) — caller falls back to a fresh empty cache.
    """
    if not _truncatable_cache(cache) or target_offset <= 0:
        return None
    try:
        import mlx.core as mx  # type: ignore
    except Exception:
        return None
    # Defensive: only clone plain KVCache. QuantizedKVCache's no-arg ctor
    # silently produces a shape-inconsistent instance, which then makes
    # BatchGenerator crash with "list index out of range" at bg.next().
    # When any layer is quantized, bail and let the caller build a fresh
    # cache. Field note (2026-05-18 night): observed in prod with kv_q8_default=true.
    for c in cache:
        if type(c).__name__ != "KVCache":
            return None
    cloned: list = []
    try:
        for c in cache:
            new_c = type(c)()  # KVCache() — args-free init is safe for this type
            src_keys = getattr(c, "keys", None)
            src_vals = getattr(c, "values", None)
            if src_keys is None or src_vals is None:
                # Layer has no tokens yet — leave new cache empty.
                cloned.append(new_c)
                continue
            # Force eager materialization of the source slice so we don't
            # hold a lazy ref into the donor's graph.
            k_slice = src_keys[..., :target_offset, :]
            v_slice = src_vals[..., :target_offset, :]
            mx.eval(k_slice, v_slice)
            # Drive the cache's own buffer allocation + slice-write path.
            # After this returns, new_c.keys / new_c.values are independent
            # buffers (mx.zeros-allocated inside update_and_fetch).
            new_c.update_and_fetch(k_slice, v_slice)
            cloned.append(new_c)
    except Exception as e:
        sys.stderr.write(f"[cache] clone failed: {e}\n")
        return None
    return cloned


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def set_prewarm_prefix(model, tokenizer, repo: str, prefix_text: str,
                       kv_q8: bool = False) -> Optional[dict]:
    """Prefill a shared cache for a constant prefix (system prompt + wiki).

    Returns a dict {"tokens": [...], "bytes": int, "model_id": str} on success
    so the API layer / caller can surface the result. On cold-start of any new
    session whose prompt begins with `prefix_text`'s tokens, the lookup path
    deep-clones this cache instead of paying the full prefill cost — that's
    the cross-session TTFT win.

    Safe to call multiple times (overrides). Empty text clears.
    """
    global _prewarm
    if not prefix_text or not prefix_text.strip():
        with _prewarm_lock:
            _prewarm = None
        sys.stderr.write("[cache] prewarm cleared (empty text)\n")
        return None
    if not _CACHE_AVAILABLE or make_prompt_cache is None:
        sys.stderr.write("[cache] prewarm skipped — mlx_lm cache not available\n")
        return None
    try:
        tokens = list(tokenizer.encode(prefix_text, add_special_tokens=False))
        if len(tokens) < _SESSION_MIN_PREFIX:
            sys.stderr.write(
                f"[cache] prewarm skipped — prefix too short "
                f"({len(tokens)} < {_SESSION_MIN_PREFIX})\n"
            )
            return None
        cache = _build_prompt_cache(model, kv_q8)
        if cache is None:
            return None
        # Run a one-shot forward to populate the cache. We use stream_generate
        # with max_tokens=1 and immediately discard the generated token —
        # the side-effect we want is the populated K/V state.
        sys.stderr.write(
            f"[cache] prewarming prefix ({len(tokens)} tokens, kv_q8={kv_q8})…\n"
        )
        t0 = time.time()
        gen = stream_generate(model, tokenizer, prefix_text,
                              max_tokens=1, prompt_cache=cache)
        for _ in gen:
            break  # discard the generated token, cache is now populated
        elapsed = time.time() - t0
        size = _cache_size_bytes(cache)
        with _prewarm_lock:
            _prewarm = {
                "cache": cache,
                "tokens": tokens,
                "model_id": repo,
                "bytes": size,
                "set_at": time.time(),
            }
        sys.stderr.write(
            f"[cache] prewarm ready: {len(tokens)} tokens, "
            f"{size / 1024**3:.2f} GB, prefilled in {elapsed:.1f}s\n"
        )
        return {"tokens": len(tokens), "bytes": size, "elapsed_s": elapsed,
                "model_id": repo}
    except Exception as e:
        sys.stderr.write(f"[cache] prewarm failed: {e}\n")
        with _prewarm_lock:
            _prewarm = None
        return None


def _try_prewarm_clone(model_id: str, prompt_tokens: list[int]
                       ) -> tuple[Optional[object], Optional[list[int]], str, int]:
    """If a prewarm cache exists and matches this model + is a prefix of the
    prompt, return a fresh clone for the caller to use as the session cache.

    Returns (cache_clone, suffix_tokens, hit_kind, prefix_len).
    On miss / no prewarm: (None, None, "no-prewarm", 0).
    """
    with _prewarm_lock:
        pw = _prewarm
    if pw is None or pw.get("model_id") != model_id:
        return None, None, "no-prewarm", 0
    pw_tokens: list[int] = pw["tokens"]
    common = _common_prefix_len(pw_tokens, prompt_tokens)
    # We require the FULL prewarm prefix to be present in the new prompt —
    # partial prefixes wouldn't save anything because we'd still have to
    # rewind the clone, defeating the point. The min-prefix threshold also
    # avoids cloning for trivial overlaps.
    if common < len(pw_tokens) or common < _SESSION_MIN_PREFIX:
        return None, None, "prewarm-divergent", common
    clone = _clone_cache_truncated(pw["cache"], common)
    if clone is None:
        return None, None, "prewarm-unclonable", common
    return clone, prompt_tokens[common:], "prewarm-clone", common


# Minimum prefix length for cross-session radix matches. Higher than the
# in-session threshold because cloning a cache has a cost (copy + GPU
# materialization) — we want the prefill savings to outweigh that. 256 tokens
# is the typical break-even on M3 Ultra for a 64-layer model.
_RADIX_MIN_PREFIX = int(os.environ.get("RUNNER_RADIX_MIN_PREFIX", "256"))

# Cross-session radix clone is OPT-IN as of 2026-05-18 night. Observed
# in production:
#   [host-a] bg.next() failed: list index out of range; clearing all slots
#   req …: 0 toks in 0.1s · session=…(session-RADIX-CLONE) · finish=error
# The clone produces a cache that BatchGenerator can't iterate — most likely
# because `type(c)()` for QuantizedKVCache doesn't pass the (group_size, bits)
# constructor args, leaving the clone shape-inconsistent. Quantized cache is
# the default since v1.5.0 so this hit nearly every cold session.
#
# Per-session continuation (same session_id, in-place offset truncation) still
# works — that path doesn't clone. Only the cross-session sharing is gated.
_RADIX_CLONE_ENABLED = os.environ.get("RUNNER_RADIX_CLONE_ENABLED", "0") == "1"


def _find_best_radix_match(model_id: str, prompt_tokens: list[int],
                           exclude_sid: Optional[str] = None
                           ) -> tuple[Optional[object], Optional[list[int]], str, int]:
    """Scan every cached session (and the prewarm cache) to find the entry
    whose stored tokens form the LONGEST common prefix with the new prompt.
    Then deep-clone that cache truncated to the common prefix and return it.

    Gated by `_RADIX_CLONE_ENABLED` — see the flag's comment. Default OFF
    since 2026-05-18 night to avoid BatchGenerator crashes on cloned
    QuantizedKVCache layers.

    This is the radix-tree behaviour without an explicit tree: with ≤ 64
    sessions cached (the byte/count budget caps us there) and prefix
    comparison being O(min(len_a, len_b)), the linear scan is well under
    10 ms even with very long contexts. A real radix tree would be needed
    if we let the cache grow to thousands of entries — not the regime we're
    in today on a personal cluster.

    Why this matters: today, a brand-new Companion conversation only
    benefits from the cache if its prompt starts with the *configured*
    prewarm prefix. With radix lookup, if conversation B happens to share
    a long prefix with cached conversation A — e.g. both quote the same
    document, both got the same wiki context injected — B gets A's KV
    state for free, no configuration required.

    Returns (cache_clone, suffix_tokens, hit_kind, common_prefix_len).
    On miss: (None, None, "radix-no-match", 0).
    """
    if not _RADIX_CLONE_ENABLED:
        # Feature gated off — short-circuit so the caller falls back to a
        # fresh empty cache. See _RADIX_CLONE_ENABLED comment.
        return None, None, "radix-disabled", 0
    best_common = 0
    best_entry = None
    best_source = ""
    # Compare against every live session entry.
    with _session_lock:
        candidates = list(_session_store.items())
    for sid, entry in candidates:
        if sid == exclude_sid:
            continue
        if entry.get("model_id") != model_id:
            continue
        stored: list[int] = entry.get("tokens") or []
        if len(stored) < _RADIX_MIN_PREFIX:
            continue
        common = _common_prefix_len(stored, prompt_tokens)
        if common > best_common:
            best_common = common
            best_entry = entry
            best_source = f"session:{sid[:8]}"
    # Prewarm is also a candidate (and a strong one if configured).
    with _prewarm_lock:
        pw = _prewarm
    if pw is not None and pw.get("model_id") == model_id:
        pw_tokens: list[int] = pw["tokens"]
        common = _common_prefix_len(pw_tokens, prompt_tokens)
        if common > best_common:
            best_common = common
            best_entry = pw
            best_source = "prewarm"

    if best_entry is None or best_common < _RADIX_MIN_PREFIX:
        return None, None, "radix-no-match", best_common
    # Generating from a cache that's longer than what the new prompt agrees
    # with would require rewinding the original — we'd rather not mutate a
    # source that other sessions might match against. So we truncate the
    # CLONE to common, leaving the donor untouched.
    clone = _clone_cache_truncated(best_entry["cache"], best_common)
    if clone is None:
        return None, None, "radix-unclonable", best_common
    sys.stderr.write(
        f"[cache] radix hit: {best_common} tokens from {best_source} "
        f"(prompt={len(prompt_tokens)}, saved prefill ≈ {best_common} toks)\n"
    )
    return clone, prompt_tokens[best_common:], "radix-clone", best_common


def _session_lookup(session_id: str, model_id: str, prompt_tokens: list[int]):
    """Return (cache, suffix_tokens, hit_kind).

    hit_kind ∈ {"hit", "hit-truncated", "miss", "divergent", "non-truncatable",
                "fresh", "prewarm-clone", "cold"}.

    Two paths to a populated cache:
      (a) Per-session continuity: stored cache from a previous turn of THIS
          session_id is mutated in place (offset truncated to common prefix).
      (b) Cross-session prewarm: no session entry exists yet (cold), but a
          global prewarm cache matches as a prefix → deep-clone it into a
          fresh cache for this session. Saves a full system-prefix prefill
          on the first turn of every new conversation.

    Otherwise (None, None, <reason>) and caller builds a fresh empty cache.
    """
    if not session_id:
        # Stateless calls can still benefit from a radix scan — any prior
        # session that happens to share a long prefix wins.
        clone, suffix, kind, _ = _find_best_radix_match(model_id, prompt_tokens)
        if clone is not None:
            return clone, suffix, kind
        return None, None, "no-session"
    with _session_lock:
        entry = _session_store.get(session_id)
    if entry is None:
        # First time we see this session_id this runner-life. Normal on
        # turn-1 of a conversation, or after a runner restart. Radix scan
        # picks the best donor — either the configured prewarm prefix, or
        # another cached session that happens to share a long prefix with
        # this prompt (e.g. same document referenced, same wiki snippet
        # injected).
        clone, suffix, kind, _ = _find_best_radix_match(
            model_id, prompt_tokens, exclude_sid=session_id
        )
        if clone is not None:
            return clone, suffix, kind
        return None, None, "cold"
    with _session_lock:
        if entry.get("model_id") != model_id:
            # Same session, but a different model was loaded under it. Cache
            # is unusable — drop it. Shouldn't happen unless the cluster was
            # reloaded with a new model while the client kept the same id.
            del _session_store[session_id]
            return None, None, "model-changed"
        stored_tokens: list[int] = entry["tokens"]
        common = _common_prefix_len(stored_tokens, prompt_tokens)
        if common < _SESSION_MIN_PREFIX:
            del _session_store[session_id]
            return None, None, "divergent"
        if common == len(prompt_tokens):
            # Re-submitting the exact same prompt with no extension — nothing to
            # generate over a hit. Treat as miss so we regenerate from scratch
            # (cheap insurance vs. weird cache state).
            del _session_store[session_id]
            return None, None, "fresh"
        cache = entry["cache"]
        if common < len(stored_tokens):
            # The stored tokens went further than what the new prompt agrees
            # with — must rewind. Only safe for plain/quantized KVCache.
            if not _truncatable_cache(cache):
                # Fallback V4-class : le cache vivant ne se rembobine pas,
                # mais si un snapshot fin-de-prompt existe et que ses tokens
                # sont un PREFIXE STRICT du nouveau prompt (garanti quand la
                # conversation etend l'historique : le prompt du tour N est
                # prefixe du tour N+1 par determinisme du template), on
                # restaure des objets frais depuis le snapshot — zero rewind.
                snap = entry.get("snap") if _SNAP_CACHE_ENABLED else None
                snap_tokens = entry.get("snap_tokens")
                if snap and snap_tokens:
                    sc = _common_prefix_len(snap_tokens, prompt_tokens)
                    if sc == len(snap_tokens) and sc < len(prompt_tokens):
                        entry["last_used"] = time.time()
                        restored = _restore_from_snap(cache, snap)
                        return restored, prompt_tokens[sc:], "snap-hit"
                del _session_store[session_id]
                return None, None, "non-truncatable"
            _truncate_cache_to(cache, common)
        entry["last_used"] = time.time()
        kind = "hit" if common == len(stored_tokens) else "hit-truncated"
        return cache, prompt_tokens[common:], kind


def _spec_session_get(session_id: str, model_id: str):
    """Hook A (spec+snap) : entree session brute pour le chemin spec.
    Chaque rang tient SA copie (SPMD) ; la coherence inter-rangs est garantie
    par la barriere all_sum du generateur, pas ici."""
    if not session_id:
        return None
    with _session_lock:
        entry = _session_store.get(session_id)
        if os.environ.get("SPEC_SESS_DEBUG"):
            _keys = list(_session_store.keys())
            _mid = entry.get("model_id") if entry else None
            sys.stderr.write(
                f"[spec-sess] GET r{os.environ.get('MLX_RANK','?')} "
                f"sid={str(session_id)[:8]} found={entry is not None} "
                f"store_keys={[k[:8] for k in _keys]} "
                f"stored_mid={str(_mid)[:40]} want_mid={str(model_id)[:40]}\n")
            sys.stderr.flush()
        if not entry or entry.get("model_id") != model_id:
            return None
        entry["last_used"] = time.time()
        return entry


def _spec_session_put(session_id: str, model_id: str, cache, tokens,
                      snap, snap_tokens, spec_ctx=None) -> None:
    """Hook A (spec+snap) : persiste le cache+snapshot du chemin spec dans le
    MEME store byte-budgete que le chemin plain. `spec_ctx` (rank0 only) =
    refs (k,v) du CtxCache drafter au point du prompt."""
    if not session_id or cache is None:
        return
    with _session_lock:
        _session_store[session_id] = {
            "cache": cache,
            "tokens": list(tokens),
            "model_id": model_id,
            "last_used": time.time(),
            # A : le cache spec est deja compte dans la RAM modele (ref vivante,
            # pas une copie) ; le snapshot ne garde que le KV au point du
            # prompt. On force bytes=0 pour l'evicteur byte-budget — sinon
            # _cache_size_bytes surevalue le shard bas (12 couches) et
            # auto-evince la seule entree des le store (bug servant 05/08).
            # Le count-cap (_SESSION_MAX) borne toujours le nombre d'entrees.
            "bytes": 0,
            "snap": snap,
            "snap_tokens": list(snap_tokens) if snap_tokens else None,
            "spec_ctx": spec_ctx,
        }
    _evict_sessions()


def _session_store_after_gen(session_id: str, model_id: str, cache,
                             all_tokens: list[int],
                             rank: int = 0, world: int = 1,
                             snap: Optional[list] = None,
                             snap_tokens: Optional[list[int]] = None) -> None:
    """Persist the populated cache + cumulative token list under this session.
    Also records the cache byte cost so the byte-budgeted evictor can do its
    job (otherwise it would never know how big each entry is).

    Then kicks off a best-effort disk persist on a background thread so the
    next runner life can restore the session. The thread is daemonized — it
    won't block runner shutdown if the pickle is mid-flight."""
    if not session_id or cache is None:
        return
    cache_bytes = _cache_size_bytes(cache)
    with _session_lock:
        _session_store[session_id] = entry = {
            "cache": cache,
            "tokens": list(all_tokens),
            "model_id": model_id,
            "last_used": time.time(),
            "bytes": cache_bytes,
            # Snapshot fin-de-prompt (modeles non-trimmables, V4) : refs O(1)
            # vers l'etat du cache AVANT generation. Le tour suivant restaure
            # ce prefixe garanti au lieu de rembobiner (impossible). Peut
            # garder en vie ~1 copie des buffers K/V au point du prompt en
            # plus du cache vivant — assume, c'est le prix du prefix-hit.
            "snap": snap,
            "snap_tokens": list(snap_tokens) if snap_tokens else None,
        }
    _evict_sessions()
    # Disk persist — opt-in via RUNNER_CACHE_DISK_ENABLED=1, single-rank only.
    # Multi-rank writes were happening without ever being restored (the
    # restore code refuses world!=1), so they wasted I/O + RAM during gen.
    # Even on single-rank, the user must opt in because of the memory-
    # pressure risk during pickling (audit 2026-05-18).
    if _DISK_CACHE_ENABLED and world == 1:
        snapshot = dict(entry)  # shallow — cache and tokens are by-ref but
                                # we only READ them in the persist function
        t = threading.Thread(
            target=_save_session_to_disk,
            args=(session_id, snapshot, rank, world),
            daemon=True,
            name=f"disk-persist-{session_id[:8]}",
        )
        t.start()


def _session_clear_all() -> None:
    """Drop every cached session AND the global prewarm cache — call on model
    change/unload. The prewarm is model-specific (its KV state was populated
    by the current weights) so it must go too."""
    global _prewarm
    with _session_lock:
        _session_store.clear()
    with _prewarm_lock:
        _prewarm = None


# ──────────────────────────────────────────────────────────────────────────────
# Disk persistence (#4)
#
# Survive runner restarts so a user's recent conversations don't pay full
# prefill on the next session. We pickle each session's KV cache to local
# disk on the mac that runs the runner (NOT to the shared /Volumes mount —
# different ranks hold different slices and shouldn't share files).
#
# Constraints honored:
#   - Single-rank only restore (world_size=1). Multi-rank requires
#     cross-rank coordination on which sessions survived; deferred.
#   - Plain KVCache only — QuantizedKVCache writes work but skip restore
#     because reconstructing the group-quant state needs more care.
#   - Files keyed by (sha16(model_id), rank, world, session_id) so loads
#     only reuse entries that match the current topology.
#   - Stale entries (>24h) auto-pruned at startup.
#
# Cost: pickle of mx.array goes through `np.array(copy=True)` per layer —
# a 20 GB cache takes ~10-20 s to write. Done outside the gen hot path
# (after `_session_store_after_gen` returns). Reads are similar but
# rare (only at model load).
# ──────────────────────────────────────────────────────────────────────────────
# Disk persistence is OPT-IN — must be enabled explicitly via env. Default
# OFF after the 2026-05-18 audit flagged it as a memory-pressure / I/O risk:
#   * np.array(mx_arr, copy=True) doubles the cache in RAM during the copy
#   * background daemon threads can run during unload/reload → competition
#     for Metal allocations at the worst moment
#   * multi-rank writes were happening but _load_sessions_from_disk refuses
#     world != 1 → pure I/O cost for zero benefit
#   * no byte cap on disk → 20+ GB pickles per session uncapped
# Default OFF on multi-rank pools — multi-rank writes were happening but
# _load_sessions_from_disk refuses world != 1, so the I/O cost was paying
# for zero benefit.
# We now require BOTH (a) the env flag set and (b) world_size==1. Multi-rank
# writes are silently skipped even if the flag is on.
_DISK_CACHE_ENABLED = os.environ.get("RUNNER_CACHE_DISK_ENABLED", "0") == "1"
_DISK_CACHE_DIR = Path(os.environ.get(
    "RUNNER_CACHE_DIR",
    os.path.expanduser("~/mlx-cluster/.cache"),
))
_DISK_CACHE_MAX_AGE_S = float(os.environ.get(
    "RUNNER_CACHE_DISK_MAX_AGE_S", str(24 * 3600.0)))
# Total bytes of pickled caches we'll keep on disk per model. Beyond this,
# the oldest files are pruned at startup. Defaults to 40 GB which mirrors
# the in-memory budget.
_DISK_CACHE_BUDGET_BYTES = int(os.environ.get(
    "RUNNER_CACHE_DISK_BUDGET_BYTES",
    str(40 * 1024 * 1024 * 1024)))


def _disk_cache_dir_for(model_id: str) -> Path:
    h = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:16]
    return _DISK_CACHE_DIR / h


def _disk_path_for(model_id: str, sid: str, rank: int, world: int) -> Path:
    # The sid is opaque to us — could contain unusual chars. Hash it to be safe.
    sid_safe = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:24]
    return _disk_cache_dir_for(model_id) / f"r{rank}_w{world}_{sid_safe}.pkl"


def _save_session_to_disk(sid: str, entry: dict, rank: int, world: int) -> None:
    """Pickle one session's cache. Best-effort — failures are logged, never raised."""
    try:
        import numpy as np  # local import; numpy may not be needed if disabled
    except Exception:
        return
    cache = entry.get("cache")
    if cache is None:
        return
    # Skip if any layer isn't a plain KVCache — quantized writes need extra
    # state (scales/biases); SSM/sliding-window can't be serialized this way.
    for c in cache:
        if type(c).__name__ != "KVCache":
            return
    try:
        layers = []
        for c in cache:
            keys = getattr(c, "keys", None)
            values = getattr(c, "values", None)
            offset = int(getattr(c, "offset", 0))
            if keys is None or values is None or offset <= 0:
                # Empty layer — store a marker so restore knows the shape later.
                layers.append({"empty": True})
                continue
            # Force eager evaluation before the np.array copy.
            mx.eval(keys, values)
            k_np = np.array(keys[..., :offset, :], copy=True)
            v_np = np.array(values[..., :offset, :], copy=True)
            layers.append({"keys": k_np, "values": v_np, "kind": "KVCache",
                           "offset": offset})
        payload = {
            "tokens": list(entry.get("tokens") or []),
            "model_id": entry.get("model_id"),
            "world": world,
            "rank": rank,
            "saved_at": time.time(),
            "layers": layers,
            "sid": sid,
        }
        path = _disk_path_for(entry["model_id"], sid, rank, world)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)  # atomic rename
        # Don't spam stderr on every gen — keep a single line at debug level.
    except Exception as e:
        sys.stderr.write(f"[cache] disk persist failed for {sid[:8]}: {e}\n")


def _load_sessions_from_disk(model_id: str, rank: int, world: int) -> int:
    """Restore disk-persisted sessions matching this model + rank topology.
    Returns the number of sessions restored. Skips stale (> max age) files
    and drops them; corrupted files are quarantined as `.bad`."""
    # Multi-rank restore would need cross-rank coordination (all ranks must
    # agree which session_ids survived). Defer that — for now, only the
    # single-rank case restores cleanly.
    if world != 1:
        return 0
    try:
        import numpy as np  # noqa
        from mlx_lm.models.cache import KVCache as _KV  # type: ignore
    except Exception:
        return 0
    cache_dir = _disk_cache_dir_for(model_id)
    if not cache_dir.exists():
        return 0
    restored = 0
    now = time.time()
    for path in cache_dir.glob(f"r{rank}_w{world}_*.pkl"):
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            sys.stderr.write(f"[cache] disk load corrupt: {path.name} ({e})\n")
            try: path.rename(path.with_suffix(".bad"))
            except Exception: pass
            continue
        if payload.get("model_id") != model_id:
            continue
        if now - float(payload.get("saved_at", 0)) > _DISK_CACHE_MAX_AGE_S:
            try: path.unlink()
            except Exception: pass
            continue
        try:
            layers = []
            for ld in payload["layers"]:
                c = _KV()
                if ld.get("empty"):
                    layers.append(c)
                    continue
                k_mx = mx.array(ld["keys"])
                v_mx = mx.array(ld["values"])
                # update_and_fetch allocates fresh buffers + slice-writes,
                # exactly what we need to restore independent state.
                c.update_and_fetch(k_mx, v_mx)
                layers.append(c)
            sid = payload.get("sid") or path.stem
            with _session_lock:
                _session_store[sid] = {
                    "cache": layers,
                    "tokens": list(payload["tokens"]),
                    "model_id": model_id,
                    "last_used": now,
                    "bytes": _cache_size_bytes(layers),
                }
            restored += 1
        except Exception as e:
            sys.stderr.write(f"[cache] disk restore {path.name} failed: {e}\n")
    if restored:
        sys.stderr.write(
            f"[cache] restored {restored} session(s) from disk "
            f"(model={model_id.split('/')[-1]}, rank={rank}/{world})\n"
        )
    return restored


def _prune_stale_disk_cache() -> None:
    """Walk the disk cache root and delete files that exceed the TTL OR
    push the total size past the byte budget. Cheap (just stat + unlink) —
    run once at runner startup.

    Two-phase enforcement:
      1. TTL: drop anything older than `_DISK_CACHE_MAX_AGE_S` (24h default).
      2. Budget: if total remaining bytes > `_DISK_CACHE_BUDGET_BYTES`,
         drop oldest-first until under budget.

    The byte cap was missing in v1.5.1 — the audit flagged that 20+ GB
    pickles per session could accumulate uncapped.
    """
    if not _DISK_CACHE_DIR.exists():
        return
    cutoff = time.time() - _DISK_CACHE_MAX_AGE_S
    # Phase 1: TTL
    survivors: list[tuple[float, int, Path]] = []  # (mtime, size, path)
    pruned_ttl = 0
    for path in _DISK_CACHE_DIR.rglob("*.pkl"):
        try:
            st = path.stat()
            if st.st_mtime < cutoff:
                path.unlink()
                pruned_ttl += 1
                continue
            survivors.append((st.st_mtime, st.st_size, path))
        except Exception:
            pass
    # Phase 2: byte budget — evict oldest until under cap
    total = sum(s for _, s, _ in survivors)
    pruned_budget = 0
    if total > _DISK_CACHE_BUDGET_BYTES:
        survivors.sort()  # ascending mtime → oldest first
        for mtime, size, path in survivors:
            if total <= _DISK_CACHE_BUDGET_BYTES:
                break
            try:
                path.unlink()
                total -= size
                pruned_budget += 1
            except Exception:
                pass
    if pruned_ttl or pruned_budget:
        sys.stderr.write(
            f"[cache] disk prune: {pruned_ttl} stale, {pruned_budget} over-budget "
            f"({total / 1024**3:.1f} GB remaining, cap {_DISK_CACHE_BUDGET_BYTES / 1024**3:.0f} GB)\n"
        )

# Make local auto_parallel + stubs + patches importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Apply exo's mlx patches (yarn_rope correctness + batch_gen async_eval optim)
try:
    from patches import apply_mlx_patches
    apply_mlx_patches()
except Exception as e:
    sys.stderr.write(f"[runner] mlx patches not applied: {e}\n")


def emit(rank: int, obj: dict) -> None:
    if rank == 0:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def log(msg: str) -> None:
    sys.stderr.write(f"[{socket.gethostname()}] {msg}\n")
    sys.stderr.flush()


def _active_gb() -> float:
    """Best-effort MLX active (wired) memory in GB. 0.0 if the API is absent."""
    try:
        if hasattr(mx, "get_active_memory"):
            return mx.get_active_memory() / (1024 ** 3)
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            return mx.metal.get_active_memory() / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def free_metal(reason: str = "") -> None:
    """Explicitly release MLX/Metal buffer cache so wired memory is reclaimed.

    The cluster's recurring "73 GB / 190 GB wired stuck until reboot" bug
    comes from the runner exiting WITHOUT ever telling Metal to drop its
    buffer cache — we were relying on implicit C++ destructors that don't
    run on SIGKILL and aren't guaranteed even on a clean exit. Calling
    `mx.clear_cache()` on every teardown (and after each model unload in
    the persistent path) makes the reclaim explicit and reliable for the
    common case where the process is still responsive.

    NOTE: this cannot reclaim memory from a process that was already
    SIGKILL'd mid-kernel — that genuinely needs a reboot. The point is to
    stop GETTING into that state on normal load/unload cycles.
    """
    before = _active_gb()
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception as e:
        log(f"free_metal: clear_cache failed ({e})")
        return
    after = _active_gb()
    log(f"free_metal{f' ({reason})' if reason else ''}: "
        f"active {before:.1f} GB -> {after:.1f} GB")


# Belt-and-suspenders: even if main() returns via an unexpected path, drop
# the Metal cache on interpreter exit. Cheap no-op when already cleared.
import atexit as _atexit
_atexit.register(lambda: free_metal("atexit"))


def shard_tensor(model, group):
    """Apply exo-style auto_parallel tensor sharding."""
    from auto_parallel import tensor_auto_parallel

    gen = tensor_auto_parallel(model, group)
    last = None
    while True:
        try:
            resp = next(gen)
            last = resp
        except StopIteration as e:
            return e.value
    return last  # unreachable


def _compute_proportional_bounds(num_layers, weights):
    """Split `num_layers` across ranks proportionally to `weights`.

    Returns cumulative bounds (len = size+1, starts at 0, ends at num_layers).
    Each rank gets `floor(num_layers * w_i / total_w)` layers; the rounding
    leftover goes to the rank with the largest weight (the one that
    can afford it). Every rank receives at least 1 layer.

    Why proportional-to-weight : when nodes have heterogeneous RAM
    (.29 512 GB + .30 256 GB), the even split assigns equal shards and
    the small node OOMs on the first forward pass. Weighting by
    available RAM (or wired_limit) approximates per-shard memory
    pressure well enough that the small node breathes.

    Layer-size variance (MoE sparsity) is NOT modelled — proportional-
    by-RAM is rough but vastly better than even split on heterogeneous
    hardware. Tune via RUNNER_LAYER_BOUNDS when the rough split misses.
    """
    size = len(weights)
    if size <= 0 or num_layers <= 0:
        return list(range(num_layers + 1))
    total_w = sum(weights)
    if total_w <= 0:
        per = num_layers // size
        return [0] + [per * i for i in range(1, size)] + [num_layers]
    raw = [num_layers * w / total_w for w in weights]
    counts = [max(1, int(r)) for r in raw]
    diff = num_layers - sum(counts)
    if diff != 0:
        order = sorted(range(size), key=lambda i: weights[i], reverse=True)
        i = 0
        step = 1 if diff > 0 else -1
        guard = 0
        while diff != 0 and guard < size * (abs(diff) + size):
            idx = order[i % size]
            if counts[idx] + step >= 1:
                counts[idx] += step
                diff -= step
            i += 1
            guard += 1
    bounds = [0]
    acc = 0
    for c in counts:
        acc += c
        bounds.append(acc)
    if bounds[-1] != num_layers:
        bounds[-1] = num_layers
    return bounds


def shard_pipeline(model, group, num_layers):
    """Apply exo-style auto_parallel pipeline sharding.

    Layer split order of precedence :
      1. RUNNER_LAYER_BOUNDS (operator override, cumulative bounds CSV).
      2. RUNNER_RAM_WEIGHTS (orchestrator-supplied per-rank weights, CSV
         of bytes — typically wired_limit_bytes or ram_bytes). The
         runner normalises and computes proportional bounds. Required
         for heterogeneous clusters where even-split OOMs small nodes.
      3. Even split (default — same shard size on every rank).
    """
    from auto_parallel import pipeline_auto_parallel
    from exo_stubs import PipelineShardMetadata

    rank = group.rank()
    size = group.size()
    bounds_env = os.environ.get("RUNNER_LAYER_BOUNDS", "")
    weights_env = os.environ.get("RUNNER_RAM_WEIGHTS", "")
    bounds = None
    split_source = "even"
    if bounds_env:
        try:
            b = [int(x) for x in bounds_env.split(",")]
        except ValueError as e:
            raise ValueError(f"RUNNER_LAYER_BOUNDS={bounds_env!r} parse error: {e}")
        if len(b) != size + 1 or b[0] != 0 or b[-1] != num_layers:
            raise ValueError(
                f"RUNNER_LAYER_BOUNDS={bounds_env!r} invalid for size={size}, "
                f"num_layers={num_layers}; expected {size+1} ints starting at 0 ending at {num_layers}"
            )
        bounds = b
        split_source = "manual"
    elif weights_env:
        try:
            weights = [int(x) for x in weights_env.split(",")]
        except ValueError:
            weights = []
        if (
            len(weights) == size
            and all(w >= 0 for w in weights)
            and sum(weights) > 0
        ):
            bounds = _compute_proportional_bounds(num_layers, weights)
            split_source = f"proportional(weights={weights})"
        else:
            log(
                f"RUNNER_RAM_WEIGHTS={weights_env!r} invalid for size={size}; "
                f"falling back to even split"
            )
    if bounds is None:
        per = num_layers // size
        bounds = [0] + [per * i for i in range(1, size)] + [num_layers]
    start, end = bounds[rank], bounds[rank + 1]
    log(f"rank {rank} pipeline shard layers [{start}, {end}) split={split_source}")
    meta = PipelineShardMetadata(
        device_rank=rank,
        world_size=size,
        start_layer=start,
        end_layer=end,
    )
    gen = pipeline_auto_parallel(model, group, meta)
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value


_TOOL_CALL_HERMES_JSON = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL,
)
_TOOL_CALL_HERMES_LIST = re.compile(
    r"<tool_calls>\s*(\[.*?\])\s*</tool_calls>", re.DOTALL,
)
# Qwen3-Coder XML: <tool_call><function=NAME><parameter=KEY>VAL</parameter>…</function></tool_call>
_TOOL_CALL_QWEN_XML = re.compile(
    r"<tool_call>\s*(<function=.*?</function>)\s*</tool_call>", re.DOTALL,
)
_QWEN_FN_NAME = re.compile(r"<function=([^>]+)>", re.DOTALL)
_QWEN_PARAM = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)

# Hy3 XML: <tool_call>NAME<tool_sep><arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>
# Observed on inferencerlabs/Hy3-preview-MLX-9bit and family. The name lives
# BEFORE <tool_sep>, args are interleaved key/value pairs after.
# The `(?::\w+)?` tolerates the per-tag suffix some Hunyuan checkpoints emit —
# odyssai/Hy3-mlx-8bit-headbf16 wraps every tag as `<tool_call:opensource>`,
# `<tool_sep:opensource>`, `<arg_key:opensource>`, … . Without it the model's
# real tool call stayed unparsed in content → finish=stop, tool_calls=null →
# agent benches gated it as non-agent (2026-08-13).
_TOOL_CALL_HY3_XML = re.compile(
    r"<tool_call(?::\w+)?>\s*([^<\s][^<]*?)\s*<tool_sep(?::\w+)?>(.*?)</tool_call(?::\w+)?>",
    re.DOTALL,
)
_HY3_ARG_PAIR = re.compile(
    r"<arg_key(?::\w+)?>\s*(.*?)\s*</arg_key(?::\w+)?>\s*"
    r"<arg_value(?::\w+)?>\s*(.*?)\s*</arg_value(?::\w+)?>",
    re.DOTALL,
)

# LongCat format: <longcat_tool_call>NAME\n<longcat_arg_key>K</longcat_arg_key>
# <longcat_arg_value>V</longcat_arg_value>…</longcat_tool_call>
# Observed on meituan-longcat/LongCat-Flash-Lite. The name is the first line
# (before any newline or tag); args are interleaved key/value pairs. Mirrors
# the model's own parse_model_response.py: value = json.loads() when decodable
# (numbers/bools/objects), else the raw string.
_TOOL_CALL_LONGCAT = re.compile(
    r"<longcat_tool_call>\s*(.*?)\s*</longcat_tool_call>", re.DOTALL,
)
_LONGCAT_NAME = re.compile(r"([^\n<]+)")
_LONGCAT_ARG_PAIR = re.compile(
    r"<longcat_arg_key>\s*(.*?)\s*</longcat_arg_key>\s*"
    r"<longcat_arg_value>\s*(.*?)\s*</longcat_arg_value>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> tuple[list[dict], str]:
    """Extract OpenAI-shaped tool_calls from a generated string.

    Recognises:
      - Hermes JSON (Qwen3, GLM-4): `<tool_call>{"name":..,"arguments":..}</tool_call>`
      - Hermes JSON list: `<tool_calls>[ {…}, … ]</tool_calls>`
      - Qwen3-Coder XML: `<tool_call><function=NAME><parameter=KEY>VAL</parameter>…</function></tool_call>`
      - Hy3 XML: `<tool_call>NAME<tool_sep><arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`
      - LongCat: `<longcat_tool_call>NAME<longcat_arg_key>K</longcat_arg_key><longcat_arg_value>V</longcat_arg_value>…</longcat_tool_call>`

    Returns (tool_calls, content_without_calls). Each call:
      `{"id": "call_xxx", "type": "function", "function": {"name", "arguments"}}`.
    Arguments are JSON-stringified per the OpenAI spec.
    """
    calls: list[dict] = []
    cleaned = text

    def add_call(name: str, args) -> None:
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:12],
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args) if not isinstance(args, str) else args,
            },
        })

    # Pass 1: Hermes JSON single-call wrappers
    for m in _TOOL_CALL_HERMES_JSON.finditer(text):
        payload = m.group(1).strip()
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            add_call(obj["name"], obj.get("arguments", {}))
            cleaned = cleaned.replace(m.group(0), "")

    # Pass 2: Hermes JSON list wrappers
    for m in _TOOL_CALL_HERMES_LIST.finditer(text):
        payload = m.group(1).strip()
        try:
            arr = json.loads(payload)
        except Exception:
            continue
        if isinstance(arr, list):
            for item in arr:
                if isinstance(item, dict) and item.get("name"):
                    add_call(item["name"], item.get("arguments", {}))
            cleaned = cleaned.replace(m.group(0), "")

    # Pass 3: Qwen3-Coder XML format
    for m in _TOOL_CALL_QWEN_XML.finditer(text):
        body = m.group(1)
        name_m = _QWEN_FN_NAME.search(body)
        if not name_m:
            continue
        name = name_m.group(1).strip()
        args = {}
        for pm in _QWEN_PARAM.finditer(body):
            key = pm.group(1).strip()
            raw = pm.group(2).strip()
            # Try to JSON-decode each value (numbers/bools/objects); fall back to str.
            try:
                args[key] = json.loads(raw)
            except Exception:
                args[key] = raw
        add_call(name, args)
        cleaned = cleaned.replace(m.group(0), "")

    # Pass 4: Hy3 XML format (`<tool_call>NAME<tool_sep>…</tool_call>`)
    # The name comes before <tool_sep>; args are interleaved <arg_key>/<arg_value>.
    for m in _TOOL_CALL_HY3_XML.finditer(text):
        name = m.group(1).strip()
        body = m.group(2)
        args: dict = {}
        for pm in _HY3_ARG_PAIR.finditer(body):
            key = pm.group(1).strip()
            raw = pm.group(2).strip()
            try:
                args[key] = json.loads(raw)
            except Exception:
                args[key] = raw
        add_call(name, args)
        cleaned = cleaned.replace(m.group(0), "")

    # Pass 5: LongCat format (`<longcat_tool_call>NAME<longcat_arg_key>…</…>`)
    # Name is the first line; args are <longcat_arg_key>/<longcat_arg_value> pairs.
    for m in _TOOL_CALL_LONGCAT.finditer(text):
        body = m.group(1)
        name_m = _LONGCAT_NAME.match(body.strip())
        if not name_m:
            continue
        name = name_m.group(1).strip()
        args: dict = {}
        for pm in _LONGCAT_ARG_PAIR.finditer(body):
            key = pm.group(1).strip()
            raw = pm.group(2).strip()
            try:
                args[key] = json.loads(raw)
            except Exception:
                args[key] = raw
        add_call(name, args)
        cleaned = cleaned.replace(m.group(0), "")

    return calls, cleaned.strip()


def _normalize_messages_for_template(messages):
    """Normalize messages before `apply_chat_template`.

    The OpenAI wire format sends `tool_calls[*].function.arguments` as a JSON
    *string*. Some chat templates (notably Hy3-preview) iterate
    `arguments.items()` and crash with `'str object' has no attribute 'items'`.

    We deep-copy and convert `arguments` to a dict when it's a string. If the
    string isn't valid JSON (rare), we fall back to `{}` so templating still
    renders.
    """
    if not isinstance(messages, list):
        return messages
    out = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        tc = msg.get("tool_calls")
        if not tc:
            out.append(msg)
            continue
        new_msg = dict(msg)
        new_calls = []
        for call in tc:
            if not isinstance(call, dict):
                new_calls.append(call)
                continue
            new_call = dict(call)
            fn = new_call.get("function")
            if isinstance(fn, dict):
                new_fn = dict(fn)
                args = new_fn.get("arguments")
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args) if args.strip() else {}
                    except Exception:
                        parsed = {}
                    if not isinstance(parsed, dict):
                        parsed = {"_value": parsed}
                    new_fn["arguments"] = parsed
                new_call["function"] = new_fn
            new_calls.append(new_call)
        new_msg["tool_calls"] = new_calls
        out.append(new_msg)
    return out


def _resolve_eos_token_seqs(tokenizer) -> list[list[int]]:
    """Return the EOS / end-of-turn token id(s) as a list-of-singleton-lists.
    BatchGenerator's stop_tokens param expects sequences; we wrap singletons.

    Background: chat-tuned models often emit a "next-turn" marker rather than
    the tokenizer's nominal `eos_token_id` when the assistant turn ends.
    Examples:
      - Qwen / Hy3:       `<|im_end|>`
      - Llama 3:          `<|eot_id|>`
      - Gemma:            `<end_of_turn>`
      - GLM-4.6 / 5.1:    `<|user|>` or `<|observation|>` (NOT <|endoftext|>)
      - Older / classic:  `</s>`, `<|endoftext|>`

    mlx-lm's `stream_generate` only stops on the single `tokenizer.eos_token_id`,
    so distributed (legacy) runs need this expanded set to know when to break.
    We also pull from `tokenizer.generation_config.eos_token_id` (often a list,
    as HF supports multi-eos since 4.30+).
    """
    seqs: list[list[int]] = []
    seen_ids: set[tuple] = set()

    def _add(tid):
        if isinstance(tid, int) and tid >= 0 and (tid,) not in seen_ids:
            seqs.append([tid])
            seen_ids.add((tid,))

    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        _add(eos)
    elif isinstance(eos, (list, tuple)):
        for t in eos:
            _add(t)

    # mlx-lm's TokenizerWrapper carries the MODEL config's `eos_token_id` — often
    # a LIST — in `eos_token_ids` (see mlx_lm/utils.py: load_tokenizer is called
    # with eos_token_ids=config["eos_token_id"]). `tokenizer.eos_token_id` above
    # only ever exposes the single primary id from tokenizer_config.json, so the
    # extra turn-enders were dropped. Laguna: config declares [2, 24] = 〈|EOS|〉 +
    # `</assistant>`, but only 2 survived — the model emitted 24, nothing stopped
    # it, and `</assistant>` leaked into the text before generation looped.
    for t in (getattr(tokenizer, "eos_token_ids", None) or ()):
        _add(t)

    # HF generation_config can declare a list of eos ids that differs from the
    # tokenizer's primary eos. Read it if present.
    try:
        gc = getattr(tokenizer, "generation_config", None) or {}
        gc_eos = gc.get("eos_token_id") if isinstance(gc, dict) else getattr(gc, "eos_token_id", None)
        if isinstance(gc_eos, int):
            _add(gc_eos)
        elif isinstance(gc_eos, (list, tuple)):
            for t in gc_eos:
                _add(t)
    except Exception:
        pass

    # Common end-of-turn markers across chat-tuned models.
    extra_names = ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>",
                   "<end_of_turn>", "</s>",
                   # GLM-4.6 / 5.1 turn boundaries
                   "<|user|>", "<|observation|>",
                   # Bailing / Ring-2.x turn boundary (eos_token is <|endoftext|>
                   # but the chat template ends assistant turns with this)
                   "<|role_end|>")
    # `convert_tokens_to_ids` returns the UNK id (not None) for a name absent
    # from the vocab, so every model missing all of the above silently gained
    # <unk> as a stop token — on Laguna that was id 0 = 〈|UNK|〉. Skip it.
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for name in extra_names:
        try:
            tid = tokenizer.convert_tokens_to_ids(name)
            if unk_id is not None and tid == unk_id:
                continue
            _add(tid)
        except Exception:
            pass
    return seqs


def _build_prompt_cache(model, kv_q8: bool):
    """Construct a fresh prompt cache for a single gen request.

    Default path: `make_prompt_cache()` so the model's own `make_cache()` hook
    runs. That gives the right cache type per layer for hybrid models (qwen3_next
    SSM, gemma sliding-window, mamba, etc.).

    With `kv_q8=True`, we replace ONLY the plain `KVCache` slots with
    `QuantizedKVCache(group_size=64, bits=8)` — other slots (SSM, rotating,
    sliding-window) keep their original type, so we don't break attention masks.
    """
    if not _CACHE_AVAILABLE or make_prompt_cache is None:
        return None
    try:
        cache = make_prompt_cache(model)
        if not kv_q8 or QuantizedKVCache is None:
            return cache
        try:
            from mlx_lm.models.cache import KVCache  # type: ignore
        except Exception:
            KVCache = None
        if KVCache is None:
            return cache
        upgraded = []
        n_q8 = 0
        for c in cache:
            if type(c) is KVCache:
                upgraded.append(QuantizedKVCache(group_size=64, bits=8))
                n_q8 += 1
            else:
                # Specialized cache (SSM/rotating/sliding) — leave intact.
                upgraded.append(c)
        sys.stderr.write(f"[runner] kv_q8: {n_q8}/{len(cache)} layers Q8-quantized "
                         f"(rest kept native)\n")
        return upgraded
    except Exception as e:
        sys.stderr.write(f"[runner] cache build failed ({e}), letting stream_generate auto-create\n")
        return None


def main() -> None:
    repo = os.environ.get("RUNNER_MODEL", "mlx-community/GLM-4.5-Air-4bit")
    mode = os.environ.get("RUNNER_MODE", "pipeline")  # pipeline | tensor | tensor_ap
    use_ap = os.environ.get("RUNNER_USE_AP", "0") == "1"
    kv_q8_default = os.environ.get("RUNNER_KV_Q8", "0") == "1"
    # Speculative decoding: small draft model proposes N tokens per step, the
    # main model verifies them in one forward. Supported only in single-rank
    # legacy mode (mlx-lm's stream_generate accepts draft_model=...).
    draft_repo = os.environ.get("RUNNER_DRAFT_MODEL", "") or None
    num_draft_tokens = int(os.environ.get("RUNNER_NUM_DRAFT_TOKENS", "4"))
    world_size_env = int(os.environ.get("MLX_WORLD_SIZE", "0") or "0")

    stop_requested = {"flag": False}

    def handle_sig(signum, _frame):
        log(f"signal {signum} received, marking stop")
        stop_requested["flag"] = True

    signal.signal(signal.SIGTERM, handle_sig)
    signal.signal(signal.SIGINT, handle_sig)

    t0 = time.time()

    # Single-node mode (world_size=1): skip JACCL entirely. The model loads in
    # this single process — no distributed init, no IBV setup, no barrier.
    if world_size_env == 1:
        group = None
        rank = 0
        size = 1
        log(f"init single-node mode (no distributed), model={repo}, mode={mode}")
    else:
        # #40 WU4 — transport is engine-selected per pool: jaccl (RDMA, the
        # perf default) or ring (TCP via MLX_HOSTFILE — no QP-degradation bug,
        # the long-run stability option). The log lines keep the exact
        # "init <backend> backend" shape the engine's phase markers grep.
        backend = os.environ.get("RUNNER_BACKEND", "jaccl").strip().lower() or "jaccl"
        log(f"init {backend} backend, model={repo}, mode={mode}, use_ap={use_ap}")
        group = mx.distributed.init(backend=backend, strict=True)
        rank = group.rank()
        size = group.size()
        log(f"rank {rank}/{size} group ready in {time.time()-t0:.2f}s")

    t1 = time.time()

    if size == 1:
        # Single-node: load full model, no sharding.
        repo_path = Path(repo) if Path(repo).exists() else hf_repo_to_path(repo)
        log(f"single-node loading model from {repo_path}")
        model, model_config = load_model(repo_path, lazy=False, strict=False)
        mx.eval(model.parameters())
        # trust_remote_code=True: some model repos (mimo_v2, deepseek-v4, …) ship
        # custom tokenizer/model Python files referenced via `auto_map` in their
        # config; without this flag transformers prompts interactively and hangs.
        # Safe here because we own/curate the model dirs on the cluster filesystem.
        # eos_token_ids: mirror mlx_lm.utils.load — the MODEL config declares the
        # full stop set (often a list), tokenizer_config.json only the primary id.
        tokenizer = tokenizer_utils.load(repo_path, tokenizer_config_extra={"trust_remote_code": True},
                                         eos_token_ids=model_config.get("eos_token_id"))
    elif use_ap:
        # exo-style: load full, then auto-parallel shard
        # Resolve HF repo to local path if needed
        repo_path = Path(repo) if Path(repo).exists() else hf_repo_to_path(repo)
        log(f"rank {rank} loading model (lazy=True) from {repo_path}")
        model, model_config = load_model(repo_path, lazy=True, strict=False)
        if mode == "tensor":
            log(f"rank {rank} applying tensor_auto_parallel")
            model = shard_tensor(model, group)
        else:
            # Multimodal models (Qwen3.5-VL/Qwen3.6 etc.) nest num_hidden_layers
            # under text_config or language_model. Cascade the lookup.
            num_layers = (
                model_config.get("num_hidden_layers")
                or model_config.get("text_config", {}).get("num_hidden_layers")
                or model_config.get("language_model", {}).get("num_hidden_layers")
                or (model_config.get("text_config", {}).get("language_model", {}) or {}).get("num_hidden_layers")
            )
            if num_layers is None:
                raise RuntimeError(
                    f"could not find num_hidden_layers in model_config; "
                    f"keys present: {list(model_config.keys())}"
                )
            log(f"rank {rank} applying pipeline_auto_parallel ({num_layers} layers)")
            model = shard_pipeline(model, group, num_layers)
        mx.eval(model.parameters())
        log(f"rank {rank} loading tokenizer")
        # trust_remote_code=True: some model repos (mimo_v2, deepseek-v4, …) ship
        # custom tokenizer/model Python files referenced via `auto_map` in their
        # config; without this flag transformers prompts interactively and hangs.
        # Safe here because we own/curate the model dirs on the cluster filesystem.
        # eos_token_ids: mirror mlx_lm.utils.load — the MODEL config declares the
        # full stop set (often a list), tokenizer_config.json only the primary id.
        tokenizer = tokenizer_utils.load(repo_path, tokenizer_config_extra={"trust_remote_code": True},
                                         eos_token_ids=model_config.get("eos_token_id"))
    else:
        if mode == "tensor":
            model, tokenizer = sharded_load(repo, tensor_group=group)
        else:
            model, tokenizer = sharded_load(repo, pipeline_group=group)

    # Sync barrier: don't signal "ready" until ALL ranks have loaded tokenizer
    # otherwise rank 0 starts its gen timer before rank 1 is actually ready.
    # Skipped in single-node mode (no peers to sync with).
    if size > 1:
        log(f"rank {rank} barrier before ready")
        _b = mx.distributed.all_sum(mx.array([1.0]), group=group)
        mx.eval(_b)

    load_s = time.time() - t1
    log(f"rank {rank} model loaded in {load_s:.1f}s")

    # Load draft model AFTER target so it lands on the same device. Only on
    # rank 0 (size==1) since speculative decoding lives in the legacy path.
    # E0 harness exception (#66 plan): RUNNER_SPEC_MULTIRANK=1 loads the
    # draft REPLICATED on every rank — validation of the multi-rank
    # accept-alignment invariant only, never a prod default.
    spec_multirank = os.environ.get("RUNNER_SPEC_MULTIRANK", "0") == "1"
    # DSpark drafters (config carries `dspark_block_size`) use the dedicated
    # spec_dspark path — single-node trimmable-pool OR multi-rank lockstep —
    # NOT mlx-lm's generic stream_generate(draft_model=). Detect early so the
    # generic draft load below skips them (loading a DSpark drafter as a plain
    # causal model would fail / mis-load).
    _draft_is_dspark = False
    if draft_repo:
        try:
            _dp = Path(draft_repo) if Path(draft_repo).exists() else hf_repo_to_path(draft_repo)
            _draft_is_dspark = "dspark_block_size" in json.load(open(Path(_dp) / "config.json"))
        except Exception:
            _draft_is_dspark = False
    draft_model = None
    if draft_repo and not _draft_is_dspark and (size == 1 or spec_multirank):
        try:
            t_draft = time.time()
            draft_path = Path(draft_repo) if Path(draft_repo).exists() else hf_repo_to_path(draft_repo)
            log(f"loading DRAFT model from {draft_path}"
                + (f" (MULTIRANK harness, rank {rank})" if size > 1 else ""))
            draft_model, _ = load_model(draft_path, lazy=False, strict=False)
            mx.eval(draft_model.parameters())
            log(f"draft model loaded in {time.time()-t_draft:.1f}s — speculative decoding ENABLED "
                f"(num_draft_tokens={num_draft_tokens})")
        except Exception as e:
            log(f"DRAFT model load failed ({e}); speculative disabled")
            draft_model = None
    elif draft_repo:
        log(f"draft_model requested but size={size} (>1) — speculative decoding only "
            f"supported single-rank (set RUNNER_SPEC_MULTIRANK=1 for the harness); "
            f"ignoring draft")

    # Native MTP (plan docs/PLAN-distributed-mtp.md): the model's own MTP
    # head(s) as drafter, loaded NEXT TO the trunk on EVERY rank — drafting
    # is local and identical per rank (PP all_gather / TP all_sum give all
    # ranks the same final hidden+logits), so no new collectives.
    native_mtp = None
    if os.environ.get("RUNNER_MTP", "off").strip().lower() == "native":
        try:
            from mtp_module import load_native_mtp
            _mtp_dir = Path(repo) if Path(repo).exists() else hf_repo_to_path(repo)
            native_mtp = load_native_mtp(
                model, _mtp_dir,
                sidecar=os.environ.get("RUNNER_MTP_SIDECAR") or None,
                quantize=os.environ.get("RUNNER_MTP_QUANT", "0") == "1",
            )
        except Exception as e:
            log(f"native MTP load failed ({e}) — serving AR only")
            native_mtp = None

    # DSpark speculative distribue (goal drafter, plan P7). Un drafter DSpark
    # SEPARE (V4DSparkDrafter) charge sur rank0 ; le bloc propose est diffuse
    # aux servants (all_sum), verify target en lockstep (module spec_dspark,
    # portage du harnais dv_g PROUVE lossless). Detecte par la config du
    # drafter qui porte `dspark_block_size`. Contrairement a mlx-lm
    # stream_generate(draft_model=) (nodes=1 only), ce chemin est multi-rang.
    #   Detection : size>1 + un draft => TOUJOURS DSpark (le draft mlx-lm est
    #   nodes=1 only, cf plus haut). Les servants n'ont PAS le dossier drafter
    #   (seul rank0 l'a) — ils entrent en mode servant sur le seul fait
    #   size>1+draft, sans lire de config. rank0 charge le drafter ; son ECHEC
    #   est FATAL (rank0 plain + servants spec = deadlock), donc on laisse
    #   propager pour faire echouer le load proprement plutot que desync.
    spec_dspark_ctx = None
    if draft_repo and _draft_is_dspark:
        import spec_dspark
        if rank == 0:
            _dpath = Path(draft_repo) if Path(draft_repo).exists() else hf_repo_to_path(draft_repo)
            _dcfg = json.load(open(Path(_dpath) / "config.json"))
            _tpath = Path(repo) if Path(repo).exists() else hf_repo_to_path(repo)
            _targs = spec_dspark.load_target_args(str(_tpath))
            _drafter = spec_dspark.load_dspark_drafter(str(_dpath), _targs, _dcfg)
            spec_dspark_ctx = {"enabled": True, "drafter": _drafter,
                               "args": _targs, "dcfg": _dcfg,
                               "single_node": size == 1}
            log(f"DSpark drafter loaded (block={_dcfg.get('dspark_block_size')}, "
                f"taps={_dcfg.get('dspark_target_layer_ids')}) — "
                f"{'single-node trimmable-pool' if size == 1 else 'multi-rank'} spec ENABLED")
        else:
            # servants exist only in multi-rank; single-node is rank0-only.
            spec_dspark_ctx = {"enabled": True, "drafter": None,
                               "args": None, "dcfg": None, "single_node": False}
            log("DSpark spec pool — servant rank (target-only, no drafter dir)")

    emit(rank, {"event": "ready", "rank": rank, "size": size, "load_s": load_s,
                "speculative": draft_model is not None or spec_dspark_ctx is not None,
                "mtp": native_mtp is not None})

    # Disk cache: only touch it when explicitly opted in. The 2026-05-18
    # audit flagged that even the prune scan + restore path can compete
    # with active gen for filesystem IO + Metal pressure on big models.
    # Default OFF until we have proper benchmarks + per-cluster opt-in.
    if _DISK_CACHE_ENABLED:
        try:
            _prune_stale_disk_cache()
            _load_sessions_from_disk(repo, rank, size)
        except Exception as e:
            log(f"disk cache restore failed: {e}")

    # ── Loop dispatcher ──────────────────────────────────────────────────────
    # size == 1: BatchGenerator-driven main loop (concurrent serving). Multi-
    # rank stays on the legacy single-stream loop so JACCL collectives stay
    # aligned across ranks at every model() call. The two paths share session
    # cache helpers so prefix-cache reuse works in both.
    # Speculative decoding (draft_model) requires the legacy path — mlx-lm's
    # BatchGenerator doesn't currently accept a draft_model.
    # minimax_m3 forces the legacy single-slot path: its MSA mask + decode
    # block-gather (OdyssAI-X#53) assume a SCALAR cache offset (B=1). The
    # BatchGenerator cache exposes a non-scalar offset → `mx.arange(offset+S)`
    # TypeError in MiniMaxM3Model.__call__. B>1 batched M3 is phase-2 scope.
    # deepseek_v4 (+ dspark) force the legacy single-slot path: its PoolingCache
    # (compressed-attention pooled KV, variable-length per request) has no
    # batched merge — mlx_lm's BatchGenerator._merge_caches calls
    # PoolingCache.merge -> `BatchPoolingCache` which isn't defined, so a plain
    # (no-drafter) load crashes with NameError at bg.next(). Single-user is the
    # supported mode; with a DSpark drafter it already takes the legacy path.
    # Batched pooled cache (B>1 concurrent) is phase-2 scope, like minimax_m3.
    # inkling_mm_model shares the constraint: its LayerCache (KVCache + 4 short-
    # conv ConvCaches per layer) has no batched merge, so BatchGenerator's
    # _merge_caches raises "does not yet support batching with history". Single-
    # stream is the supported text-only v1 mode (batched conv-state cache = later).
    _mt = (model_config or {}).get("model_type") or ""
    # REVERT 2026-08-27 — RUNNER_BATCH default flipped "1" -> "0" (legacy).
    # The batched path is the common factor behind a week of production
    # regressions the legacy loop never exhibits: the BatchGenerator poison
    # (post-long-gen 0.2-0.35 tok/s crawls needing pool reloads), the MLX
    # buffer-cache balloon hitting the ~499GB Metal limit (~7495-token
    # truncations, finish=error), the malloc-failure memory ratchet, and
    # insert-during-gen slowdowns. Three guard iterations (empty-rebuild,
    # force-rebuild, periodic clear_cache) each closed one hole and the next
    # appeared — the component is not production-ready. Single-node pools go
    # back to the battle-tested legacy single-slot loop (the same loop the
    # multi-rank path uses); benches are sequential so nothing is lost.
    # Continuous batching returns via RUNNER_BATCH=1 (explicit opt-in) once
    # fixed for real on rpi-dev.
    use_batched = (size == 1) and _BATCH_AVAILABLE and (draft_model is None) and (
        native_mtp is None                     # native MTP needs the legacy path
    ) and (
        spec_dspark_ctx is None                # single-node DSpark spec -> legacy
    ) and (
        os.environ.get("RUNNER_BATCH", "0") == "1"
    ) and ("minimax-m3" not in repo.lower()) and (
        _mt not in ("deepseek_v4", "deepseek_v4_dspark", "inkling_mm_model")
    )
    if use_batched:
        log("entering batched main loop (BatchGenerator)")
        _run_batched_main(model, tokenizer, repo, kv_q8_default, stop_requested,
                          rank, world_size=size)
    else:
        log(f"entering legacy single-stream main loop "
            f"(size={size}, batch_available={_BATCH_AVAILABLE}, "
            f"speculative={draft_model is not None}, "
            f"mtp={native_mtp is not None})")
        _run_legacy_main(model, tokenizer, repo, kv_q8_default, stop_requested,
                         rank, draft_model=draft_model,
                         num_draft_tokens=num_draft_tokens, world_size=size,
                         group=group, native_mtp=native_mtp,
                         spec_dspark_ctx=spec_dspark_ctx)

    # Explicit teardown BEFORE the process exits. Two reasons:
    #   1. Drop every model reference so the weights are deallocated, then
    #      tell Metal to release its buffer cache (free_metal). Without this
    #      the wired pages lingered until reboot (the 73-190 GB bug).
    #   2. Returning from main() lets the JACCL group's C++ destructors run
    #      ibv_destroy_qp, releasing the RDMA queue pairs cleanly so the next
    #      init doesn't inherit a corrupted state.
    try:
        model = None
        tokenizer = None
        draft_model = None
        free_metal("shutdown")
    except Exception as e:
        log(f"shutdown cleanup error: {e}")
    emit(rank, {"event": "bye"})
    log("exiting cleanly (Metal cache cleared; destructors will run ibv_destroy_qp)")


# ──────────────────────────────────────────────────────────────────────────────
# Legacy single-stream main loop (multi-rank-safe; one gen at a time, plus
# optional speculative decoding when a draft_model is passed)
# ──────────────────────────────────────────────────────────────────────────────
def _run_legacy_main(model, tokenizer, repo: str, kv_q8_default: bool,
                     stop_requested: dict, rank: int,
                     draft_model=None, num_draft_tokens: int = 4,
                     world_size: int = 1, group=None,
                     native_mtp=None, spec_dspark_ctx=None) -> None:
    # Expanded stop set. `stream_generate` already breaks on
    # tokenizer.eos_token_id, but chat-tuned models often emit a "next-turn"
    # marker first (GLM emits <|user|>, Qwen emits <|im_end|>, etc.). Without
    # this check, the model can run away until max_tokens — observed in prod
    # on GLM-5.1 where assistant turns ended with <|user|> and the runner kept
    # generating 64k tokens of bilingual garbage.
    _stop_ids: set[int] = {seq[0] for seq in _resolve_eos_token_seqs(tokenizer)
                           if len(seq) == 1}
    if rank == 0:
        log(f"legacy stop_ids = {sorted(_stop_ids)} "
            f"(tokenizer.eos={getattr(tokenizer, 'eos_token_id', None)})")

    # ── Reader thread + queue ─────────────────────────────────────────────
    # Mirror the batched path so cancel commands can arrive WHILE a gen is
    # blocking the main thread inside stream_generate. The legacy loop used
    # to read stdin synchronously between gens — any cancel sent mid-stream
    # piled up unread, defeating the point of "hard cancel". Now the reader
    # parses every line, intercepts `cancel` immediately (toggles the flag
    # set the gen loop checks), and queues everything else for the main
    # thread to drain in order.
    in_q: queue.Queue = queue.Queue()
    EOF = object()

    def reader() -> None:
        while not stop_requested["flag"]:
            line = sys.stdin.readline()
            if not line:
                in_q.put(EOF)
                return
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                log(f"bad json: {e}")
                continue
            # Hot path: cancel bypasses the queue so it can land between
            # stream_generate yields without waiting for the main thread.
            if msg.get("cmd") == "cancel":
                _mark_cancelled(msg.get("id"))
                continue
            in_q.put(msg)

    threading.Thread(target=reader, daemon=True, name=f"runner-stdin-r{rank}").start()

    while not stop_requested["flag"]:
        try:
            req = in_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if req is EOF:
            log("stdin closed, exiting loop")
            break

        cmd = req.get("cmd")
        if cmd == "stop":
            log("stop cmd received")
            break
        if cmd == "session_clear":
            sid = req.get("session_id")
            if sid:
                with _session_lock:
                    _session_store.pop(sid, None)
                log(f"cleared session {sid}")
            else:
                _session_clear_all()
                log("cleared all sessions")
            continue
        if cmd == "prewarm":
            # Build/replace the global prefix cache. All ranks must run the
            # forward together (it's a distributed prefill), but only rank 0
            # emits a result event so the API has a single source of truth.
            text = req.get("text", "")
            kv_q8 = bool(req.get("kv_q8", kv_q8_default))
            result = set_prewarm_prefix(model, tokenizer, repo, text, kv_q8=kv_q8)
            if rank == 0:
                emit(rank, {"event": "prewarm", "id": req.get("id", ""),
                            "ok": result is not None, "result": result})
            continue
        if cmd == "keepalive":
            # #40 WU2 — all ranks exercise the JACCL group together (a tiny
            # all_sum) to keep the QPT_UC connections warm and let the
            # orchestrator detect a dead/hung peer early. Only rank 0 acks. A
            # dead peer makes this all_sum hang (UC has no timeout) — that hang
            # is exactly the signal the orchestrator's keepalive timeout catches.
            if group is not None and world_size > 1:
                try:
                    _ka = mx.distributed.all_sum(mx.array([1.0]), group=group)
                    mx.eval(_ka)
                except Exception as e:
                    log(f"keepalive all_sum error: {e}")
                    if rank == 0:
                        emit(rank, {"event": "keepalive_ok", "id": req.get("id", ""),
                                    "ok": False, "error": str(e)})
                    continue
            if rank == 0:
                emit(rank, {"event": "keepalive_ok", "id": req.get("id", ""), "ok": True})
            continue
        if cmd != "gen":
            log(f"unknown cmd: {cmd}")
            continue

        req_id = req.get("id", "")
        prompt = req.get("prompt")
        messages = req.get("messages")
        tools = req.get("tools")
        max_tokens = int(req.get("max_tokens", 200))
        enable_thinking = req.get("enable_thinking", None)
        kv_q8 = bool(req.get("kv_q8", kv_q8_default))
        session_id = req.get("session_id") or None

        # Backward-compat: a bare `prompt` becomes a single user message.
        if messages is None:
            if not prompt:
                log("gen with neither messages nor prompt — skipping")
                continue
            messages = [{"role": "user", "content": prompt}]

        # Hy3-preview's chat template iterates `arguments.items()`, so the
        # OpenAI-wire JSON-string form crashes Jinja. Normalize once here and
        # reuse the result for both string + token templating below.
        messages = _normalize_messages_for_template(messages)

        chat_kwargs = {"add_generation_prompt": True, "tokenize": False}
        if enable_thinking is not None:
            chat_kwargs["enable_thinking"] = enable_thinking
            # MiniMax-M3's template reads `thinking_mode` (enabled/disabled/
            # adaptive), NOT the enable_thinking bool — map it so no-thinking
            # actually suppresses the <mm:think> prefill. Others ignore the
            # extra kwarg (Jinja drops unused vars).
            if "minimax-m3" in repo.lower():
                chat_kwargs["thinking_mode"] = "enabled" if enable_thinking else "disabled"
        # reasoning_effort: OpenAI o-series dial the template injects as a
        # "Reasoning: <effort>" system directive (Step-3.7). Harmless for
        # templates that don't read it — Jinja ignores the unused kwarg.
        reasoning_effort = req.get("reasoning_effort", None)
        if reasoning_effort:
            chat_kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            chat_kwargs["tools"] = tools
        def _apply_template_with_fallbacks():
            """apply_chat_template with a retry ladder on OPTIONAL kwargs a
            template may reject — a bad optional must degrade, never kill the
            rank (2026-07-08: the Hy3 release template VALIDATES
            reasoning_effort and raises on "medium" → whole pool died)."""
            nonlocal tools
            try:
                return tokenizer.apply_chat_template(messages, **chat_kwargs)
            except Exception as e:
                # 1. Tokenizers that reject `tools=` (template without tool
                #    support). Drop and retry.
                if tools:
                    log(f"chat template rejected tools ({e}); retrying without")
                    chat_kwargs.pop("tools", None)
                    tools = None
                    try:
                        return tokenizer.apply_chat_template(messages, **chat_kwargs)
                    except Exception as e2:
                        e = e2
                # 2. Templates that validate reasoning_effort (Hy3 release:
                #    no_think/low/high only). Drop the dial and retry —
                #    template default beats a dead pool.
                if chat_kwargs.get("reasoning_effort"):
                    log(f"chat template rejected reasoning_effort="
                        f"{chat_kwargs['reasoning_effort']!r} ({e}); retrying without")
                    chat_kwargs.pop("reasoning_effort", None)
                    return tokenizer.apply_chat_template(messages, **chat_kwargs)
                raise

        templated = _apply_template_with_fallbacks()

        # Tokenize the templated prompt. We need the full token list for prefix
        # cache lookup; on a hit we'll feed only the suffix.
        # Two paths to the same answer — apply_chat_template directly with
        # tokenize=True is preferred (handles tools/thinking consistently with
        # the string version), with encode() as fallback.
        try:
            ids_kwargs = dict(chat_kwargs); ids_kwargs["tokenize"] = True
            prompt_tokens_full = list(tokenizer.apply_chat_template(messages, **ids_kwargs))
        except Exception:
            prompt_tokens_full = list(tokenizer.encode(templated, add_special_tokens=False))

        # Prefix cache lookup BEFORE building a fresh cache. On a hit we reuse
        # the existing populated cache and feed only the suffix tokens.
        # SKIP for the DSpark spec pool: its generator owns the session store
        # (session_get/put hooks + its own snapshot restore). The plain
        # _session_lookup here would DELETE the spec-managed entry on its
        # divergent/fresh/non-truncatable branches → the spec fast-prefill saw
        # an empty store every turn (bug 2026-08-06: store 1→0 between requests).
        _spec_on_here = bool(spec_dspark_ctx and spec_dspark_ctx.get("enabled"))
        cached_cache, suffix_tokens, hit_kind = _session_lookup(
            session_id, repo, prompt_tokens_full,
        ) if (session_id and not _spec_on_here) else (None, None, "no-session")

        if cached_cache is not None:
            prompt_cache = cached_cache
            gen_input = suffix_tokens
            cache_label = "session-HIT" if hit_kind != "snap-hit" else "snap-HIT"
        else:
            # Build a per-request prompt cache. Q8 quantized cache halves memory
            # for long contexts (Qwen3-Coder-Next 32k, GLM-5.1, Hy3-preview).
            prompt_cache = _build_prompt_cache(model, kv_q8)
            gen_input = templated  # let stream_generate retokenize the string
            cache_label = ("Q8" if kv_q8 else "fp16") + (
                f"·{hit_kind}" if session_id else ""
            )

        # Snapshot fin-de-prompt pour les caches NON-trimmables (V4 :
        # PoolingCache). Sans lui, chaque tour de conversation diverge en fin
        # de prefixe (la reponse generee est re-templatisee differemment), le
        # rewind est impossible -> re-prefill INTEGRAL a chaque tour (TTFT
        # 20s+ observe en prod sur Pro). On prefill nous-memes tout sauf le
        # dernier token, on snapshotte (O(1), arrays immutables), et le tour
        # suivant restaure ce prefixe garanti. Modeles a cache trimmable :
        # chemin historique intact.
        _snap_pending = None
        _snap_tokens_pending = None
        _spec_dspark_on = bool(spec_dspark_ctx and spec_dspark_ctx.get("enabled"))
        _spec_stats: dict = {}
        if (_SNAP_CACHE_ENABLED
                and session_id and prompt_cache is not None and draft_model is None
                and not _spec_dspark_on          # le pool spec fait son propre prefill
                and not _truncatable_cache(prompt_cache)):
            _pre = (suffix_tokens if cached_cache is not None
                    else prompt_tokens_full)
            if len(_pre) > 1:
                _manual_prefill(model, prompt_cache, _pre[:-1])
            _snap_pending = _snap_cache(prompt_cache)
            _snap_tokens_pending = prompt_tokens_full[:-1]
            gen_input = prompt_tokens_full[-1:]

        if rank == 0:
            log(f"req {req_id}: session={session_id or '-'} "
                f"cache={cache_label} "
                f"prompt_toks={len(prompt_tokens_full)}"
                + (f" suffix_toks={len(suffix_tokens)}" if suffix_tokens else ""))

        ntoks = 0
        t_gen = time.time()
        emit_batch_n = int(os.environ.get("RUNNER_EMIT_BATCH", "10"))
        buf: list[str] = []
        full_text_parts: list[str] = []  # accumulate for tool-call parsing
        gen_token_ids: list[int] = []     # accumulate for session cache update
        gen_kwargs = {"max_tokens": max_tokens}
        if prompt_cache is not None:
            gen_kwargs["prompt_cache"] = prompt_cache
        # Per-model sampling overrides (see MODEL_SAMPLING_DEFAULTS).
        # Returns (None, None) when no entry matches — leaves mlx-lm
        # defaults intact for everything except listed models. This
        # prevents the MiniMax-style degenerate repetition loops without
        # changing behavior for Qwen/GLM/Hy3/etc which are fine on greedy.
        # Prose-vs-code detection (§9.1): M3 codes well but the prose
        # no_repeat_ngram ban drops operators. Flip to the code-safe profile
        # when the request looks like code (strong signals only).
        _is_code = _looks_like_code_request(messages, tools)
        _sampler, _lp = _build_sampling_for(repo, tokenizer, is_code=_is_code)
        if _sampler is not None:
            gen_kwargs["sampler"] = _sampler
        if _lp is not None:
            gen_kwargs["logits_processors"] = _lp
        # Speculative decoding: when a draft_model is loaded, pass it to
        # stream_generate. mlx-lm's stream_generate(draft_model=) draws N
        # tokens from the draft per main-model verify step.
        if draft_model is not None:
            gen_kwargs["draft_model"] = draft_model
            gen_kwargs["num_draft_tokens"] = num_draft_tokens
            # mlx-lm's speculative_generate_step expects a COMBINED
            # [target_layers + draft_layers] cache when prompt_cache is
            # passed (generate.py:527 slices the tail for the draft). Our
            # session/per-request cache is target-only → the draft slice
            # comes out EMPTY and the draft prefill crashes (IndexError,
            # E0 2026-07-06). Let mlx-lm build both caches itself; the
            # session prefix is re-prefilled on this path.
            if prompt_cache is not None:
                gen_kwargs.pop("prompt_cache", None)
                prompt_cache = None
                cached_cache = None
                suffix_tokens = None
                gen_input = templated
                cache_label = "draft-fresh"

        # Native-MTP routing (plan D6/D7). Activation travels PER REQUEST in
        # the fan-out JSONL (`"mtp": {"on":…, "depth":…}`) — identical on
        # every rank, so the decision is aligned by construction. Greedy v0:
        # any sampling profile or draft model falls back to plain AR.
        req_mtp = req.get("mtp") or {}
        use_native_mtp = (
            native_mtp is not None
            and bool(req_mtp.get("on", True))
            and _sampler is None and _lp is None
            and draft_model is None
        )
        canary_sha = hashlib.sha256()

        def _canary_line(payload: dict) -> None:
            sys.stderr.write("[canary] " + json.dumps(payload) + "\n")
            sys.stderr.flush()

        if use_native_mtp:
            from mtp_spec import native_mtp_stream_generate
            _mtp_prompt_ids = list(prompt_tokens_full)
            _mtp_prefix = (len(prompt_tokens_full) - len(suffix_tokens)
                           if cached_cache is not None and suffix_tokens is not None
                           else 0)
            gen_iter = native_mtp_stream_generate(
                model, tokenizer, _mtp_prompt_ids,
                mtp=native_mtp,
                depth=int(req_mtp.get("depth")
                          or os.environ.get("RUNNER_MTP_DEPTH", "3")),
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
                prefix_len=_mtp_prefix,
                hidden_source=os.environ.get("RUNNER_MTP_HIDDEN", "post_norm"),
                stop_ids=_stop_ids,
                canary_cb=lambda r, d, n, s: _canary_line(
                    {"rid": req_id, "rank": rank, "round": r,
                     "drafted": d, "accepted": n, "sha": s}),
            )
        elif _spec_dspark_on and world_size == 1:
            # Single-node DSpark : rollback par TRIM du pool (un-fold), zero
            # re-forward -> ~1.2-1.4x sur code, lossless. Pas de collectives,
            # drafter in-process. Le generateur fait son propre prefill.
            import spec_dspark
            gen_iter = spec_dspark.single_node_spec_stream_generate(
                model, tokenizer, list(prompt_tokens_full),
                max_tokens=max_tokens,
                drafter=spec_dspark_ctx.get("drafter"),
                target_args=spec_dspark_ctx.get("args"),
                dcfg=spec_dspark_ctx.get("dcfg"),
                stop_ids=_stop_ids, stats_out=_spec_stats,
                # fast-prefill (snap-cache) hooks — meme store byte-budgete que
                # le chemin distribue ; restore via les constructeurs frais.
                session_id=session_id if _SNAP_CACHE_ENABLED else None,
                model_id=repo,
                session_get=_spec_session_get,
                session_put=_spec_session_put,
                restore_from_snap=_restore_from_snap)
        elif _spec_dspark_on:
            # DSpark spec distribue : le generateur fait SON prefill (capture
            # les taps -> ctx drafter) et pilote la loop propose/verify en
            # lockstep multi-rang. rank0 yield les tokens acceptes ; les
            # servants tournent la boucle de reception et ne yieldent rien.
            import spec_dspark
            gen_iter = spec_dspark.spec_dspark_stream_generate(
                model, tokenizer, list(prompt_tokens_full),
                max_tokens=max_tokens, rank=rank, size=world_size, group=group,
                drafter=spec_dspark_ctx.get("drafter"),
                target_args=spec_dspark_ctx.get("args"),
                dcfg=spec_dspark_ctx.get("dcfg"),
                stop_ids=_stop_ids,
                # A (spec+snap) : hooks session — lookup/restore/store du store
                # byte-budgete du runner ; restore via les constructeurs frais
                # prouves du chemin plain (1.24.0).
                session_id=session_id if _SNAP_CACHE_ENABLED else None,
                model_id=repo,
                session_get=_spec_session_get,
                session_put=_spec_session_put,
                restore_from_snap=_restore_from_snap,
                stats_out=_spec_stats)
        else:
            gen_iter = stream_generate(model, tokenizer, gen_input, **gen_kwargs)

        # E0 harness canary: per-token cumulative sha over the emitted ids,
        # for the draft-model multi-rank alignment gate (G0). Opt-in.
        token_canary = os.environ.get("RUNNER_TOKEN_CANARY", "0") == "1"

        cancelled_mid_gen = False
        loop_detected = False
        anti_loop = bool(req.get("anti_loop", True))
        # Context-limit: per-request override else the server default. Enforced
        # per-step in the loop below (prompt + generated tokens).
        context_limit = int(req.get("context_limit", 0) or _CONTEXT_LIMIT)
        context_limit_hit = False
        _prompt_n = len(prompt_tokens_full)
        # stream_generate's LAST response carries finish_reason ("stop" on an
        # eos_token_ids hit, "length" on the max_tokens cap). Capture it so the
        # done event can tell the API layer which one it was.
        gen_finish: Optional[str] = None
        for res in gen_iter:
            tok_id = getattr(res, "token", None)
            rf = getattr(res, "finish_reason", None)
            if rf:
                gen_finish = rf
            # Extra stop check, BEFORE emitting the token's text. Chat-tuned
            # models emit a next-turn marker (<|im_end|>, <|role_end|>,
            # <|user|>…) to end the assistant turn. When that marker IS the
            # tokenizer's nominal eos (Qwen's <|im_end|>), stream_generate
            # stops on its own and never yields the marker's text. When it is
            # NOT (Ring-2.x: eos=<|endoftext|> but turns end on <|role_end|>),
            # stream_generate yields the marker as normal text — so we must
            # break BEFORE appending it, or it leaks into the completion
            # (the "391<|role_end|>" bug). See _resolve_eos_token_seqs.
            if isinstance(tok_id, int) and tok_id in _stop_ids:
                gen_finish = "stop"
                break
            buf.append(res.text)
            full_text_parts.append(res.text)
            if isinstance(tok_id, int):
                gen_token_ids.append(tok_id)
                if token_canary:
                    canary_sha.update(tok_id.to_bytes(4, "little"))
                    if (ntoks + 1) % 32 == 0:
                        _canary_line({"rid": req_id, "rank": rank,
                                      "ntoks": ntoks + 1,
                                      "sha": canary_sha.hexdigest()[:16]})
            ntoks += 1
            # Context-limit: hard cap on prompt + generated. Deterministic on
            # ntoks so multi-rank pools break in lockstep (same as anti-loop /
            # _stop_ids). finish_reason stays "length"; the done event carries
            # the distinct `context_limit` flag.
            if context_limit and _prompt_n + ntoks >= context_limit:
                context_limit_hit = True
                gen_finish = "length"
                log(f"req {req_id}: context-limit stop "
                    f"(prompt={_prompt_n} gen={ntoks} limit={context_limit})")
                break
            if len(buf) >= emit_batch_n:
                emit(rank, {"event": "token", "id": req_id, "text": "".join(buf)})
                buf.clear()
            # Anti-loop: every N tokens, test the id stream for degenerate
            # repetition. Pure function of ids identical on every rank →
            # every rank breaks at the same token (multi-rank safe, same
            # construction as the _stop_ids break above).
            if anti_loop and ntoks % ANTI_LOOP_CHECK_EVERY == 0:
                _hit = _detect_loop(gen_token_ids)
                if _hit is None and ntoks % ANTI_LOOP_L_CHECK_EVERY == 0:
                    _hit = _detect_loop_large(gen_token_ids)
                if _hit:
                    loop_detected = True
                    gen_finish = "stop"
                    log(f"req {req_id}: anti-loop stop "
                        f"(period={_hit[0]} repeats={_hit[1]} ntoks={ntoks})")
                    break
            # Hard cancel: the reader thread set our req_id in _cancelled_ids
            # when /admin/runs/{id}/cancel propagated to us. Break out of
            # the generator so MLX stops computing and we surface a clean
            # finish_reason=cancelled. Without this, cancel was a UI mute
            # — runner kept burning tokens.
            if _is_cancelled(req_id):
                cancelled_mid_gen = True
                break
            if stop_requested["flag"]:
                break
        if buf:
            emit(rank, {"event": "token", "id": req_id, "text": "".join(buf)})
        elapsed = time.time() - t_gen
        tps = ntoks / elapsed if elapsed > 0 else 0.0

        # Parse tool calls if the request enabled them. The full output may
        # contain `<tool_call>{"name":..,"arguments":..}</tool_call>` blocks
        # (Hermes-style, used by Qwen3/GLM-4) — we extract them and surface
        # them in OpenAI shape for the API layer to forward.
        tool_calls: list[dict] = []
        if tools:
            full_text = "".join(full_text_parts)
            tool_calls, _ = parse_tool_calls(full_text)

        # Persist the populated cache for this session so the next turn can
        # resume mid-prompt. The cache currently holds K/V for
        # `prompt_tokens_full` + `gen_token_ids`; that's the cumulative
        # token list we store as the session prefix.
        # Le chemin spec gere SA persistence de session DANS le generateur
        # (il possede le cache) — l'outer store ne doit pas ecraser son entree
        # avec le prompt_cache inutilise du chemin plain.
        if session_id and prompt_cache is not None and not _spec_dspark_on:
            cumulative = list(prompt_tokens_full) + gen_token_ids
            # NB (2026-08-05) : snap du PROMPT (pris avant gen). Cache TOUT
            # l'historique jusqu'au dernier message (88% mesure en prod
            # Companion) car le round-trip token EST stable -> l'histo entier
            # reste un prefixe deterministe. Les ~12% non-caches = la DERNIERE
            # reponse assistant (posterieure au snap) + le nouveau message.
            # Un snap POST-gen les capturerait (faisable, round-trip stable)
            # mais la 1re implementation buggait (match exact snap-hit
            # divergeait sur le dernier token tronque) et le gain marginal
            # (~12%) ne justifie pas le risque sur les 88% qui marchent.
            _session_store_after_gen(session_id, repo, prompt_cache, cumulative,
                                     rank=rank, world=world_size,
                                     snap=_snap_pending,
                                     snap_tokens=_snap_tokens_pending)

        # Prefix-cache hit accounting. When `cached_cache` was found and
        # `suffix_tokens` < `prompt_tokens_full`, the difference is the count
        # served from cache (no prefill cost). Mirrors an OpenAI-style `cached_tokens`
        # so the API layer can publish OpenAI-compat
        # `prompt_tokens_details.cached_tokens` and Companion's StatsRow
        # surfaces the win for local pools too, not just the cloud proxy path.
        cached_count = (
            max(0, len(prompt_tokens_full) - len(suffix_tokens))
            if (cached_cache is not None and suffix_tokens is not None)
            else 0
        )
        if _spec_dspark_on:
            # A (spec+snap) : le generateur publie son hit via stats_out.
            cached_count = int(_spec_stats.get("cached_tokens") or 0)
        done_event = {
            "event": "done",
            "id": req_id,
            "ntoks": ntoks,
            "prompt_tokens": len(prompt_tokens_full),
            "cached_tokens": cached_count,
            "elapsed_s": elapsed,
            "tps": tps,
        }
        if cancelled_mid_gen:
            done_event["finish_reason"] = "cancelled"
        else:
            # Fallback when the generator yielded no finish_reason (MTP path,
            # stop_requested break): infer from the token count vs the cap.
            done_event["finish_reason"] = gen_finish or (
                "length" if ntoks >= max_tokens else "stop")
        if loop_detected:
            done_event["loop_detected"] = True
        if context_limit_hit:
            done_event["context_limit"] = True
        if tool_calls:
            done_event["tool_calls"] = tool_calls
        if session_id:
            done_event["session"] = {
                "id": session_id,
                "cache_kind": cache_label,
                "cumulative_tokens": (len(prompt_tokens_full) + len(gen_token_ids))
                                     if prompt_cache is not None else 0,
            }
        emit(rank, done_event)
        # Drop our entry from the cancellation set so it doesn't grow unbounded.
        _clear_cancelled(req_id)
        log(f"req {req_id}: {ntoks} toks in {elapsed:.1f}s = {tps:.2f} tok/s"
            + (" · CANCELLED" if cancelled_mid_gen else "")
            + (f" · {len(tool_calls)} tool_call(s)" if tool_calls else "")
            + (f" · session={session_id}({cache_label})" if session_id else "")
            + (f" · finish={done_event['finish_reason']}" if not cancelled_mid_gen else ""))


# ──────────────────────────────────────────────────────────────────────────────
# Batched main loop (single-rank only)
#
# Drives `mlx_lm.generate.BatchGenerator` so multiple in-flight gen requests
# share each forward pass. A reader thread pulls JSONL from stdin into a
# queue; the main thread drains, classifies each request, inserts into the
# BatchGenerator (passing pre-warmed prefix caches when sessioned), then
# ticks the BG and routes per-uid responses back to their req_id.
# ──────────────────────────────────────────────────────────────────────────────
def _run_batched_main(model, tokenizer, repo: str, kv_q8_default: bool,
                      stop_requested: dict, rank: int,
                      world_size: int = 1) -> None:
    eos_seqs = _resolve_eos_token_seqs(tokenizer)
    bg_kwargs: dict = {
        "max_tokens": 4096,
        "completion_batch_size": int(os.environ.get("RUNNER_BATCH_SIZE", "8")),
        "prefill_batch_size": int(os.environ.get("RUNNER_PREFILL_BATCH", "4")),
        "prefill_step_size": 2048,
    }
    if eos_seqs:
        bg_kwargs["stop_tokens"] = eos_seqs
    bg = BatchGenerator(model, **bg_kwargs)

    in_q: queue.Queue = queue.Queue()
    EOF = object()

    def reader() -> None:
        while not stop_requested["flag"]:
            line = sys.stdin.readline()
            if not line:
                in_q.put(EOF)
                return
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"[runner] bad json: {e}\n")
                continue
            # Hot-path cancel — same pattern as legacy main loop. Cancel
            # commands toggle the shared set; the main thread picks them
            # up either during the per-uid drain or per-iteration check.
            if msg.get("cmd") == "cancel":
                _mark_cancelled(msg.get("id"))
                continue
            in_q.put(msg)

    threading.Thread(target=reader, daemon=True).start()

    # uid → per-request state
    slot: dict[int, dict] = {}
    emit_batch_n = int(os.environ.get("RUNNER_EMIT_BATCH", "10"))
    # Poison guard (2026-08-27): removing a LONG slot (thousands of generated
    # tokens) from the BatchGenerator leaves it in a degraded state where every
    # subsequent request decodes ~50x slower. Proven deterministically on
    # glm5_next Q8 single-node: fresh BG serves a 7.5k-token c03 at 21 tok/s
    # (finish=stop, perfectly healthy), then a 50-token "hello" times out at
    # >150s; a 600-token completion does NOT trigger it; no cancel involved.
    # This was the root of the bench C03/C04 "0.2 tok/s" wedges (the ghost
    # generations fixed in api v1.41.2 were the amplifier, not the source).
    # Same medicine as the bg.next() extend-cache recovery below: rebuild the
    # BG. Every removal path (natural finish, hard-cancel, anti-loop) funnels
    # through _emit_done_for / bg.remove — we mark the BG dirty when a big
    # slot is removed and rebuild as soon as the batch is EMPTY, so in-flight
    # requests are never disturbed and the cost lands between requests.
    _bg_rebuild_min_toks = int(os.environ.get("BG_REBUILD_MIN_TOKS", "1000"))
    # Force-rebuild guard (2026-08-27b): the empty-batch condition above can
    # STARVE — with concurrent traffic (bench monitor probes, overlapping
    # requests) the batch never empties, the dirty flag waits forever, and the
    # poisoned BG keeps serving everything at ~0.25 tok/s (observed live on
    # Bench R2 r01: 20 min at 0.25 with uids climbing and zero rebuilds). When
    # the BG has been dirty for > BG_REBUILD_FORCE_S AND the batch's aggregate
    # throughput is < BG_REBUILD_FORCE_TPS (i.e. it IS poisoned — a healthy
    # overlapping batch decodes 10-20+ tok/s and is never touched), finalize
    # the in-flight slots with finish=error (same contract as the
    # metal::malloc recovery below — they are crawling anyway) and rebuild.
    _bg_force_s = float(os.environ.get("BG_REBUILD_FORCE_S", "90"))
    _bg_force_tps = float(os.environ.get("BG_REBUILD_FORCE_TPS", "2.0"))
    bg_state = {"dirty": False, "since": 0.0, "toks": 0}
    # ── Upstream instrumentation + candidate source fix (2026-08-27) ──────
    # AMONT diagnosis: MLX (active+cache) memory grows ~50MB/generated-token
    # on this path and hits the ~499GB Metal resource limit at ~7495 tokens
    # (the constant magic number in every truncated run). Suspected: per-step
    # allocations of ever-growing sizes that the MLX buffer cache cannot
    # reuse, so cache_memory balloons. Instrumentation logs active/cache mem
    # + step timing every BG_MEM_LOG_EVERY generated tokens (0 = off) to
    # confirm; BG_CLEAR_CACHE_EVERY (0 = off) then bounds it by calling
    # mx.clear_cache() every N generated tokens — the candidate SOURCE fix
    # for both the 7495-token truncation ceiling and (if allocator thrash is
    # the slow path) the post-long-gen crawl. Both OFF by default until
    # measured on a controlled run.
    # Defaults ON (2026-08-27, bench unblock): the runner gets its env from
    # remote_cmd's inline assignments — arbitrary new envs don't reach it
    # without an api.py change + container restart. Baking the defaults here
    # keeps this a runner-only deploy. clear_cache every 512 generated tokens
    # bounds the buffer-cache balloon (the ~50MB/tok growth that hits the
    # 499GB Metal limit at ~7495 toks); mem log every 500 gives the curves.
    _bg_mem_log_every = int(os.environ.get("BG_MEM_LOG_EVERY", "500"))
    _bg_clear_cache_every = int(os.environ.get("BG_CLEAR_CACHE_EVERY", "512"))
    _bg_mem_ctr = {"toks": 0, "last_t": time.time(), "last_toks": 0}

    log(f"batched main: completion_batch={bg_kwargs['completion_batch_size']}, "
        f"prefill_batch={bg_kwargs['prefill_batch_size']}, "
        f"eos={[s[0] for s in eos_seqs]}")

    def _emit_done_for(uid: int, finish_reason: Optional[str],
                       resp_cache=None, resp_all_tokens=None) -> None:
        s = slot.get(uid)
        if s is None:
            return
        # Drain remaining buffer
        if s["buf"]:
            emit(rank, {"event": "token", "id": s["req_id"], "text": "".join(s["buf"])})
            s["buf"].clear()
        elapsed = time.time() - s["t_gen"]
        tps = (s["ntoks"] / elapsed) if elapsed > 0 else 0.0
        # Tool-call parsing on the full text
        tool_calls: list[dict] = []
        if s["tools"]:
            tool_calls, _ = parse_tool_calls("".join(s["full_text_parts"]))
        # Session persistence: prefer the cache from the finishing Response
        # (mlx-lm attaches it as resp.prompt_cache). Fallback to extract_cache
        # by uid (works pre-removal).
        if s["session_id"]:
            cache_obj = resp_cache
            cum_tokens: list[int] = list(resp_all_tokens or [])
            if cache_obj is None:
                try:
                    extracted = bg.extract_cache([uid])
                    ent = extracted.get(uid)
                    if ent is not None:
                        cache_obj, fallback_toks = ent
                        if not cum_tokens:
                            cum_tokens = list(fallback_toks or [])
                except Exception as e:
                    log(f"session extract_cache fallback failed for uid {uid}: {e}")
            if not cum_tokens:
                cum_tokens = list(s["prompt_tokens_full"]) + s["gen_token_ids"]
            if cache_obj is not None:
                _session_store_after_gen(s["session_id"], repo, cache_obj,
                                         cum_tokens,
                                         rank=rank, world=world_size)
            else:
                log(f"session {s['session_id']}: cache unavailable post-gen, not stored")
        done_event: dict = {
            "event": "done",
            "id": s["req_id"],
            "ntoks": s["ntoks"],
            "prompt_tokens": len(s["prompt_tokens_full"]),
            # cached_tokens populated at insert time from cache_offset (0 if
            # no session hit). Mirrors OpenAI prompt_tokens_details.cached_tokens.
            "cached_tokens": int(s.get("cached_tokens") or 0),
            "elapsed_s": elapsed,
            "tps": tps,
        }
        if finish_reason:
            done_event["finish_reason"] = finish_reason
        if s.get("loop_detected"):
            done_event["loop_detected"] = True
        if tool_calls:
            done_event["tool_calls"] = tool_calls
        if s["session_id"]:
            done_event["session"] = {
                "id": s["session_id"],
                "cache_kind": s["cache_label"],
                "cumulative_tokens": len(s["prompt_tokens_full"]) + len(s["gen_token_ids"]),
            }
        emit(rank, done_event)
        log(f"req {s['req_id']}: {s['ntoks']} toks in {elapsed:.1f}s = {tps:.2f} tok/s"
            + (f" · {len(tool_calls)} tool_call(s)" if tool_calls else "")
            + (f" · session={s['session_id']}({s['cache_label']})" if s["session_id"] else "")
            + (f" · finish={finish_reason}" if finish_reason else ""))
        # Poison guard: a big slot is leaving the BG — schedule a rebuild for
        # the next moment the batch is empty (see the flag's comment above).
        # Criterion = TOTAL context (prompt + generated), not just generated:
        # a huge-prompt test that emits few tokens (Bench P profile) poisons
        # the BG just as hard as a long generation (observed 2026-08-27: p01/
        # p02 big-prompt completions < 1000 gen toks → no dirty → p03 and the
        # whole R2 bench crawled at 0.25 with zero rebuilds).
        if (len(s["prompt_tokens_full"]) + s["ntoks"]) >= _bg_rebuild_min_toks:
            if not bg_state["dirty"]:
                bg_state["dirty"] = True
                bg_state["since"] = time.time()
                bg_state["toks"] = 0
        try:
            bg.remove([uid])
        except Exception:
            pass
        del slot[uid]

    while not stop_requested["flag"]:
        # ── Poison guard rebuild ───────────────────────────────────────────
        # Must run BEFORE the drain so the next request inserts into the
        # fresh BG, never the poisoned one. Preferred path: batch is empty —
        # in-flight slots are never disturbed.
        if bg_state["dirty"] and not slot:
            try:
                bg = BatchGenerator(model, **bg_kwargs)
                mx.clear_cache()
                log(f"BG rebuilt after big-slot removal (poison guard, "
                    f"ctx >= {_bg_rebuild_min_toks})")
            except Exception as e:
                log(f"BG poison-guard rebuild FAILED: {e}")
            bg_state["dirty"] = False
        # Force path (starvation guard): the batch never empties (concurrent
        # probes / overlapping requests keep arriving) AND its aggregate
        # throughput proves it is poisoned — cull the crawling slots with
        # finish=error (same contract as the metal::malloc recovery) and
        # rebuild. A healthy overlapping batch (>= _bg_force_tps aggregate)
        # is never touched.
        elif bg_state["dirty"] and slot:
            _age = time.time() - bg_state["since"]
            if _age > _bg_force_s and (bg_state["toks"] / max(_age, 1.0)) < _bg_force_tps:
                log(f"BG FORCE rebuild: dirty {_age:.0f}s, aggregate "
                    f"{bg_state['toks'] / max(_age, 1.0):.2f} tok/s < {_bg_force_tps} "
                    f"— culling {len(slot)} crawling slot(s)")
                for uid in list(slot.keys()):
                    _emit_done_for(uid, finish_reason="error")
                try:
                    bg = BatchGenerator(model, **bg_kwargs)
                    mx.clear_cache()
                    log("BG rebuilt (force path)")
                except Exception as e:
                    log(f"BG force rebuild FAILED: {e}")
                bg_state["dirty"] = False
        # ── Drain incoming requests (non-blocking) ─────────────────────────
        drained = 0
        while drained < 32:
            try:
                req = in_q.get_nowait()
            except queue.Empty:
                break
            drained += 1
            if req is EOF:
                stop_requested["flag"] = True
                log("stdin closed, exiting batched loop")
                break
            cmd = req.get("cmd")
            if cmd == "stop":
                stop_requested["flag"] = True
                log("stop cmd received")
                break
            if cmd == "session_clear":
                sid = req.get("session_id")
                if sid:
                    with _session_lock:
                        _session_store.pop(sid, None)
                    log(f"cleared session {sid}")
                else:
                    _session_clear_all()
                    log("cleared all sessions")
                continue
            if cmd != "gen":
                log(f"unknown cmd: {cmd}")
                continue

            req_id = req.get("id", "")
            prompt = req.get("prompt")
            messages_in = req.get("messages")
            tools = req.get("tools")
            max_tokens = int(req.get("max_tokens", 200))
            enable_thinking = req.get("enable_thinking", None)
            kv_q8 = bool(req.get("kv_q8", kv_q8_default))
            session_id = req.get("session_id") or None

            if messages_in is None:
                if not prompt:
                    log("gen with neither messages nor prompt — skipping")
                    continue
                messages_in = [{"role": "user", "content": prompt}]

            # See note in single-stream branch: Hy3 chat template expects
            # `tool_calls[*].function.arguments` as a dict, not a JSON string.
            messages_in = _normalize_messages_for_template(messages_in)

            chat_kwargs = {"add_generation_prompt": True, "tokenize": False}
            if enable_thinking is not None:
                chat_kwargs["enable_thinking"] = enable_thinking
                # M3 reads `thinking_mode`, not enable_thinking — see the
                # single-stream branch.
                if "minimax-m3" in repo.lower():
                    chat_kwargs["thinking_mode"] = "enabled" if enable_thinking else "disabled"
            # reasoning_effort: see single-stream branch — template injects it
            # as a "Reasoning: <effort>" system directive (Step-3.7).
            reasoning_effort = req.get("reasoning_effort", None)
            if reasoning_effort:
                chat_kwargs["reasoning_effort"] = reasoning_effort
            if tools:
                chat_kwargs["tools"] = tools
            try:
                templated = tokenizer.apply_chat_template(messages_in, **chat_kwargs)
            except Exception as e:
                if tools:
                    log(f"chat template rejected tools ({e}); retrying without")
                    chat_kwargs.pop("tools", None)
                    templated = tokenizer.apply_chat_template(messages_in, **chat_kwargs)
                    tools = None
                else:
                    raise

            try:
                ids_kwargs = dict(chat_kwargs); ids_kwargs["tokenize"] = True
                prompt_tokens_full = list(tokenizer.apply_chat_template(messages_in, **ids_kwargs))
            except Exception:
                prompt_tokens_full = list(tokenizer.encode(templated, add_special_tokens=False))

            cached_cache, suffix_tokens, hit_kind = (
                _session_lookup(session_id, repo, prompt_tokens_full)
                if session_id else (None, None, "no-session")
            )
            # In batched mode, kv_q8 is NOT supported (mlx-lm's
            # QuantizedKVCache "does not yet support batching with history").
            # If a cached session cache contains any Q8 layer, drop it and
            # fall back to a fresh non-Q8 cache built by BG.
            cache_has_q8 = (
                cached_cache is not None and
                any(type(c).__name__ == "QuantizedKVCache" for c in cached_cache)
            )
            if cached_cache is not None and not cache_has_q8:
                # Session HIT: reuse the populated cache, feed only the suffix.
                # Trust the lookup — it already ran the truncation safety check.
                bg_input = list(suffix_tokens)
                cache_offset = len(prompt_tokens_full) - len(suffix_tokens)
                all_toks_for_slot = list(prompt_tokens_full[:cache_offset])
                cache_for_slot = cached_cache
                cache_label = f"session-{hit_kind.upper()}"
            else:
                # No session, miss, divergent, or Q8 incompatible — fresh cache
                # via BG's default builder.
                bg_input = list(prompt_tokens_full)
                cache_for_slot = None
                all_toks_for_slot = []
                if cache_has_q8:
                    cache_label = "fp16·session-Q8-bypass"
                else:
                    cache_label = ("fp16" if not kv_q8 else "fp16(Q8-skipped)") + (
                        f"·{hit_kind}" if session_id else ""
                    )

            try:
                if cache_for_slot is not None:
                    uids = bg.insert(
                        prompts=[bg_input],
                        max_tokens=[max_tokens],
                        caches=[cache_for_slot],
                        all_tokens=[all_toks_for_slot],
                    )
                else:
                    uids = bg.insert(prompts=[bg_input], max_tokens=[max_tokens])
            except Exception as e:
                log(f"bg.insert failed for req {req_id}: {e}")
                emit(rank, {"event": "done", "id": req_id, "ntoks": 0,
                            "elapsed_s": 0.0, "tps": 0.0, "error": str(e)[:200]})
                continue

            uid = uids[0]
            # Allocate a stateful streaming detokenizer per slot so BPE merges
            # across tokens stay coherent (vs naive decode([tok]) which can
            # mangle multi-byte chars and merge boundaries).
            # cache_offset: tokens served from prefix cache (0 unless this is a
            # session hit). Stored on the slot so the done event can expose
            # OpenAI-compat `cached_tokens` per request. Defined above when we
            # selected the cache path, but `cache_for_slot is None` branches
            # skip the assignment — backfill 0 here.
            slot_cache_hit = locals().get("cache_offset", 0) if cache_for_slot is not None else 0
            slot[uid] = {
                "req_id": req_id,
                "prompt_tokens_full": prompt_tokens_full,
                "cached_tokens": slot_cache_hit,
                "gen_token_ids": [],
                "buf": [],
                "full_text_parts": [],
                "ntoks": 0,
                "t_gen": time.time(),
                "tools": tools,
                "session_id": session_id,
                "cache_label": cache_label,
                "detok": tokenizer.detokenizer,  # new instance per slot
                "anti_loop": bool(req.get("anti_loop", True)),
                "loop_detected": False,
            }
            log(f"req {req_id}: inserted as uid={uid} "
                f"session={session_id or '-'} cache={cache_label} "
                f"prompt_toks={len(prompt_tokens_full)}"
                + (f" suffix_toks={len(suffix_tokens)}" if suffix_tokens else ""))

        # ── Hard-cancel removal pass ─────────────────────────────────────
        # Before ticking BG, check if any in-flight slot got cancelled. We
        # call bg.remove([uid]) which drops the uid from all 3 stages
        # (unprocessed / prompt-batch / generation-batch) and stops billing
        # compute on it. The corresponding slot is finalised with
        # finish_reason=cancelled so the API surfaces a clean cancel.
        cancelled_uids: list[int] = []
        for uid, s in list(slot.items()):
            if _is_cancelled(s.get("req_id")):
                cancelled_uids.append(uid)
        if cancelled_uids:
            try:
                bg.remove(cancelled_uids)
            except Exception as e:
                log(f"bg.remove({cancelled_uids}) failed: {e}")
            for uid in cancelled_uids:
                req_id = slot.get(uid, {}).get("req_id")
                _emit_done_for(uid, finish_reason="cancelled")
                slot.pop(uid, None)
                _clear_cancelled(req_id)

        # ── Tick the BatchGenerator ────────────────────────────────────────
        if not slot:
            # No active work — short sleep to avoid burning CPU on bg.next()
            time.sleep(0.005)
            continue
        try:
            tick = bg.next()
        except Exception as e:
            import traceback as _tb
            log(f"bg.next() TRACEBACK: {_tb.format_exc()[:2000]}")
            # bg.next() can poison its internal _prompt_batch.prompt_cache —
            # observed with mlx-lm 0.31.3 where _extend_cache calls
            # `ca.extend(cb)` and certain cache types (Q8, MLA, some fp16
            # variants) don't have .extend → AttributeError. Clearing slots
            # isn't enough because the poisoned cache stays in _prompt_batch
            # and every subsequent request crashes the same way. Re-create
            # the BG from scratch — costs one model.reset() but unblocks the
            # serving path immediately. The first request after the reset
            # always works; the bug only fires on subsequent extends.
            log(f"bg.next() failed: {e}; resetting BatchGenerator + clearing slots")
            for uid in list(slot.keys()):
                _emit_done_for(uid, finish_reason="error")
            try:
                bg = BatchGenerator(model, **bg_kwargs)
                log("BatchGenerator re-initialised after extend-cache failure")
                bg_state["dirty"] = False  # fresh BG — poison guard satisfied
                try:
                    mx.clear_cache()  # release the ballooned buffer cache —
                    # without this each 499GB-limit cycle RATCHETS process
                    # memory up until the allocator lives at the ceiling and
                    # every request crawls (r09 2026-08-27: fresh BG on a
                    # pinned process still 0.27 tok/s; only a reload cured).
                    log("mx.clear_cache() after malloc-limit reset")
                except Exception:
                    pass
            except Exception as e2:
                log(f"BG reinit failed: {e2} — runner will die next iteration")
                raise
            time.sleep(0.05)
            continue

        # bg.next() returns either (prompt_responses, gen_responses) tuple or
        # a flat list depending on version. Normalise.
        prompt_responses: list = []
        gen_responses: list = []
        if isinstance(tick, tuple) and len(tick) == 2:
            prompt_responses, gen_responses = tick
        elif isinstance(tick, list):
            gen_responses = tick

        if not gen_responses and not prompt_responses:
            time.sleep(0.005)
            continue

        # Build the set of stop-token ids so we don't emit them as text.
        stop_ids = {seq[0] for seq in eos_seqs if len(seq) == 1}
        for resp in gen_responses:
            uid = getattr(resp, "uid", None)
            if uid is None or uid not in slot:
                continue
            s = slot[uid]
            tok = getattr(resp, "token", None)
            finish = getattr(resp, "finish_reason", None)
            if isinstance(tok, int):
                s["gen_token_ids"].append(tok)
                s["ntoks"] += 1
                if bg_state["dirty"]:
                    bg_state["toks"] += 1  # feeds the force-rebuild rate check
                # Upstream instrumentation / candidate fix (see flags above).
                _bg_mem_ctr["toks"] += 1
                if (_bg_mem_log_every
                        and _bg_mem_ctr["toks"] % _bg_mem_log_every == 0):
                    _now = time.time()
                    _dt = _now - _bg_mem_ctr["last_t"]
                    _dtoks = _bg_mem_ctr["toks"] - _bg_mem_ctr["last_toks"]
                    _bg_mem_ctr["last_t"] = _now
                    _bg_mem_ctr["last_toks"] = _bg_mem_ctr["toks"]
                    try:
                        log(f"[bg-mem] toks={_bg_mem_ctr['toks']} "
                            f"active={mx.get_active_memory() / 1e9:.1f}GB "
                            f"cache={mx.get_cache_memory() / 1e9:.1f}GB "
                            f"peak={mx.get_peak_memory() / 1e9:.1f}GB "
                            f"rate={_dtoks / max(_dt, 1e-6):.2f} tok/s "
                            f"cache_kv={bg.prompt_cache_nbytes / 1e9:.1f}GB")
                    except Exception as e:
                        log(f"[bg-mem] probe failed: {e}")
                if (_bg_clear_cache_every
                        and _bg_mem_ctr["toks"] % _bg_clear_cache_every == 0):
                    try:
                        mx.clear_cache()
                    except Exception:
                        pass
                # Stream via the slot's stateful detokenizer — handles BPE
                # merges, multi-byte chars, and special-token suppression
                # correctly across tokens. Skip stop tokens from user text;
                # they'll trigger finish_reason="stop" instead.
                if tok not in stop_ids:
                    try:
                        s["detok"].add_token(tok)
                        seg = s["detok"].last_segment or ""
                    except Exception as e:
                        # Fallback to one-shot decode (loses some BPE merge
                        # info but never crashes).
                        try:
                            seg = tokenizer.decode([tok], skip_special_tokens=True)
                        except Exception:
                            seg = ""
                    if seg:
                        s["buf"].append(seg)
                        s["full_text_parts"].append(seg)
                        if len(s["buf"]) >= emit_batch_n:
                            emit(rank, {"event": "token", "id": s["req_id"],
                                        "text": "".join(s["buf"])})
                            s["buf"].clear()
                # Anti-loop (batched path): same detector as the multi-rank
                # loop. _emit_done_for drains the buffer, emits done with the
                # loop flag, removes the uid from the BG and drops the slot —
                # later responses for this uid fall through `uid not in slot`.
                if (s["anti_loop"]
                        and s["ntoks"] % ANTI_LOOP_CHECK_EVERY == 0):
                    _hit = _detect_loop(s["gen_token_ids"])
                    if _hit is None and s["ntoks"] % ANTI_LOOP_L_CHECK_EVERY == 0:
                        _hit = _detect_loop_large(s["gen_token_ids"])
                    if _hit:
                        s["loop_detected"] = True
                        log(f"req {s['req_id']}: anti-loop stop "
                            f"(period={_hit[0]} repeats={_hit[1]} ntoks={s['ntoks']})")
                        _emit_done_for(uid, finish_reason="stop")
                        continue
            if finish:
                # Finalize detokenizer to flush any trailing bytes (e.g.
                # an incomplete multi-byte sequence at the very end).
                try:
                    s["detok"].finalize()
                    tail = s["detok"].last_segment or ""
                    if tail:
                        s["buf"].append(tail)
                        s["full_text_parts"].append(tail)
                except Exception:
                    pass
                _emit_done_for(uid, finish_reason=finish,
                               resp_cache=getattr(resp, "prompt_cache", None),
                               resp_all_tokens=getattr(resp, "all_tokens", None))


if __name__ == "__main__":
    main()
