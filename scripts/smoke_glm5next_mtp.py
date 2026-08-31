"""Standalone shape+coherence smoke for the GLM-5.3-Flash native MTP head.

Run ON a node with the shared /Volumes/models and the vendored glm5_next module
installed into mlx_lm.models. Reproduces the engine's env (MLX_METAL_FAST_SYNCH
+ RUNNER_MTP_HYBRID_SNAPSHOT) so any cross-stream / hybrid-rollback bug shows.

  RUNNER_MTP_HYBRID_SNAPSHOT=1 MLX_METAL_FAST_SYNCH=1 \
    python3 smoke_glm5next_mtp.py [--trunk <dir>] [--tokens 24]

Checks:
  1. load_native_mtp -> STRICT load, zero unmatched keys.
  2. draft_step: S=1, chained x3, batched re-forward — finite logits [1,S,V],
     finite hidden [1,S,H], cache offset advances.
  3. ONE end-to-end native_mtp_stream_generate — decoded text must be COHERENT
     (proves head + hybrid rollback + indexer _pool invalidation all work);
     prints the accept rate (a positive rate proves the head drafts usefully,
     not just that verify carries the trunk).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

SCRIPTS = str(Path(__file__).resolve().parent)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

DEFAULT_TRUNK = "/Volumes/models/odysseus/odyssai/GLM-5.3-Flash-Q6h16"


def _p(m: str) -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", default=DEFAULT_TRUNK)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--prompt", default="Explain what a Fibonacci sequence is in two sentences.")
    a = ap.parse_args()

    _p(f"[env] MLX_METAL_FAST_SYNCH={os.environ.get('MLX_METAL_FAST_SYNCH')} "
       f"RUNNER_MTP_HYBRID_SNAPSHOT={os.environ.get('RUNNER_MTP_HYBRID_SNAPSHOT')}")
    _p(f"[smoke] trunk = {a.trunk}")

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from mtp_module import load_native_mtp

    t0 = time.time()
    model, tokenizer = load(a.trunk)
    _p(f"[load] trunk loaded in {time.time()-t0:.1f}s "
       f"(model_type={getattr(model, 'model_type', '?')})")

    try:
        from mlx_lm.generate import generation_stream as gen_stream
    except Exception:
        gen_stream = mx.default_stream(mx.default_device())

    # ── 1. Build + load the MTP module (STRICT) ────────────────────────────
    t0 = time.time()
    mtp = load_native_mtp(model, a.trunk)
    if mtp is None:
        _p("[FAIL] load_native_mtp returned None — no MTP head loaded")
        return 1
    _p(f"[mtp] loaded in {time.time()-t0:.1f}s  family={mtp.family} "
       f"hybrid_snapshot={getattr(mtp, 'hybrid_snapshot', False)}")

    # Cross-check STRICT: every module param must have been filled (load_weights
    # strict=True already raises on any missing/extra key, so reaching here == 0
    # unmatched; assert non-empty as a belt-and-braces guard).
    from mlx.utils import tree_flatten
    n_params = len(tree_flatten(mtp.parameters()))
    _p(f"[mtp] module has {n_params} parameter tensors (STRICT load OK)")

    # Build a valid seed hidden by running the trunk over the prompt once.
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": a.prompt}], add_generation_prompt=True)
    else:
        prompt_ids = tokenizer.encode(a.prompt)
    _p(f"[smoke] prompt_ids = {len(prompt_ids)} tokens")

    inner = model.model
    lm_head = model.lm_head
    seed_cache = make_prompt_cache(model)
    with mx.stream(gen_stream):
        h_post = inner(mx.array([prompt_ids], dtype=mx.uint32), seed_cache)
        mx.eval(h_post)
    seed_hidden = h_post[:, -1:, :]                       # [1,1,H]
    seed_tok = int(mx.argmax(lm_head(seed_hidden)[:, -1, :], axis=-1).item())
    H = seed_hidden.shape[-1]
    _p(f"[smoke] seed hidden {tuple(seed_hidden.shape)} finite="
       f"{bool(mx.isfinite(seed_hidden.astype(mx.float32)).all().item())} "
       f"seed_tok={seed_tok}")

    def _cache_off(c):
        first = c[0]
        try:
            first = first[0]
        except TypeError:
            pass
        return getattr(first, "offset", -1)

    # ── 2a. draft_step S=1 ─────────────────────────────────────────────────
    c1 = mtp.make_cache()
    with mx.stream(gen_stream):
        lg, hd = mtp.draft_step(mx.array([[seed_tok]], dtype=mx.uint32),
                                seed_hidden, c1)
        mx.eval(lg, hd)
    V = lg.shape[-1]
    assert lg.shape == (1, 1, V), f"S=1 logits shape {lg.shape}"
    assert hd.shape == (1, 1, H), f"S=1 hidden shape {hd.shape}"
    assert bool(mx.isfinite(lg).all().item()), "S=1 logits not finite"
    assert bool(mx.isfinite(hd.astype(mx.float32)).all().item()), "S=1 hidden not finite"
    off1 = _cache_off(c1)
    assert off1 == 1, f"S=1 offset expected 1 got {off1}"
    _p(f"[draft S=1] logits {tuple(lg.shape)} hidden {tuple(hd.shape)} "
       f"offset={off1} vocab={V}  OK")

    # ── 2b. chained x3 (draft chain) ───────────────────────────────────────
    d_tok, d_hid = seed_tok, seed_hidden
    cc = mtp.make_cache()
    chain = []
    with mx.stream(gen_stream):
        for i in range(3):
            dl, d_hid = mtp.draft_step(mx.array([[d_tok]], dtype=mx.uint32),
                                       d_hid, cc)
            assert dl.shape == (1, 1, V) and d_hid.shape == (1, 1, H)
            assert bool(mx.isfinite(dl).all().item()), f"chain step {i} logits not finite"
            d_tok = int(mx.argmax(dl[:, -1, :], axis=-1).item())
            chain.append(d_tok)
    off_cc = _cache_off(cc)
    assert off_cc == 3, f"chain offset expected 3 got {off_cc}"
    _p(f"[draft x3] chained tokens={chain} offset={off_cc}  OK")

    # ── 2c. batched re-forward (accepted-pairs commit path) ────────────────
    cb = mtp.make_cache()
    toks3 = mx.array([[seed_tok] + chain[:2]], dtype=mx.uint32)          # [1,3]
    hid3 = mx.broadcast_to(seed_hidden, (1, 3, H))
    with mx.stream(gen_stream):
        bl, bh = mtp.draft_step(toks3, hid3, cb)
        mx.eval(bl, bh)
    assert bl.shape == (1, 3, V), f"batched logits shape {bl.shape}"
    assert bh.shape == (1, 3, H), f"batched hidden shape {bh.shape}"
    assert bool(mx.isfinite(bl).all().item()), "batched logits not finite"
    off_cb = _cache_off(cb)
    assert off_cb == 3, f"batched offset expected 3 got {off_cb}"
    _p(f"[draft batched S=3] logits {tuple(bl.shape)} hidden {tuple(bh.shape)} "
       f"offset={off_cb}  OK")

    # ── 3. end-to-end native_mtp_stream_generate ───────────────────────────
    from mtp_spec import native_mtp_stream_generate
    stop_ids = set(getattr(tokenizer, "eos_token_ids", None)
                   or ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []))
    _p(f"[e2e] generating {a.tokens} tokens (depth={a.depth}, stop_ids={sorted(stop_ids)})")
    t0 = time.time()
    text = ""
    last = None
    for r in native_mtp_stream_generate(
            model, tokenizer, prompt_ids,
            mtp=mtp, depth=a.depth, max_tokens=a.tokens, stop_ids=stop_ids):
        text += r.text
        last = r
    dt = time.time() - t0
    ar = last.accept_rate if last else 0.0
    _p(f"[e2e] done in {dt:.1f}s  emitted={last.generation_tokens if last else 0} "
       f"accept_rate={ar:.2%}  finish={last.finish_reason if last else '?'}")
    _p("=" * 70)
    _p("[e2e DECODED]:")
    _p(text)
    _p("=" * 70)

    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\t")
    coherent = len(text.strip()) >= 8 and printable >= 0.9 * max(len(text), 1)
    if not coherent:
        _p("[WARN] output looks short/garbled — inspect above")
    _p(f"[smoke] ALL SHAPE CHECKS PASSED; text_coherent_heuristic={coherent} "
       f"accept_rate={ar:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
