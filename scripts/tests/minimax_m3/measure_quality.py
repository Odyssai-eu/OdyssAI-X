#!/usr/bin/env python3
"""Quant-quality metrics for MiniMax-M3 / any mlx-lm model — OdyssAI-X#53.

Three numbers to compare Q6 / mixed (experts-6 + head-8) / Q8:

  1. perplexity  — global fluency. The standard number, but BLUNT for the M3
                   corruption: drift is ~0.5-3% of tokens, so a few drifted
                   tokens barely move the mean. Good for "overall quant damage".
  2. surprise    — the top-N positions where the model is most surprised by the
                   ACTUAL next token in a clean reference text. Surfaces WHERE a
                   quant struggles (proper nouns, cross-lingual words).
  3. margins     — teacher-forced  log P(good | ctx) - log P(bad | ctx)  for a
                   set of probes (Elma vs Elna, "yeux" vs "eyes", "Une" vs
                   "Una"). The SHARP, corruption-specific instrument: does
                   raising embed_tokens + lm_head to 8-bit restore the
                   bilingual logit margin that Q6 narrows? margin>0 = good wins;
                   a WIDER positive margin under mixed/Q8 than Q6 confirms the
                   logit-floor hypothesis (the piece left at "medium confidence").

Same cross-entropy convention as mlx_lm.perplexity.eval_ppl. Loads its OWN copy
of the weights (mlx_lm.load) -> run on a node with the engine UNLOADED (no room
for a 2nd ~322GB load next to the serving pool).

  python measure_quality.py --model <path> [--text fr.txt] [--probes probes_fr.json] \
      [--seq-len 1024] [--top-surprise 25]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


def _register_minimax_m3():
    """Seed sys.modules['mlx_lm.models.minimax_m3'] with our vendored class the
    SAME way the runner does (patches.apply_mlx_patches -> apply_minimax_m3),
    so mlx_lm.load() resolves model_type=minimax_m3 AND applies the per-module
    quantization block from config.json (essential to read the mixed quant)."""
    if "mlx_lm.models.minimax_m3" in sys.modules:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("M3_PATCHES_PARENT"),
        os.path.expanduser("~/mlx-cluster"),          # deployed node layout
        os.path.abspath(os.path.join(here, "..", "..")),  # repo: scripts/ holds patches/
        os.getcwd(),
    ]
    for parent in candidates:
        if parent and os.path.isdir(os.path.join(parent, "patches")):
            sys.path.insert(0, parent)
            try:
                from patches import apply_mlx_patches
                apply_mlx_patches()
                return
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"[measure_quality] patch bootstrap from {parent} failed: {exc}\n")
    sys.stderr.write("[measure_quality] WARNING: minimax_m3 patch not found; "
                     "mlx_lm.load will fail for M3. Set M3_PATCHES_PARENT.\n")


def ppl_and_surprise(model, tokenizer, text, seq_len, top_n):
    """Teacher-forced PPL over `text` + the top-N most-surprising tokens."""
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        raise SystemExit("corpus too short")
    total_nll, total_n = 0.0, 0
    surprises = []  # (nll, abs_pos_of_target, target_id)
    for s in range(0, len(ids) - 1, seq_len):
        chunk = ids[s : s + seq_len + 1]
        if len(chunk) < 2:
            break
        logits = model(mx.array(chunk[:-1])[None]).astype(mx.float32)
        tgt = mx.array(chunk[1:])[None]
        nll = nn.losses.cross_entropy(logits, tgt, reduction="none")[0]
        mx.eval(nll)
        nll_np = np.asarray(nll)
        total_nll += float(nll_np.sum())
        total_n += int(nll_np.size)
        for i, v in enumerate(nll_np):
            surprises.append((float(v), s + i + 1, int(chunk[i + 1])))
    ppl = math.exp(total_nll / total_n)
    surprises.sort(reverse=True)
    rows = []
    for v, pos, tok in surprises[:top_n]:
        ctx = tokenizer.decode(ids[max(0, pos - 8) : pos])
        rows.append((round(v, 2), ctx[-40:], tokenizer.decode([tok])))
    return ppl, total_n, rows


def seq_logprob(model, tokenizer, context, cont):
    """Sum of teacher-forced log P(cont tokens | context). Robust to multi-token
    continuations + boundary re-tokenization (common-prefix aligned)."""
    ctx_ids = tokenizer.encode(context)
    full_ids = tokenizer.encode(context + cont)
    k = 0
    while k < min(len(ctx_ids), len(full_ids)) and ctx_ids[k] == full_ids[k]:
        k += 1
    logits = model(mx.array(full_ids[:-1])[None]).astype(mx.float32)[0]
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(logp)
    total, n = 0.0, 0
    for j in range(max(k, 1), len(full_ids)):
        total += float(logp[j - 1, full_ids[j]])
        n += 1
    return total, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", help="clean reference corpus (UTF-8)")
    ap.add_argument("--probes", help="JSON list of {name, context, good, bad}")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--top-surprise", type=int, default=25)
    args = ap.parse_args()

    _register_minimax_m3()
    print(f"loading {args.model} ...", flush=True)
    model, tokenizer = load(args.model)

    if args.text:
        text = open(args.text, encoding="utf-8").read()
        ppl, n, rows = ppl_and_surprise(
            model, tokenizer, text, args.seq_len, args.top_surprise
        )
        print(f"\n=== PERPLEXITY === {ppl:.4f}  (over {n} tokens)")
        print(f"=== TOP-{args.top_surprise} SURPRISE  (nll | context | next) ===")
        for v, ctx, nxt in rows:
            print(f"  {v:6.2f} | …{ctx!r} | -> {nxt!r}")

    if args.probes:
        probes = json.load(open(args.probes, encoding="utf-8"))
        print("\n=== MARGINS  log P(good|ctx) - log P(bad|ctx)  (>0 = good wins) ===")
        for p in probes:
            g, ng = seq_logprob(model, tokenizer, p["context"], p["good"])
            b, nb = seq_logprob(model, tokenizer, p["context"], p["bad"])
            margin = g - b
            flag = "OK " if margin > 0 else "!! "
            print(
                f"  {flag}{p['name']:<22} margin={margin:+7.3f}  "
                f"good={g:+7.2f}({ng}t) bad={b:+7.2f}({nb}t)  "
                f"[{p['good']!r} vs {p['bad']!r}]"
            )


if __name__ == "__main__":
    main()
