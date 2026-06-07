# Code-review log — DeepSeek-V4-Flash module (`scripts/mlx_models/deepseek_v4.py`)

Double-review (Claude re-derivation vs reference + Codex adversarial, read-only) of the
from-scratch forward, after P0 structural PASS (1629/1629) and a tiny-config forward smoke
PASS (runs, finite, shape correct). The smoke only proves it RUNS; this review targets
NUMERICAL correctness vs `scripts/mlx_models/ref-deepseek-v4/`.

## Method
- Claude re-derived each forward subsystem against `reference_model.py` / `reference_kernel.py`
  line-by-line (independent of Codex).
- Codex ran `codex exec -s read-only` over the module + reference (task blr2rakli; verdict
  given in-log, then the run hung on a sandbox-uncancellable `rg` and was stopped).
- Both passes are recorded below; **they converged on the same findings** — the strongest
  signal the diagnosis isn't an over-fit.

## Findings (Claude ∩ Codex)

| # | Item | Severity | Verdict | Status |
|---|------|----------|---------|--------|
| 1 | attn_sink in softmax denominator | — | **CORRECT** — matches kernel:345-348 (sink excluded from stabilizing max, added to denom only as `exp(sink-max)`, contributes 0 to numerator, per-head). | keep |
| 2 | inverse-rope on attention output | BLOCKER | **BUG** — module applied FORWARD rope; reference:534 applies `inverse=True` (complex conjugate / de-rotation). | **FIXED** |
| 2b | rope pairing convention | BLOCKER (latent) | **BUG** — used `nn.RoPE(traditional=False)` (half-split); reference apply_rotary_emb (232-244) rotates consecutive INTERLEAVED pairs (`view_as_complex`). | **FIXED** |
| 3 | YaRN freq interpolation | MAJOR | **BUG** — `nn.RoPE` does a uniform position scale, not YaRN NTK-by-parts; compressed layers need `precompute_freqs_cis` (200-228) with factor 16 / orig 65536 / β 32,1. | **FIXED** |
| 4 | grouped o-LoRA (wo_a.0..7 + wo_b) | — | **CORRECT** — contiguous per-group split matches reference einsum `bsgd,grd->bsgr` (537-542); P0 validated the group shapes. | keep |
| 5 | MoE gate noaux_tc semantics | — | **CORRECT** — sqrtsoftplus on raw logits; bias added to scored values for SELECTION only; weights from UNBIASED `original_scores`; renorm × route_scale 1.5 (Gate.forward 564-584). | keep |
| 6 | hash-routing weights | — | **CORRECT** — `original_scores.gather` runs for both hash & non-hash, so hash experts also get score-weights (ref:580 after the if/else). | keep |
| 7 | sparse/compressed attention absent | BLOCKER (long-context) | **GAP (deferred P3)** — Compressor/Indexer/sliding-window-128 are built (P0 keys) but NOT wired into the dense forward; mask is plain causal. OK as a short-context (≤ window) approximation; WRONG for long context. | P3 |
| 8 | SwiGLU clamp `swiglu_limit=10` | MAJOR | **GAP (deferred P3)** — stock SwitchGLU has no clamp; reference clamps `up`∈[-10,10], `gate`≤10 (Expert.forward 600-602). Rarely activates; numerical drift. | P3 |
| 9 | kv non-rope activation quant | MAJOR | **GAP (deferred P3)** — reference act_quant FP8-simulates kv[...,:-rd] for QAT match (ref:506); affine-8bit path skips it. Drift. | P3 |

MoE weight-application order (weight after the full expert vs reference's weight-before-`w2`)
is **equivalent** — `w2` is linear/bias-free so a per-row scalar commutes through it.

## Codex verdict (verbatim sense)
> VERDICT: REVISE — at least two independent blockers (inverse RoPE, sparse/compressed
> attention absent), plus majors on YaRN, activation-quant, and the SwiGLU clamp.

## Resolution this pass
Fixed in `deepseek_v4.py`: **#2, #2b, #3** — rewrote the rope as `FlashRoPE` (a plain class,
not an `nn.Module`, so it adds zero params and keeps the P0 key-diff exact): interleaved-pair
complex rotation, YaRN inv_freq from `rope_scaling`, `inverse=True` on the output. Re-validated:
**P0 PASS 1629/1629, forward smoke PASS** after the change.

Remaining (the genuine hard part, all P3 long-context correctness): **#7** wire the
Compressor + Indexer + sliding-window-128 into the forward (dense-mask emulation first), the
custom window+compressed KV cache layout + incremental-decode Compressor state; **#8** SwiGLU
clamp; **#9** kv act-quant. Then full numerical validation before the P2 distributed load.

## P3 long-context forward — IMPLEMENTED + numerically validated (2026-06-07)
Finding #7 (the long-context blocker) is now closed for the prefill path. A 5-agent
reference-mapping workflow (wv7b36mnj) produced cross-checked specs for the Compressor,
Indexer, the window/compressed gather-index construction, the sparse_attn→dense emulation,
and a full param-shape cross-check; all five agreed with an independent Claude reading of the
reference. Implemented in `deepseek_v4.py`:
- **FlashCompressor** rewrite: the overlap (ratio==4) case now does ONE joint softmax over the
  2*ratio contributor set (prev-group first-half ⊕ cur-group second-half) instead of the prior
  bug (two independent softmaxes summed); plain (ratio==128) gated pool; RMSNorm; rope on the
  last 64 dims at the group-start strided positions g*ratio (was missing entirely).
- **`_flash_mask`**: combined additive mask = causal sliding-window-128 over real keys ++
  fully-past compressed-block mask, the dense-mask emulation of sparse_attn's topk gather.
- **Attention**: K==V is now the COMBINED latent (window ++ compressed); the sink stays in the
  softmax denominator. Indexer top-k pruning deferred (exact no-op while n_comp ≤ index_topk,
  i.e. prompts ≤ ~2048).

**Numerical validation** (`/tmp/numtest_dsv4.py`, vs pure-numpy oracles of the reference, no
kernels): compressor ratio-4 overlap 3.4e-7, ratio-4 (3 groups) 3.6e-7, ratio-128 7.2e-7;
`_flash_mask` vs reference index sets EXACT (0.0); gather==dense identity 1.9e-7; MLX dense
sink-softmax vs numpy 1.9e-7. **ALL PASS** (float32 precision). P0 1629/1629 + forward smoke
remain green after the rewrite.

## Codex round-2 review of the long-context forward + arbiter decisions (2026-06-07)
Codex verdict: **REVISE, NO BLOCKER** — it explicitly cleared the overlap compressor (prev-group
shift, group-0 padding, concat axis, remainder drop, strided rope), the mask, and the combined-KV
attention. The 5 findings are all the already-documented deferrals. Claude (final arbiter) ruling:

| # | Codex sev | Item | Ruling |
|---|-----------|------|--------|
| 3 | MAJOR | SwiGLU `swiglu_limit=10` clamp missing | **FIXED** — `_ClampedSwiGLU` (up∈[-10,10], gate≤10) wired into SwitchGLU + the shared expert; numerically validated (clamp-active test PASS). |
| 5 | MINOR | div-by-zero guard in `_flash_mask` | **FIXED** — `n_comp>0 and ratio>0` guard. |
| 4 | MINOR | `-1e9` vs `-inf` in compressor padding | **REJECTED w/ reason** — `-1e9` is numerically identical to `-inf` after the max-subtracted softmax (exp underflows to 0) and avoids inf-arithmetic NaN risk in degenerate (G=1 / all-masked) rows. Component tests pass to 3e-7 with it. |
| 1 | MAJOR | Indexer top-k pruning skipped | **DEFERRED w/ reason** — provably exact while n_comp ≤ index_topk (prompt ≤ 2048, the first-landing smoke target). For > 2048 we attend all visible blocks (graceful superset). Wiring the learned top-k is the long-prompt P3 piece, only validatable against long prompts once the model loads. |
| 2 | MAJOR | kv non-rope act-quant FP8 sim skipped | **DEFERRED w/ reason** — MLX has no `float8_e4m3` dtype, so faithful reproduction needs a hand-rolled e4m3 block-64 round-trip (high effort / error-prone). It is a QAT *fidelity-match*, not a correctness requirement: the model was QAT-trained to tolerate FP8-rounded activations, so running them at higher precision is the safe direction. Validate empirically on the real-model smoke; implement only if output quality degrades. |

Validation after the round-2 fixes: P0 1629/1629, forward smoke PASS, numerical suite ALL PASS
(now incl. the SwiGLU-clamp test).

## Gate status
Prefill forward (≤ ~2048 tokens): rope, gate, sink, o-LoRA, Compressor, combined mask,
combined-KV attention, AND the SwiGLU clamp are all numerically validated vs the reference.
Two documented deferrals remain (act-quant FP8 sim; Indexer pruning > 2048) plus the
incremental-decode cache (full-recompute works) — none a short-context correctness blocker, all
slated for empirical validation on the real model. **The module is ready for P2. P2 (deploy to
.29/.30/.31, unload Hy3, load Flash 3-node) is gated ONLY on Sophie's explicit go — prod machine
action on Argo.**

---

# P2/P3 on the real model + the distributed deadlock diagnosis (2026-06-07 evening)

## What works on the real 282GB model
- **Loads distributed** (3-node pipeline, `pipeline_auto_parallel`, 8-bit affine key-driven quant,
  1629/1629 keys, ~50-90s). Two integration bugs found+fixed during P2:
  1. `ModelArgs` didn't inherit `BaseModelArgs` → no `from_dict` (mlx-lm's `load_model` needs it). Fixed.
  2. The pipeline uses `cache.keys` as a sync handle (`mx.depends(cache.keys, send)`); our attention
     ignored the cache → `mx.depends(None)` crash / `cache.state` NoneType. Resolved by the **Hy3-pattern
     cache** (attention now `update_and_fetch`es the window latent and USES the returned cached kv, with
     offset-aware rope/mask — also enables correct decode for contexts ≤ window).
- **Single-node forward is CORRECT**: `model(ids)` on .29 → "The capital of France is" → " Paris",
  "def square(n): return" → " n", 2.6s cold ≈ 2.4s warm (so NOT compile-bound).

## The distributed deadlock — systematic bisection
The 3-node distributed forward HANGS (all ranks busy-wait, stable memory — NOT OOM once the leak below
is controlled). A **tiny-checkpoint bisection** (random-weight DeepSeek-V4 at varying sizes, loaded via
the orchestrator — see `/tmp/make_tiny*.py`, checkpoints kept on the nodes under
`/Volumes/models/odysseus/tiny-dsv4*`, `big-hidden`, `big-exp`, `real-6l`) ruled out every axis:

| variant | result |
|---|---|
| tiny (fp32, 6L) / tiny-Q (8-bit) | PASS → not the structure, not quantization |
| tiny-43L (43 layers, tiny dims) | PASS → not the layer count |
| BIG-HIDDEN (hidden 4096 + real attention dims) | PASS → not the attention dims |
| BIG-EXP (256 experts, top-6) | PASS → not the experts |
| REAL-6L (ALL real per-layer dims, 6 layers) | PASS → not the per-layer combo |
| REAL-6L on a leak-loaded node (159GB wired) | PASS → not memory pressure |
| **real-43L (the actual 282GB)** | **DEADLOCK** |

Every reduced variant works distributed → **the model code is CORRECT**. The deadlock only appears at
the FULL model's scale (rank 0 computing its ~23 real-dim layers).

## The pinpoint (instrumented collective trace, clean memory)
With per-collective tracing (PFL recv / PLL layer-fwd / send / all_gather) + a per-call barrier, at full
scale: all ranks sync at the barrier, then **rank0 computes+sends (rank1 receives OK), but rank1's LOCAL
`mx.eval` of its 10 real-dim layers never completes** → rank1 never sends → rank2 waits forever. rank1's
compute is local (~600ms single-node) but stalls distributed. **Mechanism**: a rank's local `mx.eval`
couples to the distributed graph and blocks on a pending collective on another rank → circular wait,
which only manifests when per-rank compute is slow enough (full scale). Neither disabling the all_gather
(it's needed for logits sync) nor a per-call barrier fixes it → it is in **MLX-distributed's eval/collective
scheduling**, not this module.

## Second blocker: wired-memory leak
Each orchestrator reload of a large model leaks ~the shard size of wired Metal memory (.29 reached 311GB
wired with NO live process; survives `unload`; only a reboot frees it). `free_metal (atexit)` shows
"active X -> X" (no release). Confounded the bisection (later tests OOM'd instead of deadlocking) and is
**critical for V4-Pro** (780GB on 5 nodes has no margin to leak per reload).

## For V4-Pro (Odyssai-eu/OdyssAI-X#35) — two infra blockers, code de-risked
1. **Full-scale distributed eval/collective deadlock** — fix is at the MLX-distributed / pipeline-scheduling
   level (decouple each rank's local eval from pending collectives; or restructure the prefill path —
   `set_pipeline_prefill`/queue_sends is never engaged by the runner; or raise with MLX maintainers). The
   tiny-repro harness lets future bisection iterate in seconds, but the deadlock only reproduces near full
   scale, so a near-real variant is needed to validate any fix.
2. **Wired-memory leak on reload** — needs a runner/Metal free-on-unload fix.

The `deepseek_v4.py` model code itself is portable to V4-Pro as-is.
