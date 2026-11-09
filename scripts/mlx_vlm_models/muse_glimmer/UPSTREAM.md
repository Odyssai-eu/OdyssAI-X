# muse_glimmer (vendored)

Meta Muse-Glimmer-30B (`muse_glimmer`), text + vision, as served by
`mlx_vlm.server` with its DFlash drafter Muse-Glimmer-30B-assistant
(`muse_glimmer_assistant`, shipped by the pinned mlx-vlm itself).

Origin: rewritten on .29 on 2026-11-09 (work dir `~/glimmer-30b`) over the
mlx-vlm package of the pin below, then checked against HF transformers bf16
on the real checkpoint: last-token logits corr 0.99998 (text) / 0.99997
(image), argmax identical. `processing_muse_glimmer.py` is the pin's file,
unchanged. MIT, copyright mlx-vlm authors for the parts kept from it.

It is NOT interchangeable with the pin's own `muse_glimmer`:

- the four sandwich norms per layer are plain `nn.RMSNorm`; `sanitize` adds
  1.0 to the HF weights (stored centered at 0) once, and detects an already
  shifted MLX artifact (mean |w| >= 0.5) so it does not shift twice. The pin's
  module applies (1 + w) at runtime instead. Artifacts converted with this
  module (`odyssai/Muse-Glimmer-30B-Q9h16-gs64`) hold the shifted norms and
  must be served with it;
- the target-side speculative hooks are taken from the pin
  (`capture_layer_ids` / `hidden_states` in `LanguageModelOutput`,
  `chunked_prefill_policy`, `rollback_speculative_cache`,
  `requires_uniform_batch_acceptance`). Without them the DFlash drafter
  fails on `mx.concatenate(None)` at the first round.

DFlash check on .29 (Q9h16 target, bf16 assistant, greedy, 400 tokens):
20.6 -> 40.9 tok/s (French prose), 20.7 -> 51.9 tok/s (Python). Against a
full prefill of the same tokens, plain decoding disagrees on 3.1 % of
positions and DFlash on 4.2 % (903 tokens, 3 prompts), the same order: the
target's own decode already drifts from its prefill (max |dlogit| 3.1, same
with a plain KVCache), so greedy DFlash output is not bit-identical to plain
greedy output.

overlays-pin: 87020830d4dde238f111ade6d592f204a89cd527
