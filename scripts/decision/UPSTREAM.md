# Vendored decision readout

| File | Source | Upstream SHA-256 (from the model's `SHA256SUMS`) | License |
|---|---|---|---|
| `decision_core.py` | `caiovicentino1/Eikos-27B` on Hugging Face, fetched 2026-09-27 | `9ff8d754ce99c6539fc7f5bd88c10b196357c70124f59ed3d1bd1a0d3fdbb7d0` | MIT (`LICENSE-eikos`) |
| `mlx_decide.py` | same | `b6b54867eaa2a9faf8d7f2e1d66fb2f720b11eceab1baf3a18dc23f9d5eb8564` | MIT (`LICENSE-eikos`) |

Both files are byte-identical to upstream. They carry the prompt format (SemIf), the
single-token option labels, the calibration formula and the MLX letter-logit readout, so a
decision model answers here exactly as it was trained and evaluated.

Why vendored instead of executed from the model folder: loading "any decision model"
must never run Python shipped inside a downloaded Hugging Face folder (the same risk as
`trust_remote_code`). The model folder only provides weights, `decision_config.json` and
`calib.json`; `decision_serve.py` refuses a `prompt_version` / `readout` it does not know.

To update: copy the new files from a model folder, check them against that folder's
`SHA256SUMS`, read the diff, update this table.

## Julia runtime (`julia/`)

Pure PyTorch path of Supersonic Labs' Julia decision models (mmBERT encoder + marker
head), from `SupersonicLabs/Julia-1` on Hugging Face at revision
`a85b127321d580d65176c89ced8273f305745d85`, fetched 2026-09-28. License: Apache-2.0
(declared in the model card; the repository ships no LICENSE file, so the standard text
is included as `julia/LICENSE-APACHE-2.0.txt`). Files are byte-identical to upstream:

| File | SHA-256 |
|---|---|
| `julia/__init__.py` | `02485eb4dd7dadd13907380d1a9c3398709bd8af54b3e0e12b3061a3ebb5c1da` |
| `julia/inference.py` | `79b4e716365a6e07da4580ead725d5b76d06ef3a824a3ec23738feba64704b36` |
| `julia/model.py` | `ef2ba82fe20cdf0db7bb887e9ef075476ed08b985ce9a95be0de3e26246ecc81` |
| `julia/data.py` | `e3510fa4152ec11fa193046715991f44d7c2f85fd2488a98ef11c9d3db23da4e` |
| `julia/cuda.py` | `138ab63182f25473ce3886296c47ac44e94fc50ae5270927ae4c373f7bb49137` |
| `julia/probabilities.py` | `2a0197d2e0fa5a3c4b06b93599a85706724df7b0d2cc821a13ed293aef59f206` |
| `julia/typed.py` | `ed89e66a70fcedac1339347bd8fcca69fd2cd1a31535ad0148dae42c610b544d` |

Left out on purpose: `julia/router/` (Bend trees + a C bridge compiled at runtime,
only used by `load_model`'s FastEngine). `decision_serve.py` uses
`julia.inference.TransformerEngine` directly: local tokenizer and encoder config,
`trust_remote_code=False`, no network. Weights are checked by the server against the
`weights_sha256` of the model's `inference-policy.json` before loading.

## CLM (`clm/`)

`schema.py` (question → state / candidate texts, answer assembly) and `heads.py`
(projection-head architecture and checkpoint loading) from
`github.com/Contrastive-LM/CLM` `src/clm/` at commit
`bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`, fetched 2026-09-28, Apache-2.0 (`clm/LICENSE`).
Byte-identical:

| File | SHA-256 |
|---|---|
| `clm/schema.py` | `52cec58afbf49ad7b7aa6bdb7e7476ee42bf3fd7a2703d44319dc4b565987335` |
| `clm/heads.py` | `3f3b880e940a47b45879614b140fd873f7de9b13ccb8254b07989af7ea92e093` |

`clm/__init__.py` is ours and imports nothing (upstream's pulls in their HTTP client
and server). Never called: `heads.download()` / `_count_download()` (network).
`heads.py` loads the checkpoint with `torch.load()` without `weights_only`;
`decision_serve.py` refuses torch < 2.6, where that default is not yet `True`.
The encoder is not vendored: Qwen3-8B (the official `Qwen/Qwen3-8B` bf16 weights)
runs on MLX, last-token pooled and L2-normalised like their vLLM pooling server.

## Clef (`clef/`)

`clef_mlx.py`, the MLX port of Cloudflare Clef / Clef-Flash (Qwen3.5 backbone through
mlx-vlm, joint schema head in MLX), from `mlx-community/clef-8bit` at revision
`1c8887c20827d91cb1b0c6f186e24f5bc5b0e0ce` and `mlx-community/clef-flash-8bit` at revision
`5a195dc147926b63b1d139691ed4480ca9f92c0e` (the two copies are identical), fetched
2026-10-14. License: Apache-2.0 (`clef/LICENSE`, the repository's own file). Byte-identical:

| File | SHA-256 |
|---|---|
| `clef/clef_mlx.py` | `852223c944819a32fad5cf798d9d1dff30419820eaf5ad1f10cb9698afec97d5` |

`clef/__init__.py` is ours and imports nothing. The model folder carries its own copy of
`clef_mlx.py`; it is never imported. Used: `load()`, `ClefMLX.logits()` (the encoding,
backbone and joint head); never called: `snapshot_download` (a local folder is always
passed), the CLI and its HTTP server. Clef ships no format declaration: the operator writes
`decision_config.json` = `{"prompt_version": "clef-v1", "readout": "clef-joint-head"}`.

## d3 (`d3/`)

The vLLM Semantic Router d3 runtime (Decision 3.0), from `vllm-sr/d3-ultra` on Hugging Face at
revision `ccdcfa611649a59e095e0b5e1732a15cedafadd6`, fetched 2026-11-02. License: Apache-2.0
(`d3/LICENSE-APACHE-2.0.txt` and `d3/NOTICE`, the repository's own files). Byte-identical,
SHA-256 as listed in that revision's `MODEL_MANIFEST.json`:

| File | SHA-256 |
|---|---|
| `d3/d3_runtime.py` | `810f88c0cf538680163a285516818125f3b1f3a04a709046c691e877e00e7118` |
| `d3/d3_format.py` | `5f3da74306e66224fe414128fa8307d73f58af94ff1eeebacb9825f728d50720` |
| `d3/d3_compose.py` | `d0bab5e6f400a198d39f4adae3b83a5ef06b64485f968aeca15d65be777be39e` |

The d3-ultra runtime is the newest of the family and reads every d3 package's
`decision_config.json` (the smaller sizes ship an older runtime): on the same requests it
renders the same token ids as each package's own runtime (checked for d3-lite, -nano, -mini,
-flash). Only its torch-free parts run here: `D3.prepare / run / respond`, the prompt contract
(`d3_format`), the temperature table and the permutation average. `d3/d3_mlx.py` and
`d3/__init__.py` are ours: `D3MLX` loads the backbone converted by mlx-vlm (no torch) and
replaces `probabilities` with the text layers on MLX under the package's `attention_mode`.
Never imported: `d3_fast.py` / `d3_kernels.py` (CUDA / ROCm kernels), and the model folder's
own copies of these files. The package declares its format itself:
`decision_config.json` `format_id: d3-code-readout-v1`.

The MLX folder is made by `convert_d3.py` (ours, run once per model, not installed on the nodes):
mlx-vlm conversion with the keys remapped and a strict load, plus the (1 + w) norm scales stored exact in
FP32; `D3MLX` refuses a folder whose norm scales are not FP32. Fidelity, measured against each package's
own runtime on CPU (48 questions, `bench/`): same token ids, and against the
same checkpoint in FP32, MLX bf16 stays within the order of upstream's own bf16 runtime (max |dp| 1.2-1.7e-2
vs 0.7-1.4e-2, same argmax).

## Vela 3.0 (`vela3/`)

vLLM Semantic Router Vela 3.0 (`vela3-lite` / `-mini` / `-flash` / `-turbo`): a d3 base at a pinned revision,
unchanged, plus gated LoRA adapters, span heads and label definitions. Upstream's code is torch from its first
line, so nothing of it is imported; its pure parts are **extracted verbatim** (no edit) by `vela3/extract.sh`
from `vela3-lite` (fetched 2026-11-02; `vela3_span.py`, `span_decode_v2.py`, `vela3_adapters.py` and
`vela3_labeldef.py` are byte-identical in the four sizes, `modeling_vela3.py` of lite is the superset of
mini / flash for these lines and differs from turbo's only by a comment there). License: Apache-2.0
(`vela3/LICENSE`, the package's own file).

| File | Source | SHA-256 of the source file |
|---|---|---|
| `vela3/vela3_methods.py` | `modeling_vela3.py` lines 186-401 (router gate, span routing, thresholds, `span_answer`, evidence decodes) | `c5c0c4ee06d34780b7e284f5467047fdee921793e428b330e1fed56371468592` |
| `vela3/vela3_render.py` | `vela3_span.py` line 24, lines 71-155 (word units, windows, span prompt) | `97d2448765a7ee5411dff44c601c3a073fdad9a9921f9818bac4135916081c20` |
| `vela3/vela3_labels.py` | `vela3_labeldef.py` lines 19-28, 58-61, 129-144 (definition prompt, cache key, `define`) | `b7fcefdd1a0b3e8ac22ac12338c72e76925688a989b5af6c70a104fa70122a39` |
| `vela3/span_decode_v2.py` | `span_decode_v2.py`, byte-identical | `d2c5f2f446cf12d2f8a24dbf634452477088523b0973e1e07df244c851421efb` |

Each generated file names its source and that source's SHA-256 in its docstring. `vela3/vela3_mlx.py` and
`vela3/__init__.py` are ours: the adapters (upstream's MultiLoRA math, one adapter per forward), `SpanHeadV2`,
the span pass, the definition writer (writer `base` / `grafted_head`; `labeldef_writer`, another model, is not
available: names only, as upstream does when its writer fails) and `decide` (lite / mini / flash, and turbo's
`decisions.attention` extra pass), on the d3 MLX runtime above. A Vela folder is the d3 base converted by
`convert_d3.py` plus the package's adapters, heads, `vela3_config.json` (`format: vela3/3`),
`label_definitions.json` and `router_questions.json`; the package's own Python files are never read.

## d1 (no vendored file)

LiquidAI d1 runs from the mlx-vlm package itself (`mlx_vlm.models.d1`), pinned in the
`mlxvlm` venv by `install-decision.sh`: d1 subclasses mlx-vlm's LFM2-VL model, so a copy
would mean vendoring that whole architecture. The package is installed from PyPI at an
exact version; nothing is read from the model folder but weights, tokenizer and configs.
d1 ships no format declaration: the operator writes `decision_config.json` =
`{"prompt_version": "d1-v1", "readout": "d1-answer-slot"}`.
