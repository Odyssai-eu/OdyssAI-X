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
