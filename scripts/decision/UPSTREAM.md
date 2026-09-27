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
