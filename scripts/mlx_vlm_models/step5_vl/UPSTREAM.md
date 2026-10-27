# step5_vl — vendored from mlx-vlm

Source: Blaizzy/mlx-vlm PR #2539 "Add Step 5 Preview vision-language model support",
merge commit `6e47301d184659cccde9176a90d38d818743d081` (2026-10-21). The PR was merged
into the `pc/moe-offload-nvfp4` branch after that branch had itself been merged into
`main`, so the model never reached `main` nor any release (checked against v0.7.10,
2026-10-27). License: MIT, Copyright (c) 2025 Prince Canuma (mlx-vlm).

Files: `__init__.py`, `config.py`, `language.py`, `step5_vl.py` from that commit. It reuses
`step3p7` (vision tower, processor, attention) as shipped by our mlx-vlm 0.7.3 pin; the
only step3p7 differences at 6e47301 are a `stop_gradient` in the router and a rotate-half
helper moved to `rope_utils`, neither of which changes inference.

Local changes (OdyssAI-X):
- `config.py`: `ModelConfig.norms_folded` (default False).
- `__init__.py`: registers `step5_vl` as `IMAGE_PATCH_TOKEN` in `prompt_utils.MODEL_CONFIG` (the PR
  edits prompt_utils.py itself).
- `step5_vl.py` `Model.sanitize`: router `mlp.gate.weight` renamed to `mlp.gate.gate.weight`,
  `mtp.*` dropped, and with `norms_folded` the +1 is applied to the sparse indexer's Q norm
  only. This is the layout of the mlxconvert checkpoint `Step-5-Preview-VLM-Q8h16`
  (recipe step5_vlm_q8_h16), checked tensor by tensor against the BF16 source.

Installed into the node's mlx-vlm venv by `scripts/install-mlx-vlm.sh` (step 8). The
tensor split for the distributed runner lives in `scripts/vlm_runner.py`
(`_shard_lm_step5_tensor`): the port has no shard().
