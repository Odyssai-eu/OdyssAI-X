"""Extract the qwen3_5_mtp MTP head from an original (unquantized) Qwen3.5/3.8
checkpoint into a module-layout sidecar the NativeMTPModule can load.

The trunk conversion (mlx_lm.convert → qwen3_5.sanitize) DROPS the `mtp.*`
tensors and applies a `+1.0` shift to RMSNorm weights whenever mtp weights are
present in the source. To keep the MTP head CONSISTENT with the shifted trunk it
attaches to, this extraction replays the SAME shift on the head's norms — rather
than trusting a third-party MTP-only checkpoint whose convention is unknown.

Shifted norm suffixes (mirrors qwen3_5.TextModel.sanitize):
  .input_layernorm.weight, .post_attention_layernorm.weight,
  .q_norm.weight, .k_norm.weight, and the head's own `norm.weight`.

Output keys (module layout, "mtp." prefix stripped): fc.*, norm.*,
pre_fc_norm_embedding.*, pre_fc_norm_hidden.*, layers.0.* — the same shape the
published MTP-8bit sidecar ships, so _load_source_weights/_rewrite_weights (qwen
family) map them onto enorm/hnorm/eh_proj/shared_head_norm/mtp_block.

Usage: python3 extract_qwen_mtp_sidecar.py <original_dir> <out_dir>
  writes <out_dir>/mtp-sidecar/mtp-sidecar.safetensors
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import mlx.core as mx

# Norm suffixes the trunk sanitize shifts by +1 when mtp weights are present.
_SHIFT_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def main() -> None:
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) / "mtp-sidecar"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mtp-sidecar.safetensors"

    shards = sorted(glob.glob(str(src / "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {src}")

    raw: dict[str, mx.array] = {}
    for shard in shards:
        for k, v in mx.load(shard).items():
            if k.startswith("mtp."):
                raw[k] = v
    if not raw:
        raise RuntimeError(f"no mtp.* tensors in {src} — is this the MTP build?")

    shifted = 0
    result: dict[str, mx.array] = {}
    for k, v in raw.items():
        key = k[len("mtp."):]                       # strip the mtp. prefix
        # the head's own final norm (mtp.norm) shifts too, like model.norm
        is_head_norm = key == "norm.weight"
        if v.ndim == 1 and (is_head_norm or any(key.endswith(s) for s in _SHIFT_SUFFIXES)):
            v = v + 1.0
            shifted += 1
        result[key] = v

    mx.save_safetensors(str(out), result)
    print(f"[extract] {len(result)} tensors → {out}  ({shifted} norms shifted +1)")
    print("[extract] keys:", sorted(result)[:8], "...")


if __name__ == "__main__":
    main()
