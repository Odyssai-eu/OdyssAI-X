#!/usr/bin/env python3
"""Will this checkpoint load under the runner's strict load (#116)? No weight is read.

Rebuilds each tensor of the checkpoint as a lazy zero from its safetensors header
(nothing is read from disk past the headers, nothing is allocated), runs the prod
load path (patches + mlx-lm load_model) and compares what the model expects with
what the checkpoint delivers after sanitize:

  missing  model parameters no tensor fills (strict=False left them random)
  extra    checkpoint tensors with no slot in the model
  shape    same key, different shape

Run it with the node venv, next to runner.py (patches/ on sys.path):

    ~/mlx-cluster/.venv/bin/python audit_strict_load.py <model_dir> [<model_dir>…]

Exit 0 when every checkpoint matches exactly.
"""
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patches import apply_mlx_patches  # noqa: E402

apply_mlx_patches()
import mlx_lm.utils as mlx_lm_utils  # noqa: E402

DTYPES = {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32, "F64": mx.float32,
          "U32": mx.uint32, "I32": mx.int32, "U16": mx.uint16, "I16": mx.int16,
          "U8": mx.uint8, "I8": mx.int8, "I64": mx.int64, "BOOL": mx.bool_}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def lazy_load(path, *args, **kwargs):
    return {k: mx.zeros(v["shape"], DTYPES.get(v["dtype"], mx.uint8))
            for k, v in header(path).items()}


def audit(model_dir):
    seen = {}
    load_weights = nn.Module.load_weights

    def record(self, weights, strict=True):
        if isinstance(weights, list):
            seen[id(self)] = {k: tuple(v.shape) for k, v in weights}
        return load_weights(self, weights, strict=False)

    real_load = mx.load
    mx.load, nn.Module.load_weights = lazy_load, record
    try:
        model, _ = mlx_lm_utils.load_model(Path(model_dir), lazy=True, strict=False)
    finally:
        mx.load, nn.Module.load_weights = real_load, load_weights
    loaded = seen.get(id(model), {})
    params = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    missing = sorted(set(params) - set(loaded))
    extra = sorted(set(loaded) - set(params))
    shape = sorted(k for k in set(params) & set(loaded) if params[k] != loaded[k])

    def patterns(keys):
        return Counter(re.sub(r"\.\d+\.", ".N.", k) for k in keys).most_common(8)

    return {"model": str(model_dir), "params": len(params), "missing": len(missing),
            "extra": len(extra), "shape": len(shape), "missing_keys": patterns(missing),
            "extra_keys": patterns(extra), "shape_keys": shape[:5]}


def main(dirs):
    # The qwen4_exp n-gram seed guard compares tensor VALUES with rebuilt ones;
    # lazy zeros would always trip it, and keys/shapes are what is audited here.
    try:
        import mlx_lm.models.qwen4_exp as q4
        for c in vars(q4).values():
            if isinstance(c, type) and hasattr(c, "_check_ngram_constant"):
                c._check_ngram_constant = lambda self, k, v: None
    except Exception:
        pass
    ok = True
    for d in dirs:
        try:
            r = audit(d)
        except Exception as e:
            print(json.dumps({"model": d, "error": f"{type(e).__name__}: {str(e)[:400]}"}))
            ok = False
            continue
        print(json.dumps(r))
        ok = ok and not (r["missing"] or r["extra"] or r["shape"])
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
