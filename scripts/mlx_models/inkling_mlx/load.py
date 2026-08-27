"""Load a (possibly quantized) Inkling MLX model produced by ``convert_model``."""

from __future__ import annotations

import glob
import json
import os

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import InklingConfig
from .model import InklingForConditionalGeneration


def quant_predicate(group_size: int, recipe: str = "uniform"):
    """Quantize exactly the modules the converter did, by delegating to
    ``convert.is_quant_target`` with the same ``recipe``. Guarantees the loaded
    module set matches the checkpoint (e.g. under ``experts_only``, attention and
    embed/unembed stay bf16 and must NOT be re-quantized here)."""
    from .convert import is_quant_target

    def pred(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        w = getattr(module, "weight", None)
        if w is None:
            return False
        return is_quant_target(path + ".weight", w.shape[-1], group_size, recipe)

    return pred


def load(path: str, lazy: bool = False):
    cfg_dict = json.load(open(os.path.join(path, "config.json")))
    config = InklingConfig.from_dict(cfg_dict)
    model = InklingForConditionalGeneration(config)

    q = cfg_dict.get("quantization")
    if q:
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                    class_predicate=quant_predicate(q["group_size"], q.get("recipe", "uniform")))

    # Stream shards: assign each, then release its handle. We do NOT eagerly
    # mx.eval() the whole parameter tree — for a ~500 GB model that builds one
    # enormous eval graph and trips a Metal resource limit. Weights stay lazy
    # (mmap-backed) and materialize on demand during the forward pass, exactly
    # like mlx-lm loads large models.
    loaded = set()
    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    for shard in shards:
        w = mx.load(shard)
        model.load_weights(list(w.items()), strict=False)
        if not lazy:
            # materialize THIS shard's tensors now (bounded graph) and keep them
            # resident. Avoids one enormous eval over all ~500 GB of params, which
            # trips a Metal resource limit; also prevents per-token disk paging.
            mx.eval(list(w.values()))
        loaded.update(w.keys())
        del w

    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing = expected - loaded
    if missing:
        raise ValueError(f"{len(missing)} params not found in checkpoint, e.g. {sorted(missing)[:3]}")

    model.eval()
    return model, config
