"""Loader smoke: build the NativeMTPModule for a REAL model dir + sidecar,
WITHOUT loading the trunk. Audits the weight mapping coverage that
load_weights(strict=False) would silently tolerate:

  * module parameters that received NO weight  -> would run random-init
  * mapped weights that matched NO module slot -> silently dropped
  * shape mismatches per matched pair

Usage: python3 mtp_loader_smoke.py <model_dir> <sidecar_path>
Exit 0 only if coverage is exact (norms may be listed as benign).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).parent))
from mtp_module import (MTPSpec, NativeMTPModule, _load_source_weights,
                        _rewrite_weights)


def main() -> None:
    model_dir = Path(sys.argv[1])
    sidecar = Path(sys.argv[2])
    config = json.load(open(model_dir / "config.json"))

    from mlx_lm.models.glm_moe_dsa import ModelArgs
    from mlx_lm.models.deepseek_v32 import DeepseekV32DecoderLayer
    args = ModelArgs.from_dict(config)

    n_layers = int(config["num_hidden_layers"])
    spec = MTPSpec("deepseek", n_layers, 1, "sidecar",
                   sidecar if sidecar.is_file() else sidecar / "mtp-sidecar.safetensors")

    module = NativeMTPModule(args, DeepseekV32DecoderLayer, n_layers)
    params = dict(tree_flatten(module.parameters()))
    print(f"module params: {len(params)} tensors, "
          f"{sum(v.size for v in params.values())/1e9:.2f}B")

    raw = _load_source_weights(spec, model_dir)
    print(f"sidecar tensors read: {len(raw)}")
    mapped = _rewrite_weights(raw, spec, args)
    print(f"mapped after rewrite: {len(mapped)}")

    module_keys = set(params)
    mapped_keys = set(mapped)

    missing = sorted(module_keys - mapped_keys)   # param without weight
    orphans = sorted(mapped_keys - module_keys)   # weight without slot
    mismatch = sorted(k for k in (module_keys & mapped_keys)
                      if tuple(params[k].shape) != tuple(mapped[k].shape))

    print(f"\nparams WITHOUT weight ({len(missing)}):")
    for k in missing[:25]:
        print("  ", k, tuple(params[k].shape))
    print(f"weights WITHOUT slot ({len(orphans)}):")
    for k in orphans[:25]:
        print("  ", k, tuple(mapped[k].shape))
    print(f"shape mismatches ({len(mismatch)}):")
    for k in mismatch[:25]:
        print("  ", k, tuple(mapped[k].shape), "->", tuple(params[k].shape))

    ok = not missing and not orphans and not mismatch
    print("\nLOADER SMOKE:", "PASS" if ok else "FAIL")
    if not ok or "--full" not in sys.argv:
        sys.exit(0 if ok else 1)

    # ── --full: real load + optional quantize + one draft_step forward ──
    # Fake trunk stub with the right dims: exercises bind_trunk, weight
    # materialisation, nn.quantize predicate, and the forward path, without
    # the 563GB trunk. Logits are meaningless (random embed/head) — only
    # shapes/dtypes/mechanics are under test.
    import mlx.nn as nn
    import time as _t

    class _StubInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(int(config["vocab_size"]),
                                             int(config["hidden_size"]))

    class _StubTrunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.args = args
            self.model = _StubInner()
            self.lm_head = nn.Linear(int(config["hidden_size"]),
                                     int(config["vocab_size"]), bias=False)

    quantize = "--quantize" in sys.argv
    stub = _StubTrunk()
    mx.eval(stub.parameters())
    from mtp_module import _maybe_quantize
    t0 = _t.time()
    module.bind_trunk(stub)
    if quantize:
        _maybe_quantize(module, config, mapped, when="pre_load")
    module.load_weights(list(mapped.items()), strict=False)
    if quantize:
        _maybe_quantize(module, config, mapped, when="post_load")
    mx.eval(module.parameters())
    from mlx.utils import tree_flatten as _tf
    gb = sum(v.nbytes for _, v in _tf(module.parameters())) / 1e9
    print(f"loaded (quantize={quantize}) in {_t.time()-t0:.1f}s — {gb:.2f} GB resident")

    cache = module.make_cache()
    h = mx.random.normal((1, 1, int(config["hidden_size"])))
    tok = mx.array([[42]], dtype=mx.uint32)
    t0 = _t.time()
    logits, hid = module.draft_step(tok, h, cache)
    mx.eval(logits, hid)
    print(f"draft_step[1]: logits {tuple(logits.shape)}, hidden {tuple(hid.shape)}, "
          f"{(_t.time()-t0)*1000:.0f} ms (first: compile)")
    t0 = _t.time()
    for _ in range(10):
        logits, hid = module.draft_step(
            mx.array([[7]], dtype=mx.uint32), hid[:, -1:, :], cache)
        mx.eval(logits)
    print(f"draft_step x10 warm: {(_t.time()-t0)*100:.1f} ms/step")
    print("FULL SMOKE: PASS")


if __name__ == "__main__":
    main()
