"""Loader smoke for the qwen3 MTP family — coverage + a real draft_step forward
on a stub trunk (no 50GB trunk load). Mirrors mtp_loader_smoke.py but builds the
qwen3_5 DecoderLayer at a full-attention layer_idx and resolves the trunk args
from text_config.

Usage: python3 mtp_loader_smoke_qwen.py <trunk_dir> [--full]
  expects <trunk_dir>/mtp-sidecar/mtp-sidecar.safetensors
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).parent))
from mtp_module import (MTPSpec, NativeMTPModule, _family_layer_cls,
                        _family_mtp_layer_idx, _load_source_weights,
                        _rewrite_weights)


def main() -> None:
    trunk = Path(sys.argv[1])
    config = json.load(open(trunk / "config.json"))
    tcfg = config.get("text_config") or config

    from mlx_lm.models.qwen3_5 import TextModelArgs
    args = TextModelArgs.from_dict(tcfg)
    n_layers = int(tcfg["num_hidden_layers"])
    hs = int(tcfg["hidden_size"])
    vocab = int(tcfg["vocab_size"])

    sidecar = trunk / "mtp-sidecar" / "mtp-sidecar.safetensors"
    spec = MTPSpec("qwen3", n_layers, 1, "sidecar", sidecar)
    layer_cls = _family_layer_cls("qwen3")
    layer_idx = _family_mtp_layer_idx("qwen3", args, n_layers)
    print(f"trunk: {n_layers} layers, hs={hs}, vocab={vocab}, "
          f"mtp layer_idx={layer_idx} (full_attn_interval={getattr(args,'full_attention_interval',None)})")

    module = NativeMTPModule(args, layer_cls, layer_idx, family="qwen3")
    params = dict(tree_flatten(module.parameters()))
    raw = _load_source_weights(spec, trunk)
    mapped = _rewrite_weights(raw, spec, args)
    print(f"module params: {len(params)} | sidecar read: {len(raw)} | mapped: {len(mapped)}")

    mk, xk = set(params), set(mapped)
    missing = sorted(mk - xk)
    orphans = sorted(xk - mk)
    mismatch = sorted(k for k in (mk & xk)
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

    # --full: bind a stub trunk (embed/lm_head only) + one draft_step forward.
    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, hs)

    class _LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.args = args
            self.model = _Inner()
            self.lm_head = nn.Linear(hs, vocab, bias=False)

    class _Trunk(nn.Module):        # qwen wrapper: text model under .language_model
        def __init__(self):
            super().__init__()
            self.language_model = _LM()

    stub = _Trunk()
    mx.eval(stub.parameters())
    module.bind_trunk(stub)
    module.load_weights(list(mapped.items()), strict=False)
    mx.eval(module.parameters())
    gb = sum(v.nbytes for _, v in tree_flatten(module.parameters())) / 1e9
    print(f"loaded — {gb:.2f} GB resident")

    cache = module.make_cache()
    h = mx.random.normal((1, 1, hs))
    tok = mx.array([[42]], dtype=mx.uint32)
    t0 = time.time()
    logits, hid = module.draft_step(tok, h, cache)
    mx.eval(logits, hid)
    print(f"draft_step: logits {tuple(logits.shape)} hidden {tuple(hid.shape)} "
          f"{(time.time()-t0)*1000:.0f} ms")
    assert tuple(logits.shape) == (1, 1, vocab), "logit shape wrong"
    assert mx.isfinite(logits).all().item(), "non-finite logits"
    print("FULL SMOKE: PASS")


if __name__ == "__main__":
    main()
