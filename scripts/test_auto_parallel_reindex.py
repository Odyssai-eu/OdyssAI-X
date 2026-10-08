#!/usr/bin/env python3
"""Golden tests for the per-family shard reindexing in pipeline_auto_parallel.

These pin TODAY's behaviour (characterisation, not a specification): each case
feeds stub layers to pipeline_auto_parallel and checks the indices it writes
on the model. Expected values were derived by hand from the rules in
auto_parallel.py; if the code disagrees, that is a finding to report, never a
reason to change an expectation.

No mlx group, no node, no network. Needs the mlx-cluster venv (mlx_lm 0.31+):

    /Users/sophie/mlx-cluster/.venv/bin/python scripts/test_auto_parallel_reindex.py
"""
import os
import re
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.nn as nn  # noqa: E402

import auto_parallel as ap  # noqa: E402

F, T = False, True
S, FU = "sliding_attention", "full_attention"


class Outer(nn.Module):
    """External model: exposes .model. Must NOT define make_cache in its class."""


class Inner(nn.Module):
    """Plain internal model for the duck-typed families."""


class StubGptOss(nn.Module):
    pass


class StubMimo(nn.Module):
    pass


class StubStep35(nn.Module):
    pass


class StubQwen(nn.Module):
    pass


class StubNemo(nn.Module):
    pass


def L(**attrs):
    return SimpleNamespace(**attrs)


def layers8(attr, values):
    """8 GLOBAL layers; the pipeline slices [2:6] (the local rank)."""
    assert len(values) == 8
    return [L(**{attr: v}) for v in values]


def outer_of(inner):
    outer = Outer()
    outer.model = inner
    return outer


def run(outer, start=2, end=6):
    meta = SimpleNamespace(start_layer=start, end_layer=end, device_rank=0, world_size=2)
    gen = ap.pipeline_auto_parallel(outer, mock.sentinel.group, meta)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


class ReindexBase(unittest.TestCase):
    def setUp(self):
        self.calls = []
        for target, name, value in (
            (ap.mx, "eval", lambda *a, **k: None),
            (ap, "PipelineFirstLayer", lambda layer, *a, **k: layer),
            (ap, "PipelineLastLayer", lambda layer, *a, **k: layer),
            (ap, "patch_pipeline_model", lambda model, group: model),
            (ap, "_patch_hybrid_cache", self._record_patch),
        ):
            p = mock.patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)

    def _record_patch(self, model, **kw):
        self.calls.append(dict(model=model, **kw))

    def use_class(self, name, cls):
        p = mock.patch.object(ap, name, cls)
        p.start()
        self.addCleanup(p.stop)


# ---- GptOss (layer_types list, sliced by the pipeline) ----------------------

class GptOss(ReindexBase):
    def test_G1_mixed(self):
        self.use_class("GptOssMoeModel", StubGptOss)
        inner = StubGptOss()
        inner.layer_types = [S, S, FU, S, S, FU, S, FU]
        inner.layers = [L() for _ in range(8)]
        out = run(outer_of(inner))
        self.assertEqual(inner.layer_types, [FU, S, S, FU])
        self.assertEqual(inner.swa_idx, 1)
        self.assertEqual(inner.ga_idx, 0)
        self.assertIs(out, out)

    def test_G1b_no_full_attention_falls_back_to_zero(self):
        self.use_class("GptOssMoeModel", StubGptOss)
        inner = StubGptOss()
        inner.layer_types = [FU, FU, S, S, S, S, FU, FU]
        inner.layers = [L() for _ in range(8)]
        run(outer_of(inner))
        self.assertEqual(inner.layer_types, [S, S, S, S])
        self.assertEqual(inner.swa_idx, 0)
        self.assertEqual(inner.ga_idx, 0)


# ---- MiMo (is_sliding_window, class patched in the tuple) -------------------

class MiMo(ReindexBase):
    def test_G2_mixed(self):
        self.setUp_mimo()
        inner = StubMimo()
        inner.layers = layers8("is_sliding_window", [T, T, F, T, T, F, T, F])
        run(outer_of(inner))
        self.assertEqual(inner.swa_idx, 1)
        self.assertEqual(inner.ga_idx, 0)

    def test_G2b_no_full_layer_falls_back_to_zero(self):
        self.setUp_mimo()
        inner = StubMimo()
        inner.layers = layers8("is_sliding_window", [F, F, T, T, T, T, F, F])
        run(outer_of(inner))
        self.assertEqual(inner.swa_idx, 0)
        self.assertEqual(inner.ga_idx, 0)

    def setUp_mimo(self):
        p = mock.patch.object(ap, "_MIMO_V2_INNER_CLASSES", (StubMimo,))
        p.start()
        self.addCleanup(p.stop)


# ---- Step35 (is_sliding) ---------------------------------------------------

class Step35(ReindexBase):
    def test_G3_mixed(self):
        self.use_class("Step35InnerModel", StubStep35)
        inner = StubStep35()
        inner.layers = layers8("is_sliding", [T, T, F, T, T, F, T, F])
        run(outer_of(inner))
        self.assertEqual(inner._swa_idx, 1)
        self.assertEqual(inner._full_idx, 0)

    def test_G3b_no_full_layer_falls_back_to_zero(self):
        self.use_class("Step35InnerModel", StubStep35)
        inner = StubStep35()
        inner.layers = layers8("is_sliding", [F, F, T, T, T, T, F, F])
        run(outer_of(inner))
        self.assertEqual(inner._swa_idx, 0)
        self.assertEqual(inner._full_idx, 0)


# ---- Qwen3.5 / Qwen3Next (isinstance, both names -> one stub) --------------

class Qwen(ReindexBase):
    def use_qwen(self):
        self.use_class("Qwen3_5TextModelInner", StubQwen)
        self.use_class("Qwen3NextInnerModel", StubQwen)

    def test_G4_mixed(self):
        self.use_qwen()
        inner = StubQwen()
        inner.layers = layers8("is_linear", [T, T, F, T, T, F, T, F])
        run(outer_of(inner))
        self.assertEqual(inner.fa_idx, 0)
        self.assertEqual(inner.ssm_idx, 1)
        self.assertEqual(self.calls, [])

    def test_G4b_no_full_layer_patches_cache(self):
        self.use_qwen()
        inner = StubQwen()
        inner.layers = layers8("is_linear", [F, F, T, T, T, T, F, F])
        outer = outer_of(inner)
        run(outer)
        self.assertEqual(inner.fa_idx, 0)
        self.assertEqual(inner.ssm_idx, 0)
        self.assertEqual(len(self.calls), 1)
        c = self.calls[0]
        self.assertIs(c["model"], outer)
        self.assertEqual((c["fa_idx"], c["has_full_attn"], c["ssm_idx"], c["has_linear"]),
                         (0, False, 0, True))


# ---- qwen4_exp (duck-typed: ple_layers + make_cache wrap on the OUTER model) -

class Qwen4Exp(ReindexBase):
    def test_G5_mixed(self):
        inner = Inner()
        inner.ple_layers = [99]
        inner.hyper_connection_mixer = None
        inner.hc = None
        inner.layers = [L(ple=None) if i != 3 else L(ple=object()) for i in range(8)]
        outer = outer_of(inner)
        outer.make_cache = lambda: list(range(8))
        run(outer)
        self.assertEqual(inner.ple_layers, [1])
        self.assertEqual(outer.make_cache(), [2, 3, 4, 5])


# ---- bailing_moe_linear (duck-typed on attn_idx + gla_idx + is_global) -------

class Bailing(ReindexBase):
    def test_G6_mixed(self):
        inner = Inner()
        inner.attn_idx = -1
        inner.gla_idx = -1
        inner.layers = layers8("is_global", [F, F, T, F, F, T, F, T])
        run(outer_of(inner))
        self.assertEqual(inner.attn_idx, 0)
        self.assertEqual(inner.gla_idx, 1)

    def test_G6b_no_full_layer_raises(self):
        inner = Inner()
        inner.attn_idx = -1
        inner.gla_idx = -1
        inner.layers = layers8("is_global", [T, T, F, F, F, F, T, T])
        with self.assertRaisesRegex(ValueError, re.escape(
                "bailing_moe_linear pipeline shard [2,6) contains no full-attention layer")):
            run(outer_of(inner))


# ---- kimi (ssm_idx + attn_idx, no gla_idx) ----------------------------------

class Kimi(ReindexBase):
    def test_G7_mixed(self):
        inner = Inner()
        inner.ssm_idx = -1
        inner.attn_idx = -1
        inner.layers = layers8("is_linear", [T, T, F, T, T, F, T, F])
        run(outer_of(inner))
        self.assertEqual(inner.ssm_idx, 1)
        self.assertEqual(inner.attn_idx, 0)

    def test_G7b_no_full_layer_raises(self):
        inner = Inner()
        inner.ssm_idx = -1
        inner.attn_idx = -1
        inner.layers = layers8("is_linear", [F, F, T, T, T, T, F, F])
        with self.assertRaisesRegex(ValueError, re.escape(
                "kimi pipeline shard [2,6) contains no full-attention layer")):
            run(outer_of(inner))


# ---- glm5_next (fa_idx + ssm_idx, no attn_idx, not Qwen/NemotronH) ----------

class Glm5Next(ReindexBase):
    def test_G8_mixed(self):
        inner = Inner()
        inner.fa_idx = -1
        inner.ssm_idx = -1
        inner.layers = layers8("is_linear", [T, T, F, T, T, F, T, F])
        run(outer_of(inner))
        self.assertEqual(inner.fa_idx, 0)
        self.assertEqual(inner.ssm_idx, 1)

    def test_G8b_no_full_layer_raises(self):
        inner = Inner()
        inner.fa_idx = -1
        inner.ssm_idx = -1
        inner.layers = layers8("is_linear", [F, F, T, T, T, T, F, F])
        with self.assertRaisesRegex(ValueError, re.escape(
                "hybrid pipeline shard [2,6) contains no full-attention layer")):
            run(outer_of(inner))

    def test_G8c_qwen_instance_is_excluded_from_glm5(self):
        self.use_class("Qwen3_5TextModelInner", StubQwen)
        self.use_class("Qwen3NextInnerModel", StubQwen)
        inner = StubQwen()
        inner.fa_idx = -1
        inner.ssm_idx = -1
        inner.layers = layers8("is_linear", [F, F, T, T, T, T, F, F])
        run(outer_of(inner))  # must NOT raise: the Qwen branch handles it
        self.assertEqual(inner.fa_idx, 0)
        self.assertEqual(inner.ssm_idx, 0)
        self.assertEqual(len(self.calls), 1)


# ---- NemotronH (cache-slot indices by block_type) ---------------------------

class NemotronH(ReindexBase):
    def use_nemo(self):
        self.use_class("NemotronHInnerModel", StubNemo)

    def test_G9_mixed_counts_cache_slots_not_layers(self):
        self.use_nemo()
        inner = StubNemo()
        inner.fa_idx = -1
        inner.ssm_idx = -1
        inner.layers = layers8("block_type", ["-", "E", "M", "E", "*", "-", "M", "*"])
        run(outer_of(inner))
        self.assertEqual(inner.fa_idx, 1)   # M takes slot 0, * takes slot 1 (E and - take none)
        self.assertEqual(inner.ssm_idx, 0)
        self.assertEqual(self.calls, [])

    def test_G9b_no_attention_patches_cache(self):
        self.use_nemo()
        inner = StubNemo()
        inner.fa_idx = -1
        inner.ssm_idx = -1
        inner.layers = layers8("block_type", ["*", "E", "M", "E", "M", "-", "*", "*"])
        outer = outer_of(inner)
        run(outer)
        self.assertEqual(inner.fa_idx, 0)
        self.assertEqual(inner.ssm_idx, 0)
        self.assertEqual(len(self.calls), 1)
        c = self.calls[0]
        self.assertIs(c["model"], outer)
        self.assertEqual((c["fa_idx"], c["has_full_attn"], c["ssm_idx"], c["has_linear"]),
                         (0, False, 0, True))


# ---- negative case -------------------------------------------

class Negative(ReindexBase):
    def test_X1_no_index_family_touches_nothing(self):
        inner = Inner()
        inner.layers = [L() for _ in range(8)]
        run(outer_of(inner))
        for name in ("fa_idx", "ssm_idx", "attn_idx", "gla_idx", "swa_idx", "ga_idx"):
            self.assertFalse(hasattr(inner, name), name)
        self.assertEqual(self.calls, [])



if __name__ == "__main__":
    unittest.main()
