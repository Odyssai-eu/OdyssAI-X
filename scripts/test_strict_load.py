#!/usr/bin/env python3
"""The runner refuses a checkpoint that does not fill the model (#116).

Under strict=False, a parameter no tensor fills keeps its random init and the
model serves noise without any error. `_load_model_strict` loads strict and turns
mlx's key list into one readable error. Built on a tiny real Qwen3 checkpoint.

    /Users/sophie/mlx-cluster/.venv/bin/python scripts/test_strict_load.py
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models import qwen3  # noqa: E402

from runner import _load_model_strict  # noqa: E402

CONFIG = {"model_type": "qwen3", "hidden_size": 16, "num_hidden_layers": 1,
          "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
          "rms_norm_eps": 1e-6, "vocab_size": 32, "max_position_embeddings": 128,
          "rope_theta": 10000.0, "head_dim": 8, "tie_word_embeddings": True}


def write_checkpoint(edit=None) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "config.json").write_text(json.dumps(CONFIG))
    model = qwen3.Model(qwen3.ModelArgs.from_dict(CONFIG))
    weights = dict(tree_flatten(model.parameters()))
    if edit:
        edit(weights)
    mx.save_safetensors(str(d / "model.safetensors"), weights)
    return d


class StrictLoad(unittest.TestCase):
    def test_complete_checkpoint_loads(self):
        model, config = _load_model_strict(write_checkpoint(), lazy=True)
        self.assertEqual(config["model_type"], "qwen3")

    def test_missing_tensor_is_refused(self):
        d = write_checkpoint(lambda w: w.pop("model.layers.0.mlp.up_proj.weight"))
        with self.assertRaises(ValueError) as cm:
            _load_model_strict(d, lazy=True)
        msg = str(cm.exception)
        self.assertIn("Missing 1 parameters", msg)
        self.assertIn("model.layers.0.mlp.up_proj.weight", msg)
        self.assertIn("#116", msg)

    def test_unknown_tensor_is_refused(self):
        d = write_checkpoint(lambda w: w.update({"model.layers.0.extra.weight": mx.zeros((2,))}))
        with self.assertRaises(ValueError) as cm:
            _load_model_strict(d, lazy=True)
        self.assertIn("Received 1 parameters not in model", str(cm.exception))

    def test_wrong_shape_is_refused(self):
        d = write_checkpoint(lambda w: w.update({"model.norm.weight": mx.zeros((8,))}))
        with self.assertRaises(ValueError) as cm:
            _load_model_strict(d, lazy=True)
        self.assertIn("Expected shape", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
