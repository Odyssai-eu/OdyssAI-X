#!/usr/bin/env python3
"""A missing Argo-only model module must not kill the whole patch set.

patches/__init__.py used to import longcat2_pipeline and glm_moe_dsa_model
unconditionally; both do `from mlx_lm.models import <model>` at module level,
and those model files are hand-copied into the Argo venvs only (the repo does
not vendor them). On every other node — the three 96 GB Macs since 01/10, and
any fresh install.sh — the package import died on longcat2 and runner.py
swallowed it ("[runner] mlx patches not applied"), so yarn RoPE, batch_gen
and the pipeline split fix silently stopped applying (verified on .49,
2026-10-07: ImportError cannot import name 'longcat2' from 'mlx_lm.models').

No node is reached and no real mlx/mlx_lm is required: everything mlx the
core patch modules need at import time is faked, and the Argo-only model
modules are poisoned (a None entry in sys.modules makes the from-import raise
ImportError, the exact failure mode of .49).

    .venv/bin/python scripts/test_patches_guarded_imports.py
"""
import sys
import types
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


# --- fakes: just enough import-time surface for the core patch modules.
class _PermissiveModule(types.ModuleType):
    """Any attribute read at import time yields a placeholder type — enough for
    annotations and subclassing, the patch bodies never run in this test."""
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        placeholder = type(name, (), {})
        setattr(self, name, placeholder)
        return placeholder


mlx = types.ModuleType("mlx")
mlx.core = _PermissiveModule("mlx.core")
mlx_lm = types.ModuleType("mlx_lm")
mlx_lm_generate = types.ModuleType("mlx_lm.generate")
mlx_lm_generate.GenerationBatch = type("GenerationBatch", (), {})
mlx_lm_models = types.ModuleType("mlx_lm.models")
# An empty __path__ and no model attrs reproduce .49 exactly: the submodule
# import finds nothing (ModuleNotFoundError, swallowed by _handle_fromlist)
# and IMPORT_FROM raises "cannot import name 'longcat2' from 'mlx_lm.models'".
# (Poisoning sys.modules with None does NOT work on 3.11: `from pkg import x`
# then binds None instead of raising — bpo-17636.)
mlx_lm_models.__path__ = []
mlx_lm_pipeline = types.ModuleType("mlx_lm.models.pipeline")
mlx_lm_pipeline.PipelineMixin = type("PipelineMixin", (), {})
mlx_lm_rope = _PermissiveModule("mlx_lm.models.rope_utils")  # needs YarnRoPE at import
mlx_lm_models.pipeline = mlx_lm_pipeline
mlx_lm_models.rope_utils = mlx_lm_rope
for name, mod in {
    "mlx": mlx, "mlx.core": mlx.core,
    "mlx_lm": mlx_lm, "mlx_lm.generate": mlx_lm_generate,
    "mlx_lm.models": mlx_lm_models, "mlx_lm.models.pipeline": mlx_lm_pipeline,
    "mlx_lm.models.rope_utils": mlx_lm_rope,
}.items():
    sys.modules[name] = mod

for m in [m for m in sys.modules if m == "patches" or m.startswith("patches.")]:
    del sys.modules[m]

import patches  # noqa: E402

check("the package still imports when longcat2 is missing", patches.apply_longcat2_pipeline, None)
check("… and when glm_moe_dsa/deepseek_v32 are missing", patches.apply_glm_dsa, None)
for sym in ("apply_pipeline_split_fix", "patch_yarn_rope", "apply_batch_gen_patch",
            "apply_mimo_v2_alias", "apply_bailing_hybrid", "apply_minimax_m3", "apply_g9v3"):
    check(f"core patch symbol survived: {sym}", callable(getattr(patches, sym, None)), True)

# apply_mlx_patches wiring: every core patch applies, the skipped ones don't.
CALLS = []
for sym in ("apply_pipeline_split_fix", "patch_yarn_rope", "apply_batch_gen_patch",
            "apply_mimo_v2_alias", "apply_bailing_hybrid", "apply_minimax_m3", "apply_g9v3"):
    setattr(patches, sym, (lambda name: lambda: CALLS.append(name))(sym))
patches.apply_mlx_patches()
patches.apply_mlx_patches()  # idempotent: _applied short-circuits
check("apply order keeps the pipeline split first, skipped patches never called",
      CALLS, ["apply_pipeline_split_fix", "patch_yarn_rope", "apply_batch_gen_patch",
              "apply_mimo_v2_alias", "apply_bailing_hybrid", "apply_minimax_m3", "apply_g9v3"])

# Source guard: the two model-specific imports each have their except branch
# that disables the applier instead of letting the package import die.
src = (SCRIPTS / "patches" / "__init__.py").read_text()
check("longcat2 import is guarded",
      "except ImportError as e:\n    apply_longcat2_pipeline = None" in src, True)
check("glm_moe_dsa import is guarded",
      "except ImportError as e:\n    apply_glm_dsa = None" in src, True)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
