# Portions derived from exo (https://github.com/exo-explore/exo),
# src/exo/worker/engines/mlx/patches/__init__.py, Copyright 2025 Exo Technologies Ltd.
# Upstream commit: f0d1371d89a7f899e96977014cce7307a53682f2 (2026-04-28);
# provenance in vendor/exo/UPSTREAM.md.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0 (a copy ships in
# vendor/exo/LICENSE). Unless required by applicable law or agreed to in
# writing, software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by OdyssAI (odyssai.eu), 2026: relative imports; the exo patch set
# (yarn RoPE, batch generation) extended with OdyssAI's own model patches and
# the pipeline split-coverage fix. These modifications are part of OdyssAI-X,
# licensed under the GNU AGPL v3.0 (see LICENSE and NOTICE).

import sys

from .bailing_hybrid_alias import apply_bailing_hybrid
from .g9v3_alias import apply_g9v3
from .minimax_m3_alias import apply_minimax_m3
from .mimo_v2_alias import apply_mimo_v2_alias
from .opt_batch_gen import apply_batch_gen_patch
from .pipeline_split_coverage import apply_pipeline_split_fix
from .standard_yarn_rope import patch_yarn_rope

# glm_moe_dsa and longcat2 import their target model from mlx_lm at module
# level. Those model files exist only where the model has been deployed
# (longcat2.py and glm_moe_dsa.py are hand-copied into the Argo venvs; the
# repo does not vendor them), so the import fails on every other node — the
# 96 GB Macs since 01/10, and any fresh install.sh. One missing model module
# used to kill the whole package import, and runner.py:1645 swallows it:
# "[runner] mlx patches not applied" — so yarn RoPE, batch_gen and the
# pipeline split fix silently stopped applying everywhere but Argo. Guard the
# model-specific imports; the core patches above stay hard (they only depend
# on mlx_lm core modules that always exist).
try:
    from .glm_moe_dsa_model import apply_glm_dsa
except ImportError as e:
    apply_glm_dsa = None
    sys.stderr.write(f"[patches] glm_moe_dsa model module unavailable ({e}) — patch skipped\n")
try:
    from .longcat2_pipeline import apply_longcat2_pipeline
except ImportError as e:
    apply_longcat2_pipeline = None
    sys.stderr.write(f"[patches] longcat2 model module unavailable ({e}) — patch skipped\n")

_applied = False


def apply_mlx_patches() -> None:
    global _applied
    if _applied:
        return
    _applied = True
    # En premier : corrige un trou de couverture des couches en pipeline
    # (des couches n'etaient calculees par aucun rang). Doit preceder
    # tout patch qui redefinit pipeline() pour un modele donne.
    apply_pipeline_split_fix()
    patch_yarn_rope()
    apply_batch_gen_patch()
    apply_mimo_v2_alias()
    apply_bailing_hybrid()
    apply_minimax_m3()
    apply_g9v3()
    # apply_glm_dsa()  # DEBRANCHE 2026-08-29: venv glm_moe_dsa = PR#1410 head (coherent full/shared), le patch Option A (juin, GLM-5.2) mixait mal avec le snapshot -> 3 bugs multi-node. Re-brancher SEULEMENT si regression GLM-5.2.
    if apply_longcat2_pipeline is not None:
        apply_longcat2_pipeline()
