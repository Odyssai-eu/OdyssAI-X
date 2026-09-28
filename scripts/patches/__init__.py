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

from .bailing_hybrid_alias import apply_bailing_hybrid
from .g9v3_alias import apply_g9v3
from .glm_moe_dsa_model import apply_glm_dsa
from .longcat2_pipeline import apply_longcat2_pipeline
from .minimax_m3_alias import apply_minimax_m3
from .mimo_v2_alias import apply_mimo_v2_alias
from .opt_batch_gen import apply_batch_gen_patch
from .pipeline_split_coverage import apply_pipeline_split_fix
from .standard_yarn_rope import patch_yarn_rope

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
    apply_longcat2_pipeline()
