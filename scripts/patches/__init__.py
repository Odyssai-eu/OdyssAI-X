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
