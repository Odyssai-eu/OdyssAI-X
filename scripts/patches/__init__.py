from .bailing_hybrid_alias import apply_bailing_hybrid
from .glm_moe_dsa_model import apply_glm_dsa
from .longcat2_pipeline import apply_longcat2_pipeline
from .minimax_m3_alias import apply_minimax_m3
from .mimo_v2_alias import apply_mimo_v2_alias
from .opt_batch_gen import apply_batch_gen_patch
from .standard_yarn_rope import patch_yarn_rope

_applied = False


def apply_mlx_patches() -> None:
    global _applied
    if _applied:
        return
    _applied = True
    patch_yarn_rope()
    apply_batch_gen_patch()
    apply_mimo_v2_alias()
    apply_bailing_hybrid()
    apply_minimax_m3()
    apply_glm_dsa()
    apply_longcat2_pipeline()
