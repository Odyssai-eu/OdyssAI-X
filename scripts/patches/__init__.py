from .bailing_hybrid_alias import apply_bailing_hybrid
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
