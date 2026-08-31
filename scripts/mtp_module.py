"""Native MTP (multi-token prediction) module loader for OdyssAI-X runners.

Builds the model's own MTP head(s) as a standalone MLX module, loaded NEXT TO
the mlx-lm trunk (never monkeypatched into it). The trunk stays exactly what
`sharded_load`/`load_model` produced — including pipeline/tensor sharding —
while the MTP module is small enough to be REPLICATED on every rank
(plan D1/D5, docs/PLAN-distributed-mtp.md).

Weight sources, in priority order:
  1. the model dir's own safetensors index (models whose MLX conversion kept
     the MTP tensors — e.g. LongCat `model.mtp.*` if converted with them);
  2. a SIDECAR safetensors file holding the original (bf16) MTP tensors,
     recovered with scripts/sidecar_fetch.py when the MLX conversion stripped
     them (mlx-lm `sanitize` drops layers >= num_hidden_layers — verified
     deepseek_v32.py:494). Sidecar path comes from RUNNER_MTP_SIDECAR or
     `<model_dir>/mtp-sidecar/mtp-sidecar.safetensors`.

v0 scope: the DeepSeek-family layout (GLM-5.2 `glm_moe_dsa`, DeepSeek-V3.2)
— enorm/hnorm/eh_proj + one decoder layer + shared_head_norm, with embed and
lm_head SHARED from the trunk (GLM-5.2 layers.78 ships neither — verified on
the zai-org/GLM-5.2 index). LongCat binding (own embed + own norm + LSA
indexer reuse) lands in E4 once the base port exists.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn


def _log(msg: str) -> None:
    sys.stderr.write(f"[mtp] {msg}\n")
    sys.stderr.flush()


# ──────────────────────────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────────────────────────

# model_type → family. A family fixes: layer class, weight layout, and which
# pieces are shared with the trunk. Extend here for new architectures (and
# mirror in api.py MTP_FAMILY_MODEL_TYPES for the dashboard checkbox).
#
#  deepseek : MLA attention (kv_b absorption), CacheList(KVCache, KVCache) per
#             layer, shared_head.norm -> final norm, args.n_routed_experts.
#  hy_v3    : GQA + q/k-norm (no kv_b), plain KVCache, final_layernorm -> final
#             norm, mlp.expert_bias lives under router, args.num_experts.
#             Hidden contract: PRE-norm trunk hidden (vLLM hy_v3_mtp.py) —
#             prefer hidden_source="pre" at activation.
#  qwen3    : Qwen3.5/3.8 qwen3_5_mtp head. GQA + q/k-norm, plain KVCache,
#             dense MLP. The head's single decoder layer is FULL-attention (not
#             the trunk's Mamba linear_attn) — instantiate DecoderLayer at a
#             full-attn layer_idx. Trunk lives under model.language_model (VL
#             wrapper), so args + embed/lm_head resolve one level deeper. The
#             sidecar ships module-layout keys (fc/norm/pre_fc_norm_*/layers.0)
#             with the trunk's +1 norm shift already applied (extract script).
#  qwen4_exp: Qwen3.8-Flash-Next residual_linear_shared MTP (own module class
#             Qwen4ExpMTPModule, NOT NativeMTPModule): consumes the trunk's
#             PRE-final-mixer HC multi stream, one trunk-class full-attention
#             DecoderLayer (MoE + QSA indexer) in HC space, own final mixer.
#             HYBRID trunk -> requires the #72 snapshot/restore rollback,
#             gated behind RUNNER_MTP_HYBRID_SNAPSHOT=1.
#  glm5_next: GLM-5.3-Flash native MTP head (checkpoint layer 45 ==
#             num_hidden_layers). DeepSeek-V3-lineage fusion (enorm/hnorm/
#             eh_proj concat) -> ONE Glm5NextMTPLayer (plain residual: NoPE MLA
#             + Glm5NextIndexer + DeepseekV32MoE, NO hyper-connections) ->
#             shared_head_norm -> shared lm_head. Reuses the NativeMTPModule
#             scaffold (concat order matches). The trunk is HYBRID (alternating
#             linear-attn + sparse-DSA layers) -> requires the #72 snapshot/
#             restore rollback, gated behind RUNNER_MTP_HYBRID_SNAPSHOT=1. The
#             sidecar ships MODULE-layout keys (embed_q/unembed_out absorbed,
#             experts stacked as switch_mlp) -> loaded STRICT.
_FAMILY_BY_MODEL_TYPE = {
    "glm_moe_dsa": "deepseek", "deepseek_v32": "deepseek",
    "deepseek_v3": "deepseek", "kimi_k2": "deepseek",
    "hy_v3": "hy_v3",
    "qwen3_5": "qwen3",
    "qwen4_exp": "qwen4_exp",
    "glm5_next": "glm5_next",
}


def _family_layer_cls(family: str):
    """The trunk decoder-layer class this family's MTP block instantiates.
    All share the (args, layer_idx) __init__ signature."""
    if family == "hy_v3":
        from mlx_lm.models.hy_v3 import DecoderLayer
        return DecoderLayer
    if family == "qwen3":
        from mlx_lm.models.qwen3_5 import DecoderLayer
        return DecoderLayer
    if family == "glm5_next":
        # Plain-residual sparse-DSA MTP block (NoPE MLA + Glm5NextIndexer +
        # DeepseekV32MoE, no hyper-connections). Vendored under mlx_lm.models
        # by install-model-modules.sh, imported by path like the trunk.
        from mlx_lm.models.glm5_next import Glm5NextMTPLayer
        return Glm5NextMTPLayer
    from mlx_lm.models.deepseek_v32 import DeepseekV32DecoderLayer
    return DeepseekV32DecoderLayer


def _family_cache_factory(family: str):
    """Per-layer cache shape: dsv32 layers read a CacheList (main + indexer
    KV); hy_v3 and qwen3 GQA attention consume a plain KVCache directly."""
    def _factory():
        from mlx_lm.models.cache import CacheList, KVCache
        if family in ("hy_v3", "qwen3"):
            return [KVCache()]
        return [CacheList(KVCache(), KVCache())]
    return _factory


def _family_mtp_layer_idx(family: str, args: Any, start_layer: int) -> int:
    """Which layer_idx to build the MTP block at. The idx picks behaviour in
    families whose DecoderLayer branches on it: qwen3 alternates linear vs full
    attention by `(idx+1) % full_attention_interval`, and the MTP head is a
    FULL-attention layer, so land on a full-attn slot. Others ignore the idx
    beyond identity, so the trunk depth (start_layer) is fine."""
    if family == "qwen3":
        interval = int(getattr(args, "full_attention_interval", 4) or 4)
        return interval - 1                       # (idx+1) % interval == 0 → full attn
    return start_layer


def _family_trunk(family: str, model: Any):
    """(inner, lm_head, args) for the trunk this family's head shares. qwen3
    wraps the text model under `.language_model` (VL config); others expose the
    text trunk at the top level."""
    if family == "qwen3":
        lm = model.language_model
        return lm.model, getattr(lm, "lm_head", None), lm.args
    return model.model, getattr(model, "lm_head", None), getattr(model, "args", None)


class MTPSpec:
    def __init__(self, family: str, start_layer: int, num_layers: int,
                 source: str, source_path: Path):
        self.family = family
        self.start_layer = start_layer      # trunk num_hidden_layers
        self.num_layers = num_layers        # nextn/mtp module count (1 for GLM)
        self.source = source                # "model_index" | "sidecar"
        self.source_path = source_path      # index json or sidecar safetensors

    def __repr__(self) -> str:  # for ready-event logging
        return (f"MTPSpec(family={self.family}, start={self.start_layer}, "
                f"n={self.num_layers}, source={self.source})")


def _read_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        return json.load(f)


def _index_keys(index_path: Path) -> list[str]:
    with open(index_path) as f:
        return list(json.load(f).get("weight_map", {}))


def detect_native_mtp(model_dir: str | Path,
                      sidecar_env: Optional[str] = None) -> Optional[MTPSpec]:
    """Return an MTPSpec if native MTP weights are reachable for this model.

    Order: model's own index first (conversion kept them), then sidecar.
    Returns None when the architecture is unsupported or no weights exist —
    the caller falls back to plain AR, never errors.
    """
    model_dir = Path(model_dir)
    try:
        config = _read_config(model_dir)
    except Exception as e:
        _log(f"detect: no readable config ({e})")
        return None

    model_type = str(config.get("model_type", "")).lower()
    family = _FAMILY_BY_MODEL_TYPE.get(model_type)
    if family is None:
        return None
    # qwen3's VL-wrapper config nests the trunk depth under text_config.
    tcfg = config.get("text_config") or config
    n_layers = int(config.get("num_hidden_layers")
                   or tcfg.get("num_hidden_layers")
                   or 0)
    if not n_layers:
        return None

    # Rollback guard: native-MTP speculative decoding trims the trunk cache to
    # roll back rejected drafts. A HYBRID linear-attention trunk (Qwen3.5's
    # Mamba/GatedDeltaNet layers) uses ArraysCache, whose recurrent state has
    # NO .trim — the rollback silently corrupts it (correct prefill token, then
    # garbage from round 1). Refuse → clean AR. Full-attention-only (interval 1)
    # is trimmable and fine. Verified 2026-08-15: ArraysCache.trim → AttributeError.
    if family == "qwen3":
        fai = int(tcfg.get("full_attention_interval") or 1)
        if fai > 1:
            _log(f"native MTP unsupported on hybrid linear-attn trunk "
                 f"(full_attention_interval={fai}, non-trimmable Mamba cache) — AR only")
            return None
    # qwen4_exp is ALWAYS hybrid (3/4 linear-attn layers): speculation is only
    # safe with the #72 snapshot/restore rollback. New machinery -> opt-in env
    # gate, default 0 = today's behaviour (clean AR fallback).
    if family == "qwen4_exp":
        if os.environ.get("RUNNER_MTP_HYBRID_SNAPSHOT", "0") != "1":
            _log("qwen4_exp trunk is hybrid — set RUNNER_MTP_HYBRID_SNAPSHOT=1 "
                 "to enable snapshot/restore speculation (#72) — AR only")
            return None
    # glm5_next (GLM-5.3-Flash) is likewise a HYBRID trunk (linear-attn +
    # sparse-DSA layers, ArraysCache has no .trim): speculation is only safe
    # with the #72 snapshot/restore rollback. Same opt-in gate as qwen4_exp.
    if family == "glm5_next":
        if os.environ.get("RUNNER_MTP_HYBRID_SNAPSHOT", "0") != "1":
            _log("glm5_next trunk is hybrid — set RUNNER_MTP_HYBRID_SNAPSHOT=1 "
                 "to enable snapshot/restore speculation (#72) — AR only")
            return None
    prefix = f"model.layers.{n_layers}."

    # 1. Conversion kept the MTP tensors in the model dir itself.
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        keys = _index_keys(idx_path)
        if any(k.startswith(prefix) for k in keys):
            # num layers = how many extra layer indices exist
            extra = {int(k.split(".")[2]) for k in keys
                     if k.startswith("model.layers.")
                     and int(k.split(".")[2]) >= n_layers}
            return MTPSpec(family, n_layers, len(extra),
                           "model_index", idx_path)

    # 2. Sidecar recovered from the original repo.
    candidates = []
    if sidecar_env:
        candidates.append(Path(sidecar_env))
    candidates.append(model_dir / "mtp-sidecar" / "mtp-sidecar.safetensors")
    # HF drafter repo layout dropped/symlinked into the model dir (#72:
    # `<model_dir>/mtp-sidecar/` -> kikekewl snapshot with model.safetensors).
    candidates.append(model_dir / "mtp-sidecar")
    for cand in candidates:
        if cand.is_dir():
            # Our extract scripts write mtp-sidecar.safetensors; an HF drafter
            # repo (e.g. kikekewl/...MTP-Drafter, qwen4_exp) ships a plain
            # model.safetensors — accept both layouts.
            for name in ("mtp-sidecar.safetensors", "model.safetensors"):
                if (cand / name).exists():
                    cand = cand / name
                    break
            else:
                cand = cand / "mtp-sidecar.safetensors"
        if cand.exists():
            return MTPSpec(family, n_layers, 1, "sidecar", cand)

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Module
# ──────────────────────────────────────────────────────────────────────────────

class NativeMTPModule(nn.Module):
    """One DeepSeek-family MTP step: eh_proj(concat[enorm(emb), hnorm(h)])
    -> decoder layer -> shared_head_norm -> (shared) lm_head.

    Concat order is [embedding, hidden] — the "embedding_hidden" order the
    MTPLX GLM backend ships as its only supported order (glm_mtp_patch.py).
    embed/lm_head are references INTO the trunk (never copies): asserted at
    bind time and never re-swapped (adapter hot-swap is out of contract, plan
    F-32).
    """

    # #72 rollback mode. A HYBRID trunk (glm5_next: linear-attn + sparse-DSA)
    # can't trim its ArraysCache linear state — mtp_spec switches to snapshot/
    # restore when this is True. Set per-instance from the family (default off
    # for the pure full-attention deepseek trunks that trim cleanly).
    hybrid_snapshot = False

    def __init__(self, args: Any, layer_cls: Callable, layer_idx: int,
                 family: str = "deepseek"):
        super().__init__()
        hs = int(args.hidden_size)
        eps = float(args.rms_norm_eps)
        self.family = family
        if family == "glm5_next":
            self.hybrid_snapshot = True
        self.enorm = nn.RMSNorm(hs, eps=eps)
        self.hnorm = nn.RMSNorm(hs, eps=eps)
        self.eh_proj = nn.Linear(2 * hs, hs, bias=False)
        self.mtp_block = layer_cls(args, layer_idx=layer_idx)
        # deepseek: shared_head.norm; hy_v3: final_layernorm — both load into
        # this slot via the family rename in _rewrite_weights.
        self.shared_head_norm = nn.RMSNorm(hs, eps=eps)
        # Trunk references bound at load time. They MUST stay OUT of this
        # module's tree: nn.Module.__setattr__ registers any Module child
        # (underscore or not), which would (a) drag the trunk's embed/head
        # into parameters()/quantize passes — mutating the SHARED trunk —
        # and (b) inflate the module's apparent size. object.__setattr__
        # bypasses registration (caught by the .31 full smoke: 21 GB
        # resident + quantize converting eh_proj around bf16 weights).
        object.__setattr__(self, "_embed", None)
        object.__setattr__(self, "_lm_head", None)
        object.__setattr__(self, "_cache_factory", None)

    # -- binding ------------------------------------------------------------
    def bind_trunk(self, model: Any) -> None:
        inner, lm, _ = _family_trunk(self.family, model)
        object.__setattr__(self, "_embed", inner.embed_tokens)
        # tie_word_embeddings models expose as_linear via embed; GLM/DSv32/qwen
        # have a real lm_head.
        if lm is None:
            lm = inner.embed_tokens.as_linear
        object.__setattr__(self, "_lm_head", lm)
        # Cache shape is family-dependent (CacheList for dsv32, plain KVCache
        # for hy_v3's GQA) — resolved from the registry.
        object.__setattr__(self, "_cache_factory",
                           _family_cache_factory(self.family))

    def make_cache(self) -> list:
        assert self._cache_factory is not None, "bind_trunk() first"
        return self._cache_factory()

    # -- forward ------------------------------------------------------------
    def draft_step(self, token_ids: mx.array, prev_hidden: mx.array,
                   cache: list) -> tuple[mx.array, mx.array]:
        """One MTP step over S positions (S=1 in the chained-draft loop).

        token_ids: [B, S] int32 — tokens whose K/V this step APPENDS to the
        mtp cache (one position per call in the draft chain, plan F-17).
        prev_hidden: [B, S, H] — trunk (or previous-step) hidden aligned to
        token_ids positions.
        Returns (logits [B, S, V], hidden [B, S, H]).
        """
        from mlx_lm.models.base import create_attention_mask
        emb = self._embed(token_ids)
        mixed = self.eh_proj(
            mx.concatenate([self.enorm(emb), self.hnorm(prev_hidden)], axis=-1)
        )
        # Mask convention mirrors dsv32's own __call__ exactly:
        # create_attention_mask(h, cache[0][0], return_array=True) — the
        # CacheList's first child carries the offset. Plain KVCache (tests,
        # simple layers) has no children: use it directly.
        first = cache[0] if cache else None
        if first is not None:
            try:
                first = first[0]
            except TypeError:
                pass
        mask = create_attention_mask(mixed, first, return_array=True)
        h = self.mtp_block(mixed, mask=mask, cache=cache[0])
        logits = self._lm_head(self.shared_head_norm(h))
        return logits, h


class Qwen4ExpMTPModule(nn.Module):
    """Qwen4Exp (Qwen3.8-Flash-Next) MTP step — residual_linear_shared fusion.

    Reference: vLLM vllm/models/qwen4_exp/nvidia/mtp.py (Apache-2.0),
    "scheme A". Differs from NativeMTPModule's concat fusion:
      * consumes the trunk's PRE-final-mixer MULTI stream [B, S, hc*H]
        (mtp_spec captures it via `hidden_source_override = "hc_multi"`);
      * fc_embedding projects the token embedding, fc_hidden (shared across
        branches) projects each HC branch; the embedding is added as a
        residual to EVERY branch;
      * the block is ONE trunk-class DecoderLayer forced full_attention,
        running in HC space (own _AttnCache: KV + QSA indexer — trimmable,
        so the mtp_cache trim/re-forward sequence is unchanged);
      * its OWN final mixer (use_combine=False — it carries the norm, like
        the trunk's) collapses to the sample hidden for the trunk's lm_head;
        the UN-collapsed layer output is the next step's multi stream.
    `hybrid_snapshot = True` switches mtp_spec to the #72 snapshot/restore
    rollback on the trunk (ArraysCache has no .trim). embed/lm_head are trunk
    references OUT of the module tree (same contract as NativeMTPModule).
    """

    hybrid_snapshot = True
    hidden_source_override = "hc_multi"

    def __init__(self, targs: Any):
        super().__init__()
        import copy
        from mlx_lm.models.qwen4_exp import (
            DecoderLayer, GatedResidual, RMSNorm, RotaryEmbedding)

        hs = int(targs.hidden_size)
        eps = float(targs.rms_norm_eps)
        self.family = "qwen4_exp"
        self.hc = int(targs.hc_count)
        self.d = hs
        # vLLM: GemmaRMSNorm — flat norm (embedding H, hidden hc*H); the
        # per-branch norm lives inside the mixers' hc_norm (group_size=H).
        self.pre_fc_norm_embedding = RMSNorm(hs, eps=eps)
        self.pre_fc_norm_hidden = RMSNorm(self.hc * hs, eps=eps)
        self.fc_embedding = nn.Linear(hs, hs, bias=False)
        self.fc_hidden = nn.Linear(hs, hs, bias=False)
        # One trunk DecoderLayer at a PAST-THE-END index: layer_types gets a
        # full_attention slot appended, and (idx+1) can't collide with
        # ple_layer_ids — so no PLE, no n-gram inputs on this layer.
        idx = len(targs.layer_types)
        args2 = copy.copy(targs)
        args2.layer_types = list(targs.layer_types) + ["full_attention"]
        self.layers = [DecoderLayer(args2, idx)]
        self.hyper_connection_mixer = GatedResidual(targs, use_combine=False)
        rotary_dim = int(targs.head_dim * targs.partial_rotary_factor)
        self.rope = RotaryEmbedding(rotary_dim, targs.rope_theta)
        object.__setattr__(self, "_embed", None)
        object.__setattr__(self, "_lm_head", None)

    def bind_trunk(self, model: Any) -> None:
        inner, lm, _ = _family_trunk("qwen4_exp", model)
        object.__setattr__(self, "_embed", inner.embed_tokens)
        if lm is None:
            lm = inner.embed_tokens.as_linear
        object.__setattr__(self, "_lm_head", lm)

    def make_cache(self) -> list:
        from mlx_lm.models.qwen4_exp import _AttnCache
        return [_AttnCache()]

    def draft_step(self, token_ids: mx.array, prev_hidden: mx.array,
                   cache: list) -> tuple[mx.array, mx.array]:
        """One MTP step over S positions (S=1 in the chained-draft loop).

        token_ids: [B, S] — appended to the mtp KV(+indexer) cache.
        prev_hidden: [B, S, hc*H] MULTI stream — trunk capture on prefill /
        after commit, previous draft_step output inside the chain.
        Returns (logits [B, S, V], multi stream [B, S, hc*H]).
        """
        from mlx_lm.models.base import create_attention_mask
        B, S = token_ids.shape
        emb = self.fc_embedding(
            self.pre_fc_norm_embedding(self._embed(token_ids)))
        h = self.pre_fc_norm_hidden(prev_hidden)
        h = self.fc_hidden(h.reshape(B, S, self.hc, self.d))
        h = (emb[..., None, :] + h).reshape(B, S, self.hc * self.d)
        c = cache[0]
        mask = create_attention_mask(h, c)
        h = self.layers[0](h, self.rope, mask, None, c, c.indexer, None, None)
        sample = self.hyper_connection_mixer(h)
        logits = self._lm_head(sample)
        return logits, h


# ──────────────────────────────────────────────────────────────────────────────
# Weight loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_source_weights(spec: MTPSpec, model_dir: Path) -> dict[str, mx.array]:
    """Read the raw MTP tensors (bf16 or quantized) for the spec's layer(s)."""
    # qwen3 ships a module-layout sidecar (fc/norm/pre_fc_norm_*/layers.0.*)
    # with NO `model.layers.{n}.` prefix — take every tensor in the file.
    if spec.family == "qwen3":
        return dict(mx.load(str(spec.source_path)))
    prefix = f"model.layers.{spec.start_layer}."
    raw: dict[str, mx.array] = {}
    if spec.source == "sidecar":
        raw = {k: v for k, v in mx.load(str(spec.source_path)).items()
               if k.startswith(prefix)}
    else:
        with open(spec.source_path) as f:
            wm = json.load(f)["weight_map"]
        shards = sorted({wm[k] for k in wm if k.startswith(prefix)})
        for shard in shards:
            for k, v in mx.load(str(model_dir / shard)).items():
                if k.startswith(prefix):
                    raw[k] = v
    return raw


def _rewrite_weights(raw: dict[str, mx.array], spec: MTPSpec,
                     args: Any) -> dict[str, mx.array]:
    """Map original checkpoint names onto NativeMTPModule's tree.

    Mirrors the trunk's own sanitize conventions (deepseek_v32.Model.sanitize):
    experts.{i} stacked into mlp.switch_mlp, rotary inv_freq dropped. Shared
    pieces (embed_tokens, shared_head.head/lm_head) are SKIPPED — the module
    references the trunk's.
    """
    # qwen3: module-layout sidecar → direct rename onto the NativeMTPModule tree.
    # fc→eh_proj, pre_fc_norm_embedding→enorm, pre_fc_norm_hidden→hnorm,
    # norm→shared_head_norm, layers.0.*→mtp_block.* . Dense MLP, no MLA/experts.
    if spec.family == "qwen3":
        _RENAME = {
            "pre_fc_norm_embedding": "enorm",
            "pre_fc_norm_hidden": "hnorm",
            "fc": "eh_proj",
            "norm": "shared_head_norm",
        }
        out: dict[str, mx.array] = {}
        for k, v in raw.items():
            if k.startswith("layers.0."):
                out["mtp_block." + k[len("layers.0."):]] = v
                continue
            head, _, rest = k.partition(".")
            out[_RENAME.get(head, head) + ("." + rest if rest else "")] = v
        return out

    prefix = f"model.layers.{spec.start_layer}."
    flat: dict[str, mx.array] = {}
    for k, v in raw.items():
        if "rotary_emb.inv_freq" in k:
            continue
        suffix = k[len(prefix):]
        if suffix.startswith("embed_tokens.") or suffix.startswith("shared_head.head."):
            continue  # shared with trunk
        if suffix.startswith("shared_head.norm."):
            flat["shared_head_norm." + suffix[len("shared_head.norm."):]] = v
        elif spec.family == "hy_v3" and suffix.startswith("final_layernorm."):
            # hy_v3 names its pre-lm_head norm final_layernorm — same slot.
            flat["shared_head_norm." + suffix[len("final_layernorm."):]] = v
        elif spec.family == "hy_v3" and suffix == "mlp.expert_bias":
            # Raw HF puts expert_bias directly under mlp; the vendored hy_v3
            # Router owns it (same move the trunk sanitize does).
            flat["mtp_block.mlp.router.expert_bias"] = v
        elif suffix.startswith(("enorm.", "hnorm.", "eh_proj.")):
            flat[suffix] = v
        else:
            flat["mtp_block." + suffix] = v

    # MLA kv_b absorption: DeepseekV32Attention consumes the SPLIT form
    # (embed_q [heads, kv_lora, qk_nope] + unembed_out [heads, v_head,
    # kv_lora]), not the raw fused kv_b_proj the checkpoint ships. Same
    # transform the trunk conversion applies (and MTPLX's
    # _rewrite_kv_b_projection) — caught by mtp_loader_smoke on GLM-5.2.
    kv_b_key = "mtp_block.self_attn.kv_b_proj.weight"
    if kv_b_key in flat:
        v = flat.pop(kv_b_key)
        heads = int(args.num_attention_heads)
        nope = int(args.qk_nope_head_dim)
        vdim = int(args.v_head_dim)
        v = v.reshape(heads, nope + vdim, -1)
        flat["mtp_block.self_attn.embed_q.weight"] = mx.contiguous(
            v[:, :nope, :].swapaxes(-1, -2))
        flat["mtp_block.self_attn.unembed_out.weight"] = mx.contiguous(
            v[:, nope:, :])

    # Stack per-expert tensors into the SwitchGLU layout the MoE layer uses.
    # deepseek args name it n_routed_experts; hy_v3 args name it num_experts.
    n_routed = int(getattr(args, "n_routed_experts", 0)
                   or getattr(args, "num_experts", 0) or 0)
    if n_routed and f"mtp_block.mlp.experts.0.gate_proj.weight" in flat:
        for mod in ("gate_proj", "down_proj", "up_proj"):
            for leaf in ("weight", "scales", "biases"):
                first = f"mtp_block.mlp.experts.0.{mod}.{leaf}"
                if first not in flat:
                    continue
                stacked = mx.stack([
                    flat.pop(f"mtp_block.mlp.experts.{i}.{mod}.{leaf}")
                    for i in range(n_routed)
                ])
                flat[f"mtp_block.mlp.switch_mlp.{mod}.{leaf}"] = stacked
    return flat


def _maybe_quantize(module: NativeMTPModule, config: dict,
                    weights: dict[str, mx.array], *,
                    when: str) -> None:
    """Optional trunk-aligned quantize-on-load (RUNNER_MTP_QUANT=1).

    Ordering matters (caught by the .31 full smoke):
      * source ALREADY quantized (scales present) -> convert layers BEFORE
        load_weights so the quantized tensors land in QuantizedLinear slots
        (`when="pre_load"`);
      * source bf16 (sidecar) -> load bf16 into the bf16 module FIRST, then
        nn.quantize converts layers AND quantizes the loaded weights
        (`when="post_load"`).
    Trunk-shared embed/lm_head live OUTSIDE the module tree (bind_trunk) so
    this can never touch them. The draft head tolerates quantization by
    design — drafts are proposals, the trunk verify guards exactness.
    """
    q = config.get("quantization") or config.get("quantization_config") or {}
    if not q or "bits" not in q or "group_size" not in q:
        return

    already_quantized = any(k.endswith(".scales") for k in weights)
    if (already_quantized and when != "pre_load") or (
            not already_quantized and when != "post_load"):
        return

    def predicate(path: str, m: Any):
        if not hasattr(m, "to_quantized"):
            return False
        if already_quantized:
            return f"{path}.scales" in weights
        return True

    nn.quantize(module, group_size=int(q["group_size"]), bits=int(q["bits"]),
                mode=q.get("mode", "affine"), class_predicate=predicate)


def load_native_mtp(model: Any, model_dir: str | Path, *,
                    sidecar: Optional[str] = None,
                    quantize: bool = False) -> Optional[NativeMTPModule]:
    """Build + load the MTP module next to an already-loaded trunk.

    Returns None (with a log line) instead of raising when anything is
    missing — callers treat that as "MTP unavailable, serve AR".
    """
    model_dir = Path(model_dir)
    spec = detect_native_mtp(model_dir, sidecar_env=sidecar)
    if spec is None:
        _log("no native MTP weights reachable — AR only")
        return None

    t0 = time.time()
    config = _read_config(model_dir)
    # The MTP block is built from the TRUNK's args (qwen3's live one level down
    # under .language_model); _family_trunk resolves them per family.
    _, _, args = _family_trunk(spec.family, model)
    if args is None:
        _log("trunk model has no .args — cannot build MTP block")
        return None

    # qwen4_exp: own module class (residual_linear_shared fusion in HC space)
    # — bypasses the NativeMTPModule scaffold. The kikekewl sidecar ships the
    # official `mtp.*` weights with the prefix stripped and the experts
    # already stacked as switch_mlp: keys match this module tree 1:1, so the
    # load is STRICT — any mismatch aborts to AR instead of serving garbage.
    if spec.family == "qwen4_exp":
        targs = getattr(args, "text", args)
        try:
            module = Qwen4ExpMTPModule(targs)
            module.bind_trunk(model)
            raw = dict(mx.load(str(spec.source_path)))
            if not raw:
                _log(f"spec found ({spec}) but zero tensors loaded — AR only")
                return None
            module.load_weights(list(raw.items()), strict=True)
            mx.eval(module.parameters())
        except Exception as e:
            _log(f"qwen4_exp MTP load failed ({e}) — AR only")
            return None
        if quantize:
            _log("qwen4_exp MTP: quantize requested but v1 serves the bf16 "
                 "sidecar as-is (5.2 GB) — ignored")
        from mlx.utils import tree_flatten
        gb = sum(v.nbytes for _, v in tree_flatten(module.parameters())) / 1e9
        _log(f"loaded {spec} — {len(raw)} tensors, {gb:.2f} GB bf16, "
             f"hybrid_snapshot armed, {time.time()-t0:.1f}s")
        return module

    # glm5_next: the NativeMTPModule scaffold fits (DeepSeek-V3 concat fusion),
    # but the sidecar already ships MODULE-layout keys (embed_q/unembed_out
    # absorbed by the extraction's glm5_next.sanitize reuse, experts stacked as
    # switch_mlp, indexer passthrough) — so it maps onto the module tree 1:1
    # and loads STRICT. Any mismatch aborts to AR rather than serving garbage.
    # Served bf16 as-is next to the quantized trunk (like qwen4_exp): the draft
    # head tolerates it — the trunk verify guards exactness.
    if spec.family == "glm5_next":
        try:
            layer_cls = _family_layer_cls("glm5_next")
            module = NativeMTPModule(args, layer_cls, spec.start_layer,
                                     family="glm5_next")
            module.bind_trunk(model)
            raw = dict(mx.load(str(spec.source_path)))
            if not raw:
                _log(f"spec found ({spec}) but zero tensors loaded — AR only")
                return None
            module.load_weights(list(raw.items()), strict=True)
            mx.eval(module.parameters())
        except Exception as e:
            _log(f"glm5_next MTP load failed ({e}) — AR only")
            return None
        if quantize:
            _log("glm5_next MTP: quantize requested but v1 serves the bf16 "
                 "sidecar as-is — ignored")
        from mlx.utils import tree_flatten
        gb = sum(v.nbytes for _, v in tree_flatten(module.parameters())) / 1e9
        _log(f"loaded {spec} — {len(raw)} tensors, {gb:.2f} GB bf16, STRICT, "
             f"hybrid_snapshot armed, {time.time()-t0:.1f}s")
        return module

    try:
        layer_cls = _family_layer_cls(spec.family)
    except Exception as e:
        _log(f"{spec.family} layer import failed ({e})")
        return None

    layer_idx = _family_mtp_layer_idx(spec.family, args, spec.start_layer)
    module = NativeMTPModule(args, layer_cls, layer_idx,
                             family=spec.family)
    module.bind_trunk(model)

    # FAST PATH — pre-quantized module-layout sidecar. Loading the bf16
    # sidecar per-rank does `_stack_moe_experts` (mx.stack of 768 bf16
    # experts) = a ~20GB contiguous spike ON TOP of the trunk shard, which
    # OOM-kills the heavy all-MoE pipeline ranks (E3 2026-07-06: rank0 with
    # the light dense shard survived, ranks 1/2 died silently). A one-time
    # offline `build_prequantized_sidecar` writes the module already in Q6
    # (~8GB, module-layout); loading it needs no stack and no rewrite.
    q6_path = _prequantized_path(spec, sidecar)
    if q6_path is not None and q6_path.exists():
        q6 = dict(mx.load(str(q6_path)))
        if quantize:
            _quantize_structure(module, config, q6)
        module.load_weights(list(q6.items()), strict=False)
        mx.eval(module.parameters())
        from mlx.utils import tree_flatten
        gb = sum(v.nbytes for _, v in tree_flatten(module.parameters())) / 1e9
        _log(f"loaded {spec} via PREQUANTIZED sidecar — {len(q6)} tensors, "
             f"{gb:.2f} GB, {time.time()-t0:.1f}s")
        return module

    # SLOW PATH — build from the raw (bf16) sidecar. Fine on a single node /
    # the light rank; use build_prequantized_sidecar first for multi-rank.
    raw = _load_source_weights(spec, model_dir)
    if not raw:
        _log(f"spec found ({spec}) but zero tensors loaded — AR only")
        return None
    mapped = _rewrite_weights(raw, spec, args)
    if quantize:
        _maybe_quantize(module, config, mapped, when="pre_load")
    module.load_weights(list(mapped.items()), strict=False)
    if quantize:
        _maybe_quantize(module, config, mapped, when="post_load")
    mx.eval(module.parameters())

    from mlx.utils import tree_flatten
    n_params = sum(v.size for _, v in tree_flatten(module.parameters()))
    _log(f"loaded {spec} — {len(mapped)} tensors, {n_params/1e9:.2f}B params, "
         f"{time.time()-t0:.1f}s (quantize={quantize})")
    return module


def _prequantized_path(spec: "MTPSpec", sidecar_env: Optional[str]) -> Optional[Path]:
    """Where a pre-quantized module-layout sidecar would live (next to the
    bf16 one, or under the RUNNER_MTP_SIDECAR dir)."""
    cands = []
    if sidecar_env:
        p = Path(sidecar_env)
        cands.append((p if p.is_dir() else p.parent) / "module-q6.safetensors")
    if spec.source == "sidecar":
        cands.append(spec.source_path.parent / "module-q6.safetensors")
    return cands[0] if cands else None


def _quantize_structure(module: NativeMTPModule, config: dict,
                        q6_weights: dict[str, mx.array]) -> None:
    """Convert module Linears to QuantizedLinear wherever the pre-quantized
    sidecar carries `.scales` for that path — so load_weights lands the
    quantized tensors in matching slots (no data touched here)."""
    q = config.get("quantization") or config.get("quantization_config") or {}
    if not q or "bits" not in q or "group_size" not in q:
        return

    def predicate(path: str, m: Any):
        return hasattr(m, "to_quantized") and f"{path}.scales" in q6_weights

    nn.quantize(module, group_size=int(q["group_size"]), bits=int(q["bits"]),
                mode=q.get("mode", "affine"), class_predicate=predicate)


def build_prequantized_sidecar(model_dir: str | Path,
                               sidecar: Optional[str] = None,
                               out_path: Optional[str] = None) -> Path:
    """One-time offline: read the bf16 sidecar, build+quantize the module,
    and save it in module layout (Q6, ~8GB). Run ONCE on a node with free
    RAM; the result loads per-rank with no bf16 stack spike. Uses a stub
    trunk (bind only stores embed/lm_head refs — never called here)."""
    from mlx.utils import tree_flatten

    model_dir = Path(model_dir)
    config = _read_config(model_dir)
    spec = detect_native_mtp(model_dir, sidecar_env=sidecar)
    if spec is None:
        raise RuntimeError("no native MTP sidecar found to pre-quantize")
    layer_cls = _family_layer_cls(spec.family)

    # Minimal ModelArgs for the layer + a stub trunk carrying embed/lm_head.
    if spec.family == "hy_v3":
        from mlx_lm.models.hy_v3 import ModelArgs
    else:
        from mlx_lm.models.glm_moe_dsa import ModelArgs
    args = ModelArgs.from_dict(config)

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.args = args
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(
                int(config["vocab_size"]), int(config["hidden_size"]))
            self.lm_head = nn.Linear(int(config["hidden_size"]),
                                     int(config["vocab_size"]), bias=False)

    module = NativeMTPModule(args, layer_cls, spec.start_layer,
                             family=spec.family)
    module.bind_trunk(_Stub())
    raw = _load_source_weights(spec, model_dir)
    mapped = _rewrite_weights(raw, spec, args)
    # Order matters: load bf16 into the bf16 module, THEN quantize the loaded
    # weights (post_load). Quantizing the structure first would leave the
    # bf16 mapped weights unconverted -> the saved file stays bf16 (21GB).
    module.load_weights(list(mapped.items()), strict=False)
    _maybe_quantize(module, config, mapped, when="post_load")
    mx.eval(module.parameters())

    flat = dict(tree_flatten(module.parameters()))
    out = Path(out_path) if out_path else (
        _prequantized_path(spec, sidecar)
        or spec.source_path.parent / "module-q6.safetensors")
    mx.save_safetensors(str(out), flat)
    gb = sum(v.nbytes for v in flat.values()) / 1e9
    _log(f"pre-quantized module -> {out} ({len(flat)} tensors, {gb:.2f} GB)")
    return out
