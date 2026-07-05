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
# pieces are shared with the trunk. Extend here for new architectures.
_DEEPSEEK_FAMILY = {"glm_moe_dsa", "deepseek_v32", "deepseek_v3", "kimi_k2"}


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
    if model_type not in _DEEPSEEK_FAMILY:
        return None
    n_layers = int(config.get("num_hidden_layers") or 0)
    if not n_layers:
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
            return MTPSpec("deepseek", n_layers, len(extra),
                           "model_index", idx_path)

    # 2. Sidecar recovered from the original repo.
    candidates = []
    if sidecar_env:
        candidates.append(Path(sidecar_env))
    candidates.append(model_dir / "mtp-sidecar" / "mtp-sidecar.safetensors")
    for cand in candidates:
        if cand.is_dir():
            cand = cand / "mtp-sidecar.safetensors"
        if cand.exists():
            return MTPSpec("deepseek", n_layers, 1, "sidecar", cand)

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

    def __init__(self, args: Any, layer_cls: Callable, layer_idx: int):
        super().__init__()
        hs = int(args.hidden_size)
        eps = float(args.rms_norm_eps)
        self.enorm = nn.RMSNorm(hs, eps=eps)
        self.hnorm = nn.RMSNorm(hs, eps=eps)
        self.eh_proj = nn.Linear(2 * hs, hs, bias=False)
        self.mtp_block = layer_cls(args, layer_idx=layer_idx)
        self.shared_head_norm = nn.RMSNorm(hs, eps=eps)
        # Bound at load time (trunk references, not parameters of this module):
        self._embed: Optional[Callable] = None
        self._lm_head: Optional[Callable] = None
        self._cache_factory: Optional[Callable] = None

    # -- binding ------------------------------------------------------------
    def bind_trunk(self, model: Any) -> None:
        inner = model.model
        self._embed = inner.embed_tokens
        # tie_word_embeddings models expose as_linear via embed; GLM/DSv32
        # have a real lm_head.
        self._lm_head = getattr(model, "lm_head", None)
        if self._lm_head is None:
            self._lm_head = lambda h: inner.embed_tokens.as_linear(h)
        # One CacheList(KVCache, KVCache) per dsv32-style layer.
        def _factory():
            from mlx_lm.models.cache import CacheList, KVCache
            return [CacheList(KVCache(), KVCache())]
        self._cache_factory = _factory

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


# ──────────────────────────────────────────────────────────────────────────────
# Weight loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_source_weights(spec: MTPSpec, model_dir: Path) -> dict[str, mx.array]:
    """Read the raw MTP tensors (bf16 or quantized) for the spec's layer(s)."""
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
        elif suffix.startswith(("enorm.", "hnorm.", "eh_proj.")):
            flat[suffix] = v
        else:
            flat["mtp_block." + suffix] = v

    # Stack per-expert tensors into the SwitchGLU layout the dsv32 layer uses.
    n_routed = int(getattr(args, "n_routed_experts", 0) or 0)
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
                    weights: dict[str, mx.array]) -> None:
    """Optional trunk-aligned quantize-on-load (RUNNER_MTP_QUANT=1).

    v0 default is OFF: the sidecar stays bf16 (~14 GB for GLM-5.2's MoE MTP
    layer), which every node carries comfortably; exactness first, memory
    optimization after G1 (plan §E1/D5).
    """
    q = config.get("quantization") or config.get("quantization_config") or {}
    if not q or "bits" not in q or "group_size" not in q:
        return

    already_quantized = any(k.endswith(".scales") for k in weights)

    def predicate(path: str, m: Any):
        if not hasattr(m, "to_quantized"):
            return False
        if already_quantized:
            return f"{path}.scales" in weights
        return f"{path}.weight" in weights

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
    args = getattr(model, "args", None)
    if args is None:
        _log("trunk model has no .args — cannot build MTP block")
        return None

    try:
        from mlx_lm.models.deepseek_v32 import DeepseekV32DecoderLayer
    except Exception as e:
        _log(f"deepseek_v32 layer import failed ({e})")
        return None

    module = NativeMTPModule(args, DeepseekV32DecoderLayer, spec.start_layer)
    module.bind_trunk(model)

    raw = _load_source_weights(spec, model_dir)
    if not raw:
        _log(f"spec found ({spec}) but zero tensors loaded — AR only")
        return None
    mapped = _rewrite_weights(raw, spec, args)
    if quantize:
        _maybe_quantize(module, config, mapped)
    module.load_weights(list(mapped.items()), strict=False)
    mx.eval(module.parameters())

    from mlx.utils import tree_flatten
    n_params = sum(v.size for _, v in tree_flatten(module.parameters()))
    _log(f"loaded {spec} — {len(mapped)} tensors, {n_params/1e9:.2f}B params, "
         f"{time.time()-t0:.1f}s (quantize={quantize})")
    return module
