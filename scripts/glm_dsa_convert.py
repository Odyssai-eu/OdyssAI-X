#!/usr/bin/env python3
"""glm_dsa_convert.py — conversion glm_moe_dsa (BF16 -> MLX quantifié), auditée.

2e famille du converter #48 (après bailing_hybrid) : GLM-5.1 et dérivés
(Macaron-V1). model_type glm_moe_dsa = sous-classe mlx-lm de deepseek_v32 :
MLA + indexer DSA + MoE sigmoid/noaux_tc + 1 couche MTP (nextn) à dropper.

Différences vs bailing_convert.py :
  * source BF16 (Macaron). Un GLM FP8 d'origine utilise weight_scale_inv
    par blocs 128 — NON géré ici : refus explicite plutôt que garbage.
  * absorption kv_b -> embed_q/unembed_out : math PORTÉE VERBATIM du sanitize
    mlx-lm deepseek_v32 (reshape (H, nope+v, lora), slice, swapaxes) — c'est la
    classe runtime elle-même qui fixe le format, on ne réinvente rien.
  * sortie = arbre POST-sanitize (celui de mlx_lm.convert) : experts stackés
    switch_mlp, embed_q/unembed_out absorbés, pas de kv_b ni d'experts.N.
  * indexer DSA stocké bf16 NON quantifié — le class_predicate de mlx-lm ne
    quantifie un module que si "<path>.scales" existe dans le checkpoint, donc
    aucun override de config n'est nécessaire : pas de scales => bf16 au load.
  * router MoEGate : gate.weight bf16 + e_score_correction_bias f32, bruts
    (le module n'a pas de to_quantized — jamais quantifié au runtime).
  * FUSION LoRA inline (--fuse-adapter lX) : W += (alpha/r)·B@A appliqué en f32
    AVANT stacking/absorption/quantification ; l'audit compare au poids FUSIONNÉ.
    Mapping PEFT -> base : strip "base_model.model." + remap
    ".mlp.shared_expert." -> ".mlp.shared_experts." (nommage training ≠ release).

Usage (sur .29) :
  ~/mlx-cluster/.venv/bin/python glm_dsa_convert.py \
      /Volumes/models/mlx/raw/mindlab-research/Macaron-V1-Preview-749B \
      /Volumes/models/odysseus/odyssai/Macaron-V1-Preview-749B-l0-mlx-6bit \
      --bits 6 --fuse-adapter l0
  # pilot : --limit-layers 5 (3 dense + 2 MoE, vérifie noms/shapes/audit/LoRA)
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from bailing_convert import Output, ShardReader, corr

AUDIT_THRESHOLD = 0.985


def qinto(
    out: dict[str, mx.array],
    name: str,
    w32: np.ndarray,
    gs: int,
    bits: int,
) -> tuple[np.ndarray, str]:
    """Quantifie un poids 2D/3D, retourne (dequant_np, nom) pour l'audit."""
    w = mx.array(w32).astype(mx.bfloat16)
    qw, sc, bi = mx.quantize(w, group_size=gs, bits=bits)
    out[f"{name}.weight"] = qw
    out[f"{name}.scales"] = sc
    out[f"{name}.biases"] = bi
    deq = np.array(
        mx.dequantize(qw, sc, bi, group_size=gs, bits=bits).astype(mx.float32)
    )
    return deq, name


class LoraFuser:
    """Deltas PEFT (lora_A/lora_B) appliqués au fil de la lecture.

    L'adaptateur entier tient en RAM (~2 GB) ; les noms sont remappés vers le
    nommage de la release et chaque delta consommé est compté — un delta jamais
    consommé en fin de run = drift de nommage = FAIL (pas d'à-peu-près sur 749B).
    """

    def __init__(self, adapter_dir: Path):
        acfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        self.scale = acfg["lora_alpha"] / acfg["r"]
        self.pairs: dict[str, dict[str, mx.array]] = {}
        data = mx.load(str(adapter_dir / "adapter_model.safetensors"))
        for k, v in data.items():
            for suffix, slot in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
                if k.endswith(suffix):
                    base = k[: -len(suffix)]
                    base = base.removeprefix("base_model.model.")
                    base = base.replace(".mlp.shared_expert.", ".mlp.shared_experts.")
                    self.pairs.setdefault(base, {})[slot] = v
        bad = [n for n, p in self.pairs.items() if "A" not in p or "B" not in p]
        if bad:
            raise SystemExit(f"adaptateur incomplet (A/B manquant): {bad[:4]}")
        self.applied: set[str] = set()
        print(f"adapter: {len(self.pairs)} modules cibles, scale={self.scale}", flush=True)

    def fuse(self, name: str, w32: np.ndarray) -> np.ndarray:
        p = self.pairs.get(name)
        if p is None:
            return w32
        a = np.array(p["A"].astype(mx.float32))
        b = np.array(p["B"].astype(mx.float32))
        self.applied.add(name)
        return w32 + self.scale * (b @ a)

    def report(self) -> dict:
        missed = sorted(set(self.pairs) - self.applied)
        return {
            "expected": len(self.pairs),
            "applied": len(self.applied),
            "missed": missed[:8],
        }


class NoFuser:
    scale = 0.0

    def fuse(self, name: str, w32: np.ndarray) -> np.ndarray:
        return w32

    def report(self) -> dict:
        return {"expected": 0, "applied": 0, "missed": []}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert a glm_moe_dsa BF16 checkpoint to MLX (audited)."
    )
    ap.add_argument("src", help="source dir (BF16 safetensors)")
    ap.add_argument("dst", help="output dir (MLX quantized)")
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--fuse-adapter", default=None,
                    help="sous-dossier LoRA à fuser (ex. l0) — relatif à src, ou chemin absolu")
    ap.add_argument("--limit-layers", type=int, default=0,
                    help="pilot : ne convertir que N couches puis s'arrêter (pas de top/config)")
    args = ap.parse_args()

    gs, bits = args.group_size, args.bits
    src, dst = Path(args.src), Path(args.dst)
    cfg = json.loads((src / "config.json").read_text())

    if cfg.get("model_type") != "glm_moe_dsa":
        raise SystemExit(f"model_type={cfg.get('model_type')} — ce converter est pour glm_moe_dsa.")

    H = cfg["num_attention_heads"]
    NOPE = cfg["qk_nope_head_dim"]
    VDIM = cfg["v_head_dim"]
    LORA = cfg["kv_lora_rank"]
    N_LAYERS = cfg["num_hidden_layers"]          # nextn = couche N_LAYERS, droppée
    N_EXPERTS = cfg["n_routed_experts"]
    FIRST_DENSE = cfg["first_k_dense_replace"]

    rd = ShardReader(src)
    # Refus explicite du FP8 GLM (weight_scale_inv par blocs ≠ weight_scale par canal).
    sample_shard = next(iter(set(rd.wm.values())))
    hdr, _ = rd._header(sample_shard)
    dts = {m["dtype"] for k, m in hdr.items() if k != "__metadata__"}
    if "F8_E4M3" in dts:
        raise SystemExit("source FP8 détectée — glm_dsa_convert ne gère que BF16 "
                         "(le FP8 GLM est en weight_scale_inv blockwise).")

    fuser = NoFuser()
    adapter_path = None
    if args.fuse_adapter:
        adapter_path = Path(args.fuse_adapter)
        if not adapter_path.is_absolute():
            adapter_path = src / args.fuse_adapter
        fuser = LoraFuser(adapter_path)

    out = Output(dst)
    t0 = time.time()
    worst: tuple[float, str] = (1.0, "")
    n_do = args.limit_layers or N_LAYERS
    print(f"src={src.name} bits={bits}/g{gs} layers={N_LAYERS} (run {n_do}) "
          f"experts={N_EXPERTS} dense<{FIRST_DENSE} H={H} nope={NOPE} v={VDIM} "
          f"kv_lora={LORA} adapter={adapter_path.name if adapter_path else 'none'}",
          flush=True)

    def audit(name: str, deq: np.ndarray, ref: np.ndarray) -> None:
        nonlocal worst
        c = corr(deq, ref)
        out.audit.append({"tensor": name, "corr": round(c, 6)})
        if c < worst[0]:
            worst = (c, name)
        if c < AUDIT_THRESHOLD:
            raise SystemExit(
                f"AUDIT FAIL: {name} corr={c:.4f} < {AUDIT_THRESHOLD} — conversion invalide."
            )

    def read_fused(name: str) -> np.ndarray:
        """name SANS suffixe .weight — lit + fuse l'éventuel delta LoRA."""
        return fuser.fuse(name, rd.read(f"{name}.weight"))

    for layer in range(n_do):
        tag = f"{layer:05d}"
        if out.done(tag):
            path = dst / out.shard_name(tag)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            for k, meta in hdr.items():
                if k == "__metadata__":
                    continue
                out.weight_map[k] = out.shard_name(tag)
                o0, o1 = meta["data_offsets"]
                out.total += o1 - o0
            print(f"[{layer:02d}] déjà fait — skip", flush=True)
            continue

        P = f"model.layers.{layer}"
        A = f"{P}.self_attn"
        tensors: dict[str, mx.array] = {}

        # — attention MLA (toutes les couches, pas de double layout ici) —
        for sub in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
            w = read_fused(f"{A}.{sub}")
            deq, nm = qinto(tensors, f"{A}.{sub}", w, gs, bits)
            if sub == "o_proj":
                audit(nm, deq, w)
        for sub in ("q_a_layernorm", "kv_a_layernorm"):
            tensors[f"{A}.{sub}.weight"] = mx.array(
                rd.read(f"{A}.{sub}.weight")
            ).astype(mx.bfloat16)

        # — indexer DSA : bf16 passthrough, jamais quantifié —
        for sub in ("wq_b.weight", "wk.weight", "weights_proj.weight",
                    "k_norm.weight", "k_norm.bias"):
            tensors[f"{A}.indexer.{sub}"] = mx.array(
                rd.read(f"{A}.indexer.{sub}")
            ).astype(mx.bfloat16)

        # — absorption kv_b -> embed_q / unembed_out (sanitize dsv32, verbatim) —
        kvb = read_fused(f"{A}.kv_b_proj").reshape(H, NOPE + VDIM, LORA)
        emb = np.ascontiguousarray(np.swapaxes(kvb[:, :NOPE, :], -1, -2))  # (H, LORA, NOPE)
        une = np.ascontiguousarray(kvb[:, NOPE:, :])                       # (H, V, LORA)
        deq, nm = qinto(tensors, f"{A}.embed_q", emb, gs, bits)
        audit(nm, deq, emb)
        deq, nm = qinto(tensors, f"{A}.unembed_out", une, gs, bits)
        audit(nm, deq, une)
        del kvb, emb, une

        # — mlp —
        if layer < FIRST_DENSE:
            for sub in ("gate_proj", "down_proj", "up_proj"):
                w = read_fused(f"{P}.mlp.{sub}")
                qinto(tensors, f"{P}.mlp.{sub}", w, gs, bits)
        else:
            # router MoEGate : poids bruts (pas de to_quantized côté mlx-lm)
            tensors[f"{P}.mlp.gate.weight"] = mx.array(
                rd.read(f"{P}.mlp.gate.weight")
            ).astype(mx.bfloat16)
            tensors[f"{P}.mlp.gate.e_score_correction_bias"] = mx.array(
                rd.read(f"{P}.mlp.gate.e_score_correction_bias")
            )  # f32 — précision du routing
            for sub in ("gate_proj", "down_proj", "up_proj"):
                w = read_fused(f"{P}.mlp.shared_experts.{sub}")
                qinto(tensors, f"{P}.mlp.shared_experts.{sub}", w, gs, bits)
            for sub in ("gate_proj", "down_proj", "up_proj"):
                stack = np.stack([
                    read_fused(f"{P}.mlp.experts.{e}.{sub}")
                    for e in range(N_EXPERTS)
                ])
                deq, nm = qinto(tensors, f"{P}.mlp.switch_mlp.{sub}", stack, gs, bits)
                if sub == "gate_proj":
                    audit(nm, deq, stack)
                del stack, deq

        for sub in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"{P}.{sub}.weight"] = mx.array(
                rd.read(f"{P}.{sub}.weight")
            ).astype(mx.bfloat16)

        out.write_shard(tag, tensors)
        del tensors
        mx.clear_cache()
        el = time.time() - t0
        kind = "dense" if layer < FIRST_DENSE else "moe"
        print(f"[{layer:02d}/{N_LAYERS}] {kind} ok — {el/60:.1f} min, "
              f"worst corr {worst[0]:.4f} ({worst[1]}), "
              f"lora {len(fuser.applied) if hasattr(fuser, 'applied') else 0}",
              flush=True)

    if args.limit_layers:
        rep = fuser.report()
        print(f"PILOT OK — {args.limit_layers} couches, worst corr {worst[0]:.4f} "
              f"({worst[1]}), lora applied {rep['applied']}", flush=True)
        return 0

    # ---- top-level -----------------------------------------------------------
    if not out.done("top"):
        tensors = {}
        w = rd.read("model.embed_tokens.weight")
        deq, nm = qinto(tensors, "model.embed_tokens", w, gs, bits)
        audit(nm, deq, w)
        w = read_fused("lm_head")
        deq, nm = qinto(tensors, "lm_head", w, gs, bits)
        audit(nm, deq, w)
        tensors["model.norm.weight"] = mx.array(
            rd.read("model.norm.weight")
        ).astype(mx.bfloat16)
        out.write_shard("top", tensors)
        mx.clear_cache()
        print("[top] ok", flush=True)

    # ---- garde-fou fusion : tout delta non consommé = drift de nommage ------
    rep = fuser.report()
    if rep["expected"] and rep["applied"] != rep["expected"]:
        raise SystemExit(f"FUSION FAIL: {rep['applied']}/{rep['expected']} deltas "
                         f"appliqués — manquants: {rep['missed']}")

    # ---- config / tokenizer / index / manifest ------------------------------
    cfg.pop("quantization_config", None)
    cfg.pop("auto_map", None)
    cfg["num_nextn_predict_layers"] = 0
    q = {"group_size": gs, "bits": bits, "mode": "affine"}
    q.update(out.quant_overrides)
    cfg["quantization"] = q
    cfg["quantization_config"] = dict(q)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    for f in src.glob("*"):
        if f.name.startswith("tokenizer") or f.name in (
            "special_tokens_map.json", "generation_config.json", "chat_template.jinja",
        ):
            shutil.copy(f, dst / f.name)

    index = {"metadata": {"total_size": out.total}, "weight_map": out.weight_map}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))

    manifest = {
        "tool": "glm_dsa_convert.py v1 (famille 2 odyssai-convert, Odysseus#48)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": "ultra-512",
        "source": str(src),
        "source_format": "BF16 safetensors",
        "model_type": cfg.get("model_type"),
        "fused_adapter": None if adapter_path is None else {
            "path": str(adapter_path), "scale": fuser.scale, **rep,
        },
        "recipe": {"bits": bits, "group_size": gs,
                   "indexer": "bf16 (non quantifié)",
                   "router_gate": "bf16 + e_score_correction_bias f32 (bruts)",
                   "mtp_dropped": True,
                   "absorption": "kv_b->embed_q/unembed_out (mlx-lm dsv32 verbatim)",
                   "dims": {"heads": H, "qk_nope": NOPE, "v": VDIM, "kv_lora": LORA,
                            "layers": N_LAYERS, "experts": N_EXPERTS,
                            "first_dense": FIRST_DENSE}},
        "audit": {"threshold": AUDIT_THRESHOLD,
                  "worst": {"corr": worst[0], "tensor": worst[1]},
                  "samples": out.audit},
    }
    (dst / "conversion-manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"TERMINÉ en {(time.time()-t0)/60:.1f} min — worst corr {worst[0]:.4f} ({worst[1]})")
    print(f"sortie: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
