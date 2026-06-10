#!/usr/bin/env python3
"""ring26_convert_q6.py — conversion Ring-2.6-1T FP8 -> MLX Q6/g64, auditée.

Seed du futur odyssai-convert (Odysseus#48, WU1+WU2). Tourne en local sur
ultra-512. Leçons d'avril intégrées (#43) :
  * décodage FP8 E4M3 + weight_scale par canal (LUT, jamais de cast naïf)
  * absorption kv_b_proj -> embed_q (transposé) / unembed_out — la transform
    validée par golden test torch-vs-mlx (0,2 %)
  * drop des couches MTP (nextn, layers.80)
  * AUDIT INLINE : à chaque couche, corrélation dequant(Q6) vs source — la
    conversion s'invalide d'elle-même sous le seuil
  * manifest de provenance écrit dans la sortie
  * résumable : une couche déjà écrite est sautée

Usage (sur .29):
  ~/mlx-cluster/.venv/bin/python ring26_convert_q6.py \
      /Volumes/models/mlx/safe/inclusionAI/Ring-2.6-1T \
      /Volumes/models/odysseus/odyssai/Ring-2.6-1T-mlx-6bit
"""

from __future__ import annotations

import json
import shutil
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

GS, BITS = 64, 6
GATE_GS, GATE_BITS = 64, 8
AUDIT_THRESHOLD = 0.985   # corr minimale dequant(Q6) vs source par tenseur audité
H, NOPE, VDIM, LORA = 64, 128, 128, 512   # Ring-2.6-1T (mêmes dims que 2.5)
N_LAYERS, N_EXPERTS, FIRST_DENSE = 80, 256, 4

# ---------------------------------------------------------------- lecture src

_LUT = np.zeros(256, dtype=np.float32)
for _b in range(256):
    _s = -1.0 if _b & 0x80 else 1.0
    _e = (_b >> 3) & 0xF
    _m = _b & 7
    if _e == 0:
        _v = _s * (_m / 8.0) * 2.0**-6
    elif _e == 15 and _m == 7:
        _v = np.nan
    else:
        _v = _s * (1 + _m / 8.0) * 2.0 ** (_e - 7)
    _LUT[_b] = _v


class ShardReader:
    """Lecture brute safetensors (headers parsés à la main : F8/BF16 sûrs)."""

    def __init__(self, root: Path):
        self.root = root
        self.wm = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self._headers: dict[str, tuple[dict, int]] = {}

    def _header(self, shard: str) -> tuple[dict, int]:
        if shard not in self._headers:
            with open(self.root / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._headers[shard] = (json.loads(f.read(n)), 8 + n)
        return self._headers[shard]

    def read(self, name: str) -> np.ndarray:
        """Tenseur décodé en float32 — FP8 déquantifié AVEC son weight_scale."""
        shard = self.wm[name]
        hdr, d0 = self._header(shard)
        meta = hdr[name]
        o0, o1 = meta["data_offsets"]
        with open(self.root / shard, "rb") as f:
            f.seek(d0 + o0)
            raw = f.read(o1 - o0)
        dt = meta["dtype"]
        if dt == "F8_E4M3":
            w = _LUT[np.frombuffer(raw, dtype=np.uint8)].reshape(meta["shape"])
            sname = name.replace(".weight", ".weight_scale")
            if sname in self.wm:
                w = w * self.read(sname)
            return w
        if dt == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(meta["shape"])
        if dt == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(meta["shape"]).copy()
        raise ValueError(f"{name}: dtype {dt} inattendu")

    def has(self, name: str) -> bool:
        return name in self.wm


# ---------------------------------------------------------------- écriture

class Output:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.weight_map: dict[str, str] = {}
        self.total = 0
        self.quant_overrides: dict[str, dict] = {}
        self.audit: list[dict] = []

    def shard_name(self, tag: str) -> str:
        return f"model-{tag}.safetensors"

    def write_shard(self, tag: str, tensors: dict[str, mx.array]) -> None:
        path = self.root / self.shard_name(tag)
        mx.save_safetensors(str(path), tensors)
        for k, v in tensors.items():
            self.weight_map[k] = self.shard_name(tag)
            self.total += v.nbytes

    def done(self, tag: str) -> bool:
        return (self.root / self.shard_name(tag)).exists()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def quantize_into(
    out: dict[str, mx.array],
    name: str,
    w32: np.ndarray,
    overrides: dict[str, dict],
    gs: int = GS,
    bits: int = BITS,
) -> tuple[np.ndarray, str]:
    """Quantifie un poids 2D/3D ; retourne (dequant_np, nom) pour l'audit."""
    w = mx.array(w32).astype(mx.bfloat16)
    qw, sc, bi = mx.quantize(w, group_size=gs, bits=bits)
    out[f"{name}.weight"] = qw
    out[f"{name}.scales"] = sc
    out[f"{name}.biases"] = bi
    if (gs, bits) != (GS, BITS):
        overrides[name] = {"group_size": gs, "bits": bits, "mode": "affine"}
    deq = np.array(
        mx.dequantize(qw, sc, bi, group_size=gs, bits=bits).astype(mx.float32)
    )
    return deq, name


def main() -> int:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    rd = ShardReader(src)
    out = Output(dst)
    t0 = time.time()
    worst: tuple[float, str] = (1.0, "")

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

    # ---- couches 0..79 (80 = MTP, droppée) --------------------------------
    for layer in range(N_LAYERS):
        tag = f"{layer:05d}"
        if out.done(tag):
            # reconstituer weight_map/total depuis le shard existant (reprise)
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
        tensors: dict[str, mx.array] = {}
        is_global = (layer + 1) % 8 == 0

        # — attention —
        if is_global:
            for sub in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "dense"):
                w = rd.read(f"{P}.attention.{sub}.weight")
                deq, nm = quantize_into(tensors, f"{P}.attention.{sub}", w, out.quant_overrides)
                if sub == "dense":
                    audit(nm, deq, w)
            for sub in ("q_a_layernorm", "kv_a_layernorm"):
                tensors[f"{P}.attention.{sub}.weight"] = mx.array(
                    rd.read(f"{P}.attention.{sub}.weight")
                ).astype(mx.bfloat16)
            # absorption kv_b -> embed_q / unembed_out (transform du golden test)
            kvb = rd.read(f"{P}.attention.kv_b_proj.weight").reshape(H, NOPE + VDIM, LORA)
            emb = np.swapaxes(kvb[:, :NOPE, :], -1, -2)   # (H, LORA, NOPE)
            une = kvb[:, NOPE:, :]                        # (H, V, LORA)
            deq, nm = quantize_into(tensors, f"{P}.attention.embed_q", emb, out.quant_overrides)
            audit(nm, deq, emb)
            deq, nm = quantize_into(tensors, f"{P}.attention.unembed_out", une, out.quant_overrides)
            audit(nm, deq, une)
        else:
            for sub in ("query_key_value", "g_proj", "dense"):
                w = rd.read(f"{P}.attention.{sub}.weight")
                deq, nm = quantize_into(tensors, f"{P}.attention.{sub}", w, out.quant_overrides)
                if sub == "query_key_value":
                    audit(nm, deq, w)
            for sub in ("g_norm", "query_layernorm", "key_layernorm"):
                tensors[f"{P}.attention.{sub}.weight"] = mx.array(
                    rd.read(f"{P}.attention.{sub}.weight")
                ).astype(mx.bfloat16)

        # — mlp —
        if layer < FIRST_DENSE:
            for sub in ("gate_proj", "down_proj", "up_proj"):
                w = rd.read(f"{P}.mlp.{sub}.weight")
                quantize_into(tensors, f"{P}.mlp.{sub}", w, out.quant_overrides)
        else:
            # gate (8-bit) + expert_bias (fp32 brut)
            gw = rd.read(f"{P}.mlp.gate.weight")
            quantize_into(tensors, f"{P}.mlp.gate.gate_proj", gw, out.quant_overrides,
                          gs=GATE_GS, bits=GATE_BITS)
            if rd.has(f"{P}.mlp.gate.expert_bias"):
                tensors[f"{P}.mlp.gate.expert_bias"] = mx.array(
                    rd.read(f"{P}.mlp.gate.expert_bias")
                )  # fp32 (cast_predicate du modèle)
            # shared experts
            for sub in ("gate_proj", "down_proj", "up_proj"):
                w = rd.read(f"{P}.mlp.shared_experts.{sub}.weight")
                quantize_into(tensors, f"{P}.mlp.shared_experts.{sub}", w, out.quant_overrides)
            # experts -> stack switch_mlp (audité sur gate_proj)
            for sub in ("gate_proj", "down_proj", "up_proj"):
                stack = np.stack(
                    [rd.read(f"{P}.mlp.experts.{e}.{sub}.weight") for e in range(N_EXPERTS)]
                )
                deq, nm = quantize_into(tensors, f"{P}.mlp.switch_mlp.{sub}", stack, out.quant_overrides)
                if sub == "gate_proj":
                    audit(nm, deq, stack)
                del stack, deq

        # — norms de couche —
        for sub in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"{P}.{sub}.weight"] = mx.array(
                rd.read(f"{P}.{sub}.weight")
            ).astype(mx.bfloat16)

        out.write_shard(tag, tensors)
        del tensors
        mx.clear_cache()
        el = time.time() - t0
        print(f"[{layer:02d}/{N_LAYERS}] {'A' if is_global else 'L'} ok — {el/60:.1f} min, worst corr {worst[0]:.4f} ({worst[1]})", flush=True)

    # ---- top-level ----------------------------------------------------------
    if not out.done("top"):
        tensors = {}
        w = rd.read("model.word_embeddings.weight")
        deq, nm = quantize_into(tensors, "model.word_embeddings", w, out.quant_overrides)
        audit(nm, deq, w)
        w = rd.read("lm_head.weight")
        deq, nm = quantize_into(tensors, "lm_head", w, out.quant_overrides)
        audit(nm, deq, w)
        tensors["model.norm.weight"] = mx.array(rd.read("model.norm.weight")).astype(mx.bfloat16)
        out.write_shard("top", tensors)
        mx.clear_cache()
        print("[top] ok", flush=True)

    # ---- config / tokenizer / index / manifest ------------------------------
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)
    cfg.pop("auto_map", None)
    cfg["num_nextn_predict_layers"] = 0
    q: dict = {"group_size": GS, "bits": BITS, "mode": "affine"}
    q.update(out.quant_overrides)
    cfg["quantization"] = q
    cfg["quantization_config"] = q  # convention mlx-community (les deux clés)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    for f in src.glob("*"):
        if f.name.startswith("tokenizer") or f.name in (
            "special_tokens_map.json", "generation_config.json", "chat_template.jinja",
        ):
            shutil.copy(f, dst / f.name)

    index = {
        "metadata": {"total_size": out.total},
        "weight_map": out.weight_map,
    }
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))

    manifest = {
        "tool": "ring26_convert_q6.py v0 (seed odyssai-convert, Odysseus#48)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": "ultra-512",
        "source": str(src),
        "source_format": "FP8 compressed-tensors (E4M3 + weight_scale)",
        "recipe": {"bits": BITS, "group_size": GS,
                   "gate_override": {"bits": GATE_BITS, "group_size": GATE_GS},
                   "mtp_dropped": True, "absorption": "kv_b->embed_q/unembed_out"},
        "audit": {"threshold": AUDIT_THRESHOLD,
                  "worst": {"corr": worst[0], "tensor": worst[1]},
                  "samples": out.audit},
    }
    (dst / "conversion-manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"TERMINÉ en {(time.time()-t0)/60:.1f} min — worst corr {worst[0]:.4f} ({worst[1]})")
    print(f"sortie: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
