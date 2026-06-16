#!/usr/bin/env python3
"""mistral_eagle_convert.py — draft EAGLE Mistral 3.5 (natif fp8) -> dir HF plat bf16.

Le draft EAGLE de Mistral 3.5 est livré en format **natif Mistral** : `params.json`
+ `consolidated.safetensors` (poids **fp8 e4m3**, scales par-tenseur) + `tekken.json`.
Telemak ne sait pas le charger (il exige un `config.json` HF + safetensors lisibles).

Ce script déquantifie et reformate, SANS rien réécrire d'autre :
  - **déquant fp8** : `w_bf16 = w_fp8.astype(f32) * qscale_weight` (scale SCALAIRE par
    tenseur). `qscale_act` est ignoré (quant d'activation runtime, inutile en bf16).
  - **noms de clés gardés tels quels** (le futur `Mistral3Eagle.swift` les mappe via
    `@ModuleInfo(key:)`) : `eagle_linear`, `layers.N.attention.{wq,wk,wv,wo}`,
    `layers.N.feed_forward.w{1,2,3}`, `layers.N.{attention_norm,ffn_norm}`, `norm`.
  - **config.json** synthétisé (`model_type: mistral3_eagle`) pour que MTPModelLoader
    + MTPCompatibility reconnaissent le draft (sidecar).

Le draft réutilise le **tokenizer ET la lm_head de la cible** -> on ne sort ni l'un
ni l'autre (le draft EAGLE n'a ni embed_tokens ni lm_head). Sortie = dossier plat
loadable par Telemak (chemin absolu) et par `Mistral3Eagle.swift`.

Usage (sur .32, ou le draft natif est présent) :
  ~/mlx-cluster/.venv/bin/python mistral_eagle_convert.py \
      /Volumes/models/odysseus/mistralai/Mistral-Medium-3.5-128B-EAGLE \
      /Volumes/models/odysseus/odyssai/Mistral-Medium-3.5-128B-EAGLE-mlx
"""
import argparse
import json
import os
import struct

import mlx.core as mx
import numpy as np


def _e4m3fn_table():
    """Table de décodage des 256 octets fp8 **e4m3fn** (OCP) -> float32. e4m3fn :
    1 signe / 4 exposant (biais 7) / 3 mantisse ; pas d'inf ; `S.1111.111` = NaN ;
    max normal 448. Évite la dépendance `ml_dtypes` (absente du venv cluster)."""
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if (b >> 7) & 1 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            t[b] = s * (2.0 ** -6) * (m / 8.0)              # sous-normaux
        elif e == 0xF and m == 0x7:
            t[b] = np.nan                                   # e4m3fn NaN
        else:
            t[b] = s * (2.0 ** (e - 7)) * (1.0 + m / 8.0)   # normaux
    return t


_E4M3FN = _e4m3fn_table()


def _bf16_to_f32(buf):
    """bf16 = les 16 bits de poids fort d'un float32 -> shift gauche de 16 bits."""
    return (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def read_native_safetensors(path):
    """Lecture brute du `consolidated.safetensors` natif Mistral. fp8 e4m3 et bf16
    ne sont pas gérés par numpy standard -> décodage maison. Tout promu en f32."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        data_start = 8 + hlen
        out = {}
        for k, v in header.items():
            if k == "__metadata__":
                continue
            dt, shape = v["dtype"], v["shape"]
            s, e = v["data_offsets"]
            f.seek(data_start + s)
            buf = f.read(e - s)
            if dt == "F8_E4M3":
                a = _E4M3FN[np.frombuffer(buf, dtype=np.uint8)]
            elif dt == "BF16":
                a = _bf16_to_f32(buf)
            elif dt == "F16":
                a = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
            elif dt == "F32":
                a = np.frombuffer(buf, dtype=np.float32).copy()
            else:
                raise ValueError(f"dtype non géré {dt} pour {k}")
            out[k] = a.reshape(shape) if shape else a.reshape(())
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="dossier draft EAGLE natif (params.json + consolidated)")
    ap.add_argument("dst", help="dossier de sortie MLX plat")
    args = ap.parse_args()

    with open(os.path.join(args.src, "params.json")) as fh:
        params = json.load(fh)
    raw = read_native_safetensors(os.path.join(args.src, "consolidated.safetensors"))

    weights = {}
    n_dequant = n_kept = 0
    for k, v in raw.items():
        if k.endswith(".qscale_act") or k.endswith(".qscale_weight"):
            continue  # consommés via le .weight correspondant / ignorés
        scale_key = k[: -len(".weight")] + ".qscale_weight" if k.endswith(".weight") else None
        if scale_key and scale_key in raw:
            arr = v.astype(np.float32) * float(raw[scale_key])  # déquant fp8, scale scalaire
            n_dequant += 1
        else:
            arr = v.astype(np.float32)  # normes bf16, gardées
            n_kept += 1
        weights[k] = mx.array(np.ascontiguousarray(arr)).astype(mx.bfloat16)

    os.makedirs(args.dst, exist_ok=True)
    mx.save_safetensors(os.path.join(args.dst, "model.safetensors"), weights)

    yarn = params.get("yarn")
    cfg = {
        "model_type": "mistral3_eagle",
        "architectures": ["Mistral3EagleDraftModel"],
        "hidden_size": params["dim"],
        "num_hidden_layers": params["n_layers"],
        "num_attention_heads": params["n_heads"],
        "num_key_value_heads": params["n_kv_heads"],
        "head_dim": params["head_dim"],
        "intermediate_size": params["hidden_dim"],
        "rope_theta": params["rope_theta"],
        "rms_norm_eps": params["norm_eps"],
        "vocab_size": params["vocab_size"],
        "max_position_embeddings": params["max_position_embeddings"],
        "tie_word_embeddings": params.get("tied_embeddings", False),
        "rope_scaling": ({"rope_type": "yarn", **yarn} if yarn else None),
        "torch_dtype": "bfloat16",
    }
    with open(os.path.join(args.dst, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"OK -> {args.dst}")
    print(f"  {len(weights)} tenseurs bf16 ({n_dequant} déquantifiés fp8, {n_kept} gardés)")
    print(f"  config.json: model_type=mistral3_eagle, {cfg['num_hidden_layers']} couches, "
          f"dim {cfg['hidden_size']}, GQA {cfg['num_attention_heads']}/{cfg['num_key_value_heads']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
