#!/usr/bin/env python3
"""mistral_convert.py — Mistral 3.5 (dense, model_type mistral3) -> MLX quantisé.

Mistral 3.5 est DENSE -> pas de switch_mlp/MSA a gerer (contrairement a
m3_convert, cable pour le MoE/MSA de MiniMax-M3). On s'appuie donc directement
sur `mlx_lm.convert` (qui supporte nativement `mistral3`) + un `quant_predicate`
pour la tete bf16.

Recette :
  - corps quantifie a --bits (defaut 6), group-size 64 ;
  - --head-bits 16  -> embed_tokens + lm_head gardes en **bf16** (tete pleine
    precision : le head-bf16 qui a marche sur M3, le polish du residu quant) ;
  - --head-bits 8   -> tete quantifiee a 8-bit (corps 6-bit), variante mixte.

Sortie = dossier plat MLX (config.json + *.safetensors), loadable par :
  - Python mlx-lm : par chemin ;
  - Telemak Swift : par **chemin absolu** (ModelLoader accepte le path direct,
    cf Telemak/Engine/ModelLoader.swift l.23) -> les DEUX moteurs chargent les
    memes poids, indispensable pour le bench ratio Swift/Python (WU0).

Usage (sur .32, ou le raw est telecharge) :
  ~/mlx-cluster/.venv/bin/python mistral_convert.py \
      /Volumes/models/odysseus/mistralai/Mistral-Medium-3.5-128B \
      /Volumes/models/odysseus/odyssai/Mistral-Medium-3.5-128B-MLX-6bit --bits 6
  # head-bf16 :  ... -MLX-6bit-headbf16 --bits 6 --head-bits 16
"""
import argparse

from mlx_lm import convert


def _head_predicate(head_bits: int, body_bits: int, group_size: int):
    """quant_predicate pour mlx_lm.convert : (path, module, config) -> bool|dict.
    False = ne pas quantifier (reste bf16) ; dict = quantifier avec ces params ;
    True = quantifier au --bits global. `*_` absorbe l'arite (2 ou 3 args)."""
    if head_bits == body_bits:
        return None  # uniforme, pas besoin de predicat

    def predicate(path, module, *_):
        is_head = ("embed_tokens" in path) or ("lm_head" in path)
        if not is_head:
            return True  # corps -> quantifie a body_bits
        if head_bits >= 16:
            return False  # tete -> bf16 (skip quant)
        return {"bits": head_bits, "group_size": group_size}  # tete -> head_bits

    return predicate


def main() -> int:
    ap = argparse.ArgumentParser(description="Mistral 3.5 dense -> MLX quantisé")
    ap.add_argument("src", help="dossier HF du raw bf16 (mistral3)")
    ap.add_argument("dst", help="dossier de sortie MLX")
    ap.add_argument("--bits", type=int, default=6, help="bits du corps (defaut 6)")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--head-bits", type=int, default=0,
                    help="bits pour embed_tokens + lm_head (0 = comme --bits ; "
                         ">=16 = garder bf16 / pleine precision = head-bf16)")
    args = ap.parse_args()
    head_bits = args.head_bits or args.bits

    qp = _head_predicate(head_bits, args.bits, args.group_size)
    note = "" if qp is None else (" head=bf16" if head_bits >= 16 else f" head={head_bits}b")
    print(f"src={args.src}")
    print(f"dst={args.dst}")
    print(f"bits={args.bits}/g{args.group_size}{note}  predicate={'oui' if qp else 'non'}")

    convert(
        args.src,
        args.dst,
        quantize=True,
        q_bits=args.bits,
        q_group_size=args.group_size,
        dtype="bfloat16",          # non-quantifie (tete head-bf16, normes) reste bf16
        quant_predicate=qp,
    )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
