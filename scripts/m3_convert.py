#!/usr/bin/env python3
"""m3_convert.py — conversion minimax_m3_vl (BF16 -> MLX quantisé), auditée.

3e famille du converter #48 (après bailing_hybrid et glm_moe_dsa) :
MiniMax-M3 428B A23B. Tour TEXTE seule — la tour vision et le projector sont
droppés (mlx-vlm viendra plus tard) ; aucun poids MTP dans le checkpoint.

Sortie = nommage hub conservé (model.layers.N.self_attn.*, block_sparse_moe.*)
avec les experts STACKÉS en switch_mlp.{gate,down,up}_proj quantifiés — le
layout que minimax_m3.py consomme directement (son sanitize passe les noms
stackés tels quels). Config aplati : text_config -> top-level, model_type
"minimax_m3", layer_types/mlp_layer_types EXPLICITES, champs index_* plats.

Recette : corps Q6/g64 ; indexer MSA gardé en bf16 (la sélection de blocs est
le système nerveux du modèle — 57x(512+128)x6144 poids, ~0,5 GB, le prix de
la sérénité) ; router gate + e_score_correction_bias bruts (module custom,
jamais quantifié au runtime) ; normes bf16 ; embed/lm_head quantifiés.

Audit inline par couche (corr dequant vs source >= seuil), manifest de
provenance, resumable (shard par couche).

Usage (sur .29) :
  ~/mlx-cluster/.venv/bin/python m3_convert.py \
      /Volumes/models/mlx/raw/MiniMaxAI/MiniMax-M3 \
      /Volumes/models/odysseus/odyssai/MiniMax-M3-mlx-6bit --bits 6
  # pilot : --limit-layers 4
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

AUDIT_THRESHOLD = 0.985


class ShardReader:
    """Lecture brute safetensors (BF16/F32) — headers parsés à la main."""

    def __init__(self, root: Path):
        self.root = root
        self.wm = json.loads(
            (root / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}

    def _header(self, shard: str):
        if shard not in self._headers:
            with open(self.root / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._headers[shard] = (json.loads(f.read(n)), 8 + n)
        return self._headers[shard]

    def read(self, name: str) -> np.ndarray:
        shard = self.wm[name]
        hdr, d0 = self._header(shard)
        meta = hdr[name]
        o0, o1 = meta["data_offsets"]
        with open(self.root / shard, "rb") as f:
            f.seek(d0 + o0)
            raw = f.read(o1 - o0)
        dt = meta["dtype"]
        if dt == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(meta["shape"])
        if dt == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(meta["shape"]).copy()
        raise ValueError(f"{name}: dtype {dt} inattendu (BF16/F32 attendus)")

    def has(self, name: str) -> bool:
        return name in self.wm


class Output:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.weight_map: dict[str, str] = {}
        self.total = 0
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


def qinto(out, name, w32, gs, bits):
    w = mx.array(w32).astype(mx.bfloat16)
    qw, sc, bi = mx.quantize(w, group_size=gs, bits=bits)
    out[f"{name}.weight"] = qw
    out[f"{name}.scales"] = sc
    out[f"{name}.biases"] = bi
    deq = np.array(
        mx.dequantize(qw, sc, bi, group_size=gs, bits=bits).astype(mx.float32)
    )
    return deq, name


def keep_bf16(out, name, w32):
    out[name] = mx.array(w32).astype(mx.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--limit-layers", type=int, default=0)
    args = ap.parse_args()

    gs, bits = args.group_size, args.bits
    src, dst = Path(args.src), Path(args.dst)
    full_cfg = json.loads((src / "config.json").read_text())
    if full_cfg.get("model_type") not in ("minimax_m3_vl", "minimax_m3"):
        raise SystemExit(f"model_type={full_cfg.get('model_type')} — converter minimax_m3.")
    tc = full_cfg.get("text_config", full_cfg)

    NL = tc["num_hidden_layers"]
    NE = tc["num_local_experts"]
    sc = tc.get("sparse_attention_config", {})
    sparse_freq = sc.get("sparse_attention_freq") or tc.get("layer_types")
    if sparse_freq and isinstance(sparse_freq[0], str):
        layer_types = sparse_freq
    else:
        layer_types = [
            "minimax_m3_sparse" if f else "full_attention" for f in sparse_freq
        ]
    mfreq = tc.get("moe_layer_freq")
    if isinstance(mfreq[0], str):
        mlp_layer_types = mfreq
    else:
        mlp_layer_types = ["sparse" if f else "dense" for f in mfreq]

    rd = ShardReader(src)
    out = Output(dst)
    t0 = time.time()
    worst: tuple[float, str] = (1.0, "")
    n_do = args.limit_layers or NL
    print(f"src={src.name} bits={bits}/g{gs} layers={NL} (run {n_do}) experts={NE} "
          f"sparse={sum(1 for t in layer_types if 'sparse' in t)} "
          f"moe={sum(1 for t in mlp_layer_types if t == 'sparse')}", flush=True)

    def audit(name, deq, ref):
        nonlocal worst
        c = corr(deq, ref)
        out.audit.append({"tensor": name, "corr": round(c, 6)})
        if c < worst[0]:
            worst = (c, name)
        if c < AUDIT_THRESHOLD:
            raise SystemExit(f"AUDIT FAIL: {name} corr={c:.4f} — conversion invalide.")

    H = "language_model."  # préfixe hub, strippé en sortie

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

        # — attention GQA —
        for sub in ("q_proj", "k_proj", "v_proj", "o_proj"):
            w = rd.read(f"{H}{A}.{sub}.weight")
            deq, nm = qinto(tensors, f"{A}.{sub}", w, gs, bits)
            if sub == "o_proj":
                audit(nm, deq, w)
        for sub in ("q_norm", "k_norm"):
            keep_bf16(tensors, f"{A}.{sub}.weight", rd.read(f"{H}{A}.{sub}.weight"))

        # — indexer MSA : bf16 intégral (le système nerveux de la sélection) —
        if layer_types[layer] == "minimax_m3_sparse":
            for sub in ("index_q_proj.weight", "index_k_proj.weight",
                        "index_q_norm.weight", "index_k_norm.weight"):
                keep_bf16(tensors, f"{A}.{sub}", rd.read(f"{H}{A}.{sub}"))

        # — mlp —
        if mlp_layer_types[layer] == "dense":
            for sub in ("gate_proj", "down_proj", "up_proj"):
                w = rd.read(f"{H}{P}.mlp.{sub}.weight")
                qinto(tensors, f"{P}.mlp.{sub}", w, gs, bits)
        else:
            M = f"{P}.block_sparse_moe"
            keep_bf16(tensors, f"{M}.gate.weight", rd.read(f"{H}{M}.gate.weight"))
            tensors[f"{M}.gate.e_score_correction_bias"] = mx.array(
                rd.read(f"{H}{M}.e_score_correction_bias")
            )  # fp32 brut — précision du routing
            for sub in ("gate_proj", "down_proj", "up_proj"):
                w = rd.read(f"{H}{M}.shared_experts.{sub}.weight")
                qinto(tensors, f"{M}.shared_experts.{sub}", w, gs, bits)
            for w_name, out_name in (("w1", "gate_proj"), ("w2", "down_proj"),
                                     ("w3", "up_proj")):
                stack = np.stack([
                    rd.read(f"{H}{M}.experts.{e}.{w_name}.weight")
                    for e in range(NE)
                ])
                deq, nm = qinto(tensors, f"{M}.switch_mlp.{out_name}", stack, gs, bits)
                if w_name == "w1":
                    audit(nm, deq, stack)
                del stack, deq

        for sub in ("input_layernorm", "post_attention_layernorm"):
            keep_bf16(tensors, f"{P}.{sub}.weight", rd.read(f"{H}{P}.{sub}.weight"))

        out.write_shard(tag, tensors)
        del tensors
        mx.clear_cache()
        el = time.time() - t0
        print(f"[{layer:02d}/{NL}] {mlp_layer_types[layer]} ok — {el/60:.1f} min, "
              f"worst corr {worst[0]:.4f} ({worst[1]})", flush=True)

    if args.limit_layers:
        print(f"PILOT OK — {args.limit_layers} couches, worst corr {worst[0]:.4f}", flush=True)
        return 0

    # ---- top-level -----------------------------------------------------------
    if not out.done("top"):
        tensors = {}
        w = rd.read(f"{H}model.embed_tokens.weight")
        deq, nm = qinto(tensors, "model.embed_tokens", w, gs, bits)
        audit(nm, deq, w)
        w = rd.read(f"{H}lm_head.weight")
        deq, nm = qinto(tensors, "lm_head", w, gs, bits)
        audit(nm, deq, w)
        keep_bf16(tensors, "model.norm.weight", rd.read(f"{H}model.norm.weight"))
        out.write_shard("top", tensors)
        mx.clear_cache()
        print("[top] ok", flush=True)

    # ---- config / tokenizer / index / manifest ------------------------------
    cfg = dict(tc)
    cfg["model_type"] = "minimax_m3"
    cfg["layer_types"] = layer_types
    cfg["mlp_layer_types"] = mlp_layer_types
    for flat, legacy in (("index_n_heads", "sparse_num_index_heads"),
                          ("index_head_dim", "sparse_index_dim"),
                          ("index_block_size", "sparse_block_size"),
                          ("index_topk_blocks", "sparse_topk_blocks"),
                          ("index_local_blocks", "sparse_local_block")):
        if legacy in sc:
            cfg[flat] = sc[legacy]
    rp = cfg.get("rope_parameters") or {}
    cfg.setdefault("rope_theta", rp.get("rope_theta", 5e6))
    cfg.setdefault("partial_rotary_factor", rp.get("partial_rotary_factor", 0.5))
    cfg.pop("quantization_config", None)
    q = {"group_size": gs, "bits": bits, "mode": "affine"}
    cfg["quantization"] = q
    cfg["quantization_config"] = dict(q)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    for f in src.glob("*"):
        if f.name.startswith("tokenizer") or f.name in (
            "special_tokens_map.json", "generation_config.json",
            "chat_template.jinja", "added_tokens.json", "vocab.json",
        ):
            shutil.copy(f, dst / f.name)

    index = {"metadata": {"total_size": out.total}, "weight_map": out.weight_map}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))

    manifest = {
        "tool": "m3_convert.py v1 (famille 3 odyssai-convert, Odysseus#48)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": "ultra-512",
        "source": str(src),
        "source_format": "BF16 safetensors (minimax_m3_vl)",
        "model_type": "minimax_m3",
        "scope": "text tower only — vision tower + projector dropped",
        "recipe": {"bits": bits, "group_size": gs,
                   "indexer": "bf16 (non quantifié)",
                   "router_gate": "bf16 + e_score_correction_bias f32 (bruts)",
                   "experts": "stack switch_mlp (w1->gate, w2->down, w3->up)"},
        "audit": {"threshold": AUDIT_THRESHOLD,
                  "worst": {"corr": worst[0], "tensor": worst[1]},
                  "samples": out.audit},
    }
    (dst / "conversion-manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"TERMINÉ en {(time.time()-t0)/60:.1f} min — worst corr {worst[0]:.4f} ({worst[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
