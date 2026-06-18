#!/usr/bin/env python3
"""m3_convert.py — conversion minimax_m3_vl (BF16 -> MLX quantisé), auditée.

3e famille du converter #48 (après bailing_hybrid et glm_moe_dsa) :
MiniMax-M3 428B A23B. Tour TEXTE seule — la tour vision et le projector sont
droppés (mlx-vlm viendra plus tard) ; aucun poids MTP dans le checkpoint.

Sortie = nommage hub conservé (model.layers.N.self_attn.*, block_sparse_moe.*)
avec les experts STACKÉS en switch_mlp.{gate,down,up}_proj quantifiés — le
layout que minimax_m3.py consomme directement (son sanitize passe les noms
stackés tels quels). Config aplati : text_config -> top-level, model_type
"minimax_m3", champs index_* plats. (PAS de layer_types/mlp_layer_types : le
validateur strict de transformers 5.9.0 rejette "minimax_m3_sparse" au load
tokenizer — le modèle dérive 3-full/57-sparse depuis sparse_attention_config.)

Quant mixte : --head-bits 8 (avec --bits 6) lève embed_tokens + lm_head à 8-bit
(la projection vocab bilingue) en laissant les experts à 6-bit -> fixe la
corruption logit-floor du lm_head à ~taille Q6, single-node (OdyssAI-X#53).

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
import socket
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
    if bits >= 16:
        # Full bf16 — no quantization (mx.quantize supports 2..8 bits only). The
        # source is bf16, so bf16->bf16 is EXACTLY lossless (audit corr == 1.0).
        # Same `.weight` naming as the quant path but no scales/biases; the config
        # omits the quantization block so the loader keeps these modules bf16.
        out[f"{name}.weight"] = w
        return w32, name
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
    ap.add_argument("--head-bits", type=int, default=0,
                    help="bits for embed_tokens + lm_head (0 = same as --bits). "
                         "Set 8 with --bits 6 for the MIXED quant that lifts only "
                         "the bilingual vocab projection — fixes the lm_head "
                         "logit-floor corruption at ~Q6 size / single-node "
                         "(OdyssAI-X#53). head_bits>=16 = keep bf16 / full-precision "
                         "head (e.g. --bits 8 --head-bits 16).")
    ap.add_argument("--group-size", type=int, default=64)
    # MSA selection fix (OdyssAI-X#53). The reference ships sparse_local_block=1 /
    # sparse_topk_blocks=16, which EVICTS the recent blocks carrying rare-token
    # spelling at long context (>2048) -> name drift / typos / fusions. Widening
    # the guaranteed local window to 8 (1024 recent keys) + the budget to 24 (so
    # the 8 forced blocks don't cannibalise the indexer's free slots) fixed it:
    # canary T01 scored 445 (Elma 41/41, longueur cible) vs 407.5 at 1/16, en
    # gardant ~36x de sparsité à 128k. Tunable ici.
    ap.add_argument("--local-blocks", type=int, default=8,
                    help="index_local_blocks: # of recent 128-key blocks always kept "
                         "(reference 1; 8 = MSA fix).")
    ap.add_argument("--topk-blocks", type=int, default=24,
                    help="index_topk_blocks: total selected blocks per query "
                         "(reference 16; 24 = MSA fix, laisse ~16 slots libres indexer).")
    ap.add_argument("--limit-layers", type=int, default=0)
    args = ap.parse_args()

    gs, bits = args.group_size, args.bits
    head_bits = args.head_bits or bits
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
    head_note = f" head={head_bits}" if head_bits != bits else ""
    print(f"src={src.name} bits={bits}/g{gs}{head_note} layers={NL} (run {n_do}) experts={NE} "
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
        # embed_tokens + lm_head at head_bits. mx.quantize only supports 2..8 bits,
        # so head_bits>=16 means "keep bf16" (full-precision head — the experts-Q8
        # + bf16-head variant): write the raw weight; config['quantization'] marks
        # these modules False so the loader skips quantizing them. Lossless, no audit.
        for nm in ("model.embed_tokens", "lm_head"):
            w = rd.read(f"{H}{nm}.weight")
            if head_bits >= 16:
                keep_bf16(tensors, f"{nm}.weight", w)
            else:
                deq, _ = qinto(tensors, nm, w, gs, head_bits)
                audit(nm, deq, w)
        keep_bf16(tensors, "model.norm.weight", rd.read(f"{H}model.norm.weight"))
        out.write_shard("top", tensors)
        mx.clear_cache()
        print("[top] ok", flush=True)

    # ---- config / tokenizer / index / manifest ------------------------------
    cfg = dict(tc)
    cfg["model_type"] = "minimax_m3"
    # Do NOT emit layer_types / mlp_layer_types. transformers 5.9.0's strict
    # PreTrainedConfig validator rejects the custom "minimax_m3_sparse" value at
    # AutoTokenizer-load time -> the runner's tokenizer load crashes ("3 ranks
    # died"). Pop any inherited from the source too. The vendored model derives
    # the 3-full / 57-sparse structure from sparse_attention_config + the
    # index_* fields below; the working Q6 and the validated Q8-3node both run
    # WITHOUT these keys. See OdyssAI-X#53.
    cfg.pop("layer_types", None)
    cfg.pop("mlp_layer_types", None)
    for flat, legacy in (("index_n_heads", "sparse_num_index_heads"),
                          ("index_head_dim", "sparse_index_dim"),
                          ("index_block_size", "sparse_block_size"),
                          ("index_topk_blocks", "sparse_topk_blocks"),
                          ("index_local_blocks", "sparse_local_block")):
        if legacy in sc:
            cfg[flat] = sc[legacy]
    # Override the reference 1/16 with the MSA-fix selection (see --local-blocks /
    # --topk-blocks). The reference values evict local fidelity at long context;
    # 8/24 is the validated fix (canari 445 vs 407.5). The indexer SCORING budget
    # stays as shipped; this only widens what the gather is guaranteed to keep.
    cfg["index_local_blocks"] = args.local_blocks
    cfg["index_topk_blocks"] = args.topk_blocks
    rp = cfg.get("rope_parameters") or {}
    cfg.setdefault("rope_theta", rp.get("rope_theta", 5e6))
    cfg.setdefault("partial_rotary_factor", rp.get("partial_rotary_factor", 0.5))
    cfg.pop("quantization_config", None)
    # AutoTokenizer lit ce config : sans tokenizer_class, un model_type inconnu
    # part en validation stricte PreTrainedConfig et explose (leçon du premier
    # load M3, 2026-06-12 23:42). eos/bos : le text_config ne les porte pas —
    # l'eos authentique est celui du tokenizer ([e~[ = 200020 pour M3).
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg.setdefault("eos_token_id", 200020)
    cfg.setdefault("bos_token_id", 200019)
    # Quantization config. Uniform unless head_bits differs (MIXED quant:
    # embed_tokens + lm_head at head_bits, everything else at bits). mlx_lm's
    # load_model keys per-module overrides by module PATH in config["quantization"]:
    #   def class_predicate(p, m):
    #       if p in config["quantization"]: return config["quantization"][p]
    #       ...
    # so the two head modules carry their own {group_size, bits} dict and the
    # rest fall through to the global bits.
    if bits >= 16:
        # Full bf16 model — NO quantization block at all, so load_model keeps
        # every module full-precision. Experts are still restructured into
        # switch_mlp and the indexer/router/norms stay bf16 as in the quant
        # recipes. (head_bits is also >=16 here, so the head is bf16 too.)
        cfg.pop("quantization", None)
        cfg.pop("quantization_config", None)
    else:
        q = {"group_size": gs, "bits": bits, "mode": "affine"}
        if head_bits != bits:
            if head_bits >= 16:
                # bf16 head: mark False so load_model's class_predicate skips quant
                # on these two modules (loaded full-precision).
                q["model.embed_tokens"] = False
                q["lm_head"] = False
            else:
                head_q = {"group_size": gs, "bits": head_bits, "mode": "affine"}
                q["model.embed_tokens"] = dict(head_q)
                q["lm_head"] = dict(head_q)
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
        "machine": socket.gethostname(),
        "source": str(src),
        "source_format": "BF16 safetensors (minimax_m3_vl)",
        "model_type": "minimax_m3",
        "scope": "text tower only — vision tower + projector dropped",
        "recipe": {"bits": bits, "head_bits": head_bits, "group_size": gs,
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
