#!/usr/bin/env python3
"""Pro-loader PRE-FLIGHT evaluator.

Answers, BEFORE touching a GPU, the questions a serious loader must answer:
  - taille réelle du modèle (octets sur disque)
  - format / quantization (bits, group_size, mlx-convertible)
  - modèle supporté ? (runner/module présent, tokenizer là, config saine)
  - vision_config → chemin VLM ? venv mlx-vlm présent sur le node cible ?
  - combien de NODES faut-il, et LESQUELS ? (budget wired par node, co-résidence
    des pools déjà chargés déduite ; contraintes de sharding tensor vs pipeline)
  - draft présent ? valide ? compatible (même vocab/tokenizer) ?
  → un VERDICT + un PLAN (nodes sélectionnés d'office, mode, draft), et des
    ALTERNATIVES ("tient sur 3 en tensor, ou 2 en pipeline"), avant de charger.

Pure functions: `evaluate()` takes already-gathered data and returns a dict —
no I/O, unit-testable. The thin gatherer + CLI at the bottom is for standalone
testing against real models on a node; api.py wires `evaluate()` behind
`GET /admin/clusters/{id}/preflight` and as the load gate.
"""
from __future__ import annotations

import json
import math
import shlex
import subprocess
from typing import Any, Optional

# ── model-type knowledge ────────────────────────────────────────────────
# Types the distributed text runner can PIPELINE-split (PipelineMixin present,
# in stock mlx-lm or a vendored scripts/mlx_models module). Others can still
# load single-node or tensor-split, but not pipeline across ranks.
PIPELINE_CAPABLE = frozenset({
    "deepseek_v2", "deepseek_v3", "deepseek_v32", "deepseek_v4",
    "glm4_moe", "glm4_moe_lite", "glm_moe_dsa", "ministral3", "hy_v3", "hy_v4",
    "kimi_k3", "longcat2", "qwen3_5_moe", "qwen3_5_mixed", "bailing_moe_linear",
    "minimax_m3", "g9v3",
})

# Types the cluster knows how to serve at all (stock mlx-lm families + our
# vendored/aliased modules). Unknown → warn (not a hard block: mlx-lm may have
# gained it), but surface it loud so it's not a silent KeyError at load.
KNOWN_MODEL_TYPES = PIPELINE_CAPABLE | frozenset({
    "qwen2", "qwen3", "qwen3_moe", "llama", "mistral", "mixtral", "gemma2",
    "gemma3", "gemma4", "phi3", "cohere", "mimo_v2", "mimo_v2_flash",
    "kimi_linear", "inkling_mm_model", "muse_glimmer", "minimax_m2",
    "minimax_m3_vl", "laguna",
})

DEFAULT_ACTIVATION_FACTOR = 1.10   # KV cache + activations headroom over weights


# ── model meta ──────────────────────────────────────────────────────────
def read_model_meta(config: dict, size_bytes: int) -> dict:
    """Distill a model's config.json + on-disk size into the facts the planner
    needs. Cascades into text_config/language_model like the runner does."""
    def nested(key):
        return (
            config.get(key)
            or (config.get("text_config") or {}).get(key)
            or (config.get("language_model") or {}).get(key)
            or ((config.get("text_config") or {}).get("language_model") or {}).get(key)
        )

    mt = (config.get("model_type")
          or (config.get("text_config") or {}).get("model_type") or "").lower()
    is_vision = bool(
        "_vl" in mt or "_vision" in mt or "vision" in mt
        or "vision_config" in config or "vision_tower_config" in config
    )

    # Quantization: mlx `quantization` key, else inferencerlabs-style
    # `quantization_config` (root group_size + per-layer overrides, no `bits`).
    quant = config.get("quantization")
    quant_source = "quantization"
    if not quant and config.get("quantization_config"):
        qc = config["quantization_config"]
        root = {k: v for k, v in qc.items() if not isinstance(v, dict)}
        quant = {"group_size": root.get("group_size"), "bits": root.get("bits"),
                 "_needs_mlx_key": True}       # runner needs the key written
        quant_source = "quantization_config"
    bits = (quant or {}).get("bits")
    group_size = (quant or {}).get("group_size")

    kvh = nested("num_key_value_heads")
    layers = nested("num_hidden_layers")
    hidden = nested("hidden_size")
    vocab = nested("vocab_size")

    return {
        "model_type": config.get("model_type"),
        "size_bytes": int(size_bytes or 0),
        "size_gb": round((size_bytes or 0) / 1024**3, 1),
        "quant": {"bits": bits, "group_size": group_size,
                  "source": quant_source,
                  "needs_mlx_key": bool((quant or {}).get("_needs_mlx_key"))},
        "is_vision": is_vision,
        "num_key_value_heads": kvh,
        "num_hidden_layers": layers,
        "hidden_size": hidden,
        "vocab_size": vocab,
        "pipeline_capable": mt in PIPELINE_CAPABLE,
        "supported": mt in KNOWN_MODEL_TYPES,
    }


# ── node planning ───────────────────────────────────────────────────────
def _fits_tensor(meta: dict, cap_n: dict, n: int) -> bool:
    """Tensor-parallel: even split, KV heads must divide the world size."""
    kvh = meta.get("num_key_value_heads")
    if kvh is not None and n > 1 and kvh % n != 0:
        return False
    budget = int((cap_n or {}).get("max_loadable_bytes", 0))
    return budget > 0 and meta["size_bytes"] <= budget


def _fits_pipeline(meta: dict, cap_n: dict, n: int) -> bool:
    """Pipeline-parallel: needs PipelineMixin; capacity-aware hetero split."""
    if n > 1 and not meta.get("pipeline_capable"):
        return False
    budget = int((cap_n or {}).get("max_loadable_pipeline_bytes",
                                   cap_n.get("max_loadable_bytes", 0)))
    return budget > 0 and meta["size_bytes"] <= budget


def plan_nodes(meta: dict, capacity_by_nodes: dict, max_nodes: int) -> dict:
    """Smallest node count (+ mode) that fits, plus the full menu of options.

    `capacity_by_nodes` is the shape api.py already computes per cluster:
      {"1": {"max_loadable_bytes", "max_loadable_pipeline_bytes", "per_node":[…]}, …}
    Returns {ok, nodes, mode, indices, options[], reason}.
    """
    options: list[dict] = []
    for n in range(1, max_nodes + 1):
        cap = capacity_by_nodes.get(str(n)) or capacity_by_nodes.get(n) or {}
        if not cap.get("max_loadable_bytes") and not cap.get("max_loadable_pipeline_bytes"):
            continue
        if _fits_tensor(meta, cap, n):
            options.append({"nodes": n, "mode": "tensor",
                            "headroom_gb": round((cap["max_loadable_bytes"] - meta["size_bytes"]) / 1024**3, 1)})
        if n > 1 and _fits_pipeline(meta, cap, n):
            options.append({"nodes": n, "mode": "pipeline",
                            "headroom_gb": round((cap.get("max_loadable_pipeline_bytes", 0) - meta["size_bytes"]) / 1024**3, 1)})
        elif n == 1 and _fits_pipeline(meta, cap, n):
            # single-node "pipeline" == plain load; report as tensor to avoid dup
            pass

    if not options:
        # Why doesn't it fit even at max? Report the gap at max_nodes.
        cap_max = capacity_by_nodes.get(str(max_nodes)) or {}
        best = max(int(cap_max.get("max_loadable_pipeline_bytes", 0)),
                   int(cap_max.get("max_loadable_bytes", 0)))
        gap = round((meta["size_bytes"] - best) / 1024**3, 1)
        kvh = meta.get("num_key_value_heads")
        reason = (f"ne tient sur aucune topologie (jusqu'à {max_nodes} nodes) : "
                  f"{meta['size_gb']} GB > {round(best/1024**3,1)} GB de budget max "
                  f"(manque {gap} GB)")
        if kvh and not meta.get("pipeline_capable"):
            reason += (f" — et tensor-only avec kv_heads={kvh} limite les world_size "
                       f"aux diviseurs de {kvh}")
        return {"ok": False, "nodes": None, "mode": None, "indices": None,
                "options": [], "reason": reason}

    # Prefer the FEWEST nodes; tie-break tensor over pipeline (simpler), then
    # more headroom.
    options.sort(key=lambda o: (o["nodes"], 0 if o["mode"] == "tensor" else 1,
                                -o["headroom_gb"]))
    best = options[0]
    indices = list(range(best["nodes"]))     # concrete-node selection layered below
    return {"ok": True, "nodes": best["nodes"], "mode": best["mode"],
            "indices": indices, "options": options,
            "reason": f"tient sur {best['nodes']} node(s) en {best['mode']} "
                      f"(marge {best['headroom_gb']} GB)"}


def _plan_vlm_nodes(meta: dict, free_by_index: Optional[dict],
                    per_node: Optional[list[dict]], max_nodes: int) -> dict:
    """Node plan for a VISION model, co-residence + kv-heads aware.

    A VLM is single-node mlx_vlm.server when it fits ONE free node, else it is
    distributed (vlm_runner tensor/pipeline). Find the smallest node count N
    whose N FREEST nodes each hold the even shard; for the tensor-parallel VLM
    path (minimax_m3_vl and any non-pipeline VLM) kv_heads must divide N. Picks
    the actual free node indices — never a blind [0] (the old hard-code sent a
    327GB distributed VLM at node 0, which was busy AND too small)."""
    size = meta["size_bytes"]
    kvh = meta.get("num_key_value_heads")
    npn = len(per_node) if per_node else max_nodes
    free = free_by_index or {}
    # Free RAM per index; fall back to the node's ceiling (per_node) when the
    # co-residence map is absent so a fresh cluster still plans.
    def _free(i: int) -> int:
        if i in free:
            return int(free[i])
        if per_node and i < len(per_node):
            return int(per_node[i].get("wired_limit_bytes")
                       or per_node[i].get("ram_bytes") or 0)
        return 0
    order = sorted(range(npn), key=lambda i: -_free(i))
    tensor_only = not meta.get("pipeline_capable")
    for n in range(1, min(max_nodes, npn) + 1):
        if kvh and n > 1 and tensor_only and kvh % n != 0:
            continue                                   # kv must divide world size
        chosen = order[:n]
        shard = size / n
        if all(_free(i) >= shard for i in chosen):
            mode = "vlm" if n == 1 else "vlm-dist"
            reason = ("modèle vision → mlx_vlm.server single-node" if n == 1
                      else f"VLM distribué {n} nodes (vision, "
                           f"{'kv=%d' % kvh if kvh else 'tensor/pipeline'})")
            return {"ok": True, "nodes": n, "mode": mode, "indices": chosen,
                    "options": [{"nodes": n, "mode": mode}], "reason": reason}
    kv_hint = (f" ; tensor-only kv_heads={kvh} limite N aux diviseurs de {kvh}"
               if kvh and tensor_only else "")
    return {"ok": False, "nodes": None, "mode": None, "indices": None,
            "options": [],
            "reason": (f"VLM {meta['size_gb']} GB ne tient sur aucune topologie "
                       f"de nodes LIBRES (jusqu'à {max_nodes}){kv_hint}")}


def select_concrete_nodes(indices: list[int], per_node: list[dict],
                          size_bytes: int, mode: str,
                          free_by_index: Optional[dict] = None) -> dict:
    """Pick WHICH physical nodes. Co-residence aware: `free_by_index` maps a
    node index → free wired bytes (ceiling minus what loaded pools already
    hold). Prefers the freest nodes; flags when the chosen set is tight."""
    n = len(indices)
    ranked = list(range(len(per_node)))
    if free_by_index:
        ranked.sort(key=lambda i: -free_by_index.get(i, 0))
    chosen = ranked[:n]
    shard = size_bytes / n * (DEFAULT_ACTIVATION_FACTOR if mode == "pipeline" else 1.0)
    tight = []      # weights fit but the 1.10 margin doesn't — may still run
    overflow = []   # doesn't fit at all — certain OOM
    if free_by_index:
        needed = size_bytes * DEFAULT_ACTIVATION_FACTOR
        if mode == "pipeline":
            # Pipeline split is HETEROGENEOUS: each rank's shard is proportional
            # to its budget (auto_parallel RAM-weighted split), NOT size/n. So
            # the binding constraint is the SUM of free budgets over the chosen
            # nodes — a proportional split never overflows a single node when
            # the sum fits. Testing size/n against each node (the old even-split
            # assumption) FALSELY flagged the small nodes of a hetero cluster
            # (main: .29=494 GB, .30-.33=215 GB) — it refused a 764 GB Hy4 that
            # fits fine at 3 nodes hetero (494+215+215=924 > 764).
            total_free = sum(free_by_index.get(i, 0) for i in chosen)
            if total_free < size_bytes:
                overflow = list(chosen)
            elif total_free < needed:
                tight = list(chosen)
        else:
            # Tensor-parallel splits EVENLY — every rank holds size/n, so the
            # per-node budget IS the binding constraint.
            raw_shard = size_bytes / n
            for i in chosen:
                fi = free_by_index.get(i, 0)
                if fi < raw_shard:
                    overflow.append(i)
                elif fi < raw_shard * DEFAULT_ACTIVATION_FACTOR:
                    tight.append(i)
    return {"chosen": chosen,
            "hosts": [per_node[i].get("host") for i in chosen if i < len(per_node)],
            "shard_gb": round(shard / 1024**3, 1),
            "tight_indices": tight,
            "overflow_indices": overflow}


def validate_draft(draft_meta: Optional[dict], main_meta: dict) -> Optional[dict]:
    """Speculative-decoding draft: present, readable, vocab-compatible."""
    if draft_meta is None:
        return None
    if not draft_meta.get("model_type"):
        return {"ok": False, "reason": "draft: config.json illisible/absent"}
    dv, mv = draft_meta.get("vocab_size"), main_meta.get("vocab_size")
    if dv and mv and dv != mv:
        return {"ok": False,
                "reason": f"draft vocab {dv} ≠ modèle {mv} — incompatible (rejette les tokens)"}
    return {"ok": True, "reason": f"draft OK (type {draft_meta['model_type']}, vocab match)"}


# ── the verdict ─────────────────────────────────────────────────────────
def evaluate(*, config: dict, size_bytes: int, capacity_by_nodes: dict,
             max_nodes: int, per_node: Optional[list[dict]] = None,
             free_by_index: Optional[dict] = None,
             vlm_venv_present: Optional[bool] = None,
             draft_config: Optional[dict] = None,
             draft_size_bytes: int = 0) -> dict:
    """Full pre-flight verdict + load plan. Pure — feed it gathered data."""
    meta = read_model_meta(config, size_bytes)
    warnings: list[str] = []
    blockers: list[str] = []

    # A serious loader never says "fits" when it doesn't know the size. du=0
    # means the model dir isn't on the target node (not synced yet) or the
    # models volume isn't mounted — refuse rather than load blind.
    if meta["size_bytes"] <= 0:
        blockers.append(
            "taille 0 — modèle absent du node cible (à synchroniser) ou "
            "volume /Volumes/models non monté ; refus de charger à l'aveugle")
    if not config:
        blockers.append("config.json illisible/absente sur le node cible")
    if not meta["supported"]:
        warnings.append(
            f"model_type '{meta['model_type']}' inconnu du cluster — vérifier "
            f"qu'un module runner existe (sinon KeyError au load)")
    if meta["quant"]["needs_mlx_key"]:
        warnings.append(
            "checkpoint avec `quantization_config` sans clé `quantization` mlx "
            "(style inferencerlabs) — le runner doit l'écrire avant load")
    if meta["is_vision"]:
        if vlm_venv_present is False:
            blockers.append(
                "modèle vision (mlx_vlm.server) mais le venv mlx-vlm est ABSENT "
                "du node cible — le load resterait bloqué à 95%")
        # VLM node plan: single-node mlx_vlm.server if it fits ONE free node,
        # else distributed — co-residence + kv-heads aware, picks free nodes
        # (NOT a blind [0]). Fixes: a 327GB MiniMax planned at busy node 0.
        plan = _plan_vlm_nodes(meta, free_by_index, per_node, max_nodes)
        if not plan["ok"]:
            blockers.append(plan["reason"])
        elif vlm_venv_present is False:
            plan["ok"] = False
    else:
        plan = plan_nodes(meta, capacity_by_nodes, max_nodes)
        if not plan["ok"]:
            blockers.append(plan["reason"])

    selection = None
    if plan.get("ok") and per_node:
        selection = select_concrete_nodes(plan["indices"], per_node,
                                           meta["size_bytes"], plan["mode"],
                                           free_by_index)
        if selection.get("overflow_indices"):
            blockers.append(
                f"pas assez de mémoire libre sur les nodes "
                f"{selection['overflow_indices']} pour {meta['size_gb']} GB "
                f"(OOM certain) — un pool résident les occupe, ou le cluster est "
                f"trop petit pour ce modèle. Unload le résident, ajoute des "
                f"nodes, ou force=true pour outrepasser")
        if selection["tight_indices"]:
            warnings.append(
                f"nodes {selection['tight_indices']} serrés (co-résidence) — "
                f"shard {selection['shard_gb']} GB proche du libre")

    draft = validate_draft(read_model_meta(draft_config, draft_size_bytes)
                           if draft_config else None, meta) if draft_config else None
    if draft and not draft["ok"]:
        warnings.append(draft["reason"])

    ok = plan.get("ok", False) and not blockers
    return {
        "ok": ok,
        "verdict": ("OK" if ok else "REFUSÉ"),
        "model": meta,
        "plan": plan,
        "selection": selection,
        "draft": draft,
        "warnings": warnings,
        "blockers": blockers,
        "summary": _summary(ok, meta, plan, selection, draft, blockers),
    }


def _summary(ok, meta, plan, selection, draft, blockers) -> str:
    if not ok:
        return "REFUSÉ — " + ("; ".join(blockers) or "voir warnings")
    hosts = (selection or {}).get("hosts") or []
    where = f" → {', '.join(h for h in hosts if h)}" if hosts else ""
    d = ""
    if draft and draft.get("ok"):
        d = " + draft"
    return (f"OK : {meta['size_gb']} GB sur {plan['nodes']} node(s) "
            f"en {plan['mode']}{where}{d}")


# ── standalone gatherer + CLI (testing before wiring into api.py) ────────
def _ssh_json(ssh: str, abspath: str) -> dict:
    cmd = f"cat {shlex.quote(abspath.rstrip('/') + '/config.json')} 2>/dev/null"
    try:
        out = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "-o",
                              "BatchMode=yes", ssh, cmd],
                             capture_output=True, text=True, timeout=15)
        return json.loads(out.stdout or "{}")
    except Exception:
        return {}


def _ssh_size(ssh: str, abspath: str) -> int:
    cmd = f"du -sk {shlex.quote(abspath)} 2>/dev/null | cut -f1"
    try:
        out = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "-o",
                              "BatchMode=yes", ssh, cmd],
                             capture_output=True, text=True, timeout=30)
        return int((out.stdout or "0").strip().split()[0]) * 1024
    except Exception:
        return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Pre-flight verdict for a model.")
    ap.add_argument("ssh", help="rank-0 ssh target, e.g. admin@192.168.86.29")
    ap.add_argument("model", help="absolute model path on the node")
    ap.add_argument("--capacity", help="capacity_by_nodes JSON (from /admin/clusters/{id})")
    ap.add_argument("--max-nodes", type=int, default=1)
    ap.add_argument("--draft", help="absolute draft model path")
    ap.add_argument("--vlm-venv", choices=["yes", "no"], default=None)
    args = ap.parse_args()

    cfg = _ssh_json(args.ssh, args.model)
    size = _ssh_size(args.ssh, args.model)
    cap = json.loads(args.capacity) if args.capacity else {
        str(args.max_nodes): {"max_loadable_bytes": 10**15,
                              "max_loadable_pipeline_bytes": 10**15,
                              "per_node": []}}
    per_node = (cap.get(str(args.max_nodes)) or {}).get("per_node") or []
    draft_cfg = _ssh_json(args.ssh, args.draft) if args.draft else None
    draft_size = _ssh_size(args.ssh, args.draft) if args.draft else 0
    v = evaluate(config=cfg, size_bytes=size, capacity_by_nodes=cap,
                 max_nodes=args.max_nodes, per_node=per_node,
                 vlm_venv_present={"yes": True, "no": False}.get(args.vlm_venv),
                 draft_config=draft_cfg, draft_size_bytes=draft_size)
    print(json.dumps(v, indent=2, ensure_ascii=False))
