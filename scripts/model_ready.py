#!/usr/bin/env python3
"""model_ready — LA seule façon de répondre « ce modèle est-il complet ? ».

Écrit après avoir grillé une nuit (2026-08-02/03) sur des watchers qui
comptaient des fichiers : « 44/79 » puis « 37/44 » — deux verdicts FAUX sur
deux modèles COMPLETS, parce que le nombre de fichiers d'un repo HF (index,
config, tokenizer, README) n'est PAS le nombre de shards, et que seul
`model.safetensors.index.json` fait foi.

Règle : ne jamais comparer un `ls | wc -l` à un nombre attendu. Demander ici.

  model_ready.py <dir>            -> exit 0 si complet, 1 sinon (+ verdict)
  model_ready.py <dir> --json     -> {"ready":bool,"present":n,"total":n,"missing":[...]}
  model_ready.py <dir> --wait 7200 -> bloque jusqu'à complet (poll 60s), exit 1 au timeout

Complet = tout shard listé dans l'index existe, avec la BONNE taille quand
l'index/metadata la donne, et zéro `*.incomplete` en vol. Un modèle
single-shard (pas d'index) est complet si `model.safetensors` existe.
"""
import argparse
import json
import os
import sys
import time


def check(d: str) -> dict:
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.isdir(d) and not os.path.exists(idx):
        # single-shard layout — no index to consult
        one = os.path.join(d, "model.safetensors")
        ok = os.path.exists(one)
        return {"ready": ok, "present": int(ok), "total": 1,
                "missing": [] if ok else ["model.safetensors"],
                "layout": "single", "incomplete": 0}
    if not os.path.exists(idx):
        return {"ready": False, "present": 0, "total": 0,
                "missing": ["model.safetensors.index.json"],
                "layout": "absent", "incomplete": 0}

    wm = json.load(open(idx))["weight_map"]
    shards = sorted(set(wm.values()))          # <- LA vérité, pas un ls
    missing = [s for s in shards if not os.path.exists(os.path.join(d, s))]
    # partial writes: a shard still being fetched is short — hf leaves a
    # sibling .incomplete, and the download dir keeps them until done.
    incomplete = 0
    for root, _, files in os.walk(d):
        incomplete += sum(1 for f in files if f.endswith(".incomplete"))
    return {"ready": not missing and incomplete == 0,
            "present": len(shards) - len(missing), "total": len(shards),
            "missing": missing, "layout": "sharded", "incomplete": incomplete}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--wait", type=int, default=0,
                    help="secondes à attendre la complétion (poll 60s)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    deadline = time.time() + a.wait if a.wait else None
    while True:
        r = check(a.dir)
        if r["ready"] or not deadline or time.time() > deadline:
            break
        inc = ", %d .incomplete" % r["incomplete"] if r["incomplete"] else ""
        if not a.quiet:
            print("[model_ready] %d/%d shards%s — attente…"
                  % (r["present"], r["total"], inc), flush=True)
        time.sleep(60)

    inc = ", %d .incomplete" % r["incomplete"] if r["incomplete"] else ""
    miss = ", manquants: %s" % r["missing"][:3] if r["missing"] else ""
    if a.json:
        print(json.dumps(r))
    elif not a.quiet:
        if r["ready"]:
            print("READY — %d shards (%s)" % (r["total"], r["layout"]))
        else:
            print("NOT READY — %d/%d shards%s%s"
                  % (r["present"], r["total"], inc, miss))
    sys.exit(0 if r["ready"] else 1)


if __name__ == "__main__":
    main()
