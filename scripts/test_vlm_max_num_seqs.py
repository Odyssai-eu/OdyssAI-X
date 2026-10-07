#!/usr/bin/env python3
"""VLM replica concurrency knob (#108): MLX_VLM_MAX_NUM_SEQS is exported on
EVERY mlx_vlm.server launch (empty = unbounded, the mlx_vlm default — the CLI
flag only writes the same env var, so a leftover node-profile var would
silently win when omitted). The pool's dead "batch: false" label becomes an
honest True, the cap is refused (422) outside the VLM-replica path and on
adoption, persists in the cluster state, restores via .get (legacy entries
have no key). No node is reached.

    .venv/bin/python scripts/test_vlm_max_num_seqs.py
"""
import asyncio
import json
import os
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({"repl": {"name": "R", "kind": "replica", "backend": "ring",
                    "models_dir": "/m",
                    "nodes": [{"host": f"n{i}", "ssh": f"admin@198.51.100.{i + 1}", "master": i == 0}
                              for i in range(3)]}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402
import pydantic  # noqa: E402
from fastapi import HTTPException  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


# --- builder: the env is ALWAYS set (Kimi-2 / GLM-1 of the review) ---
CMD = lambda n=None: api._vlm_launch_cmd("t", "/v/mlx-vlm", "/m/MiMo Q9", 8080, n)
check("sans cap: MLX_VLM_MAX_NUM_SEQS exporté VIDE", "MLX_VLM_MAX_NUM_SEQS= " in CMD(), True)
check("sans cap: aucun drapeau CLI", "--max-num-seqs" in CMD(), False)
check("avec 16: MLX_VLM_MAX_NUM_SEQS=16", "MLX_VLM_MAX_NUM_SEQS=16 " in CMD(16), True)
check("le reste de la commande est intact (model quoté, port, PID)",
      ("--model '/m/MiMo Q9'" in CMD() and "--port 8080" in CMD()
       and "echo VLM_PID=$!" in CMD()), True)
check("déterminisme", CMD() == CMD(), True)

# --- request validation: strict, never coerced (GLM-1/Kimi-1) ---
def req_field(v):
    try:
        r = api.ArgoLoadRequest(model="/m/x", nodes=3, vlm_max_num_seqs=v)
        return r.vlm_max_num_seqs
    except pydantic.ValidationError:
        return "422"


check("int 16 accepté", req_field(16), 16)
check("None (absent) accepté", req_field(None), None)
for bad in (0, -3, "8", True, 8.0):
    check(f"valeur {bad!r} → 422 (strict, pas de coercition)", req_field(bad), "422")

# --- pool: honest label + carried cap + adoption refusal ---
class FakeChild:
    ssh_target = "admin@x"
    port = 8080
    host = "n0"
    model_path = "/m/x"
    upstream = "http://1.2.3.4:8080"
    pid = None


pool = api.VLMReplicaPool(model_path="/m/x", cluster="repl", alias="a",
                          node_indices=[0, 1], port=8080, venv="/v",
                          max_num_seqs=16)
check("étiquette honnête: batch=True", pool.batch, True)
check("le cap est porté par le pool", pool.max_num_seqs, 16)
pool2 = api.VLMReplicaPool(model_path="/m/x", cluster="repl", alias="a",
                           node_indices=[0], port=8080, venv="/v")
check("sans cap: None (non borné)", pool2.max_num_seqs, None)

async def adoption_case(mns):
    p = api.VLMReplicaPool(model_path="/m/x", cluster="repl", alias="a",
                           node_indices=[0], port=8080, venv="/v", max_num_seqs=mns)
    p.children = [FakeChild()]
    p._ip = lambda i: "1.2.3.4"
    async def served(ip, port, mp):
        return mp                       # "already serves this model" → adoption
    api._vlm_served_model = served
    try:
        await p._start_child(0)
        return "adopted"
    except RuntimeError as e:
        return "refused"


check("adoption SANS cap: adopté (comportement inchangé)",
      asyncio.run(adoption_case(None)), "adopted")
check("adoption AVEC cap: refusé, unload d'abord (Kimi-1)",
      asyncio.run(adoption_case(16)), "refused")

# --- dispatcher: the knob 422s outside the VLM-replica path (Kimi-2) ---
async def arch_vision(ssh, path):
    return {"is_vision": False}          # TEXT replica → the knob must 422


async def preflight_ok(cid, model, draft=None, venv=None):
    return {"ok": True}


api.get_model_arch_meta = arch_vision
api._gather_preflight = preflight_ok
try:
    asyncio.run(api.admin_cluster_load(
        "repl", api.ArgoLoadRequest(model="/m/x", nodes=3, vlm_max_num_seqs=16)))
    dispatch = "NO 422"
except HTTPException as e:
    dispatch = e.status_code
check("text replica load + cap → 422 au dispatcher", dispatch, 422)

# --- restore: .get, legacy entries without the key (GLM-2/Kimi-3) ---
check("restore d'une entrée SANS la clé → None (via .get, pas KeyError)",
      {"a": 1}.get("max_num_seqs"), None)
check("restore d'une entrée AVEC null explicite → None",
      {"max_num_seqs": None}.get("max_num_seqs"), None)

# --- source guards ---
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.py")).read()
check("l'export est TOUJOURS dans la commande (pas de conditionnel)",
      "MLX_VLM_MAX_NUM_SEQS={max_seqs}" in src, True)
check("le champ requête est strict + gt=0",
      "vlm_max_num_seqs: Optional[int] = Field(default=None, strict=True, gt=0)" in src, True)
check("plus aucun self.batch = False dans VLMReplicaPool",
      src.count("self.batch = False"), 0)
check("le restore lit .get (jamais entry[...])",
      'max_num_seqs=entry.get("max_num_seqs")' in src, True)
check("la persistance emporte le cap",
      '"max_num_seqs": getattr(pool, "max_num_seqs", None)' in src, True)
dash = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")).read()
check("dashboard: champ nombre VLM max seqs (la case batch n'est plus morte pour les VL)",
      'form-default-vlm-seqs' in dash and "body.vlm_max_num_seqs" in dash, True)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
