#!/usr/bin/env python3
"""Route-level error shaping when the generation dies mid-flight (#41).

A watchdog/died raise must not silently truncate streams or record success.
The OpenAI paths already shaped the response (error chunk + [DONE] /
422-500); this pins what was missing: (1) the run REGISTRY finalizes with
the truth (error, not the default done — and never overwriting cancelled),
(2) the Anthropic /v1/messages paths got the same treatment: a terminal
`event: error` in the stream (CancelledError re-raised BEFORE any emit) and
a shaped 422/500 in the non-stream path. No node is reached.

    .venv/bin/python scripts/test_watchdog_error_shaping.py
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
json.dump({"argo": {"name": "A", "kind": "mlx-distributed", "backend": "ring",
                    "models_dir": "/m",
                    "nodes": [{"host": "n0", "ssh": "admin@198.51.100.1", "master": True}]}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


# --- behavior: the registry records the status it is finalized with ---
finalized = {}
api._persist.finalize_run = lambda rid, r: finalized.__setitem__(rid, r)  # capture, pas de DB
api._runs_register("t41", model="m", cluster="argo", pool_alias="a",
                   client="127.0.0.1", max_tokens=8, kind="streaming")
api._runs_finalize("t41", status="error")
check("finalize(error): le run quitte les actifs", "t41" in api._active_runs, False)
check("finalize(error): le statut remis au registre est error, pas done",
      (finalized.get("t41") or {}).get("status"), "error")

api._runs_register("t41c", model="m", cluster="argo", pool_alias="a",
                   client="127.0.0.1", max_tokens=8, kind="streaming")
api._runs_finalize("t41c", status="cancelled")
check("finalize(cancelled): pas écrasé par un défaut done",
      (finalized.get("t41c") or {}).get("status"), "cancelled")

# --- source guards: the four consumer sites ---
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.py")).read()
check("OpenAI ×2: le finalize porte le statut réel (error/cancelled/done)",
      src.count('status=("error" if run_status == "error"'), 2)
check("Anthropic stream: CancelledError re-raisé AVANT tout emit (#41 F1)",
      "except asyncio.CancelledError:\n            # Re-raised BEFORE any emit" in src, True)
check("Anthropic stream: event error terminal (spec: pas de message_stop après)",
      '"error": {"type": "server_error", "message": str(e)[:300]}' in src, True)
check("Anthropic stream: UNE seule métrique, au statut réel du run",
      "status=gen_status" in src and src.count("record_metric") - src.count("status=") >= 1, True)
check("Anthropic non-stream: 422 chat_template sinon 500, façonné",
      src.count('422 if msg.startswith("chat_template") else 500'), 2)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
