#!/usr/bin/env python3
"""Admin auth middleware (1.53.9): same rules as the BaseHTTPMiddleware it replaced,
and a route behind it sees the client disconnect. No node needed.

    python3 scripts/test_admin_middleware.py
"""
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({}, open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402


def _no_ssh(*a, **k):
    raise AssertionError("test tried to reach a node")


api._ssh_exec = _no_ssh
_real_run = api.subprocess.run
api.subprocess.run = lambda cmd, *a, **k: _no_ssh() if "ssh" in str(cmd) else _real_run(cmd, *a, **k)
import uvicorn  # noqa: E402
from fastapi import Request  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


seen = {}


@api.app.post("/_test/disconnect")
async def _probe(request: Request):
    await request.json()
    for i in range(10):
        await asyncio.sleep(0.3)
        if await request.is_disconnected():
            seen["after"] = i
            return {}
    seen["after"] = None
    return {}


@api.app.get("/v1/_test/ping")
async def _ping():
    return {"ok": True}


PORT = 18779
BASE = f"http://127.0.0.1:{PORT}"
# lifespan="off": the engine's startup sweeps the orphan runners of every configured
# cluster over SSH. A test must never reach a node (2026-09-28: a first version of this
# file ran the lifespan and killed live runners on .29 and .49).
threading.Thread(target=lambda: uvicorn.run(api.app, host="127.0.0.1", port=PORT, log_level="error",
                                            lifespan="off"), daemon=True).start()
for _ in range(50):
    try:
        urllib.request.urlopen(BASE + "/admin/discovery/state", timeout=1)
        break
    except Exception:
        time.sleep(0.2)


def get(path, token=None, headers=None):
    h = dict(headers or {})
    if token:
        h["Authorization"] = f"Bearer {token}"
    try:
        r = urllib.request.urlopen(urllib.request.Request(BASE + path, headers=h), timeout=5)
        return r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers)


# dev mode (no admin token): admin routes open
api.ADMIN_TOKEN = ""
check("dev mode: /admin/version open", get("/admin/version")[0], 200)

# admin token set
api.ADMIN_TOKEN = "s3cret"
code, h = get("/admin/version")
check("no bearer → 401", code, 401)
check("401 carries WWW-Authenticate: Bearer", h.get("www-authenticate") or h.get("WWW-Authenticate"), "Bearer")
check("wrong bearer → 401", get("/admin/version", token="nope")[0], 401)
check("admin bearer → 200", get("/admin/version", token="s3cret")[0], 200)
check("?token= accepted", get("/admin/version?token=s3cret")[0], 200)
check("/admin/discovery/state public", get("/admin/discovery/state")[0], 200)
check("/admin/pair closed without window or enroll secret", get("/admin/pair")[0], 401)
check("public route open", get("/v1/_test/ping")[0], 200)

# crew hook: unknown crew bearer on /v1/* is tagged, admin bearer is not
code, h = get("/v1/_test/ping", token="unknown-crew")
check("unknown crew bearer → x-odyssai-crew-revoked", (code, h.get("x-odyssai-crew-revoked")), (200, "true"))
check("admin bearer on /v1 not tagged", get("/v1/_test/ping", token="s3cret")[1].get("x-odyssai-crew-revoked"), None)
known = {"id": "c1"}
api.find_crew_by_token, orig_find = (lambda t: known if t == "crew-ok" else None), api.find_crew_by_token
bumped = []
api.update_crew_last_seen, orig_upd = (lambda cid: bumped.append(cid)), api.update_crew_last_seen
check("known crew bearer not tagged", get("/v1/_test/ping", token="crew-ok")[1].get("x-odyssai-crew-revoked"), None)
check("known crew bearer → last_seen bumped", bumped, ["c1"])
check("crew bearer can't reach /admin/*", get("/admin/version", token="crew-ok")[0], 401)
api.find_crew_by_token, api.update_crew_last_seen = orig_find, orig_upd

# the point of the rewrite: a route behind the middleware sees the client leave
api.ADMIN_TOKEN = ""
try:
    urllib.request.urlopen(urllib.request.Request(BASE + "/_test/disconnect", data=b"{}",
                           headers={"content-type": "application/json"}), timeout=0.8)
except Exception:
    pass
time.sleep(3.5)
check("route behind the middleware sees the disconnect", seen.get("after") is not None, True)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
