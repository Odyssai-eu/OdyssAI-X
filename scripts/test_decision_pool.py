#!/usr/bin/env python3
"""Engine tests for decision clusters (#95), no node and no model.

A real scripts/decision/decision_serve.py HTTP server runs on 127.0.0.1 with a fake
decider (no MLX); ssh is simulated (decision_config.json read, install check,
launch = start the fake server, kill = stop it). The engine endpoints are called in
process through httpx.ASGITransport (no lifespan: no sweep, no restore at start).

Run: ODYSSAI_X_STATE_DIR=$(mktemp -d) CLUSTER_CONFIG_FILE=$(mktemp -d)/cc.json python3 scripts/test_decision_pool.py
"""
import asyncio
import json
import os
import socket
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "decision"))
os.environ["PROMPT_STYLE"] = "semif"

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
CC = os.environ["CLUSTER_CONFIG_FILE"]
MODELS_DIR = "/Volumes/models/odysseus/test"
MODEL = f"{MODELS_DIR}/Fake-Decision-27B"
json.dump({
    "dec": {"name": "Dec", "kind": "decision", "backend": "http-proxy", "models_dir": MODELS_DIR,
            "nodes": [{"host": "n0", "ssh": "admin@127.0.0.1", "master": True}]},
    "plain": {"name": "Plain", "kind": "mlx-distributed", "backend": "ring", "models_dir": MODELS_DIR,
              "nodes": [{"host": "n1", "ssh": "admin@127.0.0.1", "master": True}]},
}, open(CC, "w"))

import httpx  # noqa: E402
import api  # noqa: E402
import decision_core  # noqa: E402
import decision_serve as ds  # noqa: E402

FAILS = []


def check(label, got, want):
    if got != want:
        FAILS.append(f"{label}: got {got!r}, want {want!r}")
        print(f"FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"OK  {label}")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class FakeDecider:
    def dist(self, state, q, opts):
        return {lab: (0.9 if i == 0 else 0.1 / (len(opts) - 1)) for i, (lab, _) in enumerate(opts)}, 7

    def dist_many_cached(self, state, items):
        return [self.dist(state, q, o) for q, o in items]


SERVERS = {}          # port -> ThreadingHTTPServer
SERVED_PATH = {}      # port -> model path the fake server reports
DCFG = {"prompt_version": "letter-v1-semif", "readout": "letter-logit", "max_one_pass": 160}
CALLS = []


def start_fake(port, model_path, name):
    w = ds.Worker(model_path, False, factory=lambda: (FakeDecider(), decision_core))
    w.start(); w.ready.wait(5)
    srv = ThreadingHTTPServer(("127.0.0.1", port), ds.make_handler(w, name, DCFG))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    SERVERS[port] = srv
    SERVED_PATH[port] = model_path


def stop_fake(port):
    srv = SERVERS.pop(port, None)
    if srv:
        srv.shutdown(); srv.server_close()


def fake_ssh(target, cmd, timeout=30):
    CALLS.append(cmd)
    if "decision_config.json" in cmd and cmd.startswith("cat "):
        if "Not-A-Decision" in cmd:
            return 1, "", ""
        if "Weird-Format" in cmd:
            return 0, json.dumps({"prompt_version": "letter-v9-x", "readout": "letter-logit"}), ""
        return 0, json.dumps(DCFG), ""
    if cmd.startswith("test -x"):
        return 0, "ok\n", ""
    if "decision_serve.py" in cmd and "nohup" in cmd:
        port = int(cmd.split("--port ")[1].split()[0])
        name = cmd.split("--name ")[1].split()[0].strip("'")
        model = cmd.split("--model ")[1].split()[0].strip("'")
        start_fake(port, model, name)
        return 0, "VLM_PID=4242\n", ""
    if "decision_serve.py.*--port" in cmd:
        port = int(cmd.split("--port ")[1].split("(")[0])
        stop_fake(port)
        return 0, "cleaned (SIGTERM)\n", ""
    if "kill -0" in cmd:
        return 0, "alive\n", ""
    return 0, "", ""


async def fake_size(ssh, path):
    return 55 * 1024 ** 3


api._ssh_exec = fake_ssh
RUNS = []
_orig_register = api._runs_register
def _rec_register(rid, **kw):
    RUNS.append((rid, kw))
    return _orig_register(rid, **kw)
api._runs_register = _rec_register
api.get_model_size_bytes = fake_size
PORT = free_port()
api.DECISION_DEFAULT_PORT = PORT
api.VLM_READY_TIMEOUT_S = 10.0


async def main():
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as c:
        # 1. untagged model on a decision cluster → 409
        r = await c.post("/admin/clusters/dec/load", json={"model": "Fake-Decision-27B"})
        check("1 untagged model on decision cluster → 409", r.status_code, 409)

        # 2. tag it through the settings
        r = await c.put("/admin/settings", json={"decision_models": ["Fake-Decision-27B"]})
        check("2 settings stores the tag", r.json().get("decision_models"), ["Fake-Decision-27B"])

        # 3. tagged model on a non-decision cluster → 409 before any preflight
        r = await c.post("/admin/clusters/plain/load", json={"model": "Fake-Decision-27B", "nodes": 1})
        check("3 tagged model on mlx-distributed cluster → 409", r.status_code, 409)

        # 4. load on the decision cluster
        r = await c.post("/admin/clusters/dec/load", json={"model": "Fake-Decision-27B"})
        check("4 load → 200", r.status_code, 200)
        body = r.json()
        check("4 dispatched as decision pool", body.get("dispatched"), "decision-pool")
        alias = body.get("alias")
        pool = api.get_pool("dec", alias)
        check("4 pool is a DecisionPool", type(pool).__name__, "DecisionPool")
        check("4 pool never is_vlm", getattr(pool, "is_vlm", None), False)
        check("4 launched on the default port", pool.port, PORT)

        # 5. /v1/models marks it as a decision model, without vision
        r = await c.get("/v1/models")
        ent = next((m for m in r.json()["data"] if m["id"] == alias), None)
        x = (ent or {}).get("x_odyssai", {})
        check("5 /v1/models kind=decision", x.get("kind"), "decision")
        check("5 /v1/models endpoint", x.get("endpoint"), "/v1/systemone")
        check("5 /v1/models no vision", x.get("supports_vision"), False)

        # 6. /v1/systemone routes to the local pool
        q = {"state": "s", "questions": {"ok": {"type": "noul", "instructions": "?",
                                                 "criteria": {"true": "y", "false": "n"}},
                                          "act": {"type": "choice", "instructions": "?",
                                                  "criteria": {"go": "g", "stop": "s"}}}}
        r = await c.post("/v1/systemone", json={**q, "model": alias})
        check("6 systemone → 200", r.status_code, 200)
        d = r.json()
        check("6 answer from the decision server", d["answers"]["act"]["choice"], "go")
        check("6 model is the alias", d["model"], alias)
        check("6 x_odyssai names the cluster", d.get("x_odyssai", {}).get("cluster"), "dec")
        check("6 registered as a run for the dashboard",
              (RUNS[-1][1].get("pool_alias"), RUNS[-1][1].get("client")) if RUNS else None, (alias, "decision"))
        check("6 run finalized (no active run left)", RUNS[-1][0] in api._active_runs, False)

        # 7. unknown model id, single decision model overall → it serves
        r = await c.post("/v1/systemone", json={**q, "model": "typesafe/jev-latest"})
        check("7 single decision model serves an unknown id", r.status_code, 200)

        # 8. a cloud systemone alias now exists → unknown id is ambiguous → 404 listing both
        r = await c.put("/admin/providers/jevtest", json={
            "api_base": "http://127.0.0.1:9/v1", "protocol": "systemone", "enabled": True,
            "published": [{"alias": "jev-test", "upstream": "jev"}]})
        check("8 cloud decision provider added", r.status_code, 200)
        r = await c.post("/v1/systemone", json={**q, "model": "typesafe/jev-latest"})
        check("8 two decision models → unknown id 404", r.status_code, 404)
        check("8 404 lists local + cloud", sorted(r.json()["detail"]["available"]), sorted([alias, "jev-test"]))
        r = await c.post("/v1/systemone", json={**q, "model": alias})
        check("8 local alias still routed locally", r.status_code, 200)

        # 9. chat and messages refuse a decision pool
        r = await c.post("/v1/chat/completions", json={"model": alias, "messages": [{"role": "user", "content": "hi"}]})
        check("9 chat → 400", r.status_code, 400)
        r = await c.post("/v1/messages", json={"model": alias, "max_tokens": 8,
                                                "messages": [{"role": "user", "content": "hi"}]})
        check("9 messages → 400", r.status_code, 400)

        # 10. alias clash with a cloud alias → 409
        r = await c.post("/admin/clusters/dec/load", json={"model": "Fake-Decision-27B", "alias": "jev-test"})
        check("10 alias clash with cloud → 409", r.status_code, 409)

        # 11. persisted as is_decision, never is_vlm
        st = json.load(open(api.state_file_for("dec")))
        ent = next(p for p in st["pools"] if p["alias"] == alias)
        check("11 state: is_decision", ent.get("is_decision"), True)
        check("11 state: no is_vlm", "is_vlm" in ent, False)
        check("11 state: decision_cfg kept", ent.get("decision_cfg", {}).get("readout"), "letter-logit")

        # 12. restore adopts the running server (no new launch)
        n_launch = sum("nohup" in x for x in CALLS)
        rp = await api._restore_decision_pool("dec", alias, ent, [0])
        check("12 restore adopts in place", type(rp).__name__, "DecisionPool")
        check("12 no relaunch when adopted", sum("nohup" in x for x in CALLS), n_launch)

        # 13. preflight on the decision cluster
        r = await c.get("/admin/clusters/dec/preflight", params={"model": "Fake-Decision-27B"})
        check("13 preflight decision ok", (r.json().get("ok"), r.json().get("plan", {}).get("mode")), (True, "decision"))
        r = await c.get("/admin/clusters/plain/preflight", params={"model": "Fake-Decision-27B"})
        check("13 preflight on plain cluster refuses the tagged model", r.json().get("ok"), False)

        # 14. unload kills by port and removes the pool
        r = await c.post(f"/admin/clusters/dec/unload?alias={alias}", json={"force": True})
        check("14 unload → 200", r.status_code, 200)
        check("14 pool gone", api.get_pool("dec", alias), None)
        check("14 server stopped", PORT in SERVERS, False)
        check("14 no runner sweep on a decision node",
              any("mlx-cluster/runner.py" in x for x in CALLS), False)

        # 15. restore relaunches when nothing answers
        n_launch = sum("nohup" in x for x in CALLS)
        rp = await api._restore_decision_pool("dec", alias, ent, [0])
        check("15 restore relaunches", (type(rp).__name__, sum("nohup" in x for x in CALLS) - n_launch),
              ("DecisionPool", 1))

        # 16. a foreign model on the port is never adopted nor killed
        n_kill = sum("decision_serve.py.*--port" in x for x in CALLS)
        SERVED_PATH[PORT] = "/Volumes/models/other/Another-Model"
        stop_fake(PORT); start_fake(PORT, "/Volumes/models/other/Another-Model", "other")
        rp = await api._restore_decision_pool("dec", alias, ent, [0])
        check("16 foreign model not adopted", rp, None)
        check("16 foreign model not killed", sum("decision_serve.py.*--port" in x for x in CALLS), n_kill)
        stop_fake(PORT)

        # 17. not a decision model / unknown format → 422
        await c.put("/admin/settings", json={"decision_models": ["Fake-Decision-27B", "Not-A-Decision", "Weird-Format"]})
        r = await c.post("/admin/clusters/dec/load", json={"model": "Not-A-Decision"})
        check("17 no decision_config.json → 422", r.status_code, 422)
        r = await c.post("/admin/clusters/dec/load", json={"model": "Weird-Format"})
        check("17 unknown decision format → 422", r.status_code, 422)

        # 18. kind validation accepts 'decision'
        check("18 validate kind decision", api.validate_cluster_def("dec2", 
            {"kind": "decision", "nodes": [{"host": "n0", "ssh": "a@b", "master": True}]}), None)


asyncio.run(main())

# 19. container shutdown keeps the nohup decision server running (adopted at boot)
src = open(os.path.join(REPO, "scripts", "api.py")).read()
fn = src[src.index("def _is_nohup_vlm(p):"):][:300]
check("19 shutdown spares decision pools", 'getattr(p, "is_decision", False)' in fn, True)

if FAILS:
    print("\n".join("FAIL " + f for f in FAILS))
    sys.exit(1)
print("all OK")
