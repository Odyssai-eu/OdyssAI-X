#!/usr/bin/env python3
"""#79/#80 — node discovery: heartbeat registry, expiry, list, add-to-cluster,
and the node-side `odyssai-x advertise --dry-run`. The engine lifespan never
runs and ssh is stubbed: no node is reached.

    python3 scripts/test_discovery.py
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({"rep": {"name": "REP", "kind": "replica", "backend": "ring",
                   "nodes": [{"host": "na", "ssh": "admin@198.51.100.1", "master": True}]}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, HERE)
import api  # noqa: E402
import httpx  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


_real_exec = asyncio.create_subprocess_exec


async def _guard(*cmd, **kw):
    if cmd and "ssh" in str(cmd[0]):
        raise AssertionError(f"test tried to reach a node: {cmd[:3]}")
    return await _real_exec(*cmd, **kw)


asyncio.create_subprocess_exec = _guard


async def req(method, path, **kw):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://t") as c:
        return await c.request(method, path, **kw)


def call(method, path, **kw):
    return asyncio.run(req(method, path, **kw))


BEAT = {"host": "nb", "user": "admin", "ip": "192.168.86.77", "chip": "Apple M3 Ultra",
        "ram_gb": 256, "rdma": 6, "version": "1.53.14"}

# heartbeat → listed
r = call("POST", "/v1/nodes/heartbeat", json=BEAT)
check("beat accepted, interval returned", (r.status_code, r.json().get("beat_s")), (200, 5))
d = call("GET", "/admin/nodes/discovered").json()
check("beating node listed", [(n["host"], n["ssh"], n["rdma"]) for n in d["data"]], [("nb", "admin@192.168.86.77", 6)])
check("expiry is 3 missed beats", d["expire_s"], 15)
txt = call("GET", "/admin/nodes/discovered?format=text").text
check("text table has the node", "nb" in txt and "admin@192.168.86.77" in txt and "256G" in txt, True)

# validation: the list only takes plausible LAN nodes
for label, bad in (("host with shell chars", {**BEAT, "host": "a;rm"}), ("user with shell chars", {**BEAT, "user": "x y"}),
                   ("non-IP", {**BEAT, "ip": "evil.example"}), ("loopback", {**BEAT, "ip": "127.0.0.1"}),
                   ("Thunderbolt link-local", {**BEAT, "ip": "169.254.10.2"})):
    check(f"refused: {label}", call("POST", "/v1/nodes/heartbeat", json=bad).status_code, 422)
r = call("POST", "/v1/nodes/heartbeat", json={**BEAT, "host": "nc", "chip": "Apple\x07 M3\n" + "x" * 200})
check("chip stripped of control chars and capped", len(api._discovered["nc"]["chip"]) <= 64 and "\n" not in api._discovered["nc"]["chip"], True)

# goodbye removes at once
call("POST", "/v1/nodes/heartbeat", json={**BEAT, "host": "nc", "bye": True})
check("goodbye removes the node", "nc" in [n["host"] for n in call("GET", "/admin/nodes/discovered").json()["data"]], False)

# no goodbye: gone after 3 missed beats
api._discovered["nb"]["seen_at"] -= 16
check("silent node gone after 15 s", call("GET", "/admin/nodes/discovered").json()["count"], 0)

# a node already in a cluster is flagged
call("POST", "/v1/nodes/heartbeat", json={**BEAT, "host": "na", "ip": "198.51.100.1"})
row = next(n for n in call("GET", "/admin/nodes/discovered").json()["data"] if n["host"] == "na")
check("node in topology flagged with its cluster", row["in_clusters"], ["rep"])

# add to cluster: refused while ssh is not ready (host key / key), with the fix
call("POST", "/v1/nodes/heartbeat", json=BEAT)
calls = []


async def ssh_fail(target, cmd, timeout=10.0):
    calls.append((target, cmd))
    return 255, ""

api._ssh_capture = ssh_fail
r = call("POST", "/admin/nodes/discovered/nb/add", json={"cluster": "rep"})
check("add refused while ssh fails → 409 ssh_not_ready", (r.status_code, r.json()["detail"]["error"]), (409, "ssh_not_ready"))
check("… the fix names the host key step", "accept its host key" in r.json()["detail"]["message"], True)
check("… the engine only ran `true` on the announced target", calls, [("admin@192.168.86.77", "true")])


async def ssh_ok(target, cmd, timeout=10.0):
    return 0, ""

api._ssh_capture = ssh_ok
r = call("POST", "/admin/nodes/discovered/nb/add", json={"cluster": "rep"})
nodes = api.get_cluster_def("rep")["nodes"]
check("add succeeds once ssh works", r.status_code, 200)
check("node appended with its ssh target, not master", [(n["host"], n["ssh"], n["master"]) for n in nodes],
      [("na", "admin@198.51.100.1", True), ("nb", "admin@192.168.86.77", False)])
check("adding it twice → 409", call("POST", "/admin/nodes/discovered/nb/add", json={"cluster": "rep"}).status_code, 409)
check("unknown discovered host → 404", call("POST", "/admin/nodes/discovered/zz/add", json={"cluster": "rep"}).status_code, 404)
check("unknown cluster → 404", call("POST", "/admin/nodes/discovered/nb/add", json={"cluster": "nope"}).status_code, 404)

# the list sits behind the admin token when one is set; the heartbeat stays public
api.ADMIN_TOKEN = "s3cret"
check("list needs the admin token", call("GET", "/admin/nodes/discovered").status_code, 401)
check("heartbeat stays public (nodes hold no admin token)", call("POST", "/v1/nodes/heartbeat", json=BEAT).status_code, 200)
api.ADMIN_TOKEN = ""

# node side: the advertiser's dry run
env = {**os.environ, "ODYSSAI_X_ENGINE": "http://engine.test:8000/"}
p = subprocess.run(["sh", os.path.join(HERE, "odyssai-x"), "advertise", "--dry-run"], capture_output=True, text=True, env=env, timeout=30)
lines = dict(l.split(": ", 1) for l in p.stdout.strip().splitlines())
check("dry run: exit 0", p.returncode, 0)
check("dry run: Bonjour on port 22, role=node", "_odyssai._tcp local 22" in lines.get("bonjour", "") and "role=node" in lines["bonjour"], True)
check("dry run: one engine, trailing slash dropped", lines.get("heartbeat"), "POST http://engine.test:8000/v1/nodes/heartbeat every 5 s")
payload = json.loads(lines["payload"])
check("dry run: payload is what the engine accepts",
      sorted(payload) == sorted(["host", "user", "ip", "chip", "ram_gb", "rdma", "version", "bye"]) and payload["ram_gb"] > 0, True)
src = open(os.path.join(HERE, "odyssai-x")).read()
check("advertiser beats every 5 s, rescans at most every 5 min", ("sleep 5 & wait $!" in src, "next_scan=$((now + 300))" in src), (True, True))

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
