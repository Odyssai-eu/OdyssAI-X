#!/usr/bin/env python3
"""#81 part 2 — `GET /admin/doctor`, cluster mode. SSH is stubbed everywhere and
the engine lifespan never runs (it sweeps runners on real nodes): no node is
ever reached.

    python3 scripts/test_doctor_cluster.py
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({"dt": {"name": "DT", "kind": "mlx-distributed", "backend": "jaccl",
                  "models_dir": "$HOME/mlx-models",
                  "nodes": [{"host": "na", "ssh": "admin@198.51.100.1", "master": True},
                            {"host": "nb", "ssh": "admin@198.51.100.2"},
                            {"host": "nc", "ssh": "admin@198.51.100.3"}]}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, HERE)
import api  # noqa: E402
from doctor_schema_check import validate  # noqa: E402

SCHEMA = json.load(open(os.path.join(REPO, "docs", "doctor.schema.json")))
FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


# Nothing may spawn ssh for real.
_real_exec = asyncio.create_subprocess_exec


async def _guard_exec(*cmd, **kw):
    if cmd and "ssh" in str(cmd[0]):
        raise AssertionError(f"test tried to reach a node: {cmd[:3]}")
    return await _real_exec(*cmd, **kw)


asyncio.create_subprocess_exec = _guard_exec

# ── edge probe fixtures ────────────────────────────────────────────────
# Three ranks, full mesh: a.en3↔b.en5, a.en4↔c.en5, b.en4↔c.en6.
TOPO = [
    {"rank": 0, "host": "na", "ssh": "admin@198.51.100.1", "rdma": [None, "rdma_en3", "rdma_en4"]},
    {"rank": 1, "host": "nb", "ssh": "admin@198.51.100.2", "rdma": ["rdma_en5", None, "rdma_en4"]},
    {"rank": 2, "host": "nc", "ssh": "admin@198.51.100.3", "rdma": ["rdma_en5", "rdma_en6", None]},
]
HEALTHY_PORTS = {
    "admin@198.51.100.1": {"rdma_en3": ("PORT_ACTIVE", "169.254.1.1"), "rdma_en4": ("PORT_ACTIVE", "169.254.1.2")},
    "admin@198.51.100.2": {"rdma_en5": ("PORT_ACTIVE", "169.254.2.1"), "rdma_en4": ("PORT_ACTIVE", "169.254.2.2")},
    "admin@198.51.100.3": {"rdma_en5": ("PORT_ACTIVE", "169.254.3.1"), "rdma_en6": ("PORT_ACTIVE", "169.254.3.2")},
}


def edge_stub(ports, unreachable=(), ssh_down=(), hang=()):
    """Fake _ssh_capture for both probe rounds. ports: target → dev → (state, ip)."""
    async def fake(target, cmd, timeout=10.0):
        if target in hang:              # like the real _ssh_capture: bounded, then raises
            await asyncio.sleep(timeout)
            raise asyncio.TimeoutError()
        if target in ssh_down:
            raise OSError("ssh: connect to host: Operation timed out")
        if cmd.startswith("for d in"):
            return 0, "".join(f"{d} {st} {ip}\n" for d, (st, ip) in ports.get(target, {}).items())
        out = []
        for part in cmd.split(" ; "):
            dev = part.rsplit("then echo ", 1)[1].split()[0].strip("'")
            en = dev[len("rdma_"):]
            out.append(f"{dev} {'unreachable' if (target, en) in unreachable else 'ok'}")
        return 0, "\n".join(out) + "\n"
    return fake


def ports_with(target, dev, state="PORT_ACTIVE", ip="keep"):
    p = json.loads(json.dumps(HEALTHY_PORTS))
    p = {t: {d: tuple(v) for d, v in ds.items()} for t, ds in p.items()}
    st, old_ip = p[target][dev]
    p[target][dev] = (state, old_ip if ip == "keep" else ip)
    return p


# ── 1. the string view is unchanged (load preflight, link watch) ───────
old_src = subprocess.run(["git", "-C", REPO, "show", "HEAD:scripts/api.py"], capture_output=True, text=True).stdout
start = old_src.index("async def _validate_rdma_edges(")
end = old_src.index("\n\n\n", start)
ns = {"asyncio": asyncio, "shlex": api.shlex, "sys": sys}
exec(old_src[start:end], ns)               # the pre-refactor function, verbatim
old_validate = ns["_validate_rdma_edges"]

scenarios = {
    "healthy": dict(ports=HEALTHY_PORTS),
    "port down both ends": dict(ports=ports_with("admin@198.51.100.2", "rdma_en5", "PORT_DOWN")),
    "no alias": dict(ports=ports_with("admin@198.51.100.1", "rdma_en4", ip="none")),
    "unreachable peer": dict(ports=HEALTHY_PORTS, unreachable={("admin@198.51.100.3", "en6")}),
    "ssh down on one node": dict(ports=HEALTHY_PORTS, ssh_down={"admin@198.51.100.2"}),
    "whole node NO_DEVICE": dict(ports={**HEALTHY_PORTS, "admin@198.51.100.3": {
        "rdma_en5": ("NO_DEVICE", "none"), "rdma_en6": ("NO_DEVICE", "none")}}),
}
for name, sc in scenarios.items():
    api._ssh_capture = edge_stub(**sc)
    ns["_ssh_capture"] = api._ssh_capture
    new = asyncio.run(api._validate_rdma_edges(TOPO))
    old = asyncio.run(old_validate(TOPO))
    check(f"_validate_rdma_edges byte-identical to before ({name})", new, old)
check("whole-node NO_DEVICE stays inconclusive (no problem reported)", old, [])

# ── 2. doctor edges: one row per cable, re-checked ─────────────────────
api._ssh_capture = edge_stub(ports=HEALTHY_PORTS)
rows = asyncio.run(api._doctor_edges(TOPO))
check("healthy mesh → one OK row, 3 links", [(r["status"], r["message"]) for r in rows], [("OK", "3 Thunderbolt links usable")])

api._ssh_capture = edge_stub(ports=ports_with("admin@198.51.100.2", "rdma_en5", "PORT_DOWN"))
rows = asyncio.run(api._doctor_edges(TOPO))
check("cable down (seen from both ends) → exactly one FAIL", [r["status"] for r in rows], ["FAIL"])
check("… naming the cable", rows[0]["subject"] in ("na rdma_en3 ↔ nb rdma_en5", "nb rdma_en5 ↔ na rdma_en3"), True)
check("… with a cable fix", "cable" in rows[0]["fix"], True)

both = ports_with("admin@198.51.100.2", "rdma_en5", "PORT_DOWN")
both["admin@198.51.100.1"]["rdma_en3"] = ("PORT_DOWN", "169.254.1.1")
api._ssh_capture = edge_stub(ports=both)
raw = asyncio.run(api._probe_rdma_edges(TOPO))
rows = asyncio.run(api._doctor_edges(TOPO))
check("cable down at both ends: the probe sees it twice", len(raw), 2)
check("… the doctor reports it once", [r["status"] for r in rows], ["FAIL"])

calls = {"n": 0}
flaky = edge_stub(ports=ports_with("admin@198.51.100.3", "rdma_en6", "NO_DEVICE"))
clean = edge_stub(ports=HEALTHY_PORTS)


async def once_bad(target, cmd, timeout=10.0):
    if cmd.startswith("for d in"):
        calls["n"] += 1
    return await (flaky if calls["n"] <= 3 else clean)(target, cmd, timeout)


api._ssh_capture = once_bad
rows = asyncio.run(api._doctor_edges(TOPO))
check("transient fault gone on re-check → OK, not FAIL", [r["status"] for r in rows], ["OK"])

api._ssh_capture = edge_stub(ports=HEALTHY_PORTS, unreachable={("admin@198.51.100.3", "en6")})
rows = asyncio.run(api._doctor_edges(TOPO))
check("unreachable peer through one cable → one FAIL", [(r["status"], "unreachable" in r["message"]) for r in rows], [("FAIL", True)])

api.DOCTOR_EDGE_TIMEOUT_S = 0.3
api._ssh_capture = edge_stub(ports=HEALTHY_PORTS, hang={"admin@198.51.100.1"})
t = time.time()
rows = asyncio.run(api._doctor_edges(TOPO))
check("a hung node does not hold the edge check past its bound", time.time() - t < 2, True)
api.DOCTOR_EDGE_TIMEOUT_S = 5.0

# ── 3. per-node rows over ssh (stubbed) ────────────────────────────────
NODE_REPORT = lambda host, build, extra=(): json.dumps({
    "schema": 1, "mode": "node", "host": host, "exit": 0, "summary": {"ok": 2, "warn": 0, "fail": 0},
    "checks": [{"check": "macos", "status": "OK", "subject": "", "message": f"macOS 26.6.1 ({build})", "fix": ""},
               {"check": "mlx", "status": "OK", "subject": "", "message": "mlx 0.32.2", "fix": ""}, *extra]})
seen_cmds = []


def node_stub(by_target):
    async def fake(target, cmd, data, timeout):
        seen_cmds.append((target, cmd, len(data)))
        r = by_target[target]
        if r == "hang":
            await asyncio.sleep(timeout + 5)
        if r == "timeout":
            raise asyncio.TimeoutError()
        return r
    return fake


NODES = {"admin@198.51.100.1": (0, NODE_REPORT("na", "25G76"), ""),
         "admin@198.51.100.2": (255, "", "ssh: connect to host 198.51.100.2 port 22: Operation timed out\n"),
         "admin@198.51.100.3": (0, "ODYSSAI_NO_VENV\n", "")}
api._ssh_capture_stdin = node_stub(NODES)
api._ssh_capture = edge_stub(ports=HEALTHY_PORTS)
rep = asyncio.run(api.run_doctor("dt"))
by = {(r["host"], r["check"]): r["status"] for r in rep["checks"]}
check("node rows carry their host", by.get(("na", "mlx")), "OK")
check("unreachable node → one ssh FAIL", by.get(("nb", "ssh")), "FAIL")
check("node without venv → python FAIL", by.get(("nc", "python")), "FAIL")
check("report exit is the worst status", rep["exit"], 2)
check("cluster report validates against docs/doctor.schema.json", validate(rep, SCHEMA), None)
check("the script travels on stdin (bytes sent)", all(n > 1000 for _, _, n in seen_cmds), True)
check("the node runs it with the venv python, reading stdin",
      all('exec "$PY" - --json --manifest-json' in c for _, c, _ in seen_cmds), True)
check("models_dir from the cluster def, expanded on the node",
      all('--models-dir "$HOME/mlx-models"' in c for _, c, _ in seen_cmds), True)

api._ssh_capture_stdin = node_stub({
    "admin@198.51.100.1": (0, NODE_REPORT("na", "25G76"), ""),
    "admin@198.51.100.2": (0, NODE_REPORT("nb", "25G76"), ""),
    "admin@198.51.100.3": (0, NODE_REPORT("nc", "25F80"), "")})
rep = asyncio.run(api.run_doctor("dt"))
check("different macOS builds → one WARN naming them",
      [(r["status"], "25F80: nc" in r["message"]) for r in rep["checks"] if r["check"] == "macos-build"], [("WARN", True)])
check("all healthy but builds → exit 1", rep["exit"], 1)

# bounded: a node that never answers costs at most DOCTOR_NODE_TIMEOUT_S
api.DOCTOR_NODE_TIMEOUT_S = 0.5


async def hang_stdin(target, cmd, data, timeout):
    await asyncio.sleep(timeout + 0.05)
    raise asyncio.TimeoutError()


api._ssh_capture_stdin = hang_stdin
t = time.time()
rep = asyncio.run(api.run_doctor("dt"))
check("hung nodes → FAIL each, bounded by the node timeout", (time.time() - t < 2, sum(r["check"] == "ssh" and r["status"] == "FAIL" for r in rep["checks"])), (True, 3))
api.DOCTOR_NODE_TIMEOUT_S = 10.0

check("remote cmd refuses a models_dir that could leave its quotes",
      "--models-dir" in api._doctor_remote_cmd("{}", '/x"; rm -rf ~; "'), False)

# ── 4. the endpoint (ASGI transport: no lifespan) ──────────────────────
import httpx  # noqa: E402

api._ssh_capture_stdin = node_stub({t_: (0, NODE_REPORT(h, "25G76"), "") for t_, h in
                                    (("admin@198.51.100.1", "na"), ("admin@198.51.100.2", "nb"), ("admin@198.51.100.3", "nc"))})
api._ssh_capture = edge_stub(ports=ports_with("admin@198.51.100.2", "rdma_en5", "PORT_DOWN"))
api.build_topology = lambda cid, count=None: TOPO


async def call(q):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://t") as c:
        return await c.get("/admin/doctor" + q)

r = asyncio.run(call("?cluster=dt&format=text"))
lines = r.text.strip().splitlines()
check("text format: 200 + X-Doctor-Exit 2", (r.status_code, r.headers.get("x-doctor-exit")), (200, "2"))
check("text format: exactly one FAIL, the cable", [l.split(" ")[0] for l in lines if l.startswith("FAIL")], ["FAIL"])
check("text lines name the host", any(l.startswith("OK   na mlx:") for l in lines), True)
r = asyncio.run(call("?cluster=dt"))
check("json format: schema-valid, header set", (validate(r.json(), SCHEMA), r.headers.get("x-doctor-exit")), (None, "2"))
check("unknown cluster → 404", asyncio.run(call("?cluster=nope")).status_code, 404)

# ── 5. doctor_node.py really runs from stdin (the cluster transport) ───
man = open(os.path.join(HERE, "doctor-manifest.json")).read()
with tempfile.TemporaryDirectory() as md:
    p = subprocess.run([sys.executable, "-", "--json", "--manifest-json", man, "--models-dir", md],
                       input=open(os.path.join(HERE, "doctor_node.py")).read(), capture_output=True, text=True, timeout=30)
    rep = json.loads(p.stdout)
check("doctor_node.py piped on stdin → schema-valid report", validate(rep, SCHEMA), None)
check("… exit code matches the report", p.returncode, rep["exit"])

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
