#!/usr/bin/env python3
"""#81 — `odyssai-x doctor`, node mode. Fixtures only: never touches a node.

    python3 scripts/test_doctor.py
"""
import copy
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import doctor_node as d  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


MANIFEST = {"mlx": "0.32.2", "mlx_lm": "0.31.3", "jaccl_sha256_16": "03ecbe4b8ad93f26",
            "modules": {"glm5_next.py": "aa", "qwen4_exp.py": "bb"},
            "patches": {"__init__.py": "cc", "opt_batch_gen.py": "dd"}}
HEALTHY = {  # an Argo node as measured on .30 (2026-09-28), all in order
    "host": "ultra-256a", "macos": "26.6.1", "macos_build": "25G76", "arch": "arm64",
    "python": {"executable": "/Users/admin/mlx-cluster/.venv/bin/python", "version": "3.11.15", "in_venv": True},
    "system_python": {"rc": 0, "stderr": ""},
    "mlx": "0.32.2", "mlx_lm": "0.31.3",
    "jaccl": {"path": "/x/libjaccl.dylib", "sha256_16": "03ecbe4b8ad93f26"},
    "modules": {"glm5_next.py": "aa", "qwen4_exp.py": "bb"},
    "patches": {"__init__.py": "cc", "opt_batch_gen.py": "dd"},
    "rdma_ctl": {"rc": 0, "out": "enabled"},
    "rdma_devices": [{"device": "rdma_en3", "interface": "en3", "state": "PORT_ACTIVE", "link_local": "169.254.250.13"},
                     {"device": "rdma_en2", "interface": "en2", "state": "PORT_DOWN", "link_local": None}],
    "wired": {"limit_mb": 204800, "ram_mb": 262144, "daemon": "com.thecompai.wired-limit"},
    "models_dir": {"asked": None, "path": "/Volumes/models/odysseus", "writable": True, "free_gb": 1322},
}


def run(mut=None):
    f = copy.deepcopy(HEALTHY)
    if mut:
        mut(f)
    rows = d.evaluate(f, MANIFEST)
    return rows, {r["check"] + (":" + r["subject"] if r["subject"] else ""): r["status"] for r in rows}


def only_bad(rows):
    return sorted((r["check"], r["status"]) for r in rows if r["status"] != "OK")


rows, st = run()
check("healthy RDMA node: every line OK", only_bad(rows), [])
check("healthy → exit 0", d.exit_code(rows), 0)

# system python blocked by the Xcode licence (fixture, never on a node)
rows, st = run(lambda f: f.update(system_python={"rc": 69, "stderr": "You have not agreed to the Xcode license agreements."}))
check("Xcode licence → named WARN", only_bad(rows), [("system-python", "WARN")])
check("WARN → exit 1", d.exit_code(rows), 1)
check("the fix names the command", "xcodebuild -license accept" in [r for r in rows if r["check"] == "system-python"][0]["fix"], True)

# not running under the venv
rows, _ = run(lambda f: f["python"].update(in_venv=False, executable="/usr/bin/python3"))
check("system python as interpreter → FAIL", only_bad(rows), [("python", "FAIL")])
check("FAIL → exit 2", d.exit_code(rows), 2)

# mlx off pin (max-64 on 2026-09-28: 0.31.2)
rows, _ = run(lambda f: f.update(mlx="0.31.2"))
check("mlx off pin → FAIL naming both versions",
      [(r["status"], "0.31.2" in r["message"] and "0.32.2" in r["message"]) for r in rows if r["check"] == "mlx"], [("FAIL", True)])
rows, _ = run(lambda f: f.update(mlx=None))
check("mlx missing → FAIL, jaccl not judged", (only_bad(rows)), [("mlx", "FAIL")])

# stock JACCL: FAIL on an RDMA node, WARN on a node without an active port
rows, _ = run(lambda f: f["jaccl"].update(sha256_16="a993b2a143e46798"))
check("stock JACCL on an RDMA node → FAIL", only_bad(rows), [("jaccl", "FAIL")])


def no_rdma(f):
    f["jaccl"]["sha256_16"] = "a993b2a143e46798"
    for dv in f["rdma_devices"]:
        dv["state"] = "PORT_DOWN"


rows, _ = run(no_rdma)
check("stock JACCL, no active port → WARN only", only_bad(rows), [("jaccl", "WARN")])
check("no active port → rdma line says ring only",
      "ring backend only" in [r for r in rows if r["check"] == "rdma"][0]["message"], True)

# module / patch drift
rows, _ = run(lambda f: f["patches"].update({"opt_batch_gen.py": "zz", "__init__.py": None}))
bad = [r for r in rows if r["check"] == "patches"][0]
check("patch drift → one FAIL listing missing and stale",
      (bad["status"], "missing __init__.py" in bad["message"], "stale opt_batch_gen.py" in bad["message"]), ("FAIL", True, True))

# active RDMA port without its link-local alias
rows, st = run(lambda f: f["rdma_devices"][0].update(link_local=None))
check("active port without 169.254 alias → FAIL naming the device", only_bad(rows), [("rdma-port", "FAIL")])
check("rdma-port row names rdma_en3", st.get("rdma-port:rdma_en3"), "FAIL")

# RDMA disabled / macOS too old
rows, _ = run(lambda f: f.update(rdma_ctl={"rc": 0, "out": "disabled"}))
check("rdma_ctl disabled → WARN", [s for c, s in only_bad(rows) if c == "rdma"], ["WARN"])
rows, _ = run(lambda f: f.update(macos="26.1"))
check("macOS 26.1 on an RDMA node → FAIL", only_bad(rows), [("macos", "FAIL")])
rows, _ = run(lambda f: (no_rdma(f), f.update(macos="15.5"), f["jaccl"].update(sha256_16="03ecbe4b8ad93f26")))
check("macOS 15.5 without RDMA link → WARN", [s for c, s in only_bad(rows) if c == "macos"], ["WARN"])

# wired limit
rows, _ = run(lambda f: f["wired"].update(daemon=None))
check("wired limit set but no daemon → WARN", only_bad(rows), [("wired-limit", "WARN")])
rows, _ = run(lambda f: f["wired"].update(limit_mb=0, daemon=None))
check("wired limit 0 → WARN", only_bad(rows), [("wired-limit", "WARN")])

# models dir
rows, _ = run(lambda f: f["models_dir"].update(writable=False))
check("models dir not writable → FAIL", only_bad(rows), [("models-dir", "FAIL")])
rows, _ = run(lambda f: f["models_dir"].update(free_gb=40))
check("low free space → WARN", only_bad(rows), [("models-dir", "WARN")])
rows, _ = run(lambda f: f.update(models_dir={"asked": "/nope", "path": None, "writable": False, "free_gb": None}))
check("asked models dir absent → FAIL", only_bad(rows), [("models-dir", "FAIL")])

# rendering: one line per check, fix only on WARN/FAIL
rows, _ = run(lambda f: f["wired"].update(daemon=None))
txt = d.render(rows).splitlines()
check("one line per check", len(txt), len(rows))
check("WARN line carries ' — ' + fix", any(l.startswith("WARN wired-limit:") and " — " in l for l in txt), True)
check("OK lines carry no fix", all(" — " not in l for l in txt if l.startswith("OK")), True)

# JSON report against the committed schema
schema = json.load(open(os.path.join(REPO, "docs", "doctor.schema.json")))


def validate(inst, sch):
    """Just enough of JSON Schema for this schema (no jsonschema dependency)."""
    if "const" in sch and inst != sch["const"]:
        return f"const {inst!r}"
    if "enum" in sch and inst not in sch["enum"]:
        return f"enum {inst!r}"
    t = sch.get("type")
    types = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float)}
    if t and not isinstance(inst, types[t]):
        return f"type {type(inst).__name__} != {t}"
    if t == "object":
        for k in sch.get("required", []):
            if k not in inst:
                return f"missing {k}"
        props = sch.get("properties", {})
        for k, v in inst.items():
            if k not in props:
                if sch.get("additionalProperties") is False:
                    return f"extra {k}"
                continue
            e = validate(v, props[k])
            if e:
                return f"{k}: {e}"
    if t == "array":
        for i, v in enumerate(inst):
            e = validate(v, sch["items"])
            if e:
                return f"[{i}] {e}"
    if t == "string" and "pattern" in sch and not re.match(sch["pattern"], inst):
        return f"pattern {inst!r}"
    return None


for label, mut in (("healthy", None), ("with a FAIL", lambda f: f.update(mlx="0.31.2"))):
    rows, _ = run(mut)
    rep = d.report("ultra-256a", rows)
    check(f"--json ({label}) validates against docs/doctor.schema.json", validate(rep, schema), None)
check("schema rejects an unknown status", validate({**d.report("h", []), "checks": [
    {"check": "x", "status": "MAYBE", "message": ""}]}, schema) is not None, True)

# manifest in the repo matches the repo (a module or patch changed → regenerate)
cur = d.build_manifest(REPO)
committed = json.load(open(os.path.join(HERE, "doctor-manifest.json")))
if cur["jaccl_sha256_16"] is None:          # no local JACCL build: compare the rest
    cur["jaccl_sha256_16"] = committed.get("jaccl_sha256_16")
check("doctor-manifest.json is up to date (else: doctor_node.py --write-manifest . > scripts/doctor-manifest.json)",
      cur == committed, True)

# read-only by construction: no mutating command in the node script, no api import
src = open(os.path.join(HERE, "doctor_node.py")).read()
code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
cmds = re.findall(r'_run\(\[\s*"([^"]+)"', code)
check("only read-only commands are run", sorted(set(cmds)),
      ["/usr/bin/python3", "ibv_devinfo", "ifconfig", "rdma_ctl", "sw_vers", "sysctl"])
check("never imports api.py", re.search(r"^\s*(import api|from api )", src, re.M) is None, True)
import io  # noqa: E402
import tokenize  # noqa: E402
# code without string literals: fix sentences may say "sudo …", calls may not
bare = " ".join(t.string for t in tokenize.generate_tokens(io.StringIO(src).readline)
                if t.type not in (tokenize.STRING, tokenize.COMMENT))
check("no write/kill/install calls", re.findall(
    r"\b(?:os\.(?:remove|unlink|kill|rename|replace|makedirs|mkdir|chmod)|shutil\.(?:rmtree|move|copy\w*)|subprocess\.Popen|os\.system)\b",
    bare), [])
opens = re.findall(r"open\(([^)]*)\)", src)
check("files only opened for reading", [o for o in opens if re.search(r"['\"][wax+]", o)], [])
check("rdma_ctl only asked for its status", re.findall(r'"rdma_ctl",\s*"(\w+)"', code), ["status"])

# the sh wrapper answers without a venv, in text and in JSON
env = {**os.environ, "ODYSSAI_X_PYTHON": "/nonexistent/python"}
p = subprocess.run(["sh", os.path.join(HERE, "odyssai-x"), "doctor"], capture_output=True, text=True, env=env)
check("wrapper, no venv: FAIL line, exit 2", (p.returncode, p.stdout.startswith("FAIL python:")), (2, True))
p = subprocess.run(["sh", os.path.join(HERE, "odyssai-x"), "doctor", "--json"], capture_output=True, text=True, env=env)
check("wrapper, no venv, --json validates", (p.returncode, validate(json.loads(p.stdout), schema)), (2, None))
p = subprocess.run(["sh", os.path.join(HERE, "odyssai-x"), "bogus"], capture_output=True, text=True, env=env)
check("wrapper: unknown command → usage, exit 64", p.returncode, 64)

# gather() on this machine: runs read-only, bounded, returns every fact evaluate() reads
with tempfile.TemporaryDirectory() as md:
    import time
    t = time.time()
    facts = d.gather(MANIFEST, models_dir=md)
    check("gather() here returns a models dir that is writable", (facts["models_dir"]["path"], facts["models_dir"]["writable"]), (md, True))
    check("gather() finishes well under the per-node budget", time.time() - t < 8, True)
    rep = d.report(facts["host"], d.evaluate(facts, MANIFEST))
    check("gather() → evaluate() → schema-valid report", validate(rep, schema), None)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
