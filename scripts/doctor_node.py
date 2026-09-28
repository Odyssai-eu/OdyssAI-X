#!/usr/bin/env python3
"""`odyssai-x doctor`, node mode (#81): is this Mac ready to serve as a cluster node?

One line per check, OK / WARN / FAIL, and the fix in one sentence. `--json` for
agents (schema: docs/doctor.schema.json). Exit code: 0 all OK, 1 at least one
WARN, 2 at least one FAIL.

    ~/mlx-cluster/odyssai-x doctor [--json] [--models-dir DIR]

Split like preflight.py: `gather()` reads the node (read-only commands, each
with a timeout), `evaluate()` turns those facts into check rows and is a pure
function, tested with fixtures in test_doctor.py. The engine's cluster mode
(`GET /admin/doctor`) pipes this same file to every node over SSH and merges
the rows with the RDMA edge checks.

Read-only by design: it never writes, kills, installs or reconfigures anything,
and it never imports api.py (whose startup sweeps runners). test_doctor.py
fails if a mutating command appears in this file.

Stdlib only, Python 3.9+: it has to run in a venv whose mlx install is broken.
Reference versions and file hashes come from doctor-manifest.json, generated
from the repo with `--write-manifest`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time

SCHEMA_VERSION = 1
MIN_MACOS_RDMA = (26, 2)        # RDMA over Thunderbolt appeared in macOS 26.2 (vendor/jaccl/README.md)
LOW_FREE_GB = 100
WIRED_LABELS = ("eu.odyssai.wiredlimit", "com.thecompai.wired-limit")   # current, legacy
MODELS_DIR_CANDIDATES = ("/Volumes/models/odysseus", "~/mlx-models")
CMD_TIMEOUT_S = 5
RANK = {"OK": 0, "WARN": 1, "FAIL": 2}


# ── gather: read the node ──────────────────────────────────────────────
def _run(cmd: list[str], timeout: float = CMD_TIMEOUT_S) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{cmd[0]}: timed out after {timeout:g} s"


def _digest(path: str, algo: str = "md5") -> str | None:
    try:
        h = hashlib.new(algo)
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _site_packages() -> str | None:
    import sysconfig
    p = sysconfig.get_paths().get("purelib")
    return p if p and os.path.isdir(p) else None


def _dist_version(name: str) -> str | None:
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:          # pragma: no cover — Python < 3.8
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def gather(manifest: dict, models_dir: str | None = None,
           cluster_dir: str = "~/mlx-cluster") -> dict:
    """Facts about this node. Every probe is read-only and bounded."""
    facts: dict = {"host": socket.gethostname().split(".")[0], "ts": time.time()}
    cluster_dir = os.path.expanduser(cluster_dir)

    rc, out, _ = _run(["sw_vers", "-productVersion"])
    facts["macos"] = out.strip() if rc == 0 else None
    rc, out, _ = _run(["sw_vers", "-buildVersion"])
    facts["macos_build"] = out.strip() if rc == 0 else None
    facts["arch"] = platform.machine()

    facts["python"] = {"executable": sys.executable, "version": platform.python_version(),
                       "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix)}
    rc, _, err = _run(["/usr/bin/python3", "-c", "pass"])
    facts["system_python"] = {"rc": rc, "stderr": err.strip()[-300:]}

    facts["mlx"] = _dist_version("mlx")
    facts["mlx_lm"] = _dist_version("mlx-lm")
    sp = _site_packages()
    lib = os.path.join(sp, "mlx", "lib", "libjaccl.dylib") if sp else None
    sha = _digest(lib, "sha256") if lib and os.path.exists(lib) else None
    facts["jaccl"] = {"path": lib, "sha256_16": sha[:16] if sha else None}

    models_src = os.path.join(sp, "mlx_lm", "models") if sp else None
    facts["modules"] = {n: _digest(os.path.join(models_src, n)) if models_src else None
                        for n in (manifest.get("modules") or {})}
    patches_dir = os.path.join(cluster_dir, "patches")
    facts["patches"] = {n: _digest(os.path.join(patches_dir, n))
                        for n in (manifest.get("patches") or {})}

    rc, out, err = _run(["rdma_ctl", "status"])
    facts["rdma_ctl"] = {"rc": rc, "out": (out or err).strip()[:200]}
    devices = []
    rc, out, _ = _run(["ibv_devinfo", "-l"])
    names = re.findall(r"\brdma_(en\d+)\b", out) if rc == 0 else []
    for en in names:
        _, dout, _ = _run(["ibv_devinfo", "-d", f"rdma_{en}"])
        m = re.search(r"state:\s*(\w+)", dout)
        _, iout, _ = _run(["ifconfig", en])
        ll = re.search(r"inet (169\.254\.\d+\.\d+)", iout)
        devices.append({"device": f"rdma_{en}", "interface": en,
                        "state": m.group(1) if m else "NO_DEVICE",
                        "link_local": ll.group(1) if ll else None})
    facts["rdma_devices"] = devices

    rc, out, _ = _run(["sysctl", "-n", "iogpu.wired_limit_mb"])
    wired_mb = int(out.strip()) if rc == 0 and out.strip().isdigit() else None
    rc, out, _ = _run(["sysctl", "-n", "hw.memsize"])
    mem_mb = int(out.strip()) // 1048576 if rc == 0 and out.strip().isdigit() else None
    facts["wired"] = {"limit_mb": wired_mb, "ram_mb": mem_mb,
                      "daemon": next((lab for lab in WIRED_LABELS
                                      if os.path.exists(f"/Library/LaunchDaemons/{lab}.plist")), None)}

    cands = [models_dir] if models_dir else list(MODELS_DIR_CANDIDATES)
    md = next((os.path.expanduser(c) for c in cands if os.path.isdir(os.path.expanduser(c))), None)
    facts["models_dir"] = {"asked": models_dir, "path": md,
                           "writable": bool(md and os.access(md, os.W_OK)),
                           "free_gb": round(shutil.disk_usage(md).free / 1e9) if md else None}
    return facts


# ── evaluate: facts → rows (pure) ──────────────────────────────────────
def _row(check: str, status: str, message: str, fix: str = "", subject: str = "") -> dict:
    return {"check": check, "status": status, "subject": subject, "message": message, "fix": fix}


def _version_tuple(v: str | None) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:3])


def evaluate(facts: dict, manifest: dict) -> list[dict]:
    rows: list[dict] = []
    active = [d for d in facts.get("rdma_devices") or [] if d.get("state") == "PORT_ACTIVE"]
    rdma_node = bool(active)

    # platform
    if facts.get("arch") != "arm64":
        rows.append(_row("platform", "FAIL", f"{facts.get('arch')} — MLX needs Apple Silicon",
                         "Use an M-series Mac."))
    mac = facts.get("macos")
    if not mac:
        rows.append(_row("macos", "WARN", "sw_vers gave no version", "Check that this is macOS."))
    elif _version_tuple(mac) < MIN_MACOS_RDMA:
        st = "FAIL" if rdma_node else "WARN"
        rows.append(_row("macos", st, f"macOS {mac}: RDMA over Thunderbolt needs 26.2 or later",
                         "Update macOS before using this node in a jaccl (RDMA) cluster."))
    else:
        rows.append(_row("macos", "OK", f"macOS {mac} ({facts.get('macos_build') or '?'})"))

    # python
    py = facts.get("python") or {}
    if not py.get("in_venv"):
        rows.append(_row("python", "FAIL", f"running under {py.get('executable')}, not the node venv",
                         "Run through ~/mlx-cluster/odyssai-x, which uses ~/mlx-cluster/.venv/bin/python."))
    else:
        rows.append(_row("python", "OK", f"venv python {py.get('version')}"))
    sp = facts.get("system_python") or {}
    if sp.get("rc") not in (0, None):
        if "licen" in (sp.get("stderr") or "").lower():
            rows.append(_row("system-python", "WARN",
                             "/usr/bin/python3 refuses to run: the Xcode licence is not accepted",
                             "Run `sudo xcodebuild -license accept` (an Xcode update re-arms it)."))
        else:
            rows.append(_row("system-python", "WARN", f"/usr/bin/python3 fails (rc {sp.get('rc')})",
                             "Only matters to tools that call the system python; the node venv is used."))

    # mlx pins
    for key, dist in (("mlx", "mlx"), ("mlx_lm", "mlx-lm")):
        want, got = manifest.get(key), facts.get(key)
        if got is None:
            rows.append(_row(dist, "FAIL", f"{dist} is not installed in the venv",
                             "Run `pip install -r ~/mlx-cluster/requirements-node.txt` in the venv."))
        elif want and got != want:
            rows.append(_row(dist, "FAIL", f"{dist} {got}, the pin is {want}",
                             f"Install {dist}=={want} (never `pip -U mlx`: it restores stock JACCL)."))
        else:
            rows.append(_row(dist, "OK", f"{dist} {got}"))

    # JACCL
    j = facts.get("jaccl") or {}
    want = manifest.get("jaccl_sha256_16")
    if facts.get("mlx") is not None:
        if not j.get("sha256_16"):
            rows.append(_row("jaccl", "FAIL" if rdma_node else "WARN", "libjaccl.dylib not found in the mlx wheel",
                             "Reinstall mlx from requirements-node.txt, then scripts/install-jaccl.sh."))
        elif not want:
            rows.append(_row("jaccl", "WARN", "no reference hash in the manifest",
                             "Regenerate doctor-manifest.json from a checkout with vendor/jaccl/build."))
        elif j["sha256_16"] != want:
            rows.append(_row("jaccl", "FAIL" if rdma_node else "WARN",
                             "stock or stale libjaccl.dylib (not OdyssAI's patched build)"
                             + ("" if rdma_node else "; only matters on an RDMA node"),
                             "Run scripts/install-jaccl.sh for this node."))
        else:
            rows.append(_row("jaccl", "OK", "patched libjaccl.dylib"))

    # vendored model modules and runtime patches
    for key, label, fix in (
            ("modules", "model-modules", "Run scripts/install-model-modules.sh for this node."),
            ("patches", "patches", "Run scripts/install-model-modules.sh for this node.")):
        ref, got = manifest.get(key) or {}, facts.get(key) or {}
        missing = sorted(n for n in ref if got.get(n) is None)
        stale = sorted(n for n in ref if got.get(n) is not None and got[n] != ref[n])
        if missing or stale:
            parts = ([f"missing {', '.join(missing)}"] if missing else []) + \
                    ([f"stale {', '.join(stale)}"] if stale else [])
            rows.append(_row(label, "FAIL", "; ".join(parts), fix))
        else:
            rows.append(_row(label, "OK", f"{len(ref)} files in sync"))

    # RDMA
    rc = facts.get("rdma_ctl") or {}
    if rc.get("rc") == 127:
        rows.append(_row("rdma", "WARN", "rdma_ctl not found: no RDMA support on this macOS",
                         "Update to macOS 26.2+ for RDMA; ring (TCP) still works."))
    elif "enabled" not in (rc.get("out") or "").lower() or "disabled" in (rc.get("out") or "").lower():
        rows.append(_row("rdma", "WARN", f"RDMA is not enabled ({rc.get('out') or 'no output'})",
                         "Enable it once from recoveryOS: `rdma_ctl enable`, then reboot."))
    else:
        devs = facts.get("rdma_devices") or []
        rows.append(_row("rdma", "OK", f"enabled, {len(active)} of {len(devs)} Thunderbolt ports active"
                         + ("" if active else " (no RDMA link: ring backend only)")))
    for d in active:
        if not d.get("link_local"):
            rows.append(_row("rdma-port", "FAIL", f"{d['device']} is active but {d['interface']} has "
                             "no link-local (169.254.x.x) address", subject=d["device"],
                             fix="Re-run the network setup (scripts/rdma-onboard.sh --check tells what is off)."))

    # wired limit
    w = facts.get("wired") or {}
    lim, ram = w.get("limit_mb"), w.get("ram_mb")
    pct = f" ({100 * lim // ram}% of RAM)" if lim and ram else ""
    if not lim:
        rows.append(_row("wired-limit", "WARN", "iogpu.wired_limit_mb is 0 (macOS default: large models get evicted)",
                         "Install the wired-limit daemon: scripts/wired-limit/install.sh."))
    elif not w.get("daemon"):
        rows.append(_row("wired-limit", "WARN", f"{lim} MB{pct} but no daemon: lost at the next reboot",
                         "Install the wired-limit daemon: scripts/wired-limit/install.sh."))
    else:
        rows.append(_row("wired-limit", "OK", f"{lim} MB{pct}, daemon {w['daemon']}"))

    # models dir
    m = facts.get("models_dir") or {}
    if not m.get("path"):
        rows.append(_row("models-dir", "FAIL" if m.get("asked") else "WARN",
                         f"no models directory ({m.get('asked') or ', '.join(MODELS_DIR_CANDIDATES)})",
                         "Create it or pass --models-dir; it must match the node's models_dir in the topology."))
    elif not m.get("writable"):
        rows.append(_row("models-dir", "FAIL", f"{m['path']} is not writable",
                         "Fix its ownership or permissions (downloads write there)."))
    elif (m.get("free_gb") or 0) < LOW_FREE_GB:
        rows.append(_row("models-dir", "WARN", f"{m['path']}: only {m.get('free_gb')} GB free",
                         "Free space before downloading a large model."))
    else:
        rows.append(_row("models-dir", "OK", f"{m['path']}, {m.get('free_gb')} GB free"))
    return rows


def exit_code(rows: list[dict]) -> int:
    return max((RANK[r["status"]] for r in rows), default=0)


def report(host: str, rows: list[dict], mode: str = "node") -> dict:
    counts = {s.lower(): sum(r["status"] == s for r in rows) for s in RANK}
    return {"schema": SCHEMA_VERSION, "mode": mode, "host": host,
            "checks": rows, "summary": counts, "exit": exit_code(rows)}


def render(rows: list[dict], host: str = "") -> str:
    out = []
    for r in rows:
        subj = f" {r['subject']}" if r.get("subject") else ""
        line = f"{r['status']:<4} {r['check']}{subj}: {r['message']}"
        if r["status"] != "OK" and r.get("fix"):
            line += f" — {r['fix']}"
        out.append(line)
    return "\n".join(out)


# ── manifest ───────────────────────────────────────────────────────────
def build_manifest(repo: str) -> dict:
    """Reference versions and hashes, from a checkout."""
    req = open(os.path.join(repo, "requirements-node.txt")).read()
    pin = lambda n: (re.search(rf"^{re.escape(n)}==(\S+)", req, re.M) or [None, None])[1]
    top = lambda d: sorted(f for f in os.listdir(os.path.join(repo, d)) if f.endswith(".py"))
    dylib = os.path.join(repo, "vendor", "jaccl", "build", "libjaccl.dylib")
    return {
        "mlx": pin("mlx"), "mlx_lm": pin("mlx-lm"),
        "jaccl_sha256_16": (_digest(dylib, "sha256") or "")[:16] or None,
        "modules": {f: _digest(os.path.join(repo, "scripts", "mlx_models", f)) for f in top("scripts/mlx_models")},
        "patches": {f: _digest(os.path.join(repo, "scripts", "patches", f)) for f in top("scripts/patches")},
    }


def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    ap = argparse.ArgumentParser(prog="odyssai-x doctor", description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    ap.add_argument("--models-dir", help="the node's models_dir (default: first of %s)"
                    % ", ".join(MODELS_DIR_CANDIDATES))
    ap.add_argument("--manifest", default=os.path.join(here, "doctor-manifest.json"))
    ap.add_argument("--manifest-json", help="inline manifest (cluster mode pipes this file over SSH)")
    ap.add_argument("--write-manifest", metavar="REPO", help="print the manifest built from a checkout")
    a = ap.parse_args(argv)
    if a.write_manifest:
        print(json.dumps(build_manifest(a.write_manifest), indent=1, sort_keys=True))
        return 0
    try:
        manifest = json.loads(a.manifest_json) if a.manifest_json else json.load(open(a.manifest))
    except (OSError, ValueError) as e:
        manifest = {}
        missing = _row("manifest", "WARN", f"no reference manifest ({e})",
                       "Copy doctor-manifest.json next to this script (the installer does).")
    else:
        missing = None
    facts = gather(manifest, a.models_dir)
    rows = ([missing] if missing else []) + evaluate(facts, manifest)
    if a.json:
        print(json.dumps(report(facts["host"], rows), indent=1))
    else:
        print(render(rows))
    return exit_code(rows)


if __name__ == "__main__":
    sys.exit(main())
