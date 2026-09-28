#!/usr/bin/env python3
"""#93 phase 1 — orphan sweep reports an incomplete run, patterns self-exclude,
the post-reboot reload waits for the models volume. No node needed.

    ODYSSAI_X_STATE_DIR=$(mktemp -d) CLUSTER_CONFIG_FILE=$(mktemp -d)/cc.json python3 scripts/test_sweep_restore.py
"""
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({"t": {"name": "T", "kind": "mlx-distributed", "backend": "ring",
                 "nodes": [{"host": "n0", "ssh": "admin@127.0.0.1", "master": True}]}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


# 1. self-excluding patterns: the regex matches a runner, not a shell whose
#    command line contains the pattern text itself
for pat, target in ((api.RUNNER_MATCH_PATTERN, "python /Users/x/mlx-cluster/runner.py"),
                    (api.VLM_RUNNER_MATCH_PATTERN, "python /Users/x/mlx-cluster/vlm_runner.py")):
    rx = re.compile(pat)
    check(f"{pat!r} matches the runner", bool(rx.search(target)), True)
    check(f"{pat!r} does not match a sweep shell", bool(rx.search(f"zsh -c pkill -f '{pat}'")), False)


# 2. sweep output parsing: complete vs incomplete
def fake_run(stdout, rc=0, stderr=""):
    return lambda *a, **k: subprocess.CompletedProcess(a, rc, stdout=stdout, stderr=stderr)


orig = api.subprocess.run
api.subprocess.run = fake_run("no orphan\nno vlm orphan\nWIRED_BYTES=4294967296\n")
r = api._sweep_orphan_runners("t")["swept"][0]
check("complete sweep", (r["ok"], r["complete"], r["result"]), (True, True, "no vlm orphan"))
api.subprocess.run = fake_run("", rc=255, stderr="Connection to 127.0.0.1 closed by remote host.\n")
r = api._sweep_orphan_runners("t")["swept"][0]
check("killed shell → incomplete, not clean", (r["ok"], r["complete"]), (False, False))
check("incomplete result carries rc and stderr", ("rc=255" in r["result"], "closed by remote host" in r["result"]), (True, True))
api.subprocess.run = orig

# 3. boot guard: an incomplete sweep puts the host in the restore guard
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.py")).read()
guard = src[src.index("_leaked_hosts: set = set()"):][:1500]
check("boot: incomplete sweep → host not restored", 'if _entry.get("complete") is False:' in guard
      and "_leaked_hosts.add(_entry.get(\"host\"))" in guard, True)


# 4. post-reboot reload waits until config.json is readable
api.save_cluster_state_v2  # state file used by _wait_models_readable
json.dump({"schema": "v2", "pools": [{"alias": "m", "model": "/Volumes/models/x/M", "node_indices": [0], "nodes": 1}]},
          open(api.state_file_for("t"), "w"))
calls = {"n": 0}


def fake_ssh(target, cmd, timeout=10):
    calls["n"] += 1
    return 0, ("ok" if calls["n"] >= 3 else "missing"), ""


api._ssh_exec = fake_ssh
ok = asyncio.run(api._wait_models_readable("t", timeout_s=5, every_s=0.05))
check("waits until the volume answers", (ok, calls["n"]), (True, 3))
calls["n"] = -10**9
ok = asyncio.run(api._wait_models_readable("t", timeout_s=0.2, every_s=0.05))
check("gives up after the timeout (restores anyway)", ok, False)
rr = src[src.index("async def _reboot_reload_after"):][:2500]
check("reboot-reload waits before restoring",
      rr.index("await _wait_models_readable(cluster_id)") < rr.index("await _restore_cluster_pools(cluster_id)"), True)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
