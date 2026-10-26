#!/usr/bin/env python3
"""Runner ownership (#215, 1.62.11): a stop, a sweep or a load guard kills only
the runners it owns.

Runs the REAL node-side shell (selectors + kill script) under zsh, like the
nodes' login shell, against fake runners started on this machine: processes
whose argv ends in `mlx-cluster/runner.py` (or vlm_runner.py) and whose
environment carries a RUNNER_LAUNCH_ID, exactly what `remote_cmd` launches.
Refuses to run if a real runner exists on this machine.

    .venv/bin/python scripts/test_runner_ownership.py
"""
import asyncio
import os
import subprocess
import sys
import tempfile
import time

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
os.environ["ODYSSAI_ORCHESTRATOR_ID"] = "testorch"
os.environ["SWEEP_SIGTERM_GRACE_S"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


if subprocess.run(["pgrep", "-f", "[m]lx-cluster/(vlm_)?runner.py"],
                  capture_output=True).returncode == 0:
    sys.exit("a runner-like process already runs here — refusing to test")

ROOT = tempfile.mkdtemp()
os.makedirs(os.path.join(ROOT, "mlx-cluster"))
for name in ("runner.py", "vlm_runner.py"):
    open(os.path.join(ROOT, "mlx-cluster", name), "w").write(
        "import signal, time\nsignal.signal(signal.SIGTERM, lambda *a: exit(0))\n"
        "while True: time.sleep(1)\n")
PROCS = []


def fake(token, script="runner.py"):
    env = {k: v for k, v in os.environ.items() if k != "RUNNER_LAUNCH_ID"}
    if token is not None:
        env["RUNNER_LAUNCH_ID"] = token
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "mlx-cluster", script)], env=env)
    PROCS.append(p)
    return p


def zsh(script):
    return subprocess.run(["zsh", "-c", script], capture_output=True, text=True, timeout=60)


def alive(p):
    return p.poll() is None


def cleanup():
    for p in PROCS:
        if alive(p):
            p.kill()
            p.wait()
    PROCS.clear()


try:
    # Exact: a pool's stop kills its own ranks only
    a, b, f, u = fake("testorch/fith/qwen3/aaaa"), fake("testorch/fith/gemma/bbbb"), \
        fake("devorch/fith/qwen3/cccc"), fake(None)
    v = fake("testorch/fith/qwen3/aaaa", "vlm_runner.py")
    time.sleep(0.5)
    sel = zsh(api._runner_sel_sh("") + "lid " + str(a.pid))
    check("lid reads the token from the environment", sel.stdout.strip(), "testorch/fith/qwen3/aaaa")
    out = zsh(api._sel_exact("testorch/fith/qwen3/aaaa") + api._kill_sel_sh("clean exit"))
    time.sleep(0.3)
    check("exact kill: result line", out.stdout.strip(), "cleaned (SIGTERM)")
    check("exact kill: its runner.py and vlm_runner.py gone", (alive(a), alive(v)), (False, False))
    check("exact kill: sibling pool, other orchestrator, untagged untouched",
          (alive(b), alive(f), alive(u)), (True, True, True))
    out = zsh(api._sel_exact("testorch/fith/qwen3/aaaa") + api._kill_sel_sh("clean exit"))
    check("exact kill with nothing left: no-op message", out.stdout.strip(), "clean exit")
    cleanup()

    # Sweep: unclaimed own tokens only; untagged only at boot; foreign counted
    own_free, own_claimed = fake("testorch/main/old/1111"), fake("testorch/main/ling/2222")
    foreign, untagged = fake("devorch/main/x/3333"), fake(None)
    time.sleep(0.5)
    script = (api._sel_orphans({"testorch/main/ling/2222"}, include_untagged=False)
              + api._foreign_count_sh() + 'echo "foreign=$foreign"; ' + api._kill_sel_sh("no orphan"))
    out = zsh(script)
    time.sleep(0.3)
    check("sweep: foreign runners counted", "foreign=1" in out.stdout, True)
    check("sweep (unload): unclaimed own token killed, claimed/foreign/untagged spared",
          (alive(own_free), alive(own_claimed), alive(foreign), alive(untagged)),
          (False, True, True, True))
    out = zsh(api._sel_orphans({"testorch/main/ling/2222"}, include_untagged=True)
              + api._kill_sel_sh("no orphan"))
    time.sleep(0.3)
    check("sweep (boot): untagged killed too, foreign still spared",
          (alive(untagged), alive(foreign), alive(own_claimed)), (False, True, True))
    cleanup()

    # Prefix: the pool being loaded (replica children included), not a
    # hot-swapped old pool of the same alias, not a longer alias
    new, rep0, old, other = (fake("testorch/amigos/coder/new1"), fake("testorch/amigos/coder#r0/n2"),
                             fake("testorch/amigos/coder/old0"), fake("testorch/amigos/coder2/z9"))
    time.sleep(0.5)
    sel_sh = api._sel_prefix("testorch/amigos/coder", exclude={"testorch/amigos/coder/old0"})
    got = set(zsh(sel_sh + "sel").stdout.split())
    check("prefix: new pool + its replicas, not the old pool or a longer alias",
          got, {str(new.pid), str(rep0.pid)})
    fp = zsh(api._runner_footprint_cmd(sel_sh)).stdout.strip()
    check("footprint of the selection is a number", fp.isdigit(), True)
    out = zsh(sel_sh + api._kill_sel_sh())
    time.sleep(0.3)
    check("load guard kill: only the loading pool",
          (alive(new), alive(rep0), alive(old), alive(other)), (False, False, True, True))
    cleanup()

    # The whole sweep command, as _sweep_orphan_runners sends it over ssh
    own_free, own_claimed, foreign = (fake("testorch/main/old/1111"),
                                      fake("testorch/main/ling/2222"), fake("devorch/main/x/3333"))
    time.sleep(0.5)
    api._LIVE_LAUNCH_IDS.clear()
    api._LIVE_LAUNCH_IDS.add("testorch/main/ling/2222")
    orig = (api.get_cluster_def, api.subprocess.run, api._mark_cluster_degraded)
    api.get_cluster_def = lambda cid: {"kind": "runner", "nodes": [{"host": "local", "ssh": "x"}]}
    api.subprocess.run = lambda argv, **kw: orig[1](["zsh", "-c", argv[-1]], **kw)
    api._mark_cluster_degraded = lambda *a, **k: None
    try:
        res = api._sweep_orphan_runners("main")["swept"][0]
    finally:
        api.get_cluster_def, api.subprocess.run, api._mark_cluster_degraded = orig
        api._LIVE_LAUNCH_IDS.clear()
    time.sleep(0.3)
    check("full sweep command: result line",
          res["result"], "cleaned (SIGTERM) (1 of another orchestrator spared)")
    check("full sweep command: wired probe still parsed", isinstance(res.get("wired_bytes"), int), True)
    check("full sweep command: only the unclaimed own runner died",
          (alive(own_free), alive(own_claimed), alive(foreign)), (False, True, True))
    cleanup()
finally:
    cleanup()

# Launch token plumbing
cmd = api.remote_cmd({"rank": 0, "ssh": "admin@198.51.100.1"},
                     [{"rank": 0, "ssh": "admin@198.51.100.1", "rdma": None}],
                     "/m/x", "pipeline", 5555, "[]", False, 10, backend="ring",
                     launch_id="testorch/c/a/1234")
check("remote_cmd puts the token in the runner env", "RUNNER_LAUNCH_ID=testorch/c/a/1234 " in cmd, True)
vcmd = api.remote_vlm_cmd({"rank": 0, "ssh": "admin@198.51.100.1"},
                          [{"rank": 0, "ssh": "admin@198.51.100.1"}], "/m/x", 5556,
                          backend="ring", launch_id="testorch/c/v/5678")
check("remote_vlm_cmd too", "RUNNER_LAUNCH_ID=testorch/c/v/5678 " in vcmd, True)
lid = api._new_launch_id("fith", "argo:minimax m3")
check("token: orchestrator/cluster/alias/nonce, no blank",
      (lid.split("/")[:3], " " in lid), (["testorch", "fith", "argo:minimax_m3"], False))
check("orchestrator id from env override", api._orchestrator_id(), "testorch")


class FakePool:
    cluster, alias = "fith", "qwen3"

    def __init__(self, fail):
        self.fail = fail

    async def _start(self):
        if self.fail:
            raise RuntimeError("load failed")


p = FakePool(fail=False)
asyncio.run(api.RunnerPool.start(p))
check("start claims a token", p.launch_id in api._LIVE_LAUNCH_IDS, True)
first = p.launch_id
asyncio.run(api.RunnerPool.start(p))
check("restart (replica heal) drops the old token",
      (first in api._LIVE_LAUNCH_IDS, p.launch_id in api._LIVE_LAUNCH_IDS), (False, True))
p.runners = []
asyncio.run(api.RunnerPool.stop(p))
check("stop releases the token", p.launch_id in api._LIVE_LAUNCH_IDS, False)
q = FakePool(fail=True)
try:
    asyncio.run(api.RunnerPool.start(q))
except RuntimeError:
    pass
check("failed start releases the token", q.launch_id in api._LIVE_LAUNCH_IDS, False)

r = object.__new__(api.RunnerProc)
r.node, r.launch_id = {"rank": 0, "ssh": "admin@198.51.100.9"}, None
orig_run = api.subprocess.run
called = []
api.subprocess.run = lambda *a, **k: called.append(a)
r._remote_pkill("test")
api.subprocess.run = orig_run
check("a rank without token never runs a node-wide kill", called, [])

print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
sys.exit(1 if FAILS else 0)
