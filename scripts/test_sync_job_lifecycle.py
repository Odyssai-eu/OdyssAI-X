#!/usr/bin/env python3
"""Sync job lifecycle (1.62.10): cancel kills on the node, nothing starts after
a cancel, a restart reattaches instead of lying "interrupted".

No node, no network: ssh helpers and subprocess spawns are replaced by fakes.

    .venv/bin/python scripts/test_sync_job_lifecycle.py
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


SRC = {"id": "ultra-512", "ssh": "admin@192.168.86.29", "models_dir": "/Volumes/models/odysseus"}
DST = {"id": "ultra-256a", "ssh": "admin@192.168.86.30", "models_dir": "/Volumes/models/odysseus"}
DST2 = {"id": "ultra-256b", "ssh": "admin@192.168.86.31", "models_dir": "/Volumes/models/odysseus"}
HOSTS = {h["id"]: h for h in (SRC, DST, DST2)}
api._resolve_host = lambda hid: HOSTS.get(hid)
api._persist.persist_job = lambda jid, payload: None


def ere_match(pattern, line):
    return subprocess.run(["grep", "-qE", "--", pattern], input=line, text=True).returncode == 0


# Patterns against real argv shapes
M = "org/Qwen2.5-Coder-7B"
rs, tar = api._sync_kill_patterns("ab12", M, SRC, DST)
argv_rsync = f"rsync -a /Volumes/models/odysseus/{M}/ admin@192.168.86.30:/Volumes/models/odysseus/{M}/"
check("rsync pattern matches its slot", ere_match(rs, argv_rsync), True)
check("rsync pattern matches openrsync's re-exec'd argv",
      ere_match(rs, "/usr/libexec/rsync/rsync.openrsync -a " + argv_rsync[len("rsync -a "):]), True)
check("rsync pattern ignores another target node",
      ere_match(rs, argv_rsync.replace("192.168.86.30", "192.168.86.31")), False)
check("rsync pattern ignores a longer model name (prefix)",
      ere_match(rs, argv_rsync.replace("7B/", "7B-8bit/")), False)
check("rsync pattern does not match its own pkill command line",
      ere_match(rs, f"pkill -f -- {rs}"), False)
check("tar pattern matches its slot",
      ere_match(tar, "tar cf - -T /tmp/tbsync_manifest_ab12_ultra-256a.txt"), True)
check("tar pattern ignores another target of the same job",
      ere_match(tar, "tar cf - -T /tmp/tbsync_manifest_ab12_ultra-256b.txt"), False)
check("TB5 list file is per target",
      api._sync_tb5_list_path("ab12", "ultra-256a") != api._sync_tb5_list_path("ab12", "ultra-256b"), True)

# Verdict by sizes
check("converged: same sizes", api._sync_converged({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 9}), True)
check("not converged: truncated file", api._sync_converged({"a": 1, "b": 2}, {"a": 1, "b": 1}), False)
check("not converged: target unreadable", api._sync_converged({"a": 1}, None), False)
check("not converged: empty source", api._sync_converged({}, {}), False)


def job_with(*statuses, status="running"):
    return {"id": "j", "model": M, "source": "ultra-512", "status": status,
            "per_target": [{"target": t, "status": st} for t, st in
                           zip(["ultra-256a", "ultra-256b"], statuses)]}


for sts, want in [(("done", "done"), "done"), (("done", "failed"), "partial"),
                  (("failed", "failed"), "failed"),
                  (("interrupted", "interrupted"), "interrupted"),
                  (("interrupted", "failed"), "failed")]:
    j = job_with(*sts)
    api._sync_conclude("j", j)
    check(f"conclude {sts} -> {want}", j["status"], want)
j = job_with("cancelled", "done", status="cancelled")
api._sync_conclude("j", j)
check("conclude keeps a cancelled job cancelled", j["status"], "cancelled")

# Nothing is spawned for a cancelled job; a forced exit stays cancelled
spawned = []


async def fake_exec(*a, **kw):
    spawned.append(a)
    raise AssertionError("spawned after cancel")
orig_exec = api.asyncio.create_subprocess_exec
api.asyncio.create_subprocess_exec = fake_exec
api._sync_jobs["c1"] = job_with("queued", status="cancelled")
slot = api._sync_jobs["c1"]["per_target"][0]
asyncio.run(api._run_one_rsync("c1", M, SRC, DST, slot))
api.asyncio.create_subprocess_exec = orig_exec
check("cancelled job: no process spawned", spawned, [])
check("cancelled job: slot cancelled", slot["status"], "cancelled")
api._sync_jobs["c2"] = job_with("running", status="cancelled")
s2 = api._sync_jobs["c2"]["per_target"][0]
api._sync_slot_end("c2", s2, "failed", "rsync killed")
check("forced exit after cancel stays cancelled", (s2["status"], s2["error"]), ("cancelled", None))

# Cancel: label first, verified remote reap, unconfirmed reported
reaped = []


def fake_reap(job_id, model, src, dst):
    reaped.append(dst["id"])
    return dst["id"] == "ultra-256b"        # still alive on .31
api._sync_reap_remote = fake_reap
api._sync_jobs["k1"] = job_with("done", "running")
res = asyncio.run(api.admin_sync_job_cancel("k1"))
k1 = api._sync_jobs["k1"]
check("cancel: job cancelled", k1["status"], "cancelled")
check("cancel: only the unfinished slot is reaped", reaped, ["ultra-256b"])
check("cancel: unconfirmed reported", res.get("unconfirmed"), ["ultra-256b"])
check("cancel: done slot untouched", k1["per_target"][0]["status"], "done")
check("cancel: open slot cancelled with a warning",
      (k1["per_target"][1]["status"], bool(k1["per_target"][1].get("error"))), ("cancelled", True))

# Boot reattach: watch while alive, then judge by sizes
api._SYNC_WATCH_EVERY_S = 0


def run_reattach(alive_seq, src_man, dst_man, statuses=("running",)):
    seq = list(alive_seq)
    api._sync_remote_alive = lambda ssh, pats: seq.pop(0) if seq else False
    calls = []

    async def fake_manifest(ssh, path, timeout=30.0):
        calls.append(ssh)
        return src_man if ssh == SRC["ssh"] else dst_man
    api._dir_manifest = fake_manifest
    payload = json.loads(json.dumps(job_with(*statuses)))
    payload["id"] = "r1"
    api._persist.load_unfinished_jobs = lambda: [payload]
    api._sync_jobs.pop("r1", None)

    async def go():
        await api._sync_reattach()
        await asyncio.gather(*list(api._sync_watchers))
    asyncio.run(go())
    return api._sync_jobs["r1"], seq


j, left = run_reattach([True, True, False], {"a": 5}, {"a": 5})
check("reattach: waits until the remote transfer is gone", left, [])
check("reattach: converged -> slot done, job done",
      (j["per_target"][0]["status"], j["status"]), ("done", "done"))
j, _ = run_reattach([False], {"a": 5}, {"a": 2})
check("reattach: not converged -> interrupted, retry hint",
      (j["per_target"][0]["status"], j["status"], "retry" in (j["per_target"][0].get("error") or "")),
      ("interrupted", "interrupted", True))
j, _ = run_reattach([], {"a": 5}, {"a": 5}, statuses=("done", "running"))
check("reattach: a slot already done stays done", j["per_target"][0]["status"], "done")

print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
sys.exit(1 if FAILS else 0)
