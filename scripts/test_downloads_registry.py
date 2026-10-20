#!/usr/bin/env python3
"""HF downloads (1.61.0): spec normalisation, exact kill pattern, persisted
registry and boot-time reattach without relaunch. No node needed: SSH is stubbed.

    python3 scripts/test_downloads_registry.py
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile

_TMP = tempfile.mkdtemp()
os.environ.setdefault("CLUSTER_CONFIG_FILE", os.path.join(_TMP, "cc.json"))
os.environ.setdefault("ODYSSAI_X_STATE_DIR", _TMP)
os.environ["ODYSSAI_X_DOWNLOADS_FILE"] = os.path.join(_TMP, "downloads.json")
json.dump({}, open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402
from fastapi import HTTPException  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("ok   " if ok else "FAIL ") + label + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


MD = "/volumes/models/odysseus"
HOST = {"id": "ultra-512", "ssh": "admin@node", "models_dir": MD}
api.HOSTS_REGISTRY[:] = [HOST]
api._resolve_host = lambda hid: HOST if hid == HOST["id"] else None
BIN = "/Users/admin/mlx-cluster/.venv/bin/hf"
api.HF_BIN_REMOTE = "$HOME/mlx-cluster/.venv/bin/hf"
PY = "/opt/homebrew/Cellar/python@3.11/3.11.15_1/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python"
DS = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
ARGV_DS = f"{PY} {BIN} download {DS} --local-dir {MD}/{DS} --max-workers 4"


def ere_match(pattern, line):
    """Match like `pgrep -f` does: an ERE against the space-joined argv."""
    return subprocess.run(["grep", "-qE", "--", pattern],
                          input=line, text=True).returncode == 0


# ── 1. spec normalisation ────────────────────────────────────────────────────
check("leading space stripped", api._normalize_hf_repo(" " + DS), DS)
check("trailing blanks + slashes stripped", api._normalize_hf_repo(f"/{DS}/ \n"), DS)
check("subfolder spec kept", api._normalize_hf_repo("org/name/6bit"), "org/name/6bit")
for bad in ["deepseek-ai", "org /name", "org/ name", "org//name", "org/../etc",
            "org/name/..", "", "   ", "org/na me", "-org/name"]:
    try:
        api._normalize_hf_repo(bad)
        check(f"refused {bad!r}", "accepted", "refused")
    except ValueError:
        check(f"refused {bad!r}", "refused", "refused")

# ── 2. exact kill pattern ────────────────────────────────────────────────────
lay = api._dl_layout(MD, DS)
pat = api._dl_kill_pattern(lay["repo_id"], lay["local_dir"], lay["include"])
check("pattern matches the live argv", ere_match(pat, ARGV_DS), True)
check("pattern does not match itself (sh -c wrapper)", ere_match(pat, f"zsh -c pkill -f -- '{pat}'"), False)
bad_lay = api._dl_layout(MD, " " + DS)
bad_pat = api._dl_kill_pattern(bad_lay["repo_id"], bad_lay["local_dir"], bad_lay["include"])
check("space-dir pattern spares the real download", ere_match(bad_pat, ARGV_DS), False)
check("space-dir pattern matches its own process",
      ere_match(bad_pat, f"{PY} {BIN} download {DS} --local-dir {MD}/ {DS} --max-workers 4"), True)
check("no match on a longer sibling dir",
      ere_match(pat, f"{PY} {BIN} download {DS} --local-dir {MD}/{DS}-old --max-workers 4"), False)
sub = api._dl_layout(MD, "org/name/6bit")
sub_pat = api._dl_kill_pattern(sub["repo_id"], sub["local_dir"], sub["include"])
sub_argv = f"{PY} {BIN} download org/name --include 6bit/* --local-dir {MD}/org/name --max-workers 4"
whole_argv = f"{PY} {BIN} download org/name --local-dir {MD}/org/name --max-workers 4"
check("subfolder pattern matches its argv", ere_match(sub_pat, sub_argv), True)
check("subfolder pattern spares the whole-repo pull", ere_match(sub_pat, whole_argv), False)
whole = api._dl_layout(MD, "org/name")
whole_pat = api._dl_kill_pattern(whole["repo_id"], whole["local_dir"], whole["include"])
check("whole-repo pattern spares the subfolder pull", ere_match(whole_pat, sub_argv), False)

# ── 3. scan parser ───────────────────────────────────────────────────────────
rx = api._dl_scan_re()
m = rx.search(ARGV_DS)
check("scan parses repo/local-dir", (m.group(1), m.group(2), m.group(3)), (DS, None, f"{MD}/{DS}"))
m = rx.search(sub_argv)
check("scan parses --include", (m.group(1), m.group(2), m.group(3)), ("org/name", "6bit/*", f"{MD}/org/name"))
m = rx.search(f"{PY} {BIN} download {DS} --local-dir {MD}/ {DS} --max-workers 4")
check("scan keeps a space-bearing dir whole", m.group(3), f"{MD}/ {DS}")


# ── 4. persistence + reattach ────────────────────────────────────────────────
class FakeNode:
    """Stands in for `_ssh_exec` on the node: a set of live argv lines."""

    def __init__(self):
        self.live = set()
        self.calls = []
        self.find_rc, self.find_out = 0, "1\n2048\n"     # partial files remain

    def ssh(self, target, cmd, timeout=12):
        self.calls.append(cmd)
        if cmd.startswith("ps -axo"):
            return 0, "".join(l + "\n" for l in self.live), ""
        if cmd.startswith("pkill -f"):
            p = cmd.split("pkill -f -- ", 1)[1].split("; ", 1)[0]
            p = p[1:-1].replace("'\"'\"'", "'") if p.startswith("'") else p
            self.live = {l for l in self.live if not ere_match(p, l)}
            return 0, "GONE\n", ""
        if cmd.startswith("pgrep -f"):
            p = cmd.split("pgrep -f -- ", 1)[1].split(" >/dev/null", 1)[0]
            p = p[1:-1].replace("'\"'\"'", "'") if p.startswith("'") else p
            alive = any(ere_match(p, l) for l in self.live)
            return 0, ("ALIVE" if alive else "DEAD") + "\n1024\n", ""
        if cmd.startswith("find "):
            return self.find_rc, self.find_out, ""
        return 0, "", ""


node = FakeNode()
api._ssh_exec = node.ssh


async def _no_size(repo, token):
    return 510311045295


api._hf_repo_total_bytes = _no_size
spawned = []
api._dl_spawn = lambda dl_id, repo, tok, targets: spawned.append((dl_id, repo))


async def boot():
    """What a container restart does to the registry: memory gone, reload."""
    for t in list(api._dl_tasks.values()):
        t.cancel()
    await asyncio.sleep(0)
    api._downloads.clear()
    api._dl_tasks.clear()
    api._dl_ready.clear()
    api._dl_last_saved = None
    await api._dl_reattach()
    api._dl_ready.set()
    await asyncio.sleep(0.05)


async def main():
    # Two orphans from before 1.61.0 (no registry file at all).
    node.live = {ARGV_DS,
                 f"{PY} {BIN} download inclusionAI/Ling-3.1-flash --local-dir {MD}/inclusionAI/Ling-3.1-flash --max-workers 4",
                 "zsh -c mkdir -p x && HF_TOKEN=secret " + ARGV_DS}
    await boot()
    jobs = sorted(api._downloads.values(), key=lambda j: j["repo"])
    check("orphans adopted", [(j["repo"], j["status"], j.get("adopted")) for j in jobs],
          [(DS, "running", True), ("inclusionAI/Ling-3.1-flash", "running", True)])
    check("adopted job has the standard layout", jobs[0].get("adopted_layout"), "standard")
    check("nothing relaunched at boot", spawned, [])
    saved = open(os.environ["ODYSSAI_X_DOWNLOADS_FILE"]).read()
    check("token never on disk", "secret" in saved or "_hf_token" in saved, False)
    ids_before = sorted(api._downloads)

    # Restart #2: same transfers → same ids, no duplicate entry.
    await boot()
    check("restart keeps the same jobs, no duplicate", sorted(api._downloads), ids_before)
    check("still running after restart",
          sorted(j["status"] for j in api._downloads.values()), ["running", "running"])

    # Creating the same spec again (with the 2026-10-20 typo) is refused.
    req = api.HFDownloadRequest(repo=" " + DS, targets=["ultra-512"])
    try:
        await api.admin_downloads_create(req)
        check("duplicate create refused", "accepted", "409")
    except HTTPException as e:
        check("duplicate create refused", e.status_code, 409)
    try:
        await api.admin_downloads_create(api.HFDownloadRequest(repo="org /name"))
        check("malformed repo refused", "accepted", "400")
    except HTTPException as e:
        check("malformed repo refused", e.status_code, 400)
    out = await api.admin_downloads_create(api.HFDownloadRequest(repo=" org/new-model ",
                                                                 targets=["ultra-512"]))
    check("create normalises the spec", (out["repo"], spawned[-1][1]), ("org/new-model", "org/new-model"))

    # Cancel an adopted job: the remote process is reaped and confirmed gone,
    # the other download is untouched.
    ds_id = next(j["id"] for j in api._downloads.values() if j["repo"] == DS)
    res = await api.admin_downloads_cancel(ds_id)
    check("cancel status", res["status"], "cancelled")
    check("cancel confirms the remote is gone", res["remote_alive"], [])
    check("DeepSeek process gone", any(DS + " --local-dir" in l for l in node.live), False)
    check("Ling process untouched", any("Ling-3.1-flash" in l for l in node.live), True)

    # Retry refuses while a transfer into the directory is alive.
    node.live.add(ARGV_DS)
    try:
        await api.admin_downloads_retry(ds_id)
        check("retry refused over a live transfer", "accepted", "409")
    except HTTPException as e:
        check("retry refused over a live transfer", e.status_code, 409)
    node.live.discard(ARGV_DS)

    # Restart with the cancelled job on disk and its process gone: stays cancelled.
    await boot()
    check("cancelled job stays cancelled", api._downloads[ds_id]["status"], "cancelled")
    # The Ling transfer dies while we are detached: verdict from the disk.
    ling_id = next(j["id"] for j in api._downloads.values() if "Ling" in j["repo"])
    node.live = set()
    for _ in range(50):
        await asyncio.sleep(0.2)
        if api._downloads[ling_id]["status"] != "running":
            break
    check("dead transfer with partial files → failed (retryable)",
          api._downloads[ling_id]["status"], "failed")
    # Shutdown freeze: once frozen, in-memory relabels never reach the file.
    api._dl_save()
    before = open(os.environ["ODYSSAI_X_DOWNLOADS_FILE"]).read()
    api._dl_persist_frozen = True
    api._downloads[ling_id]["status"] = "running"
    api._dl_save()
    check("frozen snapshot untouched", open(os.environ["ODYSSAI_X_DOWNLOADS_FILE"]).read(), before)
    api._dl_persist_frozen = False
    for t in list(api._dl_tasks.values()):
        t.cancel()

    # Verdict on a node that cannot be read (rebooting, volume not mounted):
    # no label, the watcher asks again — unless it gives up.
    slot = {"ssh": "admin@node", "status": "running", "local_dir": f"{MD}/{DS}",
            "target": f"{MD}/{DS}", "bytes": 5}
    for rc, out, label in [(255, "", "ssh down"), (0, "0\n", "volume not mounted")]:
        node.find_rc, node.find_out = rc, out
        check(f"verdict waits when {label}", (api._dl_slot_verdict(slot, 1000), slot["status"]),
              (False, "running"))
    check("verdict gives up when told to", (api._dl_slot_verdict(slot, 1000, give_up=True),
                                              slot["status"]), (True, "failed"))
    slot.update(status="running", error=None)
    node.find_rc, node.find_out = 0, "0\n1\n"          # no .incomplete, 1 KiB on disk
    check("readable and complete → completed", (api._dl_slot_verdict(slot, 1024), slot["status"]),
          (True, "completed"))

    # The launch command always takes hf's HTTP path (Xet does not resume).
    seen = []

    async def fake_exec(*argv, **kw):
        seen.append(argv[-1])

        class P:
            returncode = 0

            async def communicate(self):
                return b"", b""
        return P()
    real_exec = api.asyncio.create_subprocess_exec
    api.asyncio.create_subprocess_exec = fake_exec
    try:
        await api._hf_dl_one("x1", HOST, DS, "tok", {})
    finally:
        api.asyncio.create_subprocess_exec = real_exec
    check("launch disables Xet", "HF_HUB_DISABLE_XET=1 HF_TOKEN=tok " in seen[0], True)
    check("launched argv still matches the kill pattern",
          ere_match(pat, f"{PY} {BIN} download {DS} --local-dir {MD}/{DS} --max-workers 4")
          and f"{BIN.split('/Users/admin', 1)[1]} download {DS} --local-dir {MD}/{DS} --max-workers 4" in seen[0],
          True)


asyncio.run(main())
print(f"\n{len(FAILS)} failure(s)")
sys.exit(1 if FAILS else 0)
