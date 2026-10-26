#!/usr/bin/env python3
"""A shard over 50 GB must not stop a download (1.62.12).

huggingface_hub 1.16 refuses client-side to start a file over 50 GB on the plain
HTTP path, and the downloader forces that path (HF_HUB_DISABLE_XET=1, xet does
not resume). Step-5-Preview-BF16 (21 shards of 54.2 GB) failed on it. The
downloader now puts a sitecustomize on the `hf download` PYTHONPATH that lifts
the constant, without touching the node venv or the `hf` argv.

Part 1 (local): the launch command writes the sitecustomize, sets PYTHONPATH,
keeps HF_HUB_DISABLE_XET and keeps the argv the kill pattern matches.
Part 2 (node venv, the real library): http_get with a 60 GB expected size
raises the "too large" ValueError without the sitecustomize, and goes on to the
network (connection refused on a closed local port) with it. No HF traffic.

    .venv/bin/python scripts/test_hf_large_file_guard.py [admin@192.168.86.29]
"""
import os
import re
import shlex
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


# --- 1. the launch command ---------------------------------------------------
lay = api._dl_layout("/Volumes/models/odysseus", "stepfun-ai/Step-5-Preview-BF16")
cmd = api._hf_dl_cmd(lay, "hf_secret", {"model-00004.safetensors": 54208650568, "README.md": 9})
site = api._HF_SITE_DIR_REMOTE
check("writes the sitecustomize", f'> "{site}/sitecustomize.py"' in cmd, True)
check("sitecustomize written before hf starts",
      cmd.index("sitecustomize.py") < cmd.index(" download "), True)
check("hf runs with the site dir on PYTHONPATH", f'PYTHONPATH="{site}" ' in cmd, True)
check("xet stays disabled", "HF_HUB_DISABLE_XET=1 " in cmd, True)
check("size manifest still written", api.SIZES_MANIFEST in cmd, True)
pat = api._dl_kill_pattern(lay["repo_id"], lay["local_dir"], lay["include"])
hf_argv = cmd[cmd.index(api._dl_shell_stable_tail(api.HF_BIN_REMOTE)):]
hf_argv = hf_argv.replace("'", "")
check("kill pattern still matches the hf argv", bool(re.search(pat, hf_argv)), True)
sub = api._hf_dl_cmd(api._dl_layout("/m", "org/name/6bit"), None)
check("subfolder spec keeps --include", "--include '6bit/*'" in sub and "PYTHONPATH=" in sub, True)

# --- 2. the real guard, in a node venv ---------------------------------------
node = sys.argv[1] if len(sys.argv) > 1 else "admin@192.168.86.29"
probe = (
    "import io\n"
    "from huggingface_hub import file_download as f\n"
    "try:\n"
    "    f.http_get('http://127.0.0.1:9/x', io.BytesIO(), expected_size=60_000_000_000)\n"
    "    print('RESULT none')\n"
    "except ValueError as e:\n"
    "    print('RESULT guard' if 'too large' in str(e) else 'RESULT other-value ' + str(e)[:80])\n"
    "except Exception as e:\n"
    "    print('RESULT network ' + type(e).__name__)\n"
)
remote = (
    "set -e; d=$(mktemp -d); "
    f"printf %s {shlex.quote(api._HF_SITECUSTOMIZE)} > $d/sitecustomize.py; "
    f"echo without; ~/mlx-cluster/.venv/bin/python -c {shlex.quote(probe)}; "
    f"echo with; PYTHONPATH=$d ~/mlx-cluster/.venv/bin/python -c {shlex.quote(probe)}; "
    "rm -rf $d"
)
r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", node, remote],
                   capture_output=True, text=True, timeout=120)
out = r.stdout
res = re.findall(r"RESULT (\S+)", out)
print(f"     node {node}: {res} {r.stderr.strip()[:200]}")
check("node venv: guard refuses without the sitecustomize", res[:1], ["guard"])
check("node venv: guard lifted with it (reaches the network)", res[1:2], ["network"])

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("all checks passed")
