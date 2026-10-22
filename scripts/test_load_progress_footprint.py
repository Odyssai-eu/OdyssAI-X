#!/usr/bin/env python3
"""The load poller measures the runners' physical footprint, not their RSS.

MLX keeps the weights in Metal buffers, which RSS does not count: on a loaded
Muse rank RSS read 8.1 GiB for a 171 GiB footprint, so the poller saw a few GB
that stopped growing and flagged a healthy load as STALLED (2026-10-22). This
runs `_runner_footprint_cmd` on this Mac against a helper process that holds a
buffer, and checks the reading against the buffer size. With an mlx Python at
~/mlx-cluster/.venv the buffer is a Metal array, and RSS is checked to miss it.

    .venv/bin/python scripts/test_load_progress_footprint.py
"""
import os
import subprocess
import sys
import tempfile
import time
import uuid

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []
GIB = 1024**3


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


MLX_PYTHON = os.path.expanduser("~/mlx-cluster/.venv/bin/python")
HELPER_MLX = ("import mlx.core as mx, sys, time\n"
              "a = mx.ones((256, 1024, 1024), dtype=mx.float32); mx.eval(a)\n"
              "print('ready', flush=True); time.sleep(60)\n")
HELPER_PLAIN = ("import sys, time\n"
                "a = bytearray(1 << 30)\n"
                "for i in range(0, len(a), 16384): a[i] = 1\n"
                "print('ready', flush=True); time.sleep(60)\n")

use_mlx = os.path.exists(MLX_PYTHON) and subprocess.run(
    [MLX_PYTHON, "-c", "import mlx.core"], capture_output=True).returncode == 0
marker = f"odyssai-footprint-test-{uuid.uuid4().hex}"
helper = subprocess.Popen(
    [MLX_PYTHON if use_mlx else sys.executable, "-c", HELPER_MLX if use_mlx else HELPER_PLAIN, marker],
    stdout=subprocess.PIPE, text=True)
try:
    check("helper ready", helper.stdout.readline().strip(), "ready")
    time.sleep(0.5)
    out = subprocess.run(["bash", "-c", api._runner_footprint_cmd(marker)],
                         capture_output=True, text=True, timeout=30)
    footprint = int(out.stdout.strip() or 0)
    rss = int(subprocess.run(["ps", "-o", "rss=", "-p", str(helper.pid)],
                             capture_output=True, text=True).stdout.strip() or 0) * 1024
    print(f"     helper={'mlx Metal' if use_mlx else 'bytearray'} "
          f"footprint={footprint / GIB:.2f} GiB rss={rss / GIB:.2f} GiB")
    check("footprint counts the 1 GiB buffer", footprint >= 0.95 * GIB, True)
    check("footprint stays near the buffer (one process matched)", footprint < 2 * GIB, True)
    if use_mlx:
        check("RSS misses the Metal buffer (the bug)", rss < 0.5 * GIB, True)
    out = subprocess.run(["bash", "-c", api._runner_footprint_cmd(marker + "-absent")],
                         capture_output=True, text=True, timeout=30)
    check("no matching process -> 0", out.stdout.strip(), "0")
finally:
    helper.kill()

print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAIL'}")
sys.exit(1 if FAILS else 0)
