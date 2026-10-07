#!/usr/bin/env python3
"""Mid-generation errors must carry the stderr tails of the ranks that DIED.

2026-10-06, GLM-5.3-Flash Q6h16 pipeline: rank 1 died on its first token with
"[convert] Only length-1 arrays can be converted to Python scalars"; rank 0
exited controlled through the JACCL side channel and the surfaced message was
only "runner died mid-generation (rc=None, a peer rank died)" — the traceback
never reached the caller, only the container log. _midgen_death_report lists
every rank whose process has exited, with its stderr tail, bounded to 25
lines. No node is reached.

    .venv/bin/python scripts/test_midgen_death_report.py
"""
import json
import os
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({"argo": {"name": "A", "kind": "mlx-distributed", "backend": "ring",
                    "models_dir": "/m",
                    "nodes": [{"host": "n0", "ssh": "admin@198.51.100.1", "master": True},
                              {"host": "n1", "ssh": "admin@198.51.100.2"}]}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


class FakeProc:
    def __init__(self, rc):
        self._rc = rc

    def poll(self):
        return self._rc


class FakeRunner:
    def __init__(self, rank, rc, tail_lines):
        self.node = {"rank": rank}
        self.proc = FakeProc(rc)
        self._tail = tail_lines

    def stderr_tail(self, n=20):
        return "\n".join(self._tail[-n:])


traceback = [f"line{i}" for i in range(60)] + ["ValueError: [convert] Only length-1 arrays"]
rank1 = FakeRunner(1, 1, traceback)
rank0 = FakeRunner(0, 0, ["[jaccl] peer is gone: side channel to rank 1 closed"])
rank2_alive = FakeRunner(2, None, ["still serving"])

report = api._midgen_death_report([rank0, rank1, rank2_alive])
check("a dead rank's traceback is in the report (the 2026-10-06 case)",
      "ValueError: [convert] Only length-1 arrays" in report, True)
check("each dead rank is named with its exit code",
      ("--- rank 1 (exit=1) ---" in report, "--- rank 0 (exit=0) ---" in report), (True, True))
check("a live rank (poll None) is not reported", "rank 2" in report, False)
check("the tail is bounded to 25 lines per rank",
      len(report.split("--- rank 1 (exit=1) ---\n")[1].strip().splitlines()), 25)
check("the bounded tail keeps the END of the stderr (the exception), not the start",
      report.splitlines()[-1], "ValueError: [convert] Only length-1 arrays")
check("no exited rank → empty report (nothing appended to the error)",
      api._midgen_death_report([rank2_alive]), "")
check("a runner whose poll() throws is skipped, not fatal",
      "rank 1" in api._midgen_death_report([BrokenRunner := type("R", (), {
          "node": {"rank": 1}, "proc": None, "stderr_tail": lambda self, n=20: "x"})(), rank1]), True)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
