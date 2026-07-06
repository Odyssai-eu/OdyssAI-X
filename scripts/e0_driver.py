"""E0 harness driver — multi-rank draft-spec alignment gate (G0).

Runs ON a single node (localhost ring, no network variable). Two phases:

  A. AR reference: 1 process, world_size=1, NO draft — per-prompt token
     canaries + full text.
  B. Spec multirank: 2 processes, ring over 127.0.0.1, target TP (our
     auto_parallel), draft REPLICATED per rank (RUNNER_SPEC_MULTIRANK=1),
     RUNNER_TOKEN_CANARY=1 — per-rank canary series.

G0 passes iff, for every prompt and repetition:
  * rank0 and rank1 canary series are IDENTICAL (same (ntoks, sha) pairs);
  * phase-B rank0 text == phase-A text (greedy exact-match spec must not
    change the output).

Usage (on the node):
  .venv/bin/python3 e0_driver.py --workdir /tmp/mtp-e0 \
      --model /path/to/target --draft /path/to/draft \
      --max-tokens 500 --reps 2
Writes <workdir>/e0-report.json and exits 0 only on G0 PASS.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROMPTS = [
    "Write a Python function that parses an ISO-8601 timestamp without using "
    "external libraries, then explain its edge cases.",
    "Explain, step by step, why the sky appears blue during the day and "
    "reddish at sunset. Keep it rigorous but accessible.",
    "List and briefly describe seven practical strategies to reduce tail "
    "latency in a distributed key-value store.",
]

READY_TIMEOUT_S = 900
DONE_TIMEOUT_S = 1200


def _spawn(rank: int, world: int, args, hostfile: str, spec: bool,
           logdir: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        "MLX_RANK": str(rank),
        "MLX_WORLD_SIZE": str(world),
        "MLX_METAL_FAST_SYNCH": "1",
        "RUNNER_MODEL": args.model,
        "RUNNER_MODE": "tensor",
        "RUNNER_BACKEND": "ring",
        "RUNNER_USE_AP": "1",
        "RUNNER_KV_Q8": "0",
        "RUNNER_EMIT_BATCH": "10",
        "RUNNER_TOKEN_CANARY": "1",
        "RUNNER_BATCH": "0",          # legacy loop everywhere (uniformity)
    })
    if world > 1:
        env["MLX_HOSTFILE"] = hostfile
    if spec:
        env["RUNNER_DRAFT_MODEL"] = args.draft
        env["RUNNER_NUM_DRAFT_TOKENS"] = str(args.num_draft)
        env["RUNNER_SPEC_MULTIRANK"] = "1"
    err = open(logdir / f"rank{rank}.stderr", "w")
    # Rank>0 stdout MUST be discarded: runner emit() writes on every rank and
    # an unread PIPE back-pressures the process mid-collective — the #23
    # deadlock (reproduced tonight: rank1 blocked on stdout write, rank0
    # blocked waiting for it in the ring op, both at 0% CPU).
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "runner.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE if rank == 0 else subprocess.DEVNULL,
        stderr=err, env=env, text=True, bufsize=1,
    )


def _wait_event(proc: subprocess.Popen, name: str, timeout: float,
                collect_text: bool = False) -> tuple[dict, str]:
    """Read rank-0 stdout until event `name`; optionally collect token text."""
    t0 = time.time()
    text_parts: list[str] = []
    while time.time() - t0 < timeout:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise RuntimeError(f"runner died (rc={proc.returncode})")
            continue
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if collect_text and ev.get("event") == "token":
            text_parts.append(ev.get("text", ""))
        if ev.get("event") == name:
            return ev, "".join(text_parts)
        if ev.get("event") == "fatal":
            raise RuntimeError(f"runner fatal: {ev}")
    raise TimeoutError(f"no {name} within {timeout}s")


def _send(procs: list[subprocess.Popen], obj: dict) -> None:
    line = json.dumps(obj) + "\n"
    for p in procs:
        p.stdin.write(line)
        p.stdin.flush()


def _run_phase(args, world: int, spec: bool, label: str,
               logdir: Path) -> dict:
    hostfile = str(logdir / "hostfile.json")
    port0 = args.port
    Path(hostfile).write_text(json.dumps(
        [[f"127.0.0.1:{port0 + r}"] for r in range(world)]))
    procs = [_spawn(r, world, args, hostfile, spec, logdir)
             for r in range(world)]
    results: dict = {"label": label, "requests": []}
    try:
        _wait_event(procs[0], "ready", READY_TIMEOUT_S)
        print(f"[{label}] ready", flush=True)
        reps = args.reps if spec else 1
        for rep in range(reps):
            for pi, prompt in enumerate(PROMPTS):
                rid = f"{label}-p{pi}-r{rep}"
                _send(procs, {
                    "cmd": "gen",
                    "id": rid,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.max_tokens,
                })
                done, text = _wait_event(procs[0], "done", DONE_TIMEOUT_S,
                                         collect_text=True)
                results["requests"].append({
                    "id": rid, "prompt_idx": pi, "rep": rep,
                    "ntoks": done.get("ntoks"), "tps": done.get("tps"),
                    "text": text,
                })
                print(f"[{label}] {rid}: {done.get('ntoks')} toks "
                      f"@ {done.get('tps'):.2f} tok/s", flush=True)
        _send(procs, {"cmd": "stop"})
        for p in procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    # Parse canary series per rank from stderr files.
    results["canaries"] = {}
    for r in range(world):
        series: dict[str, list] = {}
        pat = re.compile(r"\[canary\] (\{.*\})")
        for line in (logdir / f"rank{r}.stderr").read_text().splitlines():
            m = pat.search(line)
            if not m:
                continue
            c = json.loads(m.group(1))
            series.setdefault(c.get("rid", "?"), []).append(
                [c.get("ntoks"), c.get("sha")])
        results["canaries"][f"rank{r}"] = series
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--workdir", default="/tmp/mtp-e0")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--num-draft", type=int, default=4)
    ap.add_argument("--port", type=int, default=5580)
    ap.add_argument("--skip-ar", action="store_true")
    args = ap.parse_args()

    wd = Path(args.workdir)
    report: dict = {"model": args.model, "draft": args.draft, "phases": {}}

    if not args.skip_ar:
        ar_dir = wd / "phaseA-ar"
        ar_dir.mkdir(parents=True, exist_ok=True)
        report["phases"]["ar"] = _run_phase(args, 1, False, "AR", ar_dir)

    spec_dir = wd / "phaseB-spec"
    spec_dir.mkdir(parents=True, exist_ok=True)
    report["phases"]["spec"] = _run_phase(args, 2, True, "SPEC", spec_dir)

    # ── G0 verdict ─────────────────────────────────────────────────────────
    failures: list[str] = []
    spec = report["phases"]["spec"]
    c0 = spec["canaries"].get("rank0", {})
    c1 = spec["canaries"].get("rank1", {})
    for rid, s0 in c0.items():
        s1 = c1.get(rid)
        if s1 is None:
            failures.append(f"{rid}: rank1 emitted no canaries")
        elif s0 != s1:
            failures.append(f"{rid}: canary series diverge "
                            f"(rank0 {len(s0)} pts vs rank1 {len(s1)})")
    if not c0:
        failures.append("rank0 emitted no canaries at all")

    if "ar" in report["phases"]:
        ar_by_prompt = {r["prompt_idx"]: r["text"]
                        for r in report["phases"]["ar"]["requests"]}
        for r in spec["requests"]:
            ref = ar_by_prompt.get(r["prompt_idx"])
            if ref is not None and r["text"] != ref:
                failures.append(f"{r['id']}: spec text != AR text "
                                f"(spec {len(r['text'])}ch vs AR {len(ref)}ch)")

    report["failures"] = failures
    report["g0"] = "PASS" if not failures else "FAIL"
    (wd / "e0-report.json").write_text(json.dumps(report, indent=1))
    print(f"G0: {report['g0']}")
    for f in failures:
        print(f"  FAIL: {f}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
