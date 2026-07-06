"""E3 driver — native-MTP multi-NODE gate (G2), engine-free.

Same contract as e0_driver but ranks are spawned over ssh on real nodes
(RunnerProc pattern: held ssh, stdin JSONL, rank-0 stdout only). Two runs
on the SAME topology:

  A. AR baseline  (RUNNER_MTP absent)
  B. native MTP   (RUNNER_MTP=native + sidecar), canaries per round

G2 verdict: (a) per-(rid,round) canary tuples identical across ALL ranks;
(b) MTP text == AR text per prompt (greedy exactness);
(c) speedup = tps_mtp / tps_ar reported (gate thresholds applied by the
operator: <1.1 stop, >=1.25 pass).

Prereqs on every node: workdir with runner.py/mtp_module.py/mtp_spec.py
(+ auto_parallel.py/exo_stubs.py), the model dir, and the mtp sidecar.

Usage:
  python3 e3_driver.py --nodes 192.168.86.29,192.168.86.30,... \
      --model /Volumes/models/.../GLM-5.2-Q6 \
      --sidecar /Volumes/models/.../sidecar/GLM-5.2-mtp \
      --workdir /tmp/mtp-e3 --venv '~/mlx-cluster/.venv/bin/python3' \
      --mode pipeline --max-tokens 256 --depth 3
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
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
READY_TIMEOUT_S = 1800
DONE_TIMEOUT_S = 2400


def _spawn(rank: int, ip: str, args, mtp_on: bool) -> subprocess.Popen:
    env = {
        "MLX_RANK": str(rank),
        "MLX_WORLD_SIZE": str(len(args.node_list)),
        "MLX_METAL_FAST_SYNCH": "1",
        "MLX_HOSTFILE": f"{args.workdir}/hostfile.json",
        "RUNNER_MODEL": args.model,
        "RUNNER_MODE": args.mode,
        "RUNNER_BACKEND": "ring",
        "RUNNER_USE_AP": "1" if args.use_ap else "0",
        "RUNNER_KV_Q8": "0",
        "RUNNER_EMIT_BATCH": "10",
        "RUNNER_BATCH": "0",
    }
    if mtp_on:
        env["RUNNER_MTP"] = "native"
        env["RUNNER_MTP_DEPTH"] = str(args.depth)
        env["RUNNER_MTP_SIDECAR"] = args.sidecar
        env["RUNNER_MTP_HIDDEN"] = args.hidden_source
        if args.quantize:
            env["RUNNER_MTP_QUANT"] = "1"
        if args.timing:
            env["TIMING_MTP"] = "1"
    hosts = json.dumps([[f"{n}:{args.port}"] for n in args.node_list])
    env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    # `exec VAR=x cmd` is not valid zsh (tries to exec "VAR=x" -> 127);
    # `exec env VAR=x cmd` is portable across sh/zsh.
    remote = (
        f"echo {shlex.quote(hosts)} > {args.workdir}/hostfile.json && "
        f"cd {args.workdir} && exec env {env_str} {args.venv} runner.py"
    )
    return subprocess.Popen(
        ["ssh", "-o", "ServerAliveInterval=10", f"admin@{ip}", remote],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE if rank == 0 else subprocess.DEVNULL,
        stderr=open(f"{args.logdir}/{'mtp' if mtp_on else 'ar'}-rank{rank}.stderr", "w"),
        text=True, bufsize=1,
    )


def _wait_event(proc, name, timeout, collect_text=False):
    t0 = time.time()
    parts: list[str] = []
    while time.time() - t0 < timeout:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise RuntimeError(f"rank0 ssh died (rc={proc.returncode})")
            continue
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if collect_text and ev.get("event") == "token":
            parts.append(ev.get("text", ""))
        if ev.get("event") == name:
            return ev, "".join(parts)
    raise TimeoutError(f"no {name} within {timeout}s")


def _run(args, mtp_on: bool, label: str) -> dict:
    procs = [_spawn(r, ip, args, mtp_on) for r, ip in enumerate(args.node_list)]
    out: dict = {"label": label, "requests": []}
    try:
        ev, _ = _wait_event(procs[0], "ready", READY_TIMEOUT_S)
        print(f"[{label}] ready (load {ev.get('load_s', 0):.0f}s, "
              f"mtp={ev.get('mtp')})", flush=True)
        if mtp_on and not ev.get("mtp"):
            raise RuntimeError("runner ready WITHOUT mtp module — check sidecar")
        for pi, prompt in enumerate(PROMPTS):
            rid = f"{label}-p{pi}"
            line = json.dumps({
                "cmd": "gen", "id": rid,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
            }) + "\n"
            for p in procs:
                p.stdin.write(line)
                p.stdin.flush()
            done, text = _wait_event(procs[0], "done", DONE_TIMEOUT_S, True)
            out["requests"].append({"id": rid, "prompt_idx": pi,
                                    "ntoks": done.get("ntoks"),
                                    "tps": done.get("tps"), "text": text})
            print(f"[{label}] {rid}: {done.get('ntoks')} toks "
                  f"@ {done.get('tps'):.2f} tok/s", flush=True)
        for p in procs:
            try:
                p.stdin.write(json.dumps({"cmd": "stop"}) + "\n")
                p.stdin.flush()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=90)
            except subprocess.TimeoutExpired:
                p.kill()
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    # canaries per rank from stderr files
    out["canaries"] = {}
    pat = re.compile(r"\[canary\] (\{.*\})")
    for r in range(len(args.node_list)):
        series: dict[str, list] = {}
        f = Path(args.logdir) / f"{'mtp' if mtp_on else 'ar'}-rank{r}.stderr"
        for line in f.read_text().splitlines():
            m = pat.search(line)
            if not m:
                continue
            c = json.loads(m.group(1))
            series.setdefault(c.get("rid", "?"), []).append(
                [c.get("round"), c.get("drafted"), c.get("accepted"),
                 c.get("sha")])
        out["canaries"][f"rank{r}"] = series
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, help="comma-separated LAN IPs, rank order")
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--workdir", default="/tmp/mtp-e3")
    ap.add_argument("--logdir", default=".")
    ap.add_argument("--venv", default="~/mlx-cluster/.venv/bin/python3")
    ap.add_argument("--mode", default="pipeline")
    ap.add_argument("--use-ap", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--hidden-source", default="post_norm")
    ap.add_argument("--quantize", action="store_true",
                    help="quantize the MTP module to the trunk's quant on load")
    ap.add_argument("--timing", action="store_true",
                    help="emit [mtp-timing] per-phase breakdown")
    ap.add_argument("--port", type=int, default=5590)
    ap.add_argument("--skip-ar", action="store_true")
    args = ap.parse_args()
    args.node_list = [n.strip() for n in args.nodes.split(",") if n.strip()]
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    report: dict = {"nodes": args.node_list, "model": args.model,
                    "depth": args.depth, "phases": {}}
    if not args.skip_ar:
        report["phases"]["ar"] = _run(args, False, "AR")
    report["phases"]["mtp"] = _run(args, True, "MTP")

    failures: list[str] = []
    mtp = report["phases"]["mtp"]
    ranks = list(mtp["canaries"])
    base = mtp["canaries"].get("rank0", {})
    for rid, s0 in base.items():
        for rk in ranks[1:]:
            s = mtp["canaries"][rk].get(rid)
            if s != s0:
                failures.append(f"{rid}: canary divergence rank0 vs {rk}")
    if not base:
        failures.append("no canaries from rank0")
    speed = {}
    if "ar" in report["phases"]:
        ar_by = {r["prompt_idx"]: r for r in report["phases"]["ar"]["requests"]}
        for r in mtp["requests"]:
            ref = ar_by.get(r["prompt_idx"])
            if ref and r["text"] != ref["text"]:
                failures.append(f"{r['id']}: MTP text != AR text")
            if ref and ref["tps"]:
                speed[r["id"]] = round(r["tps"] / ref["tps"], 3)
    report["failures"] = failures
    report["speedup"] = speed
    report["g2_alignment"] = "PASS" if not failures else "FAIL"
    Path(args.logdir, "e3-report.json").write_text(json.dumps(report, indent=1))
    print(f"alignment: {report['g2_alignment']}  speedups: {speed}")
    for f in failures:
        print(f"  FAIL: {f}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
