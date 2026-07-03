"""Master orchestrator for the long-lived JACCL runner.

Applies exo-style lifecycle disciplines:
  1. Random ephemeral coordinator port per session (avoids TCP TIME_WAIT collision).
  2. Long-lived runners on each node (init JACCL once, accept many prompts).
  3. Graceful teardown: send "stop" cmd, wait, then SIGTERM, then SIGKILL.
  4. Master orchestrates termination on every rank in lockstep.

Usage:
    python master.py --model mlx-community/GLM-4.5-Air-4bit --mode pipeline
    # then type prompts on stdin (one per line) — each is one inference
    # Ctrl+D to graceful-exit
"""

import argparse
import json
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

from topology import load_topology, to_nodes_dict

NODES: list[dict] = []
COORDINATOR_RANK = 0
REMOTE_CLUSTER_DIR = (os.environ.get("ODYSSAI_X_REMOTE_CLUSTER_DIR") or os.environ.get("ODYSSEUS_REMOTE_CLUSTER_DIR") or "$HOME/mlx-cluster").rstrip("/")
RUNNER_REMOTE = (os.environ.get("ODYSSAI_X_RUNNER_REMOTE") or os.environ.get("ODYSSEUS_RUNNER_REMOTE") or f"{REMOTE_CLUSTER_DIR}/runner.py")
PYTHON_REMOTE = (os.environ.get("ODYSSAI_X_PYTHON_REMOTE") or os.environ.get("ODYSSEUS_PYTHON_REMOTE") or f"{REMOTE_CLUSTER_DIR}/.venv/bin/python")


def load_nodes(cluster: str, size: int) -> list[dict]:
    topo = load_topology()
    if topo is None:
        raise SystemExit(
            "No topology.yaml found. Copy config/topology.example.yaml to "
            "~/.odysseus/topology.yaml or set ODYSSAI_X_TOPOLOGY."
        )
    if cluster not in topo.clusters:
        raise SystemExit(f"Cluster {cluster!r} not found in topology.yaml")
    pools = to_nodes_dict(topo.clusters[cluster])
    nodes = pools.get(size)
    if not nodes:
        available = ", ".join(str(k) for k in sorted(pools))
        raise SystemExit(
            f"Cluster {cluster!r} has no size={size} pool in topology.yaml "
            f"(available: {available or 'none'})"
        )
    return nodes


def random_ephemeral_port() -> int:
    # Mirror exo's random_ephemeral_port. Pick from the OS ephemeral range.
    return random.randint(49152, 65535)


def build_devices_matrix() -> list:
    return [n["rdma"] for n in sorted(NODES, key=lambda x: x["rank"])]


def remote_cmd(node: dict, model: str, mode: str, port: int, devices_json: str, use_ap: bool) -> str:
    coord_ip = next(n for n in NODES if n["rank"] == COORDINATOR_RANK)["ssh"].split("@")[1]
    env = {
        "MLX_RANK": str(node["rank"]),
        "MLX_JACCL_COORDINATOR": f"{coord_ip}:{port}",
        "MLX_IBV_DEVICES": "/tmp/mlx_jaccl_devices.json",
        "MLX_METAL_FAST_SYNCH": "1",
        "RUNNER_MODEL": model,
        "RUNNER_MODE": mode,
        "RUNNER_USE_AP": "1" if use_ap else "0",
        "RUNNER_EMIT_BATCH": os.environ.get("RUNNER_EMIT_BATCH", "10"),
        "RUNNER_LAYER_BOUNDS": os.environ.get("RUNNER_LAYER_BOUNDS", ""),
    }
    env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    write_devices = f"echo {shlex.quote(devices_json)} > /tmp/mlx_jaccl_devices.json"
    return f"{write_devices} && {env_str} {PYTHON_REMOTE} {RUNNER_REMOTE}"


class RunnerProc:
    def __init__(self, node: dict, cmd: str):
        self.node = node
        self.proc = subprocess.Popen(
            ["ssh", "-o", "ServerAliveInterval=10", node["ssh"], cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.ready = False
        self.events = []
        self._stop_streams = False
        self._lock = threading.Lock()
        # #29: signalled whenever a new event lands so the feed loop can BLOCK
        # on it instead of a 50ms busy-poll.
        self._event_signal = threading.Event()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()
        if node["rank"] == 0:
            self._stdout_thread = threading.Thread(
                target=self._drain_stdout, daemon=True
            )
            self._stdout_thread.start()

    def _drain_stderr(self):
        for line in self.proc.stderr:
            sys.stderr.write(f"[rank{self.node['rank']}] {line}")
            sys.stderr.flush()

    def _drain_stdout(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                sys.stderr.write(f"[rank0 raw] {line}\n")
                continue
            with self._lock:
                self.events.append(ev)
            self._event_signal.set()
            if ev.get("event") == "ready":
                self.ready = True

    def send(self, obj: dict):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()

    def close_stdin(self):
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception:
            pass

    def graceful_stop(self, soft_timeout=10.0, term_timeout=10.0):
        # 1. send stop cmd via stdin
        try:
            self.send({"cmd": "stop"})
        except Exception:
            pass
        self.close_stdin()
        # 2. wait soft
        t0 = time.time()
        while time.time() - t0 < soft_timeout:
            if self.proc.poll() is not None:
                return
            time.sleep(0.2)
        # 3. SIGTERM (on the SSH process, which propagates to remote python)
        sys.stderr.write(f"[rank{self.node['rank']}] soft stop timeout, SIGTERM\n")
        self.proc.terminate()
        t0 = time.time()
        while time.time() - t0 < term_timeout:
            if self.proc.poll() is not None:
                return
            time.sleep(0.2)
        # 4. SIGKILL
        sys.stderr.write(f"[rank{self.node['rank']}] SIGTERM timeout, SIGKILL\n")
        self.proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/GLM-4.5-Air-4bit")
    ap.add_argument("--mode", default="pipeline", choices=["pipeline", "tensor"])
    ap.add_argument("--use-ap", action="store_true",
                    help="Use exo-ported auto_parallel.py (required for qwen3_next, etc.)")
    ap.add_argument("--prompts-file", default=None,
                    help="JSONL file of prompts (one {prompt, max_tokens} per line). "
                         "If omitted, read prompts from stdin (one per line, blank to skip).")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--cluster", default="default",
                    help="Cluster key from topology.yaml")
    ap.add_argument("--nodes", type=int, default=1,
                    help="Pool size from topology.yaml")
    args = ap.parse_args()
    global NODES
    NODES = load_nodes(args.cluster, args.nodes)

    port = random_ephemeral_port()
    devices = build_devices_matrix()
    devices_json = json.dumps(devices)

    sys.stderr.write(
        f"[master] coordinator_port={port} devices={devices_json}\n"
        f"[master] starting runners on {len(NODES)} nodes...\n"
    )

    runners: list[RunnerProc] = []
    for node in NODES:
        cmd = remote_cmd(node, args.model, args.mode, port, devices_json, args.use_ap)
        runners.append(RunnerProc(node, cmd))

    # wait for rank 0 ready
    rank0 = next(r for r in runners if r.node["rank"] == 0)
    sys.stderr.write("[master] waiting rank 0 ready...\n")
    t0 = time.time()
    while not rank0.ready and time.time() - t0 < 600:
        if rank0.proc.poll() is not None:
            sys.stderr.write("[master] rank 0 died before ready\n")
            for r in runners:
                r.graceful_stop()
            sys.exit(1)
        time.sleep(0.5)
    if not rank0.ready:
        sys.stderr.write("[master] timeout waiting for ready\n")
        for r in runners:
            r.graceful_stop()
        sys.exit(1)
    sys.stderr.write(f"[master] rank 0 ready in {time.time()-t0:.1f}s, sending prompts\n")

    def _feed_from(iterator):
        for line in iterator:
            line = line.strip()
            if not line:
                continue
            if args.prompts_file:
                req = json.loads(line)
                prompt_text = req["prompt"]
                max_tokens = req.get("max_tokens", args.max_tokens)
            else:
                prompt_text = line
                max_tokens = args.max_tokens
            req_id = uuid.uuid4().hex[:8]
            sys.stderr.write(f"\n[master] >>> req {req_id}: {prompt_text[:60]!r}\n")
            req = {
                "cmd": "gen",
                "id": req_id,
                "prompt": prompt_text,
                "max_tokens": max_tokens,
            }
            for r in runners:
                r.send(req)
            # wait for done event from rank 0
            seen = 0
            while True:
                with rank0._lock:
                    evs = list(rank0.events)
                # find first done for our id
                done_ev = next(
                    (e for e in evs[seen:] if e.get("event") == "done" and e.get("id") == req_id),
                    None,
                )
                # print new tokens
                for e in evs[seen:]:
                    if e.get("event") == "token" and e.get("id") == req_id:
                        sys.stdout.write(e.get("text", ""))
                        sys.stdout.flush()
                seen = len(evs)
                if done_ev:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    sys.stderr.write(
                        f"[master] req {req_id}: {done_ev['ntoks']} toks "
                        f"in {done_ev['elapsed_s']:.1f}s = {done_ev['tps']:.2f} tok/s\n"
                    )
                    break
                if rank0.proc.poll() is not None:
                    sys.stderr.write("[master] rank 0 died during gen\n")
                    return
                # #29: block on the next event (up to 0.5s) instead of a 50ms
                # busy-poll. The 0.5s cap still re-checks proc liveness.
                rank0._event_signal.wait(timeout=0.5)
                rank0._event_signal.clear()

    def feed_prompts():
        # #29: `with open(...)` so the prompts file handle is released even if
        # the feed loop raises (the old code leaked it on any exception).
        if args.prompts_file:
            with open(args.prompts_file, "r") as f:
                _feed_from(f)
        else:
            sys.stderr.write("[master] reading prompts from stdin (Ctrl+D to stop)\n")
            _feed_from(sys.stdin)

    interrupted = {"flag": False}

    def handle_sigint(signum, frame):
        if interrupted["flag"]:
            sys.stderr.write("\n[master] second Ctrl+C, hard stop\n")
            for r in runners:
                r.proc.kill()
            sys.exit(130)
        interrupted["flag"] = True
        sys.stderr.write("\n[master] Ctrl+C, graceful stop (Ctrl+C again to force)\n")

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        feed_prompts()
    except Exception as e:
        sys.stderr.write(f"[master] feed loop error: {e}\n")
    finally:
        sys.stderr.write("[master] graceful stop on all runners\n")
        # #29: single stop path. graceful_stop() already sends {"cmd":"stop"} +
        # close_stdin() before waiting/escalating — the old code's manual
        # pre-send + close_stdin loops were a redundant double-stop. On a stop
        # cmd each rank exits in well under the soft timeout, so per-rank
        # graceful_stop returns fast; no parallel pre-send needed.
        for r in runners:
            r.graceful_stop()
        sys.stderr.write("[master] all runners stopped\n")


if __name__ == "__main__":
    main()
