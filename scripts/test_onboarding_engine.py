#!/usr/bin/env python3
"""Engine side of the onboarding (#101/#105): public key route and host-key pinning.

The onboarding app hands the engine's public key to a node and brings back the node's
ssh host key; `POST /admin/nodes/discovered/{host}/add` with `host_key` pins it so the
engine's first ssh succeeds without anyone accepting the key by hand. No node is reached:
the ssh check is stubbed, and the real OpenSSH only parses the drop-in (`ssh -G`).

    python3 scripts/test_onboarding_engine.py
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

T = Path(tempfile.mkdtemp())
os.environ["CLUSTER_CONFIG_FILE"] = str(T / "cc.json")
os.environ["ODYSSAI_X_STATE_DIR"] = str(T)
os.environ["ODYSSAI_X_SSH_CONFIG_D"] = str(T / "ssh_config.d")
os.environ["HOME"] = str(T / "home")
(T / "ssh_config.d").mkdir()
(T / "home" / ".ssh").mkdir(parents=True)
(T / "home" / ".ssh" / "id_ed25519.pub").write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEngineKeyForTestsOnly0000000000000000 odyssai-orchestrator\n")
json.dump({"solo": {"name": "solo", "kind": "mlx-distributed", "backend": "ring", "max_nodes": 2,
                    "nodes": [{"host": "n0", "ssh": "admin@198.51.100.10", "master": True}]}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402
from fastapi import HTTPException  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


async def no_ssh_capture(target, cmd, timeout=10):
    calls.append(target)
    return 0, ""

calls = []
api._ssh_capture = no_ssh_capture
api.save_cluster_state_v2 = lambda cid: None
NODE_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINodeHostKeyForTests1111111111111111111"

# 1. public key route
r = asyncio.run(api.admin_onboarding_pubkey())
check("pubkey route returns the engine's ed25519 public key", r["public_key"].split()[0:2],
      ["ssh-ed25519", "AAAAC3NzaC1lZDI1NTE5AAAAIEngineKeyForTestsOnly0000000000000000"])
conf = T / "ssh_config.d" / "odyssai-onboard.conf"
check("... and lays the ssh drop-in that reads the pinned keys", conf.is_file(), True)

# 2. the drop-in is valid OpenSSH config: ssh -G resolves the extra global file
out = subprocess.run(["ssh", "-G", "-F", str(conf), "198.51.100.62"], capture_output=True, text=True)
gk = next((ln for ln in out.stdout.splitlines() if ln.startswith("globalknownhostsfile")), "")
check("OpenSSH parses it: the pinned file is a global known_hosts file", str(api.ONBOARD_KNOWN_HOSTS) in gk, True)

# 3. add a discovered node with its host key
api._discovered["ultra-96d"] = {"host": "ultra-96d", "user": "odyssai", "ip": "198.51.100.62",
                               "ssh": "odyssai@198.51.100.62", "seen_at": __import__("time").time(),
                               "chip": "M3 Ultra", "ram_gb": 96}
res = asyncio.run(api.admin_nodes_discovered_add("ultra-96d", api.DiscoveredAdd(
    cluster="solo", host_key=NODE_KEY + " root@ultra-96d")))
pinned = api.ONBOARD_KNOWN_HOSTS.read_text().splitlines()
check("the node's host key is pinned for its address (comment dropped)", pinned, [f"198.51.100.62 {NODE_KEY}"])
check("the engine then checked ssh to the node", calls, ["odyssai@198.51.100.62"])
check("the node joined the cluster", [n["host"] for n in res["nodes"]], ["n0", "ultra-96d"])

# 4. re-pinning replaces, other hosts are kept
api._pin_host_key("198.51.100.63", NODE_KEY.replace("1111", "2222"))
api._pin_host_key("198.51.100.62", NODE_KEY.replace("1111", "3333"))
lines = api.ONBOARD_KNOWN_HOSTS.read_text().splitlines()
check("re-pinning a host replaces its line and keeps the others",
      sorted(ln.split()[0] for ln in lines), ["198.51.100.62", "198.51.100.63"])
check("... with the new key", [ln for ln in lines if ln.startswith("198.51.100.62")],
      [f"198.51.100.62 {NODE_KEY.replace('1111', '3333')}"])

# 5. garbage is refused
for bad in ("not a key", "ssh-ed25519 short", "ssh-ed25519 AAAA\nevil", "ssh-dss AAAAB3NzaC1kc3MAAACBAP" + "A" * 40):
    try:
        api._pin_host_key("198.51.100.64", bad)
        got = "accepted"
    except HTTPException as e:
        got = e.status_code
    check(f"a malformed host key is refused: {bad[:18]!r}", got, 422)
check("nothing was pinned for the refused keys", "198.51.100.64" in api.ONBOARD_KNOWN_HOSTS.read_text(), False)

# 6. an engine that cannot write the drop-in says so instead of half-pinning
api.SSH_CONFIG_D = T / "missing"
try:
    asyncio.run(api.admin_nodes_discovered_add("ultra-96d", api.DiscoveredAdd(cluster="solo", host_key=NODE_KEY)))
    got = "added"
except HTTPException as e:
    got = (e.status_code, e.detail.get("error") if isinstance(e.detail, dict) else e.detail)
check("without a writable ssh_config.d, host_key gets a clear 409", got, (409, "host_key_pinning_unavailable"))

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
