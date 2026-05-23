#!/usr/bin/env python3
"""Auto-discover JACCL RDMA wiring across a cluster.

The TB5 cabling between Macs is physically opaque — there's no
way for an operator (or an AI agent) to tell that the cable plugged
into the third Thunderbolt port shows up as `rdma_en5` on this node
and `rdma_en3` on the other. macOS assigns enumeration order at boot.

This script SSHes into each node, lists the PORT_ACTIVE HCAs, reads
each interface's local MAC + the peer MAC visible via NDP, and
cross-references peer MACs across the cluster to build the rdma_to:
matrix for topology.yaml.

  Usage:
    scripts/discover-rdma-wiring.py \\
      0=admin@192.168.86.29:ultra-512 \\
      1=admin@192.168.86.30:ultra-256a \\
      2=admin@192.168.86.31:ultra-256b \\
      3=admin@192.168.86.32:ultra-256c

Each positional argument is `<rank>=<ssh-target>[:<node-id>]`. The
node-id is optional — defaults to the SSH target's hostname.

Output: per-node `rdma_to:` blocks ready to paste into topology.yaml.
Exits non-zero if the cluster isn't a full mesh (some peer not
visible from some other peer), with a clear "X cannot reach Y" message.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field


SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",
]


@dataclass
class NodeInfo:
    rank: int
    ssh: str
    node_id: str
    # hca_name -> en_iface (e.g. "rdma_en3" -> "en3")
    hcas: dict[str, str] = field(default_factory=dict)
    # en_iface -> local MAC
    local_macs: dict[str, str] = field(default_factory=dict)
    # en_iface -> list of peer MACs seen via NDP
    peer_macs: dict[str, list[str]] = field(default_factory=dict)


def ssh(target: str, cmd: str, timeout: int = 30) -> str:
    """Run a command on a remote host via SSH and return stdout."""
    result = subprocess.run(
        ["ssh", *SSH_OPTS, target, cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ssh {target}: exit {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout


def list_active_hcas(target: str) -> dict[str, str]:
    """Return {hca_name: en_iface} for PORT_ACTIVE HCAs on this node."""
    out = ssh(target, "ibv_devinfo 2>&1")
    hcas: dict[str, str] = {}
    current: str | None = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("hca_id:"):
            current = s.split(":", 1)[1].strip()
        elif current and s.startswith("state:") and "PORT_ACTIVE" in s:
            # "rdma_en3" → "en3" — the JACCL convention is rdma_<iface>.
            iface = current.removeprefix("rdma_")
            hcas[current] = iface
            current = None
    return hcas


def collect_iface_data(target: str, ifaces: list[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """For each TB interface: get local MAC + force NDP discovery + read peer MACs."""
    if not ifaces:
        return {}, {}
    # One shot: ping multicast on each iface (forces neighbor discovery),
    # then dump ifconfig + ndp -an in a single SSH round-trip.
    parts = []
    for i in ifaces:
        parts.append(f"ping6 -c 1 -W 500 ff02::1%{i} >/dev/null 2>&1")
        parts.append(f"echo '--IFCONFIG {i}--'; ifconfig {i} 2>/dev/null")
    parts.append("echo '--NDP--'; ndp -an 2>/dev/null")
    out = ssh(target, " ; ".join(parts))

    local_macs: dict[str, str] = {}
    current_iface: str | None = None
    ndp_section = False
    peer_macs: dict[str, list[str]] = {iface: [] for iface in ifaces}

    for line in out.splitlines():
        m = re.match(r"--IFCONFIG (\S+)--", line)
        if m:
            current_iface = m.group(1)
            ndp_section = False
            continue
        if line.strip() == "--NDP--":
            current_iface = None
            ndp_section = True
            continue
        if current_iface and not ndp_section:
            em = re.search(r"\bether\s+([0-9a-f:]{17})\b", line, re.IGNORECASE)
            if em:
                local_macs[current_iface] = em.group(1).lower()
        elif ndp_section:
            # NDP line format:
            #   fe80::xxxx%en3   36:ce:da:e0:b4:c4   en3   23h41m42s S
            parts2 = line.split()
            if len(parts2) >= 3:
                iface_col = parts2[2]
                if iface_col in peer_macs:
                    mac = parts2[1].lower()
                    if re.fullmatch(r"[0-9a-f:]{17}", mac):
                        # Skip our own MAC (link-local addresses for the
                        # local side appear in ndp as "permanent R").
                        local = local_macs.get(iface_col)
                        if mac != local:
                            peer_macs[iface_col].append(mac)
    return local_macs, peer_macs


def probe_node(node: NodeInfo) -> None:
    """SSH into node, populate hcas/local_macs/peer_macs."""
    node.hcas = list_active_hcas(node.ssh)
    if not node.hcas:
        return
    ifaces = list(node.hcas.values())
    node.local_macs, node.peer_macs = collect_iface_data(node.ssh, ifaces)


def build_wiring(nodes: list[NodeInfo]) -> tuple[dict[int, dict[int, str]], list[str]]:
    """Cross-reference peer MACs to build the rdma_to: matrix.

    Returns (wiring, warnings). `wiring[rank][peer_rank] = hca_name`.
    """
    # Build global MAC → rank index.
    mac_to_rank: dict[str, int] = {}
    for n in nodes:
        for mac in n.local_macs.values():
            mac_to_rank[mac] = n.rank

    wiring: dict[int, dict[int, str]] = {n.rank: {} for n in nodes}
    warnings: list[str] = []

    for n in nodes:
        for hca, iface in n.hcas.items():
            peers = n.peer_macs.get(iface, [])
            matched = False
            for pmac in peers:
                peer_rank = mac_to_rank.get(pmac)
                if peer_rank is not None and peer_rank != n.rank:
                    if peer_rank in wiring[n.rank]:
                        warnings.append(
                            f"[{n.node_id}] {hca} sees peer rank {peer_rank} "
                            f"but rank {peer_rank} is already reached via "
                            f"{wiring[n.rank][peer_rank]}"
                        )
                    wiring[n.rank][peer_rank] = hca
                    matched = True
                    break
            if not matched:
                if peers:
                    warnings.append(
                        f"[{n.node_id}] {hca} ({iface}) sees MAC(s) "
                        f"{peers} but none match any known cluster node"
                    )
                else:
                    warnings.append(
                        f"[{n.node_id}] {hca} ({iface}) — no peer visible "
                        f"in NDP (cable unplugged, or peer offline?)"
                    )
    return wiring, warnings


def emit_yaml(nodes: list[NodeInfo], wiring: dict[int, dict[int, str]]) -> str:
    """Print per-node rdma_to: blocks ready to paste."""
    lines = []
    rank_to_id = {n.rank: n.node_id for n in nodes}
    for n in nodes:
        lines.append(f"# rank {n.rank} — {n.node_id} ({n.ssh})")
        lines.append(f"rdma_to:")
        peers = wiring.get(n.rank, {})
        for peer_rank in sorted(peers):
            hca = peers[peer_rank]
            peer_id = rank_to_id.get(peer_rank, f"rank-{peer_rank}")
            lines.append(f"  {peer_rank}: {hca}    # → {peer_id}")
        lines.append("")
    return "\n".join(lines)


def parse_node_spec(spec: str) -> NodeInfo:
    """Parse `<rank>=<ssh>[:<id>]` into a NodeInfo."""
    if "=" not in spec:
        raise SystemExit(f"bad spec {spec!r} (expected <rank>=<ssh>[:<id>])")
    rank_str, rhs = spec.split("=", 1)
    rank = int(rank_str)
    if ":" in rhs:
        ssh_target, node_id = rhs.rsplit(":", 1)
    else:
        ssh_target = rhs
        # Default node id = the host part of the SSH target.
        node_id = ssh_target.split("@", 1)[-1]
    return NodeInfo(rank=rank, ssh=ssh_target, node_id=node_id)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for a worked example.",
    )
    p.add_argument("nodes", nargs="+", help="<rank>=<ssh-target>[:<node-id>]")
    args = p.parse_args(argv)

    nodes = [parse_node_spec(s) for s in args.nodes]
    nodes.sort(key=lambda n: n.rank)

    print(f"# Probing {len(nodes)} nodes…", file=sys.stderr)
    for n in nodes:
        print(f"#   rank {n.rank}: {n.ssh} ({n.node_id})", file=sys.stderr)
        try:
            probe_node(n)
        except Exception as e:
            print(f"# error probing {n.ssh}: {e}", file=sys.stderr)
            return 2
        print(
            f"#     active HCAs: {', '.join(sorted(n.hcas)) or '(none)'}",
            file=sys.stderr,
        )

    wiring, warnings = build_wiring(nodes)

    if warnings:
        print(file=sys.stderr)
        print("# Warnings:", file=sys.stderr)
        for w in warnings:
            print(f"#   {w}", file=sys.stderr)

    # Sanity check: full mesh has N*(N-1) directed edges.
    expected = len(nodes) * (len(nodes) - 1)
    actual = sum(len(peers) for peers in wiring.values())
    print(file=sys.stderr)
    print(
        f"# Edges discovered: {actual} / {expected} expected for a full mesh",
        file=sys.stderr,
    )

    print(emit_yaml(nodes, wiring))

    return 0 if actual == expected else 1


if __name__ == "__main__":
    sys.exit(main())
