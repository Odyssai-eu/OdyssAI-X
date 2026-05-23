"""Cluster topology loader.

Reads `~/.odysseus/topology.yaml` (or the path in `ODYSSEUS_TOPOLOGY`)
and exposes the cluster definitions in the shape api.py wants:

  - nodes_for(cluster: str, size: int) -> list[NodeDef]
  - rdma_wiring() -> dict[host_id, dict[peer_id, interface_name]]
  - known_hosts() -> list[KnownHost]
  - default_cluster_defs() -> dict[cluster_name, ClusterDef]

If the YAML doesn't exist, returns None from the public Loader.from_env()
factory so the caller falls back to the hardcoded constants in api.py.
This is the niveau-1 migration path: hardcoded → YAML loader available
→ YAML loader required (3 versions out).

Schema validation via Pydantic — operators get clear error messages
on malformed YAML instead of cryptic KeyErrors deep in api.py.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    import yaml  # PyYAML
except ImportError as e:
    raise ImportError(
        "topology.yaml support requires PyYAML — `pip install pyyaml`"
    ) from e


# ── Pydantic schema ───────────────────────────────────────────────────────


class NodeConfig(BaseModel):
    id: Optional[str] = Field(default=None, description="Stable host id shown in the dashboard.")
    label: Optional[str] = Field(default=None, description="Human-readable host label.")
    rank: int = Field(..., ge=0, description="Rank within the pool. 0 = master.")
    ssh: str = Field(..., description="SSH target, e.g. user@host.lan")
    models_dir: Optional[str] = Field(default=None, description="Model directory on this host.")
    port: Optional[int] = Field(default=None, ge=1, le=65535, description="HTTP service port for proxy nodes.")
    # JACCL only — per-peer interface map. `{peer_rank: interface_name}`.
    # Symmetric: if rank 0 says rdma_to[1] = 'rdma_en5', rank 1 must
    # say rdma_to[0] = 'rdma_en5' (same physical cable).
    rdma_to: dict[int, str] = Field(default_factory=dict)

    @field_validator("ssh")
    @classmethod
    def _ssh_shape(cls, v: str) -> str:
        # We expand ${ENV} late, so accept it here; otherwise basic shape check.
        if "${" in v:
            return v
        if "@" not in v:
            raise ValueError(f"ssh target must include user@host: got {v!r}")
        return v


class PoolConfig(BaseModel):
    size: int = Field(..., ge=1, le=16)
    nodes: list[NodeConfig] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_pool(self):
        # nodes count must match size
        if len(self.nodes) != self.size:
            raise ValueError(
                f"pool size={self.size} but {len(self.nodes)} nodes listed"
            )
        # ranks must be 0..size-1, no duplicates
        ranks = sorted(n.rank for n in self.nodes)
        if ranks != list(range(self.size)):
            raise ValueError(
                f"pool ranks must be 0..{self.size - 1}, got {ranks}"
            )
        # JACCL symmetry: if any node declares rdma_to, all should, and
        # every (i, j) pair must have a matching entry on both sides.
        rdma_pairs: dict[tuple[int, int], str] = {}
        for n in self.nodes:
            for peer_rank, iface in n.rdma_to.items():
                if peer_rank == n.rank:
                    raise ValueError(
                        f"rank {n.rank}: rdma_to includes self — that's "
                        f"impossible (a node can't RDMA to itself)"
                    )
                if peer_rank >= self.size:
                    raise ValueError(
                        f"rank {n.rank}: rdma_to[{peer_rank}] references "
                        f"a rank that doesn't exist in this pool (size={self.size})"
                    )
                rdma_pairs[(n.rank, peer_rank)] = iface
        # Symmetry — every (i, j) needs a (j, i) too, and they refer to
        # the same physical cable so the interface NAMES can differ but
        # the existence is required on both ends.
        for (i, j) in list(rdma_pairs.keys()):
            if (j, i) not in rdma_pairs:
                raise ValueError(
                    f"asymmetric RDMA wiring: rank {i} declares rdma_to[{j}] "
                    f"but rank {j} doesn't declare rdma_to[{i}]. Each cable "
                    f"must be listed on both ends."
                )
        return self


class ClusterConfig(BaseModel):
    label: Optional[str] = None
    backend: str = Field(default="ring", pattern=r"^(jaccl|ring|http-proxy)$")
    upstream: Optional[str] = None  # for http-proxy backend
    models_dir: Optional[str] = None
    pools: list[PoolConfig] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _backend_consistency(self):
        # jaccl pools of size >= 2 must declare rdma_to maps
        if self.backend == "jaccl":
            for p in self.pools:
                if p.size >= 2:
                    for n in p.nodes:
                        if not n.rdma_to:
                            raise ValueError(
                                f"backend=jaccl but pool size={p.size} node "
                                f"rank={n.rank} has no rdma_to map — JACCL "
                                f"needs interface names per peer"
                            )
        # http-proxy needs an upstream URL
        if self.backend == "http-proxy" and not self.upstream:
            raise ValueError("backend=http-proxy requires an `upstream` URL")
        return self


class TopologyConfig(BaseModel):
    """Top-level topology.yaml schema."""
    clusters: dict[str, ClusterConfig] = Field(..., min_length=1)


# ── Loader ────────────────────────────────────────────────────────────────


_ENV_RE = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _expand_env(s: str) -> str:
    """Expand ${VAR} and ${VAR:-default} in YAML strings."""
    def _sub(m):
        var, default = m.group(1), m.group(2)
        return os.environ.get(var, default if default is not None else "")
    return _ENV_RE.sub(_sub, s)


def _expand_env_recursive(obj):
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_recursive(v) for v in obj]
    return obj


def default_topology_path() -> Path:
    """Resolve ~/.odysseus/topology.yaml unless ODYSSEUS_TOPOLOGY env is set."""
    p = os.environ.get("ODYSSEUS_TOPOLOGY")
    if p:
        return Path(p).expanduser()
    return Path.home() / ".odysseus" / "topology.yaml"


def load_topology(path: Optional[Path] = None) -> Optional[TopologyConfig]:
    """Read + parse + validate the topology YAML.

    Returns None if the file doesn't exist (caller should fall back to
    hardcoded defaults during the migration window). Raises pydantic
    ValidationError on malformed YAML — operators see what's wrong.
    """
    p = path or default_topology_path()
    if not p.exists():
        return None
    raw = yaml.safe_load(p.read_text())
    expanded = _expand_env_recursive(raw)
    return TopologyConfig.model_validate(expanded)


# ── Adapters — return the shapes api.py currently consumes ───────────────


def host_id_from_ssh(ssh: str) -> str:
    """Derive a stable host id from `user@host`.

    Operators can override this with `node.id` in topology.yaml. This fallback
    strips common LAN suffixes so `worker-a.lan` and `worker-a.local` show as
    `worker-a` in the dashboard.
    """
    h = ssh.split("@", 1)[-1]
    for sfx in (".lan", ".local"):
        if h.endswith(sfx):
            h = h[: -len(sfx)]
    return h


def to_nodes_dict(cluster: ClusterConfig) -> dict[int, list[dict]]:
    """Build the `{pool_size: [{rank, ssh, rdma}, ...]}` dict that api.py's
    legacy `{pool_size: [rank nodes...]}` shape used by api.py.

    The `rdma` list inside each node entry is the row of the per-pool RDMA
    matrix in rank order, with None on the diagonal — matching what
    runner.py / master.py / mlx_distributed_config expects.
    """
    out: dict[int, list[dict]] = {}
    for pool in cluster.pools:
        nodes = sorted(pool.nodes, key=lambda n: n.rank)
        rows = []
        for n in nodes:
            rdma_row: list[Optional[str]] = []
            for peer_rank in range(pool.size):
                if peer_rank == n.rank:
                    rdma_row.append(None)
                else:
                    rdma_row.append(n.rdma_to.get(peer_rank))
            rows.append({
                "rank": n.rank,
                "ssh": n.ssh,
                "rdma": rdma_row,
                "host": n.id or host_id_from_ssh(n.ssh),
            })
        out[pool.size] = rows
    return out


def to_rdma_wiring(topo: TopologyConfig) -> dict[str, dict[str, str]]:
    """Flatten across ALL clusters into the global RDMA wiring dict
    keyed by SSH-derived host id. Used by the dashboard's "Add node"
    flow + the validation that an arbitrary rdma collective is wireable.

    The host id derived here is the part after the last `@` in the SSH
    target, stripped of `.lan` / `.local` suffixes. Matches what the inventory
    in api.py uses.
    """
    wiring: dict[str, dict[str, str]] = {}
    for _, cluster in topo.clusters.items():
        for pool in cluster.pools:
            if pool.size < 2:
                continue
            by_rank = {n.rank: n for n in pool.nodes}
            for n in pool.nodes:
                src = n.id or host_id_from_ssh(n.ssh)
                wiring.setdefault(src, {})
                for peer_rank, iface in n.rdma_to.items():
                    peer = by_rank.get(peer_rank)
                    if peer is None:
                        continue
                    dst = peer.id or host_id_from_ssh(peer.ssh)
                    wiring[src][dst] = iface
    return wiring


def to_known_hosts(topo: TopologyConfig) -> list[dict]:
    """Distinct (host_id, ssh, rdma_wired) tuples across all clusters."""
    seen: dict[str, dict] = {}
    for _, cluster in topo.clusters.items():
        for pool in cluster.pools:
            rdma_wired = any(n.rdma_to for n in pool.nodes)
            for n in pool.nodes:
                hid = n.id or host_id_from_ssh(n.ssh)
                if hid not in seen:
                    seen[hid] = {
                        "id": hid,
                        "ssh": n.ssh,
                        "rdma_wired": rdma_wired,
                        "label": n.label or hid,
                    }
                elif rdma_wired and not seen[hid]["rdma_wired"]:
                    seen[hid]["rdma_wired"] = True
    return list(seen.values())


def to_default_cluster_defs(topo: TopologyConfig) -> dict[str, dict]:
    """Build api.py's DEFAULT_CLUSTER_DEFS from topology.yaml.

    For each cluster, the largest declared pool becomes the editable host
    inventory. Smaller pools remain available through `to_nodes_dict()` for
    explicit pool-size topologies.
    """
    out: dict[str, dict] = {}
    for cid, cluster in topo.clusters.items():
        largest = max(cluster.pools, key=lambda p: p.size)
        nodes = []
        for n in sorted(largest.nodes, key=lambda item: item.rank):
            node = {
                "host": n.id or host_id_from_ssh(n.ssh),
                "ssh": n.ssh,
                "master": n.rank == 0,
            }
            if n.port is not None:
                node["port"] = n.port
            if n.models_dir:
                node["models_dir"] = n.models_dir
            nodes.append(node)

        kind = "http-proxy" if cluster.backend == "http-proxy" else "mlx-distributed"
        backend = "http" if cluster.backend == "http-proxy" else cluster.backend
        cluster_def = {
            "name": cluster.label or cid,
            "kind": kind,
            "backend": backend,
            "nodes": nodes,
        }
        if cluster.models_dir:
            cluster_def["models_dir"] = cluster.models_dir
        if cluster.upstream:
            cluster_def["upstream"] = cluster.upstream
        out[cid] = cluster_def
    return out
