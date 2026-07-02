"""OdyssAI-X (odyssai.eu) — orchestration API + dashboard for distributed MLX inference.

Manages multiple operator-defined clusters (declared in topology.yaml),
sharing a common model layout under the configured models_dir on every node.

Serves:
  GET  /                    -> dashboard HTML
  GET  /health              -> minimal liveness
  GET  /v1/models           -> OpenAI-style listing (across all clusters)
  POST /v1/chat/completions -> OpenAI chat, routed by model id

Admin (no auth by default — see ODYSSAI_X_ADMIN_TOKEN):
  GET  /admin/clusters
  GET  /admin/clusters/{id}                 — full cluster def + capacity
  PUT  /admin/clusters/{id}                 — partial update
  GET  /admin/clusters/{id}/status
  GET  /admin/clusters/{id}/models?dir=
  POST /admin/clusters/{id}/models-dir
  POST /admin/clusters/{id}/load
  POST /admin/clusters/{id}/unload
  POST /admin/clusters/{id}/reset
  POST /admin/clusters/{id}/reboot-all
  Shared:
    GET  /admin/metrics, /admin/sessions, /admin/logs, /admin/runs

State persistence:
  scripts/state-<cluster_id>.json     (one per cluster)
  scripts/cluster-config.json         (per-cluster models_dir overrides)

Run:
  .venv/bin/python scripts/api.py                              # reload last state
  .venv/bin/python scripts/api.py --model X --mode pipeline    # force fresh
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import copy
import json
import os
import random
import re
import secrets
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Union
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request

# Local sibling module — write-through SQLite for runs + sync_jobs history.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import persistence as _persist  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

# ──────────────────────────────────────────────────────────────────────────────
# Engine env contract — ODYSSAI_X_* (the public name), with the legacy
# ODYSSEUS_* names dual-read (deprecated 2026-06-12). Deployed compose files
# and operator shells keep working; a one-time stderr note nudges the rename.
# ──────────────────────────────────────────────────────────────────────────────
_ENV_DEPRECATED_SEEN: set = set()


def env_get(suffix: str, default=None):
    """Engine env lookup: ODYSSAI_X_<suffix>, then legacy ODYSSEUS_<suffix>."""
    new = f"ODYSSAI_X_{suffix}"
    if new in os.environ:
        return os.environ[new]
    old = f"ODYSSEUS_{suffix}"
    if old in os.environ:
        if old not in _ENV_DEPRECATED_SEEN:
            _ENV_DEPRECATED_SEEN.add(old)
            sys.stderr.write(f"[api] DEPRECATED env {old} — rename to {new}\n")
        return os.environ[old]
    return default


def env_value_by_name(name: str):
    """Resolve an env var BY NAME with prefix bridging. Provider configs
    persist `api_key_env` names — a stored ODYSSEUS_* name must keep
    resolving after the operator renames the variable, and vice versa."""
    v = os.environ.get(name)
    if v is not None:
        return v
    if name.startswith("ODYSSEUS_"):
        return os.environ.get("ODYSSAI_X_" + name[len("ODYSSEUS_"):])
    if name.startswith("ODYSSAI_X_"):
        return os.environ.get("ODYSSEUS_" + name[len("ODYSSAI_X_"):])
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Cluster topology
# ──────────────────────────────────────────────────────────────────────────────
try:
    from topology import (
        host_id_from_ssh as _host_id_from_ssh,
        load_topology as _load_topology,
        to_default_cluster_defs as _to_default_cluster_defs,
        to_nodes_dict as _to_nodes_dict,
        to_rdma_wiring as _to_rdma_wiring,
        to_known_hosts as _to_known_hosts,
    )
    _TOPOLOGY = _load_topology()
except Exception as _e:
    # Missing PyYAML or malformed YAML is a configuration error when the
    # operator explicitly provides ODYSSAI_X_TOPOLOGY. Without a topology file,
    # boot with a neutral localhost single-node default instead of leaking any
    # author's LAN into the open-source runtime.
    if env_get("TOPOLOGY"):
        raise
    print(f"[topology] WARNING: could not load topology: {_e!r}; "
          f"using neutral localhost fallback", flush=True)
    _TOPOLOGY = None
    def _host_id_from_ssh(ssh: str) -> str:
        host = ssh.split("@", 1)[-1]
        for suffix in (".lan", ".local"):
            if host.endswith(suffix):
                host = host[: -len(suffix)]
        return host

if _TOPOLOGY is None and env_get("TOPOLOGY"):
    raise RuntimeError(
        f"ODYSSAI_X_TOPOLOGY is set but no topology was loaded: "
        f"{env_get('TOPOLOGY')}"
    )


_SAFE_SSH_RE = re.compile(
    r"^[A-Za-z0-9_][A-Za-z0-9._-]*@[A-Za-z0-9._-]+(:[0-9]{1,5})?$"
)


def _safe_ssh_target(target: str) -> str:
    """Defense-in-depth: reject targets that could be parsed as SSH options.

    The primary gate is NodeConfig._ssh_shape in topology.py (validated at
    config-save time). This guard prevents a target that snuck in via a
    hand-edited config file from reaching subprocess.
    """
    if "${" in target:
        return target  # unexpanded env ref, skip until expansion
    if not _SAFE_SSH_RE.match(target):
        raise ValueError(
            f"refusing ssh target {target!r}: not in user@host form, "
            "possible option injection"
        )
    return target


def _local_ssh_target() -> str:
    return env_get("LOCAL_SSH", f"{os.environ.get('USER', 'user')}@localhost")


NEUTRAL_DEFAULT_CLUSTER_ID = "default"


def _neutral_cluster_defs() -> dict[str, dict]:
    ssh = _local_ssh_target()
    host = env_get("LOCAL_HOST_ID", _host_id_from_ssh(ssh))
    return {
        NEUTRAL_DEFAULT_CLUSTER_ID: {
            "name": "Local",
            "kind": "mlx-distributed",
            "backend": "ring",
            "nodes": [{"host": host, "ssh": ssh, "master": True}],
        }
    }


def _neutral_known_hosts() -> list[dict]:
    node = _neutral_cluster_defs()[NEUTRAL_DEFAULT_CLUSTER_ID]["nodes"][0]
    return [{
        "id": node["host"],
        "ssh": node["ssh"],
        "rdma_wired": False,
        "label": node["host"],
    }]


if _TOPOLOGY is not None:
    TOPOLOGY_NODES_BY_CLUSTER = {
        cid: _to_nodes_dict(cluster) for cid, cluster in _TOPOLOGY.clusters.items()
    }
    RDMA_WIRING: dict[str, dict[str, str]] = _to_rdma_wiring(_TOPOLOGY)
    KNOWN_HOSTS = _to_known_hosts(_TOPOLOGY)
    DEFAULT_CLUSTER_DEFS: dict[str, dict] = _to_default_cluster_defs(_TOPOLOGY)
    print(f"[topology] loaded operator topology — clusters: "
          f"{list(_TOPOLOGY.clusters.keys())}", flush=True)
else:
    TOPOLOGY_NODES_BY_CLUSTER: dict[str, dict[int, list[dict]]] = {
        NEUTRAL_DEFAULT_CLUSTER_ID: {
            1: [{"rank": 0, "ssh": _local_ssh_target(), "rdma": [None],
                 "host": _neutral_known_hosts()[0]["id"]}]
        }
    }
    RDMA_WIRING = {}
    KNOWN_HOSTS = _neutral_known_hosts()
    DEFAULT_CLUSTER_DEFS = _neutral_cluster_defs()

# Default canonical model directory on every node, every cluster.
# Flat layout: org--repo dirs or symlinks.
DEFAULT_MODELS_DIR = env_get("DEFAULT_MODELS_DIR", "/Volumes/models/odysseus")

COORDINATOR_RANK = 0
REMOTE_CLUSTER_DIR = env_get("REMOTE_CLUSTER_DIR", "$HOME/mlx-cluster").rstrip("/")
RUNNER_REMOTE = env_get("RUNNER_REMOTE", f"{REMOTE_CLUSTER_DIR}/runner.py")
PYTHON_REMOTE = env_get("PYTHON_REMOTE", f"{REMOTE_CLUSTER_DIR}/.venv/bin/python")
RUNNER_MATCH_PATTERN = env_get("RUNNER_MATCH_PATTERN", "mlx-cluster/runner.py")

_HERE = Path(__file__).resolve().parent
CLUSTER_CONFIG_FILE = Path(os.environ.get("CLUSTER_CONFIG_FILE", _HERE / "cluster-config.json"))


def state_file_for(cluster_id: str) -> Path:
    """Per-cluster persisted state. Lives next to cluster-config.json so
    operators get a single backup/restore unit. Env override:
    ODYSSAI_X_STATE_DIR sets the directory; individual files are
    `state-<cluster_id>.json`."""
    state_dir = Path(env_get("STATE_DIR", _HERE))
    return state_dir / f"state-{cluster_id}.json"


# Legacy single-grid state file (kept only for the unused Nautilus path).
# Per-cluster state files come from state_file_for(cluster_id).
STATE_FILE = Path(os.environ.get("STATE_FILE", _HERE / "state.json"))
# SQLite history for runs + sync jobs. Defaults to /app/data/ (the Docker
# volume that holds state.json + cluster-config.json), so history survives
# container restarts. Env override: ODYSSAI_X_DB_PATH.
_DEFAULT_DB = Path("/app/data/odysseus.db") if Path("/app/data").is_dir() \
              else _HERE / "odysseus.db"
PERSIST_DB_PATH = Path(env_get("DB_PATH", _DEFAULT_DB))

# Server-wide policy: when a chat request doesn't explicitly set
# `enable_thinking`, what should we do? Default: false (suppress reasoning
# scratchpads in Qwen3/GLM4/Hy3/etc. for snappier responses). Clients that
# WANT thinking (Companion, advanced UIs) opt-in by sending
# `enable_thinking: true` in the OpenAI /v1/chat/completions body.
#
# Override priority: cluster-config.json `settings.enable_thinking_default`
# (set via admin UI) > THINKING_DEFAULT env var > hard default `false`.
# Use `get_enable_thinking_default()` rather than reading the constant
# directly so admin-UI changes take effect without restart.
THINKING_DEFAULT = os.environ.get("THINKING_DEFAULT", "false").lower() == "true"


def get_enable_thinking_default() -> bool:
    """Effective server-wide default for `enable_thinking` when a client
    doesn't pass the field. Looks at the persisted settings overlay first,
    then falls back to the env-driven `THINKING_DEFAULT` constant. Applies
    to BOTH the local pool path (distributed runners) and the cloud-provider
    proxy path (cloud upstreams) — single source of truth."""
    cfg = _load_cluster_config()
    settings = (cfg.get("settings") or {})
    val = settings.get("enable_thinking_default")
    if isinstance(val, bool):
        return val
    return THINKING_DEFAULT


def set_enable_thinking_default(value: bool) -> None:
    with _cluster_config_txn() as cfg:
        settings = cfg.get("settings") or {}
        settings["enable_thinking_default"] = bool(value)
        cfg["settings"] = settings
DASHBOARD_FILE = Path(os.environ.get("DASHBOARD_FILE", _HERE / "dashboard.html"))
ODYRAG_FILE    = Path(os.environ.get("ODYRAG_FILE",    _HERE / "odyrag.html"))
ICON_FILE = Path(os.environ.get("ICON_FILE", _HERE / "odysseus.png"))
# User Guide Markdown source. Tries env override first, then a list of
# plausible defaults so dev (repo `docs/user-guide/`) and container
# (`/app/docs/user-guide/`) both resolve without manual config.
def _resolve_user_guide_dir() -> Path:
    env = os.environ.get("USER_GUIDE_DIR")
    if env:
        return Path(env)
    candidates = [
        _HERE / "docs" / "user-guide",        # container: /app/docs/user-guide
        _HERE.parent / "docs" / "user-guide", # dev: <repo>/docs/user-guide
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]  # fallback, will report "not found" cleanly

USER_GUIDE_DIR = _resolve_user_guide_dir()


# ──────────────────────────────────────────────────────────────────────────────
# Per-cluster config (models_dir + per-model load history). Persisted to file.
# ──────────────────────────────────────────────────────────────────────────────
# cluster-config.json is the single flat-file store for settings, providers,
# crew tokens, discovery state, models_dir + load history. It is read on the
# hot inference path and mutated from ~12 admin sites + background sweepers,
# so access is guarded by a re-entrant lock with a short read cache and an
# atomic (tmp + os.replace) write. Mutators MUST go through
# `_cluster_config_txn()` so the read-modify-write holds the lock end-to-end
# and concurrent writers can't clobber each other (#22).
_config_lock = threading.RLock()
_config_cache: Optional[dict] = None
_config_cache_ts: float = 0.0
_CONFIG_TTL_S = 2.0


def _read_cluster_config_from_disk() -> dict:
    try:
        return json.loads(CLUSTER_CONFIG_FILE.read_text())
    except Exception:
        return {}


def _load_cluster_config() -> dict:
    """Parsed cluster-config, cached up to _CONFIG_TTL_S. Returns a deep copy
    so callers may mutate freely; mutations only persist via a txn / save."""
    global _config_cache, _config_cache_ts
    with _config_lock:
        now = time.monotonic()
        if _config_cache is not None and (now - _config_cache_ts) < _CONFIG_TTL_S:
            return copy.deepcopy(_config_cache)
        cfg = _read_cluster_config_from_disk()
        _config_cache = cfg
        _config_cache_ts = now
        return copy.deepcopy(cfg)


def _save_cluster_config(cfg: dict) -> None:
    """Persist atomically (tmp + os.replace) under the config lock and refresh
    the cache. Prefer `_cluster_config_txn()` for read-modify-write so the
    read and write share one lock hold (no lost updates)."""
    global _config_cache, _config_cache_ts
    with _config_lock:
        try:
            tmp = CLUSTER_CONFIG_FILE.with_name(CLUSTER_CONFIG_FILE.name + ".tmp")
            tmp.write_text(json.dumps(cfg, indent=2))
            os.replace(tmp, CLUSTER_CONFIG_FILE)
            _config_cache = copy.deepcopy(cfg)
            _config_cache_ts = time.monotonic()
        except Exception as e:
            sys.stderr.write(f"[api] failed to save cluster config: {e}\n")


@contextmanager
def _cluster_config_txn():
    """Atomic read-modify-write of cluster-config.json. Holds `_config_lock`
    across the whole cycle, reads a FRESH copy from disk (bypassing the TTL
    cache so we never write back a stale base), yields it for mutation, and
    persists atomically on clean exit. Must NOT `await` inside the block."""
    with _config_lock:
        cfg = _read_cluster_config_from_disk()
        yield cfg
        _save_cluster_config(cfg)


def models_dir_for(cluster: str) -> str:
    cfg = _load_cluster_config()
    return cfg.get(cluster, {}).get("models_dir", DEFAULT_MODELS_DIR)


def set_models_dir(cluster: str, path: str) -> None:
    with _cluster_config_txn() as cfg:
        cfg.setdefault(cluster, {})["models_dir"] = path


def get_load_history(cluster: str, model: str) -> Optional[dict]:
    cfg = _load_cluster_config()
    return cfg.get(cluster, {}).get("load_history", {}).get(model)


def record_load_history(cluster: str, model: str, load_s: float, size_bytes: int,
                        nodes: int) -> None:
    with _cluster_config_txn() as cfg:
        bucket = cfg.setdefault(cluster, {}).setdefault("load_history", {})
        bucket[model] = {
            "last_load_s": round(load_s, 2),
            "size_bytes": int(size_bytes),
            "nodes": nodes,
        }


def get_cluster_max_nodes(cluster: str, default: int) -> int:
    """User-configurable upper bound on the cluster's node count.
    Lets you dedicate a host to another cluster. Falls back to the topology size."""
    cfg = _load_cluster_config()
    try:
        n = int(cfg.get(cluster, {}).get("max_nodes", default))
        return max(1, min(default, n))
    except Exception:
        return default


def set_cluster_max_nodes(cluster: str, max_nodes: int) -> None:
    with _cluster_config_txn() as cfg:
        cfg.setdefault(cluster, {})["max_nodes"] = int(max_nodes)


# ──────────────────────────────────────────────────────────────────────────────
# Editable cluster definitions
# ──────────────────────────────────────────────────────────────────────────────
# RDMA_WIRING, KNOWN_HOSTS and DEFAULT_CLUSTER_DEFS are now sourced from
# topology.yaml at boot. Without topology.yaml the runtime gets a neutral
# localhost single-node fallback. No operator LAN belongs in source code.


def get_cluster_def(cluster_id: str) -> dict:
    """Effective cluster definition = defaults overlaid by cluster-config.json.
    Always returns a dict with name/kind/backend/nodes (and models_dir if set)."""
    cfg = _load_cluster_config()
    default = DEFAULT_CLUSTER_DEFS.get(cluster_id, {})
    overlay = cfg.get(cluster_id, {})
    merged = json.loads(json.dumps(default))  # deep copy
    for k in ("name", "kind", "backend", "max_nodes", "models_dir", "enabled", "upstream", "supports_vision", "_vlm_managed", "_vlm_port", "_vlm_model_path", "_vlm_pid"):
        if k in overlay:
            merged[k] = overlay[k]
    if "nodes" in overlay and overlay["nodes"]:
        merged["nodes"] = overlay["nodes"]
    return merged


def _is_cluster_tombstoned(cluster_id: str) -> bool:
    """True if cluster-config.json marked this cluster as removed via UI.

    cluster-config.json mixes cluster definitions with non-cluster
    sections at top-level (e.g. 'crew' is a list, 'discovery' is a
    bookkeeping dict). Anything whose value isn't a dict can't be a
    cluster, so it's never tombstoned. Without this guard, iterating
    `active_cluster_ids()` over a config that contains a 'crew' list
    crashes the cluster-list endpoint with
    `AttributeError: 'list' object has no attribute 'get'` and the
    dashboard renders 'No cluster configured'.
    """
    entry = _load_cluster_config().get(cluster_id)
    if not isinstance(entry, dict):
        return False
    return bool(entry.get("_removed"))


def cluster_exists(cluster_id: str) -> bool:
    """Cluster is active (visible in dashboard) when it has a definition AND
    isn't tombstoned. Definitions come from either topology.yaml seed OR a
    dashboard-added entry in cluster-config.json."""
    if _is_cluster_tombstoned(cluster_id):
        return False
    if cluster_id in DEFAULT_CLUSTER_DEFS:
        return True
    overlay = _load_cluster_config().get(cluster_id, {})
    # A meaningful overlay entry (has nodes, or has kind=telemak with upstream)
    if overlay.get("nodes"):
        return True
    if overlay.get("kind") == "telemak" and overlay.get("upstream"):
        return True
    return False


def active_cluster_ids() -> list[str]:
    """Union of topology.yaml clusters + dashboard-added entries, minus
    tombstones. Source of truth for "which clusters does OdyssAI-X publish".

    cluster-config.json carries cluster definitions alongside unrelated
    top-level sections ('crew', 'discovery', 'settings', ...) that
    were never meant to be cluster ids. Filter them out by requiring
    the value to be a dict with at least one cluster-shape field.
    """
    seen: dict[str, bool] = {}
    for cid in DEFAULT_CLUSTER_DEFS.keys():
        seen[cid] = True
    for cid, entry in _load_cluster_config().items():
        if cid in seen:
            continue
        if not isinstance(entry, dict):
            continue
        # Heuristic : a real cluster entry has at least one of these.
        # Bare 'load_history'-only entries pre-date the explicit kind
        # field — keep them too, since the legacy single-cluster file
        # used 'load_history' as the only top-level marker.
        if any(k in entry for k in (
            "nodes", "kind", "label", "models_dir", "max_nodes",
            "_removed", "load_history", "backend", "upstream",
        )):
            seen[cid] = True
    return [cid for cid in seen if not _is_cluster_tombstoned(cid)]


def _cluster_enabled(cluster_id: str) -> bool:
    """True unless cluster-config.json explicitly stored enabled=false for this
    cluster. Default True so existing config (no `enabled` field) keeps working."""
    return get_cluster_def(cluster_id).get("enabled", True) is not False


def save_cluster_def(cluster_id: str, updates: dict) -> None:
    """Persist a partial update to the cluster definition."""
    with _cluster_config_txn() as cfg:
        cur = cfg.setdefault(cluster_id, {})
        for k, v in updates.items():
            cur[k] = v


def build_rdma_matrix(host_ids: list[str]) -> list[list[Optional[str]]]:
    """Build the NxN RDMA port matrix from the physical wiring map."""
    matrix = []
    for i, src in enumerate(host_ids):
        row = []
        for j, dst in enumerate(host_ids):
            row.append(None if i == j else RDMA_WIRING.get(src, {}).get(dst))
        matrix.append(row)
    return matrix


def build_topology(cluster_id: str, count: Optional[int] = None) -> list[dict]:
    """Return the runtime topology (rank-ordered) for a cluster.

    - Master node is moved to rank 0
    - RDMA matrix is computed from the physical wiring
    - `count` slices to the first N nodes (master always included)
    """
    cd = get_cluster_def(cluster_id)
    nodes = list(cd.get("nodes", []))
    # Move master to position 0 (stable for the rest)
    master_idx = next((i for i, n in enumerate(nodes) if n.get("master")), 0)
    if master_idx != 0:
        nodes.insert(0, nodes.pop(master_idx))
    if count is not None:
        nodes = nodes[:max(1, count)]
    host_ids = [n["host"] for n in nodes]
    matrix = build_rdma_matrix(host_ids)
    return [
        {
            "rank": i,
            "ssh": n["ssh"],
            "rdma": matrix[i],
            "host": n["host"],
            # Propagate models_dir to the runtime topology so consumers
            # (preflight validator, runner spawn) can resolve relative
            # model ids without a separate cluster_def lookup. Falls
            # back to the cluster-level default models_dir when the
            # node entry didn't override it.
            "models_dir": n.get("models_dir") or cd.get("models_dir") or DEFAULT_MODELS_DIR,
        }
        for i, n in enumerate(nodes)
    ]


def require_topology(cluster_id: str, count: Optional[int] = None) -> list[dict]:
    topo = build_topology(cluster_id, count)
    if not topo:
        raise HTTPException(404, f"{cluster_id} topology is not configured")
    return topo


def rank0_ssh_for_cluster(cluster_id: str, count: Optional[int] = None) -> str:
    return require_topology(cluster_id, count)[0]["ssh"]


def build_topology_from_indices(cluster_id: str,
                                 node_indices: list[int]) -> list[dict]:
    """Build a topology from explicit host indices into the cluster def's
    `nodes` list. Used by multi-pool Default: each pool occupies a disjoint
    subset of nodes, and the first index in the list becomes that pool's
    rank-0 master (regardless of which host was the cluster's "master"
    field — within a pool, the operator chooses).

    Returns the same shape as `build_topology()`: rank-ordered dicts
    with {rank, ssh, rdma, host}. RDMA matrix is recomputed for the
    subset since wiring is per-pair.
    """
    cd = get_cluster_def(cluster_id)
    all_nodes = list(cd.get("nodes") or [])
    if not node_indices:
        raise ValueError("node_indices must contain at least one index")
    seen: set[int] = set()
    selected: list[dict] = []
    for i in node_indices:
        if not (0 <= i < len(all_nodes)):
            raise ValueError(
                f"node index {i} out of range [0..{len(all_nodes) - 1}]"
            )
        if i in seen:
            raise ValueError(f"duplicate node index {i}")
        seen.add(i)
        selected.append(all_nodes[i])
    host_ids = [n["host"] for n in selected]
    matrix = build_rdma_matrix(host_ids)
    return [
        {
            "rank": i,
            "ssh": n["ssh"],
            "rdma": matrix[i],
            "host": n["host"],
            # Same models_dir propagation as build_topology (2026-05-25 fix).
            "models_dir": n.get("models_dir") or cd.get("models_dir") or DEFAULT_MODELS_DIR,
        }
        for i, n in enumerate(selected)
    ]


def validate_cluster_def(cluster_id: str, new_def: dict) -> Optional[str]:
    """Return error string if invalid, else None."""
    kind = new_def.get("kind", "mlx-distributed")
    backend = new_def.get("backend", "jaccl")
    nodes = new_def.get("nodes") or []
    if not isinstance(nodes, list) or len(nodes) == 0:
        return "nodes must be a non-empty list"
    masters = [n for n in nodes if n.get("master")]
    if len(masters) != 1:
        return f"exactly 1 master required (got {len(masters)})"

    # Telemak (single-Mac native runtime) — http-proxy passthrough to a Swift
    # binary running on a Mac on the LAN. The OdyssAI-X orchestrator does not
    # spawn the runner — it just proxies HTTP requests. Single node by design.
    if kind == "telemak":
        if backend != "http-proxy":
            return f"telemak kind requires backend=http-proxy (got {backend})"
        if len(nodes) != 1:
            return f"telemak kind requires exactly 1 node (got {len(nodes)})"
        if not (new_def.get("upstream") or "").strip():
            return "telemak kind requires non-empty 'upstream' (e.g. http://host.lan:8003)"
        return None

    if kind != "mlx-distributed":
        return f"unknown kind: {kind}"
    if backend not in ("jaccl", "ring"):
        return f"unsupported backend for mlx-distributed: {backend} (expected 'jaccl' or 'ring')"
    # For mlx-distributed with jaccl, validate RDMA wiring exists for every pair.
    if kind == "mlx-distributed" and backend == "jaccl":
        host_ids = [n.get("host") for n in nodes]
        for src in host_ids:
            if src not in RDMA_WIRING and len(host_ids) > 1:
                return f"host '{src}' has no RDMA wiring map in topology.yaml"
            for dst in host_ids:
                if src == dst: continue
                if not RDMA_WIRING.get(src, {}).get(dst):
                    return f"no RDMA wiring from {src} to {dst}"
    # Ensure ssh field is filled (look up KNOWN_HOSTS if missing)
    for n in nodes:
        if not n.get("ssh"):
            host_id = n.get("host")
            inv = next((h for h in KNOWN_HOSTS if h["id"] == host_id), None)
            if not inv:
                return f"unknown host: {host_id}"
            n["ssh"] = inv["ssh"]
    return None


def estimate_load_s(model: str, size_bytes: int, cluster: str, nodes: int) -> float:
    """Empirical estimate. Prefer a cached prior load_s for this model+nodes,
    fall back to size-based heuristic.

    Apple-SSD-to-GPU loading is ~3-5 GB/s on M3 Ultra. Multi-node lazy load is
    closer to a barrier wait than disk read, so the estimate must include
    framework overhead.
    """
    h = get_load_history(cluster, model)
    if h and h.get("last_load_s") and h.get("nodes") == nodes:
        # Prior known duration — bump 15% margin so the bar doesn't max early.
        return max(float(h["last_load_s"]) * 1.15, 6.0)
    size_gb = (size_bytes or 0) / (1024 ** 3)
    if size_gb <= 0:
        return 30.0
    # Heuristic: ~4 GB/s effective + 5s framework overhead, with a floor.
    return max(size_gb / 4.0 + 5.0, 8.0)


async def get_model_size_bytes(ssh: str, path: str) -> int:
    """SSH to a node and `du -sk` the model dir. Returns 0 on failure."""
    cmd = f"du -sk {shlex.quote(path)} 2>/dev/null | awk '{{print $1}}'"
    try:
        out = await asyncio.to_thread(
            subprocess.run,
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh, cmd],
            capture_output=True, text=True, timeout=15,
        )
        kb = int((out.stdout or "0").strip() or "0")
        return kb * 1024
    except Exception:
        return 0


async def batch_get_model_sizes(ssh: str, paths: list[str]) -> dict[str, int]:
    """One SSH call to size N model dirs. Avoids the per-model overhead of
    N separate connections — for a node with 15 models that's ~75s vs ~3s."""
    if not paths:
        return {}
    # Quote each path and emit `<bytes>\t<path>` per line via du.
    args = " ".join(shlex.quote(p) for p in paths)
    cmd = (
        f"for p in {args}; do "
        '  s=$(du -sk "$p" 2>/dev/null | awk \'{print $1}\'); '
        '  echo "${s:-0} $p"; '
        "done"
    )
    try:
        out = await asyncio.to_thread(
            subprocess.run,
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh, cmd],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return {p: 0 for p in paths}
    result: dict[str, int] = {}
    for line in (out.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        try:
            kb = int(parts[0])
        except ValueError:
            kb = 0
        result[parts[1]] = kb * 1024
    # fill misses with 0
    for p in paths:
        result.setdefault(p, 0)
    return result


# Mirror of the tensor-parallel ShardingStrategy dispatch in auto_parallel.py
# (the isinstance() chain — search "tensor_parallel_sharding_strategy ="). The
# orchestrator can't import auto_parallel (heavy mlx_lm imports, not present in
# this container), so the list is duplicated here. Any model_type NOT in this
# set is pipeline-only for multi-node (e.g. hy_v3). When a ShardingStrategy
# branch is added in auto_parallel.py, add its model_type string here too.
TENSOR_CAPABLE_MODEL_TYPES = frozenset({
    "llama", "ministral3",
    "deepseek_v3", "deepseek_v32", "kimi_k25", "deepseek_v4",
    "minimax", "glm4_moe", "glm4_moe_lite",
    "qwen3", "qwen3_moe", "qwen3_next", "qwen3_5", "qwen3_5_moe", "qwen3_vl",
    "gpt_oss", "step3p5", "nemotron_h", "gemma4",
})


def _resolve_model_abspath(model: str, base_dir: str) -> str:
    """Resolve a model id to an absolute path on the node. A leading '/' is
    taken as-is; otherwise it's joined under the cluster's models_dir. Without
    this, du/cat on a relative model id runs in the SSH home dir and silently
    returns nothing — which made the RAM preflight degrade to 'size unknown,
    proceed' (the guard was effectively a no-op for the normal relative-id
    case)."""
    if model.startswith("/"):
        return model
    return f"{base_dir.rstrip('/')}/{model}"


async def get_model_arch_meta(ssh: str, abspath: str) -> dict:
    """SSH to a node and read the model's config.json. Returns
    {model_type, num_hidden_layers, num_key_value_heads} (None when absent).
    Used to decide mode validity (tensor vs pipeline) WITHOUT loading the model.
    Cascades into text_config/language_model like runner.py does for the
    multimodal nests."""
    miss = {"model_type": None, "num_hidden_layers": None,
            "num_key_value_heads": None, "is_vision": False}
    cmd = f"cat {shlex.quote(abspath.rstrip('/') + '/config.json')} 2>/dev/null"
    try:
        out = await asyncio.to_thread(
            subprocess.run,
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh, cmd],
            capture_output=True, text=True, timeout=15,
        )
        cfg = json.loads(out.stdout or "{}")
    except Exception:
        return miss
    if not isinstance(cfg, dict):
        return miss

    def _nested(key):
        return (
            cfg.get(key)
            or cfg.get("text_config", {}).get(key)
            or cfg.get("language_model", {}).get(key)
            or (cfg.get("text_config", {}).get("language_model", {}) or {}).get(key)
        )
    # Vision detection — same rule as _model_capabilities: model_type carries a
    # "_vl"/"vision" marker OR a vision_config nest is present. Lets the loader
    # route VL models to the single-node mlx_vlm.server flow instead of the
    # distributed text runner (which can't run them).
    _mt = (cfg.get("model_type") or cfg.get("text_config", {}).get("model_type") or "").lower()
    is_vision = bool(
        "_vl" in _mt or "_vision" in _mt or "vision" in _mt
        or "vision_config" in cfg or "vision_tower_config" in cfg
    )
    return {
        "model_type": cfg.get("model_type"),
        "num_hidden_layers": _nested("num_hidden_layers"),
        "num_key_value_heads": _nested("num_key_value_heads"),
        "is_vision": is_vision,
    }


def _validate_load_mode(model_type, num_layers, num_kv_heads,
                        nodes_count: int, mode: str) -> tuple[bool, str]:
    """Is (mode, nodes_count) valid for this model's ARCHITECTURE? Independent
    of RAM — that's `_validate_load_fits`. Single-node is always mode-valid.

      - pipeline : needs num_hidden_layers >= nodes_count (one stage per node).
      - tensor   : needs the model_type to have a ShardingStrategy AND
                   num_key_value_heads divisible by nodes_count.

    Unknown metadata degrades to permissive (returns ok) so a config we can't
    read never hard-blocks a load — the runtime remains the final arbiter."""
    if nodes_count <= 1:
        return True, ""
    if mode == "pipeline":
        if num_layers and num_layers < nodes_count:
            return False, f"pipeline impossible : {num_layers} layers < {nodes_count} nodes"
        return True, ""
    if mode == "tensor":
        if model_type and model_type not in TENSOR_CAPABLE_MODEL_TYPES:
            return False, f"{model_type} n'a pas de stratégie tensor (pipeline only)"
        if num_kv_heads and num_kv_heads % nodes_count != 0:
            return False, f"KV-heads {num_kv_heads} non divisible par {nodes_count}"
        return True, ""
    return False, f"mode inconnu : {mode}"


def _model_load_overhead_factor() -> float:
    """Multiplier applied to model size to estimate the RAM footprint during
    load. Accounts for: (1) the safetensors mmap'd alongside actually-used
    weights during sharding, (2) KV cache pre-alloc, (3) MLX scratch buffers,
    (4) Python/MLX runtime, (5) macOS file cache pressure on the shared volume.
    1.15 = empirical, observed on Hy3 + GLM loads."""
    return 1.15


def _cluster_total_ram_bytes(cluster: str, nodes_count: int) -> tuple[int, list[dict]]:
    """Sum RAM across the nodes that would be used for a load of `cluster`
    at `nodes_count`. Uses the cached telemetry probe data; falls back to
    a static map per known hardware when telemetry hasn't yet probed a node.

    Returns (total_bytes, per_node_details). per_node_details is a list of
    {host, ssh, ram_bytes, wired_limit_bytes, from_telemetry: bool}. The
    `wired_limit_bytes` is the macOS `iogpu.wired_limit_mb` × 1 MiB and
    matters for MLX/Metal — it's the actual ceiling on how much memory a
    rank can hold (model weights + KV + activations must all fit under it).
    When the user has tuned wired_limit_mb above the default ~75% of RAM,
    we want the validator to honour it instead of being conservative.
    """
    try:
        topo = build_topology(cluster, nodes_count)
    except Exception:
        topo = []
    if not topo:
        return 0, []
    # Use the live telemetry snapshot when available.
    tdata = _telemetry_cache.get("data") or {}
    by_id: dict[str, dict] = {}
    for h in (tdata.get("hosts") or []):
        by_id[h.get("id")] = h
    out: list[dict] = []
    total = 0
    for n in topo:
        host_id = n.get("host")
        ssh = n.get("ssh")
        ram = None
        wired_limit = 0
        # First-best: live telemetry
        h = by_id.get(host_id)
        if h and h.get("ram_total_bytes"):
            ram = int(h["ram_total_bytes"])
        if h and h.get("wired_limit_mb"):
            wired_limit = int(h["wired_limit_mb"]) * 1024 * 1024
        # No static RAM map here: hardware belongs in telemetry or
        # topology/config, not in source. Unknown RAM degrades to a warning
        # path in _validate_load_fits instead of guessing.
        if not ram:
            ram = 0
        total += ram
        out.append({
            "host": host_id, "ssh": ssh, "ram_bytes": ram,
            "wired_limit_bytes": wired_limit,
            "from_telemetry": bool(h and h.get("ram_total_bytes")),
        })
    return total, out


def _hetero_pipeline_ceiling(per_node: list[dict]) -> int:
    """Max model bytes loadable in PIPELINE mode under the capacity-aware
    split the loader actually performs (see ram_weights_csv in start_runners):
    each rank's shard is proportional to its weight (wired_limit | raw RAM),
    and must fit its budget (wired_limit | 0.75×RAM) with the +10%
    activations factor. Returns 0 when the loader would NOT activate the
    capacity-aware split (a weight missing, or all nodes identical — in
    which case the even-split math of the caller is already exact)."""
    weights: list[int] = []
    budgets: list[int] = []
    for nd in per_node:
        w = nd.get("wired_limit_bytes") or nd.get("ram_bytes") or 0
        b = nd.get("wired_limit_bytes") or int((nd.get("ram_bytes") or 0) * 0.75)
        if not w or not b:
            return 0
        weights.append(int(w))
        budgets.append(int(b))
    if len(weights) < 2 or len(set(weights)) == 1:
        return 0
    sw = sum(weights)
    return int(min(b * sw / (1.10 * w) for w, b in zip(weights, budgets)))


def _validate_load_fits(model_size_bytes: int, cluster: str,
                         nodes_count: int,
                         mode: Optional[str] = None) -> tuple[bool, str, dict]:
    """Returns (ok, reason, detail) for a (model, nodes) combo.

    Two checks (both must pass):
      1. Total cluster RAM ≥ model × overhead (catches obviously-too-big loads)
      2. Per-rank shard ≤ smallest node's available RAM (catches the
         GLM-5.1-on-3-nodes class of mistake where total budget is fine
         but the smallest node can't hold its 1/N shard + activations)

    Check #2 is the one that matters in practice: in pipeline-parallel,
    each rank holds ~model_size/nodes_count of weights + KV cache +
    activations. If that exceeds the smallest node's RAM, the rank dies
    OOM mid-load and the cluster deadlocks at the barrier (the symptom
    first observed 2026-05-18 with GLM-5.1 on Default nodes=3).
    """
    if not model_size_bytes:
        return True, "size unknown — proceed at your own risk", {
            "model_size_bytes": 0, "cluster_ram_bytes": 0,
        }
    total_ram, per_node = _cluster_total_ram_bytes(cluster, nodes_count)
    if not total_ram:
        return True, "cluster ram unknown — proceed at your own risk", {
            "model_size_bytes": model_size_bytes, "cluster_ram_bytes": 0,
        }

    # Per-rank budget honors the actual macOS `iogpu.wired_limit_mb` when
    # telemetry has it. That's the real ceiling on Metal allocations
    # (weights + KV + activations all must fit under it). When the user
    # has tuned wired_limit above default with `sudo sysctl iogpu.wired_limit_mb`,
    # we want to allow larger loads accordingly — not stay stuck at the
    # default-75%-of-RAM conservative bound.
    #
    # Fallback when wired_limit unknown: 0.75 × RAM (matches default macOS).
    activations_factor = 1.10  # +10% per-rank for KV + activations + scratch
    default_headroom_factor = 0.75

    per_rank_required = int((model_size_bytes / max(nodes_count, 1)) * activations_factor)

    # Smallest node's budget — that's the binding constraint
    node_budgets = []
    for n in per_node:
        if n.get("wired_limit_bytes"):
            node_budgets.append(int(n["wired_limit_bytes"]))
        elif n.get("ram_bytes"):
            node_budgets.append(int(n["ram_bytes"] * default_headroom_factor))
    per_node_budget = min(node_budgets) if node_budgets else 0
    min_node_ram = min((n["ram_bytes"] for n in per_node if n["ram_bytes"]), default=0)

    overall_required = int(model_size_bytes * _model_load_overhead_factor())
    overall_headroom = total_ram - overall_required

    # Indicate whether budget came from actual tuned wired_limit or fallback
    budget_source = "wired_limit_mb" if any(n.get("wired_limit_bytes") for n in per_node) else f"{default_headroom_factor}×RAM"
    detail = {
        "model_size_bytes": model_size_bytes,
        "model_size_gb": round(model_size_bytes / 1024**3, 1),
        "cluster_ram_bytes": total_ram,
        "cluster_ram_gb": round(total_ram / 1024**3, 1),
        "per_rank_required_bytes": per_rank_required,
        "per_rank_required_gb": round(per_rank_required / 1024**3, 1),
        "min_node_ram_bytes": min_node_ram,
        "min_node_ram_gb": round(min_node_ram / 1024**3, 1),
        "per_node_budget_bytes": per_node_budget,
        "per_node_budget_gb": round(per_node_budget / 1024**3, 1),
        "per_node_budget_source": budget_source,
        "overall_required_gb": round(overall_required / 1024**3, 1),
        "activations_factor": activations_factor,
        "per_node": per_node,
    }

    # Check #1 : overall budget
    if overall_headroom < 0:
        return False, (
            f"model {detail['model_size_gb']} GB × {_model_load_overhead_factor()} overhead = "
            f"{detail['overall_required_gb']} GB needed, only {detail['cluster_ram_gb']} GB total "
            f"on {nodes_count} node{'s' if nodes_count>1 else ''}. Use more nodes or pick a smaller model."
        ), detail

    # Check #2 : per-rank shard vs smallest node. The naive form assumes an
    # EVEN split — but in pipeline mode the loader performs a capacity-aware
    # split (shards ∝ wired_limit, see ram_weights_csv), so on heterogeneous
    # clusters the big node absorbs the difference. Mirror that here, or the
    # gate refuses loads the engine handles fine (Ling-Q8 1T on 512+4×256).
    if per_rank_required > per_node_budget and per_node_budget > 0:
        if mode == "pipeline":
            ceiling = _hetero_pipeline_ceiling(per_node)
            if ceiling and model_size_bytes <= ceiling:
                detail["capacity_aware_split"] = True
                detail["hetero_ceiling_gb"] = round(ceiling / 1024**3, 1)
                return True, (
                    f"fits via capacity-aware pipeline split "
                    f"(ceiling {detail['hetero_ceiling_gb']} GB)"
                ), detail
        src_note = "from telemetry-probed wired_limit_mb" if budget_source == "wired_limit_mb" else "from default 0.75×RAM (telemetry has no wired_limit data)"
        return False, (
            f"per-rank shard {detail['per_rank_required_gb']} GB exceeds smallest node's safe "
            f"RAM budget ({detail['per_node_budget_gb']} GB {src_note}, on {detail['min_node_ram_gb']} GB hardware). "
            f"With pipeline split across {nodes_count} node{'s' if nodes_count>1 else ''}, each rank holds "
            f"~{round(detail['model_size_gb']/nodes_count, 1)} GB weights + Metal scratch + KV cache. "
            f"Use more nodes, OR run `sudo sysctl iogpu.wired_limit_mb=<higher>` on the small nodes "
            f"and let the next telemetry probe (~5s) pick it up."
        ), detail

    return True, "ok", detail


# Module-level loading state (read by /admin/.../status without holding the
# admin lock; the load handlers update these dicts inside their critical
# sections so the UI sees consistent values).
_nautilus_loading: dict = {"in_progress": False}
_loading_state: dict[str, dict] = {}


def _loading_state_for(cluster_id: str) -> dict:
    """Per-cluster loading state. Created lazily on first use."""
    if cluster_id not in _loading_state:
        _loading_state[cluster_id] = {"in_progress": False}
    return _loading_state[cluster_id]


def _begin_loading(state: dict, model: str, nodes: int, size_bytes: int,
                   estimated_s: float) -> None:
    state.update({
        "in_progress": True,
        "model": model,
        "nodes": nodes,
        "size_bytes": int(size_bytes),
        "estimated_s": float(estimated_s),
        "started_at": time.time(),
    })


def _end_loading(state: dict) -> None:
    state.clear()
    state["in_progress"] = False


def _loading_snapshot(state: dict) -> Optional[dict]:
    """Return a UI-friendly snapshot of the current load (or None if idle)."""
    if not state.get("in_progress"):
        return None
    elapsed = time.time() - state.get("started_at", time.time())
    est = max(state.get("estimated_s") or 1.0, 1.0)
    # Cap at 95 % until completion — last 5 % reserved for the success snap.
    pct = min(95.0, (elapsed / est) * 100.0)
    return {
        "model": state.get("model"),
        "nodes": state.get("nodes"),
        "size_bytes": state.get("size_bytes"),
        "elapsed_s": round(elapsed, 2),
        "estimated_s": round(est, 2),
        "progress_pct": round(pct, 1),
    }

# ──────────────────────────────────────────────────────────────────────────────
# HF model downloader (runs `hf download` on configured hosts via SSH)
# ──────────────────────────────────────────────────────────────────────────────
# `hf` is the shipped Python entrypoint of the remote venv. User-pip installs
# vary by Python version, so we point at the configured venv path instead.
HF_BIN_REMOTE = env_get("HF_BIN_REMOTE", f"{REMOTE_CLUSTER_DIR}/.venv/bin/hf")

_downloads: dict[str, dict] = {}
# Per-job → per-host process map. Host-keyed so cancel can target one target
# without killing the whole multi-host download.
_dl_procs: dict[str, dict[str, asyncio.subprocess.Process]] = {}


async def _hf_repo_total_bytes(repo: str, token: Optional[str]) -> Optional[int]:
    """Best-effort total size of an HF repo = sum of file sizes, via the HF tree
    API. Mechanism borrowed from the HF tools app (#14) to drive a real progress
    bar. Returns None on any failure so the caller falls back to indeterminate."""
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, headers=headers)
            if r.status_code != 200:
                return None
            total = 0
            for e in r.json():
                if e.get("type") == "file":
                    sz = e.get("size") or (e.get("lfs") or {}).get("size") or 0
                    total += int(sz)
            return total or None
    except Exception:
        return None


async def _hf_dl_one(dl_id: str, host: dict, repo: str,
                     hf_token: Optional[str], slot: dict) -> dict:
    """One target download. Mutates `slot` in place so the parent job's
    aggregated `per_target` array reflects live status. Uses HF CLI's
    built-in resume (snapshot_download.resume_download=True) so re-running
    after a cancel picks up where it left."""
    ssh = host["ssh"]
    models_dir = host.get("models_dir") or "/Volumes/models/mlx-vlm"
    # Models from HF land under `<org>/<name>` so the matrix sees them at
    # the right hierarchy (e.g. `inferencerlabs/Hy3-preview-MLX-9bit`).
    # That matches the layout the rest of OdyssAI-X expects.
    target = f"{models_dir}/{repo}"
    slot["host"] = host["id"]
    slot["ssh"] = ssh
    slot["target"] = target
    slot["status"] = "running"
    slot["bytes"] = 0
    slot["started_at"] = time.time()

    env_prefix = ""
    if hf_token:
        env_prefix = f"HF_TOKEN={shlex.quote(hf_token)} "
    # `hf download` resumes partial files by default. 4 workers keeps the
    # per-file resume granular even on multi-GB safetensors shards.
    cmd = (
        f"mkdir -p {shlex.quote(target)} && "
        f"{env_prefix}{HF_BIN_REMOTE} download {shlex.quote(repo)} "
        f"--local-dir {shlex.quote(target)} "
        f"--max-workers 4"
    )
    proc = await asyncio.create_subprocess_exec(
        "ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh, cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    # Track the per-target proc so cancel can target it specifically.
    _dl_procs.setdefault(dl_id, {})[host["id"]] = proc

    async def poll_size():
        while proc.returncode is None:
            try:
                rc, out, _ = await asyncio.to_thread(
                    _ssh_exec, ssh,
                    f"du -sk {shlex.quote(target)} 2>/dev/null | cut -f1", 10
                )
                kb = int((out or "0").strip().split()[0] or 0) if out else 0
                slot["bytes"] = kb * 1024
                # Human-readable echo for the legacy UI field
                slot["size"] = _human_bytes(kb * 1024)
            except Exception:
                pass
            await asyncio.sleep(3)

    poll_task = asyncio.create_task(poll_size())
    try:
        _, stderr = await proc.communicate()
    finally:
        poll_task.cancel()
    slot["finished_at"] = time.time()
    if proc.returncode == 0:
        slot["status"] = "done"
    elif slot.get("status") == "cancelled":
        pass
    else:
        slot["status"] = "error"
        slot["error"] = (
            stderr.decode(errors="replace")[:800] if stderr else f"exit {proc.returncode}"
        )
    # Per-target proc done — remove from registry. Parent decides overall status.
    by_host = _dl_procs.get(dl_id) or {}
    by_host.pop(host["id"], None)
    return slot


def _human_bytes(n: int) -> str:
    units = ["B", "K", "M", "G", "T"]
    f = float(n or 0)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f}{u}".rstrip("0").rstrip(".") if u != "B" else f"{int(f)}{u}"
        f /= 1024
    return f"{int(n)}B"


async def _hf_dl_run(dl_id: str, repo: str, hf_token: Optional[str],
                     targets: list[dict]) -> None:
    """Fan-out download to N targets in parallel. Each target lives in
    its own slot under `per_target`. Job's top-level `status` aggregates:

      - running  : at least one target still running
      - done     : all targets done
      - cancelled: any target cancelled and no other targets running
      - partial  : some done + some failed
      - error    : all targets failed
    """
    job = _downloads[dl_id]
    job["per_target"] = [{"host": t["id"], "ssh": t["ssh"], "status": "queued"}
                         for t in targets]
    # Map host_id → slot for the runner to mutate
    by_id = {p["host"]: p for p in job["per_target"]}
    # Real progress bar (#14): each target pulls the FULL repo, so the aggregate
    # target = repo_size × N targets. Best-effort via the HF API (HF tools idea);
    # None → the UI keeps the indeterminate bar.
    repo_bytes = await _hf_repo_total_bytes(repo, hf_token)
    if repo_bytes:
        job["repo_bytes"] = repo_bytes
        job["total_bytes"] = repo_bytes * max(1, len(targets))
    coros = [_hf_dl_one(dl_id, t, repo, hf_token, by_id[t["id"]])
             for t in targets]
    results = await asyncio.gather(*coros, return_exceptions=True)
    # Aggregate status
    statuses = [r.get("status") if isinstance(r, dict) else "error"
                for r in results]
    if all(s == "done" for s in statuses):
        job["status"] = "done"
    elif all(s == "cancelled" for s in statuses):
        job["status"] = "cancelled"
    elif any(s == "done" for s in statuses):
        job["status"] = "partial"
    else:
        job["status"] = "error"
        first_err = next((r.get("error") for r in results
                          if isinstance(r, dict) and r.get("error")), None)
        if first_err:
            job["error"] = first_err
    job["finished_at"] = time.time()
    # Aggregate bytes for the top-level row.
    total_bytes = sum(int(p.get("bytes") or 0) for p in job["per_target"])
    job["bytes"] = total_bytes
    job["size"] = _human_bytes(total_bytes)
    # Drop per-job proc registry once everything's done.
    _dl_procs.pop(dl_id, None)


# ──────────────────────────────────────────────────────────────────────────────
# (Hy3 service helpers removed — its single-node use case is now a normal
#  one-node cluster load.)
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Connection tests for cluster topology + remote services
# ──────────────────────────────────────────────────────────────────────────────
async def _ssh_ping(ssh_target: str) -> dict:
    """Ping a node via SSH (cheap: just `hostname` + uname). Returns latency_ms + ok."""
    t0 = time.time()
    try:
        p = await asyncio.to_thread(
            subprocess.run,
            ["ssh", "-o", "ConnectTimeout=3", "-o", "BatchMode=yes", _safe_ssh_target(ssh_target),
             "hostname && sysctl -n hw.memsize 2>/dev/null"],
            capture_output=True, text=True, timeout=8,
        )
        dt = (time.time() - t0) * 1000
        if p.returncode == 0:
            lines = p.stdout.strip().split("\n")
            host = lines[0] if lines else ""
            mem_gb = None
            if len(lines) > 1:
                try:
                    mem_gb = round(int(lines[1]) / (1024**3))
                except Exception:
                    pass
            return {"ok": True, "latency_ms": round(dt, 1), "hostname": host, "ram_gb": mem_gb}
        return {"ok": False, "latency_ms": round(dt, 1), "error": p.stderr.strip()[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


async def _http_ping(url: str, timeout: float = 3.0) -> dict:
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(url)
            dt = (time.time() - t0) * 1000
            return {"ok": r.status_code < 500, "latency_ms": round(dt, 1),
                    "status": r.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}






def random_ephemeral_port() -> int:
    return random.randint(49152, 65535)


def remote_cmd(node: dict, nodes: list[dict], model: str, mode: str, port: int,
               devices_json: str, use_ap: bool, emit_batch: int,
               kv_q8: bool = False,
               draft_model: Optional[str] = None,
               num_draft_tokens: int = 4,
               ram_weights_csv: Optional[str] = None) -> str:
    coord_ip = next(n for n in nodes if n["rank"] == COORDINATOR_RANK)["ssh"].split("@")[1]
    world_size = len(nodes)
    env = {
        "MLX_RANK": str(node["rank"]),
        "MLX_WORLD_SIZE": str(world_size),
        "MLX_JACCL_COORDINATOR": f"{coord_ip}:{port}",
        "MLX_IBV_DEVICES": "/tmp/mlx_jaccl_devices.json",
        "MLX_METAL_FAST_SYNCH": "1",
        "RUNNER_MODEL": model,
        "RUNNER_MODE": mode,
        "RUNNER_USE_AP": "1" if use_ap else "0",
        "RUNNER_KV_Q8": "1" if kv_q8 else "0",
        "RUNNER_EMIT_BATCH": str(emit_batch),
    }
    # Capacity-aware pipeline split (#9). When the orchestrator knows per-rank
    # RAM (via telemetry), pass it as a CSV of weights so the runner can size
    # each rank's layer count proportionally instead of doing the even split
    # that OOMs heterogeneous clusters (.29 512GB + .30 256GB on Qwen397).
    # Format: "wW0,wW1,...,wWN-1" where each Wi is the relative weight for
    # rank i (the runner normalises). The runner falls back to even split
    # when the env var is missing or invalid.
    if ram_weights_csv:
        env["RUNNER_RAM_WEIGHTS"] = ram_weights_csv
    if draft_model:
        env["RUNNER_DRAFT_MODEL"] = draft_model
        env["RUNNER_NUM_DRAFT_TOKENS"] = str(num_draft_tokens)
    env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    write_devices = f"echo {shlex.quote(devices_json)} > /tmp/mlx_jaccl_devices.json"
    # If the model path is relative, run the runner from the node's
    # models_dir so `Path(repo).exists()` resolves correctly. Without
    # this, runner.py falls through to `hf_repo_to_path()` which tries
    # the HF cache with `local_files_only=True` and dies with
    # LocalEntryNotFoundError — the regression Sophie hit on 2026-05-25
    # when loading inferencerlabs/Qwen3.5-397B-A17B-MLX-9bit by
    # relative path even though the model exists at
    # $models_dir/inferencerlabs/Qwen3.5-397B-A17B-MLX-9bit on every
    # node. Absolute model paths skip the cd.
    cd_prefix = ""
    if not model.startswith("/"):
        node_models_dir = node.get("models_dir") or DEFAULT_MODELS_DIR
        cd_prefix = f"cd {shlex.quote(node_models_dir)} && "
    return f"{write_devices} && {cd_prefix}{env_str} {PYTHON_REMOTE} {RUNNER_REMOTE}"


# ──────────────────────────────────────────────────────────────────────────────
# Runner pool
# ──────────────────────────────────────────────────────────────────────────────
# Max seconds `submit` waits for the next runner event before checking whether
# rank-0 (the sole event producer) is still alive. Long enough not to trip on a
# slow multi-node prefill, short enough to fail a dead runner promptly (#21).
_GEN_IDLE_TIMEOUT_S = float(env_get("GEN_IDLE_TIMEOUT_S", "120"))
# No-progress watchdog (2026-06-07). `_rank0_alive()` only polls the LOCAL ssh
# client, which stays connected (ServerAliveInterval=10) even when the remote
# runner is wedged in an MLX/JACCL collective emitting zero tokens — that hung a
# request silently for ~5h. These cap the wall-clock with NO `token` progress,
# separately for the (necessarily silent) prefill phase and the inter-token
# decode phase. Generous prefill budget: a healthy 19k-token prefill cost ~150s
# in prod and declared contexts reach 1M tokens, so we only abort well beyond
# the legitimate window. Both env-overridable.
_GEN_PREFILL_DEADLINE_S = float(env_get("GEN_PREFILL_DEADLINE_S", "600"))
_GEN_DECODE_DEADLINE_S = float(env_get("GEN_DECODE_DEADLINE_S", "90"))
# Conservative floor prefill throughput (tok/s) used to scale the prefill
# deadline by estimated prompt size: a 1M-token context legitimately prefills
# for minutes, so a flat 600s would false-positive on it. prefill_deadline =
# max(_GEN_PREFILL_DEADLINE_S, est_prompt_tokens / _MIN_PREFILL_TPS).
_MIN_PREFILL_TPS = max(1.0, float(env_get("MIN_PREFILL_TPS", "20")))
# Metal frees wired pages ASYNC after a killed runner exits; the orphan sweep
# re-reads wired after this grace before declaring a leak (the 166.7 GB phantom
# of 2026-06-16 that read 8 GB seconds later). Env-overridable.
_WIRED_REPOLL_GRACE_S = float(env_get("WIRED_REPOLL_GRACE_S", "10"))

# At most one watchdog-triggered recovery ladder per cluster in flight: multiple
# wedged in-flight requests would otherwise each spawn a duplicate _cluster_reset
# (which cancels runs and may clear the degraded flag mid-recovery). Also holds a
# strong ref so the detached task isn't GC'd (asyncio keeps only weak refs).
_WATCHDOG_RECOVERY_BY_CLUSTER: dict = {}


class RunnerProc:
    def __init__(self, node: dict, cmd: str, on_event, cluster: str = "nautilus"):
        self.node = node
        self.cluster = cluster
        # Only rank 0 produces events we parse; ranks > 0 would otherwise fill
        # their stdout pipe buffer (no reader) and block the remote process on
        # write(), stalling the distributed barrier -> deadlock. Send their
        # stdout to DEVNULL so the kernel never back-pressures them (#23).
        _stdout = subprocess.PIPE if node["rank"] == 0 else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            ["ssh", "-o", "ServerAliveInterval=10", _safe_ssh_target(node["ssh"]), cmd],
            stdin=subprocess.PIPE, stdout=_stdout, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self.ready = threading.Event()
        self._on_event = on_event
        # Keep the last N stderr lines so death-cause can be surfaced to the
        # API caller. Without this, a runner that crashes mid-load (OOM,
        # JACCL queue pair fail, model layout mismatch) leaves the dashboard
        # showing "timeout waiting for rank 0" — useless. With this, the
        # load endpoint can report the actual Python traceback.
        self._stderr_tail: deque = deque(maxlen=80)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        if node["rank"] == 0:
            threading.Thread(target=self._drain_stdout, daemon=True).start()

    # Per-rank load phase, parsed from stderr lines. Honest progress
    # signal that doesn't depend on rank 0 alone.
    phase: str = "spawning"
    phase_at: float = 0.0

    def stderr_tail(self, n: int = 20) -> str:
        """Return the last N stderr lines joined. Safe to call from any thread."""
        try:
            lines = list(self._stderr_tail)[-n:]
        except Exception:
            lines = []
        return "\n".join(lines)

    # Map runner.py log substrings → phase tag (ordered most-specific first).
    # Used to convert noisy stderr into a single clean "where is this rank"
    # signal that the dashboard can show per-rank during load.
    _PHASE_MARKERS: list[tuple[str, str]] = [
        ("init jaccl backend",            "init"),
        ("group ready in",                "group_ready"),
        ("loading model (lazy=True)",     "loading_weights"),
        ("applying pipeline_auto_parallel","sharding"),
        ("applying tensor_auto_parallel", "sharding"),
        ("pipeline shard layers",         "sharded"),
        ("loading tokenizer",             "loading_tokenizer"),
        ("barrier before ready",          "barrier"),
        ("model loaded in",               "loaded"),
        ("Traceback",                     "error"),
        ("MemoryError",                   "oom"),
        ("RuntimeError",                  "error"),
        ("queue pair",                    "jaccl_error"),
    ]

    def _maybe_update_phase(self, line: str) -> None:
        for needle, tag in self._PHASE_MARKERS:
            if needle in line:
                self.phase = tag
                self.phase_at = time.time()
                return

    def _drain_stderr(self):
        rank = self.node["rank"]
        for line in self.proc.stderr:
            line = line.rstrip("\n")
            sys.stderr.write(f"[rank{rank}] {line}\n")
            sys.stderr.flush()
            try:
                self._stderr_tail.append(line)
            except Exception:
                pass
            try:
                self._maybe_update_phase(line)
            except Exception:
                pass
            try:
                _push_log_line(self.cluster, rank, line)
            except Exception:
                pass

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
            if ev.get("event") == "ready":
                self.ready.set()
            self._on_event(ev)

    def send(self, obj: dict) -> bool:
        """Write a command to the runner's stdin. Returns True on success,
        False if the pipe is closed/broken — the keepalive (#40) relies on this
        to detect a PARTIAL broadcast (some ranks unreachable) and bail instead
        of awaiting an ack that can never arrive. Existing callers ignore it."""
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.write(json.dumps(obj) + "\n")
                self.proc.stdin.flush()
                return True
            except (BrokenPipeError, OSError):
                return False
        return False

    def graceful_stop(self, soft=10.0, term=10.0):
        try:
            self.send({"cmd": "stop"})
        except Exception:
            pass
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception:
            pass
        t0 = time.time()
        while time.time() - t0 < soft:
            if self.proc.poll() is not None:
                return
            time.sleep(0.2)
        sys.stderr.write(f"[rank{self.node['rank']}] soft timeout, SIGTERM (local ssh)\n")
        self.proc.terminate()
        t0 = time.time()
        while time.time() - t0 < term:
            if self.proc.poll() is not None:
                self._remote_pkill("after SIGTERM local")
                return
            time.sleep(0.2)
        sys.stderr.write(f"[rank{self.node['rank']}] SIGTERM timeout, SIGKILL\n")
        self.proc.kill()
        # Always pkill the remote python process — closing the local SSH socket
        # does NOT propagate any signal to the runner that's stuck in an mlx
        # compute kernel and not polling stdin. Without this, killing the local
        # client leaves a zombie runner.py on the Apple Silicon node holding
        # tens of GB of wired RAM until the next reboot.
        self._remote_pkill("after SIGKILL local")

    def _remote_pkill(self, reason: str) -> None:
        """SSH back to the node and kill any runner.py still running, trying
        SIGTERM first so MLX/Metal can release wired memory cleanly.

        Why two-phase:
          - SIGKILL alone (the old behavior) cleaned up the Python process
            but left macOS Metal wired pages reserved until reboot. Symptom:
            after each crashed-rank-then-unload cycle, ~190 GB wired stuck
            on the killed node, requiring `sudo shutdown -r now`. Operator
            observed this 3× on 2026-05-18.
          - SIGTERM lets Python catch the signal and the MLX `__del__` /
            atexit handlers free buffers via `mx.metal.clear_cache()` and
            release wired allocations.
          - If SIGTERM didn't take effect within 5s (process truly stuck
            in a C++ MLX kernel that ignores signals), we fall back to
            SIGKILL — the wired memory will leak, but at least the process
            is gone.

        Single SSH call because each round-trip is 200-500ms. Combined shell
        gives us SIGTERM → sleep 5 → check → SIGKILL atomically.
        """
        try:
            rank = self.node["rank"]
            ssh = self.node.get("ssh")
            if not ssh:
                return
            sys.stderr.write(f"[rank{rank}] remote graceful kill on {ssh} ({reason})\n")
            # `pkill -TERM` (default signal) → wait 5s for cleanup → check
            # → SIGKILL only if still alive. Returns 0 if any process was
            # signalled (TERM phase), 1 if no match. We don't care about
            # the exact rc here, just that we tried both phases.
            pattern = shlex.quote(RUNNER_MATCH_PATTERN)
            # 24×0.5s = 12s grace so the runner's SIGTERM-driven Metal
            # cleanup (free_metal) finishes on big models before SIGKILL.
            cmd = (
                f"pkill -TERM -f {pattern} 2>/dev/null && "
                f"for i in $(seq 1 24); do "
                f"  pgrep -f {pattern} >/dev/null 2>&1 || break; "
                f"  sleep 0.5; "
                f"done; "
                f"pgrep -f {pattern} >/dev/null 2>&1 && "
                f"  (echo 'SIGTERM ignored, escalating to SIGKILL — wired memory will leak' >&2 && "
                f"   pkill -9 -f {pattern}) || "
                f"  echo 'clean exit'"
            )
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no", ssh, cmd],
                capture_output=True, text=True, timeout=15,
            )
            tail = (r.stdout + r.stderr).strip().splitlines()[-1:] if (r.stdout or r.stderr) else []
            sys.stderr.write(f"[rank{rank}] remote kill result: {tail[0] if tail else '(no output)'}\n")
        except Exception as e:
            sys.stderr.write(f"[rank{self.node.get('rank','?')}] remote kill failed: {e}\n")


def _sweep_orphan_runners(cluster_id: str,
                          timeout_per_node: float = 15.0) -> dict:
    """Two-phase pkill of any lingering runner.py on every node of `cluster_id`.

    Why this exists:
      The orchestrator container is the only thing that should launch runner.py
      on the macs. But if the container crashes/restarts while runners are still
      alive on the macs, the new container starts with `_*_pool = None` and has
      no idea those zombies exist. They keep ~200 GB wired in MLX/Metal each
      until reboot, and JACCL re-init fails because the queue pairs are still
      occupied (errno 16). Symptom on 2026-05-18: container restart → auto-load
      crashes with "Changing queue pair to RTR failed" → operator clicks Unload →
      API returns `loaded:false` instantly (pool is None) → wired RAM stays high
      on the affected nodes.

      Fix: at container startup AND in unload endpoints (when pool is None),
      sweep the cluster's nodes for orphan runner.py and kill them with the
      same SIGTERM→grace→SIGKILL pattern used by RunnerProc._remote_pkill.
      This idempotent reset is cheap (~1s per node) and lets JACCL re-init.

    Runs nodes in parallel (each SSH is independent). Returns a per-node
    summary dict for logging / UI.
    """
    cd = get_cluster_def(cluster_id)
    nodes = cd.get("nodes") or []
    if not nodes:
        return {"cluster": cluster_id, "swept": [], "note": "no nodes defined"}
    pattern = shlex.quote(RUNNER_MATCH_PATTERN)
    # The sweep does three things in one SSH round-trip per node:
    #   1. kill any runner.py (two-phase SIGTERM → grace → SIGKILL)
    #   2. probe wired memory AFTER the kill — surfaces the Metal-leak
    #      symptom where a SIGKILL'd process leaves wired pages reserved
    #      until reboot (the very thing v1.4.2's SIGTERM-first pattern
    #      tries to avoid). High residual wired post-sweep = node still
    #      has a problem the sweep couldn't fix.
    #   3. report the kill result + wired bytes so the caller can warn.
    cmd = (
        # ── phase 1: kill ────────────────────────────────────────────
        # 24×0.5s = 12s grace. The runner now frees its Metal cache in its
        # SIGTERM-driven exit path (runner.py free_metal); for a 200 GB+
        # model the array deallocation + clear_cache can take several
        # seconds, so we give it room before escalating to SIGKILL (which
        # is what leaks the wired pages).
        f"if pgrep -f {pattern} >/dev/null 2>&1; then "
        f"  pkill -TERM -f {pattern} 2>/dev/null; "
        f"  for i in $(seq 1 24); do "
        f"    pgrep -f {pattern} >/dev/null 2>&1 || break; "
        f"    sleep 0.5; "
        f"  done; "
        f"  if pgrep -f {pattern} >/dev/null 2>&1; then "
        f"    pkill -9 -f {pattern}; echo 'killed (SIGKILL)'; "
        f"  else echo 'cleaned (SIGTERM)'; fi; "
        f"else echo 'no orphan'; fi; "
        # ── phase 2: wired memory probe ──────────────────────────────
        # vm_stat reports in pages; multiply by page size for bytes.
        f"PS=$(vm_stat | head -1 | awk '{{print $8}}' | tr -d '.'); "
        f"WIRED_PAGES=$(vm_stat | awk '/Pages wired down/{{gsub(/\\./,\"\",$4); print $4}}'); "
        f"echo \"WIRED_BYTES=$((WIRED_PAGES * PS))\""
    )

    # Wired-bytes ceiling above which we warn the operator. macOS keeps a
    # baseline of wired pages for the kernel + drivers (~3-6 GB typical).
    # Anything substantially above that on a node where we just killed the
    # only Metal workload = leaked pages we'd hoped the SIGTERM path freed.
    WIRED_WARN_THRESHOLD = 15 * 1024**3  # 15 GB

    def _run_one(node: dict) -> dict:
        ssh = node.get("ssh")
        host = node.get("host") or ssh or "?"
        if not ssh:
            return {"host": host, "ok": False, "result": "no ssh"}
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                 ssh, cmd],
                capture_output=True, text=True, timeout=timeout_per_node,
            )
            # Parse the two lines: kill result + WIRED_BYTES=N
            kill_result = "(no output)"
            wired_bytes = None
            for line in (r.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("WIRED_BYTES="):
                    try:
                        wired_bytes = int(line.split("=", 1)[1])
                    except (ValueError, IndexError):
                        pass
                else:
                    kill_result = line  # last non-WIRED line wins
            # Metal frees wired pages ASYNC after the killed process exits, so an
            # immediate read false-positives a leak (166.7 GB at kill → 8 GB a few
            # seconds later, 2026-06-16). If the first read is high, wait out the
            # async free and re-read once (the probe is idempotent — orphans are
            # already gone) before judging.
            if wired_bytes and wired_bytes > WIRED_WARN_THRESHOLD:
                time.sleep(_WIRED_REPOLL_GRACE_S)
                try:
                    rr = subprocess.run(
                        ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                         ssh, cmd],
                        capture_output=True, text=True, timeout=timeout_per_node,
                    )
                    for line in (rr.stdout or "").splitlines():
                        if line.strip().startswith("WIRED_BYTES="):
                            try:
                                wired_bytes = int(line.strip().split("=", 1)[1])
                            except (ValueError, IndexError):
                                pass
                            break
                except Exception:
                    pass
            wired_warn = bool(wired_bytes and wired_bytes > WIRED_WARN_THRESHOLD)
            return {
                "host": host,
                "ok": r.returncode == 0,
                "result": kill_result,
                "rc": r.returncode,
                "wired_bytes": wired_bytes,
                "wired_gb": round(wired_bytes / 1024**3, 1) if wired_bytes else None,
                "wired_warn": wired_warn,
            }
        except subprocess.TimeoutExpired:
            return {"host": host, "ok": False, "result": "ssh timeout"}
        except Exception as e:
            return {"host": host, "ok": False, "result": f"error: {e}"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        results = list(pool.map(_run_one, nodes))
    # Log summary + flag warnings prominently. Wired-leak detection here is
    # the missing post-sweep check from the 2026-05-18 audit — without it,
    # the operator only finds out their RAM is stuck when they try to load
    # the next model and OOM.
    summary = []
    warnings: list[str] = []
    for r in results:
        wired = f" wired={r['wired_gb']}GB" if r.get("wired_gb") is not None else ""
        warn = " ⚠️ WIRED LEAK SUSPECTED" if r.get("wired_warn") else ""
        summary.append(f"{r['host']}={r['result']}{wired}{warn}")
        if r.get("wired_warn"):
            warnings.append(
                f"{r['host']} retains {r['wired_gb']} GB wired after sweep — "
                f"Metal pages may be leaked; reboot may be needed before next load"
            )
    sys.stderr.write(f"[api] orphan sweep on {cluster_id}: " + ", ".join(summary) + "\n")
    for w in warnings:
        sys.stderr.write(f"[api] ⚠️  {w}\n")
    # If the sweep found leaked wired memory on any node, that cluster is
    # not safe to reload until the operator resets or reboots — mark it.
    if warnings:
        _mark_cluster_degraded(
            cluster_id,
            reason="wired memory leak suspected post-sweep",
            details={"warnings": warnings,
                     "nodes": [{"host": r["host"], "wired_gb": r.get("wired_gb")}
                               for r in results if r.get("wired_warn")]},
        )
    return {"cluster": cluster_id, "swept": results, "warnings": warnings}


class RunnerPool:
    def __init__(self, model: str, mode: str, use_ap: bool, nodes_count: int = 2,
                 emit_batch: int = 10, cluster: str = "nautilus",
                 kv_q8: bool = False,
                 draft_model: Optional[str] = None,
                 num_draft_tokens: int = 4,
                 alias: Optional[str] = None,
                 node_indices: Optional[list[int]] = None):
        self.model = model
        self.mode = mode
        self.use_ap = use_ap
        self.cluster = cluster
        self.nodes_count = nodes_count
        self.kv_q8 = kv_q8
        self.draft_model = draft_model
        self.num_draft_tokens = num_draft_tokens
        # Default alias = cluster name (back-compat: single pool per cluster
        # stays at alias=="default" / "nautilus"). Additional pool aliases
        # pools pass an explicit alias like "default-big".
        self.alias = alias or cluster
        # `node_indices` (Default only) selects which subset of the cluster's
        # nodes this pool occupies. When absent we fall back to the legacy
        # contiguous-from-0 behaviour using `nodes_count`. The order of the
        # list IS the rank order — first index becomes rank 0.
        self.node_indices: Optional[list[int]] = (
            list(node_indices) if node_indices else None
        )
        if node_indices:
            topo = build_topology_from_indices(cluster, list(node_indices))
        else:
            topo = build_topology(cluster, nodes_count)
        if not topo:
            raise ValueError(f"{cluster}: no topology configured for nodes_count {nodes_count}")
        self.nodes = topo
        self.emit_batch = emit_batch
        self.runners: list[RunnerProc] = []
        # broadcast_lock: small critical section around the per-runner send so
        # concurrent submits don't interleave their bytes across ranks. The
        # bulk of the request lifetime (waiting for tokens) is unlocked.
        self.broadcast_lock = asyncio.Lock()
        self._listeners: dict[str, asyncio.Queue] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.load_s: Optional[float] = None
        self.started_at: Optional[float] = None
        # Idle TTL bookkeeping. `last_used_at` is touched on every submit; the
        # background ttl sweeper auto-unloads when (now - last_used_at) exceeds
        # `ttl_seconds`. ttl_seconds=0 disables. Default sourced from
        # `settings.pool_ttl_seconds_default` at start time.
        self.last_used_at: float = time.time()
        # Pool-level liveness for the no-progress watchdog: monotonic time of the
        # last token emitted for ANY request. A request queued behind a long one
        # on the serialised multi-rank runner reads this so it isn't mistaken for
        # a wedge (head-of-line blocking false-positive, 2026-06-16).
        self.last_token_at: float = time.monotonic()
        self.ttl_seconds: int = 0
        # #40 foundation — pool-level busy/idle tracking + maintenance gate.
        # `busy_count` counts in-flight distributed forwards (gen + prewarm)
        # at the POOL level, so idle-gating works for EVERY protocol (OpenAI,
        # Anthropic, …) — unlike `_active_runs`, which some routes skip.
        # The keepalive (WU2) and preventive reload (WU1) only act when a pool
        # is idle, and claim it via `_try_claim_maintenance()` so no forward can
        # start mid-action. `_idle_gate` is SET when the pool is free to accept
        # forwards, CLEARED while a maintenance action holds the pool.
        self.busy_count: int = 0
        self.maintenance: bool = False
        self._idle_gate: asyncio.Event = asyncio.Event()
        self._idle_gate.set()
        # WU2/WU3 keepalive health: consecutive missed keepalives. Reset to 0 on
        # any successful keepalive; WU3 fires controlled recovery at threshold.
        self.keepalive_fails: int = 0
        self.last_keepalive_ok_at: Optional[float] = None
        # WU4 groundwork: which transport this pool actually initialised with.
        # Defaults to the cluster backend; only jaccl pools have the QP bug, so
        # the keepalive/preventive-reload machinery targets jaccl pools.
        self.backend: str = "jaccl"
        # Degraded state — set when we detect a JACCL queue-pair death, a
        # rank that crashed mid-gen with a recognizable RDMA errno, or a
        # post-sweep wired-memory leak. Reloads that would reuse the
        # degraded state are refused until the operator explicitly
        # `/admin/{cluster}/reset`s the pool. Pattern from the
        # 2026-05-18 audit: "ne pas recharger en boucle un pool sale".
        self.degraded: bool = False
        self.degraded_reason: Optional[str] = None
        self.degraded_at: Optional[float] = None

    def _on_event(self, ev: dict):
        # Any token (for ANY request) proves the runner is making progress and is
        # not wedged — stamp the pool-level liveness clock the no-progress
        # watchdog reads (head-of-line fix, 2026-06-16).
        if ev.get("event") == "token":
            self.last_token_at = time.monotonic()
        req_id = ev.get("id")
        if req_id and req_id in self._listeners:
            q = self._listeners[req_id]
            if self._loop is not None:
                self._loop.call_soon_threadsafe(q.put_nowait, ev)

    async def start(self):
        self._loop = asyncio.get_running_loop()
        # Pre-flight: validate the model is PRESENT and COMPLETE on EVERY target
        # node (not just rank-0) before sshing the runners. A model rsync'd to
        # rank-0 but missing/half-copied on the others used to pass and then
        # crash mid-load with a cryptic per-rank HFValidationError (bf16 5-node,
        # rsync unfinished, 2026-06-14). Reject upfront, naming the bad node(s).
        # Per node best-effort: an SSH/probe ERROR (not a model-absent verdict)
        # logs and is skipped so exotic topologies don't false-block.
        if self.model:
            async def _probe_node(n):
                ssh_t = (
                    n.get("ssh")
                    or (f"{n.get('user','admin')}@{n['ip']}" if n.get("ip") else None)
                )
                if not ssh_t:
                    return None
                try:
                    ok, err = await _validate_model_layout(
                        ssh_t, self.model, models_dir=n.get("models_dir")
                    )
                except Exception as e:
                    sys.stderr.write(
                        f"[api] layout probe skipped on {n.get('host', ssh_t)} ({e})\n"
                    )
                    return None
                return None if ok else f"{n.get('host', ssh_t)} → {err}"
            problems = [
                r for r in await asyncio.gather(*[_probe_node(n) for n in self.nodes])
                if isinstance(r, str)
            ]
            if problems:
                raise RuntimeError(
                    "model not present/complete on all target nodes — "
                    + "; ".join(problems)
                    + ". Rsync the model to every node before loading."
                )
        port = random_ephemeral_port()
        devices = [n["rdma"] for n in sorted(self.nodes, key=lambda x: x["rank"])]
        devices_json = json.dumps(devices)
        # Capacity-aware split (#9). Gather per-rank RAM via telemetry so the
        # runner can split layers proportionally rather than evenly. Falls
        # back to even split when telemetry is missing (single-node, fresh
        # cluster) — runner detects empty/invalid env and ignores it.
        ram_weights_csv: Optional[str] = None
        if self.nodes_count > 1 and self.mode == "pipeline":
            try:
                _total, per_node = _cluster_total_ram_bytes(
                    self.cluster, self.nodes_count
                )
                # _cluster_total_ram_bytes returns entries in rank order
                # (builds via build_topology). Use wired_limit when set,
                # otherwise raw RAM — wired_limit is the actual Metal
                # ceiling and is what JACCL/MLX can spend.
                weights: list[int] = []
                for entry in per_node:
                    w = entry.get("wired_limit_bytes") or entry.get("ram_bytes") or 0
                    weights.append(int(w))
                # Only enable when EVERY rank has a non-zero weight AND
                # there's actual heterogeneity (otherwise even-split is
                # already correct and we save a code path).
                if all(w > 0 for w in weights) and len(set(weights)) > 1:
                    ram_weights_csv = ",".join(str(w) for w in weights)
                    sys.stderr.write(
                        f"[api] capacity-aware split: per-rank weights = {weights} "
                        f"(bytes; runner will normalise)\n"
                    )
            except Exception as e:
                sys.stderr.write(
                    f"[api] capacity-aware split: telemetry probe failed ({e}); "
                    f"falling back to even split\n"
                )
        sys.stderr.write(
            f"[api] starting {self.nodes_count} runners "
            f"(model={self.model}, mode={self.mode}, ap={self.use_ap}, port={port})\n"
        )
        t0 = time.time()
        for node in self.nodes:
            cmd = remote_cmd(node, self.nodes, self.model, self.mode, port, devices_json,
                             self.use_ap, self.emit_batch, kv_q8=self.kv_q8,
                             draft_model=self.draft_model,
                             num_draft_tokens=self.num_draft_tokens,
                             ram_weights_csv=ram_weights_csv)
            self.runners.append(RunnerProc(node, cmd, self._on_event, cluster=self.cluster))
        rank0 = next(r for r in self.runners if r.node["rank"] == 0)

        # Wait for rank 0 to emit `ready` while watching EVERY rank for death.
        # Old behavior: only checked rank 0, so a dead rank-1 left rank 0
        # blocked on `mx.distributed.barrier()` and we'd wait the full 600s
        # before raising a misleading "timeout waiting for rank 0" — the
        # symptom first observed on the 2026-05-18 GLM-5.1 3-node load.
        load_timeout_s = 600.0
        no_progress_grace_s = 60.0    # how long to wait after a rank death
                                       # before surrendering — to let other
                                       # ranks emit their tracebacks too
        deaths: list[tuple[int, int, str]] = []  # (rank, exit_code, stderr_tail)
        while True:
            elapsed = time.time() - t0
            if rank0.ready.is_set():
                break
            if elapsed >= load_timeout_s:
                raise RuntimeError(
                    f"load timeout: rank 0 didn't reach ready in {load_timeout_s:.0f}s. "
                    f"rank 0 stderr tail:\n{rank0.stderr_tail()}"
                )
            # Scan all ranks for death
            for r in self.runners:
                rc = r.proc.poll()
                if rc is not None:
                    rank = r.node["rank"]
                    if not any(d[0] == rank for d in deaths):
                        deaths.append((rank, rc, r.stderr_tail()))
                        sys.stderr.write(
                            f"[api] rank {rank} died during startup (exit={rc}) — collecting stderr\n"
                        )
            if deaths:
                # Wait briefly for other ranks to also die / emit logs, then
                # fail with a structured report.
                death_t = time.time()
                while time.time() - death_t < no_progress_grace_s:
                    if rank0.ready.is_set():
                        # rare but possible: a non-rank-0 died after rank 0
                        # finished its part. Treat rank 0 ready as success
                        # even with deaths — actually no, the cluster needs
                        # ALL ranks at the barrier. If rank 0 is "ready" but
                        # others died, the pool is broken.
                        break
                    # Collect more deaths
                    for r in self.runners:
                        rc = r.proc.poll()
                        if rc is not None and not any(d[0] == r.node["rank"] for d in deaths):
                            deaths.append((r.node["rank"], rc, r.stderr_tail()))
                    await asyncio.sleep(0.5)
                # Build a single error message with all collected info.
                death_report = "\n\n".join(
                    f"--- rank {rk} (exit={rc}) ---\n{tail}"
                    for rk, rc, tail in sorted(deaths)
                )
                # Flip the cluster to degraded when the failure pattern
                # looks JACCL/RDMA — otherwise the next load attempt
                # crashes the same way and we trash the macs further.
                # Surface only the first matching tail for the reason.
                combined = "\n".join(tail for _, _, tail in deaths)
                if _looks_like_jaccl_error(combined):
                    first_tail = next(
                        (t for _, _, t in deaths if _looks_like_jaccl_error(t)),
                        combined,
                    )
                    _mark_cluster_degraded(
                        self.cluster,
                        reason="JACCL/RDMA error during load",
                        details={"dead_ranks": [d[0] for d in deaths],
                                 "tail": first_tail[-500:]},
                    )
                raise RuntimeError(
                    f"{len(deaths)} rank(s) died during load — pool unusable.\n{death_report}"
                )
            await asyncio.sleep(0.5)

        # Belt-and-braces: even if rank 0 is "ready", verify no other rank
        # quietly died. (Shouldn't happen since rank 0 ready implies the
        # barrier crossed, but distributed code is sneaky.)
        for r in self.runners:
            rc = r.proc.poll()
            if rc is not None:
                raise RuntimeError(
                    f"rank {r.node['rank']} died right at ready barrier (exit={rc}).\n{r.stderr_tail()}"
                )

        self.load_s = time.time() - t0
        self.started_at = time.time()
        sys.stderr.write(f"[api] ready in {self.load_s:.1f}s\n")

    async def stop(self):
        sys.stderr.write("[api] stopping runners\n")
        for r in self.runners:
            try:
                r.send({"cmd": "stop"})
            except Exception:
                pass
        for r in self.runners:
            try:
                if r.proc.stdin and not r.proc.stdin.closed:
                    r.proc.stdin.close()
            except Exception:
                pass
        # Stop every rank concurrently and isolate failures: a single rank
        # raising in graceful_stop() must not leave the others running (orphaned
        # runners hold wired GPU memory), and a hung SSH on one rank must not
        # serialise the whole teardown (#25).
        def _stop_all(runners):
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(runners))
            ) as ex:
                futs = {ex.submit(r.graceful_stop): r for r in runners}
                for fut in concurrent.futures.as_completed(futs):
                    try:
                        fut.result()
                    except Exception as e:
                        host = futs[fut].node.get("ssh", "?")
                        sys.stderr.write(f"[api] graceful_stop failed on {host}: {e}\n")
        await asyncio.to_thread(_stop_all, list(self.runners))
        sys.stderr.write("[api] all runners stopped\n")
        self.runners.clear()

    def alive_count(self) -> int:
        return sum(1 for r in self.runners if r.proc.poll() is None)

    def _rank0_alive(self) -> bool:
        """True if the rank-0 runner (the sole event producer for `submit`)
        is still running. Once it dies, no further events will ever arrive."""
        for r in self.runners:
            if r.node.get("rank") == 0:
                return r.proc.poll() is None
        return False

    def is_idle(self) -> bool:
        """No in-flight distributed forwards and not already under maintenance."""
        return self.busy_count == 0 and not self.maintenance

    def _try_claim_maintenance(self) -> bool:
        """Atomically claim the pool for a maintenance action (keepalive /
        preventive reload) — returns True only if the pool was idle. Fully
        synchronous (no await), so it can't interleave with submit's own
        busy-claim. While held, `_idle_gate` is cleared so new forwards wait."""
        if self.maintenance or self.busy_count > 0:
            return False
        self.maintenance = True
        self._idle_gate.clear()
        return True

    def _release_maintenance(self) -> None:
        self.maintenance = False
        self._idle_gate.set()

    async def _acquire_busy(self) -> None:
        """Wait out any maintenance window, then atomically mark the pool busy.
        The sync re-check after the gate await is what closes the race against a
        maintenance claim that lands while we were parked on the gate."""
        while True:
            await self._idle_gate.wait()
            if not self.maintenance:
                self.busy_count += 1
                return

    def _release_busy(self) -> None:
        self.busy_count = max(0, self.busy_count - 1)

    async def submit(self, prompt: Optional[str], max_tokens: int,
                     enable_thinking: Optional[bool],
                     messages: Optional[list[dict]] = None,
                     tools: Optional[list[dict]] = None,
                     session_id: Optional[str] = None,
                     request_id: Optional[str] = None,
                     reasoning_effort: Optional[str] = None) -> AsyncIterator[dict]:
        # Concurrent submits are allowed: the runner side handles serialisation
        # (single-rank uses BatchGenerator for true parallelism; multi-rank
        # serialises in the gen loop but tokens are routed by req_id).
        # Touch the activity clock for the TTL sweeper.
        # `request_id` lets the caller propagate its OUTER id (e.g.
        # OpenAI completion_id) so hard cancel via pool.cancel(rid)
        # targets the right slot on every rank.
        self.last_used_at = time.time()
        req_id = request_id or uuid.uuid4().hex[:8]
        q: asyncio.Queue = asyncio.Queue()
        self._listeners[req_id] = q
        req: dict = {"cmd": "gen", "id": req_id, "max_tokens": max_tokens}
        if messages is not None:
            req["messages"] = messages
        elif prompt is not None:
            req["prompt"] = prompt
        if tools:
            req["tools"] = tools
        # Server-wide default: thinking OFF unless the client explicitly opts in.
        # Models that don't support thinking ignore the field, so this is safe.
        # Default is dashboard-editable via /admin/settings (cluster-config.json
        # `settings.enable_thinking_default`), env fallback `THINKING_DEFAULT`.
        if enable_thinking is None:
            enable_thinking = get_enable_thinking_default()
        req["enable_thinking"] = enable_thinking
        # reasoning_effort: forwarded as a chat-template kwarg (Step-3.7 reads
        # it). Only set when non-empty so models that don't read it are untouched.
        if reasoning_effort:
            req["reasoning_effort"] = reasoning_effort
        if session_id:
            req["session_id"] = session_id
        # #40: mark the pool busy (idle-gating for keepalive + preventive reload)
        # only now that the request is fully built and about to broadcast, so a
        # build-time error can't leak the counter. Released in the finally below.
        await self._acquire_busy()
        try:
            # Tight broadcast window so concurrent submits don't interleave
            # bytes on the runner stdins.
            async with self.broadcast_lock:
                for r in self.runners:
                    r.send(req)
            # Bound the wait so a runner that dies mid-generation (rank-0 SSH
            # process killed, node panic, JACCL queue-pair death) can't hang the
            # request forever. If nothing arrives within the idle window AND the
            # producer (rank 0) is gone, surface an error instead of blocking on
            # a queue nobody will ever feed (#21).
            #
            # No-progress watchdog: `_rank0_alive()` only proves the local ssh
            # client is up, NOT that the remote runner is making progress — a
            # runner wedged in an MLX/JACCL collective keeps ssh connected while
            # emitting zero tokens (the ~5h silent hang of 2026-06-07). So we
            # also bound the wall-clock with no `token` progress and, on breach,
            # mark the pool+cluster degraded, best-effort cancel, fire the
            # recovery ladder, and fail the request. `token` is the only
            # real-progress event (runner emits no heartbeat), so it is the only
            # thing that resets the deadline.
            # Prompt-size-aware prefill budget: a 1M-token context legitimately
            # prefills for minutes, so scale the prefill deadline by an estimated
            # prompt size at a conservative floor throughput — only a TRUE hang
            # exceeds it. Decode, once started, gets the tight _GEN_DECODE budget.
            _est_prompt_toks = (
                len(prompt or "")
                + sum(len(str(m.get("content", ""))) for m in (messages or []))
            ) / 4.0
            prefill_deadline = max(_GEN_PREFILL_DEADLINE_S, _est_prompt_toks / _MIN_PREFILL_TPS)
            last_progress = time.monotonic()
            seen_token = False
            while True:
                deadline = _GEN_DECODE_DEADLINE_S if seen_token else prefill_deadline
                # Pool-aware: the runner is WEDGED only if it emits no token for
                # ANY request within the deadline. Without this, a request queued
                # behind a long-running one on the serialised multi-rank runner
                # false-fires a phantom "wedged in prefill" + recovery (the
                # head-of-line crash of 2026-06-16). `self.last_token_at` is bumped
                # by _on_event on every token from any request.
                stalled = time.monotonic() - max(last_progress, self.last_token_at)
                # No-progress watchdog (checked every iteration, not only on
                # timeout, so a stream of non-token events can't mask a wedge):
                # ssh up (so _rank0_alive passes) but the remote runner is stuck
                # emitting no tokens past the budget.
                if stalled >= deadline and self._rank0_alive():
                    phase = "decode" if seen_token else "prefill"
                    # Mark degraded FIRST (refuses new loads 409 + flags this
                    # pool), best-effort cancel, then fire the recovery ladder
                    # DETACHED — it stops+sweeps this very pool, so it cannot be
                    # awaited from inside this pool's own generator — and fail.
                    self.degraded = True
                    self.degraded_reason = "gen no-progress watchdog"
                    self.degraded_at = time.time()
                    _mark_cluster_degraded(
                        self.cluster, "gen no-progress watchdog",
                        {"phase": phase, "no_progress_s": round(stalled, 1),
                         "request_id": req_id, "alias": self.alias},
                    )
                    try:
                        await self.cancel(req_id)
                    except Exception:
                        pass
                    # One recovery ladder per cluster at a time; concurrent
                    # wedged requests just fail without spawning duplicates.
                    if self.cluster not in _WATCHDOG_RECOVERY_BY_CLUSTER:
                        _t = asyncio.create_task(_cluster_reset(self.cluster))
                        _WATCHDOG_RECOVERY_BY_CLUSTER[self.cluster] = _t
                        _t.add_done_callback(
                            lambda _done, c=self.cluster: _WATCHDOG_RECOVERY_BY_CLUSTER.pop(c, None)
                        )
                    raise RuntimeError(
                        f"runner wedged mid-generation: no token for "
                        f"{stalled:.0f}s in {phase} phase — pool marked "
                        f"degraded, recovery triggered"
                    )
                try:
                    # Wake at the deadline (so the check above fires promptly),
                    # capped at the rank-0 liveness cadence.
                    wait = min(_GEN_IDLE_TIMEOUT_S, max(0.5, deadline - stalled))
                    ev = await asyncio.wait_for(q.get(), timeout=wait)
                except asyncio.TimeoutError:
                    if not self._rank0_alive():
                        raise RuntimeError(
                            "runner died mid-generation "
                            "(no events and rank-0 process is gone)"
                        )
                    continue  # loop top re-checks the no-progress deadline
                if ev.get("event") == "token":
                    last_progress = time.monotonic()
                    seen_token = True
                yield ev
                if ev.get("event") == "done":
                    return
        finally:
            self._listeners.pop(req_id, None)
            self._release_busy()

    async def cancel(self, request_id: str) -> int:
        """Broadcast a hard-cancel command for `request_id` to every rank.

        The runner's reader thread intercepts `{"cmd":"cancel"}` and adds
        the id to the shared `_cancelled_ids` set. The active gen loop
        breaks at the next token boundary (legacy) or the slot is removed
        from BatchGenerator on the next tick (batched). Either way, the
        gen stops billing compute — unlike the old soft-cancel which
        merely stopped the API from forwarding tokens.

        Returns the number of ranks the cancel was actually sent to."""
        if not request_id:
            return 0
        sent = 0
        cmd = {"cmd": "cancel", "id": request_id}
        async with self.broadcast_lock:
            for r in self.runners:
                try:
                    r.send(cmd)
                    sent += 1
                except Exception as e:
                    sys.stderr.write(
                        f"[pool {self.cluster}] cancel→rank {r.node.get('rank','?')} "
                        f"failed: {e}\n"
                    )
        return sent

    async def prewarm(self, text: str, kv_q8: Optional[bool] = None,
                      timeout_s: float = 600.0) -> dict:
        """Prefill a shared cross-session prefix cache on every rank.

        All ranks must execute the distributed forward together, but only rank
        0 emits the result event back to us. We wait up to `timeout_s` for it.
        Empty text clears the prewarm.

        Returns the per-event payload from rank 0:
          {ok: bool, result: {tokens, bytes, elapsed_s, model_id} | null}
        """
        req_id = uuid.uuid4().hex[:8]
        q: asyncio.Queue = asyncio.Queue()
        self._listeners[req_id] = q
        req: dict = {"cmd": "prewarm", "id": req_id, "text": text}
        if kv_q8 is not None:
            req["kv_q8"] = bool(kv_q8)
        await self._acquire_busy()
        try:
            async with self.broadcast_lock:
                for r in self.runners:
                    r.send(req)
            try:
                ev = await asyncio.wait_for(q.get(), timeout=timeout_s)
            except asyncio.TimeoutError:
                return {"ok": False, "error": f"prewarm timeout after {timeout_s}s"}
            return {"ok": bool(ev.get("ok")), "result": ev.get("result")}
        finally:
            self._listeners.pop(req_id, None)
            self._release_busy()

    async def keepalive(self, timeout_s: float = 20.0) -> dict:
        """WU2 — exercise the JACCL group with a tiny all_sum to (a) keep the
        QPT_UC connections warm and (b) surface a dead/hung peer EARLY, while
        the orchestrator can still drive a controlled recovery (vs the silent
        ~45h crash that leaves a wired-memory orphan).

        Caller MUST hold the maintenance claim (`_try_claim_maintenance`) so no
        gen is queued ahead of (or behind) this collective — otherwise the
        all_sum could sit behind a long gen and time out on a healthy pool.

        Returns {ok, rtt_ms} or {ok: False, error, reason}. `reason` is:
          - 'partial_broadcast' : the keepalive reached only some ranks → the
            ones it reached will block in all_sum → treat as a peer failure.
          - 'timeout'           : no rank-0 ack within timeout → peer hung/dead.
        Single-node pools (size 1) short-circuit to ok (no group to probe)."""
        if len(self.runners) <= 1:
            return {"ok": True, "rtt_ms": 0.0, "reason": "single_node"}
        req_id = uuid.uuid4().hex[:8]
        q: asyncio.Queue = asyncio.Queue()
        self._listeners[req_id] = q
        req = {"cmd": "keepalive", "id": req_id}
        t0 = time.monotonic()
        try:
            # Fail-fast broadcast: r.send swallows BrokenPipe and returns False
            # on a dead stdin. A PARTIAL broadcast desyncs the collective (the
            # ranks that got it block in all_sum), so we must NOT then await an
            # ack that can't come — report partial_broadcast immediately.
            async with self.broadcast_lock:
                sent = sum(1 for r in self.runners if r.send(req))
            if sent != len(self.runners):
                return {"ok": False, "error": f"keepalive reached {sent}/{len(self.runners)} ranks",
                        "reason": "partial_broadcast"}
            try:
                ev = await asyncio.wait_for(q.get(), timeout=timeout_s)
            except asyncio.TimeoutError:
                return {"ok": False, "error": f"keepalive timeout after {timeout_s}s",
                        "reason": "timeout"}
            rtt_ms = (time.monotonic() - t0) * 1000.0
            self.last_keepalive_ok_at = time.time()
            return {"ok": bool(ev.get("ok", True)), "rtt_ms": round(rtt_ms, 1)}
        finally:
            self._listeners.pop(req_id, None)


# ──────────────────────────────────────────────────────────────────────────────
# State + metrics
# ──────────────────────────────────────────────────────────────────────────────
def _state_path_for_cluster(cluster: str) -> Path:
    """Where save_state writes the legacy v1 state file. Nautilus keeps its
    historical filename for back-compat; every other cluster uses the
    standard state-<id>.json convention."""
    if cluster == "nautilus":
        return STATE_FILE
    return state_file_for(cluster)


def save_state(model: str, mode: str, use_ap: bool, nodes_count: int,
               cluster: str = "nautilus", kv_q8: bool = False) -> None:
    path = _state_path_for_cluster(cluster)
    try:
        path.write_text(json.dumps({
            "model": model, "mode": mode, "use_ap": use_ap, "nodes": nodes_count,
            "cluster": cluster, "kv_q8": kv_q8,
        }))
    except Exception as e:
        sys.stderr.write(f"[api] failed to save {cluster} state: {e}\n")


def load_state(cluster: str = "nautilus") -> Optional[dict]:
    path = _state_path_for_cluster(cluster)
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────
# Default state-file v2 — multi-pool persistence (2026-05-19)
#
# v1 shape: {"model": …, "mode": …, "use_ap": …, "nodes": N, "kv_q8": …}
# v2 shape: {"schema":"v2", "pools":[{"alias":…, "model":…, "node_indices":…, …}, …]}
#
# Migration is automatic on read: v1 files are treated as one entry with
# alias='default' and node_indices=[0..N-1]. v2 files preserve every pool.
#
# The save side always writes v2 — the schema marker lets older API
# versions reading the same file fall through to its except clause and
# treat it as 'no state' (acceptable degradation when downgrading).
# ─────────────────────────────────────────────────────────────────────────


def save_cluster_state_v2(cluster_id: str) -> None:
    """Persist EVERY loaded pool for this cluster. Called after any
    successful load/unload/reset so the next container startup can restore
    the full multi-pool topology."""
    pools_payload = []
    for alias, pool in list_pools(cluster_id):
        if getattr(pool, "node_indices", None):
            indices = list(pool.node_indices)
        else:
            indices = []
            for n in pool.nodes:
                i = _host_to_index(cluster_id, n.get("host"))
                if i is not None:
                    indices.append(i)
        pools_payload.append({
            "alias": alias,
            "model": pool.model,
            "mode": pool.mode,
            "use_ap": pool.use_ap,
            "nodes": pool.nodes_count,
            "node_indices": indices,
            "kv_q8": pool.kv_q8,
            "draft_model": pool.draft_model,
            "num_draft_tokens": pool.num_draft_tokens,
        })
    payload = {"schema": "v2", "pools": pools_payload}
    sf = state_file_for(cluster_id)
    try:
        if not pools_payload:
            if sf.exists():
                sf.unlink()
            return
        sf.write_text(json.dumps(payload, indent=2))
    except Exception as e:
        sys.stderr.write(f"[api] failed to save {cluster_id} state v2: {e}\n")


def load_cluster_state_v2(cluster_id: str) -> list[dict]:
    """Read this cluster's state file. Returns a list of pool dicts ready to
    feed restore code. Empty list = nothing to restore. Handles both v2
    (native multi-pool) and v1 (single pool, migrated on the fly)."""
    sf = state_file_for(cluster_id)
    try:
        raw = json.loads(sf.read_text())
    except Exception:
        return []
    if isinstance(raw, dict) and raw.get("schema") == "v2":
        out = []
        for p in raw.get("pools") or []:
            if not isinstance(p, dict) or not p.get("model"):
                continue
            out.append(p)
        return out
    # v1 fallback — single default-alias pool, contiguous indices.
    if isinstance(raw, dict) and raw.get("model"):
        n = int(raw.get("nodes") or 1)
        return [{
            "alias": DEFAULT_ALIAS,
            "model": raw["model"],
            "mode": raw.get("mode", "pipeline"),
            "use_ap": bool(raw.get("use_ap", True)),
            "nodes": n,
            "node_indices": list(range(n)),
            "kv_q8": bool(raw.get("kv_q8", False)),
            "draft_model": raw.get("draft_model"),
            "num_draft_tokens": int(raw.get("num_draft_tokens") or 4),
        }]
    return []


# Ring buffer of recent request metrics (global — feeds the cross-cluster
# throughput stats). NOTE: a single high-volume cluster (e.g. the autocomplete
# service on tele-fast) can fill all 50 slots in minutes, evicting every other
# cluster's rows — which left "Recent activity" empty on every other Telemak
# card (#32). The per-cluster buffers below are what the dashboard panel reads.
_metrics: deque = deque(maxlen=50)

# Per-cluster recent metrics so a chatty cluster can't crowd the others out of
# the dashboard's "Recent activity" panel (#32). Keyed by cluster id, created
# on demand in record_metric().
_metrics_by_cluster: dict[str, deque] = {}

# Runner stderr line buffers + SSE subscribers, per cluster. Each line is
# {ts, cluster, rank, text}. Subscribers receive new lines via asyncio queues.
# The seed entries are the historical clusters; any custom cluster id (e.g.
# 'main' from topology.yaml, 'telemak-max64' from dashboard-added entries)
# is initialised on demand via _ensure_log_cluster() below — without that,
# /admin/logs?cluster=main 400s and the dashboard runner-logs panel stays
# empty even though stderr is flowing.
_log_buffers: dict[str, deque] = {
    "nautilus": deque(maxlen=500),
    "default": deque(maxlen=500),
}
_log_subscribers: dict[str, list[asyncio.Queue]] = {
    "nautilus": [], "default": [],
}
_log_lock = threading.Lock()


def _ensure_log_cluster(cluster: str) -> None:
    """Idempotent : init the per-cluster log buffer + subscriber list if
    we haven't seen this cluster before. Cheap (one dict lookup) so we
    call it both on the producer side (_push_log_line) and on the
    consumer side (admin_logs) so either path bootstraps the entries
    for new clusters."""
    if cluster in _log_buffers:
        return
    with _log_lock:
        if cluster not in _log_buffers:
            _log_buffers[cluster] = deque(maxlen=500)
            _log_subscribers[cluster] = []


def _push_log_line(cluster: str, rank: int, text: str) -> None:
    """Called from the per-rank stderr drain thread. Appends to the buffer +
    notifies any active SSE subscribers. Thread-safe."""
    _ensure_log_cluster(cluster)
    line = {"ts": time.time(), "cluster": cluster, "rank": rank, "text": text}
    with _log_lock:
        _log_buffers[cluster].append(line)
        subs = list(_log_subscribers[cluster])
    for q in subs:
        # Non-blocking; if the queue is full we drop (slow consumer).
        try:
            q.put_nowait(line)
        except Exception:
            pass

# Active prefix-cache sessions tracker (lightweight, single-user OK).
# Key: (cluster, session_id). Value: {last_seen, cumulative_tokens, cache_kind, model}.
# Sessions older than _SESSION_STALE_S are considered inactive (matches runner's
# 1 h TTL).
_SESSION_STALE_S = 3600.0
_active_sessions: dict[tuple[str, str], dict] = {}


def _touch_session(cluster: str, sess: dict, model: str) -> None:
    sid = sess.get("id")
    if not sid:
        return
    _active_sessions[(cluster, sid)] = {
        "session_id": sid,
        "cluster": cluster,
        "model": model,
        "cache_kind": sess.get("cache_kind"),
        "cumulative_tokens": sess.get("cumulative_tokens", 0),
        "last_seen": time.time(),
    }


def _live_sessions() -> list[dict]:
    now = time.time()
    return [s for s in _active_sessions.values()
            if now - s.get("last_seen", 0) < _SESSION_STALE_S]


def record_metric(client: str, ntoks: int, elapsed_s: float, ttft_s: Optional[float],
                  prompt_chars: int, model: str,
                  cluster: Optional[str] = None,
                  tool_calls: int = 0,
                  session_kind: Optional[str] = None,
                  status: str = "completed") -> None:
    row = {
        "ts": time.time(),
        "client": client,
        "cluster": cluster,
        "model": model,
        "ntoks": ntoks,
        "elapsed_s": round(elapsed_s, 3),
        "ttft_s": round(ttft_s, 3) if ttft_s is not None else None,
        "tps": round(ntoks / elapsed_s, 2) if elapsed_s > 0 else 0.0,
        "prompt_chars": prompt_chars,
        "tool_calls": tool_calls,
        "session_kind": session_kind,
        "status": status,
    }
    _metrics.appendleft(row)
    # Also retain per-cluster so the dashboard can show each cluster's own
    # recent activity regardless of a noisy neighbour's volume (#32).
    if cluster:
        dq = _metrics_by_cluster.get(cluster)
        if dq is None:
            dq = _metrics_by_cluster[cluster] = deque(maxlen=100)
        dq.appendleft(row)


# ──────────────────────────────────────────────────────────────────────────────
# Active runs tracking (in-flight inference visibility + soft cancel)
# ──────────────────────────────────────────────────────────────────────────────
# Tracks every inference that's currently being generated so the dashboard can
# show progress + offer cancel. Soft cancel: setting the event tells the
# streaming loop to stop forwarding tokens; the underlying runner.py keeps
# generating until completion (true hard-cancel needs runner support).
_active_runs: dict[str, dict] = {}
_active_run_cancels: dict[str, asyncio.Event] = {}


# ──────────────────────────────────────────────────────────────────────────────
# <think>…</think> stream filter (#enable_thinking=False compat)
#
# Some chat templates ignore the `enable_thinking` Jinja variable and the
# model emits `<think>…</think>` blocks anyway — MiniMax M2.7 is the
# documented case (2026-05-18). Per-stream state machine that:
#
#   • routes content inside `<think>…</think>` to delta.reasoning_content
#     (Companion + most OpenAI-compatible clients pick this up as the
#     reasoning channel, displayed collapsed),
#   • routes everything else to delta.content as before,
#   • swallows the `<think>` / `</think>` markers themselves.
#
# Markers may straddle chunk boundaries (model emits a few tokens per
# event), so the function carries up to `len(longest_marker)-1` chars
# forward to next invocation. The carry is the LATEST tail that could
# still be a partial marker prefix — never a confirmed marker.
#
# When `enable_thinking=True` (user asked for thinking visible), we
# bypass the filter and pass content through unchanged, since the client
# explicitly wanted to see the markers.
# ──────────────────────────────────────────────────────────────────────────────
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_MAX_PARTIAL = max(len(_THINK_OPEN), len(_THINK_CLOSE)) - 1  # 7

# Per-model think markers. Most reasoning models wrap in <think>…</think>;
# MiniMax-M3 uses its own <mm:think>…</mm:think> pair (token 200059/200060).
# Without the right pair, the splitter seeds in_think=True, hunts for the
# wrong close marker forever, and traps the WHOLE answer in reasoning_content
# (content=null) — exactly M3's first-smoke symptom (2026-06-13).
_THINK_MARKERS = {
    "minimax-m3": ("<mm:think>", "</mm:think>"),
}


def _model_think_markers(model_id: Optional[str]) -> tuple[str, str]:
    needle = (model_id or "").lower()
    for key, pair in _THINK_MARKERS.items():
        if key in needle:
            return pair
    return _THINK_OPEN, _THINK_CLOSE


def _seed_in_think(model_id: Optional[str], enable_thinking) -> bool:
    """Should a filtered stream START already inside a think block?

    True for templates that PREFILL the open tag into the prompt (the model
    emits only the body + close): the classic auto-open family (M2, Qwen3.5/6,
    Step-3.7). M3 only prefills `<mm:think>` when thinking is EXPLICITLY
    enabled; in adaptive mode it emits its own open tag, so seeding True would
    eat the literal tag — seed False there and let the filter catch it.
    """
    if "minimax-m3" in (model_id or "").lower():
        return enable_thinking is True
    return True

# Models whose chat_template auto-prefills `<think>\n` at the end of the
# prompt, leaving the model to emit just the reasoning body and a closing
# `</think>` tag. Without compensation, our filter never sees the open
# marker, stays in `in_think=False`, and lets the whole think body land in
# delta.content — exactly what the user wants to suppress.
#
# When the model_id matches one of these substrings (case-insensitive),
# the per-stream state starts with `in_think=True`. The first emitted
# `</think>` then transitions to visible-content mode as expected, and
# all reasoning before it gets routed to delta.reasoning_content.
#
# MiniMax M2's official docs (2026-05-20 update) explicitly say:
#   "The model's reasoning is wrapped in <think> tags within the
#    content field. Do not modify the content field."
# i.e. their stance is "don't disable thinking" — we route around them
# by treating the auto-opened block as reasoning and stripping the close.
# GLM-5.2 (glm_moe_dsa): template auto-opens <think> ; HONORS enable_thinking
# (off -> empty <think></think> baked in the prompt, no output block), like
# Qwen3.5/3.6 -> goes HERE only, NOT in _MODELS_IGNORE_ENABLE_THINKING_FLAG.
# Substring "glm-5.2" matches the concrete HF path (kernelpool/GLM-5.2-*, all quants).
_MODELS_AUTO_OPEN_THINK = ("minimax", "qwen3.5", "qwen3.6", "step-3.7", "step3p7", "glm-5.2")
# Subset of _MODELS_AUTO_OPEN_THINK that IGNORES the `enable_thinking`
# kwarg and always wraps reasoning in <think>...</think>. Per MiniMax M2
# docs (2026-05-20 update): "The model's reasoning is wrapped in <think>
# tags within the content field. Do not modify the content field."
# Qwen3.5 and Qwen3.6 honor the flag — when Companion sends
# enable_thinking=false they actually stop thinking, no filter needed.
# MiniMax-M2 doesn't honor it, so we MUST keep the filter on for it even
# when callers ask for no-thinking, otherwise the reasoning text leaks
# into `content` verbatim (`</think>` literal visible to the user).
# minimax-m3 added 2026-07-02: the mlx-vlm VL serving path (m3vl) does NOT
# honor enable_thinking=false — the model keeps thinking in adaptive mode and
# leaks a raw <mm:think>...</mm:think> block into content. (The text M3 on the
# jaccl runner DOES honor it, but the shared filter must cover the worst case.)
# Keeping the filter ON when off is safe against ghosting: _seed_in_think()
# returns False for M3 when thinking is off, so we never seed in_think and only
# strip a block IF the model actually emits the tags; a genuine no-think answer
# flows through untouched.
_MODELS_IGNORE_ENABLE_THINKING_FLAG = ("minimax-m2", "minimax-m3", "step-3.7", "step3p7")

# Models whose chat template reads a `reasoning_effort` system directive
# (OpenAI o-series convention: minimal/low/medium/high). Step-3.7-Flash is a
# reasoning-first "agent" model that ALWAYS opens a <think> block — there is
# no enable_thinking off-switch — but it honors a "Reasoning: <effort>" line
# injected into the system prompt. Without it the model defaults to heavy
# reasoning (Sophie, 2026-05-31: "énorme thinking sans qu'on puisse le mettre
# en false"). So default these to "minimal" when the caller didn't specify an
# effort, keeping them fast/agentic by default while still overridable per
# request (Companion Inference settings → reasoning_effort).
_MODELS_REASONING_EFFORT_DEFAULT = {
    "step-3.7": "minimal",
    "step3p7": "minimal",
}


def _model_auto_opens_think(model_id: Optional[str]) -> bool:
    if not model_id:
        return False
    needle = model_id.lower()
    return any(key in needle for key in _MODELS_AUTO_OPEN_THINK)


def _model_ignores_enable_thinking_flag(model_id: Optional[str]) -> bool:
    if not model_id:
        return False
    needle = model_id.lower()
    return any(key in needle for key in _MODELS_IGNORE_ENABLE_THINKING_FLAG)


def _should_filter_think(model_id: Optional[str], enable_thinking) -> bool:
    """Whether to split a <think> block off the wire into reasoning_content.

    Single source of truth shared by the local-pool path AND the Telemak
    proxy path so a model behaves identically however it's served (Sophie,
    2026-05-31: Step-3.7 dérouled the think on Argo-local while the proxy
    filtered it). Filter — and seed in_think=True, since these templates
    auto-OPEN the block with no leading tag — whenever the model auto-opens
    think:

      * thinking ON  → route the reasoning into the collapsed channel
        instead of letting it déroule into the answer.
      * thinking OFF but the model IGNORES the flag (MiniMax, Step-3.7) →
        it emits the block anyway, so keep filtering or `</think>` leaks
        into `content`.

    The ONLY no-filter case is a model that HONORS the flag with thinking
    explicitly off (Qwen3.5/3.6): it emits no block, so seeding in_think
    would trap its direct answer in reasoning_content → empty content →
    Companion ghost (observed 2026-05-30 on Qwen3.5-397B).
    """
    if enable_thinking is False and not _model_ignores_enable_thinking_flag(model_id):
        return False
    return _model_auto_opens_think(model_id)


def _default_reasoning_effort(model_id: Optional[str]) -> Optional[str]:
    """Per-model default `reasoning_effort` when the caller didn't pass one.

    Keeps reasoning-first models (Step-3.7) fast by default instead of running
    their heavy built-in default. Returns None for models that don't read the
    directive — passing it to them would just inject an unused system line.
    An explicit caller value always wins over this; this only fills the gap.
    """
    if not model_id:
        return None
    needle = model_id.lower()
    for key, eff in _MODELS_REASONING_EFFORT_DEFAULT.items():
        if key in needle:
            return eff
    return None


def _split_think_stream(text: str, state: dict) -> tuple[str, str]:
    """Take one chunk of streamed text, return (visible, reasoning).

    `state` is a mutable dict caller maintains between chunks. Initialize as
    `{"in_think": False, "carry": ""}` for a new stream. The function mutates
    it in place. On stream end, caller should flush state["carry"] into
    whichever bucket matches state["in_think"] — `_flush_think_stream` does
    that.
    """
    open_m = state.get("open", _THINK_OPEN)
    close_m = state.get("close", _THINK_CLOSE)
    max_partial = max(len(open_m), len(close_m)) - 1
    text = state.get("carry", "") + (text or "")
    state["carry"] = ""
    visible_parts: list[str] = []
    reasoning_parts: list[str] = []

    while text:
        if state.get("in_think"):
            idx = text.find(close_m)
            if idx == -1:
                # No close marker — emit body up to the last few chars that
                # could be the start of the close tag. Hold those back.
                tail_len = min(len(text), max_partial)
                if len(text) > tail_len:
                    reasoning_parts.append(text[:-tail_len])
                state["carry"] = text[-tail_len:] if tail_len else ""
                text = ""
            else:
                reasoning_parts.append(text[:idx])
                text = text[idx + len(close_m):]
                state["in_think"] = False
        else:
            idx = text.find(open_m)
            if idx == -1:
                tail_len = min(len(text), max_partial)
                if len(text) > tail_len:
                    visible_parts.append(text[:-tail_len])
                state["carry"] = text[-tail_len:] if tail_len else ""
                text = ""
            else:
                visible_parts.append(text[:idx])
                text = text[idx + len(open_m):]
                state["in_think"] = True

    return "".join(visible_parts), "".join(reasoning_parts)


def _flush_think_stream(state: dict) -> tuple[str, str]:
    """Flush any final carry — called once after the model's `done` event.
    Returns (visible, reasoning) for the residual."""
    carry = state.get("carry", "")
    state["carry"] = ""
    if not carry:
        return "", ""
    if state.get("in_think"):
        # Stream ended inside a think block — model didn't close it. Treat
        # remainder as reasoning so it isn't lost.
        return "", carry
    return carry, ""


def _runs_register(rid: str, *, model: str, cluster: str, client: str,
                   max_tokens: int, kind: str = "streaming",
                   pool_alias: Optional[str] = None) -> asyncio.Event:
    """Register a new in-flight run. Returns an asyncio.Event the caller can
    poll inside its streaming loop to detect cancellation.

    `pool_alias` lets the dashboard distinguish multi-pool Default runs: e.g.
    a request to alias `default-extra` carries `cluster='default'` AND
    `pool_alias='default-extra'`. Default alias = cluster name (back-compat
    for the single Nautilus pool and the default `default` pool)."""
    ev = asyncio.Event()
    _active_runs[rid] = {
        "id": rid,
        "request_id": rid,
        "model": model,
        "cluster": cluster,
        "pool_alias": pool_alias or cluster,
        "client": client,
        "started_at": time.time(),
        "max_tokens": max_tokens,
        "output_tokens": 0,
        "tok_per_s": 0.0,
        "elapsed_s": 0.0,
        "status": kind,
    }
    _active_run_cancels[rid] = ev
    _persist.persist_run(rid, _active_runs[rid], force=True)
    return ev


def _runs_tick(rid: str, text: str = "") -> None:
    """Update an active run's progress. We approximate tokens as len(text)/4
    when the runner emits text chunks; switch to exact counting once
    runner.py exposes per-event ntoks."""
    r = _active_runs.get(rid)
    if not r:
        return
    delta = max(1, len(text) // 4) if text else 1
    r["output_tokens"] += delta
    elapsed = time.time() - r["started_at"]
    r["elapsed_s"] = round(elapsed, 2)
    if elapsed > 0:
        r["tok_per_s"] = round(r["output_tokens"] / elapsed, 2)
    _persist.persist_run(rid, r)  # throttled to 1Hz inside


def _runs_finalize(rid: str, *, status: str = "done") -> None:
    r = _active_runs.get(rid)
    if r:
        r = dict(r)
        r["status"] = status
        r["finished_at"] = time.time()
        _persist.finalize_run(rid, r)
    _active_runs.pop(rid, None)
    _active_run_cancels.pop(rid, None)


# Maximum time a run is allowed to sit in 'cancelling' before we
# force-finalize it. After cancel-all + broadcast, a healthy runner
# ACKs within a couple seconds and the normal _runs_finalize path fires.
# When the runner is SIGKILL'd, deadlocked in prefill, or the node has
# rebooted, the ACK never comes and the run rotted in _active_runs
# forever. 30s is plenty for a real cancel ACK and short enough that
# the dashboard "Active runs" pane reflects truth.
CANCELLING_FINALIZE_GRACE_S = 30.0


def _finalize_stuck_cancelling() -> int:
    """Reap any run that's been 'cancelling' for more than the grace
    period. Returns the count reaped. Called from a 10s background
    sweeper so the dashboard never shows ghost CANCELLING entries.
    Idempotent — runs that finalize legitimately via the ACK path
    won't be in _active_runs anymore by the time we look."""
    now = time.time()
    reaped: list[str] = []
    for rid, r in list(_active_runs.items()):
        if r.get("status") != "cancelling":
            continue
        # Use cancel_started_at when present, else fall back to started_at.
        # The Event was set in admin_run_cancel/admin_runs_cancel_all, so
        # the run has been 'cancelling' from at least that moment.
        cancelled_at = r.get("cancelled_at") or r.get("started_at") or now
        if (now - cancelled_at) < CANCELLING_FINALIZE_GRACE_S:
            continue
        reaped.append(rid)
    for rid in reaped:
        _runs_finalize(rid, status="cancelled")
    if reaped:
        sys.stderr.write(
            f"[cancel-sweeper] force-finalized {len(reaped)} stuck cancelling "
            f"run(s): {', '.join(reaped)}\n"
        )
    return len(reaped)


def _runs_is_cancelled(rid: str) -> bool:
    ev = _active_run_cancels.get(rid)
    return bool(ev and ev.is_set())


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible models
# ──────────────────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    # OpenAI vision spec: content is either a plain string OR a list of
    # content parts (text + image_url + tool_result + etc.). Vision-capable
    # clients send the list form. We pass it through unchanged for cloud
    # proxy routing; the local distributed runner accepts only the string
    # form today.
    content: Optional[Union[str, list[dict]]] = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = Field(default=512)
    # OpenAI's newer canonical output cap. Modern OpenAI-compatible clients
    # (e.g. graphify) send max_completion_tokens instead of max_tokens; without
    # this alias they get silently truncated to the max_tokens default (512).
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False
    enable_thinking: Optional[bool] = None
    # OpenAI o-series reasoning dial (minimal/low/medium/high). Forwarded to
    # the chat template as a "Reasoning: <effort>" system directive for models
    # that read it (Step-3.7). None → per-model default (_default_reasoning_effort).
    reasoning_effort: Optional[str] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Any] = None
    session_id: Optional[str] = None  # opt-in prefix-cache key (also: X-Session-Id header)

    @model_validator(mode="after")
    def _alias_max_completion_tokens(self) -> "ChatCompletionRequest":
        # max_completion_tokens (newer OpenAI param) wins when explicitly set.
        if self.max_completion_tokens is not None:
            self.max_tokens = self.max_completion_tokens
        return self


def _now() -> int:
    return int(time.time())


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    if len(messages) == 1 and messages[0].role == "user":
        return messages[0].content
    parts = [f"[{m.role}] {m.content}" for m in messages]
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Admin: discover models on the cluster (rank 0)
# ──────────────────────────────────────────────────────────────────────────────
def discover_models_on_node(ssh: str, models_dir: str) -> list[str]:
    """SSH into a node and list model directories under `models_dir`.

    Two layouts supported (so we can migrate without breaking):
      - flat:    `<models_dir>/<org>--<repo>/config.json`     (legacy)
      - 2-level: `<models_dir>/<org>/<repo>/config.json`      (preferred — mirrors HF / Inferencer)

    Any directory containing a `config.json` is a candidate model. The
    bash probe walks one level for flat, two for nested, accepts both
    in the same run. Returns absolute paths usable as RUNNER_MODEL —
    callers can derive a display name via the relative path under
    `models_dir`.

    !! We force `bash -c` because the default shell on the remote node
    may be zsh (macOS default since Catalina). zsh's nomatch option
    aborts the entire script when an inner glob expands to nothing
    (e.g. an empty org dir like `mistralai/`), which silently drops
    every subsequent org folder. Bash treats empty globs as the
    literal pattern, the `-d` test fails, and we continue. Discovered
    2026-05-28 — picker was showing partial lists.
    """
    inner = (
        f"for d in {shlex.quote(models_dir)}/*; do "
        "  [ -d \"$d\" ] || continue ; "
        # Flat layout: config.json sits directly under <d>.
        "  if [ -f \"$d/config.json\" ]; then "
        "    echo P:$d ; continue ; "
        "  fi ; "
        # 2-level: <d> is the org folder; look one level deeper.
        "  for sub in \"$d\"/*; do "
        "    [ -d \"$sub\" ] || continue ; "
        "    [ -f \"$sub/config.json\" ] || continue ; "
        "    echo P:$sub ; "
        "  done ; "
        "done 2>/dev/null"
    )
    cmd = f"bash -c {shlex.quote(inner)}"
    try:
        out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", ssh, cmd],
            capture_output=True, text=True, timeout=15,
        )
        models: list[str] = []
        for line in out.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            _, _, value = line.partition(":")
            if value:
                models.append(value)
        return sorted(set(models))
    except Exception as e:
        sys.stderr.write(f"[api] discover failed on {ssh}: {e}\n")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan + global state
# ──────────────────────────────────────────────────────────────────────────────
_pool: Optional[RunnerPool] = None

# Generic pool registry: cluster_id → alias → RunnerPool.
#
# A cluster may host multiple concurrent pools, each occupying a disjoint
# subset of its nodes (enforced at load time). The unique-pool case uses
# alias = DEFAULT_ALIAS; extra pools get operator-chosen aliases.
_pools: dict[str, dict[str, RunnerPool]] = {}
_admin_locks: dict[str, asyncio.Lock] = {}
DEFAULT_ALIAS = "default"


def get_pool(cluster_id: str, alias: str = DEFAULT_ALIAS) -> Optional[RunnerPool]:
    return _pools.get(cluster_id, {}).get(alias)


def list_pools(cluster_id: str) -> list[tuple[str, RunnerPool]]:
    """All loaded pools for one cluster, default alias first when present."""
    d = _pools.get(cluster_id, {})
    out: list[tuple[str, RunnerPool]] = []
    if DEFAULT_ALIAS in d:
        out.append((DEFAULT_ALIAS, d[DEFAULT_ALIAS]))
    for alias, p in d.items():
        if alias == DEFAULT_ALIAS:
            continue
        out.append((alias, p))
    return out


def list_all_pools() -> list[tuple[str, str, RunnerPool]]:
    """Every loaded pool across every cluster. (cluster_id, alias, pool)."""
    return [(cid, a, p) for cid in _pools for a, p in list_pools(cid)]


def set_pool(cluster_id: str, alias: str, pool: RunnerPool) -> None:
    _pools.setdefault(cluster_id, {})[alias] = pool


def del_pool(cluster_id: str, alias: str) -> None:
    bucket = _pools.get(cluster_id)
    if not bucket:
        return
    bucket.pop(alias, None)
    if not bucket:
        _pools.pop(cluster_id, None)


def pool_aliases(cluster_id: str) -> list[str]:
    return [a for a, _ in list_pools(cluster_id)]


def nodes_in_use(cluster_id: str) -> set[int]:
    """Union of node indices occupied by any loaded pool on this cluster."""
    used: set[int] = set()
    for _, pool in list_pools(cluster_id):
        for node in pool.nodes:
            host = node.get("host")
            if host is None:
                continue
            idx = _host_to_index(cluster_id, host)
            if idx is not None:
                used.add(idx)
    return used


def _host_to_index(cluster_id: str, host: str) -> Optional[int]:
    """Map a host id to its index in the cluster def's node list."""
    cd = get_cluster_def(cluster_id)
    for i, n in enumerate(cd.get("nodes") or []):
        if n.get("host") == host:
            return i
    return None


def get_admin_lock(cluster_id: str) -> asyncio.Lock:
    if cluster_id not in _admin_locks:
        _admin_locks[cluster_id] = asyncio.Lock()
    return _admin_locks[cluster_id]


_args: Optional[argparse.Namespace] = None
_admin_lock = asyncio.Lock()

# ──────────────────────────────────────────────────────────────────────────────
# Cluster-level degraded state (recovery ladder)
#
# When we detect a JACCL queue-pair failure, a residual wired-memory leak
# after sweep, or a rank crash with a known RDMA errno, we set the cluster
# as `degraded`. Subsequent loads on that cluster are refused (HTTP 409)
# until the operator explicitly calls /admin/{cluster}/reset — which runs
# the full recovery ladder (cancel → stop → sweep → clear sessions →
# verify wired → clear degraded).
#
# This implements the audit recommendation: "ne pas recharger en boucle
# un pool sale". Before this, a load that crashed JACCL would silently
# accept a retry that crashed on the same QP — and so on, accumulating
# zombies on the macs until the operator rebooted.
# ──────────────────────────────────────────────────────────────────────────────
_cluster_degraded: dict[str, dict] = {}  # cluster_id → {reason, at, details}

# Regex / substrings that indicate a JACCL / RDMA-level failure rather
# than a Python-level error. Used to decide whether to flip the degraded
# bit when a rank dies.
_JACCL_ERROR_PATTERNS = (
    "jaccl",
    "Changing queue pair to RTR",
    "errno=2",
    "errno=16",
    "errno=96",
    "ibv_",
    "rdma",
)


def _looks_like_jaccl_error(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(p.lower() in low for p in _JACCL_ERROR_PATTERNS)


def _mark_cluster_degraded(cluster_id: str, reason: str,
                           details: Optional[dict] = None) -> None:
    """Flip a cluster to degraded. Idempotent — repeated calls update the
    reason / details but preserve the original timestamp."""
    existing = _cluster_degraded.get(cluster_id) or {}
    _cluster_degraded[cluster_id] = {
        "reason": reason,
        "at": existing.get("at") or time.time(),
        "details": details or existing.get("details"),
    }
    sys.stderr.write(f"[degraded] {cluster_id} → {reason}\n")


def _cluster_is_degraded(cluster_id: str) -> bool:
    return cluster_id in _cluster_degraded


def _clear_cluster_degraded(cluster_id: str) -> None:
    if cluster_id in _cluster_degraded:
        del _cluster_degraded[cluster_id]
        sys.stderr.write(f"[degraded] {cluster_id} → cleared\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    # SQLite history for runs + sync_jobs. Failure-tolerant: init returns
    # False on any error and all _persist.* calls become no-ops.
    try:
        PERSIST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    _persist.init_db(str(PERSIST_DB_PATH))
    # Any sync_jobs left in 'running' state are orphaned from a previous
    # container life — mark them as interrupted so the UI shows truth.
    _persist.mark_orphans_interrupted()
    # Trim runs history to a reasonable upper bound (keep last 5000).
    _persist.prune_runs(keep=5000)
    # Reap any runner.py left running on cluster nodes from a previous container
    # life. Without this, JACCL re-init fails (errno 16, queue pair busy) and
    # ~200 GB stays wired per zombie node until reboot. The sweep is idempotent
    # and cheap (~1s/node when nothing to kill).
    # Run the per-cluster orphan sweeps OFF the event loop and in parallel:
    # each does blocking SSH (up to ~15s/node), so calling them synchronously
    # here stalls startup (and /health) for N*timeout when a node is down (#24).
    _sweep_cids = list(DEFAULT_CLUSTER_DEFS.keys())
    _sweep_results = await asyncio.gather(
        *[asyncio.to_thread(_sweep_orphan_runners, _cid) for _cid in _sweep_cids],
        return_exceptions=True,
    )
    for _cid, _res in zip(_sweep_cids, _sweep_results):
        if isinstance(_res, Exception):
            sys.stderr.write(f"[api] orphan sweep ({_cid}) failed: {_res}\n")
    initial = _initial_config()
    if initial is not None:
        try:
            _pool = RunnerPool(**initial)
            await _pool.start()
        except Exception as e:
            sys.stderr.write(f"[api] startup load (nautilus) failed: {e}\n")
            _pool = None
    # Multi-pool restore (state file v2): iterate every cluster declared in
    # topology.yaml, restore whatever pools were running at last shutdown.
    # Within a cluster, sort the default alias first so any extra-alias load
    # that depended on it sees a clean baseline.
    for cid in DEFAULT_CLUSTER_DEFS.keys():
        saved_pools = load_cluster_state_v2(cid)
        if not saved_pools:
            continue
        saved_pools.sort(key=lambda p: 0 if p.get("alias") == DEFAULT_ALIAS else 1)
        for entry in saved_pools:
            alias = entry.get("alias", DEFAULT_ALIAS)
            try:
                indices = entry.get("node_indices") or list(range(int(entry.get("nodes") or 1)))
                pool = RunnerPool(
                    model=entry["model"],
                    mode=entry.get("mode", "pipeline"),
                    use_ap=bool(entry.get("use_ap", True)),
                    nodes_count=len(indices),
                    cluster=cid,
                    kv_q8=bool(entry.get("kv_q8", False)),
                    draft_model=entry.get("draft_model"),
                    num_draft_tokens=int(entry.get("num_draft_tokens") or 4),
                    alias=alias,
                    node_indices=indices,
                )
                await pool.start()
                set_pool(cid, alias, pool)
            except Exception as e:
                sys.stderr.write(f"[api] startup load ({cid}:{alias}) failed: {e}\n")
    # Apply persisted default TTL to any pool that just started up.
    _apply_default_ttl_to_pools()
    # Background TTL sweeper — auto-unloads pools idle for > ttl_seconds.
    # Frees nodes for other pools without manual ops.
    ttl_task = asyncio.create_task(_pool_ttl_sweeper())
    # Background dead-pool sweeper — drops pools whose runners have all
    # died (OOM, panic, node reboot) so the dashboard reflects reality
    # before the operator hits the load button. See #5.
    dead_task = asyncio.create_task(_dead_pool_sweeper())
    # Background cancelling-sweeper — force-finalises runs stuck
    # in 'cancelling' past the grace period (runner never ACK'd
    # because it was kill -9'd or the node rebooted). Without this,
    # ghost runs sit in /admin/runs forever.
    cancel_task = asyncio.create_task(_cancelling_sweeper())
    # #40 — JACCL stability loop: keepalive health (WU2) → controlled recovery
    # (WU3), plus age-based preventive reload (WU1) for long-running jaccl pools.
    jaccl_task = asyncio.create_task(_jaccl_stability_loop())
    try:
        yield
    finally:
        ttl_task.cancel()
        dead_task.cancel()
        cancel_task.cancel()
        jaccl_task.cancel()
        # Await the cancelled sweepers before tearing pools down — otherwise a
        # sweeper mid-operation (e.g. holding a cluster admin lock) is abandoned
        # and the subsequent pool.stop() races its half-finished state (#24).
        await asyncio.gather(ttl_task, dead_task, cancel_task, return_exceptions=True)
        if _pool is not None:
            await _pool.stop()
        for _cid, _alias, pool in list_all_pools():
            try:
                await pool.stop()
            except Exception:
                pass


def _apply_default_ttl_to_pools() -> None:
    """Set `pool.ttl_seconds` from the persisted default on each live pool.
    Called at startup and after admin settings updates."""
    cfg = _load_cluster_config()
    ttl = int((cfg.get("settings") or {}).get("pool_ttl_seconds_default") or 0)
    if _pool is not None:
        _pool.ttl_seconds = ttl
    for _cid, _alias, pool in list_all_pools():
        pool.ttl_seconds = ttl


async def _dead_pool_sweeper() -> None:
    """Every 30s, scan every cluster for pools whose runners have all
    died and drop them. Same purge as the at-load path (#5 issue), just
    pro-active so the dashboard reflects reality without the operator
    having to attempt a load first.

    Without this, an OOM crash or node reboot leaves the pool registry
    "loaded:true" on the dashboard while every rank process is gone.
    Operator sees a healthy pool tile, clicks chat, and gets
    "model_not_loaded" because alive_count() == 0 at routing time. The
    sweeper closes that gap.

    Probe budget: just `proc.poll() is None`. No SSH, no TCP probe —
    those would add latency and risk false positives on a flaky link.
    A surviving runner is plenty robust under transient network
    hiccups; only a dead local SSH child triggers purge.
    """
    while True:
        try:
            await asyncio.sleep(30)
            seen_clusters = set()
            for cid, _alias, _pool in list_all_pools():
                seen_clusters.add(cid)
            for cid in seen_clusters:
                purged = await _purge_dead_pools(cid)
                if purged:
                    sys.stderr.write(
                        f"[dead-pool-sweeper] {cid}: purged {len(purged)} "
                        f"dead pool(s): {', '.join(purged)}\n"
                    )
        except asyncio.CancelledError:
            return
        except Exception as e:
            sys.stderr.write(f"[dead-pool-sweeper] error: {e}\n")


async def _cancelling_sweeper() -> None:
    """Every 10s, reap runs that have been 'cancelling' past the grace
    period (the runner never ACK'd — usually because it was SIGKILL'd
    or the node rebooted). Without this, ghost CANCELLING entries
    stay forever in /admin/runs and confuse the dashboard into
    thinking the cluster is busy. See _finalize_stuck_cancelling().
    """
    while True:
        try:
            await asyncio.sleep(10)
            _finalize_stuck_cancelling()
        except asyncio.CancelledError:
            return
        except Exception as e:
            sys.stderr.write(f"[cancel-sweeper] error: {e}\n")


async def _pool_ttl_sweeper() -> None:
    """Every 30s, scan local pools and auto-unload any that's been idle past
    its `ttl_seconds`. No-op when ttl_seconds=0 (the default — opt-in).

    Why this lives in OdyssAI-X: a 3-node Default costs 3 nodes' worth of RAM
    held for nothing during quiet hours. With ttl=1800 the cluster auto-
    yields so other pools can claim the RAM. operator keeps manual
    control via pinning (TODO) or via reloading explicitly.

    We never touch `_pool` (the legacy nautilus pool — unused since 2026-05).
    """
    while True:
        try:
            await asyncio.sleep(30)
            now = time.time()
            for cid, _alias, pool in list_all_pools():
                ttl = int(getattr(pool, "ttl_seconds", 0) or 0)
                if ttl <= 0:
                    continue
                idle = now - getattr(pool, "last_used_at", now)
                if idle >= ttl:
                    await _auto_unload_cluster(
                        cid, reason=f"idle {int(idle)}s ≥ ttl {ttl}s"
                    )
        except asyncio.CancelledError:
            return
        except Exception as e:
            sys.stderr.write(f"[ttl-sweeper] error: {e}\n")


# ──────────────────────────────────────────────────────────────────────────────
# #40 — JACCL stability: keepalive health (WU2) + preventive reload (WU1) +
# controlled recovery (WU3). The UNCONTROLLED QPT_UC death bounds its PROBABILITY
# (reset QPs before the window) and its BLAST RADIUS (detect a dead/hung peer in
# minutes → recover automatically, vs a 187GB-wired orphan sitting until a human
# notices). Tunables are env-overridable.
# CALIBRATION (2026-06-14): the original 30h preventive-reload window assumed a
# ~45h crash. Measured on M3 Q8 3-node under real load, the queue pairs degrade
# in ~4h IDLE (keepalive fully failed after ~4-5h of misses). 30h never fired
# before the death — Sophie kept hitting dead pools. Dropped to 3h so the
# (idle-gated, reload-not-reboot, pool-nodes-only) refresh runs before the
# degradation window, not after. The QPT death rate is load-dependent; 3h is the
# idle-decay margin, not a guarantee for a single very-long active generation.
# ──────────────────────────────────────────────────────────────────────────────
_JACCL_STABILITY_ENABLED = os.environ.get("JACCL_STABILITY_ENABLED", "1") == "1"
_JACCL_KEEPALIVE_INTERVAL_S = float(os.environ.get("JACCL_KEEPALIVE_INTERVAL_S", "90"))
_JACCL_KEEPALIVE_TIMEOUT_S = float(os.environ.get("JACCL_KEEPALIVE_TIMEOUT_S", "20"))
_JACCL_KEEPALIVE_FAIL_THRESHOLD = int(os.environ.get("JACCL_KEEPALIVE_FAIL_THRESHOLD", "2"))
_JACCL_PREVENTIVE_RELOAD_AGE_S = float(
    os.environ.get("JACCL_PREVENTIVE_RELOAD_AGE_S", str(3 * 3600)))
# WU3 auto-recovery (reboot the POOL'S nodes + reload on a keepalive-detected
# hang). Default ON since 2026-06-14: the recovery is now PER-NODE — only the
# wedged pool's own nodes reboot (never reboot-all), so it cannot touch a node
# outside this pool. A node is rebooted iff it belongs to this pool (.32 included
# on a 5-node pool, left alone otherwise). Set JACCL_AUTO_RECOVERY_ENABLED=0 for
# detect-only (logs the keepalive failure, takes no action).
_JACCL_AUTO_RECOVERY_ENABLED = os.environ.get("JACCL_AUTO_RECOVERY_ENABLED", "1") == "1"


def _jaccl_log(cluster_id: str, text: str) -> None:
    """Operator-visible line (dashboard runner-log panel) + stderr."""
    line = f"[jaccl-stability] {text}"
    try:
        _push_log_line(cluster_id, 0, line)
    except Exception:
        pass
    sys.stderr.write(line + "\n")


def _pool_reload_request(pool: RunnerPool) -> ArgoLoadRequest:
    """Snapshot EVERY live load param so a reload reproduces the exact pool, not
    ArgoLoadRequest defaults (Codex #8). force=True skips the preflight size
    check — the same model fit a moment ago."""
    return ArgoLoadRequest(
        model=pool.model, mode=pool.mode, use_ap=pool.use_ap,
        nodes=pool.nodes_count, kv_q8=pool.kv_q8,
        draft_model=pool.draft_model, num_draft_tokens=pool.num_draft_tokens,
        force=True, alias=pool.alias, node_indices=pool.node_indices,
    )


async def _preventive_reload(cluster_id: str, pool: RunnerPool) -> None:
    """WU1 — controlled unload+reload of ONE pool to reset its QPs before the
    ~45h window. Reuses admin_cluster_load (stop-old-start-new for the same
    alias) so registry/state/locks are handled and OTHER pools are untouched
    (Codex #7). Caller holds the maintenance claim."""
    req = _pool_reload_request(pool)
    age_h = (time.time() - (pool.started_at or time.time())) / 3600.0
    _jaccl_log(cluster_id, f"preventive reload {pool.alias} (age {age_h:.1f}h, "
               f"model={pool.model}, nodes={pool.node_indices or pool.nodes_count})")
    try:
        res = await admin_cluster_load(cluster_id, req)
        _jaccl_log(cluster_id, f"preventive reload done: loaded={res.get('loaded')} "
                   f"load_s={res.get('load_s')}")
    except Exception as e:
        _jaccl_log(cluster_id, f"preventive reload FAILED: {e}")


async def _wait_nodes_reachable(cluster_id: str, timeout_s: float = 360.0,
                                host_ids: Optional[list[str]] = None) -> bool:
    """Poll SSH until the target hosts respond — needed after a reboot, which
    returns as soon as the reboots are ISSUED, not when the nodes are back.
    host_ids=None → every cluster host; else only those hosts (per-node scope)."""
    member_ids = host_ids if host_ids is not None else _cluster_host_ids(cluster_id)
    targets = [h.get("ssh") for h in (_resolve_host(hid) for hid in member_ids)
               if h and h.get("ssh")]
    if not targets:
        return False
    await asyncio.sleep(25)  # let the macs actually go down before polling
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pings = await asyncio.gather(*[_ssh_ping(t) for t in targets])
        if all(p.get("ok") for p in pings):
            return True
        await asyncio.sleep(10)
    return False


def _pool_host_ids(cluster_id: str, req: ArgoLoadRequest) -> list[str]:
    """Host ids of a pool's OWN nodes (per-node scope) — `node_indices` into the
    cluster def, or the legacy contiguous `range(nodes)`. The recovery/refresh
    acts only on these, never the whole cluster: a node is touched iff it is
    actually in this pool (so .32 etc. is included on a 5-node pool, left alone
    when it belongs to a different pool / service)."""
    cd = get_cluster_def(cluster_id)
    nodes = cd.get("nodes") or []
    idxs = list(req.node_indices) if req.node_indices else list(range(int(req.nodes or 0)))
    return [nodes[i]["host"] for i in idxs
            if 0 <= i < len(nodes) and nodes[i].get("host")]


async def _keepalive_recovery_ladder(cluster_id: str, req: ArgoLoadRequest) -> None:
    """WU3 — recovery for a keepalive-detected hang. The surviving ranks are
    stuck in a C++ all_sum (dead UC peer, no timeout), so a graceful stop can't
    clear the QPs — only a reboot does. PER-NODE: reboot ONLY this pool's nodes
    (the QPs that wedged are among them), never reboot-all — other pools/nodes
    are untouched. Ladder: reboot pool nodes → wait them up → reload snapshot →
    clear degraded on success."""
    host_ids = _pool_host_ids(cluster_id, req)
    hosts = [h for h in (_resolve_host(hid) for hid in host_ids) if h]
    if not hosts:
        _jaccl_log(cluster_id, "recovery: no pool hosts resolved — left degraded")
        return
    try:
        _jaccl_log(cluster_id,
                   f"recovery: reboot pool nodes {[h['id'] for h in hosts]} (hung QP, dead peer)")
        results = await asyncio.gather(*[_reboot_one(h) for h in hosts])
        _jaccl_log(cluster_id, f"recovery reboot methods: {[r.get('method') for r in results]}")
    except Exception as e:
        _jaccl_log(cluster_id, f"recovery reboot error: {e}")
        return
    if not await _wait_nodes_reachable(cluster_id, host_ids=host_ids):
        _jaccl_log(cluster_id, "recovery: pool nodes did not return in time — left degraded")
        return
    # /Volumes/models lags SSH by a few seconds after reboot — a reload fired too
    # early fails 'config.json missing or empty'. Grace + one retry (seen 2026-06-14).
    await asyncio.sleep(12)
    for attempt in (1, 2):
        try:
            res = await admin_cluster_load(cluster_id, req)
            _jaccl_log(cluster_id, f"recovery reload: loaded={res.get('loaded')} "
                       f"load_s={res.get('load_s')}")
            if res.get("loaded"):
                _clear_cluster_degraded(cluster_id)
            return
        except Exception as e:
            if attempt == 1:
                _jaccl_log(cluster_id, f"recovery reload attempt 1 failed ({e}); "
                           "retry in 15s (mount race?)")
                await asyncio.sleep(15)
                continue
            _jaccl_log(cluster_id, f"recovery reload error: {e}")
            return


def _fire_keepalive_recovery(cluster_id: str, pool: RunnerPool, res: dict) -> None:
    """WU3 trigger — mark degraded and launch the recovery ladder, deduped per
    cluster via the existing watchdog map. Snapshots the reload config BEFORE
    the pool is torn down."""
    if cluster_id in _WATCHDOG_RECOVERY_BY_CLUSTER:
        return
    if not _JACCL_AUTO_RECOVERY_ENABLED:
        # Early-warning only: surface the failure loudly, take NO disruptive
        # action (no degraded flag, no reboot) until the recovery ladder is
        # validated and explicitly armed via JACCL_AUTO_RECOVERY_ENABLED=1.
        _jaccl_log(cluster_id, f"keepalive FAILED threshold ({pool.alias}, "
                   f"{pool.keepalive_fails} misses) — auto-recovery DISARMED, "
                   f"per-node reboot+reload of this pool's nodes would run if armed. "
                   f"Manual: POST /admin/clusters/{cluster_id}/reset.")
        return
    pool.degraded = True
    pool.degraded_reason = "keepalive timeout — peer unresponsive"
    pool.degraded_at = time.time()
    _mark_cluster_degraded(cluster_id, "keepalive timeout — peer unresponsive",
                           {"alias": pool.alias, "fails": pool.keepalive_fails,
                            "reason": res.get("reason"), "detail": res.get("error")})
    req = _pool_reload_request(pool)
    _jaccl_log(cluster_id, f"keepalive recovery FIRED ({pool.alias}): "
               f"{pool.keepalive_fails} consecutive misses")
    t = asyncio.create_task(_keepalive_recovery_ladder(cluster_id, req))
    _WATCHDOG_RECOVERY_BY_CLUSTER[cluster_id] = t
    t.add_done_callback(
        lambda _d, c=cluster_id: _WATCHDOG_RECOVERY_BY_CLUSTER.pop(c, None))


async def _jaccl_stability_loop() -> None:
    """#40 — per-tick health for every loaded distributed (jaccl) pool:
      WU1 preventive reload when age > threshold AND idle;
      WU2 keepalive probe (idle only — else it queues behind a gen and could
          time out on a healthy pool);
      WU3 controlled recovery on KEEPALIVE_FAIL_THRESHOLD consecutive misses.
    Skips pools already degraded or under recovery."""
    if not _JACCL_STABILITY_ENABLED:
        sys.stderr.write("[jaccl-stability] disabled via env\n")
        return
    while True:
        try:
            await asyncio.sleep(_JACCL_KEEPALIVE_INTERVAL_S)
            for cluster_id, alias, pool in list_all_pools():
                if getattr(pool, "backend", "jaccl") != "jaccl":
                    continue
                if pool.degraded or _cluster_is_degraded(cluster_id):
                    continue
                if cluster_id in _WATCHDOG_RECOVERY_BY_CLUSTER:
                    continue
                # WU1 — preventive reload (idle + old). Claim maintenance so no
                # gen starts mid-reload; if busy, defer to a later tick.
                age = time.time() - (pool.started_at or time.time())
                if age >= _JACCL_PREVENTIVE_RELOAD_AGE_S:
                    if pool._try_claim_maintenance():
                        try:
                            await _preventive_reload(cluster_id, pool)
                        finally:
                            pool._release_maintenance()
                    continue  # reloaded (or busy) — skip keepalive this tick
                # WU2 — keepalive probe, idle only.
                if not pool._try_claim_maintenance():
                    continue
                try:
                    res = await pool.keepalive(timeout_s=_JACCL_KEEPALIVE_TIMEOUT_S)
                finally:
                    pool._release_maintenance()
                if res.get("ok"):
                    if pool.keepalive_fails:
                        _jaccl_log(cluster_id,
                                   f"keepalive recovered ({alias}, rtt {res.get('rtt_ms')}ms)")
                    pool.keepalive_fails = 0
                else:
                    pool.keepalive_fails += 1
                    _jaccl_log(cluster_id,
                        f"keepalive miss {pool.keepalive_fails}/"
                        f"{_JACCL_KEEPALIVE_FAIL_THRESHOLD} ({alias}): "
                        f"{res.get('reason')} {res.get('error')}")
                    if pool.keepalive_fails >= _JACCL_KEEPALIVE_FAIL_THRESHOLD:
                        _fire_keepalive_recovery(cluster_id, pool, res)  # WU3
        except asyncio.CancelledError:
            return
        except Exception as e:
            sys.stderr.write(f"[jaccl-stability] loop error: {e}\n")


def _initial_config() -> Optional[dict]:
    if _args.model:
        return {
            "model": _args.model, "mode": _args.mode,
            "use_ap": _args.use_ap, "nodes_count": _args.nodes,
            "kv_q8": getattr(_args, "kv_q8", False),
        }
    state = load_state()
    if state and state.get("model"):
        return {
            "model": state["model"], "mode": state.get("mode", "pipeline"),
            "use_ap": state.get("use_ap", False),
            "nodes_count": state.get("nodes", 2),
            "kv_q8": state.get("kv_q8", False),
        }
    return None


def _initial_default_config() -> Optional[dict]:
    state = load_state(cluster="default")
    if state and state.get("model"):
        return {
            "model": state["model"], "mode": state.get("mode", "pipeline"),
            "use_ap": state.get("use_ap", True),
            "nodes_count": state.get("nodes", 3),
            "kv_q8": state.get("kv_q8", False),
        }
    return None



# Version is the single source of truth for the server identity. Bump at
# each meaningful release (engine behaviour change, API contract change,
# user-visible feature). Surfaced via /admin/version + dashboard About tab.
#
# Bump conventions:
#   patch (1.7.2 → 1.7.3) — bugfix only
#   minor (1.7.2 → 1.8.0) — new feature, new endpoint, behaviour change
#   major (1.7.2 → 2.0.0) — breaking API or topology change
#
# Use `./scripts/bump-version.sh patch|minor|major` to bump + auto-commit.
APP_VERSION = "1.9.0"

app = FastAPI(
    title="OdyssAI-X (odyssai.eu)",
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.get("/admin/version")
async def admin_version():
    """Identity + runtime info for the About tab. Cheap (no probes)."""
    import platform as _platform
    info: dict = {
        "name": "OdyssAI-X",
        "version": APP_VERSION,
        "python": _platform.python_version(),
        "platform": _platform.platform(),
    }
    # Best-effort upstream lib versions — useful for diagnosing engine drift.
    try:
        import mlx
        info["mlx"] = getattr(mlx, "__version__", "?")
    except Exception:
        pass
    try:
        import mlx_lm
        info["mlx_lm"] = getattr(mlx_lm, "__version__", "?")
    except Exception:
        pass
    return info


# ──────────────────────────────────────────────────────────────────────────────
# Admin auth middleware (opt-in)
# ──────────────────────────────────────────────────────────────────────────────
# OdyssAI-X is a LAN-bound engine by default. /admin/* is OPEN unless the
# operator deliberately opts into Bearer-token auth by setting
# ADMIN_TOKEN in the environment.
#
# When to set it:
#   - You're exposing the engine beyond your LAN (Cloudflare tunnel,
#     port forward, multi-tenant deployment, …).
#   - You want a second layer of access control on top of network
#     segmentation.
#
# When NOT to set it:
#   - Single-operator LAN install (the common case). Whoever can reach
#     `:8000` is by definition the operator. A token here adds friction
#     without protecting against any realistic threat.
#
# EventSource (used by /admin/logs follow stream) can't set custom
# headers in browsers, so we also accept the token as a `?token=…` query
# param when auth is enabled.

ADMIN_TOKEN = (env_get("ADMIN_TOKEN") or "").strip()

if ADMIN_TOKEN:
    sys.stderr.write(
        f"[api] /admin/* protected by Bearer token (length {len(ADMIN_TOKEN)})\n"
    )
else:
    _ADMIN_OPEN_WARNING = (
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  WARNING: /admin/* routes are OPEN — no auth required.      ║\n"
        "║  Anyone on the network can load/unload models, read stats,  ║\n"
        "║  and change cluster config.                                  ║\n"
        "║  Set ODYSSAI_X_ADMIN_TOKEN=<secret> to enable Bearer auth.  ║\n"
        "║  Safe on a trusted LAN; harden before any WAN exposure.     ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )
    sys.stderr.write(_ADMIN_OPEN_WARNING)


@app.middleware("http")
async def _admin_token_middleware(request: Request, call_next):
    path = request.url.path

    # Extract bearer once (used by multiple branches below).
    bearer = None
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        bearer = auth[7:].strip()
    if not bearer:
        bearer = request.query_params.get("token")

    # ── Per-route auth rules ──────────────────────────────────────────────
    # Public always (no auth check):
    public_always = (
        path == "/admin/discovery/state"        # watcher poll
    )
    # Public only while the discovery gate is open:
    if path == "/admin/pair" and get_discovery_state().get("active"):
        public_always = True
    # /admin/crew/self accepts a crew bearer (not admin):
    if path == "/admin/crew/self" and request.method == "DELETE" and bearer:
        if find_crew_by_token(bearer):
            return await call_next(request)
        # fall through to admin-token branch (admin can also self-revoke
        # any crew via /admin/crew/{id} — not /self — so this is rejected
        # unless admin token is provided which is silly but not harmful).

    if not ADMIN_TOKEN:
        # Dev mode — admin routes open. We still update crew last_seen below.
        response = await call_next(request)
        _maybe_update_crew_last_seen(path, bearer, response)
        return response

    if public_always:
        return await call_next(request)

    if path.startswith("/admin/"):
        if bearer != ADMIN_TOKEN:
            return JSONResponse(
                {"detail": "missing or invalid admin token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    # Not /admin/* — public route. Update crew last_seen if a crew bearer was sent.
    response = await call_next(request)
    _maybe_update_crew_last_seen(path, bearer, response)
    return response


def _maybe_update_crew_last_seen(path: str, bearer: Optional[str], response) -> None:
    """If the request carried a crew bearer on /v1/* or similar, bump the crew
    member's last_seen and tag the response if the token has been revoked."""
    if not bearer or not path.startswith("/v1/"):
        return
    # Ignore admin token on /v1/* (works but isn't a crew member)
    if bearer == ADMIN_TOKEN:
        return
    entry = find_crew_by_token(bearer)
    if entry:
        update_crew_last_seen(entry["id"])
    else:
        # Companion sent something looking like a crew token but it's unknown.
        # Tell it via response header so it can prompt re-pair.
        try:
            response.headers["x-odyssai-crew-revoked"] = "true"
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Public endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/odysseus.png")
async def serve_odysseus_icon():
    # Try multiple candidate paths so the icon works both inside the
    # container (/app/odysseus.png) and from local dev (sibling of api.py).
    candidates = [
        Path("/app/odysseus.png"),
        Path(env_get("ICON", "")) if env_get("ICON") else None,
        _HERE / "odysseus.png",
        _HERE.parent / "logo" / "odysseus.png",
    ]
    for p in candidates:
        if p and p.exists():
            return FileResponse(str(p), media_type="image/png")
    raise HTTPException(404, "odysseus icon missing")


@app.get("/")
async def dashboard():
    if not DASHBOARD_FILE.exists():
        return HTMLResponse("<h1>dashboard.html missing</h1>", status_code=500)
    return HTMLResponse(DASHBOARD_FILE.read_text())


WALL_FILE = Path(os.environ.get("WALL_FILE", _HERE / "wall.html"))


@app.get("/odyrag")
async def odyrag():
    """OdyRAG — knowledge graph management dashboard (LightRAG-based).
    Served fresh per request so docker cp deploys without restart."""
    if not ODYRAG_FILE.exists():
        return HTMLResponse("<h1>odyrag.html missing</h1>", status_code=500)
    return HTMLResponse(ODYRAG_FILE.read_text())


@app.api_route("/odyrag/api/{shard_id}/{path:path}", methods=["GET", "POST", "DELETE"])
async def odyrag_proxy(shard_id: str, path: str, request: Request):
    """Proxy requests to LightRAG shard instances — avoids browser CORS restrictions."""
    _SHARD_PORTS: dict[str, int] = {
        "company": 8767,
        "shard1": 8768, "shard2": 8769, "shard3": 8770, "shard4": 8771,
    }
    port = _SHARD_PORTS.get(shard_id)
    if port is None:
        raise HTTPException(status_code=404, detail=f"Unknown shard: {shard_id!r}")
    qs = str(request.query_params)
    target = f"http://192.168.86.39:{port}/{path}"
    if qs:
        target = f"{target}?{qs}"
    body = await request.body()
    fwd_headers: dict[str, str] = {}
    if body:
        ct = request.headers.get("content-type", "")
        if ct:
            fwd_headers["content-type"] = ct
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.request(
                request.method, target,
                content=body or None,
                headers=fwd_headers,
            )
        try:
            return JSONResponse(resp.json(), status_code=resp.status_code)
        except Exception:
            return PlainTextResponse(resp.text, status_code=resp.status_code)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.get("/wall")
async def status_wall():
    """Standalone status wall — portrait black display for ops monitoring.
    Same-origin so it can poll /admin/clusters without CORS. Read fresh per
    request → hot-deploys via `docker cp` without a container restart."""
    if not WALL_FILE.exists():
        return HTMLResponse("<h1>wall.html missing</h1>", status_code=500)
    return HTMLResponse(WALL_FILE.read_text())


# ──────────────────────────────────────────────────────────────────────────────
# User Guide — Markdown content served straight from disk so the operator can edit
# without redeploying. Public (no auth) by design — it's documentation.
# ──────────────────────────────────────────────────────────────────────────────


def _user_guide_topic_meta(path: Path) -> dict:
    """Parse the leading title line + numeric prefix to feed the picker."""
    stem = path.stem  # e.g. "12-providers-and-aliases"
    parts = stem.split("-", 1)
    try:
        order = int(parts[0])
        slug = parts[1] if len(parts) > 1 else stem
    except ValueError:
        order = 999
        slug = stem
    title = slug.replace("-", " ").title()
    try:
        # First non-blank line starting with "# " wins; falls back to slug.
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                break
    except Exception:
        pass
    return {"slug": slug, "title": title, "order": order, "file": path.name}


@app.get("/help/topics")
async def help_topics():
    """List the user-guide topics in declared order."""
    if not USER_GUIDE_DIR.is_dir():
        return {"topics": [], "note": f"user-guide dir not found: {USER_GUIDE_DIR}"}
    topics = []
    for p in sorted(USER_GUIDE_DIR.glob("*.md")):
        topics.append(_user_guide_topic_meta(p))
    return {"topics": topics}


@app.get("/help/topic/{slug}")
async def help_topic(slug: str):
    """Return the raw Markdown for a topic by slug. Browses by filename
    suffix so the route stays stable even when files are renumbered."""
    if not USER_GUIDE_DIR.is_dir():
        raise HTTPException(404, "user-guide directory missing")
    # Sanity-check the slug to prevent traversal.
    if "/" in slug or ".." in slug or "\x00" in slug:
        raise HTTPException(400, "bad slug")
    # Match by suffix after the numeric prefix.
    for p in sorted(USER_GUIDE_DIR.glob("*.md")):
        stem = p.stem
        parts = stem.split("-", 1)
        candidate = parts[1] if len(parts) > 1 else stem
        if candidate == slug:
            return PlainTextResponse(p.read_text(), media_type="text/markdown")
    raise HTTPException(404, f"topic '{slug}' not found")


@app.get("/health")
async def health():
    base = {
        "version": APP_VERSION,
        # Surfaces admin auth posture so operators can detect open installs
        # programmatically (e.g. Companion shows a warning in Settings).
        "admin_auth_enabled": bool(ADMIN_TOKEN),
    }
    if _pool is None:
        return {"status": "idle", **base}
    alive = _pool.alive_count()
    return {
        "status": "ok" if alive == len(_pool.runners) else "degraded",
        "model": _pool.model, "alive": alive, "nodes": len(_pool.runners),
        **base,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Capability contract — engine + per-model declarations
# ──────────────────────────────────────────────────────────────────────────────
# Clients (Companion, Continue.dev, raw curl) ask for capabilities instead of
# guessing them by string-matching on model id. Two endpoints:
#   GET /.well-known/inference-engine.json   — engine metadata
#   GET /v1/models                            — per-model, OpenAI-compat with `x_odyssai` extension
# See docs/CAPABILITIES.md (TODO) for the full spec.

ENGINE_VERSION = "0.7.3"

_config_json_cache: dict[str, dict] = {}     # full path → parsed HF config.json
_caps_cache: dict[str, dict] = {}             # static caps by model_id


def _quant_from_name(name: str) -> Optional[str]:
    """Extract quantization from the model name. Strong file-naming convention:
    `-Nbit`, `-N-bit`, `-qN`. Returns string like '8-bit', '4-bit', or None."""
    if not name:
        return None
    s = name.lower()
    m = re.search(r"-(\d+)[ -]?bit\b", s)
    if m:
        return f"{m.group(1)}-bit"
    m = re.search(r"-q(\d+)\b", s)
    if m:
        return f"{m.group(1)}-bit"
    if "-fp16" in s or "-bf16" in s:
        return "fp16"
    return None


# Prefix whitelist of HF model_type values known to support tool/function
# calling via their chat templates. Prefix-based to absorb minor naming
# variants (qwen3_5_moe, qwen3_next, deepseek_v32, etc.). Conservative —
# under-declare rather than lie. Always overridden by an explicit
# "tool" mention in the chat_template if config.json embeds one.
_TOOLS_SUPPORTED_PREFIXES = (
    "qwen", "llama", "mistral", "mixtral", "ministral",
    "deepseek", "glm", "phi", "gemma",
    "hy_v", "minimax", "command", "yi",
)


def _model_type_supports_tools(mt: str) -> bool:
    if not mt:
        return False
    return any(mt.startswith(p) for p in _TOOLS_SUPPORTED_PREFIXES)


def _enrich_caps_from_config(caps: dict, config: dict) -> None:
    """Pull context_length, modalities, supports_* from a parsed HF config.json."""
    # Context length — may live at root or under text_config / language_model.
    text_cfg = config.get("text_config") or {}
    lang_cfg = (text_cfg.get("language_model")
                or config.get("language_model") or {})
    ctx = (config.get("max_position_embeddings")
           or text_cfg.get("max_position_embeddings")
           or lang_cfg.get("max_position_embeddings"))
    if ctx:
        caps["context_length"] = int(ctx)

    # Model family / vision detection from model_type.
    mt = (config.get("model_type")
          or text_cfg.get("model_type") or "").lower()
    caps["family"] = mt or None

    # Vision: model_type pattern OR presence of vision_config / image processors.
    is_vision = ("_vl" in mt or "_vision" in mt or "vision" in mt
              or "vision_config" in config
              or "vision_tower_config" in config)
    if is_vision:
        caps["supports_vision"] = True
        caps["modalities"] = sorted(set((caps.get("modalities") or []) + ["text", "image"]))
    else:
        caps["supports_vision"] = False

    # Tools: model_type prefix whitelist + presence of `tool` in chat_template.
    base_type = mt.split("_vl")[0].split("_vision")[0]
    if _model_type_supports_tools(base_type) or _model_type_supports_tools(mt):
        caps["supports_tools"] = True
    chat_template = config.get("chat_template") or ""
    if isinstance(chat_template, str) and "tool" in chat_template.lower():
        caps["supports_tools"] = True
    # Default to False if model_type known but unsupported, leave None if unknown.
    if caps.get("supports_tools") is None and mt:
        caps["supports_tools"] = False

    # JSON mode: nearly universal via prompt; declare True for known instruct families.
    if caps.get("supports_tools"):
        caps["supports_json_mode"] = True


async def _validate_model_layout(ssh_target: str, model_path: str,
                                  models_dir: Optional[str] = None,
                                  timeout: float = 8.0) -> tuple[bool, Optional[str]]:
    """Pre-flight check that a model dir is fully usable before cluster spawn.

    Returns (ok, error_message). On failure, error is a short human-readable
    string suitable for surfacing in the dashboard ("config.json missing",
    "no safetensors found", etc.) so we don't fail mid-load with a cryptic
    KeyError (cf. Mistral Medium 3.5 saga 2026-05-15).

    Cheap : single SSH round-trip that runs the checks in shell.

    Regression fix 2026-05-25 : when `model_path` is a relative id like
    `inferencerlabs/Qwen3.5-397B-A17B-MLX-9bit`, SSH defaults to $HOME on
    the rank-0 node and the existence check resolves nothing. Caller now
    passes `models_dir` (the rank-0 node's models_dir from topology) so
    we can `cd` into it before checking. Absolute paths (`/Volumes/...`)
    are checked as-is regardless of models_dir.
    """
    p = model_path.rstrip("/")
    # One SSH round-trip running a python3 probe on the node. Checks PRESENCE
    # (config.json + >=1 safetensors + tokenizer) AND COMPLETENESS (no in-progress
    # *.hfdl download markers, and every shard listed in model.safetensors.index.json
    # actually on disk). The completeness half catches a half-finished rsync — a
    # partial copy used to pass the old shell probe and crash mid-load per-rank.
    remote_py = (
        "import json,os,glob,sys\n"
        "p=" + json.dumps(p) + "\n"
        "md=" + json.dumps(models_dir or "") + "\n"
        "d=p if os.path.isabs(p) else (os.path.join(md,p) if md else p)\n"
        "cj=os.path.join(d,'config.json')\n"
        "if not (os.path.isfile(cj) and os.path.getsize(cj)>0): print('MISSING_CONFIG'); sys.exit()\n"
        "st=glob.glob(os.path.join(d,'*.safetensors'))\n"
        "if not st: print('MISSING_WEIGHTS'); sys.exit()\n"
        "if not any(os.path.isfile(os.path.join(d,t)) and os.path.getsize(os.path.join(d,t))>0 "
        "for t in ('tokenizer.json','tokenizer.model','tokenizer_config.json')): print('MISSING_TOKENIZER'); sys.exit()\n"
        "if glob.glob(os.path.join(d,'*.hfdl')): print('INCOMPLETE_DOWNLOAD'); sys.exit()\n"
        "ix=os.path.join(d,'model.safetensors.index.json')\n"
        "sharded=any('-of-' in os.path.basename(f) for f in st) or len(st)>1\n"
        "if sharded:\n"
        " if not (os.path.isfile(ix) and os.path.getsize(ix)>0): print('INCOMPLETE_INDEX'); sys.exit()\n"
        " want=set(json.load(open(ix)).get('weight_map',{}).values())\n"
        " have=set(os.path.basename(f) for f in st)\n"
        " if want-have: print('INCOMPLETE_SHARDS:%d'%len(want-have)); sys.exit()\n"
        "print('OK')\n"
    )
    cmd = ["ssh", "-o", "ConnectTimeout=4", "-o", "BatchMode=yes",
           ssh_target, "python3 -c " + shlex.quote(remote_py)]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out = (stdout or b"").decode("utf-8", "ignore").strip()
        if out == "OK":
            return True, None
        if out == "MISSING_CONFIG":
            return False, f"config.json missing or empty at {p}"
        if out == "MISSING_WEIGHTS":
            return False, f"no *.safetensors files at {p}"
        if out == "MISSING_TOKENIZER":
            return False, f"no tokenizer (tokenizer.json/model) at {p}"
        if out == "INCOMPLETE_DOWNLOAD":
            return False, f"download still in progress (.hfdl markers) at {p}"
        if out == "INCOMPLETE_INDEX":
            return False, f"sharded model missing model.safetensors.index.json at {p} (rsync unfinished?)"
        if out.startswith("INCOMPLETE_SHARDS"):
            n = out.split(":", 1)[1] if ":" in out else "?"
            return False, f"incomplete — {n} safetensors shard(s) missing at {p} (rsync unfinished?)"
        # SSH itself failed (host down, perms…). Conservative: don't block.
        err = (stderr or b"").decode("utf-8", "ignore").strip()[:200]
        return False, f"ssh check failed ({ssh_target}): {err or 'unknown'}"
    except asyncio.TimeoutError:
        return False, f"layout check timeout ({timeout}s) on {ssh_target}"
    except Exception as e:
        return False, f"layout check error: {e}"


async def _read_model_config(ssh_target: str, model_path: str, timeout: float = 6.0) -> Optional[dict]:
    """Read `config.json` of a model dir over SSH. Cached forever per model path."""
    cache_key = f"{ssh_target}::{model_path}"
    if cache_key in _config_json_cache:
        return _config_json_cache[cache_key]
    cfg_path = f"{model_path.rstrip('/')}/config.json"
    cmd = ["ssh", "-o", "ConnectTimeout=4", "-o", "BatchMode=yes",
           ssh_target, f"cat {shlex.quote(cfg_path)} 2>/dev/null"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0 or not stdout:
            return None
        try:
            cfg = json.loads(stdout.decode("utf-8", "ignore"))
        except Exception:
            return None
        _config_json_cache[cache_key] = cfg
        return cfg
    except Exception:
        return None


def _avg_tps_for(model: str, window: int = 20) -> Optional[float]:
    """Recent average tokens/s for a model from the metrics ring."""
    vals = [m["tps"] for m in list(_metrics)[:window]
            if m.get("tps", 0) > 0 and m.get("model") == model]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 1)


def _load_history_entry(cluster: str, model: str) -> Optional[dict]:
    """Return the recorded load_history entry for (cluster, model) or None.

    The shape is `{"last_load_s": float, "size_bytes": int, "nodes": int}`.
    Tolerates lookup by full path *and* by basename (rsync of the same model
    is recorded under both forms in older histories)."""
    cfg = _load_cluster_config()
    hist = (cfg.get(cluster, {}) or {}).get("load_history") or {}
    if not isinstance(hist, dict):
        return None
    entry = hist.get(model)
    if isinstance(entry, dict):
        return entry
    # fallback to basename match
    base = model.split("/")[-1]
    for k, v in hist.items():
        if not isinstance(v, dict):
            continue
        if k == base or k.split("/")[-1] == base:
            return v
    return None


def _estimated_load_s_for(cluster: str, model: str) -> Optional[float]:
    entry = _load_history_entry(cluster, model)
    if not entry:
        return None
    val = entry.get("last_load_s") or entry.get("load_s")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _size_bytes_for(cluster: str, model: str) -> Optional[int]:
    entry = _load_history_entry(cluster, model)
    if not entry:
        return None
    try:
        return int(entry["size_bytes"]) if entry.get("size_bytes") else None
    except (TypeError, ValueError):
        return None


async def _model_capabilities(
    model_id: str,
    *,
    pool: Optional["RunnerPool"] = None,
    pool_name: Optional[str] = None,
    cluster_for_unloaded: Optional[str] = None,
    size_bytes: Optional[int] = None,
) -> dict:
    """Compose the x_odyssai dict for one model.

    If `pool` is given, the model is currently loaded — live state available.
    Otherwise we try to read `config.json` from the model dir on the cluster's
    master (best effort; capability declaration falls back gracefully).
    """
    caps: dict = {
        "loaded": pool is not None,
        "loading": False,
        "pool": pool_name,
        "backend": None,
        "nodes": None,
        "context_length": None,
        "max_output_tokens": None,
        "quantization": _quant_from_name(model_id),
        "size_bytes": size_bytes,
        "modalities": ["text"],
        "family": None,
        "supports_tools": None,
        "supports_vision": None,
        "supports_json_mode": None,
        "supports_streaming": True,
        "estimated_tps": None,
        "estimated_load_s": None,
        "warm": pool is not None,
        "kv_cache_q8": False,
        "admin_loadable": True,
    }

    # Live state for loaded models
    if pool is not None:
        cd = get_cluster_def(pool_name) if pool_name else {}
        caps["backend"] = cd.get("backend")
        caps["nodes"] = pool.nodes_count
        caps["kv_cache_q8"] = bool(pool.kv_q8)
        caps["estimated_tps"] = _avg_tps_for(pool.model)
        # cluster_label + kind let clients (Companion picker) render the
        # group heading using the operator-chosen display name ("Argo"
        # instead of "main", "TeleCoder" instead of "telemak-code-next").
        # kind tells the picker which group bucket to drop the entry in
        # (telemak vs mlx-distributed vs cloud).
        if cd:
            caps["cluster_label"] = cd.get("name") or pool_name
            caps["kind"] = cd.get("kind")

    # Try config.json — for loaded: from rank0 of its pool; for unloaded: from cluster master
    ssh_target = None
    model_path = None
    if pool is not None and pool.nodes:
        # rank 0 of the loaded pool
        rank0 = next((n for n in pool.nodes if n.get("rank") == 0), pool.nodes[0])
        ssh_target = rank0["ssh"]
        model_path = pool.model if pool.model.startswith("/") else None
        if not model_path:
            mdir = models_dir_for(pool_name) if pool_name else None
            if mdir:
                model_path = f"{mdir.rstrip('/')}/{pool.model}"
    elif cluster_for_unloaded:
        topo = build_topology(cluster_for_unloaded, 1)
        if topo:
            ssh_target = topo[0]["ssh"]
        mdir = models_dir_for(cluster_for_unloaded)
        model_path = f"{mdir.rstrip('/')}/{model_id}"

    if ssh_target and model_path:
        cfg = await _read_model_config(ssh_target, model_path)
        if cfg:
            _enrich_caps_from_config(caps, cfg)

    # Load duration + size from history (works for loaded and unloaded)
    cluster_for_lookup = pool_name or cluster_for_unloaded
    if cluster_for_lookup:
        if caps.get("estimated_load_s") is None:
            caps["estimated_load_s"] = _estimated_load_s_for(cluster_for_lookup, model_id)
        if caps.get("size_bytes") is None:
            caps["size_bytes"] = _size_bytes_for(cluster_for_lookup, model_id)

    return caps


def _engine_metadata() -> dict:
    """Static engine metadata exposed at /.well-known/inference-engine.json."""
    return {
        "name": "OdyssAI-X",
        "vendor": "odyssai.eu",
        "version": ENGINE_VERSION,
        "api_compat": ["openai/v1", "anthropic/v1"],
        "auth": {
            "required": bool(ADMIN_TOKEN),
            "scheme": "bearer",
            "scope": "/admin/*",
            "public_routes": ["/v1/*", "/health", "/.well-known/*"],
            "eventsource_query_fallback": True,
        },
        "features": [
            "sse-streaming",
            "tool-calling",
            "vision",
            "prefix-cache",
            "speculative-decoding",
            "distributed-inference",
            "kv-cache-q8",
            "openai-models-extended",
            "cloud-passthrough",
            "fallback-routing",
        ],
        "limits": {
            "max_concurrent_requests": 16,
            "max_prompt_chars": 1_000_000,
        },
        "admin": {
            "pools_endpoint": "/admin/pools",
            "clusters_endpoint": "/admin/clusters/{id}",
            "inventory_endpoint": "/admin/inventory",
            "runs_endpoint": "/admin/runs",
            "sync_matrix_endpoint": "/admin/sync/matrix",
            "models_dir_configurable": True,
            "supports_pool_load_unload": True,
            "supports_hot_swap": False,
        },
        "extensions": {
            "x_odyssai": {
                "doc": "https://odyssai.eu/contract/x_odyssai",
                "fields": [
                    "loaded", "loading", "pool", "backend", "nodes",
                    "context_length", "max_output_tokens", "quantization",
                    "size_bytes", "modalities", "family",
                    "supports_tools", "supports_vision", "supports_json_mode",
                    "supports_streaming", "estimated_tps", "estimated_load_s",
                    "warm", "kv_cache_q8", "admin_loadable",
                ],
            },
        },
    }


@app.get("/.well-known/inference-engine.json")
async def well_known_engine():
    return _engine_metadata()


# ──────────────────────────────────────────────────────────────────────────────
# Cloud providers — OdyssAI-X-as-gateway for cloud-hosted models
# ──────────────────────────────────────────────────────────────────────────────
# Companion / external clients hit /v1/chat/completions with model="or:foo"
# and OdyssAI-X proxies to OpenRouter (or any OpenAI-compat upstream) using
# the API key stored in env. All OpenAI-compat — no protocol translation.
# Config lives in cluster-config.json under "cloud_providers".
#
# Convention: alias prefix = provider id colon (e.g. "or:claude-haiku" for
# OpenRouter, "oa:gpt-4o" for OpenAI direct). No slashes (avoid confusion
# with model paths).

DEFAULT_CLOUD_PROVIDERS: dict[str, dict] = {
    # Example shape — actual user config goes in cluster-config.json:
    # "openrouter": {
    #   "api_base": "https://openrouter.ai/api/v1",
    #   "api_key_env": "ODYSSAI_X_OPENROUTER_KEY",
    #   "published": [
    #     {"alias": "or:claude-haiku",
    #      "upstream": "anthropic/claude-haiku-4-5",
    #      "caps": {"context_length": 200000, "supports_tools": True,
    #               "supports_vision": False, "modalities": ["text"]}},
    #   ],
    # },
}


def get_cloud_providers() -> dict:
    cfg = _load_cluster_config()
    return cfg.get("cloud_providers", {}) or {}


def save_cloud_providers(providers: dict) -> None:
    with _cluster_config_txn() as cfg:
        cfg["cloud_providers"] = providers


def find_cloud_alias(model_id: Optional[str]) -> Optional[tuple[str, dict, dict]]:
    """Look up which provider serves an alias. Returns (provider_id, provider_cfg, alias_entry).

    Skips providers with `enabled=false` — they're toggled off without delete,
    so their aliases shouldn't resolve to a backend. Clients calling a disabled
    alias get a 404 from the proxy router as if the alias didn't exist.
    """
    if not model_id:
        return None
    providers = get_cloud_providers()
    for prov_id, prov in providers.items():
        if not _provider_enabled(prov):
            continue
        for entry in (prov.get("published") or []):
            if entry.get("alias") == model_id:
                return prov_id, prov, entry
    return None


# Canonical Claude tier names that Claude Code / the Anthropic SDK send
# (claude-opus-4-x, claude-sonnet-4-x, claude-3-5-haiku-x, etc.). We map
# each tier to an operator-chosen local model so `claude` works
# plug-and-play against a custom ANTHROPIC_BASE_URL without the user
# setting ANTHROPIC_MODEL. Inspired by free-claude-code's tier routing.
_ANTHROPIC_TIER_ENV = {
    "opus":   "ANTHROPIC_OPUS",
    "sonnet": "ANTHROPIC_SONNET",
    "haiku":  "ANTHROPIC_HAIKU",
}


def _first_servable_model() -> Optional[str]:
    """The model id a tier should fall back to when nothing is configured:
    the first loaded local pool, else the first loaded Telemak cluster."""
    for cid, alias, _pool in list_all_pools():
        return alias if alias != DEFAULT_ALIAS else cid
    for cid in active_cluster_ids():
        cd = get_cluster_def(cid)
        if cd.get("kind") == "telemak":
            return cid
    return None


def _resolve_anthropic_tier(model_id: Optional[str]) -> Optional[str]:
    """Map a canonical Claude tier name → an operator-chosen local model.

    Resolution order, per tier (opus/sonnet/haiku):
      1. tier-specific env var (ODYSSAI_X_ANTHROPIC_{OPUS,SONNET,HAIKU})
      2. catch-all env var ODYSSAI_X_ANTHROPIC_MODEL
      3. plug-and-play fallback = first servable local model

    Returns `model_id` unchanged when it isn't a Claude tier name, or when
    it already resolves to a published cloud alias (operator intent wins).
    This is purely additive: a `claude-*` name that previously 404'd now
    routes to a local model; nothing else changes path.
    """
    if not model_id:
        return model_id
    low = model_id.lower()
    if not low.startswith("claude"):
        return model_id
    # If the operator explicitly published a cloud alias by this exact
    # name, respect it — don't hijack to a local model.
    if find_cloud_alias(model_id):
        return model_id
    tier = (
        "opus" if "opus" in low
        else "sonnet" if "sonnet" in low
        else "haiku" if "haiku" in low
        else None
    )
    if tier:
        env_target = env_get(_ANTHROPIC_TIER_ENV[tier])
        if env_target:
            return env_target
    catch_all = env_get("ANTHROPIC_MODEL")
    if catch_all:
        return catch_all
    return _first_servable_model() or model_id


def _cloud_provider_key(prov: dict) -> Optional[str]:
    """Resolve the provider's API key.

    Priority:
      1. `api_key` field in the config (set via UI, stored in cluster-config.json).
      2. `api_key_env` env var name (fallback for ops who don't want secrets in config).
    """
    direct = (prov.get("api_key") or "").strip()
    if direct:
        return direct
    env_var = prov.get("api_key_env")
    if not env_var:
        return None
    return (env_value_by_name(env_var) or "").strip() or None


# Known upstream quirks we work around in the proxy. Keep in sync with
# `_cloud_entries_for_v1_models()` so clients see the same list via
# `x_odyssai.backend_quirks` on /v1/models.
PROVIDER_QUIRKS: dict[str, list[str]] = {
    "mlx-vlm": [
        "stream_tools_empty_deltas",      # tokens generated, deltas stay empty
        "finish_reason_stop_with_tools",  # returns "stop" instead of "tool_calls"
    ],
}


def _has_quirk(prov_id: str, quirk: str) -> bool:
    return quirk in PROVIDER_QUIRKS.get(prov_id, [])


async def _proxy_chat_completion(prov_id: str, prov: dict, entry: dict,
                                  body: dict) -> Any:
    """Proxy OpenAI-compat chat completion to an upstream provider.

    Streaming pass-through: when body.stream=True we relay SSE bytes as-is.
    Non-streaming: we wait, parse JSON, return JSONResponse with the upstream
    status code.

    Auth is optional: if no api_key resolves (neither stored nor env), we
    proxy without an Authorization header — useful for local sidecars
    (mlx-vlm on the LAN, an internal vLLM, etc.) that don't require auth.

    Protocol mismatch: if the provider is `protocol=anthropic` (only api.
    anthropic.com today), we transparently translate the OpenAI request
    to Anthropic /v1/messages and translate the response back. Same UX
    as upstreams that speak both wire formats natively. Lets a client
    client (OpenAI shape) call Claude through OdyssAI-X without knowing
    about the protocol split.
    """
    if _provider_protocol(prov) == "anthropic":
        return await _proxy_chat_completion_via_anthropic(prov_id, prov, entry, body)
    api_key = _cloud_provider_key(prov)

    upstream = entry.get("upstream")
    if not upstream:
        raise HTTPException(500, f"cloud alias '{entry.get('alias')}' has no upstream model id")

    # Override model id with the upstream-native name. Strip our internal fields.
    fwd = dict(body)
    fwd["model"] = upstream
    fwd.pop("session_id", None)
    # `enable_thinking` is our canonical name (mlx-lm + mlx-vlm convention).
    # Some upstreams use `thinking` (boolean, false = skip the <think>…</think>
    # block). Translate before popping so users get consistent behaviour
    # regardless of the upstream provider's preferred field name. Unknown
    # providers ignore unknown fields, so it's safe to send both.
    #
    # Layered policy:
    #   1. Client explicit `enable_thinking` or `thinking` → wins (passed through)
    #   2. Neither field present → inject the server-wide default
    #      (`settings.enable_thinking_default` via admin, env `THINKING_DEFAULT`
    #      fallback). This stops the upstream's own default — many upstreams
    #      defaults thinking ON, which wastes tokens on `<think>` for general
    #      chat. Client always retains override via the body.
    et = fwd.pop("enable_thinking", None)
    # Resolve the effective thinking intent: an explicit client `thinking` /
    # `enable_thinking` wins; otherwise inject the server-wide default (many
    # upstreams default thinking ON, wasting tokens on <think> for plain chat).
    _t = fwd.get("thinking", None)
    if isinstance(_t, dict):
        think_on = None  # client sent a structured config — leave it untouched
    elif isinstance(_t, bool):
        think_on = _t
    elif et is not None:
        think_on = bool(et)
    else:
        think_on = get_enable_thinking_default()
    if think_on is not None:
        # MiniMax's OpenAI-compatible API validates `thinking` as a
        # ThinkingConfig OBJECT ({"type":"enabled"|"disabled"}), not the bare
        # boolean that other upstreams accept or ignore. Translate per-upstream.
        if "minimax" in str(upstream).lower():
            fwd["thinking"] = {"type": "enabled" if think_on else "disabled"}
        else:
            fwd["thinking"] = think_on
    # OpenAI spec: streaming responses do NOT include `usage` in their final
    # chunk unless the client opts in via `stream_options.include_usage`.
    # Without it, clients (Companion) can't render prompt/completion tokens
    # or tok/s in their metrics box — they only see chunk count and duration.
    # We always opt in here on streaming requests so the experience is
    # consistent regardless of provider. Local mlx-lm runners already report
    # usage natively; the cloud proxy needs to ask for it explicitly.
    if fwd.get("stream"):
        existing_opts = fwd.get("stream_options") or {}
        if not isinstance(existing_opts, dict):
            existing_opts = {}
        if "include_usage" not in existing_opts:
            existing_opts["include_usage"] = True
        fwd["stream_options"] = existing_opts

    url = f"{prov['api_base'].rstrip('/')}/chat/completions"
    headers = {
        "content-type": "application/json",
        # OpenRouter-specific (ignored by other providers):
        "http-referer": "https://odyssai.eu",
        "x-title": "OdyssAI-X",
    }
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    is_stream = bool(fwd.get("stream"))

    # mlx-vlm has a known bug: streaming + tools yields empty deltas
    # (output_tokens increment but content/tool_calls stay []). We work around
    # it by forcing non-stream upstream and re-emitting the full response as a
    # single SSE chunk to the client. The client keeps its streaming code
    # path; we just deliver one big delta instead of N small ones.
    needs_stream_to_unary = (
        is_stream
        and bool(fwd.get("tools"))
        and _has_quirk(prov_id, "stream_tools_empty_deltas")
    )

    if needs_stream_to_unary:
        unary_fwd = dict(fwd)
        unary_fwd["stream"] = False
        async def gen_unary() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    r = await client.post(url, headers=headers, json=unary_fwd)
                except Exception as e:
                    err = {"error": {"message": f"upstream {prov_id} unreachable: {e}",
                                      "provider": prov_id}}
                    yield ("data: " + json.dumps(err) + "\n\n").encode()
                    return
                if r.status_code >= 400:
                    txt = r.text or ""
                    err = {"error": {"message": txt[:300], "code": r.status_code,
                                      "provider": prov_id}}
                    yield ("data: " + json.dumps(err) + "\n\n").encode()
                    return
                try:
                    payload = r.json()
                except Exception:
                    err = {"error": {"message": "non-JSON upstream response",
                                      "provider": prov_id}}
                    yield ("data: " + json.dumps(err) + "\n\n").encode()
                    return
                # Re-emit the upstream payload as a single SSE chunk.
                # Build a chat.completion.chunk wrapper around choice[0].message.
                created = payload.get("created") or int(time.time())
                cid = payload.get("id") or f"chatcmpl-{uuid.uuid4().hex[:24]}"
                model_id = payload.get("model") or upstream
                choices = payload.get("choices") or []
                first = choices[0] if choices else {}
                msg = first.get("message") or {}
                tool_calls = msg.get("tool_calls") or []
                # mlx-vlm returns finish_reason="stop" even when tool_calls are
                # present. OpenAI spec says it must be "tool_calls" so that
                # agent loops (e.g. Companion) trigger tool execution.
                fixes_finish = _has_quirk(prov_id, "finish_reason_stop_with_tools")
                finish = (
                    "tool_calls" if (tool_calls and fixes_finish)
                    else (first.get("finish_reason") or "stop")
                )
                delta = {
                    "role": msg.get("role") or "assistant",
                    "content": msg.get("content") or "",
                }
                if tool_calls:
                    delta["tool_calls"] = tool_calls
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": delta,
                                  "finish_reason": None}],
                }
                if payload.get("usage"):
                    chunk["usage"] = payload["usage"]
                yield ("data: " + json.dumps(chunk) + "\n\n").encode()
                # Final chunk with finish_reason.
                final = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {},
                                  "finish_reason": finish}],
                }
                yield ("data: " + json.dumps(final) + "\n\n").encode()
                yield b"data: [DONE]\n\n"

        return StreamingResponse(gen_unary(), media_type="text/event-stream")

    if is_stream:
        async def gen() -> AsyncIterator[bytes]:
            timeout = httpx.Timeout(60.0, read=None)  # no read timeout for SSE
            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    async with client.stream(
                        "POST", url, headers=headers, json=fwd,
                    ) as r:
                        if r.status_code >= 400:
                            txt = (await r.aread()).decode("utf-8", "ignore")
                            err_payload = {"error": {"message": txt[:300],
                                                      "code": r.status_code,
                                                      "provider": prov_id}}
                            yield ("data: " + json.dumps(err_payload) + "\n\n").encode()
                            return
                        # Forward bytes verbatim — keeps upstream usage fields
                        # (`prompt_tokens_details.cached_tokens` from compatible upstreams,
                        # `cache_read_input_tokens` from Anthropic /v1/messages)
                        # transparent to the client. Companion StatsRow then
                        # surfaces the prefix-cache win without extra work.
                        async for chunk in r.aiter_bytes():
                            if chunk:
                                yield chunk
                except Exception as e:
                    err_payload = {"error": {"message": str(e)[:300], "provider": prov_id}}
                    yield ("data: " + json.dumps(err_payload) + "\n\n").encode()

        return StreamingResponse(gen(), media_type="text/event-stream")

    # Non-streaming
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.post(url, headers=headers, json=fwd)
        except Exception as e:
            raise HTTPException(502, f"upstream {prov_id} unreachable: {e}")
        try:
            payload = r.json()
        except Exception:
            raise HTTPException(502, f"upstream {prov_id} returned non-JSON (status {r.status_code})")
        # Normalise finish_reason="tool_calls" for providers that return "stop"
        # even when tool_calls are present (breaks agent loops). Quirk
        # declared in PROVIDER_QUIRKS for clients to see.
        if _has_quirk(prov_id, "finish_reason_stop_with_tools") and isinstance(payload, dict):
            for ch in (payload.get("choices") or []):
                msg = ch.get("message") if isinstance(ch, dict) else None
                if isinstance(msg, dict) and msg.get("tool_calls"):
                    ch["finish_reason"] = "tool_calls"
        return JSONResponse(payload, status_code=r.status_code)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-protocol translation: OpenAI ⇄ Anthropic
# ──────────────────────────────────────────────────────────────────────────────
# Companion (and most clients) speak OpenAI /v1/chat/completions. When the
# user picks an alias on an anthropic-protocol provider (api.anthropic.com),
# we translate the OpenAI request to Anthropic /v1/messages shape, call the
# upstream, then translate the response back. End result: any client → any
# backend, regardless of wire format.
#
# Translation tables:
#   request:    OpenAI                          ↔  Anthropic
#               messages[role=system]           →  system (string)
#               messages[role=user/assistant]   ↔  messages
#               messages[role=tool]             →  user msg w/ tool_result block
#               assistant.tool_calls            →  assistant content tool_use blocks
#               tools[].function.parameters     ↔  tools[].input_schema
#               tool_choice                     ↔  tool_choice (shape differs)
#               stop                            ↔  stop_sequences
#               max_tokens (optional)           →  max_tokens (required, default 4096)
#
#   response:   Anthropic                       ↔  OpenAI
#               content blocks (text)           →  message.content (string)
#               content blocks (tool_use)       →  message.tool_calls[]
#               stop_reason                     ↔  finish_reason
#                  end_turn → stop
#                  tool_use → tool_calls
#                  max_tokens → length
#               usage.input_tokens              →  usage.prompt_tokens
#               usage.output_tokens             →  usage.completion_tokens
#               usage.cache_read_input_tokens   →  usage.prompt_tokens_details.cached_tokens
#
# Streaming differs more substantially (Anthropic uses typed event names like
# message_start / content_block_delta, OpenAI uses plain ChatCompletionChunk
# objects). See `_translate_anthropic_sse_to_openai` below.

_ANTC_STOP_REASON_TO_OPENAI = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def _openai_to_anthropic_request(body: dict, upstream_model: str) -> dict:
    """Build an Anthropic /v1/messages body from an OpenAI ChatCompletion body.

    Best-effort: unknown OpenAI params (frequency_penalty, presence_penalty,
    logit_bias, response_format, …) are dropped — Anthropic doesn't have
    equivalents. Clients that need them stick to OpenAI-protocol providers.
    """
    system_parts: list[str] = []
    msgs: list[dict] = []
    for m in body.get("messages") or []:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        system_parts.append(b.get("text", ""))
            continue
        if role == "tool":
            # OpenAI tool result → Anthropic user message with tool_result block.
            msgs.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id") or "",
                    "content": content if isinstance(content, str) else json.dumps(content),
                }],
            })
            continue
        if role == "assistant" and m.get("tool_calls"):
            # OpenAI assistant with tool_calls → Anthropic assistant with mixed
            # text + tool_use content blocks. Preserve any leading text.
            blocks: list[dict] = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        blocks.append({"type": "text", "text": b.get("text", "")})
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args_raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or ("toolu_" + uuid.uuid4().hex[:24]),
                    "name": fn.get("name") or "",
                    "input": args if isinstance(args, dict) else {},
                })
            msgs.append({"role": "assistant", "content": blocks})
            continue
        if role in ("user", "assistant"):
            # Plain text message — Anthropic accepts string content for these.
            if isinstance(content, str):
                msgs.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Pass through multimodal content blocks (text/image). The
                # OpenAI image_url shape needs to be massaged to Anthropic's
                # image source shape, but that's vision-territory and out of
                # scope for the chat path. Keep what's compatible.
                msgs.append({"role": role, "content": content})
            else:
                msgs.append({"role": role, "content": ""})

    out: dict = {
        "model": upstream_model,
        "messages": msgs,
        # Anthropic REQUIRES max_tokens. OpenAI defaults to model max when omitted.
        "max_tokens": int(body.get("max_tokens") or 4096),
    }
    if system_parts:
        out["system"] = "\n\n".join(system_parts)
    for k in ("temperature", "top_p", "top_k", "stream", "metadata"):
        if body.get(k) is not None:
            out[k] = body[k]
    if body.get("stop") is not None:
        s = body["stop"]
        out["stop_sequences"] = s if isinstance(s, list) else [s]

    if body.get("tools"):
        antc_tools: list[dict] = []
        for t in body["tools"]:
            fn = (t or {}).get("function") if isinstance(t, dict) else None
            if not fn:
                continue
            antc_tools.append({
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        if antc_tools:
            out["tools"] = antc_tools

    tc = body.get("tool_choice")
    if tc:
        if tc == "auto":
            out["tool_choice"] = {"type": "auto"}
        elif tc == "none":
            out["tool_choice"] = {"type": "none"}
        elif tc == "required":
            out["tool_choice"] = {"type": "any"}
        elif isinstance(tc, dict) and tc.get("type") == "function":
            name = ((tc.get("function") or {}).get("name")) or ""
            if name:
                out["tool_choice"] = {"type": "tool", "name": name}
    return out


def _anthropic_to_openai_response(payload: dict, openai_model: str) -> dict:
    """Translate a non-stream Anthropic /v1/messages response → OpenAI shape."""
    blocks = payload.get("content") or []
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            text_parts.append(b.get("text", "") or "")
        elif t == "tool_use":
            tool_calls.append({
                "id": b.get("id") or ("call_" + uuid.uuid4().hex[:24]),
                "type": "function",
                "function": {
                    "name": b.get("name") or "",
                    "arguments": json.dumps(b.get("input") or {}),
                },
            })
        # thinking blocks dropped — OpenAI shape has no equivalent that
        # clients reliably parse. Future: surface as reasoning_content.

    msg: dict = {"role": "assistant", "content": ("".join(text_parts) or None)}
    finish = _ANTC_STOP_REASON_TO_OPENAI.get(payload.get("stop_reason") or "", "stop")
    if tool_calls:
        msg["tool_calls"] = tool_calls
        finish = "tool_calls"

    u = payload.get("usage") or {}
    prompt_t = int(u.get("input_tokens") or 0)
    completion_t = int(u.get("output_tokens") or 0)
    usage: dict = {
        "prompt_tokens": prompt_t,
        "completion_tokens": completion_t,
        "total_tokens": prompt_t + completion_t,
    }
    cached = int(u.get("cache_read_input_tokens") or 0)
    if cached:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}

    return {
        "id": payload.get("id") or ("chatcmpl-" + uuid.uuid4().hex[:24]),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": openai_model,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": finish,
        }],
        "usage": usage,
    }


async def _translate_anthropic_sse_to_openai(
    anth_stream: AsyncIterator[bytes], openai_model: str
) -> AsyncIterator[bytes]:
    """Consume Anthropic SSE events, emit OpenAI ChatCompletionChunk SSE bytes.

    Anthropic event sequence (simplified):
      message_start (delta has role + usage with input_tokens)
      content_block_start[idx]  (type=text or tool_use)
      content_block_delta[idx]  (text_delta.text  or  input_json_delta.partial_json)
      content_block_stop[idx]
      … more blocks …
      message_delta             (delta.stop_reason + usage update with output_tokens)
      message_stop

    OpenAI emits a stream of chat.completion.chunk objects. We map:
      message_start          → first chunk with {role:"assistant"}
      text_delta             → chunk with delta.content = text
      input_json_delta       → chunk with delta.tool_calls[{index, function.arguments=partial}]
      tool_use block_start   → chunk with delta.tool_calls[{index, id, function.name, type}]
      message_delta          → final chunk with finish_reason + usage (if include_usage)
    """
    chat_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    buf = b""
    # Track per-content-block context. Anthropic indexes blocks; OpenAI uses
    # the same index for tool_calls. Text blocks emit content deltas only.
    block_kind: dict[int, str] = {}      # idx → "text" | "tool_use"
    tool_call_meta: dict[int, dict] = {} # idx → {id, name}
    final_usage: Optional[dict] = None
    final_stop: Optional[str] = None
    started = False

    def emit_chunk(delta: dict, finish: Optional[str] = None,
                   usage: Optional[dict] = None) -> bytes:
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": openai_model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage is not None:
            chunk["usage"] = usage
        return ("data: " + json.dumps(chunk) + "\n\n").encode()

    try:
        async for chunk_bytes in anth_stream:
            buf += chunk_bytes
            # SSE events end with \n\n. Process complete events; keep the tail.
            while b"\n\n" in buf:
                raw, buf = buf.split(b"\n\n", 1)
                event_type = None
                data_lines: list[str] = []
                for line in raw.split(b"\n"):
                    s = line.decode("utf-8", "ignore")
                    if s.startswith("event:"):
                        event_type = s[6:].strip()
                    elif s.startswith("data:"):
                        data_lines.append(s[5:].strip())
                if not data_lines:
                    continue
                data_str = "\n".join(data_lines)
                try:
                    ev = json.loads(data_str)
                except Exception:
                    continue
                etype = event_type or ev.get("type")

                if etype == "message_start":
                    if not started:
                        yield emit_chunk({"role": "assistant"})
                        started = True
                    # Capture initial usage (input_tokens). Output completes
                    # later in message_delta.
                    msg = ev.get("message") or {}
                    u = msg.get("usage") or {}
                    if u:
                        final_usage = {
                            "prompt_tokens": int(u.get("input_tokens") or 0),
                            "completion_tokens": 0,
                            "total_tokens": int(u.get("input_tokens") or 0),
                        }
                        cached = int(u.get("cache_read_input_tokens") or 0)
                        if cached:
                            final_usage["prompt_tokens_details"] = {"cached_tokens": cached}

                elif etype == "content_block_start":
                    idx = ev.get("index", 0)
                    block = ev.get("content_block") or {}
                    if block.get("type") == "text":
                        block_kind[idx] = "text"
                    elif block.get("type") == "tool_use":
                        block_kind[idx] = "tool_use"
                        tool_call_meta[idx] = {
                            "id": block.get("id") or ("call_" + uuid.uuid4().hex[:24]),
                            "name": block.get("name") or "",
                        }
                        # Emit a "header" chunk announcing the tool call.
                        yield emit_chunk({
                            "tool_calls": [{
                                "index": idx,
                                "id": tool_call_meta[idx]["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_call_meta[idx]["name"],
                                    "arguments": "",
                                },
                            }],
                        })

                elif etype == "content_block_delta":
                    idx = ev.get("index", 0)
                    delta = ev.get("delta") or {}
                    if block_kind.get(idx) == "text" and delta.get("type") == "text_delta":
                        yield emit_chunk({"content": delta.get("text", "")})
                    elif block_kind.get(idx) == "tool_use" and delta.get("type") == "input_json_delta":
                        yield emit_chunk({
                            "tool_calls": [{
                                "index": idx,
                                "function": {"arguments": delta.get("partial_json", "")},
                            }],
                        })

                elif etype == "content_block_stop":
                    # Nothing to emit — OpenAI doesn't bracket blocks.
                    pass

                elif etype == "message_delta":
                    d = ev.get("delta") or {}
                    sr = d.get("stop_reason")
                    if sr:
                        final_stop = _ANTC_STOP_REASON_TO_OPENAI.get(sr, "stop")
                        if any(k == "tool_use" for k in block_kind.values()):
                            final_stop = "tool_calls"
                    u = ev.get("usage") or {}
                    if u and final_usage is not None:
                        out_t = int(u.get("output_tokens") or 0)
                        final_usage["completion_tokens"] = out_t
                        final_usage["total_tokens"] = final_usage["prompt_tokens"] + out_t

                elif etype == "message_stop":
                    # Final chunk: empty delta + finish_reason + usage.
                    yield emit_chunk({}, finish=(final_stop or "stop"), usage=final_usage)
                    yield b"data: [DONE]\n\n"
                    return

                elif etype == "error":
                    err = ev.get("error") or {}
                    err_chunk = {"error": {
                        "message": err.get("message") or "anthropic upstream error",
                        "type": err.get("type") or "upstream_error",
                    }}
                    yield ("data: " + json.dumps(err_chunk) + "\n\n").encode()
                    return

                # ping / unknown → ignore
    except Exception as e:
        err_chunk = {"error": {"message": f"translation error: {e}",
                                "type": "internal_error"}}
        yield ("data: " + json.dumps(err_chunk) + "\n\n").encode()
        return

    # Stream ended without a message_stop — emit a graceful close.
    if started:
        yield emit_chunk({}, finish=(final_stop or "stop"), usage=final_usage)
    yield b"data: [DONE]\n\n"


async def _proxy_chat_completion_via_anthropic(prov_id: str, prov: dict,
                                                  entry: dict, body: dict):
    """OpenAI /v1/chat/completions client → Anthropic /v1/messages upstream.

    Wraps the request translation, upstream call, and response translation.
    Same auth contract as `_proxy_anthropic_messages` (x-api-key +
    anthropic-version).
    """
    api_key = _cloud_provider_key(prov)
    if not api_key:
        raise HTTPException(401, f"provider '{prov_id}' has no API key set")
    upstream_model = entry.get("upstream")
    if not upstream_model:
        raise HTTPException(500, f"cloud alias '{entry.get('alias')}' has no upstream model id")

    antc_body = _openai_to_anthropic_request(body, upstream_model)
    is_stream = bool(body.get("stream"))
    # Mirror stream flag onto the anthropic request body.
    antc_body["stream"] = is_stream

    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    url = f"{prov['api_base'].rstrip('/')}/v1/messages"
    # The visible alias the client sent — we echo it back in the OpenAI
    # response's `model` field so they see what they asked for, not the
    # upstream-native id.
    openai_model = entry.get("alias") or body.get("model") or upstream_model

    if is_stream:
        async def gen() -> AsyncIterator[bytes]:
            timeout = httpx.Timeout(60.0, read=None)
            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    async with client.stream("POST", url, headers=headers, json=antc_body) as r:
                        if r.status_code >= 400:
                            txt = (await r.aread()).decode("utf-8", "ignore")
                            err = {"error": {"message": txt[:300],
                                             "code": r.status_code,
                                             "provider": prov_id}}
                            yield ("data: " + json.dumps(err) + "\n\n").encode()
                            return
                        async for out_chunk in _translate_anthropic_sse_to_openai(
                            r.aiter_bytes(), openai_model
                        ):
                            yield out_chunk
                except Exception as e:
                    err = {"error": {"message": str(e)[:300], "provider": prov_id}}
                    yield ("data: " + json.dumps(err) + "\n\n").encode()
        return StreamingResponse(gen(), media_type="text/event-stream")

    # Non-streaming
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.post(url, headers=headers, json=antc_body)
        except Exception as e:
            raise HTTPException(502, f"upstream {prov_id} unreachable: {e}")
        try:
            payload = r.json()
        except Exception:
            raise HTTPException(502, f"upstream {prov_id} returned non-JSON (status {r.status_code})")
        if r.status_code >= 400:
            # Forward upstream error in OpenAI-ish error shape — clients
            # already know how to display this.
            return JSONResponse(
                {"error": payload.get("error") or {"message": str(payload), "type": "upstream_error"}},
                status_code=r.status_code,
            )
        return JSONResponse(_anthropic_to_openai_response(payload, openai_model))


async def _proxy_anthropic_messages(prov_id: str, prov: dict, entry: dict,
                                     req: "AnthropicMessagesRequest",
                                     request: Request):
    """Proxy a /v1/messages call to an Anthropic-protocol upstream.

    Today this is api.anthropic.com — the only upstream that speaks
    /v1/messages natively. We forward the body almost verbatim (rewriting
    `model` to the upstream's native id), set the Anthropic auth header
    contract (`x-api-key` + `anthropic-version`), and stream-pass-through
    so cache fields (`cache_read_input_tokens`, `cache_creation_input_tokens`)
    reach the client transparently — same passthrough story as the
    /v1/chat/completions cloud proxy.
    """
    api_key = _cloud_provider_key(prov)
    if not api_key:
        raise HTTPException(401, f"provider '{prov_id}' has no API key set")
    upstream_model = entry.get("upstream")
    if not upstream_model:
        raise HTTPException(500, f"cloud alias '{entry.get('alias')}' has no upstream model id")

    # Build the upstream body. Pydantic gives us a clean dict; we just swap
    # the model id and drop our internal fields.
    body = req.model_dump(exclude_none=True)
    body["model"] = upstream_model
    body.pop("metadata", None)  # Anthropic accepts it, but we strip session-id hints

    # Anthropic uses `x-api-key` + `anthropic-version` rather than Bearer auth.
    # `anthropic-version` is required; pick a recent stable date that's been
    # GA for months so we don't break on key rotation.
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    # Pass through anthropic-beta if the client sent one (extended thinking,
    # 1M context, etc.). Clients know better than us which betas they want.
    beta = request.headers.get("anthropic-beta")
    if beta:
        headers["anthropic-beta"] = beta

    url = f"{prov['api_base'].rstrip('/')}/v1/messages"
    is_stream = bool(body.get("stream"))

    if is_stream:
        async def gen() -> AsyncIterator[bytes]:
            timeout = httpx.Timeout(60.0, read=None)
            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    async with client.stream("POST", url, headers=headers, json=body) as r:
                        if r.status_code >= 400:
                            txt = (await r.aread()).decode("utf-8", "ignore")
                            err = {"type": "error",
                                   "error": {"type": "upstream_error",
                                             "message": txt[:300],
                                             "code": r.status_code,
                                             "provider": prov_id}}
                            yield ("event: error\ndata: " + json.dumps(err) + "\n\n").encode()
                            return
                        async for chunk in r.aiter_bytes():
                            if chunk:
                                yield chunk
                except Exception as e:
                    err = {"type": "error",
                           "error": {"type": "upstream_unreachable",
                                     "message": str(e)[:300],
                                     "provider": prov_id}}
                    yield ("event: error\ndata: " + json.dumps(err) + "\n\n").encode()

        return StreamingResponse(gen(), media_type="text/event-stream")

    # Non-streaming
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.post(url, headers=headers, json=body)
        except Exception as e:
            raise HTTPException(502, f"upstream {prov_id} unreachable: {e}")
        try:
            payload = r.json()
        except Exception:
            raise HTTPException(502, f"upstream {prov_id} returned non-JSON (status {r.status_code})")
        return JSONResponse(payload, status_code=r.status_code)


async def _list_upstream_models(prov: dict) -> list[dict]:
    """Probe the provider's /v1/models endpoint to help the user pick which
    models to publish. Returns the raw upstream list (may be empty if the
    provider rate-limits us or doesn't expose /models).

    Auth optional — for keyed providers we send the bearer, for local
    sidecars (mlx-vlm etc.) we probe anonymously.
    """
    api_key = _cloud_provider_key(prov)
    url = f"{prov['api_base'].rstrip('/')}/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, headers=headers)
            if r.status_code == 200:
                d = r.json()
                return d.get("data", []) if isinstance(d, dict) else []
    except Exception:
        pass
    return []


def _cloud_entries_for_v1_models() -> list[dict]:
    """Return /v1/models entries for every published cloud alias, with x_odyssai
    composed from the declared caps + cloud defaults."""
    out: list[dict] = []
    providers = get_cloud_providers()
    for prov_id, prov in providers.items():
        if not _provider_enabled(prov):
            continue   # disabled providers don't publish aliases
        has_key = _cloud_provider_key(prov) is not None
        for entry in (prov.get("published") or []):
            alias = entry.get("alias")
            if not alias:
                continue
            declared = entry.get("caps") or {}
            # Backend quirks: declared in PROVIDER_QUIRKS up top. Clients can
            # read this to apply their own defensive logic if needed (we already
            # work them around in the proxy).
            quirks = list(PROVIDER_QUIRKS.get(prov_id, []))
            caps = {
                "loaded": has_key,        # available if key is set
                "loading": False,
                "pool": prov_id,
                "backend": "http-proxy",
                "backend_quirks": quirks,
                "nodes": None,
                "context_length": declared.get("context_length"),
                "max_output_tokens": declared.get("max_output_tokens"),
                "quantization": declared.get("quantization"),
                "size_bytes": None,
                "modalities": declared.get("modalities") or ["text"],
                "family": declared.get("family"),
                "supports_tools": declared.get("supports_tools"),
                "supports_vision": declared.get("supports_vision"),
                "supports_json_mode": declared.get("supports_json_mode"),
                "supports_streaming": True,
                "estimated_tps": declared.get("estimated_tps"),
                "estimated_load_s": 0,    # cloud = no load
                "warm": has_key,
                "kv_cache_q8": False,
                "admin_loadable": False,
                "upstream": entry.get("upstream"),
            }
            out.append({
                "id": alias,
                "object": "model",
                "created": _now(),
                "owned_by": f"odyssai-cloud-{prov_id}",
                "x_odyssai": caps,
            })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Admin: cloud providers config
# ──────────────────────────────────────────────────────────────────────────────
class CloudPublishedEntry(BaseModel):
    alias: str
    upstream: str
    caps: Optional[dict] = None


class CloudProviderUpdate(BaseModel):
    api_base: Optional[str] = None
    api_key: Optional[str] = None        # direct value, stored in cluster-config.json
    api_key_env: Optional[str] = None    # optional env var fallback
    clear_api_key: Optional[bool] = None # if true, wipe stored api_key
    published: Optional[list[CloudPublishedEntry]] = None
    enabled: Optional[bool] = None       # toggle without delete; defaults to True
    # Wire protocol the upstream speaks. "openai" = /v1/chat/completions
    # (most clouds + local OpenAI-compatible). "anthropic" = /v1/messages
    # (api.anthropic.com only, today). Determines which proxy path runs.
    # Defaults to "openai" so existing providers keep working.
    protocol: Optional[str] = None       # "openai" | "anthropic"


def _provider_enabled(prov: dict) -> bool:
    """A provider is enabled unless explicitly set to False. Default True so
    existing config (which has no `enabled` field) keeps working."""
    return prov.get("enabled", True) is not False


def _provider_protocol(prov: dict) -> str:
    """Wire protocol the upstream speaks. Default 'openai' for backward
    compat with all existing providers (openrouter, openai, anthropic, …)."""
    p = (prov.get("protocol") or "openai").lower()
    return p if p in ("openai", "anthropic") else "openai"


def _redact_provider(prov_id: str, prov: dict) -> dict:
    """Build a safe-to-return view of a provider. NEVER includes the api_key."""
    return {
        "id": prov_id,
        "api_base": prov.get("api_base"),
        "api_key_env": prov.get("api_key_env"),
        "api_key_set": _cloud_provider_key(prov) is not None,
        "api_key_source": (
            "config" if (prov.get("api_key") or "").strip()
            else ("env" if prov.get("api_key_env") and (env_value_by_name(prov["api_key_env"]) or "").strip()
                  else "none")
        ),
        "published": prov.get("published") or [],
        "enabled": _provider_enabled(prov),
        "protocol": _provider_protocol(prov),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Server-wide settings (admin-editable defaults)
# ──────────────────────────────────────────────────────────────────────────────
# Persistent toggles applied across all clusters + cloud providers. Today this
# only covers `enable_thinking_default`, but the shape supports adding more
# toggles (per-pool TTL default, cache hints, etc.) without a schema bump.
class ServerSettingsUpdate(BaseModel):
    enable_thinking_default: Optional[bool] = None
    pool_ttl_seconds_default: Optional[int] = None
    # KV cache controls — set globally, override per request via the load
    # endpoint. Q8 default ON for big-MoE workloads (Qwen397B, Hy3, GLM-5.1)
    # where 128k context cache at fp16 doesn't fit alongside the model. Set
    # `false` to revert to fp16 for quality-critical eval runs.
    kv_q8_default: Optional[bool] = None
    # System prefix prewarm — text that's prefilled once at model load into a
    # shared KV cache. New conversations whose prompt starts with these tokens
    # skip the prefill cost for this prefix (cross-session TTFT win). Typical
    # content: the system message + tools list + any constant wiki/context
    # injected by Companion. Empty disables prewarm.
    system_prefix_text: Optional[str] = None


def get_kv_q8_default() -> bool:
    """Cluster-wide Q8 KV cache default. Honors cluster-config.json
    `settings.kv_q8_default`; falls back to env `RUNNER_KV_Q8` (`1`/`0`);
    final default True (halves cache memory for big-context MoE workloads
    we run today). Code paths can call this when the request doesn't
    explicitly set kv_q8."""
    cfg = _load_cluster_config()
    val = (cfg.get("settings") or {}).get("kv_q8_default")
    if val is not None:
        return bool(val)
    env = os.environ.get("RUNNER_KV_Q8")
    if env is not None:
        return env == "1"
    return True


def get_system_prefix_text() -> str:
    """The configured prewarm prefix text. Empty string = prewarm disabled."""
    cfg = _load_cluster_config()
    return str((cfg.get("settings") or {}).get("system_prefix_text") or "")


async def _maybe_auto_prewarm(pool, cluster_name: str) -> None:
    """If a system prefix is configured, prewarm the pool's shared cache.
    Best-effort: failure is logged, never raised — the model is still usable
    without the prewarm, just slower on first turn.

    Called right after a successful load. Runs the prewarm inline (blocks the
    load endpoint) so the API caller knows when the cluster is fully warm —
    UI cleanliness > shaving a few seconds off the load response.
    """
    text = get_system_prefix_text()
    if not text.strip():
        return
    try:
        res = await pool.prewarm(text, kv_q8=get_kv_q8_default(),
                                 timeout_s=300.0)
        if res.get("ok"):
            r = res.get("result") or {}
            sys.stderr.write(
                f"[load] {cluster_name} auto-prewarm OK: "
                f"{r.get('tokens')} tok, "
                f"{(r.get('bytes') or 0) / 1024**3:.2f} GB, "
                f"{r.get('elapsed_s', 0):.1f}s\n"
            )
        else:
            sys.stderr.write(
                f"[load] {cluster_name} auto-prewarm not ok: {res}\n"
            )
    except Exception as e:
        sys.stderr.write(f"[load] {cluster_name} auto-prewarm error: {e}\n")


@app.get("/admin/settings")
async def admin_settings_get():
    cfg = _load_cluster_config()
    s = cfg.get("settings") or {}
    return {
        "enable_thinking_default": get_enable_thinking_default(),
        "pool_ttl_seconds_default": int(s.get("pool_ttl_seconds_default") or 0),
        "kv_q8_default": get_kv_q8_default(),
        "system_prefix_text": get_system_prefix_text(),
    }


@app.post("/admin/restart")
async def admin_restart():
    """Restart the OdyssAI-X FastAPI process. The Docker container's
    `restart: unless-stopped` policy brings us back up immediately. Useful
    when you've edited cluster-config.json on disk or pulled new code via
    `docker cp` and don't want to SSH into the Docker host just to bounce the
    container.

    Exits with code 0 — the container supervisor handles the restart. We
    schedule the exit on the event loop so the HTTP response goes out
    before we tear down.
    """
    async def _bye():
        await asyncio.sleep(0.3)  # let the 200 response flush
        # Use os._exit so we skip atexit hooks that might block on cluster
        # cleanup (we're being restarted, not gracefully shutting down).
        os._exit(0)
    asyncio.create_task(_bye())
    return {"ok": True, "message": "restart scheduled in 300ms"}


@app.put("/admin/settings")
async def admin_settings_update(req: ServerSettingsUpdate):
    with _cluster_config_txn() as cfg:
        s = cfg.get("settings") or {}
        if req.enable_thinking_default is not None:
            s["enable_thinking_default"] = bool(req.enable_thinking_default)
        if req.pool_ttl_seconds_default is not None:
            # 0 disables the TTL sweeper for new pools.
            s["pool_ttl_seconds_default"] = max(0, int(req.pool_ttl_seconds_default))
        if req.kv_q8_default is not None:
            s["kv_q8_default"] = bool(req.kv_q8_default)
        if req.system_prefix_text is not None:
            s["system_prefix_text"] = str(req.system_prefix_text)
        cfg["settings"] = s
    # Propagate to live pools so the change takes effect immediately, not
    # only after next load.
    _apply_default_ttl_to_pools()
    # If the prefix text was edited and a pool is currently loaded, re-prewarm
    # it now so the user doesn't have to unload+reload to see the effect.
    if req.system_prefix_text is not None:
        targets: list[tuple[RunnerPool, str]] = []
        if _pool is not None:
            targets.append((_pool, "nautilus"))
        for cid, alias, pool in list_all_pools():
            targets.append((pool, f"{cid}:{alias}"))
        for pool, name in targets:
            try:
                asyncio.create_task(
                    pool.prewarm(req.system_prefix_text,
                                 kv_q8=get_kv_q8_default()))
                sys.stderr.write(f"[settings] re-prewarm scheduled for {name}\n")
            except Exception as e:
                sys.stderr.write(f"[settings] re-prewarm failed for {name}: {e}\n")
    return await admin_settings_get()


# ── Coeos router config (RFC #63) ─────────────────────────────────────────────

class CoeosConfig(BaseModel):
    # CoeOS config = the "TMB Settings" the operator imports. Everything is data:
    # the taxonomy (axes) AND the per-axis model bindings AND the decider.
    enabled: Optional[bool] = None
    name: Optional[str] = None                   # e.g. "TMB Settings — Best of all (cloud) v0.1"
    regime: Optional[str] = None                 # "local" | "cloud" (informational)
    updated: Optional[str] = None                # settings vintage (monthly refresh)
    note: Optional[str] = None
    decider_model: Optional[str] = None          # the model that CLASSIFIES the request
    default_axis: Optional[str] = None
    axes: Optional[list] = None                  # [{key, label, model(=logical), description?}, …]
    models: Optional[dict] = None                # logical name → {name, endpoint} registry
    cold_boot_autoload: Optional[bool] = None


@app.get("/admin/coeos")
async def admin_coeos_get():
    return get_coeos_config()


@app.get("/admin/coeos/available-models")
async def admin_coeos_available_models():
    return {"data": list_routable_model_ids()}


@app.get("/admin/coeos/decisions")
async def admin_coeos_decisions():
    """Routing decision counts (model × axis × fallback) for operator visibility."""
    return {"decisions": [
        {"model": k[0], "axis": k[1], "fallback": k[2], "count": v}
        for k, v in sorted(_coeos_decisions.items(), key=lambda kv: -kv[1])]}


@app.put("/admin/coeos")
async def admin_coeos_update(req: CoeosConfig):
    # Importing a TMB Settings file = a PUT with {name, decider_model,
    # default_axis, axes, …}. Validate the axes shape + reserved id.
    if req.axes is not None:
        if not isinstance(req.axes, list):
            raise HTTPException(400, detail={"error": "bad_axes",
                "message": "axes must be a list of {key, label, model} objects."})
        seen = set()
        for ax in req.axes:
            if not isinstance(ax, dict) or not ax.get("key"):
                raise HTTPException(400, detail={"error": "bad_axis",
                    "message": "each axis needs a non-empty 'key'."})
            k = str(ax["key"]).strip().lower()
            if k in seen:
                raise HTTPException(400, detail={"error": "dup_axis",
                    "message": f"duplicate axis key: {k!r}."})
            seen.add(k)
            m = ax.get("model")
            if m and str(m).strip().lower() == COEOS_MODEL_ID:
                raise HTTPException(400, detail={"error": "reserved_id",
                    "message": "'coeos' is the router's own id and can't be bound to an axis."})
    with _cluster_config_txn() as cfg:
        c = cfg.get("coeos_config") or {}
        for field in ("enabled", "name", "regime", "updated", "note", "decider_model",
                      "default_axis", "axes", "models", "cold_boot_autoload"):
            val = getattr(req, field)
            if val is not None:
                c[field] = bool(val) if field in ("enabled", "cold_boot_autoload") else val
        cfg["coeos_config"] = c
    return get_coeos_config()


# Provider templates — preset configs for common upstreams. The dashboard
# "Add provider" form offers these as a dropdown so a user can pick e.g.
# "Anthropic direct" and get api_base + protocol + api_key_env pre-filled.
# Order matters for UI: the first item is the default the dropdown selects.
# Generic — no engine-specific entries (vLLM, Ollama, mlx-vlm are
# all OpenAI-compatible; users pick "OpenAI" and point api_base at their
# local host). No site-specific defaults — templates stay portable.
PROVIDER_TEMPLATES = [
    {
        "id": "custom",
        "label": "Custom",
        "api_base": "",
        "protocol": "openai",
        "api_key_env": "",
        "hint": "Any OpenAI-compatible endpoint. Fill the fields by hand.",
    },
    {
        "id": "openai",
        "label": "OpenAI",
        "api_base": "https://api.openai.com/v1",
        "protocol": "openai",
        "api_key_env": "ODYSSAI_X_OPENAI_KEY",
        "hint": "OpenAI's API, or any OpenAI-compatible upstream (vLLM, "
                "Ollama, LM Studio, mlx-vlm…). Replace api_base if pointing to "
                "a local server.",
        # Templates can pre-fill the published aliases list so the user has
        # working models the moment they paste their API key. Edit/extend
        # freely afterwards. alias == upstream means clients call the OpenAI
        # model id directly — no translation hop.
        "default_aliases": [
            {"alias": "gpt-5", "upstream": "gpt-5",
             "caps": {"context_length": 400000, "supports_tools": True}},
            {"alias": "gpt-4o", "upstream": "gpt-4o",
             "caps": {"context_length": 128000, "supports_tools": True,
                      "supports_vision": True}},
            {"alias": "gpt-4o-mini", "upstream": "gpt-4o-mini",
             "caps": {"context_length": 128000, "supports_tools": True}},
        ],
    },
    {
        "id": "anthropic",
        "label": "Anthropic",
        "api_base": "https://api.anthropic.com",
        "protocol": "anthropic",
        "api_key_env": "ODYSSAI_X_ANTHROPIC_KEY",
        "hint": "Anthropic's API. Speaks /v1/messages natively — clients get a "
                "Claude-shape passthrough. Default aliases are the current "
                "Claude tiers; edit if Anthropic releases newer ones.",
        # Identity mapping — alias name == Anthropic's model id, so
        # `model: "claude-haiku-4-5"` in the client just works. Caps from
        # public Anthropic spec at time of writing.
        "default_aliases": [
            {"alias": "claude-haiku-4-5", "upstream": "claude-haiku-4-5",
             "caps": {"context_length": 200000, "supports_tools": True,
                      "supports_vision": True}},
            {"alias": "claude-sonnet-4-6", "upstream": "claude-sonnet-4-6",
             "caps": {"context_length": 1000000, "supports_tools": True,
                      "supports_vision": True}},
            {"alias": "claude-opus-4-7", "upstream": "claude-opus-4-7",
             "caps": {"context_length": 1000000, "supports_tools": True,
                      "supports_vision": True}},
        ],
    },
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "api_base": "https://openrouter.ai/api/v1",
        "protocol": "openai",
        "api_key_env": "ODYSSAI_X_OPENROUTER_KEY",
        "hint": "Aggregator for many cloud models (Anthropic, DeepSeek, Qwen, "
                "Mistral, …). Single key, OpenAI shape. Default aliases use the "
                "`or:` prefix convention — extend with any model from "
                "openrouter.ai/models.",
        "default_aliases": [
            {"alias": "or:claude-haiku", "upstream": "anthropic/claude-haiku-4-5",
             "caps": {"context_length": 200000, "supports_tools": True}},
            {"alias": "or:claude-sonnet", "upstream": "anthropic/claude-sonnet-4-6",
             "caps": {"context_length": 1000000, "supports_tools": True}},
            {"alias": "or:hy3-preview", "upstream": "tencent/hy3-preview",
             "caps": {"context_length": 262144, "supports_tools": True}},
        ],
    },
]


@app.get("/admin/providers/templates")
async def admin_providers_templates():
    """Preset configs for the 'Add provider' UI. Pure metadata — applying
    a template just pre-fills the form; the user can still edit any field."""
    return {"data": PROVIDER_TEMPLATES, "count": len(PROVIDER_TEMPLATES)}


@app.get("/admin/providers")
async def admin_providers_list():
    """List configured cloud providers. The actual API key value is NEVER
    returned — only `api_key_set` (bool) and `api_key_source` ('config'|'env'|'none')."""
    providers = get_cloud_providers()
    out = [_redact_provider(pid, p) for pid, p in providers.items()]
    return {"data": out, "count": len(out)}


@app.put("/admin/providers/{provider_id}")
async def admin_providers_upsert(provider_id: str, req: CloudProviderUpdate):
    """Create or update a cloud provider. Partial updates supported.

    `api_key`: passing a non-empty value stores it in cluster-config.json
    (volume-mounted, host-protected). Passing it as `null`/absent leaves the
    existing key untouched. To remove a stored key, pass `clear_api_key: true`.

    Either `api_key` (config-stored) or `api_key_env` (env var name) must
    resolve to a non-empty value at runtime — but you can save a provider
    with neither and add the key later.
    """
    if not provider_id or not provider_id.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(400, "provider_id must be alphanumeric (- and _ allowed)")
    providers = get_cloud_providers()
    cur = providers.get(provider_id, {})
    if req.api_base is not None:
        cur["api_base"] = req.api_base
    if req.api_key_env is not None:
        # empty string clears the env-var pointer
        cur["api_key_env"] = req.api_key_env or None
        if cur["api_key_env"] is None:
            cur.pop("api_key_env", None)
    if req.clear_api_key:
        cur.pop("api_key", None)
    elif req.api_key is not None and req.api_key.strip():
        cur["api_key"] = req.api_key.strip()
    if req.published is not None:
        cur["published"] = [e.model_dump(exclude_none=True) for e in req.published]
    if req.enabled is not None:
        cur["enabled"] = bool(req.enabled)
    if req.protocol is not None:
        p = req.protocol.lower()
        if p not in ("openai", "anthropic"):
            raise HTTPException(400, "protocol must be 'openai' or 'anthropic'")
        cur["protocol"] = p
    if "api_base" not in cur or not cur["api_base"]:
        raise HTTPException(400, "api_base required")
    providers[provider_id] = cur
    save_cloud_providers(providers)
    return _redact_provider(provider_id, cur)


@app.delete("/admin/providers/{provider_id}")
async def admin_providers_delete(provider_id: str):
    providers = get_cloud_providers()
    if provider_id not in providers:
        raise HTTPException(404, f"unknown provider {provider_id}")
    del providers[provider_id]
    save_cloud_providers(providers)
    return {"ok": True}


@app.post("/admin/providers/{provider_id}/test")
async def admin_providers_test(provider_id: str):
    """Verify the provider is reachable by hitting their /v1/models. Works
    for both keyed providers (OpenRouter, OpenAI) and unkeyed local ones
    (mlx-vlm, internal vLLM, etc.)."""
    providers = get_cloud_providers()
    prov = providers.get(provider_id)
    if not prov:
        raise HTTPException(404, f"unknown provider {provider_id}")
    has_key = _cloud_provider_key(prov) is not None
    models = await _list_upstream_models(prov)
    if not models:
        return {"ok": False, "models_count": 0,
                "error": ("API key not set" if not has_key and prov.get("api_key_env")
                          else "upstream unreachable or empty /models")}
    return {"ok": True, "models_count": len(models),
            "auth_used": has_key,
            "sample": [m.get("id") for m in models[:10]]}


@app.get("/admin/providers/{provider_id}/upstream-models")
async def admin_providers_upstream(provider_id: str):
    """Fetch the provider's full model list so the UI can show a picker."""
    providers = get_cloud_providers()
    prov = providers.get(provider_id)
    if not prov:
        raise HTTPException(404, f"unknown provider {provider_id}")
    return {"data": await _list_upstream_models(prov)}


# ──────────────────────────────────────────────────────────────────────────────
# Crew & pairing — companion clients discover and pair with OdyssAI-X
# ──────────────────────────────────────────────────────────────────────────────
# Discovery model: the operator opens the gate ("Enable discovery"), the host-side
# mDNS watcher advertises OdyssAI-X on `_odyssai-engine._tcp.local.`, the
# first Companion that calls POST /admin/pair gets a crew token, gate
# auto-closes. No PIN — the open gate IS the auth, scoped to LAN.
#
# Crew tokens are NOT access-control on /v1/* (that route is public). They
# serve as client identifiers so the Crew tab can show "Companion @ host
# last seen 30s ago" + revoke.
#
# Storage: cluster-config.json["crew"] = list of entries, ["discovery"] = state
# Plain tokens are returned once at pair time, only SHA256 hash stored.

import hashlib
import secrets

DISCOVERY_TIMEOUT_S = 5 * 60  # safety auto-close after 5 min if nobody pairs


def _hash_token(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def get_crew() -> list[dict]:
    cfg = _load_cluster_config()
    return cfg.get("crew", []) or []


def save_crew(crew: list[dict]) -> None:
    with _cluster_config_txn() as cfg:
        cfg["crew"] = crew


def get_discovery_state() -> dict:
    cfg = _load_cluster_config()
    state = cfg.get("discovery") or {}
    if not isinstance(state, dict):
        state = {}
    # Auto-expire if elapsed
    expires_at = state.get("expires_at")
    if state.get("active") and expires_at and time.time() > expires_at:
        state = {"active": False, "started_at": None, "expires_at": None}
        set_discovery_state(state)
    return {
        "active": bool(state.get("active")),
        "started_at": state.get("started_at"),
        "expires_at": state.get("expires_at"),
    }


def set_discovery_state(state: dict) -> None:
    with _cluster_config_txn() as cfg:
        cfg["discovery"] = state


def find_crew_by_token(plain_token: str) -> Optional[dict]:
    if not plain_token:
        return None
    h = _hash_token(plain_token)
    for entry in get_crew():
        if entry.get("token_hash") == h:
            return entry
    return None


def update_crew_last_seen(crew_id: str) -> None:
    """Bump last_seen for a crew member (best effort, swallow errors)."""
    try:
        crew = get_crew()
        for entry in crew:
            if entry.get("id") == crew_id:
                entry["last_seen"] = time.time()
                save_crew(crew)
                return
    except Exception:
        pass


# Public endpoint for the host-side mDNS watcher to poll.
@app.get("/admin/discovery/state")
async def admin_discovery_state():
    """Return the current discovery state. PUBLIC — the host-side watcher
    polls this without auth to know when to advertise on mDNS."""
    return get_discovery_state()


class DiscoveryEnableRequest(BaseModel):
    duration_s: Optional[int] = None  # default DISCOVERY_TIMEOUT_S


@app.post("/admin/discovery/enable")
async def admin_discovery_enable(req: Optional[DiscoveryEnableRequest] = None):
    """Open the gate. The next /admin/pair within the window auto-pairs and
    closes the gate. Admin token required."""
    duration = (req.duration_s if req and req.duration_s else DISCOVERY_TIMEOUT_S)
    now = time.time()
    state = {
        "active": True,
        "started_at": now,
        "expires_at": now + duration,
    }
    set_discovery_state(state)
    sys.stderr.write(f"[discovery] gate OPEN for {duration}s\n")
    return state


@app.post("/admin/discovery/disable")
async def admin_discovery_disable():
    """Force-close the gate. Admin token required."""
    set_discovery_state({"active": False, "started_at": None, "expires_at": None})
    sys.stderr.write("[discovery] gate CLOSED (manual)\n")
    return get_discovery_state()


class PairRequest(BaseModel):
    client_id: str
    client_name: str
    user_label: Optional[str] = None


@app.post("/admin/pair")
async def admin_pair(req: PairRequest):
    """Pair a new client. Only callable while discovery is active. Returns
    a crew token (plaintext, ONCE) and the engine metadata for auto-config."""
    state = get_discovery_state()
    if not state.get("active"):
        raise HTTPException(403, "discovery gate is closed — ask the operator to enable it")

    if not req.client_id or not req.client_name:
        raise HTTPException(400, "client_id and client_name required")

    # Generate token
    plain = "crew_" + secrets.token_urlsafe(24)
    crew_id = "crew-" + secrets.token_hex(8)
    now = time.time()
    entry = {
        "id": crew_id,
        "client_id": req.client_id,
        "client_name": req.client_name,
        "user_label": req.user_label,
        "token_hash": _hash_token(plain),
        "token_preview": plain[-6:],
        "paired_at": now,
        "last_seen": now,
        "scopes": ["chat"],
    }
    crew = get_crew()
    # Dedup by client_id: if same client re-pairs, replace its old entry
    crew = [c for c in crew if c.get("client_id") != req.client_id]
    crew.append(entry)
    save_crew(crew)

    # Auto-close the gate
    set_discovery_state({"active": False, "started_at": None, "expires_at": None})
    sys.stderr.write(
        f"[discovery] gate CLOSED (pair success: {req.client_name} → {crew_id})\n"
    )

    return {
        "ok": True,
        "token": plain,                      # ONCE — Companion must persist
        "crew_id": crew_id,
        "engine": _engine_metadata(),
        "expires": None,                     # crew tokens don't expire (revoke to invalidate)
    }


@app.get("/admin/crew")
async def admin_crew_list():
    """List paired companions. Token hashes never returned, only previews."""
    crew = get_crew()
    out = []
    for entry in crew:
        out.append({
            "id": entry.get("id"),
            "client_id": entry.get("client_id"),
            "client_name": entry.get("client_name"),
            "user_label": entry.get("user_label"),
            "token_preview": entry.get("token_preview"),
            "paired_at": entry.get("paired_at"),
            "last_seen": entry.get("last_seen"),
            "scopes": entry.get("scopes") or [],
        })
    return {"data": out, "count": len(out)}


@app.delete("/admin/crew/self")
async def admin_crew_self_revoke(request: Request):
    """Companion-initiated disconnect: revokes the entry matching its bearer.
    Requires a crew bearer token (not admin)."""
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(401, "bearer token required")
    plain = auth[7:].strip()
    entry = find_crew_by_token(plain)
    if not entry:
        raise HTTPException(401, "unknown crew token")
    crew = get_crew()
    crew = [c for c in crew if c.get("id") != entry["id"]]
    save_crew(crew)
    sys.stderr.write(f"[crew] self-revoked {entry['id']}\n")
    return {"ok": True, "revoked": entry["id"]}


@app.delete("/admin/crew/{crew_id}")
async def admin_crew_revoke(crew_id: str):
    """Revoke a crew member. Admin token required."""
    crew = get_crew()
    if not any(c.get("id") == crew_id for c in crew):
        raise HTTPException(404, f"unknown crew member {crew_id}")
    crew = [c for c in crew if c.get("id") != crew_id]
    save_crew(crew)
    sys.stderr.write(f"[crew] revoked {crew_id}\n")
    return {"ok": True, "revoked": crew_id}


# Discover models on each cluster's master (cached briefly).
_discovery_cache: dict = {"data": None, "ts": 0.0}
_DISCOVERY_TTL = 60.0


def _short_model_name(path: str, models_dir: str) -> str:
    """Display id for a model — strips the provider/org prefix so clients
    show just the model name. The provider is implementation noise; users
    care about the model name + quant + family, not who packaged it.

    Layout-aware:
      `<md>/inferencerlabs/Hy3-preview-MLX-9bit`     → `Hy3-preview-MLX-9bit`
      `<md>/inferencerlabs--Hy3-preview-MLX-9bit`    → `Hy3-preview-MLX-9bit`
      `<md>/Hy3-preview-MLX-9bit`                    → `Hy3-preview-MLX-9bit`
      anything not under models_dir                  → basename fallback

    Collision risk (two providers shipping the same model name) is handled
    by the caller via `_dedupe_with_provider`.
    """
    if not path:
        return path
    md = (models_dir or "").rstrip("/")
    if md and path.startswith(md + "/"):
        rel = path[len(md) + 1:]
        # 2-level: 'provider/model' → 'model'
        if "/" in rel:
            return rel.split("/", 1)[1]
        # 1-level flat: 'provider--model' → 'model'
        if "--" in rel:
            return rel.split("--", 1)[1]
        return rel
    return path.split("/")[-1]


def _provider_of(path: str, models_dir: str) -> str:
    """Inverse of `_short_model_name` for collision disambiguation —
    returns the provider/org segment ('inferencerlabs', 'mlx-community',
    'kernelpool', …) or '' if not derivable."""
    if not path:
        return ""
    md = (models_dir or "").rstrip("/")
    if not (md and path.startswith(md + "/")):
        return ""
    rel = path[len(md) + 1:]
    if "/" in rel:
        return rel.split("/", 1)[0]
    if "--" in rel:
        return rel.split("--", 1)[0]
    return ""


async def _discover_unloaded_models() -> list[tuple[str, str, Optional[int]]]:
    """Return [(model_id, cluster, size_bytes?), ...] for models present on the
    master of each cluster, excluding the ones currently loaded. Cached 60s.

    `model_id` is just the model name (provider stripped) — clients show
    `Hy3-preview-MLX-9bit` instead of `inferencerlabs/Hy3-preview-MLX-9bit`.
    When two providers ship a model with the same name, we keep the
    provider on the losing entry (`provider/model`) so the user can still
    tell them apart. Rare in practice but safe.
    """
    now = time.time()
    if _discovery_cache["data"] is not None and (now - _discovery_cache["ts"]) < _DISCOVERY_TTL:
        return _discovery_cache["data"]
    loaded: set[str] = set()
    if _pool is not None:
        loaded.add(_short_model_name(_pool.model, models_dir_for("nautilus")))
    for cid, _alias, p in list_all_pools():
        loaded.add(_short_model_name(p.model, models_dir_for(cid)))

    # First pass: gather every (path, cluster, mdir) candidate across every
    # configured cluster. Dedupe by absolute path — clusters that share a
    # models_dir mount expose the same model and that isn't a collision.
    raw: list[tuple[str, str, str]] = []
    seen_paths: set[str] = set()
    for cluster in DEFAULT_CLUSTER_DEFS.keys():
        topo = build_topology(cluster, 1)
        if not topo:
            continue
        ssh = topo[0]["ssh"]
        mdir = models_dir_for(cluster)
        try:
            models = await asyncio.to_thread(discover_models_on_node, ssh, mdir)
        except Exception:
            models = []
        for m in models:
            if m in seen_paths:
                continue
            seen_paths.add(m)
            raw.append((m, cluster, mdir))

    # True collisions only: two distinct paths give the same short name
    # (e.g. inferencerlabs/Qwen3.5-X-MLX-9bit and mlx-community/Qwen3.5-X-MLX-9bit
    # both stripping to `Qwen3.5-X-MLX-9bit`). Prefix the provider on both
    # so the user can tell them apart.
    short_counts: dict[str, int] = {}
    for m, _c, mdir in raw:
        s = _short_model_name(m, mdir)
        short_counts[s] = short_counts.get(s, 0) + 1

    found: list[tuple[str, str, Optional[int]]] = []
    seen: set[str] = set()
    for m, cluster, mdir in raw:
        short = _short_model_name(m, mdir)
        if short_counts.get(short, 0) > 1:
            prov = _provider_of(m, mdir)
            if prov:
                short = f"{prov}/{short}"
        if short in loaded or short in seen:
            continue
        seen.add(short)
        found.append((short, cluster, None))
    _discovery_cache["data"] = found
    _discovery_cache["ts"] = now
    return found


@app.get("/v1/models")
async def list_models(include_unloaded: bool = False):
    """OpenAI-compatible model listing.

    **Default behaviour (2026-05-18+)**: only **currently servable** models
    are advertised — loaded pools (default / nautilus aliases) and
    published cloud providers. Models that exist on disk but aren't loaded
    are NOT exposed as model_ids, because:

      1. Auto-swap was never implemented (only routing to already-loaded
         pools — see `_route_pool`). Publishing unloaded models implied
         a capability we didn't have.
      2. With multi-pool loading (a model per node-subset on a cluster),
         the operator decides upfront what's hot. Inventory belongs on
         `/admin/{cluster}/models`, not `/v1/models`.
      3. Field note (2026-05-18 night): "on publie les modeles loaded seulement,
         on enlève l'auto-swap, on peut load plusieurs modèles."

    `?include_unloaded=true` restores the legacy listing for tools that
    want a full inventory view (Companion's "advanced mode" model picker
    can opt in here). Each unloaded entry carries `x_odyssai.ready=false`
    so the client distinguishes.
    """
    data: list[dict] = []

    # Helper to push a record with caps
    async def _push_loaded(pool, pool_name: str, alias: str):
        if pool is None:
            return
        caps = await _model_capabilities(pool.model, pool=pool, pool_name=pool_name)
        # Expose the alias only. We used to also emit the concrete model
        # path (e.g. `/Volumes/models/odysseus/inferencerlabs--Hy3-preview-…`)
        # but it (a) duplicated the alias entry in client UIs like Companion's
        # model picker, and (b) leaked infra details (filesystem paths) that
        # have no business reaching API consumers. The alias always routes
        # to whatever's currently loaded on the pool — that's the stable
        # public contract.
        alias_caps = dict(caps)
        alias_caps["alias_for"] = pool.model
        alias_caps["ready"] = True
        data.append({
            "id": alias, "object": "model",
            "created": _now(), "owned_by": f"odyssai-{pool_name}",
            # OpenAI-standard `root` = the concrete model the alias resolves to,
            # so standard clients (and the dashboard) can show what `default`
            # actually serves without reading our x_ extensions (#62). `id`
            # stays the alias — routing on model="default" is unchanged.
            "root": pool.model,
            "x_concrete": pool.model,
            "x_odyssai": alias_caps,
        })

    await _push_loaded(_pool, "nautilus", "nautilus")
    # Emit one entry per loaded pool across every cluster.
    for cid, alias, pool in list_all_pools():
        await _push_loaded(pool, cid, alias)

    # CoeOS router: advertise the virtual `CoeOS` id when enabled (agents target
    # it, and clients like Companion show it as an endpoint alongside Argo /
    # Telemak). It's not a pool — it resolves to an axis's model per request.
    _coeos_cfg = get_coeos_config()
    _coeos_ax_models = _coeos_axis_models(_coeos_cfg)
    if _coeos_cfg.get("enabled") and _coeos_ax_models:
        data.append({
            "id": COEOS_DISPLAY_ID, "object": "model",
            "created": _now(), "owned_by": "odyssai-coeos",
            "root": COEOS_DISPLAY_ID, "x_concrete": COEOS_DISPLAY_ID,
            "x_odyssai": {"ready": True, "kind": "router",
                          "axes": _coeos_ax_models},
        })

    # kind=telemak clusters: query the upstream's /v1/models. Multi-model
    # routing (V1):
    #   - 1 model loaded → emit `cluster_id` (back-compat).
    #   - N>1 models loaded → emit N entries `cluster_id:<short_id>`. The
    #     bare cluster_id alias disappears so clients are forced to
    #     disambiguate (Companion picks the right entry from the list).
    for cid in active_cluster_ids():
        cd = get_cluster_def(cid)
        if cd.get("kind") != "telemak":
            continue
        loaded = await _telemak_loaded_models(cid, cd)
        if not loaded:
            continue
        # Auto-discover capabilities from upstream /.well-known/. Falls back
        # to legacy hardcoded (stream=True, tools=False) when the endpoint
        # is missing or unreachable (older Telemak builds).
        caps = (await _telemak_capabilities(cid, cd)).get("capabilities") or {}
        stream_cap = caps.get("stream", True)
        tools_cap = caps.get("tools", False)
        # Vision capability. The upstream /.well-known/ is absent on a plain
        # mlx_vlm.server, so caps has no vision flag -> Companion reads
        # x_odyssai.supports_vision as false and warns "may not support
        # images" for a working VLM. Resolve it at the source (the engine is
        # the capability truth), not via a client-side allowlist: honor an
        # explicit cluster-def override, else infer from the checkpoint name
        # (VL builds carry "vl"/"vision").
        def _telemak_vision(name: str) -> bool:
            if caps.get("vision") or cd.get("supports_vision"):
                return True
            n = (name or "").lower()
            return any(t in n for t in ("-vl", "vl-", "_vl", "vision"))
        # Distinct `kind` marker the dashboard reads to draw a VLM badge. The
        # transport stays http-proxy (backend below), but engine-managed VLMs
        # (VLM_MANAGED_KEY) — or any telemak cluster serving a vision model —
        # report kind="vlm" so the VL model is attributable + badged as such.
        def _telemak_kind(name: str) -> str:
            if cd.get(VLM_MANAGED_KEY) or _telemak_vision(name):
                return "vlm"
            return "telemak"
        # cluster_label + family + quantization let the Companion picker
        # render telemak rows with the same depth as local pool rows:
        # title = cluster_label (e.g. "TeleCoder"), subtitle = family · quant.
        cluster_label = cd.get("name") or cid
        if len(loaded) == 1:
            short = _telemak_short_id(loaded[0])
            data.append({
                "id": cid, "object": "model",
                "created": _now(), "owned_by": "odyssai-telemak",
                "root": loaded[0],
                "x_concrete": loaded[0],
                "x_odyssai": {
                    "ready": True,
                    "loaded": True,
                    "alias_for": loaded[0],
                    "kind": _telemak_kind(loaded[0]),
                    "cluster_label": cluster_label,
                    "family": short,
                    "quantization": _quant_from_name(loaded[0]),
                    "stream": stream_cap,
                    "tools": tools_cap,
                    "supports_vision": _telemak_vision(loaded[0]),
                    "backend": "http-proxy",
                    "upstream": cd.get("upstream"),
                },
            })
        else:
            for upstream_model in loaded:
                short = _telemak_short_id(upstream_model)
                data.append({
                    "id": f"{cid}:{short}", "object": "model",
                    "created": _now(), "owned_by": "odyssai-telemak",
                    "root": upstream_model,
                    "x_concrete": upstream_model,
                    "x_odyssai": {
                        "ready": True,
                        "loaded": True,
                        "alias_for": upstream_model,
                        "kind": _telemak_kind(upstream_model),
                        "cluster_label": cluster_label,
                        "family": short,
                        "quantization": _quant_from_name(upstream_model),
                        "cluster": cid,
                        "short_id": short,
                        "stream": stream_cap,
                        "tools": tools_cap,
                        "supports_vision": _telemak_vision(upstream_model),
                        "backend": "http-proxy",
                        "upstream": cd.get("upstream"),
                    },
                })

    # Published cloud aliases (OdyssAI-X-as-gateway). Always advertised
    # because they're always servable — the upstream takes care of itself.
    cloud = _cloud_entries_for_v1_models()
    for entry in cloud:
        entry.setdefault("x_odyssai", {})["ready"] = True
    data.extend(cloud)

    # Opt-in: inventory of on-disk models that aren't currently loaded.
    # Marked ready=false so clients know they can't be selected without
    # an admin load step first.
    if include_unloaded:
        try:
            unloaded = await _discover_unloaded_models()
        except Exception:
            unloaded = []
        for short, cluster, sz in unloaded:
            caps = await _model_capabilities(
                short, pool=None, pool_name=None,
                cluster_for_unloaded=cluster, size_bytes=sz,
            )
            caps = dict(caps); caps["ready"] = False
            data.append({
                "id": short, "object": "model",
                "created": _now(), "owned_by": f"odyssai-{cluster}",
                "x_odyssai": caps,
            })

    return {"object": "list", "data": data}


def _cluster_fallback_for(model_id: Optional[str]) -> Optional[str]:
    """If `model_id` is a local cluster alias (default/nautilus) that has a
    configured fallback cloud alias in its cluster_def, return it. Else None.

    Cluster def field: `fallback: "or:hy3-preview"`.
    """
    if not model_id:
        return None
    m = model_id.strip().lower()
    if m not in DEFAULT_CLUSTER_DEFS:
        return None
    try:
        cd = get_cluster_def(m)
    except Exception:
        return None
    fb = cd.get("fallback")
    return fb if isinstance(fb, str) and fb else None


def classify_request(
    *,
    model_id: Optional[str],
    max_tokens: Optional[int],
    prompt_chars: int,
    has_tools: bool,
    header_hint: Optional[str] = None,
) -> str:
    """Coarse request class for routing + protection rules.

    Classes:
      - `probe`    : tiny / structured outputs (titles, classifiers,
                     tool-decision steps, health checks). Default refuses.
      - `agent`    : tool calls enabled — multi-round tool use likely.
                     Default accepts, but routing prefers single-node engines
                     so the tool loop doesn't tie up distributed compute.
      - `longform` : large completions (essays, long code blocks).
      - `chat`     : the default interactive case.
      - `compile`  : memory-compile job (header-tagged). Default refuses.

    Header-hint takes precedence — Companion's memory pipeline sets
    `x-odyssai-job=compile` to opt in to compile-class routing without
    relying on heuristics.

    Heuristic ranges chosen to match Companion's existing probe gate
    (max_tokens<=20 short) and the audit's 'micro-task' definition
    (max_tokens<=32 was the threshold introduced 2026-05-18).
    """
    if header_hint:
        h = header_hint.strip().lower()
        if h in ("probe", "chat", "agent", "longform", "compile"):
            return h
    if max_tokens is not None:
        if max_tokens <= 32 and prompt_chars < 4000:
            return "probe"
        if max_tokens > 4096:
            return "longform"
    if has_tools:
        return "agent"
    return "chat"


# Per-cluster acceptance map. False = refuse with a structured 400 message.
# Default refuses probe + compile (those are routing mistakes). Other clusters and
# nautilus accept everything because they're already single-node so the
# overhead of running a probe there is minimal.
_CLUSTER_ACCEPTS: dict[str, dict[str, bool]] = {
    "default": {
        "probe":    False,
        "chat":     True,
        "agent":    True,
        "longform": True,
        "compile":  False,
    },
}


def _refuse_message_for(cluster: str, klass: str) -> Optional[dict]:
    """Build a structured 400 detail when `cluster` refuses `klass`.
    Returns None when the request is accepted. Hint phrases mention the
    obvious alternatives (LAN endpoint, cloud probe model)."""
    cluster_rules = _CLUSTER_ACCEPTS.get(cluster, {})
    if cluster_rules.get(klass, True):
        return None
    hint = {
        "probe":   "Route probes/titles to a fast LAN endpoint or a cloud probe model.",
        "compile": "Memory-compile jobs go to Companion's memory worker, not the distributed engine.",
    }.get(klass, f"This cluster doesn't accept request class={klass}.")
    return {
        "error": "cluster_refuses_class",
        "cluster": cluster,
        "class": klass,
        "message": f"Cluster {cluster!r} refuses requests classified as {klass!r}.",
        "hint": hint,
    }


def _route_pool(model: Optional[str]) -> Optional[RunnerPool]:
    """Pick which RunnerPool handles a chat request based on model id.

    Routing rules (in order):
      1. Match a cluster id → that cluster's default-alias pool
      2. Match an alias (across all clusters) → that pool
      3. Match a concrete model path against any loaded pool's `.model`
      4. "nautilus" → legacy single grid (_pool)

    Returns None when no match — caller surfaces 404 with `ready_models`.
    """
    if not model:
        return None
    m = model.strip().lower()
    if m == "nautilus":
        return _pool
    # 1. Direct cluster-id match
    pool = get_pool(m)
    if pool is not None:
        return pool
    # 2. Alias match anywhere
    for _cid, alias, pool in list_all_pools():
        if alias.lower() == m:
            return pool
    # 3. Concrete model path
    for _cid, _alias, pool in list_all_pools():
        if model == pool.model:
            return pool
    if _pool is not None and model == _pool.model:
        return _pool
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Coeos — smart LLM-router (RFC OdyssAI-X#63). A virtual model id `coeos` that
# reads the request, picks the best candidate via a JSON routing instruction
# (deterministic rules first, a light LLM only when ambiguous AND already hot),
# rewrites req.model, and lets the normal routing chain serve it. For AGENTS
# (Omnigent/Cline/Aider/Hermes), routing PER SKILL AXIS. The taxonomy + bindings
# are DATA (the "TMB Settings" file the operator imports), never hard-coded.
# Config lives in cluster-config.json["coeos_config"].
# ──────────────────────────────────────────────────────────────────────────────

COEOS_MODEL_ID = "coeos"          # canonical id used for case-insensitive matching
COEOS_DISPLAY_ID = "CoeOS"        # public id emitted in /v1/models (Sophie's spelling)

# CoeOS is a benchmark-composed virtual model: the routing TAXONOMY (skill axes)
# and the per-axis model bindings live ENTIRELY in the config ("TMB Settings"
# file the operator imports) — NO hard-coded categories/rules in code. The
# decider model classifies each request into one configured axis, then we serve
# that axis's bound model. Adding an axis (e.g. Swift) = editing the file.

# Per-(model, axis, fallback) decision counter — operator visibility.
_coeos_decisions: dict[tuple, int] = {}


def get_coeos_config() -> dict:
    cfg = _load_cluster_config()
    return cfg.get("coeos_config") or {}


def _coeos_axes(cfg: dict) -> list:
    """The configured skill axes (data-driven taxonomy). Each = {key, label,
    model, ...}. Comes from the imported TMB Settings — never hard-coded."""
    axes = cfg.get("axes")
    return [a for a in axes if isinstance(a, dict) and a.get("key")] if isinstance(axes, list) else []


def _coeos_axis_models(cfg: dict) -> dict:
    """axis key → bound model (a LOGICAL name resolved via the registry, or a
    literal endpoint in legacy configs). Non-empty, not the reserved coeos id."""
    out = {}
    for ax in _coeos_axes(cfg):
        k, m = ax.get("key"), ax.get("model")
        if k and m and str(m).lower() != COEOS_MODEL_ID:
            out[k] = m
    return out


def _coeos_model_registry(cfg: dict) -> dict:
    """Logical model name → {name: <public display>, endpoint: <operator's id>}.
    The registry is the ONLY operator-specific join: axes bind portable logical
    names, the registry maps each to the local endpoint. Empty endpoint = not
    mapped yet. Absent registry = legacy alias-based config (see resolve)."""
    reg = cfg.get("models")
    return reg if isinstance(reg, dict) else {}


def _coeos_resolve_endpoint(cfg: dict, logical) -> tuple:
    """Logical model name → (endpoint_id, public_display_name).
    If a registry entry exists, use its endpoint + name. If there's no registry
    entry for this name, treat the binding itself as the endpoint (back-compat
    with legacy alias-based TMB Settings)."""
    if not logical:
        return "", ""
    entry = _coeos_model_registry(cfg).get(logical)
    if isinstance(entry, dict):
        return (entry.get("endpoint") or ""), (entry.get("name") or logical)
    # No registry entry → the binding IS the endpoint (legacy config).
    return logical, logical


def _coeos_hot_model_ids() -> set:
    """Ids servable RIGHT NOW, in the SAME id-space /v1/models publishes:
    loaded local pool aliases (and their cluster ids), always-ready cloud
    aliases, and nautilus. Telemak cluster ids are checked separately in
    `_coeos_is_servable` (they're proxies, not pools). Read-only."""
    hot: set = set()
    for cid, alias, _p in list_all_pools():
        hot.add(alias)   # the id /v1/models emits
        hot.add(cid)     # tolerate the cluster-id form too
    if _pool is not None:
        hot.add("nautilus")
    try:
        for e in _cloud_entries_for_v1_models():
            hot.add(e["id"])
    except Exception:
        pass
    return hot


def _coeos_is_servable(mid, hot: set) -> bool:
    """True if `mid` can be served right now — covers loaded local pools,
    cloud aliases, and ACTIVE Telemak clusters (id `cid` or `cid:short`),
    matching what /v1/models advertises. No async upstream calls."""
    if not mid:
        return False
    if mid in hot or find_cloud_alias(mid) is not None:
        return True
    base = str(mid).split(":")[0]
    cd = get_cluster_def(base)
    if cd and cd.get("kind") == "telemak" and base in set(active_cluster_ids()):
        return True
    return False


def list_routable_model_ids() -> list:
    """Published local + cloud ids (excludes Telemak proxies — the dashboard
    picker reads the full /v1/models list directly for those)."""
    return sorted(_coeos_hot_model_ids())


def _coeos_header_axis(request, keys: list) -> Optional[str]:
    """Explicit axis from the agent. `x-coeos-axis` wins; `x-coeos-category` is a
    back-compat alias. Returned only if it's a CONFIGURED axis key."""
    for h in ("x-coeos-axis", "x-coeos-category"):
        v = (request.headers.get(h) or "").strip().lower()
        if v in keys:
            return v
    return None


def _coeos_parse_axis(text: str, keys: list) -> Optional[str]:
    """Extract the chosen axis key from the decider's (possibly multi-token,
    reasoned) reply. Priority: an explicit final `AXIS: <key>` line → a bare reply
    whose first token is a key (legacy tag-style) → the LAST word-bounded key
    mentioned anywhere. Returns None if no configured key is found."""
    import re
    if not text or not text.strip():
        return None
    low = text.lower()
    keyset = {k.lower() for k in keys}
    # 1. explicit `AXIS: <key>` (or `AXIS = key`, `AXIS:"key"`) — last one wins.
    for m in reversed(list(re.finditer(r"axis\s*[:=]\s*[`\"']?([a-z0-9_]+)", low))):
        if m.group(1) in keyset:
            return m.group(1)
    # 2. legacy: the whole reply is just the key.
    first = low.strip().split()[0].strip('`"\',.') if low.strip() else ""
    if first in keyset:
        return first
    # 3. last word-bounded key mention anywhere in the reasoning.
    best, best_pos = None, -1
    for k in keys:
        for m in re.finditer(r"\b" + re.escape(k.lower()) + r"\b", low):
            if m.start() > best_pos:
                best, best_pos = k, m.start()
    return best


async def _coeos_llm_classify(decider_id, axes, messages) -> Optional[str]:
    """Ask the (already-hot) decider to UNDERSTAND the request and classify it into
    ONE configured axis. The taxonomy (keys + labels + per-axis descriptions) is
    passed from config — nothing about the skills or the winning models is
    hard-coded here. The decider is treated as a reasoning router, not a tag
    matcher: it receives the full last message + each axis's frontier description
    and room to think, then emits the key on a final `AXIS:` line.

    Dispatch mirrors the chat endpoint so a Telemak-proxy decider (e.g. a Qwen3.5
    122B served by a Telemak Swift binary) is reachable — `_route_pool` only
    resolves LOCAL pools and returns None for proxies, which is why the old
    pool-only path silently skipped proxy deciders and every request fell to
    `default_axis`. Order: Telemak upstream → local pool. `enable_thinking=False`
    so a reasoning-first decider answers directly instead of burning its budget on
    a `<think>` block that never reaches the key."""
    last_content = ""
    if messages:
        c = messages[-1].get("content")
        if isinstance(c, str):
            last_content = c[:8000]
        elif isinstance(c, list):  # multimodal — keep the text parts
            last_content = " ".join(
                p.get("text", "") for p in c
                if isinstance(p, dict) and p.get("type") == "text")[:8000]

    def _axis_line(ax):
        line = f"- {ax['key']}: {ax.get('label', ax['key'])}"
        desc = ax.get("description") or ax.get("hint")
        if desc:
            line += f" — {desc}"
        return line

    menu = "\n".join(_axis_line(ax) for ax in axes if ax.get("key"))
    keys = [ax["key"] for ax in axes if ax.get("key")]
    prompt = (
        "You are CoeOS's routing classifier. UNDERSTAND the request — its true "
        "intent and the nature of the deliverable (target language, domain) — then "
        "pick the SINGLE best-matching skill axis from the menu. Prefer the MOST "
        "SPECIFIC axis that applies; choose a generic bucket (e.g. code_general) "
        "ONLY when no specific axis fits. Honour each axis's frontier notes "
        "(the '— …' clause, including its 'not here if …' guidance).\n\n"
        f"Axes:\n{menu}\n\n"
        f"Request:\n{last_content}\n\n"
        "Reason in at most two short sentences, then end your reply with a final "
        f"line exactly: `AXIS: <key>` where <key> is one of: {', '.join(keys)}")
    dmsg = [{"role": "user", "content": prompt}]
    buf = ""
    try:
        # 1. Telemak proxy decider — forward to its upstream, same as the chat
        #    passthrough. `_route_pool` can't see proxies, so we resolve the
        #    cluster directly and reuse the proven proxy helper.
        tele_cluster, tele_short = _telemak_split_alias(decider_id)
        if tele_cluster and cluster_exists(tele_cluster) \
                and get_cluster_def(tele_cluster).get("kind") == "telemak":
            body = {"model": decider_id, "messages": dmsg, "max_tokens": 160,
                    "enable_thinking": False, "stream": False}
            out = await _telemak_proxy_chat_completion(
                tele_cluster, get_cluster_def(tele_cluster), body, False,
                requested_short_id=tele_short)
            buf = ((out.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
        else:
            # 2. Local pool decider (back-compat, e.g. a small local classifier).
            pool = _route_pool(decider_id)
            if pool is None:
                return None
            async for ev in pool.submit(None, 160, False, messages=dmsg):
                if ev.get("event") == "token":
                    buf += ev.get("text", "")
                elif ev.get("event") == "done":
                    break
    except Exception as e:
        sys.stderr.write(f"[coeos] decider error: {e}\n")
        return None
    return _coeos_parse_axis(buf, keys)


async def coeos_resolve(req, request) -> tuple:
    """Resolve `coeos` → (concrete model id, axis). Classify the request into one
    CONFIGURED axis (explicit `x-coeos-axis` header → decider LLM if it's hot →
    `default_axis`), then serve that axis's bound model. Falls back to any hot
    bound model (never routes to a cold model); 503 if none is hot (unless
    cold_boot_autoload). Taxonomy + bindings are data — see the TMB Settings."""
    cfg = get_coeos_config()
    if not cfg.get("enabled"):
        raise HTTPException(status_code=400, detail={
            "error": "coeos_disabled",
            "message": "CoeOS router is disabled. Enable it in Settings → CoeOS."})
    axes = _coeos_axes(cfg)
    axis_models = _coeos_axis_models(cfg)
    keys = [ax.get("key") for ax in axes if ax.get("key")]
    default_axis = cfg.get("default_axis")
    if default_axis not in keys:
        default_axis = next((k for k in keys if k in axis_models),
                            (keys[0] if keys else None))
    hot = _coeos_hot_model_ids()

    # Classify: explicit header → decider LLM (only if already hot) → default.
    axis = _coeos_header_axis(request, keys)
    if not axis:
        decider_id = cfg.get("decider_model")
        # Servability gate (Telemak-aware): only invoke an already-hot decider —
        # never cold-boot one just to classify. `_coeos_is_servable` recognises
        # loaded local pools, cloud aliases AND active Telemak proxies, unlike the
        # old `_route_pool(...).alive_count() > 0` which was None/0 for proxies.
        if decider_id and _coeos_is_servable(decider_id, hot):
            messages = [m.model_dump(exclude_none=True) for m in req.messages]
            axis = await _coeos_llm_classify(decider_id, axes, messages)
    if axis not in keys:
        axis = default_axis
    # Resolve the axis binding through the model registry: axes bind a LOGICAL
    # model name (portable, e.g. "minimax-m3"); the registry maps it to the
    # operator's endpoint + a public display name. Legacy configs without a
    # registry treat the binding as a literal endpoint (back-compat).
    logical = axis_models.get(axis) if axis else None
    endpoint, display = _coeos_resolve_endpoint(cfg, logical)

    # No silent fallback to a different model. If the recommended model isn't
    # mapped or isn't loaded, surface it ("<name> — not loaded") so the operator
    # loads it or maps an endpoint. cold_boot_autoload may opt to load it instead
    # of erroring.
    fallback = False
    if endpoint and _coeos_is_servable(endpoint, hot):
        chosen = endpoint
    elif endpoint and cfg.get("cold_boot_autoload"):
        sys.stderr.write(f"[coeos] cold boot — loading {endpoint!r} for axis {axis} (autoload on)\n")
        chosen, fallback = endpoint, True
    else:
        raise HTTPException(status_code=503, detail={
            "error": "coeos_model_not_loaded",
            "axis": axis,
            "recommended": display or logical or "?",
            "endpoint": endpoint or None,
            "message": f"{display or logical or 'recommended model'} — not loaded. "
                       "Map an endpoint for this model in Settings → CoeOS, or load "
                       "it. CoeOS does not silently route to a different model."})

    k = (chosen, axis or "?", fallback)
    _coeos_decisions[k] = _coeos_decisions.get(k, 0) + 1
    return chosen, (axis or "")


_TOOL_BLOCK_RE = [
    __import__("re").compile(r"<tool_call>.*?</tool_call>", __import__("re").DOTALL),
    __import__("re").compile(r"<tool_calls>.*?</tool_calls>", __import__("re").DOTALL),
]


def _strip_tool_calls_from_text(text: str) -> str:
    """Remove `<tool_call>...</tool_call>` (and `<tool_calls>...`) blocks from
    the user-visible content, since we surface them as structured `tool_calls`.
    Handles both Hermes JSON and Qwen3-Coder XML inner forms.
    """
    out = text
    for pat in _TOOL_BLOCK_RE:
        out = pat.sub("", out)
    return out.strip()


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    # Coeos router (RFC #63): a virtual model id that picks a concrete candidate
    # and rewrites req.model, so the normal chain below (Telemak/cloud/local)
    # serves the chosen model. Resolved FIRST so no other branch short-circuits.
    coeos_routed: Optional[str] = None     # clean id CoeOS routed to (for the response label)
    coeos_axis: Optional[str] = None
    if req.model and req.model.strip().lower() == COEOS_MODEL_ID:
        coeos_routed, coeos_axis = await coeos_resolve(req, request)
        req.model = coeos_routed
    # 0. Telemak passthrough? If the model id matches a kind=telemak cluster,
    # proxy the request to that cluster's upstream Swift binary. Supports
    # both `cluster_id` (1-model back-compat) and `cluster_id:short_id`
    # (multi-model V1) forms.
    if req.model:
        tele_cluster, tele_short = _telemak_split_alias(req.model)
        if tele_cluster and cluster_exists(tele_cluster):
            cd_t = get_cluster_def(tele_cluster)
            if cd_t.get("kind") == "telemak":
                body = req.model_dump(exclude_none=True)
                return await _telemak_proxy_chat_completion(
                    tele_cluster, cd_t, body, bool(req.stream),
                    requested_short_id=tele_short,
                )

    # 1. Cloud passthrough? If the model id matches a published cloud alias
    # (e.g. "or:claude-haiku"), proxy to the configured upstream. No local
    # compute. OpenAI-compat in / out — no protocol translation.
    cloud = find_cloud_alias(req.model)
    if cloud:
        prov_id, prov, entry = cloud
        body = req.model_dump(exclude_none=True)
        return await _proxy_chat_completion(prov_id, prov, entry, body)

    # 2. Local pool route (existing behaviour).
    pool = _route_pool(req.model)
    if pool is None or pool.alive_count() == 0:
        # Fallback: if the user routed at a local alias but no pool is loaded,
        # check for a cluster-level fallback to a cloud alias.
        fb_alias = _cluster_fallback_for(req.model)
        if fb_alias:
            fb_cloud = find_cloud_alias(fb_alias)
            if fb_cloud:
                prov_id, prov, entry = fb_cloud
                body = req.model_dump(exclude_none=True)
                sys.stderr.write(
                    f"[fallback] {req.model!r} unavailable → routing to "
                    f"cloud alias {fb_alias!r} ({prov_id})\n"
                )
                return await _proxy_chat_completion(prov_id, prov, entry, body)
        # Structured error: tell the client exactly which models are ready
        # to serve right now so it doesn't have to guess. `/v1/models` is
        # the canonical source — we mirror its `ready=true` ids here.
        ready = []
        # Every loaded Default alias is servable (default "default" + extras).
        for cid, alias, _ in list_all_pools():
            ready.append(alias if alias != DEFAULT_ALIAS else cid)
        if _pool is not None:         ready.append("nautilus")
        # Cloud aliases are always ready.
        try:
            ready.extend(e["id"] for e in _cloud_entries_for_v1_models())
        except Exception:
            pass
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_loaded",
                "message": f"Model {req.model!r} is not currently loaded. "
                           f"Auto-swap is disabled — load explicitly via the "
                           f"admin dashboard or pick a ready model.",
                "ready_models": ready,
                "hint": "GET /v1/models lists everything ready to serve "
                        "(add ?include_unloaded=true for inventory).",
            },
        )
    # Request classification + per-cluster acceptance. Replaces the older
    # plain "Default refuses max_tokens<=32" gate with a broader probe/chat/
    # agent/longform/compile taxonomy. Default refuses probe + compile; other
    # classes pass through. Header `x-odyssai-job` overrides the heuristic.
    prompt_chars_pre = sum(
        len(m.content) if isinstance(m.content, str)
        else sum(len(p.text or "") for p in (m.content or [])
                 if getattr(p, "type", None) == "text")
        for m in (req.messages or [])
    )
    req_class = classify_request(
        model_id=req.model,
        max_tokens=req.max_tokens,
        prompt_chars=prompt_chars_pre,
        has_tools=bool(req.tools),
        header_hint=request.headers.get("x-odyssai-job"),
    )
    refuse = _refuse_message_for(pool.cluster, req_class)
    if refuse:
        raise HTTPException(status_code=400, detail=refuse)
    # Pass the full message structure to the runner so tool messages, system
    # messages, multi-turn etc. are preserved by `apply_chat_template`.
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = _now()
    model_id = pool.model
    client_ip = request.client.host if request.client else "?"
    prompt_chars = sum(len(m.get("content") or "") for m in messages)
    # Prefix cache: header takes precedence over body field; both are optional.
    session_id = (request.headers.get("x-session-id")
                  or request.headers.get("X-Session-Id")
                  or req.session_id)
    session_meta: dict = {"id": session_id} if session_id else {}

    if not req.stream:
        text_parts: list[str] = []
        ntoks = 0
        nstream_prompt_tokens: Optional[int] = None
        nstream_cached_tokens: int = 0
        elapsed_s = 0.0
        ttft_s: Optional[float] = None
        tool_calls: list[dict] = []
        t_start = time.time()
        cancel_event = _runs_register(completion_id, model=model_id, cluster=pool.cluster,
                                       pool_alias=pool.alias,
                                       client=client_ip, max_tokens=req.max_tokens or 512,
                                       kind="streaming")
        run_status = "completed"
        try:
            async for ev in pool.submit(None, req.max_tokens or 512, req.enable_thinking,
                                        messages=messages, tools=req.tools,
                                        session_id=session_id,
                                        request_id=completion_id,
                                        reasoning_effort=(req.reasoning_effort
                                                          or _default_reasoning_effort(model_id))):
                if cancel_event.is_set():
                    run_status = "cancelled"
                    break
                if ev.get("event") == "token":
                    if ttft_s is None:
                        ttft_s = time.time() - t_start
                    text_parts.append(ev.get("text", ""))
                    _runs_tick(completion_id, ev.get("text", ""))
                elif ev.get("event") == "done":
                    ntoks = ev.get("ntoks", 0)
                    elapsed_s = ev.get("elapsed_s", 0.0)
                    tool_calls = ev.get("tool_calls", []) or []
                    session_meta.update(ev.get("session", {}) or {})
                    nstream_prompt_tokens = ev.get("prompt_tokens")
                    nstream_cached_tokens = int(ev.get("cached_tokens") or 0)
        finally:
            _runs_finalize(completion_id)
        _touch_session(pool.cluster, session_meta, model_id)
        record_metric(client_ip, ntoks, elapsed_s, ttft_s, prompt_chars, model_id,
                      cluster=pool.cluster, tool_calls=len(tool_calls),
                      session_kind=session_meta.get("cache_kind"),
                      status=run_status)
        content = "".join(text_parts)
        if tool_calls:
            content = _strip_tool_calls_from_text(content)
        # Same <think>…</think> split as the streaming path. When the caller
        # asked enable_thinking=False, the model may still emit them
        # (MiniMax M2.7-style chat templates ignore the Jinja flag).
        # Move that content into `reasoning_content` so OpenAI-compat
        # clients render it as the reasoning channel, not the answer.
        reasoning_content_full = ""
        # Same wire filter as the streaming path, same shared decision as the
        # Telemak proxy (_should_filter_think): filter when thinking is ON
        # (route the block to reasoning_content) AND when the model ignores
        # enable_thinking=False (MiniMax/Step-3.7). model_id is pool.model,
        # the concrete HF path, so the family match holds for pool aliases.
        if _should_filter_think(model_id, req.enable_thinking) and content:
            open_m, close_m = _model_think_markers(model_id)
            # Seed in_think=True only when the open tag was PREFILLED into the
            # prompt (output begins mid-thought, no leading open). If the model
            # emitted its own open — M3 adaptive mode — seed False and let the
            # filter catch it. Full content in hand here, so detect directly.
            seed = not content.lstrip().startswith(open_m)
            ts: dict = {"in_think": seed, "carry": "",
                        "open": open_m, "close": close_m}
            visible_full, reasoning_full = _split_think_stream(content, ts)
            fl_vis, fl_reason = _flush_think_stream(ts)
            content = visible_full + fl_vis
            reasoning_content_full = reasoning_full + fl_reason
        message_obj: dict = {"role": "assistant", "content": content or None}
        if reasoning_content_full:
            message_obj["reasoning_content"] = reasoning_content_full
        if tool_calls:
            message_obj["tool_calls"] = tool_calls
        p_tokens = nstream_prompt_tokens if nstream_prompt_tokens is not None else max(1, prompt_chars // 4)
        usage_obj: dict = {
            "prompt_tokens": int(p_tokens),
            "completion_tokens": int(ntoks),
            "total_tokens": int(p_tokens) + int(ntoks),
        }
        # Prefix cache hit (session-HIT in mlx-lm prompt_cache terms). Surface
        # as OpenAI-compat `prompt_tokens_details.cached_tokens` so Companion's
        # StatsRow shows the win for local requests, not just the cloud proxy.
        if nstream_cached_tokens > 0:
            usage_obj["prompt_tokens_details"] = {"cached_tokens": int(nstream_cached_tokens)}
        body = {
            "id": completion_id, "object": "chat.completion",
            "created": created, "model": model_id,
            "choices": [{
                "index": 0,
                "message": message_obj,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": usage_obj,
            "x_mlx_cluster": {
                "elapsed_s": elapsed_s, "ttft_s": ttft_s,
                "tps": ntoks / elapsed_s if elapsed_s else 0.0,
            },
        }
        if session_meta:
            body["x_mlx_cluster"]["session"] = session_meta
        if coeos_routed:
            body["x_odyssai_routed"] = {"router": COEOS_DISPLAY_ID,
                                        "routed_to": coeos_routed,
                                        "axis": coeos_axis,
                                        "concrete": model_id}
        return JSONResponse(body)

    async def stream() -> AsyncIterator[bytes]:
        ttft_s: Optional[float] = None
        t_start = time.time()
        ntoks_total = 0
        elapsed_total = 0.0
        tool_calls_count = 0
        sess: dict = {}
        run_status = "completed"
        cancel_event = _runs_register(completion_id, model=model_id, cluster=pool.cluster,
                                       pool_alias=pool.alias,
                                       client=client_ip, max_tokens=req.max_tokens or 512,
                                       kind="streaming")
        try:
            first = {"id": completion_id, "object": "chat.completion.chunk",
                     "created": created, "model": model_id,
                     "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
            if coeos_routed:
                first["x_odyssai_routed"] = {"router": COEOS_DISPLAY_ID,
                                             "routed_to": coeos_routed,
                                             "axis": coeos_axis,
                                             "concrete": model_id}
            yield f"data: {json.dumps(first)}\n\n".encode()
            # Per-stream <think>…</think> wire filter. Decision is shared
            # with the Telemak proxy via _should_filter_think so a model
            # behaves identically local vs proxied — filter when thinking is
            # ON (route the block to the collapsed reasoning channel) AND when
            # the model ignores enable_thinking=False (MiniMax/Step-3.7).
            # model_id is pool.model (the concrete HF path), so the family
            # match works even when the caller dialed a pool alias ("default").
            think_filter_active = _should_filter_think(model_id, req.enable_thinking)
            # Seed in_think + pick the right markers per model. Most templates
            # auto-OPEN (prefilled tag) → seed True; M3 adaptive emits its own
            # open → seed False (see _seed_in_think). _should_filter_think
            # already excludes the honor-flag-thinking-off ghost case.
            _open_m, _close_m = _model_think_markers(model_id)
            think_state: dict = {
                "in_think": think_filter_active and _seed_in_think(model_id, req.enable_thinking),
                "carry": "", "open": _open_m, "close": _close_m,
            }
            async for ev in pool.submit(None, req.max_tokens or 512, req.enable_thinking,
                                        messages=messages, tools=req.tools,
                                        session_id=session_id,
                                        request_id=completion_id,
                                        reasoning_effort=(req.reasoning_effort
                                                          or _default_reasoning_effort(model_id))):
                if await request.is_disconnected():
                    run_status = "disconnected"
                    break
                if cancel_event.is_set():
                    run_status = "cancelled"
                    break
                if ev.get("event") == "token":
                    if ttft_s is None:
                        ttft_s = time.time() - t_start
                    raw_text = ev.get("text", "")
                    if think_filter_active:
                        visible, reasoning = _split_think_stream(raw_text, think_state)
                    else:
                        visible, reasoning = raw_text, ""
                    # Emit reasoning delta first if any, then content. This
                    # preserves the temporal ordering of the model's output
                    # for clients that interleave the two channels visually.
                    if reasoning:
                        chunk = {"id": completion_id, "object": "chat.completion.chunk",
                                 "created": created, "model": model_id,
                                 "choices": [{"index": 0,
                                              "delta": {"reasoning_content": reasoning},
                                              "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n".encode()
                    if visible:
                        chunk = {"id": completion_id, "object": "chat.completion.chunk",
                                 "created": created, "model": model_id,
                                 "choices": [{"index": 0,
                                              "delta": {"content": visible},
                                              "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n".encode()
                    # _runs_tick tracks the run's character count; pass the
                    # full raw text so the stats reflect what the model
                    # produced, not just what we surfaced.
                    _runs_tick(completion_id, raw_text)
                elif ev.get("event") == "done":
                    # Flush any final carry from the think-filter so we don't
                    # drop the model's last few characters when they landed
                    # inside the lookahead buffer.
                    if think_filter_active:
                        fl_vis, fl_reason = _flush_think_stream(think_state)
                        if fl_reason:
                            chunk = {"id": completion_id, "object": "chat.completion.chunk",
                                     "created": created, "model": model_id,
                                     "choices": [{"index": 0,
                                                  "delta": {"reasoning_content": fl_reason},
                                                  "finish_reason": None}]}
                            yield f"data: {json.dumps(chunk)}\n\n".encode()
                        if fl_vis:
                            chunk = {"id": completion_id, "object": "chat.completion.chunk",
                                     "created": created, "model": model_id,
                                     "choices": [{"index": 0,
                                                  "delta": {"content": fl_vis},
                                                  "finish_reason": None}]}
                            yield f"data: {json.dumps(chunk)}\n\n".encode()
                    ntoks_total = ev.get("ntoks", 0)
                    elapsed_total = ev.get("elapsed_s", 0.0)
                    tool_calls = ev.get("tool_calls", []) or []
                    tool_calls_count = len(tool_calls)
                    sess = ev.get("session", {}) or {}
                    # Prompt tokens from runner if available; cheap fallback otherwise.
                    p_tokens = ev.get("prompt_tokens")
                    if p_tokens is None:
                        p_tokens = max(1, prompt_chars // 4)  # ~OpenAI rule of thumb
                    cached_toks = int(ev.get("cached_tokens") or 0)
                    delta: dict = {}
                    if tool_calls:
                        delta["tool_calls"] = tool_calls
                    usage_payload: dict = {
                        "prompt_tokens": int(p_tokens),
                        "completion_tokens": int(ntoks_total),
                        "total_tokens": int(p_tokens) + int(ntoks_total),
                    }
                    if cached_toks > 0:
                        usage_payload["prompt_tokens_details"] = {"cached_tokens": cached_toks}
                    final = {"id": completion_id, "object": "chat.completion.chunk",
                             "created": created, "model": model_id,
                             "choices": [{"index": 0, "delta": delta,
                                          "finish_reason": "tool_calls" if tool_calls else "stop"}],
                             # OpenAI-compat usage in the final chunk so clients
                             # (Companion etc.) can render prompt/completion counts
                             # regardless of stream_options.include_usage.
                             "usage": usage_payload,
                             "x_mlx_cluster": {"elapsed_s": elapsed_total,
                                               "ttft_s": ttft_s,
                                               "tps": ev.get("tps"),
                                               "ntoks": ntoks_total}}
                    yield f"data: {json.dumps(final)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        finally:
            _runs_finalize(completion_id)
            record_metric(client_ip, ntoks_total, elapsed_total, ttft_s, prompt_chars, model_id,
                          cluster=pool.cluster, tool_calls=tool_calls_count,
                          session_kind=sess.get("cache_kind"),
                          status=run_status)

    return StreamingResponse(stream(), media_type="text/event-stream")


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible /v1/responses  (shim → /v1/chat/completions)
#
# The Responses API (openai.responses.*) is what Omnigent speaks. We don't store
# state: this is a thin, STATELESS adapter that translates the Responses request
# into a ChatCompletionRequest, reuses `chat_completions` (so CoeOS routing,
# Telemak/cloud passthrough, tool quirks and EOS handling all come for free),
# then translates the chat result back into Responses shape — a Response object
# (non-stream) or the typed `response.*` SSE event sequence (stream), including
# function_call items.
#
# Mapped: input (str | items: message / function_call / function_call_output),
#   instructions→system, tools (flat→nested), tool_choice, max_output_tokens,
#   reasoning.effort. Ignored (stateless): previous_response_id, store, metadata.
# Not yet: built-in tools (web_search/file_search), input audio.
# ──────────────────────────────────────────────────────────────────────────────
class ResponsesRequest(BaseModel):
    model: Optional[str] = None
    input: Optional[Any] = None            # str OR list of input items
    instructions: Optional[str] = None     # system prompt
    stream: Optional[bool] = False
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Any] = None
    max_output_tokens: Optional[int] = None
    reasoning: Optional[dict] = None        # {"effort": "low|medium|high"}
    model_config = {"extra": "allow"}       # tolerate previous_response_id, store, …


def _responses_flatten_content(content: Any):
    """Responses content (str | list of parts) → chat content (str, or the
    OpenAI parts list when images are present)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        parts: list[dict] = []
        has_img = False
        for p in content:
            if isinstance(p, str):
                texts.append(p)
                parts.append({"type": "text", "text": p})
                continue
            if not isinstance(p, dict):
                continue
            pt = p.get("type")
            if pt in ("input_text", "output_text", "text", "summary_text"):
                tx = p.get("text") or ""
                texts.append(tx)
                parts.append({"type": "text", "text": tx})
            elif pt in ("input_image", "image_url", "image"):
                url = p.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                url = url or p.get("url")
                if url:
                    has_img = True
                    parts.append({"type": "image_url", "image_url": {"url": url}})
        return parts if has_img else "\n".join(t for t in texts if t)
    return ""


def _responses_build_messages(instructions: Optional[str], inp: Any) -> list[dict]:
    """Responses (instructions + input) → chat messages list."""
    msgs: list[dict] = []
    if instructions:
        msgs.append({"role": "system", "content": instructions})
    if inp is None:
        return msgs
    if isinstance(inp, str):
        msgs.append({"role": "user", "content": inp})
        return msgs
    if isinstance(inp, list):
        for item in inp:
            if isinstance(item, str):
                msgs.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t in (None, "message"):
                msgs.append({"role": item.get("role", "user"),
                             "content": _responses_flatten_content(item.get("content"))})
            elif t == "function_call":
                msgs.append({"role": "assistant", "content": None,
                             "tool_calls": [{
                                 "id": item.get("call_id") or item.get("id"),
                                 "type": "function",
                                 "function": {"name": item.get("name"),
                                              "arguments": item.get("arguments") or "{}"}}]})
            elif t == "function_call_output":
                out = item.get("output")
                if not isinstance(out, str):
                    out = json.dumps(out)
                msgs.append({"role": "tool",
                             "tool_call_id": item.get("call_id") or item.get("id"),
                             "content": out})
            # unknown item types (reasoning, …) are ignored
    return msgs


def _responses_tools_to_chat(tools: Optional[list]) -> Optional[list]:
    """Responses tools (flat {type:function, name, …}) → chat ({type:function,
    function:{…}}). Tolerates already-nested. Skips built-in tools."""
    out: list[dict] = []
    for t in tools or []:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        if isinstance(t.get("function"), dict):
            out.append(t)
            continue
        fn = {"name": t.get("name"), "description": t.get("description"),
              "parameters": t.get("parameters") or {"type": "object", "properties": {}}}
        if "strict" in t:
            fn["strict"] = t["strict"]
        out.append({"type": "function",
                    "function": {k: v for k, v in fn.items() if v is not None}})
    return out or None


def _responses_tool_choice(tc: Any) -> Any:
    """Responses tool_choice → chat tool_choice."""
    if tc is None:
        return None
    if isinstance(tc, str):           # "auto" | "none" | "required"
        return tc
    if isinstance(tc, dict):
        if tc.get("type") == "function":
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            if name:
                return {"type": "function", "function": {"name": name}}
        return tc.get("type") or "auto"
    return None


def _chat_to_responses_obj(chat_body: dict, model_label: str) -> dict:
    """Chat completion JSON → Responses Response object."""
    choice = (chat_body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    # Reasoning models (MiniMax-M3 local) put the answer in `reasoning_content`
    # and leave `content` null. The chat endpoint keeps them separate on purpose
    # (Companion's thinking UI) — but an agent (Omnigent) needs one text, so the
    # shim falls back to reasoning_content when content is empty.
    text = msg.get("content") or msg.get("reasoning_content") or ""
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    output: list[dict] = []
    if text:
        output.append({"type": "message", "id": "msg_" + uuid.uuid4().hex[:24],
                       "status": "completed", "role": "assistant",
                       "content": [{"type": "output_text", "text": text, "annotations": []}]})
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        output.append({"type": "function_call", "id": "fc_" + uuid.uuid4().hex[:24],
                       "call_id": tc.get("id") or ("call_" + uuid.uuid4().hex[:12]),
                       "name": fn.get("name"), "arguments": fn.get("arguments") or "{}",
                       "status": "completed"})
    usage = chat_body.get("usage") or {}
    resp = {"id": "resp_" + uuid.uuid4().hex[:24], "object": "response",
            "created_at": chat_body.get("created") or _now(), "status": "completed",
            "model": chat_body.get("model") or model_label, "output": output,
            "output_text": text,
            "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                      "output_tokens": usage.get("completion_tokens", 0),
                      "total_tokens": usage.get("total_tokens", 0)}}
    if chat_body.get("x_odyssai_routed"):
        resp["x_odyssai_routed"] = chat_body["x_odyssai_routed"]
    return resp


async def _responses_stream(chat_iter, model_label: str):
    """Wrap the chat-completions SSE byte stream → typed Responses SSE events."""
    seq = 0

    def ev(typ: str, payload: dict) -> bytes:
        nonlocal seq
        out = {"type": typ, "sequence_number": seq, **payload}
        seq += 1
        return f"event: {typ}\ndata: {json.dumps(out)}\n\n".encode()

    rid = "resp_" + uuid.uuid4().hex[:24]
    msg_id = "msg_" + uuid.uuid4().hex[:24]
    base = {"id": rid, "object": "response", "created_at": _now(),
            "model": model_label, "status": "in_progress", "output": []}
    yield ev("response.created", {"response": base})
    yield ev("response.in_progress", {"response": base})

    text_started = False
    full_text = ""
    tool_calls: list[dict] = []
    routed = None
    buf = b""
    async for raw in chat_iter:
        buf += raw if isinstance(raw, (bytes, bytearray)) else str(raw).encode()
        while b"\n\n" in buf:
            block, buf = buf.split(b"\n\n", 1)
            line = block.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            if chunk.get("x_odyssai_routed"):
                routed = chunk["x_odyssai_routed"]
            delta = ((chunk.get("choices") or [{}])[0]).get("delta") or {}
            # content first; fall back to reasoning_content (MiniMax-M3 streams the
            # answer there with content empty) so agent streaming isn't blank.
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if piece:
                if not text_started:
                    text_started = True
                    yield ev("response.output_item.added", {"output_index": 0,
                             "item": {"type": "message", "id": msg_id, "status": "in_progress",
                                      "role": "assistant", "content": []}})
                    yield ev("response.content_part.added", {"item_id": msg_id, "output_index": 0,
                             "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})
                full_text += piece
                yield ev("response.output_text.delta", {"item_id": msg_id, "output_index": 0,
                         "content_index": 0, "delta": piece})
            if delta.get("tool_calls"):
                tool_calls = delta["tool_calls"]

    output_items: list[dict] = []
    out_index = 0
    if text_started:
        yield ev("response.output_text.done", {"item_id": msg_id, "output_index": 0,
                 "content_index": 0, "text": full_text})
        yield ev("response.content_part.done", {"item_id": msg_id, "output_index": 0,
                 "content_index": 0, "part": {"type": "output_text", "text": full_text, "annotations": []}})
        msg_item = {"type": "message", "id": msg_id, "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": full_text, "annotations": []}]}
        yield ev("response.output_item.done", {"output_index": 0, "item": msg_item})
        output_items.append(msg_item)
        out_index = 1

    for tc in tool_calls:
        fn = tc.get("function") or {}
        fc_id = "fc_" + uuid.uuid4().hex[:24]
        call_id = tc.get("id") or ("call_" + uuid.uuid4().hex[:12])
        args = fn.get("arguments") or "{}"
        if not isinstance(args, str):
            args = json.dumps(args)
        added = {"type": "function_call", "id": fc_id, "call_id": call_id,
                 "name": fn.get("name"), "arguments": "", "status": "in_progress"}
        yield ev("response.output_item.added", {"output_index": out_index, "item": added})
        yield ev("response.function_call_arguments.delta", {"item_id": fc_id,
                 "output_index": out_index, "delta": args})
        yield ev("response.function_call_arguments.done", {"item_id": fc_id,
                 "output_index": out_index, "arguments": args})
        done = {**added, "arguments": args, "status": "completed"}
        yield ev("response.output_item.done", {"output_index": out_index, "item": done})
        output_items.append(done)
        out_index += 1

    final = {**base, "status": "completed", "output": output_items, "output_text": full_text}
    if routed:
        final["x_odyssai_routed"] = routed
    yield ev("response.completed", {"response": final})


@app.post("/v1/responses")
async def responses(req: ResponsesRequest, request: Request):
    messages = _responses_build_messages(req.instructions, req.input)
    if not messages:
        raise HTTPException(status_code=400, detail={
            "error": "invalid_request",
            "message": "`input` is required (string or list of input items)."})
    kwargs: dict = {"model": req.model, "messages": messages, "stream": bool(req.stream)}
    if req.max_output_tokens:
        kwargs["max_tokens"] = req.max_output_tokens
    tools = _responses_tools_to_chat(req.tools)
    if tools:
        kwargs["tools"] = tools
    tc = _responses_tool_choice(req.tool_choice)
    if tc is not None:
        kwargs["tool_choice"] = tc
    if isinstance(req.reasoning, dict) and req.reasoning.get("effort"):
        kwargs["reasoning_effort"] = req.reasoning["effort"]
    chat_req = ChatCompletionRequest(**kwargs)
    model_label = req.model or "odyssai"

    result = await chat_completions(chat_req, request)   # reuse the whole routing chain
    if bool(req.stream):
        return StreamingResponse(_responses_stream(result.body_iterator, model_label),
                                 media_type="text/event-stream")
    chat_body = json.loads(result.body)
    return JSONResponse(_chat_to_responses_obj(chat_body, model_label))


# ──────────────────────────────────────────────────────────────────────────────
# Anthropic-compatible /v1/messages
#
# Translates Anthropic Messages API to our OpenAI-style runner protocol so
# Claude Code, Aider in Anthropic mode, and other Anthropic SDK clients can hit
# the cluster directly.
#
# Coverage:
#   - Body: model, max_tokens, system (str|blocks), messages, tools, stream
#   - Content blocks: text (input), text + tool_use (output), tool_result (input)
#   - Streaming SSE: message_start / content_block_* / message_delta / message_stop
#   - Tool conversion: Anthropic {name, description, input_schema} →
#     OpenAI {type:"function", function:{name, description, parameters}}
#
# Not yet supported: image/document content blocks, prompt caching control,
# extended thinking, batch API, files API.
# ──────────────────────────────────────────────────────────────────────────────
class AnthropicMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: Any  # str OR list[block]


class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: dict


class AnthropicMessagesRequest(BaseModel):
    model: Optional[str] = None
    max_tokens: int = 1024
    messages: list[AnthropicMessage]
    system: Optional[Any] = None  # str | list[{type:"text",text:"..."}]
    tools: Optional[list[AnthropicTool]] = None
    tool_choice: Optional[Any] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    stop_sequences: Optional[list[str]] = None
    metadata: Optional[dict] = None  # we read user_id as a session id when present


def _antc_text_from_blocks(content: Any) -> str:
    """Anthropic content can be str or list of blocks. Extract concatenated text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(str(b.get("text", "")))
                elif b.get("type") == "tool_result":
                    inner = b.get("content")
                    if isinstance(inner, str):
                        parts.append(inner)
                    elif isinstance(inner, list):
                        for ib in inner:
                            if isinstance(ib, dict) and ib.get("type") == "text":
                                parts.append(str(ib.get("text", "")))
        return "\n".join(parts)
    return ""


def _antc_to_openai_messages(req: AnthropicMessagesRequest) -> list[dict]:
    """Flatten Anthropic system + messages into OpenAI-shaped messages.

    Assistant tool_use blocks → assistant message with tool_calls.
    User tool_result blocks   → tool message(s) (one per tool_use_id).
    """
    out: list[dict] = []
    sys_text = ""
    if isinstance(req.system, str):
        sys_text = req.system
    elif isinstance(req.system, list):
        sys_text = _antc_text_from_blocks(req.system)
    if sys_text:
        out.append({"role": "system", "content": sys_text})

    for m in req.messages:
        role = m.role
        content = m.content
        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text":
                    text_parts.append(str(b.get("text", "")))
                elif t == "tool_use":
                    tool_calls.append({
                        "id": b.get("id") or ("call_" + uuid.uuid4().hex[:12]),
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": json.dumps(b.get("input", {})),
                        },
                    })
            msg: dict = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        elif role == "user" and isinstance(content, list):
            # Split tool_result blocks into separate `tool` messages, keep text together.
            text_parts: list[str] = []
            tool_msgs: list[dict] = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text":
                    text_parts.append(str(b.get("text", "")))
                elif t == "tool_result":
                    inner = b.get("content")
                    if isinstance(inner, list):
                        inner_text = "\n".join(
                            str(ib.get("text", "")) for ib in inner
                            if isinstance(ib, dict) and ib.get("type") == "text"
                        )
                    else:
                        inner_text = str(inner) if inner is not None else ""
                    tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id", ""),
                        "content": inner_text,
                    })
            if text_parts:
                out.append({"role": "user", "content": "\n".join(text_parts)})
            out.extend(tool_msgs)
        else:
            out.append({"role": role, "content": _antc_text_from_blocks(content)})
    return out


def _antc_tools_to_openai(tools: Optional[list[AnthropicTool]]) -> Optional[list[dict]]:
    if not tools:
        return None
    return [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description or "",
            "parameters": t.input_schema,
        },
    } for t in tools]


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(req: AnthropicMessagesRequest):
    """Anthropic Messages token-count endpoint.

    Claude Code probes this at startup (and before each turn) to manage
    its context window — without it, Claude Code refuses to talk to a
    custom ANTHROPIC_BASE_URL. The official API returns `{input_tokens: N}`.

    We don't run the target model's tokenizer here (it lives remote on the
    runner, and count_tokens must be cheap + synchronous), so we return a
    char-based estimate. ~4 chars/token is the standard heuristic for
    English/code; it's close enough for context-budget decisions, which is
    all the caller uses it for. We round UP (ceil) so we never *under*-count
    and let the client overflow the window.
    """
    # Flatten system + messages to text via the same path /v1/messages uses.
    text_parts: list[str] = []
    try:
        for m in _antc_to_openai_messages(req):
            c = m.get("content")
            if isinstance(c, str):
                text_parts.append(c)
            # tool_calls carry JSON arguments that also cost tokens
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                text_parts.append(str(fn.get("name", "")))
                text_parts.append(str(fn.get("arguments", "")))
    except Exception:
        # Defensive: fall back to a raw text extraction if the flatten path
        # trips on an unexpected block shape.
        if isinstance(req.system, str):
            text_parts.append(req.system)
        for m in req.messages:
            text_parts.append(_antc_text_from_blocks(m.content))

    # Tool schemas are part of the prompt the model sees — count them too.
    if req.tools:
        try:
            text_parts.append(json.dumps(_antc_tools_to_openai(req.tools)))
        except Exception:
            pass

    total_chars = sum(len(p) for p in text_parts if p)
    input_tokens = max(1, -(-total_chars // 4))  # ceil(total_chars / 4)
    return {"input_tokens": input_tokens}


@app.post("/v1/messages")
async def anthropic_messages(req: AnthropicMessagesRequest, request: Request):
    # 0a. Tier routing: Claude Code / Anthropic SDK send canonical tier
    # names (claude-opus/sonnet/haiku-*). Map them to an operator-chosen
    # local model so `claude` works against ANTHROPIC_BASE_URL without the
    # user pinning ANTHROPIC_MODEL. No-op for non-claude model ids.
    req.model = _resolve_anthropic_tier(req.model)

    # 0. Telemak passthrough? kind=telemak clusters expose /v1/messages
    # natively (V1, Block 4). Same cluster_id / cluster:short alias
    # convention as /v1/chat/completions.
    if req.model:
        tele_cluster, tele_short = _telemak_split_alias(req.model)
        if tele_cluster and cluster_exists(tele_cluster):
            cd_t = get_cluster_def(tele_cluster)
            if cd_t.get("kind") == "telemak":
                body = req.model_dump(exclude_none=True)
                return await _telemak_proxy_messages(
                    tele_cluster, cd_t, body, bool(req.stream),
                    requested_short_id=tele_short,
                )

    # If the model alias maps to a cloud provider, proxy to upstream.
    # Anthropic-protocol upstreams (api.anthropic.com) get a direct
    # passthrough — same wire format on both ends. OpenAI-protocol
    # upstreams could be reached via translation, but that's a bigger
    # surface and not needed today (clients that want OpenAI shape hit
    # /v1/chat/completions; clients that want Anthropic shape hit
    # /v1/messages and want a native Anthropic upstream behind it).
    cloud_match = find_cloud_alias(req.model)
    if cloud_match:
        prov_id, prov, entry = cloud_match
        if _provider_protocol(prov) == "anthropic":
            return await _proxy_anthropic_messages(prov_id, prov, entry, req, request)
        raise HTTPException(
            400,
            f"alias '{req.model}' is on an OpenAI-protocol provider ('{prov_id}'). "
            f"Use POST /v1/chat/completions or pick an alias on an anthropic-protocol provider."
        )

    pool = _route_pool(req.model)
    if pool is None or pool.alive_count() == 0:
        # Same 404 shape as /v1/chat/completions — see comment there.
        ready = []
        # Every loaded Default alias is servable (default "default" + extras).
        for cid, alias, _ in list_all_pools():
            ready.append(alias if alias != DEFAULT_ALIAS else cid)
        if _pool is not None:         ready.append("nautilus")
        try:
            ready.extend(e["id"] for e in _cloud_entries_for_v1_models())
        except Exception:
            pass
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_loaded",
                "message": f"Model {req.model!r} is not currently loaded.",
                "ready_models": ready,
                "hint": "GET /v1/models lists everything ready.",
            },
        )

    # Request classification (same logic as /v1/chat/completions).
    # Anthropic shape always has max_tokens, no need to guard for None.
    prompt_chars_pre = sum(
        len(b.text or "")
        for m in (req.messages or [])
        if isinstance(getattr(m, "content", None), list)
        for b in m.content if getattr(b, "type", None) == "text"
    ) + sum(
        len(m.content) for m in (req.messages or [])
        if isinstance(getattr(m, "content", None), str)
    )
    req_class = classify_request(
        model_id=req.model,
        max_tokens=req.max_tokens,
        prompt_chars=prompt_chars_pre,
        has_tools=bool(req.tools),
        header_hint=request.headers.get("x-odyssai-job"),
    )
    refuse = _refuse_message_for(pool.cluster, req_class)
    if refuse:
        raise HTTPException(status_code=400, detail=refuse)

    oa_messages = _antc_to_openai_messages(req)
    oa_tools = _antc_tools_to_openai(req.tools)
    msg_id = "msg_" + uuid.uuid4().hex[:24]
    model_id = pool.model
    client_ip = request.client.host if request.client else "?"
    prompt_chars = sum(len(m.get("content") or "") for m in oa_messages)
    # Prefix-cache session id: header > metadata.user_id (Anthropic convention).
    session_id = (request.headers.get("x-session-id")
                  or request.headers.get("X-Session-Id")
                  or (req.metadata or {}).get("user_id")
                  or (req.metadata or {}).get("session_id"))

    if not req.stream:
        text_parts: list[str] = []
        ntoks = 0
        elapsed_s = 0.0
        ttft_s: Optional[float] = None
        tool_calls: list[dict] = []
        session_meta: dict = {}
        t_start = time.time()
        async for ev in pool.submit(None, req.max_tokens, None,
                                    messages=oa_messages, tools=oa_tools,
                                    session_id=session_id,
                                    request_id=msg_id,
                                    reasoning_effort=_default_reasoning_effort(model_id)):
            if ev.get("event") == "token":
                if ttft_s is None:
                    ttft_s = time.time() - t_start
                text_parts.append(ev.get("text", ""))
            elif ev.get("event") == "done":
                ntoks = ev.get("ntoks", 0)
                elapsed_s = ev.get("elapsed_s", 0.0)
                tool_calls = ev.get("tool_calls", []) or []
                session_meta.update(ev.get("session", {}) or {})
        _touch_session(pool.cluster, session_meta, model_id)
        record_metric(client_ip, ntoks, elapsed_s, ttft_s, prompt_chars, model_id,
                      cluster=pool.cluster, tool_calls=len(tool_calls),
                      session_kind=session_meta.get("cache_kind"))
        # Build Anthropic content blocks
        blocks: list[dict] = []
        text = _strip_tool_calls_from_text("".join(text_parts)) if tool_calls else "".join(text_parts)
        if text:
            blocks.append({"type": "text", "text": text})
        for tc in tool_calls:
            try:
                inp = json.loads(tc["function"]["arguments"]) if tc["function"].get("arguments") else {}
            except Exception:
                inp = {"_raw": tc["function"].get("arguments", "")}
            blocks.append({
                "type": "tool_use",
                "id": "toolu_" + tc["id"].replace("call_", "")[:24],
                "name": tc["function"]["name"],
                "input": inp,
            })
        stop_reason = "tool_use" if tool_calls else "end_turn"
        return JSONResponse({
            "id": msg_id, "type": "message", "role": "assistant",
            "model": model_id,
            "content": blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_chars // 4, "output_tokens": ntoks},  # cheap estimate
            "x_mlx_cluster": {
                "elapsed_s": elapsed_s, "ttft_s": ttft_s,
                "tps": ntoks / elapsed_s if elapsed_s else 0.0,
            },
        })

    async def antc_stream() -> AsyncIterator[bytes]:
        ttft_s: Optional[float] = None
        t_start = time.time()
        ntoks_total = 0
        elapsed_total = 0.0
        text_parts_local: list[str] = []
        text_block_open = False
        tool_calls_final: list[dict] = []
        try:
            # message_start
            ms = {
                "type": "message_start",
                "message": {
                    "id": msg_id, "type": "message", "role": "assistant",
                    "content": [], "model": model_id,
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
            yield f"event: message_start\ndata: {json.dumps(ms)}\n\n".encode()

            # content_block_start (text)
            cbs = {"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}}
            yield f"event: content_block_start\ndata: {json.dumps(cbs)}\n\n".encode()
            text_block_open = True

            tool_calls_final: list[dict] = []
            async for ev in pool.submit(None, req.max_tokens, None,
                                        messages=oa_messages, tools=oa_tools,
                                        session_id=session_id,
                                        request_id=msg_id,
                                        reasoning_effort=_default_reasoning_effort(model_id)):
                if await request.is_disconnected():
                    break
                if ev.get("event") == "token":
                    if ttft_s is None:
                        ttft_s = time.time() - t_start
                    txt = ev.get("text", "")
                    text_parts_local.append(txt)
                    delta = {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "text_delta", "text": txt}}
                    yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode()
                elif ev.get("event") == "done":
                    ntoks_total = ev.get("ntoks", 0)
                    elapsed_total = ev.get("elapsed_s", 0.0)
                    tool_calls_final = ev.get("tool_calls", []) or []

            # close text block
            if text_block_open:
                yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n".encode()

            # tool_use blocks (one per call) — emitted as start/stop without deltas
            for i, tc in enumerate(tool_calls_final, start=1):
                try:
                    inp = json.loads(tc["function"]["arguments"]) if tc["function"].get("arguments") else {}
                except Exception:
                    inp = {}
                tu_block = {"type": "tool_use",
                            "id": "toolu_" + tc["id"].replace("call_", "")[:24],
                            "name": tc["function"]["name"],
                            "input": inp}
                yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':i,'content_block':tu_block})}\n\n".encode()
                yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':i})}\n\n".encode()

            # message_delta (stop_reason + usage)
            stop_reason = "tool_use" if tool_calls_final else "end_turn"
            md = {"type": "message_delta",
                  "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                  "usage": {"output_tokens": ntoks_total}}
            yield f"event: message_delta\ndata: {json.dumps(md)}\n\n".encode()

            # message_stop
            yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n".encode()
        finally:
            record_metric(client_ip, ntoks_total, elapsed_total, ttft_s, prompt_chars, model_id,
                          cluster=pool.cluster, tool_calls=len(tool_calls_final))

    return StreamingResponse(antc_stream(), media_type="text/event-stream")


# ──────────────────────────────────────────────────────────────────────────────
# Admin endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/admin/status")
async def admin_status():
    loading = _loading_snapshot(_nautilus_loading)
    # Honest "is anything loaded". The legacy single `_pool` (Nautilus) has
    # been dissolved since 2026-05 and is always None, so reporting it alone
    # made this endpoint claim loaded:false even when a CLUSTER pool (Argo/
    # main, telemak…) was loaded. Aggregate the real pools via list_all_pools()
    # — the same source /v1/models uses — so the signal stops lying.
    cluster_pools = list_all_pools()  # [(cluster_id, alias, RunnerPool), …]
    pools_out = [
        {
            "cluster": cid,
            "alias": alias,
            "model": pool.model,
            "nodes": pool.nodes_count,
            "alive": pool.alive_count(),
        }
        for cid, alias, pool in cluster_pools
    ]
    if _pool is None and not cluster_pools:
        return {"loaded": False, "loading": loading, "pools": []}
    if _pool is not None:
        uptime = time.time() - (_pool.started_at or time.time())
        recent_tps = [m["tps"] for m in list(_metrics)[:10] if m["tps"] > 0]
        return {
            "loaded": True,
            "loading": loading,
            "model": _pool.model, "mode": _pool.mode,
            "use_ap": _pool.use_ap, "nodes": _pool.nodes_count,
            "kv_q8": _pool.kv_q8,
            "draft_model": _pool.draft_model,
            "num_draft_tokens": _pool.num_draft_tokens if _pool.draft_model else None,
            "alive": _pool.alive_count(),
            "load_s": _pool.load_s, "uptime_s": uptime,
            "recent_avg_tps": round(sum(recent_tps) / len(recent_tps), 2) if recent_tps else None,
            "recent_count": len(_metrics),
            "pools": pools_out,
        }
    # Cluster-pool-only reality (current): report honestly from the loaded pools.
    return {
        "loaded": True,
        "loading": loading,
        "model": cluster_pools[0][2].model,
        "pools": pools_out,
    }


@app.get("/admin/models")
async def admin_models(dir: Optional[str] = None):
    """Discover models on Nautilus rank 0 (under `dir` or the saved models_dir)."""
    rank0 = rank0_ssh_for_cluster("nautilus")
    target_dir = dir or models_dir_for("nautilus")
    if dir:
        # Persist the override on explicit query
        set_models_dir("nautilus", dir)
    models = await asyncio.to_thread(discover_models_on_node, rank0, target_dir)
    annotated = []
    for m in models:
        annotated.append({
            "id": m,
            "kind": "path",
            "is_loaded": _pool is not None and _pool.model == m,
        })
    return {"data": annotated, "models_dir": target_dir}


class ModelsDirRequest(BaseModel):
    dir: str


@app.post("/admin/models-dir")
async def admin_models_dir(req: ModelsDirRequest):
    set_models_dir("nautilus", req.dir)
    return {"cluster": "nautilus", "models_dir": req.dir}


class LoadRequest(BaseModel):
    model: str
    mode: str = "pipeline"  # "pipeline" | "tensor"
    use_ap: bool = False
    nodes: int = 2  # 2 | 3 | 4
    # None = take the cluster-wide default from /admin/settings (kv_q8_default).
    # Explicit true/false still wins per request — important for eval runs that
    # want a known cache type regardless of the global setting.
    kv_q8: Optional[bool] = None
    draft_model: Optional[str] = None  # speculative decoding, single-rank only
    num_draft_tokens: int = 4
    # Hot-swap: when False (default since 2026-05-18 audit), the old pool
    # is stopped BEFORE the new one starts — no double-allocation in RAM.
    # Set True to overlap the load with the old serving (faster cutover,
    # but doubles RAM transiently). Default False prevents OOM on big models.
    force_hot_swap: bool = False


@app.post("/admin/load")
async def admin_load(req: LoadRequest):
    if req.mode not in ("pipeline", "tensor"):
        raise HTTPException(400, f"invalid mode {req.mode}")
    if "nautilus" not in DEFAULT_CLUSTER_DEFS:
        raise HTTPException(404, "nautilus topology is not configured")
    max_nodes = len(get_cluster_def("nautilus").get("nodes", [])) or 1
    if req.nodes < 1 or req.nodes > max_nodes:
        raise HTTPException(400, f"nautilus: invalid nodes {req.nodes}")

    # Resolve kv_q8: explicit request value wins; fall back to cluster default.
    kv_q8 = req.kv_q8 if req.kv_q8 is not None else get_kv_q8_default()

    rank0_ssh = rank0_ssh_for_cluster("nautilus", req.nodes)
    size_bytes = await get_model_size_bytes(rank0_ssh, req.model)
    estimated_s = estimate_load_s(req.model, size_bytes, "nautilus", req.nodes)

    global _pool
    async with _admin_lock:
        _begin_loading(_nautilus_loading, req.model, req.nodes, size_bytes, estimated_s)
        try:
            old = _pool
            # Stop-old-before-start-new by default. Hot-swap kept its old
            # name but the gate is now opt-in. See LoadRequest.force_hot_swap.
            if old is not None and not req.force_hot_swap:
                sys.stderr.write("[load] nautilus: stopping old pool before starting new (no hot-swap)\n")
                await old.stop()
                old = None  # mark consumed
                _pool = None
            new_pool = RunnerPool(
                model=req.model, mode=req.mode,
                use_ap=req.use_ap, nodes_count=req.nodes,
                kv_q8=kv_q8,
                draft_model=req.draft_model,
                num_draft_tokens=req.num_draft_tokens,
            )
            try:
                await new_pool.start()
            except Exception as e:
                try:
                    await new_pool.stop()
                except Exception:
                    pass
                raise HTTPException(500, f"load failed: {e}")
            if old is not None:
                # Only reached when force_hot_swap=True — the old pool ran
                # alongside the new one's load.
                await old.stop()
            _pool = new_pool
            save_state(req.model, req.mode, req.use_ap, req.nodes, kv_q8=kv_q8)
            record_load_history("nautilus", req.model, _pool.load_s or 0.0,
                                size_bytes, req.nodes)
        finally:
            _end_loading(_nautilus_loading)
    await _maybe_auto_prewarm(_pool, "nautilus")
    return {"loaded": True, "model": _pool.model, "load_s": _pool.load_s}


@app.post("/admin/unload")
async def admin_unload():
    global _pool
    async with _admin_lock:
        if _pool is None:
            # See admin_cluster_unload for the rationale: reap orphans even
            # when pool state is None.
            sweep = await asyncio.to_thread(_sweep_orphan_runners, "nautilus")
        else:
            await _pool.stop()
            _pool = None
            sweep = await asyncio.to_thread(_sweep_orphan_runners, "nautilus")
    # Always clear persisted state so next start doesn't auto-reload.
    try:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
    except Exception:
        pass
    return {"loaded": False, "sweep": sweep}


@app.get("/admin/logs")
async def admin_logs(cluster: str = "default", tail: int = 100, follow: bool = False,
                     request: Request = None):
    """Runner stderr logs.

    - `tail` only: returns last N lines as JSON (no streaming).
    - `follow=true`: SSE stream, replays last `tail` lines then sends every
      new line as a JSON event. Closes on client disconnect.
    """
    # Lazy-init: cluster may be a topology.yaml-defined id ('main',
    # 'telemak-max64', …) that wasn't pre-seeded into _log_buffers.
    # Only refuse the request if the cluster id is truly unknown to
    # the orchestrator.
    if cluster not in _log_buffers and not cluster_exists(cluster):
        raise HTTPException(400, f"unknown cluster {cluster}")
    _ensure_log_cluster(cluster)
    if not follow:
        rows = list(_log_buffers[cluster])[-tail:]
        return {"cluster": cluster, "data": rows}

    async def stream() -> AsyncIterator[bytes]:
        # 1) Replay the tail
        seen = list(_log_buffers[cluster])[-tail:]
        for line in seen:
            yield f"data: {json.dumps(line)}\n\n".encode()
        # 2) Subscribe to new lines via an asyncio queue
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        with _log_lock:
            _log_subscribers[cluster].append(q)
        try:
            while True:
                if request is not None and await request.is_disconnected():
                    break
                try:
                    line = await asyncio.wait_for(q.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    # Keep-alive comment so proxies don't drop the connection.
                    yield b": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(line)}\n\n".encode()
        finally:
            with _log_lock:
                try:
                    _log_subscribers[cluster].remove(q)
                except ValueError:
                    pass

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/admin/sessions")
async def admin_sessions(cluster: Optional[str] = None):
    """Live prefix-cache sessions (last seen < 1 h). Lightweight tracker
    populated from every chat-completions done event with `session.id`.
    """
    rows = _live_sessions()
    if cluster:
        rows = [r for r in rows if r.get("cluster") == cluster]
    return {
        "data": sorted(rows, key=lambda r: -r["last_seen"]),
        "count": len(rows),
    }


@app.get("/admin/metrics")
async def admin_metrics(cluster: Optional[str] = None, limit: int = 50,
                        per_cluster: int = 20):
    """Recent request metrics, optionally filtered by cluster.

    Without a `cluster` filter the response is balanced PER cluster (up to
    `per_cluster` rows each), not the global last-`limit`. This is the #32 fix:
    a high-volume cluster (the autocomplete service) used to fill the global
    50-slot ring buffer in minutes and crowd every other cluster out of the
    dashboard's "Recent activity" panel.

    Each entry: ts, client, cluster, model, ntoks, elapsed_s, ttft_s, tps,
    prompt_chars, tool_calls, session_kind.
    """
    if cluster:
        dq = _metrics_by_cluster.get(cluster)
        if dq is not None:
            rows = list(dq)[:limit]
        else:
            # Cluster hasn't recorded since restart — fall back to the global
            # buffer so the contract still returns whatever's there.
            rows = [r for r in _metrics if r.get("cluster") == cluster][:limit]
        return {"data": rows, "filter": {"cluster": cluster, "limit": limit}}
    # No filter: take up to per_cluster from each cluster, merge, sort by ts —
    # every cluster gets representation, none can monopolise the panel.
    merged: list[dict] = []
    for dq in _metrics_by_cluster.values():
        merged.extend(list(dq)[:per_cluster])
    merged.sort(key=lambda r: r.get("ts", 0), reverse=True)
    rows = merged if merged else list(_metrics)
    return {"data": rows, "filter": {"cluster": None, "limit": limit,
                                     "per_cluster": per_cluster}}


class HFDownloadRequest(BaseModel):
    repo: str  # "org/name"
    hf_token: Optional[str] = None
    # Host ids from HOSTS_REGISTRY. Each target uses its own models_dir.
    # Empty list means "first configured host" so open-source installs do
    # not inherit any author's LAN target.
    targets: list[str] = Field(default_factory=list)


@app.post("/admin/downloads")
async def admin_downloads_create(req: HFDownloadRequest):
    """Start a HuggingFace download to one or more cluster nodes in parallel.

    Each target uses its node's `models_dir` (declared in topology.yaml or
    edited via the dashboard). The HF CLI resumes partial files by default,
    so POSTing the same `repo` + `targets` after a cancel picks up where it
    left off — see /admin/downloads/{id}/resume for the explicit one-click flow.
    """
    if "/" not in req.repo:
        raise HTTPException(400, "repo must look like 'org/name'")
    targets = req.targets or ([HOSTS_REGISTRY[0]["id"]] if HOSTS_REGISTRY else [])
    if not targets:
        raise HTTPException(400, "at least one configured target required")
    resolved = []
    for tid in targets:
        h = next((x for x in HOSTS_REGISTRY if x["id"] == tid), None)
        if not h:
            raise HTTPException(400, f"unknown target {tid}")
        resolved.append(h)
    dl_id = uuid.uuid4().hex[:8]
    _downloads[dl_id] = {
        "id": dl_id,
        "repo": req.repo,
        "started_at": time.time(),
        "finished_at": None,
        "status": "running",
        "size": "0B",
        "bytes": 0,
        "targets": [h["id"] for h in resolved],
        "per_target": [],
        "hf_token_set": bool(req.hf_token),
        "error": None,
    }
    asyncio.create_task(_hf_dl_run(dl_id, req.repo, req.hf_token, resolved))
    return {"id": dl_id}


@app.get("/admin/downloads")
async def admin_downloads_list():
    return {"data": list(_downloads.values())}


@app.get("/admin/hf/search")
async def admin_hf_search(q: str, limit: int = 20):
    """Proxy HF model search (borrowed from the HF tools app). Returns a trimmed
    list the dashboard's Hugging Face tab renders into clickable download targets."""
    q = (q or "").strip()
    if not q:
        return {"data": []}
    params = {
        "search": q,
        "limit": max(1, min(limit, 50)),
        "sort": "downloads",
        "direction": -1,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get("https://huggingface.co/api/models", params=params)
            if r.status_code != 200:
                raise HTTPException(502, f"HF search failed: {r.status_code}")
            out = [
                {
                    "id": m.get("id") or m.get("modelId"),
                    "downloads": m.get("downloads"),
                    "likes": m.get("likes"),
                    "pipeline_tag": m.get("pipeline_tag"),
                    "updated": m.get("lastModified"),
                }
                for m in r.json()
                if (m.get("id") or m.get("modelId"))
            ]
            return {"data": out}
    except httpx.HTTPError as e:
        raise HTTPException(502, f"HF search error: {e}")


@app.delete("/admin/downloads/{dl_id}")
async def admin_downloads_cancel(dl_id: str, host: Optional[str] = None):
    """Cancel a download. If `host` is given, only cancel that target;
    otherwise cancel every running target. Cancelled targets can be
    resumed via POST /admin/downloads/{id}/resume — partial files on
    disk are kept and `hf download` continues from where it left."""
    d = _downloads.get(dl_id)
    if not d:
        raise HTTPException(404, "no such download")
    by_host = _dl_procs.get(dl_id) or {}
    targets = [host] if host else list(by_host.keys())
    for h in targets:
        proc = by_host.get(h)
        if proc and proc.returncode is None:
            try:
                proc.terminate()
            except Exception:
                pass
            # Mark the per-target slot so the runner doesn't relabel as error.
            for slot in (d.get("per_target") or []):
                if slot.get("host") == h:
                    slot["status"] = "cancelled"
    # If we cancelled them all, mirror at the top level so the UI shows
    # a Resume button. Otherwise leave it running for the surviving targets.
    if not host and not any(p.returncode is None for p in by_host.values()):
        d["status"] = "cancelled"
    return {"ok": True}


@app.post("/admin/downloads/{dl_id}/resume")
async def admin_downloads_resume(dl_id: str):
    """Restart a previously cancelled / failed download. Picks up partial
    files via HF CLI's snapshot_download resume_download=True. We keep the
    original `repo` + `targets` and re-fan-out; targets already complete
    are skipped (HF CLI is a no-op when local files already match remote)."""
    d = _downloads.get(dl_id)
    if not d:
        raise HTTPException(404, "no such download")
    if d.get("status") == "running":
        raise HTTPException(409, "already running")
    resolved = []
    for tid in d.get("targets", []):
        h = next((x for x in HOSTS_REGISTRY if x["id"] == tid), None)
        if h:
            resolved.append(h)
    if not resolved:
        raise HTTPException(400, "no valid targets recorded on this download")
    # Reset top-level state. Per-target slots are re-initialised by the runner.
    d["status"] = "running"
    d["error"] = None
    d["finished_at"] = None
    d["started_at"] = time.time()
    d["per_target"] = []
    # We don't store the HF token (security) — caller must re-pass it via
    # /admin/downloads if their resume needs auth. For unauthenticated repos
    # this is fine.
    asyncio.create_task(_hf_dl_run(dl_id, d["repo"], None, resolved))
    return {"ok": True}


# ──────────────────────────────────────────────────────────────────────────────
# Admin: cross-host sync matrix (model presence per node + free space)
# ──────────────────────────────────────────────────────────────────────────────
def _build_hosts_registry() -> list[dict]:
    """Build the managed-host registry from topology/default cluster defs.

    This drives telemetry, downloads, sync, deletes and reboot. It must never
    contain author-specific LAN defaults; operators declare hosts in
    topology.yaml or edit clusters through the dashboard.
    """
    inventory = {h["id"]: h for h in KNOWN_HOSTS}
    by_id: dict[str, dict] = {}
    cluster_sets: dict[str, set[str]] = {}

    for cluster_id, cd in DEFAULT_CLUSTER_DEFS.items():
        cluster_models_dir = cd.get("models_dir") or models_dir_for(cluster_id)
        for node in cd.get("nodes") or []:
            ssh = node.get("ssh") or ""
            host_id = node.get("host") or _host_id_from_ssh(ssh)
            if not host_id:
                continue
            inv = inventory.get(host_id, {})
            entry = by_id.setdefault(host_id, {
                "id": host_id,
                "ssh": ssh or inv.get("ssh"),
                "models_dir": node.get("models_dir") or cluster_models_dir or DEFAULT_MODELS_DIR,
                "cluster": cluster_id,
                "label": inv.get("label") or host_id,
            })
            if not entry.get("ssh") and inv.get("ssh"):
                entry["ssh"] = inv["ssh"]
            if node.get("models_dir"):
                entry["models_dir"] = node["models_dir"]
            cluster_sets.setdefault(host_id, set()).add(cluster_id)

    for host_id, inv in inventory.items():
        if host_id not in by_id:
            by_id[host_id] = {
                "id": host_id,
                "ssh": inv.get("ssh"),
                "models_dir": DEFAULT_MODELS_DIR,
                "cluster": "",
                "label": inv.get("label") or host_id,
            }

    # Also walk UI-added clusters from cluster-config.json so Telemak (and any
    # other) clusters registered through the "+ Add Telemak" dashboard form
    # contribute their nodes to the host registry. Without this, the matrix
    # probe never sees those hosts → Models card stays empty.
    try:
        for cluster_id, entry in _load_cluster_config().items():
            if cluster_id in DEFAULT_CLUSTER_DEFS:
                continue  # already handled above
            if not isinstance(entry, dict) or entry.get("_removed"):
                continue
            if not entry.get("nodes"):
                continue
            cluster_models_dir = entry.get("models_dir") or models_dir_for(cluster_id)
            for node in entry.get("nodes") or []:
                ssh = node.get("ssh") or ""
                host_id = node.get("host") or _host_id_from_ssh(ssh)
                if not host_id or not ssh:
                    continue
                inv = inventory.get(host_id, {})
                rec = by_id.setdefault(host_id, {
                    "id": host_id,
                    "ssh": ssh,
                    "models_dir": node.get("models_dir") or cluster_models_dir or DEFAULT_MODELS_DIR,
                    "cluster": cluster_id,
                    "label": inv.get("label") or host_id,
                })
                if not rec.get("ssh"):
                    rec["ssh"] = ssh
                if node.get("models_dir"):
                    rec["models_dir"] = node["models_dir"]
                cluster_sets.setdefault(host_id, set()).add(cluster_id)
    except Exception:
        # Probe must not crash on a malformed cluster-config.json — fall
        # back to topology-only hosts in that case.
        pass

    for host_id, entry in by_id.items():
        clusters = cluster_sets.get(host_id)
        if clusters:
            entry["cluster"] = "+".join(sorted(clusters))
    return [h for h in by_id.values() if h.get("ssh")]


# Module-import snapshot. The matrix endpoint calls _build_hosts_registry()
# fresh so UI-added clusters (cluster-config.json) are picked up without a
# container restart — this constant is kept for legacy callers that may
# import HOSTS_REGISTRY directly.
HOSTS_REGISTRY = _build_hosts_registry()

# In-memory cache: { "matrix": {...}, "ts": epoch }.
_sync_matrix_cache: dict = {"ts": 0.0, "data": None}
_SYNC_MATRIX_TTL_S = 60.0


def _ssh_exec(ssh_target: str, cmd: str, timeout: int = 12) -> tuple[int, str, str]:
    """Shared SSH helper for the matrix endpoint."""
    p = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=4", "-o", "BatchMode=yes", _safe_ssh_target(ssh_target), cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return p.returncode, p.stdout, p.stderr


async def _probe_host(host: dict) -> dict:
    """For one host: free bytes at models_dir + every direct subdir with sizeBytes."""
    ssh = host["ssh"]
    md  = host["models_dir"]
    out = {
        "id": host["id"],
        "label": host["label"],
        "ssh": ssh,
        "models_dir": md,
        "cluster": host["cluster"],
        "reachable": False,
        "free_bytes": None,
        "models": [],   # [{name, size_bytes}]
        "error": None,
    }
    # Combined command: df + du in one round-trip to keep latency low.
    # `du -sb` is GNU; on macOS use `du -sk` (1024-byte blocks) and convert.
    #
    # Two-level walk for the new layout (2026-05-18): models now live under
    # org folders like `{md}/inferencerlabs/Hy3-preview-MLX-9bit/`. A folder
    # is treated as a model if it contains a `config.json` (HF/MLX convention);
    # otherwise we list ITS children one level deeper and emit names as
    # `org/model`. Top-level model folders (legacy layout) still work.
    # The probe must NOT run in the node's login shell: zsh aborts the WHOLE
    # command on an unmatched glob ("no matches found" — e.g. an org dir that
    # exists but is still empty on a fresh node), so the node showed zero
    # models despite having some. POSIX sh leaves unmatched globs literal and
    # the [ -d ] guards filter them — wrap everything in /bin/sh -c.
    inner = (
        f"if [ ! -d {shlex.quote(md)} ]; then echo ___NOMODELS___; exit 0; fi; "
        f"FREE=$(df -k {shlex.quote(md)} | awk 'NR==2 {{print $4}}'); "
        f"echo \"___FREE___$FREE\"; "
        f"for d in {shlex.quote(md)}/*/; do "
        f"  [ -d \"$d\" ] || continue; "
        f"  if [ -f \"$d/config.json\" ]; then "
        f"    SZ=$(du -sk \"$d\" 2>/dev/null | awk '{{print $1}}'); "
        f"    NAME=$(basename \"$d\"); "
        f"    echo \"$SZ:$NAME\"; "
        f"  else "
        f"    PARENT=$(basename \"$d\"); "
        f"    for sub in \"$d\"*/; do "
        f"      [ -d \"$sub\" ] || continue; "
        f"      SZ=$(du -sk \"$sub\" 2>/dev/null | awk '{{print $1}}'); "
        f"      NAME=$(basename \"$sub\"); "
        f"      echo \"$SZ:$PARENT/$NAME\"; "
        f"    done; "
        f"  fi; "
        f"done"
    )
    cmd = f"/bin/sh -c {shlex.quote(inner)}"
    try:
        rc, stdout, stderr = await asyncio.to_thread(_ssh_exec, ssh, cmd, 14)
        if rc != 0:
            out["error"] = (stderr or "ssh failed")[:200]
            return out
        if "___NOMODELS___" in stdout:
            out["reachable"] = True
            return out
        out["reachable"] = True
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("___FREE___"):
                try:
                    out["free_bytes"] = int(line.split("___", 2)[2]) * 1024
                except Exception:
                    pass
                continue
            if ":" in line:
                size_kb, name = line.split(":", 1)
                try:
                    out["models"].append({"name": name.strip(), "size_bytes": int(size_kb) * 1024})
                except Exception:
                    pass
    except subprocess.TimeoutExpired:
        out["error"] = "timeout"
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


def _pivot_matrix(per_host: list[dict]) -> dict:
    """Pivot per-host listings into a per-model matrix.
    Output schema:
      hosts: [{id, label, ssh, models_dir, cluster, reachable, free_bytes, error}, ...]
      models: [
        { name, total_bytes, presence: {host_id: size_bytes}, present_on: [host_ids] }
      ]
    """
    hosts = []
    model_index: dict[str, dict] = {}
    for h in per_host:
        hosts.append({k: h[k] for k in ("id","label","ssh","models_dir","cluster","reachable","free_bytes","error")})
        for m in h["models"]:
            name = m["name"]
            sz = m["size_bytes"]
            entry = model_index.setdefault(name, {"name": name, "total_bytes": 0, "presence": {}, "present_on": []})
            entry["presence"][h["id"]] = sz
            entry["present_on"].append(h["id"])
            entry["total_bytes"] = max(entry["total_bytes"], sz)
    # Sort models by size desc to put biggest first
    models = sorted(model_index.values(), key=lambda m: -m["total_bytes"])
    return {"hosts": hosts, "models": models}


_telemetry_cache: dict = {"ts": 0.0, "data": None}
_TELEMETRY_TTL_S = 4.0  # short cache since the dashboard polls every 5s


async def _probe_telemetry(host: dict) -> dict:
    """Per-host RAM usage + SSH latency + runner & RDMA observability.

    Returns:
      { id, label, ssh, reachable, latency_ms,
        ram_total_bytes, ram_used_bytes, ram_wired_bytes, ram_pct,
        wired_limit_mb,
        runner_count,                # number of runner.py procs alive
        runner_procs: [              # ps detail per runner (top 4 by RSS)
          {pid:int, cpu_pct:float, rss_bytes:int, etime:str}
        ],
        rdma_ports: [                # ibv_devinfo state per HCA port
          {name:str, port:int, state:str}     # ACTIVE / DOWN / INIT / ARMED
        ] | None,                    # None on hosts without RDMA
        error }

    Memory model on macOS: "used" = active + wired + compressed (Activity Monitor convention).

    Why each field:
      - runner_count   → orphan detection ("loaded:false" but procs alive)
      - runner_procs   → spot the rank holding 200 GB when others don't
      - rdma_ports     → JACCL "errno 16" pre-flight: any port not ACTIVE = reboot
                         needed before next load. Only collected if ibv_devinfo
                         exists; gracefully None on non-RDMA hosts.
    """
    ssh = host["ssh"]
    out = {
        "id": host["id"],
        "label": host["label"],
        "ssh": ssh,
        "reachable": False,
        "latency_ms": None,
        "ram_total_bytes": None,
        "ram_used_bytes": None,
        "ram_wired_bytes": None,
        "ram_pct": None,
        "wired_limit_mb": None,
        "runner_count": None,
        "runner_procs": None,
        "rdma_ports": None,
        "error": None,
    }
    # One-shot probe: vm_stat + sysctls + runner.py inventory + ibv_devinfo.
    # All sections are prefixed with `SECTION:<name>` so the parser can split
    # cleanly without ambiguity.
    cmd = (
        "PS=$(vm_stat | head -1 | awk '{print $8}' | tr -d '.'); "
        "TOT=$(sysctl -n hw.memsize); "
        "WL=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0); "
        "echo 'SECTION:mem'; "
        "vm_stat | awk -v ps=\"$PS\" '"
        "  /Pages free/      { gsub(/\\./,\"\",$3); print \"free=\"$3*ps } "
        "  /Pages active/    { gsub(/\\./,\"\",$3); print \"active=\"$3*ps } "
        "  /Pages inactive/  { gsub(/\\./,\"\",$3); print \"inactive=\"$3*ps } "
        "  /Pages speculative/{gsub(/\\./,\"\",$3); print \"speculative=\"$3*ps } "
        "  /Pages wired down/{gsub(/\\./,\"\",$4); print \"wired=\"$4*ps } "
        "  /occupied by compressor/{gsub(/\\./,\"\",$5); print \"compressed=\"$5*ps } "
        "'; "
        "echo \"total=$TOT\"; "
        "echo \"wired_limit=$WL\"; "
        # ── runner.py inventory ────────────────────────────────────────────
        # ps -o gives: pid, %cpu, rss(KB), etime, command. Filter to runner.py
        # rows. Use awk with field-9+ join to handle commands with spaces.
        "echo 'SECTION:runners'; "
        "ps -axo pid,pcpu,rss,etime,command 2>/dev/null | "
        "  awk '/runner\\.py/ && !/awk/ "
        "       {printf \"%s|%s|%s|%s\\n\",$1,$2,$3,$4}' | head -8; "
        # ── RDMA / JACCL ports ─────────────────────────────────────────────
        # ibv_devinfo is only present on macs with the JACCL/IB stack
        # (the 4 ultras). Gracefully report nothing on others.
        "echo 'SECTION:rdma'; "
        "command -v ibv_devinfo >/dev/null 2>&1 && "
        "  ibv_devinfo 2>/dev/null | "
        "  awk '/^hca_id:/{hca=$2} "
        "       /port:[[:space:]]*[0-9]+/{p=$2} "
        "       /state:/{s=$2; print hca\"|\"p\"|\"s}' "
        "  || echo NO_RDMA"
    )
    try:
        t0 = time.perf_counter()
        rc, stdout, stderr = await asyncio.to_thread(_ssh_exec, ssh, cmd, 6)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        out["latency_ms"] = latency_ms
        if rc != 0:
            out["error"] = (stderr or "ssh failed")[:200]
            return out
        out["reachable"] = True
        # Split by SECTION markers — each section has a known parser.
        sections: dict[str, list[str]] = {}
        cur = None
        for raw in stdout.splitlines():
            line = raw.strip()
            if line.startswith("SECTION:"):
                cur = line.split(":", 1)[1]
                sections[cur] = []
            elif cur is not None:
                sections[cur].append(line)

        # ── memory section ─────────────────────────────────────────────
        kv = {}
        for line in sections.get("mem", []):
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    kv[k] = int(v)
                except Exception:
                    pass
        total = kv.get("total")
        wired = kv.get("wired", 0)
        active = kv.get("active", 0)
        compressed = kv.get("compressed", 0)
        used = active + wired + compressed
        out["ram_total_bytes"] = total
        out["ram_used_bytes"] = used
        out["ram_wired_bytes"] = wired
        if total and total > 0:
            out["ram_pct"] = round(used / total * 100, 1)
        out["wired_limit_mb"] = kv.get("wired_limit") or None

        # ── runner inventory ──────────────────────────────────────────
        # Format per line: pid|pcpu|rss_kb|etime
        procs: list[dict] = []
        for line in sections.get("runners", []):
            if not line or "|" not in line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            try:
                procs.append({
                    "pid": int(parts[0]),
                    "cpu_pct": float(parts[1]),
                    "rss_bytes": int(parts[2]) * 1024,  # ps -o rss is KB
                    "etime": parts[3],
                })
            except (ValueError, IndexError):
                continue
        # Sort by RSS desc so the heavy runner shows first
        procs.sort(key=lambda p: -p["rss_bytes"])
        out["runner_count"] = len(procs)
        out["runner_procs"] = procs[:4]  # top 4 — rest is noise

        # ── RDMA ports ────────────────────────────────────────────────
        rdma_lines = sections.get("rdma", [])
        if rdma_lines and rdma_lines != ["NO_RDMA"]:
            ports: list[dict] = []
            for line in rdma_lines:
                if not line or "|" not in line:
                    continue
                parts = line.split("|")
                if len(parts) < 3:
                    continue
                # ibv_devinfo emits state like "PORT_ACTIVE (4)" — keep
                # everything after the colon, which awk gives us as $2,
                # so we only have "PORT_ACTIVE" here. Strip the prefix.
                state = parts[2].replace("PORT_", "")
                ports.append({
                    "name": parts[0],
                    "port": int(parts[1]) if parts[1].isdigit() else parts[1],
                    "state": state,
                })
            out["rdma_ports"] = ports
        # else: leave rdma_ports = None (host has no RDMA stack)
    except subprocess.TimeoutExpired:
        out["error"] = "timeout"
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


@app.get("/admin/nodes/telemetry")
async def admin_nodes_telemetry(fresh: bool = False):
    """Per-host RAM usage + SSH latency. Cached for 4 s."""
    now = time.time()
    if (not fresh) and _telemetry_cache["data"] and (now - _telemetry_cache["ts"] < _TELEMETRY_TTL_S):
        return _telemetry_cache["data"]
    results = await asyncio.gather(*[_probe_telemetry(h) for h in HOSTS_REGISTRY])
    payload = {"hosts": results, "ts": now}
    _telemetry_cache["data"] = payload
    _telemetry_cache["ts"] = now
    return payload


@app.get("/admin/sync/matrix")
async def admin_sync_matrix(fresh: bool = False):
    """Return cross-host model presence matrix.

    Caches for 60 s by default; pass ?fresh=1 to force a re-probe.
    Each cell shows the model's on-disk size (bytes) per host, with missing
    hosts implied by absence from the presence dict.
    """
    now = time.time()
    if (not fresh) and _sync_matrix_cache["data"] and (now - _sync_matrix_cache["ts"] < _SYNC_MATRIX_TTL_S):
        return _sync_matrix_cache["data"]

    # Rebuild the host registry each call so UI-added clusters (Telemak
    # registered via "+ Add Telemak", future http-proxy kinds) show up in
    # the matrix without requiring a container restart.
    hosts_registry = _build_hosts_registry()
    # Probe all hosts in parallel.
    results = await asyncio.gather(*[_probe_host(h) for h in hosts_registry])
    payload = _pivot_matrix(results)
    payload["cached_at"] = now
    _sync_matrix_cache["data"] = payload
    _sync_matrix_cache["ts"] = now
    return payload


# ──────────────────────────────────────────────────────────────────────────────
# Admin: active runs (in-flight inferences) + soft cancel
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/admin/runs")
async def admin_runs(history: int = 0):
    """List currently in-flight inferences. Each entry has:
      id, model, cluster, client, started_at, max_tokens,
      output_tokens (approx), tok_per_s, elapsed_s, status.

    Query params:
      `history=N` (0..500) — append the N most recent finished runs from
      the SQLite history (deduped against in-flight by id). Default 0 to
      keep the polling endpoint cheap.
    """
    active = list(_active_runs.values())
    history = max(0, min(int(history or 0), 500))
    if history:
        active_ids = {r["id"] for r in active}
        past = [r for r in _persist.recent_runs(limit=history * 2)
                if r.get("id") not in active_ids][:history]
        data = active + past
    else:
        data = active
    return {"data": data, "count": len(data),
            "active": len(active), "history": len(data) - len(active)}


def _pool_for_cluster_name(name: Optional[str]):
    """Map a cluster name to its default-alias pool. Returns None for unknown / unloaded."""
    if not name:
        return None
    if name == "nautilus":
        return _pool
    return get_pool(name)


@app.post("/admin/runs/{run_id}/cancel")
async def admin_run_cancel(run_id: str):
    """Hard cancel — soft cancel + runner-side break.

    Soft layer: sets the asyncio Event the streaming handler polls, so the
    HTTP response closes cleanly even if the runner is mid-prefill.

    Hard layer: broadcasts `{"cmd":"cancel","id":<run_id>}` to every rank
    of the pool that's serving this run. The runner's reader thread marks
    the id in `_cancelled_ids`; the gen loop breaks at the next token
    (legacy) or the slot is removed from BatchGenerator on the next tick
    (batched). Compute actually stops — previously cancel was a mute.

    Run_id == OpenAI completion_id == runner req_id (threaded through
    submit's `request_id` since 2026-05-19), so the runners can match.
    """
    ev = _active_run_cancels.get(run_id)
    if not ev:
        raise HTTPException(404, "no such run")
    ev.set()
    r = _active_runs.get(run_id)
    if r:
        r["status"] = "cancelling"
        r["cancelled_at"] = time.time()
    # Hard cancel — best-effort broadcast to the right pool's runners.
    cluster = (r or {}).get("cluster")
    pool = _pool_for_cluster_name(cluster)
    sent = 0
    if pool is not None:
        try:
            sent = await pool.cancel(run_id)
        except Exception as e:
            sys.stderr.write(f"[runs] hard-cancel broadcast failed for {run_id}: {e}\n")
    return {"ok": True, "run_id": run_id, "hard_cancel_sent": sent}


@app.post("/admin/runs/cancel-all")
async def admin_runs_cancel_all(cluster: Optional[str] = None):
    """Hard cancel in-flight runs. With `?cluster=<id>`,
    only runs whose `cluster` field matches are cancelled. Without, all
    runs are cancelled. Each cancel triggers both the soft Event and the
    runner-side broadcast (see admin_run_cancel)."""
    cancelled = 0
    hard_sent_total = 0
    now = time.time()
    # Build list of (rid, pool) first to broadcast outside the iteration.
    targets: list[tuple[str, Optional[Any]]] = []
    for rid, ev in list(_active_run_cancels.items()):
        r = _active_runs.get(rid)
        if cluster and (not r or r.get("cluster") != cluster):
            continue
        ev.set()
        cancelled += 1
        if r:
            r["status"] = "cancelling"
            r["cancelled_at"] = now
        targets.append((rid, _pool_for_cluster_name((r or {}).get("cluster"))))
    for rid, pool in targets:
        if pool is None:
            continue
        try:
            hard_sent_total += await pool.cancel(rid)
        except Exception as e:
            sys.stderr.write(f"[runs] hard-cancel-all broadcast failed for {rid}: {e}\n")
    return {"ok": True, "cancelled": cancelled, "cluster": cluster,
            "hard_cancel_sent_total": hard_sent_total}


# ──────────────────────────────────────────────────────────────────────────────
# Admin: model rsync jobs (push a model dir from one host to others)
# ──────────────────────────────────────────────────────────────────────────────
_sync_jobs: dict[str, dict] = {}
_sync_procs: dict[str, list] = {}  # job_id -> list of asyncio.subprocess.Process


async def _measure_dir_kb(ssh_target: str, path: str, timeout: float = 10.0) -> Optional[int]:
    """Return size in KB of `path` on `ssh_target`, or None on error.

    Uses `du -sk` — fast, no network bandwidth, gives a stable byte-level read."""
    cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh_target,
           f"du -sk {shlex.quote(path)} 2>/dev/null | awk '{{print $1}}'"]
    try:
        p = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(p.communicate(), timeout=timeout)
        s = (out or b"").decode("utf-8", "ignore").strip()
        return int(s) if s.isdigit() else None
    except Exception:
        return None


async def _poll_sync_progress(job_id: str, model: str, src: dict, targets: list[dict]) -> None:
    """Background poller: every ~4s, du -sk on each running target's model dir
    and update the corresponding slot's bytes_transferred. Stops when the job
    is no longer running."""
    target_by_id = {d["id"]: d for d in targets}
    while True:
        job = _sync_jobs.get(job_id)
        if not job:
            return
        if job.get("status") != "running":
            return
        for slot in job.get("per_target", []):
            if slot.get("status") != "running":
                continue
            dst = target_by_id.get(slot["target"])
            if not dst:
                continue
            dst_path = f"{dst['models_dir'].rstrip('/')}/{model}"
            kb = await _measure_dir_kb(dst["ssh"], dst_path)
            if kb is not None:
                slot["bytes_transferred"] = kb * 1024
        await asyncio.sleep(4)


class RsyncJobRequest(BaseModel):
    model: str            # subdir name under models_dir (e.g. "inferencerlabs--Hy3-preview-MLX-9bit")
    source: str           # configured host id
    targets: list[str]    # target host ids
    delete: bool = False  # rsync --delete on target (default False = safe)


def _resolve_host(host_id: str) -> Optional[dict]:
    for h in HOSTS_REGISTRY:
        if h["id"] == host_id:
            return h
    return None


async def _run_one_rsync(job_id: str, model: str, src: dict, dst: dict, slot: dict) -> dict:
    """Run rsync from src to dst host for the given model directory.

    `slot` is the per_target dict for this destination — mutated in place so
    /admin/sync/jobs reflects live status (running → done/failed) without
    waiting for the whole job to finish.
    """
    src_path = f"{src['models_dir'].rstrip('/')}/{model}/"
    dst_path = f"{dst['models_dir'].rstrip('/')}/{model}/"
    # Model names can now contain a `/` (org/model layout). rsync creates the
    # last component but not intermediate parents — make sure the org dir
    # exists on the destination before transferring.
    dst_parent = os.path.dirname(dst_path.rstrip("/")) or dst["models_dir"]
    # rsync only creates the LEAF dir, not intermediate parents (the `org/` part
    # of an `org/model` path). Create the parent ON THE DESTINATION first. The
    # previous code ran `mkdir -p` inside the SOURCE-side command, so it created
    # the org dir on the source and the receiver still failed with "mkdir … No
    # such file or directory" on any node that had never received a model under
    # that org (e.g. a node new to that org, 2026-06-18). The orchestrator
    # can ssh every node directly. NB: the ultra nodes ship openrsync (Apple's),
    # which lacks --mkpath, so an explicit mkdir on the target is the portable fix.
    inner = f"rsync -a {shlex.quote(src_path)} {dst['ssh']}:{shlex.quote(dst_path)}"
    cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", src["ssh"], inner]
    slot["status"] = "running"
    slot["started_at"] = time.time()
    slot["finished_at"] = None
    slot["error"] = None
    try:
        mkproc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
            dst["ssh"], f"mkdir -p {shlex.quote(dst_parent)}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _sync_procs.setdefault(job_id, []).append(mkproc)
        mko, mke = await mkproc.communicate()
        if mkproc.returncode != 0:
            slot["status"] = "failed"
            slot["error"] = (
                "mkdir on target failed: "
                + (mke.decode("utf-8", "ignore") or mko.decode("utf-8", "ignore"))
            )[:400]
            slot["finished_at"] = time.time()
            return slot
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _sync_procs.setdefault(job_id, []).append(proc)
        stdout, stderr = await proc.communicate()
        slot["finished_at"] = time.time()
        if proc.returncode == 0:
            slot["status"] = "done"
        else:
            slot["status"] = "failed"
            slot["error"] = (stderr.decode("utf-8", "ignore") or stdout.decode("utf-8", "ignore"))[:400]
    except Exception as e:
        slot["status"] = "failed"
        slot["error"] = str(e)[:400]
        slot["finished_at"] = time.time()
    return slot


async def _drive_sync_job(job_id: str, model: str, source_id: str, target_ids: list[str]) -> None:
    src = _resolve_host(source_id)
    if not src:
        _sync_jobs[job_id]["status"] = "failed"
        _sync_jobs[job_id]["error"] = f"unknown source {source_id}"
        _sync_jobs[job_id]["finished_at"] = time.time()
        _persist.persist_job(job_id, _sync_jobs[job_id])
        return
    targets = []
    for tid in target_ids:
        dst = _resolve_host(tid)
        if dst and dst["id"] != src["id"]:
            targets.append(dst)
    if not targets:
        _sync_jobs[job_id]["status"] = "failed"
        _sync_jobs[job_id]["error"] = "no valid targets"
        _sync_jobs[job_id]["finished_at"] = time.time()
        _persist.persist_job(job_id, _sync_jobs[job_id])
        return

    job = _sync_jobs.get(job_id)
    if not job:
        return

    # Probe the source for total bytes so the frontend can show a progress bar.
    src_path = f"{src['models_dir'].rstrip('/')}/{model}"
    total_kb = await _measure_dir_kb(src["ssh"], src_path, timeout=20.0)
    if total_kb is not None:
        job["bytes_total"] = total_kb * 1024

    # Map each target to its per_target slot (mutated live by _run_one_rsync).
    per_target = job.get("per_target") or []
    slot_by_id = {p["target"]: p for p in per_target}

    # Start a background poller that updates each slot's bytes_transferred.
    poller = asyncio.create_task(_poll_sync_progress(job_id, model, src, targets))

    # Run all target rsyncs in parallel; each one mutates its slot directly.
    results = await asyncio.gather(*[
        _run_one_rsync(job_id, model, src, dst, slot_by_id[dst["id"]]) for dst in targets
    ])
    # Stop the poller (idempotent — it will exit on next tick when status != running).
    try:
        poller.cancel()
    except Exception:
        pass

    # Final size readings so the bar shows 100% on completion.
    for dst in targets:
        slot = slot_by_id.get(dst["id"])
        if slot and slot.get("status") == "done":
            kb = await _measure_dir_kb(dst["ssh"], f"{dst['models_dir'].rstrip('/')}/{model}", timeout=10.0)
            if kb is not None:
                slot["bytes_transferred"] = kb * 1024

    job["finished_at"] = time.time()
    failed = [r for r in results if r["status"] != "done"]
    if not failed:
        job["status"] = "done"
    elif len(failed) == len(results):
        job["status"] = "failed"
    else:
        job["status"] = "partial"
    _persist.persist_job(job_id, job)
    # Invalidate matrix cache so next /admin/sync/matrix sees the new files.
    _sync_matrix_cache["data"] = None
    _sync_matrix_cache["ts"] = 0.0


@app.post("/admin/sync/rsync")
async def admin_sync_rsync(req: RsyncJobRequest):
    """Push <model> from <source> to one or more <targets> via rsync over SSH.
    Returns a job id; poll /admin/sync/jobs for status.

    Model names follow the matrix layout: either flat (`repo`) or two-
    level (`org/repo`) — the probe walks both depths. Reject paths that
    could escape the models_dir (leading `/`, `..` segments, null/back-
    slash) but allow a single intermediate `/` for the org/repo case.
    """
    name = (req.model or "").strip()
    if not name:
        raise HTTPException(400, "model required")
    if name.startswith("/") or name.startswith("~"):
        raise HTTPException(400, "model must be a relative path under models_dir")
    if "\\" in name or "\x00" in name:
        raise HTTPException(400, "invalid characters in model name")
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise HTTPException(400, "model path must not contain empty / '.' / '..' segments")
    if len(parts) > 2:
        # The matrix probe only walks 2 levels (org/repo). Refusing deeper
        # paths catches accidental over-nesting and keeps mkdir -p semantics
        # predictable.
        raise HTTPException(400, "model path supports at most one '/' (org/repo)")
    src = _resolve_host(req.source)
    if not src:
        raise HTTPException(400, f"unknown source host: {req.source}")
    valid_targets = [t for t in req.targets if _resolve_host(t) and t != req.source]
    if not valid_targets:
        raise HTTPException(400, "no valid targets (must differ from source)")

    job_id = uuid.uuid4().hex[:8]
    _sync_jobs[job_id] = {
        "id": job_id,
        "model": req.model,
        "source": req.source,
        "targets": valid_targets,
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "per_target": [
            {"target": t, "status": "queued", "started_at": None, "finished_at": None,
             "bytes_transferred": 0, "error": None}
            for t in valid_targets
        ],
        "bytes_total": None,
        "error": None,
        # NOTE: this dict is mutated in-place by _drive_sync_job + helpers.
        # We persist at terminal transitions (done/failed/partial) via
        # _persist.persist_job() at the end of _drive_sync_job.
    }
    _persist.persist_job(job_id, _sync_jobs[job_id])
    asyncio.create_task(_drive_sync_job(job_id, req.model, req.source, valid_targets))
    return {"id": job_id}


@app.get("/admin/sync/jobs")
async def admin_sync_jobs():
    """List all sync jobs (active in-memory + recent history from SQLite).
    Deduped by id; capped at last 50 by started_at descending."""
    mem = list(_sync_jobs.values())
    mem_ids = {j["id"] for j in mem}
    past = [j for j in _persist.recent_jobs(limit=100) if j.get("id") not in mem_ids]
    jobs = sorted(mem + past, key=lambda j: -(j.get("started_at") or 0))[:50]
    return {"data": jobs, "count": len(jobs)}


# ──────────────────────────────────────────────────────────────────────────────
# Admin: node reboot (per-host + cluster-wide)
# ──────────────────────────────────────────────────────────────────────────────
# Reboot uses `sudo -n /sbin/shutdown -r now` over SSH. For this to work
# non-interactively, each node needs a sudoers line:
#   admin ALL=(ALL) NOPASSWD: /sbin/shutdown
# (configure once via `sudo visudo` per node).
#
# We background the command so SSH disconnect during reboot doesn't hang
# the orchestrator. Returns rc=0 on accepted command, else stderr in payload.

def _cluster_host_ids(cluster_id: str) -> list[str]:
    cd = get_cluster_def(cluster_id)
    return [n.get("host") for n in (cd.get("nodes") or []) if n.get("host")]


def _vlm_occupied_hosts() -> dict[str, str]:
    """Map host_id -> vlm_cluster_id for every node currently hosting an
    engine-managed VLM. A VLM cluster def is only persisted AFTER its
    mlx_vlm.server became ready (see /admin/vlm/load step d), so an existing
    (non-tombstoned) VLM_MANAGED_KEY cluster means that host is occupied.
    Used to hard-block text pools from claiming a VLM-occupied node."""
    occupied: dict[str, str] = {}
    for cid in _vlm_managed_cluster_ids():
        cd = get_cluster_def(cid)
        for n in cd.get("nodes") or []:
            h = n.get("host")
            if h:
                occupied[h] = cid
    return occupied


async def _reboot_one(host: dict) -> dict:
    out = {"host": host["id"], "ssh": host["ssh"], "rc": None, "error": None, "method": None}
    # Try sudo -n shutdown first (works with NOPASSWD config), fall back to
    # osascript (works for the logged-in admin session on macOS).
    cmd = (
        "(sudo -n /sbin/shutdown -r now </dev/null >/dev/null 2>&1 && "
        "echo METHOD=sudo) || "
        "(osascript -e 'tell application \"System Events\" to restart' </dev/null >/dev/null 2>&1 && "
        "echo METHOD=osascript) || "
        "echo METHOD=failed"
    )
    try:
        rc, stdout, stderr = await asyncio.to_thread(_ssh_exec, host["ssh"], cmd, 6)
        out["rc"] = rc
        if "METHOD=sudo" in stdout:
            out["method"] = "sudo"
        elif "METHOD=osascript" in stdout:
            out["method"] = "osascript"
        else:
            out["method"] = "failed"
            out["error"] = (stderr or "no reboot method worked — configure NOPASSWD sudo for /sbin/shutdown")[:300]
    except subprocess.TimeoutExpired:
        # Timeout is often a SIGNAL that the host is actually rebooting (SSH dies).
        out["rc"] = 0
        out["method"] = "timeout-likely-rebooting"
    except Exception as e:
        out["error"] = str(e)[:300]
    return out


@app.post("/admin/nodes/{host_id}/reboot")
async def admin_node_reboot(host_id: str):
    """Reboot a single host. Returns the method that worked (sudo / osascript / failed)."""
    host = _resolve_host(host_id)
    if not host:
        raise HTTPException(404, f"unknown host: {host_id}")
    res = await _reboot_one(host)
    return res


class ClusterNodeIn(BaseModel):
    host: str
    ssh: Optional[str] = None
    master: bool = False
    port: Optional[int] = None


class ClusterConfigUpdate(BaseModel):
    # Legacy / settings fields
    max_nodes: Optional[int] = None
    models_dir: Optional[str] = None
    # Full editable definition
    name: Optional[str] = None
    kind: Optional[str] = None        # "mlx-distributed" | "telemak"
    backend: Optional[str] = None     # "jaccl" | "ring" | "http-proxy"
    nodes: Optional[list[ClusterNodeIn]] = None
    # Upstream URL — required for kind=telemak (http-proxy passthrough to a
    # Swift single-node runtime). Empty / None for mlx-distributed.
    upstream: Optional[str] = None
    # Optional cloud fallback (cloud alias to redirect to when the pool is
    # unreachable). Empty string clears it.
    fallback: Optional[str] = None
    # Soft-disable a cluster without removing its definition. Disabled clusters
    # refuse load via the admin API and are hidden from the dashboard.
    enabled: Optional[bool] = None


def _pool_for_cluster(cluster_id: str):
    """Return the default-alias pool for this cluster, if loaded. Generic
    over any cluster_id in topology.yaml."""
    return get_pool(cluster_id)


async def _purge_dead_pools(cluster_id: str) -> list[str]:
    """Drop pools whose ranks have all died.

    A runner can die after the load endpoint returned `loaded:true`:
      - OOM during the first forward pass (Metal OOM, common on under-RAMed nodes)
      - segfault / panic
      - node reboot (Sophie pulls the plug to fix RDMA cables)
      - any silent JACCL crash that doesn't take the orchestrator down

    The pool registry keeps claiming those node indices, so the next
    `POST /load` on overlapping indices fails with HTTP 409
    "already in use by another loaded pool", forcing the operator to
    manually `unload {alias}` + `reset` before retrying. Friction with
    no value — the runner is dead, we know it.

    This helper sweeps `list_pools(cluster_id)` and drops every pool
    with `alive_count() == 0`. Returns the list of purged aliases so
    the caller can log them.

    Best-effort: pool.stop() failures don't block the purge — if the
    SSH connection is gone the runner is already dead.
    """
    purged: list[str] = []
    for alias, pool in list(list_pools(cluster_id)):
        if pool.alive_count() > 0:
            continue
        try:
            await pool.stop()
        except Exception as e:
            print(
                f"[purge] {cluster_id}[{alias}] stop failed during dead-pool "
                f"purge: {e}", flush=True
            )
        del_pool(cluster_id, alias)
        purged.append(alias)
        print(
            f"[purge] {cluster_id}[{alias}]: pool was 'loaded' in registry but "
            f"every rank had exited — auto-unloaded to free node indices "
            f"{sorted(pool.node_indices) if pool.node_indices else '[]'}",
            flush=True,
        )
    if purged:
        try:
            save_cluster_state_v2(cluster_id)
        except Exception as e:
            print(f"[purge] {cluster_id} persist after purge failed: {e}", flush=True)
    return purged


async def _auto_unload_cluster(cluster_id: str, reason: str) -> None:
    """Unload every pool on a cluster in response to admin settings changes
    that invalidate the current load (topology change, max_nodes reduction,
    TTL idle expiry).

    No-op if no pool is loaded — caller can fire-and-forget even when state
    was already clean.
    """
    print(f"[admin] auto-unload {cluster_id}: {reason}", flush=True)
    async with get_admin_lock(cluster_id):
        for alias, pool in list(list_pools(cluster_id)):
            try:
                await pool.stop()
            except Exception as e:
                print(f"[admin] {cluster_id}[{alias}] auto-unload stop failed: {e}", flush=True)
            del_pool(cluster_id, alias)
        try:
            sf = state_file_for(cluster_id)
            if sf.exists():
                sf.unlink()
        except Exception:
            pass


@app.get("/admin/inventory")
async def admin_inventory():
    """Return the static inventory of known hosts the user can compose into a cluster."""
    return {"hosts": KNOWN_HOSTS}


@app.get("/admin/clusters")
async def admin_clusters_list():
    """Compact list of every cluster OdyssAI-X publishes — id, display name,
    backend, master SSH/host, total node count, enabled flag. Used by
    external bench/cockpit UIs (odyssai-services) so they don't have to
    hardcode the cluster IDs.

    Includes BOTH topology.yaml-seeded clusters AND dashboard-added entries
    (kind=telemak, etc.). Tombstoned clusters (DELETE'd via UI) are hidden.
    """
    out = []
    for cid in active_cluster_ids():
        cd = get_cluster_def(cid)
        nodes = cd.get("nodes") or []
        master = next((n for n in nodes if n.get("master")), nodes[0] if nodes else {})
        cluster_max = len(nodes) or 1
        saved_max = get_cluster_max_nodes(cid, default=cluster_max)
        # Engine-managed VLM marker — `kind` stays as-is (telemak) so the
        # http-proxy renderers keep working, but the dashboard reads `is_vlm`
        # (+ supports_vision) to draw a distinct "VLM" badge and attribute the
        # VL model to its host.
        is_vlm = bool(cd.get(VLM_MANAGED_KEY) or cd.get("supports_vision"))
        out.append({
            "id": cid,
            "name": cd.get("name", cid),
            "kind": cd.get("kind"),
            "is_vlm": is_vlm,
            "supports_vision": bool(cd.get("supports_vision")),
            "backend": cd.get("backend"),
            "upstream": cd.get("upstream") or None,
            "enabled": _cluster_enabled(cid),
            "master_host": master.get("host"),
            "master_ssh": master.get("ssh"),
            "node_count": len(nodes),
            "max_nodes": min(saved_max, cluster_max),
        })
    return {"data": out}


@app.get("/admin/clusters/{cluster_id}")
async def admin_cluster_get(cluster_id: str):
    if not cluster_exists(cluster_id):
        raise HTTPException(404, f"unknown cluster {cluster_id}")
    cd = get_cluster_def(cluster_id)
    cluster_max = len(cd.get("nodes", [])) or 1
    saved_max = get_cluster_max_nodes(cluster_id, default=cluster_max)
    effective_max = min(saved_max, cluster_max)
    # Compute RAM budget per legal node count so the dashboard can show a
    # capacity bar AND know which nodes_count values are too small for a
    # given model. Cheap — uses cached telemetry + static hardware map.
    capacity_by_nodes: dict[int, dict] = {}
    for n in range(1, effective_max + 1):
        try:
            total, per_node = _cluster_total_ram_bytes(cluster_id, n)
            # Per-rank ceiling honors actual wired_limit_mb when telemetry has
            # it, else falls back to 0.75×RAM. Mirrors _validate_load_fits().
            node_budgets = []
            for nd in per_node:
                if nd.get("wired_limit_bytes"):
                    node_budgets.append(int(nd["wired_limit_bytes"]))
                elif nd.get("ram_bytes"):
                    node_budgets.append(int(nd["ram_bytes"] * 0.75))
            min_budget = min(node_budgets) if node_budgets else 0
            min_node = min((nd["ram_bytes"] for nd in per_node if nd["ram_bytes"]), default=0)
            overall_max = int(total / _model_load_overhead_factor())
            per_rank_max = int(min_budget / 1.10) if min_budget else 0
            max_loadable = min(overall_max, per_rank_max * n) if per_rank_max else overall_max
            # Pipeline mode splits capacity-aware on heterogeneous clusters —
            # its ceiling is higher than the even-split bound above.
            hetero = _hetero_pipeline_ceiling(per_node) if n > 1 else 0
            max_loadable_pipeline = min(overall_max, hetero) if hetero else max_loadable
            capacity_by_nodes[str(n)] = {
                "total_ram_bytes": total,
                "total_ram_gb": round(total / 1024**3, 1),
                "min_node_ram_bytes": min_node,
                "min_node_ram_gb": round(min_node / 1024**3, 1),
                "min_node_budget_gb": round(min_budget / 1024**3, 1),
                "max_loadable_bytes": max_loadable,
                "max_loadable_gb": round(max_loadable / 1024**3, 1),
                "max_loadable_pipeline_bytes": max_loadable_pipeline,
                "max_loadable_pipeline_gb": round(max_loadable_pipeline / 1024**3, 1),
                "per_rank_max_gb": round(per_rank_max / 1024**3, 1),
                "per_node": per_node,
            }
        except Exception:
            capacity_by_nodes[str(n)] = {"total_ram_bytes": 0}
    return {
        "id": cluster_id,
        "name": cd.get("name", cluster_id),
        "kind": cd.get("kind"),
        "backend": cd.get("backend"),
        "nodes": cd.get("nodes", []),
        "max_nodes": effective_max,
        "models_dir": models_dir_for(cluster_id),
        # upstream is meaningful for kind=telemak (http-proxy passthrough);
        # exposing it here lets the dashboard editor pre-fill the form
        # without an extra status fetch.
        "upstream": cd.get("upstream"),
        "fallback": cd.get("fallback") or None,
        "enabled": _cluster_enabled(cluster_id),
        "capacity_by_nodes": capacity_by_nodes,
        "model_overhead_factor": _model_load_overhead_factor(),
    }


@app.put("/admin/clusters/{cluster_id}")
async def admin_cluster_update(cluster_id: str, req: ClusterConfigUpdate):
    """Update editable cluster settings.

    Accepts partial updates: any subset of {name, kind, backend, nodes, max_nodes,
    models_dir}. If `nodes` is provided it must satisfy validate_cluster_def
    (exactly 1 master, known hosts, RDMA wiring sufficient for backend=jaccl
    with >1 node, etc).

    Refuses topology changes while a pool is loaded — user must unload first.

    Upsert: if cluster_id doesn't exist (neither in topology.yaml nor in
    cluster-config.json), CREATE it. Used by the "+ Add Telemak" flow in
    the dashboard.
    """
    creating = not cluster_exists(cluster_id)
    if creating:
        # On create, validate the cluster_id slug shape (URL-safe, no weird chars).
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", cluster_id):
            raise HTTPException(400, f"cluster_id {cluster_id!r} must match [a-z0-9][a-z0-9-]{{0,40}} (URL-safe slug)")
        # Lift any tombstone if one existed for this id.
        with _cluster_config_txn() as _cfg:
            _cfg[cluster_id] = {}

    pool = _pool_for_cluster(cluster_id)
    current = get_cluster_def(cluster_id) if not creating else {}

    # Build the candidate (full) def by merging requested fields onto current.
    candidate = json.loads(json.dumps(current))
    topo_changed = False
    settings_changed = False

    if req.name is not None:
        candidate["name"] = req.name
        settings_changed = True
    if req.kind is not None:
        candidate["kind"] = req.kind
        topo_changed = True
    if req.backend is not None:
        candidate["backend"] = req.backend
        topo_changed = True
    if req.nodes is not None:
        candidate["nodes"] = [n.model_dump() for n in req.nodes]
        topo_changed = True
    if req.upstream is not None:
        # http-proxy passthrough target — only meaningful for kind=telemak.
        candidate["upstream"] = req.upstream.strip()
        topo_changed = True

    # On create, kind/backend/nodes are required. Stamp them on the candidate.
    if creating:
        if not candidate.get("kind"):
            raise HTTPException(400, "create requires `kind` (mlx-distributed or telemak)")
        if not candidate.get("backend"):
            raise HTTPException(400, "create requires `backend`")
        if not candidate.get("nodes"):
            raise HTTPException(400, "create requires `nodes`")
        topo_changed = True

    # Validate when topology fields changed
    if topo_changed:
        err = validate_cluster_def(cluster_id, candidate)
        if err:
            raise HTTPException(400, err)
        # Auto-unload when a topology-affecting change requires it. Previously
        # we 409'd and made the user click "Unload" then re-save — friction
        # that led to stale config silently kept in memory. Pattern lifted
        # auto-unload-on-settings-change pattern.
        if pool is not None and getattr(pool, "loaded", False):
            await _auto_unload_cluster(cluster_id, reason="topology change")

    # max_nodes (cap on how many nodes can be claimed at load time)
    cluster_max = len(candidate.get("nodes", [])) or 1
    if req.max_nodes is not None:
        new_max = int(req.max_nodes)
        if new_max < 1 or new_max > cluster_max:
            raise HTTPException(400, f"max_nodes must be 1..{cluster_max}")
        # If reducing max below the current loaded nodes_count, unload first.
        if pool is not None and getattr(pool, "loaded", False) and pool.nodes_count > new_max:
            await _auto_unload_cluster(cluster_id, reason=f"max_nodes reduced below loaded ({pool.nodes_count}→{new_max})")
        set_cluster_max_nodes(cluster_id, new_max)

    # Persist topology/identity fields if present
    persist: dict = {}
    if req.name is not None: persist["name"] = req.name
    if req.kind is not None: persist["kind"] = req.kind
    if req.backend is not None: persist["backend"] = req.backend
    if req.nodes is not None: persist["nodes"] = candidate["nodes"]
    if req.enabled is not None:
        # Refuse disable if currently loaded — caller must unload first to avoid
        # leaving a phantom pool unreachable from the dashboard.
        if req.enabled is False and pool is not None and getattr(pool, "loaded", False):
            raise HTTPException(409, f"{cluster_id} is loaded — unload first to disable")
        persist["enabled"] = bool(req.enabled)
    if req.upstream is not None: persist["upstream"] = candidate["upstream"]
    if req.fallback is not None:
        # Empty string clears the fallback; non-empty validates against published cloud aliases.
        if req.fallback == "":
            persist["fallback"] = ""
        else:
            if not find_cloud_alias(req.fallback):
                raise HTTPException(400, f"fallback {req.fallback!r} is not a published cloud alias")
            persist["fallback"] = req.fallback
    if persist:
        save_cluster_def(cluster_id, persist)

    if req.models_dir is not None:
        set_models_dir(cluster_id, req.models_dir)

    return await admin_cluster_get(cluster_id)


@app.delete("/admin/clusters/{cluster_id}")
async def admin_cluster_delete(cluster_id: str):
    """Remove a cluster via UI. Auto-unloads first if loaded.

    Tombstone semantics: the cluster's overlay entry in cluster-config.json is
    overwritten with `{_removed: true}`. At next list/lookup, active_cluster_ids()
    filters it out. This works for both dashboard-added clusters AND for
    topology.yaml-declared clusters (the tombstone shadows the seed). To
    un-remove a topology.yaml cluster, the operator edits cluster-config.json
    by hand or re-adds it via the UI.
    """
    if not cluster_exists(cluster_id):
        raise HTTPException(404, f"unknown cluster {cluster_id}")
    pool = _pool_for_cluster(cluster_id)
    if pool is not None and getattr(pool, "loaded", False):
        await _auto_unload_cluster(cluster_id, reason="cluster removed via UI")
    with _cluster_config_txn() as cfg:
        cfg[cluster_id] = {"_removed": True}
    return {"ok": True, "removed": cluster_id}


@app.post("/admin/clusters/{cluster_id}/reboot-all")
async def admin_cluster_reboot_all(cluster_id: str):
    """Reboot every host in this cluster in parallel."""
    member_ids = _cluster_host_ids(cluster_id)
    if not member_ids:
        raise HTTPException(404, f"unknown cluster: {cluster_id}")
    hosts = [_resolve_host(hid) for hid in member_ids]
    hosts = [h for h in hosts if h]
    if not hosts:
        raise HTTPException(404, "no resolved hosts")
    results = await asyncio.gather(*[_reboot_one(h) for h in hosts])
    return {"ok": True, "cluster": cluster_id, "results": results}


class BulkDeleteRequest(BaseModel):
    items: list[dict]  # [{ "host": "host-id", "model": "name" }, ...]


def _safe_model_name(name: str) -> bool:
    """Validate that model name is one path component, optionally prefixed by
    a single org folder (`org/model`). Rejects path traversal, leading dots,
    leading slash, and deeper nesting.

    Examples:
      ok:   "Qwen2.5-Coder-1.5B-bf16"
      ok:   "inferencerlabs/Hy3-preview-MLX-9bit"
      bad:  "../etc"      → traversal
      bad:  "/abs/path"   → absolute
      bad:  "a/b/c"       → too deep
      bad:  ".hidden"     → hidden
    """
    if not name or ".." in name or name.startswith("/") or name.startswith("."):
        return False
    parts = name.split("/")
    if len(parts) > 2:
        return False
    for p in parts:
        if not p or p.startswith(".") or p in ("", "."):
            return False
    return True


async def _delete_one(host: dict, model: str) -> dict:
    """rm -rf the model dir on a single host."""
    out = {"host": host["id"], "model": model, "ok": False, "error": None}
    if not _safe_model_name(model):
        out["error"] = "invalid model name"
        return out
    target = f"{host['models_dir'].rstrip('/')}/{model}"
    # Defensive: ensure target stays inside models_dir.
    cmd = (
        f"D={shlex.quote(target)}; "
        f"case \"$D\" in {shlex.quote(host['models_dir'].rstrip('/'))}/*) "
        f"rm -rf \"$D\" && echo OK;; *) echo BAD_PATH;; esac"
    )
    try:
        rc, stdout, stderr = await asyncio.to_thread(_ssh_exec, host["ssh"], cmd, 30)
        if rc == 0 and "OK" in stdout:
            out["ok"] = True
        elif "BAD_PATH" in stdout:
            out["error"] = "refused: target outside models_dir"
        else:
            out["error"] = (stderr or stdout or "rm failed")[:300]
    except Exception as e:
        out["error"] = str(e)[:300]
    return out


@app.delete("/admin/sync/host/{host_id}/model/{model:path}")
async def admin_sync_delete_one(host_id: str, model: str):
    """Delete a model dir on a single host."""
    host = _resolve_host(host_id)
    if not host:
        raise HTTPException(404, f"unknown host: {host_id}")
    if not _safe_model_name(model):
        raise HTTPException(400, "invalid model name")
    res = await _delete_one(host, model)
    # Invalidate matrix cache so next refresh reflects the deletion.
    _sync_matrix_cache["data"] = None
    _sync_matrix_cache["ts"] = 0.0
    return res


@app.post("/admin/sync/bulk-delete")
async def admin_sync_bulk_delete(req: BulkDeleteRequest):
    """Delete multiple (host, model) pairs in parallel."""
    tasks = []
    for it in req.items:
        h = _resolve_host(it.get("host", ""))
        m = it.get("model", "")
        if not h or not _safe_model_name(m):
            continue
        tasks.append(_delete_one(h, m))
    if not tasks:
        return {"ok": False, "results": [], "error": "no valid items"}
    results = await asyncio.gather(*tasks)
    _sync_matrix_cache["data"] = None
    _sync_matrix_cache["ts"] = 0.0
    return {"ok": True, "results": results, "deleted": sum(1 for r in results if r["ok"])}


_SYNC_TERMINAL = {"done", "failed", "partial", "cancelled", "interrupted", "error"}


@app.post("/admin/sync/jobs/clear")
async def admin_sync_jobs_clear():
    """Purge every terminal sync job (done/failed/partial/cancelled/…) from both
    the in-memory list and the SQLite history. Running/queued jobs are kept.
    This is the dashboard 'Clear' button (#57)."""
    removed = [jid for jid, j in _sync_jobs.items() if j.get("status") in _SYNC_TERMINAL]
    for jid in removed:
        _sync_jobs.pop(jid, None)
    past = _persist.clear_terminal_jobs()
    return {"ok": True, "removed_active": len(removed), "removed_history": past}


@app.delete("/admin/sync/jobs/{job_id}")
async def admin_sync_job_cancel(job_id: str):
    """A running job is cancelled (rsync terminated). A terminal job is removed
    from the list — the dashboard '(x)' per job (#57)."""
    job = _sync_jobs.get(job_id)
    if job and job["status"] == "running":
        for proc in _sync_procs.get(job_id, []) or []:
            try:
                if proc.returncode is None:
                    proc.terminate()
            except Exception:
                pass
        job["status"] = "cancelled"
        job["finished_at"] = time.time()
        _persist.persist_job(job_id, job)
        return {"ok": True, "cancelled": True}
    # Terminal (or only in history): remove the entry entirely.
    _sync_jobs.pop(job_id, None)
    _persist.delete_job(job_id)
    return {"ok": True, "removed": True}


@app.post("/admin/sync/jobs/{job_id}/retry")
async def admin_sync_job_retry(job_id: str):
    """Re-run a failed/partial/cancelled job with the same model/source/targets.
    Spawns a NEW job id (the old one stays as history) (#57)."""
    job = _sync_jobs.get(job_id) or next(
        (j for j in _persist.recent_jobs(limit=100) if j.get("id") == job_id), None
    )
    if not job:
        raise HTTPException(404, "no such job")
    if job.get("status") == "running":
        raise HTTPException(409, "job is still running")
    model = job.get("model")
    source = job.get("source")
    targets = [t for t in (job.get("targets") or []) if _resolve_host(t) and t != source]
    if not _resolve_host(source) or not targets:
        raise HTTPException(400, "original source/targets no longer resolvable")
    new_id = uuid.uuid4().hex[:8]
    _sync_jobs[new_id] = {
        "id": new_id, "model": model, "source": source, "targets": targets,
        "status": "running", "started_at": time.time(), "finished_at": None,
        "per_target": [
            {"target": t, "status": "queued", "started_at": None, "finished_at": None,
             "bytes_transferred": 0, "error": None}
            for t in targets
        ],
        "bytes_total": None, "error": None, "retry_of": job_id,
    }
    _persist.persist_job(new_id, _sync_jobs[new_id])
    asyncio.create_task(_drive_sync_job(new_id, model, source, targets))
    return {"id": new_id, "retry_of": job_id}


# ──────────────────────────────────────────────────────────────────────────────
# Admin: Default cluster
# ──────────────────────────────────────────────────────────────────────────────
def _pool_per_rank_phases(pool) -> Optional[list[dict]]:
    """Per-rank load phase snapshot. Used by /admin/.../status during a
    load so the dashboard can show "rank 0: barrier · rank 1: sharding ·
    rank 2: dead" instead of a fake 95% bar. None if no pool."""
    if pool is None or not pool.runners:
        return None
    out = []
    for r in sorted(pool.runners, key=lambda x: x.node.get("rank", 0)):
        rc = r.proc.poll()
        out.append({
            "rank": r.node.get("rank"),
            "host": r.node.get("host"),
            "phase": "dead" if rc is not None else getattr(r, "phase", "spawning"),
            "phase_age_s": (round(time.time() - r.phase_at, 1) if r.phase_at else None),
            "exit_code": rc,
            "ready": getattr(r, "ready", None) and r.ready.is_set(),
        })
    return out


@app.get("/admin/clusters/{cluster_id}/status")
async def admin_cluster_status(cluster_id: str):
    if not cluster_exists(cluster_id):
        raise HTTPException(404, f"unknown cluster {cluster_id}")
    # kind=telemak: short-circuit to a single-node "what's loaded" view.
    # The upstream is the source of truth — we query it once and shape the
    # response into OdyssAI-X' status format so the dashboard reuses the
    # mlx-distributed renderers without crashing.
    cd_t = get_cluster_def(cluster_id)
    if cd_t.get("kind") == "telemak":
        return await _telemak_status(cluster_id, cd_t)
    loading = _loading_snapshot(_loading_state_for(cluster_id))
    default_pool = get_pool(cluster_id)
    all_loaded = list_pools(cluster_id)
    # During a load, expose per-rank phase. After load completes the snapshot
    # disappears and clients fall back to the main `loaded` status.
    if loading is not None:
        # Prefer the default pool when present (single-pool back-compat),
        # otherwise pick the first loaded pool so the dashboard renders
        # per-rank phases for multi-pool loads too.
        anchor_pool = default_pool or (all_loaded[0][1] if all_loaded else None)
        ranks_view = _pool_per_rank_phases(anchor_pool)
        if ranks_view:
            loading["ranks"] = ranks_view
    cd = get_cluster_def(cluster_id)
    cluster_max = len(cd.get("nodes", [])) or 1
    effective_max = get_cluster_max_nodes(cluster_id, default=cluster_max)
    effective_max = min(effective_max, cluster_max)
    avail_counts = list(range(1, effective_max + 1))
    # Empty-cluster path : no pool of any alias is loaded. Dashboard renders
    # the load form, nothing else.
    if default_pool is None and not all_loaded:
        return {
            "loaded": False, "cluster": cluster_id,
            "loading": loading,
            "available_node_counts": avail_counts,
            "max_nodes": effective_max,
            "topologies": {
                str(n): [{"rank": h["rank"], "ssh": h["ssh"]} for h in build_topology(cluster_id, n)]
                for n in avail_counts
            },
            "models_dir": models_dir_for(cluster_id),
            "degraded": _cluster_degraded.get(cluster_id),
        }
    # Multi-pool: build per-pool views. Top-level fields (model / nodes /
    # mode / etc.) reflect the DEFAULT alias for back-compat with single-
    # pool clients. The `pools` array contains the full picture.
    def _pool_view(alias: str, pool: RunnerPool) -> dict:
        uptime_p = time.time() - (pool.started_at or time.time())
        recent_tps_p = [m["tps"] for m in list(_metrics)[:10]
                        if m["tps"] > 0 and m.get("model") == pool.model]
        return {
            "alias": alias,
            "model": pool.model,
            "mode": pool.mode,
            "use_ap": pool.use_ap,
            "nodes": pool.nodes_count,
            "node_indices": list(pool.node_indices)
                            if getattr(pool, "node_indices", None)
                            else [_host_to_index(cluster_id, n.get("host"))
                                  for n in pool.nodes if n.get("host")],
            "kv_q8": pool.kv_q8,
            "draft_model": pool.draft_model,
            "num_draft_tokens": pool.num_draft_tokens if pool.draft_model else None,
            "alive": pool.alive_count(),
            "load_s": pool.load_s,
            "uptime_s": uptime_p,
            "recent_avg_tps": round(sum(recent_tps_p) / len(recent_tps_p), 2)
                              if recent_tps_p else None,
            "topology": [{"rank": n["rank"], "ssh": n["ssh"], "host": n.get("host")}
                         for n in pool.nodes],
        }

    pools = [_pool_view(a, p) for a, p in all_loaded]
    # Pick a "primary" pool for the flat back-compat fields :
    #   - default alias when loaded (single-pool clusters keep current shape),
    #   - otherwise the first pool alphabetically — gives multi-pool clusters
    #     a deterministic anchor for old clients that only read flat fields.
    # Without this anchor, dashboards that don't yet parse `pools[]` show
    # nothing at all when an operator uses custom aliases (no `default`).
    primary_pool = default_pool or all_loaded[0][1]
    uptime = time.time() - (primary_pool.started_at or time.time())
    recent_tps = [m["tps"] for m in list(_metrics)[:10]
                  if m["tps"] > 0 and m.get("model") == primary_pool.model]
    topo = build_topology(cluster_id, primary_pool.nodes_count)
    return {
        "loaded": True, "cluster": cluster_id,
        "loading": loading,
        "model": primary_pool.model, "mode": primary_pool.mode,
        "use_ap": primary_pool.use_ap, "nodes": primary_pool.nodes_count,
        "kv_q8": primary_pool.kv_q8,
        "draft_model": primary_pool.draft_model,
        "num_draft_tokens": primary_pool.num_draft_tokens if primary_pool.draft_model else None,
        "alive": primary_pool.alive_count(),
        "load_s": primary_pool.load_s, "uptime_s": uptime,
        "recent_avg_tps": round(sum(recent_tps) / len(recent_tps), 2) if recent_tps else None,
        "topology": [{"rank": n["rank"], "ssh": n["ssh"]} for n in topo],
        "pools": pools,
        "aliases": [a for a, _ in all_loaded],
        "nodes_in_use": sorted(nodes_in_use(cluster_id)),
        "available_node_counts": avail_counts,
        "max_nodes": effective_max,
        "models_dir": models_dir_for(cluster_id),
        "degraded": _cluster_degraded.get(cluster_id),
    }


@app.get("/admin/clusters/{cluster_id}/models")
async def admin_cluster_models(cluster_id: str, dir: Optional[str] = None):
    """Discover models on the cluster's rank-0 host, under its models_dir.

    Each entry includes `size_bytes` (du -sk on the model dir) so the dashboard
    can show the user how big each model is BEFORE they attempt to load it.
    Sizes are gathered in one batched SSH call to avoid the per-model fanout.

    For kind=telemak clusters, SSH discovery is replaced by a proxy call to
    the upstream's /admin/models/available endpoint.
    """
    if not cluster_exists(cluster_id):
        raise HTTPException(404, f"unknown cluster {cluster_id}")
    cd = get_cluster_def(cluster_id)
    if cd.get("kind") == "telemak":
        upstream = (cd.get("upstream") or "").rstrip("/")
        if not upstream:
            raise HTTPException(502, f"{cluster_id}: telemak cluster has no upstream URL")
        # Master push: ensure the Telemak server scans the dir this cluster is
        # configured for (auto-heals a freshly-deployed node whose config.json
        # isn't set yet). Idempotent — no-op when already in sync.
        await _telemak_reconcile_models_dir(cluster_id, cd)
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{upstream}/admin/models/available")
            if r.status_code != 200:
                raise HTTPException(502, f"{cluster_id}: upstream /admin/models/available returned {r.status_code}")
            payload = r.json()
        except httpx.RequestError as exc:
            raise HTTPException(502, f"{cluster_id}: upstream unreachable: {exc}")
        loaded = await _telemak_loaded_models(cluster_id, cd)
        loaded_set = set(loaded)
        annotated = [{
            "id": m["id"],
            "kind": "telemak",
            "size_bytes": int(m.get("size_gb", 0) * 1_073_741_824),
            "family": m.get("family"),
            "is_loaded": m["id"] in loaded_set,
        } for m in payload.get("models", []) if m.get("id")]
        return {"data": annotated, "models_dir": None, "upstream": upstream}
    rank0 = rank0_ssh_for_cluster(cluster_id, 1)
    target_dir = dir or models_dir_for(cluster_id)
    if dir:
        set_models_dir(cluster_id, dir)
    models = await asyncio.to_thread(discover_models_on_node, rank0, target_dir)
    sizes = await batch_get_model_sizes(rank0, models)
    pool = get_pool(cluster_id)
    annotated = [{
        "id": m,
        "kind": "path",
        "size_bytes": sizes.get(m, 0),
        "is_loaded": pool is not None and pool.model == m,
    } for m in models]
    return {"data": annotated, "models_dir": target_dir}


@app.post("/admin/clusters/{cluster_id}/models-dir")
async def admin_cluster_models_dir(cluster_id: str, req: ModelsDirRequest):
    if not cluster_exists(cluster_id):
        raise HTTPException(404, f"unknown cluster {cluster_id}")
    set_models_dir(cluster_id, req.dir)
    return {"cluster": cluster_id, "models_dir": req.dir}


class ArgoLoadRequest(BaseModel):
    model: str
    mode: str = "pipeline"  # "pipeline" | "tensor"
    use_ap: bool = True     # most non-trivial models need auto_parallel multi-node
    nodes: int = 3          # 1 | 2 | 3 | 4 — legacy contiguous-from-0 selection
    # None = use cluster-wide default; explicit value overrides per-request.
    kv_q8: Optional[bool] = None     # Q8 quantized KV cache (halves memory for long contexts)
    draft_model: Optional[str] = None  # speculative decoding (nodes=1 only)
    num_draft_tokens: int = 4
    # Power-user override for the preflight size check. Use only when you
    # know the apparent du size is overestimating (e.g. dedup / sparse files).
    force: bool = False
    # Stop-old-before-start-new by default (audit 2026-05-18). Set True to
    # overlap loads — only safe when both models fit simultaneously in RAM.
    force_hot_swap: bool = False
    # Multi-pool (2026-05-19): alias names the pool. Default 'default' = the
    # singleton pool (one-pool clusters keep working identically). Pass a
    # different alias to spawn a SECOND concurrent pool on a disjoint
    # subset of nodes.
    alias: Optional[str] = None
    # Multi-pool: explicit node indices (into the cluster def's node list).
    # Order = rank order; first index = rank 0. When provided, `nodes` is
    # ignored. When absent, falls back to the legacy contiguous-from-0
    # interpretation of `nodes`.
    # Examples:
    #   {alias:"default",     nodes:1}                       → legacy, rank 0 only
    #   {alias:"default",     node_indices:[0]}              → same
    #   {alias:"default-big", node_indices:[1,2,3]}          → 2nd pool on ranks 1..3
    node_indices: Optional[list[int]] = None


@app.get("/admin/clusters/{cluster_id}/load-options")
async def admin_cluster_load_options(cluster_id: str, model: str):
    """Enumerate the VALID (nodes, mode) configurations for `model` on this
    cluster — the data that lets the UI propose configs ("1", "1+2 pipeline",
    "1+2+3 tensor", …) instead of a blind node-count picker.

    For each node-count N (1..max), and each applicable mode, returns whether it
    `fits` (RAM via _validate_load_fits AND mode-validity via _validate_load_mode)
    plus a human reason when it doesn't. Infeasible options are INCLUDED (with
    fits:false + reason) so the UI can grey them out rather than hide them."""
    if not cluster_exists(cluster_id):
        raise HTTPException(404, f"unknown cluster {cluster_id}")
    cd = get_cluster_def(cluster_id)
    if cd.get("kind") == "telemak":
        # Single-node http-proxy: no node selection.
        return {"data": [{"label": "1", "nodes": [0], "mode": "solo",
                          "fits": True, "reason": ""}], "model_meta": {}}
    cluster_max = len(cd.get("nodes", []))
    effective_max = min(get_cluster_max_nodes(cluster_id, default=cluster_max), cluster_max)

    # Read model size + arch once, from node index 0 (rank 0 of the full range).
    topo = build_topology_from_indices(cluster_id, list(range(max(effective_max, 1))))
    rank0_ssh = topo[0]["ssh"]
    base_dir = topo[0].get("models_dir") or models_dir_for(cluster_id)
    model_abspath = _resolve_model_abspath(model, base_dir)
    arch = await get_model_arch_meta(rank0_ssh, model_abspath)
    size_bytes = await get_model_size_bytes(rank0_ssh, model_abspath)
    mt = arch.get("model_type")
    layers = arch.get("num_hidden_layers")
    kv = arch.get("num_key_value_heads")

    options = []
    for n in range(1, effective_max + 1):
        label_nodes = "+".join(str(i + 1) for i in range(n))
        modes = ["solo"] if n == 1 else ["pipeline", "tensor"]
        for mode in modes:
            # Fit is mode-dependent: pipeline gets the capacity-aware split.
            fit_ok, fit_reason, fit_detail = _validate_load_fits(
                size_bytes, cluster_id, n,
                mode="pipeline" if mode in ("solo", "pipeline") else mode)
            # solo reuses the pipeline arch-validity (always ok for n==1).
            mode_ok, mode_reason = _validate_load_mode(
                mt, layers, kv, n, "pipeline" if mode == "solo" else mode)
            fits = bool(fit_ok and mode_ok)
            reason = mode_reason or ("" if fit_ok else fit_reason)
            options.append({
                "label": label_nodes if mode == "solo" else f"{label_nodes} {mode}",
                "nodes": list(range(n)),
                "nodes_count": n,
                "mode": mode,
                "fits": fits,
                "reason": reason,
                "fit_detail": fit_detail,
            })
    return {
        "data": options,
        "model_meta": {
            "model": model,
            "model_type": mt,
            "num_hidden_layers": layers,
            "num_key_value_heads": kv,
            "size_bytes": size_bytes,
            "size_gb": round(size_bytes / 1024**3, 1) if size_bytes else None,
            "tensor_capable": (mt in TENSOR_CAPABLE_MODEL_TYPES) if mt else None,
        },
    }


@app.post("/admin/clusters/{cluster_id}/load")
async def admin_cluster_load(cluster_id: str, req: ArgoLoadRequest):
    if not cluster_exists(cluster_id):
        raise HTTPException(404, f"unknown cluster {cluster_id}")
    if not _cluster_enabled(cluster_id):
        raise HTTPException(409, f"{cluster_id} is disabled — toggle enable in cluster settings to load")
    # kind=telemak: proxy load to the Swift binary's /admin/load endpoint.
    # Telemak is single-node http-proxy — no pipeline/tensor parallel, no node
    # selection. We just POST {model} upstream and return its response.
    cd = get_cluster_def(cluster_id)
    if cd.get("kind") == "telemak":
        await _telemak_reconcile_models_dir(cluster_id, cd)
        return await _telemak_proxy_load(cluster_id, cd, req)
    # Block reload if the cluster is currently flagged degraded — a JACCL
    # queue-pair stuck in TIME_WAIT or a wired-memory leak makes the next
    # load almost certain to crash the same way. Operator runs
    # /admin/clusters/{id}/reset (or sets force=True) to clear it explicitly.
    if _cluster_is_degraded(cluster_id) and not getattr(req, "force", False):
        info = _cluster_degraded.get(cluster_id, {})
        raise HTTPException(
            status_code=409,
            detail={
                "error": "cluster_degraded",
                "message": f"{cluster_id} is in a degraded state — reload refused to avoid "
                           f"compounding the failure. POST /admin/clusters/{cluster_id}/reset "
                           "to run the recovery ladder, or pass force=true to override.",
                "reason": info.get("reason"),
                "since": info.get("at"),
                "details": info.get("details"),
            },
        )
    if req.mode not in ("pipeline", "tensor"):
        raise HTTPException(400, f"invalid mode {req.mode}")
    cd = get_cluster_def(cluster_id)
    cluster_max = len(cd.get("nodes", []))
    effective_max = get_cluster_max_nodes(cluster_id, default=cluster_max)
    effective_max = min(effective_max, cluster_max)
    avail = list(range(1, effective_max + 1))

    # Resolve alias + node_indices.
    alias = (req.alias or DEFAULT_ALIAS).strip().lower()
    if not alias or "/" in alias or " " in alias or len(alias) > 64:
        raise HTTPException(400, f"invalid alias {alias!r}")
    if req.node_indices is not None and len(req.node_indices) > 0:
        node_indices = list(req.node_indices)
        for i in node_indices:
            if not (0 <= i < effective_max):
                raise HTTPException(
                    400,
                    f"node index {i} out of range [0..{effective_max - 1}]",
                )
        if len(set(node_indices)) != len(node_indices):
            raise HTTPException(400, "duplicate node index in node_indices")
        nodes_count = len(node_indices)
    else:
        if req.nodes not in avail:
            raise HTTPException(
                400,
                f"{cluster_id}: unsupported nodes={req.nodes} (available: {avail}, max={effective_max})"
            )
        node_indices = list(range(req.nodes))
        nodes_count = req.nodes

    # Purge any pool whose runners have all died (OOM, panic, node reboot).
    # Without this, the next load on overlapping indices gets HTTP 409
    # "already in use" even though nobody's actually using them. The user
    # has to manually unload + reset to break the deadlock — friction with
    # no value. See #5. Safe to run unconditionally : pools with alive
    # ranks are skipped.
    purged_aliases = await _purge_dead_pools(cluster_id)
    if purged_aliases:
        print(
            f"[load] {cluster_id}: dead pools auto-purged before load "
            f"({', '.join(purged_aliases)})",
            flush=True,
        )

    existing_aliases = {a for a, _ in list_pools(cluster_id)}
    if alias in existing_aliases and not req.force_hot_swap and alias != DEFAULT_ALIAS:
        raise HTTPException(
            409,
            f"{cluster_id} alias {alias!r} is already loaded — unload it first or "
            f"pick a different alias"
        )

    used = set()
    for a, p in list_pools(cluster_id):
        if a == alias:
            continue
        for n in p.nodes:
            host = n.get("host")
            i = _host_to_index(cluster_id, host) if host else None
            if i is not None:
                used.add(i)
    overlap = set(node_indices) & used
    if overlap:
        raise HTTPException(
            409,
            f"{cluster_id}: requested nodes {sorted(overlap)} are already in use by "
            f"another loaded pool. Pick a disjoint subset."
        )

    topo = build_topology_from_indices(cluster_id, node_indices)

    # Hard capacity enforcement (mirror of /admin/vlm/load): refuse to build a
    # text pool that would claim a node currently hosting an engine-managed
    # VLM. Co-residence can OOM (e.g. a text pool + M3-VL 327GB > node RAM).
    # Overridable with force=true (same flag that overrides degraded/mode/RAM
    # checks) or the VLM_ALLOW_CORESIDENCE env.
    vlm_hosts = _vlm_occupied_hosts()
    if vlm_hosts and not (getattr(req, "force", False) or VLM_ALLOW_CORESIDENCE):
        clashes = [
            f"node {n.get('host')} hosts VLM '{vlm_hosts[n.get('host')]}'"
            for n in topo if n.get("host") in vlm_hosts
        ]
        if clashes:
            raise HTTPException(
                409,
                f"{cluster_id}: refusing to load — " + "; ".join(clashes)
                + ". Unload the VLM (POST /admin/vlm/unload) first, pick nodes "
                "that are free, or override with force=true "
                "(or set VLM_ALLOW_CORESIDENCE).",
            )

    rank0_ssh = topo[0]["ssh"]
    # Resolve to an absolute path before sizing/reading config — a relative
    # model id would run du/cat in the SSH home dir and silently return nothing,
    # turning the RAM preflight into a no-op (see _resolve_model_abspath).
    base_dir = topo[0].get("models_dir") or models_dir_for(cluster_id)
    model_abspath = _resolve_model_abspath(req.model, base_dir)

    # Mode/architecture preflight: reject combinations the model can't do
    # (tensor on a pipeline-only model_type, KV-heads not divisible by N,
    # pipeline with more nodes than layers) BEFORE spawning runners that would
    # just die at the barrier. Independent of the RAM check below.
    arch = await get_model_arch_meta(rank0_ssh, model_abspath)

    # Unified load auto-detect. The distributed text runner can't run a vision
    # model — VL models serve single-node via mlx_vlm.server. When the requested
    # model's config.json is a vision model, transparently DISPATCH to the
    # engine-managed VLM launch flow (admin_vlm_load) targeting this pool's
    # rank-0 node, so "load model X on cluster/node N" just works for both.
    #
    # Seam: this creates a SEPARATE engine-managed telemak/http-proxy cluster
    # (id "<cluster_id>-vlm", routing to mlx_vlm.server on rank-0) rather than a
    # pool inside this mlx-distributed cluster — the two runtimes are distinct.
    # The VLM then appears in the dashboard with the "vlm" badge attributed to
    # rank-0's host. Set force=true to bypass detection and attempt a (doomed)
    # text load anyway. Multi-node VL requests collapse to single-node (rank-0).
    if arch.get("is_vision") and not getattr(req, "force", False):
        vlm_id = f"{cluster_id}-vlm"
        vlm_host = topo[0].get("host") or _host_id_from_ssh(rank0_ssh)
        vlm_req = VLMLoadRequest(
            id=re.sub(r"[^a-z0-9-]", "-", vlm_id.lower())[:41].strip("-") or "vlm",
            host=rank0_ssh,
            model=req.model,
            name=f"{cd.get('name', cluster_id)} VLM",
            models_dir=base_dir,
        )
        result = await admin_vlm_load(vlm_req)
        return {
            **result,
            "dispatched": "vlm",
            "note": (
                f"{req.model} is a vision model — the distributed text runner "
                f"can't serve it. Launched an engine-managed VLM "
                f"(mlx_vlm.server) on {vlm_host} instead; it appears as cluster "
                f"'{result.get('id')}' with the VLM badge. Pass force=true to "
                "attempt a text load anyway."
            ),
        }

    mode_ok, mode_reason = _validate_load_mode(
        arch.get("model_type"), arch.get("num_hidden_layers"),
        arch.get("num_key_value_heads"), nodes_count, req.mode,
    )
    if not mode_ok and not getattr(req, "force", False):
        raise HTTPException(400, {
            "error": "invalid_mode_for_model",
            "message": mode_reason,
            "detail": {"mode": req.mode, "nodes": nodes_count, **arch},
            "hint": "Pick a valid (nodes, mode) combo — see GET "
                    f"/admin/clusters/{cluster_id}/load-options?model={req.model}",
        })

    size_bytes = await get_model_size_bytes(rank0_ssh, model_abspath)
    ok, reason, detail = _validate_load_fits(size_bytes, cluster_id, nodes_count,
                                             mode=req.mode)
    if not ok and not getattr(req, "force", False):
        raise HTTPException(400, {
            "error": "model_too_big_for_cluster",
            "message": reason,
            "detail": detail,
            "hint": "Reload with more nodes or pick a smaller model.",
        })
    estimated_s = estimate_load_s(req.model, size_bytes, cluster_id, nodes_count)

    kv_q8 = req.kv_q8 if req.kv_q8 is not None else get_kv_q8_default()

    loading_state = _loading_state_for(cluster_id)
    async with get_admin_lock(cluster_id):
        _begin_loading(loading_state, req.model, nodes_count, size_bytes, estimated_s)
        try:
            old = get_pool(cluster_id, alias)
            if old is not None and not req.force_hot_swap:
                sys.stderr.write(
                    f"[load] {cluster_id}[{alias}]: stopping old pool before starting new\n"
                )
                await old.stop()
                old = None
                del_pool(cluster_id, alias)
            new_pool = RunnerPool(
                model=req.model, mode=req.mode, use_ap=req.use_ap,
                nodes_count=nodes_count, cluster=cluster_id,
                kv_q8=kv_q8,
                draft_model=req.draft_model,
                num_draft_tokens=req.num_draft_tokens,
                alias=alias,
                node_indices=node_indices,
            )
            try:
                await new_pool.start()
            except Exception as e:
                try:
                    await new_pool.stop()
                except Exception:
                    pass
                raise HTTPException(500, f"{cluster_id} load failed: {e}")
            if old is not None:
                await old.stop()
            set_pool(cluster_id, alias, new_pool)
            save_cluster_state_v2(cluster_id)
            record_load_history(cluster_id, req.model, new_pool.load_s or 0.0,
                                size_bytes, nodes_count)
        finally:
            _end_loading(loading_state)
    _apply_default_ttl_to_pools()
    await _maybe_auto_prewarm(new_pool, cluster_id)
    return {"loaded": True, "cluster": cluster_id, "alias": alias,
            "model": new_pool.model, "nodes": new_pool.nodes_count,
            "node_indices": node_indices,
            "load_s": new_pool.load_s}


class _ArgoUnloadBody(BaseModel):
    alias: Optional[str] = None


_TELEMAK_MODELS_CACHE: dict[str, tuple[float, list[str]]] = {}
_TELEMAK_CACHE_TTL_S = 8.0  # seconds — debounce dashboard polls without staleness


def _telemak_short_id(hf_id: str) -> str:
    """Short alias for a Telemak-loaded HF id — kebab-cased, no org prefix.

    `mlx-community/Qwen3-0.6B-4bit` → `qwen3-0.6b-4bit`. URL-safe; used as
    the colon-suffix in `<cluster_id>:<short_id>` for multi-model routing
    (v1.7-v1.8 of Telemak V1)."""
    tail = hf_id.rsplit("/", 1)[-1]
    return tail.lower()


def _telemak_split_alias(model: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Split `cluster_id:short_model_id` → (cluster_id, short_id). If no colon,
    returns (model, None). Used by chat_completions routing to support both
    bare-cluster (back-compat, 1 model) and colon-suffix (N models) forms."""
    if not model:
        return (None, None)
    if ":" not in model:
        return (model, None)
    cluster, short = model.split(":", 1)
    return (cluster, short or None)


async def _telemak_reconcile_models_dir(cluster_id: str, cd: dict) -> None:
    """OdyssAI-X (master) asserts the cluster's configured models_dir onto the
    Telemak server (slave). Idempotent: GET the server's effective dir and POST
    only when it differs from cd['models_dir'] (no re-scan churn on polls).
    No-op when the cluster has no models_dir. Best-effort — a transient failure
    is swallowed; the proxy call that follows surfaces real outages. We do NOT
    auto-create a missing dir (a non-existent configured dir is a config error to
    surface, not to silently create empty)."""
    upstream = (cd.get("upstream") or "").rstrip("/")
    want = (cd.get("models_dir") or "").strip()
    if not upstream or not want:
        return
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{upstream}/admin/models-dir")
            cur = (r.json().get("dir") or "") if r.status_code == 200 else ""
            if cur.rstrip("/") == want.rstrip("/"):
                return  # already in sync
            await client.post(
                f"{upstream}/admin/models-dir",
                json={"dir": want, "managed": True},
            )
    except httpx.RequestError:
        return


async def _telemak_loaded_models(cluster_id: str, cd: dict, force: bool = False) -> list[str]:
    """Query the Telemak upstream's /v1/models — return the list of model ids
    currently loaded. Empty list if upstream unreachable or has no models.

    Cached for ~8s per cluster to avoid hammering the upstream during dashboard
    refreshes (every status poll, /v1/models call, etc.). Set force=True to
    bypass the cache — used right after /admin/load and /admin/unload."""
    now = time.time()
    cached = _TELEMAK_MODELS_CACHE.get(cluster_id)
    if not force and cached and (now - cached[0]) < _TELEMAK_CACHE_TTL_S:
        return cached[1]
    upstream = (cd.get("upstream") or "").rstrip("/")
    if not upstream:
        return []
    import httpx
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(f"{upstream}/v1/models")
            if r.status_code != 200:
                models = []
            else:
                data = r.json().get("data", [])
                models = [m.get("id") for m in data if m.get("id")]
    except Exception:
        models = []
    _TELEMAK_MODELS_CACHE[cluster_id] = (now, models)
    return models


def _telemak_cache_invalidate(cluster_id: str) -> None:
    """Invalidate the cached /v1/models list for a cluster — call after
    load/unload so the next status poll sees the new state."""
    _TELEMAK_MODELS_CACHE.pop(cluster_id, None)
    _TELEMAK_CAPS_CACHE.pop(cluster_id, None)


_TELEMAK_CAPS_CACHE: dict[str, tuple[float, dict]] = {}
_TELEMAK_CAPS_TTL_S = 60.0


async def _telemak_capabilities(cluster_id: str, cd: dict) -> dict:
    """Fetch `/.well-known/inference-engine.json` from the Telemak upstream.

    Cached 60s per cluster. Returns `{}` on any failure — the caller falls
    back to the legacy hardcoded `stream: True, tools: False` shape, so
    older Telemak builds without the well-known endpoint keep working.

    Schema produced by Telemak >= 0.2.0:
      {engine: "telemak", version, capabilities: {stream, tools, vision,
       max_context, session_cache, openai_compat, anthropic_compat}, models}
    """
    now = time.time()
    cached = _TELEMAK_CAPS_CACHE.get(cluster_id)
    if cached and (now - cached[0]) < _TELEMAK_CAPS_TTL_S:
        return cached[1]
    upstream = (cd.get("upstream") or "").rstrip("/")
    if not upstream:
        return {}
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{upstream}/.well-known/inference-engine.json")
            if r.status_code != 200:
                _TELEMAK_CAPS_CACHE[cluster_id] = (now, {})
                return {}
            data = r.json()
    except Exception:
        _TELEMAK_CAPS_CACHE[cluster_id] = (now, {})
        return {}
    _TELEMAK_CAPS_CACHE[cluster_id] = (now, data)
    return data


async def _telemak_proxy_chat_completion(
    cluster_id: str,
    cd: dict,
    body: dict,
    stream: bool,
    requested_short_id: Optional[str] = None,
):
    """Proxy a /v1/chat/completions request to the Telemak upstream.

    Returns either a JSON dict (non-streaming) or a StreamingResponse (SSE
    streaming). The `model` field in the body is rewritten to the actually-
    loaded upstream model id — the client used `cluster_id` as the alias.

    Multi-model routing (V1):
      - `requested_short_id=None` (bare cluster alias) + 1 loaded → use it.
      - `requested_short_id=None` + N loaded → 400 ambiguous.
      - `requested_short_id="<short>"` → match against loaded[i].short_id.
        404 if no match.

    If the upstream model auto-opens a `<think>` block (Qwen3.5, Qwen3.6,
    MiniMax), this proxy filters reasoning out of `content` and routes it
    into `reasoning_content`, both for the non-stream and stream paths.
    """
    upstream = (cd.get("upstream") or "").rstrip("/")
    if not upstream:
        raise HTTPException(400, f"{cluster_id}: missing upstream URL")
    loaded = await _telemak_loaded_models(cluster_id, cd)
    if not loaded:
        raise HTTPException(503, f"{cluster_id}: no model loaded on upstream — POST /admin/clusters/{cluster_id}/load first")

    if requested_short_id:
        match = next((m for m in loaded if _telemak_short_id(m) == requested_short_id), None)
        if not match:
            available = [f"{cluster_id}:{_telemak_short_id(m)}" for m in loaded]
            raise HTTPException(
                404,
                f"{cluster_id}: no loaded model matches short id '{requested_short_id}'. available: {available}",
            )
        upstream_model = match
    elif len(loaded) > 1:
        available = [f"{cluster_id}:{_telemak_short_id(m)}" for m in loaded]
        raise HTTPException(
            400,
            f"{cluster_id}: ambiguous model id — {len(loaded)} models loaded. use cluster:model form. available: {available}",
        )
    else:
        upstream_model = loaded[0]
    # Auto-think heuristic: Qwen3.5/3.6/MiniMax templates auto-open <think>
    # at prompt time, so reasoning streams BEFORE any visible </think> tag.
    # When that's the case we seed the filter with `in_think=True` and
    # route everything to `reasoning_content` until the model finally
    # closes with `</think>`.
    #
    # `enable_thinking: false` from the client means "I don't want
    # thinking in the output". Qwen3.5/3.6 honor the flag at the chat
    # template level — the model genuinely stops emitting <think> blocks,
    # so we MUST disable the filter (otherwise it eats the visible
    # output as reasoning_content — empty-content ghost bug).
    #
    # MiniMax M2 docs explicitly say it IGNORES the flag and always
    # wraps reasoning in <think>. If we trust the flag for MiniMax, the
    # reasoning + `</think>` literal leaks into `content` (Companion
    # shows "The user is asking…</think>\n\n# Real story" in chat).
    # So for MiniMax (and any other model in the ignore list), keep the
    # filter on regardless of the flag.
    # Shared decision with the local-pool path — see _should_filter_think.
    auto_think = _should_filter_think(upstream_model, body.get("enable_thinking"))
    forward_body = dict(body)
    # Per-model reasoning_effort default (Step-3.7 → minimal) when the client
    # didn't specify one — keeps the proxied model fast by default, same as the
    # local pool path. An explicit client value in body already carries over.
    if not forward_body.get("reasoning_effort"):
        _re_default = _default_reasoning_effort(upstream_model)
        if _re_default:
            forward_body["reasoning_effort"] = _re_default
    forward_body["model"] = upstream_model
    if stream:
        # Ask the upstream for a trailing usage chunk (mlx_vlm.server and
        # Telemak both honor stream_options.include_usage) so the client can
        # compute tok/s. The cloud proxy already does this; the telemak path
        # did not, so VLM rows showed "Chunks" instead of a speed metric.
        _so = dict(forward_body.get("stream_options") or {})
        _so.setdefault("include_usage", True)
        forward_body["stream_options"] = _so
    import httpx
    if not stream:
        _t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                r = await client.post(f"{upstream}/v1/chat/completions", json=forward_body)
        except Exception as e:
            raise HTTPException(502, f"telemak upstream unreachable: {e}")
        _elapsed = max(0.001, time.time() - _t0)
        if r.status_code >= 400:
            raise HTTPException(r.status_code, f"telemak upstream error: {r.text[:300]}")
        out = r.json()
        # Filter <think>...</think> out of content. Seed in_think=True for
        # auto-opening models so reasoning before the first </think> close
        # gets routed to reasoning_content instead of leaking into content.
        if auto_think:
            for choice in out.get("choices", []) or []:
                msg = choice.get("message") or {}
                raw = msg.get("content") or ""
                _om, _cm = _model_think_markers(upstream_model)
                state = {"in_think": not raw.lstrip().startswith(_om),
                         "carry": "", "open": _om, "close": _cm}
                vis, reas = _split_think_stream(raw, state)
                vis2, reas2 = _flush_think_stream(state)
                visible = (vis + vis2).lstrip()
                reasoning = (reas + reas2).strip()
                msg["content"] = visible
                if reasoning:
                    msg["reasoning_content"] = reasoning
                choice["message"] = msg
        out["model"] = f"{cluster_id}:{requested_short_id}" if requested_short_id else cluster_id
        # Record metrics so Recent activity appears for Telemak clusters (#32).
        # Non-stream: wall-clock = full latency. The client sees nothing until
        # the whole response lands, so observed TTFT == elapsed (no separate
        # first-token event exists for a non-streamed completion).
        _usage = out.get("usage") or {}
        record_metric(
            client="telemak-proxy",
            ntoks=_usage.get("completion_tokens") or 0,
            elapsed_s=_elapsed,
            ttft_s=_elapsed,
            prompt_chars=(_usage.get("prompt_tokens") or 0) * 4,
            model=upstream_model,
            cluster=cluster_id,
        )
        return out

    # Streaming path — parse SSE chunks, route delta.content through the
    # think filter, re-emit content + reasoning_content as separate deltas.
    async def _gen():
        if auto_think:
            _om, _cm = _model_think_markers(upstream_model)
            state = {"in_think": _seed_in_think(upstream_model, body.get("enable_thinking")),
                     "carry": "", "open": _om, "close": _cm}
        else:
            state = None
        _ttft: list = []          # mutable cell for TTFT
        _t0 = time.time()
        _ntoks = 0
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{upstream}/v1/chat/completions", json=forward_body) as r:
                if r.status_code >= 400:
                    err_body = (await r.aread()).decode(errors="replace")[:500]
                    yield f"data: {json.dumps({'error': {'message': f'telemak upstream {r.status_code}: {err_body}'}})}\n\n".encode()
                    return
                if not auto_think:
                    # No filter needed — pipe bytes through, count tokens from
                    # usage chunk for metrics.
                    async for chunk in r.aiter_raw():
                        if chunk:
                            if not _ttft:
                                _ttft.append(time.time() - _t0)
                            # Parse usage from SSE lines as they stream
                            for line in chunk.split(b"\n"):
                                s = line.strip()
                                if s.startswith(b"data: ") and s != b"data: [DONE]":
                                    try:
                                        obj = json.loads(s[6:])
                                        u = obj.get("usage") or {}
                                        if u.get("completion_tokens"):
                                            _ntoks = u["completion_tokens"]
                                    except Exception:
                                        pass
                            yield chunk
                    record_metric(
                        client="telemak-proxy",
                        ntoks=_ntoks,
                        elapsed_s=max(0.001, time.time() - _t0),
                        ttft_s=_ttft[0] if _ttft else None,
                        prompt_chars=0,
                        model=upstream_model,
                        cluster=cluster_id,
                    )
                    return
                # Auto-think filter mode: line-buffer the SSE stream so we
                # can parse each `data: {...}` event, extract delta.content,
                # filter, and re-emit as content + reasoning_content deltas.
                #
                # Important: flush any residual think-stream carry BEFORE
                # passing `data: [DONE]` through, so the final reasoning
                # bytes never land after the [DONE] sentinel (which some
                # clients treat as end-of-stream and discard later events).
                buf = b""
                done_seen = False
                async for chunk in r.aiter_raw():
                    if not chunk:
                        continue
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line.strip() == b"data: [DONE]":
                            # Flush + emit DONE last.
                            async for f in _telemak_emit_flush(state, cluster_id):
                                yield f
                            yield b"data: [DONE]\n"
                            done_seen = True
                            record_metric(
                                client="telemak-proxy",
                                ntoks=_ntoks,
                                elapsed_s=max(0.001, time.time() - _t0),
                                ttft_s=_ttft[0] if _ttft else None,
                                prompt_chars=0,
                                model=upstream_model,
                                cluster=cluster_id,
                            )
                            break
                        # Usage-only chunk (stream_options.include_usage):
                        # choices:[] + usage:{...}, no content delta. The think
                        # filter would drop it, so the client never gets tok/s.
                        # Pass it through verbatim (its delta is empty) and
                        # capture completion_tokens for our own metrics.
                        _st = line.strip()
                        if _st.startswith(b"data: ") and _st != b"data: [DONE]":
                            try:
                                _uo = json.loads(line[6:])
                            except Exception:
                                _uo = None
                            if isinstance(_uo, dict) and _uo.get("usage"):
                                _u = _uo.get("usage") or {}
                                if _u.get("completion_tokens"):
                                    _ntoks = _u["completion_tokens"]
                                if not _ttft:
                                    _ttft.append(time.time() - _t0)
                                yield line + b"\n"
                                continue
                        out_line = _telemak_filter_sse_line(line, state, cluster_id)
                        if out_line is not None:
                            if not _ttft:
                                _ttft.append(time.time() - _t0)
                            # Track completion tokens from usage field
                            try:
                                obj = json.loads(line[6:]) if line.strip().startswith(b"data: ") else {}
                                u = obj.get("usage") or {}
                                if u.get("completion_tokens"):
                                    _ntoks = u["completion_tokens"]
                            except Exception:
                                pass
                            yield out_line
                    if done_seen:
                        break
                if not done_seen:
                    # Upstream closed without [DONE] — flush what we have.
                    if buf:
                        out_line = _telemak_filter_sse_line(buf, state, cluster_id)
                        if out_line is not None:
                            yield out_line
                    async for f in _telemak_emit_flush(state, cluster_id):
                        yield f
    from fastapi.responses import StreamingResponse
    return StreamingResponse(_gen(), media_type="text/event-stream")


async def _telemak_proxy_messages(
    cluster_id: str,
    cd: dict,
    body: dict,
    stream: bool,
    requested_short_id: Optional[str] = None,
):
    """Proxy a /v1/messages (Anthropic shape) request to the Telemak upstream.

    Mirror of `_telemak_proxy_chat_completion` for the Anthropic surface
    (Telemak V1 Block 4 ships /v1/messages natively). We don't translate
    the body — Telemak speaks Anthropic on the same model.

    For streaming, Anthropic SSE events (`message_start`, `content_block_*`,
    `message_*`) are piped through verbatim. No `<think>` filtering for now —
    that's a `/v1/chat/completions` concern.
    """
    upstream = (cd.get("upstream") or "").rstrip("/")
    if not upstream:
        raise HTTPException(400, f"{cluster_id}: missing upstream URL")
    loaded = await _telemak_loaded_models(cluster_id, cd)
    if not loaded:
        raise HTTPException(503, f"{cluster_id}: no model loaded on upstream")

    if requested_short_id:
        match = next((m for m in loaded if _telemak_short_id(m) == requested_short_id), None)
        if not match:
            available = [f"{cluster_id}:{_telemak_short_id(m)}" for m in loaded]
            raise HTTPException(
                404,
                f"{cluster_id}: no loaded model matches '{requested_short_id}'. available: {available}",
            )
        upstream_model = match
    elif len(loaded) > 1:
        available = [f"{cluster_id}:{_telemak_short_id(m)}" for m in loaded]
        raise HTTPException(
            400,
            f"{cluster_id}: ambiguous model — {len(loaded)} loaded. use cluster:short form. available: {available}",
        )
    else:
        upstream_model = loaded[0]

    forward_body = dict(body)
    forward_body["model"] = upstream_model
    # Per-model reasoning_effort default (Step-3.7 → minimal), same as the chat
    # proxy + local pool path. Explicit client value in body carries over.
    if not forward_body.get("reasoning_effort"):
        _re_default = _default_reasoning_effort(upstream_model)
        if _re_default:
            forward_body["reasoning_effort"] = _re_default

    import httpx
    if not stream:
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                r = await client.post(f"{upstream}/v1/messages", json=forward_body)
        except Exception as e:
            raise HTTPException(502, f"telemak upstream unreachable: {e}")
        if r.status_code >= 400:
            raise HTTPException(r.status_code, f"telemak upstream error: {r.text[:300]}")
        out = r.json()
        out["model"] = f"{cluster_id}:{requested_short_id}" if requested_short_id else cluster_id
        return out

    async def _gen():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{upstream}/v1/messages", json=forward_body) as r:
                if r.status_code >= 400:
                    err_body = (await r.aread()).decode(errors="replace")[:500]
                    yield f"event: error\ndata: {json.dumps({'type':'error','error':{'type':'api_error','message': f'telemak upstream {r.status_code}: {err_body}'}})}\n\n".encode()
                    return
                async for chunk in r.aiter_raw():
                    if chunk:
                        yield chunk

    from fastapi.responses import StreamingResponse
    return StreamingResponse(_gen(), media_type="text/event-stream")


async def _telemak_emit_flush(state: dict, cluster_id: str):
    """Yield the residual carry from the think-stream filter as a final delta
    chunk, before [DONE]. No-op if there's nothing to flush."""
    vis, reas = _flush_think_stream(state)
    if not vis and not reas:
        return
    delta: dict = {}
    if vis: delta["content"] = vis
    if reas: delta["reasoning_content"] = reas
    final = {
        "id": "chatcmpl-telemak-flush",
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": cluster_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n".encode()


def _telemak_filter_sse_line(line: bytes, state: dict, cluster_id: str) -> Optional[bytes]:
    """Parse one SSE line. If it's a `data: {...}` chat-completion chunk with a
    delta.content, run the content through the think-stream filter and re-emit
    with content + reasoning_content separated. Other lines (blank, [DONE],
    role-only deltas, finish-reason chunks) pass through unchanged.

    Returns the line to emit (bytes including the trailing `\n`), or None if
    fully consumed (e.g. content all routed to carry, no visible output yet)."""
    raw = line.rstrip(b"\r")
    if not raw.startswith(b"data: "):
        # Pass through unchanged (blank lines between events, comments, etc.)
        return raw + b"\n"
    payload = raw[len(b"data: "):]
    if payload.strip() == b"[DONE]":
        return raw + b"\n"
    try:
        obj = json.loads(payload)
    except Exception:
        return raw + b"\n"
    # Find the delta.content if present.
    try:
        choices = obj.get("choices") or []
        if not choices:
            obj["model"] = cluster_id
            return (b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n")
        choice0 = choices[0]
        delta = choice0.get("delta") or {}
        content = delta.get("content")
        if not isinstance(content, str) or content == "":
            # Pass through (role-only chunk, finish_reason chunk, etc.)
            obj["model"] = cluster_id
            return (b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n")
        vis, reas = _split_think_stream(content, state)
        if not vis and not reas:
            return None  # all carried over, nothing to emit yet
        new_delta = {k: v for k, v in delta.items() if k != "content"}
        if vis:
            new_delta["content"] = vis
        if reas:
            new_delta["reasoning_content"] = reas
        new_obj = dict(obj)
        new_obj["model"] = cluster_id
        new_obj["choices"] = [dict(choice0, delta=new_delta)]
        return (b"data: " + json.dumps(new_obj, ensure_ascii=False).encode() + b"\n")
    except Exception:
        return raw + b"\n"


async def _telemak_status(cluster_id: str, cd: dict) -> dict:
    """Return a status snapshot for a kind=telemak cluster.

    Queries /v1/models for the loaded-models list and /health for memory +
    throughput metrics. Telemak holds N models concurrently (V1 Block 1+);
    we surface the full list as `models_loaded` while keeping `model`
    (singular) populated with the first entry so older dashboard code that
    only checks "any model loaded?" still works.

    Shape stays compatible with OdyssAI-X' mlx-distributed cluster status so
    the dashboard renderers don't crash on the kind=telemak branch.
    """
    upstream = (cd.get("upstream") or "").rstrip("/")
    models_loaded: list[str] = []
    # Per-model metadata derived from Telemak's /v1/models + capability
    # contract. Shape : {model_id: {kind: "llm"|"embedder"|"vlm",
    # mtp_enabled: bool, draft_model: str|null, num_draft_tokens: int|null}}.
    # Default values mean "Telemak didn't tell us, treat as LLM with no
    # speculative decoding" — keeps old Telemak builds compatible.
    models_details: dict[str, dict] = {}
    reachable = False
    wired_used_gb = None
    wired_free_gb = None
    avg_tok_s_recent = None
    requests_served = None
    uptime_s = None
    upstream_version = None
    spec_modes: list[str] = []
    # Activity readout. Two sources, in priority order:
    #   1. /admin/activity (Telemak 0.6.15+) — gives the canonical
    #      current_phase / current_model / current_tok_s straight from
    #      the engine. When available, `busy` derives from
    #      current_phase != "idle" (no heuristic window).
    #   2. /admin/sessions (older Telemak) — fallback, infers busy from
    #      last_used_s < 5s on any KV-cached session.
    BUSY_WINDOW_S = 5.0
    active_sessions_count: int = 0
    last_request_seconds_ago: Optional[float] = None
    busy = False
    sessions_summary: list[dict] = []
    # /admin/activity fields (Telemak 0.6.15+). All optional — None
    # means the upstream is too old to report.
    current_phase: Optional[str] = None
    current_model: Optional[str] = None
    current_request_started_at: Optional[float] = None
    current_generated_tokens: Optional[int] = None
    current_tok_s: Optional[float] = None
    active_requests: Optional[int] = None
    last_error: Optional[str] = None
    if upstream:
        import httpx

        async def _safe_get(client: "httpx.AsyncClient", url: str):
            try:
                r = await client.get(url)
                return r if r.status_code == 200 else None
            except Exception:
                return None

        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                # Fire all 5 upstream GETs in parallel — previously sequential,
                # each waiting for the previous before starting (issue #26).
                # Wall-clock latency = max(individual) instead of sum.
                r_models, r_health, r_cap, r_activity, r_sessions = await asyncio.gather(
                    _safe_get(client, f"{upstream}/v1/models"),
                    _safe_get(client, f"{upstream}/health"),
                    _safe_get(client, f"{upstream}/.well-known/inference-engine.json"),
                    _safe_get(client, f"{upstream}/admin/activity"),
                    _safe_get(client, f"{upstream}/admin/sessions"),
                    return_exceptions=False,
                )

                if r_models is not None:
                    reachable = True
                    for m in r_models.json().get("data", []):
                        mid = m.get("id")
                        if not mid:
                            continue
                        models_loaded.append(mid)
                        x = m.get("x_telemak") or {}
                        models_details[mid] = {
                            "kind": x.get("kind") or "llm",
                            "draft_model": x.get("draft_model"),
                            "num_draft_tokens": x.get("num_draft_tokens"),
                        }

                if r_health is not None:
                    reachable = True
                    hj = r_health.json()
                    if not models_loaded:
                        models_loaded = hj.get("models_loaded", []) or []
                    wired_used_gb = hj.get("wired_memory_used_gb")
                    wired_free_gb = hj.get("wired_memory_free_gb")
                    avg_tok_s_recent = hj.get("avg_tok_s_recent")
                    requests_served = hj.get("requests_served")
                    uptime_s = hj.get("uptime_s")
                    upstream_version = hj.get("version")

                if r_cap is not None:
                    cj = r_cap.json()
                    spec_modes = (cj.get("capabilities", {}).get("speculative_decoding") or {}).get("modes") or []

                if r_activity is not None:
                    aj = r_activity.json() or {}
                    current_phase = aj.get("current_phase")
                    current_model = aj.get("current_model")
                    current_request_started_at = aj.get("current_request_started_at")
                    current_generated_tokens = aj.get("current_generated_tokens")
                    current_tok_s = aj.get("current_tok_s")
                    active_requests = aj.get("active_requests")
                    last_error = aj.get("last_error")
                    if current_phase and current_phase != "idle":
                        busy = True

                if r_sessions is not None:
                    sess_list = (r_sessions.json() or {}).get("sessions") or []
                    active_sessions_count = len(sess_list)
                    if sess_list:
                        recents = [
                            float(x.get("last_used_s"))
                            for x in sess_list
                            if x.get("last_used_s") is not None
                        ]
                        if recents:
                            last_request_seconds_ago = min(recents)
                            if current_phase is None:
                                busy = last_request_seconds_ago < BUSY_WINDOW_S
                        sessions_summary = [
                            {
                                "id": str(x.get("id"))[:12],
                                "model": x.get("model"),
                                "kv_size_mb": x.get("kv_size_mb"),
                                "last_used_s": x.get("last_used_s"),
                            }
                            for x in sess_list
                        ]
        except Exception:
            reachable = False
    # VLM attribution. A plain mlx_vlm.server does NOT emit x_telemak.kind, so
    # every loaded model above defaulted to kind="llm" — the dashboard would
    # then badge an engine-managed VLM as "chat". The engine IS the capability
    # truth here (it launched the server as a VLM), so when this telemak cluster
    # is engine-managed as a VLM (VLM_MANAGED_KEY) or explicitly flags vision,
    # stamp kind="vlm" onto every loaded model. Only takes effect while a model
    # is actually loaded (models_details is empty otherwise) — the badge shows
    # the VLM "in action", not on an idle/empty cluster.
    is_vlm_cluster = bool(cd.get(VLM_MANAGED_KEY) or cd.get("supports_vision"))
    if is_vlm_cluster:
        for det in models_details.values():
            det["kind"] = "vlm"
    # Derive mtp_enabled per model : engine reports an MTP mode AND model
    # is an LLM (embedders / VLMs don't benefit from MTP today).
    has_mtp_mode = any("mtp" in (m or "").lower() for m in spec_modes)
    for mid, det in models_details.items():
        det["mtp_enabled"] = bool(
            has_mtp_mode and (det.get("kind") or "llm") == "llm"
        )
    nodes = cd.get("nodes") or []
    master_ssh = (nodes[0] or {}).get("ssh") if nodes else None
    loaded_model = models_loaded[0] if models_loaded else None
    return {
        "loaded": bool(models_loaded),
        "cluster": cluster_id,
        "loading": None,
        "available_node_counts": [1],
        "max_nodes": 1,
        "nodes": 1 if models_loaded else 0,
        "model": loaded_model,                     # back-compat (singular)
        "models_loaded": models_loaded,            # canonical (list)
        "models_details": models_details,          # per-model kind / MTP / draft
        "spec_modes": spec_modes,                  # engine-level speculative_decoding modes
        "wired_memory_used_gb": wired_used_gb,
        "wired_memory_free_gb": wired_free_gb,
        "avg_tok_s_recent": avg_tok_s_recent,
        "requests_served": requests_served,
        "uptime_s": uptime_s,
        "upstream_version": upstream_version,
        # Activity readout. `busy` is the truthy summary derived from
        # /admin/activity when available (Telemak 0.6.15+), falling back
        # to a recent-session heuristic for older builds. The richer
        # fields below come from /admin/activity directly:
        #
        #   current_phase             "idle" | "prefill" | "decode" | "streaming"
        #   current_model             absolute path to the model serving the live turn
        #   current_request_started_at unix seconds since the turn began
        #   current_generated_tokens   token count emitted so far in the live turn
        #   current_tok_s              live decode rate
        #   active_requests            concurrent in-flight HTTP requests
        #   last_error                 last exception message, if any, else null
        #
        # All can be null when the upstream is too old (< 0.6.15) or
        # currently idle.
        "busy": busy,
        "active_sessions_count": active_sessions_count,
        "last_request_seconds_ago": last_request_seconds_ago,
        "sessions": sessions_summary,
        "current_phase": current_phase,
        "current_model": current_model,
        "current_request_started_at": current_request_started_at,
        "current_generated_tokens": current_generated_tokens,
        "current_tok_s": current_tok_s,
        "active_requests": active_requests,
        "last_error": last_error,
        "mode": "telemak",
        "topologies": {"1": [{"rank": 0, "ssh": master_ssh}]} if master_ssh else {"1": []},
        "models_dir": cd.get("models_dir"),
        "degraded": None,
        "kind": "telemak",
        # Engine-managed VLM marker. Distinct from `kind` (which stays "telemak"
        # so the http-proxy renderers keep working); the dashboard reads
        # `is_vlm` + `supports_vision` to draw the VLM badge and attribute the
        # VL model to its host. Only "in action" when a model is loaded.
        "is_vlm": is_vlm_cluster,
        "supports_vision": bool(cd.get("supports_vision")),
        "vlm_ready": bool(is_vlm_cluster and models_loaded and reachable),
        "host": (nodes[0] or {}).get("host") if nodes else None,
        "upstream": upstream,
        "upstream_reachable": reachable,
    }


async def _telemak_proxy_load(cluster_id: str, cd: dict, req) -> dict:
    """Proxy POST /admin/load to the Telemak upstream and reshape the response
    into OdyssAI-X' standard load-response shape so the dashboard renders it
    without a special-case."""
    upstream = (cd.get("upstream") or "").rstrip("/")
    if not upstream:
        raise HTTPException(400, f"{cluster_id}: missing upstream URL")
    if not req.model:
        raise HTTPException(400, "model is required")
    import httpx
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(f"{upstream}/admin/load", json={"model": req.model})
    except Exception as e:
        raise HTTPException(502, f"telemak upstream unreachable: {e}")
    if r.status_code >= 400:
        raise HTTPException(r.status_code, f"telemak upstream error: {r.text[:200]}")
    body = r.json() if r.text else {}
    # Stash the loaded model in cluster-config so /status & dashboard can show it
    save_cluster_def(cluster_id, {"_telemak_loaded_model": req.model})
    _telemak_cache_invalidate(cluster_id)
    return {
        "loaded": True,
        "cluster": cluster_id,
        "model": req.model,
        "nodes": 1,
        "mode": "telemak",
        "load_s": float(body.get("load_s") or 0.0),
        "upstream_response": body,
    }


async def _telemak_proxy_unload(
    cluster_id: str, cd: dict, model: Optional[str] = None
) -> dict:
    """Proxy POST /admin/unload to the Telemak upstream.

    `model=None` (or empty) → unload everything (`{"all": true}`).
    `model="<hf-id>"`        → unload that one model only, leaving any
    others in the registry untouched. The dashboard's per-row Unload
    button sends the specific id; the legacy "Unload" header button (no
    model attached) falls back to all.
    """
    upstream = (cd.get("upstream") or "").rstrip("/")
    if not upstream:
        raise HTTPException(400, f"{cluster_id}: missing upstream URL")
    payload: dict
    if model:
        payload = {"model": model}
    else:
        payload = {"all": True}
    import httpx
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{upstream}/admin/unload", json=payload)
    except Exception as e:
        raise HTTPException(502, f"telemak upstream unreachable: {e}")
    _telemak_cache_invalidate(cluster_id)
    if r.status_code >= 400 and r.status_code != 404:
        raise HTTPException(r.status_code, f"telemak upstream error: {r.text[:200]}")
    if not model:
        save_cluster_def(cluster_id, {"_telemak_loaded_model": None})
    return {"unloaded": True, "cluster": cluster_id, "model": model}


# ── Telemak service lifecycle (launchd over SSH) ──────────────────────────
#
# Telemak runs as a per-node launchd *user agent* `eu.odyssai.telemak`
# (KeepAlive=true) — the same service the Telemak menu-bar app drives. That
# app can only Start/Stop when running locally (launchctl targets the user's
# gui/$uid domain). OdyssAI-X lifts that limit by running the exact same
# launchctl verbs over SSH against the node. Verified: `launchctl print
# gui/$uid/eu.odyssai.telemak` answers over SSH, so the gui domain is
# reachable for an SSH'd command while the user is logged in.
#
# KeepAlive semantics (decided with the operator):
#   start   → bootstrap (load + run); falls back to kickstart if already loaded
#   stop    → SIGTERM the instance; KeepAlive relaunches it (a soft bounce)
#   restart → kickstart -k (kill + relaunch atomically) — the wedged-slot fix
#   quit    → bootout (remove the job from the domain; stays down until start)
TELEMAK_LAUNCHD_LABEL = "eu.odyssai.telemak"
TELEMAK_LAUNCHD_PLIST = "~/Library/LaunchAgents/eu.odyssai.telemak.plist"
TELEMAK_LIFECYCLE_ACTIONS = ("start", "stop", "restart", "quit")


def _telemak_lifecycle_cmd(action: str) -> Optional[str]:
    """Map a lifecycle action to a launchctl command line run on the node.
    `gui/$(id -u)` resolves to the logged-in user's GUI launchd domain."""
    label = TELEMAK_LAUNCHD_LABEL
    dom = "gui/$(id -u)"
    if action == "start":
        return (
            f"launchctl bootstrap {dom} {TELEMAK_LAUNCHD_PLIST} 2>/dev/null "
            f"|| launchctl kickstart {dom}/{label}"
        )
    if action == "stop":
        return f"launchctl kill SIGTERM {dom}/{label}"
    if action == "restart":
        return f"launchctl kickstart -k {dom}/{label}"
    if action == "quit":
        return f"launchctl bootout {dom}/{label}"
    return None


class TelemakLifecycleRequest(BaseModel):
    action: str  # start | stop | restart | quit


@app.post("/admin/clusters/{cluster_id}/telemak/lifecycle")
async def admin_telemak_lifecycle(cluster_id: str, req: TelemakLifecycleRequest):
    """Start / stop / restart / quit the Telemak launchd service on its node.

    Telemak-only (http-proxy single node). Resolves the node's SSH target
    from the cluster def and runs the matching `launchctl` verb remotely.
    """
    if not cluster_exists(cluster_id):
        raise HTTPException(404, f"unknown cluster {cluster_id}")
    cd = get_cluster_def(cluster_id)
    if cd.get("kind") != "telemak":
        raise HTTPException(400, f"{cluster_id} is not a telemak cluster")
    action = (req.action or "").strip().lower()
    cmd = _telemak_lifecycle_cmd(action)
    if not cmd:
        raise HTTPException(
            400,
            f"invalid action {req.action!r} — expected one of "
            f"{', '.join(TELEMAK_LIFECYCLE_ACTIONS)}",
        )
    nodes = cd.get("nodes") or []
    ssh = (nodes[0] or {}).get("ssh") if nodes else None
    if not ssh:
        raise HTTPException(400, f"{cluster_id}: telemak node has no ssh target")
    loop = asyncio.get_event_loop()
    try:
        rc, out, err = await loop.run_in_executor(None, _ssh_exec, ssh, cmd, 20)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, f"launchctl {action} on {ssh} timed out")
    except Exception as e:
        raise HTTPException(502, f"ssh to {ssh} failed: {e}")
    _telemak_cache_invalidate(cluster_id)
    # launchctl returns non-zero for benign no-ops (bootstrap when already
    # loaded, bootout when already out). Surface rc + streams so the dashboard
    # can show a precise message rather than a blanket failure.
    return {
        "cluster": cluster_id,
        "action": action,
        "ok": rc == 0,
        "rc": rc,
        "ssh": ssh,
        "stdout": (out or "").strip()[:400],
        "stderr": (err or "").strip()[:400],
    }


@app.post("/admin/clusters/{cluster_id}/unload")
async def admin_cluster_unload(
    cluster_id: str,
    request: Request,
    alias: Optional[str] = None,  # ?alias=…  (query param)
):
    """Unload a pool from this cluster.

    - `?alias=default` (default) → unload the singleton pool.
    - `?alias=<other>`           → unload that specific extra pool.
    - `?alias=*`                 → unload ALL pools on this cluster.

    Accepts `alias` in the query string OR in a JSON body
    (`{"alias": "..."}`). Query wins if both are sent.

    Orphan sweep runs on the cluster's nodes regardless of which aliases
    were live, so a stale runner from a previous container life gets killed
    even if our state says nothing was loaded.
    """
    if not cluster_exists(cluster_id):
        raise HTTPException(404, f"unknown cluster {cluster_id}")
    # kind=telemak: proxy to upstream /admin/unload. Optional `model` in
    # the body targets a specific loaded model; absent → unload all.
    cd_proxy = get_cluster_def(cluster_id)
    if cd_proxy.get("kind") == "telemak":
        telemak_model: Optional[str] = None
        try:
            raw = await request.body()
            if raw:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    val = payload.get("model")
                    if isinstance(val, str) and val:
                        telemak_model = val
        except Exception:
            pass
        return await _telemak_proxy_unload(cluster_id, cd_proxy, telemak_model)
    body_alias: Optional[str] = None
    if alias is None:
        try:
            raw = await request.body()
            if raw:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    val = payload.get("alias")
                    if isinstance(val, str) and val:
                        body_alias = val
        except Exception:
            pass
    target = (alias or body_alias or DEFAULT_ALIAS).strip().lower()
    async with get_admin_lock(cluster_id):
        if target == "*":
            for a, p in list_pools(cluster_id):
                try:
                    await p.stop()
                except Exception as e:
                    sys.stderr.write(f"[unload] {cluster_id}[{a}] stop error: {e}\n")
                del_pool(cluster_id, a)
        else:
            pool = get_pool(cluster_id, target)
            if pool is not None:
                await pool.stop()
                del_pool(cluster_id, target)
        sweep = await asyncio.to_thread(_sweep_orphan_runners, cluster_id)
    save_cluster_state_v2(cluster_id)
    return {
        "loaded": False, "cluster": cluster_id,
        "unloaded_alias": target,
        "remaining_aliases": pool_aliases(cluster_id),
        "sweep": sweep,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Recovery ladder — explicit reset endpoint per cluster
#
# Runs the full sequence the audit recommended:
#   1. cancel in-flight runs (soft + hard via pool.cancel)
#   2. stop the pool (if any)
#   3. orphan sweep (kills zombies + probes wired memory)
#   4. clear session caches on each rank
#   5. clear the cluster_degraded flag if the sweep came back clean
#
# Endpoint returns a structured report so the dashboard can show what
# happened at each step. The 'cleared' field tells the caller whether
# it's safe to load again.
# ──────────────────────────────────────────────────────────────────────────────
async def _cluster_reset(cluster_id: str) -> dict:
    """Run the recovery ladder for one cluster. Used by the per-cluster
    reset endpoints below. Always best-effort: failures at any step are
    reported but never raised — the goal is to get the cluster as clean
    as possible regardless of partial progress."""
    report: dict = {"cluster": cluster_id, "steps": []}

    def _step(name: str, ok: bool, detail: Any = None) -> None:
        report["steps"].append({"name": name, "ok": ok, "detail": detail})

    # 1. Cancel in-flight runs scoped to this cluster
    try:
        cancelled = 0
        hard_sent = 0
        pool = _pool_for_cluster_name(cluster_id)
        for rid, ev in list(_active_run_cancels.items()):
            r = _active_runs.get(rid)
            if r and r.get("cluster") == cluster_id:
                ev.set()
                cancelled += 1
                r["status"] = "cancelling"
                if pool is not None:
                    try:
                        hard_sent += await pool.cancel(rid)
                    except Exception:
                        pass
        _step("cancel_runs", True, {"cancelled": cancelled, "hard_sent": hard_sent})
    except Exception as e:
        _step("cancel_runs", False, str(e))

    # 2. Stop the pool (handles state file + admin lock)
    try:
        if cluster_id == "nautilus":
            global _pool
            async with _admin_lock:
                if _pool is not None:
                    await _pool.stop()
                    _pool = None
            try:
                if STATE_FILE.exists():
                    STATE_FILE.unlink()
            except Exception:
                pass
        else:
            async with get_admin_lock(cluster_id):
                # Stop ALL pools on this cluster (default + extras). Reset is
                # a cluster-level recovery — partial reset doesn't make sense
                # when JACCL is sick across all ranks.
                for a, p in list_pools(cluster_id):
                    try:
                        await p.stop()
                    except Exception as e:
                        sys.stderr.write(f"[reset] {cluster_id}[{a}] stop error: {e}\n")
                    del_pool(cluster_id, a)
            save_cluster_state_v2(cluster_id)
        _step("stop_pool", True)
    except Exception as e:
        _step("stop_pool", False, str(e))

    # 3. Orphan sweep — also re-probes wired memory and may re-mark
    #    degraded if the leak is still there.
    sweep_result = None
    try:
        sweep_result = await asyncio.to_thread(_sweep_orphan_runners, cluster_id)
        _step("orphan_sweep", True, {
            "warnings": sweep_result.get("warnings", []),
            "nodes": [{"host": r["host"], "result": r.get("result"),
                       "wired_gb": r.get("wired_gb"),
                       "wired_warn": r.get("wired_warn")}
                      for r in sweep_result.get("swept", [])],
        })
    except Exception as e:
        _step("orphan_sweep", False, str(e))

    # 4. Wipe in-process session caches — they referenced KV state on the
    #    runners we just killed. Without this, the next load would route
    #    stale session_ids to non-existent caches.
    # (The session_store lives in the runner process, so killing them
    # already dropped the in-memory caches. This step is a no-op today
    # but kept as a placeholder for the multi-pool future where the
    # API may also keep cluster-level cache metadata.)
    _step("clear_sessions", True, {"note": "killed with the runner processes"})

    # 5. Decide whether the cluster is clean. The sweep re-marked degraded
    # if it still saw wired leaks; otherwise we clear the flag.
    still_degraded = (sweep_result is not None
                      and bool(sweep_result.get("warnings")))
    if not still_degraded:
        _clear_cluster_degraded(cluster_id)
        report["cleared"] = True
    else:
        report["cleared"] = False
        report["still_degraded"] = _cluster_degraded.get(cluster_id)
    return report


@app.post("/admin/clusters/{cluster_id}/reset")
async def admin_cluster_reset(cluster_id: str):
    if not cluster_exists(cluster_id):
        raise HTTPException(404, f"unknown cluster {cluster_id}")
    return await _cluster_reset(cluster_id)


@app.post("/admin/reset")
async def admin_nautilus_reset():
    return await _cluster_reset("nautilus")


@app.post("/admin/clusters/{cluster_id}/keepalive")
async def admin_cluster_keepalive(cluster_id: str):
    """#40 — manual JACCL keepalive probe. Exercises the group with a tiny
    all_sum and reports rtt; this is the same RunnerPool.keepalive() the
    background health loop (WU1–WU3) drives. Claims the pool's maintenance gate
    so the probe can't race an in-flight generation."""
    if not cluster_exists(cluster_id):
        raise HTTPException(404, f"unknown cluster {cluster_id}")
    pool = _pool_for_cluster_name(cluster_id)
    if pool is None:
        raise HTTPException(404, f"{cluster_id}: no pool loaded")
    if not pool._try_claim_maintenance():
        return {"cluster": cluster_id, "ok": False, "skipped": True,
                "error": "pool busy (in-flight gen or maintenance)"}
    try:
        res = await pool.keepalive()
    finally:
        pool._release_maintenance()
    return {"cluster": cluster_id, **res}


# ──────────────────────────────────────────────────────────────────────────────
# Cache prewarm — explicit re-trigger of the shared system-prefix cache.
# Useful when the user edits the prefix text in settings and wants the new
# version active without unloading the model. Auto-runs after every load
# (see _maybe_auto_prewarm), so this endpoint is for manual refresh.
# ──────────────────────────────────────────────────────────────────────────────
class PrewarmRequest(BaseModel):
    text: Optional[str] = None  # None → use the saved system_prefix_text
    kv_q8: Optional[bool] = None  # None → cluster default


@app.post("/admin/{cluster}/prewarm")
async def admin_cluster_prewarm(cluster: str, req: PrewarmRequest):
    pool = _pool_for_cluster(cluster)
    if pool is None:
        raise HTTPException(404, f"{cluster} not loaded")
    text = req.text if req.text is not None else get_system_prefix_text()
    kv_q8 = req.kv_q8 if req.kv_q8 is not None else get_kv_q8_default()
    return await pool.prewarm(text, kv_q8=kv_q8)


class SessionClearRequest(BaseModel):
    session_id: Optional[str] = None  # None = clear all


def _broadcast_to_pool(pool: Optional[RunnerPool], obj: dict) -> None:
    if pool is None:
        return
    for r in pool.runners:
        try:
            r.send(obj)
        except Exception:
            pass


@app.post("/admin/sessions/clear")
async def admin_sessions_clear(req: SessionClearRequest):
    """Drop the prefix cache for a session (or all sessions). Broadcasts to
    every runner in every loaded pool so multi-rank caches stay coherent.
    """
    payload = {"cmd": "session_clear"}
    if req.session_id:
        payload["session_id"] = req.session_id
    _broadcast_to_pool(_pool, payload)
    for _cid, _alias, p in list_all_pools():
        _broadcast_to_pool(p, payload)
    return {"ok": True, "session_id": req.session_id, "scope": "all_clusters"}


# ──────────────────────────────────────────────────────────────────────────────
# Connection test — probes cluster nodes + remote services
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/admin/connection-test")
async def admin_connection_test(nodes: int = 1, cluster: str = "default"):
    """Run a parallel ping over SSH for each node of the named cluster.

    cluster: any configured cluster id.
    nodes:   requested pool size for that cluster.
    """
    if cluster in DEFAULT_CLUSTER_DEFS:
        max_nodes = len(get_cluster_def(cluster).get("nodes", [])) or 1
        if nodes < 1 or nodes > max_nodes:
            raise HTTPException(400, f"{cluster}: invalid nodes {nodes}")
        topo = require_topology(cluster, nodes)
    else:
        raise HTTPException(404, f"unknown cluster {cluster}")

    ssh_targets = [n["ssh"] for n in topo]
    ssh_results = await asyncio.gather(*[_ssh_ping(t) for t in ssh_targets])

    return {
        "cluster_name": cluster,
        "cluster": [
            {"rank": n["rank"], "ssh": n["ssh"], **r}
            for n, r in zip(topo, ssh_results)
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Engine-managed single-node VLM serving
#
# mlx-vlm is single-node only (no distributed serving path), so a VLM = exactly
# one node. Today the routing layer already works: a telemak-kind cluster with
# backend=http-proxy proxies chat/messages to an mlx_vlm.server, handling vision
# passthrough, <think> split, usage chunk and supports_vision. The ONLY thing
# missing was the operator having to (a) create a python3.12 venv, (b) launch
# mlx_vlm.server by hand, and (c) PUT a telemak cluster def. These endpoints
# fold (b) + (c) into the engine so `POST /admin/vlm/load` does the whole thing;
# (a) is provisioned once per node by scripts/install-mlx-vlm.sh.
#
# This is ADDITIVE: it reuses validate_cluster_def / save_cluster_def /
# get_cluster_def and the same telemak cluster shape the http-proxy router
# already consumes. It does NOT touch _telemak_proxy_chat_completion, the
# generation path, or any existing routing.
#
# Lifecycle:
#   load   → ssh nohup launch mlx_vlm.server → poll /v1/models until data
#            non-empty → upsert a telemak cluster def marked `_vlm_managed`
#   unload → ssh two-phase kill (SIGTERM → grace → SIGKILL) matching the port
#            AND the checkpoint (so an unrelated VLM on the node survives) →
#            tombstone the cluster def (same {_removed: true} mechanism as
#            DELETE /admin/clusters/{id})
# ──────────────────────────────────────────────────────────────────────────────
VLM_DEFAULT_VENV = env_get("VLM_VENV", "/Users/admin/.venvs/mlx-vlm")
VLM_DEFAULT_PORT = int(env_get("VLM_PORT", "8080") or "8080")
# 600s default: a 327GB 6-bit VL takes ~200-240s to load and a bigger VL
# (Q8 ~450GB, or a cold first Metal compile) needs more — 180 false-timed-out
# on the m3vl 6-bit (2026-07-02). Per-request ready_timeout_s still overrides.
VLM_READY_TIMEOUT_S = float(env_get("VLM_READY_TIMEOUT_S", "600") or "600")
# Marker key stamped on cluster-config entries the engine launched, so
# /admin/vlm/status and unload can distinguish engine-managed VLMs from
# hand-added telemak clusters.
VLM_MANAGED_KEY = "_vlm_managed"
# Hard co-residence enforcement. A VLM co-resident with a loaded text pool (or
# another VLM) on the same node can OOM (e.g. M3-VL 327GB + a text pool > 512GB).
# By default /admin/vlm/load REFUSES such a placement; set this env (or pass
# force=true on the request) to override and allow co-residence deliberately.
VLM_ALLOW_CORESIDENCE = (env_get("VLM_ALLOW_CORESIDENCE", "") or "").lower() in ("1", "true", "yes", "on")


def _vlm_ip_from_ssh(ssh_target: str) -> str:
    """Derive the upstream host (IP or hostname) from a `user@host[:port]` ssh
    target. This is the SSH host, NOT hardcoded — it comes from the request.

    `admin@192.168.86.30`      → `192.168.86.30`
    `admin@node-a.lan:2222`    → `node-a.lan`  (ssh port is not the http port)
    """
    host = ssh_target.split("@", 1)[-1]
    # Strip an ssh port suffix if present (`host:port`); the VLM http port is
    # a separate parameter. Guard against IPv6 (no bare colon form supported).
    if host.count(":") == 1:
        host = host.split(":", 1)[0]
    return host


def _vlm_log_path(vlm_id: str) -> str:
    """Remote log path for a managed VLM. Kept under $HOME so the launch env
    (HOME=/Users/admin) resolves it; validated slug means no shell metachars."""
    return f"~/mlx-vlm-{vlm_id}.log"


def _vlm_launch_cmd(vlm_id: str, venv: str, model_path: str, port: int) -> str:
    """Build the proven nohup launch command with a full exported env block.

    Mirrors the manual launch that ran on .29 tonight: export a clean env
    (HOME/USER/TMPDIR/PATH) so mlx_vlm.server finds its venv + HF cache the
    same way an interactive login shell would, then nohup the server detached
    with stdout+stderr to the per-id log. Echoes the launched PID on stdout.
    """
    venv_bin = f"{venv.rstrip('/')}/bin"
    server_bin = f"{venv_bin}/mlx_vlm.server"
    log = _vlm_log_path(vlm_id)
    # HOME/USER fixed to the admin account the nodes run under (matches the
    # ssh target `admin@...`); PATH puts the venv first. All interpolated
    # values are shlex.quote'd — model_path may contain '/', slug is validated.
    return (
        f"export HOME=/Users/admin USER=admin TMPDIR=/tmp "
        f"PATH={shlex.quote(venv_bin)}:/usr/bin:/bin:/usr/sbin:/sbin && "
        f"nohup {shlex.quote(server_bin)} "
        f"--model {shlex.quote(model_path)} "
        f"--host 0.0.0.0 --port {int(port)} "
        f"--trust-remote-code "
        f"> {log} 2>&1 & "
        f"echo VLM_PID=$!"
    )


def _vlm_kill_cmd(port: int, model_path: str) -> str:
    """Two-phase SIGTERM → grace → SIGKILL of the mlx_vlm.server for THIS port
    (and, defensively, matching the checkpoint) — mirrors _remote_pkill /
    _sweep_orphan_runners so Metal wired pages get a chance to free cleanly.

    Matching on `--port <port>` AND the model path avoids killing an unrelated
    mlx_vlm.server the operator may have on the same node. The pgrep pattern is
    an extended regex over the full command line.
    """
    # Escape regex metacharacters in the model path so it matches literally in
    # pgrep -f (which treats its pattern as a regex). Then shell-quote the whole
    # pattern for the remote shell.
    model_rx = re.escape(model_path)
    # Require both the server module and the exact port token; include the model
    # path as a further guard. `--port <port> ` ensures 8080 doesn't match 80801.
    pattern = shlex.quote(
        f"mlx_vlm.server.*--port {int(port)}( .*{model_rx})?"
    )
    return (
        f"if pgrep -f {pattern} >/dev/null 2>&1; then "
        f"  pkill -TERM -f {pattern} 2>/dev/null; "
        f"  for i in $(seq 1 24); do "
        f"    pgrep -f {pattern} >/dev/null 2>&1 || break; "
        f"    sleep 0.5; "
        f"  done; "
        f"  if pgrep -f {pattern} >/dev/null 2>&1; then "
        f"    pkill -9 -f {pattern}; echo 'killed (SIGKILL)'; "
        f"  else echo 'cleaned (SIGTERM)'; fi; "
        f"else echo 'no process'; fi"
    )


async def _vlm_probe_ready(ip: str, port: int, timeout: float = 5.0) -> Optional[bool]:
    """Health probe: GET http://<ip>:<port>/v1/models and check the `data`
    array is NON-EMPTY. mlx_vlm.server's own /health is unreliable (a zombie
    holds the port with data:[]), so the model list is the real readiness gate.

    Returns True (ready), False (reachable but empty → still loading / zombie),
    or None (unreachable → not up yet).
    """
    import httpx
    url = f"http://{ip}:{int(port)}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
    except Exception:
        return None
    if r.status_code >= 400:
        return False
    try:
        body = r.json()
    except Exception:
        return False
    data = body.get("data") if isinstance(body, dict) else None
    return bool(data)


async def _vlm_log_tail(ssh_target: str, vlm_id: str, lines: int = 40) -> str:
    """SSH-read the tail of a managed VLM's log — surfaced in the 503 body on
    ready-timeout so the operator sees WHY (OOM, bad checkpoint, import error)
    the same way RunnerProc.stderr_tail does for distributed runners."""
    log = _vlm_log_path(vlm_id)
    cmd = f"tail -n {int(lines)} {log} 2>/dev/null || true"
    try:
        rc, out, err = await asyncio.to_thread(_ssh_exec, ssh_target, cmd, 12)
        return (out or err or "").strip()[-4000:]
    except Exception as e:
        return f"(could not read log tail: {e})"


def _vlm_managed_cluster_ids() -> list[str]:
    """cluster-config ids the engine launched as VLMs (marked VLM_MANAGED_KEY),
    excluding tombstoned ones."""
    out: list[str] = []
    for cid, entry in _load_cluster_config().items():
        if not isinstance(entry, dict):
            continue
        if entry.get("_removed"):
            continue
        if entry.get(VLM_MANAGED_KEY):
            out.append(cid)
    return out


def _vlm_capacity_reasons(host_id: str, ssh_target: str, exclude_id: str) -> list[str]:
    """Co-residence reasons: is the target node already a member of a LOADED
    mlx-distributed pool, or already hosting another engine-managed VLM?
    Co-residence can OOM (e.g. M3-VL 327GB + a text pool > 512GB). Returns the
    list of concrete reasons (empty when the node is free). The caller decides
    whether to hard-block (default) or warn (override)."""
    reasons: list[str] = []
    # 1. Member of a loaded mlx-distributed pool?
    for cid, _alias, pool in list_all_pools():
        if not getattr(pool, "loaded", False):
            continue
        if host_id in _cluster_host_ids(cid):
            reasons.append(f"node {host_id} is in loaded pool '{cid}'")
    # 2. Already hosting another managed VLM (same ssh target / host)?
    for cid in _vlm_managed_cluster_ids():
        if cid == exclude_id:
            continue
        cd = get_cluster_def(cid)
        for n in cd.get("nodes") or []:
            if n.get("ssh") == ssh_target or n.get("host") == host_id:
                reasons.append(f"node {host_id} already hosts VLM '{cid}'")
                break
    return reasons


def _vlm_capacity_message(reasons: list[str]) -> str:
    return "co-residence detected — this can OOM the node: " + "; ".join(reasons)


class VLMLoadRequest(BaseModel):
    id: str                      # cluster slug [a-z0-9-]
    host: str                    # ssh target, e.g. "admin@192.168.86.30"
    model: str                   # path rel to models_dir OR absolute
    name: Optional[str] = None   # display name (defaults to id)
    port: Optional[int] = None   # default VLM_DEFAULT_PORT (8080)
    venv: Optional[str] = None   # default VLM_DEFAULT_VENV
    models_dir: Optional[str] = None   # override; else DEFAULT_MODELS_DIR
    ready_timeout_s: Optional[float] = None
    # Override the hard co-residence check (default False). Also overridable
    # cluster-wide via VLM_ALLOW_CORESIDENCE env. When set, co-residence is
    # allowed and only surfaced as a `warning` in the response.
    force: bool = False


class VLMUnloadRequest(BaseModel):
    id: str


def _vlm_resolve_model_path(model: str, models_dir: Optional[str]) -> str:
    """Absolute path passed to mlx_vlm.server --model.

    Absolute input (starts with '/') is used verbatim. A relative id is joined
    onto models_dir (default DEFAULT_MODELS_DIR) so `odyssai/MiniMax-M3-VL-...`
    resolves the same way the distributed loader resolves relative model ids.
    """
    m = model.strip()
    if m.startswith("/"):
        return m
    base = (models_dir or DEFAULT_MODELS_DIR).rstrip("/")
    return f"{base}/{m}"


@app.post("/admin/vlm/load")
async def admin_vlm_load(req: VLMLoadRequest):
    """Launch mlx_vlm.server on a chosen node and route to it via a telemak
    http-proxy cluster the engine creates.

    Steps:
      a. validate the slug + resolve ssh target and upstream host/IP
      b. ssh-launch mlx_vlm.server (nohup + full env), log to ~/mlx-vlm-<id>.log
      c. poll GET /v1/models until data non-empty (up to ready_timeout_s);
         on timeout, ssh-read the log tail and return 503 with it
      d. on ready, upsert a telemak cluster def {kind:telemak,
         backend:http-proxy, upstream, supports_vision:true, _vlm_managed:true}
      e. return {status:"loaded", id, upstream, model}
    """
    vlm_id = (req.id or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", vlm_id):
        raise HTTPException(400, f"id {req.id!r} must match [a-z0-9][a-z0-9-]{{0,40}} (URL-safe slug)")
    ssh_target = (req.host or "").strip()
    try:
        _safe_ssh_target(ssh_target)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not (req.model or "").strip():
        raise HTTPException(400, "model is required")

    port = int(req.port or VLM_DEFAULT_PORT)
    venv = (req.venv or VLM_DEFAULT_VENV).strip()
    ready_timeout = float(req.ready_timeout_s or VLM_READY_TIMEOUT_S)
    model_path = _vlm_resolve_model_path(req.model, req.models_dir)
    ip = _vlm_ip_from_ssh(ssh_target)
    host_id = _host_id_from_ssh(ssh_target)
    upstream = f"http://{ip}:{port}"

    # Hard capacity enforcement. A VLM co-resident with a loaded text pool (or
    # another loaded VLM) on the same node can OOM. REFUSE by default (409);
    # allow only when explicitly overridden (per-request force or the
    # VLM_ALLOW_CORESIDENCE env), in which case it degrades to a warning.
    reasons = _vlm_capacity_reasons(host_id, ssh_target, exclude_id=vlm_id)
    warning = None
    if reasons:
        override = bool(req.force) or VLM_ALLOW_CORESIDENCE
        msg = _vlm_capacity_message(reasons)
        if not override:
            raise HTTPException(
                409,
                f"{msg}. Refusing to load — unload the co-resident pool/VLM "
                f"first, pick a free node, or override with force=true "
                f"(or set VLM_ALLOW_CORESIDENCE).",
            )
        warning = msg + " (overridden)"

    # If already reachable on this port with a model loaded, treat as an early
    # zombie/duplicate — surface it rather than launching a second server that
    # fights for the port. The operator can /admin/vlm/unload first.
    already = await _vlm_probe_ready(ip, port)
    if already is True:
        raise HTTPException(
            409,
            f"{ip}:{port} already serving a model — unload it first "
            f"(POST /admin/vlm/unload) or pick another port",
        )

    # (b) launch
    launch = _vlm_launch_cmd(vlm_id, venv, model_path, port)
    launched_pid: Optional[str] = None
    try:
        rc, out, err = await asyncio.to_thread(_ssh_exec, ssh_target, launch, 20)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, f"ssh launch on {ssh_target} timed out")
    except Exception as e:
        raise HTTPException(502, f"ssh to {ssh_target} failed: {e}")
    if rc != 0:
        raise HTTPException(
            502,
            f"launch failed on {ssh_target} (rc={rc}): "
            f"{(err or out or '').strip()[:300]}",
        )
    for line in (out or "").splitlines():
        if line.startswith("VLM_PID="):
            launched_pid = line.split("=", 1)[1].strip()

    # (c) poll for readiness — data non-empty on /v1/models.
    deadline = time.time() + ready_timeout
    ready = False
    while time.time() < deadline:
        state = await _vlm_probe_ready(ip, port)
        if state is True:
            ready = True
            break
        await asyncio.sleep(3.0)
    if not ready:
        tail = await _vlm_log_tail(ssh_target, vlm_id)
        raise HTTPException(
            503,
            "mlx_vlm.server did not become ready within "
            f"{int(ready_timeout)}s on {upstream}. Log tail:\n{tail}",
        )

    # (d) upsert the telemak cluster def routing to the launched server.
    candidate = {
        "name": (req.name or vlm_id),
        "kind": "telemak",
        "backend": "http-proxy",
        "upstream": upstream,
        "supports_vision": True,
        "nodes": [{"host": host_id, "ssh": ssh_target, "master": True}],
    }
    verr = validate_cluster_def(vlm_id, candidate)
    if verr:
        # Server is up but the def is bad — kill what we launched so we don't
        # leave an unreachable-from-the-dashboard server holding the node.
        try:
            await asyncio.to_thread(_ssh_exec, ssh_target, _vlm_kill_cmd(port, model_path), 20)
        except Exception:
            pass
        raise HTTPException(400, f"cluster def invalid: {verr}")
    # Persist the def + engine-managed bookkeeping. save_cluster_def overlays
    # onto (and lifts any tombstone on) the cluster-config entry.
    save_cluster_def(vlm_id, {
        **candidate,
        VLM_MANAGED_KEY: True,
        "_vlm_port": port,
        "_vlm_model_path": model_path,
        "_vlm_pid": launched_pid,
        "_telemak_loaded_model": model_path,
    })
    _telemak_cache_invalidate(vlm_id)

    return {
        "status": "loaded",
        "id": vlm_id,
        "upstream": upstream,
        "model": model_path,
        "pid": launched_pid,
        "warning": warning,
    }


@app.post("/admin/vlm/unload")
async def admin_vlm_unload(req: VLMUnloadRequest):
    """Kill the mlx_vlm.server for a managed VLM and tombstone its cluster def.

    Two-phase SIGTERM → grace → SIGKILL matching the port + checkpoint (reused
    from the orphan-sweep pattern) so an unrelated VLM on the same node is left
    alone; then tombstone the cluster def with {_removed: true}, the same
    mechanism DELETE /admin/clusters/{id} uses.
    """
    vlm_id = (req.id or "").strip().lower()
    cd = get_cluster_def(vlm_id)
    if not cd or cd.get("kind") != "telemak":
        raise HTTPException(404, f"unknown VLM cluster {vlm_id}")
    nodes = cd.get("nodes") or []
    ssh_target = (nodes[0] or {}).get("ssh") if nodes else None
    if not ssh_target:
        raise HTTPException(400, f"{vlm_id}: no ssh target on cluster node")
    port = int(cd.get("_vlm_port") or VLM_DEFAULT_PORT)
    model_path = cd.get("_vlm_model_path") or cd.get("_telemak_loaded_model") or ""

    kill = _vlm_kill_cmd(port, model_path)
    kill_result = ""
    try:
        rc, out, err = await asyncio.to_thread(_ssh_exec, ssh_target, kill, 20)
        lines = (out or err or "").strip().splitlines()
        kill_result = lines[-1] if lines else "(no output)"
    except subprocess.TimeoutExpired:
        kill_result = "ssh kill timed out"
    except Exception as e:
        kill_result = f"ssh kill error: {e}"

    # Tombstone the cluster def — same {_removed: true} mechanism as
    # DELETE /admin/clusters/{id}. active_cluster_ids() / cluster_exists()
    # then filter it out.
    with _cluster_config_txn() as cfg:
        cfg[vlm_id] = {"_removed": True}
    _telemak_cache_invalidate(vlm_id)

    return {
        "status": "unloaded",
        "id": vlm_id,
        "ssh": ssh_target,
        "port": port,
        "kill_result": kill_result,
    }


@app.get("/admin/vlm/status")
async def admin_vlm_status():
    """List engine-managed VLM clusters + their /v1/models reachability."""
    out: list[dict] = []
    for cid in _vlm_managed_cluster_ids():
        cd = get_cluster_def(cid)
        upstream = (cd.get("upstream") or "").rstrip("/")
        nodes = cd.get("nodes") or []
        ssh_target = (nodes[0] or {}).get("ssh") if nodes else None
        host_id = (nodes[0] or {}).get("host") if nodes else None
        port = int(cd.get("_vlm_port") or VLM_DEFAULT_PORT)
        ip = _vlm_ip_from_ssh(ssh_target) if ssh_target else None
        reachable = await _vlm_probe_ready(ip, port) if ip else None
        out.append({
            "id": cid,
            "name": cd.get("name") or cid,
            "host": host_id,
            "ssh": ssh_target,
            "upstream": upstream,
            "port": port,
            "model": cd.get("_vlm_model_path") or cd.get("_telemak_loaded_model"),
            "pid": cd.get("_vlm_pid"),
            # True = serving, False = up but empty (loading/zombie), None = down
            "ready": reachable,
        })
    return {"vlms": out}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    global _args
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="If omitted, reload from state.json (or wait for /admin/load)")
    ap.add_argument("--mode", default="pipeline", choices=["pipeline", "tensor"])
    ap.add_argument("--use-ap", action="store_true")
    ap.add_argument("--nodes", type=int, default=2, choices=[2, 3, 4])
    ap.add_argument("--emit-batch", type=int, default=10)
    ap.add_argument("--host", default=os.environ.get("API_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("API_PORT", "8000")))
    _args = ap.parse_args()

    import uvicorn
    uvicorn.run(app, host=_args.host, port=_args.port, log_level="info")


if __name__ == "__main__":
    main()
