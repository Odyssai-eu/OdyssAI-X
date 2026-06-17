# AGENTS.md — OdyssAI-X (project orientation)

> Orientation for an agent (or human) landing in this repo: **what it is, where
> everything lives, how to navigate.** This is *not* an install runbook — to
> install the stack, see [`INSTALL-CLUSTER.md`](INSTALL-CLUSTER.md). For internal
> dev conventions and the live cluster state, see [`CLAUDE.md`](CLAUDE.md).

---

## What OdyssAI-X is

The **engine** layer of the **odyssai.eu** ecosystem: a distributed MLX inference
engine for Apple Silicon clusters, built directly on `mlx` / `mlx-lm` (no `exo`).
It exposes an OpenAI- and Anthropic-compatible HTTP API and an admin dashboard.

**Key principle:** the orchestrator *routes*, it never runs inference itself — it
dispatches to MLX cluster nodes (JACCL/RDMA over Thunderbolt 5) **or** proxies to
a single-Mac Telemak runtime. That is why a cheap Mac mini can drive expensive
Mac Studios.

```
Clients (Companion · IDE agents · OpenAI/Anthropic SDKs)
   ↓ HTTP  /v1/chat/completions · /v1/messages
OdyssAI-X  (control plane + dashboard, THIS repo)   :8000
   ↓ SSH (starts long-lived runners)        ↘ http-proxy
Cluster (Apple Silicon nodes, MLX)            Telemak (single Mac)
   ↔ JACCL / RDMA over Thunderbolt 5
```

---

## The odyssai.eu family — where everything lives

| Component | Repo | Role |
|---|---|---|
| **OdyssAI-X** (engine) | this repo — `github.com/Odyssai-eu/Odysseus` | distributed MLX inference, OpenAI/Anthropic API, admin dashboard |
| **Companion** (client) | `github.com/Odyssai-eu/Companion` | React chat client + memory (consumes this engine) |
| **Telemak** (runtime) | `github.com/Odyssai-eu/telemak` | native single-Mac Swift runtime (Solo mode upstream) |
| **OdyssAI Configurator** (installer) | `github.com/Odyssai-eu/Odyssai-config` | macOS DMG that installs all three + builds the RDMA topology |

---

## Install

Don't install by hand. Use the **OdyssAI Configurator** (macOS DMG, drag-to-
Applications, no Developer ID — build local or clear quarantine).
[`INSTALL-CLUSTER.md`](INSTALL-CLUSTER.md) covers all three paths:
the **GUI wizard** (human, primary), the **`odyssai-configure` CLI** (agents /
scripted macOS — §12), and the **raw scripts** (Linux orchestrator / CI / headless
— appendix).

---

## Repo map — where things are

**`scripts/`** — the engine itself:
- `api.py` — the OdyssAI-X orchestrator (FastAPI): `/v1/*`, `/admin/*`, serves the dashboard, cluster/pool/model lifecycle.
- `dashboard.html` — the admin SPA, served fresh per request by `api.py` at `/` (hot-deployable via `docker cp`, no restart).
- `runner.py` — the long-lived per-node MLX runner, launched over SSH on each cluster node.
- `master.py`, `inference.py`, `inference_pipe.py`, `auto_parallel.py`, `persistence.py` — distributed inference plumbing (pipeline/tensor parallel, KV cache).
- `topology.py` — `TopologyConfig`, the schema + validator for `~/.odysseus/topology.yaml`.
- `bootstrap-node.sh`, `discover-rdma-wiring.py`, `provision-node-local.sh`, `odyssai-network-setup.sh`, `rdma-onboard.sh`, `lean-node.sh` — node provisioning + RDMA wiring discovery (what the Configurator wraps).
- `*_convert.py` (`m3_convert.py`, `glm_dsa_convert.py`, `mistral_convert.py`, …) — model conversion helpers.

**Other top-level:**
- `config/topology.example.yaml` — the topology template (copy it, never write from scratch).
- `Dockerfile` + `docker-compose.yml` — the orchestrator container (`odyssai-odysseus`).
- `docs/` — `ABOUT.md` (product), `API.md` (endpoints), `DEPLOY.md` (deploy patterns), `PRODUCTION.md`, `SECURITY-POSTURE.md`, `HARDWARE.md`, `GETTING-STARTED.md`, and `docs/sessions/` (the narrative SESSION log).
- `CLAUDE.md` — internal dev guide: live cluster state, conventions, gotchas.
- `INSTALL-CLUSTER.md` — the install doc (GUI / CLI / headless).

---

## Runtime shape (quick reference)

- Container **`odyssai-odysseus`** on **`:8000`**; dashboard at `/`.
- **`/v1/*`** (chat/completions, messages, models) is always **public**.
- **`/admin/*`** is **open by default** (trusted-LAN); set `ODYSSAI_X_ADMIN_TOKEN`
  (legacy alias `ODYSSEUS_ADMIN_TOKEN`) to require `Authorization: Bearer`.
- Topology lives at **`~/.odysseus/topology.yaml`** (clusters → pools → nodes;
  `jaccl` needs `rdma_to`, `http-proxy` needs `upstream`).
- Backends: `ring` (TCP, always works) · `jaccl` (RDMA TB5, ~2× faster, queue-pair
  bug after long sessions → reboot nodes). Sharding: tensor (KV-heads ÷ world_size)
  · pipeline (big MoEs).

---

## Conventions (full version in `CLAUDE.md`)

- Estimate in **Fibonacci scrum points**, never time units.
- **Direct-push to `main`** is the default; deploy follows the commit (server == main).
- Conventional Commits + HEREDOC, never `--no-verify`, no emojis in code/commits.
- **OdyssAI-X is the inference rail** (gateway / `engine_url`). **LiteLLM is a legacy
  fallback** — never present it as central.

## What you should NOT do

- **Don't run inference logic from the orchestrator** — it routes only; the engine runs on the nodes.
- **Don't hand-write `~/.odysseus/topology.yaml`** — use `odyssai-configure topology build` (or copy the example).
- **Don't rename physical model paths** (`/Volumes/models/…`) when rebranding — they are real volumes on the nodes; renaming breaks model loading.
- **Don't modify `runner.py` / `api.py` / the Dockerfile** as part of an install — they are release artefacts.
