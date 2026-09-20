# OdyssAI-X

> *Mobilis in Mobile*

**Distributed MLX inference engine for Apple Silicon clusters.** Built directly on `mlx` and `mlx-lm` — no `exo`, no extra orchestrator. Three ways to put Macs to work: **distributed** (one model split across nodes, JACCL/RDMA over Thunderbolt 5 or TCP), **replica** (N copies of one model, continuous batching, data-parallel throughput), and **VLM** (vision models served through `mlx-vlm`). OpenAI- and Anthropic-compatible HTTP on top; one control plane, many pools, many models at once.

OdyssAI-X is the **engine** layer of [**OdyssAI**](https://odyssai.eu), the open-source local AI ecosystem. The orchestrator *routes*, it never runs inference itself: it SSHes into the nodes to spawn long-lived runners, so a Mac mini can drive a rack of Mac Studios.

```
┌──────────────────────────────────────────────────────────────┐
│  Clients  (Claude Code · IDE agents · OpenAI/Anthropic SDKs  │
│            · any HTTP client — directly, or through CoeOS)   │
│         ↓  HTTP  ─  /v1/chat/completions  ·  /v1/messages    │
├──────────────────────────────────────────────────────────────┤
│  OdyssAI-X  (control plane + dashboard, this repo)  :8000    │
│         ↓  SSH  ─  starts long-lived runners per node        │
├──────────────────────────────────────────────────────────────┤
│  Nodes  (Apple Silicon, MLX + mlx-lm)                        │
│    distributed  ↔ JACCL/RDMA TB5  or  ring/TCP               │
│    replica      ═ N independent copies, batched              │
│    VLM          → mlx-vlm venv, proxied                      │
└──────────────────────────────────────────────────────────────┘
```

## What's in the box

- **OpenAI- and Anthropic-compatible HTTP API** — drop-in for anything that speaks `chat/completions` or `messages`, Claude Code included.
- **Replica mode (data-parallel)** — N single-node copies of a model, each with continuous batching, least-busy dispatch, session affinity (KV cache stays on the conversation's home replica), self-healing replicas, capacity guard. Add a Mac, gain throughput; no inter-node collective, so no node cap.
- **Distributed mode** — pipeline or tensor parallel (by the model's KV-head divisibility) over JACCL/RDMA Thunderbolt 5 or TCP ring, for models too big for one node.
- **Multi-pool, multi-model** — any number of clusters in `topology.yaml`, different models loaded side by side, all behind one `/v1`.
- **Extended `mlx-lm`** — vendored model modules (`scripts/mlx_models/`: DeepSeek V4, GLM-5 Next / GLM MoE DSA, Kimi K3, Qwen4-exp, Hy3/Hy4, Laguna, Inkling) plus runtime patches (`scripts/patches/`), installed on the nodes by the bootstrap script and kept in sync by `scripts/install-model-modules.sh` (`--check` reports per-file drift). Native MTP speculative decoding on supported families; per-model reasoning-tag handling (`reasoning_content` split).
- **VLM serving** — vision models through a dedicated `mlx-vlm` venv, fronted by the same API.
- **Ops that survive real use** — model-layout preflight, orphan/zombie runner sweep, dead-pool sweeper, unload guard, state restore across restarts, per-node capacity accounting.
- **Live admin dashboard** — clusters, pools, models, loads, Hugging Face download (incl. quant sub-folders), sync between nodes, logs.
- **Capability contract** — `/.well-known/inference-engine.json` and per-model `x_odyssai` blocks (vision, tools, stream, context length).

## OdyssAI — two components

OdyssAI is two pieces: **OdyssAI-X**, the engine, and **CoeOS**, the client — shipped as **Nemo**, the app, and the **CoeOS box**, the smart router behind it.

| Component | Repo | Role |
|---|---|---|
| **OdyssAI-X** (engine) | this repo | distributed / replica / VLM MLX inference on Apple Silicon; OpenAI + Anthropic API; dashboard. AGPL-3.0 |
| **Nemo** (the CoeOS client) | [Odyssai-eu/coeos](https://github.com/Odyssai-eu/coeos) | a desktop AI operating system for one person: chat with visible reasoning and personal memory, cowork on documents, code with a panel of agents; every turn shows which model served it. Signed, notarized macOS app ([releases](https://github.com/Odyssai-eu/coeos/releases)). MIT |
| **CoeOS box** (the smart router) | [Odyssai-eu/coeos-box](https://github.com/Odyssai-eu/coeos-box) | every request goes to the model proven best at that skill — local on the engine, or cloud with your own keys. Users, tokens, quotas, the CoeOS console. MIT |

Also in the organisation: [Guardian](https://github.com/Odyssai-eu/odyssai-guardian) (confidential-content detection before anything leaves for a cloud provider, MIT), [CodeOS](https://github.com/Odyssai-eu/CodeOS) (a 100 % coding app, a version of opencode that keeps following upstream; planner, coder, reviewer and sceptic on four different models routed by the CoeOS box, MIT), [odyssai-services](https://github.com/Odyssai-eu/odyssai-services) (bench + sidecar tooling), [mlx-swift-lm](https://github.com/Odyssai-eu/mlx-swift-lm) (MIT).

## Install

**macOS / Apple Silicon only.** Two roles: one **server** (the orchestrator container — a Mac mini is plenty) and one or more **nodes** (the Macs that hold the models; the server can be a node too). No installer app: a terminal, or an AI agent following [`AGENTS.md`](AGENTS.md), does the whole thing.

**Prerequisites**
- Apple Silicon Macs. Nodes: **Python 3.11** (`brew install python@3.11`); **Python 3.12** as well if the node should serve vision models (installed on demand). Server: **Docker Desktop**.
- **Remote Login** enabled on every node (System Settings → General → Sharing) and a passwordless SSH key from the server to each node (`ssh-copy-id`).
- Enough unified memory for the model you intend to serve (weights ≈ file size, plus ~20 % for KV cache and scratch). Disk on the nodes: models live **per node** under `<models_dir>/<org>/<name>/` — volumes are not shared.

**1. Bootstrap each node** — one command, idempotent:

```bash
scripts/bootstrap-node.sh user@node.lan            # default models dir: ~/mlx-models
scripts/bootstrap-node.sh user@node.lan /Volumes/models/odyssai
```

It checks SSH + Python, creates `~/mlx-cluster` with a pinned venv (`requirements-node.txt`: `mlx 0.32.2`, `mlx-lm 0.31.3`, `transformers 5.10.0`, …), copies the runner and patches, **installs the vendored model modules into the venv's `mlx_lm/models/`** and the **patched JACCL** (`vendor/jaccl/`, a drop-in `libjaccl.dylib` — see `vendor/jaccl/PATCHES.md`), runs a smoke import, and sets up the optional `mlx-vlm` venv (`~/.venvs/mlx-vlm`, Python 3.12). Re-run it after any `pip` upgrade on the node — upgrading `mlx-lm` silently removes the vendored modules, upgrading `mlx` puts the stock JACCL back.

**2. Pin the GPU memory budget on each node** (persistent `iogpu.wired_limit_mb`; needs the node's password, so it is a deliberate step, not automated):

```bash
scripts/wired-limit/install.sh user@node.lan 250880    # e.g. 245 GB on a 256 GB Mac
```

**3. Describe your cluster** on the server, then start the orchestrator:

```bash
mkdir -p ~/.odysseus && cp config/topology.example.yaml ~/.odysseus/topology.yaml
$EDITOR ~/.odysseus/topology.yaml      # ssh targets, models_dir, backend (ring | jaccl)
docker compose up -d
curl -s http://localhost:8000/health   # {"status":"idle","version":"…"}
```

Single Mac? Use `host.docker.internal` as the ssh target (the container can't reach the Mac through `localhost`) — the example topology shows it.

**4. First model.** Download from the dashboard (`http://localhost:8000/`, Models → Download; `org/name` or `org/name/<quant-subfolder>`), load it on a cluster, chat:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"<alias from /v1/models>","messages":[{"role":"user","content":"hi"}]}'
```

For throughput, create the cluster with **Kind = replica** in the dashboard (one entry per node holding the model) and load it with batching:

```bash
curl -s -X POST http://localhost:8000/admin/clusters/<id>/load -H 'content-type: application/json' \
  -d '{"model":"<org>/<name>","nodes":4,"batch":true}'
```

**Optional — RDMA over Thunderbolt 5** for distributed pools: cable the nodes in a full mesh, run `scripts/rdma-onboard.sh` on each (it provisions the Thunderbolt network the way `exo` proved it), declare `backend: jaccl` and the port wiring in `topology.yaml`. Every edge is checked before a load (port state, link-local alias, reachability through that cable) and a rank that dies mid-run is reported to the survivors in under a second instead of hanging them — see `AGENTS.md` §7 and `vendor/jaccl/PATCHES.md`. TCP `ring` needs none of this and always works.

Full walkthrough, node roles, budgets and gotchas: [`AGENTS.md`](AGENTS.md) and [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md).

## Documentation

- [`AGENTS.md`](AGENTS.md) — the install checklist an agent (or you) executes step by step, with the traps.
- [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md) — narrative guide: single node → TCP multi-node → RDMA.
- [`docs/API.md`](docs/API.md) — endpoints (`/v1/*`, `/admin/*`), cluster kinds, routing, capability contract.
- [`docs/DEPLOY.md`](docs/DEPLOY.md) — deploying code changes (container vs nodes); [`docs/RUNBOOK-argo-v4.md`](docs/RUNBOOK-argo-v4.md) — operating a JACCL cluster.
- [`docs/user-guide/`](docs/user-guide/) — multi-user serving (replica), CoeOS; [`docs/bug-reports/`](docs/bug-reports/) — upstream issues we hit and how.
- [Nemo](https://github.com/Odyssai-eu/coeos) — the CoeOS client that sits on this engine, and the [CoeOS box](https://github.com/Odyssai-eu/coeos-box), its smart router. Docs site: [odyssai.eu/docs](https://odyssai.eu/docs/).

## Status

**Pre-release, running in production internally** (six Macs, several clusters, replica + distributed + VLM pools side by side). The 1.x cycle is about operator onboarding on someone else's hardware. Licensed under the **GNU Affero General Public License v3.0** — see [LICENSE](LICENSE). Third-party components are attributed in [NOTICE](NOTICE).

## Contributing

Pull requests welcome — bug fixes, model support, capability blocks, performance work. See [CONTRIBUTING.md](CONTRIBUTING.md). Estimates are in Fibonacci scrum points, never time.

## Acknowledgments

Built on Apple's [MLX](https://github.com/ml-explore/mlx), [`mlx-lm`](https://github.com/ml-explore/mlx-lm) and MLX distributed (JACCL — vendored at `vendor/jaccl/` from MLX v0.32.2, MIT, with our patches listed in `vendor/jaccl/PATCHES.md`). The Thunderbolt/RDMA network recipe in `scripts/odyssai-network-setup.sh` is vendored from [exo](https://github.com/exo-explore/exo) (Apache-2.0, attribution in the file header). Vision serving uses [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm).
