# OdyssAI-X

> **Built on MLX, not waiting for it.**

**OdyssAI-X serves large language and vision models across a cluster of Apple Silicon Macs, behind one OpenAI- and Anthropic-compatible API.**

**97.6%** for GLM-5.3 (Q8, max effort) served locally by OdyssAI-X: the best local model on the [TMB scoreboard](https://themonoclebear.com/en/scoreboard/), 3rd of 81 ranked, 0.2 pt behind Fable 5 (97.8%) and ahead of Claude Opus 4.8 (97.0%) (2026-09-28).

## Install

On each Mac that will hold models (a node), in Terminal:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Odyssai-eu/OdyssAI-X/main/install.sh)"
```

Nothing to install first: it brings its own Python (uv), the pinned MLX, the patched JACCL and the GPU memory setting, times each stage and ends with `odyssai-x doctor`, one line per check with the fix when something is off. Run it again any time: `already up to date`. Or hand this repo to your coding agent: it reads [`AGENTS.md`](AGENTS.md). Then [start the server](#other-install-methods) (step 3 onwards).

## Model support

| Family (mode) | Status | Last verified |
|---|---|---|
| GLM-5.3 744B (distributed, 4-5 nodes, RDMA) | `stable` | 2026-09-20 |
| MiMo-V2.6 (vision, replica) | `stable` | 2026-09-26 |
| Qwen3.8-Flash-Next (replica, distributed) | `runs with caveats` | 2026-09-26 |
| Gemma 4 (single node, vision) | `runs with caveats` | 2026-09-26 |
| DeepSeek V4 / V4.1 Flash (single node, pipeline) | `runs with caveats` | 2026-09-11 |
| MiniMax M2.7 (replica) | `runs with caveats` | 2026-09-06 |
| GLM-5.3-Flash (single node, MTP) | `runs with caveats` | 2026-08-26 |
| Qwen3.5 / Qwen3.5-MoE 397B (single node; vision distributed) | `runs with caveats` | 2026-08-26 |
| Hunyuan Hy3 (single node) | `runs with caveats` | 2026-08-13 |
| Kimi K3 (distributed) | `unsupported` | 2026-07-28 |
| LongCat 2.0 (pipeline) | `unsupported` | 2026-07-07 |

`stable`: served in production with a passing run since 2026-09-01 and no open blocking bug. `runs with caveats`: served, with older evidence or a known limit. `unsupported`: tried, did not hold. Other architectures the loader accepts (Llama, GLM-4, DeepSeek V3, gpt-oss, Nemotron, Kimi K2) are unverified on the current fleet.

OdyssAI-X is open source. **Deploying it in your company?** Contact us: [odyssai.eu](https://odyssai.eu).

## Coming from exo

- **Same:** Apple Silicon Macs, MLX, pipeline and tensor parallel over Thunderbolt 5 RDMA (JACCL) or TCP, an OpenAI-compatible API, a web dashboard.
- **Different:**
  - `odyssai-x doctor` checks every node and every RDMA cable from both ends, and names the failing one (node, port, fix in one sentence); the same edge check runs before every load.
  - A rank that dies mid-run is reported to every survivor in 0.25 s (measured on process death), instead of a silent hang.
  - No auto-join: a Mac serves only once it is declared in `topology.yaml`.
  - Model families adapted to MLX in this repo: DeepSeek V4, GLM-5.3-Flash, Qwen3.8-Flash-Next, Hunyuan Hy3, Inkling (`scripts/mlx_models/`).
- **Not covered:** CUDA or Linux nodes, an iOS app, bandwidth-based placement, a radix (prefix-tree) KV cache, auto-join (a discovered node joins a cluster when you add it).

## What it is

Three ways to put Macs to work: **distributed** (one model split across nodes, JACCL/RDMA over Thunderbolt 5 or TCP), **replica** (N copies of one model, continuous batching, data-parallel throughput), and **VLM** (vision models served through `mlx-vlm`). Built directly on `mlx` and `mlx-lm`. One control plane, many pools, many models at once.

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

Also in the organisation: [Guardian](https://github.com/Odyssai-eu/odyssai-guardian) (confidential-content detection before anything leaves for a cloud provider, MIT), [odyssai-services](https://github.com/Odyssai-eu/odyssai-services) (bench + sidecar tooling), [mlx-swift-lm](https://github.com/Odyssai-eu/mlx-swift-lm) (MIT).

## Other install methods

**macOS / Apple Silicon only.** Two roles: one **server** (the orchestrator container — a Mac mini is plenty) and one or more **nodes** (the Macs that hold the models; the server can be a node too). [`AGENTS.md`](AGENTS.md) is the full step-by-step flow, with the expected output and the failure branch of every step; this section is its manual path, driven from a checkout on the server.

**Prerequisites**
- Apple Silicon Macs. Nodes installed with `install.sh` need nothing else. For this manual path: **Python 3.11** on each node (`brew install python@3.11`); **Python 3.12** as well if the node should serve vision models (installed on demand). Server: **Docker Desktop**.
- **Remote Login** enabled on every node (System Settings → General → Sharing) and a passwordless SSH key from the server to each node (`ssh-copy-id`).
- Enough unified memory for the model you intend to serve (weights ≈ file size, plus ~20 % for KV cache and scratch). Disk on the nodes: models live **per node** under `<models_dir>/<org>/<name>/` — volumes are not shared.

**1. Bootstrap each node** — one command, idempotent:

```bash
scripts/bootstrap-node.sh admin@node-a.local            # default models dir: ~/mlx-models
scripts/bootstrap-node.sh admin@node-a.local /Volumes/models/odyssai
```

It checks SSH + Python, creates `~/mlx-cluster` with a pinned venv (`requirements-node.txt`: `mlx 0.32.2`, `mlx-lm 0.31.3`, `transformers 5.10.0`, …), copies the runner and patches, **installs the vendored model modules into the venv's `mlx_lm/models/`** and the **patched JACCL** (`vendor/jaccl/`, a drop-in `libjaccl.dylib` — see `vendor/jaccl/PATCHES.md`), runs a smoke import, and sets up the optional `mlx-vlm` venv (`~/.venvs/mlx-vlm`, Python 3.12). Re-run it after any `pip` upgrade on the node — upgrading `mlx-lm` silently removes the vendored modules, upgrading `mlx` puts the stock JACCL back.

**2. Pin the GPU memory budget on each node** (persistent `iogpu.wired_limit_mb`; `install.sh` does it, this is the manual equivalent and needs the node's password):

```bash
scripts/wired-limit/install.sh admin@node-a.local 250880    # e.g. 245 GB on a 256 GB Mac
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

**Optional — RDMA over Thunderbolt 5** for distributed pools: enable RDMA once per node in recoveryOS (`rdma_ctl enable`), cable the nodes in a full mesh, run `sudo scripts/rdma-onboard.sh --apply --console` at each node's console (it provisions the Thunderbolt network; attribution in `NOTICE`), map the ports with `scripts/discover-rdma-wiring.py`, and declare `backend: jaccl` and the port wiring in `topology.yaml`. Every edge is checked before a load (port state, link-local alias, reachability through that cable) and a rank that dies mid-run is reported to the survivors in under a second instead of hanging them — see `AGENTS.md` section E and `vendor/jaccl/PATCHES.md`. TCP `ring` needs none of this and always works.

Narrative guide, node roles, budgets and gotchas: [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md).

## Documentation

- [`AGENTS.md`](AGENTS.md) — the install flow an agent (or you) executes step by step, with the expected output and the failure branch of each step.
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

Built on Apple's [MLX](https://github.com/ml-explore/mlx), [`mlx-lm`](https://github.com/ml-explore/mlx-lm) and MLX distributed (JACCL — vendored at `vendor/jaccl/` from MLX v0.32.2, MIT, with our patches listed in `vendor/jaccl/PATCHES.md`). The pipeline/tensor sharding in `scripts/auto_parallel.py`, two `mlx-lm` runtime patches and the Thunderbolt/RDMA network recipe in `scripts/odyssai-network-setup.sh` are derived from [exo](https://github.com/exo-explore/exo) (Apache-2.0; per-file headers, file list in [NOTICE](NOTICE), provenance in `vendor/exo/UPSTREAM.md`). Vision serving uses [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm).
