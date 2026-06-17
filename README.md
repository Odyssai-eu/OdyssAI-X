# OdyssAI-X

> *Mobilis in Mobile*

**Distributed MLX inference engine for Apple Silicon clusters.** Built directly on `mlx` and `mlx-lm` — no `exo`, no extra orchestrator. JACCL backend over Thunderbolt 5 RDMA for in-cluster traffic; OpenAI- and Anthropic-compatible HTTP for clients.

OdyssAI-X is the **engine** layer of [**OdyssAI**](https://odyssai.eu) — the open-source local AI ecosystem. The orchestrator *routes*, it never runs inference itself: it dispatches to MLX cluster nodes **or** proxies to a single-Mac Telemak runtime, so a cheap Mac mini can drive expensive Mac Studios.

```
┌─────────────────────────────────────────────────────────┐
│  Clients  (Companion · IDE agents · OpenAI/Anthropic     │
│            SDKs · any HTTP client)                       │
│         ↓  HTTP  ─  /v1/chat/completions                │
│         ↓        ─  /v1/messages                        │
├─────────────────────────────────────────────────────────┤
│  OdyssAI-X  (control plane + dashboard, this repo)      │
│         ↓  SSH  ─  starts long-lived runners            │
├─────────────────────────────────────────────────────────┤
│  Cluster  (Apple Silicon nodes, MLX + mlx-lm)           │
│         ↔  JACCL / RDMA over Thunderbolt 5              │
└─────────────────────────────────────────────────────────┘
```

## What's in the box

- **OpenAI- and Anthropic-compatible HTTP API** — drop-in for any client that speaks `chat/completions` or `messages`.
- **Multi-pool orchestration** — declare any number of clusters in `topology.yaml` with the IDs you want (`default`, `chat`, `coder`, …), assign different models to each, load/unload from the dashboard.
- **Pipeline + tensor parallel** — either, depending on the model's KV-head divisibility. Pipeline-AP for big MoEs that JACCL's pipeline mode handles.
- **KV prefix cache** — `session_id`-based reuse across turns. Big TTFT wins on the same conversation.
- **Live admin dashboard** — runs, models, pool wiring, sync from Hugging Face, logs.
- **Capability contract** — `/.well-known/inference-engine.json` and per-model `x_odyssai` blocks so clients can introspect what's actually supported (vision, tools, stream, context length).

## The OdyssAI family — where everything lives

| Component | Repo | Role |
|---|---|---|
| **OdyssAI-X** (engine) | this repo | distributed MLX inference, OpenAI/Anthropic API, admin dashboard |
| **Companion** (client) | [Odyssai-eu/Companion](https://github.com/Odyssai-eu/Companion) | React chat client + memory; consumes this engine |
| **Telemak** (runtime) | [Odyssai-eu/telemak](https://github.com/Odyssai-eu/telemak) | native single-Mac Swift runtime (Solo-mode upstream) |
| **OdyssAI Configurator** (installer) | [Odyssai-eu/Odyssai-config](https://github.com/Odyssai-eu/Odyssai-config) | macOS DMG that installs all three + builds the RDMA topology |

## Install

**Use the [OdyssAI Configurator](https://github.com/Odyssai-eu/Odyssai-config)** — a
native macOS app (DMG, drag-to-Applications, no terminal). A wizard installs each
component on the right machine and builds the RDMA topology for you:

- **Engine** → a Mac Studio (Apple Silicon)
- **Serveur** → a Mac mini (Apple Silicon): OdyssAI-X + Companion
- **Telemak** → any one Mac (Solo mode)

in two modes — **Cluster** (Serveur + N Mac Studio over RDMA/TB5) or **Solo**
(Serveur + 1 Telemak Mac). The Configurator isn't signed with an Apple Developer
ID (open-source, outside the App Store): build it locally, or clear the
quarantine bit on a downloaded DMG — see its README.

Prefer the terminal, or installing with an AI agent? [`INSTALL-CLUSTER.md`](INSTALL-CLUSTER.md)
covers the GUI wizard, the `odyssai-configure` CLI, and a headless/scripted path
(Linux orchestrator / CI). For project orientation — what lives where —
see [`AGENTS.md`](AGENTS.md).

## Documentation

- [`AGENTS.md`](AGENTS.md) — project orientation (what OdyssAI-X is, where everything lives).
- [`INSTALL-CLUSTER.md`](INSTALL-CLUSTER.md) — install (GUI / CLI / headless).
- [`docs/`](docs/) — `ABOUT.md`, `API.md`, `DEPLOY.md`, `PRODUCTION.md`, `SECURITY-POSTURE.md`.
- [Companion](https://github.com/Odyssai-eu/Companion) — the recommended client.
- Full docs site: [odyssai.eu/docs](https://odyssai.eu/docs/).

## Status

**Pre-release.** The engine runs internally in production; the 0.x cycle stabilises operator onboarding (topology config, hardware discovery, first-time setup) ahead of a 1.0 cut. Apache 2.0 licensed — see [LICENSE](LICENSE).

## Contributing

Pull requests welcome — bug fixes, model support, capability blocks, performance work. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and dev setup.

## Acknowledgments

Built on Apple's [MLX](https://github.com/ml-explore/mlx) and [`mlx-lm`](https://github.com/ml-explore/mlx-lm). JACCL is part of `mlx-distributed`. Pipeline auto-parallel patterns informed by [exo](https://github.com/exo-explore/exo)'s lifecycle work.
