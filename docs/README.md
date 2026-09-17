# Odysseus docs — repo subset

This directory ships a small subset of the Odysseus documentation —
enough to install, deploy, and recover without a network connection.

The **canonical user-facing docs** live on the website at
[**odyssai.eu/docs**](https://odyssai.eu/docs/) (Starlight, with search,
prettier rendering, and last-updated metadata). When the two diverge,
the website is the source of truth.

## What's in this directory

| File | What it covers |
|---|---|
| [`GETTING-STARTED.md`](GETTING-STARTED.md) | Three install paths (single-node / multi-TCP / multi-JACCL) with concrete commands, prereq checks, and a smoke test. The long-form operator walkthrough. |
| [`DEPLOY.md`](DEPLOY.md) | Production deployment patterns — Docker host setup, hot-reload vs full rebuild, log access, persistence, discovery + pairing flow. |
| [`user-guide/`](user-guide/) | Operator guides that mirror the website: [multi-user serving](user-guide/multi-user-serving.md) (replica mode + continuous batching) and [CoeOS on this engine](user-guide/coeos.md). |
| [`API.md`](API.md) | Endpoints (`/v1/*`, `/admin/*`), cluster kinds, routing, capability contract. |
| [`bug-reports/`](bug-reports/) | Factual debugging notes for known upstream issues (`hy3` chat template, `mlx-vlm` `finish_reason` quirk, `mlx-vlm` streaming-tools deltas). |

## Where the rest lives

- [**odyssai.eu/docs/getting-started**](https://odyssai.eu/docs/getting-started/) — same as `GETTING-STARTED.md` here, prettier.
- [**odyssai.eu/docs/architecture**](https://odyssai.eu/docs/architecture/overview/) — the stack, the cluster, inference modes.
- [**odyssai.eu/docs/api**](https://odyssai.eu/docs/api/endpoints/) — endpoint reference + auth model.
- [**odyssai.eu/docs/operate**](https://odyssai.eu/docs/operate/deploy/) — deploy, cluster health, troubleshooting.
- [**CoeOS**](https://github.com/Odyssai-eu/coeos) — the gateway/router that sits on this engine; its user guide is [`doc/USER-GUIDE.md`](https://github.com/Odyssai-eu/coeos/blob/main/doc/USER-GUIDE.md) in that repo.

## How to install Odysseus

If you're here to install: read [`../AGENTS.md`](../AGENTS.md) at the
repo root. It's the runbook designed for AI coding agents to follow
top-to-bottom (and is also the most concise human reference).

## How to contribute docs

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md). User-facing docs are
authored on the website source (Astro/Starlight) and a curated subset
mirrors into this directory at release time. If you spot something
that should be in the repo but isn't, open an issue.
