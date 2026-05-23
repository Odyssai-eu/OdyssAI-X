# Odysseus

> *Mobilis in Mobile*

**Distributed MLX inference engine for Apple Silicon clusters.** Built directly on `mlx` and `mlx-lm` — no `exo`, no extra orchestrator. JACCL backend over Thunderbolt 5 RDMA for in-cluster traffic; OpenAI- and Anthropic-compatible HTTP for clients.

Part of [**OdyssAI**](https://odyssai.eu) — the open-source local AI ecosystem. Odysseus is the **engine** layer (this repo). Its sibling client is [**Companion**](https://github.com/Odyssai-eu/Companion).

```
┌─────────────────────────────────────────────────────────┐
│  Clients  (Companion · IDE agents · OpenAI/Anthropic     │
│            SDKs · any HTTP client)                       │
│         ↓  HTTP  ─  /v1/chat/completions                │
│         ↓        ─  /v1/messages                        │
├─────────────────────────────────────────────────────────┤
│  Odysseus  (control plane + dashboard, this repo)       │
│         ↓  SSH  ─  starts long-lived runners            │
├─────────────────────────────────────────────────────────┤
│  Cluster  (Apple Silicon nodes, MLX + mlx-lm)           │
│         ↔  JACCL / RDMA over Thunderbolt 5              │
└─────────────────────────────────────────────────────────┘
```

## What's in the box

- **OpenAI- and Anthropic-compatible HTTP API** — drop-in for any client that speaks `chat/completions` or `messages`.
- **Multi-pool orchestration** — declare any number of clusters in `topology.yaml` with the IDs you want (`default`, `chat`, `coder`, whatever fits), assign different models to each, load/unload from the dashboard.
- **Pipeline + tensor parallel** — use either depending on the model's KV-head divisibility. Pipeline-AP for big MoEs that JACCL's pipeline mode handles.
- **KV prefix cache** — `session_id`-based reuse across turns. Big TTFT wins on the same conversation.
- **Live admin dashboard** — runs, models, pool wiring, sync from Hugging Face, logs.
- **Capability contract** — `/.well-known/inference-engine.json` and per-model `x_odyssai` blocks so clients can introspect what's actually supported (vision, tools, stream, context length).

## Install

**Easiest path.** Open Claude Code, Codex, Cursor, or any other coding
agent in this folder and tell it what you have:

- *"I have N Macs, install the full cluster end-to-end"* →
  the agent reads [`INSTALL-CLUSTER.md`](INSTALL-CLUSTER.md), a 7-stage
  runbook that walks SSH bootstrap, node setup, RDMA discovery,
  orchestrator + Companion install, models dir, and first-model
  download — all in one session.
- *"Install Odysseus on this machine"* (single component) → the agent
  reads [`AGENTS.md`](AGENTS.md), which covers the 6 granular install
  patterns (orchestrator only / node only / single-Mac full stack /
  etc.) for when you already have some pieces in place.

Either way, the agent adapts to your OS, your topology, and your
existing tools.

**Manual path** (single-node — the same Mac plays both orchestrator and
cluster-node roles; Apple Silicon required for the cluster role):

> If your setup is **multi-node** (a separate orchestrator host + 1-N
> Apple Silicon nodes), use [`AGENTS.md`](AGENTS.md) instead. The
> orchestrator-only host does NOT need MLX — only the cluster nodes do.

```bash
git clone https://github.com/Odyssai-eu/Odysseus.git
cd Odysseus

# 1. Enable Remote Login on your Mac
#    System Settings → General → Sharing → Remote Login → ON
#    Then add your own SSH key for container-to-host auth:
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# 2. Bootstrap this Mac as a cluster node — i.e. install mlx + mlx-lm
#    + runner.py under ~/mlx-cluster/. Single-node only: the same Mac
#    plays both roles, so we bootstrap it. Skip this step on a separate
#    orchestrator host (multi-node deployments).
scripts/bootstrap-node.sh "$USER@localhost"

# 3. Configure topology — single-node example
mkdir -p ~/.odysseus
cp config/topology.example.yaml ~/.odysseus/topology.yaml
# edit ~/.odysseus/topology.yaml — set ssh: $USER@host.docker.internal

# 4. Start the engine + dashboard
docker compose up -d
open http://localhost:8000

# 5. Download a model into the models_dir on your Mac
pip install --user --upgrade huggingface_hub   # if not already installed
huggingface-cli download \
  mlx-community/Qwen3-7B-MLX-8bit \
  --local-dir ~/mlx-models/mlx-community/Qwen3-7B-MLX-8bit

# 6. Load + chat
#    /admin/* is open by default (LAN install). To require Bearer auth
#    on a publicly-reachable deployment, set ODYSSEUS_ADMIN_TOKEN in
#    your env and add `-H "Authorization: Bearer $ODYSSEUS_ADMIN_TOKEN"`.
curl -X POST http://localhost:8000/admin/default/load \
  -H 'Content-Type: application/json' \
  -d '{"model": "mlx-community/Qwen3-7B-MLX-8bit"}'

curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

For multi-node TCP and JACCL RDMA paths, see [`AGENTS.md`](AGENTS.md) or
the full operator walkthrough in
[`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md).

## Documentation

- [Companion](https://github.com/Odyssai-eu/Companion) — the recommended client
- Full docs site: [docs.odyssai.eu](https://odyssai.eu/docs/).

## Status

**Pre-release.** The engine is used internally in production but has rough edges around operator onboarding (cluster topology config, hardware discovery, first-time setup). The 0.x cycle stabilises those before a 1.0 cut.

Apache 2.0 licensed. See [LICENSE](LICENSE).

## Contributing

We welcome pull requests — bug fixes, model support, capability blocks, performance work. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and the development setup.

## Acknowledgments

Built on Apple's [MLX](https://github.com/ml-explore/mlx) and the [`mlx-lm`](https://github.com/ml-explore/mlx-lm) library. JACCL is part of `mlx-distributed`. Pipeline auto-parallel patterns informed by [exo](https://github.com/exo-explore/exo)'s lifecycle work.
