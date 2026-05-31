# AGENTS.md — Odysseus

> This file is a runbook for AI coding agents (Claude Code, Codex, Cursor,
> Aider, …) installing Odysseus on the user's machine. It is meant to be
> executed top-to-bottom. The user has already cloned this repo and pointed
> their agent at it.
>
> If you are a human, you can read it too — it just won't apologize for
> being literal.
>
> **Building a full multi-node cluster end-to-end?** Use
> [`INSTALL-CLUSTER.md`](INSTALL-CLUSTER.md) instead — it's a 7-stage
> runbook that drives SSH bootstrap, node setup, RDMA discovery,
> orchestrator + Companion install, models dir, and first-model
> download in one session. This `AGENTS.md` covers the 6 granular
> install patterns for when only some pieces are needed.

## 0. What you are about to install

**Odysseus** is a distributed MLX inference engine for Apple Silicon
clusters. One Mac runs the orchestrator (a Docker container); 1–N Macs in
the same LAN host the actual MLX runners. Clients talk to the orchestrator
over the standard OpenAI (`/v1/chat/completions`) and Anthropic
(`/v1/messages`) HTTP surfaces.

Concretely, after a successful install you have:

- a single container `odyssai-odysseus` listening on `:8000`
- a dashboard at `http://<docker-host>:8000/`
- one or more *clusters* of Macs publishing aliases (e.g. `my-cluster`)
  that resolve to MLX models loaded on demand
- an OpenAI- and Anthropic-compatible API any client can hit

What makes it different from "just run a model on a Mac":

| | Ollama / LM Studio | Odysseus |
|---|---|---|
| Multi-Mac | single host | 1–N Macs, model sharded across them |
| Largest model | RAM of 1 Mac | sum of cluster RAM (up to ~2 TB on 4× M3 Ultra 512 GB) |
| Sharding | n/a | pipeline parallel + tensor parallel, per-pool |
| Cluster fabric | n/a | JACCL RDMA over Thunderbolt 5 OR TCP ring |
| Cloud passthrough | usually no | yes (`or:*`, `anthropic/*`, `openai/*` aliases in the same catalogue) |
| Cross-protocol translation | no | yes — OpenAI clients can hit Anthropic upstreams transparently |
| Standards adhered to | OpenAI chat | OpenAI chat + completions + Anthropic Messages + capability contract (`/.well-known/inference-engine.json`) |

If a model fits on a single Apple Silicon Mac, Ollama and LM Studio are
simpler — pick them. Odysseus is for the workload where one Mac is not
enough, or where the user wants OpenAI+Anthropic surfaces with cloud
passthrough behind a single endpoint.

## Three components, six install patterns

> **STOP — agents read this first.**
>
> Do NOT pick a pattern yourself. Do NOT run `bootstrap-node.sh`, do
> NOT `docker compose up`, do NOT clone Companion until the user has
> told you, **in chat**, which of the six patterns below applies. If
> they said only "install Odysseus" without specifying, the *only*
> correct first action is to show them the table below and ask. Picking
> Pattern 1 because the machine *could* run everything is wrong — many
> operators have the orchestrator on a small box driving cluster nodes
> that already exist, and would not appreciate a fresh MLX venv being
> installed locally.

The Odyssai stack has three independent components. Pick what to
install on the machine you're working with.

| Component | What it runs | Hardware | Install entry |
|---|---|---|---|
| **node** | MLX runtime + `runner.py` for one rank | Apple Silicon Mac (arm64) | `scripts/bootstrap-node.sh` |
| **orchestrator** | The Docker container (`odyssai-odysseus`) — API, dashboard, SSHes to nodes | Any OS with Docker | `docker compose up -d` from this repo |
| **client** | Companion (chat UI, memory, projects, MCP) | Any OS with Docker | **Separate repo** — `https://github.com/Odyssai-eu/Companion`, follow its `AGENTS.md` |

Six install patterns map onto sub-steps below. Read the **"What's on
this machine"** column carefully — it tells you what to install
*here*. Anything else lives on a different machine.

| # | Pattern | What's on this machine | When to pick it | Sub-steps from this AGENTS.md |
|---|---|---|---|---|
| 1 | **Full single-Mac** | node + orchestrator + client | One Apple Silicon Mac, the user wants everything in one place to try things out. | 2a + 2b → 3 (bootstrap localhost) → 4 → 5 → then clone Companion + its AGENTS.md |
| 2 | **Orchestrator + client** | orchestrator + client (nodes are on OTHER machines) | A small box (Mac mini, Linux) drives existing cluster nodes and also serves Companion. | 2a → 4 → 5 → then clone Companion + its AGENTS.md |
| 3 | **Orchestrator only** | orchestrator (no node, no client here) | The orchestrator host in a setup where the chat client lives elsewhere (a teammate's machine, a server, no client at all because you'll use the API directly). Nodes still live on other machines — bootstrap them with pattern 6. | 2a → 4 → 5 |
| 4 | **Client only** | client | The orchestrator is already running somewhere (or the user is going to point Companion at Ollama / LM Studio / cloud). | nothing in *this* repo — clone Companion directly and run its AGENTS.md |
| 5 | **Orchestrator + local node** | orchestrator + node (no client here) | One Mac that both orchestrates and runs a runner; the chat client lives elsewhere. | 2a + 2b → 3 (bootstrap localhost) → 4 → 5 |
| 6 | **Pure node** | node only (orchestrator is on another machine) | One of N Apple Silicon Macs in a cluster — every cluster node needs this exact install. | 2b → 3 (bootstrap from the orchestrator host or your laptop) |

> The labels are deliberately about **what runs here**, not about what
> the full deployment looks like. Pattern 3 ("orchestrator only") is a
> common valid choice — the user installs the engine on the box that
> will drive the cluster, then walks pattern 6 on each cluster node.
> Both halves are needed for a working setup; this AGENTS.md just
> tells you what to do on the host you're currently on.

Ask the user which pattern applies before going further. If they want
a multi-node cluster (patterns 2, 3, 5), also ask which inter-node
transport — that decides the `backend:` field in `topology.yaml`:

| Transport | Hardware needed | When |
|---|---|---|
| **Single-node** (no transport) | One Apple Silicon Mac (orchestrator + node = same machine) | Default for pattern 1. No SSH-between-Macs, no RDMA. |
| **Multi-node TCP (`ring` backend)** | 2–N Apple Silicon nodes on the same LAN | The user has more than one Mac as nodes but no Thunderbolt 5 cabling between them. Slower than RDMA but works on any network. |
| **Multi-node JACCL (RDMA over TB5)** | 2–N Apple Silicon nodes with full-mesh TB5 cabling | Production path. Faster (~2× on big MoEs). Requires the operator to cable nodes and discover the interface names. |

You can always start with `ring` and migrate to `jaccl` later by
editing `topology.yaml`.

## 2. Prerequisites — check the right hosts for the right things

The prereqs depend on the role each machine plays. Run each check on
the appropriate host; stop and report what's missing, don't silently
install missing dependencies.

### 2a. Orchestrator host (where the Docker container will run)

The orchestrator needs Docker and SSH-out capability. It does **not**
need MLX, Python ML libs, or Apple Silicon — an Intel Mac mini, a
Linux box with Docker, or even an Apple Silicon Mac all work.

```bash
# Docker (Desktop on macOS, daemon on Linux)
docker version >/dev/null 2>&1 || echo "ERROR: Docker not installed or not running."

# SSH client (to reach cluster nodes)
which ssh >/dev/null || echo "ERROR: SSH client missing."

# Disk for the Docker image (~500 MB) + the state volume (~100 MB).
df -h /
```

If single-node (orchestrator + node are the same Mac), also run the
node checks below on the same host.

### 2b. Cluster node (each Apple Silicon Mac that will run MLX)

Run these on **every** node you'll add to `topology.yaml`. The
orchestrator host SSHes here and spawns `runner.py` per rank.

```bash
# Apple Silicon required for MLX
[ "$(uname -m)" = "arm64" ] || echo "ERROR: cluster nodes must be Apple Silicon (arm64)."

# macOS 14+ (Sonoma) recommended
sw_vers -productVersion

# Python 3.11+ — bootstrap-node.sh creates the venv with this
python3.11 --version 2>/dev/null || python3 --version || echo "ERROR: Python 3.11+ required."

# Remote Login enabled? (System Settings → General → Sharing → Remote Login)
# Test by SSHing from the orchestrator host:
#   ssh -o BatchMode=yes user@<this-node> hostname

# Enough disk for the venv (~5 GB after mlx + mlx-lm install) + your
# models. Plan ~1.2× the model file size for KV cache + runtime overhead.
df -h /Volumes 2>/dev/null || df -h /
```

**Disk + RAM budget — pick the model class that fits the cluster total:**

| Pool size | Model class it fits (9-bit MLX) | Suggested unified memory total |
|---|---|---|
| 1 node, 32–64 GB | 7-13 B (Qwen3-7B, Llama-3-8B) | 32 GB |
| 1 node, 96 GB | 30-40 B chat / 35 B vision | 96 GB |
| 2-3 nodes, ≥256 GB total | 100 B+ MoE (Qwen3-Coder-Next, Hy3-preview) | 256 GB cumulative |
| 3-4 nodes, ≥512 GB total | 400 B+ (Qwen3.5-397B, Qwen3-Coder-480B) | 512 GB cumulative |

**Single-node specific — important gotcha:**

The orchestrator runs **inside a Docker container** and SSHes out to
cluster nodes. Even on a single-Mac install, the container treats the
Mac host as a "remote node" reached over SSH — `${USER}@localhost`
won't work because `localhost` inside the container is the container
itself, not the Mac.

For single-node, the user must:

1. **Enable Remote Login** on the Mac:
   *System Settings → General → Sharing → Remote Login → ON*
   (sets up sshd on port 22).

2. **Allow SSH key-based auth from the container to the host**:
   ```bash
   # On the Mac host:
   [ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
   cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys

   # Test from the container (after `docker compose up`):
   docker exec odyssai-odysseus ssh -o StrictHostKeyChecking=accept-new \
     "$USER@host.docker.internal" hostname
   ```

3. **Use `host.docker.internal` as the SSH target in `topology.yaml`**:
   ```yaml
   nodes:
     - rank: 0
       ssh: ${ODYSSEUS_NODE_USER}@host.docker.internal
   ```

   `${ODYSSEUS_NODE_USER}` is set by `docker-compose.yml` from the host
   environment (`$USER` by default). Do NOT use `${USER}` directly —
   topology.yaml is parsed inside the container, where the process
   runs as root.

   If you start the orchestrator from a context where `$USER` isn't
   set (cron, systemd, CI), export `ODYSSEUS_NODE_USER` explicitly:
   ```bash
   ODYSSEUS_NODE_USER=alice docker compose up -d
   ```

`host.docker.internal` is Docker's built-in alias for the host running
the daemon. The `extra_hosts: ["host.docker.internal:host-gateway"]`
block in `docker-compose.yml` keeps it working on Linux too.

**Multi-node — additional checks:**

```bash
# SSH key-based auth from the orchestrator host to every cluster node.
# Test for each node target the user gives:
ssh -o BatchMode=yes user@<node-host> hostname

# Models directory at the same path on every node — created by the
# bootstrap step below.
```

**JACCL only — additional checks:**

```bash
# On each node, list the PORT_ACTIVE RDMA HCAs JACCL sees. There
# should be one PORT_ACTIVE entry per TB5 cable plugged in.
ssh user@<node-host> 'ibv_devinfo 2>&1 | awk "/^hca_id/{n=\$2} /state:/{print n,\$2}"'
```

For the actual cabling map (which HCA reaches which peer), use the
discovery script in step 4 — don't try to fill `rdma_to` by hand.

If any of the above fail, stop and tell the user *which* check failed and
*how* to fix it (install Docker Desktop, enable Remote Login, run
`ssh-copy-id`, plug TB5 cable in the right port, etc.). Do not work
around prerequisites silently.

## 3. Bootstrap each cluster node — NOT the orchestrator host

> **Critical:** run `bootstrap-node.sh` ONLY against machines that are
> cluster nodes (the ones that will actually run MLX). Do NOT run it
> against an orchestrator-only host — the orchestrator never executes
> MLX, so it has no use for `~/mlx-cluster/.venv` and no business
> downloading `mlx` + `mlx-lm`.
>
> Single-node deploy: the same Mac plays both roles, so yes, bootstrap
> it. Multi-node deploy with a separate orchestrator (Mac mini, Linux
> box, …): bootstrap the cluster nodes only, leave the orchestrator
> alone.

Every node the orchestrator talks to must have an MLX runtime at a
known path. The orchestrator expects:

  - `~/mlx-cluster/runner.py` — the runner script
  - `~/mlx-cluster/.venv/bin/python` — a Python 3.11+ venv with
    `mlx` + `mlx-lm` installed
  - a few helper modules (`auto_parallel.py`, `exo_stubs.py`, `patches/`)

The `scripts/bootstrap-node.sh` script puts those in place. Run it once
per node — typically from the orchestrator host (or from your laptop
if you have SSH to the nodes from there):

```bash
# Single-node deploy: the orchestrator Mac is also the node. SSH back
# to itself via host.docker.internal once the container is up; for the
# bootstrap step we can use localhost from the shell.
scripts/bootstrap-node.sh "$USER@localhost"

# Multi-node deploy: only the cluster nodes, NOT the orchestrator host.
scripts/bootstrap-node.sh user@host-a.lan
scripts/bootstrap-node.sh user@host-b.lan
scripts/bootstrap-node.sh user@host-c.lan

# Optional second arg: per-node models dir (default ~/mlx-models)
scripts/bootstrap-node.sh user@host-a.lan /Volumes/external/models
```

The script is idempotent — re-running it on a node just re-syncs the
scripts and re-checks the venv. It verifies the node is Apple Silicon,
creates `~/mlx-cluster/`, sets up the venv, installs `mlx` + `mlx-lm`,
and runs a smoke import.

If the user has clusters already configured (existing Macs they've
been using for ML), the bootstrap is still required — it lays down the
exact runner.py + venv layout the orchestrator expects.

## 4. Configure the cluster topology

Odysseus reads `~/.odysseus/topology.yaml` at container boot. You create
it before starting the engine.

```bash
mkdir -p ~/.odysseus
cp config/topology.example.yaml ~/.odysseus/topology.yaml
```

The example ships with a single-node cluster keyed `default`.

**Cluster IDs are arbitrary.** The keys under `clusters:` in
`topology.yaml` are free-form identifiers chosen by the operator — they
appear in admin routes (`/admin/clusters/<id>/…`) and in the dashboard.
Pick short, role-descriptive names — `default`, `chat`, `coder`,
`reasoner`, `vision`, `mon-mac`, or any free-form identifier (the
engine treats them as opaque strings, doesn't bake any specific name
in). **Ask the user** what to call their cluster(s) before editing
the file. The model *alias* published in
`/v1/models` is separate and editable from the dashboard.

Common edits in `topology.yaml`:

- **`ssh` target.** For single-node, set to
  `${ODYSSEUS_NODE_USER}@host.docker.internal` (see step 2 on why the
  env-var form is necessary). For multi-node, use the SSH spec you
  bootstrapped in step 3 (`user@host-a.lan`) — no env expansion needed
  for plain hostnames.
- **`models_dir`.** Path on every node where MLX models live. Default
  `/Volumes/models/odysseus` — change to match what `bootstrap-node.sh`
  used (default `~/mlx-models`).
- **`backend`.** Set to `ring` for multi-node TCP, `jaccl` for RDMA TB5.
  Omit for single-node pools (`size: 1`).
- **`rdma_to` map (JACCL only).** For each node, list which RDMA HCA
  reaches each peer rank. Don't fill this by hand — run:
  ```bash
  scripts/discover-rdma-wiring.py 0=user@host-a 1=user@host-b 2=user@host-c
  ```
  It SSHes into each node, lists PORT_ACTIVE HCAs, cross-references
  peer MACs via NDP, and prints the `rdma_to:` blocks ready to paste.

  > **Note for AI install agents** : you are the one writing
  > `topology.yaml` — you know the cluster the operator is
  > deploying because you just bootstrapped it (step 3) and got
  > the `user@host` mappings from the operator. The rank → ssh
  > arguments you pass to `discover-rdma-wiring.py` MUST match the
  > rank → ssh entries you put in `topology.yaml`. Same source of
  > truth, no duplication risk : you held both ends.

The example file has commented blocks for 2-node ring and 3-node JACCL —
adapt them, don't write topology.yaml from scratch.

There is no separate `validate` CLI today — the orchestrator surfaces
parsing errors in its boot logs (step 5). If `docker compose up -d` exits
or the container fails health, `docker logs odyssai-odysseus` will show
the topology error.

## 5. Start the engine

```bash
docker compose up -d

# Wait for the API to come up (max 30 s)
for i in {1..30}; do
  curl -sf http://localhost:8000/health && break || sleep 1
done
```

Expected response from `/health` (idle, no model loaded yet):

```json
{"status": "idle", "version": "…"}
```

Once a model is loaded (step 7) the same endpoint returns a richer
shape:

```json
{"status": "ok", "version": "…", "model": "…", "alive": 1, "nodes": 1}
```

Open the dashboard in the user's browser to confirm visually:
`http://localhost:8000/` (or `http://<docker-host>:8000/` if the engine
is on a different machine).

### Admin auth (optional, opt-in)

`/admin/*` is **open by default**. Odysseus is meant for a trusted-LAN
install — whoever can reach `:8000` is the operator, and a token would
add friction without protecting against any realistic threat.

If the user is exposing the engine beyond their LAN (Cloudflare tunnel,
port-forward, multi-tenant deployment), set `ODYSSEUS_ADMIN_TOKEN`
explicitly. Any non-empty value flips `/admin/*` to require
`Authorization: Bearer <token>`:

```bash
# Pin your own token (any reasonably-long random string works)
echo "ODYSSEUS_ADMIN_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" > .env
docker compose up -d  # picks up the new env var

# Then every /admin/* call needs the bearer
ODYSSEUS_ADMIN_TOKEN=$(grep ODYSSEUS_ADMIN_TOKEN .env | cut -d= -f2)
curl -H "Authorization: Bearer $ODYSSEUS_ADMIN_TOKEN" \
  http://localhost:8000/admin/clusters
```

For a single-operator LAN install (the common case), skip this entirely.
`/v1/*` endpoints (`/v1/chat/completions`, `/v1/messages`, `/v1/models`)
are always public regardless.

## 6. Get a model onto the cluster

A model has to physically exist under `models_dir` on every node before
the orchestrator can load it. There are three ways:

### 6a. Single-node — download from Hugging Face

The simplest path. The user runs `huggingface-cli` directly on the
Mac, into the `models_dir` set in `topology.yaml`:

```bash
# On the Mac host
pip install --user huggingface_hub   # if not installed
huggingface-cli download \
  mlx-community/Qwen3-7B-MLX-8bit \
  --local-dir ~/mlx-models/mlx-community/Qwen3-7B-MLX-8bit
```

Two-level layout (`org/repo`) matches what Odysseus expects.

### 6b. Multi-node — sync from one node to the rest

If you bootstrapped node A with the model in 6a, the orchestrator's
`/admin/sync/rsync` endpoint pushes it to other nodes over SSH:

```bash
curl -X POST http://<docker-host>:8000/admin/sync/rsync \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen3-7B-MLX-8bit",
    "source": "<host-a-id>",
    "targets": ["<host-b-id>", "<host-c-id>"]
  }'
```

Host identifiers come from `topology.yaml` `nodes[].id` (or the SSH
target if no `id` is set). Watch progress in dashboard → **Sync
matrix**. If you set `ODYSSEUS_ADMIN_TOKEN`, add
`-H "Authorization: Bearer $ODYSSEUS_ADMIN_TOKEN"` here too.

### 6c. From the dashboard

Open `http://<docker-host>:8000/`, go to **Sync matrix**, click
**Download from Hugging Face**, enter the repo id, pick targets. Same
mechanism as 6a+6b, with progress UI.

## 7. Load the model + smoke-test inference

```bash
# Substitute <cluster> with the key you set in topology.yaml — `default`
# if you started from the example, or whatever the user chose. If
# ODYSSEUS_ADMIN_TOKEN is set, add `-H "Authorization: Bearer
# $ODYSSEUS_ADMIN_TOKEN"` to every /admin/* call.
curl -X POST http://<docker-host>:8000/admin/<cluster>/load \
  -H 'Content-Type: application/json' \
  -d '{"model": "mlx-community/Qwen3-7B-MLX-8bit"}'
# Expect a streaming progress response. First load is 30 s–5 min depending
# on disk speed and node count.

# Smoke a completion (/v1/* is always public).
curl -X POST http://<docker-host>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<cluster>",
    "messages": [{"role":"user","content":"Say hello in one short sentence."}]
  }'
```

You should see a streaming SSE response ending with `data: [DONE]`. If
yes: **the engine is installed**. Tell the user.

## 8. Add cloud providers (optional)

Odysseus can route to OpenRouter / Anthropic / OpenAI under unified
aliases. The user pastes a key once, gets a catalogue entry like
`or:claude-haiku`, and any OpenAI/Anthropic-compatible client can hit it
through the same `:8000` endpoint.

In the dashboard: **Settings → Cloud providers → Add provider**. Pick a
template (OpenRouter / Anthropic / OpenAI), paste the API key, save. The
aliases appear in `/v1/models` immediately.

Skip this step if the user only wants local models.

## 9. What to install next

Odysseus is the engine. Most users also want **Companion**, the chat
client that lives on top of it. Companion is a separate repo:
https://github.com/Odyssai-eu/Companion (clone next to this one).

To wire Companion to this engine after Companion is installed, the user
opens **Settings → Infrastructure → Engine** in Companion, enters
`http://<docker-host>:8000`, and clicks Test endpoint. Discovery + pairing
is documented in Companion's `AGENTS.md`.

## 10. Common failure modes

**`docker compose up` exits immediately**
→ `docker logs odyssai-odysseus` to see the lifespan error. Most common:
invalid `~/.odysseus/topology.yaml` or unreachable SSH targets. The boot
log points at the offending key.

**Engine starts, `/v1/models` is empty**
→ No model loaded yet. Run step 7 (load) — the picker fills as soon as
a model is resident.

**SSH from container fails on a single-node install**
→ The container is trying to `ssh user@host.docker.internal` but Remote
Login isn't on, the SSH key isn't in `~/.ssh/authorized_keys`, or
`host.docker.internal` isn't resolving. Walk back through step 2.

**Load fails with `Shape mismatch` at runner init**
→ The model needs pipeline parallel but the topology requested tensor
parallel (or vice versa). Tensor parallel requires KV-head count
divisible by `world_size`; pipeline parallel requires the model to
implement `PipelineMixin` (deepseek_v2/v3/v32, glm4_moe, hy_v3, …). Add
`"sharding": "pipeline"` to the load payload for big MoEs, or pick a
different `pools[].size`.

**Cluster gives errno 16 / 96 / 2 after several sessions**
→ JACCL queue-pair degradation, known upstream MLX/JACCL bug on
RDMA-backed pools. Reboot the affected nodes — the dashboard has a
**Reboot all** button that orchestrates this.

**Model paths mismatch between nodes**
→ `models_dir` in topology.yaml must point to the same path on every
node. Either NFS-mount a shared directory or use the dashboard's
**Sync matrix** to rsync the model onto every node.

**SSH from container fails after install**
→ Confirm `~/.ssh:/root/.ssh:ro` is mounted in `docker-compose.yml` and
that `~/.ssh/known_hosts` already trusts every node (run `ssh user@node`
once from the host before starting the container).

**Continuous batching with Q8 KV cache silently corrupts output**
→ Known upstream issue. Use Q8 KV in legacy single-stream mode, or use
the `BatchGenerator` path with fp16 KV. Don't mix.

## 11. Where to learn more

- **README.md** — short pitch + 1-page quick-start (overlaps with this
  doc; this is the deeper one).
- **docs/GETTING-STARTED.md** — long-form operator walkthrough with all
  three install paths in detail.
- **docs/DEPLOY.md** — production deployment patterns (hot-reload, full
  rebuild, log access, discovery + pairing flow).
- **docs/bug-reports/** — factual notes on known upstream issues
  (`hy3` chat template, `mlx-vlm` quirks).
- **https://odyssai.eu/docs/** — public docs site (Starlight). Search,
  architecture, full API reference, prettier rendering.
- **Companion** — the recommended client. Clone
  https://github.com/Odyssai-eu/Companion next to this repo and run its
  `AGENTS.md`.

## 12. What you should NOT do

- **Do not pin model versions in this AGENTS.md.** The user picks what
  to run. Quote model names only as smoke-test examples.
- **Do not write `~/.odysseus/topology.yaml` from scratch.** Copy
  `config/topology.example.yaml` and edit. The example is the source of
  truth for valid keys.
- **Do not silently downgrade prerequisites.** If the user is missing
  Docker, say so and stop — don't `brew install` it without asking.
- **Do not skip the node bootstrap (step 3) — for the actual nodes.**
  Without `~/mlx-cluster/.venv` and `runner.py` on each node, every load
  attempt fails with `runner not found`. Single-node deploy: bootstrap
  the Mac. Multi-node deploy: bootstrap each cluster node.
- **Do not bootstrap the orchestrator-only host as a node.** If the
  orchestrator is a separate machine (Mac mini, Linux box, …), it never
  runs MLX itself — only Docker. Running `bootstrap-node.sh` on it just
  installs mlx + mlx-lm + a venv it'll never use. Skip it.
- **Do not install LiteLLM as part of Odysseus.** Cloud passthrough is
  built into the engine. LiteLLM is only useful as a fallback rail for
  legacy clients (rare).
- **Do not modify `runner.py`, `api.py`, or the Dockerfile during
  install.** They're release artefacts. If something doesn't work,
  diagnose and tell the user — don't patch around it.

## 13. Tell the user when you're done

When step 7 returns a streaming completion successfully, tell the user
in this shape:

> Odysseus is installed and running.
>
> - Dashboard: http://<docker-host>:8000/
> - API base (OpenAI): http://<docker-host>:8000/v1
> - API base (Anthropic): http://<docker-host>:8000
> - Cluster: `<cluster-name>` — `<N>` node(s), backend `<backend>`
> - Loaded model: `<model-id>`
>
> To use a chat UI, install Companion next: clone
> https://github.com/Odyssai-eu/Companion and run its `AGENTS.md`.

That's the end of the install.

---

## Security posture (LAN-first self-hosted)

odyssai.eu is self-hosted, **LAN-first, single-operator**. Secure the CODE;
do NOT impose a network-security policy on the operator. Triage every security
finding into two buckets — full version in Odysseus `docs/SECURITY-POSTURE.md`.

**Bucket A — always fix (code hygiene, not policy):**
- Exploitable code: injection (SSH option/ProxyCommand, path traversal,
  spawning a process from request input), RCE, SSRF, unsafe deserialization.
  Network-independent → fix it.
- No hardcoded SHARED secret. BUT a documented generic default password the
  operator changes on first login is fine and intended — do NOT replace it
  with a per-install random (worse UX: different every install, undocumentable,
  log-buried, easy lockout). Keep the env override.

**Bucket B — operator's choice (option + docs, NEVER force):**
- Bind interface (LAN vs localhost), mandatory API key, WAN exposure.
- Default LAN-friendly + usable with zero config; hardening is a documented
  opt-in, not a default. WAN exposure is the operator's job (tunnel/firewall);
  advise (Cloudflare Tunnel, IP/MAC allowlist, reverse-proxy auth), don't
  implement a policy for them.

**Cross-repo constraint:** Odysseus reaches Telemak nodes over the LAN and
sends no key — so Telemak must not force localhost-only on orchestrated nodes,
and making the Telemak key mandatory requires first teaching Odysseus to send
`Authorization: Bearer` on every upstream call. Otherwise the stack breaks.

Golden rule: **secure the code, yes; impose a network posture on the client,
no — give options and advice.**
