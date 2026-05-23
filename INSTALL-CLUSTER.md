# INSTALL-CLUSTER.md — guided multi-node setup

> This file is a runbook for AI coding agents (Claude Code, Codex,
> Cursor, Aider, …). When the user wants to set up an Odysseus cluster
> end-to-end — 1 to 5 Apple Silicon nodes, plus an orchestrator host,
> plus optionally Companion as the chat client — follow this doc
> top-to-bottom.
>
> If the user only wants ONE component installed (engine only, client
> only, single node only), use `AGENTS.md` instead — it has the
> per-component runbook with the 6 install patterns.
>
> This doc is for the "I have N Macs, drive me through the whole
> install in one session" case.

## What you will install

| Component | Where | Why |
|---|---|---|
| **Cluster nodes** (1-5) | Each Apple Silicon Mac that will run MLX runners | Compute. One pool per cluster, but pools share nodes via JACCL/ring. |
| **Orchestrator** | One Docker host (can be one of the nodes, or a separate Mac mini / Linux box) | The Odysseus container — REST API + dashboard + SSHes to nodes |
| **Companion** (optional) | Same Docker host or another | The chat client — web UI, memory, projects, MCP |
| **Models directory** | Each node OR a shared external SSD | Where MLX weights live on disk |

You will end this runbook with a working `/v1/chat/completions` endpoint
serving a real open-weights model, with `/admin/clusters/<your-name>`
visible to the dashboard.

## Stage 0 — Plan the deployment

Ask the user the following before touching anything. Build a plan
table in the conversation; once the user confirms, proceed.

```
Q1. How many cluster nodes?
    1 — one Mac (single-Mac install)
    2-5 — multi-Mac cluster

Q2. For each node, the SSH target:
    e.g. "admin@192.168.1.42", "sophie@studio-1.lan"

Q3. Are the nodes Thunderbolt-5-cabled in a full mesh?
    yes  → JACCL backend (faster, requires per-peer interface map)
    no   → ring backend (TCP, works on any LAN)

Q4. Where does the orchestrator run?
    one of the nodes / a separate Mac / a Linux box
    (anything with Docker reachable from all nodes)

Q5. Install Companion (chat client) alongside?
    yes → same host as orchestrator? or different?
    no  → user will hit the API directly or pair Companion later

Q6. Cluster name — short, role-descriptive (e.g. "my-cluster", "lab-fast")
    This becomes the cluster_id in /admin/clusters/<name>/* and the
    YAML key in topology.yaml. Whatever the user picks IS the name —
    the engine no longer reserves any reserved word.

Q7. Models directory plan
    - external SSD (recommended for serious clusters, 2-4 TB)
    - internal SSD on each node (default ~/mlx-models)
    - NFS-mounted shared volume

Q8. First model to load?
    Skip this for now — Stage 7 has a `suggest-models.py` that picks
    candidates based on the cluster RAM you collect in Stage 2.
```

Echo the plan back to the user as a summary before starting Stage 1.
Don't proceed without confirmation.

## Stage 1 — SSH bootstrap (one-time per node)

The orchestrator container will SSH into each node to spawn runners.
You need key-based auth from the orchestrator host (or your laptop, if
you're driving the install from there) to each node.

For each node target the user gave in Q2:

```bash
# 1. Generate a key on the orchestrator host (or your laptop) if there
#    isn't one yet — skip if ~/.ssh/id_ed25519 already exists.
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519

# 2. Copy the key to the node. The user types their macOS account
#    password ONCE per node — never store it, never read it back.
ssh-copy-id <ssh-target>
# e.g. ssh-copy-id admin@192.168.1.42

# 3. Verify key-based auth works without password.
ssh -o BatchMode=yes -o ConnectTimeout=5 <ssh-target> 'hostname; uname -m'
# Expect: <hostname>\narm64
```

**Critical:** never put the password in the conversation, the script,
or any file. The user pastes it directly into their `ssh-copy-id`
prompt — Claude never sees or stores it.

**Pre-req on each node:**
- macOS with Apple Silicon (the smoke command above confirms `arm64`)
- **Remote Login** enabled (System Settings → General → Sharing →
  Remote Login → ON). If `ssh-copy-id` fails with `Connection refused`,
  this is why. Tell the user; don't try to fix it yourself.

If a node has SSH key auth set up already from a prior install (you can
tell by `ssh -o BatchMode=yes ... hostname` working without
`ssh-copy-id`), skip it for that node.

## Stage 2 — Bootstrap each cluster node

Now install the MLX runtime on each node — Python venv with
`mlx + mlx-lm`, plus the `runner.py` script the orchestrator will
spawn. The `scripts/bootstrap-node.sh` shipped in this repo does
exactly that.

For each node:

```bash
# Run from the directory of this repo. Default models dir is
# '$HOME/mlx-models' on the remote node — override with the path the
# user chose in Q7 if it's an external SSD.
scripts/bootstrap-node.sh <ssh-target>

# OR with an explicit models dir on the node (quote the variable so
# the dollar sign resolves on the REMOTE shell):
scripts/bootstrap-node.sh <ssh-target> '$HOME/mlx-models'
scripts/bootstrap-node.sh admin@studio-1.lan /Volumes/External/models
```

The script is idempotent — re-running it just re-syncs scripts and
re-checks the venv. Smoke at the end verifies `mlx + mlx-lm` import.

### Optional but recommended — `lean-node.sh`

If the Mac will only serve as an MLX node (no human use), strip the
background services that hold wired memory + churn CPU on idle :

```bash
ssh <ssh-target> 'bash -s' < scripts/lean-node.sh
# or with an explicit models dir to disable Spotlight on:
ssh <ssh-target> 'bash -s' < scripts/lean-node.sh -- --models-dir /Volumes/models
```

What it does, briefly :

1. Hides ~25 unused Apple apps from Launchpad (Mail, Maps, FaceTime,
   News, Stocks, Music, Photos, iWork suite, …) via `chflags hidden` —
   reversible.
2. Turns off Spotlight indexing on the models SSD (a multi-TB volume
   gains nothing from being indexed).
3. Disables Time Machine on the node.
4. `launchctl disable`s iCloud / Photos / News / Music background
   daemons — `photoanalysisd` alone can hold 1-2 GB of wired memory
   on a freshly-installed Mac.
5. `pmset -a sleep 0 disksleep 0 hibernatemode 0` — a node that sleeps
   mid-load = killed pool, JACCL queue-pair errors.

All actions are reversible and the script is idempotent. Add
`--dry-run` to preview without applying. Ask the user first if the Mac
is also a workstation — the hidden-apps step would be intrusive on a
shared box.

**Collect from each node** (you need it for Stage 3 + 7):

```bash
# RAM per node (we'll sum across nodes for the model-fit calc later)
ssh <ssh-target> 'sysctl -n hw.memsize | awk "{print int(\$1/1024/1024/1024)}"'
# → e.g. 96, 256, 512 (GB)

# Models dir on the node
ssh <ssh-target> 'echo $HOME/mlx-models'
# OR the path the user set in Q7
```

Tally the total cluster RAM. Save the per-node values for the topology
file in Stage 4.

## Stage 3 — RDMA discovery (only if Q3 was "yes JACCL")

JACCL needs to know, for each node, which RDMA HCA talks to which
peer. macOS enumerates Thunderbolt ports differently per boot — so on
one node the cable to peer rank 1 might be `rdma_en5`, and on another
the cable to peer rank 1 might be `rdma_en3`. The mapping is
hardware-specific and **physically opaque** — neither the operator
nor the agent can tell which `rdma_en<N>` corresponds to which
chassis port by looking at the box.

Skip this stage entirely if Q3 was "ring". Continue to Stage 4.

### Auto-discovery

Use the bundled script. It SSHes into each node, lists the
PORT_ACTIVE HCAs, then cross-references the peer MAC addresses
visible via NDP (IPv6 neighbor discovery on each TB5 interface) to
reconstruct who's connected to whom:

```bash
scripts/discover-rdma-wiring.py \
  0=<ssh-of-rank0>[:<id>] \
  1=<ssh-of-rank1>[:<id>] \
  ...
```

Worked example for a 4-node cluster:

```bash
scripts/discover-rdma-wiring.py \
  0=admin@10.0.0.1:host-a \
  1=admin@10.0.0.2:host-b \
  2=admin@10.0.0.3:host-c \
  3=admin@10.0.0.4:host-d
```

Output is a per-node `rdma_to:` block, ready to paste under each node
in `topology.yaml`:

```
# rank 0 — host-a (admin@10.0.0.1)
rdma_to:
  1: rdma_en5    # → host-b
  2: rdma_en4    # → host-c
  3: rdma_en3    # → host-d
```

The script exits non-zero (with warnings on stderr) if the cluster
isn't a full mesh — typically a cable unplugged or a node off. Fix
the cabling and re-run.

### When auto-discovery fails

If the script fails repeatedly (NDP table not populating, IPv6 disabled
on the TB bridge, etc.) — fall back to `ring` backend in Stage 4. Ring
is TCP-only, no wiring map needed, slightly slower on big MoEs.
JACCL is an optimization; ring is a working install.

## Stage 4 — Write `topology.yaml` + start the orchestrator

On the orchestrator host (the one from Q4), do:

```bash
# 1. Clone Odysseus on the orchestrator host (if not done yet).
git clone https://github.com/Odyssai-eu/Odysseus.git ~/Odysseus
cd ~/Odysseus

# 2. Write the topology config. ~/.odysseus is the canonical location
#    the container reads (mounted into the container at runtime).
mkdir -p ~/.odysseus
```

Build `~/.odysseus/topology.yaml` from Q1-Q7. Use the user's cluster
name from Q6 as the YAML key (NOT "default" unless the user explicitly
said "default"). Single-node example:

```yaml
clusters:
  <cluster-name>:                # e.g. my-cluster
    label: "<short description>" # e.g. "Mac Studio cluster"
    backend: ring                # or jaccl if Q3 was yes
    pools:
      - size: 1
        nodes:
          - rank: 0
            id: node-0
            ssh: admin@192.168.1.42
            models_dir: /Users/admin/mlx-models   # from Q7 / Stage 2
```

Multi-node ring (Q3 = no):

```yaml
clusters:
  <cluster-name>:
    label: "<description>"
    backend: ring
    pools:
      - size: <N>          # number of nodes
        nodes:
          - rank: 0
            id: node-0
            ssh: <target-0>
            models_dir: <models-dir-0>
          - rank: 1
            id: node-1
            ssh: <target-1>
            models_dir: <models-dir-1>
          # … one per node
```

Multi-node JACCL (Q3 = yes): same as ring but add `backend: jaccl` and
a `rdma_to:` block per node, populated from the Stage 3 matrix:

```yaml
- rank: 0
  id: node-0
  ssh: <target-0>
  rdma_to:
    1: rdma_en5      # interface on node-0 that reaches node-1
    2: rdma_en4      # interface on node-0 that reaches node-2
```

Then start the orchestrator:

```bash
docker compose up -d

# Wait for /health to come up
for i in {1..30}; do
  curl -sf http://localhost:8000/health && break || sleep 1
done
# Expect: {"status":"idle","version":"…"}

# Confirm the cluster is registered
curl -s http://localhost:8000/admin/clusters | jq '.data[].id'
# Expect: "<cluster-name>"  (and nothing else if you defined only one)
```

If `/admin/clusters` returns an empty list or the wrong id, the
container couldn't parse the topology — `docker logs odyssai-odysseus`
will show the error. Fix the YAML, re-up, re-check.

**Admin auth note:** `/admin/*` is open by default for trusted-LAN
installs (whoever can reach `:8000` is the operator). If the user is
exposing the engine beyond their LAN, set `ODYSSEUS_ADMIN_TOKEN` in
the env at this stage — see `AGENTS.md` step 5.

## Stage 5 — Install Companion (only if Q5 was "yes")

Companion is in a separate repo. On the host the user chose for
Companion (often the same as the orchestrator host):

```bash
git clone https://github.com/Odyssai-eu/Companion.git ~/Companion
cd ~/Companion

cp .env.example .env
# Defaults are sensible — host port binds to 127.0.0.1 only, session
# secret auto-generates on first boot. Edit only if you need
# customization (e.g. HOST_BIND=0.0.0.0 for LAN exposure).

docker compose up -d

# Wait for /api/health
for i in {1..30}; do
  curl -sf http://localhost:3000/api/health && break || sleep 1
done
# Expect: {"status":"ok","version":"0.1.0","engines":0}
```

Tell the user:

> Open http://localhost:3000/, click **Sign up**, enter email +
> password + name (8-char minimum). The first signup on an empty DB
> becomes the workspace admin automatically. After signup, go to
> **Settings → Infrastructure → Engine** and click **Discover** —
> Companion will find the Odysseus engine you started in Stage 4.

If the orchestrator and Companion are on different machines, the user
will need to enter `http://<orchestrator-host>:8000` manually instead
of using Discover (which only scans the local LAN).

## Stage 6 — Confirm models directory + sync if multi-node

Each cluster node needs the same model files at the same path on its
local disk (or via a shared mount).

```bash
# Verify the models_dir exists on each node and is writable
for target in <ssh-target-0> <ssh-target-1> …; do
  ssh "$target" "mkdir -p '<models-dir>' && ls -ld '<models-dir>'"
done
```

External SSD recommendation: if any node has an external SSD with
≥500 GB free (`ls /Volumes/`), suggest moving models there before the
disk fills. Big MoEs are 300-700 GB each.

For multi-node: when you load a model in Stage 7, the orchestrator
will need the file on EVERY node. Either:

- **Download on one node, sync to the rest** via `/admin/sync/rsync`
  (works from the dashboard or via curl). The orchestrator manages
  the rsync over SSH.
- **NFS-mount a shared models directory** — operators with a serious
  storage setup often have this already.

Don't try to be clever — just tell the user the options.

## Stage 7 — First model: pick + download + load

What gates model size is **cumulated RAM across the pool**, not node
count. 4 × 32 GB Macs total 128 GB of usable memory — that doesn't
unlock a 1.5 TB model just because there are 4 nodes. Sum the per-node
unified-memory values you captured in Stage 2 (or re-collect them via
`ssh admin@<ip> 'sysctl -n hw.memsize'` divided by 2³⁰).

Then ask the helper which models fit, keeping 20% margin for KV cache:

```bash
# From the repo root
scripts/suggest-models.py --ram-gb <total>
```

Example output for a 96 GB single-Mac install:

```
mlx-community/Qwen3.6-35B-A3B-8bit       38G  chat   Latest Qwen MoE...
```

Example for a 4-node pool of 256 GB Ultras (`--ram-gb 1024`):

```
mlx-community/Qwen3.6-35B-A3B-8bit                38G  chat
mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit    85G  chat
mlx-community/Qwen3-Coder-Next-8bit               85G  code
mlx-community/Qwen3.5-122B-A10B-8bit             131G  chat
mlx-community/MiniMax-M2-8bit                    243G  chat
mlx-community/Qwen3.5-397B-A17B-8bit             422G  reasoner
mlx-community/Qwen3-Coder-480B-A35B-Instruct-8bit 540G code
mlx-community/DeepSeek-V3.1-8bit                 713G  reasoner
```

Show the user the table. Let them pick. Don't impose.

Then download + load. On a single-node install:

```bash
# Make sure huggingface-cli is on the orchestrator host (or wherever
# the models_dir actually lives — usually the node itself).
pip install --user --upgrade huggingface_hub

# Download into the models_dir from Q7 / Stage 2
huggingface-cli download <repo-id> \
  --local-dir <models-dir>/<repo-id>
# e.g. ~/mlx-models/mlx-community/Qwen3-7B-MLX-8bit
```

For multi-node, download on node 0, then sync to the others:

```bash
curl -X POST http://<docker-host>:8000/admin/sync/rsync \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<repo-id>",
    "source": "node-0",
    "targets": ["node-1", "node-2"]
  }'
# Watch progress in dashboard → Sync matrix.
```

Then load the model on the cluster:

```bash
curl -X POST http://<docker-host>:8000/admin/clusters/<cluster-name>/load \
  -H 'Content-Type: application/json' \
  -d '{"model": "<repo-id>"}'
# Streaming progress response. First load is 30 s–5 min depending on
# disk speed and node count.
```

Smoke-test the completion:

```bash
curl -X POST http://<docker-host>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<cluster-name>",
    "messages": [{"role":"user","content":"Say hello in one short sentence."}]
  }'
```

You should get a streaming SSE response. If yes: **the cluster is
fully operational**.

If the user opted into Companion (Q5), tell them:

> Open Companion → start a new chat → pick **<cluster-name>** in the
> model picker → send a message. The reply comes through Odysseus,
> through your `<cluster-name>` cluster, off the model you just loaded.

## Stage 8 — Tell the user what they just built

Hand the user a concise success summary:

```
✓ Odysseus cluster installed.

  Cluster:       <cluster-name>
  Backend:       <ring|jaccl>
  Nodes:         <N>  (<total-RAM>GB total unified memory)
                 - node-0 @ <target-0>  <ram-0>GB  models: <dir-0>
                 - node-1 @ <target-1>  <ram-1>GB  models: <dir-1>
                 …

  Orchestrator:  <docker-host>:8000  (dashboard, API)
  Companion:     <companion-host>:3000  (optional, only if installed)

  Loaded model:  <repo-id>
  Endpoints:
    OpenAI:      http://<docker-host>:8000/v1/chat/completions
    Anthropic:   http://<docker-host>:8000/v1/messages

  Next moves:
    - Browse the dashboard at http://<docker-host>:8000/
    - Add more models via Sync matrix → Download from Hugging Face
    - Add another cluster: edit ~/.odysseus/topology.yaml, restart container
    - Expose beyond LAN: set ODYSSEUS_ADMIN_TOKEN, configure reverse proxy
```

## Failure modes — what to do when something breaks

**SSH key auth still fails after `ssh-copy-id`**
→ Remote Login isn't enabled on that node. User flips it on in
System Settings. Don't fight the OS — tell them, wait, retry.

**`bootstrap-node.sh` reports "Python 3.11+ required"**
→ The node has only an older Python. Install via `brew install
python@3.11` on the node, then re-run the bootstrap.

**`docker logs odyssai-odysseus` shows "topology validation failed"**
→ Your `~/.odysseus/topology.yaml` has a typo or wrong key. The log
quotes the offending field. Fix the YAML, `docker compose up -d`
again (no rebuild needed).

**`/admin/clusters` is empty after Stage 4**
→ Topology file isn't being read. Check the bind mount in
`docker-compose.yml`: `${HOME}/.odysseus:/root/.odysseus:ro` should
exist. If you're on a system where `${HOME}` resolves weirdly, replace
with the absolute path.

**Load fails with `Connection refused` to a node**
→ The orchestrator's SSH key isn't trusted on that node. Re-run
`ssh-copy-id` from inside the container:
`docker exec odyssai-odysseus ssh-copy-id <ssh-target>`.

**Load fails with `Shape mismatch` at runner init**
→ The model needs pipeline parallel but the pool was started with
tensor parallel (or vice versa). Add `"sharding": "pipeline"` to the
load payload for big MoEs, or pick a smaller pool size.

**JACCL pool fails with `Changing queue pair to RTR failed`**
→ Known upstream MLX/JACCL bug — queue-pair degradation after many
sessions. Reboot the affected nodes (dashboard → **Reboot all**).

## What you should NOT do

- **Do not collect or store passwords.** SSH key auth via
  `ssh-copy-id` is the only authentication step where the user types
  a password, and it goes directly into their terminal, not into the
  conversation.
- **Do not bootstrap the orchestrator-only host as a node.** Stage 2
  only applies to machines listed in `topology.yaml` as cluster nodes.
  A separate Mac mini orchestrating remote Mac Studios doesn't need
  MLX installed locally.
- **Do not invent the RDMA matrix.** If the user can't draw it,
  fall back to `ring`. Wrong wiring data makes JACCL fail at
  `init_distributed` with cryptic errors.
- **Do not pick a model for the user.** Show what fits, let them
  choose. Their hardware, their preferences.
- **Do not pin model versions in this doc.** The `suggest-models.py`
  catalog is curated; if the user asks for something not in it,
  download it anyway as long as it's a published MLX model on HF.

## Where to go from here

- **`AGENTS.md`** — granular component install (the 6 patterns from
  yesterday). Use when the user already has parts of the stack and
  just needs to add one piece.
- **`docs/GETTING-STARTED.md`** — long-form operator walkthrough,
  more narrative than this runbook.
- **https://odyssai.eu/docs/** — public docs site (Starlight), with
  architecture deep-dives, API reference, and the Companion user
  guide.
- **Companion repo** — `https://github.com/Odyssai-eu/Companion`
  with its own `AGENTS.md` for client-side workflow patterns
  (semantic routing, MCP servers, agent tokens, etc.).
