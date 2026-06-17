# INSTALL-CLUSTER.md — install the odyssai.eu stack with the OdyssAI Configurator

> **Audience: a human operator.** This is the recommended, no-terminal path:
> download the **OdyssAI Configurator** (a native macOS app, DMG / drag-to-
> Applications), and click through a wizard that installs each component on the
> right machine and builds the RDMA topology for you.
>
> Installing with an **AI agent** or a scripted macOS setup? Jump to **§12 —
> Install via the CLI** (`odyssai-configure`). Need the bare scripted path
> (Linux orchestrator, CI, no macOS GUI)? See the **Advanced / headless**
> appendix at the end. For project orientation, see [`AGENTS.md`](AGENTS.md).

---

## 1. What you are building

The **odyssai.eu** stack is three components, each on its own machine. The
orchestrator (**OdyssAI-X**) *never runs inference itself* — it routes requests
to MLX cluster nodes **or** to a single-Mac Telemak runtime. That is why a cheap
Mac mini can drive expensive Mac Studios.

| Component | What it is | Target machine | How the Configurator installs it |
|---|---|---|---|
| **Engine** | the MLX engine (`runner.py` + `venv` + `mlx`/`mlx-lm`) | **Mac Studio** (Apple Silicon) | SSH provisioning via `bootstrap-node.sh` |
| **Serveur** | **OdyssAI-X** (orchestrator) **+ Companion** (chat client + memory) | **Mac mini** (Apple Silicon) | Docker stack (`app` + Postgres `db` + `nemo-memory` embedder) |
| **Telemak** | single-Mac runtime | any one Mac | drag-and-drop of `Telemak.app` |

Two deployment modes:

- **Cluster** — Serveur **+ N Mac Studio** nodes wired over **RDMA / Thunderbolt 5**. Backend `jaccl`, multi-node topology.
- **Solo** — Serveur **+ 1 Mac running Telemak**. OdyssAI-X drives Telemak over `http-proxy`. No RDMA.

> The single-Mac case is **always Telemak**, never OdyssAI-X alone — OdyssAI-X is
> always the orchestrator-on-a-server.

---

## 2. Hardware & network you need

- **Serveur** → a **Mac mini Apple Silicon**. It can't be a cheap x86 box: the
  memory embedder (`nemo-memory` = `lightrag-mlx` + an MLX model) requires Apple
  Silicon.
- **Engine** → one or more **Mac Studio** (Apple Silicon). macOS 14+ recommended,
  Python 3.11+ (the bootstrap creates the venv).
- **Telemak** → any Mac (Solo mode).
- **Network**:
  - Cluster mode → one **TB5 cable per node-to-node link** for the RDMA mesh, plus normal LAN for SSH/API.
  - **Remote Login** (System Settings → General → Sharing) enabled on every Engine node, reachable by SSH key from the Serveur.
- **Disk**: ~5 GB per node for the venv after `mlx`+`mlx-lm`, plus your models (plan ~1.2× the model file size for KV cache + overhead).

---

## 3. Step 0 — Get the Configurator

The Configurator is **not signed with an Apple Developer ID** (deliberate:
open-source, outside the App Store). Two ways to get it:

- **Build locally (recommended for a technical operator)** — a locally built
  `.app` has no quarantine bit and opens directly:
  ```bash
  git clone https://github.com/Odyssai-eu/Odyssai-config.git
  cd Odyssai-config
  sh scripts/package-dmg.sh        # → dist/OdyssAI-Configurator-<ver>.dmg
  ```
- **Download the DMG** (GitHub release / AirDrop) — macOS sets the quarantine
  bit and Gatekeeper blocks the first launch. Unblock **without reinstalling**:
  - **No terminal**: System Settings → Privacy & Security → **“Open Anyway”** after a first open attempt.
  - **One line**: `xattr -dr com.apple.quarantine "/Applications/OdyssAI Configurator.app"`

Then drag **OdyssAI Configurator.app** to **Applications** and launch it. The
wizard flow is: **Profile → Dependencies → Configuration → Installation →
Topology → Validation.**

---

## 4. Mode Cluster — Serveur + N Mac Studio (RDMA)

### Step 1 — Install the Serveur (on the Mac mini)

1. Launch the Configurator **on the Mac mini**, pick profile **Serveur**.
2. **Dependencies**: the wizard checks for Docker and guides the install if missing.
3. **Installation**: it runs the Companion Docker stack (`docker compose up -d`,
   `--build` because the images are `:local`): `app`, `db` (Postgres 17),
   `nemo-memory` (MLX embedder).
4. Result: **OdyssAI-X on `:8000`** and **Companion on `:3100`**.

Verify the engine is up:
```bash
for i in {1..30}; do curl -sf http://localhost:8000/health && break || sleep 1; done
# idle, no model yet:  {"status":"idle","version":"…"}
```
Open the dashboard at `http://localhost:8000/` (or `http://<server-ip>:8000/`).

### Step 2 — Install the Engine on each Mac Studio

For **every** Mac Studio node:

1. Make sure **Remote Login** is on and the Serveur can SSH to it with a key
   (no password prompt):
   ```bash
   ssh -o BatchMode=yes admin@<node-ip> hostname    # expect: <hostname>
   ```
   If it prompts for a password, copy a key first: `ssh-copy-id admin@<node-ip>`
   (you type the node’s macOS password **once** — it is never stored).
2. In the Configurator, pick profile **Engine**, enter the node’s SSH target and
   (optionally) the **models directory** — the same path on every node. The
   wizard runs `bootstrap-node.sh` over SSH: it provisions `~/mlx-cluster/` with
   `runner.py`, the helpers/patches, and a pinned venv (`mlx`/`mlx-lm`), then
   smoke-imports MLX. **Idempotent** — safe to re-run.

### Step 3 — Build the topology

1. In the Configurator’s **Topology** step, enter each node as `rank = ssh-target`
   (optionally `:id`), choose backend **`jaccl`**.
2. Click **Build**. It probes the RDMA wiring via NDP (IPv6 neighbour discovery
   on each TB5 link), generates the `rdma_to:` matrix, **validates mesh symmetry**
   (every cable listed on both ends; `N·(N−1)` edges expected), and writes
   **`~/.odysseus/topology.yaml`** — backing up the previous file to `.bak` and
   **preserving any other clusters** already defined.
3. If a cable is unplugged or a node is off, validation fails with a clear
   “X cannot reach Y”. Fix the cabling and Build again.

Confirm the cluster registered:
```bash
curl -s http://localhost:8000/admin/clusters    # expect your cluster name
```

### Step 4 — First model + smoke test

See **§6** below.

---

## 5. Mode Solo — Serveur + 1 Telemak Mac

### Step 1 — Install the Serveur

Same as Cluster Step 1 (Mac mini, profile **Serveur**).

### Step 2 — Install Telemak

On the single Mac that will run the model, pick profile **Telemak**. The
Configurator copies `Telemak.app` into `/Applications/` (locally or over SSH with
`--node`). Launch Telemak and note the HTTP endpoint it listens on (e.g.
`http://<telemak-ip>:8003`).

### Step 3 — Wire Solo (http-proxy, no RDMA)

In the Topology step, choose backend **`http-proxy`** and set the **upstream** to
Telemak’s URL. The Configurator writes a Solo cluster into `~/.odysseus/topology.yaml`
(no `rdma_to`, just `upstream:`) and validates it.

### Step 4 — Smoke

OdyssAI-X now proxies to Telemak. Run the completion smoke from §6 (Telemak owns
its own model lifecycle — load the model in Telemak itself).

---

## 6. First model — download, load, smoke

A model must physically exist under `models_dir` on every node before the
orchestrator can load it.

```bash
# 6a. Download into the models_dir (two-level org/repo layout)
huggingface-cli download \
  mlx-community/Qwen3-7B-MLX-8bit \
  --local-dir ~/mlx-models/mlx-community/Qwen3-7B-MLX-8bit

# 6b. Multi-node: push it from one node to the others (or use the dashboard → Sync matrix)
curl -X POST http://<server-ip>:8000/admin/sync/rsync \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3-7B-MLX-8bit","source":"<node-a-id>","targets":["<node-b-id>"]}'

# 7. Load on the cluster (<cluster> = the key you set in topology, e.g. `default`)
curl -X POST http://<server-ip>:8000/admin/<cluster>/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3-7B-MLX-8bit"}'
# First load: 30 s – 5 min depending on disk + node count.

# Smoke a completion (/v1/* is always public)
curl -X POST http://<server-ip>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<cluster>","messages":[{"role":"user","content":"Say hello in one short sentence."}]}'
```
A streaming SSE response ending in `data: [DONE]` means **the stack is installed**.

> The models directory is **editable per cluster** in the dashboard (cluster
> screen → *Models directory* — a single shared path used by every node) and is
> what the Models matrix reads. You can also download from the dashboard →
> **Sync matrix → Download from Hugging Face**.

---

## 7. Rebuild the topology (cable moved / node added)

When a TB5 cable changes or you add a node, open the Configurator → **Topology →
Rebuild**. It re-probes the wiring, shows a **visual diff** of the cabling
before/after, re-validates the mesh, and rewrites `~/.odysseus/topology.yaml`
(old file kept as `.bak`). No need to hand-edit anything.

---

## 8. Optional — cloud providers & Companion

- **Cloud passthrough** is built into OdyssAI-X. Dashboard → **Settings → Cloud
  providers → Add provider** (OpenRouter / Anthropic / OpenAI), paste the key —
  aliases like `or:claude-haiku` appear in `/v1/models` immediately. Do **not**
  install LiteLLM; it is only a legacy fallback rail.
- **Companion** is already installed by the Serveur profile (`:3100`). To point
  it at this engine: Companion → **Settings → Infrastructure → Engine** →
  `http://<server-ip>:8000` → Test endpoint.

---

## 9. Admin auth (optional, opt-in)

`/admin/*` is **open by default** — odyssai.eu is a trusted-LAN, single-operator
install. Only if you expose the engine beyond your LAN (tunnel, port-forward),
set a token. Any non-empty value flips `/admin/*` to require `Authorization:
Bearer <token>`:
```bash
echo "ODYSSAI_X_ADMIN_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
docker compose up -d
```
> `ODYSSEUS_ADMIN_TOKEN` is still read as a legacy alias, but `ODYSSAI_X_ADMIN_TOKEN`
> is the current name. `/v1/*` endpoints stay public regardless.

---

## 10. Common failure modes

| Symptom | Fix |
|---|---|
| `docker compose up` exits immediately | `docker logs odyssai-odysseus` — usually an invalid `~/.odysseus/topology.yaml` or an unreachable SSH target. The boot log names the bad key. |
| Engine up, `/v1/models` empty | No model loaded — run the load step (§6). |
| Topology Build fails “X cannot reach Y” | A TB5 cable is unplugged or a node is off. Fix cabling, Build again. |
| Load fails `Shape mismatch` at runner init | Sharding mismatch. Tensor parallel needs KV-heads divisible by `world_size`; big MoEs need pipeline (`"sharding":"pipeline"` in the load payload). |
| `errno 16 / 96 / 2` after several sessions | JACCL queue-pair degradation (known upstream MLX/JACCL bug). Reboot the affected nodes — dashboard has a **Reboot all** button. |
| Model paths mismatch between nodes | `models_dir` must be the same path on every node — use the dashboard **Sync matrix** to rsync, or a shared mount. |

---

## 11. Security posture (LAN-first, self-hosted)

odyssai.eu is **LAN-first, single-operator**. Defaults are usable with zero
config; hardening is an opt-in, not a forced policy. Bind interface, mandatory
API key, and WAN exposure are **your choice** (advice: Cloudflare Tunnel,
IP/MAC allowlist, reverse-proxy auth — the stack doesn’t impose them). The
documented default admin password is meant to be changed on first login. Full
version in `docs/SECURITY-POSTURE.md`.

---

## 12. Install via the CLI (agents / scripted macOS)

The same install, driven by **`odyssai-configure`** — the idempotent CLI that the
GUI calls under the hood (JSON output, ships in the app bundle, or
`swift run odyssai-configure` from the [Configurator repo](https://github.com/Odyssai-eu/Odyssai-config)).
This is the path for an AI agent or a scripted macOS setup. All flags below are
verified against the binary.

```bash
# Prerequisites (JSON — parse, don't eyeball)
odyssai-configure check-deps          # ssh / python / docker / vendored scripts
odyssai-configure setup-deps          # ~/.odyssai/venv (pydantic + pyyaml) for topology validation
odyssai-configure versions            # installed component versions vs the bundle's payloads

# Mac mini (Serveur) — TWO installs: orchestrator THEN server
odyssai-configure install orchestrator [--rebuild]                       # OdyssAI-X (Docker, :8000), vendored, no clone
odyssai-configure install server [--app-home <dir>] [--bind <iface>] \
                                 [--port <n>] [--skip-nemo]              # Companion stack (:3100) + native MLX nemo. Defaults: 0.0.0.0 / 3100

# Each Mac Studio (Engine) — remote bootstrap over SSH
odyssai-configure install engine --node admin@<node-ip> [-m <models-dir>]   # default models dir $HOME/mlx-models
#   …or provision the CURRENT Mac locally (no remote SSH):
odyssai-configure node-setup base        # MLX runtime: embedded python + vendored wheels (no network, no brew)
odyssai-configure node-setup network     # RDMA recipe on this Mac (root + local console)
odyssai-configure node-setup check       # read-only RDMA provisioning status

# Telemak (Solo mode)
odyssai-configure install telemak [--node admin@<mac>] [--force]            # omit --node for local /Applications/

# Topology — probes RDMA, validates mesh, backs up to .bak, PRESERVES other clusters
odyssai-configure topology build --cluster <name> --backend jaccl \
  --node 0=admin@<node-a>[:id] --node 1=admin@<node-b>[:id] …               # Cluster (RDMA)
odyssai-configure topology build --cluster <name> --backend http-proxy \
  --upstream http://<telemak-ip>:8003                                       # Solo
odyssai-configure topology rebuild [--cluster <name>] [--dry-run]           # re-probe + before/after diff
odyssai-configure topology show          # current clusters as JSON
odyssai-configure validate               # validate ~/.odysseus/topology.yaml against the schema
```

Then download/load a model and smoke-test as in §6. For project orientation
(what lives where, conventions), see [`AGENTS.md`](AGENTS.md).

---

## Appendix — Advanced / headless install (no DMG)

Use this when the Configurator GUI isn’t an option: a **Linux orchestrator**
(no macOS `.app`), CI, or a fully headless setup. The Configurator is just a
façade over these scripts — they remain the source of truth and still ship in
the engine repo.

```bash
# 1. SSH bootstrap each node (one-time)
ssh-copy-id admin@<node-ip>                       # key auth, password typed once

# 2. Provision each MLX node (idempotent) — from the engine repo
./scripts/bootstrap-node.sh admin@<node-ip> ~/mlx-models

# 3. Discover the RDMA wiring (Cluster mode only)
python scripts/discover-rdma-wiring.py 0=admin@<node-a> 1=admin@<node-b> …
#    → prints ready-to-paste topology.yaml blocks; non-zero exit if the mesh is incomplete

# 4. Write the topology by COPYING the example (never from scratch)
cp config/topology.example.yaml ~/.odysseus/topology.yaml
#    edit nodes / rdma_to (jaccl) or upstream (http-proxy); validated by topology.py (TopologyConfig)

# 5. Start the Serveur stack
docker compose up -d
```
For an **agent-driven** equivalent of this appendix using the `odyssai-configure`
CLI (`install`, `topology build/rebuild`, `validate`), see [`AGENTS.md`](AGENTS.md).

---

## Where to learn more

- [`AGENTS.md`](AGENTS.md) — project orientation: what OdyssAI-X is, where everything lives, conventions.
- `docs/DEPLOY.md` — production deployment patterns (hot-reload, rebuild, logs).
- `https://odyssai.eu/docs/` — public docs site (architecture, full API reference).
- Configurator repo: `https://github.com/Odyssai-eu/Odyssai-config` (`README.md` + `CLAUDE.md`).
