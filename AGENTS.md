# AGENTS.md — install the odyssai.eu stack (agent, via `odyssai-configure`)

> This file is a runbook for an **AI coding agent** (Claude Code, Codex, Cursor,
> Aider, …) installing the odyssai.eu stack. Execute it top-to-bottom.
>
> **Premise: prefer the `odyssai-configure` CLI** — it is idempotent, emits JSON,
> and wraps the bootstrap/topology scripts the right way. Drive raw scripts only
> from the **Appendix** (when the CLI isn't available). A human operator using
> the **OdyssAI Configurator** GUI instead should follow [`INSTALL-CLUSTER.md`](INSTALL-CLUSTER.md).
>
> Do not invent hostnames, IPs, models, or paths — the user provides those. When
> a prerequisite is missing, stop and tell the user; do not `brew install` things
> without asking.

---

## 1. What you are installing

Three components, each on its own machine. **OdyssAI-X (the orchestrator) never
runs inference** — it routes to MLX nodes **or** to a single-Mac Telemak runtime.

| Component | What | Target | CLI install |
|---|---|---|---|
| **Engine** | MLX engine (`runner.py` + venv + `mlx`/`mlx-lm`) | Mac Studio (Apple Silicon) | `install engine --node <ssh>` (remote) or `node-setup base` (this Mac) |
| **Orchestrator** | OdyssAI-X (`api.py`) — routing only, no inference | Mac mini (Apple Silicon) | `install orchestrator` |
| **Server** | Companion (chat client + memory: app + db + memory compiler + company LightRAG + native nemo) | same Mac mini | `install server` |
| **Telemak** | single-Mac runtime | any Mac | `install telemak` |

> The Mac mini “Serveur” hosts **both** OdyssAI-X and Companion → run `install
> orchestrator` **and** `install server` on it. (The GUI’s single “Serveur”
> profile does both for a human; the CLI splits them.)

Two modes: **Cluster** (Serveur + N Mac Studio, RDMA/TB5, backend `jaccl`) ·
**Solo** (Serveur + 1 Telemak Mac, backend `http-proxy`). The single-Mac case is
always Telemak, never OdyssAI-X alone.

> The Serveur **must** be Apple Silicon: the `nemo-memory` embedder
> (`lightrag-mlx` + an MLX model) won't run on x86.

---

## 2. Get the CLI + check prerequisites

The `odyssai-configure` binary ships inside the **OdyssAI Configurator.app**
bundle, or you build it from the Configurator repo:
```bash
git clone https://github.com/Odyssai-eu/Odyssai-config.git && cd Odyssai-config
swift build && swift run odyssai-configure --help     # Swift 6.1+ / macOS 14+
```

Then check the local prerequisites (JSON output — parse it, don't eyeball):
```bash
odyssai-configure check-deps      # ssh / python / docker / vendored scripts
odyssai-configure setup-deps      # creates ~/.odyssai/venv (pydantic + pyyaml) if needed
odyssai-configure versions        # installed component versions vs this bundle's payloads (JSON)
```

Per-profile prerequisites the CLI expects:
- **Engine node**: Apple Silicon, macOS 14+, Python 3.11+. For a **remote** install (`install engine --node`): **Remote Login on** + SSH key auth from where you run the CLI — verify `ssh -o BatchMode=yes admin@<node> hostname`. To provision the **current** Mac instead, use `node-setup base` (embedded python + vendored wheels — no network, no brew) and `node-setup network` (RDMA recipe; root + local console).
- **Orchestrator / Server**: Docker (Desktop on macOS), ~500 MB image + state volume. The Server profile also runs the native MLX `nemo` memory service → Apple Silicon.

---

## 3. Install — per profile

Run each as needed for the target mode. All sub-commands are **idempotent** and
return JSON; check `ok`/error fields rather than scraping stdout.

```bash
# Mac mini — OdyssAI-X orchestrator (Docker, vendored build context, no clone).
odyssai-configure install orchestrator [--rebuild]

# Mac mini — Companion stack (app + db + memory compiler + company LightRAG, Docker;
# + native MLX nemo memory service). Defaults: --bind 0.0.0.0, --port 3100.
odyssai-configure install server [--app-home <dir>] [--bind <iface>] [--port <n>] [--skip-nemo]

# Each Mac Studio — bootstrap an MLX engine over SSH (provisions ~/mlx-cluster/).
odyssai-configure install engine --node admin@<node-ip> [--models-dir <path>]   # -m alias; default $HOME/mlx-models

# Telemak (Solo mode) — drops Telemak.app locally (/Applications) or on a remote node.
odyssai-configure install telemak [--node admin@<mac>] [--force]
```

- `--models-dir` (engine) is the path **on that node**; the cluster-level shared
  path is set in the topology (`topology build --models-dir`) and is editable per
  cluster in the dashboard.
- After `install orchestrator`, OdyssAI-X is on `:8000`; after `install server`,
  Companion is on `:3100`. Confirm the engine:
  ```bash
  for i in {1..30}; do curl -sf http://localhost:8000/health && break || sleep 1; done
  # {"status":"idle","version":"…"}   → orchestrator up, no model yet
  ```

---

## 4. Build the topology

The topology lives in **`~/.odysseus/topology.yaml`** (schema `TopologyConfig` in
`topology.py`). Build it with the CLI — it probes RDMA, validates mesh symmetry,
backs up the old file to `.bak`, and **preserves other clusters**. Never
hand-write it when the CLI is available.

```bash
# Cluster mode (RDMA/TB5): probe wiring via NDP, write rdma_to matrix.
odyssai-configure topology build --cluster <name> --backend jaccl \
  --node 0=admin@<node-a>[:id] --node 1=admin@<node-b>[:id] …

# Solo mode: OdyssAI-X → Telemak over HTTP, no RDMA.
odyssai-configure topology build --cluster <name> --backend http-proxy \
  --upstream http://<telemak-ip>:8003

# After a cable moves or a node is added: re-probe + diff before/after.
odyssai-configure topology rebuild [--cluster <name>] [--dry-run]

odyssai-configure topology show         # current wiring
odyssai-configure validate              # validate ~/.odysseus/topology.yaml against the schema
```
`topology build` exits non-zero with a clear “X cannot reach Y” if a TB5 cable is
unplugged or a node is off — fix the cabling and re-run. Confirm registration:
```bash
curl -s http://localhost:8000/admin/clusters     # expect your cluster name
```

---

## 5. Get a model, load it, smoke-test

A model must exist under `models_dir` on every node before a load.

```bash
# Download into the models_dir (two-level org/repo layout)
huggingface-cli download mlx-community/Qwen3-7B-MLX-8bit \
  --local-dir ~/mlx-models/mlx-community/Qwen3-7B-MLX-8bit

# Multi-node: push from one node to the rest (or dashboard → Sync matrix)
curl -X POST http://<server-ip>:8000/admin/sync/rsync -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3-7B-MLX-8bit","source":"<node-a-id>","targets":["<node-b-id>"]}'

# Load (<cluster> = the key from topology, e.g. `default`)
curl -X POST http://<server-ip>:8000/admin/<cluster>/load -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3-7B-MLX-8bit"}'      # first load 30 s – 5 min

# Smoke (/v1/* is always public)
curl -X POST http://<server-ip>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<cluster>","messages":[{"role":"user","content":"Say hello in one short sentence."}]}'
```
A streaming SSE response ending in `data: [DONE]` means the stack works.

> Host ids come from `topology.yaml` `nodes[].id` (or the SSH target). For big
> MoEs that need pipeline parallel, add `"sharding":"pipeline"` to the load payload.

---

## 6. Admin auth (optional, opt-in)

`/admin/*` is **open by default** — trusted-LAN, single-operator. Only set a token
if exposing the engine beyond the LAN. Any non-empty value requires `Authorization:
Bearer <token>` on every `/admin/*` call:
```bash
echo "ODYSSAI_X_ADMIN_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
docker compose up -d
```
`ODYSSEUS_ADMIN_TOKEN` is read as a legacy alias; `ODYSSAI_X_ADMIN_TOKEN` is
current. `/v1/*` stays public.

---

## 7. Optional — cloud providers & Companion

- Cloud passthrough is built into OdyssAI-X (dashboard → **Settings → Cloud
  providers**). Do **not** install LiteLLM — it's only a legacy fallback rail.
- Companion ships with the Serveur profile (`:3100`). Wire it to the engine:
  Companion → **Settings → Infrastructure → Engine** → `http://<server-ip>:8000`.

---

## 8. Common failure modes

| Symptom | Fix |
|---|---|
| `docker compose up` exits immediately | `docker logs odyssai-odysseus` — invalid `~/.odysseus/topology.yaml` or unreachable SSH target; the boot log names the key. |
| `/v1/models` empty after start | No model loaded — run the load step (§5). |
| `topology build` fails “X cannot reach Y” | TB5 cable unplugged / node off. Fix cabling, re-run. |
| Load `Shape mismatch` at runner init | Sharding mismatch — tensor parallel needs KV-heads divisible by `world_size`; big MoEs need `"sharding":"pipeline"`. |
| `errno 16 / 96 / 2` after several sessions | JACCL queue-pair degradation (known upstream bug). Reboot the affected nodes (dashboard **Reboot all**). |
| `models_dir` mismatch between nodes | Same path on every node — Sync matrix rsync, or a shared mount. |

---

## 9. What you should NOT do

- **Don't pin model versions here.** The user chooses; quote models only as smoke examples.
- **Don't hand-write `~/.odysseus/topology.yaml`** when the CLI is available — use `topology build`. (Bare-script appendix only as last resort, and then copy `config/topology.example.yaml`, never from scratch.)
- **Don't bootstrap the orchestrator host as a node.** The Serveur runs Docker only; it never runs MLX. `install engine` is for the Mac Studios.
- **Don't install LiteLLM.** Cloud passthrough is built in.
- **Don't modify `runner.py`, `api.py`, or the Dockerfile** during install — they are release artefacts. Diagnose and report instead of patching.
- **Don't silently downgrade prerequisites** — if Docker/Remote Login is missing, say so and stop.

---

## 10. Tell the user when you're done

When §5 returns a streaming completion:

> The odyssai.eu stack is installed and running.
> - Dashboard: `http://<server-ip>:8000/`
> - API (OpenAI): `http://<server-ip>:8000/v1` · (Anthropic): `http://<server-ip>:8000`
> - Companion (chat UI): `http://<server-ip>:3100/`
> - Cluster: `<cluster>` — `<N>` node(s), backend `<backend>` · Loaded: `<model-id>`

---

## Appendix — raw scripts (when the CLI isn't available)

`odyssai-configure` wraps these engine scripts (`scripts/` in this repo). Use them
directly only on a headless/Linux orchestrator or in CI:

```bash
./scripts/bootstrap-node.sh admin@<node-ip> ~/mlx-models           # = install engine
python scripts/discover-rdma-wiring.py 0=admin@<a> 1=admin@<b> …    # = topology build (probe)
cp config/topology.example.yaml ~/.odysseus/topology.yaml          # then edit + validate via topology.py
docker compose up -d                                               # = install server (stack)
```
- `bootstrap-node.sh` is idempotent: `scp`s `runner.py` + helpers/patches, builds the pinned venv, smoke-imports MLX into `~/mlx-cluster/`.
- `discover-rdma-wiring.py` prints ready-to-paste `topology.yaml` blocks and exits non-zero if the mesh is incomplete (`N·(N−1)` edges).
- `topology.py` (`TopologyConfig`) validates mesh symmetry + backend coherence (`jaccl`⇒`rdma_to`, `http-proxy`⇒`upstream`).

---

## Security posture (LAN-first, self-hosted)

Secure the **code**; do not impose a network policy on the operator.
- **Always fix**: exploitable code (injection, RCE, SSRF, path traversal, unsafe deserialization) and hardcoded shared secrets. A documented default admin password the operator changes on first login is fine — keep the env override.
- **Operator's choice (option + docs, never force)**: bind interface (LAN vs localhost), mandatory API key, WAN exposure. Default LAN-friendly + zero-config; hardening is an opt-in.
- **Cross-repo**: OdyssAI-X reaches Telemak over the LAN with no key — so Telemak must not force localhost-only on orchestrated nodes, and making the Telemak key mandatory first requires teaching OdyssAI-X to send `Authorization: Bearer` upstream. Full version: `docs/SECURITY-POSTURE.md`.
