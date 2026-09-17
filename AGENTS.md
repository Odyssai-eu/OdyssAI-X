# AGENTS.md — OdyssAI-X: install and operate it, step by step

> For an agent (or a human at a terminal) landing here with a Mac and a goal:
> **a working `/v1/chat/completions` served by your own Apple Silicon machines.**
> Every step below is a command, what it must print, and the trap behind it.
> Internal dev conventions and the live production layout are in `CLAUDE.md`.

**macOS / Apple Silicon only.** There is no installer app; this file is the installer.

---

## 0. What you are building

- **Server** — one Mac (a Mac mini is enough) running the orchestrator in Docker
  (`odyssai-odysseus`, port **8000**): API, dashboard, cluster/pool lifecycle. It
  never runs inference; it SSHes into nodes to spawn runners.
- **Nodes** — the Macs that hold the models. Each gets `~/mlx-cluster` (pinned
  Python 3.11 venv + `runner.py` + vendored model modules), optionally
  `~/.venvs/mlx-vlm` (Python 3.12) for vision models. The server may also be a node.
- **Three pool kinds**, chosen per cluster:
  - `mlx-distributed` — one model split across nodes (pipeline or tensor parallel),
    backend `ring` (TCP, always works) or `jaccl` (RDMA over Thunderbolt 5, ~2× faster,
    needs cabling).
  - `replica` — N independent single-node copies, continuous batching, least-busy
    dispatch + session affinity. The throughput mode. No inter-node collective.
  - VLM — a vision model served by `mlx-vlm` on one node, proxied by the orchestrator.

Versions this repo is validated against (`requirements-node.txt`): **mlx 0.32.0 ·
mlx-lm 0.31.3 · transformers 5.10.0**. Do not float them.

---

## 1. Prerequisites — check before anything else

```bash
uname -m                      # arm64  (Intel is unsupported)
python3.11 --version          # on every node; `brew install python@3.11` if missing
docker --version              # on the server: Docker Desktop, running
ssh user@node.lan hostname    # from the server, no password prompt, for every node
```

If `ssh` prompts for a password: `ssh-copy-id user@node.lan`. Remote Login must be ON
on each node (System Settings → General → Sharing → Remote Login).

**Trap — the Xcode licence gate.** After an Xcode / Command Line Tools update a node's
`/usr/bin/python3` and `git` refuse to run until the licence is re-accepted; automation
then fails with "You have not agreed to the Xcode license agreements". Test with
`ssh user@node.lan 'git --version'`; fix (needs the node password):
`ssh -t user@node.lan 'sudo xcodebuild -license accept'`. The orchestrator's own
probes use the venv Python and are immune, but `brew`/`git`/`pip` on the node are not.

---

## 2. Bootstrap every node

```bash
scripts/bootstrap-node.sh user@node.lan                     # models dir defaults to ~/mlx-models
scripts/bootstrap-node.sh user@node.lan /Volumes/models/odyssai
```

Expected: `[1/5]`…`[6/6]` then `✓ node bootstrapped.` The steps: SSH+Python check →
copy `runner.py` + helpers + `patches/` + `requirements-node.txt` → pinned venv →
**`install-model-modules.sh`** (copies `scripts/mlx_models/*.py` into the venv's
`site-packages/mlx_lm/models/`) → smoke import (`mlx_lm.models.glm5_next`, `qwen4_exp`,
`deepseek_v4` + patches) → `mlx-vlm` venv (best-effort; a warning here only disables
vision models on that node).

**Trap — vendored modules live in `site-packages`.** Any `pip install -U mlx-lm` on a
node deletes them; models then fail with an unknown `model_type`. Never upgrade
`mlx-lm` outside `requirements-node.txt`; after any pip change re-run
`scripts/install-model-modules.sh user@node.lan` (or the whole bootstrap — idempotent).

**Trap — `models_dir` is per node.** Volumes are local; a model must be present at the
same `<models_dir>/<org>/<name>/` on every node of the pool that serves it. The
dashboard's *Sync* rsyncs from one node to the others.

---

## 3. Pin the GPU memory budget (per node, deliberate)

`iogpu.wired_limit_mb` decides how much unified memory MLX may wire; it does **not**
survive a reboot, so a tuned node silently falls back to the macOS default. Install
the LaunchDaemon (asks for the node password — that is why it is not automated):

```bash
scripts/wired-limit/install.sh user@node.lan 250880     # ≈245 GB on a 256 GB Mac
scripts/wired-limit/install.sh user@node.lan 491520     # ≈480 GB on a 512 GB Mac
```

The orchestrator reads the effective limit through telemetry and refuses loads that
would not fit (`preflight_refused`) instead of letting macOS jetsam the runner.

---

## 4. Describe the cluster and start the server

```bash
mkdir -p ~/.odysseus
cp config/topology.example.yaml ~/.odysseus/topology.yaml
$EDITOR ~/.odysseus/topology.yaml
docker compose up -d
curl -s http://localhost:8000/health        # {"status":"idle","version":"…"}
```

`topology.yaml` = clusters → pools → nodes (`ssh`, `models_dir`), plus `backend`
(`ring` | `jaccl`) and, for `jaccl`, the Thunderbolt port wiring. Cluster ids are yours
(`default`, `chat`, `reasoner`, …) and become `/admin/clusters/<id>/…`.

- **Single Mac:** the ssh target is `${ODYSSEUS_NODE_USER}@host.docker.internal` —
  inside the container `localhost` is the container, not your Mac.
- `/admin/*` is open on a trusted LAN. Exposing :8000 beyond it? set
  `ODYSSAI_X_ADMIN_TOKEN` (legacy `ODYSSEUS_ADMIN_TOKEN`) → `Authorization: Bearer`.
- The container mounts `~/.ssh` (read-only) to reach the nodes and `~/.odysseus` for
  the topology; persisted state lives in the `odysseus-data` volume.

Verify the mesh before loading anything:

```bash
curl -s http://localhost:8000/admin/nodes/telemetry | jq '.hosts[] | {host, ssh_ok, ram_total_bytes}'
```

Every node must be `ssh_ok: true` with a non-zero RAM figure.

---

## 5. First model

1. **Download** — dashboard `http://localhost:8000/` → Models → Download, or
   `POST /admin/downloads`. Accepts `org/name` and `org/name/<quant-subfolder>`
   (many MLX repos ship `4bit/`, `6bit/`, `8bit/` in one repo). Files land at
   `<models_dir>/<spec>/` with `config.json` at the root of the model directory.
2. **Check fit** — `GET /admin/clusters/<id>/load-options?model=<org/name>`: lists node
   counts that fit with the per-rank budget and why one does not.
3. **Load** — `POST /admin/clusters/<id>/load` `{"model":"<org/name>","nodes":1}`.
   First load: 30 s – 5 min. `GET /admin/status` → the pool shows `alive = nodes`.
4. **Chat** —

```bash
curl -s http://localhost:8000/v1/models | jq '.data[].id'
curl -s http://localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"<alias>","messages":[{"role":"user","content":"Say ok."}],"max_tokens":20}'
```

Reasoning models return their thinking in `reasoning_content` and the answer in
`content` (per-model tag handling is built in — GLM, DeepSeek, Qwen, Kimi…).

---

## 6. Replica mode — throughput

Create the cluster with **Kind = replica** (dashboard → Argo form → Kind), one node
entry per Mac that holds the model, then:

```bash
curl -s -X POST http://localhost:8000/admin/clusters/<id>/load -H 'content-type: application/json' \
  -d '{"model":"<org/name>","nodes":4,"batch":true}'
```

`batch: true` turns on continuous batching inside each replica (without it a replica
serves one request at a time). Requests are dispatched least-busy; a `session_id`
keeps a conversation on its home replica so its KV cache is reused. Dead replicas are
restarted with backoff and re-admitted; the pool stays up as long as one replica lives.
Known limit: batching interleaves *decode*, not *prefill* — a very large prompt
(~10k+ tokens) occupies its replica for the prefill duration.

---

## 7. Optional — RDMA over Thunderbolt 5 (distributed pools)

Only for `mlx-distributed` clusters that need more than TCP. Full mesh of TB5 cables
(N nodes → N·(N−1)/2 cables), then on each node:

```bash
scripts/rdma-onboard.sh              # provisions the Thunderbolt network (vendored exo recipe)
scripts/discover-rdma-wiring.py      # tells you which port sees which node → topology wiring
```

Declare `backend: jaccl` + the wiring in `topology.yaml`. **Trap — queue-pair
degradation:** after many load/unload cycles JACCL fails at init (`Couldn't allocate
protection domain`, `Recv failed with errno=2`, or `Changing queue pair to RTR failed`).
It is an upstream MLX/JACCL bug; reboot the affected nodes (dashboard → *Reboot all*,
or `POST /admin/clusters/<id>/reboot-all`). `ring` never has this problem.

---

## 8. Operating rules that prevent the classic incidents

- **Never load a second model on a node that is serving one** unless both fit with
  headroom: macOS jetsam kills the *loading* runner silently (`exit=255`, no traceback).
  An unload that answers `409` means "busy" — stop, do not force.
- **One orchestrator per set of physical nodes.** Two servers pointing at the same Macs
  fight for memory and purge each other's pools.
- **Xcode updates re-arm the licence** on every node that has Xcode.app (see §1).
- **Model dirs are per node** (§2); a `preflight_refused: taille 0` means the model is
  missing on *that* node or the path has a stale sub-folder suffix.
- **Do not edit `runner.py` / `api.py` / the Dockerfile to install** — they are release
  artefacts; configuration is `topology.yaml`, env vars, and the scripts above.
- Deploying a code change: `api.py` → `docker cp` into the container + restart (drops
  live pools, they restore); `runner.py` → `scp` to every node's `~/mlx-cluster/`
  (takes effect at the next runner spawn). Keep the nodes identical.

---

## 9. Repo map

- `scripts/api.py` — the orchestrator (FastAPI): `/v1/*`, `/admin/*`, dashboard, pools
  (`RunnerPool`, `ReplicaPool`, VLM proxy pools), downloader, preflight, persistence.
- `scripts/runner.py` — per-node MLX runner (spawned over SSH); `scripts/patches/` —
  runtime model aliases; `scripts/mlx_models/` — vendored model modules.
- `scripts/dashboard.html` — the admin SPA (served per request; hot-deployable).
- `scripts/bootstrap-node.sh`, `install-model-modules.sh`, `install-mlx-vlm.sh`,
  `wired-limit/`, `rdma-onboard.sh`, `odyssai-network-setup.sh`, `discover-rdma-wiring.py`
  — node provisioning.
- `scripts/topology.py`, `config/topology.example.yaml` — topology schema + template.
- `Dockerfile`, `docker-compose.yml`, `requirements.txt` (container),
  `requirements-node.txt` (nodes, pinned).
- `docs/` — `GETTING-STARTED.md`, `API.md`, `DEPLOY.md`, `RUNBOOK-argo-v4.md`,
  `user-guide/` (replica, CoeOS), `bug-reports/`. (Internal notes are not shipped.)

## Conventions

Fibonacci scrum points, never time. Conventional Commits + HEREDOC, no `--no-verify`,
no emojis in code or commits. Direct push to `main`; what runs on the server is what
is on `main`.
