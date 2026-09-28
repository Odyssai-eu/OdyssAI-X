# AGENTS.md: install and operate OdyssAI-X, step by step

> Node discovery over Bonjour (`odyssai nodes`, #79/#80) is not shipped yet: declare each node in the topology by its ssh target (A3). Every command below exists today.

For an agent (or a human at a terminal) with a set of Macs and one goal: **a working
`/v1/chat/completions` served by those Macs.** Every step is one command, the output
it must print, and what to do when it does not. Run the steps in order. Do not skip
a step and do not improvise one: when a failure branch says "stop", report the
failing line to the human and wait.

---

## Assumes

Check each item before step 1. If one is not true, stop and tell the human which one.

| Item | Requirement |
|---|---|
| Hardware | Apple Silicon Macs (`uname -m` prints `arm64`). One **server** runs the orchestrator; one or more **nodes** hold the models. The server may also be a node. |
| macOS | Same macOS version on every node. For RDMA over Thunderbolt: macOS 26.2 or later. |
| Accounts | An **admin** account on every node. The human knows its password: the installer asks for it once (the GPU memory LaunchDaemon). |
| Management LAN | Every node and the server on the same Ethernet LAN; the nodes reach github.com and astral.sh (the installer downloads from them). |
| Node software | Nothing: the installer brings uv and Python 3.11 (no Homebrew, no Xcode tools). The manual path (§B) needs Homebrew's `python@3.11`. |
| SSH | Remote Login ON on every node; from the server, `ssh admin@<node>` works without a password prompt. |
| Server | Docker Desktop installed and running; `git`, `curl` and `jq` available. |
| TB5 wiring (RDMA only) | Thunderbolt 5 cables in a full mesh (N nodes, N(N-1)/2 cables) and RDMA enabled once per node in recoveryOS (`rdma_ctl enable`, then reboot). Without this, use `backend: ring` (TCP), which needs nothing. |

Names used below: nodes `node-a.local`, `node-b.local` (their Bonjour names; any
ssh target works), user `admin`, cluster id `default`. Replace them with yours.

---

## The sequence at a glance

```bash
# on each node (from the server, over ssh)
ssh -t admin@node-a.local '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Odyssai-eu/OdyssAI-X/main/install.sh)"'
# on the server
git clone https://github.com/Odyssai-eu/OdyssAI-X.git && cd OdyssAI-X
mkdir -p ~/.odysseus && cp config/topology.example.yaml ~/.odysseus/topology.yaml
docker compose up -d
curl -s http://localhost:8000/health
scripts/discover-rdma-wiring.py 0=admin@node-a.local 1=admin@node-b.local   # RDMA only
$EDITOR ~/.odysseus/topology.yaml
docker compose restart
scripts/odyssai-x doctor --engine http://localhost:8000                     # stop on any FAIL
curl -s -X POST http://localhost:8000/admin/downloads -H 'content-type: application/json' \
  -d '{"repo":"mlx-community/Qwen3-30B-A3B-4bit","targets":["node-a"]}'
curl -s -X POST http://localhost:8000/admin/clusters/default/load -H 'content-type: application/json' \
  -d '{"model":"mlx-community/Qwen3-30B-A3B-4bit","nodes":1}'
curl -s http://localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"<alias>","messages":[{"role":"user","content":"Say ok."}],"max_tokens":20}'
```

The steps below give, for each line, the expected output and the failure branch.

---

## A. Install

### A1. Bootstrap every node

Run once per node, from the server:

```bash
ssh -t admin@node-a.local '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Odyssai-eu/OdyssAI-X/main/install.sh)"'
```

It installs, on that node: uv and Python 3.11, the pinned venv (`~/mlx-cluster/.venv`,
versions from `requirements-node.txt`), the runner and its helpers, the vendored model
modules and runtime patches, the patched JACCL (downloaded and checked against its
sha256), the GPU wired-memory daemon, then runs `~/mlx-cluster/odyssai-x doctor` (node
mode) and prints the node's topology entry. Each stage prints `ok` or `changed` and
its duration (about 6 minutes on a fresh node, mostly package downloads).

- **Expected:** eight stage lines, the doctor lines (one per check), the topology entry.
  Exit code 0. A second run prints `already up to date` and changes nothing.
- **Password prompt:** sudo asks for the node's admin password once, for the GPU
  memory daemon (skipped when the node already has one). An agent does not type
  passwords: hand the prompt to the human, or have the human run the same command in
  their own terminal. Without a terminal (`ssh` without `-t`) that stage is skipped
  with a warning.
- **`FAIL …: this node is serving a model`:** the installer refuses to change a node
  that serves. Unload its pools first, or, for a deliberate change, rerun with
  `ODYSSAI_X_FORCE=1` in front of `/bin/bash`.
- **`WARN` (exit 1):** read the fix sentence on the line, report it, continue.
- **`FAIL` (exit 2):** stop. Report the `FAIL` line (it names the check and the fix)
  to the human. After the fix, re-run the same command; it is idempotent.
- **`ssh` asks for a password:** the SSH row of Assumes is not met. Stop.

### A2. Start the server

```bash
git clone https://github.com/Odyssai-eu/OdyssAI-X.git && cd OdyssAI-X
mkdir -p ~/.odysseus && cp config/topology.example.yaml ~/.odysseus/topology.yaml
docker compose up -d
curl -s http://localhost:8000/health
```

- **Expected:** `{"status":"idle","version":"…","admin_auth_enabled":false}`.
- **Connection refused:** the container is still starting. Retry every 5 s for 60 s;
  then run `docker compose logs --tail 50` and stop with that output.
- `/admin/*` is open on a trusted LAN. If port 8000 is reachable beyond it, set
  `ODYSSAI_X_ADMIN_TOKEN` before `docker compose up -d` and send
  `Authorization: Bearer <token>` on every `/admin/*` call.

### A3. Collect the nodes

Each A1 run ends with that node's topology entry (host, ssh target, `models_dir`). Keep
them for A4. (Discovery over Bonjour, `odyssai nodes`, is not shipped yet.)

- **A node's doctor showed a `FAIL`:** fix it first (the line names the fix) and re-run
  A1 on that node.

### A4. Describe the cluster

Write `~/.odysseus/topology.yaml` with one entry per node collected in A3. The ssh target
is `admin@<host>`; `models_dir` is the directory on that node where models live.

TCP (`ring`), two nodes:

```yaml
clusters:
  default:
    backend: ring
    pools:
      - size: 1
        nodes:
          - {rank: 0, id: node-a, ssh: admin@node-a.local, models_dir: /Users/admin/mlx-models}
      - size: 2
        nodes:
          - {rank: 0, id: node-a, ssh: admin@node-a.local, models_dir: /Users/admin/mlx-models}
          - {rank: 1, id: node-b, ssh: admin@node-b.local, models_dir: /Users/admin/mlx-models}
```

RDMA (`jaccl`): first, **the human**, at the console of each node (the script
refuses to run over SSH), provisions the Thunderbolt network once from a checkout of
this repository:

```bash
sudo scripts/rdma-onboard.sh --apply --console --expect <number of cabled TB ports on this node>
```

Exit code 0 means ready, 10 means reboot this node once and re-run, 1 means a guard
refused: stop and report its output. Then, from the server:

```bash
scripts/discover-rdma-wiring.py 0=admin@node-a.local 1=admin@node-b.local
```

- **Expected:** one `rdma_to:` block per node. Set `backend: jaccl` and paste each
  block under its node in the pool that uses those nodes.
- **Non-zero exit, `X cannot reach Y`:** the cable between X and Y is missing, loose
  or on a disabled port. Stop and report the line.

Single Mac as its own node: the ssh target is
`${ODYSSEUS_NODE_USER}@host.docker.internal` (inside the container `localhost` is the
container); the example topology shows it.

Apply the file:

```bash
docker compose restart
```

### A5. Doctor, cluster mode: stop on any FAIL

```bash
scripts/odyssai-x doctor --engine http://localhost:8000          # add --json for agents
```

Runs the checks in the engine (`GET /admin/doctor`, same data with `?format=text`): every
node reachable, venv and pinned `mlx`, patched JACCL, model modules and patches in sync
with the release, wired-memory limit, models directory and free space, macOS build
alike on every node and, for every Thunderbolt cable of the topology, both ends checked
(port active, link-local alias, peer reachable through that cable) with one line per
cable. One entry per check with `OK`, `WARN` or `FAIL` and a one-sentence fix; the
JSON follows `docs/doctor.schema.json`. The exit code is the worst status (header
`X-Doctor-Exit` on the HTTP call).

- **Exit 0:** continue.
- **Exit 1 (`WARN` only):** report the warnings, continue.
- **Exit 2 (any `FAIL`):** stop. Do not load a model. Report each `FAIL` entry
  verbatim; a cable failure reads like
  `FAIL rdma-edge node-a rdma_en3 ↔ node-b rdma_en5: … PORT_DOWN — Check the Thunderbolt cable on node-a en3 (port down) at both ends.`
  After the fix, run A5 again.

### A6. First model

A small model that fits one node, used as the install check:

```bash
curl -s -X POST http://localhost:8000/admin/downloads -H 'content-type: application/json' \
  -d '{"repo":"mlx-community/Qwen3-30B-A3B-4bit","targets":["node-a"]}'
curl -s http://localhost:8000/admin/downloads | jq '.data[] | {repo, status, size, error}'
```

- **Expected:** `{"id":"…"}`, then `status` goes from `running` to `completed` (poll
  every 30 s). `targets` are host ids: the ssh host without its domain suffix
  (`admin@node-a.local` gives `node-a`). Gated repositories need `"hf_token":"…"`.
- **`status: failed`:** stop and report `error`.

```bash
curl -s "http://localhost:8000/admin/clusters/default/load-options?model=mlx-community/Qwen3-30B-A3B-4bit" | jq
curl -s -X POST http://localhost:8000/admin/clusters/default/load -H 'content-type: application/json' \
  -d '{"model":"mlx-community/Qwen3-30B-A3B-4bit","nodes":1}'
curl -s http://localhost:8000/admin/clusters/default/status | jq '{loaded, loading, model, nodes}'
```

- **Expected:** `load-options` lists `1` among the node counts that fit. The load
  returns, then `status` shows `"loaded": true` and `"loading": null` (30 s to 5 min).
- **HTTP 422 `preflight_refused`:** the load would fail; `blockers` says why (model
  missing on that node, does not fit, vision venv missing). Stop and report `blockers`.
  Do not pass `force`.
- **HTTP 409 `cluster_degraded`:** stop and report `message`.

### A7. First request

```bash
curl -s http://localhost:8000/v1/models | jq -r '.data[].id'
curl -s http://localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"<alias from /v1/models>","messages":[{"role":"user","content":"Say ok."}],"max_tokens":20}'
```

- **Expected:** a `choices[0].message.content` that answers. Reasoning models put their
  thinking in `reasoning_content` and the answer in `content`.
- The install is done. Anything that speaks OpenAI `chat/completions` or Anthropic
  `messages` can use `http://<server>:8000/v1`.

---

## B. Manual path (no installer)

Use this to drive the nodes from a checkout on the server (it pushes over ssh). It
replaces A1; the other steps are unchanged.

1. **Bootstrap each node from the server** (pushes over ssh, idempotent):

   ```bash
   scripts/bootstrap-node.sh admin@node-a.local                  # models dir: ~/mlx-models
   scripts/bootstrap-node.sh admin@node-a.local /Volumes/models  # or your own
   ```

   Expected: `[1/5]` to `[6/6]`, a doctor report, then `✓ admin@node-a.local bootstrapped.` A line with
   `⚠` about JACCL means the stock JACCL was kept (RDMA pools lose the patches listed
   in `vendor/jaccl/PATCHES.md`); a line with `⚠` about `mlx-vlm` disables vision
   models on that node only. `ERROR: python3.11 not found`: `brew install python@3.11`
   on the node, re-run.

2. **Pin the GPU memory budget** (asks for the node password; `install.sh` does this):

   ```bash
   scripts/wired-limit/install.sh admin@node-a.local 250880    # about 245 GB on a 256 GB Mac
   ```

3. **Check:** the doctor of A5 works the same on nodes bootstrapped this way; on one
   node: `ssh admin@node-a.local '~/mlx-cluster/odyssai-x doctor'`.

---

## C. Traps behind the steps

- **Xcode licence.** After an Xcode or Command Line Tools update, a node's
  `/usr/bin/python3` and `git` refuse to run until the licence is accepted again, and
  `brew`/`git`/`pip` automation on that node fails. Test: `ssh admin@node-a.local 'git
  --version'`. Fix (human, needs the password): `ssh -t admin@node-a.local 'sudo
  xcodebuild -license accept'`. The orchestrator's own probes use the venv Python and
  are not affected.
- **Never `pip install -U mlx` or `mlx-lm` on a node.** Upgrading `mlx-lm` deletes the
  vendored model modules; upgrading `mlx` puts the stock `libjaccl.dylib` back. Versions
  come from `requirements-node.txt` only. After any pip change, re-run A1 (or B1).
- **Keep the nodes identical.** Runtime patches live in `~/mlx-cluster/patches/` and
  are read at each runner spawn. After any change to `scripts/mlx_models/` or
  `scripts/patches/`, sync every node of the pool (`scripts/install-model-modules.sh
  admin@<node>…`): ranks running different versions of a model file corrupt output
  silently. A running runner keeps what it imported; a synced file is used at the next
  load.
- **`models_dir` is per node.** Volumes are not shared; a model must exist at the same
  `<models_dir>/<org>/<name>/` on every node of the pool that serves it. The
  dashboard's *Sync* copies it from one node to the others.
- **The wired-memory limit** (`iogpu.wired_limit_mb`) does not survive a reboot on its
  own; the daemon from A1 (or B2) re-applies it. The orchestrator reads the effective
  limit and refuses loads that would not fit (`preflight_refused`) instead of letting
  macOS kill the runner.

---

## D. Throughput: replica mode

Create the cluster with **Kind = replica** (dashboard, cluster form, Kind), one node
entry per Mac that holds the model, then:

```bash
curl -s -X POST http://localhost:8000/admin/clusters/<id>/load -H 'content-type: application/json' \
  -d '{"model":"<org/name>","nodes":4,"batch":true}'
```

`batch: true` turns on continuous batching inside each replica (without it a replica
serves one request at a time). Requests go to the least busy replica; a `session_id`
keeps a conversation on its home replica so its KV cache is reused. Dead replicas are
restarted with backoff and re-admitted; the pool stays up while one replica lives.
Known limit: batching interleaves *decode*, not *prefill*: a very large prompt (about
10k tokens and more) occupies its replica for the prefill duration.

---

## E. What RDMA gives you, and its limits

Measured with the libibverbs probes in `scripts/jaccl/` on two M3 Ultras (2026-09-18):

- **Before a load**, every edge of the wiring is checked from both ends (port
  `PORT_ACTIVE`, link-local `169.254.x.x` alias present, peer alias reachable through
  that interface). A bad edge refuses the load and names it, for example
  `rdma link(s) not usable - node-b rdma_en6 -> node-c rdma_en7: PORT_DOWN`. Fix the
  cable or port, or reboot that node (a reboot renegotiates the Thunderbolt link).
- **During a run**, a rank that dies (crash, SIGKILL, memory pressure) is reported on
  every survivor within a second (`[jaccl] peer is gone: side channel to rank N
  closed…`) instead of a silent spin at 100 % CPU. The runner exits and the
  orchestrator's rank-death path recovers. A collective that makes no progress at all
  fails after `JACCL_PROGRESS_TIMEOUT_S` (default 600).
- When several ranks die at once, the load error starts with `CAUSE -> rank N: …`: the
  rank with the link error; the others died of the closed side channel.

Limits: 10 queue pairs per device (shared by all processes on the node), UC only (no
RC, no retransmission), receive buffers must match the message size. `ring` (TCP)
needs none of this.

---

## F. Operating rules that prevent the classic incidents

- **Never load a second model on a node that is serving one** unless both fit with
  headroom: macOS kills the *loading* runner silently (`exit=255`, no traceback). An
  unload that answers `409` means "busy": stop, do not force.
- **One orchestrator per set of physical nodes.** Two servers pointing at the same Macs
  fight for memory and purge each other's pools.
- **Do not edit `runner.py`, `api.py` or the Dockerfile to install.** Configuration is
  `topology.yaml`, environment variables and the scripts above.
- Deploying a code change: `api.py` goes into the container (`docker cp` + restart;
  live pools are restored); `runner.py` goes to every node's `~/mlx-cluster/` (used at
  the next runner spawn). Keep the nodes identical.

---

## G. Repo map

- `scripts/api.py`: the orchestrator (FastAPI): `/v1/*`, `/admin/*`, dashboard, pools
  (`RunnerPool`, `ReplicaPool`, VLM proxy pools), downloader, preflight, persistence.
- `scripts/runner.py`: per-node MLX runner (spawned over SSH); `scripts/patches/`:
  runtime model patches; `scripts/mlx_models/`: vendored model modules.
- `scripts/dashboard.html`: the admin page (served per request).
- Node install: `install.sh` (on the node), `scripts/odyssai-x` + `doctor_node.py` +
  `doctor-manifest.json` (the doctor; `docs/doctor.schema.json`).
- Node provisioning from a checkout: `scripts/bootstrap-node.sh`, `install-model-modules.sh`,
  `install-jaccl.sh`, `build-jaccl.sh`, `install-mlx-vlm.sh`, `wired-limit/`,
  `rdma-onboard.sh`, `odyssai-network-setup.sh`, `discover-rdma-wiring.py`.
- `vendor/jaccl/`: JACCL (MLX v0.32.2) and our patches (`PATCHES.md`, `UPSTREAM.md`);
  `scripts/jaccl/`: libibverbs probes and the 2-rank smoke test behind section E.
- `vendor/exo/`: licence and provenance of the files derived from exo (see `NOTICE`).
- `scripts/topology.py`, `config/topology.example.yaml`: topology schema and template.
- `Dockerfile`, `docker-compose.yml`, `requirements.txt` (container),
  `requirements-node.txt` (nodes, pinned).
- `docs/`: `GETTING-STARTED.md`, `API.md`, `DEPLOY.md`, `RUNBOOK-argo-v4.md`,
  `user-guide/`, `bug-reports/`.

## Conventions

Fibonacci scrum points, never time. Conventional Commits + HEREDOC, no `--no-verify`,
no emojis in code or commits. Direct push to `main`; what runs on the server is what
is on `main`.
