# Deployment — Odysseus engine

The Odysseus engine runs as a single Docker container on a Linux/macOS
host with SSH access to your Apple Silicon cluster nodes. Auto-restart
is on. This document covers a single-host deployment; multi-engine /
HA setups are out of scope.

## Architecture

```
clients (Companion, curl, Continue.dev, Aider, Claude Code, …)
    ↓ HTTP :8000   (single entry point)
<docker-host> — Docker container 'odyssai-odysseus'
    ↓ SSH (your operator user) — uses host's ~/.ssh mounted ro
<cluster pool A>   spawn runner.py
<cluster pool B>   spawn runner.py
    ↓ JACCL (TB5 RDMA) or ring (TCP) per pool
inference
```

The Docker host does **no MLX compute** — it's pure orchestration. The
container:
- Listens on :8000 (host port mapped)
- SSHes into the cluster to spawn `runner.py`
- Serves the dashboard on `/`
- Persists state in a Docker volume (`/app/data/`)

Standalone Apple Silicon services you may run alongside (mlx-vlm, an
LM Studio instance, …) are **independent** of the container — they're
not orchestrated by Odysseus unless you wire them in as engine
providers.

## Container details

- **Name** : `odyssai-odysseus`
- **Image** : `odyssai-odysseus:latest` (built from this repo)
- **Persistent volume** : Docker-managed, mounted at `/app/data/`
- **Bind mount** : `~/.ssh:/root/.ssh:ro` (your operator user's keys)
- **Restart policy** : `unless-stopped`
- **Network mode** : `bridge` (cluster on LAN reachable via the default
  bridge)
- **Admin auth** : `/admin/*` is **open by default**. Odysseus is built
  for trusted-LAN installs — whoever can reach `:8000` is the operator.
  `/v1/*` and `/health` are always public.
  - Set `ODYSSEUS_ADMIN_TOKEN=<any-non-empty-value>` if you expose the
    engine beyond your LAN (Cloudflare tunnel, port-forward, multi-tenant
    deploy). With it set, every `/admin/*` call requires
    `Authorization: Bearer <token>` (EventSource also accepts
    `?token=…`).
  - Generate a strong token:

    ```bash
    python3 -c "import secrets; print(secrets.token_urlsafe(32))"
    ```

State files in `/app/data/` :
- `state-<pool>.json` — last loaded model per pool
- `cluster-config.json` — `models_dir` per cluster + `load_history`
- `topology.yaml` — your cluster topology (see `config/topology.example.yaml`)

## Useful commands

```bash
# SSH to the Docker host
ssh <operator>@<docker-host>

# Then, in the repo checkout:
cd ~/odyssai-odysseus

# Live logs
docker logs -f odyssai-odysseus

# Status
docker ps --filter name=odyssai-odysseus

# Restart (preserves volume + writable layer)
docker restart odyssai-odysseus

# Stop / start
docker compose down
docker compose up -d

# Rebuild image after Dockerfile change (preserves volume)
docker compose up -d --build
```

## Deployment workflow (code update)

### 1. Push from your laptop
```bash
git add scripts/api.py scripts/dashboard.html
git commit -m "..."
git push origin main
```

### 2A. Hot reload (no image rebuild)

For `api.py` / `dashboard.html` only:
```bash
ssh <operator>@<docker-host> "
  cd ~/odyssai-odysseus && git pull
  docker cp scripts/api.py odyssai-odysseus:/app/api.py
  docker cp scripts/dashboard.html odyssai-odysseus:/app/dashboard.html
  docker restart odyssai-odysseus
"
```

Restart ≈ 5 s + reload state (cluster runners restart if `state-*.json`
files are present).

### 2B. Full rebuild (Dockerfile change)
```bash
ssh <operator>@<docker-host> "
  cd ~/odyssai-odysseus && git pull
  docker compose up -d --build
"
```

### 3. Deploy `runner.py` (engine changes)

`runner.py` runs on the cluster nodes, not in the container. For each
node in your topology:

```bash
for ip in <node-ip-1> <node-ip-2> <node-ip-3> <node-ip-4>; do
  scp -q scripts/runner.py <operator>@$ip:~/mlx-cluster/runner.py &
done
wait

# Reload the affected pools via dashboard or API
curl -X POST http://<docker-host>:8000/admin/<pool>/unload
curl -X POST http://<docker-host>:8000/admin/<pool>/load \
  -d '{"model":"...","nodes":1,...}'
```

If your install has `ODYSSEUS_ADMIN_TOKEN` set (exposed deployments),
add `-H "Authorization: Bearer $ODYSSEUS_ADMIN_TOKEN"` to each
`/admin/*` call above.

## Verification after deploy

```bash
# API ready?
curl -s http://<docker-host>:8000/health

# Pool status
curl -s http://<docker-host>:8000/admin/<pool>/status \
  | jq '{loaded,model,nodes,alive}'

# Active sessions
curl -s http://<docker-host>:8000/admin/sessions

# Smoke generation
curl -s -X POST http://<docker-host>:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"<your-alias>","messages":[{"role":"user","content":"hi"}],"max_tokens":10}'
```

## Endpoints

- **Dashboard** : `http://<docker-host>:8000/`
- **OpenAI chat** : `POST http://<docker-host>:8000/v1/chat/completions`
- **Anthropic Messages** : `POST http://<docker-host>:8000/v1/messages`
- **Admin** : `/admin/<pool>/{status,load,unload,...}` etc.

See https://odyssai.eu/docs/api/endpoints/ for the full list.

## Aliases published

The full catalogue is at `GET /v1/models` (editable via dashboard or
`PUT /admin/providers/{id}`). You publish whatever alias names you want
per pool (e.g. `my-cluster`, `<lab>-fast`, …) and per cloud provider
(`or:claude-haiku`, `or:gpt-5`, etc.). One alias = one Companion
picker entry.

## Persistence

### State volume

Docker-managed volume mounted on `/app/data/`. Survives container
restarts AND image rebuilds. Contains state JSON files + cluster
config + topology.

To wipe (loses load history):
```bash
docker compose down -v
```

### Auto-start on host boot

- **Linux** : `systemctl enable docker` is usually already on; the
  container's `restart: unless-stopped` takes care of the rest.
- **macOS** : Docker Desktop must have *Start Docker Desktop when you
  log in* enabled, with auto-login. Container `restart: unless-stopped`
  then brings Odysseus back; the lifespan reloads `state-*.json`.

### Discovery & pairing (zero install)

No system service to install. The mechanism:

1. **Operator opens the gate** via Dashboard → Settings → Crew →
   "Open gate". This puts the engine in pairing mode for 5 minutes
   (configurable).
2. **Companion (or any client) scans the LAN** over HTTP:
   - Detects its local interfaces → list of subnets (typically /24)
   - Probes each IP on port 8000, endpoint
     `/.well-known/inference-engine.json` (timeout 1.5 s per IP,
     concurrency 50)
   - A `200` with `vendor === "odyssai.eu"` → engine found
3. **Companion calls `POST /admin/pair`** without admin auth (gate
   open) → receives a crew token + engine metadata
4. **Gate auto-closes** on first successful pair (or after 5 min)

No mDNS, no system service, no files on the host. A /24 scan takes
about 5-10 s.

**Manual test**:

```bash
# 1. Open the gate (if you set ODYSSEUS_ADMIN_TOKEN, also pass
#    `-H "Authorization: Bearer $ODYSSEUS_ADMIN_TOKEN"`)
curl -X POST http://<docker-host>:8000/admin/discovery/enable -d '{}'

# 2. Probe from anywhere on the LAN
curl -s http://<docker-host>:8000/.well-known/inference-engine.json | jq '.vendor'
# → "odyssai.eu"

# 3. Pair (gate is open — no admin token needed for this call)
curl -X POST -H "content-type: application/json" \
  http://<docker-host>:8000/admin/pair \
  -d '{"client_id":"test","client_name":"Test Client"}'
# → { token: "crew_...", engine: {...} }

# Gate is now auto-closed. Next pair attempt → 403.
```

## Troubleshooting

**Container exits with code 0 immediately**
→ `docker logs odyssai-odysseus` for the error. Often a model
referenced in `state-<pool>.json` that no longer exists. Solution:
```bash
docker exec odyssai-odysseus rm /app/data/state-<pool>.json
docker restart odyssai-odysseus
```

**Pool reload fails at boot**
→ The lifespan log says why (model not found, RDMA down, …). Fallback:
empty pool, load manually via dashboard.

**SSH from container fails**
→ Check the `~/.ssh:/root/.ssh:ro` mount in `docker-compose.yml`. Test:
```bash
docker exec odyssai-odysseus ssh -o BatchMode=yes <operator>@<node-ip> hostname
```

**Container can't reach the LAN**
→ `network_mode: bridge` (default) should route via the host. If broken:
`docker network inspect bridge`. A reboot of the Docker host often
fixes transient bridge issues.

**Volume empty after rebuild**
→ Confirm the Docker Compose project name hasn't drifted:
`docker volume ls`. If you renamed the project or repo, the volume
name moves with it — keep the project name pinned across rebuilds
with `-p <name>` if you want the same volume to follow.
