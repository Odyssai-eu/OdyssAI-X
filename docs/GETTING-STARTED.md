# Getting started with Odysseus

This guide walks a fresh operator from "I have a Mac" to "first chat completion served by my own cluster". Three modes, picked by hardware:

1. **Single node** — one Apple Silicon Mac. No mesh, no JACCL. Fastest path to a working `/v1/chat/completions`.
2. **Multi-node TCP (ring backend)** — multiple Macs on the same LAN, no Thunderbolt cabling required. Slower than RDMA but works on any network.
3. **Multi-node JACCL (RDMA over Thunderbolt 5)** — full mesh between Apple Silicon nodes with TB5 cables. Maximum throughput for big models.

If you're new to Odysseus, **start with single node**. You can move to multi-node once you have something working.

## Prerequisites

- macOS with Apple Silicon (M1 / M2 / M3 / M4 — the M-series). Intel Macs aren't supported.
- Docker Desktop installed and running.
- Python 3.11+ available on each cluster node (for the runner processes).
- For multi-node: SSH access from the master node to every other node (`ssh user@host` works without password — set up via `ssh-copy-id`).
- For JACCL multi-node: Thunderbolt 5 cables between nodes (each Apple Silicon Mac has 6 TB5 ports; for a full mesh of N nodes, you need N×(N-1)/2 cables).

## Disk + memory budget

| Pool size | Suggested total RAM | Notes |
|---|---|---|
| Small (7-13B model, single node) | 32 GB | Fits a Qwen3-7B or Llama-3-8B comfortably |
| Medium (30-40B model, single node) | 96 GB | M3 Ultra base config or M4 Max with extra |
| Large (100B+ MoE, 2-3 nodes) | 256 GB+ | Distributed pipeline parallelism |
| XL (390B+ MoE, 3-4 nodes) | 512 GB+ | Production reasoner setups |

Each loaded model needs its weights resident in unified memory. Plan ~1.2× the model file size to account for KV cache and runtime overhead.

## Mode 1 — Single node

Easiest path. The cluster is just your own machine. JACCL is skipped because `world_size=1`.

> **Heads-up on the SSH path.** Odysseus runs the orchestrator in a Docker
> container that SSHes out to "cluster nodes" to spawn the MLX runner.
> Even on a single-Mac install, the orchestrator treats your Mac as a
> remote node. So we need: Remote Login enabled on the Mac, an SSH key
> the container can use, and `host.docker.internal` as the target
> instead of `localhost` (which inside the container resolves to the
> container itself).

```bash
# 1. Clone
git clone https://github.com/Odyssai-eu/Odysseus.git
cd Odysseus

# 2. Enable Remote Login on macOS
#    System Settings → General → Sharing → Remote Login → ON
#    Then make sure your own SSH key authorises container-to-host:
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# 3. Bootstrap your Mac as a cluster node (creates ~/mlx-cluster with
#    venv + runner.py + helpers).
scripts/bootstrap-node.sh "$USER@$(hostname)"

# 4. Create your topology config
mkdir -p ~/.odysseus
cp config/topology.example.yaml ~/.odysseus/topology.yaml

# 5. Edit topology.yaml — set the ssh target to host.docker.internal
$EDITOR ~/.odysseus/topology.yaml
```

Minimal `topology.yaml` for single-node:

```yaml
clusters:
  argo:
    pools:
      - size: 1
        nodes:
          - rank: 0
            ssh: ${ODYSSEUS_NODE_USER}@host.docker.internal
            models_dir: /Users/${ODYSSEUS_NODE_USER}/mlx-models
```

> `${ODYSSEUS_NODE_USER}` is set by `docker-compose.yml` from the host
> `$USER`. Override it (`ODYSSEUS_NODE_USER=alice docker compose up -d`)
> if you launch the orchestrator from a context where `$USER` isn't
> set (cron, systemd, CI).

Then:

```bash
# 6. Start the engine + dashboard
docker compose up -d

# 7. Verify it's alive
curl http://localhost:8000/health
# {"status":"idle","version":"…"}    # before a model is loaded

# 8. Download a model into models_dir on the Mac host
huggingface-cli download \
  mlx-community/Qwen3-7B-MLX-8bit \
  --local-dir ~/mlx-models/mlx-community/Qwen3-7B-MLX-8bit

# 9. Load it onto the argo cluster
#    /admin/* is open on a default LAN install. To require Bearer auth
#    on a publicly-reachable deployment, set ODYSSEUS_ADMIN_TOKEN in
#    your env and pass `-H "Authorization: Bearer $ODYSSEUS_ADMIN_TOKEN"`.
curl -X POST http://localhost:8000/admin/argo/load \
  -H 'Content-Type: application/json' \
  -d '{"model": "mlx-community/Qwen3-7B-MLX-8bit"}'

# 10. Chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "argo",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

You should see a streaming SSE response. Open `http://localhost:8000` for the dashboard view of what's loaded, who's chatting, and what the cluster's doing.

## Mode 2 — Multi-node TCP (ring backend)

Use this when you have multiple Macs on a LAN but no Thunderbolt cables. Slower than JACCL — distributed traffic goes over your normal Ethernet — but works anywhere.

Prerequisites:
- SSH key auth from the Docker host to every cluster node
- `scripts/bootstrap-node.sh` run once per node (creates `~/mlx-cluster/` with venv + runner)
- A models directory at the same path on every node (use the dashboard's **Sync matrix** to rsync, or NFS-mount)

```bash
# Bootstrap each node
scripts/bootstrap-node.sh user@host-a.lan
scripts/bootstrap-node.sh user@host-b.lan
```

`topology.yaml`:

```yaml
clusters:
  argo:
    backend: ring                 # TCP, not JACCL
    pools:
      - size: 2
        nodes:
          - rank: 0
            id: host-a
            ssh: user@host-a.lan
            models_dir: /Users/user/mlx-models
          - rank: 1
            id: host-b
            ssh: user@host-b.lan
            models_dir: /Users/user/mlx-models
```

Then `docker compose up -d` on the Docker host. The container reads the topology, SSHes to each worker, starts the runner, and orchestrates over TCP.

To push a model from one node to others, use `/admin/sync/rsync` (which requires `model`, `source`, `targets` — all three) or the dashboard's Sync matrix UI:

```bash
curl -X POST http://localhost:8000/admin/sync/rsync \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen3-7B-MLX-8bit",
    "source": "host-a",
    "targets": ["host-b"]
  }'
```

(If you set `ODYSSEUS_ADMIN_TOKEN` for an exposed deployment, add
`-H "Authorization: Bearer $ODYSSEUS_ADMIN_TOKEN"` to every `/admin/*`
call.)

Expect ~10 G Ethernet throughput between nodes. For models that fit comfortably on a single node, you don't gain from multi-node. Where it matters: models bigger than any single node's RAM.

## Mode 3 — Multi-node JACCL (RDMA over Thunderbolt 5)

Full mesh cabling between nodes. Bandwidth approaches 80 Gb/s per link, RTT in single-digit microseconds.

This is the production path but the setup is more involved:

1. **Cable the mesh.** For N nodes, plug N×(N-1)/2 TB5 cables — each pair gets one cable. (Diagram: see `config/topology.example.yaml` JACCL block and the wiring guidance on https://odyssai.eu/docs/architecture/cluster/.)
2. **Discover interface names.** Run on each node:

   ```bash
   ifconfig | grep -B1 "thunderbolt"
   # → en4, en5, en6 etc. depending on which port you plugged
   ```

   These names depend on macOS port enumeration at boot. They differ from one node to the next — that's expected.

3. **Generate the topology** with the discovery tool (TODO — currently you edit `topology.yaml` by hand from the `ifconfig` output):

   ```yaml
   clusters:
     argo:
       backend: jaccl
       pools:
         - size: 3
           nodes:
             - rank: 0
               ssh: user@host-a.lan
               rdma_to:
                 1: rdma_en5    # interface on host-a that talks to rank 1
                 2: rdma_en4    # interface on host-a that talks to rank 2
             - rank: 1
               ssh: user@host-b.lan
               rdma_to:
                 0: rdma_en5
                 2: rdma_en4
             - rank: 2
               ssh: user@host-c.lan
               rdma_to:
                 0: rdma_en5
                 1: rdma_en4
   ```

4. **Verify mesh health** before loading a model:

   ```bash
   curl -s http://localhost:8000/admin/nodes/telemetry \
     | jq '.hosts[] | {host, ssh_ok, ram_total_bytes}'
   ```

   Every node should answer `ssh_ok: true` with non-zero `ram_total_bytes`. A node that's missing or `ssh_ok: false` means the orchestrator can't reach it — fix SSH first; JACCL needs a working SSH channel to start the runner on each rank.

   On JACCL pools, if `init_distributed` later fails with `Changing queue pair to RTR failed`, that's the upstream queue-pair degradation bug. Reboot the affected nodes (dashboard → **Reboot all**, or `POST /admin/clusters/<id>/reboot-all`).

5. **Load + chat** as in single-node mode. Pool `size` in the request determines which entry in `topology.yaml` is used.

## Known gotchas

- **JACCL queue pair degradation** after many consecutive sessions. Symptom: a model that loaded fine yesterday fails today with `Changing queue pair to RTR failed`. Reboot the affected nodes — there's a "Reboot all" button in the dashboard.
- **Continuous batching with quantized KV cache** is currently incompatible (silent corruption). Either use Q8 KV with the legacy `stream_generate` path, or use BatchGenerator with fp16 KV.
- **Pipeline parallel** requires the model to implement `PipelineMixin` (e.g. `deepseek_v2`, `deepseek_v3`, `glm4_moe`, `hy_v3`, `ministral3`). Tensor parallel requires KV-head count divisible by `world_size`. Mixing these wrong is the most common cause of `RuntimeError: shape mismatch` at load.
- **Model paths**: the model id in the API (`mlx-community/Foo-MLX-8bit`) must match a directory on every node at the same path. The dashboard's sync UI rsyncs from `host-a` to the others, but the master expects models at `${ODYSSEUS_MODELS_DIR:-~/mlx-models}` by default.

## Next steps

- **Pair Companion** — install [Companion](https://github.com/Odyssai-eu/Companion) and point it at this engine for a chat UI with memory, projects, skills, and the rest.
- **Multi-cluster setups** — the API recognises three cluster keys today: `argo`, `hades`, `vlm`. You can run more than one (e.g. `argo` for a big reasoner pool, `hades` for code, `vlm` for vision) by defining several entries in `topology.yaml`. The user-facing aliases in `/v1/models` are editable separately from the dashboard.
- **Capability contract** — your engine publishes `/.well-known/inference-engine.json` and `/v1/models` with `x_odyssai` blocks. Clients (Companion, IDE plugins) use this to know what each model supports.

If you get stuck, open an issue with your `topology.yaml` (redact SSH targets if sharing) and the output of `curl /admin/version` + the error you're seeing.
