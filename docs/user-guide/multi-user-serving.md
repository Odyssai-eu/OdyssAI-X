---
title: Multi-user serving
description: Replica mode plus continuous batching — one endpoint, one full copy of the model per Mac, up to eight conversations per Mac at once.
---

# Multi-user serving

> Sharding makes a model fit. Replicas make it serve a crowd. Put a full copy on
> every Mac, let each Mac batch several conversations per forward pass, and hand
> the whole thing to your users as one model name.

Pipeline and tensor parallel (see [Inference modes](./inference-modes.md)) split
one model across the cluster so it fits. They serve **one request at a time**.
When the model already fits one Mac and what you need is *users*, not *size*,
use a **replica cluster**.

## What it does

- **One full copy per node.** Every Mac in the cluster loads the whole model and
  serves on its own. No collective between nodes, no RDMA wiring, no node cap.
- **One alias, one endpoint.** Clients call `/v1/chat/completions` with the pool
  alias as usual. OdyssAI-X picks the node.
- **Least-busy dispatch.** A request goes to the replica with the fewest
  requests in flight.
- **Session affinity.** A conversation stays on the replica that holds its KV
  cache — no transfer, no re-prefill. It moves only if that replica dies or is
  saturated.
- **Continuous batching inside each replica** (the **Batch** option). A node
  serves up to eight requests in the same forward pass instead of one after the
  other. Graceful degradation: a dead replica is dropped, the pool stays up as
  long as one replica lives.

## What to expect

Measured on four Mac Studio 256 GB, twenty simultaneous requests, ~100-token
answers:

| Model | Without Batch | With Batch |
|---|---|---|
| Qwen3.8 Flash Next 6-bit | 16.0 s wall · 124 tok/s aggregate · 5 waves | **9.0 s · 216 tok/s · 1 wave** |
| MiniMax M2.7 6-bit | — | **11.9 s · 254 tok/s · 60 tok/s per stream · 1 wave** |

Two things to know before you read those numbers as a promise:

- **Batching costs per-stream speed on some models.** Qwen3.8 Flash Next goes
  from 45 tok/s alone to 21 tok/s with five streams on the node — the node still
  delivers 2.4× more tokens. MiniMax M2.7 barely slows down. Hybrid attention
  models pay more than classic ones.
- **The ceiling is eight decoding sequences per node.** Beyond 8 × nodes
  simultaneous requests, the rest queue.

## Set it up

### 1. Create a replica cluster

Dashboard → **Add Argo cluster** → **Kind: `replica (1 copie par node, débit
concurrent)`**. No backend to pick, no wiring. Every node must hold the model
files locally, at the same path.

### 2. Load with Batch

Select the cluster, pick the model, tick **Batch — Continuous batching in
replica mode**, **Load**. The load fans out to every node of the cluster; the
loader shows bytes materialised per node. The pool row then carries the
**replica · batch** pill.

Leave the box unticked and the pool still works — one request at a time per
node. The box only appears on replica clusters.

### 3. Or use the API

```bash
curl -X POST http://<server>:8000/admin/clusters/<cluster>/load \
  -H 'Content-Type: application/json' \
  -d '{"model": "/Volumes/models/odysseus/odyssai/Qwen3.8-Flash-Next-Q6", "batch": true}'
```

The response lists `replicas[]` with `live` and `busy_count` per node, plus
`batch`. `/admin/clusters/<cluster>/status` exposes `pools[].is_replica` and
`pools[].batch`. The setting is persisted: a restart reloads the pool the same
way.

## Requirements and caveats

- **The model's MLX module must support batched positions.** Continuous
  batching hands the model one position offset *per sequence*. A model folder
  that ships its own `model_file` module written for single-stream serving fails
  every request at prefill (`arange(): incompatible function arguments`, zero
  tokens, `finish=error`). Prefer checkpoints without a bundled module, served
  by the `mlx_lm` port; if you must convert one, convert from the bf16 source
  rather than swapping the module (norm conventions differ between ports).
- **Memory per node** = one full copy + KV cache for the sequences in flight.
  The replica preflight uses the *smallest* node as the limit.
- **Not batched:** models whose caches have no batched merge (MiniMax-M3,
  DeepSeek-V4, Inkling), and any load with a drafter, DSpark or native MTP. Those
  fall back to one request at a time per replica.
- **Long generations.** The batched loop was pulled from production once
  (August 2026) for post-long-generation slowdowns and Metal buffer-cache growth.
  It is back as an explicit opt-in with a periodic cache purge. Watch a stream
  that crawls after a long answer, or node memory that keeps climbing.
- **Quantisation matters for speed.** Small-output projections (a few rows)
  should stay bf16; quantised, they halve the per-stream speed on Qwen3.8 Flash
  Next. The `odyssai/…-Q6` checkpoints follow that rule.

## Knobs

| Variable | Default | What it does |
|---|---|---|
| `RUNNER_BATCH_SIZE` | 8 | Sequences decoding together per node |
| `RUNNER_PREFILL_BATCH` | 4 | Prompts prefilled together per step |
| `BG_CLEAR_CACHE_EVERY` | 512 | Metal buffer-cache purge every N tokens |
| `REPLICA_MAX_CONCURRENT` | 16 | Affinity threshold before rebalancing |
| `REPLICA_AFFINITY_TTL_S` | 1800 | Idle lifetime of a session's affinity |

## Read next

- [Inference modes](./inference-modes.md) — when you need sharding instead.
- [The cluster](./cluster.md) — transports, wiring, topology.
- [HTTP API](./api.md) — the load endpoint and the rest of the surface.
