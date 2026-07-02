# PLAN — Distributed VLM serving (branch `vlm-distributed`)

> Handoff for the distributed-VLM effort. The single-node VLM kind is DONE and
> live (main, OdyssAI-X v1.8.1: `/admin/vlm/load|unload|status`, engine spawns
> `mlx_vlm.server` on one Argo node + telemak http-proxy). This branch adds the
> DISTRIBUTED (multi-node tensor-parallel) path, for a VL model too big for one
> node. Grounded in the G4 feasibility evidence (2026-07-02).

## Why (scope)

Today every VLM = one node. The biggest VL we serve (MiniMax-M3 427B in 4/6-bit,
~240–330 GB) fits one ultra-512, so single-node is enough. Distributed is only
needed for a VL that does NOT fit one node — e.g. a **Q8 VL (~454 GB)**, or a
future larger VLM. Do NOT build this until such a model is actually wanted;
the single-node kind covers current needs.

## What upstream (Blaizzy/mlx-vlm @ecc457b, installed) ALREADY ships

- `models/minimax_m3_vl/language.py` has a **complete tensor-parallel `shard()`**:
  attention q/k/v (`all-to-sharded`), o_proj (`sharded-to-all`), the MSA indexer
  (sharded, with a divisibility guard), and the MoE `switch_mlp` experts
  (`shard_inplace` + `mx.distributed.all_sum` on the block output). Router gate,
  embed_tokens, lm_head, norms, shared_experts = replicated.
- `mlx_vlm.utils.sharded_load(model_path, tensor_group=g)` uses the memory-safe
  **lazy-load → shard → eval-this-rank-only** pattern (mmap; each rank
  materializes only its slice of the LM). A 256 GB node CAN join a 241 GB model.
- Max TP degree = 4 (MiniMax-M3 has 4 KV heads → group size must divide 4).

## What is MISSING (the work)

1. **`Model.shard()` delegator (2 lines).** `sharded_load` gates TP on
   `hasattr(model, "shard")` at the TOP-LEVEL Model, but `minimax_m3_vl.Model`
   lacks it (unlike `kimi_k25.py:134` / `qwen3_vl_moe.py:189` which have the
   2-line `def shard(self, group=None): self.language_model.shard(group)`).
   Without it `sharded_load(..., tensor_group=g)` raises ValueError.
   → Add it upstream (fork) OR call `model.language_model.shard(group)` directly
   from our runner. **This is the smallest unblock + a good upstream PR.**
2. **No distributed serving entrypoint.** `sharded_load` has ZERO callers in the
   package; `mlx_vlm.server`/`generate`/`chat` know nothing about `mx.distributed`.
   → Write a distributed VLM runner (mirror our `scripts/runner.py`: `mx.distributed.init(backend=...)`, `sharded_load(repo, tensor_group=group)`, then a serving loop). Expose an OpenAI HTTP endpoint on **rank 0** so it plugs into the existing telemak http-proxy (same as the single-node kind).
3. **Vision tower is NEVER sharded** (`minimax_m3_vl.py`/`vision.py` have no shard/
   distributed code). Each rank holds + runs a **full replica** of the vision
   tower + projectors. That's ~1.7 GB (negligible) — acceptable to replicate.
   Confirm the image pre-embeddings are computed consistently across ranks (run
   vision on rank 0 and broadcast, OR run identically on all ranks — verify no
   divergence). This is the main correctness unknown.

## Engine integration (mirror the single-node kind)

The single-node kind (main) is the template: `/admin/vlm/load` SSH-spawns one
`mlx_vlm.server` + registers a telemak http-proxy cluster. For distributed, the
engine must instead spawn the distributed VLM runner across N nodes (like it
spawns `runner.py` for text pools via `RunnerProc` — ssh Popen per node,
`mx.distributed` init inside), rank-0 serving HTTP, exposed as the proxy upstream.
Likely a new kind (`mlx-vlm-distributed`) or a `nodes: [>1]` variant of the VLM
load endpoint. Reuse: `RunnerProc` ssh-spawn + orphan-sweep kill, the telemak
proxy routing, `install-mlx-vlm.sh` (mlx-vlm venv already on every node).

## Verification gates (per Sophie's rules — empirical, end-to-end)

1. Offline: `sharded_load` a VL model across 2 nodes (.30+.31, ring backend to
   start — JACCL only once stable), single-image forward → coherent caption.
   Prove the 2-line delegator + vision replication produce correct output vs the
   single-node baseline (same image, compare).
2. Server: distributed runner serves on rank-0 HTTP, curl image → coherent.
3. E2E: through OdyssAI-X (`m3vl-dist` cluster) → coherent image answer.
4. Perf sanity: distributed TP should not be dramatically slower than single-node
   (watch for the MoE-gather wall seen on the Swift side; different stack here).

## Constraints / gotchas

- Backend: start with **ring (TCP)** — stable. JACCL (RDMA) has the queue-pair
  degradation bug (reboot to reset); only after ring works.
- The full checkpoint must exist on **every** participating node's local
  `/Volumes/models/odysseus` (mmap per rank). Copy cost for a 240–450 GB model.
- TP degree divides 4 (KV heads). 2-node and 4-node OK; 3-node NOT (4 % 3 ≠ 0).
- Effort ≈ **5–8 pts**. Single biggest risk = vision-replication correctness +
  first-load bring-up of the day-old upstream distributed path.

## Status

Branch created off main (single-node VLM kind = v1.8.1, live). No code yet —
this is the map. Assign Fable.
