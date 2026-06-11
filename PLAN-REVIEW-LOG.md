# Plan Review Log: #40 JACCL stability (full package)

Pivot validated empirically (3× controlled unload+reload, 0 errno, wired freed, ~5s). Design written to PLAN.md, then cross-model adversarial review by Codex (read-only). MAX_ROUNDS=1 (ad-hoc, not a full grill loop).

## Round 1 — Codex (VERDICT: REVISE)

Codex raised 10 findings against the prod-automation plan. Verified the two highest-impact ones in code:

| # | Finding | Arbitration |
|---|---|---|
| 1 | `ring` backend not wired — `runner.py:1497` hardcodes `backend="jaccl"`; config-level `ring` is fiction. | **ACCEPT.** WU4 is not docs-only: must plumb backend through `remote_cmd` → runner `mx.distributed.init(backend=...)` + store `pool.backend`. Verified the hardcode. |
| 2 | Idle detection via `_active_runs` unsafe — Anthropic `/v1/messages` doesn't register runs → traffic looks idle. | **ACCEPT.** Foundational fix: a pool-level in-flight counter (`busy_count`) incremented in `RunnerPool.submit`/`prewarm`, used by ALL protocols. Replaces `_active_runs` for idle-gating. |
| 3 | TOCTOU between idle-check and reload/keepalive (request arrives in the gap). | **ACCEPT.** Pool `maintenance` flag: scheduler/keepalive set it, recheck `busy_count==0`, then act; `submit` refuses/queues while maintenance. |
| 4 | `broadcast_lock` only serializes stdin writes, not no-gen. | **ACCEPT** (same gate as #2/#3 — keepalive requires `busy_count==0` under the maintenance gate). |
| 5 | Partial broadcast silent — `send()` swallows `BrokenPipeError` → keepalive reaches some ranks → they block in all_sum. | **ACCEPT.** Keepalive broadcast must be fail-fast with per-rank send result; partial delivery → immediate degraded+recover, don't await a doomed all_sum. |
| 6 | `_cluster_reset` does cancel→stop→sweep, NOT reboot-all/reload. | **ACCEPT.** Verified (api.py:9505-9598). WU3 recovery for a hung QP needs: degraded → reboot-all (clears orphan QP that stop can't) → node-up wait → reload from captured snapshot. Build the explicit ladder; don't assume `_cluster_reset` recovers service. |
| 7 | Don't reuse `_auto_unload_cluster` (unloads EVERY pool + unlinks state). | **ACCEPT.** Alias-scoped reload under `get_admin_lock(cluster_id)`, preserving other pools. |
| 8 | Reload config snapshot must include ALL `RunnerPool` fields (use_ap, draft_model, num_draft_tokens, emit_batch, kv_q8, alias, node_indices). | **ACCEPT.** Snapshot every constructor field before stop; reload from snapshot, not `ArgoLoadRequest` defaults. |
| 9 | Pool state lacks `backend`; some topology checks use nonexistent `pool.loaded` (7918/7962). | **ACCEPT** for adding `pool.backend`. Note the `pool.loaded` reference for a separate hygiene check. |
| 10 | Keepalive delivery valid only if rank-0 emits with the id; implement `RunnerPool.keepalive()` like `prewarm()` with listener registered BEFORE broadcast. | **ACCEPT** (matches the `_listeners`/`_on_event` pattern at 1626/1646). |

### Claude's response
All 10 accepted — the review is correct and material. Net effect: the "full package" is larger and more prod-correctness-sensitive than the 8-pt estimate. Two findings change scope: WU4 (ring) needs real backend plumbing, and WU3 recovery needs an explicit reboot-all+reload ladder (not `_cluster_reset`). Findings #2/#3/#4 converge on a **foundational prerequisite**: a pool-level busy-counter + maintenance gate, which must land FIRST.

Revised implementation order:
0. **Foundational** — `RunnerPool.busy_count` + `maintenance` flag, incremented/checked in `submit`/`prewarm` across all protocols (OpenAI + Anthropic). (addresses #2/#3/#4)
1. **WU2 keepalive** — `RunnerPool.keepalive()` (listener-before-broadcast, per-rank send-result, fail-fast on partial) + runner `keepalive` cmd. (addresses #5/#10)
2. **WU1 scheduler** — alias-scoped reload from a full pool-field snapshot, gated on busy_count==0 + maintenance, age via `started_at`. (addresses #7/#8)
3. **WU3 recovery** — explicit ladder: consecutive-fail threshold → degraded → reboot-all → node-up wait → reload from snapshot. (addresses #6)
4. **WU4 ring** — plumb `backend` through remote_cmd → runner init + `pool.backend`; then document. (addresses #1/#9)

VERDICT carried: REVISE → re-scoped. Single round.
