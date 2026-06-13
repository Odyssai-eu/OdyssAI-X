# Plan: #40 — JACCL 45h stability (full package WU1–WU4)
_Pivot validated empirically 2026-06-09: controlled unload+reload resets RDMA without reboot (0 errno over 3 cycles, wired freed 16.5G→1.6G, ~5s reload). #35 is on-hold (upstream mlx)._

## Goal
Stop the recurring ~45h JACCL QP-degradation crash on Argo. Two levers, both leveraging existing orchestrator/runner infrastructure: (1) reduce the PROBABILITY of hitting the silent QPT_UC death by resetting QPs preventively before the degradation window; (2) bound the IMPACT when it happens anyway — detect a dying peer in minutes (not hours) and fire automatic controlled recovery instead of leaving a 187GB-wired orphan until a human notices.

Honest framing: WU2/WU3 do NOT make a dead-peer all_sum hang gracefully recoverable without a reset (UC has no timeout; a hung rank is stuck in C++ → SIGKILL → orphan QP → needs reboot-all to fully clear RDMA). What they buy is **early automatic detection + automatic controlled recovery**, converting "indefinite hang, manual discovery hours later, hours-long orphan" into "detected in ~minutes → auto reboot-all + reload → back in service". WU1 reduces how often we get there at all.

## Approach

### WU1 [2] — Preventive reload scheduler (api.py, orchestrator-only, low risk)
- New background asyncio task started at app startup (alongside the existing watchdog tasks).
- Every PREVENTIVE_RELOAD_CHECK_S (default 300s) iterate loaded pools. For each pool where:
  - backend is `jaccl` (skip `ring` — no QP bug), AND
  - `time.time() - pool.started_at` > PREVENTIVE_RELOAD_AGE_S (default 30h, configurable via global-settings), AND
  - cluster is NOT degraded, AND
  - **idle**: no entry in `_active_runs` for this cluster,
  → controlled unload+reload of that exact pool (same model/node_indices/mode/alias), logging before/after. Reuses the proven controlled path (pivot: ~5s, clean).
- Idle-gating is the safety: never reload mid-generation. If never idle in the window, retry next tick (defer a busy cluster).

### WU2 [3] — Keepalive (runner.py + api.py)
- **runner.py** (minimal, mirrors the `prewarm` cmd at L1714): handle `{"cmd":"keepalive","id":...}` → ALL ranks call `mx.distributed.all_sum(mx.array([1.0]), group)` + `mx.eval`; rank 0 emits `{"event":"keepalive_ok","id":...,"rtt_ms":...}`. Single-node (size==1): no-op ok. Collective-safe because the orchestrator broadcasts it to every rank exactly like `gen`/`prewarm`.
- **api.py**: background task, every KEEPALIVE_INTERVAL_S (default 90s), for each idle jaccl pool: broadcast `keepalive` to all ranks (existing `broadcast_lock` + per-runner stdin), await the `keepalive_ok` event with timeout KEEPALIVE_TIMEOUT_S (default 20s). Only when idle (don't perturb an active gen; gen forwards already exercise the QPs).

### WU3 [5] — Early-warning → controlled recovery (api.py, highest stakes — false-positive guarded)
- If a keepalive does NOT get its `keepalive_ok` within timeout → a peer is hung/dead.
- **False-positive guard**: require KEEPALIVE_FAIL_THRESHOLD (default 2) CONSECUTIVE timeouts before acting — a single transient (orchestrator GC pause, scheduling hiccup) must not reboot a healthy cluster.
- On threshold breach: `_mark_cluster_degraded(cluster, "keepalive timeout — peer unresponsive")` + fire `_cluster_reset(cluster)`, deduped via the existing `_WATCHDOG_RECOVERY_BY_CLUSTER` map (mirror the no-progress watchdog at L1954-1968 exactly). `_cluster_reset` already runs the controlled recovery ladder (cancel → stop → sweep → reboot-all if needed → reload).
- A successful keepalive resets the consecutive-fail counter to 0.

### WU4 [2] — Ring fallback documented (docs)
- Document `ring` (TCP) backend as the explicit per-cluster option for runs where >24h stability outweighs throughput (no QP bug). Where: docs/PRODUCTION.md + the backend field. No code (backend switch already exists, api.py L593 validation).

## Key decisions & tradeoffs
- **Orchestrator-driven keepalive, not a runner background thread.** Collectives are synchronous across all ranks; an independent per-rank thread calling all_sum would desync with the gen loop and deadlock. Broadcasting a `keepalive` cmd reuses the proven gen/prewarm lockstep delivery. Cost: keepalive only runs when idle (fine — busy pools exercise QPs anyway).
- **Idle-gating everywhere.** Both the preventive reload and the keepalive only act on idle pools. Never perturb an in-flight generation.
- **Consecutive-failure threshold on WU3.** The expensive action (reboot-all) needs ≥2 consecutive keepalive misses. Tunable. Main false-positive defense.
- **Reuse, don't reinvent recovery.** WU3 hooks the EXISTING `_cluster_reset` + degraded-flag + dedup map. No new recovery path.
- **WU2 runner change is additive and minimal** (one new cmd handler, no change to the gen/teardown paths).

## Risks / open questions
- **False-positive reboot (WU3).** Mitigated by the consecutive-fail threshold + idle-only keepalive. Open: is 2 enough, or 3? Lean conservative if reboot-all is disruptive.
- **Keepalive on a hung pool blocks the surviving ranks in all_sum.** Expected — that's the detection. Confirm `_cluster_reset`/reboot-all doesn't itself wait on the hung runner's graceful stop indefinitely (reboot-all should be a hard reset). VERIFY in code.
- **Scheduler reload picks the wrong pool config.** Must capture the EXACT live load params (model, node_indices, mode, alias, kv_q8), not a guessed default. VERIFY the pool object carries these.
- **Broadcasting keepalive racing a concurrent submit.** broadcast_lock serializes; idle-gating means no concurrent gen. Confirm lock scope.
- **48h endurance done-criterion** can't be validated in-session; needs a long Argo run with WU1 active. Follow-up.

## Out of scope
- Patching libjaccl / MLX upstream — contournement par lifecycle, comme exo.
- Removing reboot-all — stays the ultimate safety net (and WU3's recovery action).
- Changing the default backend — `jaccl` stays default; `ring` is the documented long-run opt-in.
