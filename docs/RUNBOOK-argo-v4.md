# Argo/V4 recovery runbook

Operational runbook for bringing the Argo cluster back to a servable
state after an incident (stuck prefill, leaked wired memory, killed
collective run). Written from the recovery sequence rediscovered by
trial three times in one night on 2026-08-04 — this document exists so
a fourth rediscovery is never necessary.

Scope: Argo cluster, V4-family models (DeepSeek-V4 / V4-Pro, Kimi K3).
Pairs with `docs/DEPLOY.md` (container/deploy reference) and
`docs/HARDWARE.md` (node inventory) — this doc does not repeat cluster
addresses or topology; those are config-driven (see
`config/topology.example.yaml`) and must never be hardcoded here or in
any script.

## 1. Return to service after an incident

Proven sequence (succeeded 3 times on 2026-08-04):

1. Restart the engine container (`odyssai-odysseus`, Docker context
   `desktop-linux`) on the production Docker host.
2. Reboot every node in the Argo cluster (5 nodes).
3. Wait until each node satisfies the clean-state predicates in
   Section 2 — do not proceed on any node still failing them.
4. Issue an explicit `POST /admin/clusters/main/load`. Do not rely on
   auto-load.
5. Smoke-test with a single request before resuming normal traffic.

An unload/reload cycle that skips steps 1-2 is not sufficient after a
hard kill. The container and the nodes can both be left in a dirty
state (stale JACCL pool handles, leaked wired memory, orphaned GPU
pages) that a plain reload does not clear.

## 2. Clean-state predicates (per node, before any load or run)

Check all three on every node before trusting it to join a pool:

- **Uptime < 1 minute** — verify the *actual* uptime, not just that a
  reboot command returned success. A `sudo reboot` can silently fail
  to restart the node. `sudo shutdown -r now` has been reliable across
  all 3 recoveries on 2026-08-04 and is the preferred command.
- **Wired memory < 20 GB** (`vm_stat`). Anything higher means the
  previous process's memory was not actually released.
- **APIPA interface count == 4** — four `inet 169.254.x.x` link-local
  interfaces present. This is the address range JACCL routes over; it
  is absent for roughly the first 10 seconds after boot, so check
  after that window rather than immediately on boot completion.

## 3. The six-minute rule

A prefill that runs longer than 6 minutes is a failure, not something
to wait out. Kill it and reboot per Section 1.

This is an operator rule fixed on 2026-08-04: waiting past 6 minutes
has never once recovered a stuck prefill — it has only delayed the
reboot that was already needed. Treat 6 minutes as a hard timeout, not
a guideline.

## 4. Self-healing vs. operator action

After killing a standalone run that was mid-collective on native
JACCL, the run leaks wired memory (160-215 GB per node observed in
practice). In that situation, **always restart the engine container
before reloading a pool**.

Skipping the container restart risks the `jaccl-stability`
self-healing monitor triggering its own reboot cycle while a freshly
loaded pool is serving live requests — the monitor cannot distinguish
a stale pre-existing leak from a load in progress. Watch for this log
signature:

```
[jaccl-stability] leak recovery: [...]
```

Restarting the container first clears the monitor's leak-tracking
state, so it does not fire underneath the next load.

## 5. Forbidden actions

- **Never `pkill` a run that is mid native-collective.** The Metal GPU
  pages it holds are reboot-only to reclaim — a kill leaves the node
  in a state no in-process cleanup can fix.
- **Never run a heavy job** (dequantization, format conversion, or any
  similar batch job) **on a node belonging to a currently loaded
  pool.** It competes for the same memory and compute the pool depends
  on to serve requests.
- **Never hardcode an IP, host, or port** in scripts or docs — cluster
  topology is config-driven. See `config/topology.example.yaml` and
  `docs/DEPLOY.md`.
