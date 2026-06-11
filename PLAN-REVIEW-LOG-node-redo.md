# Plan Review Log: RDMA node onboarding — refait depuis exo
Act 1 (grill-with-docs) complete — plan locked, CONTEXT.md + ADR-0001 updated.
Iron constraints (gravé dans le marbre): never disable Wi-Fi, never modify en0.
MAX_ROUNDS=5. PLAN_FILE=PLAN-node-redo.md.

## Round 1 — Codex

Material blockers:

- `ifconfig bridge0 destroy` is not TB-scoped; if `bridge0` contains `en0`, Wi-Fi, or a stale bad config, it can still cut management. Fix: inspect `ifconfig bridge0` members first and abort unless every member is a verified Thunderbolt interface and none is `en0`/Wi-Fi.
- “Retrait persistant de bridge0 des prefs” is dangerous: direct plist edits can rewrite service order/current set and bounce `configd`. Fix: do not edit prefs in v1; use LaunchDaemon-only teardown, add plist surgery only if exo proves it is required.
- If plist editing remains, the plan lacks a strict diff contract. Fix: mutate an offline copy, `plutil -lint`, compare before/after, and allow only removal of the exact Thunderbolt Bridge service UUID plus its references.
- “Wrapper exo” is not safe until audited; exo is a reference, not a trusted binary dependency. Fix: first retrieve `disable_bridge.sh`, statically scan for `scselect`, `networksetup`, `en0`, Wi-Fi/airport, location, and plist writes; abort on any match outside the allowlist.
- The plan says “leave node exactly as found” after failure, but runtime bridge destroy and daemon install are already mutations. Fix: use two-phase flow: preflight, runtime-only test, postcheck, then persist; uninstall daemon on any failed persistence postcheck.
- Boot race is unresolved: bridge0 may reappear after RunAtLoad and before JACCL starts. Fix: daemon must retry with settle/backoff until bridge0 is absent and TB HCAs are active for N consecutive samples.
- LaunchDaemon semantics are underspecified. Fix: define `/Library/LaunchDaemons` label, root:wheel ownership, `0644` plist, executable `0755`, `RunAtLoad`, `StartInterval`, logs, `launchctl bootstrap system`, `kickstart`, and verification.
- The plan does not gate Odysseus/JACCL load on RDMA readiness. Fix: add a pre-load check that rejects `backend=jaccl` if expected `rdma_to` HCAs are missing or not `PORT_ACTIVE`.
- “Ports TB: IPv6 link-local seul, jamais IPv4” may tempt networksetup mutations on TB services. Fix: do not configure TB IP methods at all; success criterion is per-port `inet6 fe80` and HCA `PORT_ACTIVE`, ignoring TB IPv4/APIPA.
- “APIPA = échec” is overbroad and conflicts with TB behavior. Fix: treat `169.254.x` as failure only on the management interface, not on Thunderbolt ports.
- Interface identification is vague; `enN` numbering changes and “Thunderbolt” grep is brittle. Fix: identify TB ports from `ibv_devinfo rdma_en* -> en*` plus `ifconfig`/IORegistry, not from guessed names.
- Management protection is too narrow if only hardcoding `en0`. Fix: snapshot the active management route/interface/service UUID before mutation and protect both that interface and literal `en0`.
- Wi-Fi protection is declarative, not verified. Fix: pre/post snapshot Wi-Fi power, service UUID, device, SSID/IP if associated; abort if Wi-Fi is off before starting.
- SSH could be riding the interface being modified. Fix: before mutation, `route get <ssh-client-or-orchestrator-ip>` and abort unless the route uses protected management or Wi-Fi, never bridge/TB.
- Existing `bridge0` might not be the macOS Thunderbolt Bridge service. Fix: require both service identity/name and member validation before touching it.
- Observability is insufficient for a brick-risk operation. Fix: emit a JSON report with pre/post `scselect`, services, service order, `ifconfig bridge0`, TB `ifconfig`, Wi-Fi state, routes, `ibv_devinfo`, launchctl status, and exact mutations.
- No reboot validation is required before declaring success. Fix: require apply -> reboot -> SSH reachable -> Wi-Fi still on -> no bridge0 -> all expected HCAs active -> discovery full mesh.
- The plan assumes exo exists on healthy nodes and maps to this repo’s current Odysseus flow. Fix: vendor the audited minimal script into this repo or document the exact source hash; do not runtime-depend on exo being installed.
- Simpler safer path is not prioritized. Fix: implement daemon-only teardown first, prove it across reboot, and leave prefs untouched unless empirical evidence shows macOS re-creates bridge0 despite the daemon.
- Rollback backup is mentioned only for prefs, not daemon/runtime state. Fix: define explicit cleanup: unload/remove daemon, restore plist only from validated backup, and never attempt to recreate bridge0 remotely.

VERDICT: REVISE
### Claude's response (round 1)

Accepted essentially all findings — they sharpen the plan and are consistent with ADR-0001. Folded into PLAN-node-redo.md: (1) preflight member-inspection of bridge0, abort if any member is en0/Wi-Fi/non-TB; (2) daemon-only teardown in v1, NO prefs edit unless exo proves it required (then strict offline-diff contract); (3) static audit of exo's disable_bridge.sh against an allowlist before wrapping, and VENDOR the audited minimal script into the repo (no runtime dependency on exo being installed); (4) two-phase apply: preflight -> runtime-only test -> postcheck -> persist, uninstall daemon on failed persistence; (5) boot-race: daemon retries with settle/backoff until bridge0 absent AND TB HCAs PORT_ACTIVE for N samples; (6) full LaunchDaemon spec; (7) gate JACCL load on RDMA readiness; (8) do NOT configure TB IP methods at all — success = per-port inet6 fe80 + HCA PORT_ACTIVE; (9) APIPA=failure ONLY on management iface, not TB; (10) identify TB ports via ibv_devinfo rdma_en*->en*, not name grep; (11) protect active mgmt route/service-UUID + literal en0 via snapshot; (12) Wi-Fi pre/post snapshot, abort if Wi-Fi off at start; (13) SSH/orchestrator route guard — abort unless route uses protected mgmt/Wi-Fi; (14) verify bridge0 identity is the macOS Thunderbolt Bridge; (15) JSON pre/post observability report; (16) reboot-gated success criteria; (17) rollback cleanup for daemon/runtime, not just prefs. Nothing rejected.

## Round 2 — Codex

Round-1 findings are mostly addressed: no prefs in v1, no network location, TB/member validation, pre/post snapshots, daemon-only persistence, reboot gate, JACCL load gate, no TB IP config.

Remaining material issues:

- Phase 1 rollback is unsafe/undefined: “revert du teardown runtime” implies recreating `bridge0` remotely, which conflicts with “never recreate bridge0 remotely” and can re-enter the dangerous path. Fix: do not recreate `bridge0`; on runtime failure, leave TB de-bridged, remove any installed artifacts, emit report, and require manual console only if restoration is truly needed.
- If `en0` or Wi-Fi changes after runtime teardown, the plan says “revert runtime + abort”; that performs another network mutation after detecting management drift. Fix: on management/Wi-Fi drift, stop all further network mutations and report immediately.
- Wi-Fi guard only requires “power on”; that is not a usable out-of-band lifeline. Fix: require Wi-Fi on + associated + has non-APIPA IP or explicitly mark “no OOB” and abort before mutation.
- `route get <ip-orchestrateur>` may test the wrong peer if the script is run from a laptop, CI box, or existing SSH session. Fix: validate route to both the intended orchestrator IP and the active SSH client from `$SSH_CONNECTION`.
- Phase 0 TB identification still leans on `ibv_devinfo`; on a bridged/bad node, HCAs may be missing or inactive before teardown. Fix: identify TB member interfaces from IORegistry/SystemConfiguration first, use `ibv_devinfo` only as postcheck/gate.
- “en0 inchangé (même service/bail)” is too strict and may false-fail on DHCP lease renewal, triggering rollback logic. Fix: compare service UUID/device, IP not APIPA, router, and route; do not require identical lease metadata.
- “Fallback ring sinon” under JACCL load gate can silently mask RDMA failure and run over management LAN. Fix: reject `backend=jaccl` load with an explicit health error; ring must be an operator-selected config change.
- Daemon retry loop has no ceiling or degraded state. Fix: cap retries per boot cycle, then leave a clear failed status/log; keep StartInterval for later retry without tight looping forever.
- Success criterion checks SSH `.3x` and Wi-Fi on, but not that the active SSH/management route remained off TB after reboot. Fix: add post-reboot `route get` verification that management SSH still uses protected management/Wi-Fi, never TB/bridge.

VERDICT: REVISE
### Claude's response (round 2)

Accepted all 9 refinements: (1) Phase-1 failure never recreates bridge0 — leave TB de-bridged, remove artifacts, report, physical-console-only restore; (2) on mgmt/Wi-Fi drift, STOP all network mutations and report immediately (no revert mutation); (3) Wi-Fi guard requires on+associated+non-APIPA IP (real OOB) or explicit no-OOB abort; (4) SSH-route guard checks both the orchestrator IP and $SSH_CONNECTION client; (5) TB identification via IORegistry/SystemConfiguration first, ibv_devinfo only as postcheck/gate; (6) en0-unchanged compares service UUID/device + non-APIPA IP + router + route, NOT lease metadata (no DHCP-renewal false-fail); (7) JACCL load gate raises an explicit health error, ring is operator-selected only (never silent fallback over mgmt LAN); (8) daemon retry capped per boot cycle with a degraded failed-status, StartInterval for later retry; (9) post-reboot route get verifies mgmt SSH still rides protected mgmt/Wi-Fi, never TB/bridge. Nothing rejected.

## Round 3 — Codex

Prior round-2 findings are addressed. The plan now blocks the original brick path: no location switch, no prefs edit in v1, no TB IP config, no bridge recreation over SSH, Wi-Fi must be usable, management/SSH routes are guarded, and JACCL load fails explicitly instead of falling back.

Remaining issues are wording/implementation traps, not material blockers:

- Phase 1 is still titled “mutation réversible”, but rollback says never recreate `bridge0`; fix the wording to “mutation non-persistente, non-restaurée à distance”.
- Phase 2 rollback says “node = état phase 0”, which is false after Phase 1 de-bridged `bridge0`; fix to “daemon removed, network left in current de-bridged state, report emitted”.
- If `bridge0` is already absent at preflight, the plan does not define idempotent behavior; fix: treat as runtime-success only if TB postcheck passes, then install/verify daemon.
- The “no-OOB explicite” exception is dangerous if automated; fix: require physical-console confirmation for that override, not just a flag.
- “TB actif” should be tied to expected peer count/rank topology; fix: postcheck expected HCAs from planned mesh, not every possible TB port on the Mac.

VERDICT: APPROVED
### Claude's response (round 3) — CONVERGED

APPROVED. Folded the 5 non-blocking wording/safety refinements: (1) Phase 1 retitled 'mutation non-persistante, non restaurée à distance'; (2) Phase 2 + rollback now say network stays in current de-bridged state (never recreate bridge0), not 'état phase 0'; (3) idempotent path when bridge0 already absent at preflight — runtime-OK only if TB postcheck passes, then install/verify daemon; (4) 'no-OOB' override requires physical-console confirmation, not a flag; (5) TB postcheck scoped to expected HCAs per planned mesh topology, not every TB port on the Mac. Converged in 3 rounds.

## Round 4 — Codex (review of the CODE: scripts/rdma-onboard.sh)

Findings:

- `scripts/rdma-onboard.sh:184-193` — `mgmt_fingerprint()` does not include the management IP and treats empty IP as `apipa=no`, so `en0` can lose/change its DHCP lease without drift detection; fix: include exact non-empty IP, device, service identifier, router, and route in the fingerprint.
- `scripts/rdma-onboard.sh:66-68,137-143,226-231` — the code claims to protect literal `en0` but only checks `$mgmt`; if `mgmt_device()` is not `en0`, a bridge containing `en0` is not hard-aborted; fix: always reject members equal to `$mgmt`, `en0`, and `$wifi`.
- `scripts/rdma-onboard.sh:66-68,164-175` — management identity is hardcoded as Hardware Port `Ethernet`, not the active management route/session interface; fix: derive the protected management iface from route to `$SSH_CONNECTION`/orchestrator plus literal `en0`.
- `scripts/rdma-onboard.sh:164-175,337` — if `SSH_CONNECTION` is stripped by `sudo` and no orchestrator IP is passed, the live SSH-route guard silently no-ops; fix: require at least one route peer for `--apply`, unless a physical-console-only mode is explicitly implemented.
- `scripts/rdma-onboard.sh:138-145` — preflight never verifies that `bridge0` is the macOS “Thunderbolt Bridge” service, only that members look TB; fix: map `Device: bridge0` from service order/SystemConfiguration and abort unless its service identity is exactly Thunderbolt Bridge.
- `scripts/rdma-onboard.sh:221-235` — TOCTOU remains between member validation/deletion and `ifconfig bridge0 destroy`; fix: immediately before destroy, re-read current members and abort unless all remaining members are verified TB and not `en0`/Wi-Fi.
- `scripts/rdma-onboard.sh:232-235` — mutation failures are swallowed with `|| true` and the script logs “bridge0 torn down” even if delete/destroy failed; fix: capture failures, log exact failing command, and return nonzero unless bridge0 is already absent.
- `scripts/rdma-onboard.sh:73-85` — TB identification depends on `Hardware Port: Thunderbolt [0-9]`, which can miss bridged/default macOS Thunderbolt interfaces and localized/non-numbered names; fix: identify TB `en*` from IORegistry BSD Name/Thunderbolt parent first, with SystemConfiguration only as corroboration.
- `scripts/rdma-onboard.sh:150-158` — `assoc="$(... | grep -ci ...)”` can exit the script under `set -euo pipefail` before `die()` logs the intended failure; fix: append `|| true` inside the command substitution or use an `if networksetup ... | grep -qi ...; then` guard.
- `scripts/rdma-onboard.sh:169-170,212-213` — `route ... | awk` and `ibv_devinfo | awk` assignments can abort under `set -euo pipefail` before controlled checks run; fix: make these probes failure-tolerant with `|| true` and explicitly validate empty output.
- `scripts/rdma-onboard.sh:248-249` — post-teardown Wi-Fi drift check only verifies power `On`, not associated/non-APIPA IP as the plan requires; fix: reuse the full Wi-Fi lifeline check after teardown.
- `scripts/rdma-onboard.sh:200-218` — `tb_postcheck()` validates counts, not expected topology/HCA names, so the wrong active HCAs can satisfy `EXPECT`; fix: accept expected `rdma_en*`/peer mapping or run discovery and validate the planned mesh, not just counts.
- `scripts/rdma-onboard.sh:25-28,44-49,333-340` — usage advertises `--expect N`, but the parser does not implement it and treats `--expect` as the orchestrator IP; fix: add real option parsing for `--expect` and orchestrator IP.
- `scripts/rdma-onboard.sh:97-124` — snapshot JSON is built with raw `echo` and unescaped command output, so quotes/backslashes in location, service, SSID, or logs produce invalid JSON; fix: generate snapshots via `python3 -c 'json.dump(...)'` or `jq -n --arg`.
- `scripts/rdma-onboard.sh:298-316` — daemon mode runs teardown without the preflight service-identity and route/Wi-Fi guards, relying only on member checks; fix: add a daemon-safe preflight that at least validates bridge identity, literal `en0`/Wi-Fi exclusion, and TB identity before each mutation.
- `scripts/rdma-onboard.sh:31-33,80-85` — PLAN WU1 requires audited/vendored exo behavior before use, but the script ships “faithful equivalent” unaudited; fix: vendor the audited exo teardown or block `--apply` until the audit hash/source is present.

VERDICT: REVISE
### Claude's response (round 4)

Accepted all 16 code findings; rewriting scripts/rdma-onboard.sh to address: mgmt fingerprint includes exact non-empty IP (empty=drift); protected set = literal en0 + mgmt-port device + route ifaces to all peers, members rejected against the full set; bridge0 identity verified == 'Thunderbolt Bridge' service; --apply requires at least one route peer (SSH_CONNECTION or orchestrator) unless --console; bridge0 service-identity + member re-validation immediately before destroy (TOCTOU); teardown checks bridge0 actually gone, no success-claim on swallowed failure; TB id via IORegistry Thunderbolt/USB4 ancestry first, hardware-port as corroboration, excluding en0/wifi/bridge; all pipe probes made set -e-safe with || true + explicit empty handling; post-teardown reuses the FULL Wi-Fi lifeline check (on+associated+non-APIPA); real --expect parsing; snapshot JSON emitted via python3 json.dumps (proper escaping); daemon-run runs a daemon-safe guard (bridge identity + en0/wifi/TB validation) before each mutation; --apply blocked unless an exo-audit hash marker is present. Nothing rejected.

## Round 5 — Codex (code review #2)

Findings:

- `scripts/rdma-onboard.sh:229-247` — preflight does not require management `en0` to currently have a non-empty non-APIPA IP before mutation, so an already-broken/empty management lease can be captured as the “before” state and pass drift checks; fix: abort preflight unless the protected management interface has exact IP + non-APIPA + service + route/router.
- `scripts/rdma-onboard.sh:175-183` — `mgmt_fingerprint()` encodes empty IP as `apipa=no`, which makes “no IP before/no IP after” look healthy; fix: include `healthy=yes/no` where empty or APIPA is unhealthy, and require `healthy=yes` before Phase 1.
- `scripts/rdma-onboard.sh:235-236` — abort path expands `$SSH_CONNECTION` under `set -u`, so no SSH peer can crash with “unbound variable” instead of controlled `die`; fix: use `${SSH_CONNECTION:-}` in the message.
- `scripts/rdma-onboard.sh:213-218` — TOCTOU recheck is before the `deletem` loop, not immediately before `destroy`; a protected member could appear between deletion and destroy; fix: re-run `validate_bridge_safe` after the `deletem` loop and immediately before `ifconfig bridge0 destroy`.
- `scripts/rdma-onboard.sh:188-204` — `--expect N` still validates only counts, not the planned topology/HCA names, so wrong cables/peers can pass onboarding; fix: accept expected `rdma_en*` names or peer map and validate those exact HCAs.
- `scripts/rdma-onboard.sh:230,341-344` — `--apply` is blocked by the exo audit marker, but direct `--daemon-run` can still perform unaudited teardown; fix: require the audit marker in `daemon_run` too, or only allow daemon mode when invoked from the installed audited script path.
- `scripts/rdma-onboard.sh:71-80` — IORegistry lookup still uses `ioreg -r -n "$d"`, which may not find interfaces by BSD name on macOS; if fallback hardware-port names miss bridged TB ports, safe bridges are falsely rejected; fix: query IORegistry for `BSD Name == enX` and inspect parent ancestry.
- `scripts/rdma-onboard.sh:333-338` — option parsing allows `--expect`/`--orchestrator` without a value and then `shift 2` can abort unclearly; fix: validate required option arguments and numeric `EXPECT` before shifting.
- `scripts/rdma-onboard.sh:262-281` — LaunchDaemon plist is written and bootstrapped without `plutil -lint`; malformed XML or bad values leave a root plist artifact before failure; fix: lint the plist before `launchctl bootstrap`, removing it on lint/bootstrap failure.
- `scripts/rdma-onboard.sh:121-145` — snapshot generation silently succeeds/logs even if `python3` fails or the JSON file was not written; fix: make snapshot failure explicit for `--check/--apply`, or at least log `snapshot failed` with nonzero status.

VERDICT: REVISE
### Claude's response (round 5)

Applied 9/10: mgmt_fingerprint now carries healthy=yes/no (empty or APIPA IP = unhealthy) and preflight+phase1 require healthy=yes before any mutation; ${SSH_CONNECTION:-none} in the die message (no set -u crash); validate_bridge_safe re-run after the deletem loop, immediately before destroy (TOCTOU); daemon_run requires the exo-audit marker too; is_tb_device uses hardware-port Thunderbolt match as primary + IORegistry BSD-Name ancestry as corroboration; --expect/--orchestrator validate their argument (numeric/non-empty); install_daemon runs plutil -lint and removes the plist on lint/bootstrap failure; snapshot logs an explicit WARNING on python3/json failure. DEFERRED (arbiter decision, finding #5): --expect remains a fast count-based gate; full per-HCA-name topology validation is intentionally delegated to the reboot-gated discover-rdma-wiring.py step already in the success criteria, to avoid duplicating the authoritative mesh check in two places. Will surface this as a counter-position if re-raised.

## Round 6 — Codex (code review #3) — CONVERGED

No material findings. Prior blockers are fixed in the script: no network location/prefs/IP-method mutation path, protected device checks cover `en0`/mgmt/Wi-Fi/route ifaces, bridge identity + member validation happen before mutation and before destroy, live-route guard is enforced, management/Wi-Fi drift checks are now meaningful, daemon mode is marker-gated and guarded, snapshots are valid JSON with explicit failure logging, and `--expect` as a fast count gate is acceptable with `discover-rdma-wiring.py` as the authoritative reboot-gated mesh validation.

VERDICT: APPROVED
VERDICT: APPROVED. Plan (3 rounds) + code (3 rounds) both converged. 6 Codex rounds total.
