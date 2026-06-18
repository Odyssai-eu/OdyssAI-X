# 1. RDMA node onboarding: vendored exo recipe, node-local, health-asserted

Date: 2026-06-09. **Rewritten 2026-06-10** after the exo source analysis (see
`Odyssai-config/RAPPORT-exo-analyse.md`) — the original premises rested on a
misdiagnosis and are superseded.

Status: Accepted (supersedes the 2026-06-09 version)

## Context

Adding the 5th Argo node (a 256GB Mac Studio) went badly: the node lost its
management IPv4 and its Wi-Fi during a from-scratch network-setup script run
**over SSH**, and was recovered only by an OS reinstall. The original version of
this ADR — written mid-incident — concluded that the onboarding step must never
touch `en0`, Wi-Fi, network locations, or preferences.

**Superseded diagnosis.** The final root cause of the IPv4 loss was a **dead
switch-uplink cable** (proven: an unrelated MacBook went APIPA at the same
moment, and the node re-acquired its address the instant the uplink was fixed). The
script's real faults were narrower: it did not recreate Wi-Fi in its new
location (losing the out-of-band lifeline), it accepted an APIPA address as
success, and it ran over SSH with no operator at the console.

**The proven recipe.** The 4 production ultra nodes run exo's network
setup to this day (verified live 2026-06-10): a dedicated network **location
"exo"** containing recreated services for every port (Ethernet, Wi-Fi, one DHCP
service per Thunderbolt port), `bridge0` torn down and removed from the SC
prefs, the Thunderbolt Bridge service disabled, all re-asserted by a root
LaunchDaemon (`io.exo.networksetup`, RunAtLoad + every 1786 s). The per-TB-port
services are what make each port carry an IPv6 link-local (fe80) — the address
family JACCL GIDs and our NDP-based wiring discovery consume. A "minimal
toggle" (only disabling the bridge service) is unproven and likely insufficient
(a service-less port may stay address-less).

## Decision

1. **Vendor exo's recipe, do not reinvent it.** The onboarding step is
   `scripts/odyssai-network-setup.sh` — exo's embedded daemon script (Apache
   2.0, commit `09f9ea313f72e261f40a94cea4c0e3681b31af23`) with Odyssai
   identity (location `odyssai`, services `Odyssai Thunderbolt N`, daemon
   `eu.odyssai.networksetup`) and targeted fixes (bridge service resolved
   dynamically **before** teardown — exo hardcodes the English name; bridge
   port skipped by device; management + Wi-Fi services recreated **first**;
   guarded `setdhcp`; exact-match idempotence guards; kill-switch marker).
2. **`en0` and Wi-Fi are protected by mandatory post-apply health assertions,
   not by untouchability.** The recipe *recreates* both services in the new
   location (as exo does). The driver (`scripts/rdma-onboard.sh`) must assert,
   with a poll-and-deadline (≥120 s), that after the switch: `en0` holds a
   **non-APIPA** lease and the default route exists; and report Wi-Fi
   re-association. An APIPA on the management NIC is always a failure.
3. **Node-local only, operator present, never SSH.** The initial apply runs at
   the machine's console (or Screen Sharing) — the driver refuses when
   `SSH_CONNECTION` is set. The installed daemon's **unattended idempotent
   re-asserts are explicitly authorized** (that is the persistence mechanism).
4. **Fail loudly, never half-done.** Exit-code contract 0/10/11/1; at most ONE
   reboot proposal; `applied-no-daemon` is a derived, recoverable state; revert
   restores the **persisted** prior location (never a hardcoded name) and
   proves it before dismantling anything.
5. **Never two network daemons.** Applying on a machine that still has exo's
   daemon/location is blocked until exo is migrated off.

## Consequences

- Onboarding = `sudo rdma-onboard.sh --apply --console` (or
  `odyssai-configure node-setup network` from the Configurator DMG), at the
  console of the target Mac.
- Verification asserts management/Wi-Fi health *because* the recipe touches
  them; the old "never asserts anything about en0/Wi-Fi" consequence is
  superseded.
- The recipe is re-asserted forever by the daemon; opting out = the
  `network-setup.disabled` kill-switch or `--revert`.
- `ring` remains the fallback transport when the mesh cannot come up
  (INSTALL-CLUSTER.md rule) — an explicit operator choice, never silent.
