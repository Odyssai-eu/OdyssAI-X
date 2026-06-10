#!/usr/bin/env bash
#
# odyssai-network-setup.sh — make this Mac's Thunderbolt ports RDMA/JACCL-ready.
#
# Portions derived from exo (https://github.com/exo-explore/exo), specifically the
# LaunchDaemon setup script embedded in app/EXO/EXO/Services/NetworkSetupHelper.swift
# (lines 16-69), Copyright 2025 Exo Technologies Ltd, Apache License 2.0.
# Source commit: 09f9ea313f72e261f40a94cea4c0e3681b31af23 (2026-06-03). MODIFIED:
# Odyssai identity (location/services/support dir), dynamic bridge-service resolution
# before teardown (fixes exo's hardcoded-English-name locale bug), bridge port skipped
# by device not by localized name, management+Wi-Fi services recreated before the TB
# ports, guarded setdhcp, exact-match idempotence guards, kill-switch marker,
# ODYSSAI_SKIP_SLEEP knob.
#
# WHAT IT DOES (the recipe proven on the 4 production ultras, via exo):
#   1. tear down the aggregated bridge0 (runtime) and drop it from the SC prefs
#   2. create + switch to the dedicated network location "odyssai"
#   3. recreate the management (Ethernet) and Wi-Fi services FIRST (lifeline), then
#      one DHCP service per Thunderbolt port ("Odyssai Thunderbolt N") — the failing
#      DHCP on each TB port yields APIPA IPv4 + IPv6 link-local (fe80), which is what
#      JACCL GIDs / NDP discovery consume
#   4. disable the (dynamically resolved) Thunderbolt Bridge service
#
# Installed by rdma-onboard.sh at /Library/Application Support/Odyssai/network-setup.sh
# and run by the LaunchDaemon eu.odyssai.networksetup (RunAtLoad + StartInterval 1786).
# Idempotent: safe to re-run at every boot / interval.

set -euo pipefail

SUPPORT="/Library/Application Support/Odyssai"
# Kill-switch: operator opt-out without uninstalling the daemon.
[ -f "${SUPPORT}/network-setup.disabled" ] && exit 0

LOCATION="odyssai"
TB_PREFIX="Odyssai"
PREFS="/Library/Preferences/SystemConfiguration/preferences.plist"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Wait for macOS to finish its own network setup after boot (exo does this; the
# driver's synchronous first run skips it via ODYSSAI_SKIP_SLEEP=1).
[ "${ODYSSAI_SKIP_SLEEP:-0}" = "1" ] || sleep 20

# --- helpers ----------------------------------------------------------------

# Exact-match guards. Services need the '*' (disabled) prefix stripped + header
# dropped; -listlocations has NO header and NO '*' so it gets a plain exact grep.
service_exists()  { networksetup -listallnetworkservices 2>/dev/null | sed '1d;s/^\*//' | grep -Fxq -- "$1"; }
location_exists() { networksetup -listlocations 2>/dev/null | grep -Fxq -- "$1"; }

# Network-service name bound to a device (enX/bridgeX) in the CURRENT location.
service_for_device() {
  networksetup -listnetworkserviceorder 2>/dev/null | awk -v d="$1" '
    /^\([0-9*]+\) /{n=$0; sub(/^\([0-9*]+\) /,"",n)}
    $0 ~ ("Device: " d ")") {print n; exit}'
}

# Hardware-port name for a device (the IOKit-stable name, e.g. "Thunderbolt 1").
port_for_device() {
  networksetup -listallhardwareports 2>/dev/null | awk -v d="$1" '
    /^Hardware Port:/ {p=substr($0,16)}
    /^Device:/ && substr($0,9)==" "d {print substr(p,2); exit}'
}

# --- 0. resolve the bridge service BEFORE destroying bridge0 ----------------
# (after the destroy there are no members left to inspect — exo resolves nothing
# and falls back to a hardcoded English name; we capture it now, fail-soft)
BRIDGE_SVC=""
BRIDGE_PORT=""
if ifconfig bridge0 >/dev/null 2>&1; then
  BRIDGE_SVC="$(service_for_device bridge0 || true)"
  BRIDGE_PORT="$(port_for_device bridge0 || true)"
  log "resolved bridge0: service='${BRIDGE_SVC:-?}' port='${BRIDGE_PORT:-?}'"
fi

# --- 1. tear down bridge0 (runtime) ------------------------------------------
if ifconfig bridge0 >/dev/null 2>&1; then
  if ifconfig bridge0 | grep -q 'member'; then
    ifconfig bridge0 | awk '/member/ {print $2}' | xargs -n1 ifconfig bridge0 deletem 2>/dev/null || true
  fi
  ifconfig bridge0 destroy 2>/dev/null || true
  log "bridge0 torn down"
fi

# --- 2. drop the bridge from the persistent SC prefs -------------------------
/usr/libexec/PlistBuddy -c "Delete :VirtualNetworkInterfaces:Bridge:bridge0" "$PREFS" 2>/dev/null || true

# --- 3. dedicated location ----------------------------------------------------
location_exists "$LOCATION" || networksetup -createlocation "$LOCATION"
networksetup -switchtolocation "$LOCATION" >/dev/null
log "switched to location '${LOCATION}'"

# --- 4. recreate services: management + Wi-Fi FIRST (lifeline), TB ports after.
# Hardware-port names from IOKit are locale-stable ("Ethernet", "Wi-Fi",
# "Thunderbolt N"); only the BRIDGE port/service name is localized — that one is
# matched by the resolved name/device, never by a literal.
# Pass 1: every non-TB, non-bridge, non-dongle port (Ethernet, Wi-Fi, ...).
networksetup -listallhardwareports \
  | awk -F': ' '/Hardware Port: / {print $2}' \
  | while IFS= read -r name; do
      case "$name" in
        "Ethernet Adapter"*) ;;                      # transient USB/TB dongles: skip
        "Thunderbolt "*) ;;                          # TB ports: pass 2
        *)
          [ -n "$BRIDGE_PORT" ] && [ "$name" = "$BRIDGE_PORT" ] && continue   # bridge port, by resolved name
          service_exists "$name" \
            || networksetup -createnetworkservice "$name" "$name" 2>/dev/null \
            || continue
          ;;
      esac
    done

# Pass 2: one DHCP service per Thunderbolt port.
networksetup -listallhardwareports \
  | awk -F': ' '/Hardware Port: / {print $2}' \
  | while IFS= read -r name; do
      case "$name" in
        "Thunderbolt Bridge") ;;                     # defensive (EN name)
        "Thunderbolt "*)
          [ -n "$BRIDGE_PORT" ] && [ "$name" = "$BRIDGE_PORT" ] && continue
          svc="${TB_PREFIX} ${name}"
          service_exists "$svc" \
            || networksetup -createnetworkservice "$svc" "$name" 2>/dev/null \
            || continue
          networksetup -setdhcp "$svc" 2>/dev/null \
            || log "warn: setdhcp failed for '${svc}' (continuing)"
          ;;
      esac
    done
log "services recreated (mgmt/Wi-Fi first, TB ports DHCP)"

# --- 5. disable the Thunderbolt Bridge service (resolved name, never literal) --
# configd may have re-materialized a bridge service in the new location; re-resolve
# now, fall back to the pre-teardown name captured in step 0.
FINAL_SVC="$(service_for_device bridge0 || true)"
[ -n "$FINAL_SVC" ] || FINAL_SVC="$BRIDGE_SVC"
if [ -n "$FINAL_SVC" ] && service_exists "$FINAL_SVC"; then
  networksetup -setnetworkserviceenabled "$FINAL_SVC" off 2>/dev/null \
    && log "disabled bridge service '${FINAL_SVC}'" \
    || log "warn: could not disable '${FINAL_SVC}'"
else
  log "no bridge service in location '${LOCATION}' — nothing to disable"
fi

log "done."
