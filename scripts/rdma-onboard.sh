#!/usr/bin/env bash
# rdma-onboard.sh — driver: provision this Mac's Thunderbolt network for RDMA/JACCL.
#
# Wraps the vendored exo recipe (odyssai-network-setup.sh — see its header for the
# Apache 2.0 attribution) in a guard battery + state machine. Hardened across 10 Codex
# review rounds + a 4-lens adversarial panel (2026-06-08..10). Mechanism credit: exo
# (github.com/exo-explore/exo, commit 09f9ea313f72e261f40a94cea4c0e3681b31af23).
#
# THE RECIPE (proven on the 4 production ultras): tear down bridge0, dedicated network
# location "odyssai", per-TB-port DHCP services (failing DHCP -> APIPA + IPv6 fe80 ->
# JACCL GIDs / NDP discovery), Thunderbolt Bridge service disabled, root LaunchDaemon
# eu.odyssai.networksetup (RunAtLoad + StartInterval 1786) re-asserts forever.
#
# EXIT CODES (the GUI/CLI contract — never parse prose):
#   0  = ready        (local-ready; mesh-ready too if every cabled TB port has
#                      fe80 + its rdma_enX HCA PORT_ACTIVE)
#   10 = needs_reboot (recipe applied + daemon loaded, but cabled TB ports lack fe80;
#                      proposed AT MOST ONCE — a second consecutive 10 becomes 1)
#   11 = needs_apply  (--check only: node not yet provisioned)
#   1  = blocked      (a guard refused, or diagnosis needed; nothing half-done)
#
# USAGE:
#   rdma-onboard.sh --check                                   # read-only; non-root OK
#   sudo rdma-onboard.sh --apply --console [--expect N] [--allow-active-node]
#   sudo rdma-onboard.sh --revert
#
# Local-console only for mutations: --apply/--revert REFUSE under SSH (SSH_CONNECTION).
set -euo pipefail

SUPPORT="/Library/Application Support/Odyssai"
INSTALLED_SCRIPT="${SUPPORT}/network-setup.sh"
STATE_PRIOR_LOCATION="${SUPPORT}/prior-location.name"
STATE_BRIDGE_SVC="${SUPPORT}/bridge-service.name"
STATE_REBOOT_COUNT="${SUPPORT}/reboot-proposals.count"
MARKER_NO_DAEMON="${SUPPORT}/applied-no-daemon"
KILL_SWITCH="${SUPPORT}/network-setup.disabled"
PLIST_LABEL="eu.odyssai.networksetup"
PLIST_DEST="/Library/LaunchDaemons/${PLIST_LABEL}.plist"
DAEMON_INTERVAL=1786
LOCATION="odyssai"
LOG="/var/log/eu.odyssai.rdma-onboard.log"

EXIT_READY=0; EXIT_NEEDS_REBOOT=10; EXIT_NEEDS_APPLY=11
EXPECT=0; CONSOLE=0; ALLOW_ACTIVE=0
CONVERGE_DEADLINE="${ODYSSAI_CONVERGE_DEADLINE:-120}"

is_root() { [ "$(id -u)" -eq 0 ]; }
# Non-root (--check): reports + log go to TMPDIR, never /Library or /var/log.
if ! is_root; then
  LOG="${TMPDIR:-/tmp}/eu.odyssai.rdma-onboard.log"
  REPORT_DIR="${TMPDIR:-/tmp}/odyssai-reports"
else
  REPORT_DIR="${SUPPORT}/reports"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG" >&2; }
die() { log "BLOCKED: $*"; exit 1; }

# The vendored recipe script ships next to this driver (or via env override).
vendored_script() {
  if [ -n "${ODYSSAI_NETWORK_SETUP:-}" ]; then echo "$ODYSSAI_NETWORK_SETUP"; return; fi
  echo "$(cd "$(dirname "$0")" && pwd)/odyssai-network-setup.sh"
}

# --- identity helpers (all set -e safe) -------------------------------------

port_device() {
  networksetup -listallhardwareports 2>/dev/null | awk -v p="Hardware Port: $1" '
    $0 == p { getline; if ($1 == "Device:") print $2; exit }' || true
}
wifi_device() { port_device "Wi-Fi"; }
mgmt_device() { port_device "Ethernet"; }

service_for_device() {
  networksetup -listnetworkserviceorder 2>/dev/null | awk -v d="$1" '
    /^\([0-9*]+\) /{n=$0; sub(/^\([0-9*]+\) /,"",n)}
    $0 ~ ("Device: " d ")") {print n; exit}' || true
}
service_exists()  { networksetup -listallnetworkservices 2>/dev/null | sed '1d;s/^\*//' | grep -Fxq -- "$1"; }
location_exists() { networksetup -listlocations 2>/dev/null | grep -Fxq -- "$1"; }
current_location(){ networksetup -getcurrentlocation 2>/dev/null || true; }
default_iface()   { route -n get default 2>/dev/null | awk '/interface:/{print $2}' || true; }
is_apipa()        { case "${1:-}" in 169.254.*) return 0;; *) return 1;; esac; }

is_tb_device() {
  local d="$1"; [ -n "$d" ] || return 1
  [ "$d" = "$(mgmt_device)" ] && return 1
  [ "$d" = "$(wifi_device)" ] && return 1
  case "$d" in bridge*) return 1;; esac
  networksetup -listallhardwareports 2>/dev/null \
    | awk '/^Hardware Port: Thunderbolt [0-9]/{getline; if($1=="Device:")print $2}' \
    | grep -qx "$d"
}
tb_devices() { local d; for d in $(ifconfig -l 2>/dev/null | tr ' ' '\n' | grep -E '^en[0-9]+$'); do is_tb_device "$d" && echo "$d"; done; }
tb_cabled()  { local d; for d in $(tb_devices); do ifconfig "$d" 2>/dev/null | grep -q 'status: active' && echo "$d"; done; }

mgmt_healthy() {
  local mgmt ip; mgmt="$(mgmt_device)"; ip="$(ipconfig getifaddr "$mgmt" 2>/dev/null || true)"
  [ -n "$ip" ] && ! is_apipa "$ip" && [ -n "$(default_iface)" ]
}
wifi_usable() {
  local wifi wip; wifi="$(wifi_device)"; [ -n "$wifi" ] || return 1
  [ "$(networksetup -getairportpower "$wifi" 2>/dev/null | awk '{print $NF}' || true)" = "On" ] || return 1
  wip="$(ipconfig getifaddr "$wifi" 2>/dev/null || true)"
  [ -n "$wip" ] && ! is_apipa "$wip"
}
daemon_loaded() { launchctl print "system/${PLIST_LABEL}" >/dev/null 2>&1; }

# --- snapshots (valid JSON via python3) --------------------------------------

snapshot_json() {
  mkdir -p "$REPORT_DIR" 2>/dev/null || true
  SNAP_TAG="$1" \
  SNAP_LOCATION="$(current_location)" \
  SNAP_BR_PRESENT="$(ifconfig bridge0 >/dev/null 2>&1 && echo present || echo absent)" \
  SNAP_BR_SVC="$(service_for_device bridge0)" \
  SNAP_MGMT_IP="$(ipconfig getifaddr "$(mgmt_device)" 2>/dev/null || true)" \
  SNAP_WIFI_IP="$(ipconfig getifaddr "$(wifi_device)" 2>/dev/null || true)" \
  SNAP_DEFAULT_IFACE="$(default_iface)" \
  SNAP_TB="$(tb_devices | tr '\n' ' ')" \
  SNAP_TB_CABLED="$(tb_cabled | tr '\n' ' ')" \
  SNAP_DAEMON="$(daemon_loaded && echo loaded || echo not-loaded)" \
  SNAP_IBV="$(ibv_devinfo 2>/dev/null | awk '/hca_id/{h=$2} /state:/{print h"="$2}' | tr '\n' ' ' || true)" \
  python3 -c '
import json, os, datetime
d = {k[5:].lower(): os.environ.get(k,"") for k in os.environ if k.startswith("SNAP_")}
d["stamp"]=datetime.datetime.now().isoformat(timespec="seconds")
print(json.dumps(d, indent=2))' > "${REPORT_DIR}/${1}.json" 2>/dev/null || true
  [ -s "${REPORT_DIR}/${1}.json" ] && log "snapshot -> ${REPORT_DIR}/${1}.json" \
    || log "WARNING: snapshot ${1} failed (python3?) — continuing"
}

# --- readiness ----------------------------------------------------------------

have_ibv() { command -v ibv_devinfo >/dev/null 2>&1; }

hca_active() {  # hca_active rdma_enX -> 0 if PORT_ACTIVE
  ibv_devinfo 2>/dev/null | awk -v h="$1" '
    /hca_id/{cur=$2} /state:/{if(cur==h && /PORT_ACTIVE/) found=1} END{exit !found}'
}

# Per-port mesh diagnosis on stdout: "enX ok|no-fe80|hca-down|hca-unknown"
mesh_report() {
  local d fe80
  for d in $(tb_cabled); do
    fe80="$(ifconfig "$d" 2>/dev/null | awk '/inet6 fe80/{print $2; exit}' || true)"
    if [ -z "$fe80" ]; then echo "$d no-fe80"; continue; fi
    if have_ibv; then
      hca_active "rdma_${d}" && echo "$d ok" || echo "$d hca-down"
    else
      echo "$d hca-unknown"   # degraded: fe80-only criterion (P4)
    fi
  done
}

mesh_ready() {
  local line bad=0 n=0
  while IFS= read -r line; do
    n=$((n+1))
    case "$line" in *" ok"|*" hca-unknown") ;; *) bad=$((bad+1));; esac
  done < <(mesh_report)
  [ "$EXPECT" -gt 0 ] && [ "$n" -lt "$EXPECT" ] && return 1
  [ "$n" -gt 0 ] && [ "$bad" -eq 0 ]
}

local_ready() {
  [ "$(current_location)" = "$LOCATION" ] || return 1
  daemon_loaded || return 1
  # bridge gone OR its resolved service disabled (after the recipe the service may
  # simply not exist in the fresh location — requiring "disabled" would be unfalsifiable)
  if ifconfig bridge0 >/dev/null 2>&1; then
    local svc; svc="$(service_for_device bridge0)"
    [ -n "$svc" ] || return 1
    [ "$(networksetup -getnetworkserviceenabled "$svc" 2>/dev/null || true)" = "Disabled" ] || return 1
  fi
  return 0
}

# --- guards -------------------------------------------------------------------

guard_not_ssh() {
  [ -n "${SSH_CONNECTION:-}${SSH_TTY:-}" ] \
    && die "refusing network mutation over SSH — run at the console/GUI of this Mac (the SSH vector is what stranded .33)."
  # sudo strips SSH_* from the environment ('ssh node sudo …' would sail through
  # the env check) — walk the process ancestry looking for sshd instead.
  local pid=$$ comm hops=0
  while [ "${pid:-1}" -gt 1 ] && [ "$hops" -lt 20 ]; do
    comm="$(ps -o comm= -p "$pid" 2>/dev/null | awk '{print $1}')" || break
    case "$comm" in *sshd*) die "refusing network mutation over SSH (sshd ancestor detected) — run at the console/Screen Sharing of this Mac.";; esac
    pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')" || break
    hops=$((hops+1))
  done
  return 0
}

guard_no_exo() {
  local hits=""
  [ -f /Library/LaunchDaemons/io.exo.networksetup.plist ] && hits="${hits} plist"
  launchctl print system/io.exo.networksetup >/dev/null 2>&1 && hits="${hits} loaded-job"
  location_exists "exo" && hits="${hits} location-exo"
  ls "/Library/Application Support/EXO/"disable_bridge*.sh >/dev/null 2>&1 && hits="${hits} legacy-script"
  [ -z "$hits" ] || die "exo network setup detected on this Mac (${hits# }). Two daemons would fight over the network location every ~30 min. Migrate first: sudo launchctl bootout system/io.exo.networksetup ; remove /Library/LaunchDaemons/io.exo.networksetup.plist and '/Library/Application Support/EXO' ; then re-run."
  return 0
}

guard_mgmt_not_static() {
  local svc; svc="$(service_for_device "$(mgmt_device)")"
  [ -n "$svc" ] || return 0
  if networksetup -getinfo "$svc" 2>/dev/null | grep -q '^Manual Configuration'; then
    die "management service '${svc}' uses a STATIC IPv4 config — the new location would lose it. Note the config, plan its recreation, then re-run (or keep this node on DHCP like the rest of the fleet)."
  fi
  return 0
}

guard_not_active_node() {
  [ "$ALLOW_ACTIVE" = "1" ] && return 0
  local d peers=0
  for d in $(tb_cabled); do
    if ndp -an 2>/dev/null | grep -q "%${d} " ; then peers=$((peers+1)); fi
  done
  [ "$peers" -eq 0 ] \
    || die "this node has ${peers} live Thunderbolt peer(s) — it looks like an ACTIVE cluster member; applying would bounce every interface and kill in-flight inference. Re-run with --allow-active-node to proceed deliberately."
  return 0
}

guard_forbidden_port() {
  # Mac Studio: the TB port next to the Ethernet port cannot do RDMA (rdma_en2).
  if tb_cabled | grep -qx "en2"; then
    die "a Thunderbolt cable is plugged into the port next to Ethernet (en2/rdma_en2) — that port does NOT support RDMA on Mac Studio. Move the cable to one of the three leftmost ports, then re-run."
  fi
  return 0
}

guard_route_not_tb() {
  local di; di="$(default_iface)"
  if [ -n "$di" ]; then
    { is_tb_device "$di" || [ "$di" = "bridge0" ]; } \
      && die "the default route rides ${di} (Thunderbolt/bridge) — applying would cut this Mac off. Fix the management network first."
  fi
  return 0
}

# --- subcommands ---------------------------------------------------------------

check() {
  snapshot_json "check"
  local loc; loc="$(current_location)"

  # Derive applied-no-daemon (marker alone misses SIGKILL/power-loss mid-apply).
  if [ "$loc" = "$LOCATION" ] && [ -f "$INSTALLED_SCRIPT" ] && ! daemon_loaded; then
    log "state: applied-no-daemon — the recipe ran but the re-assert daemon is not loaded."
    log "recover: sudo $0 --apply --console   (idempotent; finishes the daemon install)"
    exit 1
  fi

  if local_ready; then
    local cabled; cabled="$(tb_cabled | wc -l | tr -d ' ')"
    if [ "$cabled" -eq 0 ]; then
      log "ready (local): node provisioned; no Thunderbolt cables detected yet — cable the mesh, then re-check."
      have_ibv || log "note: ibv_devinfo not found — mesh checks will be fe80-only (degraded)."
      exit "$EXIT_READY"
    fi
    if mesh_ready; then
      log "ready (mesh): all ${cabled} cabled TB port(s) have fe80 + active HCAs."
      exit "$EXIT_READY"
    fi
    # applied but not converged: at most ONE reboot proposal, then diagnose.
    local count=0; [ -s "$STATE_REBOOT_COUNT" ] && count="$(cat "$STATE_REBOOT_COUNT")"
    if [ "$count" -ge 1 ]; then
      log "still not mesh-ready AFTER a reboot — a reboot will not fix this. Per-port diagnosis:"
      mesh_report | while IFS= read -r l; do log "  $l"; done
      log "likely causes: cable unplugged/dead, peer node powered off, forbidden port (en2), mismatched macOS builds across nodes, rdma_ctl disabled."
      exit 1
    fi
    is_root && echo "1" > "$STATE_REBOOT_COUNT" 2>/dev/null || true
    log "needs_reboot: recipe applied + daemon loaded, but cabled TB ports lack fe80. Reboot once, then re-check."
    mesh_report | while IFS= read -r l; do log "  $l"; done
    exit "$EXIT_NEEDS_REBOOT"
  fi

  log "needs_apply: node not provisioned (location='${loc}', daemon $(daemon_loaded && echo loaded || echo not-loaded))."
  exit "$EXIT_NEEDS_APPLY"
}

apply() {
  is_root || die "--apply requires root (sudo)."
  guard_not_ssh
  [ "$CONSOLE" = "1" ] || die "--apply requires --console (mutations are local-console only)."
  [ -f "$KILL_SWITCH" ] && die "kill-switch present (${KILL_SWITCH}) — remove it to re-enable provisioning."
  guard_no_exo
  guard_route_not_tb
  guard_mgmt_not_static
  guard_forbidden_port
  guard_not_active_node
  mgmt_healthy || die "management NIC is not healthy (needs a non-APIPA IPv4 + default route) — fix the LAN first; applying now would leave no way back."
  wifi_usable || log "note: Wi-Fi is not a usable lifeline right now — you are at the console, proceeding."

  local src; src="$(vendored_script)"
  [ -f "$src" ] || die "vendored network script not found at ${src} (set ODYSSAI_NETWORK_SETUP)."
  bash -n "$src" || die "vendored network script fails syntax check — refusing."

  snapshot_json "preflight"

  # ---- persist EVERYTHING before the first mutation --------------------------
  mkdir -p "$SUPPORT"
  local cur; cur="$(current_location)"
  if [ "$cur" != "$LOCATION" ] && [ ! -s "$STATE_PRIOR_LOCATION" ]; then
    printf '%s' "$cur" > "$STATE_PRIOR_LOCATION"
  fi
  local bsvc; bsvc="$(service_for_device bridge0)"
  [ -n "$bsvc" ] && printf '%s' "$bsvc" > "$STATE_BRIDGE_SVC"
  echo "0" > "$STATE_REBOOT_COUNT"

  # Stable copy (NEVER point the daemon at the bundle/DMG path) + self-heal.
  local fresh_copy=0
  if [ ! -f "$INSTALLED_SCRIPT" ] || ! cmp -s "$src" "$INSTALLED_SCRIPT"; then
    install -o root -g wheel -m 0755 "$src" "$INSTALLED_SCRIPT"
    fresh_copy=1
    log "network script installed -> ${INSTALLED_SCRIPT}"
  fi

  # Phase-aware trap: before mutation -> clean what we staged; after -> NEVER
  # delete the re-assert mechanism, leave an explicit recovery state instead.
  local PHASE="pre"
  on_err() {
    if [ "$PHASE" = "pre" ]; then
      [ "$fresh_copy" = "1" ] && [ ! -f "$PLIST_DEST" ] && rm -f "$INSTALLED_SCRIPT"
      log "BLOCKED: failed before any network mutation — nothing changed."
    else
      touch "$MARKER_NO_DAEMON" 2>/dev/null || true
      log "BLOCKED: failed AFTER the network mutation — state: applied-no-daemon."
      log "recover: sudo $0 --apply --console   (idempotent) ; or sudo $0 --revert"
    fi
    exit 1
  }
  trap on_err ERR

  # ---- mutate -----------------------------------------------------------------
  PHASE="post"
  log "running the recipe synchronously (skip-sleep)…"
  ODYSSAI_SKIP_SLEEP=1 bash "$INSTALLED_SCRIPT" 2>&1 | while IFS= read -r l; do log "  recipe: $l"; done

  # Convergence poll — the switch drops en0/Wi-Fi transiently; single-shot checks lie.
  log "waiting for convergence (deadline ${CONVERGE_DEADLINE}s)…"
  local t=0 ok=0
  while [ "$t" -lt "$CONVERGE_DEADLINE" ]; do
    if [ "$(current_location)" = "$LOCATION" ] && mgmt_healthy; then ok=1; break; fi
    sleep 3; t=$((t+3))
  done
  [ "$ok" = "1" ] || die "management did not converge within ${CONVERGE_DEADLINE}s (location=$(current_location), en0=$(ipconfig getifaddr "$(mgmt_device)" 2>/dev/null || echo none)). State: applied-no-daemon."
  log "management converged: en0=$(ipconfig getifaddr "$(mgmt_device)" 2>/dev/null) location=$(current_location)"
  wifi_usable && log "Wi-Fi lifeline re-associated: $(ipconfig getifaddr "$(wifi_device)" 2>/dev/null)" \
              || log "note: Wi-Fi not (yet) re-associated — check Known Networks if you need the lifeline."

  # ---- daemon -------------------------------------------------------------------
  cat > "$PLIST_DEST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>${PLIST_LABEL}</string>
  <key>ProgramArguments</key><array>
    <string>/bin/bash</string><string>${INSTALLED_SCRIPT}</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>${DAEMON_INTERVAL}</integer>
  <key>StandardOutPath</key><string>/var/log/${PLIST_LABEL}.log</string>
  <key>StandardErrorPath</key><string>/var/log/${PLIST_LABEL}.err.log</string>
</dict></plist>
PLIST
  chown root:wheel "$PLIST_DEST"; chmod 0644 "$PLIST_DEST"
  plutil -lint "$PLIST_DEST" >/dev/null
  launchctl bootout "system/${PLIST_LABEL}" 2>/dev/null || true
  launchctl bootstrap system "$PLIST_DEST"
  launchctl enable "system/${PLIST_LABEL}" 2>/dev/null || true
  rm -f "$MARKER_NO_DAEMON"
  log "daemon ${PLIST_LABEL} installed + loaded (re-assert every ${DAEMON_INTERVAL}s)."
  trap - ERR

  snapshot_json "post-apply"

  # ---- classify -------------------------------------------------------------------
  local cabled; cabled="$(tb_cabled | wc -l | tr -d ' ')"
  if [ "$cabled" -eq 0 ]; then
    log "ready (local): provisioned; no TB cables detected — cable the mesh, then --check."
    exit "$EXIT_READY"
  fi
  if mesh_ready; then
    log "ready (mesh): all ${cabled} cabled TB port(s) carry fe80 + active HCAs. RDMA ready."
    exit "$EXIT_READY"
  fi
  echo "1" > "$STATE_REBOOT_COUNT"
  log "needs_reboot: recipe + daemon in place, TB ports not converged. Reboot ONCE, then --check."
  mesh_report | while IFS= read -r l; do log "  $l"; done
  exit "$EXIT_NEEDS_REBOOT"
}

revert() {
  is_root || die "--revert requires root (sudo)."
  guard_not_ssh
  snapshot_json "pre-revert"

  # 1. stop re-asserts — but KEEP the assets until restoration is proven.
  launchctl bootout "system/${PLIST_LABEL}" 2>/dev/null || true

  # 2. restore the persisted prior location (never a hardcoded literal — FR macOS
  #    names the default "Automatique", not "Automatic").
  local prior=""
  [ -s "$STATE_PRIOR_LOCATION" ] && prior="$(cat "$STATE_PRIOR_LOCATION")"
  [ -n "$prior" ] || die "no persisted prior location — cannot identify the rollback target. Restore manually (networksetup -switchtolocation <name>), assets left in place."
  location_exists "$prior" || die "persisted prior location '${prior}' no longer exists — restore manually, assets left in place."
  networksetup -switchtolocation "$prior" >/dev/null
  [ "$(current_location)" = "$prior" ] \
    || die "location switch to '${prior}' did NOT take (current: $(current_location)) — assets left in place for recovery."
  log "restored location '${prior}'."

  # 3. restoration proven: now dismantle.
  location_exists "$LOCATION" && networksetup -deletelocation "$LOCATION" >/dev/null 2>&1 || true
  local bsvc=""; [ -s "$STATE_BRIDGE_SVC" ] && bsvc="$(cat "$STATE_BRIDGE_SVC")"
  [ -n "$bsvc" ] || bsvc="$(service_for_device bridge0)"
  if [ -n "$bsvc" ] && service_exists "$bsvc"; then
    networksetup -setnetworkserviceenabled "$bsvc" on 2>/dev/null \
      && log "re-enabled bridge service '${bsvc}'." \
      || log "warn: could not re-enable '${bsvc}'."
  fi
  rm -f "$PLIST_DEST" "$INSTALLED_SCRIPT" "$MARKER_NO_DAEMON" \
        "$STATE_PRIOR_LOCATION" "$STATE_BRIDGE_SVC" "$STATE_REBOOT_COUNT"

  # 4. verify bridge0 comes back with TB members (often needs a reboot) — honest
  #    partial-rollback report, never a false success. (The PlistBuddy-deleted prefs
  #    entry is not restorable by script; macOS recreates it with the service.)
  local t=0
  while [ "$t" -lt 30 ]; do
    ifconfig bridge0 >/dev/null 2>&1 && break
    sleep 3; t=$((t+3))
  done
  if ifconfig bridge0 >/dev/null 2>&1; then
    log "revert complete: bridge0 is back."
  else
    log "revert PARTIAL: location restored + daemon removed, but bridge0 has not re-materialized yet — it normally returns after a reboot. If not: networksetup -setnetworkserviceenabled '<bridge service>' on, then reboot."
  fi
  snapshot_json "post-revert"
}

usage() { echo "usage: $0 [--check | --apply --console [--expect N] [--allow-active-node] | --revert]" >&2; exit 2; }

main() {
  local cmd="${1:-}"; [ -n "$cmd" ] || usage; shift || true
  while [ $# -gt 0 ]; do
    case "$1" in
      --expect)            [ -n "${2:-}" ] || die "--expect needs a number"; case "$2" in *[!0-9]*) die "--expect must be numeric";; esac; EXPECT="$2"; shift 2 ;;
      --console)           CONSOLE=1; shift ;;
      --allow-active-node) ALLOW_ACTIVE=1; shift ;;
      *) usage ;;
    esac
  done
  case "$cmd" in
    --check)  check ;;
    --apply)  apply ;;
    --revert) revert ;;
    *) usage ;;
  esac
}
main "$@"
