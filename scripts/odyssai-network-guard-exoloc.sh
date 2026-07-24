#!/usr/bin/env bash
# odyssai-network-guard-exoloc.sh — remplaçant drop-in de l'exo disable_bridge.sh
# pour les nodes ENCORE sur la location réseau "exo" (.30/.31/.32).
#
# Pourquoi ce script et pas la recette complète (odyssai-network-setup.sh) :
# la recette bascule en location "odyssai" (create+switch) — sur un node dont la
# location active est "exo", ce switch abandonnerait les services existants
# (mgmt NIC compris) pour une location vide → node injoignable (classe
# d'incident ADR-0001 / .33 bricking). Ici : AUCUN changement de location,
# AUCUNE création de service — on garde les effets utiles de l'exo script
# (bridge0 destroy, Thunderbolt Bridge off) et on remplace son
# `setdhcp` inconditionnel (le clobber qui tuait le fix APIPA-GID) par
# l'assertion static-from-conf du fix JACCL (2026-07-24).
#
# Conf : /Library/Application Support/Odyssai/static-ips.conf
#   lignes `SERVICE NAME|IP` (+ `TB_PREFIX|EXO` ignoré ici, naming lu du conf).
# Sans conf ou sans entrée pour un service : NE TOUCHE PAS le service (pas de
# re-DHCP — un port hors mesh garde sa config quelle qu'elle soit).
#
# Tourne via le LaunchDaemon exo existant (io.exo.networksetup, RunAtLoad +
# 1786s) → ré-assert le static à chaque cycle : le daemon devient le gardien.

set -uo pipefail

# Attendre la fin du setup réseau macOS après boot (comportement exo conservé)
sleep 20

PREFS="/Library/Preferences/SystemConfiguration/preferences.plist"
STATIC_CONF="/Library/Application Support/Odyssai/static-ips.conf"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# --- 1. bridge0 : détruire (comportement exo conservé) ------------------------
if ifconfig bridge0 &>/dev/null; then
  if ifconfig bridge0 | grep -q 'member'; then
    ifconfig bridge0 | awk '/member/ {print $2}' \
      | xargs -n1 ifconfig bridge0 deletem 2>/dev/null || true
  fi
  ifconfig bridge0 destroy 2>/dev/null || true
fi
/usr/libexec/PlistBuddy -c "Delete :VirtualNetworkInterfaces:Bridge:bridge0" "$PREFS" 2>/dev/null || true

# --- 2. statics par port TB depuis le conf (le cœur du fix) -------------------
# Idempotent : ne réécrit que si la méthode ou l'IP a dérivé. Jamais de
# setdhcp : l'absence d'entrée = ne pas toucher.
if [ -f "$STATIC_CONF" ]; then
  while IFS='|' read -r svc ip; do
    case "$svc" in ""|\#*|TB_PREFIX) continue ;; esac
    [ -z "$ip" ] && continue
    cur_info=$(networksetup -getinfo "$svc" 2>/dev/null) || continue
    if ! printf '%s' "$cur_info" | grep -q "^Manual" \
       || ! printf '%s' "$cur_info" | grep -q "^IP address: ${ip}$"; then
      networksetup -setmanual "$svc" "$ip" 255.255.255.0 2>/dev/null \
        && log "static asserted: ${svc} -> ${ip}" \
        || log "warn: setmanual failed for '${svc}'"
    fi
  done < "$STATIC_CONF"
else
  log "no static-ips.conf — leaving TB services untouched"
fi

# --- 3. Thunderbolt Bridge service off (comportement exo conservé) ------------
if networksetup -listallnetworkservices 2>/dev/null | grep -q "Thunderbolt Bridge"; then
  networksetup -setnetworkserviceenabled "Thunderbolt Bridge" off 2>/dev/null || true
fi

log "done."
