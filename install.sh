#!/bin/bash
# OdyssAI-X node installer (#82). Run it ON the Mac that will be a cluster node:
#
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Odyssai-eu/OdyssAI-X/main/install.sh)"
#
# What it does, each stage timed, each stage a no-op when already in place:
#   1. checks      Apple Silicon, macOS, no model being served on this node
#   2. fetch       the OdyssAI-X sources (a tarball: no git, no Xcode tools)
#   3. python      uv + Python 3.11 + ~/mlx-cluster/.venv (no Homebrew)
#   4. packages    the pinned node requirements (requirements-node.txt)
#   5. files       runner, helpers, runtime patches, vendored model modules
#   6. jaccl       OdyssAI's patched libjaccl.dylib, checked against its sha256
#   7. wired-limit the GPU wired-memory limit, kept across reboots (asks sudo);
#                  ODYSSAI_X_WIRED_MB changes it even when its daemon exists
#   8. network     the Thunderbolt network for RDMA (scripts/rdma-onboard.sh,
#                  asks sudo; only on the Mac's own console, it can cut SSH)
#   9. vision      the vision venv ~/.venvs/mlx-vlm (scripts/install-mlx-vlm.sh,
#                  Python 3.12 from uv) with the same patched libjaccl
#  10. discovery   the node announces itself: Bonjour `_odyssai._tcp` and a
#                  heartbeat every 5 s to its orchestrator (launchd, background)
#  11. doctor      `odyssai-x doctor`: what this node still lacks, if anything
#
# Out of scope: RDMA activation, which only recoveryOS can do
# (`rdma_ctl enable`, see README "Enable RDMA"). The network stage waits for it.
#
# Environment:
#   ODYSSAI_X_REF         git ref to install (default: main)
#   ODYSSAI_X_DIR         node directory (default: ~/mlx-cluster)
#   ODYSSAI_X_MODELS_DIR  models directory (default: /Volumes/models/odysseus
#                         if it exists, else ~/mlx-models)
#   ODYSSAI_X_WIRED_MB    wired limit in MB (default: the current value if set,
#                         else RAM minus the larger of 8 GiB and 6.25 %)
#   ODYSSAI_X_ENGINE      this node's orchestrator, e.g. http://mini.local:8000
#                         (default: the node finds an OdyssAI-X engine on its /24)
#   ODYSSAI_X_FORCE=1     change a node even while it serves a model
#   ODYSSAI_X_VISION=0    skip the vision venv (default: installed)
#   ODYSSAI_X_NETWORK=0   skip the Thunderbolt network stage
#
# Everything sits in main(), called on the last line: a truncated download
# runs nothing.
set -euo pipefail

REPO="Odyssai-eu/OdyssAI-X"
JACCL_TAG="jaccl-0.32.2-odyssai.1"
PY_VERSION="3.11"

T0=$(date +%s)
CHANGED=0
STAGE_N=0
STAGES=11
SUMMARY=""

say()  { printf '%s\n' "$*"; }
warn() { printf 'WARN %s\n' "$*"; }
die()  { printf 'FAIL %s\n' "$*" >&2; exit 2; }

stage_begin() { STAGE_N=$((STAGE_N + 1)); STAGE_NAME="$1"; STAGE_T=$(date +%s); STAGE_STATE="ok"; }
stage_changed() { STAGE_STATE="changed"; CHANGED=1; }
stage_end() {
  local dt=$(( $(date +%s) - STAGE_T ))
  local line
  line=$(printf '[%d/%d] %-12s %-8s %3ss%s' "$STAGE_N" "$STAGES" "$STAGE_NAME" "$STAGE_STATE" "$dt" "${1:+  $1}")
  say "$line"
  SUMMARY="${SUMMARY}${line}
"
}

serving() {
  pgrep -f "[m]lx-cluster/(vlm_)?runner.py|[m]lx_vlm.server|[d]ecision_serve.py" >/dev/null 2>&1
}

# A change on a serving node needs ODYSSAI_X_FORCE=1 (a pip or dylib swap under
# a live runner can crash it).
guard_change() {
  if [ "$SERVING" = 1 ] && [ "${ODYSSAI_X_FORCE:-0}" != 1 ]; then
    die "$STAGE_NAME: this node is serving a model and $1 would change. Unload it first (dashboard), or rerun with ODYSSAI_X_FORCE=1."
  fi
}

# OdyssAI's patched libjaccl into the mlx of a venv (the node venv, then the
# vision venv: distributed VL runs there). Sets JACCL_NOTE; dies on failure.
patch_jaccl() {   # patch_jaccl <venv python> <manifest>
  local py="$1" want site lib tmp
  want=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["jaccl_sha256"])' "$2")
  site=$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  lib="$site/mlx/lib/libjaccl.dylib"
  [ -f "$lib" ] || die "mlx in $py has no lib/libjaccl.dylib: the mlx wheel is incomplete"
  if [ "$RDMA_OS" != 1 ]; then
    JACCL_NOTE="skipped: macOS $MACOS, stock JACCL kept"
  elif [ "$(shasum -a 256 "$lib" | cut -d' ' -f1)" = "$want" ]; then
    JACCL_NOTE="patched"
  else
    guard_change "libjaccl.dylib"
    tmp=$(mktemp -d)
    curl -fsSL -o "$tmp/libjaccl.dylib" "https://github.com/$REPO/releases/download/$JACCL_TAG/libjaccl.dylib" \
      || die "cannot download $JACCL_TAG/libjaccl.dylib"
    [ "$(shasum -a 256 "$tmp/libjaccl.dylib" | cut -d' ' -f1)" = "$want" ] \
      || die "downloaded libjaccl.dylib does not match the manifest sha256: refusing it (is mlx on the node pin?)"
    [ -f "$lib.orig" ] || cp -p "$lib" "$lib.orig"
    cp "$tmp/libjaccl.dylib" "$lib.odx-new" && mv "$lib.odx-new" "$lib"
    rm -rf "$tmp"
    if ! "$py" -c 'import mlx.core as mx; mx.distributed.is_available()' >/dev/null 2>&1; then
      cp -p "$lib.orig" "$lib.odx-new" && mv "$lib.odx-new" "$lib"
      die "mlx does not import with the patched JACCL: stock one restored"
    fi
    stage_changed
    JACCL_NOTE="patched ($JACCL_TAG)"
  fi
}

can_sudo() { sudo -n true 2>/dev/null || [ -t 0 ] || { : </dev/tty; } 2>/dev/null; }

copy_if_changed() {   # src dst → returns 0 when it copied
  if [ -f "$2" ] && cmp -s "$1" "$2"; then return 1; fi
  cp "$1" "$2.odx-new" && mv "$2.odx-new" "$2"
  return 0
}

main() {
  local DIR="${ODYSSAI_X_DIR:-$HOME/mlx-cluster}"
  local REF="${ODYSSAI_X_REF:-main}"
  local MODELS_DIR="${ODYSSAI_X_MODELS_DIR:-}"
  if [ -z "$MODELS_DIR" ]; then
    # The shared volume only if this user can write to it: another account's
    # /Volumes/models/odysseus (admin:staff 755) is readable, not writable.
    if [ -d /Volumes/models/odysseus ] && [ -w /Volumes/models/odysseus ]; then
      MODELS_DIR=/Volumes/models/odysseus
    else
      MODELS_DIR="$HOME/mlx-models"
    fi
  fi
  say "OdyssAI-X node install — $(hostname -s), ref $REF, into $DIR"

  # 1. checks ────────────────────────────────────────────────────────────
  stage_begin checks
  [ "$(uname -m)" = arm64 ] || die "this Mac is $(uname -m): OdyssAI-X nodes need Apple Silicon"
  local MACOS; MACOS=$(sw_vers -productVersion)
  local MAJOR=${MACOS%%.*} MINOR; MINOR=$(printf '%s' "$MACOS" | cut -d. -f2); MINOR=${MINOR:-0}
  RDMA_OS=0
  if [ "$MAJOR" -gt 26 ] || { [ "$MAJOR" -eq 26 ] && [ "$MINOR" -ge 2 ]; }; then RDMA_OS=1; fi
  SERVING=0; if serving; then SERVING=1; fi
  command -v curl >/dev/null && command -v tar >/dev/null && command -v shasum >/dev/null \
    || die "curl, tar and shasum are needed (they ship with macOS)"
  mkdir -p "$DIR" "$DIR/patches"
  [ -d "$MODELS_DIR" ] || { mkdir -p "$MODELS_DIR" && stage_changed; }
  local note="macOS $MACOS"
  [ "$RDMA_OS" = 1 ] || note="$note (RDMA needs 26.2+: stock JACCL kept)"
  [ "$SERVING" = 1 ] && note="$note, serving a model"
  stage_end "$note"

  # 2. fetch ─────────────────────────────────────────────────────────────
  stage_begin fetch
  local SHA
  SHA=$(curl -fsSL -H "Accept: application/vnd.github.sha" "https://api.github.com/repos/$REPO/commits/$REF") \
    || die "cannot resolve $REF on github.com/$REPO"
  SRC="$DIR/.src/$SHA"
  if [ ! -f "$SRC/.complete" ]; then
    rm -rf "$DIR/.src"; mkdir -p "$SRC"
    curl -fsSL "https://codeload.github.com/$REPO/tar.gz/$SHA" | tar -xz -C "$SRC" --strip-components 1 \
      || die "download of $REPO@$SHA failed"
    touch "$SRC/.complete"
    stage_changed
  fi
  stage_end "${SHA:0:7}"

  # 3. python ────────────────────────────────────────────────────────────
  stage_begin python
  local UV; UV=$(command -v uv || true)
  [ -n "$UV" ] || { [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"; } || true
  PY="$DIR/.venv/bin/python"
  if ! "$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11)' >/dev/null 2>&1; then
    guard_change "the venv"
    if [ -z "$UV" ]; then
      curl -fsSL https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null \
        || die "uv install failed (https://astral.sh/uv)"
      UV="$HOME/.local/bin/uv"
    fi
    "$UV" python install "$PY_VERSION" >/dev/null 2>&1 || die "uv could not install Python $PY_VERSION"
    [ -d "$DIR/.venv" ] && mv "$DIR/.venv" "$DIR/.venv.broken-$(date +%s)"
    "$UV" venv -q --python "$PY_VERSION" "$DIR/.venv" || die "venv creation failed"
    stage_changed
  fi
  stage_end "$("$PY" -V 2>&1)"

  # 4. packages ──────────────────────────────────────────────────────────
  stage_begin packages
  local REQ="$SRC/requirements-node.txt"
  if [ -n "$UV" ] || [ -x "$HOME/.local/bin/uv" ]; then
    UV="${UV:-$HOME/.local/bin/uv}"
    local plan; plan=$("$UV" pip install --dry-run --python "$PY" -r "$REQ" 2>&1) || die "requirements do not resolve: $plan"
    if printf '%s' "$plan" | grep -q "Would install"; then
      guard_change "Python packages"
      "$UV" pip install -q --python "$PY" -r "$REQ" || die "package install failed"
      stage_changed
    fi
  else
    # A venv that predates this installer (created with pip): keep using pip.
    if ! "$PY" -m pip install --dry-run -q -r "$REQ" 2>/dev/null | grep -q "Would install"; then :; else
      guard_change "Python packages"
      "$PY" -m pip install -q -r "$REQ" || die "package install failed"
      stage_changed
    fi
  fi
  stage_end "mlx $("$PY" -c 'import importlib.metadata as m; print(m.version("mlx"))' 2>/dev/null || echo '?')"

  # 5. files ─────────────────────────────────────────────────────────────
  stage_begin files
  local SITE; SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  local n=0 f
  local pending=""
  for f in runner.py auto_parallel.py exo_stubs.py inference.py inference_pipe.py persistence.py \
           doctor_node.py doctor-manifest.json odyssai-x; do
    [ -f "$DIR/$f" ] && cmp -s "$SRC/scripts/$f" "$DIR/$f" || pending="$pending $f"
  done
  cmp -s "$REQ" "$DIR/requirements-node.txt" 2>/dev/null || pending="$pending requirements-node.txt"
  for f in "$SRC"/scripts/patches/*.py; do cmp -s "$f" "$DIR/patches/$(basename "$f")" 2>/dev/null || pending="$pending patches/$(basename "$f")"; done
  for f in "$SRC"/scripts/mlx_models/*.py; do cmp -s "$f" "$SITE/mlx_lm/models/$(basename "$f")" 2>/dev/null || pending="$pending models/$(basename "$f")"; done
  if [ -n "$pending" ]; then
    guard_change "node files"
    for f in runner.py auto_parallel.py exo_stubs.py inference.py inference_pipe.py persistence.py \
             doctor_node.py doctor-manifest.json odyssai-x; do
      copy_if_changed "$SRC/scripts/$f" "$DIR/$f" && n=$((n + 1)) || true
    done
    copy_if_changed "$REQ" "$DIR/requirements-node.txt" && n=$((n + 1)) || true
    chmod +x "$DIR/odyssai-x"
    touch "$DIR/.advertise-restart"
    for f in "$SRC"/scripts/patches/*.py; do copy_if_changed "$f" "$DIR/patches/$(basename "$f")" && n=$((n + 1)) || true; done
    for f in "$SRC"/scripts/mlx_models/*.py; do copy_if_changed "$f" "$SITE/mlx_lm/models/$(basename "$f")" && n=$((n + 1)) || true; done
    stage_changed
  fi
  local VER; VER=$(sed -n 's/^APP_VERSION = "\(.*\)"/\1/p' "$SRC/scripts/api.py" | head -1)
  if [ -n "$VER" ] && [ "$(cat "$DIR/version" 2>/dev/null)" != "$VER" ]; then
    printf '%s\n' "$VER" > "$DIR/version"; n=$((n + 1)); stage_changed; touch "$DIR/.advertise-restart"
  fi
  if [ -n "${ODYSSAI_X_ENGINE:-}" ] && [ "$(cat "$DIR/engine-url" 2>/dev/null)" != "${ODYSSAI_X_ENGINE%/}" ]; then
    printf '%s\n' "${ODYSSAI_X_ENGINE%/}" > "$DIR/engine-url"; n=$((n + 1)); stage_changed; touch "$DIR/.advertise-restart"
  fi
  stage_end "$n file(s) updated"

  # 6. jaccl ─────────────────────────────────────────────────────────────
  stage_begin jaccl
  patch_jaccl "$PY" "$DIR/doctor-manifest.json"
  stage_end "$JACCL_NOTE"

  # 7. wired-limit ───────────────────────────────────────────────────────
  stage_begin wired-limit
  local label
  for label in eu.odyssai.wiredlimit com.thecompai.wired-limit; do
    [ -f "/Library/LaunchDaemons/$label.plist" ] && break
    label=""
  done
  local WANT_MB="${ODYSSAI_X_WIRED_MB:-}" HAVE_MB=""
  [ -n "$label" ] && HAVE_MB=$(sed -n 's/.*iogpu.wired_limit_mb=\([0-9]*\).*/\1/p' "/Library/LaunchDaemons/$label.plist" | head -1)
  if [ -n "$label" ] && { [ -z "$WANT_MB" ] || [ "$WANT_MB" = "$HAVE_MB" ]; }; then
    stage_end "$(sysctl -n iogpu.wired_limit_mb) MB, daemon $label"
  elif [ -n "$label" ] && [ "$label" != eu.odyssai.wiredlimit ]; then
    warn "wired limit: $label is not this installer's daemon; change its value by hand to $WANT_MB"
    stage_end "$(sysctl -n iogpu.wired_limit_mb) MB, daemon $label (not changed)"
  else
    local RAM_MB CUR VAL RES
    RAM_MB=$(( $(sysctl -n hw.memsize) / 1048576 ))
    CUR=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
    RES=$(( RAM_MB / 16 )); [ "$RES" -lt 8192 ] && RES=8192
    # An existing daemon is rewritten only when ODYSSAI_X_WIRED_MB asks for a new
    # value (.42/.49 kept 80 GB while a replica needed 87, 2026-10-01).
    VAL="$WANT_MB"
    [ -n "$VAL" ] || { [ "$CUR" -gt 0 ] && VAL=$CUR; } || VAL=$(( RAM_MB - RES ))
    [ "$VAL" -lt "$RAM_MB" ] || die "wired limit $VAL MB >= $RAM_MB MB of RAM"
    local PLIST=/Library/LaunchDaemons/eu.odyssai.wiredlimit.plist
    local CMD="sudo tee $PLIST >/dev/null && sudo chown root:wheel $PLIST && sudo chmod 644 $PLIST && sudo launchctl bootstrap system $PLIST"
    if can_sudo; then
      say "      wired limit ${VAL} MB on ${RAM_MB} MB of RAM — sudo asks for this Mac's password:"
      cat <<XML | sudo tee "$PLIST" >/dev/null
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>eu.odyssai.wiredlimit</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/sbin/sysctl</string>
        <string>iogpu.wired_limit_mb=${VAL}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/var/log/eu.odyssai.wiredlimit.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/eu.odyssai.wiredlimit.log</string>
</dict>
</plist>
XML
      sudo launchctl bootout system/eu.odyssai.wiredlimit 2>/dev/null || true
      sudo chown root:wheel "$PLIST" && sudo chmod 644 "$PLIST" && sudo launchctl bootstrap system "$PLIST" \
        || die "wired-limit daemon install failed"
      sudo sysctl "iogpu.wired_limit_mb=$VAL" >/dev/null || true
      stage_changed
      if [ -n "$label" ]; then stage_end "$(sysctl -n iogpu.wired_limit_mb) MB, daemon updated (was ${HAVE_MB:-?})"
      else stage_end "$(sysctl -n iogpu.wired_limit_mb) MB, daemon installed"; fi
    else
      warn "no terminal for sudo: rerun this installer from a terminal (or ssh -t) to set the wired limit"
      stage_end "skipped (no terminal for sudo)"
    fi
  fi

  # 8. network ───────────────────────────────────────────────────────────
  # The Thunderbolt network RDMA needs (scripts/rdma-onboard.sh wraps the exo
  # recipe). It rebuilds network services, so it runs only on the Mac's own
  # console (rdma-onboard refuses under SSH), never under a serving runner, and
  # never over an exo-managed network without the operator's choice.
  stage_begin network
  local NET_OUT NET_RC=0 ONB="$SRC/scripts/rdma-onboard.sh"
  if [ "${ODYSSAI_X_NETWORK:-1}" = 0 ]; then
    stage_end "skipped (ODYSSAI_X_NETWORK=0)"
  elif [ "$RDMA_OS" != 1 ]; then
    stage_end "skipped: macOS $MACOS has no RDMA"
  elif ! rdma_ctl status 2>/dev/null | grep -q enabled; then
    stage_end "waiting: RDMA not enabled (recoveryOS, see README \"Enable RDMA\")"
  else
    NET_OUT=$(bash "$ONB" --check 2>&1) || NET_RC=$?
    case "$NET_RC" in
      0)  stage_end "ready" ;;
      10) warn "network: applied but the Thunderbolt ports have no link-local address yet: reboot this Mac once"
          stage_end "needs one reboot" ;;
      11) if printf '%s' "$NET_OUT" | grep -q "exo present"; then
            warn "network: this Mac's Thunderbolt network is managed by exo. Keep it, or replace it: sudo bash $ONB --apply --console --migrate-exo (on this Mac's console)"
            stage_end "exo-managed: not changed"
          elif [ "$SERVING" = 1 ]; then
            warn "network: this node is serving a model; unload it, then rerun the installer here"
            stage_end "skipped (serving)"
          elif [ -n "${SSH_CONNECTION:-}" ]; then
            warn "network: must run on this Mac's own console (it rebuilds network services and could cut SSH): open Terminal on $(hostname -s) and rerun the installer"
            stage_end "skipped (over SSH)"
          elif can_sudo; then
            say "      Thunderbolt network for RDMA — sudo asks for this Mac's password:"
            NET_RC=0; sudo bash "$ONB" --apply --console || NET_RC=$?
            case "$NET_RC" in
              0)  stage_changed; stage_end "applied" ;;
              10) stage_changed; warn "network: applied; reboot this Mac once so the Thunderbolt ports get their link-local addresses"
                  stage_end "applied, needs one reboot" ;;
              *)  warn "network: rdma-onboard refused or failed (exit $NET_RC); its log says why"
                  stage_end "blocked (exit $NET_RC)" ;;
            esac
          else
            warn "no terminal for sudo: rerun this installer from a terminal on this Mac to set up the Thunderbolt network"
            stage_end "skipped (no terminal for sudo)"
          fi ;;
      *)  warn "network: $(printf '%s' "$NET_OUT" | grep -v '^$' | tail -2 | tr '\n' ' ')"
          stage_end "blocked (exit $NET_RC)" ;;
    esac
  fi

  # 9. vision ────────────────────────────────────────────────────────────
  # mlx-vlm at the repo pin in ~/.venvs/mlx-vlm (Python 3.12 from uv), with the
  # same patched libjaccl: single-node and distributed VL pools run there.
  stage_begin vision
  local VLOG="$DIR/install-vision.log" VPY="${VLM_VENV:-$HOME/.venvs/mlx-vlm}/bin/python" VRC=0
  if [ "${ODYSSAI_X_VISION:-1}" = 0 ]; then
    stage_end "skipped (ODYSSAI_X_VISION=0)"
  else
    PATH="$HOME/.local/bin:$PATH" bash "$SRC/scripts/install-mlx-vlm.sh" --local >"$VLOG" 2>&1 || VRC=$?
    if [ "$VRC" = 3 ]; then
      warn "vision: a VL server runs on this node and its venv would change; unload it, then rerun"
      stage_end "skipped (serving)"
    elif [ "$VRC" != 0 ]; then
      warn "vision: install failed (exit $VRC): $(grep -v '^$' "$VLOG" | tail -1)"
      stage_end "failed, see $VLOG"
    else
      grep -q "nothing to install" "$VLOG" && grep -q "pins OK" "$VLOG" && grep -q "already applied" "$VLOG" || stage_changed
      if [ "$SERVING" = 1 ]; then JACCL_NOTE="libjaccl not checked (serving)"; else patch_jaccl "$VPY" "$DIR/doctor-manifest.json"; fi
      stage_end "mlx-vlm $("$VPY" -c 'import importlib.metadata as m; print(m.version("mlx-vlm"))' 2>/dev/null || echo '?'), libjaccl ${JACCL_NOTE}"
    fi
  fi

  # 10. discovery ────────────────────────────────────────────────────────
  # A LaunchDaemon (starts at boot, no login needed on a headless node) running
  # `odyssai-x advertise` as this user, at launchd's lowest priority
  # (ProcessType Background). It touches no runner, so it needs no FORCE.
  stage_begin discovery
  local DPL=/Library/LaunchDaemons/eu.odyssai.x.node.plist DXML
  DXML=$(cat <<XML
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>eu.odyssai.x.node</string>
    <key>UserName</key>
    <string>$(id -un)</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>$DIR/odyssai-x</string>
        <string>advertise</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>30</integer>
    <key>ProcessType</key>
    <string>Background</string>
    <key>StandardErrorPath</key>
    <string>/tmp/eu.odyssai.x.node.log</string>
</dict>
</plist>
XML
)
  if [ -f "$DPL" ] && [ "$(cat "$DPL")" = "$DXML" ]; then
    if [ -f "$DIR/.advertise-restart" ]; then
      sudo -n launchctl kickstart -k system/eu.odyssai.x.node 2>/dev/null && rm -f "$DIR/.advertise-restart"
    fi
    stage_end "announcing ($(cat "$DIR/engine-url" 2>/dev/null || echo 'engine: looking on the LAN'))"
  elif can_sudo; then
    printf '%s\n' "$DXML" | sudo tee "$DPL" >/dev/null && sudo chown root:wheel "$DPL" && sudo chmod 644 "$DPL" \
      || die "discovery daemon install failed"
    sudo launchctl bootout system "$DPL" 2>/dev/null || true
    sudo launchctl bootstrap system "$DPL" || die "discovery daemon did not start"
    rm -f "$DIR/.advertise-restart"
    stage_changed
    stage_end "daemon installed ($(cat "$DIR/engine-url" 2>/dev/null || echo 'engine: looking on the LAN'))"
  else
    warn "no terminal for sudo: rerun from a terminal (or ssh -t) to install the discovery daemon"
    stage_end "skipped (no terminal for sudo)"
  fi

  # 11. doctor ───────────────────────────────────────────────────────────
  stage_begin doctor
  say ""
  local rc=0
  "$DIR/odyssai-x" doctor --models-dir "$MODELS_DIR" || rc=$?
  say ""
  stage_end "exit $rc"

  say ""
  if [ "$CHANGED" = 0 ]; then say "already up to date"; fi
  say "total $(( $(date +%s) - T0 ))s"
  say ""
  say "Next: add this node to a pool of your topology (~/.odysseus/topology.yaml on the server):"
  say "    - {rank: <n>, id: $(hostname -s), ssh: $(id -un)@$(hostname -s).local, models_dir: $MODELS_DIR}"
  say "Check it again any time: $DIR/odyssai-x doctor"
  if [ "$RDMA_OS" = 1 ] && ! rdma_ctl status 2>/dev/null | grep -q enabled; then
    say "RDMA over Thunderbolt: enable it once from recoveryOS (see README \"Enable RDMA\"), then rerun this installer on this Mac."
  fi
  return 0
}

main "$@"
