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
#   7. wired-limit the GPU wired-memory limit, kept across reboots (asks sudo)
#   8. discovery   the node announces itself: Bonjour `_odyssai._tcp` and a
#                  heartbeat every 5 s to its orchestrator (launchd, background)
#   9. doctor      `odyssai-x doctor`: what this node still lacks, if anything
#
# Out of scope: RDMA activation (recoveryOS: `rdma_ctl enable`), the
# Thunderbolt network setup (scripts/rdma-onboard.sh, run as root), the vision
# venv (scripts/install-mlx-vlm.sh).
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
STAGES=9
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
    if [ -d /Volumes/models/odysseus ]; then MODELS_DIR=/Volumes/models/odysseus; else MODELS_DIR="$HOME/mlx-models"; fi
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
  local WANT; WANT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["jaccl_sha256"])' "$DIR/doctor-manifest.json")
  local LIB="$SITE/mlx/lib/libjaccl.dylib"
  [ -f "$LIB" ] || die "mlx has no lib/libjaccl.dylib: the mlx wheel is incomplete"
  if [ "$RDMA_OS" != 1 ]; then
    stage_end "skipped: macOS $MACOS, stock JACCL kept"
  elif [ "$(shasum -a 256 "$LIB" | cut -d' ' -f1)" = "$WANT" ]; then
    stage_end "patched"
  else
    guard_change "libjaccl.dylib"
    local TMP; TMP=$(mktemp -d)
    curl -fsSL -o "$TMP/libjaccl.dylib" "https://github.com/$REPO/releases/download/$JACCL_TAG/libjaccl.dylib" \
      || die "cannot download $JACCL_TAG/libjaccl.dylib"
    [ "$(shasum -a 256 "$TMP/libjaccl.dylib" | cut -d' ' -f1)" = "$WANT" ] \
      || die "downloaded libjaccl.dylib does not match the manifest sha256: refusing it"
    [ -f "$LIB.orig" ] || cp -p "$LIB" "$LIB.orig"
    cp "$TMP/libjaccl.dylib" "$LIB.odx-new" && mv "$LIB.odx-new" "$LIB"
    rm -rf "$TMP"
    if ! "$PY" -c 'import mlx.core as mx; mx.distributed.is_available()' >/dev/null 2>&1; then
      cp -p "$LIB.orig" "$LIB.odx-new" && mv "$LIB.odx-new" "$LIB"
      die "mlx does not import with the patched JACCL: stock one restored"
    fi
    stage_changed
    stage_end "patched ($JACCL_TAG)"
  fi

  # 7. wired-limit ───────────────────────────────────────────────────────
  stage_begin wired-limit
  local label
  for label in eu.odyssai.wiredlimit com.thecompai.wired-limit; do
    [ -f "/Library/LaunchDaemons/$label.plist" ] && break
    label=""
  done
  if [ -n "$label" ]; then
    stage_end "$(sysctl -n iogpu.wired_limit_mb) MB, daemon $label"
  else
    local RAM_MB CUR VAL RES
    RAM_MB=$(( $(sysctl -n hw.memsize) / 1048576 ))
    CUR=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
    RES=$(( RAM_MB / 16 )); [ "$RES" -lt 8192 ] && RES=8192
    VAL="${ODYSSAI_X_WIRED_MB:-}"
    [ -n "$VAL" ] || { [ "$CUR" -gt 0 ] && VAL=$CUR; } || VAL=$(( RAM_MB - RES ))
    [ "$VAL" -lt "$RAM_MB" ] || die "wired limit $VAL MB >= $RAM_MB MB of RAM"
    local PLIST=/Library/LaunchDaemons/eu.odyssai.wiredlimit.plist
    local CMD="sudo tee $PLIST >/dev/null && sudo chown root:wheel $PLIST && sudo chmod 644 $PLIST && sudo launchctl bootstrap system $PLIST"
    if sudo -n true 2>/dev/null || [ -t 0 ] || { : </dev/tty; } 2>/dev/null; then
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
      sudo chown root:wheel "$PLIST" && sudo chmod 644 "$PLIST" && sudo launchctl bootstrap system "$PLIST" \
        || die "wired-limit daemon install failed"
      stage_changed
      stage_end "$(sysctl -n iogpu.wired_limit_mb) MB, daemon installed"
    else
      warn "no terminal for sudo: rerun this installer from a terminal (or ssh -t) to set the wired limit"
      stage_end "skipped (no terminal for sudo)"
    fi
  fi

  # 8. discovery ─────────────────────────────────────────────────────────
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
  elif sudo -n true 2>/dev/null || [ -t 0 ] || { : </dev/tty; } 2>/dev/null; then
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

  # 9. doctor ────────────────────────────────────────────────────────────
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
    say "RDMA over Thunderbolt: enable it once from recoveryOS (\`rdma_ctl enable\`), then run scripts/rdma-onboard.sh as root."
  fi
  return 0
}

main "$@"
