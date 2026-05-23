#!/bin/bash
#
# scripts/lean-node.sh — strip background services + hide unused Apple
# apps on a macOS cluster node so MLX gets the wired memory and CPU it
# actually needs. Safe to run multiple times. Reversible: no destructive
# operations (no app deletion, no SIP changes).
#
# Target: a Mac that will only ever run MLX inference (a node already
# bootstrapped by scripts/bootstrap-node.sh). Skip on a workstation
# that also serves human use.
#
#   Usage:
#     scripts/lean-node.sh [--models-dir <path>] [--dry-run]
#     ssh admin@node 'bash -s' < scripts/lean-node.sh
#
#   Examples:
#     scripts/lean-node.sh
#     scripts/lean-node.sh --models-dir /Volumes/models --dry-run
#
# What it does:
#   1. Hides ~15 unused Apple apps from Launchpad/Finder via chflags
#      (Mail, Maps, FaceTime, News, Stocks, Music, Photos, iWork, …)
#   2. Turns off Spotlight indexing on the models SSD (huge volume,
#      zero value being mdworker'd over)
#   3. Disables Time Machine (worker nodes don't get backed up)
#   4. launchctl disables iCloud / Photos / News / Music background
#      daemons that hold wired memory on idle
#   5. pmset off all sleep modes (sleeping mid-inference = dead pool)
#
# All actions are reversible — see the "Restore" section printed at
# the end.

set -euo pipefail

MODELS_DIR="/Volumes/models"
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//' | sed '$d'; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
  esac
done

# Sanity check: macOS only.
if [ "$(uname -s)" != "Darwin" ]; then
  echo "lean-node.sh: macOS only (detected $(uname -s))" >&2
  exit 1
fi

# Wrap actions so --dry-run echoes instead of executing.
run() {
  if [ "$DRY" -eq 1 ]; then
    echo "[dry] $*"
  else
    echo "[+]   $*"
    "$@" 2>/dev/null || echo "      (not applicable on this system — skipped)"
  fi
}

echo "=== lean-node.sh ==="
echo "    host:       $(hostname)"
echo "    macOS:      $(sw_vers -productVersion)"
echo "    models_dir: $MODELS_DIR"
echo "    dry-run:    $([ "$DRY" -eq 1 ] && echo yes || echo no)"
echo ""

# ─── 1. Hide unused Apple apps ────────────────────────────────────────
# `chflags hidden` only hides from Finder + Launchpad; the .app bundle
# stays on disk and stays launchable from spotlight or `open -a`. To
# undo : `sudo chflags nohidden /path/to/<App>.app`.
echo "── 1. Hiding unused Apple apps ──"
APPS_TO_HIDE=(
  Mail Maps FaceTime Music News Stocks Photos
  "Voice Memos" "Photo Booth" "Image Capture"
  Chess Stickies "Time Machine" "Find My"
  Reminders Calendar Contacts Notes Books TV
  GarageBand iMovie Keynote Numbers Pages
  Freeform "Tips" Podcasts
)
for app in "${APPS_TO_HIDE[@]}"; do
  for prefix in /System/Applications /Applications; do
    path="${prefix}/${app}.app"
    if [ -d "$path" ]; then
      run sudo chflags hidden "$path"
    fi
  done
done
# Rebuild the Launchpad cache so the hidden flag takes effect immediately.
run defaults write com.apple.dock ResetLaunchPad -bool true
run killall Dock

# ─── 2. Spotlight off on the models SSD ───────────────────────────────
# Indexing a multi-TB models directory wastes mdworker CPU + disk I/O
# and contributes nothing — we don't search model weight files.
echo ""
echo "── 2. Spotlight indexing on models dir ──"
if [ -d "$MODELS_DIR" ]; then
  run sudo mdutil -i off "$MODELS_DIR"
else
  echo "[skip] $MODELS_DIR not mounted — will need to re-run after the SSD is plugged in"
fi

# ─── 3. Time Machine off ──────────────────────────────────────────────
# Cluster nodes aren't backed up — state lives elsewhere.
echo ""
echo "── 3. Time Machine ──"
run sudo tmutil disable

# ─── 4. Disable background daemons ────────────────────────────────────
# Each of these holds anywhere from 50 MB to several GB of wired memory
# on idle (photoanalysisd is especially heavy on freshly-installed Macs
# with no Photos library — it churns on the empty index).
echo ""
echo "── 4. Background daemons ──"
USER_ID=$(id -u)
AGENTS=(
  com.apple.photoanalysisd        # Photos face/scene recognition
  com.apple.mediaanalysisd        # Media (incl. video) analysis
  com.apple.parsec-fbf            # Apple News content fetcher
  com.apple.AMPLibraryAgent       # Music library indexer
  com.apple.bird                  # iCloud Drive sync
  com.apple.cloudd                # CloudKit sync
  com.apple.commerce              # App Store background updates
  com.apple.knowledge-agent       # Siri / Spotlight learning
  com.apple.suggestd              # Spotlight suggestions
  com.apple.assistantd            # Siri
)
for agent in "${AGENTS[@]}"; do
  run launchctl disable "user/${USER_ID}/${agent}"
done

# ─── 5. Power management ──────────────────────────────────────────────
# A node that sleeps mid-load = killed pool, JACCL queue-pair errors,
# the works. Display can still sleep (saves nothing on a headless node
# but doesn't hurt).
echo ""
echo "── 5. Power management ──"
run sudo pmset -a sleep 0
run sudo pmset -a disksleep 0
run sudo pmset -a hibernatemode 0
run sudo pmset -a displaysleep 10

# ─── Summary + reversal ───────────────────────────────────────────────
echo ""
echo "=== done ==="
echo ""
echo "Reversal cheatsheet :"
echo "  Show an app again :   sudo chflags nohidden /Applications/<App>.app"
echo "  Spotlight back on :   sudo mdutil -i on $MODELS_DIR"
echo "  Time Machine back :   sudo tmutil enable"
echo "  Daemon back on :      launchctl enable user/${USER_ID}/<agent-id>"
echo "  Restore sleep :       sudo pmset -a sleep 1 disksleep 10 hibernatemode 3"
echo ""
echo "Expected gains on a freshly-installed Mac Studio cluster node :"
echo "  - wired memory : ~2-4 GB freed (photoanalysisd alone is often 1-2 GB)"
echo "  - background CPU : noticeably less mdworker / cloudd / parsec-fbf churn"
echo "  - power : node stays awake through long model loads"
