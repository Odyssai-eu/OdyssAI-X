#!/usr/bin/env bash
# Build the OdyssAI-patched JACCL (vendor/jaccl, base MLX v0.32.2 + PATCHES.md)
# as a drop-in libjaccl.dylib for the mlx 0.32.2 wheel.
#
# The build runs on a cluster node (needs Xcode/CLT with the macOS 26.2+ SDK and
# cmake — /opt/homebrew/bin/cmake on the Ultras); the resulting dylib is copied
# back to vendor/jaccl/build/libjaccl.dylib (gitignored) for install-jaccl.sh.
#
# Usage: scripts/build-jaccl.sh <ssh-build-host>     (or JACCL_BUILD_HOST=…)
set -euo pipefail
HOST="${1:-${JACCL_BUILD_HOST:-}}"; [ -n "$HOST" ] || { echo "usage: $0 <ssh-build-host>   (an Apple Silicon node with Xcode/CLT and cmake)" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/vendor/jaccl"
OUT="$SRC/build"
mkdir -p "$OUT"
echo "[build-jaccl] syncing $SRC → $HOST:/tmp/jaccl-src"
rsync -a --delete --exclude build "$SRC/" "$HOST:/tmp/jaccl-src/"
ssh "$HOST" 'set -e; cd /tmp/jaccl-src && rm -rf build && mkdir build && cd build
  CMAKE=$(command -v cmake || echo /opt/homebrew/bin/cmake)
  "$CMAKE" .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=26.2 -DCMAKE_INSTALL_NAME_DIR=@rpath >/dev/null
  make -j8 2>&1 | grep -E "error|warning|Built target" || true
  test -f libjaccl.dylib
  otool -D libjaccl.dylib | tail -1 | grep -q "@rpath/libjaccl.dylib"
  codesign --force --sign - libjaccl.dylib >/dev/null 2>&1 || true
  codesign -dv libjaccl.dylib 2>&1 | grep -q adhoc'
scp -q "$HOST:/tmp/jaccl-src/build/libjaccl.dylib" "$OUT/libjaccl.dylib"
(cd "$SRC" && git rev-parse --short HEAD 2>/dev/null || true) > "$OUT/BUILT-FROM" || true
echo "[build-jaccl] $OUT/libjaccl.dylib ($(stat -f %z "$OUT/libjaccl.dylib") bytes)"
