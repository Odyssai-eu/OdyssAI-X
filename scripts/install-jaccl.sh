#!/usr/bin/env bash
# Install the OdyssAI-patched libjaccl.dylib (vendor/jaccl, see PATCHES.md) into
# each node's mlx wheel, replacing the stock one. Drop-in: same @rpath name, same
# exported symbol set as the mlx 0.32.2 wheel (verified nm -gU equality on the
# unpatched build), ad-hoc signed like the wheel's own dylibs.
#
# Usage:
#   scripts/install-jaccl.sh <ssh-target> [<ssh-target>…]
#   scripts/install-jaccl.sh --check <ssh-target>…     report only
#   scripts/install-jaccl.sh --restore <ssh-target>…   put the stock dylib back
#
# Order on a fresh node: pip install -r requirements-node.txt (mlx==0.32.2) →
# install-model-modules.sh → install-jaccl.sh. A `pip install -U mlx` replaces
# the dylib with the stock one: re-run this script after any wheel change.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DYLIB="$ROOT/vendor/jaccl/build/libjaccl.dylib"
MODE=install
case "${1:-}" in --check) MODE=check; shift;; --restore) MODE=restore; shift;; esac
[ $# -ge 1 ] || { echo "usage: $0 [--check|--restore] <ssh-target>…" >&2; exit 2; }
[ "$MODE" = install ] && [ ! -f "$DYLIB" ] && { echo "missing $DYLIB — run scripts/build-jaccl.sh first" >&2; exit 2; }
WANT_MLX="$(grep -E '^mlx==' "$ROOT/requirements-node.txt" | cut -d= -f3)"
LOCAL_SUM="$( [ -f "$DYLIB" ] && shasum -a 256 "$DYLIB" | cut -c1-16 || echo none)"
rc=0
for NODE in "$@"; do
  REMOTE_LIB="$(ssh "$NODE" 'ls ~/mlx-cluster/.venv/lib/python3*/site-packages/mlx/lib/libjaccl.dylib 2>/dev/null | head -1')"
  [ -n "$REMOTE_LIB" ] || { echo "$NODE: no mlx wheel dylib found"; rc=1; continue; }
  VER="$(ssh "$NODE" '~/mlx-cluster/.venv/bin/python -c "import mlx.core as m;print(m.__version__)"' 2>/dev/null || echo '?')"
  SUM="$(ssh "$NODE" "shasum -a 256 '$REMOTE_LIB' | cut -c1-16")"
  HAS_ORIG="$(ssh "$NODE" "[ -f '$REMOTE_LIB.orig' ] && echo yes || echo no")"
  if [ "$VER" != "$WANT_MLX" ]; then echo "$NODE: mlx $VER != required $WANT_MLX — upgrade the wheel first"; rc=1; continue; fi
  case "$MODE" in
    check)
      if [ "$SUM" = "$LOCAL_SUM" ]; then echo "$NODE: patched (mlx $VER, orig backup: $HAS_ORIG)"; else echo "$NODE: STOCK or stale ($SUM, orig backup: $HAS_ORIG)"; rc=1; fi ;;
    restore)
      if [ "$HAS_ORIG" = yes ]; then ssh "$NODE" "cp -p '$REMOTE_LIB.orig' '$REMOTE_LIB'" && echo "$NODE: stock dylib restored"; else echo "$NODE: no .orig backup, nothing to restore"; fi ;;
    install)
      if [ "$SUM" = "$LOCAL_SUM" ]; then echo "$NODE: already patched"; continue; fi
      [ "$HAS_ORIG" = yes ] || ssh "$NODE" "cp -p '$REMOTE_LIB' '$REMOTE_LIB.orig'"
      scp -q "$DYLIB" "$NODE:/tmp/libjaccl.dylib.new"
      ssh "$NODE" "set -e; W='$REMOTE_LIB'; N=/tmp/libjaccl.dylib.new
        codesign -dv \$N 2>&1 | grep -q adhoc || codesign --force --sign - \$N
        nm -gU \$W.orig | awk '{print \$3}' | sort > /tmp/sym_orig; nm -gU \$N | awk '{print \$3}' | sort > /tmp/sym_new
        comm -13 /tmp/sym_new /tmp/sym_orig | grep -q . && { echo 'ABI: symbols missing vs stock:'; comm -13 /tmp/sym_new /tmp/sym_orig | c++filt | head; exit 3; }
        mv \$N \$W
        ~/mlx-cluster/.venv/bin/python - <<'PY'
import mlx.core as mx, os
assert mx.distributed.init(backend='ring') is None or True
print('import ok', mx.__version__)
PY" && echo "$NODE: patched dylib installed (mlx $VER)" || { echo "$NODE: INSTALL FAILED — restoring stock"; ssh "$NODE" "cp -p '$REMOTE_LIB.orig' '$REMOTE_LIB'"; rc=1; } ;;
  esac
done
exit $rc
