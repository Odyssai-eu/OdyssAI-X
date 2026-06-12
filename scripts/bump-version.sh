#!/usr/bin/env bash
# Bump APP_VERSION in scripts/api.py.
#
# Usage:
#   ./scripts/bump-version.sh patch    # 1.7.2 → 1.7.3
#   ./scripts/bump-version.sh minor    # 1.7.2 → 1.8.0 (default)
#   ./scripts/bump-version.sh major    # 1.7.2 → 2.0.0
#
# Prints the new version on stdout. Does NOT commit — the caller does
# that as part of its deploy step so the bump lands in the same commit
# as the change that justifies it.

set -euo pipefail

KIND=${1:-minor}
case "$KIND" in
  patch|minor|major) ;;
  *)
    echo "✗ unknown bump kind '$KIND' — use patch, minor or major" >&2
    exit 1
    ;;
esac

cd "$(dirname "$0")/.."

CUR=$(grep -E '^APP_VERSION = "' scripts/api.py | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
if [[ -z "$CUR" ]]; then
  echo "✗ couldn't read current APP_VERSION from scripts/api.py" >&2
  exit 1
fi

IFS='.' read -r MAJ MIN PATCH <<<"$CUR"
case "$KIND" in
  patch) NEW="$MAJ.$MIN.$((PATCH + 1))" ;;
  minor) NEW="$MAJ.$((MIN + 1)).0" ;;
  major) NEW="$((MAJ + 1)).0.0" ;;
esac

# Use a portable sed (works on macOS + GNU). Match the line precisely so
# we don't touch the documentation example above the constant.
if [[ "$(uname)" == "Darwin" ]]; then
  sed -i '' "s|^APP_VERSION = \"$CUR\"|APP_VERSION = \"$NEW\"|" scripts/api.py
else
  sed -i "s|^APP_VERSION = \"$CUR\"|APP_VERSION = \"$NEW\"|" scripts/api.py
fi

# Sanity check the edit landed.
if ! grep -qE "^APP_VERSION = \"$NEW\"" scripts/api.py; then
  echo "✗ failed to write new version $NEW into scripts/api.py" >&2
  exit 1
fi

echo "$NEW"
