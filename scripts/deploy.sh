#!/usr/bin/env bash
set -euo pipefail

# Deploy scripts/api.py + scripts/dashboard.html to the running Odysseus
# containers on the configured hosts. ALWAYS bumps ODYSSEUS_VERSION first
# (unless `skip`), commits the bump, pushes to GitHub, then hot-patches
# the containers and restarts only when needed.
#
# Why this script exists
# ──────────────────────
# Cardinal rule per `~/.claude/CLAUDE.md` "Deploy operations" :
#   "ce qui est sur le serveur = ce qui est sur main"
#
# The hand-rolled `scp + docker cp + docker restart` flow drifted from that
# invariant repeatedly :
#  - hot-patches landed on the container but the version field in api.py
#    on `main` was never bumped, so the dashboard kept showing the old
#    number even after a real ship
#  - Sophie couldn't tell from the UI whether what she was seeing on .39
#    was last week's code or tonight's
#
# This script enforces the bump as a hard precondition. There is no path
# from "I edited api.py" to "deployed" that doesn't increment a version.
#
# Usage
# ─────
#   ./scripts/deploy.sh           # patch bump (default), push, deploy
#   ./scripts/deploy.sh patch     # explicit patch
#   ./scripts/deploy.sh minor     # 1.8.0 → 1.9.0
#   ./scripts/deploy.sh major     # 1.8.0 → 2.0.0
#   ./scripts/deploy.sh skip      # already bumped+committed, just deploy
#
# Configuration via env vars (defaults reflect Sophie's prod layout) :
#
#   ODYSSEUS_DEPLOY_HOSTS   space-separated list of SSH targets running an
#                            Odysseus container (default: the two known
#                            production hosts).
#   ODYSSEUS_CONTAINER       container name (default: odyssai-odysseus)
#   ODYSSEUS_REMOTE_DOCKER   path to docker on the remote, in case
#                            $PATH doesn't include it under non-interactive
#                            SSH (default: /usr/local/bin/docker)
#   ODYSSEUS_VERIFY_URL      one of the deploy hosts' API base URL for the
#                            post-deploy health probe (default: first host).
#
# Anything else (cluster-config.json, weights, RDMA wiring) is data —
# never touched here.

BUMP=${1:-patch}
case "$BUMP" in
  patch|minor|major|skip) ;;
  *)
    echo "✗ unknown bump kind '$BUMP' — use patch (default), minor, major, or skip" >&2
    exit 1
    ;;
esac

# Default hosts = Sophie's two production Odysseus orchestrators. Operators
# running this on a different LAN should set ODYSSEUS_DEPLOY_HOSTS in their
# env. Override is space-separated so `ODYSSEUS_DEPLOY_HOSTS="a@x b@y"` works.
HOSTS=${ODYSSEUS_DEPLOY_HOSTS:-"admin@192.168.86.39 admin@192.168.86.141"}
CONTAINER=${ODYSSEUS_CONTAINER:-odyssai-odysseus}
DOCKER=${ODYSSEUS_REMOTE_DOCKER:-/usr/local/bin/docker}
VERIFY_HOST=$(echo "$HOSTS" | awk '{print $1}')
VERIFY_URL=${ODYSSEUS_VERIFY_URL:-"http://${VERIFY_HOST#*@}:8000"}

cd "$(dirname "$0")/.."

# ── Pre-flight: working tree clean enough to deploy ──────────────────────
#
# The script will auto-stage scripts/api.py (after bumping the version line)
# during a non-skip run. Any OTHER uncommitted change means the deploy
# would push container-side code that isn't on main — exactly the drift
# this script is built to prevent. Refuse.
#
# Allowance : if BUMP=skip the caller has already committed everything,
# so the working tree must be 100% clean.
if [ "$BUMP" = "skip" ]; then
  DIRTY=$(git status --porcelain || true)
else
  # Filter out scripts/api.py because we're about to mutate it via
  # bump-version.sh and stage it ourselves.
  DIRTY=$(git status --porcelain | grep -v -E '^.M scripts/api\.py$' || true)
fi
if [ -n "$DIRTY" ]; then
  echo "✗ Refusing to deploy — uncommitted changes detected :"
  echo
  echo "$DIRTY" | sed 's/^/  /'
  echo
  echo "  Either commit them (so they land on main and ship with the bump),"
  echo "  or 'git stash' to set them aside."
  echo
  echo "  The invariant : ce qui est sur le serveur = ce qui est sur main."
  exit 1
fi

# ── Bump ──────────────────────────────────────────────────────────────────
if [ "$BUMP" != "skip" ]; then
  NEW_VERSION=$(./scripts/bump-version.sh "$BUMP")
  git add scripts/api.py
  git commit -m "chore: bump to v$NEW_VERSION"
  echo "→ bumped to v$NEW_VERSION"
else
  NEW_VERSION=$(grep -E '^ODYSSEUS_VERSION = "' scripts/api.py | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
  echo "→ deploying current v$NEW_VERSION (no bump)"
fi

git push github main

# ── Hot-deploy to each host ──────────────────────────────────────────────
#
# Pattern : scp the two updated files to /tmp on the remote, then
# `docker cp` them into the container (preserves the container's image —
# we don't rebuild). Restart only when api.py changed (the dashboard.html
# is served as a static file and picks up the new content on next reload).
#
# We always copy both because the cost is sub-second and it eliminates the
# "deployed half of the change" failure mode.
for HOST in $HOSTS; do
  echo
  echo "→ deploying to $HOST"
  scp scripts/api.py scripts/dashboard.html "$HOST:/tmp/"
  ssh "$HOST" "$DOCKER cp /tmp/api.py $CONTAINER:/app/api.py"
  ssh "$HOST" "$DOCKER cp /tmp/dashboard.html $CONTAINER:/app/dashboard.html"
  ssh "$HOST" "$DOCKER restart $CONTAINER" >/dev/null
  echo "  ✓ api.py + dashboard.html copied, container restarted"
done

# ── Post-deploy smoke ────────────────────────────────────────────────────
#
# Verify the live version field matches what we just shipped. If the
# field doesn't match, the deploy silently failed (e.g. docker cp landed
# on a stopped container, or a syntax error in api.py crashed startup).
# Loud failure here is the whole point.
#
# Be patient on the wait : when a cluster has loaded pools, the container
# replays orphan sweeps + runner startup before FastAPI accepts requests
# (observed up to ~45s on .39 when 3 ranks are coming back). Total budget
# ~90s, polled every 4s.
echo
echo "→ verifying $VERIFY_URL"
LIVE_VERSION=""
for attempt in $(seq 1 22); do
  sleep 4
  LIVE_VERSION=$(curl -s --max-time 3 "$VERIFY_URL/health" 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('version',''))" 2>/dev/null || echo "")
  if [ -n "$LIVE_VERSION" ]; then
    break
  fi
done

if [ -z "$LIVE_VERSION" ]; then
  echo "✗ couldn't reach $VERIFY_URL/health after deploy — container may have failed to start"
  echo "  check: ssh $VERIFY_HOST '$DOCKER logs --tail=50 $CONTAINER'"
  exit 1
fi
if [ "$LIVE_VERSION" != "$NEW_VERSION" ]; then
  echo "✗ live version mismatch — expected v$NEW_VERSION, got v$LIVE_VERSION"
  echo "  the container is probably running stale code despite docker cp succeeding."
  echo "  check: ssh $VERIFY_HOST '$DOCKER logs --tail=50 $CONTAINER'"
  exit 1
fi

echo
echo "✓ v$NEW_VERSION live on $VERIFY_URL (and the other hosts)"
echo "  next : comment 'Verified live ✅' on the GitHub issue you just closed."
