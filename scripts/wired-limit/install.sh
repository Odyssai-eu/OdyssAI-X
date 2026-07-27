#!/usr/bin/env bash
# Make iogpu.wired_limit_mb survive reboots on one cluster node.
#
# Installs a LaunchDaemon that re-applies the value at every boot. Needs root
# ON THE NODE, so it prompts for that node's password — which is why no
# automation runs this for you.
#
#   scripts/wired-limit/install.sh <ssh-target> <value-in-mb>
#
# e.g. scripts/wired-limit/install.sh admin@192.168.86.29 491520   # 480 GiB
#      scripts/wired-limit/install.sh admin@192.168.86.30 250880   # 245 GiB

set -euo pipefail

NODE="${1:-}"
VALUE="${2:-}"
LABEL="eu.odyssai.wiredlimit"
PLIST="/Library/LaunchDaemons/${LABEL}.plist"

if [ -z "$NODE" ] || [ -z "$VALUE" ]; then
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
if ! [[ "$VALUE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: value must be an integer number of MB, got '$VALUE'" >&2
  exit 2
fi

RAM_MB="$(ssh "$NODE" 'echo $(( $(sysctl -n hw.memsize) / 1048576 ))')"
if [ "$VALUE" -ge "$RAM_MB" ]; then
  echo "ERROR: $VALUE MB >= the node's $RAM_MB MB of RAM — refusing" >&2
  exit 2
fi
echo "→ $NODE: ${VALUE} MB wired limit on ${RAM_MB} MB of RAM"
echo "   (leaves $(( (RAM_MB - VALUE) / 1024 )) GiB to macOS)"

PLIST_XML=$(cat <<XML
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/sbin/sysctl</string>
        <string>iogpu.wired_limit_mb=${VALUE}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/var/log/${LABEL}.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/${LABEL}.log</string>
</dict>
</plist>
XML
)

echo "   installing $PLIST (sudo on the node — it will ask for the password)"
ssh -t "$NODE" "sudo tee '$PLIST' >/dev/null <<'PLISTEOF'
${PLIST_XML}
PLISTEOF
sudo chown root:wheel '$PLIST' && sudo chmod 644 '$PLIST'
sudo launchctl bootout system '$PLIST' 2>/dev/null || true
sudo launchctl bootstrap system '$PLIST'
echo -n '   now: '; sysctl -n iogpu.wired_limit_mb"

echo "✓ $NODE done — survives reboot"
