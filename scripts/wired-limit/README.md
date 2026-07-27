# Persistent `iogpu.wired_limit_mb`

`sysctl iogpu.wired_limit_mb` does **not** survive a reboot. That matters more
than it looks: the wired-leak self-healing shipped in 1.18.0 reboots nodes on
its own, so a cluster can silently drop back to the macOS default between two
loads. A model sized for a tuned limit then pages instead of failing loudly.

These files make the setting stick. They are **not installed by any automation**
— installing a LaunchDaemon needs root, so it is a deliberate act.

## Install (per node, asks for the node's password)

```bash
scripts/wired-limit/install.sh admin@192.168.86.29 491520   # ultra-512
scripts/wired-limit/install.sh admin@192.168.86.30 250880   # ultra-256a
scripts/wired-limit/install.sh admin@192.168.86.31 250880   # ultra-256b
scripts/wired-limit/install.sh admin@192.168.86.32 250880   # ultra-256c
scripts/wired-limit/install.sh admin@192.168.86.33 250880   # ultra-256d
```

The value is in **MB**. 491520 = 480 GiB on a 512 GiB machine; 250880 = 245 GiB
on a 256 GiB machine — the sizing behind the Kimi K3 load (docs/adr/0004).

## Verify

```bash
for ip in 29 30 31 32 33; do
  ssh admin@192.168.86.$ip 'echo -n "$(hostname): "; sysctl -n iogpu.wired_limit_mb'
done
```

## Remove

```bash
ssh -t admin@<node> 'sudo launchctl bootout system /Library/LaunchDaemons/eu.odyssai.wiredlimit.plist && sudo rm /Library/LaunchDaemons/eu.odyssai.wiredlimit.plist'
```

A reboot then returns the node to the macOS default.
