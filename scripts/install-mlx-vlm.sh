#!/usr/bin/env bash
#
# install-mlx-vlm.sh — provision the single-node VLM serving venv on a node.
#
# Mirrors the manual steps that stood up mlx_vlm.server on .29:
#   1. create a dedicated python3.12 venv at ~/.venvs/mlx-vlm on the node
#      (NOT the python3.11 cluster venv ~/mlx-cluster/.venv — never touched)
#   2. pip install mlx-vlm pinned to the merged VL commit + torch/torchvision
#   3. apply scripts/patches/mlx_vlm_thinking_mode_disabled.patch (see below)
#   4. smoke-import mlx_vlm + the minimax_m3_vl model module
#
# Idempotent: re-running skips venv creation if it already imports cleanly,
# skips pip entirely when mlx-vlm is already at MLX_VLM_REF (direct_url.json),
# and the patch step skips if already applied. Refuses to reinstall under a
# running mlx_vlm.server unless MLX_VLM_FORCE=1.
#
# scripts/patches/mlx_vlm_thinking_mode_disabled.patch (2026-07-08):
# upstream mlx_vlm/prompt_utils.py only maps enable_thinking=True to the
# chat-template kwarg thinking_mode="enabled" — the False case is never
# mapped to thinking_mode="disabled", so it's left undefined and MiniMax-M3's
# template falls through to its adaptive-thinking default regardless of what
# the caller requests. Confirmed still present on Blaizzy/mlx-vlm main as of
# 2026-07-08 (no matching issue/PR upstream) — root cause of the M3 overthink
# bug (~1h30 / 59k think tokens on tasks that should answer immediately).
# Not upstreamed yet (Sophie: verify no duplicate issue first); tracked here
# so every node install carries the fix instead of a hand-patched venv.
#
# This is what the node installer calls so mlx-vlm ships with the cluster
# (provision-at-node-setup, rather than a hand-rolled venv per operator).
#
# Usage:
#   scripts/install-mlx-vlm.sh <ssh-target>
#   scripts/install-mlx-vlm.sh admin@192.168.86.30
#
# Env overrides:
#   VLM_VENV      target venv path      (default <remote $HOME>/.venvs/mlx-vlm)
#   MLX_VLM_REF   git ref of mlx-vlm    (default 8702083 = main 2026-09-24, 0.7.3: MiMo-V2.6 text #2327,
#                 native MXFP4 load #2337, image/video/audio #2338, batching #2339,
#                 MiMo audio across server threads #2352;
#                 0.6.3 + mlx 0.32 crashes Qwen3.5 in server mode, mlx-vlm #1614)
#   MLX_VLM_SPEC  full pip spec, overrides the archive URL built from MLX_VLM_REF
#                 (e.g. a wheel path on the node when its GitHub link is slow)
#   PY312         python3.12 executable (default python3.12)
set -euo pipefail

SSH_TARGET="${1:-}"
if [[ -z "$SSH_TARGET" ]]; then
  echo "usage: $0 <ssh-target>   e.g. $0 admin@192.168.86.30" >&2
  exit 2
fi

# Resolve the node's HOME once (the operator user is whatever SSH_TARGET says —
# not necessarily `admin`). The remote script below is an UNQUOTED heredoc, so
# these expand locally into a plain string; a literal $HOME would not survive
# the Python patch step that gets this path substituted in.
REMOTE_HOME="$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_TARGET" 'printf %s "$HOME"')"
REMOTE_USER="$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_TARGET" 'id -un')"
VLM_VENV="${VLM_VENV:-$REMOTE_HOME/.venvs/mlx-vlm}"
MLX_VLM_REF="${MLX_VLM_REF:-87020830d4dde238f111ade6d592f204a89cd527}"
PY312="${PY312:-python3.12}"
# A GitHub archive of the pinned commit, not `git+https`: pip then needs no git,
# and a Mac without Xcode's command-line tools has none (fresh ultra-96c,
# 2026-09-30: `git version` popped the Xcode install dialog and failed).
MLX_VLM_SPEC="${MLX_VLM_SPEC:-mlx-vlm @ https://github.com/Blaizzy/mlx-vlm/archive/${MLX_VLM_REF}.tar.gz}"

echo "[install-mlx-vlm] target=$SSH_TARGET venv=$VLM_VENV ref=$MLX_VLM_REF"

# The remote script runs under a clean env. We do the whole thing in one SSH
# round-trip so the venv activation persists across the pip + smoke steps.
# Single-quoted heredoc: expand our vars locally into a plain script string
# first, then send it (no remote var-expansion surprises).
REMOTE_SCRIPT=$(cat <<REMOTE
set -euo pipefail
export HOME=${REMOTE_HOME} USER=${REMOTE_USER} TMPDIR=/tmp
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin

VENV="${VLM_VENV}"
SPEC="${MLX_VLM_SPEC}"
REF="${MLX_VLM_REF}"
FORCE="${MLX_VLM_FORCE:-0}"

# Locate python3.12 (Homebrew installs it as python3.12; fall back to a probe).
PY="\$(command -v ${PY312} || true)"
if [[ -z "\$PY" ]]; then
  for cand in /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
    [[ -x "\$cand" ]] && PY="\$cand" && break
  done
fi
# A node set up by install.sh has uv but no Homebrew or python.org Python
# (fresh ultra-96c, 2026-09-30): let uv provide 3.12, as install.sh does for 3.11.
if [[ -z "\$PY" ]]; then
  UV="\$(command -v uv || true)"
  [[ -z "\$UV" && -x "\$HOME/.local/bin/uv" ]] && UV="\$HOME/.local/bin/uv"
  if [[ -n "\$UV" ]]; then
    echo "[install-mlx-vlm] no python3.12 on this node, installing it with uv"
    "\$UV" python install 3.12
    PY="\$("\$UV" python find 3.12 || true)"
  fi
fi
if [[ -z "\$PY" ]]; then
  echo "[install-mlx-vlm] ERROR: python3.12 not found on this node (and no uv to install it: run install.sh first)" >&2
  exit 1
fi
echo "[install-mlx-vlm] python3.12 = \$PY (\$(\$PY --version 2>&1))"

# 1. venv (create only if missing)
if [[ ! -x "\$VENV/bin/python" ]]; then
  echo "[install-mlx-vlm] creating venv at \$VENV"
  mkdir -p "\$(dirname "\$VENV")"
  "\$PY" -m venv "\$VENV"
else
  echo "[install-mlx-vlm] venv already exists at \$VENV"
fi

# Already at the pinned commit (pip records it in direct_url.json)? Then leave
# the venv alone: the forced reinstall below used to run on EVERY call, i.e.
# under a live mlx_vlm.server on the replica nodes whenever a node was
# re-bootstrapped. A reinstall while a server runs needs MLX_VLM_FORCE=1.
# A git install records the commit in vcs_info; an archive install (the
# default now) records the archive URL, which carries the commit.
HAVE="\$("\$VENV/bin/python" -c 'import importlib.metadata as m, json
try:
    d = json.loads(m.distribution("mlx-vlm").read_text("direct_url.json") or "{}")
    print(d.get("vcs_info", {}).get("commit_id") or d.get("url", ""))
except Exception:
    print("")' 2>/dev/null || true)"
if [[ -n "\$HAVE" && ( "\$HAVE" == "\$REF"* || "\$HAVE" == *"/\$REF.tar.gz" ) && "\$FORCE" != 1 ]]; then
  echo "[install-mlx-vlm] mlx-vlm already at \${REF:0:7}, nothing to install"
else
  if pgrep -f "[m]lx_vlm.server" >/dev/null 2>&1 && [[ "\$FORCE" != 1 ]]; then
    echo "[install-mlx-vlm] REFUSED: mlx_vlm.server is running on this node and mlx-vlm is at \${HAVE:0:7}, not \${REF:0:7}." >&2
    echo "[install-mlx-vlm] Unload its pool first, or rerun with MLX_VLM_FORCE=1." >&2
    exit 3
  fi
  "\$VENV/bin/python" -m pip install --upgrade pip >/dev/null

  # 2. install mlx-vlm (pinned) + torch/torchvision.
  echo "[install-mlx-vlm] pip install \$SPEC torch torchvision"
  "\$VENV/bin/python" -m pip install "\$SPEC" torch torchvision
  # Several refs share the version string "0.7.2" (the v0.7.2 tag and the
  # b5952d7 main commit): pip then sees the requirement as satisfied and keeps
  # the OLD code (mimo_v2 was missing on .30-.33 on 2026-09-24). Reinstall
  # mlx-vlm itself from the exact ref, deps untouched. The patch step below
  # runs after, so it is re-applied on the fresh files.
  "\$VENV/bin/python" -m pip install --force-reinstall --no-deps "\$SPEC"
fi

# 3. smoke import — fails loudly if the VL model module isn't present.
echo "[install-mlx-vlm] smoke import"
"\$VENV/bin/python" -c "import mlx_vlm; from mlx_vlm.models import minimax_m3_vl; from mlx_vlm.models.mimo_v2 import audio; print('[install-mlx-vlm] OK', mlx_vlm.__version__ if hasattr(mlx_vlm,'__version__') else '(no __version__)')"
REMOTE
)

ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_TARGET" "bash -s" <<<"$REMOTE_SCRIPT"

# 4. thinking_mode=disabled patch (idempotent, applied locally over SSH since
#    the fix is a small in-place string replace, not a line-numbered diff that
#    could drift against the pinned ref).
echo "[install-mlx-vlm] applying thinking_mode-disabled patch"
PATCH_SCRIPT=$(cat <<'PYEOF'
path = "VENV_PLACEHOLDER/lib/python3.12/site-packages/mlx_vlm/prompt_utils.py"
with open(path) as f:
    content = f.read()
old = '''        if (
            "thinking_mode" not in template_kwargs
            and template_kwargs.get("enable_thinking") is True
            and _template_references_kw(template_processor, "thinking_mode")
        ):
            template_kwargs["thinking_mode"] = "enabled"'''
new = '''        if (
            "thinking_mode" not in template_kwargs
            and template_kwargs.get("enable_thinking") is True
            and _template_references_kw(template_processor, "thinking_mode")
        ):
            template_kwargs["thinking_mode"] = "enabled"
        elif (
            "thinking_mode" not in template_kwargs
            and template_kwargs.get("enable_thinking") is False
            and _template_references_kw(template_processor, "thinking_mode")
        ):
            template_kwargs["thinking_mode"] = "disabled"'''
if new in content:
    print("[install-mlx-vlm] patch already applied, skipping")
elif old not in content:
    print("[install-mlx-vlm] WARNING: patch target not found (upstream file changed?) — skipping")
else:
    import shutil
    shutil.copy(path, path + ".bak-thinking-mode-patch")
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("[install-mlx-vlm] patch applied")
PYEOF
)
PATCH_SCRIPT="${PATCH_SCRIPT//VENV_PLACEHOLDER/$VLM_VENV}"
ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_TARGET" "$VLM_VENV/bin/python -" <<<"$PATCH_SCRIPT"

echo "[install-mlx-vlm] done on $SSH_TARGET"
