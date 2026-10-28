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
#   5. qwen3_coder tool-parser detection without the newline (MiMo-V2.6)
#   6. empty-delta progress chunk while a tool call is withheld (#112)
#   7. anti-loop (scripts/loop_detect.py) + token counts in the progress chunk (#113)
#   8. vendored model packages (scripts/mlx_vlm_models/*: unlimited_ocr from PR #2601,
#      Unlimited-OCR-2) over mlx_vlm/models/, only over the pin they were checked on
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

# 5. qwen3_coder detection without the newline (idempotent, same in-place
#    replace as step 4). mlx_vlm/tools/registry.py picks the tool parser from
#    markers in the chat template; qwen3_coder only matches
#    `<tool_call>\n<function=`. MiMo-V2.6's template renders
#    `<tool_call><function=NAME>` with no newline, so the generic json_tools
#    parser won: non-stream returned the call as text (tool_calls null), stream
#    returned nothing (json_tools withheld the block, then failed to parse it).
#    With the qwen3_coder parser both modes return clean tool_calls (measured
#    2026-10-15 on .30). Still the same marker on Blaizzy/mlx-vlm main (0.7.8).
#    Takes effect at the next mlx_vlm.server start.
echo "[install-mlx-vlm] applying qwen3_coder no-newline marker patch"
PATCH_SCRIPT=$(cat <<'PYEOF'
path = "VENV_PLACEHOLDER/lib/python3.12/site-packages/mlx_vlm/tools/registry.py"
with open(path) as f:
    content = f.read()
old = r'''        (("<tool_call>\\n<function=",), ("<tool_call>\n<function=",)),'''
new = r'''        (("<tool_call>\\n<function=",), ("<tool_call>\n<function=",), ("<tool_call><function=",)),'''
if new in content:
    print("[install-mlx-vlm] qwen3_coder marker patch already applied, skipping")
elif old not in content:
    print("[install-mlx-vlm] WARNING: qwen3_coder marker not found (upstream file changed?) — skipping")
else:
    import shutil
    shutil.copy(path, path + ".bak-qwen3-coder-marker")
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("[install-mlx-vlm] qwen3_coder marker patch applied")
PYEOF
)
PATCH_SCRIPT="${PATCH_SCRIPT//VENV_PLACEHOLDER/$VLM_VENV}"
ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_TARGET" "$VLM_VENV/bin/python -" <<<"$PATCH_SCRIPT"

# 6. Progress chunk while a tool call is withheld (#112; idempotent, same
#    in-place replace as steps 4-5). server/openai.py drops every chunk whose
#    delta is empty, and the tool-call parser empties the delta for the whole
#    call: a 250-line write_file on MiMo-V2.6 streamed nothing for 65.9 s
#    (2026-10-15), past the desktop's 90 s silence cut on longer calls. Each
#    token that advanced while the parser holds the markup may now yield an
#    empty-delta chunk, at most one per 2 s — tokens only, never a timer, so a
#    wedged server still goes silent. Takes effect at the next mlx_vlm.server
#    start.
echo "[install-mlx-vlm] applying withheld-tool-call progress chunk patch"
PATCH_SCRIPT=$(cat <<'PYEOF'
path = "VENV_PLACEHOLDER/lib/python3.12/site-packages/mlx_vlm/server/openai.py"
with open(path) as f:
    content = f.read()
mark = "# OdyssAI-X #112"
a_old = "                        tool_call_state = ToolCallStreamState(tc_start, tc_end)\n"
a_new = a_old + "                        _odx_last_chunk = time.monotonic()  # OdyssAI-X #112\n"
b_old = """                            # Skip empty deltas (e.g. suppressed tool-call tokens)
                            has_payload = (
                                bool(delta_content)
                                or bool(delta_reasoning)
                                or chunk_logprobs is not None
                            )
                            if has_payload:
"""
b_new = b_old.replace("                            if has_payload:\n", """                            # OdyssAI-X #112: a token advanced but the tool-call
                            # parser holds it back: an empty delta keeps the
                            # client informed, at most one per 2 s.
                            if not has_payload and time.monotonic() - _odx_last_chunk >= 2.0:
                                yield "data: " + json.dumps({
                                    "id": request_id, "object": "chat.completion.chunk",
                                    "created": int(time.time()), "model": request.model,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                                }) + "\\n\\n"
                                _odx_last_chunk = time.monotonic()
                            if has_payload:
                                _odx_last_chunk = time.monotonic()
""")
if mark in content:
    print("[install-mlx-vlm] progress chunk patch already applied, skipping")
elif content.count(a_old) != 1 or content.count(b_old) != 1:
    print("[install-mlx-vlm] WARNING: stream loop anchors not found (upstream file changed?) — skipping")
else:
    import shutil
    shutil.copy(path, path + ".bak-progress-chunk")
    content = content.replace(a_old, a_new).replace(b_old, b_new)
    compile(content, path, "exec")
    with open(path, "w") as f:
        f.write(content)
    print("[install-mlx-vlm] progress chunk patch applied")
PYEOF
)
PATCH_SCRIPT="${PATCH_SCRIPT//VENV_PLACEHOLDER/$VLM_VENV}"
ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_TARGET" "$VLM_VENV/bin/python -" <<<"$PATCH_SCRIPT"

# 7. Anti-loop + token counts (#113; idempotent, same in-place replace).
#    mlx_vlm.server has no loop detector: on 2026-10-15 a MiMo-V2.6 request
#    (max_tokens 500000) decoded 95,020 tokens in 49.5 min, all held back by the
#    tool parser, until the client hung up. The runner's detector (one source:
#    scripts/loop_detect.py, copied as mlx_vlm/odyssai_loop_detect.py) now runs
#    on the chat stream and non-stream decode loops: a proven loop ends the turn
#    with finish_reason "stop", a chunk carrying x_mlx_cluster.loop_detected
#    (stream), and a log line with the tail; every 8192 tokens of a long stream
#    the tail is logged too, so a runaway that is not an exact loop leaves a trace. The step-6 progress chunk also
#    carries x_odyssai_progress {prompt_tokens, completion_tokens}. Takes effect
#    at the next mlx_vlm.server start.
echo "[install-mlx-vlm] installing the anti-loop module and patch"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
scp -q "$SCRIPT_DIR/loop_detect.py" "$SSH_TARGET:$VLM_VENV/lib/python3.12/site-packages/mlx_vlm/odyssai_loop_detect.py"
PATCH_SCRIPT=$(cat <<'PYEOF'
path = "VENV_PLACEHOLDER/lib/python3.12/site-packages/mlx_vlm/server/openai.py"
with open(path) as f:
    content = f.read()
mark = "# OdyssAI-X #113"
ids_expr = ('(getattr(token, "token_ids", None) or '
            '([token.token] if getattr(token, "token", None) is not None else []))')
edits = [
    ("from ..generate import generate, stream_generate\n",
     "from ..generate import generate, stream_generate\n"
     "from ..odyssai_loop_detect import LoopGuard as _odx_LoopGuard  # OdyssAI-X #113\n"),
    ("                        tool_call_state = ToolCallStreamState(tc_start, tc_end)\n",
     "                        tool_call_state = ToolCallStreamState(tc_start, tc_end)\n"
     "                        _odx_guard, _odx_loop, _odx_next_log = _odx_LoopGuard(), None, 8192  # OdyssAI-X #113\n"),
    ("                            output_tokens += getattr(token, \"token_count\", 1)\n"
     "                            full_output += token.text\n"
     "                            chunk_rate = metrics.record_chunk(token)\n",
     "                            output_tokens += getattr(token, \"token_count\", 1)\n"
     "                            full_output += token.text\n"
     "                            chunk_rate = metrics.record_chunk(token)\n"
     "                            # OdyssAI-X #113: the runner's anti-loop, same thresholds.\n"
     f"                            _odx_hit = _odx_guard.feed({ids_expr})\n"
     "                            if _odx_hit:\n"
     "                                _odx_loop = (_odx_hit[0], _odx_hit[1], output_tokens)\n"
     "                                logging.warning(\n"
     "                                    \"OdyssAI-X anti-loop: request=%s period=%d repeats=%d \"\n"
     "                                    \"after %d tokens; tail=%r\", request_id, _odx_hit[0],\n"
     "                                    _odx_hit[1], output_tokens, full_output[-300:])\n"
     "                                finish_reason = \"stop\"\n"
     "                                break\n"
     "                            if output_tokens >= _odx_next_log:  # a long turn leaves a trace\n"
     "                                _odx_next_log += 8192\n"
     "                                logging.info(\"OdyssAI-X long generation: request=%s tokens=%d tail=%r\",\n"
     "                                             request_id, output_tokens, full_output[-300:])\n"),
    ("                        # Parse tool calls from full output and emit final chunk\n",
     "                        if _odx_loop:  # OdyssAI-X #113: say why the turn stopped\n"
     "                            yield \"data: \" + json.dumps({\n"
     "                                \"id\": request_id, \"object\": \"chat.completion.chunk\",\n"
     "                                \"created\": int(time.time()), \"model\": request.model,\n"
     "                                \"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": None}],\n"
     "                                \"x_mlx_cluster\": {\"loop_detected\": True, \"loop_period\": _odx_loop[0],\n"
     "                                                  \"loop_repeats\": _odx_loop[1], \"ntoks\": _odx_loop[2]},\n"
     "                            }) + \"\\n\\n\"\n"
     "                        # Parse tool calls from full output and emit final chunk\n"),
    ("                                    \"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": None}],\n"
     "                                }) + \"\\n\\n\"\n",
     "                                    \"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": None}],\n"
     "                                    \"x_odyssai_progress\": {  # OdyssAI-X #113\n"
     "                                        \"prompt_tokens\": getattr(ctx, \"prompt_tokens\", None),\n"
     "                                        \"completion_tokens\": output_tokens},\n"
     "                                }) + \"\\n\\n\"\n"),
    ("                        for token in token_iter:\n"
     "                            text += token.text\n"
     "                            gt += getattr(token, \"token_count\", 1)\n",
     "                        _odx_g = _odx_LoopGuard()  # OdyssAI-X #113\n"
     "                        for token in token_iter:\n"
     "                            text += token.text\n"
     "                            gt += getattr(token, \"token_count\", 1)\n"
     f"                            _odx_hit = _odx_g.feed({ids_expr})\n"
     "                            if _odx_hit:\n"
     "                                logging.warning(\n"
     "                                    \"OdyssAI-X anti-loop (non-stream): period=%d repeats=%d \"\n"
     "                                    \"after %d tokens; tail=%r\", _odx_hit[0], _odx_hit[1], gt,\n"
     "                                    text[-300:])\n"
     "                                fr = \"stop\"\n"
     "                                break\n"),
]
if mark in content:
    print("[install-mlx-vlm] anti-loop patch already applied, skipping")
elif any(content.count(old) != 1 for old, _ in edits):
    missing = [old.strip().splitlines()[0][:60] for old, _ in edits if content.count(old) != 1]
    print(f"[install-mlx-vlm] WARNING: anti-loop anchors not found {missing} (upstream file changed, or step 6 not applied?) — skipping")
else:
    import shutil
    shutil.copy(path, path + ".bak-anti-loop")
    for old, new in edits:
        content = content.replace(old, new)
    compile(content, path, "exec")
    with open(path, "w") as f:
        f.write(content)
    print("[install-mlx-vlm] anti-loop patch applied")
PYEOF
)
PATCH_SCRIPT="${PATCH_SCRIPT//VENV_PLACEHOLDER/$VLM_VENV}"
ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_TARGET" "$VLM_VENV/bin/python -" <<<"$PATCH_SCRIPT"

# 8. Vendored model packages. Each UPSTREAM.md names the mlx-vlm pin it was
#    checked against (`overlays-pin:`); on any other pin it is not installed,
#    since a newer mlx-vlm may ship the package itself.
for pkg in "$SCRIPT_DIR"/mlx_vlm_models/*/; do
  name="$(basename "$pkg")"
  pin="$(awk '/^overlays-pin:/{print $2}' "$pkg/UPSTREAM.md")"
  if [ "$pin" != "$MLX_VLM_REF" ]; then
    echo "[install-mlx-vlm] $name vendored for $pin, pin is $MLX_VLM_REF: not installed"
    continue
  fi
  dest="$VLM_VENV/lib/python3.12/site-packages/mlx_vlm/models/$name"
  echo "[install-mlx-vlm] installing vendored model package $name"
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_TARGET" "rm -rf '$dest' && mkdir -p '$dest'"
  scp -q "$pkg"*.py "$pkg"UPSTREAM.md "$SSH_TARGET:$dest/"
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_TARGET" \
    "cd /tmp && $VLM_VENV/bin/python -c 'import mlx_vlm.models.$name as m; print(\"[install-mlx-vlm] import ok:\", m.Model.__module__)'"
done

echo "[install-mlx-vlm] done on $SSH_TARGET"
