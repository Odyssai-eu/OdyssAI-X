#!/usr/bin/env python3
"""#94 — the chat-template fallback ladder shared by the legacy and batched loops.

A fake tokenizer whose template validates reasoning_effort (as Hy3, GLM-5.3 and
Qwen3.8-Flash-Next templates do) and optionally refuses tools.

    python3 scripts/test_template_render.py
"""
import os
import sys

sys.argv = ["runner.py"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


class TemplateError(Exception):
    pass


class StrictTok:
    """Template that raises on an effort outside `allowed` and, when
    tools_ok=False, on any tools kwarg. allowed=None = accepts anything
    (GLM-5.3 style: silent)."""

    def __init__(self, allowed=None, tools_ok=True, always_fail=False):
        self.allowed, self.tools_ok, self.always_fail = allowed, tools_ok, always_fail
        self.calls = 0

    def apply_chat_template(self, messages, **kw):
        self.calls += 1
        if self.always_fail:
            raise TemplateError("Conversation roles must alternate user/assistant")
        if kw.get("tools") and not self.tools_ok:
            raise TemplateError("tools not supported")
        eff = kw.get("reasoning_effort")
        if eff is not None and self.allowed is not None and eff not in self.allowed:
            raise TemplateError(f"Unexpected reasoning effort {eff}")
        return f"<prompt eff={eff} tools={bool(kw.get('tools'))}>"


MSG = [{"role": "user", "content": "hi"}]

# Hy3 (no_think/low/high) receiving Companion's "medium"
kw = {"reasoning_effort": "medium"}
out, dropped = runner._render_chat_template(StrictTok({"no_think", "low", "high"}), MSG, kw)
check("Hy3 medium → rendered without effort", (out, dropped), ("<prompt eff=None tools=False>", ["reasoning_effort"]))
check("rejected kwarg removed from chat_kwargs (reused for tokenize)", "reasoning_effort" in kw, False)

# Qwen3.8-Flash-Next (xhigh/medium/low) receiving the bench's "high"
out, dropped = runner._render_chat_template(StrictTok({"xhigh", "medium", "low"}), MSG, {"reasoning_effort": "high"})
check("Qwen3.8 high → rendered, effort dropped", dropped, ["reasoning_effort"])

# accepted effort passes through untouched
out, dropped = runner._render_chat_template(StrictTok({"xhigh", "medium", "low"}), MSG, {"reasoning_effort": "xhigh"})
check("accepted effort kept", (out, dropped), ("<prompt eff=xhigh tools=False>", []))

# GLM-5.3 style: silent template, nothing to drop
out, dropped = runner._render_chat_template(StrictTok(None), MSG, {"reasoning_effort": "medium"})
check("silent template: nothing dropped", dropped, [])

# tools refused AND effort refused → both dropped, in order
kw = {"reasoning_effort": "high", "tools": [{"type": "function"}]}
out, dropped = runner._render_chat_template(StrictTok({"low"}, tools_ok=False), MSG, kw)
check("tools then effort dropped", (dropped, out), (["tools", "reasoning_effort"], "<prompt eff=None tools=False>"))

# a template that fails regardless → raises (the batched loop turns it into done+error)
try:
    runner._render_chat_template(StrictTok(always_fail=True), MSG, {"reasoning_effort": "high"})
    check("unrecoverable template raises", "no error", "TemplateError")
except TemplateError as e:
    check("unrecoverable template raises the template's error", "alternate" in str(e), True)

# the batched loop's rejection path emits done+error and continues (source check:
# the per-request catch sits before any slot/session/BatchGenerator mutation)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")).read()
b = src[src.index("def _run_batched_main"):]
i_render = b.index("_render_chat_template(tokenizer, messages_in, chat_kwargs)")
check("batched: render caught before session lookup", i_render < b.index("_session_lookup(session_id"), True)
check("batched: render caught before bg.insert", i_render < b.index("bg.insert("), True)
check("batched: rejection emits done with error and continues",
      '"error": f"chat_template: {e}"' in b[i_render:i_render + 900] and "continue" in b[i_render:i_render + 900], True)

# API: a done carrying `error` is raised, never yielded as a normal end
api_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.py")).read()
g = api_src[api_src.index("class RunnerRequestError"):]
check("api: done+error raises RunnerRequestError", 'raise RunnerRequestError(str(ev.get("error")))' in api_src, True)
check("api: non-stream maps it to 422/500", "except RunnerRequestError as e:" in api_src, True)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
