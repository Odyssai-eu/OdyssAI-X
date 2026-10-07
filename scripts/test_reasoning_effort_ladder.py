#!/usr/bin/env python3
"""A reasoning_effort the chat template rejects must degrade — never kill the
rank-0 batch (#94, 3 occurrences: Hy3 2026-07-08 "medium", Qwen3.8-Flash-Next
2026-09-26 "high", GLM 2026-08-27 silently mapped).

The runner imports mlx (absent from this venv), so the ladder is FROZEN here
verbatim from scripts/runner.py and a source guard asserts the repo copy still
matches — editing one without the other fails this test. The API-side alias
map (_map_reasoning_effort) is tested live. No node is reached.

    .venv/bin/python scripts/test_reasoning_effort_ladder.py
"""
import json
import os
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({"argo": {"name": "A", "kind": "mlx-distributed", "backend": "ring",
                    "models_dir": "/m",
                    "nodes": [{"host": "n0", "ssh": "admin@198.51.100.1", "master": True}]}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


# ---------------------------------------------------------------- the ladder
def log(msg):
    print(msg, file=sys.stderr)


# The function below is a frozen copy of scripts/runner.py; the guard at the
# bottom fails this test if the repo version drifts from it.
# FROZEN COPY
def _render_chat_template(tokenizer, messages, chat_kwargs: dict):
    """apply_chat_template with a retry ladder on OPTIONAL kwargs a template may
    reject — a bad optional must degrade, never kill the rank. Shared by the
    legacy and the batched loops (#94: the batched loop only had the tools step,
    so a template refusing reasoning_effort killed the whole batch — Hy3
    2026-07-08, Qwen3.8-Flash-Next 2026-09-26). Drops rejected kwargs from
    `chat_kwargs` IN PLACE (the caller reuses it for the tokenize pass) and
    returns (templated, dropped_kwarg_names). Raises when even the bare render
    fails."""
    dropped: list = []
    try:
        return tokenizer.apply_chat_template(messages, **chat_kwargs), dropped
    except Exception as e:
        err = e
    # 1. Templates without tool support reject `tools=`.
    if chat_kwargs.get("tools"):
        log(f"chat template rejected tools ({err}); retrying without")
        chat_kwargs.pop("tools", None)
        dropped.append("tools")
        try:
            return tokenizer.apply_chat_template(messages, **chat_kwargs), dropped
        except Exception as e2:
            err = e2
    # 2. Templates that validate reasoning_effort: drop the dial — the template
    #    default beats a dead pool; the caller reports it (effort_dropped).
    if chat_kwargs.get("reasoning_effort"):
        log(f"chat template rejected reasoning_effort="
            f"{chat_kwargs['reasoning_effort']!r} ({err}); retrying without")
        chat_kwargs.pop("reasoning_effort", None)
        dropped.append("reasoning_effort")
        return tokenizer.apply_chat_template(messages, **chat_kwargs), dropped
    raise err
# END FROZEN COPY


class FakeTokenizer:
    """apply_chat_template that rejects kwargs outside its vocabulary — the
    jinja2.TemplateError behavior of the three historical models."""
    def __init__(self, allowed_effort=None, allows_tools=True, bare_fails=False):
        self.allowed = allowed_effort
        self.allows_tools = allows_tools
        self.bare_fails = bare_fails

    def apply_chat_template(self, messages, **kwargs):
        if "tools" in kwargs and not self.allows_tools:
            raise ValueError("tools not supported")
        if "reasoning_effort" in kwargs and (
                self.allowed is None or kwargs["reasoning_effort"] not in self.allowed):
            raise ValueError(f"bad reasoning_effort: {kwargs['reasoning_effort']}")
        if not messages and self.bare_fails:
            raise ValueError("bare render failed")
        return "OK"


T_HY3 = FakeTokenizer(allowed_effort={"no_think", "low", "high"}, allows_tools=False)
T_QWEN38 = FakeTokenizer(allowed_effort={"xhigh", "medium", "low"})
T_BARE = FakeTokenizer(bare_fails=True)   # refuses every kwarg; bare fails on empty msgs

# Case 1 — Hy3 occurrence: "medium" (client default) + tools → BOTH dropped,
# request SERVED (finding GLM-2/MiMo of the panel: distinct flags, no error).
kw = {"reasoning_effort": "medium", "tools": [{"name": "f"}]}
check("Hy3 'medium'+tools: servie, les DEUX droppés (flags distincts)",
      _render_chat_template(T_HY3, [{"role": "u"}], kw), ("OK", ["tools", "reasoning_effort"]))
check("… mutation in place: le dict ne porte plus les kwargs rejetés",
      sorted(kw.keys()), [])
# Case 2-4 — vocabularies: in-vocab values pass untouched (remap happens
# BEFORE the ladder, api-side — finding GLM-1).
check("Qwen3.8 'xhigh' (post-remap): acceptée, rien droppé",
      _render_chat_template(T_QWEN38, [{"role": "u"}], {"reasoning_effort": "xhigh"}), ("OK", []))
check("Qwen3.8 'medium': dans le vocabulaire, acceptée",
      _render_chat_template(T_QWEN38, [{"role": "u"}], {"reasoning_effort": "medium"}), ("OK", []))
check("Hy3 'high': dans le vocabulaire, acceptée",
      _render_chat_template(T_HY3, [{"role": "u"}], {"reasoning_effort": "high"}), ("OK", []))
# Case 5 — a template refusing every kwarg still serves on real messages.
check("template nu-hostile: effort droppé, requête servie",
      _render_chat_template(T_BARE, [{"role": "u"}], {"reasoning_effort": "high"}), ("OK", ["reasoning_effort"]))
# Case 6 — bare render failure RAISES: the caller rejects THAT request alone.
try:
    _render_chat_template(T_BARE, [], {"reasoning_effort": "high"})
    check("render nu définitif: l'exception se propage (rejet par requête)", "no-raise", "raise")
except ValueError:
    check("render nu définitif: l'exception se propage (rejet par requête)", "raise", "raise")
# Case 7-8 — determinism + idempotence of the mutation.
kw2 = {"reasoning_effort": "medium", "tools": [{"name": "f"}]}
r1 = _render_chat_template(T_HY3, [{"role": "u"}], kw2)
kw2.update({"reasoning_effort": "medium", "tools": [{"name": "f"}]})
check("déterminisme: même tuple au rejeu", r1,
      _render_chat_template(T_HY3, [{"role": "u"}], kw2))

# ------------------------------------------------- the API-side alias map
check("map: qwen3.8-flash-next high → xhigh (le bench du 26/09)",
      api._map_reasoning_effort("qwen3.8-flash-next", "high"), "xhigh")
check("map: qwen3.8-flash-next minimal → low",
      api._map_reasoning_effort("qwen3.8-flash-next", "minimal"), "low")
check("map: valeur inconnue d'un modèle non mappé passe telle quelle",
      api._map_reasoning_effort("some-unknown-model", "high"), "high")

# ---------------------------------------------------------- source guards
here = os.path.dirname(os.path.abspath(__file__))
runner_src = open(os.path.join(here, "runner.py")).read()
ladder_src = open(os.path.join(__file__)).read()
frozen = ladder_src.split("# FROZEN COPY", 1)[1].split("# END FROZEN COPY", 1)[0]
check("garde anti-dérive: l'échelle du repo est TOUJOURS la copie figée",
      frozen.strip() in runner_src, True)
check("la boucle batchée enveloppe le render (rejet par requête, batch vivant)",
      "chat template render failed" in runner_src
      and '"finish_reason": "error"' in runner_src, True)

print("all OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
