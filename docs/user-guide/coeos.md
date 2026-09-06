---
title: CoeOS — the "best at everything" virtual model
description: A benchmark-driven model router, exposed as a single OpenAI-compatible model.
---

# CoeOS

> **In one sentence:** CoeOS is a single **virtual model** that sends every
> request to the model our benchmarks proved **best at that specific skill** —
> from Python code to GDPR to long-form writing. Your agents call one model,
> `CoeOS`, and get the best of the whole fleet, automatically.

---

## How it works

CoeOS doesn't run inference itself. It **classifies** the request, then
**relays** it to the right model:

```
Agent request  →  CoeOS
                    │  1. Which skill? (Python? GDPR? creative?…)
                    │     → a small "decider" model classifies it into 1 axis
                    │  2. Which model is best at that axis?
                    │     → the TMB Settings table says so (from our benchmarks)
                    ▼
              the winning model (local or cloud)  →  response
```

Three concepts:

| Term | Definition |
|---|---|
| **Skill axis** | A *benchmarked* category: `python`, `react`, `swift`, `legal_rgpd`, `creative`, `debug`, `plan_spec`… 15 axes in total. |
| **Decider** | A small, fast model (e.g. NEX-N2-mini) whose only job is to **classify** each request into one axis. |
| **TMB Settings** | The file that defines, per axis, **which model to serve** — derived from our TMB scoreboards. This is the "recipe". |

Classification follows this order:

1. **Explicit header** — if the agent knows its skill, it states it
   (`x-coeos-axis: coding`) and CoeOS uses it directly.
2. **Decider** — otherwise the decider model classifies the request into an axis.
3. **Default axis** — if the decider is unavailable or unsure, CoeOS routes to
   the default axis (the generalist).

Then: axis → bound model. If that model isn't warm, CoeOS automatically falls
back to another available model (and logs it) instead of failing.

---

## Advantages

- **The best of the whole fleet, effortlessly.** One endpoint; each skill goes
  to its champion. No per-agent configuration to maintain.
- **Evidence-based, not marketing.** The choices come from *our* TMB benchmarks
  (real per-task scores), not a generic leaderboard.
- **Sovereign or "best of all", your choice.** A **local** profile (100% on
  your machines) or a **cloud** profile (we allow the best online models where
  they dominate).
- **No code to change to evolve.** The taxonomy and the model choices live in a
  file. A new model wins an axis? Edit the file.
- **Transparent.** The response reports the model actually used
  (`CoeOS · python — MiniMax-M3`).

---

## Use cases

CoeOS is built for **agents and orchestrators** (Omnigent, Cline, Aider,
Hermes…), not for chat (the chat client keeps its own router).

- **A multi-step agent** that plans, codes, debugs, writes: each step goes to
  the model that scores best on it, without the agent having to know.
- **A plan → code pipeline**: the planning phase goes to the best planner, the
  coding phase to the best coder.
- **A heterogeneous fleet** (several local + cloud models) you want to exploit
  optimally behind a single address.

---

## Configuring CoeOS

In the **OdyssAI-X dashboard → Settings → CoeOS**.

### 1. Import a TMB Settings file

Easiest path: **Import TMB Settings…** and pick a shipped file:

- **Sovereign (local)** — everything on your machines.
- **Best of all (cloud)** — allows cloud models where they win.

The import fills everything at once: the 15 axes, the decider, the default axis.

### 2. Adjust (optional)

Everything stays editable after import:

- **Axis grid (3 × 5)** — one dropdown per skill. Pick the model from every
  published one (local + cloud). The same model can cover several axes. An axis
  left empty falls back to another.
- **Decider** — the model that classifies. Small and fast. *It must stay loaded*
  (see the note).
- **Default axis** — where to route when in doubt (typically the generalist).

Every change is **saved immediately**. **Export** lets you download your profile
as a file.

### 3. Enable

The **Enabled toggle**. Once on, `CoeOS` appears in the model list
(`/v1/models`) and in clients as an endpoint, alongside Argo or Telemak.

> **⚠️ The decider must stay warm.** Per-skill routing depends on the decider
> model: if it's **not loaded**, *all* requests fall to the default axis (no
> fine-grained routing). Reserve a slot for it — it's small.

> **Cold-boot autoload** (optional): if an axis's model isn't warm, load the
> default axis's model anyway instead of returning an error. Off = strict.

---

## Using CoeOS

CoeOS speaks the **OpenAI API**. Point any agent at it, model = `CoeOS`.

```bash
export OPENAI_BASE_URL="http://<your-odyssai-x>:8000/v1"
export OPENAI_API_KEY="dummy"          # no key required on a LAN

curl "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"CoeOS","messages":[{"role":"user","content":"Write a Python script that…"}]}'
```

CoeOS classifies ("Python") and serves the best Python coder in your profile.

### Forcing an axis (optional)

If your agent **already knows** the step's skill, it can bypass the decider with
a header — faster and safer:

```bash
  -H "x-coeos-axis: legal_rgpd"
```

Available axes: `creative`, `legal_rgpd`, `legal_complex`, `reasoning`, `calc`,
`python`, `code_general`, `debug`, `react`, `swift`, `refactoring`,
`plan_decompo`, `plan_spec`, `plan_judgment`, `fast_tools`.

### Seeing which model answered

The response exposes the model actually used. Compatible clients (e.g.
Companion) display it: **`CoeOS · python — MiniMax-M3`**.

---

## The TMB Settings — the "recipe"

This is the heart of CoeOS, and a deliverable in its own right: a file that
states, per skill, **which model is best** according to our benchmarks. We
**update it monthly** as new benchmarks and new models land. Format:

```json
{
  "name": "TMB Settings — Sovereign (local) · v0.1",
  "regime": "local",
  "decider_model": "telomnis:nex-n2-mini-6bit",
  "default_axis": "code_general",
  "axes": [
    { "key": "python", "label": "Python / scripts", "model": "default",
      "bench": "T03 48.5 / T04 48.5 (M3)", "verified": true },
    { "key": "swift",  "label": "Swift", "model": "",
      "bench": "to benchmark", "verified": false }
  ]
}
```

- empty `model` = **axis not yet filled** (waiting for the best benchmarked model).
- `bench` = the **evidence** behind the choice (the TMB test and the score).
- `verified` = `false` when the binding is estimated rather than measured by us.

Adding a skill = adding a line. No code to change.

---

## In short

| | |
|---|---|
| **What** | A virtual model `CoeOS` that routes per skill. |
| **For whom** | Agents / orchestrators (not chat). |
| **How** | Decider classifies → TMB Settings names the best model → relay. |
| **Config** | Settings → CoeOS: import a TMB Settings, adjust, enable. |
| **Usage** | OpenAI endpoint, `model: "CoeOS"`, optional `x-coeos-axis` header. |
| **The edge** | The best of the whole fleet, grounded in real benchmarks. |
