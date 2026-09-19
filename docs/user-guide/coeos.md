---
title: CoeOS on this engine
description: CoeOS is a separate gateway that routes each request to the best model per skill. This page is the engine side — what OdyssAI-X exposes to it and how to wire the two.
---

# CoeOS on this engine

> **In one sentence:** [CoeOS](https://github.com/Odyssai-eu/coeos) is the
> **smart client** of OdyssAI — a complete AI operating system with the smart
> router built in. Your tools call one model, `CoeOS`; it classifies the request
> on a **competence axis** (Python, debugging, GDPR, writing, planning…) and
> sends it to the model proven best there — served by **this engine**, or by a
> cloud provider with your own keys. The inference itself happens here.

CoeOS used to live inside the OdyssAI-X dashboard. It is now two public pieces:
**Nemo**, the client ([Odyssai-eu/coeos](https://github.com/Odyssai-eu/coeos), a signed and notarized macOS app), and the router
that runs as its own container — the **box**, [Odyssai-eu/coeos-box](https://github.com/Odyssai-eu/coeos-box), port 4600, console *Theseus*.
Everything about configuring axes, keys, users and quotas is in the box guide:
[`doc/USER-GUIDE.md`](https://github.com/Odyssai-eu/coeos-box/blob/main/doc/USER-GUIDE.md).
This page covers only the engine side.

## What this engine exposes to CoeOS

| Surface | Used for |
|---|---|
| `POST /v1/chat/completions`, `POST /v1/messages` | the relay itself (OpenAI and Anthropic wire formats; CoeOS forwards the client's format) |
| `GET /v1/models` with `x_odyssai` blocks | which models are **ready**, their capabilities (tools, vision, context) — CoeOS only routes an axis to a model that is loaded |
| `GET /.well-known/inference-engine.json` | the capability contract CoeOS mirrors to its own clients |
| `enable_thinking` | **the only** thinking flag this engine reads. CoeOS translates `thinking`/`reasoning` from cloud-style clients into it; sending `thinking:false` directly is ignored and a reasoning model spends its whole budget thinking |
| `reasoning_content` in deltas | reasoning is split from the answer per model; CoeOS passes both through |

No key is required on a trusted LAN: CoeOS's built-in local provider (`odyssai`)
is keyless by design.

## Wire CoeOS to this engine

1. Make sure the models you want CoeOS to route to are **loaded** here (a
   replica pool is the natural fit — see [Multi-user serving](multi-user-serving.md)).
2. Give the CoeOS box (`coeos-box`, the router container of the product) the engine's address:

```bash
curl -X PUT http://<coeos-host>:4600/admin/platform/providers/odyssai \
  -H 'content-type: application/json' \
  -d '{"api_base":"http://<this-engine>:8000/v1"}'
```

3. In CoeOS's *TMB Settings*, bind axes to the models this engine serves
   (registry entries reference the model by its engine alias as an `endpoint`
   model). An axis bound to a model that is not loaded here shows as
   unservable in CoeOS's *Routing table* until you load it.

## Using it

Point your tools at **CoeOS**, not at this engine:

```bash
export OPENAI_BASE_URL=http://<coeos-host>:4600/v1
export OPENAI_API_KEY=ck_…
curl "$OPENAI_BASE_URL/chat/completions" -H "authorization: Bearer $OPENAI_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"model":"CoeOS","messages":[{"role":"user","content":"Write a Python script that…"}]}'
```

The response headers say who answered: `x-coeos-axis`, `x-coeos-model`,
`x-coeos-provider`. An agent that already knows the step's skill can force it:
`-H 'x-coeos-axis: legal_rgpd'`.

## When to call the engine directly instead

- You want a **specific** model, not "the best per skill" — call this engine's
  `/v1` with the model alias.
- You are benchmarking a model (routing would hide which one you measured).
- Single-user, single-model setups: CoeOS adds a hop for no gain.

## Read next

- [Multi-user serving](multi-user-serving.md) — replica pools, the usual backend for CoeOS axes.
- [API](../API.md) — the full `/v1` and `/admin` surface of this engine.
- [CoeOS user guide](https://github.com/Odyssai-eu/coeos-box/blob/main/doc/USER-GUIDE.md) — axes, keys, users, guarantees.
