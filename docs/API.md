# OdyssAI-X — API

OpenAI + Anthropic compatible HTTP server, multi-cluster, multi-model.
Reference for the current release (see `APP_VERSION` in `scripts/api.py`).

---

## Démarrage

```bash
# Production — the orchestrator container (see docker-compose.yml)
docker compose up -d                       # first start
docker restart odyssai-odysseus            # after a hot-patch of api.py

# Dev local (no model; pools restore from the persisted state)
python scripts/api.py --port 8000
```

**Préférence** : démarrer sans `--model`. Le lifespan restaure les pools
persistés automatiquement, puis charger via le dashboard ou
`POST /admin/clusters/{id}/load`.

---

## Topologie

L'API orchestre **N clusters** (définis dans `topology.yaml`, persistés dans
`cluster-config.json`). Chaque cluster a un **kind** :

| Kind | Ce que c'est | Nodes | Exemple d'id |
|---|---|---|---|
| `mlx-distributed` | un modèle réparti sur les nodes (pipeline / tensor parallel), backend `ring` ou `jaccl` | 1-N | `default`, `reasoner` |
| `replica` | N copies mono-node du même modèle, continuous batching, dispatch least-busy + affinité de session | 1-N | `chat` |
| `telemak` | pool **http-proxy** : le cluster est un endpoint OpenAI-compatible amont (`upstream`) que l'orchestrateur proxifie — serveur `mlx_vlm.server` pour la vision, ou tout serveur local | 1 | `vision`, `coder` |

(`telemak` est le nom historique du kind http-proxy ; il reste le littéral de l'API.)

Routing `/v1/chat/completions` :
1. Alias cluster (ex. `"default"`, `"chat"`) → match direct
2. `cluster:short_id` (ex. `"chat:Qwen3.8-Flash-Next-Q6"`) → multi-modèle
3. Alias cloud (ex. `"or:claude-haiku"`, `"anthropic:claude-3-5-sonnet"`)
4. Fallback : premier pool chargé

---

## Endpoints

### Public (OpenAI / Anthropic compat)

#### `GET /v1/models`

Liste tous les modèles loadés + aliases cloud. Chaque entrée inclut
`x_odyssai` (capability contract) :

```json
{
  "object": "list",
  "data": [
    {
      "id": "default",
      "object": "model",
      "owned_by": "odysseus-mlx",
      "x_odyssai": {
        "ready": true,
        "loaded": true,
        "alias_for": "mlx-community/Step-3.7-Flash-8bit",
        "family": "Step-3.7-Flash-8bit",
        "quantization": "8-bit",
        "kind": "mlx-distributed",
        "cluster_label": "Argo",
        "backend": "jaccl",
        "nodes": 1,
        "context_length": 131072,
        "supports_tools": true,
        "supports_vision": false
      }
    },
    {
      "id": "vision",
      "owned_by": "odysseus-telemak",
      "x_odyssai": {
        "ready": true,
        "loaded": true,
        "alias_for": "mlx-community/Qwen3-VL-32B-8bit",
        "kind": "telemak",
        "cluster_label": "Vision",
        "upstream": "http://vision-node.lan:8013"
      }
    }
  ]
}
```

Le **model picker de CoeOS** affiche `family` (nom court sans org + quant)
comme label primaire, et la pastille de runtime `kind` (http-proxy / distribué / replica /
cloud).

#### `POST /v1/chat/completions`

OpenAI standard + extensions vendor.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role":"user","content":"def fib(n):"}],
    "max_tokens": 512,
    "stream": true,
    "enable_thinking": false,
    "reasoning_effort": "minimal",
    "session_id": "conv-abc-123"
  }'
```

Champs supportés (au-delà du standard OpenAI) :
- `enable_thinking` (bool) — active/désactive le bloc `<think>` pour les
  modèles qui le supportent (Qwen3.5/3.6). Ignoré pour les always-think
  (MiniMax, Step-3.7).
- `reasoning_effort` (`"none"` | `"minimal"` | `"low"` | `"medium"` |
  `"high"` | `"xhigh"`) — cadran d'effort de raisonnement OpenAI o-series.
  Injecté dans le chat template comme `Reasoning: <effort>`. Pour Step-3.7
  (always-think), **défaut automatique `"minimal"`** côté Odysseus — aucune
  config requise, mais overridable par requête.
- `session_id` — opt-in prefix cache. Aussi via header `X-Session-Id`.

**Filtrage `<think>` automatique** : pour les modèles always-think
(Step-3.7, MiniMax), Odysseus route le contenu `<think>…</think>` vers
`delta.reasoning_content` (canal replié) au lieu du content visible.
Aucune config client requise.

Réponse standard + `x_mlx_cluster` :
```json
{
  "choices": [{"message": {...}, "finish_reason": "stop"}],
  "usage": {"completion_tokens": 80},
  "x_mlx_cluster": {
    "elapsed_s": 2.71, "ttft_s": 0.63, "tps": 29.5,
    "session": {"id": "conv-abc-123", "cache_kind": "session-HIT",
                "cumulative_tokens": 124}
  }
}
```

#### `POST /v1/messages`

Anthropic Messages API compat.

```bash
curl -X POST http://localhost:8000/v1/messages \
  -d '{"model":"default","max_tokens":100,"system":"...",
       "messages":[{"role":"user","content":"..."}],
       "metadata":{"session_id":"conv-abc"}}'
```

Body : `system`, `messages`, `tools` (format Anthropic), `stream`,
`metadata.session_id` ou header `X-Session-Id`.

---

### Admin (LAN-only, no auth par défaut)

> `/admin/*` est **ouvert sur LAN** (`ODYSSEUS_ADMIN_TOKEN` non set).
> Protéger avec `ODYSSEUS_ADMIN_TOKEN=<secret>` avant toute exposition WAN.
> `GET /health` expose `admin_auth_enabled: bool` pour détection programmatique.

#### Clusters — generic (`/admin/clusters`)

- `GET /admin/clusters` — liste tous les clusters définis
- `GET /admin/clusters/{id}/status` — snapshot statut (loaded, model,
  nodes, busy, sessions, wired memory…)
- `GET /admin/clusters/{id}/models` — discover models sur le nœud master
- `POST /admin/clusters/{id}/load` — charger un modèle
  ```json
  {"model": "/Volumes/models/odysseus/mlx-community/...",
   "mode": "pipeline", "use_ap": true, "nodes": 2, "kv_q8": false}
  ```
- `POST /admin/clusters/{id}/unload` — décharger
- `GET /admin/status` — état agrégé de tous les pools actifs

#### Clusters http-proxy (kind `telemak`) — lifecycle

- `POST /admin/clusters/{id}/telemak/lifecycle`
  ```json
  {"action": "stop"}   // "stop" | "start" | "restart" | "quit"
  ```
  - `stop` : bounce (launchd relance automatiquement)
  - `quit` : bootout (reste à terre jusqu'au prochain reboot)

#### Métriques / sessions / logs

- `GET /admin/metrics?cluster=&limit=` — inférences récentes
  `{ts, client, cluster, model, ntoks, elapsed_s, ttft_s, tps, session_kind}`
- `GET /admin/sessions?cluster=` — sessions prefix-cache actives (< 1h)
- `POST /admin/sessions/clear` `{session_id?: "..."}` — vider session(s)
- `GET /admin/logs?cluster=&tail=&follow=` — runner stderr (SSE si `follow=true`)

#### Health

`GET /health` :
```json
{"status":"ok","version":"1.7.28","admin_auth_enabled":false,
 "model":"...","alive":1,"nodes":1}
```

---

## Prefix cache (sessions)

Opt-in via `session_id` (body) ou `X-Session-Id` (header).

**Mesures réelles (Qwen3-Coder-Next, Argo 2-node JACCL)** :
- T1 fresh : TTFT 0.86 s
- T2 session-HIT : TTFT **0.11 s** (8× plus rapide)

**Mesures conversation 4 tours (Step-3.7-Flash, pool http-proxy)** :
- T1 : Prompt 12 561 tok, TTFT 12 s (vault injecté)
- T2-T4 : Prompt 48-124 tok, TTFT 1-2 s (KV cache prefix reuse)

LRU + TTL 1 h + max 32 sessions par runner.

---

## `reasoning_effort` — toujours-penseurs

Certains modèles ouvrent **toujours** un bloc `<think>` (Step-3.7-Flash,
MiniMax) et ignorent `enable_thinking:false`. Le seul levier est
`reasoning_effort`, qui injecte `Reasoning: <effort>` dans le system
prompt via le chat template.

Odysseus applique un **défaut par-modèle** :

| Modèle | Défaut Odysseus |
|---|---|
| Step-3.7-Flash | `minimal` |
| Autres | `none` (pas d'injection) |

Overridable par requête : `"reasoning_effort": "high"` pour des tâches
qui bénéficient d'un raisonnement long (fiction contrainte, preuves,
code complexe). Non overridable pour les modèles qui honorent
`enable_thinking` (Qwen3.5, Qwen3.6) — utiliser `enable_thinking:false`
à la place.

---

## Filtrage `<think>` — filtre fil de pensée

Tous les modèles always-think (et les modèles en mode thinking) ont leur
contenu `<think>…</think>` **extrait** du `content` visible et routé dans
`delta.reasoning_content` (canal replié dans le client CoeOS).

Décision partagée `_should_filter_think(model_id, enable_thinking)` :
- `enable_thinking:false` + modèle honore le flag (Qwen3.5/3.6) → pas de
  filtre (pas de bloc émis → pas de ghost answer)
- `enable_thinking:false` + modèle ignore le flag (MiniMax, Step-3.7) →
  filtre quand même
- `enable_thinking:true` (ou always-think) → filtre, route vers reasoning

Comportement identique sur le chemin local (pool Argo) et le chemin proxy
(pool http-proxy).

---

## Tool calling

`x_odyssai.backend_quirks` dans `/v1/models` documente les bugs upstream
**déjà corrigés** côté proxy — les clients n'ont rien à faire.

| Backend | Stream + tools | Quirks corrigés |
|---|---|---|
| `jaccl` (Argo) | OK, deltas incrémentaux | aucun |
| `http-proxy` cloud | Pass-through verbatim | aucun |
| `http-proxy` LAN | Stream-to-unary si tools | `stream_tools_empty_deltas`, `finish_reason_stop_with_tools` |

Formats tool call reconnus par le runner :

| Format | Modèles |
|---|---|
| Hermes JSON `<tool_call>{"name":..}` | Qwen3, GLM-4 |
| Qwen3-Coder XML `<function=NAME><parameter=KEY>` | qwen3_coder |
| Hy3 XML | Hy3-preview |

---

## Caveats

1. **Submits concurrents** : `broadcast_lock` micro-section. Concurrence
   OK sur single-rank (BatchGenerator) ; multi-rank serial.
2. **Q8 KV cache** désactive BatchGenerator (mode legacy single-stream).
3. **Speculative decoding** désactive aussi BatchGenerator.
4. **Disconnect mid-stream** : stop flag posé, le runner finit le tick
   courant (pas annulable au niveau Metal/MLX).
5. **`prompt_tokens` toujours 0** dans `usage` — le runner ne remonte
   pas le count d'entrée (tracked dans `/admin/metrics`).

---

## Architecture interne (rappel)

```
HTTP request → FastAPI (api.py)
  ↓ route_pool() → cluster lookup
RunnerPool.submit(messages, tools, session_id, reasoning_effort)
  ↓ broadcast_lock micro-section
  ↓ apply_chat_template (inject reasoning_effort si défini)
SSH-spawned runner.py (long-lived) sur chaque rank
  ↓ JSONL stdin/stdout
mlx-lm stream_generate OU BatchGenerator (single-rank)
  ↓ JACCL TB5 collectives multi-rank + KV prefix cache
tokens streamed back → api.py → client SSE
```

Voir [DEPLOY.md](DEPLOY.md) pour la procédure de déploiement.
