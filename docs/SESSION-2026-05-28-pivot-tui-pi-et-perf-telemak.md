# Session 2026-05-28 — pivot TUI Pi, et Telemak natif bat Argo

> Journée à deux fils qui s'entrecroisent. Pi : le bridge HTTP qu'on a
> chassé toute la nuit du 27 reste bloqué par le bug "long-lived
> parent → Connection error" qu'on n'a jamais cracké. Sophie tranche
> en une phrase — *"Pi est l'outil, pas les outils de Pi"* — et on
> pivote sur une archi TUI : ttyd + tmux sur `.50`, iframe dans
> Companion. Le terminal Pi natif rendu tel quel. Telemak : on
> instrumente l'activité côté Odysseus (pill "generating" + sessions
> KV), puis Sophie compare MiniMax-M2.7 sur Kolos vs Argo et trouve
> un 19× d'écart en défaveur du natif Telemak. Le 1.5 tok/s révèle
> deux bugs distincts (chunks 1-pour-1 côté Telemak Swift, leak du
> `<think>` côté Odysseus proxy). Une fois fixés, **Telemak natif tape
> 43.3 tok/s decode vs Argo 28 tok/s** — la prédiction de Sophie
> *"on doit faire mieux vu que c'est en natif"* est validée
> empiriquement.
>
> Versions de sortie : Odysseus `internal/main` v1.7.8 → suite avec
> `d73da66` + `9a2d73f`, Companion v0.2.2 (deux commits sur main).

---

## TL;DR — la journée en 4 axes

| Axe | Livré |
|---|---|
| **Pi pivot TUI** | brew install ttyd + tmux sur `.50`, ttyd:7681 expose `tmux attach -t pi`, nouveau composant `PiPanel` iframe dans Companion, `/pi` route vers ce panel quand `activeAgent==="pi"`. Bypass total du bug bridge d'hier. Sophie : *"ca marche, il a écrit le fichier"*. |
| **Hermes default model** | Switch de `or:hy3-preview` (via LiteLLM) vers `telemak-code-next` (via Odysseus). Schema YAML Hermes corrigé (`base_url` pas `api_base`). Gateway restart. Smoke 200 OK. |
| **Telemak activity sur Odysseus** | `/admin/clusters/{id}/status` enrichi : `busy`, `active_sessions_count`, `last_request_seconds_ago`, `sessions[]`, `requests_served`, `uptime_s`, `upstream_version`. Pills "● generating" pulsant + "N sessions" sur les cards. Commit `d73da66`. |
| **MiniMax 1.5 → 43 tok/s** | Deux bugs en série : Telemak Swift émettait 1 chunk SSE par token (Sophie installe 0.6.10 avec batching upstream), et Odysseus skippait le `<think>` filter dès que `enable_thinking=false` alors que MiniMax ignore ce flag. Fix `9a2d73f` + `_MODELS_IGNORE_ENABLE_THINKING_FLAG`. Bonus : Companion affiche maintenant `Decode: X tok/s` (commit `e2c875a`) excluant TTFT du calcul. |

---

## 1. Matin — pivot TUI pour `/pi`

La nuit du 27 s'était terminée sur un mur : trois architectures de
bridge HTTP testées, toutes bloquées par le même quirk macOS — quand
Pi est spawné par un process parent long-lived, ses fetch outbound
vers le LAN partent en `Connection error` après ~10 ms. Standalone ça
marche, dans un host long-lived ça meurt. Bug pas cracké, STATE doc
écrit, code Companion `addon-pi.ts` + `agent-pi.ts` shippé mais
dormant (commit `51194ba`, addon `enabled=false` en DB).

Sophie hier soir avait posé le recadrage qui éclaire toute la journée :

> *"Pi est l'outil, pas les outils de pi."*

Aujourd'hui elle reformule en plus actionnable :

> *"est ce que l'approche TUI ne serait pas plus simple ?"*

Et oui, c'est radicalement plus simple. Pi a un TUI natif (default
mode interactif). Si on l'expose via [ttyd][1] derrière un `tmux
attach`, n'importe quel navigateur peut s'y connecter en
WebSocket+xterm.js. Companion devient juste un **dock** qui embed le
terminal Pi tel quel — pas de réinterprétation d'events, pas de
traduction `text_delta` → `sessionUpdate`, pas de bridge à maintenir.
Et surtout : le `pi` lancé via tmux dans le shell ssh interactif n'a
JAMAIS reproduit le bug Connection error (parent-process context
correct).

[1]: https://github.com/tsl0922/ttyd

### Setup sur `.50`

`brew install ttyd tmux` (broken initialement — `/opt/homebrew` had
mauvaise ownership, `sudo chown -R admin /opt/homebrew` fixe), puis :

```bash
tmux new-session -d -s pi -c /Volumes/Big_Twenty/Workspace \
  'pi --provider odysseus --model telemak-code-next'
ttyd -p 7681 -W tmux attach -t pi
```

ttyd répond `HTTP 200`, Pi tourne dans la session tmux, attaché au
workspace SMB partagé. Tout l'edifice en 4 commandes shell.

### Companion side

Trois fichiers patchés, 0 frontend complexe :

- `src/components/chat/PiPanel.tsx` — composant simple : header
  "Pi · terminal · tape directement dans la fenêtre" + iframe vers
  `bridgeUrl` (la ttyd URL).
- `src/hooks/useChat.ts` — fetch `piAddonInfo` au mount, expose
  `piBridgeUrl`. `/pi` slash command ne fait plus que toggle
  `activeAgent = "pi"` (pas d'invoke SSE). Composer en mode `pi`
  affiche "Type directly in the Pi terminal above" et n'envoie rien.
- `src/layouts/ChatLayout.tsx` — `chat.activeAgent === "pi"` rend
  `<PiPanel url={piBridgeUrl} />`, sinon AgentBubble classique.

DB : addon "Pi Agent" inserté avec `bridgeUrl =
http://192.168.86.50:7681`, `enabled=true`. Commit `27c74df` côté
Companion (TUI pivot), build → deploy v0.2.2 sur `.39`.

### Smoke

Sophie tape `/pi` puis dans le terminal : *"create de new folder
'test03'"* → `mkdir -p test03` exécuté, dossier créé. Puis *"write a
story of 50 words and store it inside in .md"* → `story.md` créé,
275 octets, contenu cohérent. Sophie :

> *"ca marche, il a écrit le fichier"*

Pivot validé. La vraie victoire : on a remplacé ~600 lignes de bridge
Python + extension TypeScript + traduction d'events par 50 lignes de
composant iframe et 3 commandes brew. Et bonus, Pi reste Pi — l'user
voit le terminal natif avec ses commandes `/model`, `/login`, ses
skills, son plan mode. Companion ne le réinterprète plus.

### Le bug "long-lived parent" reproduit dans la foulée

Dans la matinée, j'ai voulu relancer le tmux pour fixer le cwd
(première session lancée depuis `$HOME` au lieu de
`/Volumes/Big_Twenty/Workspace` → Pi hallucinait *"Workplace"* dans
ses mkdir). Je relance via ssh non-interactif + nohup → Pi reprend
le bug d'hier soir : Connection error sur toutes ses fetch vers
Odysseus. Sophie relance manuellement depuis son ssh interactif → tout
remarche.

**Donc le bug est bien lié au process parent.** Le pivot TUI le
contourne plutôt que le résoudre. J'ai aussi écrit un plist launchd
(`~/Library/LaunchAgents/com.thecompai.pi-ttyd.plist` +
`~/.local/bin/pi-ttyd-launcher.sh`) pour auto-restart au boot —
testé, **launchd reproduit le bug aussi**. Plist garde sur `.50` pour
le futur, pas utilisé.

### Pi 0.75.5 → 0.76 + switch modèle

Premier essai live, Pi en boucle infinie sur `mkdir Test02` — le
modèle (Gemma 4 26B-A4B = Codermac) ne reconnaît pas le succès
silencieux d'Unix. Update vers Pi 0.76.0 (notification "Update
Available" en haut du TUI) + switch du modèle par défaut vers
`telemak-code-next` (Qwen3-Coder-Next sur `.32`, déjà chargé). Nouveau
test : Pi écrit le fichier correctement. La distinction entre "Pi en
boucle" et "Pi marche" était entièrement liée au choix du modèle
backend.

---

## 2. Midi — Hermes sur Codermac

Sophie demande :

> *"tu peux mettre le modele par default de Hermes sur
> telemak-code-next : inferencerlabs/Qwen3-Coder-Next-MLX-9bit"*

L'archi Hermes par défaut routait via LiteLLM (`.44:4000`) avec
`model.default: tencent/hy3-preview` (qui s'était cassé hier avec
"Invalid model name"). On a maintenant un Odysseus en local qui sert
`telemak-code-next`, plus propre de bypasser LiteLLM.

`~/.hermes/config.yaml` patch (backup en
`config.yaml.bak-pre-tele-code-next-20260528-135757`) :

```yaml
model:
  provider: odyssai
  base_url: http://192.168.86.39:8000/v1
  api_key: ''
  tool_call_parser: qwen3
  default: telemak-code-next
providers:
  odyssai:
    base_url: http://192.168.86.39:8000/v1
    api_key: ''
fallback_providers:
- provider: odyssai
  model: telemakcoder
  context_length: 131072
```

Premier essai : `WARNING hermes_cli.config: providers.odyssai:
unknown config keys ignored: api_base, type`. Le schéma Hermes attend
`base_url` (pas `api_base`) et ne connaît pas `type:`. Patch v2 sans
ces clés → restart `hermes gateway run --replace` → `POST
.50:8642/v1/chat/completions` retourne 200 OK avec "Hello! How can I
help you today?".

Hermes (gateway) ne passe plus par LiteLLM. Il appelle Odysseus en
direct, qui route vers le cluster `telemak-code-next` sur ultra-256c.

---

## 3. Après-midi — instrumenter l'activité Telemak

Question Sophie :

> *"est ce que telemak expose son activité ? on peut voir sur Odysseus
> ce qu'il fait (running, ...)"*

Telemak expose `/health` (model, wired, avg_tok_s_recent, requests) et
`/admin/sessions` (KV-cached sessions avec `last_used_s`). Mais pas de
flag "is_generating" direct. L'idée : agréger côté Odysseus.

Patch `_telemak_status` dans `scripts/api.py` — ajoute en parallèle
de la requête `/v1/models` + `/health` un appel `/admin/sessions`,
calcule :

```python
BUSY_WINDOW_S = 5.0
busy = any(session.last_used_s < BUSY_WINDOW_S for session in sessions)
active_sessions_count = len(sessions)
last_request_seconds_ago = min(s.last_used_s for s in sessions)
```

Exposés dans la réponse de `/admin/clusters/{id}/status` :

```json
{
  "busy": false,
  "active_sessions_count": 2,
  "last_request_seconds_ago": 450.5,
  "sessions": [{"id": "f62a1b61", "model": "…", "kv_size_mb": 95, "last_used_s": 450.5}],
  "requests_served": 25,
  "uptime_s": 3341.0,
  "upstream_version": "0.6.5"
}
```

Côté dashboard, deux nouveaux pills :

```javascript
const busyPill = a.busy
  ? `<div class="pill-status" style="background:#fff8e6;...color:var(--amber-deep)">
       <span style="animation:pulse 1s infinite">●</span> generating
     </div>` : "";
const sessionsPill = a.active_sessions_count > 0
  ? `<div class="pill-status" title="last touched ${a.last_request_seconds_ago.toFixed(1)}s ago">
       ${a.active_sessions_count} sessions</div>` : "";
```

`@keyframes pulse` ajouté. Hot-deploy `api.py` + `dashboard.html`
dans le container, restart, push commit `d73da66`. Sur les cards
Telemak du dashboard, quand on envoie un prompt, le pill orange
"● generating" apparaît + clignote, puis disparaît ~5s après la fin
de la génération. "N sessions" reste visible tant que le KV cache
tient.

---

## 4. Soir — MiniMax-M2.7 sur Kolos : 19× plus lent qu'Argo

Sophie lance MiniMax-M2.7-8bit sur Kolos (`telemak512` = ultra-512) et
me partage le bench :

```
TTFT: 3978ms · Duration: 129.51s · Completion: 189 tok ·
Speed: 1.5 tok/s · Chunks: 189 · Model: telemak512 — MiniMax-M2.7-8bit
```

Pendant qu'Argo (le même modèle, distribué sur 2 nodes) sort :

```
TTFT: 2957ms · Duration: 15.29s · Completion: 429 tok ·
Speed: 28.1 tok/s · Chunks: 42 · Model: default — MiniMax-M2.7-8bit
```

**19× plus lent côté Telemak natif.** Sophie ne lâche pas :

> *"on doit faire mieux vu que c'est en natif"*

Elle a raison. Single-node MLX-Swift natif sans JACCL inter-node
overhead devrait taper plus haut, pas plus bas. Diagnostic en
profondeur.

### Bug A — chunks 1-pour-1 côté Telemak Swift

Sur Argo : **42 chunks pour 429 tokens** = ~10 tokens/chunk. Sur
Telemak : **189 chunks pour 189 tokens** = 1 token/chunk. Chaque
chunk SSE = (sérialisation JSON + frame HTTP + flush + RTT réseau) ×
3 hops (Telemak → Odysseus → Companion → browser). À 189 chunks ×
~600 ms chacun = ~113 s d'overhead protocole. Match la durée
observée.

Source confirmée dans `Sources/Telemak/Server/ChatCompletions.swift` :

```swift
for try await gen in session.streamDetails(to: userPrompt, ...) {
    switch gen {
    case .chunk(let piece):
        let chunk = ChatCompletionChunk(...)
        try await send(chunk)   // ← un send par token
```

Pas de batching. Sophie a lancé l'upgrade Telemak `0.6.10` upstream
avec le fix :

> *"il installe une mise a jour. avec envois par lot"*

Après update, retest : **47 chunks pour 507 tokens** (~10/chunk) →
batching actif, comme Argo.

### Bug B — `<think>` leak dans `content` pour MiniMax

Sophie m'envoie deux JSON Companion downloadés côté à côté. Argo :
content propre, story directement. Telemak : content qui commence par
*"The user is asking me to write a story of 100 words in French. Let
me write a short story.\n</think>\n\n# Le Vieil Homme et la Lampe"*.
Le reasoning + le tag `</think>` literal **leakent en visible**.

Cause racine dans `scripts/api.py:_telemak_proxy_chat_completion` :

```python
enable_thinking = body.get("enable_thinking")
if enable_thinking is False:
    auto_think = False          # skip filter
else:
    auto_think = _model_auto_opens_think(upstream_model)
```

La logique : si le client dit "pas de thinking", Telemak template
suppress le `<think>` block, donc pas besoin de filter. **Vrai pour
Qwen3.5 / Qwen3.6, faux pour MiniMax M2** — les docs MiniMax disent
explicitement :

> *"The model's reasoning is wrapped in <think> tags within the
> content field. Do not modify the content field."*

MiniMax IGNORE le flag `enable_thinking`. Companion par contre envoie
`enable_thinking: false` par défaut pour tous les modèles. Résultat :
le model émet quand même son `<think>...</think>`, Odysseus skip le
filter, le reasoning + `</think>` literal lande dans `content`.

Fix `9a2d73f` — split `_MODELS_AUTO_OPEN_THINK` en deux sets :

```python
_MODELS_AUTO_OPEN_THINK = ("minimax", "qwen3.5", "qwen3.6")
# Subset qui IGNORE enable_thinking — always filter, même avec false
_MODELS_IGNORE_ENABLE_THINKING_FLAG = ("minimax",)

# Telemak proxy logic :
ignores_flag = _model_ignores_enable_thinking_flag(upstream_model)
if enable_thinking is False and not ignores_flag:
    auto_think = False
else:
    auto_think = _model_auto_opens_think(upstream_model)
```

Smoke direct stream=true → `delta.reasoning_content` reçoit le
reasoning, `delta.content` reste propre. Filter actif.

### Résultat combiné

Avec les deux fixes en place, Sophie retest MiniMax sur Kolos :

```
TTFT: 3503ms · Duration: 15.20s · Completion: 507 tok ·
Speed: 33.4 tok/s · Decode: 43.3 tok/s · Chunks: 47
```

**43.3 tok/s decode pur sur natif** vs 28 tok/s sur Argo distribué.
**1.5× plus rapide qu'Argo**. Sophie avait raison sur le principe :
single-node MLX-Swift gagne sur distribué Python + JACCL pour les
modèles qui tiennent en RAM.

---

## 5. Bonus — métrique Decode tok/s dans Companion

Pendant l'investigation, Sophie regarde un bench Mistral-Medium :

```
TTFT: 4808ms · Duration: 34.00s · Completion: 147 tok ·
Speed: 4.3 tok/s
```

Inferencer annonce 5 tok/s pour ce modèle. Elle remarque l'écart.
Calcul rapide : `147 / (34.00 - 4.808) = 5.04 tok/s` → exactement la
spec. La différence vient du **TTFT compté dans le dénominateur** du
"Speed" Companion.

Patch `server/routes/chat.ts` — ajoute un champ `decodeSpeed` :

```typescript
decodeSpeed:
  st.totalMs && st.ttftMs !== null && st.completionTokens
    && st.totalMs - st.ttftMs > 0
    ? `${((st.completionTokens / (st.totalMs - st.ttftMs)) * 1000).toFixed(1)} tok/s`
    : undefined,
```

`StatsRow` affiche `Decode: X tok/s` à côté de `Speed` quand les deux
diffèrent. Sur les modèles à grand prompt eval, l'écart devient
explicite ; sur les modèles à TTFT négligeable, le pill Decode est
masqué (no clutter). Commit `e2c875a`.

---

## 6. Bémol — Mistral-Medium-3.5 multimodal pas supporté par Telemak

Tentative de charger `inferencerlabs/Mistral-Medium-3.5-MLX-9bit` sur
un Telemak → erreur :

```
configurationDecodingError("config.json", "staged-models/--Volumes--…")
```

Vérif du config.json :

```json
"architectures": ["Mistral3ForConditionalGeneration"],
"model_type": "mistral3",
"vision_config": { ... },
"text_config": { "model_type": "ministral3" }
```

C'est un modèle **multimodal** (texte + vision). `mlx-swift-lm` 3.x
décode le `ministral3` text-only mais pas le wrapper
`Mistral3ForConditionalGeneration`. Toutes les variantes
`odyssai/Mistral-Medium-3.5-128B-*` sont aussi multimodales. Le seul
ministral3 text-only dispo dans le model dir est
`Devstral-2-123B-Instruct-2512-bf16`.

À filer en feature request mlx-swift-lm (support
`Mistral3ForConditionalGeneration`). Pas un blocker — Devstral
remplace pour le coding text-only.

---

## Fichiers modifiés / créés

**Companion** (`~/Claude/code/thecompai/app/`)
- `src/components/chat/PiPanel.tsx` — nouveau, iframe wrapper Pi terminal
- `src/hooks/useChat.ts` — `piBridgeUrl` state, `/pi` slash fait juste toggle, persistent mode no-op
- `src/layouts/ChatLayout.tsx` — render `<PiPanel>` quand `activeAgent==="pi"`
- `src/components/chat/Messages.tsx` — affiche `Decode` à côté de `Speed`
- `server/routes/chat.ts` — calcule `decodeSpeed` côté backend

**Odysseus** (`~/Claude/code/MLX Distributed/scripts/`)
- `api.py` — `_telemak_status` enrichi (busy/sessions/requests_served/upstream_version),
  `_MODELS_IGNORE_ENABLE_THINKING_FLAG` ajouté, `_telemak_proxy_chat_completion` honore le split
- `dashboard.html` — pills `busyPill` + `sessionsPill` sur les cards Telemak, `@keyframes pulse`

**Pi host** (`.50`)
- `/Users/admin/.local/bin/pi-ttyd-launcher.sh` — wrapper tmux+ttyd
- `/Users/admin/Library/LaunchAgents/com.thecompai.pi-ttyd.plist` — déposé, désactivé
  (launchd reproduit le bug Connection error, à creuser plus tard)
- `~/.hermes/config.yaml` — provider odyssai en primary,
  default `telemak-code-next`, fallback `telemakcoder`. Backup en
  `config.yaml.bak-pre-tele-code-next-20260528-135757`.

---

## Numbers de la journée

- **Commits** : 2 sur Odysseus (`d73da66` + `9a2d73f`), 2 sur Companion (`27c74df` + `e2c875a`). 1 commit dormant d'hier (`51194ba`) côté Companion — bridge-pattern Pi routes kept-around.
- **Lignes diff** : Odysseus +111 / -8 (2 fichiers), Companion +136 / -22 (5 fichiers).
- **Push** : `internal/main` (Odysseus) + Companion deploy v0.2.2 sur `.39` via `deploy-prod.sh skip`.
- **Smoke** : `/pi` end-to-end (test03/story.md écrit), `/hermes` 200 OK sur Codermac, MiniMax sur Kolos 43.3 tok/s decode confirmé, Hermes gateway répond après config switch.
- **Bug fixés** : 2 (chunks 1-token côté Telemak Swift → upstream 0.6.10, think leak côté Odysseus → `9a2d73f`).
- **Bug contournés** : 1 (long-lived parent Connection error sur Pi — TUI pivot bypasse au lieu de fixer).

---

## TODO direct (par ordre)

1. **Pi launchd plist** — actuellement déposé mais inactif (reproduit le bug). Soit on crack le mystère long-lived parent, soit on utilise un autre mécanisme (launchctl as user-scope sur foreground ?). Sophie doit relancer manuellement après reboot.
2. **Decode metric pour `/hermes` et `/pi`** — ces paths ne passent pas par `chat.ts`, donc le champ `decodeSpeed` n'est pas calculé pour eux. À étendre.
3. **Mistral3ForConditionalGeneration upstream** — feature request mlx-swift-lm pour le multimodal mistral3 (Mistral-Medium-3.5).
4. **Renderer Pi-native dans Companion** — pour l'instant on iframe le terminal ttyd ; à terme, parser les events Pi natifs (`pi.text_delta`, `pi.tool_call`) pour un rendu intégré au chat. Pas urgent — le iframe fait le job.
5. **Companion bridge-pattern Pi routes** — `addon-pi.ts` + `agent-pi.ts` restent en tree, dormants. À retirer quand le bug upstream est fixé ou à conserver si on veut un fallback structuré. À décider.

---

## Lessons learned

**Reframe > fix.** Le pivot TUI a court-circuité 8 heures de debug
réseau hier soir et toute la matinée d'aujourd'hui. La phrase de
Sophie *"Pi est l'outil, pas les outils de Pi"* a posé un cadre qui
a rendu le bridge HTTP optionnel. Plus efficace que de chercher le
syscall qui foire.

**La preuve par le perf.** Sophie a poussé deux fois sur la vitesse
("on doit faire mieux vu que c'est en natif") parce qu'elle savait
intuitivement qu'un single-node natif doit battre du distribué
Python. La intuition a tenu — 43 vs 28 tok/s. Sans le push, on
aurait accepté un Telemak Swift mediocre vs Argo. Le bench compare
"même modèle, même prompt, deux backends" est la métrique la plus
honnête.

**Les flags optionnels n'ont pas tous le même contrat.** Le pattern
`enable_thinking=false` est traité différemment par chaque famille
de modèle. Qwen le respecte au template, MiniMax l'ignore. Le
proxy Odysseus ne peut pas trust le flag uniformément — il faut une
table des comportements par famille. Le commit `9a2d73f` matérialise
ce pattern (`_MODELS_IGNORE_ENABLE_THINKING_FLAG`), réutilisable
quand d'autres familles apparaîtront.
