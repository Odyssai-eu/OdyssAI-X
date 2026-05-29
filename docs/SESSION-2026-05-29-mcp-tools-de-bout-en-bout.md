# Session 2026-05-29 — MCP tools, de bout en bout

> Journée qui démarre sur le carve-out propre d'un chantier
> (DeepSeek V4 parked en attendant le support MLX upstream) et qui
> bascule sur l'UX produit. Refonte du model picker côté Companion
> (catégorie TELEMAK + rename MAIN → ARGO + display labels + dropdowns
> Auto Router), nettoyage d'un bouton mort sur l'admin Telemak côté
> Odysseus. Puis Sophie active un MCP dans Companion et tout s'effondre :
> Tavily refuse de répondre, les modèles bouclent, le loop-guard affiche
> un message trompeur sur Haiku. La cascade qui suit débogue cinq
> couches successives (cruft macOS dans la mémoire, schemas JSON null
> qui crashent le Jinja Swift, MAX_TOOL_ITERATIONS trop bas, JSON
> double-encodé, cap 4k qui coupe la JSON tavily en plein milieu) et
> ship cinq versions de Companion en file. Le bug final résiste : Coder-
> Next sur Telemak boucle même avec des résultats clean, alors qu'il
> synthesise au premier coup sur Argo. Diagnostic structurel transmis
> en commentaire d'issue, Telemak side reçoit 5/5, ship `0.6.20`
> exactement sur le root cause théorisé : le path "flatten tool history
> en texte" perdait l'historique structuré des `assistant.tool_calls` +
> `role: "tool"`. Le modèle souffrait d'amnésie. Validation Companion-
> side en fin de journée : TeleCoder + Tavily + Qdrant + Corpus = full
> MCP stack sur petits modèles locaux. La proposition de valeur
> Companion existe en prod.
>
> Versions de sortie : Odysseus v1.7.8 → **v1.7.12** (4 bumps),
> Companion v0.2.2 → **v0.2.8** (5 bumps), Telemak `0.6.20` (livré
> par leur side sur la base du diag du jour).

---

## TL;DR — la journée en 4 axes

| Axe | Livré |
|---|---|
| **Carve-out DeepSeek V4** | Reproduit le mur quant hétérogène (`Received 1423 parameters not in model`), park propre du chantier : `deepseek_v4.py` déplacé hors du venv vers `~/mlx-cluster/parked-models/` sur les 4 ultras, 282 GB de weights gardés sur `.30` + `.31`, task #46 documentée "ON HOLD waiting MLX update". Sophie : *"on attend le support officiel de mlx"*. |
| **Model picker rework** | Backend Odysseus `x_odyssai` enrichi (`cluster_label`, `kind`, `family`, `quantization` sur les entrées Telemak). Companion picker chat regroupe en TELEMAK + ARGO + CLOUD, display labels du contrat à la place des IDs de cluster, extraction de `ModelDropdown.tsx` partagé. Auto Router add-on remplace les 3 text inputs par des dropdowns (Chat / Deep / Code). `Manage models →` retiré des écrans admin Telemak (jamais valide pour les clusters http-proxy). Odysseus v1.7.10 → v1.7.12, Companion v0.2.3. |
| **MCP cascade — 5 fixes Companion** | (1) Filter macOS resource forks (`__MACOSX/*`, `._*`, `.DS_Store`) à la lecture ET à l'import du user vault, purge de 56 rows polluées en DB live ; (2) sanitiseur récursif des schemas JSON `null`-typés avant envoi à Telemak (crash Jinja Gemma-4) ; (3) `MAX_TOOL_ITERATIONS` 3 → 8 pour les workflows multi-step légitimes ; (4) stop double-JSON-encoding des MCP tool results (le `JSON.stringify` sur des strings déjà sérialisées induisait Coder-Next en erreur) ; (5) cap MCP 4k → 16k + marqueur explicite `…[result truncated, do not retry]`. v0.2.4 → **v0.2.8**, 5 commits, 5 vrais bugs trouvés. |
| **Diag Telemak + leur ship `0.6.20`** | Issue #52 ouverte avec cartographie 4-modèles, repro curl minimal, comparatif Argo vs Telemak sur même model file. Smoking gun isolé : 3 tool_calls byte-identiques côté Telemak alors qu'Argo synthesise au 1er résultat → amnésie du modèle sur son propre `tool_calls` précédent. Telemak side ship `0.6.20` avec exactement la root cause (bypass du path "flatten en texte" pour les requêtes avec `tool_calls` / `role: tool`, préservation via `UserInput(messages:)`). Validation Companion : *"yesss"* + Tour de France 2025 synthétisé proprement par TeleCoder. Qdrant + Corpus testés en bonus : tout marche. |

---

## 1. Matin — carve-out DeepSeek V4

La nuit du 28 s'était arrêtée sur DeepSeek V4 : Rapid-MLX vendoré dans
`mlx_lm.models.deepseek_v4`, deps présentes côté Argo
(`MultiLinear`, `PipelineMixin`, `BatchRotatingKVCache`), 282 GB
téléchargés sur `.30` (et copie en cours vers `.31`). Le mur arrive
au load : `model.load_weights` rejette 1423 paramètres orphelins
(`embed.scales`, `embed.biases`, `head.scales`, etc.). Le `sanitize()`
de Rapid-MLX n'a pas le remap de quantization-companions au-delà du
weight principal — l'engine mlx-lm vanilla ne sait pas non plus
appliquer le `make_quantization_config(model)` per-module que Rapid-
MLX construit dans son propre runtime.

Deux murs documentés (l'ancien `hc_head` shape mismatch d'avant +
le nouveau load_weights), Sophie tranche :

> *"on met en pause, on a d'autres chantiers. On attends le support
> officiel de mlx."*

Cleanup propre :

- `deepseek_v4.py` retiré des 4 ultras (move `mlx_lm/models/` →
  `~/mlx-cluster/parked-models/deepseek_v4.py.parked-2026-05-29-rapid-mlx`)
- import `from mlx_lm.models.deepseek_v4 import Model` → vérifié
  `ModuleNotFoundError` ✓ partout
- 282 GB weights gardés sur `.30` + `.31` (le jour où l'upstream
  arrive, on a juste à pop le fichier)
- task #46 status `pending`, subject "ON HOLD waiting MLX update",
  description avec les 2 murs + cleanup snapshot

Pas d'issue GitHub Odysseus filée — la task `#46` interne suffit
pour reprise, l'absence d'issue publique évite la dette de
maintenance "quand est-ce qu'on referme ça". Si l'upstream MLX
arrive avec un `mlx_lm.models.deepseek_v4` natif, on pop le fichier
parked, load tout marche.

---

## 2. Refonte du model picker (Companion + Odysseus)

Sophie envoie deux screenshots — le picker chat affiche les Telemak
dans le groupe `CLOUD` (faux : ce sont des Swift natifs locaux), et
`MAIN` héberge `default · qwen3_next · 9-bit · loaded` (Argo).
Demande en six points :

1. Catégorie `TELEMAK` distincte
2. Rename `MAIN` → `ARGO` (le pool id "main" a perdu son sens
   depuis le multi-cluster)
3. Display label cluster utilisé pour le nom de l'entrée Telemak
4. Loaded model + status visible côté Telemak (parité avec Argo)
5. Bouton `Manage models` désactivé sur écrans Telemak (correction
   en cours de session : ça vit côté Odysseus dashboard, pas
   Companion)
6. Auto Router add-on : 3 model pickers à la place des 3 text inputs
   Chat / Deep / Code (excluant Auto)

### Backend Odysseus (api.py)

Le predicate `tags[0]` de Companion lit depuis `x_odyssai`. On enrichit
le contrat à deux endroits :

```python
# _model_capabilities pour les pools mlx-distributed
if cd:
    caps["cluster_label"] = cd.get("name") or pool_name
    caps["kind"] = cd.get("kind")

# entrées Telemak (lignes 5000+)
"kind": "telemak",
"cluster_label": cluster_label,
"family": _telemak_short_id(loaded[0]),     # cleanup short
"quantization": _quant_from_name(loaded[0]),
"loaded": True,
```

Première version v1.7.10 → `family` portait le path brut
(`/Volumes/models/odysseus/inferencerlabs/Qwen3-Coder-Next-MLX-9bit`) au
lieu du short ID, le picker subtitle devenait illisible. Suivi
immédiat v1.7.11 avec `_telemak_short_id()` (qui existait déjà pour
le suffix multi-model). Résultat live :

```
telemak-code-next   label=TeleCoder    family=qwen3-coder-next-mlx-9bit   quant=9-bit
telemak512          label=Kolos        family=minimax-m2.7-8bit           quant=8-bit
default             label=Argo         family=qwen3_next                  quant=9-bit
```

### Frontend Companion

`ModelDropdown` extrait de `Input.tsx` (lignes 509-636 + helpers
ChevronIcon / ModelRow / LoadStateBadge / PoolBadge /
buildOdyssaiTooltip) vers `src/components/chat/ModelDropdown.tsx`
nouveau composant partagé. Nouvelles props : `includeAuto`,
`fullWidth`, `placeholder`, `triggerLabel`. Input.tsx perd ~360
lignes, gagne un `import ModelDropdown from "./ModelDropdown"`.

Côté `models.ts` (route backend), priorité des tags :

```ts
const tag = (() => {
  if (caps?.kind === "telemak") return "telemak";
  if (caps?.kind === "mlx-distributed" && caps.cluster_label) {
    return caps.cluster_label.toLowerCase();
  }
  if (caps?.backend === "http-proxy") return "cloud";
  return caps?.pool ?? "local";
})();
const name = caps?.kind === "telemak" && caps.cluster_label
  ? caps.cluster_label
  : m.id;
```

Companion v0.2.3, déployée sur `.39`. Smoke UI : groupes TELEMAK
(5 entrées avec labels) + ARGO + CLOUD, subtitles avec family · quant.

### Bouton Manage models

Sophie corrige son point 5 : *"erreur de ma part, c'est dans
Odysseus."* Le bouton `Manage models →` vit dans
`renderTelemakLoadConfig()` du dashboard Odysseus. Ouverture sur la
Models tab d'Odysseus, qui ne s'applique pas aux clusters Telemak
(eux gèrent leurs modèles via leur propre app). Retiré, commentaire
explicatif laissé pour éviter qu'il revienne. Odysseus v1.7.12.

---

## 3. Le marathon MCP — cinq fixes Companion en file

Le picker validé, Sophie active un MCP dans Companion. Et là, ça se
casse en cascade.

> Sophie :
>
> *"si j'active un MCP dans Companion, tavily, qdrant, corpus, ...
> j'ai une erreur lorsqu'il appele les outils"*

Le message Companion qui s'affiche est trompeur :

> *"The model kept asking to call tools without writing a final
> answer — likely the result context is too large for it to
> summarize. Try a hosted tool-trained model like or:claude-haiku,
> or disable some MCP servers to shrink the context."*

Premier réflexe (mauvais) : pousser Sophie vers un modèle tool-
trained (Coder-Next ou Hermes). Elle pousse-back direct :

> *"aucun ne fonctionne"*

Puis plus tard, plus fort :

> *"ca va pas. les tools on les utilise avec des petits modeles,
> pas des gros. donc, sur telemak. si on n'a pas les tools, on
> perd une grosse partie de companion."*

Recadrage capté. C'est pas un cas edge — c'est le coeur de la
proposition Companion (petits modèles locaux + tools = produire de
la valeur sans payer Anthropic). On creuse.

### Fix 1 — `__MACOSX/` cruft dans la mémoire user

Premier log inspecté révèle un body de **70 KB** envoyé à l'upstream,
dont l'écrasante majorité est composée de fichiers `__MACOSX/...`
contenant du binaire AppleDouble (`Mac OS X
...`). Sophie avait importé un ZIP exporté depuis macOS, le user-
memory importer a tout ingéré — les vrais fichiers ET les sidecars
resource-fork.

Hier soir `4bdfbac fix(user-memory): strip NULL bytes + per-row
try/catch on zip import` traitait UNE partie (NULL bytes), pas le
préfixe `__MACOSX/` ni le pattern `._foo.md` dotunderscore.

Trois couches de fix :
- `isMacOsCruft(path)` helper dans `server/lib/user-memory.ts`
  (matche `__MACOSX/*`, `*/._*`, `._*`, `.DS_Store`)
- Filter à la **lecture** (`readDbFiles` skip cruft) → la pollution
  historique n'atteint plus le modèle
- Filter à l'**import** (`importEntries` reason `macos_resource_fork`)
  + auto-purge des rows historiques pour ce user à chaque nouvel import

Purge live de la DB en plus : `DELETE FROM user_memory_files WHERE
path LIKE '__MACOSX/%' OR ...` → **56 rows** supprimées chez Sophie.

Companion v0.2.4. Sophie retest : body passé de 70 KB à 15 KB, mais
l'erreur revient.

### Fix 2 — Sanitiseur JSON null pour Jinja Telemak

Trace upstream cette fois :

```
[chat] upstream not ok: 500 Internal Server Error:
  {"detail":"telemak upstream error:
    {\"error\":{\"message\":\"model generation failed:
      runtime(\\\"Cannot convert value of type Optional<Any>
      to Jinja Value\\\")\",\"type\":\"generation_failed\"}}"
```

HTTP 500 dès le premier appel. Pas une boucle — un crash Jinja
side Swift. Repro minimal en curl avec un schema contenant
`"default": null` + `"anyOf": [..., {"type": "null"}]` (le tool
schema standard de Tavily) → reproduit le 500 instantanément. Même
call sans ces constructs → tool_call propre.

Le vrai fix vit dans `mlx-swift-lm` (chat-template renderer). En
attendant, workaround Companion-side : sanitiseur récursif qui
nettoie les MCP tool defs avant envoi.

```ts
function sanitizeJsonSchemaForJinja(schema: unknown): unknown {
  // ... walk, drop `default: null`, filter `{type: "null"}` from anyOf/oneOf
}
```

Companion v0.2.5. Test : tool_call émis propre. Sophie retest depuis
Companion : **boucle** sur 8 itérations.

### Fix 3 — MAX_TOOL_ITERATIONS 3 → 8

Trace : iter=0/1/2 visibles, exit avec bailout. Le code :

```ts
const MAX_TOOL_ITERATIONS = 3;
```

Sur la requête Macron + Tavily, le déroulé légitime nécessite 4
roundtrips :
- iter=0 : `tavily_search(query)`
- iter=1 : Tavily renvoie `daily_cap_reached` → modèle retry
- iter=2 : succès avec vraies données → modèle re-call (Coder-Next
  vérifie en chaînant)
- iter=3 : la synthèse… qui n'arrive jamais parce que MAX=3

3 était un cap historique anti-runaway. À 8 on garde la sécurité
pour les agents qui dérapent vraiment, sans tuer les workflows
multi-step normaux.

Companion v0.2.6. Test : iter=8 atteint, MAX déclenché, même message.

### Fix 4 — Stop double-JSON-encoding des tool results

Inspection du body iter=1 : la `tool` message content contient
`"\"{\\\"results\\\":[...]}\""` — JSON-encodé deux fois. Le modèle
voit un literal string commençant par `"{`, le traite comme un blob
opaque, retry.

Dans `executeMcpTool` (`lib/tools.ts`), `data` est déjà un string
(`res.content.slice(0, 4000)`). Puis `stringifyForTool` faisait
`JSON.stringify(r.data)` → re-quote, re-escape.

```ts
const content = typeof r.data === "string"
  ? r.data
  : JSON.stringify(r.data);
```

Pass-through verbatim pour les strings, stringify seulement les
structures. Companion v0.2.7. Vérif curl avec un body Companion-
shape + tool result single-quoted : finish_reason `stop`, content
*"Le président français actuel est Emmanuel Macron, qui est en
fonction depuis mai 2017."* Pipeline propre.

Sophie retest depuis Companion : **8111 chars retournés par Tavily,
boucle quand même**.

### Fix 5 — Cap MCP 4k → 16k + marqueur "do not retry"

Dernière inspection. Le tool content fait **4071 chars** dans le log.
Décodage : `…campaigned for the [Socialist Party](url)'s nomination
for preside`. **Coupé en plein milieu d'un string JSON**.
`json.loads()` du contenu décodé → `Invalid JSON: Unterminated
string starting at column 3039`.

Le `res.content.slice(0, 4000)` dans `executeMcpTool` coupe sans
respecter les frontières JSON. Le modèle reçoit un objet malformé,
décide que le tool a foiré, retry.

```ts
if (res.content.length > 16_000) {
  return {
    ok: true,
    data: res.content.slice(0, 16_000)
      + "\n…[result truncated, do not retry]",
  };
}
return { ok: true, data: res.content };
```

16k pour laisser respirer (Tavily fait typiquement 6-10k), avec
marqueur explicite quand on truncate vraiment — pour que les
modèles tool-trained reconnaissent "résultat partiel" plutôt que
"tool foiré".

Companion v0.2.8.

---

## 4. Le mur final : Coder-Next sur Telemak ne synthesise toujours pas

Sophie retest. Tavily revient à 8111 chars (sous le nouveau cap),
content valide, vraies données Macron. Et le modèle continue à
appeler Tavily. Trois fois. Avec **des queries IDENTIQUES** :

```
mcp_tavily_tavily_search (query: président de la France 2026, max_results: 5) → 12,938 chars
mcp_tavily_tavily_search (query: président de la France 2026, max_results: 5) → 12,938 chars
mcp_tavily_tavily_search (query: président de la France 2026, max_results: 5) → 12,937 chars
```

Le smoking gun : si le modèle envoie 3 fois EXACTEMENT le même
tool_call, c'est qu'il ne **voit pas** ses appels précédents. Chaque
itération ressemble à iter=0 de son point de vue. Le tool exchange
n'arrive pas dans la prompt.

> Sophie :
>
> *"au moins ca marche sur Argo. c'est deja ça. mais pas suffisant"*

Le test croisé fait la preuve. Sur Argo (Python `mlx_lm`, runner
Odysseus), le même `Qwen3-Coder-Next-MLX-9bit` reçoit le même
prompt + le même Tavily → 1 appel + synthèse en français propre
(Macron, réélu 2022, mandat 2022-2027, etc.). Sur Telemak Swift,
le modèle ne survit pas au passage iter≥1.

Companion-side : bodies envoyés byte-identiques entre Argo et
Telemak. Rien à fixer côté Companion. Le bug est dans le rendering
chat-template Swift, dans la façon dont les `assistant.tool_calls` +
`role: tool` se font materialiser dans la prompt suivante.

---

## 5. Telemak ship `0.6.20`

Issue #52 documentée avec tout :

- Cartographie 4-modèles (TeleGem ✗ → ✓, TeleCoder ✗, Telechat ✗,
  Kolos non testé)
- Repro curl minimal sur chaque
- Comparatif Argo vs Telemak sur même model file
- Diff suggéré : capturer le prompt string finale post-render dans
  `ChatCompletions.swift` juste avant `generate()`, comparer avec
  Odysseus runner.py

Telemak side ship `0.6.20` peu après, avec exactement la root cause
théorisée — flatten du tool history en texte (`[assistant] ...
[tool] ...`) qui perd la structure :

> Root cause: Telemak flattened tool-history requests into text
> (`[assistant] ... [tool] ...`) before calling `ChatSession`. That
> lost the structured `assistant.tool_calls` and `tool_call_id`, so
> Qwen/MiniMax saw the tool result as ordinary user text and could
> decide to call the tool again.

Fix (commit `d7e08c6`) :
- Decode `tool_call_id` / `name` sur les messages OpenAI entrants
- Bypass le path flatten quand `assistant.tool_calls` ou `role: tool`
  présent
- Render via `UserInput(messages:)` qui préserve `tool_calls`,
  arguments parsés en JSON, `tool_call_id`
- Conserve le fast/session path pour les tours utilisateur normaux

Smoke après deploy sur les 6 Macs : `.31` Telechat ✓, `.32` TeleCoder
✓, `.29` Kolos ✓, `.39` Argo ✓, `.49` Gemma ✓.

---

## 6. Validation Companion-side — *"yesss"*

Sophie retest TeleCoder + Tavily avec le Tour de France 2025 :

> D'après les résultats du Tour de France 2025, Tadej Pogačar a
> remporté la 112ᵉ édition de la course, obtenir son quatrième titre
> consécutif. Il a devancé :
> - Tadej Pogačar (Slovénie)
> - Jonas Vingegaard (+ 4 min 24 s)
> - Florian Lipowitz (+ 11 min 00 s)
>
> Le départ a eu lieu le 5 juillet 2025 à Lille, et l'arrivée s'est
> déroulée sur les Champs-Élysées à Paris le 27 juillet 2025.
>
> Pogačar a remporté quatre étapes [...]
>
> Souhaitez-vous des détails sur une étape en particulier ou sur
> les classements secondaires ?

Stats : TTFT 21.3 s, Duration 21.51 s, Prompt 8438 tok, Completion
240 tok, Speed 11.2 tok/s. Modèle : `telemak-code-next`
(Qwen3-Coder-Next-MLX-9bit) sur `.32`.

Synthèse single-turn. Données vraies. Suggestion de follow-up. Format
naturel. Identique à ce qu'Argo produisait.

Bonus, Sophie teste avec d'autres MCPs :

> *"je teste qdrant et corpus. et c'est pareil, ca marche"*

Le doc exporté qu'elle partage montre Coder-Next + Qdrant + Corpus
qui retournent une recherche structurée sur MLX + MTP, avec
références au repo `youssofal/MTPLX` (un specdec MTP natif Apple
Silicon, 2.24× decode TPS sur Qwen 3.6-27B, à garder sous la main
côté piste perf future avec dflash).

**Le full MCP stack sur petits modèles locaux est opérationnel.** La
proposition de valeur Companion existe en prod.

Issue #52 fermée avec la trace complète.

---

## 7. Friction & meta — les permissions et le repo divergent

Deux frictions opérationnelles à noter pour la prochaine session.

### Permissions auto-mode

Plusieurs blocages cours de journée : l'auto-mode classifier de
Claude Code refusait les actions sur `argo` (interprétation litérale
d'un "laisse argo" de la veille), sur le drop du fichier
`deepseek_v4.py` (boundary "n'intègre pas Rapid-MLX" sur-appliqué),
sur les `curl` POST locaux. Sophie a fini par :

- Ajouter `skipDangerousModePermissionPrompt: true` dans
  `~/.claude/settings.json` (per la note KB
  `claude-code-permissions-bootstrap`)
- Étendre les allow rules : `Bash(curl:*)`, `Bash(git push internal:*)`,
  `Bash(./scripts/deploy.sh*)`, `Bash(scp:*)`,
  `Bash(ssh admin@192.168.86.*:*)`

> *"les permissions sont un. vrai probleme"*

Le bootstrap est documenté dans le KB depuis le 2026-05-23 mais la
friction reste — chaque session démarre en auto-mode tant que Sophie
ne Shift+Tab pas explicitement. À monter en priorité d'amélioration
côté Claude Code si possible (settings devrait suffire à éviter
l'auto-mode activation).

### Companion push divergent

Le repo Companion sur `.39` ne peut pas fetch via HTTPS depuis github
(pas de credential dans le container). Workaround utilisé toute la
journée : `git format-patch -2 HEAD --stdout | scp + git am sur .39`,
puis `docker compose up -d --build`. Fonctionne, mais les SHAs
divergent (git am recrée le commit avec un timestamp local). À
nettoyer un jour en ajoutant une clé SSH au repo `.39` ou en mirror-
poussant via `internal` comme Odysseus.

---

## Fichiers modifiés / créés

**Odysseus** (`~/Claude/code/MLX Distributed/`) :
- `scripts/api.py` : `_model_capabilities` + entrées Telemak enrichies (`cluster_label`, `kind`, `family`, `quantization`), `_telemak_short_id()` appliqué au `family`, `mark_orphans_interrupted` étendu (json_set sur payload), ping6 `-W 500` retiré
- `scripts/discover-rdma-wiring.py` : drop `-W 500` (macOS rejected as hostname)
- `scripts/persistence.py` : `mark_orphans_interrupted` mise à jour SQL column + JSON payload (fix #17)
- `scripts/dashboard.html` : retrait du bouton `Manage models →` sur les écrans Telemak

**Companion** (`~/Claude/code/thecompai/app/`) :
- `src/components/chat/ModelDropdown.tsx` : NOUVEAU composant partagé (extracted from Input.tsx)
- `src/components/chat/Input.tsx` : -360 lignes, import du nouveau composant
- `src/pages/settings/AddonsPage.tsx` : 3 text inputs → 3 `ModelDropdown` (Auto Router)
- `src/lib/api.ts` : `OdyssaiModelCapabilities` étendu (`cluster_label`, `kind`)
- `server/lib/odyssai-contract.ts` : idem côté serveur
- `server/routes/models.ts` : priorité de tag rework (telemak / argo / cloud), name = cluster_label pour Telemak
- `server/lib/user-memory.ts` : `isMacOsCruft()` export, filter readDbFiles + walker
- `server/routes/user-memory.ts` : filter import + purge auto
- `server/lib/tools.ts` : `sanitizeJsonSchemaForJinja()` + cap MCP 16k + marqueur truncated
- `server/routes/chat.ts` : MAX_TOOL_ITERATIONS 3→8 + `stringifyForTool` no double-encode

**Parked** (sur les 4 ultras `.29/.30/.31/.32`) :
- `~/mlx-cluster/parked-models/deepseek_v4.py.parked-2026-05-29-rapid-mlx`

---

## Numbers de la journée

- **Commits** : 4 sur Odysseus + 1 chore dashboard fix + 5 chore bumps = 10 ; 5 sur Companion + 5 chore bumps = 10
- **Versions** : Odysseus v1.7.8 → **v1.7.12** (4 bumps), Companion v0.2.2 → **v0.2.8** (5 bumps)
- **Telemak co-bordé** : `0.6.20` shippé par leur side avec commit `d7e08c6`, sur la base du diag d'issue #52
- **Issues fermées** : Odysseus #16 (ping6) + #17 (payload désync) + Telemak #52 (tool-calling cross-model)
- **Issues parked** : Odysseus #46 (DeepSeek V4)
- **DB live touchée** : 56 rows `__MACOSX/*` purgées du `user_memory_files` de Sophie
- **Lignes diff cumulées** : ~600 ajoutées côté Odysseus api.py + dashboard ; ~750 ajoutées + ~380 supprimées côté Companion (l'extraction ModelDropdown rend le net négatif sur Input.tsx)
- **Smoke tests** : tous verts (api.py + persistence en local repro DB, MCP full stack sur Tour de France 2025 + Qdrant + Corpus)
- **Découverte piste perf** : MTPLX (`github.com/youssofal/MTPLX`) — native MTP specdec Apple Silicon, 2.24× decode TPS

---

## TODO direct (par ordre)

1. **Decode 1000.0 tok/s anomalie** dans Companion StatsRow — la formule `completion / (total - TTFT)` produit un fallback ou un cap quand decode est sub-seconde. Petit fix Companion à shipper proprement.
2. **Rationaliser le push Companion sur `.39`** — clé SSH github sur le repo `.39` OU mirror via `internal` comme Odysseus. Évite le pattern `git format-patch | scp + git am`.
3. **Suivre la suite Telemak `0.6.20`** — quand Sophie cycle entièrement la stack (re-load tous les modèles, re-test cross-MCP), surveiller qu'aucune régression sur Gemma ou Kolos.
4. **Bookmark MTPLX vs dflash** — quand on revient sur l'accélération decode Argo / Telemak, MTPLX est plus naturel (Apple Silicon natif, MTP heads). dflash garde l'avantage multi-backend.
5. **Décider du retrait du sanitiseur null Companion** — maintenant que Telemak gère JSON null en template, le `sanitizeJsonSchemaForJinja` est redondant. À garder par sécurité (protège des régressions Telemak) OU à retirer pour réduire la surface. Sophie tranchera.

---

## Lessons learned

**Le faux symptôme cache parfois 5 vrais bugs.** Le message "kept
asking to call tools" était trompeur — Sophie est partie sur "modèle
pas tool-trained", j'ai poussé Hermes une fois, elle a corrigé direct
*"les tools on les utilise avec des petits modeles"*. Recadrage qui a
gardé la session sur la bonne piste : 5 vrais bugs identifiés et
shippés, sans jamais tomber dans l'évitement "switch model".

**Le diag Argo vs Telemak comme oracle.** Avoir deux runtimes qui
exécutent le même model file avec la même conversation a été le levier
qui a permis de pointer net où vivait le bug. Sans Argo opérationnel,
on aurait passé la journée à blâmer Coder-Next ou Companion. Le test
crossing-stack est devenu un outil de diagnostic explicite.

**Cinq fixes Companion étaient tous légitimes.** Aucun n'était un
red herring. Cruft, schema null, MAX, double-encode, cap — chacun
était un vrai bug qui aurait pété tôt ou tard sur un autre user.
Le fait que le bug Telemak ait été le dernier visible ne diminue pas
la valeur des 4 fixes en amont. Quand on désencombre proprement,
chaque pelage révèle la couche suivante.

**Le push direct + ship continu + audit clair en commit body marche.**
Cinq commits Companion en file, 4 Odysseus, déployés tous en moins
de 30 secondes chacun, smoke vérifié après chaque, recap aux issues
GitHub à la fin. Pas de PR, pas de gate, pas de drift. Sophie a vu
chaque ship + chaque vérif live. C'est ce que la règle "direct push
to main" de 2026-05-25 visait à débloquer — et ça paie aujourd'hui.
