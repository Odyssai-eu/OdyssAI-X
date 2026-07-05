# Review log — PLAN-distributed-mtp.md

Review adversariale cross-model (MiniMax-M3, agentic read-only sur le repo,
harnais `run_review.py`, MAX_TOOL_CALLS 25). Claude arbitre final.

## Round 1 — MiniMax : VERDICT REVISE (~15 findings matériels)

### Acceptés → intégrés au plan (rev. 2)

| # | Finding | Intégration |
|---|---|---|
| F-1 | Seed hidden inter-rounds non spécifié | Invariant explicite E1 : seed = `verify_hidden[n]`, n = dernière position acceptée (bonus = index 0), assert debug |
| F-2 | Hook « model.model + norm/head à la main » non nommé | Mécanisme unique : wrapper de capture sur `model.model.norm` (input = pré-norm), toutes familles, flag D8 choisit input/output |
| F-3 | Arithmétique offset × session-cache silencieusement fragile | Assert fin de gen toujours ON : `cache.offset == prompt_len + tokens_émis` + test 3 tours (R6) |
| F-5 | Resume sur session-cache chaud : pas de hidden stocké | Convention « toujours laisser 1 token à forwarder » : trim à prompt_len-1 + re-forward du dernier token → (hidden, bonus) |
| F-6/F-12 | E0 direct-cluster trop cher ; paire Qwen non vérifiée | Échelle graduée : localhost ring 2-proc .29 d'abord (pattern GATE-0 VLM) ; inventaire disque + sha tokenizer identique exigé |
| F-8 | Agrégateur canaris : faux positifs de cadence | Comparaison keyée par round, barrier « tous les rangs ont émis le round i », timeout = warning pas trip |
| F-9/F-27 | `_stderr_tail` = tail-buffer, pas de parseur | `CanaryAggregator` = deliverable E3 explicite |
| F-10/F-23 | Auto-disable via env impossible (lu au start) | D7 réécrit : on/off + depth PAR REQUÊTE dans le JSONL fan-out ; auto-disable = l'engine cesse d'envoyer `mtp.on` |
| F-11 | Échec G1 ne bloquait que mollement | G1 = bloquant dur, 2 tentatives de fix max, « ne pas aller en E3 pour voir » |
| F-15 | D8 résoluble dès E2 sur GLM (gratuit) | E2 étape 3 : A/B pre/post-norm sur GLM-5.2 ; V2 devient confirmation |
| F-16 | Fallback « buffer Telemak » = chantier caché | R4 durci : trim = exigence du contrat port, pas de promesse de fallback, sinon E4 bloqué |
| F-17 | Sémantique cache mtp implicite | E1 : `draft_step` appende UNE position par appel |
| F-22 | Pas de critère de parcage E4 si le port glisse | Condition d'entrée E4 : gate du port passé ; sinon parqué (gates, pas dates) |
| F-30 | Layout kv_b LongCat ≠ DSv3 possible | Fold dans V2 |
| F-7 | Déterminisme conditionnel, pas « par construction » | Précondition documentée (shapes fixes + ordre de réduction fixe) ; canari = alarme |

### Rejetés — avec preuve

| # | Finding | Réfutation |
|---|---|---|
| F-4/F-18 | « Le bonus est déjà dans le cache ; verify = drafts seuls ; avance cache = +D » | **Faux.** mlx-lm 0.31.3 `generate.py:617-618` : `y = mx.concatenate([y, draft_tokens])` puis `_step(model, model_cache, y, num_draft + 1)` — le verify traite `[bonus, drafts]` = D+1 positions, le K/V du bonus est écrit PAR le verify. L'asymétrie du trim draft (`max(D-n-1,0)`, l.591) n'existe QUE parce que le draft a déjà consommé le bonus, pas le trunc. Le plan collait au source ; l'invariant est désormais explicite dans E1 pour tuer l'ambiguïté. |
| F-20 | « L'activity pill #61 n'existe pas (forward-reference) » | Shippée v1.13.2 (#61 closed, dashboard.html) — invisible au grep naïf car le dashboard exige `grep -a` (fichier à contenu binaire/unicode, gotcha connu). |
| F-28 | Composition du panel-à-3 à redéfinir | Le panel (direct/alternative/sceptique) est la règle de base de Sophie — pas un paramètre du plan. |
| F-13/F-14/F-19/F-21/F-24/F-26/F-29 | Vérifications convergentes sans changement demandé | Confirmations — rien à faire. |

## Round 2 — re-soumission du plan révisé (rev. 2)

MiniMax a relu le plan révisé + l'arbitrage :

- **F-4/F-18 RETIRÉS par le reviewer** : « I asserted [the bonus was
  already in the cache] — that was inferential speculation, not based on
  the cited lines. The user's rejection is sound. My F-4 was wrong.
  I withdraw that finding. » L'invariant explicite reste dans E1.
- **Toutes les intégrations round-1 vérifiées présentes** (F-1→F-30 : ✓
  ligne par ligne dans son rapport).
- **7 nits résiduels remontés**, arbitrage :

| # | Finding | Arbitrage |
|---|---|---|
| F-31 | Test de parité à nommer précisément | ACCEPTÉ — test par POSITION (pas sha final) dans E1.3, D ∈ {1,2,3} |
| F-32 | Aliasing lm_head partagé vs hot-swap | ACCEPTÉ — assert d'aliasing E1.3, hot-swap hors contrat v0 |
| F-33 | Assert d'offset seulement en fin de gen | ACCEPTÉ — assert périodique toutes les 64 positions |
| F-34 | Point d'émission canari E0 à ancrer | ACCEPTÉ — boucle consommatrice `runner.py:2193` |
| F-35 | Warmup absent des protocoles de bench | ACCEPTÉ — warmup ~200 tokens jeté avant CHAQUE mesure (G1/G2/G3) |
| F-36 | Coût du wrapper de capture pré-norm (fusion) | ACCEPTÉ — profil avec/sans en E2, seuil 3 % |
| F-37 | « 1.1 GB Q4 » suppose l'embed quantifié | ACCEPTÉ (le vrai du lot) — §1.3 réécrit : budget SELON la politique embed (Q4 ≈ 1.1 GB, bf16 ≈ 3.2 GB), à FIXER dans la recette de conversion ; sans douleur dans tous les cas |

Verdict formel round 2 : REVISE (« approvable with these nits fixed; not
blocking but should be addressed before E1 work begins »). Les 7 nits sont
appliqués dans **rev. 3** — le plan intègre 100 % des findings survivants.
Arbitre final : Claude (règle du skill). Les décisions D1-D8 n'ont pas
bougé sur les deux rounds ; ce qui a bougé : les invariants sont désormais
ÉCRITS (bonus/trim/seed/resume), le toggle MTP est par-requête (pas env),
l'agrégateur canaris est spécifié (barrier par round) et budgété comme
deliverable, les protocoles de bench ont warmup + parité par position.

**Clôture** : convergence effective. Le panel-à-3 de la règle de base
reste dû AVANT le premier commit de code (session d'exécution) — cette
review cross-model n'en tient pas lieu, elle l'alimente.
