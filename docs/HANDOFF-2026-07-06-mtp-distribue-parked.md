# HANDOFF — 2026-07-06 — MTP distribué : PARQUÉ (ceiling = parité)

> Chantier MTP distribué (plan `docs/PLAN-distributed-mtp.md`). Résultat net :
> **alignement prouvé, perf plafonne à la parité** pour du MTP-MoE distribué —
> le même plafond qu'Inferencer (wash MoE). Parqué sur décision Sophie.
> Evidence détaillée matin : `docs/EVIDENCE-2026-07-06-distributed-mtp-G2.md`.

## État actuel (vérifié)

**Ce qui MARCHE et est banké :**
- **E1** : `mtp_module.py` + `mtp_spec.py`, 11/11 tests unitaires. Committé.
- **Alignement distribué PROUVÉ** : 77 rounds GLM-5.2-Q6 3-node, 0 divergence
  inter-rang (canaris identiques). L'invariant D1/D3 du plan tient.
- **Le GROS finding — `wired_limit` = un SETTING, 48×** : un forward `inner()`
  distribué passait de **4237ms → 87ms** en enveloppant la génération dans
  `mlx_lm.generate.wired_limit(model, [generation_stream])`
  (= `mx.set_wired_limit(device_info["max_recommended_working_set_size"])`).
  Sans lui, les experts MoE (différents par token) se pagent depuis le disque.
  AR l'avait déjà (via stream_generate) ; mon loop custom MTP ne l'avait pas —
  c'était TOUT l'écart des 20×. **Intuition Sophie confirmée : c'était un
  settings dans le MLX pur, pas un algo.**
- **Sidecar pré-quantizé Q6** (`build_prequantized_sidecar`, module-q6.safetensors
  8GB) sur .30/.31/.32 (+.29) : évite le pic bf16 `mx.stack` 768-experts (OOM).
- Intégration engine (knob pool `mtp`, CanaryAggregator, badge, v1.14.0) —
  committée, **PAS déployée** (.39 = v1.13.2 intouchée).

**Ce qui PLAFONNE (le verdict) :**
- Perf MTP distribué GLM-5.2-Q6 4-node : **0,68-0,81× l'AR** (D=1), **0,05× puis
  ~parité** selon la config. Jamais un win franc.
- **Cause structurelle, PAS un setting manquant** : le verify multi-token sur MoE
  coûte le gather d'experts supplémentaire, qui bouffe le gain d'acceptance.
  Théorique max ~1,3× *seulement si* on tue aussi l'overhead Python par round.
- **Confirmé par 2 sources indépendantes** : (a) les docs RE d'Inferencer
  (`/Volumes/models/workplace/Opencode/inferencer/inferencer-reverse/`) montrent
  que leur MTP est la **boucle standard** (draft heads natifs → verify → accept
  → trim) — zéro sauce secrète ; (b) le head-to-head 2026-06-21 mesure leur
  MTP en **wash sur MoE** (MiniMax-M3). **La parité EST le plafond honnête du
  MTP-MoE distribué.**

**Bug ouvert (si reprise) :** `MTP text != AR text` sur le vrai modèle (le
greedy-exact du toy-test ne tient pas distribué). À corriger avant toute mesure
propre — mais le corriger ne rend RIEN plus rapide.

## Décisions gelées (NE PAS rouvrir)
- **`wired_limit` est la découverte durable** : tout loop de génération custom
  (MTP, futur) DOIT envelopper dans `wired_limit` sinon paging → 50× lent.
- MTP off par défaut en prod. .39 intouchée.
- Alignement distribué prouvé — ne pas re-tester.
- Base LongCat distribué AVANT tout binding MTP LongCat.

## Écarté (testé, NE PAS re-proposer)
- **Chercher une sauce secrète dans le blob mypyc d'Inferencer** : inutile, leur
  MTP EST la boucle standard (confirmé par les docs RE). Ne PAS décompiler.
- **Draft-model séparé multi-rang** : spec mlx-lm 0.31.3 WEDGE en TP-ring.
- **Sidecar bf16 chargé par rang** : OOM (`mx.stack` 768 experts) → sidecar Q6.
- **3-node GLM-5.2-Q6 + wired_limit** : OOM (195GB shard sur 256GB, pas de room
  pour épingler). 4-node (.29 dedans) = room, pas d'OOM.
- **D=3 sur une head à 1 couche nextn** : sur-draft, acceptance s'effondre d2/d3.
  D=1 est le bon match (mais toujours perte).

## Ouvert / next (SI reprise — rendement décroissant assumé)
1. Fix le bug greedy `MTP text != AR text` (verify/accept ou seed_hidden).
2. Tuer l'overhead Python par round (les `.item()` bloquants → async_eval,
   batch argmax) — seul chemin vers le ~1,3× théorique. Rendement décroissant.
3. Sinon : **capitaliser** — `wired_limit` bénéficie à tout loop custom futur ;
   l'infra MTP + la méthode canaris servent au port LongCat.

## Invariants
- Alignement collectifs = sacré (prouvé). Perf = le juge. Prod .39 intouchée.
- Gotcha : `pgrep -f "runner.py"` (PAS "mtp-e3.*runner.py" — cwd absent de la
  cmdline). Orchestrer en ssh tenu, jamais nohup.
- LongCat en download HF sur .29 (relancer si reboot .29).

## À promouvoir (post-session)
- → wiki LLM (DURABLE, haute valeur) : **« Tout loop de génération MLX custom
  doit s'envelopper dans `wired_limit`, sinon les poids (surtout MoE) se pagent
  → ~50× lent. C'est LE setting perf. »** + « MTP-MoE distribué plafonne à la
  parité (verify multi-token coûte le gather experts) — Inferencer aussi. »
- Mémoire `inferencer_architecture` : MAJ faite (mlx-swift, ring, mtp standard).
- Mémoire `mtp_native_speculative` : à MAJ avec le verdict G2 (parité = plafond
  MoE) + le finding wired_limit.

## Suggested skills (reprise)
- `session-doc` pour clore la journée (arc riche : paging saga + intuition Sophie
  + RE Inferencer + verdict cheval-mort).
