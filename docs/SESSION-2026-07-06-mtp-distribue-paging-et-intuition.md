# Session 2026-07-06 — MTP distribué : le paging, l'intuition, et le plafond

> Journée-marathon sur un seul objectif : faire décoller le MTP spéculatif en
> serving distribué. Le matin, on prouve l'inconnue #1 (l'alignement multi-rang
> tient — 77 rounds, 0 divergence). L'après-midi vire au polar perf : 20× trop
> lent, trois fausses pistes mesurées puis réfutées. Le tournant n'est pas un
> tok/s — c'est **Sophie qui tranche à l'instinct** : *« c'est mlx, comme nous,
> donc c'est un settings »*. Elle avait raison deux fois : le setting existait
> (`wired_limit`, 48×), ET *« je suis pas certaine qu'on soit en avance sur
> Inferencer »* — le MTP-MoE distribué plafonne à la parité, exactement où
> Inferencer plafonne aussi. On a tout compris, il n'y avait rien à voler.

---

## TL;DR — la journée en 3 axes

| Axe | Livré |
|---|---|
| **Preuve** | Alignement MTP distribué PROUVÉ — 77 rounds GLM-5.2-Q6 3-node, **0 divergence inter-rang**. L'invariant D1/D3 du plan tient : heads répliqués par rang → accept-count identique par construction. |
| **Le polar perf** | 20× trop lent diagnostiqué jusqu'à la racine : ce n'était **pas** la boucle (fausse piste 1), **pas** le stream (fausse piste 2), c'était le **paging des experts MoE**. Fix = `wired_limit` (setting mlx-lm), forward **4237ms → 87ms**. |
| **Le plafond** | Même corrigé, le MTP-MoE distribué reste à **0,68-0,81× l'AR** (parité). Confirmé par les docs RE d'Inferencer (boucle MTP standard) + leur wash MoE au head-to-head. **PARQUÉ.** |

Versions : OdyssAI-X v1.13.2 → **v1.14.0** (intégration MTP, non déployée). Prod
.39 intouchée.

---

## 1. Matin — l'alignement, l'inconnue qu'on prouve

Le plan `docs/PLAN-distributed-mtp.md` (rev.3, 3 rounds MiniMax APPROVED la
veille) verrouillait un pari : les heads MTP **répliqués sur chaque rang**
draftent à l'identique, et le verify greedy déterministe rend l'accept-count
identique par construction → **pas de desync, sans nouveau collectif**. C'était
l'inconnue #1 du chantier.

E1 d'abord : `scripts/mtp_module.py` (NativeMTPModule, famille deepseek/GLM —
enorm/hnorm/eh_proj + `DeepseekV32DecoderLayer`, embed/lm_head partagés du
trunk) + `scripts/mtp_spec.py` (`native_mtp_stream_generate`, invariants
bonus/trim/seed gravés). **11/11 tests unitaires** (parité par position vs AR
greedy D1-3, oracle 100%-accept, accepts partiels, warm-resume). Commits
`e468451`, `473d2e2`.

Puis l'intégration engine : knob pool `mtp:{enabled,depth}`, activation
par-requête dans le JSONL fan-out, `CanaryAggregator` (barrier par round,
auto-disable sur trip), badge dashboard, bump **v1.14.0**. Commit `4d337c6`.

E3 sur GLM-5.2-Q6 3-node ring : **le résultat qui compte.** 77 rounds comparés
sur les 3 rangs, **0 divergence** — canaris `(drafted, accepted, sha)` identiques
round pour round. L'alignement est prouvé. La partie novatrice/risquée du
chantier est derrière nous.

Sauf que le tok/s : **0,6**. Contre ~12 en AR. 20× trop lent.

## 2. Le premier STOP — et le « on continue »

Fidèle à la règle du gate (plan G2 : *< 1,1× → cheval mort, rapport*), je stoppe
et j'écris l'evidence (`docs/EVIDENCE-2026-07-06-distributed-mtp-G2.md`) :
alignement PASS, perf FAIL, diagnostic préliminaire = « boucle verify
non-pipelinée ». Reco : parquer ou fixer, ton appel.

Sophie : *« OK, on continue, E3 »*. On creuse.

## 3. Le polar perf — trois fausses pistes, mesurées

C'est ici que la règle de base a payé : **preuve, pas probabilité.** Chaque
hypothèse a été instrumentée puis réfutée, au lieu de patcher à l'aveugle.

- **Fausse piste 1 — la boucle non-pipelinée.** Instrumentation par phase :
  `draft=68ms, verify=4570ms`. Le draft est rapide (le smoke standalone disait
  9,6ms) ; **tout** est dans le verify. Mais mon hypothèse « boucle qui bloque »
  ne colle pas : un verify supposé ~0,08s ne devrait donner que ~0,7×, pas 0,05×.
- **Fausse piste 2 — le multi-token / l'indexeur DSA.** Sweep S=1,2,4,8 : coût
  **CONSTANT ~4,2s quel que soit S**. Donc pas le multi-token. Un forward nu
  d'UN token = 4,2s.
- **Le bisect décisif.** cache-peuplé vs cache-frais : `inner=4237ms` dans les
  deux cas, cache correct (19-26 entrées), `lm_head=7ms`. **Un forward
  distribué nu = 4,2s, alors qu'AR fait le MÊME appel (`model.model(x,cache)`)
  à 0,08s.**

Je bute. Un composant de mon path mange 4,2s qu'AR n'a pas, et je ne le trouve
pas en lisant le code.

## 4. Le tournant — l'intuition de Sophie

Deux interventions qui recadrent tout.

D'abord, sur Inferencer (le concurrent), quand je prétends qu'on est devant :

> Sophie : *« tu dis qu'on est en avance sur lui, je suis pas certaine. »*

Le hook intuition sonne — et elle a raison de douter. Mon « avance » reposait
sur un head-to-head single-node où les deux avaient un MTP-wash. Ça ne testait
ni le distribué ni un MTP qui marche. Je retire la prétention.

Puis, le coup de génie :

> Sophie : *« je suis aussi convaincue que c'est dans le code pur. c'est mlx,
> comme nous, donc, c'est un settings. je sais qu'ils n'ont pas réinventé le mtp
> sur mlx, il a juste compris comment faire. »*

C'est mlx-lm, le même que nous. Si AR est rapide et mon loop lent sur le **même
appel**, le delta est un **setting**. Je lis le setup de `stream_generate` :

```python
# mlx_lm/generate.py — le wired_limit qui enveloppe TOUTE génération
old_limit = mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
```

**Voilà.** mlx-lm épingle le modèle au *max recommended working set* du GPU → les
poids restent résidents. Sur un MoE 768-experts, sans ça, **les experts
(différents par token) se pagent depuis le disque** — 4,2s de paging par forward.
AR l'avait (via stream_generate) ; mon loop custom, non. **Tout l'écart des 20×
tenait dans une ligne.**

## 5. Le fix, l'OOM, et les 4 nodes

`wired_limit` ajouté au loop. Premier essai 3-node : **Metal OOM sur .31** — le
node est up, c'est le runner qui meurt. Sophie le voit avant moi :

> Sophie : *« 256b a laché, je vois »*

Cause : shard trunk 187GB + module MTP répliqué 8GB = 195GB, et `wired_limit`
tente d'épingler tout ça sur 256GB → dépassement. **Serré des deux côtés** : sans
wired_limit ça pagine, avec ça OOM. Pas de headroom sur 3 nodes.

> Sophie : *« essaie avec 4 nodes, prend .29 dedans, ca laisse plus de marge. GO »*

4-node (.29 ultra-512 inclus), shard ~140GB/rang, ~108GB de marge. Le bisect :

```
POPULATED S=1: inner=87ms   (était 4237ms)   → 48× plus rapide
FRESH S=1:     inner=138ms  (était 4786ms)
```

**Le mystère est résolu, et c'était bien un setting.** Zéro OOM (room pour
épingler), cache_entries=19 (78/4). Son intuition, validée noir sur blanc.

Fixes racine posés au passage : sidecar pré-quantizé Q6
(`build_prequantized_sidecar`, module-q6.safetensors 8GB) pour éviter le pic bf16
du `mx.stack` des 768 experts (l'OOM du chargement) ; crash cache draft
préexistant ; deadlock #23 du driver (stdout rang>0 non-DEVNULL). Commit
`547f8de`, `ec74602`.

## 6. Inferencer — l'apprentissage, pas le piratage

Sophie m'autorise explicitement à comprendre comment Inferencer fait :

> Sophie : *« je t'autorise à retroengeneerer l'app. le but n'est pas de reprendre
> le code mais de comprendre comment ils procèdent. c'est pas du piratage, c'est
> de l'apprentissage. »*

Ligne tenue : inspection de surface + comportement OUI, décompiler le blob mypyc
NON (règle de base + risque IP + de toute façon inutile). Le bundle : coquille
Swift (mlx-swift) + moteur Python mypyc obfusqué dans le Helper, build MLX custom
0.31.2 dual v15/v26 avec `libjaccl` embarqué. Sophie corrige : *« inferencer est
en ring pas jaccl. ce qui nous intéresse c'est le MTP. »*

Puis elle partage des docs RE (des reconstructions, pas le binaire) qui
**tranchent** : leur MTP est la **boucle standard** — `generate_mtp_logits` +
`SpeculativeTokenIterator` = draft heads natifs → verify → accept → trim,
**structurellement identique à mon `mtp_spec.py`.** Zéro sauce secrète. Et détail
clé : ils tournent `num_nextn_predict_layers=1`.

## 7. Le plafond — l'autre intuition qui gagne

Ce détail (1 couche nextn) explique notre dernière erreur : on draftait à **D=3**,
forçant une head à 1 couche à prédire t+2, t+3 récursivement, hors de sa zone →
acceptance qui s'effondre. Test **D=1** (avec baseline AR 4-node cette fois) :

```
AR 4-node : ~10 tok/s     MTP D=1 : 5,6-8,8 tok/s     speedup : 0,68-0,81×
```

Mieux que D=3 (ratio 0,75 vs 0,6 — la depth était bien un setting mal réglé),
mais **toujours une perte.** Le verify multi-token sur MoE coûte le gather
d'experts supplémentaire, qui bouffe le gain d'acceptance. Plafond théorique
~1,3× *seulement si* on tue aussi l'overhead Python par round.

Et la deuxième intuition de Sophie se referme : le head-to-head le disait déjà —
**Inferencer lui-même fait un wash MoE.** La parité EST le plafond honnête du
MTP-MoE distribué. On n'était pas « en retard » sur une sauce secrète ; il n'y a
pas de sauce. On n'était pas « en avance » non plus. On est au même mur qu'eux.

> Sophie : *« mets tout ca en FJ, fait un handoff... on va faire une pause. ca
> suffit »*

On parque. Proprement.

## Fichiers modifiés / créés

- `scripts/mtp_spec.py` — `native_mtp_stream_generate` + le fix `wired_limit` +
  `generation_stream` + instrumentation timing/sweep/bisect.
- `scripts/mtp_module.py` — NativeMTPModule + `build_prequantized_sidecar`
  (fast-path Q6, fix OOM) + rewrite kv_b MLA.
- `scripts/e0_driver.py`, `scripts/e3_driver.py` — harnais G0/G2, flags
  `--depth`/`--quantize`/`--timing`.
- `scripts/sidecar_fetch.py` — récup range-HTTP des poids MTP strippés (19,9 GB).
- `docs/EVIDENCE-2026-07-06-distributed-mtp-G2.md`,
  `docs/HANDOFF-2026-07-06-mtp-distribue-parked.md`.

## Numbers de la journée

- **Commits** : 6 sur OdyssAI-X (`e468451`, `473d2e2`, `8985eaf`, `4d337c6`,
  `547f8de`, `ec74602`), poussés **FJ (forge) + GitHub**.
- **Version** : v1.13.2 → **v1.14.0** (intégration MTP ; non déployée, MTP off).
- **Le chiffre de la journée** : `inner()` distribué **4237ms → 87ms (48×)** via
  `wired_limit`.
- **Alignement** : 77 rounds, 0 divergence inter-rang.
- **MTP D=1** : 0,68-0,81× l'AR (parité). Acceptance ~0,55-0,7.
- **Prod .39** : intouchée (v1.13.2).

## TODO direct (par ordre)

1. **LongCat-2.0** — port + conversion Q4 head-bf16 (download HFDownload.app en
   cours sur .29). Le vrai chantier suivant.
2. **MTP, si reprise** (rendement décroissant assumé) : fix bug greedy
   `MTP text != AR text` + async loop pour viser ~1,3×.
3. **Promouvoir `wired_limit` au wiki** — setting durable : tout loop de
   génération MLX custom doit s'y envelopper, sinon paging → ~50× lent.
4. **#28** split api.py god-module (backlog).

## Lessons learned

- **`wired_limit` est LE setting perf des loops custom.** `mx.set_wired_limit(
  max_recommended_working_set_size)` — sans lui, les poids (surtout MoE) se
  pagent depuis le disque. AR l'a gratuitement via stream_generate ; tout loop
  maison DOIT l'ajouter. Le finding le plus réutilisable de la journée.
- **L'intuition de Sophie est un fait, pas une opinion.** Deux fois de suite :
  « c'est un settings » (→ wired_limit, 48×) et « je suis pas certaine qu'on soit
  en avance » (→ parité = plafond MoE). Quand elle sent quelque chose, c'est du
  signal réel que mon contexte étroit n'a pas encore trouvé.
- **Preuve, pas probabilité.** Le mur perf a coûté trois hypothèses — mais chacune
  a été *mesurée* (timing par phase, sweep S, bisect cache) et réfutée, jamais
  patchée à l'aveugle. C'est ce qui a fini par isoler le paging.
- **Le cheval mort n'est pas toujours un bug.** Ici l'approche MARCHE (alignement
  prouvé, perf débloquée) — c'est le PLAFOND du domaine (MTP-MoE distribué) qui
  est la parité. Savoir s'arrêter à un plafond structurel, pas seulement à un bug.
- **Apprendre ≠ pirater.** On a compris comment Inferencer procède (boucle
  standard + wired_limit + kernels Swift) sans toucher leur blob obfusqué — par
  le comportement, le bundle lisible, et des reconstructions. Il n'y avait rien
  à voler : la vitesse est publique (un setting mlx-lm), l'algo est standard.
