# Session 2026-07-05 — AGPL public release + éval CoeOS

> Journée courte, deux décisions lourdes. Le matin : CoeOS a été benché (la
> veille) et **réussit avec mention** — 97.4 % moyen, #1 du scoreboard local ;
> on consigne le résultat en le cadrant juste (l'edge = coût/local/lisibilité,
> pas un gain qualité brut). L'après-midi : **virage de licence** — toute la
> stack passe **propriétaire → AGPL-3.0** et on **ouvre les repos au public**.
> Un faux-pas au milieu : je ressors une fausse alerte « 121 commits vont
> fuiter » basée sur un ref git périmé de mon poste ; Sophie tique (« avec quoi
> tu viens ? »), je vérifie, je la retire. La leçon du jour n'est pas un tok/s :
> c'est vérifier l'état réel avant d'alerter.

---

## TL;DR — Avant / Après

| Aspect | Avant | Après |
|---|---|---|
| Licence stack OdyssAI | Apache 2.0 (engine) / « planned » (Telemak) / rien (Companion…) | **AGPL-3.0** partout (copyleft fort protège le moat) |
| mlx-swift-lm | MIT (fork ml-explore), non reconnu GitHub | **MIT gardé** (fork ⇒ pas relicençable), reconnu GitHub |
| CoeOS SE | dans l'engine (`api.py`) | **MIT prévu** mais bloqué : à extraire hors de l'AGPL (lié #28) |
| Telemak / odyssai-services | privés GitHub | **publics** (flippés après secret-scan propre) |
| Stratégie release (mémoire) | « privé/propriétaire, PAS open-source » (2026-06-20) | **open-core AGPL** (le modèle commercial services/on-prem inchangé) |
| Éval CoeOS | RFC #63, table déployée mais non mesurée | **mesurée + validée** : 97.4 % #1 local, gate « routing ≠ dégradation » franchi |

Versions : pas de bump runtime (journée docs/licence). OdyssAI-X reste **v1.13.2**.

## Phase 1 — CoeOS : l'éval consignée

Sophie envoie la synthèse du bench (fait la veille, suite TMB générale T01-T07,
100 % local, évaluateur Sonnet 4.6) : **CoeOS routé = 97.4 % moyen T01-T05, #1
du scoreboard**, devant MiniMax-M3 seul (97.2 %) et Qwen3.5-397B (94.4 %) ; 7/7
tests, T03/T04/T07 = 50/50, T05 = 2e parfait après M3.

Le cadrage honnête que je pose sur #63 : le gate de la question ouverte #2
(**routing vs single-model**) est franchi — le routage **ne dégrade pas** la
qualité, il atteint le frontier-local en choisissant le bon modèle par axe. Mais
l'écart +0.2 pt vs M3 seul est faible : ça **confirme la thèse du RFC** (« un seul
gros modèle ferait ~95 % du job, l'edge = coût/local/lisibilité, pas la qualité
brute »). Le bench prouve la bonne chose.

Sophie recadre, et c'est le bon angle :

> Sophie : *"+0.2% c'est déjà bien en local où on est limité sur le panel de
> modèle qu'on lui met à disposition. On avait Minimax et Ornith ensemble
> seulement. Donc c'est très positif pour moi."*

C'est le plancher, pas le plafond : deux cerveaux proches laissent peu de marge
au routage. Les tests **hybride** + **cloud-100 %** à venir élargissent le panel
→ c'est là que l'edge doit s'ouvrir. Consigné sur #63 (`issuecomment-526`) +
mémoire `coeos_smart_router`. Restent : métrique scoring-aveugle (Phase 2),
binding agentic (TMB Agentic bench).

## Phase 2 — Le virage AGPL + public release

Sophie : *« la licence pour tout OdyssAI est AGPL-3.0, sauf CoeOS SE qui sera seul
à être MIT (pas encore sur github) »* + *« on go en public release »*.

**Le flag stratégie** : la mémoire `release_licensing_strategy` (2026-06-20)
disait **« privé/propriétaire, PAS open-source, ne JAMAIS open-sourcer l'engine »**.
AGPL-3.0 EST open-source. Je le signale sans le trancher — c'est cohérent en
réalité : le copyleft fort empêche un concurrent de prendre le moteur en SaaS
fermé, donc **protège le moat tout en étant ouvert**. Open-core : AGPL + services/
support/on-prem. Le modèle commercial ne change pas, seule la licence.

**Appliqué** (texte AGPL-3.0 canonique via `gh api /licenses/agpl-3.0`) :
- **OdyssAI-X** : `LICENSE` Apache 2.0 → AGPL-3.0, README + CONTRIBUTING mis à
  jour, `NOTICE` préserve l'attribution exo (Apache 2.0, one-way compatible
  AGPL). Commit `3e4ab76`, mirror GitHub reconnaît **AGPL-3.0**.
- **Telemak** (`d25bd30`), **odyssai-services** (`e4b6809`), **thecompai-memory**
  (`da2aa2f`), **odyssai-launch** (local, pas de remote) : `LICENSE` AGPL-3.0.
- **Companion** : posé **via le primary FJ** (`a6a379b` sur FJ) — mon checkout
  local avait divergé, donc branche sur le vrai HEAD FJ `64c3706`, commit,
  push forge → le mirror propage vers GitHub (vérifié AGPL-3.0). `package.json`
  license → `AGPL-3.0-or-later`.
- **mlx-swift-lm** : reste **MIT** (fork).
- **CoeOS SE** : rien — vit dans l'engine AGPL, sa MIT exige de l'extraire (#28).

**Public flip** : Telemak + odyssai-services étaient privés. Secret-scan avant
exposition (le filet) : **zéro `.env`** (tree + historique), aucun pattern de clé
(sk-/AKIA/ghp_/PRIVATE KEY/api_key) ; seules des IPs LAN privées RFC1918 (non
routables, sans valeur tierce). Flippés **publics**. État final : OdyssAI-X,
Companion, Telemak, odyssai-services = **public · AGPL-3.0** ; mlx-swift-lm =
public · MIT.

## Le faux-pas — la fausse alerte sur ref périmé

Avant de pousser Companion, j'alerte : « 121 commits vont fuiter en public ».
Sophie coupe :

> Sophie : *"on a déjà réglé tout ça, on a synchro github et FJ. avec quoi tu
> viens ?"*

Je vérifie au lieu de re-affirmer : mon `github/main` **local** pointait sur
`3e407e9` (vieux de semaines), le vrai GitHub était à `64c3706` (tenu par le
mirror FJ→GitHub). Mon « 121 en avance » comparait mon HEAD contre un **tracking
ref périmé de mon poste**, pas la réalité. La synchro + la déperso étaient bien
faites, comme elle disait. Erreur retirée. Corollaire appliqué juste après : la
LICENSE Companion posée via FJ (le primary), pas depuis mon checkout dérivé.

## Numbers de la journée

- **Commits** : 1 sur OdyssAI-X main (`3e4ab76`) + 4 repos frères (Telemak,
  odyssai-services, memory, Companion-via-FJ) + odyssai-launch local.
- **Version** : inchangée (v1.13.2 — journée licence/docs, zéro runtime).
- **Repos publics + AGPL-3.0** : OdyssAI-X, Companion, Telemak, odyssai-services.
- **Secret-scan** : propre (0 `.env`, 0 clé) sur les 2 repos flippés.

## TODO direct (par ordre)

1. **CoeOS SE en MIT** — l'extraire hors de l'engine AGPL (`api.py`) dans son
   repo, prérequis à sa licence permissive. Couplé au split #28.
2. **Tests CoeOS hybride + cloud-100 %** (Sophie) — élargir le panel, ouvrir
   l'edge du routage ; ajouter au fil #63.
3. **#28 split god-module** — seam par seam avec smoke humain (débloque aussi #1).
4. Décider le sort des autres privés (Imager, hf-*, hermes-bridge, OdyRAG,
   voice×2, obsidian-plugin, omnigent-bridge) : public+AGPL ou internes.
5. `thecompai-memory` : remote sur l'ancienne org `thecompai/memory` — migrer
   sous `Odyssai-eu` si voulu.

## Lessons learned

- **Vérifier l'état RÉEL avant d'alerter** : un ref git local périmé m'a fait
  crier à la fuite de 121 commits. Un `git fetch` / une lecture de l'état distant
  AVANT l'alerte aurait évité le faux-pas. Même famille que le « degraded
  périmé » de mercredi.
- **AGPL ≠ renoncer au moat** : le copyleft fort est un choix stratégique, pas
  une capitulation — il empêche la reprise en SaaS fermé. Open-core cohérent avec
  vendre services + on-prem.
- **On ne relicencie pas un fork** : mlx-swift-lm reste MIT (code upstream
  ml-explore) — l'inclure dans un blanket-AGPL aurait été une faute juridique.
- **Poser un fichier sur le bon primary** : le checkout dérivé ne pousse pas ;
  la LICENSE Companion est passée par FJ (primary) → mirror, pas par mon poste.
