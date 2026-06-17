# Plan: Servir Mistral Medium 3.5 128B en local — Telemak Swift single-node (primaire), Odysseus fallback
_Locked via grill-with-docs — Claude + Sophie, 2026-06-16 (rév. 2 après MiniMax round 1 + pivot Telemak). Termes per CONTEXT.md (Serving & vitesse). Tickets: telemak#73 (primaire), OdyssAI-X#54 (fallback)._

## Goal
Servir Mistral 3.5 128B (dense, multilingue, texte) en local par le chemin le plus **robuste** : **Telemak Swift single-node, JACCL-free**. Vitesse interactive — honnêtement **~12-17 tok/s** single-node (Q6 + EAGLE), soit 2,5-3,5× les ~5 tok/s bandwidth-bound d'un Q8. Le **20+** n'est PAS garanti single-node (cf. round 1 ci-dessous) → il passe par le **fallback** Odysseus multi-node TP, qui rouvre JACCL et n'est tiré que si l'usage l'exige.

**Pourquoi Telemak Swift en primaire :** un 128B dense Q6 (~96 GB) tient *large* single-node sur un 512GB Mac — splitter en multi-node (JACCL) pour un gain de bande passante marginal est le mauvais trade (Sophie). Et Telemak = Metal natif, **zéro JACCL/watchdog distribué** : il esquive toute la fragilité du moteur Python qu'on a débogué le 2026-06-16 (head-of-line `6464540`, fuite wired, JACCL #40). **Le modèle est déjà porté** : `Mistral3Text.swift` + `mistral3` registry dans le fork `mlx-swift-lm-odyssai`.

## Approach
1. **WU0 [2] — Charger + parité (Telemak).** Conv MLX Q6 de Mistral 3.5 128B (mlx-community si dispo, sinon convert depuis le raw bf16), charger dans `Mistral3Text` single-node sur un 512GB Mac, FR cohérent, **parité greedy vs Python** sur set fixe, sanitize round-trip.
2. **WU1 [3] — Bench single-node.** tok/s décode Q6 vs plafond bandwidth (~6,7) + vs Python. Confirmer **pas de pénalité Swift** (dense → Swift = Python ; ≠ le cas MoE #72).
3. **WU2 [5] — Spéculatif EAGLE (le levier vitesse).** Brancher le draft EAGLE officiel de Mistral (`mistralai/Mistral-Medium-3.5-128B-EAGLE`, ~1,5 B, 1 couche, format natif vLLM/SGLang) dans Telemak-MTP/MTPLX (têtes denses, 1,71× prouvé). Mesurer l'**acceptance** sur Q6 + la vitesse → cible ~12-17.
4. **WU3 [3] — Sampling + observabilité.** Profil de sampling Mistral côté Telemak (multilingue, **pas de cjk_lock** par défaut) ; **logger acceptance EAGLE + tok/s réel** (round-1 #14, sinon « 20 ou pas » n'a pas de définition mesurable).
5. **Fallback [#54] — Odysseus Python multi-node TP.** Si le single-node+EAGLE plafonne sous le besoin réel : TP **2 ou 4 nodes** (KV=8 ; jamais 3, 8∤3 — round-1 #5/#11), critère d'arrêt ≥ cible ou TP4 épuisé. Rouvre JACCL → tiré sciemment, pas par défaut. Pré-requis Odysseus : câbler `mistral3` dans les listes engine (`api.py:733`, `auto_parallel.py:57` n'ont que `ministral3` — round-1 #2).

## Key decisions & tradeoffs
- **Single-node Swift primaire, multi-node TP fallback** — un 128B qui tient single-node ne se splitte pas pour un gain marginal. → ADR-0004 (à créer).
- **Telemak (JACCL-free) > Odysseus (fragilité distribuée)** pour Mistral — la leçon du 2026-06-16.
- **Q6, pas Q4** — meilleure acceptance EAGLE (draft entraîné sur le full) + qualité. → ADR-0003.
- **20 single-node n'est PAS garanti** (round-1 #6 : 1,71× × 6,7 ≈ 11,6 ; réaliste 12-17). On l'assume : la cible primaire est 12-17 robuste ; 20+ = fallback. Pas de sur-promesse.
- **Multilingue, pas de cjk_lock** par défaut (≠ M3 ; leçon §9.5). Lock armable si leak.
- **M3 (MoE, multimodal)** reste Python pour le texte ; Telemak pour le **VLM** (le seul chemin vers M3 *avec vision*, telemak#72) — hors scope ici.

## Risks / open questions
- **Acceptance EAGLE sur Q6** : inconnue jusqu'à WU2. Basse → EAGLE sous-livre → on s'approche du fallback.
- **Port EAGLE → Telemak-MTP/MTPLX** : le draft est en format natif Mistral (pas HF), pour vLLM/SGLang (CUDA-first → pas Metal) → portage requis. Inconnu technique #1.
- **Le reviewer (MiniMax) ne voit pas le fork Swift** (`mlx-swift-lm-odyssai`/`telemak` = repos séparés) → les claims Swift (Mistral3Text porté, registry) sont **seedés, pas vérifiables par le code** depuis ce repo. (Round-1 #7 « EAGLE inexistant » venait de la même cécité : le draft existe bien sur HF.)
- **Compat conv MLX → `Mistral3Text`** : vérifier que le sanitize Swift avale la conv (dense → plus simple que le cas MoE déjà validé en #71).

## Out of scope
- Vision/multimodal Mistral (mlx-vlm plus tard) ; M3-on-Telemak texte (#72, parqué).
- Provider Companion + badge FR/EU → thecompai/app #27.
- Mistral Large.
