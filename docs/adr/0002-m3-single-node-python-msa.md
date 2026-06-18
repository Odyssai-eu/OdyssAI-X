# 2. MiniMax-M3 servi single-node en Python ; MSA crackée en Python ; Telemak/Swift différé

Date: 2026-06-13.

Status: Accepted

## Context

M3 (428B A23B, `minimax_m3`) tourne aujourd'hui sur Argo en **pipeline 2-node**
(a 2-node Argo pool). Mesures de la nuit : TTFT 12,3 s sur 427 tokens, decode 18,8 tok/s
— et **deux crashs JACCL** dans la session (keepalive queue-pair dégradé +
leak Metal wired). Le port MSA de la classe `minimax_m3` réalise la sélection de
blocs via un **masque additif dense** : fonctionnellement correct mais
quadratique (la sparsité est logique, pas physique). Objectif : vitesse
d'inférence à 128K de contexte (TTFT + decode), 1M plus tard.

Trois faits ont tranché :

1. **M3 tient single-node.** Q6 = 322,8 GB ; the 512GB node budget wired ~460 GB →
   `nodes=1` fits (355 GB requis). Vérifié via `/admin/clusters/.../load-options`.
2. **Single-node est plus rapide pour le flux interactif.** Le pipeline 2-node
   sérialise les nœuds sur un seul flux (bubble + comms inter-nœuds) sans gain de
   débit ; Companion est mono-flux interactif. Single-node supprime le bubble ET
   le JACCL (cause des deux crashs).
3. **Aucun blocage technique ne force Swift.** La classe Python existante tourne
   en solo telle quelle.

Alternatives considérées : (a) garder le 2-node distribué — rejeté (bubble +
instabilité JACCL, sans bénéfice mono-flux) ; (b) porter M3 sur Telemak/Swift
single-node tout de suite — rejeté pour CE crack (port complet de l'archi M3 en
Swift = multi-sessions ; et Swift est déjà rapide — c'est Python qui a besoin
d'aide, donc les ressources MTP/perf vont au Python distribué, pas au
MTP-mlx-swift).

## Decision

- **M3 est servi `nodes=1` (single-node Python)** pour le travail de
  performance, sur un nœud 512 GB.
- **Le crack MSA block-sparse se fait dans la classe `minimax_m3` Python.**
- **Telemak/Swift est différé ET déprioritisé** : plafond de vitesse supérieur
  et foyer du spéculatif, mais Swift est déjà rapide ; on n'y étend que si le
  Python single-node + MSA y arrive et qu'un besoin le justifie. Issue séparée.

## Consequences

- Réutilise le travail de la nuit (classe + converter), zéro port Swift.
- Élimine la classe de crashs JACCL de la boucle de perf (gain de stabilité).
- Le bench MSA se mesure single-node, single-stream — design simplifié (pas de
  contraintes pipeline/distribuées dans l'attention).
- Le 2-node reste disponible (la classe garde `pipeline()`) pour les modèles qui
  ne tiennent pas en solo (Q8 ~470 GB sera à la limite — à vérifier).
- Le rêve MTP de M3 reste vivant mais ailleurs : têtes EAGLE à entraîner (M3 ne
  livre aucun poids MTP — voir le constat du 2026-06-13), chantier recherche
  distinct.
