# Plan: RDMA node onboarding — refait depuis exo, en0 & Wi-Fi inviolables
_Locked via grill-with-docs — by Claude + Sophie. Terms per CONTEXT.md. Hardened by Codex round 1._

## Goal
Un Mac neuf rejoint le mesh RDMA d'Argo (backend JACCL) en obtenant un IPv6
link-local sur chaque port Thunderbolt — en défaisant l'agrégation `bridge0`,
**et rien d'autre**. L'étape opère en place, ne touche QUE les interfaces
Thunderbolt, et ne modifie **jamais** le management NIC (`en0`) ni le Wi-Fi
(voir [ADR-0001](docs/adr/0001-rdma-onboarding-thunderbolt-only.md)). On
**réplique exo**, on ne réinvente pas.

## Les problèmes (pourquoi `.33` a brické)
- **P1 — bascule de location réseau.** Le script basculait sur une location vide →
  recréation du service `en0` → `en0` perd son bail DHCP, reste en APIPA `169.254`,
  node injoignable en `.33`. *Violation : touche `en0`.*
- **P2 — location sans Wi-Fi.** Plus de service Wi-Fi → perte du lien out-of-band →
  node stranded quand `en0` tombe. *Violation : tue le Wi-Fi.*
- **P3 — ports TB en IPv4 DHCP** → chaque port tombait en `169.254`, encombrait le
  routage, contribuait à virer `en0` de son réseau.
- **P4 — `verify_mgmt` acceptait une APIPA comme succès** → masquait l'échec au boot.
- **P5 — rewrite from-scratch d'exo** au lieu d'une réplication fidèle (cause racine).

## Approche — flux à deux phases, garde-fous d'abord

### Phase 0 — Préflight (lecture seule, AUCUNE mutation)
1. **Snapshot complet** (rapport JSON pré-état) : `scselect`, liste des services +
   service order, `ifconfig bridge0` + membres, `ifconfig` de chaque TB, état Wi-Fi
   (power, service UUID, device, SSID/IP si associé), table de routes, `ibv_devinfo`,
   statut launchctl.
2. **Identifier les ports TB de façon robuste** : d'abord via **IORegistry /
   SystemConfiguration** (sur un node encore ponté, `ibv_devinfo` peut être vide/inactif
   avant teardown) ; `ibv_devinfo` (`rdma_en*` → `en*`) sert de **postcheck/gate** après.
   **Jamais** par grep de nom "Thunderbolt" (les `enN` bougent).
3. **Vérifier l'identité de `bridge0`** : c'est bien le service macOS "Thunderbolt
   Bridge" ET tous ses membres sont des interfaces TB vérifiées. **Abort** si `bridge0`
   contient `en0`, le Wi-Fi, ou toute interface non-TB.
4. **Garde management** : snapshot de l'interface/route/service-UUID de management
   actif ; protéger cette interface **et** le littéral `en0`.
5. **Garde Wi-Fi** : exiger un Wi-Fi **réellement utilisable comme bouée** — allumé,
   **associé**, avec une IP **non-APIPA**. Sinon **abort** — un override "no-OOB" n'est
   admis que via **confirmation console physique**, jamais un simple flag. Partir d'une
   bouée déjà coupée est interdit.
6. **Garde SSH/orchestrateur** : `route get` vers **l'IP orchestrateur attendue ET le
   client SSH actif** (`$SSH_CONNECTION`) — **abort** sauf si chaque route passe par le
   management protégé ou le Wi-Fi ; **jamais** par bridge/TB.
7. **Garde mutation générique** : avant toute commande mutante, refuser si la cible
   n'est pas un port TB vérifié.

### Phase 1 — Test runtime (mutation non-persistante, non restaurée à distance)
8. Défaire `bridge0` **en runtime uniquement** (pas de prefs, pas de location) —
   réplication exacte du teardown d'exo, scoping TB validé en phase 0. Si `bridge0`
   est **déjà absent** au préflight : **idempotent** — sauter le teardown, considérer le
   runtime OK **seulement si le postcheck TB passe**, puis installer/vérifier le daemon.
9. **Postcheck TB** : les HCAs **attendus selon la topologie mesh planifiée** (pas tous
   les ports TB possibles du Mac) ont `inet6 fe80` et sont `PORT_ACTIVE`.
   On ne configure **aucune méthode IP** sur les ports TB (ni DHCP ni v4off) — le
   critère de succès est purement IPv6 link-local + HCA active.
10. **Re-vérifier les gardes** : `en0` toujours sain — comparer **service UUID/device,
    IP non-APIPA, router, route** (PAS les métadonnées de bail : un renouvellement DHCP
    ne doit pas faux-déclencher) ; Wi-Fi toujours utilisable. Si dérive management/Wi-Fi
    détectée → **arrêter TOUTE mutation réseau et reporter immédiatement** (ne PAS faire
    une mutation de plus pour "revert").

### Phase 2 — Persistance (seulement si phase 1 verte)
11. Installer le **LaunchDaemon** (mécanisme exo, PAS une location) :
    `/Library/LaunchDaemons/<label>.plist`, `root:wheel`, plist `0644`, script `0755`,
    `RunAtLoad` + `StartInterval` (re-vérif périodique), logs dédiés ;
    `launchctl bootstrap system` + `kickstart`.
12. Le daemon **gère la course de boot** : retry avec settle/backoff jusqu'à
    `bridge0` absent ET HCAs TB `PORT_ACTIVE` sur **N échantillons consécutifs**, avec un
    **plafond de retries par cycle de boot** ; au-delà, laisser un **statut/log d'échec
    clair** (état dégradé) et s'en remettre au `StartInterval` pour réessayer plus tard —
    pas de boucle serrée infinie.
13. **Postcheck persistance** ; si échec → `launchctl bootout` + suppression du daemon ;
    le réseau reste dans son **état dé-ponté courant** (on ne recrée pas `bridge0`),
    rapport émis.

### Pas de prefs en v1
- **Aucune édition de `preferences.plist` en v1.** Teardown daemon-only. On n'ajoute de
  chirurgie plist que si exo prouve empiriquement que macOS recrée `bridge0` malgré le
  daemon — et alors sous **contrat de diff strict** : muter une copie offline,
  `plutil -lint`, comparer avant/après, n'autoriser que le retrait de l'UUID exact du
  service Thunderbolt Bridge et de ses références.

## Réplication d'exo (vérité terrain)
- **WU1** — récupérer le `disable_bridge.sh` réel depuis un node sain (`.32`) en
  **lecture seule**.
- **Audit statique avant tout usage** : scanner pour `scselect`, `networksetup`, `en0`,
  Wi-Fi/`airport`, `location`, écritures plist → **abort sur tout match hors allowlist**.
- **Vendoriser** la version minimale auditée **dans ce repo** (ou figer le hash source).
  Pas de dépendance runtime à "exo est installé".

## Gate de chargement
- **Odysseus refuse `backend=jaccl`** au load si les HCAs `rdma_to` attendus sont absents
  ou pas `PORT_ACTIVE` — **erreur de santé explicite**, pas de bascule silencieuse. Passer
  en `ring` doit être un **choix opérateur** (changement de config), jamais un fallback
  automatique qui masque l'échec RDMA et fait tourner l'inférence sur le LAN management.

## Critère de succès (gaté par reboot)
apply → **reboot** → SSH joignable en `.3x` → **Wi-Fi toujours on** → **`en0` inchangé**
→ **`route get` post-reboot : le SSH/management passe toujours par le management protégé
ou le Wi-Fi, jamais par TB/bridge** → pas de `bridge0` → tous les HCAs attendus
`PORT_ACTIVE` → `discover-rdma-wiring.py` montre le mesh complet. Tant que ce chemin n'est
pas vert, l'onboarding n'est PAS "réussi".

## Key decisions & tradeoffs
- **Pas de location réseau** ([ADR-0001](docs/adr/0001-rdma-onboarding-thunderbolt-only.md)) :
  choix structurant. Persistance via LaunchDaemon, pas via location.
- **Daemon-only d'abord, prefs jamais** (sauf preuve empirique) : on privilégie le chemin
  le plus simple et le moins risqué, prouvé au reboot, avant toute chirurgie plist.
- **Vendoriser exo audité** plutôt que wrapper un script externe non audité.
- **`169.254` = échec uniquement sur le management**, jamais sur les ports TB (qui n'ont
  pas d'IPv4 du tout).

## Rollback / cleanup (explicite)
- Phase 1 échoue → **ne PAS recréer `bridge0`** (recréer à distance ré-entre dans le
  chemin dangereux). Laisser les ports TB dé-pontés, retirer tout artefact installé,
  émettre le rapport ; restauration de `bridge0` uniquement en **console physique** si
  vraiment nécessaire.
- Phase 2 échoue → `launchctl bootout` + suppression du daemon ; le réseau reste dans son
  **état dé-ponté courant** (on ne recrée pas `bridge0`), rapport émis.
- Si (et seulement si) des prefs ont été touchées → restaurer **uniquement** depuis un
  backup validé. **Jamais** tenter de recréer `bridge0` à distance.

## Risks / open questions
- Le teardown runtime de `bridge0` persiste-t-il sans location ? (exo = daemon RunAtLoad —
  confirmer en WU1.)
- macOS peut ré-agréger `bridge0` au boot avant le daemon → géré par retry/settle (étape 12),
  à valider empiriquement.
- exo absent/différent sur les nodes sains → le vendoring auditté couvre ce cas.

## Out of scope
- Ajout du rank dans `cluster-config.json` + load du modèle (marchent déjà).
- Réseau des 4 nodes existants (ne pas y toucher).
- Backend `ring` (aucun bridge nécessaire — fallback si JACCL échoue).
