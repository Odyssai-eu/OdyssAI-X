# Runbook — .33 (ultra-256d) → Argo rank 4 — execution, sans kwaks
_Date cible: 2026-06-10. S'appuie sur PLAN-node-redo.md (approuvé Codex 6 rounds),
ADR-0001, scripts/rdma-onboard.sh (approuvé Codex). Ordre strict, read-only-first._

## État de départ (vérifié read-only, nuit du 2026-06-10 — clé SSH ré-installée)
- `.33` joignable à **192.168.86.33** (uplink filaire réparé → bail DHCP d'en0 repris).
- OS **macOS 26.5.1** réinstallé propre ; location réseau = **Automatic** (aucun reliquat
  "odyssai" — le dégât de la veille est effacé) ; **aucun** daemon/artefact `eu.odyssai`.
- en0 = **192.168.86.33** actif 10Gbase-T ; Wi-Fi en1 = **On, .73** (bouée OK).
- **bridge0 présent**, agrège les 6 ports TB (état canonique avant onboarding). Câbles
  vivants : **en3/en4/en5/en7 actifs** (4 liens mesh), en2/en6 vides. **Aucun fe80** sur
  les TB (bridgés) ; `rdma_en3/4/5/7` = PORT_ACTIVE.
- **Ring-2.5 PRÉSERVÉ** sur le TerraMaster (`/Volumes/models`, disk6 8 TB) → **pas de
  re-rsync**. Étape 1 "vérifier Ring-2.5" = ✅ déjà fait.
- **venv mlx absent** → re-provision (bootstrap) nécessaire.
- 4 nodes sains (.29–.32) intacts — **ne pas y toucher**.

## Ordre d'exécution

### Étape 0 — Accès (Sophie, une fois)
- Ré-autoriser la clé SSH sur l'OS neuf : `ssh-copy-id admin@192.168.86.33` (mot de passe
  interactif) ou via console. Confirmer `ssh admin@192.168.86.33 hostname`.
- Re-poser le sudo sans mot de passe (`/etc/sudoers.d/odyssai`) comme avant.

### Étape 1 — Re-provision base
- `bootstrap-node.sh` (brew, python3.11, mlx 0.31.2 / mlx-lm 0.31.3). **Fixer le bug
  `$HOME` scp** : `ODYSSEUS_REMOTE_CLUSTER_DIR=/Users/admin/mlx-cluster`.
- Vérifier Ring-2.5 présent sur le TerraMaster (`diskutil list external physical` + `ls`
  du model dir). Re-rsync seulement si le SSD a été effacé.

### Étape 2 — Onboarding RDMA (le chemin durci)
1. **WU1 d'abord — auditer + vendoriser exo** : récupérer `disable_bridge.sh` depuis un
   node sain (`.32`) en **lecture seule** ; audit statique (scan allowlist : refuser
   `scselect`/`networksetup` sur en0/Wi-Fi/location/écritures plist) ; vendoriser la
   forme minimale auditée ; créer le marqueur
   `/Library/Application Support/Odyssai/exo-teardown.audited`.
   **`rdma-onboard.sh --apply` REFUSE de tourner sans ce marqueur.**
2. Copier `scripts/rdma-onboard.sh` sur `.33`.
3. `sudo ./rdma-onboard.sh --check` — snapshot read-only ; confirmer : bridge0 = Thunderbolt
   Bridge avec membres TB uniquement, Wi-Fi bouée (on+associé+non-APIPA), en0 sain à `.33`.
4. `sudo ./rdma-onboard.sh --apply --orchestrator <ip-orchestrateur> [--expect N]` —
   préflight (gardes en0/Wi-Fi/route SSH) → teardown runtime → daemon. Abort sûr si une
   mutation devait toucher en0/Wi-Fi.
5. **REBOOT.**
6. **Gate post-reboot** : `ssh .33` → Wi-Fi on → en0 toujours `.33` → `route get` pas
   TB/bridge → pas de bridge0 → HCAs TB `PORT_ACTIVE`.

### Étape 3 — Vérif mesh
- `discover-rdma-wiring.py 0=…29 1=…30 2=…31 3=…32 4=…33` → attendre **20/20**. Si 19/20
  (`.30↔.33`), bouger ce **seul** câble TB et re-run. (Vérité terrain du mesh — pas le
  count-gate de l'étape 2.)

### Étape 4 — Cluster config
- Ajouter `.33` comme **rank 4** au cluster "main"/Argo avec la **matrice `rdma_to`
  complète** émise par la discovery (jamais inventée — AGENTS.md / INSTALL-CLUSTER.md).

### Étape 5 — Load
- Charger **Ring-2.5 sur le pipeline JACCL 5-node**. Smoke (tok/s, TTFT). Si HCAs pas
  prêts → **erreur de santé explicite**, pas de fallback `ring` silencieux.

## Garde-fous "sans kwaks"
- **Jamais** en0, **jamais** Wi-Fi, **jamais** de location, **jamais** de prefs, **jamais**
  d'IP sur les ports TB (ADR-0001, codé dans `rdma-onboard.sh`).
- `--apply` bloqué tant que le marqueur d'audit exo n'existe pas.
- **Read-only d'abord à chaque étape** ; mutation seulement après vérif explicite.
- En cas de dérive management/Wi-Fi → **STOP, aucune mutation de "réparation"** (la leçon
  de la nuit du 09).

## Rollback
- Échec onboarding → daemon retiré, réseau laissé **dé-ponté** (bridge0 **jamais** recréé
  à distance) ; node reste joignable (Wi-Fi + en0).

## Out of scope
- Réseau des 4 nodes existants. Backend `ring` (fallback opérateur seulement).
- Deepseek-v4-5-nodes (goal séparé).
