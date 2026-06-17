# CONTEXT — JACCL / RDMA distributed (glossaire)

Glossaire du domaine pour le diagnostic du crash JACCL d'Argo. Termes uniquement,
pas de détails d'implémentation.

- **JACCL** — backend distribué RDMA-sur-Thunderbolt de MLX (vit dans `ml-explore/mlx`,
  `mlx/distributed/jaccl/`). Odysseus le consomme directement (`mx.distributed.init(backend="jaccl")`),
  sans exo.
- **Argo** — notre cluster d'inférence distribué : 3 nœuds M3 Ultra (`.29` rank0 master,
  `.30` rank1, `.31` rank2) reliés en mesh Thunderbolt 5, backend `jaccl`, mode pipeline.
- **QP (Queue Pair)** — l'endpoint d'une connexion RDMA. Type **UC (Unreliable Connection,
  `IBV_QPT_UC`)** chez JACCL : pas de heartbeat ni de détection automatique de peer mort.
- **RTR (Ready To Receive)** — un état de transition d'un QP. *"Changing queue pair to RTR
  failed"* = le QP n'a pas pu atteindre RTR.
- **GID (Global Identifier)** — l'adresse RDMA utilisée pour câbler la destination d'un QP.
  Apple Thunderbolt RDMA n'expose que des GID **link-local IPv6 (`fe80::…`)**, pas des
  GID RoCE v2 (`::ffff:x.x.x.x`).
- **errno 22 / 60 / 2** — les codes d'échec JACCL observés : 22 = EINVAL (le plus fréquent,
  sur le RTR), 60 = ETIMEDOUT (Recv timeout), 2 = ENOENT-ish (Recv failed / connection lost).
  **errno 22 est générique** — plusieurs causes distinctes peuvent le produire.
- **Runner orphelin** — quand les workers (ranks N>0) meurent mais que le master (rank 0)
  survit en tenant le modèle en mémoire wired ; le cluster passe de N à 1 nœud "vivant".
- **Recovery = reboot** — état observé : une fois le crash survenu, seul un reboot OS des
  nœuds remet le RDMA en état (rien ne libère l'état du driver sans power-cycle).
- **Régression vs chronique** — *régression* = a marché puis cassé à une date/commit précis ;
  *chronique* = n'a jamais été stable. La distinction est ouverte pour notre cas.

## Onboarding d'un node RDMA (termes)

- **Thunderbolt Bridge / `bridge0`** — le service réseau macOS qui agrège TOUS les
  ports Thunderbolt en une seule interface pontée. Tant que les ports sont dans ce
  pont, ils n'ont PAS d'IPv6 link-local individuel → NDP ne voit aucun peer → JACCL
  ne découvre personne. Le défaire est *toute* la tâche d'onboarding RDMA.
- **Management NIC (`en0`)** — le 10GbE intégré qui porte l'IP de management du node
  (`192.168.86.3x`) en DHCP. **Recréé par la recette d'onboarding** dans la location
  "odyssai", protégé par des **assertions de santé post-apply** (bail non-APIPA + route,
  poll avec deadline) — voir ADR-0001 (réécrit 2026-06-10).
- **Wi-Fi (lien out-of-band)** — la voie de secours qui garde un node joignable si
  `en0` tombe. Recréé par la recette (comme chez exo) ; sa ré-association est vérifiée
  post-apply. Ne jamais l'omettre d'une location (la faute du script de la nuit du 08).
- **APIPA (`169.254.x`)** — adresse auto-assignée quand le DHCP échoue. Sur le
  **management NIC** = échec, jamais succès. Sur les **ports Thunderbolt** = voulu
  (le DHCP par port échoue par design → APIPA IPv4 + IPv6 fe80, ce que JACCL consomme).
- **exo (référence)** — l'implémentation éprouvée tournant sur les 4 nodes sains
  (location "exo", services par port TB, daemon `io.exo.networksetup` RunAtLoad +
  1786 s). Source Apache 2.0 analysée (commit 09f9ea3) et **vendorisée** en
  `scripts/odyssai-network-setup.sh` ; driver = `scripts/rdma-onboard.sh`.
- **Location "odyssai"** — la network location dédiée créée par la recette (équivalent
  Odyssai de la location "exo"). Sert aussi de mécanisme de rollback (`--revert`
  rebascule sur la location persistée d'origine).

## Serving & vitesse (glossaire)

Termes du domaine vitesse d'inférence (entrés via le grill Mistral #54, 2026-06-16).

- **Decode bandwidth-bound** — pour un modèle DENSE, le decode est borné par la bande
  passante mémoire : chaque token re-streame TOUS les poids. tok/s ≈ bande_passante ÷
  taille_modèle. D'où un dense lent là où un MoE (ne streame que ses params actifs) vole.
  Repère : M3 Ultra ~819 GB/s ; un 128B en Q8 ~128 GB → plafond ~6,4 tok/s.
- **EAGLE draft** — petit modèle de tête (1 couche, ~1,5 B pour Mistral 3.5) qui PROPOSE
  plusieurs tokens d'avance ; le target les VÉRIFIE en un forward. Spéculatif **lossless**
  (la sortie reste exactement celle du target). Mistral le livre en format natif, pour
  vLLM/SGLang.
- **Speculative acceptance** — fraction des tokens proposés par le draft que le target
  accepte ; détermine le speedup réel. Peut baisser si le target est quantifié alors que
  le draft a été entraîné sur le full.
- **MTPLX** — notre infra de décodage spéculatif (repo séparé `mtplx-odyssai`) : têtes
  type MTP/EAGLE sur modèles **denses** (1,71× prouvé). C'est le véhicule pour porter EAGLE
  chez nous — vLLM/SGLang étant CUDA-first, ils ne tournent pas sur Metal.
- **Tensor-parallel (TP) vs pipeline** — TP shard les poids INTRA-couche (chaque node tient
  une tranche de têtes ; KV-heads divisibles par world_size requis) → scale la bande passante
  agrégée. Pipeline shard PAR couches : c'est le mode d'**Argo**. Un dense comme Mistral
  (KV=8) peut faire du TP sur 2 ou 4 nodes ; M3 était pipeline-only.
