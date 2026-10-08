# Upstream tracker

Every PR, issue and comment we have open on upstream projects, with what we
expect from it and when to chase it. Read and updated by the `upstream-watch`
skill (`~/.claude/skills/upstream-watch/`): it queries GitHub for each row,
reports what moved since `Last seen`, and rewrites that column.

Rules for the table (the skill parses it):

- one row per item; `Repo` is `owner/name`, `#` is the number, `Kind` is
  `pr`, `issue` (ours) or `comment` (someone else's issue we posted on);
- `Last seen` is the ISO date of the last activity we have read; leave empty
  for a new row and the skill fills it;
- `Chase on` is the date after which silence becomes a ping (`-` = never);
- move closed or merged rows to the Archive section, keep the outcome.

## Open

| Repo | # | Kind | Subject | What we want | Last seen | Next action | Chase on |
|---|---|---|---|---|---|---|---|
| ml-explore/mlx | 3910 | issue | JACCL MeshImpl::recv spins forever on peer loss | none (#4530 refused 2026-09-29) | 2026-10-08 | none: no upstream action (decision 2026-10-08) | - |
| ml-explore/mlx | 4278 | comment | JACCL never detects a lost peer (survivors at 100% CPU) | none (#4530 refused 2026-09-29) | 2026-10-08 | none: no upstream action (decision 2026-10-08) | - |
| ml-explore/mlx | 4192 | comment | SIGSEGV in tbt_post_recv on Thunderbolt link loss (Apple libthunderboltrdma) | anyone's result on macOS 27; Apple fix | 2026-09-21 | internal only: test one node in 27 when free (docs/PLAN-jaccl); no comment upstream | - |
| ml-explore/mlx | 3467 | comment | RTR errno 22 after GID selection regression | fix merged upstream (our patch carries the workaround) | 2026-09-21 | none | - |
| ml-explore/mlx-lm | 1788 | comment | Qwen3.8-Flash-Next (qwen4_exp) support | merge; we run our own carve-out (mlx_models/qwen4_exp) until then; seed 0 vs reference 1234 (sje397, 2026-09-26): confirmed on our checkpoints, fixed in 3d81293 (forge #92) | 2026-09-26 | re-test against the branch when it merges, drop our carve-out | - |
| exo-explore/exo | 1847 | issue | jaccl RDMA crashes on M3 Ultra (errno 2/60/22) | nothing more; solved on our side, kept for visibility (Alex Cheema call 2026-09-17) | 2026-09-21 | close if Sophie decides; no pointer to #4530 (refused) | - |
| exo-explore/exo | 1831 | issue | Mistral Large 3 support | dormant | 2026-09-21 | close if still silent at next pass | 2026-10-15 |
| exo-explore/exo | 1825 | issue | load private models from a local path while online | dormant | 2026-09-21 | close if still silent at next pass | 2026-10-15 |
| exo-explore/exo | 1788 | issue | mDNS discovery broken from git source | root cause on record: Mearman (2026-10-07) reproduced without exo — an ungranted process launched over ssh loses LAN multicast (EHOSTUNREACH) once the ssh session ends, fine in the foreground; Apple TN3179 `AllowedEthernetLocalNetworkAddresses` untested. Matches our TCC finding (no kTCCServiceLocalNetwork entry) | 2026-10-07 | no longer dormant: keep open; check whether our own headless paths (node daemon `eu.odyssai.x.node`, runners launched over ssh) depend on LAN multicast; reply only on Sophie's go | - |

## Archive

| Repo | # | Kind | Subject | Outcome | Closed |
|---|---|---|---|---|---|
| ml-explore/mlx | 4530 | pr | JACCL: report a dead peer instead of spinning; name device/port/peer in errors | closed unmerged 2026-09-29 by zcbenz: refused (AI-written, hacky, hard to maintain; not a priority). Our vendored patch in vendor/jaccl stays. No further upstream action on MLX (decision 2026-10-08) | 2026-09-29 |
| Blaizzy/mlx-vlm | 2352 | pr | MiMo-V2.6 audio requests fail across threads in mlx_vlm.server | merged 87020830 (0.7.3); pin bumped, local step 5 dropped | 2026-09-24 |
| exo-explore/exo | 1930 | issue | Gemma 4 not tagged `vision` in /v1/models | fixed upstream | 2026-05 |
| exo-explore/exo | 1824 | issue | event_router nack loop with 100k+ events | fixed upstream | 2026-05 |
| exo-explore/exo | 1792 | issue | build from source: mDNS discovery fails on Tahoe 26.3 | closed as duplicate of 1788 | 2026-05 |
