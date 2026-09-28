# Upstream: exo

Source: https://github.com/exo-explore/exo
Licence: Apache License 2.0, `Copyright 2025 Exo Technologies Ltd` (copy of the
upstream `LICENSE`, byte-identical, in this directory). exo ships no `NOTICE`
file and its source files carry no per-file licence header, so there is no
upstream header or NOTICE text to carry over; each derived file below gets an
Apache-2.0 attribution header and a `Modified by OdyssAI` line instead.

The derived files stay where the runtime expects them (`scripts/`, copied to
`~/mlx-cluster/` on each node); this directory only holds the licence and this
provenance record. Our modifications are licensed under the AGPL-3.0 as part of
OdyssAI-X (see `LICENSE` and `NOTICE` at the repository root).

## Derived files

| File in this repo | Upstream path | Upstream commit | Relation |
|---|---|---|---|
| `scripts/auto_parallel.py` | `src/exo/worker/engines/mlx/auto_parallel.py` | `f0d1371d89a7f899e96977014cce7307a53682f2` (2026-04-28) | upstream file nearly verbatim (~84% of our non-blank lines are exo's), OdyssAI additions |
| `scripts/patches/opt_batch_gen.py` | `src/exo/worker/engines/mlx/patches/opt_batch_gen.py` | `f0d1371d89a7f899e96977014cce7307a53682f2` | byte-identical below the header (136 lines) |
| `scripts/patches/standard_yarn_rope.py` | `src/exo/worker/engines/mlx/patches/standard_yarn_rope.py` | `f0d1371d89a7f899e96977014cce7307a53682f2` | byte-identical below the header (118 lines) |
| `scripts/patches/__init__.py` | `src/exo/worker/engines/mlx/patches/__init__.py` | `f0d1371d89a7f899e96977014cce7307a53682f2` | exo's 13-line skeleton, extended |
| `scripts/odyssai-network-setup.sh` | script embedded in `app/EXO/EXO/Services/NetworkSetupHelper.swift` (lines 16-69) | `09f9ea313f72e261f40a94cea4c0e3681b31af23` (2026-06-03) | 10 of 96 code lines identical, structure follows exo |
| `scripts/odyssai-network-guard-exoloc.sh` | same embedded script | `09f9ea313f72e261f40a94cea4c0e3681b31af23` | 3 of 32 code lines identical |

### Modifications

- `auto_parallel.py`: at import (initial public release, 2026-05-23) the only
  changes were the three `exo.*` imports replaced by the local `exo_stubs.py`
  shim and the PEP 695 generics (`def f[T]`) replaced by a module-level
  `TypeVar` so the file parses under Python 3.11 (1587 of the 1593 upstream
  lines unchanged). Since then: sharding strategies added or adapted for
  DeepSeek V4, Bailing/Ling, MiMo, GLM DSA, Kimi K3, qwen4_exp and the
  Qwen3.5-MoE vision pipeline, plus the pipeline prefill transport. See
  `git log --follow scripts/auto_parallel.py`.
- `patches/opt_batch_gen.py`, `patches/standard_yarn_rope.py`: attribution
  header added; everything below it is byte-identical to upstream.
- `patches/__init__.py`: same `_applied` guard and `apply_mlx_patches()` entry
  point as upstream, with relative imports and OdyssAI's own model patches and
  pipeline split-coverage fix added to the list.
- `odyssai-network-setup.sh`: the file header lists the changes (OdyssAI
  identity, dynamic bridge-service resolution, device-based bridge skip,
  management and Wi-Fi services recreated first, guarded `setdhcp`, exact-match
  idempotence guards, kill-switch, `ODYSSAI_SKIP_SLEEP`).
- `odyssai-network-guard-exoloc.sh`: keeps the boot wait, the bridge0 teardown
  and the Thunderbolt Bridge disable; drops the location switch and service
  creation; replaces the unconditional `setdhcp` with a static-IP assertion.

### Measured overlap (2026-09-28)

Re-checked against exo's source on GitHub at the audited commits and at `main`:
the two patches are byte-identical below our header, and identical to exo
`main` as well. `auto_parallel.py`: 1,376 of our 1,645 non-blank lines are
lines of exo's file, which is contained almost entirely. Network scripts:
stripped non-comment lines compared with the script embedded in
`NetworkSetupHelper.swift` (lines 16-69).

## Checked and not derived

| File | Evidence |
|---|---|
| `scripts/exo_stubs.py` | Written for OdyssAI-X as an import shim for `auto_parallel.py`. It declares two plain dataclasses with the field names `auto_parallel.py` reads (`device_rank`, `world_size`, `start_layer`, `end_layer`, `immediate_exception`, `should_timeout`; `layers_loaded`, `total`) and a stdlib logger. Upstream defines these as pydantic models with other fields, properties and base classes (`src/exo/shared/types/worker/shards.py`, `runner_response.py`); the only identical lines are two field declarations (`immediate_exception: bool = False`, `should_timeout: float \| None = None`), which the interface requires. Listed here because its docstring names the upstream types. |
| `scripts/rdma-onboard.sh` | Installs and guards the network recipe above; credits the mechanism in its header. Shared lines are the shebang, `set -euo pipefail`, the Apple plist DOCTYPE and a `launchctl bootstrap` call. |
| `scripts/master.py` | `random_ephemeral_port()` is a one-line `random.randint` over the OS ephemeral range, a different implementation from `src/exo/utils/ports.py`. |
| `scripts/runner.py`, `scripts/vlm_runner.py` | Import `auto_parallel` and `patches`; the "exo-style" comments describe that use. Fewer than 5 shared lines of 25+ characters per upstream file, all generic Python (`if __name__ == "__main__":`, `except StopIteration as e:`). |
| `scripts/wired-limit/install.sh` | Shared lines are Apple plist boilerplate only. |

## Method (2026-09-27)

1. Every tracked `.py`, `.sh`, `.swift`, `.c`, `.h`, `.js`, `.html`, `.ts` and
   `.rs` file was compared line by line (stripped lines of 25+ characters,
   imports excluded) with every source file of exo at `f0d1371`, `09f9ea3` and
   `21a54c5ea0230a3bec1e1a786d200126c7e34ec6` (upstream `main` on 2026-09-27).
   Files sharing 5 or more lines were read by hand; the results are the two
   tables above.
2. `scripts/auto_parallel.py` as first committed here (`d2eb07f`) was diffed
   against every upstream revision of `auto_parallel.py`: `f0d1371` is the
   closest (similarity 0.994), and upstream `main` still carries that exact
   file, as it does for both patches.
3. Upstream `LICENSE`, root listing and file heads were checked for a `NOTICE`
   file and per-file headers: none.

## Rule for future reuse

Any code taken from exo (for example by the planned Bonjour node discovery)
gets the same header, a row in the first table with the upstream path
and commit, and a line in the root `NOTICE`.
