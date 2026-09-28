# Upstream

Source: https://github.com/ml-explore/mlx — path `mlx/distributed/jaccl/lib`
Tag: v0.32.2 (2026-08-25), commit 1f8e74e3f12f31365464a6867c6579f0e9b29d85
Licence: MIT (MLX), text in `LICENSE-MLX`. Local patches are listed in PATCHES.md.
The build fetches nlohmann/json v3.11.3 (MIT, text in `LICENSE-nlohmann-json`),
compiled into the library.

## Binary release

`install.sh` does not build JACCL (that needs the Xcode tools and cmake): it
downloads `libjaccl.dylib` from the GitHub pre-release `jaccl-0.32.2-odyssai.1`
and refuses it unless its sha256 equals `jaccl_sha256` in
`scripts/doctor-manifest.json`. That binary is `scripts/build-jaccl.sh` output
from this directory; rebuilt from commit 0ed82e4 on 2026-09-28 (SDK 27.0) it is
byte-identical to the one built on 2026-09-18 and installed on the nodes. The
release carries both licence texts. After any change here: rebuild, publish a
new pre-release tag, regenerate the manifest, bump `JACCL_TAG` in install.sh.
