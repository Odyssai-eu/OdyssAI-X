# Upstream

Source: https://github.com/ml-explore/mlx — path `mlx/distributed/jaccl/lib`
Tag: v0.32.4 (2026-10-14), commit 36969c094354086e15d2a327a9c50feb20b9cf11
(rebased from v0.32.2, commit 1f8e74e3f12f31365464a6867c6579f0e9b29d85, on 2026-10-20)
Licence: MIT (MLX), text in `LICENSE-MLX`. Local patches are listed in PATCHES.md.
The build fetches nlohmann/json v3.11.3 (MIT, text in `LICENSE-nlohmann-json`),
compiled into the library.

## Binary release

`install.sh` does not build JACCL (that needs the Xcode tools and cmake): it
downloads `libjaccl.dylib` from the GitHub pre-release `jaccl-0.32.4-odyssai.1`
and refuses it unless its sha256 equals `jaccl_sha256` in
`scripts/doctor-manifest.json`. That binary is built from this directory with
the flags of `scripts/build-jaccl.sh` (Apple clang 21.0.0, SDK 27.0, deployment
target 26.2, ad-hoc signed) on 2026-10-20. The release carries both licence
texts. After any change here: rebuild,
publish a new pre-release tag, regenerate the manifest, bump `JACCL_TAG` in
install.sh. The previous release, `jaccl-0.32.2-odyssai.1`, only fits the
mlx 0.32.2 wheel.
