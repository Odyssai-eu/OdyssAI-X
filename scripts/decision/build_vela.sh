#!/bin/zsh
# Assemble a Vela 3.0 MLX folder: the d3 base converted by convert_d3.py (APFS clone) + the vela package's own files
# (adapters, heads, configs, definitions, docs). The vela package's config.json is not copied: the base's is the MLX one.
# usage: build_vela.sh <vela package dir> <converted d3 base dir> <out dir>
VELA=${1:?usage: build_vela.sh <vela package> <converted base> <out>}; BASE=${2:?}; OUT=${3:?}
[ -e "$OUT" ] && { echo "$OUT exists"; exit 1; }
cp -c -R "$BASE" "$OUT" || cp -R "$BASE" "$OUT" || exit 1
for f in "$VELA"/*_lora.safetensors "$VELA"/*_head.safetensors "$VELA"/vela3_config.json "$VELA"/label_definitions.json \
         "$VELA"/router_questions.json "$VELA"/calibration.json "$VELA"/README.md "$VELA"/EVALUATION.md "$VELA"/LICENSE; do
  [ -f "$f" ] && cp "$f" "$OUT"/
done
echo "built $OUT ($(ls "$OUT" | wc -l | tr -d ' ') files)"
