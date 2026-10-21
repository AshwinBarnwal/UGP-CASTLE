#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
ZIP="$ROOT/data/raw_janesick/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip"
RAW="$ROOT/data/raw_janesick/outs"
OUT="$ROOT/data/castle_xenium_breast_raw"
URL="https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip"
mkdir -p "$(dirname "$ZIP")" "$RAW" "$OUT" "$ROOT/logs/castle"

if [ ! -f "$ZIP.complete" ]; then
  curl -L --fail --retry 5 --continue-at - -o "$ZIP" "$URL"
  touch "$ZIP.complete"
fi
if [ ! -f "$RAW/cell_feature_matrix.h5" ] && [ ! -f "$RAW/outs/cell_feature_matrix.h5" ]; then
  unzip -q "$ZIP" -d "$RAW"
fi
RAW_BUNDLE="$RAW"
if [ -f "$RAW/outs/cell_feature_matrix.h5" ]; then RAW_BUNDLE="$RAW/outs"; fi
if [ ! -f "$OUT/manifest.json" ]; then
  cd "$ROOT"
  PYTHONPATH="$ROOT/castle_refree/src" \
  conda run --no-capture-output -n ugp-nicheformer \
    python castle_refree/scripts/export_xenium_raw.py "$RAW_BUNDLE" \
    --output-dir "$OUT" --skip-transcripts \
    2>&1 | tee "$ROOT/logs/castle/prepare_janesick_raw.log"
fi
