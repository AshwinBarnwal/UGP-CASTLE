#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
LIMIT="${2:-}"
BATCH="${NICHEFORMER_BATCH_SIZE:-4}"
PRECISION="${NICHEFORMER_PRECISION:-fp16}"
OUTPUT_SUBDIR="${NICHEFORMER_OUTPUT_SUBDIR:-nicheformer}"
export CUDA_VISIBLE_DEVICES="${FOUNDATION_GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LIMIT_ARGS=()
if [ -n "$LIMIT" ]; then LIMIT_ARGS=(--limit-cells "$LIMIT"); fi

mkdir -p "$ROOT/results/foundation_models/$OUTPUT_SUBDIR" "$ROOT/logs/foundation_models"
cd "$ROOT"
conda run -n ugp-nicheformer python foundation_models/run_nicheformer_streaming.py \
  --adata data/cosmx_frontal_cortex.h5ad \
  --mapping results/foundation_models/input/nicheformer_gene_mapping.tsv \
  --technology-mean external/nicheformer/data/model_means/cosmx_mean_script.npy \
  --output-dir "results/foundation_models/$OUTPUT_SUBDIR" \
  --batch-size "$BATCH" \
  --precision "$PRECISION" \
  --resume \
  "${LIMIT_ARGS[@]}" \
  2>&1 | tee "$ROOT/logs/foundation_models/${OUTPUT_SUBDIR}.log"

