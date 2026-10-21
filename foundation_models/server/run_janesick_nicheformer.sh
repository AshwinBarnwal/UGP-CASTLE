#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
export CUDA_VISIBLE_DEVICES="${FOUNDATION_GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
OUT="$ROOT/results/foundation_models/nicheformer_janesick"
mkdir -p "$OUT" "$ROOT/logs/foundation_models"
cd "$ROOT"
conda run --no-capture-output -n ugp-nicheformer \
  python foundation_models/run_nicheformer_streaming.py \
  --adata data/janesick_breast.h5ad \
  --mapping results/foundation_models/janesick_input/nicheformer_gene_mapping.tsv \
  --technology-mean external/nicheformer/data/model_means/xenium_mean_script.npy \
  --technology-token 9 \
  --output-dir "$OUT" --batch-size "${NICHEFORMER_BATCH_SIZE:-4}" \
  --precision fp16 --resume \
  2>&1 | tee "$ROOT/logs/foundation_models/nicheformer_janesick.log"
