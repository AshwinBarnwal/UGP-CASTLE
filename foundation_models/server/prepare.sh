#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
cd "$ROOT"
mkdir -p results/foundation_models/input
conda run -n ugp-nicheformer python foundation_models/prepare_cosmx_inputs.py \
  --adata data/cosmx_frontal_cortex.h5ad \
  --model-vocabulary external/nicheformer/data/model_means/model.h5ad \
  --hgnc foundation_models/resources/hgnc_complete_set.txt \
  --technology-mean external/nicheformer/data/model_means/cosmx_mean_script.npy \
  --output-dir results/foundation_models/input

