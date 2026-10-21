#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
cd "$ROOT"
mkdir -p results/foundation_models/janesick_input logs/foundation_models

if [ ! -f data/janesick_breast.h5ad ]; then
  conda run --no-capture-output -n ugp-nicheformer \
    python foundation_models/prepare_janesick_h5ad.py \
    --bundle data/castle_xenium_breast \
    --labels data/xenium_r1_supervised_labels.csv \
    --output data/janesick_breast.h5ad
fi

conda run --no-capture-output -n ugp-nicheformer \
  python foundation_models/prepare_cosmx_inputs.py \
  --adata data/janesick_breast.h5ad \
  --model-vocabulary external/nicheformer/data/model_means/model.h5ad \
  --hgnc foundation_models/resources/hgnc_complete_set.txt \
  --technology-mean external/nicheformer/data/model_means/xenium_mean_script.npy \
  --technology Xenium \
  --output-dir results/foundation_models/janesick_input \
  2>&1 | tee logs/foundation_models/prepare_janesick.log
