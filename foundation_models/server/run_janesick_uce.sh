#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
export CUDA_VISIBLE_DEVICES="${FOUNDATION_GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
OUT="$ROOT/results/foundation_models/uce_janesick"
mkdir -p "$OUT" "$ROOT/logs/foundation_models"
cd "$ROOT/external/UCE"
conda run --no-capture-output -n ugp-uceserver \
  python "$ROOT/foundation_models/uce_entrypoint.py" \
  --adata_path "$ROOT/data/janesick_breast.h5ad" --dir "$OUT/" \
  --species human \
  --model_loc "$ROOT/external/UCE/model_files/33l_8ep_1024t_1280.torch" \
  --nlayers 33 --batch_size "${UCE_BATCH_SIZE:-8}" \
  --spec_chrom_csv_path "$ROOT/external/UCE/model_files/species_chrom.csv" \
  --token_file "$ROOT/external/UCE/model_files/all_tokens.torch" \
  --protein_embeddings_dir "$ROOT/external/UCE/model_files/protein_embeddings/" \
  --offset_pkl_path "$ROOT/external/UCE/model_files/species_offsets.pkl" \
  2>&1 | tee "$ROOT/logs/foundation_models/uce_janesick.log"
