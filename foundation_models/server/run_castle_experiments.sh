#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
export CUDA_VISIBLE_DEVICES="${CASTLE_GPU:-1}"
export PYTHONPATH="$ROOT/castle_refree/src"
mkdir -p "$ROOT/logs/castle"
cd "$ROOT"

run_castle () {
  local experiment="$1" bundle="$2" output="$3" dataset="$4" adaptation="$5"
  local log_key="${output//\//_}"
  local extra=()
  if [ "$adaptation" = "cosmx" ]; then extra=(--cosmx-adaptations); fi
  conda run --no-capture-output -n ugp-nicheformer \
    python castle_refree/scripts/run_castle_identity_experiment.py "$experiment" \
    --bundle "$bundle" --output "$output" --dataset-name "$dataset" \
    --device cuda --threads "${CASTLE_THREADS:-8}" "${extra[@]}" \
    2>&1 | tee "$ROOT/logs/castle/${log_key}.log"
}

# Experiment i already exists for both datasets. Train only the new experiment
# ii configurations; the benchmark script reads the existing experiment-i runs.
# Janesick uses exact segmentation-mask contacts.
run_castle ii data/castle_xenium_breast_raw results/embedding_janesick/castle_ii \
  "Janesick Xenium breast Rep1" janesick

# CosMx has no segmentation mask. Keep experiment-i decontamination unchanged
# and use the explicitly documented proxy graph only for contact-step context.
if [ ! -f data/cosmx_frontal_cortex_hvg2000/edges_contact.csv ]; then
  conda run --no-capture-output -n ugp-nicheformer \
    python castle_refree/scripts/build_proxy_contacts.py \
    --bundle data/cosmx_frontal_cortex_hvg2000
fi
run_castle ii data/cosmx_frontal_cortex_hvg2000 results/embedding_cosmx_hvg2000/castle_ii \
  "CosMx frontal cortex 2000 HVGs" cosmx
