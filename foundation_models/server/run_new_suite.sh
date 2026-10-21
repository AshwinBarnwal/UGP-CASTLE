#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
cd "$ROOT"

for required in \
  results/embedding_full/castle_i/z_c.npy \
  results/embedding_full/castle_i/cells.tsv \
  results/embedding_cosmx_hvg2000/castle_i/z_c.npy \
  results/embedding_cosmx_hvg2000/castle_i/cells.tsv
do
  if [ ! -f "$required" ]; then
    echo "Missing existing CASTLE experiment-i artifact: $required" >&2
    echo "Copy/pull the completed run; this suite deliberately will not retrain it." >&2
    exit 2
  fi
done

bash foundation_models/server/prepare_janesick.sh "$ROOT"
bash foundation_models/server/prepare_janesick_castle_raw.sh "$ROOT"
bash foundation_models/server/run_castle_experiments.sh "$ROOT"
bash foundation_models/server/run_janesick_nicheformer.sh "$ROOT"
bash foundation_models/server/run_janesick_uce.sh "$ROOT"
bash foundation_models/server/benchmark_new_suite.sh "$ROOT"
