#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
export PYTHONPATH="$ROOT/foundation_models:$ROOT/castle_refree/src"
mkdir -p "$ROOT/results/benchmark_v2" "$ROOT/logs/benchmark_v2"
cd "$ROOT"

bench_npy () {
  local source="$1" embedding="$2" cells="$3" method="$4" output="$5"
  conda run --no-capture-output -n ugp-nicheformer \
    python foundation_models/benchmark_single_embedding.py \
    --source "$source" --embedding-npy "$embedding" --cells "$cells" \
    --method "$method" --output-dir "$output" \
    2>&1 | tee "$ROOT/logs/benchmark_v2/$(basename "$output").log"
}

bench_npy data/janesick_breast.h5ad \
  results/embedding_full/castle_i/z_c.npy \
  results/embedding_full/castle_i/cells.tsv \
  "CASTLE experiment i — Janesick" results/benchmark_v2/janesick_castle_i

bench_npy data/janesick_breast.h5ad \
  results/embedding_janesick/castle_ii/z_c.npy \
  results/embedding_janesick/castle_ii/cells.tsv \
  "CASTLE experiment ii — Janesick" results/benchmark_v2/janesick_castle_ii

bench_npy data/janesick_breast.h5ad \
  results/foundation_models/nicheformer_janesick/nicheformer_embeddings.npy \
  results/foundation_models/nicheformer_janesick/cells.tsv \
  "Nicheformer — Janesick" results/benchmark_v2/janesick_nicheformer

conda run --no-capture-output -n ugp-nicheformer \
  python foundation_models/benchmark_single_embedding.py \
  --source data/janesick_breast.h5ad \
  --embedding-h5ad results/foundation_models/uce_janesick/janesick_breast_uce_adata.h5ad \
  --obsm-key X_uce --method "UCE-33 — Janesick" \
  --output-dir results/benchmark_v2/janesick_uce \
  2>&1 | tee "$ROOT/logs/benchmark_v2/janesick_uce.log"

bench_npy data/cosmx_frontal_cortex.h5ad \
  results/embedding_cosmx_hvg2000/castle_i/z_c.npy \
  results/embedding_cosmx_hvg2000/castle_i/cells.tsv \
  "CASTLE experiment i — CosMx 2000 HVGs" results/benchmark_v2/cosmx_castle_i

bench_npy data/cosmx_frontal_cortex.h5ad \
  results/embedding_cosmx_hvg2000/castle_ii/z_c.npy \
  results/embedding_cosmx_hvg2000/castle_ii/cells.tsv \
  "CASTLE experiment ii — CosMx 2000 HVGs" results/benchmark_v2/cosmx_castle_ii

conda run --no-capture-output -n ugp-nicheformer python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("results/benchmark_v2")
metrics, direct = [], []
for folder in sorted(root.iterdir()):
    if (folder / "ari_nmi_silhouette.csv").exists():
        metrics.append(pd.read_csv(folder / "ari_nmi_silhouette.csv"))
        direct.append(pd.read_csv(folder / "embedding_direct_metrics.csv"))
pd.concat(metrics, ignore_index=True).to_csv(root / "all_clustering_metrics.csv", index=False)
pd.concat(direct, ignore_index=True).to_csv(root / "all_direct_embedding_metrics.csv", index=False)
print(pd.concat(metrics, ignore_index=True).to_string(index=False))
print(pd.concat(direct, ignore_index=True).to_string(index=False))
PY
