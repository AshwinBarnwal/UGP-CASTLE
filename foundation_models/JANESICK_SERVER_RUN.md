# Janesick and CASTLE experiment-(ii) server run

## Planned runs

| Dataset | Representation | Input panel | Output evaluated |
|---|---|---:|---|
| Janesick Xenium breast Rep1 | CASTLE experiment i (existing; benchmark only) | 313 genes | intrinsic `z_c` only |
| Janesick Xenium breast Rep1 | CASTLE experiment ii | 313 genes | intrinsic `z_c` only |
| Janesick Xenium breast Rep1 | Nicheformer | all mapped panel genes | 512-D cell embedding |
| Janesick Xenium breast Rep1 | UCE-33 | all UCE-supported panel genes | 1280-D cell embedding |
| CosMx frontal cortex | CASTLE experiment ii | fixed 2,000 HVGs | intrinsic `z_c` only |
| CosMx frontal cortex | CASTLE experiment i (existing; benchmark only) | fixed 2,000 HVGs | intrinsic `z_c` only |

The Janesick target is the publication's supervised label transfer from matched
scFFPE-seq, not histology-verified ground truth. `Unlabeled` is renamed
`Unknown` and excluded from label-based scores.

Experiment ii trains CASTLE's contact-step niche branch, but niche embeddings
are never scored or plotted. Only the intrinsic identity embedding is used.
Janesick uses exact contacts from the official Xenium segmentation mask. CosMx
has no corresponding mask in the distributed H5AD, so it uses a documented
Delaunay/radius proxy graph only for contact-step context; its contamination
configuration remains identical to CosMx experiment i.

Experiment i is not trained again. The suite requires and benchmarks the
existing `results/embedding_full/castle_i` Janesick output and
`results/embedding_cosmx_hvg2000/castle_i` CosMx output.

## Transfer from Windows PowerShell

```powershell
cd D:\ugp\UGP-CASTLE

scp -r foundation_models `
  nischay@172.27.21.79:~/UGP-CASTLE/

scp -r castle_refree/scripts castle_refree/src `
  nischay@172.27.21.79:~/UGP-CASTLE/castle_refree/

scp data/janesick_breast.h5ad data/janesick_breast.manifest.json `
  nischay@172.27.21.79:~/UGP-CASTLE/data/

scp -r data/cosmx_frontal_cortex_hvg2000 `
  nischay@172.27.21.79:~/UGP-CASTLE/data/
```

These recursive copies merge the matching directories into the remote
repository; they do not replace unrelated files.

## Launch once on the server

```bash
ssh nischay@172.27.21.79
cd "$HOME/UGP-CASTLE"
mkdir -p logs
tmux new -s janesick-suite
set -o pipefail
bash foundation_models/server/run_new_suite.sh "$PWD" 2>&1 | tee logs/janesick_suite.log
```

Detach with `Ctrl-B`, then `D`. The master script is sequential so UCE,
Nicheformer, and CASTLE do not compete for the same GPU. It is safe to rerun:
downloads resume, Nicheformer and CASTLE checkpoint/resume, and UCE skips a
completed output.

## Monitor

```bash
tail -f logs/janesick_suite.log
tail -f logs/castle/results_embedding_janesick_castle_i.log
tail -f logs/castle/results_embedding_janesick_castle_ii.log
tail -f logs/foundation_models/nicheformer_janesick.log
tail -f logs/foundation_models/uce_janesick.log
```

## Results

Every benchmark directory under `results/benchmark_v2/` contains:

- `umap_leiden_optimized_vs_truth.png`
- `umap_leiden_fixed_vs_truth.png`
- `umap_kmeans_vs_truth.png`
- `ari_nmi_silhouette.csv`
- `embedding_direct_metrics.csv`
- `leiden_parameter_sweep.csv`
- `cell_assignments.csv.gz`
- `benchmark_manifest.json`

Combined score tables are written to:

- `results/benchmark_v2/all_clustering_metrics.csv`
- `results/benchmark_v2/all_direct_embedding_metrics.csv`
