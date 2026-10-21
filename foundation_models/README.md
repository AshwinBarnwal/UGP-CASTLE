# Foundation-model run: Nicheformer and UCE

This bundle targets `data/cosmx_frontal_cortex.h5ad`: 188,686 segmented CosMx
cells, 6,278 assayed genes, raw integer counts in `.X`, published cell labels in
`obs["cell_type"]`, and physical coordinates in `obsm["spatial"]`.

## What is being run

- **Nicheformer** is the spatially pretrained foundation model. We use the
  authors' zero-shot 512-dimensional cell embedding from the final transformer
  layer. Its input vocabulary is fixed at 20,310 Ensembl genes. The adapter maps
  our gene symbols to that order, inserts zeroes conceptually for unmeasured
  genes, applies the authors' CosMx technology mean, ranks expressed genes, and
  supplies the `human`, `CosMx`, and `spatial` context tokens. It does not use
  our published cell-type labels or physical coordinates to create embeddings.
- **UCE** is the non-spatial foundation model. We use the paper-scale 33-layer
  checkpoint and obtain one 1,280-dimensional embedding per cell. UCE receives
  raw counts and gene symbols and internally retains genes for which its human
  protein embeddings and chromosome metadata exist. It does not receive cell
  labels or coordinates.

The 2,000-HVG file is intentionally **not** used. Both pretrained models define
their own gene handling; preselecting HVGs would discard measured signal before
their tokenizers see it and would depart from their intended zero-shot inputs.

## Expected server

Recommended: Linux, NVIDIA Ampere-or-newer GPU, CUDA-capable PyTorch, at least
40 GB GPU memory (80 GB is ideal for UCE-33), 64 GB RAM, and 40 GB free disk.
Smaller GPUs may work with batch size 1-4, but UCE-33 is the limiting job.
Nicheformer uses bf16 by default, so an Ampere-or-newer GPU is preferred.

## One-time server sequence

From the cloned `UGP-CASTLE` repository root:

```bash
bash foundation_models/server/check_server.sh
# Stop here and inspect the GPU/driver output before choosing the CUDA wheel.
# bootstrap.sh defaults to PyTorch cu121; override TORCH_INDEX_URL if needed.
bash foundation_models/server/bootstrap.sh "$PWD"
bash foundation_models/server/download_uce_assets.sh "$PWD"
bash foundation_models/server/prepare.sh "$PWD"
```

First run small smoke tests:

```bash
FOUNDATION_GPU=1 NICHEFORMER_BATCH_SIZE=4 NICHEFORMER_PRECISION=fp16 \
  NICHEFORMER_OUTPUT_SUBDIR=nicheformer_smoke \
  bash foundation_models/server/run_nicheformer.sh "$PWD" 256
FOUNDATION_GPU=1 UCE_BATCH_SIZE=1 bash foundation_models/server/run_uce.sh "$PWD" \
  "$PWD/results/foundation_models/input/cosmx_smoke_256.h5ad"
```

If GPU memory is adequate, run UCE on the full input with a larger batch. Its
preprocessing artifacts are reusable. For the full Nicheformer run, remove the
cell limit and use its default output directory. If interrupted, it resumes
from `progress.json`:

```bash
FOUNDATION_GPU=1 NICHEFORMER_BATCH_SIZE=4 NICHEFORMER_PRECISION=fp16 \
  bash foundation_models/server/run_nicheformer.sh "$PWD"
FOUNDATION_GPU=1 UCE_BATCH_SIZE=1 bash foundation_models/server/run_uce.sh "$PWD"
```

If `check_server.sh` shows a pre-Ampere GPU, set
`NICHEFORMER_PRECISION=fp16`; bf16 is deliberately rejected on unsupported
hardware.

For unattended SSH work, run each command inside `tmux` or submit the equivalent
shell script through the server's scheduler. Do not rely on a plain SSH shell.

## Outputs

- `results/foundation_models/input/input_manifest.json`: validated dataset and
  mapping counts.
- `results/foundation_models/nicheformer/nicheformer_embeddings.npy`: rows are
  cells from `cells.tsv`; columns are the 512 Nicheformer features.
- `results/foundation_models/nicheformer/progress.json`: resumable progress and
  resolved run settings.
- `results/foundation_models/uce/*_uce_adata.h5ad`: processed cells/genes plus
  the 1,280-dimensional embedding in `obsm["X_uce"]`.
- `logs/foundation_models/*.log`: complete console logs.

After both embeddings exist, benchmark each independently with the repository's
standard Leiden grid (neighbors 10, 15, 30, 50; resolutions 0.2-2.0 by 0.2),
K-means, ARI/NMI against non-Unknown published labels, and sampled silhouette.
Any Leiden parameters selected using labels must be reported as supervised
parameter selection. Nicheformer and UCE will not be quantitatively mixed with
methods trained on a different gene-selection cohort without an explicit cohort
label.

```bash
conda run -n ugp-nicheformer python foundation_models/benchmark_embeddings.py \
  --source data/cosmx_frontal_cortex.h5ad \
  --nicheformer results/foundation_models/nicheformer/nicheformer_embeddings.npy \
  --nicheformer-cells results/foundation_models/nicheformer/cells.tsv \
  --uce results/foundation_models/uce/cosmx_frontal_cortex_uce_adata.h5ad \
  --output-dir results/foundation_models/benchmark
```

## Pinned upstream sources

- UCE Git commit: `9c416007be15ad6753dc84af4468c1dc10421ab9`
- Nicheformer source snapshot inspected: Git commit
  `485cadbc5caa15119adfd54228f8a8af835fcabc`
- Inference weights: official Hugging Face model `theislab/Nicheformer`, pinned
  at revision `0ba4ba162f9ad3d6c8b713761954c31be2fa69ed`
- UCE assets: official Figshare article `24320806`, version 5

