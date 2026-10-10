# CASTLE-RefFree

This repository implements the reference-free, joint denoising-and-embedding
core of the CASTLE design specification, plus an optional expression-guided
spillover modification for the Janesick Xenium breast-cancer dataset.

It does **not** use the Chromium matrix, Chromium cell labels, RCTD weights, or
SPLIT outputs. All expression profiles used to explain contamination are learned
from the same Xenium count matrix being embedded.

## What changed from the CASTLE text

The specification already couples denoising and embedding through an unrolled
purification encoder: a provisional embedding reconstructs each cell, the
generative contamination model produces own-transcript responsibilities, and
the encoder reads the resulting purified counts again.

The deliberate restriction in the text is that the transfer weights
`w[j -> i]` are functions of geometry only. The optional modification here adds
a bounded expression-residual score:

```text
edge_logit(j -> i)
  = -distance(j, i) / distance_scale
    + expression_strength * clipped_cosine(residual_i, learned_profile_j)
```

`residual_i` is the positive part of the observed normalized expression not
explained by cell `i`'s current learned profile. Both residual and donor profile
are projected to a fixed low-dimensional gene space before the edge score is
calculated. The projection is only a computational device; it is not a
reference atlas.

This makes expression evidence alter **where leaked signal probably came
from**, while the negative-binomial model still decides how much signal is
consistent with contamination. The correction is capped because an unrestricted
expression-dependent kernel could misclassify genuine spatial biology as noise.

## Removable switches

In `config/xenium_breast.yaml`:

```yaml
model:
  contamination:
    enabled: true
    expression_guided:
      enabled: true
```

- `expression_guided.enabled: false` runs the geometry-only CASTLE ablation.
- `contamination.enabled: false` runs the embedding without contamination
  correction.

The same switches are available without editing YAML:

```powershell
# Geometry-only kNN contamination ablation
castle-refree --config castle_refree/config/xenium_breast.yaml --geometry-only

# Embedding ablation with the full contamination module removed
castle-refree --config castle_refree/config/xenium_breast.yaml --no-contamination
```

The ablation flags automatically use suffixed result directories so they do not
overwrite the modified model. Use `--output-dir` to choose another location.

Run all three settings with the same seeds before claiming that the modification
helps.

## Implemented model

- Sparse MatrixMarket input; count rows are densified only by minibatch.
- Self-dataset intrinsic encoder and additive intrinsic/niche decoder.
- Self-excluded, three-scale spatial-neighbour context.
- Negative-binomial count likelihood with gene-specific dispersion.
- Directional, mass-conserving spillover weights.
- Geometry-predicted leakage fractions capped by `eta_max`.
- Area-linked regularization of each cell's latent transcript content, preventing
  library size from freely cancelling the leakage estimate.
- Two-step unrolled purification so denoising and embedding update each other.
- Asymmetric intrinsic/niche KL costs, niche sparsity, niche-sufficiency loss,
  and an HSIC penalty.
- Alternating cached-neighbour training to fit within workstation memory.
- Fractional purified count matrix, intrinsic and niche embeddings, estimated
  leakage, own-source responsibilities, and segmentation confidence.

Not yet implemented from the full design document: polygon/shared-boundary
features, transcript-coordinate refinement, explicit merged-cell likelihood
ratio testing, conditional technical-covariate priors, azimuthal Fourier
features, local-isometry trajectory loss, and the communication null simulator.
Those are separate modules and should not be silently approximated.

## Breast Xenium input

Use the same compact Xenium object already cached for the RCTD/SPLIT work. Run
R from the repository root:

```r
source("castle_refree/R/export_xenium_breast.R")
```

This writes a portable bundle to `data/castle_xenium_breast`. It does not
download a dataset and does not read the Chromium reference.

## Refined ProSeg breast Xenium input

The refined H5AD can use the same CASTLE training path. Export its raw
`layers/counts` matrix and the refined `x`/`y` coordinates into the portable
bundle format. The H5AD `cell_type` annotation is saved for evaluation only and
is never supplied to the trainer. Cell areas are absent from this H5AD, so the
exporter records and uses the existing area-of-1 fallback.

From the repository root, install the H5AD extra if needed and export:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\castle_refree[h5ad,evaluation]"
.\.venv\Scripts\python.exe .\castle_refree\scripts\export_h5ad.py `
  .\adata_for_tuto_s1.h5ad `
  --output-dir .\data\castle_xenium_proseg_breast
```

Dry-run, train, analyze, and evaluate the beta-2 experiment:

```powershell
.\.venv\Scripts\castle-refree.exe --config .\castle_refree\config\xenium_proseg_breast_beta2.yaml --dry-run
.\.venv\Scripts\castle-refree.exe --config .\castle_refree\config\xenium_proseg_breast_beta2.yaml
.\.venv\Scripts\python.exe .\castle_refree\scripts\analyze_full.py --input .\data\castle_xenium_proseg_breast --results .\results\castle_refree_xenium_proseg_breast_beta2 --clusters 18
.\.venv\Scripts\python.exe .\castle_refree\scripts\evaluate_proseg_labels.py --input .\data\castle_xenium_proseg_breast --results .\results\castle_refree_xenium_proseg_breast_beta2
```

The exporter reads only the selected sparse count layer rather than materializing
all the dense matrices also stored in the H5AD. The output directory is separate
from the original Xenium bundle and existing experiments.

## Evaluate the existing SPLIT result against ProSeg labels

This reuses the saved SPLIT result; it does not rerun SPLIT. Export its
`purified_counts` if the Python sparse export does not already exist:

```powershell
Rscript .\castle_refree\R\export_split_matrix_for_python.R
```

The H5AD has synthetic `query_*` cell names, while the SPLIT output has original
numeric Xenium barcodes. The ProSeg evaluator matches cells through mutual
nearest spatial coordinates using the original Xenium spatial table. It keeps
pairs within 10 coordinate units by default, writes the matched IDs and
distances for review, excludes `Unlabeled`/blank labels, and evaluates the
remaining 17 classes:

```powershell
.\.venv\Scripts\python.exe .\castle_refree\src\castle_refree\evaluate_split_proseg_labels.py
```

Outputs go to `results/split_proseg_evaluation_python/`. Set
`--max-coordinate-distance` to change the matching radius. Review
`spatial_cell_matches.csv` and the distance summary because this mapping across
different cell segmentations is approximate.

## Run scVIVA directly on the refined ProSeg H5AD

The scVIVA path reads `adata_for_tuto_s1.h5ad` directly and does not use the
CASTLE MatrixMarket export. Its dedicated environment keeps scVIVA dependencies
out of the CASTLE environment. Use Python 3.13 (or another version supported by
the installed scVIVA release, Python 3.12 or newer per its installation guide):

```powershell
.\.venv\Scripts\python.exe -m venv .venv-scviva
```

```powershell
.\.venv-scviva\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
.\.venv-scviva\Scripts\python.exe -m pip install -r .\castle_refree\requirements-scviva.txt
```

Train with the tutorial's data keys and settings (counts layer, sample batch,
cell_type labels, spatial coordinates, X_scANVI embedding, k=20, up to 600
epochs, batch size 512, and learning rate 5e-4):

```powershell
.\.venv-scviva\Scripts\python.exe .\castle_refree\scripts\run_scviva_proseg.py
```

Evaluate the saved `X_scVIVA` latent after excluding `Unlabeled`/`Unlabelled` and
blank labels, using K-means K=17 and Leiden resolutions 0.2 through 2.0 in 0.2
increments:

```powershell
.\.venv-scviva\Scripts\python.exe .\castle_refree\scripts\evaluate_scviva_proseg.py
```

The run writes `model/`, `adata_scviva.h5ad` (including `obsm['X_scVIVA']`),
`training_history.csv`, `run_metadata.json`, and the cell-type UMAP under
`results/scviva_xenium_proseg_breast/`. Evaluation outputs, including the full
Leiden sweep and best ARI/NMI summaries, go under its `label_evaluation/`
subdirectory. scVIVA uses `cell_type` to construct neighborhood features, so
ARI/NMI against those same labels are label-informed descriptive comparisons,
not independent held-out scores.

## Run unsupervised scVI on the refined ProSeg breast H5AD

This separate workflow trains standard scVI directly from the H5AD raw-count
layer. It does not use `cell_type`, `spatial`, or `X_scANVI` as model inputs.
`cell_type` remains in the saved H5AD for evaluation and plots. The first run
uses all 313 genes, a 10-dimensional latent, ZINB likelihood, 512-cell batches,
and early stopping within a 200-epoch maximum. The `sample` column is registered
for batch correction only when it contains at least two categories.

Create a separate environment (scvi-tools currently supports Python 3.12–3.14):

```powershell
.\.venv\Scripts\python.exe -m venv .venv-scvi
```

```powershell
.\.venv-scvi\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
.\.venv-scvi\Scripts\python.exe -m pip install -r .\castle_refree\requirements-scvi.txt
```

Train scVI and save its latent as `obsm['X_scVI']`:

```powershell
.\.venv-scvi\Scripts\python.exe .\castle_refree\scripts\run_scvi_proseg.py
```

Evaluate only labeled cells with K-means K=17 and Leiden resolutions 0.2 through
2.0 in 0.2 increments:

```powershell
.\.venv-scvi\Scripts\python.exe .\castle_refree\scripts\evaluate_scvi_proseg.py
```

Training outputs go to `results/scvi_xenium_proseg_breast/`; evaluation outputs,
including best-ARI and best-NMI Leiden UMAP comparisons, go to its
`label_evaluation/` subdirectory. The evaluator excludes blank and
`Unlabeled`/`Unlabelled` labels, but scVI trains on all cells without using those
labels. Its ARI/NMI scores are therefore independent of label input to training.

If you instead want to export a saved SpatialExperiment:

```r
source("castle_refree/R/export_xenium_breast.R")
```

For command-line R, the optional arguments are output directory and input RDS:

```text
Rscript castle_refree/R/export_xenium_breast.R data/castle_xenium_breast path/to/object.rds
```

## Install and run

From the repository root, create a local virtual environment and install the
package in editable mode:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .\castle_refree
```

Check input and memory path without training:

```powershell
.\.venv\Scripts\castle-refree.exe --config .\castle_refree\config\xenium_breast.yaml --dry-run
```

Run training:

```powershell
.\.venv\Scripts\castle-refree.exe --config .\castle_refree\config\xenium_breast.yaml
```

Resume an interrupted run from the most recent automatic checkpoint:

```powershell
.\.venv\Scripts\castle-refree.exe --config .\castle_refree\config\xenium_breast.yaml --resume .\results\castle_refree_xenium_breast\checkpoint.pt
```

Start with the existing 15,000-cell crop or reduce the exported bundle before a
167,780-cell run. CPU execution is supported but the full dataset will be slow;
CUDA is selected automatically when available.

## Outputs

The configured result directory receives:

- `purified_counts_cells_by_genes.npz`
- `own_responsibility_cells_by_genes.npz`
- `castle_embeddings_and_qc.npz` with `z_intrinsic`, `z_niche`, `eta`,
  `segmentation_confidence`, and `niche_effect_l1`
- `cells.tsv` and `genes.tsv`
- `training_history.csv`
- `castle_refree_model.pt`
- `resolved_config.json`

## Required comparison

The first result is an experiment, not evidence that the modification works.
At minimum compare the three switches above using:

1. held-out reconstruction likelihood;
2. stability over multiple seeds;
3. contamination-marker reduction in small cells near tumour cells;
4. cell/gene/count retention;
5. intrinsic-cluster quality and niche/domain coherence;
6. semi-synthetic known transcript transfers between adjacent cells.

If expression guidance improves fit but worsens semi-synthetic source recovery
or removes plausible niche programs, leave it disabled. That outcome would
support CASTLE's original geometry-only identifiability restriction.
