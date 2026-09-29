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
