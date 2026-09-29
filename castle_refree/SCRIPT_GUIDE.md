# CASTLE-RefFree script guide

This document describes the executable scripts, package modules, configuration
files, their inputs and outputs, and how they fit together. Paths are relative
to the repository root.

## End-to-end flow

1. `castle_refree/R/export_xenium_breast.R` converts the cached Xenium object
   into a portable count-and-coordinate bundle under `data/`.
2. A YAML file in `castle_refree/config/` selects that input bundle and defines
   graph, model, loss, and training settings.
3. The `castle-refree` command enters through
   `castle_refree/src/castle_refree/cli.py`, loads the data, constructs the
   spatial graph, trains the model, and exports embeddings and purified counts.
4. `analyze_crop.py` or `analyze_full.py` produces quality-control statistics,
   exploratory clusters, UMAP coordinates, and plots.
5. RCTD labels are exported separately and used only by
   `evaluate_crop_labels.py` or `evaluate_full_labels.py` to measure whether the
   learned intrinsic embedding agrees with an external cell-type annotation.
6. `summarize_crop_ablations.py` combines the crop experiments into one table
   and comparison figure.

## Data preparation scripts

### `castle_refree/R/export_xenium_breast.R`

Purpose: export Xenium counts and spatial metadata in a format the Python code
can read without requiring Bioconductor during model training.

Input:

- By default, the cached
  `STexampleData::Janesick_breastCancer_Xenium_rep1()` object.
- Optionally, the first command-line argument specifies the output directory.
- Optionally, the second argument specifies a local RDS spatial object instead
  of loading `STexampleData`.

Processing:

- Selects the `counts` assay when present.
- Extracts cell coordinates from `SpatialExperiment::spatialCoords()` or known
  coordinate columns in `colData`.
- Extracts cell area when available; otherwise writes area `1` and warns.
- Does not fabricate negative-control measurements and does not use Chromium or
  RCTD reference data.

Output bundle:

- `counts.mtx`: genes by cells sparse raw counts.
- `genes.tsv`: gene names in matrix-row order.
- `cells.tsv`: cell identifiers in matrix-column order.
- `spatial.csv`: cell identifier, coordinates, and area.
- `manifest.R`: provenance and dimensions.

### `castle_refree/R/export_full_rctd_labels.R`

Purpose: extract primary RCTD labels for evaluation of the full CASTLE run.

Input: `data/split_inputs_full.rds`, specifically its `primary` vector and
`weights` row names as a fallback source of cell identifiers.

Output:
`results/castle_refree_xenium_breast_beta2/rctd_labels.csv`, containing
`cell_id` and `primary`. These labels are not fed into CASTLE training.

## Package entry point and configuration

### `castle_refree/src/castle_refree/cli.py`

Purpose: command-line entry point installed as `castle-refree`.

It loads a YAML configuration, optionally overrides the output directory,
applies ablation switches, loads and filters the spatial data, creates a
`CastleTrainer`, and either performs a dry run or trains and exports results.

Important switches:

- `--config`: required YAML configuration.
- `--dry-run`: validates loading and graph construction without training.
- `--resume`: resumes an interrupted run from `checkpoint.pt`.
- `--geometry-only`: disables expression-guided edge correction while retaining
  the contamination/purification model.
- `--no-contamination`: disables the complete purification module.
- `--output-dir`: overrides the configured result directory.

### `castle_refree/src/castle_refree/config.py`

Purpose: define all configuration fields as nested dataclasses and load a YAML
file into them.

The groups are:

- `data`: input/output directories and count filters.
- `graph`: neighbor count and distance-kernel controls.
- `model`: network dimensions, dropout, and contamination settings.
- `loss`: weights for reconstruction, latent regularization, niche separation,
  contamination, and library-size anchoring.
- `train`: phase lengths, batch sizes, optimizer settings, seed, device, and
  checkpoint interval.

Unknown YAML keys are rejected. Relative data paths are resolved relative to
the YAML file, so configurations continue to work after moving the repository.

### `castle_refree/config/*.yaml`

Each YAML file is a complete experiment specification:

- `xenium_breast.yaml`: full-data baseline with intrinsic KL weight ending at
  beta 4.
- `xenium_breast_beta2.yaml`: selected full-data run with beta ending at 2.
- `xenium_breast_15000.yaml`: 15,000-cell beta-4 crop.
- `xenium_breast_15000_beta2.yaml`: 15,000-cell beta-2 crop.
- `xenium_breast_15000_eta_reg.yaml`: crop with a stronger leakage penalty
  (`lambda_eta = 0.05`).
- `xenium_breast_15000_smoke.yaml`: short five-epoch integration run.

The geometry-only crop uses the appropriate crop YAML plus the
`--geometry-only` command-line switch, rather than a separate YAML file.

### `castle_refree/src/castle_refree/__init__.py`

Purpose: expose the package's public Python API—configuration loading, spatial
data loading, and `CastleTrainer`—and define package version `0.1.0`.

## Core model implementation

### `castle_refree/src/castle_refree/data.py`

Purpose: read the portable export into a `SpatialCountData` object.

It recognizes either genes-by-cells or cells-by-genes MatrixMarket orientation
and standardizes internally to cells by genes. It aligns coordinates by cell
identifier, filters cells below `min_counts`, filters genes detected in fewer
than `min_cells_per_gene` cells, validates finite geometry, and preserves
optional cell-level negative-control totals.

### `castle_refree/src/castle_refree/graph.py`

Purpose: build the spatial neighborhood graph and geometry features.

`build_spatial_graph()` uses a KD-tree to find the configured number of nearest
spatial neighbors for every cell. Extremely long edges are capped at a chosen
distance quantile. It stores both outgoing neighbor arrays and an incoming-edge
index, because contamination is modeled as transcripts travelling from donor
cells into recipient cells.

`geometry_features()` derives standardized log cell area, mean neighbor
distance, nearest-neighbor distance, and local-density features. These features
are used to estimate each cell's leakage fraction.

### `castle_refree/src/castle_refree/model.py`

Purpose: define the neural model, observation model, purification calculation,
and loss.

Main quantities:

- `z_intrinsic`: latent representation intended to retain cell-intrinsic gene
  expression.
- `z_niche`: latent representation of local spatial context.
- `mu`: decoded per-cell gene probability profile.
- `eta`: geometry-predicted fraction of transcripts a cell contributes as
  spillover, bounded by `eta_max`.
- `incoming`: expected neighbor-derived contamination for each recipient cell.
- `own_responsibility`: for every cell and gene, the expected own-cell rate
  divided by total expected rate.
- `purified`: observed counts multiplied by own responsibility. These are
  fractional estimated own-cell counts, not library-normalized values.

The count likelihood is negative binomial. The total loss combines count
reconstruction, KL penalties for both latent spaces, niche-effect sparsity,
niche-neighbor reconstruction, HSIC separation of intrinsic and niche
representations, an eta penalty, and a library-size prior penalty. Training is
stochastic, while exported embeddings use posterior means.

### `castle_refree/src/castle_refree/trainer.py`

Purpose: orchestrate scalable graph-coupled training and export all results.

The trainer:

- selects CPU or CUDA and fixes random seeds;
- builds the spatial graph and initializes cell library sizes;
- creates cached donor embeddings, decoded profiles, leakage estimates, and
  edge weights so a full graph does not have to be differentiated at once;
- optionally adjusts distance-based edge weights with expression evidence: a
  neighbor is upweighted when its decoded profile resembles the recipient's
  positive expression residual;
- refreshes spatial context from purified rather than raw cell states after
  warm-up;
- trains in three phases: intrinsic warm-up, contamination estimation, then
  joint intrinsic/niche learning;
- saves resumable checkpoints at the configured interval.

Exports include:

- `purified_counts_cells_by_genes.npz`;
- `own_responsibility_cells_by_genes.npz`;
- `castle_embeddings_and_qc.npz` with intrinsic/niche embeddings, eta,
  segmentation confidence, and niche-effect magnitude;
- aligned cell and gene identifiers;
- training history, resolved configuration, and final model checkpoint.

## Analysis and clustering scripts

### `castle_refree/scripts/analyze_crop.py`

Purpose: analyze the 15,000-cell experiments.

It runs K-means directly on the learned latent vectors—not on UMAP coordinates:

- intrinsic embedding: requested cluster count, default 17;
- niche embedding: fixed at 12 clusters;
- 20 K-means initializations and seed 7.

UMAP is a visualization step only (`n_neighbors=30`, `min_dist=0.25`, cosine
distance, seed 7). The script calculates count-retention and leakage summaries,
and evaluates intrinsic K-means internally with silhouette, Davies–Bouldin, and
Calinski–Harabasz scores. Silhouette uses a reproducible sample of at most 5,000
cells; the other two crop scores use all crop cells.

Outputs: `crop_diagnostics.json`, `embedding_coordinates.csv.gz`, the six-panel
embedding overview, the training-diagnostic plot, and printed diagnostics.

### `castle_refree/scripts/analyze_full.py`

Purpose: perform the equivalent analysis without forcing full-data UMAP into
memory.

It fits MiniBatchKMeans to all latent vectors (default 17 intrinsic clusters,
12 niche clusters; batch size 4,096; 10 initializations; seed 7). It then draws
a reproducible sample of at most 30,000 cells for UMAP and plotting. Silhouette,
Davies–Bouldin, and Calinski–Harabasz are calculated on a separate sample of at
most 5,000 cells.

Outputs: `full_diagnostics.json`, sampled embedding coordinates, the full-run
overview figure, and the training-diagnostic figure.

### `castle_refree/scripts/evaluate_crop_labels.py`

Purpose: compare crop embeddings to external RCTD cell-type labels.

It aligns cells by identifier, evaluates the 16-dimensional CASTLE intrinsic
embedding, and builds a dimension-matched raw-expression baseline by total-count
normalizing each cell to 10,000, applying `log1p`, and running 16-component
truncated SVD. For each representation it:

- runs 17-class (or, exactly, number-of-observed-types) K-means with 20 starts;
- compares cluster IDs to RCTD primary labels using ARI and NMI;
- calculates 15-nearest-neighbor label purity;
- calculates silhouette using the RCTD labels on at most 5,000 cells.

It also colors the precomputed CASTLE UMAP by RCTD primary type and RCTD
confidence class. Outputs are `rctd_label_embedding_metrics.json` and
`castle_crop_rctd_label_overlay.png`.

### `castle_refree/scripts/evaluate_full_labels.py`

Purpose: run the same external-label comparison on a reproducible full-data
sample (30,000 cells by default).

It uses MiniBatchKMeans rather than standard K-means and uses 10 starts, but
otherwise calculates the same ARI, NMI, 15-NN purity, and label-silhouette
statistics against a 16-component raw-expression baseline. It writes
`rctd_label_embedding_metrics.json`.

### `castle_refree/scripts/summarize_crop_ablations.py`

Purpose: compare the four completed crop experiments: expression-guided beta 4,
geometry-only, expression-guided beta 2, and stronger eta regularization.

It reads each experiment's diagnostics, RCTD-label metrics, and final training
epoch. It writes `results/castle_refree_crop_ablation_summary.csv` and a
six-panel bar chart covering removal fraction, reconstruction, niche variation,
ARI, NMI, and neighbor-label purity. Dashed lines show the raw-expression PCA
baseline for label-based panels.

## Test and packaging files

### `castle_refree/tests/test_smoke.py`

Purpose: fast end-to-end correctness test on a synthetic 24-cell, 8-gene
dataset with two expression profiles. It runs one epoch per training phase,
exports results, and verifies that purified counts exist and that intrinsic
embeddings have the expected finite shape.

### `castle_refree/pyproject.toml`

Purpose: package and installation metadata. It declares the Python dependencies,
maps packages from `src/`, and installs `castle-refree` as the command-line
entry point for `castle_refree.cli:main`.

### `castle_refree/requirements.txt`

Purpose: convenient environment requirements for running training and the
analysis scripts, including plotting, UMAP, scikit-learn, and testing packages
that are not all mandatory in the minimal install metadata.

## Interpreting the current clustering comparison

The plotted clusters and the RCTD agreement scores answer different questions.

- The exploratory plots use unsupervised K-means or MiniBatchKMeans fitted to
  CASTLE's latent vectors. UMAP only places points in two dimensions for display;
  it does not determine the clusters.
- Silhouette, Davies–Bouldin, and Calinski–Harabasz in the analysis scripts are
  internal geometry measures using the model's own cluster assignments. They do
  not establish that a cluster is a correct biological cell type.
- ARI and NMI compare unsupervised cluster assignments with RCTD primary labels.
- Label silhouette ignores K-means assignments and asks whether cells sharing an
  RCTD label occupy compact, separated regions in the embedding.
- 15-NN label purity asks what fraction of each cell's 15 nearest embedding
  neighbors share its RCTD label.
- The raw baseline is deliberately reduced to the same 16 dimensions as the
  CASTLE intrinsic embedding before applying the same metrics.

These RCTD-based results are useful diagnostic comparisons, but they are not yet
an exact reproduction of the SPLIT paper's scIB benchmark. A paper-matched
comparison must use identical cells and labels for every method, the same count
normalization and 50-PC representation where appropriate, the same malignant
cell exclusions, and the same scIB Leiden/K-means and biological-conservation
metrics.
