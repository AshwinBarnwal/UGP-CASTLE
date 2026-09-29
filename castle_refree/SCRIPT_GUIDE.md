# CASTLE-RefFree implementation, scripts, and results guide

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

## What differs between the completed implementations

All four serious crop experiments use exactly the same 14,996 filtered Xenium
cells, 312 retained genes, 12-neighbor spatial graph, network dimensions,
training schedule, optimizer, random seed, and maximum leakage fraction of
0.35. Consequently, their differences can be attributed to the listed ablation
settings rather than different input cells.

The common raw baseline is also identical in every crop comparison: raw counts
are normalized to 10,000 counts per cell, transformed with `log1p`, and reduced
to 16 dimensions with truncated SVD. CASTLE is evaluated through its native
16-dimensional intrinsic embedding.

| Result name | Expression-guided edges | Final intrinsic beta | Eta penalty | What changed |
|---|---:|---:|---:|---|
| `expression_beta4` | Yes | 4 | 0.01 | Original reference-free implementation used as the first serious baseline. |
| `geometry_only` | No | 4 | 0.01 | Keeps denoising and spillover estimation, but edge weights depend only on distance. |
| `expression_beta2` | Yes | 2 | 0.01 | Weakens compression of the intrinsic latent representation. This was selected for the full run. |
| `expression_beta4_eta05` | Yes | 4 | 0.05 | Penalizes predicted leakage five times more strongly than the beta-4 baseline. |
| `15000_smoke` | Yes | 4 | 0.01 | Only five total epochs. This checks that the pipeline executes; it is not a benchmark result. |
| full `expression_beta2` | Yes | 2 | 0.01 | Applies the selected crop configuration to 164,995 filtered cells and 313 genes. |

Here, intrinsic beta is the coefficient on the intrinsic latent KL term. A
higher value forces the latent distribution closer to a standard normal and
therefore imposes a stronger information bottleneck. Lowering the final value
from 4 to 2 lets the intrinsic embedding retain more expression information.
The eta penalty discourages the model from explaining counts as spatial
spillover. Expression-guided edges alter which nearby donor is considered most
plausible by comparing a donor's decoded expression profile with the positive
residual expression in a neighboring recipient.

## Meaning of every reported measure

### Count-removal and model-fit measures

- **Fraction of counts removed:** the total fractional count mass assigned to
  contamination divided by the original count total. More is not automatically
  better: zero can mean no correction, while excessive removal can erase true
  biology.
- **Eta:** a cell-level model estimate controlling how much of that cell's
  expression can contribute to neighboring cells. It is capped at 0.35 here.
  Eta is a model parameter, not a measured ground-truth contamination rate.
- **Segmentation confidence / cell-count retention:** purified total counts
  divided by raw total counts for a cell. A value of 0.78 means approximately
  78% of its observed count mass was attributed to the cell itself.
- **Negative-binomial reconstruction loss:** disagreement between observed
  counts and the model's own-expression plus incoming-contamination expectation.
  Lower is better for fit, but a flexible model can fit well without learning
  the desired biology.
- **Intrinsic KL:** strength of departure of the intrinsic latent distribution
  from its standard-normal prior. Its raw value is not a quality score. It is
  expected to rise when beta is lowered because the model is allowed to encode
  more information.
- **Niche KL and mean niche-dimension standard deviation:** indicate how much
  information/variation is being used in the niche latent. Values near zero can
  indicate that the niche branch has collapsed or is contributing very little.

### Unsupervised cluster-geometry measures

- **Silhouette:** for each point, compares cohesion with its own assigned cluster
  against distance to the nearest alternative cluster. It ranges from -1 to 1;
  higher is better. Here it assesses K-means cluster geometry, not biological
  correctness.
- **Davies–Bouldin index:** compares within-cluster scatter to between-cluster
  separation. Lower is better. Its scale depends on the data and number of
  clusters.
- **Calinski–Harabasz index:** between-cluster dispersion divided by
  within-cluster dispersion. Higher is better, but it scales with sample size
  and therefore should not be compared directly between the crop and full run.

### RCTD-label agreement measures

- **Adjusted Rand index (ARI):** agreement between pairs of cells placed
  together/apart by K-means and by RCTD, corrected for chance. One is perfect,
  zero is chance-level expectation, and negative values are possible.
- **Normalized mutual information (NMI):** shared information between K-means
  clusters and RCTD labels, normalized to 0–1. It is less sensitive than ARI to
  some differences in cluster size and fragmentation.
- **15-NN label purity:** fraction of the 15 nearest neighbors in embedding space
  that share the query cell's RCTD label. Higher means stronger local cell-type
  organization; it can be inflated by abundant cell types.
- **RCTD-label silhouette:** silhouette calculated using RCTD labels themselves
  rather than K-means assignments. It asks whether externally labelled cell
  types form compact, separated regions. This is a stricter test than merely
  obtaining compact K-means clusters.

RCTD labels are evaluation targets only; the training code never receives
them. Nevertheless, RCTD is still an estimated label source rather than
experimental ground truth, and it is derived from the same Xenium expression
data plus a Chromium reference. These measurements show agreement with RCTD,
not proof of perfect biological identity or perfect decontamination.

## Crop results and interpretation

| Implementation | Removed | Reconstruction | Unsupervised silhouette | ARI | NMI | 15-NN purity | Label silhouette |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw 16D expression baseline | — | — | — | 0.2876 | 0.5658 | 0.7422 | 0.0511 |
| Expression beta 4 | 20.72% | 189.03 | 0.2887 | 0.2858 | 0.5556 | 0.7545 | 0.0443 |
| Geometry only | 19.41% | 189.09 | 0.3641 | 0.2855 | 0.5531 | 0.7571 | 0.0476 |
| Expression beta 2 | 17.65% | 185.00 | 0.3181 | 0.3347 | 0.6114 | 0.7761 | 0.1034 |
| Expression beta 4 + stronger eta penalty | 20.18% | 188.99 | 0.3167 | 0.2616 | 0.5513 | 0.7549 | 0.0447 |

### Expression-guided beta 4

This implementation removes 20.72% of total count mass. Relative to raw 16D
expression, local 15-NN label purity rises from 0.7422 to 0.7545, but ARI, NMI,
and label silhouette are slightly worse. Therefore the first expression-guided
implementation improves local same-label neighborhoods but does not improve
global recovery or separation of the 17 RCTD types. Its stronger beta-4
bottleneck is a plausible reason that useful cell-type information was lost.

### Geometry-only beta 4

This changes only the edge-weight mechanism: contamination still exists, but a
nearby donor is judged by distance without expression-residual evidence. It
removes slightly less count mass than expression beta 4 (19.41% versus 20.72%).
Its unsupervised K-means silhouette rises from 0.2887 to 0.3641, yet ARI and NMI
remain essentially unchanged and still do not beat raw expression. This is an
important distinction: it creates more compact model-defined clusters without
making them more biologically concordant with RCTD. Expression guidance at
beta 4 therefore did not demonstrate a meaningful label-level advantage in
this crop.

### Expression-guided beta 2

Lowering only the final intrinsic beta from 4 to 2 produces the clearest gain.
Compared with raw expression:

- ARI rises by 0.0471, from 0.2876 to 0.3347;
- NMI rises by 0.0457, from 0.5658 to 0.6114;
- 15-NN purity rises by 0.0339, from 0.7422 to 0.7761;
- label silhouette rises by 0.0523, from 0.0511 to 0.1034.

It also has the lowest reconstruction loss (185.00) and removes less count mass
(17.65%) than any other serious crop variant. This supports the interpretation
that beta 4 over-compressed the intrinsic embedding and that beta 2 preserved
more cell-type signal while requiring less aggressive correction.

There is a tradeoff: mean niche-dimension standard deviation falls from about
0.206 in expression beta 4 to 0.049, and final niche KL falls from 0.779 to
0.029. The beta-2 run's niche representation is therefore weak or partly
collapsed. It is the best current **intrinsic cell-type embedding**, but it is
not the strongest demonstration of niche representation learning.

### Expression-guided beta 4 with stronger eta penalty

Increasing `lambda_eta` from 0.01 to 0.05 modestly lowers estimated leakage and
count removal relative to expression beta 4 (20.18% versus 20.72%), as intended.
It does not improve reconstruction or RCTD agreement. ARI falls to 0.2616, while
NMI and label silhouette remain below the raw baseline. The change is too small
to solve beta-4 over-compression, and the worse ARI suggests that globally
discouraging leakage is not by itself the required fix.

### Why the raw baseline is repeated

Every crop variant is compared with the same cells and same raw count matrix,
so the raw baseline values are identical in the summary. This is deliberate:
the changing values come from CASTLE's learned embedding, not from resampling or
changing the reference labels.

## Selected full-run result

The beta-2 implementation was carried forward because it was the only crop
variant that improved all four RCTD-agreement measures over raw expression. The
full run contains 164,995 filtered cells and 313 genes. It removes 19.42% of
total count mass, with median cell retention of 0.7831.

The label comparison uses the same seed to select 30,000 of the 163,849 cells
having RCTD labels and compares 17 cell types:

| Representation | ARI | NMI | 15-NN purity | Label silhouette |
|---|---:|---:|---:|---:|
| Raw normalized 16D expression | 0.3295 | 0.5478 | 0.7240 | 0.0328 |
| Full CASTLE beta-2 intrinsic | 0.4057 | 0.6031 | 0.7668 | 0.0966 |
| Absolute improvement | +0.0763 | +0.0553 | +0.0428 | +0.0639 |

Thus the full intrinsic embedding agrees more strongly with RCTD at both global
cluster level (ARI/NMI) and local neighborhood/separation level (purity and
label silhouette). The improvements are larger than in the crop for ARI and
label silhouette. This could reflect the benefit of training on more cells and
more examples of each population, but crop and full results are not a controlled
single-variable comparison because MiniBatchKMeans and sampling are used for
the full evaluation.

The full run's internal K-means scores are silhouette 0.1942,
Davies–Bouldin 1.5509, and Calinski–Harabasz 700.73 on a 5,000-cell sample.
These establish that clusters exist in the latent geometry, but the RCTD-based
metrics are the more relevant evidence that those clusters correspond to the
reference cell-type structure.

## What the current results do and do not establish

The current evidence supports three specific statements:

1. Lowering intrinsic beta from 4 to 2 materially improved preservation of
   RCTD cell-type structure.
2. On this dataset, expression-guided edge correction at beta 4 did not clearly
   outperform geometry-only correction on label-level metrics.
3. Stronger global eta regularization did not improve the embedding.

It does not yet establish that CASTLE outperforms SPLIT or scVIVA. The present
raw-versus-CASTLE comparison uses a custom 16-dimensional K-means benchmark.
SPLIT's paper evaluation uses a broader scIB-style protocol and corrected-count
representations. A fair method comparison must hold the cells, RCTD labels,
normalization, PCA dimensionality, clustering algorithm, sampling, and excluded
cell types constant for raw, SPLIT, CASTLE, and scVIVA.
