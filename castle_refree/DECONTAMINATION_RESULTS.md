# CASTLE decontamination: experiments, results and conclusions

This document records the decontamination work on CASTLE: the data built for it,
the model changes, the tests, every result produced so far, and the conclusions.

**Scope.** Single dataset: Janesick Xenium breast cancer replicate 1 (10x
segmentation, XOA v1.0.1, 15 µm nucleus expansion). Development used one
contiguous 2.2 × 2.2 mm crop. Unless marked otherwise, results are one training
seed (7). The over-correction comparison of M6, λ = 3 and λ = 5 was repeated
with 3 seeds. Nothing has yet been confirmed on the full slide.

---

## 1. Data

### 1.1 Raw export (`scripts/export_xenium_raw.py` → `data/castle_xenium_breast_raw/`)

Built from the 10x output bundle (`Xenium_breast_cancer/`):

| Output | Source |
|---|---|
| `edges_contact.csv`: cell pairs that touch, shared boundary length, centroid distance | segmentation label mask (`cells.zarr.zip`, `masks/1`) |
| `geometry.csv`: area, nucleus/cell ratio, circularity, solidity, contact degree, contact boundary fraction, contact distances, distance to tissue edge | `cells.csv.gz`, masks, polygons |
| `controls.mtx`, control counts in `spatial.csv` | `cell_feature_matrix.h5`, `cells.csv.gz` |
| `counts_nucleus.mtx`: transcripts over the nucleus (evaluation only) | `transcripts.csv.gz` (`overlaps_nucleus`, Q ≥ 20) |

**How "touching" is defined.** Xenium assigns transcripts with a pixel label
mask (0.2125 µm pixels), not the simplified 12-vertex display polygons. Two
cells touch if any of their mask pixels are horizontally or vertically
adjacent. No distance threshold is needed. Shared boundary = adjacent pixel
pairs × 0.2125 µm.

**Checks:**
- mask area vs. 10x `cell_area`: r = 1.0;
- counts rebuilt from transcripts match the feature matrix in every cell (0 mismatches);
- counts identical to the earlier STexampleData export.

**Contact graph:**
- 472,672 touching pairs, median 6 contacts per cell, 472 cells with none;
- only **45%** of the old 12 nearest neighbours actually touch;
- 41% of transcripts lie over the nucleus.

### 1.2 Crop (`scripts/make_xenium_crop.py` → `data/castle_xenium_breast_raw_crop/`)

- 19,615 cells (≥ 5 counts); 18,738 scored, excluding a 30 µm border.
- Window chosen for cell-type mixing: stromal 33%, epithelial 22%, myoepithelial 11%, myeloid 11%, T cells 10%, endothelial 6%, B cells 4%.
- Labels (Janesick, `data/xenium_r1_supervised_labels.csv`) are used only to choose the crop and to score, never for training.

### 1.3 Test datasets (`scripts/simulate_contamination.py` → `data/castle_decontam_sim/`)

| Dataset | Contents | Ground truth |
|---|---|---|
| `null` | the real crop | none (unknown real contamination) |
| `spill` | real crop + transcripts moved between touching cells; leak fraction η set by geometry (mean 0.12), split by shared boundary; 10.9% of transcripts moved | moved counts per cell and gene, η_true |
| `niche` | real crop + LUM, POSTN, CXCL12, CCDC80, PTGDS, MMP2 doubled in 412 stromal cells touching epithelium (no transcript movement) | added counts; must **not** be removed |
| `clean0` | real geometry and total counts; every cell regenerated from its label's pooled profile with NB noise; **no contamination** | correct removal = 0 |
| `cleanspill` | `clean0` + the same geometric spillover (10.9% moved, mean η 0.12) | moved counts, η_true |

### 1.4 Data facts that shaped the model

- **NB, not ZINB.** Within each cell type, NB predicts the observed zero
  fraction almost exactly (e.g. epithelial 0.611 vs 0.611; stromal 0.725 vs
  0.725); only 2 of 870 gene–type combinations exceed it by > 5%. Poisson
  under-predicts zeros. Consistent with Genome Biology 2022 ("Modeling zero
  inflation is not necessary for spatial transcriptomics").
- **Nucleus-only counts are not a replacement for the data.** They keep 40% of
  transcripts (median 59 vs 163 per cell; 34 vs 66 genes; 10% of cells under 20
  counts). They are biased by cell shape: stromal/adipocyte genes keep 13–16%,
  immune genes 73–78%. They are also not clean (3.40% vs 3.67% foreign markers).

---

## 2. Model changes (switches in `config.py`; defaults reproduce the original model)

| Version | Change |
|---|---|
| M0 | original code: 12-NN spillover graph, fixed distance kernel, η learned only from each cell's own counts, free per-cell library size, no background |
| M1 | + spillover only between touching cells (`spillover_graph: contact`) |
| M2 | + learned kernel (distance, shared boundary, relative size) and full geometry features |
| M3 | + donor η and kernel computed live with gradients (`donor_gradient`) |
| M4 | + amortised library size q(ℓ\|x) with an area-based log-normal prior |
| M5 | + background from negative-control probes (spec eq. 13) |
| M6 | + corrected unroll (K real re-encodings), expression-guided spillover off, no η penalty |
| Pλ | M6 + exponential prior on η (`loss.lambda_eta` = λ) |
| Cx | M6 with lower `eta_max` |
| M6_nullcal | M6 with η replaced by η̂ − η̂₀, where η̂₀ is estimated on a contamination-free "null twin" generated from the fitted model |

**Ablations of M6:**
- A1: expression-guided spillover on;
- A2: niche off (same total epochs);
- A3: no contamination model;
- A4: ZINB likelihood.

The niche part is on in every version except A2. Training is 5 warm-up + 15
contamination + 20 joint epochs, β_c 1 → 2.

---

## 3. Tests (`scripts/evaluate_decontamination.py`, `scripts/compare_decontamination.py`)

| Test | What it measures |
|---|---|
| Injected removed (`spill`) | extra removal vs. the `null` fit, as a share of moved transcripts |
| Per-cell / per-gene match | correlation of extra removal with the true moved counts |
| Niche wrongly removed | extra removal of the planted niche counts vs. the baseline removal rate of those genes |
| Counts removed | median per-cell share removed on real data (over-correction guard) |
| Foreign markers | share of a cell's counts from markers of other cell types (markers defined on nucleus counts), split by whether the cell touches a different type |
| Contamination test | Mitchel et al. / SPLIT paper: logistic regression predicting proximity (≤ 15 µm) to type j; number of j's top-20 markers among the top-20 coefficients, averaged over 31 type pairs (lower = cleaner) |
| Nucleus gain | change in per-cell correlation with nucleus-only profiles |
| Clustering ARI | k-means on PCA of corrected counts vs. Janesick labels |
| Chromium similarity | cosine similarity of per-type pseudo-bulk with the Chromium reference |
| clean0 false removal | removal on contamination-free tissue (should be 0) |
| cleanspill recall / precision | share of moved transcripts removed / share of removed transcripts truly moved |

**Caveat:** the Janesick labels were transferred from the Chromium reference
that SPLIT uses, so clustering ARI and Chromium similarity favour SPLIT by
construction.

---

## 4. Results

### 4.1 All methods (seed 7)

| Method | Injected removed | Per-cell / per-gene match | Niche wrongly removed (extra / baseline) | Counts removed | Genes / cell | Foreign markers (other / same type) | Contam. test ↓ | Nucleus gain | k-means ARI | Chromium sim. (T cells) |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | – | – | – | 0% | 65 | 3.7 / 2.2% | 2.84 | – | 0.339 | 0.936 |
| nucleus-only | – | – | – | 58% | 33 | 3.5 / 2.3% | 1.97 | – | 0.299 | 0.942 |
| M0 | 22% | 0.26 / 0.60 | 0.09 / 0.08 | 22% | 60 | 3.0 / 2.2% | 1.97 | +0.006 | 0.365 | 0.943 |
| M1 | 19% | 0.18 / 0.57 | 0.13 / 0.13 | 27% | 59 | 3.0 / 2.3% | 1.84 | +0.007 | 0.387 | 0.943 |
| M2 | 17% | 0.28 / 0.57 | 0.17 / 0.16 | 28% | 58 | 3.0 / 2.3% | 1.81 | +0.008 | 0.372 | 0.943 |
| M3 | 42% | 0.53 / 0.70 | 0.13 / 0.13 | 25% | 60 | 3.1 / 2.3% | 1.90 | +0.007 | 0.354 | 0.943 |
| M4 | 30% | 0.38 / 0.64 | 0.17 / 0.15 | 29% | 59 | 3.0 / 2.3% | 1.77 | +0.008 | 0.361 | 0.942 |
| M6 | 41% | 0.47 / 0.66 | 0.13 / 0.14 | 29% | 58 | 3.0 / 2.2% | 1.77 | +0.008 | 0.388 | 0.943 |
| M6, 80 epochs | 41% | 0.50 / 0.68 | 0.10 / 0.14 | 28% | 58 | – | 1.74 | – | 0.392 | – |
| M6, 160 epochs | 42% | 0.58 / 0.71 | 0.10 / 0.15 | 28% | 59 | 3.0 / 2.2% | 1.61 | +0.008 | 0.363 | 0.944 |
| A1 expression-guided | 33% | 0.36 / 0.61 | 0.18 / 0.15 | 29% | 58 | 3.0 / 2.3% | 1.81 | +0.008 | 0.337 | 0.943 |
| A2 niche off | **50%** | **0.61 / 0.69** | **0.09 / 0.13** | 27% | 59 | 3.0 / 2.2% | 1.81 | +0.008 | 0.341 | 0.944 |
| A3 no contamination | 0% | – | 0 | 0.4% | 65 | 3.7 / 2.2% | 2.77 | 0 | 0.328 | 0.936 |
| A4 ZINB | 19% | 0.36 / 0.61 | 0.12 / 0.17 | 32% | 57 | 3.0 / 2.3% | 1.65 | +0.009 | 0.337 | 0.944 |
| SPLIT default | 8% | 0.02 / 0.16 | 0.03 / 0.15 | 29% | 49 | **2.3 / 1.4%** | 2.65 | −0.046 | 0.372 | 0.957 |
| SPLIT spatial | 9% | 0.03 / 0.13 | 0.01 / 0.12 | 17% | 54 | 2.8 / 2.0% | 2.36 | −0.020 | **0.436** | **0.959** |
| resolVI (10 or 20 neighbours) | ≈0% | ≈0 | ≈0 | 1% | 64 | 3.6 / 2.2% | 2.84–2.90 | +0.002 | 0.347 | 0.936 |
| resolVI native (10 / 20) | ≈0% | 0.00 / 0.41 | **0.39 / 0.37** vs 0 | 0% | 67–68 | 3.7–3.8 / 2.2% | 1.55–1.68 | **−0.18** | 0.345 | 0.936 |

Notes:
- **SPLIT:** RCTD doublet mode + SPLIT v0.3.0 with the Xenium vignette settings, run on each dataset (R 4.6.1, spacexr 2.2.1). It corrected 19,376 of 19,615 cells; RCTD rejects and cells with < 10 counts pass through uncorrected, and so do 6 genes missing from the reference.
- **resolVI:** scvi-tools 1.5.1, unsupervised, 50 epochs. "resolVI" = observed counts × own-source share (same definition as CASTLE); "native" = the tutorial's `generated_expression` (`model_corrected` `px_rate` median).
- **T cells contaminated by tumour cells** (the SPLIT paper's headline pair) could not be scored: only 61 T cells lie within 15 µm of epithelium (minimum 100).

### 4.2 Clustering of embeddings on the crop (best Leiden over resolutions 0.2–2.0, and k-means)

| Embedding / data | Uses labels? | k-means ARI | Best Leiden ARI |
|---|---|---|---|
| raw counts → PCA | no | 0.323 | 0.476 |
| M6 cleaned counts → PCA | no | 0.309 | 0.506 |
| SPLIT spatial cleaned counts → PCA | reference | 0.375 | 0.670 |
| CASTLE z_c (M0 / M6 / A2 / A3) | no | 0.32–0.38 | 0.436–0.457 |
| resolVI latent | no | 0.316 | 0.461 |
| scVI | no | 0.308 | 0.451 |
| scANVI | yes | 0.355 | 0.541 |
| scVIVA | yes | 0.351 | 0.672 |

scVI and scVIVA use the same settings as the ProSeg runs. scANVI was trained
from scVI with the Janesick labels to give scVIVA its expression embedding.

### 4.3 Over-correction (synthetic clean tissue), seed 7

| | clean0 false removal | cleanspill recall | precision | per-cell match | Real removal | Real spill recovered | Real spill per-cell |
|---|---|---|---|---|---|---|---|
| M6 | 11.4% | 61% | 30% | 0.78 | 28.7% | 41% | 0.47 |
| A2 niche off | 8.5% | 57% | 32% | 0.78 | 26.6% | 50% | 0.61 |
| Prior λ=2 | 6.0% | 52% | 33% | 0.79 | 24.7% | 57% | 0.65 |
| Prior λ=3 | 4.1% | 45% | 36% | 0.80 | 22.2% | 65% | 0.76 |
| Prior λ=5 | 1.6% | 31% | 39% | 0.82 | 16.0% | 67% | 0.84 |
| Prior λ=10 | 0.5% | 11% | 43% | 0.74 | 4.6% | 30% | 0.81 |
| Prior λ=30 / 100 | 0.5% | 1% | 18% | 0.01 | 0.4% | 0% | 0.03–0.07 |
| Cap η_max=0.20 | 10.7% | 53% | 33% | 0.79 | 20.6% | 4% | 0.04 |
| Cap η_max=0.10 | 8.8% | 34% | 37% | 0.71 | 11.0% | 1% | −0.17 |
| M6 + null calibration | 4.0% | 45% | 36% | 0.84 | 22.7% | 31% | 0.19 |

The null twin estimated a floor of η ≈ 0.084 on `clean0`, where the true floor
is 0.121, and 0.066–0.076 on the real-data datasets.

### 4.4 Over-correction, 3 seeds (7, 11, 23), mean ± SD

| | M6 | Prior λ=3 | Prior λ=5 |
|---|---|---|---|
| clean0 false removal | 11.2 ± 0.5% | 4.3 ± 0.3% | **1.7 ± 0.1%** |
| cleanspill recall | **62.0 ± 0.6%** | 44.5 ± 0.5% | 31.6 ± 0.6% |
| cleanspill precision | 30.0 ± 0.2% | 35.7 ± 0.2% | **39.1 ± 0.2%** |
| cleanspill per-cell match | 0.78 ± 0.01 | 0.81 ± 0.01 | **0.83 ± 0.01** |
| cleanspill η ranking vs truth | 0.49 ± 0.01 | **0.51 ± 0.05** | 0.33 ± 0.04 |
| cleanspill mean η (true 0.12) | 0.25 | **0.14** | 0.08 |
| Real median cell removed | 28.8 ± 0.3% | 22.5 ± 0.6% | **16.5 ± 0.5%** |
| Real spill injected removed | 39.9 ± 0.5% | 64.3 ± 4.1% | **70.4 ± 2.8%** |
| Real spill per-cell match | 0.48 ± 0.02 | 0.74 ± 0.03 | **0.83 ± 0.02** |
| Real spill per-gene match | 0.68 ± 0.02 | 0.75 ± 0.01 | **0.76 ± 0.01** |
| Niche wrongly removed (extra / baseline) | 0.11 / 0.14 | 0.07 / 0.11 | **0.06 / 0.08** |

Every seed reproduces the same ranking (e.g. clean0 false removal: 10.7–11.7%,
4.1–4.6% and 1.6–1.9%).

---

## 5. Conclusions

The decontamination part of CASTLE now works, and it is the only method tested
that finds contamination we planted ourselves. The niche part still gets in its
way, and the latent z_c is not yet better than plain PCA.

**What works:**
1. **Contacts from the segmentation mask.** Leakage goes only to cells that physically touch.
2. **η trained through donors (M3)** is the fix that makes contamination identifiable: recovery 22% → ~40% (SPLIT 8–9%, resolVI ≈ 0%).
3. **Geometry-only spillover and plain NB** are confirmed. Expression-guided spillover (A1) and ZINB (A4) both recover less planted contamination, and A1 removes more real niche signal.
4. **An exponential prior on η (λ = 3–5) fixes over-correction.** M6 removed about 11% of counts from contamination-free tissue: a constant false floor (η ≈ 2× too high). The prior cuts it to 2–4%, lowers real-data removal from 29% to 17–22%, and improves placement of planted contamination (per-cell 0.48 → 0.74–0.83). Stable across 3 seeds. Capping η_max leaves η stuck at the cap; null calibration underestimates the floor and is noisier.
5. **The correction is spatially specific.** Foreign markers drop only in cells touching a different type (3.7% → 3.0%), not in cells touching their own type (2.2% unchanged).

**What doesn't work yet:**
1. **The niche term competes with spillover.** Niche off recovers 50% vs 41% and removes less real niche signal. Not a training-length effect: M6 at 80 or 160 epochs still recovers 41–42%.
2. **The latent z_c is no better than PCA.** On the same crop: z_c 0.457, scVI 0.451, raw PCA 0.476 (best Leiden ARI). scVI's earlier 0.80 came from the ProSeg dataset and its labels.
3. **Recall vs. false removal.** Stronger priors miss more real contamination on synthetic tissue (recall 62% → 32%). No setting gives both zero false removal and full recall.

**Comparators:**
- **SPLIT** leads on clustering and Chromium similarity, partly because its reference shares origin with the labels. It also removes foreign markers from cells with no foreign neighbour (2.2% → 1.4%) and loses the most genes (65 → 49 per cell).
- **resolVI** as recommended (native regenerated counts) erases about 38% of the planted niche effect and moves profiles away from the nucleus reference (−0.18). Reweighted, it barely corrects (≈ 1–5%). 10 vs 20 neighbours makes no difference.
- **Nucleus-only** discards 60% of the data and clusters worst.

**Next steps:**
1. Make the η prior the default: λ = 5 to minimise over-removal, λ = 3 for sensitivity (λ = 4 untested).
2. Fix the niche/spillover competition, e.g. freeze η and the kernel during the joint phase.
3. Confirm on the full slide, including T cells next to tumour cells.
4. Work on z_c separately (β_c ≥ 1 and training length) before claiming any clustering advantage.

---

## 6. Reproduction

Run from `UGP-CASTLE/`. CASTLE uses `.venv`; resolVI `.venv-scvi`; scVI/scANVI/scVIVA `.venv-scviva`; SPLIT R 4.6.1.

```bash
.venv/Scripts/python.exe castle_refree/scripts/export_xenium_raw.py ../Xenium_breast_cancer --output-dir data/castle_xenium_breast_raw
```

```bash
.venv/Scripts/python.exe castle_refree/scripts/make_xenium_crop.py
```

```bash
.venv/Scripts/python.exe castle_refree/scripts/simulate_contamination.py
```

```bash
.venv/Scripts/python.exe castle_refree/scripts/simulate_contamination.py --only-clean
```

```bash
.venv/Scripts/python.exe castle_refree/scripts/run_decontamination_suite.py --seed 7 --variants null spill niche clean0 cleanspill
```

```bash
.venv-scvi/Scripts/python.exe castle_refree/scripts/run_resolvi_decontam.py --seed 7
```

```bash
"/c/Program Files/R/R-4.6.1/bin/Rscript.exe" castle_refree/R/run_split_decontam.R data/castle_decontam_sim results/decontamination/seed7 4
```

```bash
.venv/Scripts/python.exe castle_refree/scripts/import_split_decontam.py
```

```bash
.venv/Scripts/python.exe castle_refree/scripts/null_calibrate.py --version M6
```

```bash
.venv-scviva/Scripts/python.exe castle_refree/scripts/run_scvi_scviva_crop.py
```

```bash
.venv/Scripts/python.exe castle_refree/scripts/compare_decontamination.py --output results/decontamination/comparison_seed7.csv
```

**Result files** (`results/decontamination/`):
- `summary.csv`: semi-synthetic and clean-tissue scores, all versions and seeds;
- `comparison_seed7.csv`: real-data metrics (last full run before the λ-prior variants);
- `seed*/<version>/<dataset>/`: per-fit outputs;
- `crop_embeddings/embedding_scores.csv`: scVI / scANVI / scVIVA;
- `*.log`: run logs.

**Known issue:** `compare_decontamination.py` currently fails because
`summary.csv` now holds several seeds per model. It needs to filter to the
seed of the fits folder before the λ-prior and null-calibration variants can get
real-data metrics.
