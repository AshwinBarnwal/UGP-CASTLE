# SPLIT versus CASTLE clustering benchmark

## What was compared

The benchmark uses the same Xenium breast-cancer cells, the same 306 shared genes, and the same RCTD primary-cell labels for every representation. The original 15,000-cell development crop used to choose the CASTLE configuration was excluded. The primary benchmark then samples 100,000 of the remaining cells with seed 0.

Four representations are compared:

1. **Raw PCA50:** raw Xenium counts, library-size normalized to 10,000 counts per cell, `log1p` transformed, then reduced to 50 dimensions with Scanpy PCA.
2. **SPLIT PCA50:** SPLIT-purified counts with exactly the same normalization and PCA50 procedure.
3. **CASTLE-purified PCA50:** CASTLE's reconstructed/purified counts with exactly the same normalization and PCA50 procedure.
4. **CASTLE intrinsic z16:** CASTLE's learned 16-dimensional intrinsic posterior mean, used directly. No PCA or SVD is applied to it.

This fixes the earlier fairness concern: the count-based baselines are no longer compressed with a convenient 16-dimensional truncated SVD. They all receive the same standard 50-PC preprocessing. CASTLE's native vector remains 16-dimensional because that is the actual model output being evaluated.

## Clustering protocol

The protocol follows the clustering/evaluation structure used by SPLIT's published analysis:

- scIB-metrics/JAX K-means with `K` equal to the number of RCTD reference types.
- A 15-nearest-neighbour graph followed by Leiden clustering.
- Leiden resolution searched from 0.2 through 2.0 in increments of 0.2; the result with the highest NMI against the RCTD labels is reported.
- A 90-nearest-neighbour graph for cLISI.
- ARI, NMI, label silhouette, cLISI, isolated-label score, Calinski-Harabasz, and Davies-Bouldin metrics.

K-means, Leiden, graph construction, cLISI, Calinski-Harabasz, and Davies-Bouldin use the whole evaluation cohort. Because pairwise silhouette calculations are quadratic and this machine has 16 GB RAM, silhouette and isolated-label metrics use the same fixed 10,000-cell subsample for every representation. SPLIT's published code calculates those metrics on the full cohort, so this is the one deliberate resource-driven deviation.

## Primary result: tumours included

Cells: 100,000. Genes: 306. RCTD types / K-means clusters: 17.

| Representation | K-means ARI | K-means NMI | Leiden ARI | Leiden NMI | Label silhouette | cLISI | Isolated labels | Label CH | Label DB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw PCA50 | 0.367 | 0.569 | 0.457 | 0.618 | 0.510 | 0.979 | 0.521 | 5,176 | 3.651 |
| SPLIT PCA50 | **0.454** | **0.702** | **0.722** | **0.809** | 0.527 | **0.996** | 0.536 | 6,419 | 3.157 |
| CASTLE-purified PCA50 | 0.384 | 0.587 | 0.438 | 0.613 | 0.511 | 0.981 | 0.525 | 5,517 | 3.509 |
| CASTLE intrinsic z16 | 0.443 | 0.624 | 0.462 | 0.615 | **0.554** | 0.992 | **0.599** | **9,368** | **1.681** |

Higher is better for every column except Davies-Bouldin (DB), where lower is better.

SPLIT gives the clearest agreement with the RCTD labels, especially with Leiden. CASTLE's intrinsic embedding almost reaches SPLIT's K-means ARI (0.443 versus 0.454), but it does not reproduce SPLIT's strong graph-clustering agreement. Conversely, CASTLE intrinsic has the strongest continuous geometry: its reference types are more compact/separated according to silhouette, isolated-label, CH, and DB scores.

CASTLE-purified counts show a modest K-means improvement over raw counts, but not a consistent Leiden improvement. The useful CASTLE result is therefore primarily its learned intrinsic embedding, not its reconstructed-count representation.

## Sensitivity result: tumour/DCIS labels excluded

The SPLIT repository's code filters labels containing the literal word `malignant`. Our RCTD labels do not contain that word, so the primary run retains tumour cells. A second explicit sensitivity run removes `Invasive_Tumor`, `Prolif_Invasive_Tumor`, `DCIS 1`, and `DCIS 2`.

Cells: 86,364. Genes: 306. RCTD types / K-means clusters: 13.

| Representation | K-means ARI | K-means NMI | Leiden ARI | Leiden NMI | Label silhouette | cLISI | Isolated labels | Label CH | Label DB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw PCA50 | 0.248 | 0.488 | 0.375 | 0.553 | 0.517 | 0.983 | 0.529 | 3,217 | 3.294 |
| SPLIT PCA50 | **0.527** | **0.757** | **0.880** | **0.868** | 0.538 | **0.999** | 0.546 | 4,618 | 2.758 |
| CASTLE-purified PCA50 | 0.309 | 0.531 | 0.375 | 0.563 | 0.518 | 0.986 | 0.532 | 3,522 | 3.150 |
| CASTLE intrinsic z16 | 0.359 | 0.581 | 0.326 | 0.563 | **0.557** | 0.994 | **0.611** | **8,198** | **1.606** |

The same conclusion survives this sensitivity check: SPLIT best reproduces the RCTD-defined partition, while CASTLE intrinsic gives the best continuous embedding geometry.

## Interpretation and limitation

The RCTD labels are not independent ground truth for SPLIT. SPLIT directly receives RCTD weights, reference profiles, and primary labels, and then purifies the counts using them. Scoring against those same primary labels therefore favours SPLIT by design. This benchmark is still useful because it reproduces the paper's biological-conservation test and measures whether CASTLE preserves the same cell-type structure, but it should not be presented as an independent proof that SPLIT is biologically more correct.

The fair summary is:

- **Best match to RCTD labels:** SPLIT PCA50.
- **Best continuous embedding separation:** CASTLE intrinsic z16.
- **CASTLE reconstructed counts:** small improvement over raw counts in some metrics, not a decisive result.
- **No unfair 16D reduction of baselines:** all count matrices use PCA50; only CASTLE's native learned representation is 16D.

## Reproduction

The SPLIT export uses `Rterm.exe` because `Rscript.exe` crashes on this Windows installation:

```powershell
$env:R_LIBS_USER = 'D:/R/library/4.4'
& 'D:\R\R-4.4.2\bin\x64\Rterm.exe' --vanilla --slave -f castle_refree/R/export_split_benchmark_counts.R
```

Create the isolated benchmark environment and run both evaluations:

```powershell
py -3.12 -m venv D:/ugp/.venv-split-benchmark
D:/ugp/.venv-split-benchmark/Scripts/python.exe -m pip install -r castle_refree/requirements-benchmark.txt

D:/ugp/.venv-split-benchmark/Scripts/python.exe castle_refree/scripts/benchmark_split_castle.py --output results/benchmark_split_castle_full

D:/ugp/.venv-split-benchmark/Scripts/python.exe castle_refree/scripts/benchmark_split_castle.py --exclude-tumour --output results/benchmark_split_castle_full_tumour_excluded
```

Machine-readable results, embeddings, exact cell lists, figures, and manifests are stored in the two output directories. The manifests record the cohort sizes, preprocessing, sampling deviation, and installed versions.
