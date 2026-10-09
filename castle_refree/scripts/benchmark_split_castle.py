"""Paper-matched biological-conservation benchmark for raw, SPLIT and CASTLE.

Count representations use the SPLIT paper's downstream transformation:
total-count normalization to 10,000, log1p and 50-component Scanpy PCA.
CASTLE's native intrinsic embedding is evaluated without an artificial SVD.

The clustering metrics use scib-metrics 0.5.5, matching Bilous et al. 2026.
The expensive O(n^2) silhouette metrics use one shared fixed-size sample on a
CPU workstation; all clustering, neighbor, cLISI, CH and DB metrics use the
full paper-sized evaluation sample.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scib_metrics
from scipy import sparse
from scipy.io import mmread
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from scib_metrics.metrics._nmi_ari import (
    _compute_clustering_kmeans,
    _compute_clustering_leiden,
    _compute_nmi_ari_cluster_labels,
)
from scib_metrics.nearest_neighbors import pynndescent


COUNT_METHODS = ("raw_pca50", "split_pca50", "castle_purified_pca50")
ALL_METHODS = COUNT_METHODS + ("castle_intrinsic_z16",)
TUMOUR_LABELS = {
    "Invasive_Tumor",
    "Prolif_Invasive_Tumor",
    "DCIS 1",
    "DCIS 2",
}


def log(message: str) -> None:
    print(message, flush=True)


def read_vector(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep="\t", header=None, dtype=str).iloc[:, 0].to_numpy()


def load_matrix_market(path: Path, n_cells: int, n_genes: int) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(mmread(path), dtype=np.float32)
    if matrix.shape == (n_genes, n_cells):
        matrix = matrix.T.tocsr()
    elif matrix.shape != (n_cells, n_genes):
        raise ValueError(
            f"{path} has shape {matrix.shape}; expected {(n_genes, n_cells)} "
            f"or {(n_cells, n_genes)}"
        )
    matrix.eliminate_zeros()
    return matrix


def normalize_log_pca50(matrix: sparse.csr_matrix, name: str) -> np.ndarray:
    log(f"[{name}] normalize_total(1e4), log1p, Scanpy PCA50")
    obj = ad.AnnData(matrix.astype(np.float32, copy=True))
    sc.pp.normalize_total(obj, target_sum=1e4)
    sc.pp.log1p(obj)
    sc.tl.pca(obj, n_comps=50)
    embedding = np.asarray(obj.obsm["X_pca"], dtype=np.float32)
    if embedding.shape != (matrix.shape[0], 50) or not np.isfinite(embedding).all():
        raise ValueError(f"Invalid PCA result for {name}: {embedding.shape}")
    del obj
    gc.collect()
    return embedding


def joint_count_qc(
    matrices: dict[str, sparse.csr_matrix], min_counts: int = 10,
    min_genes: int = 5, min_cells: int = 5,
) -> tuple[dict[str, sparse.csr_matrix], np.ndarray, np.ndarray]:
    n_cells = next(iter(matrices.values())).shape[0]
    cell_keep = np.ones(n_cells, dtype=bool)
    for name, matrix in matrices.items():
        keep = np.asarray(matrix.sum(axis=1)).ravel() >= min_counts
        keep &= np.diff(matrix.indptr) >= min_genes
        log(f"[{name}] count-QC cells passing alone: {int(keep.sum()):,}/{n_cells:,}")
        cell_keep &= keep

    cell_filtered = {name: matrix[cell_keep].tocsr() for name, matrix in matrices.items()}
    n_genes = next(iter(cell_filtered.values())).shape[1]
    gene_keep = np.ones(n_genes, dtype=bool)
    for name, matrix in cell_filtered.items():
        detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
        keep = detected >= min_cells
        log(f"[{name}] gene-QC genes passing alone: {int(keep.sum()):,}/{n_genes:,}")
        gene_keep &= keep
    return (
        {name: matrix[:, gene_keep].tocsr() for name, matrix in cell_filtered.items()},
        cell_keep,
        gene_keep,
    )


def metric_sample(n_cells: int, max_cells: int, seed: int) -> np.ndarray:
    if n_cells <= max_cells:
        return np.arange(n_cells)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_cells, max_cells, replace=False))


def benchmark_embedding(
    name: str,
    embedding: np.ndarray,
    labels: np.ndarray,
    silhouette_indices: np.ndarray,
) -> dict[str, float | int]:
    log(f"[{name}] benchmarking {embedding.shape[0]:,} cells x {embedding.shape[1]} dimensions")
    label_codes = pd.Categorical(labels).codes
    n_types = int(np.unique(label_codes).size)

    log(f"[{name}] exact scib-metrics KMeans with K={n_types}")
    kmeans_labels = _compute_clustering_kmeans(embedding, n_types)
    kmeans_nmi = normalized_mutual_info_score(labels, kmeans_labels, average_method="arithmetic")
    kmeans_ari = adjusted_rand_score(labels, kmeans_labels)

    log(f"[{name}] 90-NN graph for cLISI and 15-NN Leiden")
    neighbors90 = pynndescent(
        embedding, n_neighbors=90, random_state=0, n_jobs=-1
    )
    clisi = float(scib_metrics.clisi_knn(neighbors90, labels))
    neighbors15 = neighbors90.subset_neighbors(n=15)
    graph15 = neighbors15.knn_graph_connectivities

    log(f"[{name}] Leiden resolution search 0.2..2.0")
    resolutions = np.arange(0.2, 2.01, 0.2)
    leiden_candidates: list[tuple[float, float, float]] = []
    for resolution in resolutions:
        nmi, ari = _compute_nmi_ari_cluster_labels(
            graph15, labels, resolution=float(resolution), seed=42
        )
        leiden_candidates.append((float(resolution), float(nmi), float(ari)))
    best_resolution, leiden_nmi, leiden_ari = max(
        leiden_candidates, key=lambda value: value[1]
    )
    leiden_labels = _compute_clustering_leiden(graph15, best_resolution, seed=42)

    small_x = embedding[silhouette_indices]
    small_labels = labels[silhouette_indices]
    small_kmeans = kmeans_labels[silhouette_indices]
    log(f"[{name}] silhouette metrics on shared {len(silhouette_indices):,}-cell CPU sample")
    label_silhouette = float(scib_metrics.silhouette_label(small_x, small_labels))
    isolated = float(
        scib_metrics.isolated_labels(
            small_x,
            small_labels,
            np.repeat("single_xenium_sample", len(small_labels)),
        )
    )

    result: dict[str, float | int] = {
        "cells": int(embedding.shape[0]),
        "dimensions": int(embedding.shape[1]),
        "reference_types": n_types,
        "kmeans_clusters": int(np.unique(kmeans_labels).size),
        "kmeans_ari": float(kmeans_ari),
        "kmeans_nmi": float(kmeans_nmi),
        "leiden_clusters": int(np.unique(leiden_labels).size),
        "leiden_best_resolution": float(best_resolution),
        "leiden_ari": float(leiden_ari),
        "leiden_nmi": float(leiden_nmi),
        "label_silhouette_rescaled": label_silhouette,
        "clisi": clisi,
        "isolated_labels": isolated,
        "label_calinski_harabasz": float(calinski_harabasz_score(embedding, label_codes)),
        "label_davies_bouldin": float(davies_bouldin_score(embedding, label_codes)),
        "kmeans_silhouette_unscaled_sample": float(
            silhouette_score(small_x, small_kmeans)
        ),
        "kmeans_calinski_harabasz": float(
            calinski_harabasz_score(embedding, kmeans_labels)
        ),
        "kmeans_davies_bouldin": float(
            davies_bouldin_score(embedding, kmeans_labels)
        ),
        "silhouette_sample_cells": int(len(silhouette_indices)),
        "leiden_resolution_candidates": [
            {"resolution": r, "nmi": nmi, "ari": ari}
            for r, nmi, ari in leiden_candidates
        ],
    }
    del neighbors90, neighbors15, graph15, kmeans_labels, leiden_labels
    gc.collect()
    return result


def make_plots(results: pd.DataFrame, output_dir: Path) -> None:
    display_names = {
        "raw_pca50": "Raw\nPCA50",
        "split_pca50": "SPLIT\nPCA50",
        "castle_purified_pca50": "CASTLE purified\nPCA50",
        "castle_intrinsic_z16": "CASTLE intrinsic\nz16",
    }
    metric_specs = [
        ("kmeans_ari", "KMeans ARI", True),
        ("kmeans_nmi", "KMeans NMI", True),
        ("leiden_ari", "Leiden ARI", True),
        ("leiden_nmi", "Leiden NMI", True),
        ("label_silhouette_rescaled", "Silhouette label", True),
        ("clisi", "cLISI", True),
        ("isolated_labels", "Isolated labels", True),
        ("label_calinski_harabasz", "Calinski-Harabasz", True),
        ("label_davies_bouldin", "Davies-Bouldin", False),
    ]
    raw = results.loc[list(ALL_METHODS), [m[0] for m in metric_specs]].to_numpy(float)
    goodness = np.zeros_like(raw)
    for j, (_, _, higher) in enumerate(metric_specs):
        column = raw[:, j]
        lo, hi = np.nanmin(column), np.nanmax(column)
        scaled = np.ones_like(column) if hi == lo else (column - lo) / (hi - lo)
        goodness[:, j] = scaled if higher else 1.0 - scaled

    fig, ax = plt.subplots(figsize=(15, 5.5), constrained_layout=True)
    image = ax.imshow(goodness, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metric_specs)), [m[1] for m in metric_specs], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ALL_METHODS)), [display_names[m] for m in ALL_METHODS])
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            value = raw[i, j]
            text = f"{value:.3f}" if abs(value) < 100 else f"{value:.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if goodness[i, j] > 0.62 else "black")
    ax.set_title("SPLIT-paper-style biological conservation benchmark\n"
                 "Color is column-scaled goodness; annotations are raw scores")
    fig.colorbar(image, ax=ax, label="Relative goodness within metric")
    fig.savefig(output_dir / "split_castle_benchmark_heatmap.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(len(ALL_METHODS))
    width = 0.36
    for ax, prefix, title in zip(axes, ("kmeans", "leiden"), ("KMeans", "Leiden")):
        ax.bar(x - width / 2, results.loc[list(ALL_METHODS), f"{prefix}_ari"], width, label="ARI")
        ax.bar(x + width / 2, results.loc[list(ALL_METHODS), f"{prefix}_nmi"], width, label="NMI")
        ax.set_xticks(x, [display_names[m] for m in ALL_METHODS], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"{title} agreement with RCTD labels")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
    fig.savefig(output_dir / "split_castle_ari_nmi.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cells", type=int, default=100_000)
    parser.add_argument("--silhouette-cells", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-development-crop", action="store_true",
        help="Include the 15k cells used while selecting beta; excluded by default.",
    )
    parser.add_argument(
        "--exclude-tumour", action="store_true",
        help="Sensitivity analysis excluding this dataset's four tumour/DCIS labels.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "results" / "benchmark_split_castle_full",
    )
    args = parser.parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = REPO_ROOT / "data" / "castle_xenium_breast"
    split_dir = REPO_ROOT / "data" / "benchmark_split_castle"
    castle_dir = REPO_ROOT / "results" / "castle_refree_xenium_breast_beta2"

    log("Loading identifiers and sparse matrices")
    raw_cells, raw_genes = read_vector(raw_dir / "cells.tsv"), read_vector(raw_dir / "genes.tsv")
    split_cells = read_vector(split_dir / "split_cells.tsv")
    split_genes = read_vector(split_dir / "split_genes.tsv")
    castle_cells = read_vector(castle_dir / "cells.tsv")
    castle_genes = read_vector(castle_dir / "genes.tsv")

    raw = load_matrix_market(raw_dir / "counts.mtx", len(raw_cells), len(raw_genes))
    split = load_matrix_market(
        split_dir / "split_purified_counts.mtx", len(split_cells), len(split_genes)
    )
    castle = sparse.load_npz(castle_dir / "purified_counts_cells_by_genes.npz").tocsr().astype(np.float32)
    castle_z = np.load(castle_dir / "castle_embeddings_and_qc.npz")["z_intrinsic"].astype(np.float32)
    if castle.shape != (len(castle_cells), len(castle_genes)):
        raise ValueError("CASTLE purified matrix identifiers do not match its shape")
    if castle_z.shape[0] != len(castle_cells):
        raise ValueError("CASTLE embedding and cell identifiers differ")

    labels_frame = pd.read_csv(castle_dir / "rctd_labels.csv", dtype={"cell_id": str})
    label_column = "primary" if "primary" in labels_frame else "first_type"
    labels_series = labels_frame.drop_duplicates("cell_id").set_index("cell_id")[label_column]

    raw_cell_index, castle_cell_index = pd.Index(raw_cells), pd.Index(castle_cells)
    label_index = pd.Index(labels_series.index)
    cells = split_cells[
        (pd.Index(split_cells).isin(raw_cell_index))
        & (pd.Index(split_cells).isin(castle_cell_index))
        & (pd.Index(split_cells).isin(label_index))
    ]
    if not args.include_development_crop:
        crop_cells = set(read_vector(REPO_ROOT / "data" / "castle_xenium_breast_15000" / "cells.tsv"))
        cells = cells[~pd.Index(cells).isin(crop_cells)]
    labels = labels_series.loc[cells].astype(str).to_numpy()
    valid = pd.notna(labels) & ~pd.Series(labels).str.contains("malignant", case=False, regex=False).to_numpy()
    if args.exclude_tumour:
        valid &= ~np.isin(labels, list(TUMOUR_LABELS))
    cells, labels = cells[valid], labels[valid]

    genes = split_genes[
        pd.Index(split_genes).isin(pd.Index(raw_genes))
        & pd.Index(split_genes).isin(pd.Index(castle_genes))
    ]
    raw_rows, split_rows, castle_rows = (
        raw_cell_index.get_indexer(cells),
        pd.Index(split_cells).get_indexer(cells),
        castle_cell_index.get_indexer(cells),
    )
    raw_cols, split_cols, castle_cols = (
        pd.Index(raw_genes).get_indexer(genes),
        pd.Index(split_genes).get_indexer(genes),
        pd.Index(castle_genes).get_indexer(genes),
    )
    if min(raw_rows.min(), split_rows.min(), castle_rows.min(), raw_cols.min(), split_cols.min(), castle_cols.min()) < 0:
        raise ValueError("Identifier alignment failed")

    matrices = {
        "raw": raw[raw_rows][:, raw_cols].tocsr(),
        "split": split[split_rows][:, split_cols].tocsr(),
        "castle_purified": castle[castle_rows][:, castle_cols].tocsr(),
    }
    aligned_z = castle_z[castle_rows]
    del raw, split, castle, castle_z
    gc.collect()

    matrices, cell_keep, gene_keep = joint_count_qc(matrices)
    cells, labels, aligned_z = cells[cell_keep], labels[cell_keep], aligned_z[cell_keep]
    genes = genes[gene_keep]

    evaluation_indices = metric_sample(len(cells), args.max_cells, args.seed)
    cells, labels, aligned_z = (
        cells[evaluation_indices], labels[evaluation_indices], aligned_z[evaluation_indices]
    )
    matrices = {name: matrix[evaluation_indices].tocsr() for name, matrix in matrices.items()}
    silhouette_indices = metric_sample(len(cells), args.silhouette_cells, args.seed + 17)

    log(f"Final shared cohort: {len(cells):,} cells, {len(genes):,} genes, "
        f"{pd.Series(labels).nunique()} RCTD types")
    pd.DataFrame({"cell_id": cells, "rctd_primary": labels}).to_csv(
        output_dir / "evaluation_cells.csv.gz", index=False
    )
    pd.DataFrame({"gene": genes}).to_csv(output_dir / "evaluation_genes.csv", index=False)
    pd.Series(labels).value_counts().rename_axis("rctd_primary").rename("cells").to_csv(
        output_dir / "label_counts.csv"
    )
    pd.DataFrame({"evaluation_row": silhouette_indices}).to_csv(
        output_dir / "silhouette_sample_rows.csv", index=False
    )

    embeddings = {
        "raw_pca50": normalize_log_pca50(matrices["raw"], "raw"),
        "split_pca50": normalize_log_pca50(matrices["split"], "split"),
        "castle_purified_pca50": normalize_log_pca50(
            matrices["castle_purified"], "castle_purified"
        ),
        "castle_intrinsic_z16": aligned_z,
    }
    np.savez_compressed(output_dir / "benchmark_embeddings.npz", **embeddings)
    del matrices, aligned_z
    gc.collect()

    metadata = {
        "protocol": "SPLIT/scib-metrics biological conservation",
        "development_crop_excluded": not args.include_development_crop,
        "tumour_labels_excluded": bool(args.exclude_tumour),
        "official_malignant_substring_filter_applied": True,
        "evaluation_cells": int(len(cells)),
        "evaluation_genes": int(len(genes)),
        "reference_types": int(pd.Series(labels).nunique()),
        "max_cells": int(args.max_cells),
        "silhouette_sample_cells": int(len(silhouette_indices)),
        "silhouette_sampling_deviation": (
            "SPLIT computes O(n^2) silhouette on the full <=100k cohort; this CPU run "
            "uses one shared fixed sample for silhouette and isolated-label metrics only."
        ),
        "count_preprocessing": "normalize_total(target_sum=1e4), log1p, Scanpy PCA n_comps=50",
        "native_castle_preprocessing": "none; exported 16D intrinsic posterior mean",
        "versions": {
            package: version(package)
            for package in (
                "numpy", "scipy", "pandas", "scikit-learn", "scanpy",
                "anndata", "scib-metrics", "jax", "jaxlib", "igraph", "pynndescent",
            )
        },
        "python": platform.python_version(),
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    all_results: dict[str, dict[str, float | int]] = {}
    for name in ALL_METHODS:
        all_results[name] = benchmark_embedding(
            name, embeddings[name], labels, silhouette_indices
        )
        (output_dir / "benchmark_results_partial.json").write_text(
            json.dumps(all_results, indent=2), encoding="utf-8"
        )

    results = pd.DataFrame(all_results).T
    results.to_csv(output_dir / "benchmark_results.csv", index_label="representation")
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    make_plots(results, output_dir)
    log("\nFinal benchmark results:\n" + results.to_string())
    log(f"\nSaved benchmark to {output_dir}")


if __name__ == "__main__":
    main()
