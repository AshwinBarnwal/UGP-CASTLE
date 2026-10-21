"""Evaluate SPLIT purified counts with CASTLE's Python evaluation settings."""

import argparse
import random
import sys
from pathlib import Path

# Avoid this package directory's random.py shadowing Python's stdlib module.
_module_dir = str(Path(__file__).resolve().parent)
sys.path = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != _module_dir]

import igraph
import numpy as np
import pandas as pd
import scipy.sparse as sp
import umap
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scib_metrics.nearest_neighbors import pynndescent
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from tqdm import tqdm


DEFAULT_SILHOUETTE_SAMPLE_SIZE = 2000


def sampled_silhouette(X, labels, sample_size, seed):
    """Estimate Euclidean silhouette on a reproducible sample to avoid quadratic full-data cost."""
    labels = np.asarray(labels)
    n_all_clusters = np.unique(labels).size
    if n_all_clusters < 2 or n_all_clusters >= len(labels):
        return float("nan")
    sample_size = min(sample_size, len(labels))
    if sample_size == len(labels):
        return float(silhouette_score(X, labels))
    indices = np.random.default_rng(seed).choice(len(labels), size=sample_size, replace=False)
    sampled_labels = labels[indices]
    n_clusters = np.unique(sampled_labels).size
    if n_clusters < 2 or n_clusters >= sample_size:
        return float("nan")
    return float(silhouette_score(X[indices], sampled_labels))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SPLIT counts against Janesick labels with CASTLE's clustering settings."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--export-dir", type=Path,
        help="Binary CSC export from export_split_matrix_for_python.R.",
    )
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--castle-cells", type=Path)
    parser.add_argument(
        "--output", type=Path,
        help="Output directory (default: results/split_janesick_evaluation_python).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--leiden-neighbors", type=int, default=15)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.3)
    parser.add_argument("--scale-factor", type=float, default=10_000)
    parser.add_argument("--silhouette-sample-size", type=int, default=DEFAULT_SILHOUETTE_SAMPLE_SIZE)
    return parser.parse_args()


def load_export(export_dir: Path):
    shape = np.loadtxt(export_dir / "shape.txt", dtype=np.int64).reshape(-1)
    if shape.size != 2:
        raise ValueError("shape.txt must contain the gene and cell dimensions.")
    n_genes, n_cells = map(int, shape)
    data = np.fromfile(export_dir / "counts_data.bin", dtype="<f8")
    indices = np.fromfile(export_dir / "counts_indices.bin", dtype="<i4")
    indptr = np.fromfile(export_dir / "counts_indptr.bin", dtype="<i4")
    genes = pd.read_csv(export_dir / "genes.tsv", sep="\t", header=None, dtype=str)[0]
    barcodes = pd.read_csv(export_dir / "barcodes.tsv", sep="\t", header=None, dtype=str)[0]

    if len(genes) != n_genes or len(barcodes) != n_cells:
        raise ValueError("Exported gene/barcode counts do not match shape.txt.")
    if len(indptr) != n_cells + 1 or indptr[-1] != len(data) or len(indices) != len(data):
        raise ValueError("The exported sparse CSC arrays are inconsistent.")
    counts_gene_by_cell = sp.csc_matrix(
        (data, indices, indptr), shape=(n_genes, n_cells)
    )
    return counts_gene_by_cell.T.tocsr(), genes.to_numpy(), barcodes.to_numpy()


def plot_labeling(ax, coordinates, values, title, legend_title):
    values = pd.Series(values).astype(str).to_numpy()
    categories = sorted(pd.unique(values))
    category_codes = pd.Categorical(values, categories=categories).codes
    cmap = plt.get_cmap("tab20", max(len(categories), 1))
    ax.scatter(
        coordinates[:, 0], coordinates[:, 1], c=category_codes, cmap=cmap,
        vmin=-0.5, vmax=max(len(categories) - 0.5, 0.5),
        s=1.2, linewidths=0, rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5,
               color=cmap(i), label=category)
        for i, category in enumerate(categories)
    ]
    ax.legend(handles=handles, title=legend_title, loc="center left",
              bbox_to_anchor=(1.01, 0.5), fontsize=6, frameon=False)


def save_umap_pair(path, coordinates, predicted, truth, title):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    plot_labeling(axes[0], coordinates, predicted, title, "Cluster")
    plot_labeling(axes[1], coordinates, truth, "Ground-truth cell type", "Cell type")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.silhouette_sample_size < 2:
        raise ValueError("--silhouette-sample-size must be at least 2.")
    root = args.project_root.resolve()
    export_dir = (args.export_dir or root / "results" / "split_janesick_python_export").resolve()
    labels_path = (args.labels or root / "data" / "xenium_r1_supervised_labels.csv").resolve()
    castle_cells_path = (
        args.castle_cells
        or root / "results" / "castle_refree_xenium_breast_beta2" / "cells.tsv"
    ).resolve()
    output_dir = (
        args.output or root / "results" / "split_janesick_evaluation_python"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (export_dir / "shape.txt", labels_path, castle_cells_path):
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/9] Loading exported sparse SPLIT counts...", flush=True)
    counts, genes, split_barcodes = load_export(export_dir)
    if pd.Index(split_barcodes).has_duplicates:
        raise ValueError("Duplicate barcodes in the SPLIT export.")
    if counts.data.size and (not np.isfinite(counts.data).all() or (counts.data < 0).any()):
        raise ValueError("SPLIT counts contain negative or non-finite values.")

    print("[2/9] Reading labels and CASTLE cell IDs...", flush=True)
    labels = pd.read_csv(labels_path, dtype={"Barcode": str}, keep_default_na=False)
    required = {"Barcode", "Cluster"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Ground-truth CSV must contain columns {sorted(required)}.")
    labels["Barcode"] = labels["Barcode"].astype(str).str.strip()
    labels["Cluster"] = labels["Cluster"].astype(str).str.strip()
    if labels["Barcode"].duplicated().any():
        raise ValueError("Duplicate Barcode values in ground-truth CSV.")
    unlabeled = labels["Cluster"].str.casefold().isin({"unlabeled", "unlabelled"}) | (
        labels["Cluster"] == ""
    )
    labeled = labels.loc[~unlabeled, ["Barcode", "Cluster"]].copy()
    target_k = int(labeled["Cluster"].nunique())
    if target_k < 2:
        raise ValueError(f"Need at least two labeled classes; found {target_k}.")
    castle_ids = pd.read_csv(castle_cells_path, header=None, dtype=str)[0].str.strip()
    if castle_ids.duplicated().any():
        raise ValueError("Duplicate barcodes in CASTLE cells.tsv.")

    print("[3/9] Applying the paper-style cell and gene QC filters...", flush=True)
    cell_totals = np.asarray(counts.sum(axis=1)).ravel()
    detected_genes = np.asarray((counts > 0).sum(axis=1)).ravel()
    qc_cells = (cell_totals >= 10) & (detected_genes >= 5)
    if not qc_cells.any():
        raise ValueError("No SPLIT cells remain after the cell QC filters.")
    counts_qc = counts[qc_cells]
    barcodes_qc = split_barcodes[qc_cells]
    gene_keep = np.asarray((counts_qc > 0).sum(axis=0)).ravel() >= 10
    if gene_keep.sum() < 2:
        raise ValueError("Fewer than two genes remain after gene QC.")
    counts_qc = counts_qc[:, gene_keep]

    split_rows = pd.Series(np.arange(len(barcodes_qc)), index=barcodes_qc)
    castle_set = set(castle_ids)
    shared = labeled.loc[
        labeled["Barcode"].isin(split_rows.index) & labeled["Barcode"].isin(castle_set)
    ].copy()
    if shared.empty:
        raise ValueError("No labeled barcodes are shared by SPLIT and CASTLE after QC.")
    truth = shared["Cluster"].to_numpy()
    if pd.Series(truth).nunique() != target_k:
        raise ValueError("The shared QC-filtered cells do not contain every ground-truth class.")
    row_indices = split_rows.loc[shared["Barcode"]].to_numpy()
    counts_shared = counts_qc[row_indices]

    print(f"[4/9] Normalizing and log-transforming {len(shared):,} shared cells...", flush=True)
    totals = np.asarray(counts_shared.sum(axis=1)).ravel()
    dense = counts_shared.toarray().astype(np.float32, copy=False)
    dense *= (args.scale_factor / np.maximum(totals, 1))[:, None].astype(np.float32)
    np.log1p(dense, out=dense)

    n_pcs = min(args.n_pcs, dense.shape[1] - 1, dense.shape[0] - 1)
    if n_pcs < 2:
        raise ValueError("Not enough cells or genes to calculate PCA.")
    print(f"[5/9] Computing centered, unscaled PCA ({n_pcs} PCs)...", flush=True)
    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=args.seed)
    X = pca.fit_transform(dense).astype(np.float32, copy=False)
    del dense

    print("[6/9] Running K-means and building the CASTLE PyNNDescent graph...", flush=True)
    kmeans = KMeans(n_clusters=target_k, random_state=args.seed, n_init=10).fit_predict(X)
    kmeans_silhouette = sampled_silhouette(X, kmeans, args.silhouette_sample_size, args.seed)
    cell_type_silhouette = sampled_silhouette(X, truth, args.silhouette_sample_size, args.seed)
    neighbors = pynndescent(
        X,
        n_neighbors=args.leiden_neighbors,
        random_state=args.seed,
    )

    print("[7/9] Leiden resolution sweep...", flush=True)
    resolution_rows = []
    leiden_by_resolution = {}
    resolutions = np.round(np.arange(0.2, 2.01, 0.2), 1)
    for resolution in tqdm(resolutions, desc="Leiden", unit="resolution"):
        igraph.set_random_number_generator(random.Random(args.seed))
        graph = igraph.Graph.Weighted_Adjacency(
            neighbors.knn_graph_connectivities, mode="directed"
        )
        graph.to_undirected(mode="each")
        partition = graph.community_leiden(
            objective_function="modularity",
            weights="weight",
            resolution=float(resolution),
        )
        predicted = np.asarray(partition.membership)
        leiden_by_resolution[float(resolution)] = predicted
        resolution_rows.append({
            "resolution": float(resolution),
            "n_clusters": int(pd.Series(predicted).nunique()),
            "ARI": float(adjusted_rand_score(truth, predicted)),
            "NMI": float(normalized_mutual_info_score(
                truth, predicted, average_method="arithmetic"
            )),
            "silhouette": sampled_silhouette(
                X, predicted, args.silhouette_sample_size, args.seed
            ),
        })
    resolution_metrics = pd.DataFrame(resolution_rows)
    best_ari_row = resolution_metrics.loc[resolution_metrics["ARI"].idxmax()]
    best_nmi_row = resolution_metrics.loc[resolution_metrics["NMI"].idxmax()]
    best_silhouette_row = resolution_metrics.loc[resolution_metrics["silhouette"].idxmax()]
    best_ari_resolution = float(best_ari_row["resolution"])
    best_nmi_resolution = float(best_nmi_row["resolution"])

    print("[8/9] Computing UMAP with CASTLE's settings...", flush=True)
    umap_coordinates = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
    ).fit_transform(X)

    print("[9/9] Saving matched labels, metrics, and UMAP plots...", flush=True)
    barcodes_original_index = pd.Series(np.arange(len(split_barcodes)), index=split_barcodes)
    original_positions = barcodes_original_index.loc[shared["Barcode"]].to_numpy()
    matched = shared.copy()
    matched["split_matrix_column_zero_based"] = original_positions
    matched["split_matrix_column_one_based"] = original_positions + 1
    matched["kmeans_cluster"] = kmeans.astype(str)
    matched["leiden_cluster_best_nmi"] = leiden_by_resolution[best_nmi_resolution].astype(str)
    matched["umap_1"] = umap_coordinates[:, 0]
    matched["umap_2"] = umap_coordinates[:, 1]
    matched.to_csv(output_dir / "matched_labels.csv", index=False)
    shared[["Barcode", "Cluster"]].to_csv(
        output_dir / "shared_labels_for_castle.csv", index=False
    )
    resolution_metrics.to_csv(output_dir / "leiden_resolution_metrics.csv", index=False)
    shared["Barcode"].to_csv(
        output_dir / "shared_labeled_barcodes.txt", index=False, header=False
    )

    metrics = {
        "n_ground_truth_labeled": len(labeled),
        "n_split_cells": len(split_barcodes),
        "n_castle_cells": len(castle_ids),
        "n_common_labeled_after_qc": len(shared),
        "n_split_labeled_missing": int((~labeled["Barcode"].isin(set(split_barcodes))).sum()),
        "unlabeled_rows_excluded": int(unlabeled.sum()),
        "n_ground_truth_classes": target_k,
        "n_genes_after_qc": int(gene_keep.sum()),
        "n_pcs": n_pcs,
        "kmeans_K": target_k,
        "leiden_neighbors": args.leiden_neighbors,
        "umap_neighbors": args.umap_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "silhouette_sample_size": min(args.silhouette_sample_size, len(truth)),
        "cell_type_silhouette": cell_type_silhouette,
        "seed": args.seed,
        "kmeans_ARI": adjusted_rand_score(truth, kmeans),
        "kmeans_NMI": normalized_mutual_info_score(truth, kmeans, average_method="arithmetic"),
        "kmeans_silhouette": kmeans_silhouette,
        "leiden_best_ARI": float(best_ari_row["ARI"]),
        "leiden_best_ARI_resolution": best_ari_resolution,
        "leiden_best_NMI": float(best_nmi_row["NMI"]),
        "leiden_best_NMI_resolution": best_nmi_resolution,
        "leiden_silhouette_at_best_ARI_resolution": float(best_ari_row["silhouette"]),
        "leiden_silhouette_at_best_NMI_resolution": float(best_nmi_row["silhouette"]),
        "leiden_best_silhouette": float(best_silhouette_row["silhouette"]),
        "leiden_best_silhouette_resolution": float(best_silhouette_row["resolution"]),
        "leiden_ARI_at_best_NMI_resolution": float(best_nmi_row["ARI"]),
        "leiden_NMI_at_best_NMI_resolution": float(best_nmi_row["NMI"]),
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    save_umap_pair(
        output_dir / "umap_kmeans_vs_ground_truth.png",
        umap_coordinates, kmeans, truth, f"K-means clusters (K = {target_k})",
    )
    save_umap_pair(
        output_dir / "umap_leiden_vs_ground_truth.png",
        umap_coordinates,
        leiden_by_resolution[best_nmi_resolution],
        truth,
        f"Leiden clusters (best-NMI resolution {best_nmi_resolution:.1f})",
    )

    print("\nSPLIT evaluation results")
    print("------------------------")
    print(f"Labeled ground-truth classes (K): {target_k}")
    print(f"Labeled cells shared after QC: {len(shared):,}")
    print(f"Genes / PCs used: {int(gene_keep.sum())} / {n_pcs}")
    print(f"K-means ARI/NMI: {metrics['kmeans_ARI']:.6f} / {metrics['kmeans_NMI']:.6f}")
    print(f"K-means silhouette: {metrics['kmeans_silhouette']:.6f}")
    print(f"Cell-type silhouette: {metrics['cell_type_silhouette']:.6f}")
    print(
        "Best Leiden ARI: "
        f"{metrics['leiden_best_ARI']:.6f} at resolution {best_ari_resolution:.1f}"
    )
    print(
        "Best Leiden NMI: "
        f"{metrics['leiden_best_NMI']:.6f} at resolution {best_nmi_resolution:.1f}"
    )
    print(
        "Leiden silhouette at best-NMI resolution: "
        f"{metrics['leiden_silhouette_at_best_NMI_resolution']:.6f}"
    )
    print(
        "Best Leiden silhouette: "
        f"{metrics['leiden_best_silhouette']:.6f} at resolution "
        f"{metrics['leiden_best_silhouette_resolution']:.1f} "
        f"(sample size {metrics['silhouette_sample_size']})"
    )
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
