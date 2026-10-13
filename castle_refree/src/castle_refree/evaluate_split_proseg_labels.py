"""Evaluate existing SPLIT purified counts against ProSeg H5AD cell labels.

The H5AD uses synthetic query_* observation names while the saved SPLIT result
uses original numeric Xenium barcodes. This evaluator aligns the two by mutual
nearest cell-center coordinates, with a configurable maximum distance.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Avoid this package directory's random.py shadowing Python's stdlib module.
_module_dir = str(Path(__file__).resolve().parent)
sys.path = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != _module_dir]

import anndata
import igraph
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import umap
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scib_metrics.nearest_neighbors import pynndescent
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
N_CELL_TYPES = 17
UNLABELED_NAMES = {"unlabeled", "unlabelled"}
DEFAULT_SILHOUETTE_SAMPLE_SIZE = 2000


def sampled_silhouette(X: np.ndarray, labels: np.ndarray, sample_size: int, seed: int) -> float:
    """Estimate Euclidean silhouette on a reproducible sample to avoid quadratic full-data cost."""
    labels = np.asarray(labels)
    sample_size = min(sample_size, len(labels))
    n_all_clusters = np.unique(labels).size
    if n_all_clusters < 2 or n_all_clusters >= len(labels):
        return float("nan")
    if sample_size == len(labels):
        return float(silhouette_score(X, labels))
    indices = np.random.default_rng(seed).choice(len(labels), size=sample_size, replace=False)
    sampled_labels = labels[indices]
    n_clusters = np.unique(sampled_labels).size
    if n_clusters < 2 or n_clusters >= sample_size:
        return float("nan")
    return float(silhouette_score(X[indices], sampled_labels))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing SPLIT purified counts against ProSeg H5AD labels "
            "using mutual-nearest spatial matching."
        )
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--input-h5ad", type=Path, default=ROOT / "adata_for_tuto_s1.h5ad")
    parser.add_argument(
        "--split-export-dir", type=Path,
        default=ROOT / "results" / "split_janesick_python_export",
    )
    parser.add_argument(
        "--split-spatial", type=Path,
        default=ROOT / "data" / "castle_xenium_breast" / "spatial.csv",
        help="Original Xenium cell_id-to-coordinate map used by the SPLIT barcodes.",
    )
    parser.add_argument(
        "--castle-cells", type=Path,
        default=ROOT / "results" / "castle_refree_xenium_proseg_breast_beta2" / "cells.tsv",
        help="Restrict evaluation to cells present in the ProSeg CASTLE run.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "split_proseg_evaluation_python")
    parser.add_argument("--label-column", default="cell_type")
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--max-coordinate-distance", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--leiden-neighbors", type=int, default=15)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.3)
    parser.add_argument("--silhouette-sample-size", type=int, default=DEFAULT_SILHOUETTE_SAMPLE_SIZE)
    parser.add_argument("--scale-factor", type=float, default=10_000)
    return parser.parse_args()


def load_split_export(export_dir: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    shape = np.loadtxt(export_dir / "shape.txt", dtype=np.int64).reshape(-1)
    if shape.size != 2:
        raise ValueError("shape.txt must contain gene and cell dimensions.")
    n_genes, n_cells = map(int, shape)
    data = np.fromfile(export_dir / "counts_data.bin", dtype="<f8")
    indices = np.fromfile(export_dir / "counts_indices.bin", dtype="<i4")
    indptr = np.fromfile(export_dir / "counts_indptr.bin", dtype="<i4")
    genes = pd.read_csv(export_dir / "genes.tsv", sep="\t", header=None, dtype=str)[0]
    barcodes = pd.read_csv(export_dir / "barcodes.tsv", sep="\t", header=None, dtype=str)[0]
    if len(genes) != n_genes or len(barcodes) != n_cells:
        raise ValueError("SPLIT export gene/barcode counts do not match shape.txt.")
    if len(indptr) != n_cells + 1 or indptr[-1] != len(data) or len(indices) != len(data):
        raise ValueError("SPLIT sparse arrays are inconsistent.")
    counts_gene_by_cell = sp.csc_matrix((data, indices, indptr), shape=(n_genes, n_cells))
    return counts_gene_by_cell.T.tocsr(), genes.to_numpy(), barcodes.astype(str).str.strip().to_numpy()


def save_umap_pair(path: Path, coordinates: np.ndarray, predicted: np.ndarray, truth: np.ndarray, title: str) -> None:
    def plot(ax: object, values: np.ndarray, panel_title: str, legend_title: str) -> None:
        values = pd.Series(values).astype(str).to_numpy()
        categories = sorted(pd.unique(values))
        codes = pd.Categorical(values, categories=categories).codes
        cmap = plt.get_cmap("tab20", max(len(categories), 1))
        ax.scatter(
            coordinates[:, 0], coordinates[:, 1], c=codes, cmap=cmap,
            vmin=-0.5, vmax=max(len(categories) - 0.5, 0.5),
            s=1.2, linewidths=0, rasterized=True,
        )
        ax.set(title=panel_title, xlabel="UMAP 1", ylabel="UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        handles = [
            Line2D([], [], marker="o", linestyle="", markersize=5, color=cmap(i), label=label)
            for i, label in enumerate(categories)
        ]
        ax.legend(handles=handles, title=legend_title, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=6, frameon=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    plot(axes[0], predicted, title, "Cluster")
    plot(axes[1], truth, "H5AD cell_type (unlabeled excluded)", "Cell type")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.silhouette_sample_size < 2:
        raise ValueError("--silhouette-sample-size must be at least 2.")
    for path in (args.input_h5ad, args.split_spatial, args.castle_cells):
        if not path.exists():
            raise FileNotFoundError(path)
    for filename in ("shape.txt", "counts_data.bin", "counts_indices.bin", "counts_indptr.bin", "genes.tsv", "barcodes.tsv"):
        if not (args.split_export_dir / filename).exists():
            raise FileNotFoundError(args.split_export_dir / filename)
    args.output.mkdir(parents=True, exist_ok=True)

    print("[1/9] Loading ProSeg labels and spatial coordinates...", flush=True)
    adata = anndata.read_h5ad(args.input_h5ad, backed="r")
    if args.label_column not in adata.obs:
        raise KeyError(f"H5AD obs has no {args.label_column!r} column.")
    if args.spatial_key not in adata.obsm:
        raise KeyError(f"H5AD obsm has no {args.spatial_key!r} coordinates.")
    h5ad_ids = pd.Index(adata.obs_names.astype(str))
    h5ad_xy = np.asarray(adata.obsm[args.spatial_key], dtype=np.float64)
    labels = adata.obs[args.label_column].fillna("").astype(str).str.strip().to_numpy()
    if h5ad_xy.shape != (adata.n_obs, 2) or not np.isfinite(h5ad_xy).all():
        raise ValueError(f"Expected finite two-dimensional coordinates, got {h5ad_xy.shape}.")
    if h5ad_ids.has_duplicates:
        raise ValueError("H5AD observation names are not unique.")

    print("[2/9] Loading the saved SPLIT purified-count matrix...", flush=True)
    split_counts, genes, split_ids = load_split_export(args.split_export_dir)
    if pd.Index(split_ids).has_duplicates:
        raise ValueError("SPLIT export contains duplicate cell barcodes.")
    if split_counts.data.size and (not np.isfinite(split_counts.data).all() or (split_counts.data < 0).any()):
        raise ValueError("SPLIT purified counts contain negative or non-finite values.")

    spatial = pd.read_csv(args.split_spatial, dtype={"cell_id": str})
    if not {"cell_id", "x", "y"}.issubset(spatial.columns):
        raise ValueError("SPLIT spatial file must have cell_id, x, and y columns.")
    spatial["cell_id"] = spatial["cell_id"].astype(str).str.strip()
    if spatial["cell_id"].duplicated().any():
        raise ValueError("Duplicate cell IDs in SPLIT spatial file.")
    split_xy = spatial.set_index("cell_id").reindex(split_ids)[["x", "y"]]
    if split_xy.isna().any().any():
        missing = int(split_xy.isna().any(axis=1).sum())
        raise ValueError(f"No spatial coordinates for {missing} SPLIT cells.")
    split_xy_array = split_xy.to_numpy(dtype=np.float64)

    print("[3/9] Matching cells one-to-one by mutual nearest coordinates...", flush=True)
    h_to_split_distance, h_to_split = cKDTree(split_xy_array).query(h5ad_xy, k=1)
    _, split_to_h = cKDTree(h5ad_xy).query(split_xy_array, k=1)
    h_rows = np.arange(adata.n_obs)
    mutual = split_to_h[h_to_split] == h_rows
    coordinate_match = mutual & (h_to_split_distance <= args.max_coordinate_distance)
    mapping = pd.DataFrame(
        {
            "h5ad_cell_id": h5ad_ids.to_numpy()[coordinate_match],
            "split_cell_id": split_ids[h_to_split[coordinate_match]],
            "cell_type": labels[coordinate_match],
            "coordinate_distance": h_to_split_distance[coordinate_match],
        }
    )
    castle_ids = set(pd.read_csv(args.castle_cells, header=None, dtype=str)[0].str.strip())
    mapping = mapping[mapping["h5ad_cell_id"].isin(castle_ids)].copy()
    mapping.to_csv(args.output / "spatial_cell_matches.csv", index=False)

    unlabeled_mask = mapping["cell_type"].str.casefold().isin(UNLABELED_NAMES) | mapping["cell_type"].eq("")
    n_unlabeled_excluded = int(unlabeled_mask.sum())
    mapping = mapping.loc[~unlabeled_mask].copy()
    if mapping["h5ad_cell_id"].duplicated().any() or mapping["split_cell_id"].duplicated().any():
        raise ValueError("Spatial mapping failed the one-to-one cell ID check.")
    if mapping["cell_type"].nunique() != N_CELL_TYPES:
        counts_by_type = mapping["cell_type"].value_counts().to_dict()
        raise ValueError(
            f"Expected all {N_CELL_TYPES} labeled classes after spatial matching; "
            f"found {mapping['cell_type'].nunique()}: {counts_by_type}. "
            "Consider increasing --max-coordinate-distance, then review the reported distances."
        )

    print("[4/9] Applying the SPLIT evaluation QC filters...", flush=True)
    cell_totals = np.asarray(split_counts.sum(axis=1)).ravel()
    detected_genes = np.asarray((split_counts > 0).sum(axis=1)).ravel()
    qc_cells = (cell_totals >= 10) & (detected_genes >= 5)
    if not qc_cells.any():
        raise ValueError("No SPLIT cells remain after the count and detected-gene filters.")
    qc_positions = np.flatnonzero(qc_cells)
    qc_ids = split_ids[qc_positions]
    gene_keep = np.asarray((split_counts[qc_cells] > 0).sum(axis=0)).ravel() >= 10
    if gene_keep.sum() < 2:
        raise ValueError("Fewer than two genes remain after SPLIT gene QC.")
    split_rows = pd.Series(qc_positions, index=qc_ids)
    mapping = mapping[mapping["split_cell_id"].isin(split_rows.index)].copy()
    if mapping.empty:
        raise ValueError("No spatially matched labeled cells pass the SPLIT QC filters.")

    print(f"[5/9] Preparing {len(mapping):,} matched labeled cells...", flush=True)
    row_indices = split_rows.loc[mapping["split_cell_id"]].to_numpy(dtype=np.int64)
    counts_shared = split_counts[row_indices][:, gene_keep]
    totals_shared = np.asarray(counts_shared.sum(axis=1)).ravel()
    dense = counts_shared.toarray().astype(np.float32, copy=False)
    dense *= (args.scale_factor / np.maximum(totals_shared, 1))[:, None].astype(np.float32)
    np.log1p(dense, out=dense)

    truth = mapping["cell_type"].to_numpy(dtype=str)
    if pd.Series(truth).nunique() != N_CELL_TYPES:
        raise ValueError("SPLIT QC removed at least one of the 17 labeled classes.")
    n_pcs = min(args.n_pcs, dense.shape[1] - 1, dense.shape[0] - 1)
    if n_pcs < 2:
        raise ValueError("Not enough matched cells or genes for PCA.")
    print(f"[6/9] Computing centered, unscaled PCA ({n_pcs} PCs)...", flush=True)
    X = PCA(n_components=n_pcs, svd_solver="randomized", random_state=args.seed).fit_transform(dense)
    X = X.astype(np.float32, copy=False)
    del dense

    print("[7/9] Running K-means and building the Leiden graph...", flush=True)
    kmeans = KMeans(n_clusters=N_CELL_TYPES, random_state=args.seed, n_init=10).fit_predict(X)
    kmeans_silhouette = sampled_silhouette(X, kmeans, args.silhouette_sample_size, args.seed)
    cell_type_silhouette = sampled_silhouette(X, truth, args.silhouette_sample_size, args.seed)
    neighbors = pynndescent(X, n_neighbors=args.leiden_neighbors, random_state=args.seed)
    resolution_rows: list[dict[str, float | int]] = []
    leiden_by_resolution: dict[float, np.ndarray] = {}
    for resolution in tqdm(np.round(np.arange(0.2, 2.01, 0.2), 1), desc="Leiden", unit="resolution"):
        igraph.set_random_number_generator(random.Random(args.seed))
        graph = igraph.Graph.Weighted_Adjacency(neighbors.knn_graph_connectivities, mode="directed")
        graph.to_undirected(mode="each")
        partition = graph.community_leiden(
            objective_function="modularity", weights="weight", resolution=float(resolution)
        )
        predicted = np.asarray(partition.membership)
        resolution = float(resolution)
        leiden_by_resolution[resolution] = predicted
        resolution_rows.append(
            {
                "resolution": resolution,
                "n_clusters": int(pd.Series(predicted).nunique()),
                "ARI": float(adjusted_rand_score(truth, predicted)),
                "NMI": float(normalized_mutual_info_score(truth, predicted, average_method="arithmetic")),
                "silhouette": sampled_silhouette(
                    X, predicted, args.silhouette_sample_size, args.seed
                ),
            }
        )
    resolution_metrics = pd.DataFrame(resolution_rows)
    best_ari = resolution_metrics.loc[resolution_metrics["ARI"].idxmax()]
    best_nmi = resolution_metrics.loc[resolution_metrics["NMI"].idxmax()]
    best_silhouette = resolution_metrics.loc[resolution_metrics["silhouette"].idxmax()]
    best_nmi_resolution = float(best_nmi["resolution"])

    print("[8/9] Computing UMAP and saving metrics/plots...", flush=True)
    umap_coordinates = umap.UMAP(
        n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist,
        metric="euclidean", random_state=args.seed,
    ).fit_transform(X)
    mapping["split_matrix_column_zero_based"] = row_indices
    mapping["kmeans_cluster"] = kmeans.astype(str)
    mapping["leiden_cluster_best_nmi"] = leiden_by_resolution[best_nmi_resolution].astype(str)
    mapping["umap_1"] = umap_coordinates[:, 0]
    mapping["umap_2"] = umap_coordinates[:, 1]
    mapping.to_csv(args.output / "matched_labels.csv", index=False)
    resolution_metrics.to_csv(args.output / "leiden_resolution_metrics.csv", index=False)

    metrics = {
        "method": "SPLIT purified counts; normalized log counts followed by centered unscaled PCA",
        "cell_matching": "mutual nearest spatial coordinates; approximate across segmentation IDs",
        "max_coordinate_distance": args.max_coordinate_distance,
        "n_spatial_matches_before_label_filter": int(len(pd.read_csv(args.output / "spatial_cell_matches.csv"))),
        "n_unlabeled_cells_excluded": n_unlabeled_excluded,
        "n_matched_labeled_cells_after_qc": int(len(mapping)),
        "n_ground_truth_classes": N_CELL_TYPES,
        "n_genes_after_qc": int(gene_keep.sum()),
        "n_pcs": int(n_pcs),
        "silhouette_sample_size": min(args.silhouette_sample_size, len(truth)),
        "cell_type_silhouette": cell_type_silhouette,
        "kmeans_K": N_CELL_TYPES,
        "kmeans_ARI": float(adjusted_rand_score(truth, kmeans)),
        "kmeans_NMI": float(normalized_mutual_info_score(truth, kmeans, average_method="arithmetic")),
        "kmeans_silhouette": kmeans_silhouette,
        "leiden_neighbors": args.leiden_neighbors,
        "leiden_best_ARI": float(best_ari["ARI"]),
        "leiden_best_ARI_resolution": float(best_ari["resolution"]),
        "leiden_best_NMI": float(best_nmi["NMI"]),
        "leiden_best_NMI_resolution": best_nmi_resolution,
        "leiden_silhouette_at_best_ARI_resolution": float(best_ari["silhouette"]),
        "leiden_silhouette_at_best_NMI_resolution": float(best_nmi["silhouette"]),
        "leiden_best_silhouette": float(best_silhouette["silhouette"]),
        "leiden_best_silhouette_resolution": float(best_silhouette["resolution"]),
        "leiden_ARI_at_best_NMI_resolution": float(best_nmi["ARI"]),
        "leiden_NMI_at_best_NMI_resolution": float(best_nmi["NMI"]),
        "umap_neighbors": args.umap_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "seed": args.seed,
        "unlabeled_rows_excluded": True,
    }
    pd.DataFrame([metrics]).to_csv(args.output / "metrics.csv", index=False)
    save_umap_pair(
        args.output / "umap_kmeans_vs_ground_truth.png", umap_coordinates, kmeans, truth,
        f"SPLIT K-means clusters (K={N_CELL_TYPES})",
    )
    save_umap_pair(
        args.output / "umap_leiden_vs_ground_truth.png", umap_coordinates,
        leiden_by_resolution[best_nmi_resolution], truth,
        f"SPLIT Leiden clusters (best-NMI resolution {best_nmi_resolution:.1f})",
    )

    distances = mapping["coordinate_distance"]
    print("SPLIT / ProSeg evaluation")
    print("-------------------------")
    print(f"Matched labeled cells: {len(mapping):,} (K={N_CELL_TYPES}, unlabeled excluded)")
    print(f"Coordinate distance median / p95: {distances.median():.3f} / {distances.quantile(0.95):.3f}")
    print(f"K-means ARI/NMI: {metrics['kmeans_ARI']:.6f} / {metrics['kmeans_NMI']:.6f}")
    print(f"K-means silhouette: {metrics['kmeans_silhouette']:.6f}")
    print(f"Cell-type silhouette: {metrics['cell_type_silhouette']:.6f}")
    print(f"Best Leiden ARI: {metrics['leiden_best_ARI']:.6f} at {metrics['leiden_best_ARI_resolution']:.1f}")
    print(f"Best Leiden NMI: {metrics['leiden_best_NMI']:.6f} at {best_nmi_resolution:.1f}")
    print(
        f"Leiden silhouette at best-NMI resolution: "
        f"{metrics['leiden_silhouette_at_best_NMI_resolution']:.6f}"
    )
    print(
        f"Best Leiden silhouette: {metrics['leiden_best_silhouette']:.6f} "
        f"at {metrics['leiden_best_silhouette_resolution']:.1f} "
        f"(sample size {metrics['silhouette_sample_size']})"
    )
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
