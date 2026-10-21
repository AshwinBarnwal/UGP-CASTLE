"""Evaluate an scVI embedding against the refined ProSeg cell-type labels."""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


N_CELL_TYPES = 17
DEFAULT_SILHOUETTE_SAMPLE_SIZE = 2000
DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "results" / "scvi_xenium_proseg_breast" / "adata_scvi.h5ad"


def save_png(fig: object, path: Path, *, dpi: int = 220) -> None:
    """Encode plots in memory, then write with a Windows-compatible binary file handle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    path.write_bytes(buffer.getvalue())


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate labeled cells in X_scVI with K-means K=17 and Leiden resolutions 0.2–2.0."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--label-column", default="cell_type")
    parser.add_argument("--embedding-key", default="X_scVI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leiden-neighbors", type=int, default=15)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.3)
    parser.add_argument("--silhouette-sample-size", type=int, default=DEFAULT_SILHOUETTE_SAMPLE_SIZE)
    args = parser.parse_args()
    if args.silhouette_sample_size < 2:
        raise ValueError("--silhouette-sample-size must be at least 2.")

    if not args.input.is_file():
        raise FileNotFoundError(f"Input H5AD does not exist: {args.input}")
    if args.leiden_neighbors < 2 or args.umap_neighbors < 2:
        raise ValueError("Neighbor counts must be at least 2.")
    import scanpy as sc

    adata = sc.read_h5ad(args.input)
    if args.label_column not in adata.obs:
        raise KeyError(f"Label column {args.label_column!r} is absent from obs.")
    if args.embedding_key not in adata.obsm:
        raise KeyError(f"Embedding {args.embedding_key!r} is absent from obsm.")

    labels = adata.obs[args.label_column].astype("string").fillna("").astype(str).str.strip()
    excluded = labels.eq("") | labels.str.casefold().isin({"unlabeled", "unlabelled"})
    keep = ~excluded.to_numpy()
    y = labels.loc[keep].to_numpy(dtype=str)
    X = np.asarray(adata.obsm[args.embedding_key][keep], dtype=np.float32)
    n_types = int(pd.Series(y).nunique())
    if n_types != N_CELL_TYPES:
        raise ValueError(
            f"Expected {N_CELL_TYPES} labeled cell types after excluding Unlabeled/Unlabelled and blanks, "
            f"found {n_types}."
        )
    if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all():
        raise ValueError(f"Embedding must be finite 2D data aligned to labels; got {X.shape}.")
    if len(y) <= max(args.leiden_neighbors, args.umap_neighbors):
        raise ValueError("Too few labeled cells for the requested neighborhood sizes.")

    predicted = KMeans(n_clusters=N_CELL_TYPES, random_state=args.seed, n_init=10).fit_predict(X)
    kmeans_ari = float(adjusted_rand_score(y, predicted))
    kmeans_nmi = float(normalized_mutual_info_score(y, predicted))
    kmeans_silhouette = sampled_silhouette(X, predicted, args.silhouette_sample_size, args.seed)
    cell_type_silhouette = sampled_silhouette(X, y, args.silhouette_sample_size, args.seed)

    try:
        import igraph
        from scib_metrics.nearest_neighbors import pynndescent
    except ImportError as exc:
        raise ImportError(
            "Leiden evaluation requires scib-metrics and igraph; install from requirements-scvi.txt."
        ) from exc

    neighbors = pynndescent(X, n_neighbors=args.leiden_neighbors, random_state=args.seed)
    resolution_rows: list[dict[str, float | int]] = []
    leiden_by_resolution: dict[float, np.ndarray] = {}
    for resolution in np.round(np.arange(0.2, 2.01, 0.2), 1):
        igraph.set_random_number_generator(random.Random(args.seed))
        graph = igraph.Graph.Weighted_Adjacency(neighbors.knn_graph_connectivities, mode="directed")
        graph.to_undirected(mode="each")
        partition = graph.community_leiden(
            objective_function="modularity", weights="weight", resolution=float(resolution)
        )
        cluster_labels = np.asarray(partition.membership)
        resolution = float(resolution)
        leiden_by_resolution[resolution] = cluster_labels
        resolution_rows.append(
            {
                "resolution": resolution,
                "n_clusters": int(pd.Series(cluster_labels).nunique()),
                "ARI": float(adjusted_rand_score(y, cluster_labels)),
                "NMI": float(normalized_mutual_info_score(y, cluster_labels, average_method="arithmetic")),
                "silhouette": sampled_silhouette(
                    X, cluster_labels, args.silhouette_sample_size, args.seed
                ),
            }
        )

    leiden_scores = pd.DataFrame(resolution_rows)
    best_ari_row = leiden_scores.loc[leiden_scores["ARI"].idxmax()]
    best_nmi_row = leiden_scores.loc[leiden_scores["NMI"].idxmax()]
    best_silhouette_row = leiden_scores.loc[leiden_scores["silhouette"].idxmax()]
    best_ari_resolution = float(best_ari_row["resolution"])
    best_nmi_resolution = float(best_nmi_row["resolution"])
    leiden_best_ari = leiden_by_resolution[best_ari_resolution]
    leiden_best_nmi = leiden_by_resolution[best_nmi_resolution]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import umap

    umap_xy = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
    ).fit_transform(X)

    def plot_labeling(ax: object, values: np.ndarray, title: str) -> None:
        values = pd.Series(values).astype(str).to_numpy()
        categories = sorted(pd.unique(values))
        codes = pd.Categorical(values, categories=categories).codes
        cmap = plt.get_cmap("tab20", max(len(categories), 1))
        ax.scatter(
            umap_xy[:, 0], umap_xy[:, 1], c=codes, cmap=cmap,
            vmin=-0.5, vmax=max(len(categories) - 0.5, 0.5),
            s=1.2, linewidths=0, rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        handles = [
            Line2D([], [], marker="o", linestyle="", markersize=5, color=cmap(i), label=category)
            for i, category in enumerate(categories)
        ]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6, frameon=False)

    output = args.output or args.input.parent / "label_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "cell_id": adata.obs_names[keep].astype(str),
            args.label_column: y,
            "kmeans_cluster": predicted,
            "leiden_cluster_best_ari": leiden_best_ari,
            "leiden_cluster_best_nmi": leiden_best_nmi,
            "umap_1": umap_xy[:, 0],
            "umap_2": umap_xy[:, 1],
        }
    ).to_csv(output / "matched_labels.csv", index=False)
    leiden_scores.to_csv(output / "leiden_resolution_metrics.csv", index=False)
    pd.DataFrame(
        [{
            "embedding_key": args.embedding_key,
            "n_cells": len(y),
            "n_cell_types": n_types,
            "unlabeled_or_blank_cells_excluded": int(excluded.sum()),
            "seed": args.seed,
            "leiden_neighbors": args.leiden_neighbors,
            "umap_neighbors": args.umap_neighbors,
            "umap_min_dist": args.umap_min_dist,
            "silhouette_sample_size": min(args.silhouette_sample_size, len(y)),
            "cell_type_silhouette": cell_type_silhouette,
            "kmeans_K": N_CELL_TYPES,
            "kmeans_ARI": kmeans_ari,
            "kmeans_NMI": kmeans_nmi,
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
            "cell_type_used_in_training": False,
            "scores_are_independent_of_training_labels": True,
        }]
    ).to_csv(output / "metrics.csv", index=False)

    ground_truth_title = "Ground-truth cell type (unlabeled excluded; not used in scVI training)"

    def save_pair(filename: str, clusters: np.ndarray, title: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        plot_labeling(axes[0], clusters, title)
        plot_labeling(axes[1], y, ground_truth_title)
        save_png(fig, output / filename)
        plt.close(fig)

    save_pair("umap_kmeans_vs_ground_truth.png", predicted, f"K-means clusters (K={N_CELL_TYPES})")
    save_pair(
        "umap_leiden_best_ari_vs_ground_truth.png",
        leiden_best_ari,
        f"Leiden clusters (ARI-best resolution {best_ari_resolution:.1f})",
    )
    save_pair(
        "umap_leiden_best_nmi_vs_ground_truth.png",
        leiden_best_nmi,
        f"Leiden clusters (NMI-best resolution {best_nmi_resolution:.1f})",
    )

    print(f"Excluded {int(excluded.sum())} unlabeled/blank cells before clustering and scoring.")
    print(f"K-means K={N_CELL_TYPES}: ARI={kmeans_ari:.6f}, NMI={kmeans_nmi:.6f}")
    print(f"K-means silhouette={kmeans_silhouette:.6f}; cell-type silhouette={cell_type_silhouette:.6f}")
    print(
        f"Leiden best ARI={best_ari_row['ARI']:.6f} at {best_ari_resolution:.1f}; "
        f"best NMI={best_nmi_row['NMI']:.6f} at {best_nmi_resolution:.1f}; "
        f"silhouette there={best_nmi_row['silhouette']:.6f}"
    )
    print(
        f"Leiden best silhouette={best_silhouette_row['silhouette']:.6f} "
        f"at {best_silhouette_row['resolution']:.1f} "
        f"(sample size {min(args.silhouette_sample_size, len(y))})"
    )
    print(f"Saved evaluation to {output.resolve()}")


if __name__ == "__main__":
    main()
