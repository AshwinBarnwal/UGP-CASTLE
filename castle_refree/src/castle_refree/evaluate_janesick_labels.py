"""Evaluate CASTLE embeddings against Janesick/Xenium cell labels.

Barcode values are matched directly to the values in the embedding run's
cells.tsv. This matters because an embedding may contain fewer cells than the
label table after CASTLE filtering.
"""

import argparse
import random
import sys
from pathlib import Path

# This module sits beside the project's ``random.py``. When Python executes a
# file by path, it puts this directory first on sys.path, which can shadow the
# standard-library ``random`` module imported by NumPy. It is not needed here.
_module_dir = str(Path(__file__).resolve().parent)
sys.path = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != _module_dir]

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


UNLABELED_NAMES = {"unlabeled", "unlabelled"}
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CASTLE embeddings against Janesick/Xenium labels with K-means and Leiden."
    )
    parser.add_argument("--labels", default="data/xenium_r1_supervised_labels.csv")
    parser.add_argument("--embeddings", required=True,
                        help="NPZ file containing the embedding matrix.")
    parser.add_argument("--cells", required=True,
                        help="cells.tsv from the same run as --embeddings.")
    parser.add_argument("--embedding-key", default="z_intrinsic")
    parser.add_argument("--output", default="results/janesick_evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--leiden-neighbors", type=int, default=15,
        help="Number of neighbors for the scIB graph used by Leiden (default: 15).",
    )
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.3)
    parser.add_argument("--silhouette-sample-size", type=int, default=DEFAULT_SILHOUETTE_SAMPLE_SIZE)
    args = parser.parse_args()
    if args.silhouette_sample_size < 2:
        raise ValueError("--silhouette-sample-size must be at least 2.")

    labels = pd.read_csv(args.labels, dtype={"Barcode": str})
    required_columns = {"Barcode", "Cluster"}
    missing = required_columns - set(labels.columns)
    if missing:
        raise ValueError(f"Label CSV is missing columns: {sorted(missing)}")
    labels["Barcode"] = labels["Barcode"].str.strip()
    labels["Cluster"] = labels["Cluster"].fillna("").astype(str).str.strip()
    if labels["Barcode"].duplicated().any():
        raise ValueError("Duplicate Barcode values in label CSV.")

    # Determine K from all labeled classes in the CSV, before intersecting with
    # cells retained in this embedding run.
    is_unlabeled = labels["Cluster"].str.casefold().isin(UNLABELED_NAMES) | (labels["Cluster"] == "")
    labeled = labels.loc[~is_unlabeled].copy()
    n_classes = labeled["Cluster"].nunique()
    if n_classes < 2:
        raise ValueError(f"Need at least two labeled classes; found {n_classes}.")

    cells = pd.read_csv(args.cells, sep="\t", header=None, dtype=str).iloc[:, 0]
    cells = cells.str.strip()
    if cells.duplicated().any():
        raise ValueError("Duplicate cell barcodes in cells.tsv.")

    npz = np.load(args.embeddings, allow_pickle=False)
    print("Available embedding arrays:", npz.files)
    if args.embedding_key not in npz.files:
        raise KeyError(f"{args.embedding_key!r} not found; available arrays: {npz.files}")
    embeddings = npz[args.embedding_key]
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got {embeddings.shape}.")
    if len(cells) != len(embeddings):
        raise ValueError(
            f"cells.tsv has {len(cells)} rows but the embedding has {len(embeddings)} rows. "
            "Use cells.tsv from the same result directory as the NPZ."
        )

    # Row number in cells.tsv is the row number in the embedding; the barcode
    # itself is the join key. Never use Barcode as an embedding row index.
    embedding_cells = pd.DataFrame({"Barcode": cells, "embedding_row": np.arange(len(cells))})
    matched = labeled.merge(embedding_cells, on="Barcode", how="inner", validate="one_to_one")
    if matched.empty:
        raise ValueError("No labeled barcodes matched cells.tsv; check both files belong to this dataset.")
    if n_classes > len(matched):
        raise ValueError(
            f"K={n_classes} (distinct labeled classes in CSV) exceeds the "
            f"{len(matched)} matched labeled cells."
        )

    rows = matched["embedding_row"].to_numpy()
    X = embeddings[rows]
    y = matched["Cluster"].to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("Selected embeddings contain NaN or infinite values.")

    predicted = KMeans(n_clusters=n_classes, random_state=args.seed, n_init=10).fit_predict(X)
    kmeans_ari = adjusted_rand_score(y, predicted)
    kmeans_nmi = normalized_mutual_info_score(y, predicted)
    kmeans_silhouette = sampled_silhouette(X, predicted, args.silhouette_sample_size, args.seed)
    cell_type_silhouette = sampled_silhouette(X, y, args.silhouette_sample_size, args.seed)

    # Leiden operates on a neighbor graph, not directly on a fixed number of
    # clusters. Use scib-metrics' graph construction and Leiden implementation
    # to keep the sweep close to the protocol used in the SPLIT paper.
    try:
        import igraph
        from scib_metrics.nearest_neighbors import pynndescent
    except ImportError as exc:
        raise ImportError(
            "Leiden evaluation requires scib-metrics and igraph. Install them with "
            "`python -m pip install scib-metrics igraph`."
        ) from exc

    neighbors = pynndescent(
        np.asarray(X, dtype=np.float32),
        n_neighbors=args.leiden_neighbors,
        random_state=args.seed,
    )
    resolution_rows = []
    leiden_labels_by_resolution = {}
    for resolution in np.round(np.arange(0.2, 2.01, 0.2), 1):
        # Match scib-metrics' Leiden graph clustering implementation, while
        # retaining predicted labels for the requested UMAP visualization.
        igraph.set_random_number_generator(random.Random(args.seed))
        graph = igraph.Graph.Weighted_Adjacency(
            neighbors.knn_graph_connectivities,
            mode="directed",
        )
        graph.to_undirected(mode="each")
        partition = graph.community_leiden(
            objective_function="modularity",
            weights="weight",
            resolution=float(resolution),
        )
        leiden_labels = np.asarray(partition.membership)
        leiden_labels_by_resolution[float(resolution)] = leiden_labels
        resolution_rows.append({
            "resolution": float(resolution),
            "n_clusters": int(pd.Series(leiden_labels).nunique()),
            "ARI": float(adjusted_rand_score(y, leiden_labels)),
            "NMI": float(normalized_mutual_info_score(y, leiden_labels, average_method="arithmetic")),
            "silhouette": sampled_silhouette(
                X, leiden_labels, args.silhouette_sample_size, args.seed
            ),
        })
    leiden_scores = pd.DataFrame(resolution_rows)
    best_ari_row = leiden_scores.loc[leiden_scores["ARI"].idxmax()]
    best_nmi_row = leiden_scores.loc[leiden_scores["NMI"].idxmax()]
    best_silhouette_row = leiden_scores.loc[leiden_scores["silhouette"].idxmax()]

    # The scIB Leiden metric selects resolution by NMI and reports both scores
    # from that same clustering. Keep this paired result alongside independent
    # maxima requested by the user.
    paired_at_best_nmi = best_nmi_row
    best_nmi_resolution = float(best_nmi_row["resolution"])
    leiden_best_nmi_labels = leiden_labels_by_resolution[best_nmi_resolution]

    # Both comparison figures use the same UMAP coordinates, so their label
    # colorings are directly comparable. UMAP only visualizes the embedding;
    # it does not affect the K-means or Leiden scores above.
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ImportError(
            "UMAP plotting requires umap-learn and matplotlib. Install them with "
            "`python -m pip install umap-learn matplotlib`."
        ) from exc

    umap_xy = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
    ).fit_transform(np.asarray(X, dtype=np.float32))

    def plot_labeling(ax, values, title):
        values = pd.Series(values).astype(str).to_numpy()
        categories = sorted(pd.unique(values))
        category_codes = pd.Categorical(values, categories=categories).codes
        cmap = plt.get_cmap("tab20", max(len(categories), 1))
        ax.scatter(
            umap_xy[:, 0], umap_xy[:, 1],
            c=category_codes, cmap=cmap,
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
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=6, frameon=False, ncol=1)

    def save_umap_pair(path, predicted_labels, cluster_title):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        plot_labeling(axes[0], predicted_labels, cluster_title)
        plot_labeling(axes[1], y, "Ground-truth cell type")
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    matched["kmeans_cluster"] = predicted
    matched["leiden_cluster_best_nmi"] = leiden_best_nmi_labels
    matched["umap_1"] = umap_xy[:, 0]
    matched["umap_2"] = umap_xy[:, 1]

    print("\nEvaluation results")
    print("-------------------")
    print(f"Labeled classes in CSV (K): {n_classes}")
    print(f"Unlabeled rows excluded:    {int(is_unlabeled.sum())}")
    print(f"Matched labeled cells:      {len(matched)}")
    print(f"Labeled cells not embedded: {len(labeled) - len(matched)}")
    print(f"Embedding key/shape:        {args.embedding_key} / {X.shape}")
    print(f"K-means ARI (K={n_classes}):   {kmeans_ari:.6f}")
    print(f"K-means NMI (K={n_classes}):   {kmeans_nmi:.6f}")
    print(f"K-means silhouette:            {kmeans_silhouette:.6f}")
    print(f"Cell-type silhouette:          {cell_type_silhouette:.6f}")
    print(
        f"Leiden best ARI:             {best_ari_row['ARI']:.6f} "
        f"(resolution {best_ari_row['resolution']:.1f})"
    )
    print(
        f"Leiden best NMI:             {best_nmi_row['NMI']:.6f} "
        f"(resolution {best_nmi_row['resolution']:.1f})"
    )
    print(
        f"Leiden at best-NMI res:      ARI={paired_at_best_nmi['ARI']:.6f}, "
        f"NMI={paired_at_best_nmi['NMI']:.6f} "
        f"silhouette={paired_at_best_nmi['silhouette']:.6f} "
        f"(resolution {paired_at_best_nmi['resolution']:.1f})"
    )
    print(
        f"Leiden best silhouette:        {best_silhouette_row['silhouette']:.6f} "
        f"(resolution {best_silhouette_row['resolution']:.1f}; "
        f"sample size {min(args.silhouette_sample_size, len(y))})"
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    matched.to_csv(out / "matched_labels.csv", index=False)
    leiden_scores.to_csv(out / "leiden_resolution_metrics.csv", index=False)
    pd.DataFrame([{
        "embedding_key": args.embedding_key,
        "n_cells": len(matched),
        "n_classes_csv": n_classes,
        "unlabeled_rows_excluded": int(is_unlabeled.sum()),
        "labeled_cells_not_embedded": len(labeled) - len(matched),
        "seed": args.seed,
        "leiden_neighbors": args.leiden_neighbors,
        "umap_neighbors": args.umap_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "silhouette_sample_size": min(args.silhouette_sample_size, len(y)),
        "cell_type_silhouette": cell_type_silhouette,
        "kmeans_K": n_classes,
        "kmeans_ARI": kmeans_ari,
        "kmeans_NMI": kmeans_nmi,
        "kmeans_silhouette": kmeans_silhouette,
        "leiden_best_ARI": float(best_ari_row["ARI"]),
        "leiden_best_ARI_resolution": float(best_ari_row["resolution"]),
        "leiden_best_NMI": float(best_nmi_row["NMI"]),
        "leiden_best_NMI_resolution": float(best_nmi_row["resolution"]),
        "leiden_silhouette_at_best_ARI_resolution": float(best_ari_row["silhouette"]),
        "leiden_silhouette_at_best_NMI_resolution": float(best_nmi_row["silhouette"]),
        "leiden_best_silhouette": float(best_silhouette_row["silhouette"]),
        "leiden_best_silhouette_resolution": float(best_silhouette_row["resolution"]),
        "leiden_ARI_at_best_NMI_resolution": float(paired_at_best_nmi["ARI"]),
        "leiden_NMI_at_best_NMI_resolution": float(paired_at_best_nmi["NMI"]),
    }]).to_csv(out / "metrics.csv", index=False)
    save_umap_pair(
        out / "umap_kmeans_vs_ground_truth.png",
        predicted,
        f"K-means clusters (K={n_classes})",
    )
    save_umap_pair(
        out / "umap_leiden_vs_ground_truth.png",
        leiden_best_nmi_labels,
        f"Leiden clusters (NMI-best resolution {best_nmi_resolution:.1f})",
    )
    print(f"\nSaved results to {out.resolve()}")
    print("Leiden sweep saved to leiden_resolution_metrics.csv")
    print("UMAP figures saved to umap_kmeans_vs_ground_truth.png and umap_leiden_vs_ground_truth.png")


if __name__ == "__main__":
    main()
