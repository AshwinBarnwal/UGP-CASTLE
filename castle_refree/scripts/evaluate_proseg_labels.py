"""Evaluate ProSeg CASTLE embeddings with K-means and a Leiden resolution sweep."""

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
UNLABELED_NAMES = {"unlabeled", "unlabelled"}
DEFAULT_SILHOUETTE_SAMPLE_SIZE = 2000


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
        description=(
            "Evaluate CASTLE embeddings against H5AD cell_type labels with "
            "unlabeled cells excluded, K-means K=17, and Leiden resolutions 0.2 to 2.0."
        )
    )
    parser.add_argument("--input", required=True, help="Exported CASTLE bundle directory")
    parser.add_argument("--results", required=True, help="CASTLE result directory")
    parser.add_argument("--label-column", default="cell_type")
    parser.add_argument("--embedding-key", default="z_intrinsic")
    parser.add_argument("--output", default=None, help="Defaults to <results>/label_evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leiden-neighbors", type=int, default=15)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.3)
    parser.add_argument("--silhouette-sample-size", type=int, default=DEFAULT_SILHOUETTE_SAMPLE_SIZE)
    args = parser.parse_args()
    if args.silhouette_sample_size < 2:
        raise ValueError("--silhouette-sample-size must be at least 2.")

    input_dir = Path(args.input)
    result_dir = Path(args.results)
    annotation_path = input_dir / "cell_annotations.csv"
    annotations = pd.read_csv(annotation_path, dtype={"cell_id": str})
    required_columns = {"cell_id", args.label_column}
    missing_columns = required_columns - set(annotations.columns)
    if missing_columns:
        raise ValueError(f"Annotation table is missing columns: {sorted(missing_columns)}")
    annotations["cell_id"] = annotations["cell_id"].str.strip()
    annotations[args.label_column] = annotations[args.label_column].fillna("").astype(str).str.strip()
    if annotations["cell_id"].duplicated().any():
        raise ValueError("Duplicate cell_id values in cell_annotations.csv.")

    unlabeled_mask = (
        annotations[args.label_column].str.casefold().isin(UNLABELED_NAMES)
        | annotations[args.label_column].eq("")
    )
    n_unlabeled_excluded = int(unlabeled_mask.sum())
    annotations = annotations.loc[~unlabeled_mask].copy()

    # Exclude explicit Unlabeled/Unlabelled and blank annotations from all
    # clustering, scores, and plots. The remaining ProSeg labels define K=17.
    n_types = annotations[args.label_column].nunique()
    if n_types != N_CELL_TYPES:
        raise ValueError(
            f"Expected {N_CELL_TYPES} labeled cell_type classes after excluding unlabeled cells, "
            f"found {n_types}. Check the H5AD annotation column and exported bundle."
        )

    cells = pd.read_csv(result_dir / "cells.tsv", sep="\t", header=None, dtype=str).iloc[:, 0]
    cells = cells.str.strip()
    if cells.duplicated().any():
        raise ValueError("Duplicate cell IDs in result cells.tsv.")
    npz = np.load(result_dir / "castle_embeddings_and_qc.npz", allow_pickle=False)
    if args.embedding_key not in npz.files:
        raise KeyError(f"{args.embedding_key!r} not found; available arrays: {npz.files}")
    embeddings = npz[args.embedding_key]
    if embeddings.ndim != 2 or len(cells) != len(embeddings):
        raise ValueError(
            f"cells.tsv and embedding rows must match; got {len(cells)} cells and "
            f"embedding shape {embeddings.shape}."
        )

    embedding_cells = pd.DataFrame({"cell_id": cells, "embedding_row": np.arange(len(cells))})
    matched = annotations.merge(
        embedding_cells, on="cell_id", how="inner", validate="one_to_one"
    )
    if matched.empty:
        raise ValueError("No H5AD cell IDs matched result cells.tsv.")
    if matched[args.label_column].nunique() != N_CELL_TYPES:
        absent = sorted(
            set(annotations[args.label_column]) - set(matched[args.label_column])
        )
        raise ValueError(
            f"The CASTLE result is missing one or more of the {N_CELL_TYPES} labeled classes: {absent}"
        )

    rows = matched["embedding_row"].to_numpy(dtype=np.int64)
    X = embeddings[rows]
    y = matched[args.label_column].to_numpy(dtype=str)
    if not np.isfinite(X).all():
        raise ValueError("Selected embeddings contain NaN or infinite values.")

    predicted = KMeans(n_clusters=N_CELL_TYPES, random_state=args.seed, n_init=10).fit_predict(X)
    kmeans_ari = adjusted_rand_score(y, predicted)
    kmeans_nmi = normalized_mutual_info_score(y, predicted)
    kmeans_silhouette = sampled_silhouette(X, predicted, args.silhouette_sample_size, args.seed)
    cell_type_silhouette = sampled_silhouette(X, y, args.silhouette_sample_size, args.seed)

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
    resolution_rows: list[dict[str, float | int]] = []
    leiden_labels_by_resolution: dict[float, np.ndarray] = {}
    for resolution in np.round(np.arange(0.2, 2.01, 0.2), 1):
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
        resolution_key = float(resolution)
        leiden_labels_by_resolution[resolution_key] = leiden_labels
        resolution_rows.append(
            {
                "resolution": resolution_key,
                "n_clusters": int(pd.Series(leiden_labels).nunique()),
                "ARI": float(adjusted_rand_score(y, leiden_labels)),
                "NMI": float(
                    normalized_mutual_info_score(
                        y, leiden_labels, average_method="arithmetic"
                    )
                ),
                "silhouette": sampled_silhouette(
                    X, leiden_labels, args.silhouette_sample_size, args.seed
                ),
            }
        )
    leiden_scores = pd.DataFrame(resolution_rows)
    best_ari_row = leiden_scores.loc[leiden_scores["ARI"].idxmax()]
    best_nmi_row = leiden_scores.loc[leiden_scores["NMI"].idxmax()]
    best_silhouette_row = leiden_scores.loc[leiden_scores["silhouette"].idxmax()]
    best_nmi_resolution = float(best_nmi_row["resolution"])
    leiden_best_nmi_labels = leiden_labels_by_resolution[best_nmi_resolution]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap
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

    def plot_labeling(ax, values: np.ndarray, title: str) -> None:
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

    def save_umap_pair(path: Path, cluster_labels: np.ndarray, cluster_title: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        plot_labeling(axes[0], cluster_labels, cluster_title)
        plot_labeling(axes[1], y, "Ground-truth cell type (unlabeled excluded)")
        save_png(fig, path)
        plt.close(fig)

    matched["kmeans_cluster"] = predicted
    matched["leiden_cluster_best_nmi"] = leiden_best_nmi_labels
    matched["umap_1"] = umap_xy[:, 0]
    matched["umap_2"] = umap_xy[:, 1]

    output_dir = Path(args.output) if args.output else result_dir / "label_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_dir / "matched_labels.csv", index=False)
    leiden_scores.to_csv(output_dir / "leiden_resolution_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "embedding_key": args.embedding_key,
                "n_cells": len(matched),
                "n_cell_types": N_CELL_TYPES,
                "unlabeled_cells_excluded": n_unlabeled_excluded,
                "annotated_cells_not_embedded": len(annotations) - len(matched),
                "seed": args.seed,
                "leiden_neighbors": args.leiden_neighbors,
                "umap_neighbors": args.umap_neighbors,
                "umap_min_dist": args.umap_min_dist,
                "silhouette_sample_size": min(args.silhouette_sample_size, len(y)),
                "cell_type_silhouette": cell_type_silhouette,
                "kmeans_K": N_CELL_TYPES,
                "kmeans_ARI": float(kmeans_ari),
                "kmeans_NMI": float(kmeans_nmi),
                "kmeans_silhouette": kmeans_silhouette,
                "leiden_best_ARI": float(best_ari_row["ARI"]),
                "leiden_best_ARI_resolution": float(best_ari_row["resolution"]),
                "leiden_best_NMI": float(best_nmi_row["NMI"]),
                "leiden_best_NMI_resolution": best_nmi_resolution,
                "leiden_silhouette_at_best_ARI_resolution": float(best_ari_row["silhouette"]),
                "leiden_silhouette_at_best_NMI_resolution": float(best_nmi_row["silhouette"]),
                "leiden_best_silhouette": float(best_silhouette_row["silhouette"]),
                "leiden_best_silhouette_resolution": float(best_silhouette_row["resolution"]),
                "leiden_ARI_at_best_NMI_resolution": float(best_nmi_row["ARI"]),
                "leiden_NMI_at_best_NMI_resolution": float(best_nmi_row["NMI"]),
            }
        ]
    ).to_csv(output_dir / "metrics.csv", index=False)
    save_umap_pair(
        output_dir / "umap_kmeans_vs_ground_truth.png",
        predicted,
        f"K-means clusters (K={N_CELL_TYPES})",
    )
    save_umap_pair(
        output_dir / "umap_leiden_vs_ground_truth.png",
        leiden_best_nmi_labels,
        f"Leiden clusters (NMI-best resolution {best_nmi_resolution:.1f})",
    )

    print("Evaluation results")
    print("-------------------")
    print(f"Ground-truth cell types (K): {N_CELL_TYPES} (unlabeled excluded)")
    print(f"Matched labeled cells:       {len(matched)}")
    print(f"Unlabeled cells excluded:    {n_unlabeled_excluded}")
    print(f"Labeled cells not embedded:  {len(annotations) - len(matched)}")
    print(f"Embedding key/shape:         {args.embedding_key} / {X.shape}")
    print(f"K-means ARI (K={N_CELL_TYPES}):          {kmeans_ari:.6f}")
    print(f"K-means NMI (K={N_CELL_TYPES}):          {kmeans_nmi:.6f}")
    print(f"K-means silhouette:                    {kmeans_silhouette:.6f}")
    print(f"Cell-type silhouette:                  {cell_type_silhouette:.6f}")
    print(
        f"Leiden best ARI:             {best_ari_row['ARI']:.6f} "
        f"(resolution {best_ari_row['resolution']:.1f})"
    )
    print(
        f"Leiden best NMI:             {best_nmi_row['NMI']:.6f} "
        f"(resolution {best_nmi_resolution:.1f})"
    )
    print(
        f"Leiden at best-NMI res:      ARI={best_nmi_row['ARI']:.6f}, "
        f"NMI={best_nmi_row['NMI']:.6f}, silhouette={best_nmi_row['silhouette']:.6f}"
    )
    print(
        f"Best Leiden silhouette:      {best_silhouette_row['silhouette']:.6f} "
        f"(resolution {best_silhouette_row['resolution']:.1f}; sample size {min(args.silhouette_sample_size, len(y))})"
    )
    print(f"Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
