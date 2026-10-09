"""Create a presentation-ready SPLIT PCA50 cluster overview."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO_ROOT / "results" / "benchmark_split_castle_full"
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scib_metrics.metrics._nmi_ari import _compute_clustering_kmeans


def main() -> None:
    embedding = np.load(RESULT_DIR / "benchmark_embeddings.npz")["split_pca50"]
    cells = pd.read_csv(RESULT_DIR / "evaluation_cells.csv.gz")
    if len(cells) != len(embedding):
        raise ValueError("Embedding and evaluation-cell table differ")

    print("Clustering full SPLIT embedding into K=17", flush=True)
    kmeans = np.asarray(_compute_clustering_kmeans(embedding, 17)).astype(str)

    rng = np.random.RandomState(7)
    sample = np.sort(rng.choice(len(embedding), min(30_000, len(embedding)), replace=False))
    obj = ad.AnnData(embedding[sample].astype(np.float32))
    print("Computing display-only UMAP for 30,000 cells", flush=True)
    sc.pp.neighbors(obj, n_neighbors=30, metric="cosine", random_state=7)
    sc.tl.umap(obj, min_dist=0.25, random_state=7)
    xy = np.asarray(obj.obsm["X_umap"])

    plot_data = pd.DataFrame({
        "cell_id": cells.iloc[sample]["cell_id"].to_numpy(),
        "rctd_primary": cells.iloc[sample]["rctd_primary"].astype(str).to_numpy(),
        "kmeans_cluster": kmeans[sample],
        "umap_1": xy[:, 0],
        "umap_2": xy[:, 1],
    })
    plot_data.to_csv(RESULT_DIR / "split_cluster_umap_coordinates.csv.gz", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), constrained_layout=True)
    cluster_values = sorted(plot_data["kmeans_cluster"].unique(), key=int)
    type_values = sorted(plot_data["rctd_primary"].unique())
    cluster_cmap = plt.get_cmap("tab20", len(cluster_values))
    type_cmap = plt.get_cmap("tab20", len(type_values))

    for idx, value in enumerate(cluster_values):
        take = plot_data["kmeans_cluster"].to_numpy() == value
        axes[0].scatter(xy[take, 0], xy[take, 1], s=2.2, alpha=0.75,
                        color=cluster_cmap(idx), label=f"Cluster {value}", rasterized=True)
    axes[0].set_title("SPLIT PCA50: K-means clusters (K=17)")
    axes[0].legend(markerscale=4, fontsize=7, ncol=2, frameon=False,
                   loc="upper left", bbox_to_anchor=(1.0, 1.0))

    for idx, value in enumerate(type_values):
        take = plot_data["rctd_primary"].to_numpy() == value
        axes[1].scatter(xy[take, 0], xy[take, 1], s=2.2, alpha=0.75,
                        color=type_cmap(idx), label=value, rasterized=True)
    axes[1].set_title("Same SPLIT embedding: RCTD reference types")
    axes[1].legend(markerscale=4, fontsize=7, ncol=2, frameon=False,
                   loc="upper left", bbox_to_anchor=(1.0, 1.0))

    for ax in axes:
        ax.set_xlabel("UMAP 1 (display only)")
        ax.set_ylabel("UMAP 2 (display only)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("SPLIT-purified Xenium cells", fontsize=16)
    fig.savefig(RESULT_DIR / "split_pca50_cluster_overview.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
