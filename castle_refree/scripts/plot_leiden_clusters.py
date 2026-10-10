"""Plot exact benchmark Leiden clusters for SPLIT PCA50 and CASTLE z16."""

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
from scib_metrics.metrics._nmi_ari import _compute_clustering_leiden
from scib_metrics.nearest_neighbors import pynndescent


def leiden_labels(x: np.ndarray, resolution: float = 0.6) -> np.ndarray:
    nn90 = pynndescent(x, n_neighbors=90, random_state=0, n_jobs=-1)
    graph15 = nn90.subset_neighbors(n=15).knn_graph_connectivities
    return np.asarray(_compute_clustering_leiden(graph15, resolution, seed=42)).astype(str)


def display_umap(x: np.ndarray) -> np.ndarray:
    obj = ad.AnnData(x.astype(np.float32))
    sc.pp.neighbors(obj, n_neighbors=30, metric="cosine", random_state=7)
    sc.tl.umap(obj, min_dist=0.25, random_state=7)
    return np.asarray(obj.obsm["X_umap"])


def scatter_clusters(ax, xy: np.ndarray, labels: np.ndarray, title: str) -> None:
    values = sorted(np.unique(labels), key=int)
    cmap = plt.get_cmap("tab20", len(values))
    for idx, value in enumerate(values):
        take = labels == value
        ax.scatter(xy[take, 0], xy[take, 1], s=2.3, alpha=0.78,
                   color=cmap(idx), label=f"Cluster {value}", rasterized=True)
    ax.set_title(f"{title}\n{len(values)} Leiden communities")
    ax.set_xlabel("UMAP 1 (display only)")
    ax.set_ylabel("UMAP 2 (display only)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(markerscale=4, fontsize=7, ncol=2, frameon=False,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))


def main() -> None:
    arrays = np.load(RESULT_DIR / "benchmark_embeddings.npz")
    cells = pd.read_csv(RESULT_DIR / "evaluation_cells.csv.gz")
    split = arrays["split_pca50"]
    castle = arrays["castle_intrinsic_z16"]

    print("Reconstructing exact SPLIT Leiden labels", flush=True)
    split_labels = leiden_labels(split)
    print("Reconstructing exact CASTLE Leiden labels", flush=True)
    castle_labels = leiden_labels(castle)

    rng = np.random.RandomState(7)
    sample = np.sort(rng.choice(len(cells), min(30_000, len(cells)), replace=False))
    print("Computing display UMAPs", flush=True)
    split_xy = display_umap(split[sample])
    castle_xy = display_umap(castle[sample])

    pd.DataFrame({
        "cell_id": cells.iloc[sample]["cell_id"].to_numpy(),
        "rctd_primary": cells.iloc[sample]["rctd_primary"].to_numpy(),
        "split_leiden": split_labels[sample],
        "castle_leiden": castle_labels[sample],
        "split_umap_1": split_xy[:, 0],
        "split_umap_2": split_xy[:, 1],
        "castle_umap_1": castle_xy[:, 0],
        "castle_umap_2": castle_xy[:, 1],
    }).to_csv(RESULT_DIR / "split_castle_leiden_umap_coordinates.csv.gz", index=False)

    pd.DataFrame({
        "cell_id": cells["cell_id"],
        "rctd_primary": cells["rctd_primary"],
        "split_leiden": split_labels,
        "castle_leiden": castle_labels,
    }).to_csv(RESULT_DIR / "split_castle_leiden_assignments.csv.gz", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.4), constrained_layout=True)
    scatter_clusters(axes[0], split_xy, split_labels[sample], "SPLIT PCA50")
    scatter_clusters(axes[1], castle_xy, castle_labels[sample], "CASTLE intrinsic z16")
    fig.suptitle("Leiden clustering of the same Xenium cells\n15-neighbour graph, resolution 0.6", fontsize=16)
    fig.savefig(RESULT_DIR / "split_castle_leiden_cluster_overview.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
