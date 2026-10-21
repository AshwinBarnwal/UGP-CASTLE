"""Create the requested lung L1 CASTLE/RCTD/SPLIT comparison figures.

RCTD primary labels are the external comparison labels. CASTLE is evaluated in
its native 16-dimensional intrinsic embedding. SPLIT counts receive the paper's
downstream transform: library-size normalization, log1p, then PCA50.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import umap
from scipy import sparse
from scipy.io import mmread
from scib_metrics.metrics._nmi_ari import _compute_clustering_leiden
from scib_metrics.nearest_neighbors import pynndescent
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


SEED = 42
SILHOUETTE_CELLS = 10_000


def read_vector(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep="\t", header=None, dtype=str).iloc[:, 0].to_numpy()


def split_pca50(matrix_path: Path, cells: np.ndarray, selected: np.ndarray) -> np.ndarray:
    x = sparse.csr_matrix(mmread(matrix_path), dtype=np.float32)
    if x.shape[1] == len(cells):
        x = x.T.tocsr()
    elif x.shape[0] != len(cells):
        raise ValueError(f"Unexpected SPLIT matrix shape {x.shape}")
    x = x[pd.Index(cells).get_indexer(selected)]
    obj = ad.AnnData(x)
    sc.pp.normalize_total(obj, target_sum=1e4)
    sc.pp.log1p(obj)
    sc.tl.pca(obj, n_comps=min(50, x.shape[1] - 1), random_state=SEED)
    return np.asarray(obj.obsm["X_pca"], dtype=np.float32)


def umap2(x: np.ndarray) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=15, min_dist=0.3, n_components=2,
        metric="euclidean", random_state=SEED, low_memory=True,
    ).fit_transform(x).astype(np.float32)


def clusters_and_metrics(x: np.ndarray, truth: np.ndarray, name: str):
    n_types = pd.Series(truth).nunique()
    km = KMeans(n_clusters=n_types, random_state=SEED, n_init=10).fit_predict(x)

    neighbors = pynndescent(x, n_neighbors=15, random_state=SEED, n_jobs=-1)
    graph = neighbors.knn_graph_connectivities
    candidates = []
    for resolution in np.arange(0.2, 2.01, 0.2):
        labels = _compute_clustering_leiden(graph, float(resolution), seed=SEED)
        nmi = normalized_mutual_info_score(truth, labels, average_method="arithmetic")
        candidates.append((nmi, float(resolution), labels))
    _, best_resolution, leiden = max(candidates, key=lambda item: item[0])

    rng = np.random.RandomState(SEED + 17)
    sample = np.arange(len(x)) if len(x) <= SILHOUETTE_CELLS else np.sort(
        rng.choice(len(x), SILHOUETTE_CELLS, replace=False)
    )
    rows = []
    for algorithm, labels in (("K-means", km), ("Leiden", leiden)):
        rows.append({
            "method": name,
            "clustering": algorithm,
            "cells": len(x),
            "dimensions": x.shape[1],
            "clusters": int(np.unique(labels).size),
            "ARI": adjusted_rand_score(truth, labels),
            "NMI": normalized_mutual_info_score(truth, labels, average_method="arithmetic"),
            "silhouette": silhouette_score(x[sample], labels[sample]),
            "silhouette_cells": len(sample),
            "leiden_neighbors": 15 if algorithm == "Leiden" else np.nan,
            "leiden_resolution": best_resolution if algorithm == "Leiden" else np.nan,
        })
    return km, leiden, best_resolution, rows


def categorical_scatter(ax, xy, labels, title, size=1.0, legend=False, invert_y=False):
    labels = np.asarray(labels).astype(str)
    categories = sorted(pd.unique(labels))
    cmap = plt.get_cmap("turbo", max(len(categories), 2))
    for i, category in enumerate(categories):
        mask = labels == category
        ax.scatter(
            xy[mask, 0], xy[mask, 1], s=size, c=[cmap(i)], label=category,
            linewidths=0, alpha=0.82, rasterized=True,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    if invert_y:
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
    if legend:
        ax.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
            markerscale=5, fontsize=7, ncol=1,
        )


def save_pair(path, left_xy, left_labels, left_title, right_xy, right_labels,
              right_title, spatial=False, right_legend=False):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    size = 0.75 if spatial else 1.2
    categorical_scatter(axes[0], left_xy, left_labels, left_title, size=size, invert_y=spatial)
    categorical_scatter(
        axes[1], right_xy, right_labels, right_title, size=size,
        legend=right_legend, invert_y=spatial,
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    castle_dir = ROOT / "results" / "castle_refree_xenium_lung_L1_beta2_100epochs"
    rctd_dir = ROOT / "results" / "lung_L1_rctd_split"
    split_dir = ROOT / "data" / "benchmark_lung_split_castle"
    spatial_dir = ROOT / "data" / "castle_xenium_lung_L1"
    out = ROOT / "results" / "lung_L1_castle_split_comparison"
    out.mkdir(parents=True, exist_ok=True)

    castle_cells = read_vector(castle_dir / "cells.tsv")
    z = np.load(castle_dir / "castle_embeddings_and_qc.npz")["z_intrinsic"].astype(np.float32)
    labels_df = pd.read_csv(rctd_dir / "rctd_primary_labels.csv", dtype=str)
    labels_df = labels_df.dropna(subset=["primary_cell_type"]).drop_duplicates("cell_id")
    truth_series = labels_df.set_index("cell_id")["primary_cell_type"]
    split_cells = read_vector(split_dir / "split_cells.tsv")

    common = castle_cells[
        pd.Index(castle_cells).isin(truth_series.index)
        & pd.Index(castle_cells).isin(pd.Index(split_cells))
    ]
    castle_rows = pd.Index(castle_cells).get_indexer(common)
    truth = truth_series.loc[common].to_numpy(str)
    castle_z = z[castle_rows]
    split_x = split_pca50(split_dir / "split_purified_counts.mtx", split_cells, common)

    spatial_df = pd.read_csv(spatial_dir / "spatial.csv", dtype={"cell_id": str}).set_index("cell_id")
    physical_xy = spatial_df.loc[common, ["x", "y"]].to_numpy(np.float32)

    print(f"Shared cohort: {len(common):,} cells; {pd.Series(truth).nunique()} RCTD types", flush=True)
    castle_km, castle_leiden, castle_res, castle_metrics = clusters_and_metrics(
        castle_z, truth, "CASTLE intrinsic z16"
    )
    split_km, split_leiden, split_res, split_metrics = clusters_and_metrics(
        split_x, truth, "SPLIT PCA50"
    )

    print("Computing CASTLE UMAP", flush=True)
    castle_umap = umap2(castle_z)
    print("Computing SPLIT UMAP", flush=True)
    split_umap = umap2(split_x)

    # Two requested direct comparisons: one CASTLE UMAP, recolored on the right
    # by RCTD's primary label (there is intentionally no PCA on the RCTD side).
    save_pair(
        out / "01_castle_kmeans_vs_rctd_primary_umap.png",
        castle_umap, castle_km, "CASTLE UMAP — K-means clusters",
        castle_umap, truth, "Same CASTLE UMAP — RCTD primary type",
        right_legend=True,
    )
    save_pair(
        out / "02_castle_leiden_vs_rctd_primary_umap.png",
        castle_umap, castle_leiden, "CASTLE UMAP — Leiden clusters",
        castle_umap, truth, "Same CASTLE UMAP — RCTD primary type",
        right_legend=True,
    )

    # Four requested CASTLE-vs-SPLIT comparisons.
    save_pair(
        out / "03_castle_vs_split_kmeans_umap.png",
        castle_umap, castle_km, "CASTLE intrinsic UMAP — K-means",
        split_umap, split_km, "SPLIT PCA50 UMAP — K-means",
    )
    save_pair(
        out / "04_castle_vs_split_leiden_umap.png",
        castle_umap, castle_leiden, "CASTLE intrinsic UMAP — Leiden",
        split_umap, split_leiden, "SPLIT PCA50 UMAP — Leiden",
    )
    save_pair(
        out / "05_castle_vs_split_kmeans_spatial.png",
        physical_xy, castle_km, "CASTLE — K-means in physical space",
        physical_xy, split_km, "SPLIT — K-means in physical space",
        spatial=True,
    )
    save_pair(
        out / "06_castle_vs_split_leiden_spatial.png",
        physical_xy, castle_leiden, "CASTLE — Leiden in physical space",
        physical_xy, split_leiden, "SPLIT — Leiden in physical space",
        spatial=True,
    )

    metrics = pd.DataFrame(castle_metrics + split_metrics)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({
        "cell_id": common,
        "rctd_primary": truth,
        "castle_kmeans": castle_km,
        "castle_leiden": castle_leiden,
        "split_kmeans": split_km,
        "split_leiden": split_leiden,
        "castle_umap_1": castle_umap[:, 0],
        "castle_umap_2": castle_umap[:, 1],
        "split_umap_1": split_umap[:, 0],
        "split_umap_2": split_umap[:, 1],
    }).to_csv(out / "cell_assignments_and_umap.csv.gz", index=False)
    manifest = {
        "dataset": "GSE311609 / GSM9509145 / NSCLC Lung L1",
        "cells": len(common),
        "rctd_primary_types": int(pd.Series(truth).nunique()),
        "rctd_mode": "doublet; primary type used as comparison label",
        "castle_representation": "native 16D intrinsic embedding; no PCA",
        "split_representation": "purified counts -> normalize_total(10000) -> log1p -> PCA50",
        "umap": {"neighbors": 15, "min_dist": 0.3, "random_seed": SEED},
        "kmeans": {"k": int(pd.Series(truth).nunique()), "random_seed": SEED, "n_init": 10},
        "leiden": {
            "neighbors": 15,
            "resolution_grid": "0.2 to 2.0 by 0.2; selected by maximum NMI to RCTD primary labels",
            "castle_resolution": castle_res,
            "split_resolution": split_res,
            "random_seed": SEED,
        },
        "silhouette": f"Euclidean silhouette on one fixed {min(len(common), SILHOUETTE_CELLS)}-cell sample",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    print(f"Saved requested outputs to {out}", flush=True)


if __name__ == "__main__":
    main()
