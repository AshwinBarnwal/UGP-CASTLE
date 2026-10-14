"""Create lightweight, directly comparable CosMx cortex figures.

The script deliberately uses one fixed cell sample and one plotting pipeline for
every embedding.  It currently plots CASTLE intrinsic/niche and scVI, the
methods whose full outputs are complete.  It does not disturb the training
pipeline running in another process.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "cosmx_frontal_cortex.h5ad"
CASTLE_DIR = ROOT / "results" / "castle_refree_cosmx_frontal_cortex_beta2"
SCVI_FILE = ROOT / "results" / "scvi_cosmx_frontal_cortex" / "adata_scvi.h5ad"
OUT = ROOT / "results" / "cosmx_frontal_cortex_simple_figures"
SEED = 42
PLOT_CELLS = 30_000


def embed_for_plot(x: np.ndarray) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        n_components=2,
        metric="euclidean",
        random_state=SEED,
        low_memory=True,
    ).fit_transform(x)


def scatter_panel(ax, xy, values, palette, title, size=1.1):
    values = np.asarray(values).astype(str)
    for value in sorted(np.unique(values)):
        mask = values == value
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=size,
            c=[palette[value]],
            label=value,
            linewidths=0,
            alpha=0.72,
            rasterized=True,
        )
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    sns.despine(ax=ax, left=True, bottom=True)


def save_legend(fig, handles, labels, ncol=4):
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=ncol,
        frameon=False,
        markerscale=5,
        fontsize=9,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    source = ad.read_h5ad(DATA, backed="r")
    labels = source.obs["cell_type"].astype(str).to_numpy()
    spatial = np.asarray(source.obsm["spatial"])
    source_ids = source.obs_names.to_numpy().astype(str)

    castle_ids = pd.read_csv(CASTLE_DIR / "cells.tsv", header=None)[0].astype(str).to_numpy()
    if not np.array_equal(source_ids, castle_ids):
        raise ValueError("CASTLE cells.tsv is not in the same order as the CosMx H5AD.")
    castle = np.load(CASTLE_DIR / "castle_embeddings_and_qc.npz")
    z_intrinsic = castle["z_intrinsic"]
    z_niche = castle["z_niche"]

    scvi_data = ad.read_h5ad(SCVI_FILE, backed="r")
    if not np.array_equal(source_ids, scvi_data.obs_names.to_numpy().astype(str)):
        raise ValueError("scVI cells are not in the same order as the CosMx H5AD.")
    z_scvi = np.asarray(scvi_data.obsm["X_scVI"])

    rng = np.random.default_rng(SEED)
    plot_idx = np.sort(rng.choice(len(labels), size=min(PLOT_CELLS, len(labels)), replace=False))
    embeddings = {
        "CASTLE intrinsic": z_intrinsic,
        "CASTLE niche": z_niche,
        "scVI": z_scvi,
    }

    print(f"Computing UMAPs on the same {len(plot_idx):,} cells...")
    umaps = {name: embed_for_plot(x[plot_idx]) for name, x in embeddings.items()}
    np.savez_compressed(OUT / "plot_umap_coordinates.npz", cell_indices=plot_idx, **{
        name.lower().replace(" ", "_"): xy for name, xy in umaps.items()
    })

    type_names = sorted(np.unique(labels[plot_idx]))
    type_colors = dict(zip(type_names, sns.color_palette("tab10", len(type_names))))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=False)
    for ax, (name, xy) in zip(axes, umaps.items()):
        scatter_panel(ax, xy, labels[plot_idx], type_colors, name)
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend:
            legend.remove()
    fig.suptitle("CosMx frontal cortex: embeddings coloured by published cell type", fontsize=16, weight="bold")
    save_legend(fig, handles, legend_labels, ncol=4)
    fig.subplots_adjust(bottom=0.17, top=0.88, wspace=0.08)
    fig.savefig(OUT / "01_completed_methods_umap_by_published_type.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    known = labels != "Unknown"
    metrics = []
    clusterings = {}
    for name, x in embeddings.items():
        km = MiniBatchKMeans(n_clusters=7, random_state=SEED, batch_size=4096, n_init=10)
        clusters = km.fit_predict(x)
        clusterings[name] = clusters
        sil_idx = rng.choice(len(labels), size=min(10_000, len(labels)), replace=False)
        metrics.append({
            "method": name,
            "clustering": "MiniBatchKMeans",
            "n_clusters": 7,
            "ARI_vs_published_type_excluding_unknown": adjusted_rand_score(labels[known], clusters[known]),
            "NMI_vs_published_type_excluding_unknown": normalized_mutual_info_score(labels[known], clusters[known]),
            "silhouette_embedding_10000_cell_sample": silhouette_score(x[sil_idx], clusters[sil_idx]),
        })
    pd.DataFrame(metrics).to_csv(OUT / "completed_methods_kmeans_metrics.csv", index=False)

    cluster_names = [str(i) for i in range(7)]
    cluster_colors = dict(zip(cluster_names, sns.color_palette("tab10", 7)))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, xy) in zip(axes, umaps.items()):
        scatter_panel(ax, xy, clusterings[name][plot_idx].astype(str), cluster_colors, name)
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend:
            legend.remove()
    fig.suptitle("Same cells, clustered independently in each embedding (K-means, K=7)", fontsize=16, weight="bold")
    save_legend(fig, handles, legend_labels, ncol=7)
    fig.subplots_adjust(bottom=0.15, top=0.88, wspace=0.08)
    fig.savefig(OUT / "02_completed_methods_umap_by_kmeans.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, name in zip(axes, embeddings):
        values = clusterings[name][plot_idx].astype(str)
        for value in cluster_names:
            mask = values == value
            ax.scatter(spatial[plot_idx[mask], 0], spatial[plot_idx[mask], 1], s=0.7,
                       c=[cluster_colors[value]], linewidths=0, alpha=0.75, rasterized=True)
        ax.set_title(name, fontsize=13, weight="bold")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
    fig.suptitle("Where each embedding's K-means clusters lie in the tissue", fontsize=16, weight="bold")
    fig.subplots_adjust(top=0.88, wspace=0.04)
    fig.savefig(OUT / "03_completed_methods_spatial_by_kmeans.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "cells_total": int(len(labels)),
        "cells_plotted_same_sample": int(len(plot_idx)),
        "published_label_column": "cell_type",
        "unknown_excluded_from_ari_nmi": True,
        "kmeans_clusters": 7,
        "random_seed": SEED,
        "methods": list(embeddings),
    }
    (OUT / "figure_method.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved figures and metrics to {OUT}")


if __name__ == "__main__":
    main()
