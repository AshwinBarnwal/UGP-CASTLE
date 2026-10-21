"""Supplementary Leiden, composition, label, and training plots for CosMx."""

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "cosmx_frontal_cortex_simple_figures"
DATA = ROOT / "data" / "cosmx_frontal_cortex.h5ad"
CASTLE = ROOT / "results" / "castle_refree_cosmx_frontal_cortex_beta2"
SCVI = ROOT / "results" / "scvi_cosmx_frontal_cortex" / "adata_scvi.h5ad"
SCVIVA = ROOT / "results" / "scviva_cosmx_frontal_cortex" / "adata_scviva.h5ad"
SEED = 42


def scatter(ax, xy, values, title):
    values = np.asarray(values).astype(str)
    names = sorted(np.unique(values), key=lambda x: int(x) if x.isdigit() else x)
    palette = dict(zip(names, sns.color_palette("husl", len(names))))
    for value in names:
        keep = values == value
        ax.scatter(xy[keep, 0], xy[keep, 1], s=1.0, c=[palette[value]],
                   linewidths=0, alpha=0.75, rasterized=True)
    ax.set_title(f"{title} ({len(names)} clusters)", weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)
    return palette


def heatmap(ax, truth, clusters, title):
    keep = truth != "Unknown"
    table = pd.crosstab(
        pd.Series(truth[keep], name="Published type"),
        pd.Series(np.asarray(clusters)[keep].astype(str), name="Cluster"),
        normalize="index",
    ) * 100
    sns.heatmap(table, cmap="Blues", vmin=0, vmax=100, ax=ax, cbar=False)
    ax.set_title(title, weight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("")


def main():
    source = ad.read_h5ad(DATA, backed="r")
    labels = source.obs["cell_type"].astype(str).to_numpy()
    spatial = np.asarray(source.obsm["spatial"])
    castle = np.load(CASTLE / "castle_embeddings_and_qc.npz")
    scvi = ad.read_h5ad(SCVI, backed="r")
    scviva = ad.read_h5ad(SCVIVA, backed="r")
    embeddings = {
        "CASTLE intrinsic": castle["z_intrinsic"],
        "CASTLE niche": castle["z_niche"],
        "scVI": np.asarray(scvi.obsm["X_scVI"]),
        "scVIVA": np.asarray(scviva.obsm["X_scVIVA"]),
    }
    cache = np.load(OUT / "plot_umap_coordinates.npz")
    idx = cache["cell_indices"]
    umaps = {name: cache[name.lower().replace(" ", "_")] for name in embeddings}

    leiden = {}
    rows = []
    rng = np.random.default_rng(SEED)
    sil = rng.choice(len(idx), size=min(5_000, len(idx)), replace=False)
    for name, values in embeddings.items():
        sample = ad.AnnData(X=np.asarray(values[idx], dtype=np.float32))
        sc.pp.neighbors(sample, n_neighbors=30, metric="euclidean", random_state=SEED)
        sc.tl.leiden(sample, resolution=1.0, random_state=SEED, key_added="cluster")
        clusters = sample.obs["cluster"].astype(str).to_numpy()
        leiden[name] = clusters
        known = labels[idx] != "Unknown"
        rows.append({
            "method": name,
            "clustering": "Leiden",
            "scope": "fixed_30000_cell_sample",
            "n_neighbors": 30,
            "resolution": 1.0,
            "n_clusters": len(np.unique(clusters)),
            "ARI_vs_published_type_excluding_unknown": adjusted_rand_score(labels[idx][known], clusters[known]),
            "NMI_vs_published_type_excluding_unknown": normalized_mutual_info_score(labels[idx][known], clusters[known]),
            "silhouette_embedding_5000_cell_sample": silhouette_score(values[idx][sil], clusters[sil]),
        })
    pd.DataFrame(rows).to_csv(OUT / "completed_methods_leiden_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    palettes = {}
    for ax, name in zip(axes.flat, embeddings):
        palettes[name] = scatter(ax, umaps[name], leiden[name], name)
    fig.suptitle("CosMx embeddings clustered with Leiden (30 neighbours, resolution 1.0)", weight="bold", fontsize=15)
    fig.subplots_adjust(top=0.92, wspace=0.08, hspace=0.12)
    fig.savefig(OUT / "04_completed_methods_umap_by_leiden.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, name in zip(axes.flat, embeddings):
        for value, color in palettes[name].items():
            keep = leiden[name] == value
            ax.scatter(spatial[idx[keep], 0], spatial[idx[keep], 1], s=0.7, c=[color],
                       linewidths=0, alpha=0.75, rasterized=True)
        ax.set_title(f"{name} ({len(palettes[name])} clusters)", weight="bold")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
    fig.suptitle("Where each embedding's Leiden clusters lie in the tissue", weight="bold", fontsize=15)
    fig.subplots_adjust(top=0.92, wspace=0.04, hspace=0.08)
    fig.savefig(OUT / "05_completed_methods_spatial_by_leiden.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    kmeans = {
        name: MiniBatchKMeans(n_clusters=7, random_state=SEED, batch_size=4096, n_init=10).fit_predict(values)
        for name, values in embeddings.items()
    }
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    for col, name in enumerate(embeddings):
        heatmap(axes[0, col], labels, kmeans[name], name)
        heatmap(axes[1, col], labels[idx], leiden[name], name)
    axes[0, 0].set_ylabel("K-means\nPublished cell type")
    axes[1, 0].set_ylabel("Leiden\nPublished cell type")
    fig.suptitle("Cluster composition: row-normalized percentage of each published cell type", weight="bold", fontsize=15)
    fig.subplots_adjust(top=0.92, wspace=0.35, hspace=0.45)
    fig.savefig(OUT / "06_cluster_vs_published_type_heatmaps.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    type_names = sorted(np.unique(labels))
    colors = dict(zip(type_names, sns.color_palette("tab10", len(type_names))))
    fig, ax = plt.subplots(figsize=(10, 9))
    for value in type_names:
        keep = labels == value
        ax.scatter(spatial[keep, 0], spatial[keep, 1], s=0.35, c=[colors[value]], label=value,
                   linewidths=0, alpha=0.65, rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("CosMx frontal cortex: published cell-type annotations", weight="bold", fontsize=15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, markerscale=6)
    sns.despine(ax=ax, left=True, bottom=True)
    fig.savefig(OUT / "07_spatial_published_cell_types.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    histories = [
        ("CASTLE", pd.read_csv(CASTLE / "training_history.csv"), "loss", None),
        ("scVI", pd.read_csv(SCVI.parent / "training_history.csv"), "train_loss", "validation_loss"),
        ("scVIVA", pd.read_csv(SCVIVA.parent / "training_history.csv"), "train_loss", "validation_loss"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, frame, train, validation) in zip(axes, histories):
        ax.plot(np.arange(len(frame)) + 1, frame[train], label="total" if validation is None else "train")
        if validation is not None:
            ax.plot(np.arange(len(frame)) + 1, frame[validation], label="validation")
        ax.set_title(name, weight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False)
        sns.despine(ax=ax)
    fig.suptitle("Training diagnostics for completed CosMx methods", weight="bold", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "08_training_diagnostics.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved supplementary figures to {OUT}")


if __name__ == "__main__":
    main()
