"""Re-render cached tuned-CosMx figures with a legend for every colour mapping."""

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "cosmx_tuned_castle_hvg2000_figures"
SOURCE = ROOT / "data" / "cosmx_frontal_cortex.h5ad"


def make_palette(values, truth=False):
    values = np.asarray(values).astype(str)
    names = sorted(np.unique(values), key=lambda x: int(x) if x.isdigit() else x)
    colors = dict(zip(names, sns.color_palette("tab10" if truth else "husl", len(names))))
    if truth and "Unknown" in colors:
        colors["Unknown"] = (0.68, 0.68, 0.68)
    return names, colors


def panel(ax, xy, values, title, truth=False, spatial=False):
    values = np.asarray(values).astype(str)
    names, colors = make_palette(values, truth=truth)
    for value in names:
        keep = values == value
        ax.scatter(xy[keep, 0], xy[keep, 1], s=0.8 if spatial else 1.2,
                   c=[colors[value]], label=value, linewidths=0, alpha=0.76,
                   rasterized=True)
    ax.set_title(title, fontsize=12.5, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if spatial:
        ax.set_aspect("equal")
        ax.invert_yaxis()
    else:
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4,
              frameon=False, markerscale=5, fontsize=7.5,
              title="Published type" if truth else "Cluster")
    sns.despine(ax=ax, left=True, bottom=True)


def save_three(path, xy, panels, title, spatial=False):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.4))
    for ax, (values, subtitle, truth) in zip(axes, panels):
        panel(ax, xy, values, subtitle, truth=truth, spatial=spatial)
    fig.suptitle(title, fontsize=16, weight="bold")
    fig.subplots_adjust(top=0.86, bottom=0.27, wspace=0.10)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    cache = np.load(OUT / "plot_data.npz", allow_pickle=True)
    metrics = pd.read_csv(OUT / "ari_nmi_silhouette.csv")
    castle_best = metrics[(metrics.method == "CASTLE tuned intrinsic") & (metrics.clustering == "Leiden")].iloc[0]
    scvi_best = metrics[(metrics.method == "scVI") & (metrics.clustering == "Leiden")].iloc[0]

    save_three(
        OUT / "01_castle_umap_leiden_kmeans_published.png", cache["castle_umap"],
        [
            (cache["castle_leiden"], f"Selected Leiden\nkNN={int(castle_best.neighbors)}, resolution={castle_best.resolution:.1f}", False),
            (cache["castle_kmeans"], "K-means (K=7)", False),
            (cache["published_labels"], "Published cell-type labels", True),
        ],
        "Tuned CASTLE intrinsic embedding — three views of the same UMAP",
    )

    for algorithm, keys, bests, filename in (
        ("Selected Leiden", ("castle_leiden", "scvi_leiden"), (castle_best, scvi_best), "02_castle_vs_scvi_umap_leiden.png"),
        ("K-means (K=7)", ("castle_kmeans", "scvi_kmeans"), (None, None), "03_castle_vs_scvi_umap_kmeans.png"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
        for ax, name, xy_key, labels_key, best in zip(
            axes, ("CASTLE tuned intrinsic", "scVI"), ("castle_umap", "scvi_umap"), keys, bests
        ):
            suffix = "" if best is None else f"\nkNN={int(best.neighbors)}, resolution={best.resolution:.1f}"
            panel(ax, cache[xy_key], cache[labels_key], name + suffix)
        fig.suptitle(f"CosMx 2,000-HVG embeddings — {algorithm}", fontsize=16, weight="bold")
        fig.subplots_adjust(top=0.85, bottom=0.24, wspace=0.10)
        fig.savefig(OUT / filename, dpi=260, bbox_inches="tight")
        plt.close(fig)

    source = ad.read_h5ad(SOURCE, backed="r")
    physical = np.asarray(source.obsm["spatial"], dtype=np.float32)[cache["source_rows"]]
    save_three(
        OUT / "04_castle_spatial_leiden_kmeans_published.png", physical,
        [
            (cache["castle_leiden"], "Selected Leiden in physical space", False),
            (cache["castle_kmeans"], "K-means (K=7) in physical space", False),
            (cache["published_labels"], "Published labels in physical space", True),
        ],
        "Tuned CASTLE clusters and published labels across the tissue", spatial=True,
    )
    print("Re-rendered all cached figures with per-panel legends.")


if __name__ == "__main__":
    main()
