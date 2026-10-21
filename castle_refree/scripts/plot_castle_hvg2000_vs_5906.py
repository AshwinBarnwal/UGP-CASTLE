"""Compare CASTLE intrinsic embeddings for 2,000 HVGs versus 5,906 genes."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "cosmx_frontal_cortex.h5ad"
OUT = ROOT / "results" / "cosmx_castle_hvg2000_vs_5906"
SEED = 42


NEIGHBORS = (10, 15, 30, 50)
RESOLUTIONS = tuple(np.round(np.arange(0.2, 2.01, 0.2), 1))


def load_row(result_dir: Path, cache_dir: Path, cache_key: str, method_name: str):
    source = ad.read_h5ad(SOURCE, backed="r")
    source_ids = source.obs_names.to_numpy().astype(str)
    labels = source.obs["cell_type"].astype(str).to_numpy()
    cache = np.load(cache_dir / "plot_data.npz")
    source_rows = cache["source_rows"]
    xy = cache[cache_key]
    selected_ids = source_ids[source_rows]

    castle_ids = pd.read_csv(result_dir / "cells.tsv", sep="\t", header=None, dtype=str)[0].to_numpy()
    castle = np.load(result_dir / "castle_embeddings_and_qc.npz")["z_intrinsic"]
    lookup = pd.Series(np.arange(len(castle_ids)), index=castle_ids)
    embedding = np.asarray(castle[lookup.loc[selected_ids].to_numpy()], dtype=np.float32)

    truth = labels[source_rows]
    known = truth != "Unknown"
    sweep = []
    best = None
    for n_neighbors in NEIGHBORS:
        graph = ad.AnnData(X=embedding)
        sc.pp.neighbors(graph, n_neighbors=n_neighbors, metric="euclidean", random_state=SEED)
        for resolution in RESOLUTIONS:
            sc.tl.leiden(
                graph, resolution=float(resolution), random_state=SEED,
                key_added="cluster", flavor="igraph", n_iterations=2, directed=False,
            )
            clusters = graph.obs["cluster"].astype(str).to_numpy()
            ari = adjusted_rand_score(truth[known], clusters[known])
            nmi = normalized_mutual_info_score(truth[known], clusters[known])
            row = {
                "method": method_name,
                "neighbors": n_neighbors,
                "resolution": float(resolution),
                "n_clusters": len(np.unique(clusters)),
                "ARI": float(ari),
                "NMI": float(nmi),
            }
            sweep.append(row)
            candidate = (ari, nmi, -abs(len(np.unique(clusters)) - 7), clusters.copy(), row)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    clusters = best[3]
    best_row = dict(best[4])
    rng = np.random.default_rng(SEED + 17)
    sil = np.sort(rng.choice(len(embedding), min(5000, len(embedding)), replace=False))
    best_row["silhouette"] = float(silhouette_score(embedding[sil], clusters[sil]))
    return xy, clusters, truth, sweep, best_row


def draw(ax, xy, values, colors, title, label_order):
    values = np.asarray(values).astype(str)
    for value in label_order:
        keep = values == value
        if keep.any():
            ax.scatter(xy[keep, 0], xy[keep, 1], s=1.15, c=[colors[value]],
                       linewidths=0, alpha=0.76, rasterized=True, label=value)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        (
            "CASTLE — 2,000 HVGs",
            ROOT / "results" / "castle_refree_cosmx_frontal_cortex_hvg2000_beta2",
            ROOT / "results" / "cosmx_identity_comparison_hvg2000",
            "umap_castle_intrinsic_(40_epoch)",
        ),
        (
            "CASTLE — 5,906 genes",
            ROOT / "results" / "castle_refree_cosmx_frontal_cortex_beta2",
            ROOT / "results" / "cosmx_identity_comparison_shared5906",
            "umap_castle_intrinsic",
        ),
    ]
    loaded = [(name, *load_row(result, cache, key, name)) for name, result, cache, key in rows]

    all_types = sorted(set(np.concatenate([x[3] for x in loaded])))
    type_palette = dict(zip(all_types, sns.color_palette("tab10", len(all_types))))
    if "Unknown" in type_palette:
        type_palette["Unknown"] = (0.68, 0.68, 0.68)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for row, (name, xy, clusters, truth, _sweep, best) in enumerate(loaded):
        cluster_names = sorted(np.unique(clusters), key=int)
        cluster_palette = dict(zip(cluster_names, sns.color_palette("husl", len(cluster_names))))
        draw(axes[row, 0], xy, clusters, cluster_palette,
             f"{name}\nBest-ARI Leiden: kNN={best['neighbors']}, res={best['resolution']:.1f} "
             f"({len(cluster_names)} clusters)", cluster_names)
        draw(axes[row, 1], xy, truth, type_palette,
             f"{name}\nPublished cell-type labels", all_types)

    handles, legend_labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=4, frameon=False, markerscale=5, fontsize=9)
    fig.suptitle(
        "CASTLE intrinsic embedding: 2,000 HVGs versus 5,906 genes\n"
        "Leiden structure compared with published CosMx annotations",
        fontsize=16, weight="bold",
    )
    fig.subplots_adjust(top=0.90, bottom=0.12, hspace=0.22, wspace=0.10)
    target = OUT / "castle_hvg2000_vs_5906_leiden_and_true_labels.png"
    fig.savefig(target, dpi=260, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame([row for item in loaded for row in item[4]]).to_csv(
        OUT / "leiden_parameter_sweep.csv", index=False
    )
    pd.DataFrame([item[5] for item in loaded]).to_csv(
        OUT / "selected_leiden_metrics.csv", index=False
    )
    print(target)
    print(pd.DataFrame([item[5] for item in loaded]).to_string(index=False))


if __name__ == "__main__":
    main()
