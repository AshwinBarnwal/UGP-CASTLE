"""Generate the required no-NICHE figures for tuned CASTLE on CosMx HVG2000."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import umap
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "cosmx_frontal_cortex.h5ad"
CASTLE = ROOT / "results" / "embedding_cosmx_hvg2000" / "castle_i"
SCVI = ROOT / "results" / "scvi_cosmx_frontal_cortex_hvg2000" / "adata_scvi.h5ad"
OUT = ROOT / "results" / "cosmx_tuned_castle_hvg2000_figures"
SEED = 42
PLOT_CELLS = 30_000
SIL_CELLS = 5_000
NEIGHBORS = (10, 15, 30, 50)
RESOLUTIONS = tuple(np.round(np.arange(0.2, 2.01, 0.2), 1))


def read_ids(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep="\t", header=None, dtype=str)[0].to_numpy()


def umap2(values: np.ndarray) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=30, min_dist=0.3, n_components=2, metric="euclidean",
        random_state=SEED, low_memory=True,
    ).fit_transform(values)


def leiden_sweep(method: str, embedding: np.ndarray, truth: np.ndarray):
    known = truth != "Unknown"
    sweep, best = [], None
    for n_neighbors in NEIGHBORS:
        graph = ad.AnnData(X=np.asarray(embedding, dtype=np.float32))
        sc.pp.neighbors(graph, n_neighbors=n_neighbors, metric="euclidean", random_state=SEED)
        for resolution in RESOLUTIONS:
            sc.tl.leiden(
                graph, resolution=float(resolution), random_state=SEED,
                key_added="cluster", flavor="igraph", n_iterations=2, directed=False,
            )
            labels = graph.obs["cluster"].astype(str).to_numpy()
            ari = float(adjusted_rand_score(truth[known], labels[known]))
            nmi = float(normalized_mutual_info_score(truth[known], labels[known]))
            row = {
                "method": method,
                "neighbors": n_neighbors,
                "resolution": float(resolution),
                "n_clusters": int(len(np.unique(labels))),
                "ARI": ari,
                "NMI": nmi,
            }
            sweep.append(row)
            score = (ari, nmi, -abs(row["n_clusters"] - 7))
            if best is None or score > best[0]:
                best = (score, labels.copy(), row.copy())
    rng = np.random.default_rng(SEED + 17)
    sil = np.sort(rng.choice(len(embedding), min(SIL_CELLS, len(embedding)), replace=False))
    best[2]["silhouette"] = float(silhouette_score(embedding[sil], best[1][sil]))
    best[2]["selection"] = "maximum ARI against non-Unknown published labels"
    return best[1], best[2], sweep


def palette(values, truth=False):
    names = sorted(np.unique(np.asarray(values).astype(str)), key=lambda x: int(x) if x.isdigit() else x)
    colors = dict(zip(names, sns.color_palette("tab10" if truth else "husl", len(names))))
    if truth and "Unknown" in colors:
        colors["Unknown"] = (0.68, 0.68, 0.68)
    return names, colors


def scatter(ax, xy, values, title, colors=None, spatial=False, legend=False):
    values = np.asarray(values).astype(str)
    names, generated = palette(values, truth="Published" in title)
    colors = generated if colors is None else colors
    for value in names:
        keep = values == value
        ax.scatter(
            xy[keep, 0], xy[keep, 1], s=0.8 if spatial else 1.2,
            c=[colors[value]], label=value, linewidths=0, alpha=0.76, rasterized=True,
        )
    ax.set_title(title, fontsize=12.5, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if spatial:
        ax.set_aspect("equal")
        ax.invert_yaxis()
    else:
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False,
                  markerscale=5, fontsize=8, title="Colour label")
    sns.despine(ax=ax, left=True, bottom=True)


def save_three(path, coordinates, panels, title, spatial=False):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.4))
    for ax, (values, subtitle, colors) in zip(axes, panels):
        scatter(ax, coordinates, values, subtitle, colors=colors, spatial=spatial, legend=False)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=min(5, len(labels)), frameon=False, markerscale=5, fontsize=8)
    fig.suptitle(title, fontsize=16, weight="bold")
    fig.subplots_adjust(top=0.87, bottom=0.19, wspace=0.08)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ad.read_h5ad(SOURCE, backed="r")
    source_ids = source.obs_names.to_numpy().astype(str)
    source_labels = source.obs["cell_type"].astype(str).to_numpy()
    source_spatial = np.asarray(source.obsm["spatial"], dtype=np.float32)

    castle_ids = read_ids(CASTLE / "cells.tsv")
    castle_z = np.asarray(np.load(CASTLE / "z_c.npy", mmap_mode="r"), dtype=np.float32)
    scvi = ad.read_h5ad(SCVI, backed="r")
    scvi_ids = scvi.obs_names.to_numpy().astype(str)
    scvi_z = np.asarray(scvi.obsm["X_scVI"], dtype=np.float32)

    common = set(source_ids) & set(castle_ids) & set(scvi_ids)
    source_keep = np.array([cell in common for cell in source_ids])
    common_ids = source_ids[source_keep]
    truth = source_labels[source_keep]
    physical = source_spatial[source_keep]
    castle_rows = pd.Series(np.arange(len(castle_ids)), index=castle_ids).loc[common_ids].to_numpy()
    scvi_rows = pd.Series(np.arange(len(scvi_ids)), index=scvi_ids).loc[common_ids].to_numpy()
    embeddings = {"CASTLE tuned intrinsic": castle_z[castle_rows], "scVI": scvi_z[scvi_rows]}

    rng = np.random.default_rng(SEED)
    plot_rows = np.sort(rng.choice(len(common_ids), min(PLOT_CELLS, len(common_ids)), replace=False))
    truth_plot = truth[plot_rows]
    physical_plot = physical[plot_rows]
    umaps = {name: umap2(values[plot_rows]) for name, values in embeddings.items()}

    leiden_labels, leiden_best, sweep_rows = {}, {}, []
    kmeans_labels, metric_rows = {}, []
    k = len(np.unique(truth[truth != "Unknown"]))
    for name, values in embeddings.items():
        leiden_labels[name], leiden_best[name], sweep = leiden_sweep(name, values[plot_rows], truth_plot)
        sweep_rows.extend(sweep)
        metric_rows.append({"clustering": "Leiden", **leiden_best[name]})

        km = MiniBatchKMeans(n_clusters=k, random_state=SEED, batch_size=4096, n_init=10).fit_predict(values)
        kmeans_labels[name] = km
        known = truth != "Unknown"
        sil = np.sort(rng.choice(len(common_ids), min(SIL_CELLS, len(common_ids)), replace=False))
        metric_rows.append({
            "method": name, "clustering": "K-means", "n_clusters": k,
            "ARI": float(adjusted_rand_score(truth[known], km[known])),
            "NMI": float(normalized_mutual_info_score(truth[known], km[known])),
            "silhouette": float(silhouette_score(values[sil], km[sil])),
            "neighbors": np.nan, "resolution": np.nan,
            "selection": "K fixed to number of non-Unknown published classes",
        })

    pd.DataFrame(sweep_rows).to_csv(OUT / "leiden_parameter_sweep.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics["order"] = metrics["clustering"].map({"Leiden": 0, "K-means": 1})
    metrics.sort_values(["order", "method"]).drop(columns="order").to_csv(
        OUT / "ari_nmi_silhouette.csv", index=False
    )

    castle_name = "CASTLE tuned intrinsic"
    castle_umap = umaps[castle_name]
    castle_leiden = leiden_labels[castle_name]
    castle_kmeans = kmeans_labels[castle_name][plot_rows]
    type_names, type_colors = palette(truth_plot, truth=True)
    leiden_names, leiden_colors = palette(castle_leiden)
    kmeans_names, kmeans_colors = palette(castle_kmeans)
    save_three(
        OUT / "01_castle_umap_leiden_kmeans_published.png", castle_umap,
        [
            (castle_leiden, f"Selected Leiden\nkNN={leiden_best[castle_name]['neighbors']}, "
                             f"resolution={leiden_best[castle_name]['resolution']:.1f}", leiden_colors),
            (castle_kmeans, f"K-means (K={k})", kmeans_colors),
            (truth_plot, "Published cell-type labels", type_colors),
        ],
        "Tuned CASTLE intrinsic embedding — three views of the same UMAP",
    )

    for algorithm, labels_by_method, filename in (
        ("selected Leiden", leiden_labels, "02_castle_vs_scvi_umap_leiden.png"),
        ("K-means", {n: v[plot_rows] for n, v in kmeans_labels.items()}, "03_castle_vs_scvi_umap_kmeans.png"),
    ):
        fig, axes = plt.subplots(1, len(embeddings), figsize=(7 * len(embeddings), 6.2))
        for ax, name in zip(np.atleast_1d(axes), embeddings):
            values = labels_by_method[name]
            _, colors = palette(values)
            detail = ""
            if algorithm == "selected Leiden":
                detail = f"\nkNN={leiden_best[name]['neighbors']}, res={leiden_best[name]['resolution']:.1f}"
            scatter(ax, umaps[name], values, f"{name}{detail}", colors=colors)
        fig.suptitle(f"CosMx 2,000-HVG embeddings — {algorithm}", fontsize=16, weight="bold")
        fig.subplots_adjust(top=0.86, wspace=0.10)
        fig.savefig(OUT / filename, dpi=260, bbox_inches="tight")
        plt.close(fig)

    save_three(
        OUT / "04_castle_spatial_leiden_kmeans_published.png", physical_plot,
        [
            (castle_leiden, "Selected Leiden in physical space", leiden_colors),
            (castle_kmeans, f"K-means (K={k}) in physical space", kmeans_colors),
            (truth_plot, "Published labels in physical space", type_colors),
        ],
        "Tuned CASTLE clusters and published labels across the tissue",
        spatial=True,
    )

    score = metrics.copy()
    score["row"] = score["clustering"] + " — " + score["method"]
    score = score.set_index("row")[["ARI", "NMI", "silhouette"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.heatmap(score, annot=True, fmt=".3f", cmap="YlGnBu", vmin=-0.05, vmax=1,
                linewidths=0.5, cbar_kws={"label": "Score"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("CosMx 2,000-HVG clustering scores", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "05_ari_nmi_silhouette.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        OUT / "plot_data.npz", source_rows=np.flatnonzero(source_keep)[plot_rows],
        castle_umap=castle_umap, scvi_umap=umaps["scVI"],
        castle_leiden=castle_leiden, scvi_leiden=leiden_labels["scVI"],
        castle_kmeans=castle_kmeans, scvi_kmeans=kmeans_labels["scVI"][plot_rows],
        published_labels=truth_plot,
    )
    (OUT / "figure_method.json").write_text(json.dumps({
        "cells_common": int(len(common_ids)), "cells_plotted": int(len(plot_rows)),
        "niche_included": False, "leiden_neighbors": list(NEIGHBORS),
        "leiden_resolutions": list(RESOLUTIONS),
        "leiden_selection": "maximum ARI against non-Unknown published labels, independently per method",
        "missing_compatible_methods": ["scVIVA HVG2000 incomplete", "SPLIT+PCA50 incomplete"],
    }, indent=2), encoding="utf-8")
    print(metrics.sort_values(["order", "method"]).drop(columns="order").to_string(index=False), flush=True)
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
