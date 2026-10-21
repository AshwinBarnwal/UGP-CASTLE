"""Plot CosMx embedding comparisons without the deprecated NICHE embedding.

Outputs two honest cohorts instead of mixing feature-selection protocols:

* ``hvg2000``: CASTLE intrinsic and scVI trained on the fixed 2,000 HVGs.
  The tuned CASTLE experiment-i output is included automatically once z_c.npy
  exists.
* ``shared5906``: the earlier CASTLE intrinsic, scVI and scVIVA runs trained on
  the larger shared-gene input.

Leiden is the primary clustering view.  K-means is retained as a secondary
baseline.  ARI/NMI exclude the dataset's ``Unknown`` annotations, while
silhouette is an unsupervised cluster-separation score in each native latent
space.
"""

from __future__ import annotations

import argparse
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
SEED = 42
PLOT_CELLS = 30_000
SILHOUETTE_CELLS = 5_000


def read_ids(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep="\t", header=None, dtype=str)[0].to_numpy()


def load_castle(directory: Path, key: str = "z_intrinsic") -> tuple[np.ndarray, np.ndarray]:
    ids = read_ids(directory / "cells.tsv")
    if (directory / "z_c.npy").exists():
        values = np.load(directory / "z_c.npy", mmap_mode="r")
    else:
        values = np.load(directory / "castle_embeddings_and_qc.npz")[key]
    return ids, np.asarray(values, dtype=np.float32)


def load_anndata_embedding(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    obj = ad.read_h5ad(path, backed="r")
    return obj.obs_names.to_numpy().astype(str), np.asarray(obj.obsm[key], dtype=np.float32)


def align(source_ids: np.ndarray, methods: dict[str, tuple[np.ndarray, np.ndarray]]):
    common = set(source_ids)
    for ids, _ in methods.values():
        common.intersection_update(ids)
    keep = np.array([cell in common for cell in source_ids])
    common_ids = source_ids[keep]
    aligned = {}
    for name, (ids, values) in methods.items():
        row = pd.Series(np.arange(len(ids)), index=ids)
        aligned[name] = values[row.loc[common_ids].to_numpy()]
    return keep, common_ids, aligned


def leiden(values: np.ndarray) -> tuple[np.ndarray, int]:
    obj = ad.AnnData(X=np.asarray(values, dtype=np.float32))
    sc.pp.neighbors(obj, n_neighbors=30, metric="euclidean", random_state=SEED)
    sc.tl.leiden(obj, resolution=1.0, random_state=SEED, key_added="cluster")
    labels = obj.obs["cluster"].astype(str).to_numpy()
    return labels, len(np.unique(labels))


def umap2(values: np.ndarray) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=30, min_dist=0.3, n_components=2, metric="euclidean",
        random_state=SEED, low_memory=True,
    ).fit_transform(values)


def scatter(ax, xy, labels, title, spatial=False):
    labels = np.asarray(labels).astype(str)
    names = sorted(np.unique(labels), key=lambda x: int(x) if x.isdigit() else x)
    colors = dict(zip(names, sns.color_palette("husl", len(names))))
    for value in names:
        use = labels == value
        ax.scatter(xy[use, 0], xy[use, 1], s=0.75 if spatial else 1.2,
                   c=[colors[value]], linewidths=0, alpha=0.75, rasterized=True)
    ax.set_title(f"{title} ({len(names)} clusters)", weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if spatial:
        ax.set_aspect("equal")
        ax.invert_yaxis()
    else:
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    sns.despine(ax=ax, left=True, bottom=True)


def metric_row(method, algorithm, truth, predicted, embedding, sil_rows, **extra):
    known = truth != "Unknown"
    return {
        "method": method,
        "clustering": algorithm,
        "n_clusters": int(len(np.unique(predicted))),
        "ARI_vs_published_type_excluding_unknown": float(adjusted_rand_score(truth[known], predicted[known])),
        "NMI_vs_published_type_excluding_unknown": float(normalized_mutual_info_score(truth[known], predicted[known])),
        "silhouette_native_embedding_5000_cell_sample": float(
            silhouette_score(embedding[sil_rows], predicted[sil_rows])
        ),
        **extra,
    }


def plot_row(methods, coordinates, clusters, path, title, spatial=False):
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6), squeeze=False)
    for ax, name in zip(axes[0], methods):
        scatter(ax, coordinates[name] if isinstance(coordinates, dict) else coordinates,
                clusters[name], name, spatial=spatial)
    fig.suptitle(title, fontsize=16, weight="bold")
    fig.subplots_adjust(top=0.88, wspace=0.08)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def metrics_figure(frame: pd.DataFrame, path: Path):
    show = frame.copy()
    show["row"] = show["clustering"] + " — " + show["method"]
    show = show.set_index("row")[
        ["ARI_vs_published_type_excluding_unknown",
         "NMI_vs_published_type_excluding_unknown",
         "silhouette_native_embedding_5000_cell_sample"]
    ]
    show.columns = ["ARI", "NMI", "Silhouette"]
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.6 * len(show) + 1.6)))
    sns.heatmap(show, annot=True, fmt=".3f", cmap="YlGnBu", vmin=-0.05, vmax=1,
                linewidths=0.5, cbar_kws={"label": "Score"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Clustering scores (higher is better)", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run_cohort(name: str, source_path: Path, methods, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    source = ad.read_h5ad(source_path, backed="r")
    source_ids = source.obs_names.to_numpy().astype(str)
    source_labels = source.obs["cell_type"].astype(str).to_numpy()
    source_spatial = np.asarray(source.obsm["spatial"], dtype=np.float32)
    source_keep, common_ids, embeddings = align(source_ids, methods)
    truth = source_labels[source_keep]
    physical = source_spatial[source_keep]

    rng = np.random.default_rng(SEED)
    plot_rows = np.sort(rng.choice(len(common_ids), min(PLOT_CELLS, len(common_ids)), replace=False))
    sil_rows = np.sort(rng.choice(len(plot_rows), min(SILHOUETTE_CELLS, len(plot_rows)), replace=False))
    umaps = {method: umap2(values[plot_rows]) for method, values in embeddings.items()}

    # Leiden is intentionally first and is evaluated on the exact same 30k-cell
    # graph used for visualization; doing full-188k Leiden is a different graph.
    leiden_labels = {}
    rows = []
    for method, values in embeddings.items():
        labels, n_clusters = leiden(values[plot_rows])
        leiden_labels[method] = labels
        rows.append(metric_row(
            method, "Leiden", truth[plot_rows], labels, values[plot_rows], sil_rows,
            evaluation_scope=f"fixed_{len(plot_rows)}_cell_sample",
            leiden_neighbors=30, leiden_resolution=1.0,
        ))
    plot_row(embeddings, umaps, leiden_labels, output / "01_umap_leiden.png",
             f"{name}: intrinsic embeddings — Leiden (primary comparison)")
    plot_row(embeddings, physical[plot_rows], leiden_labels, output / "02_spatial_leiden.png",
             f"{name}: Leiden clusters in physical tissue", spatial=True)

    k = len(np.unique(truth[truth != "Unknown"]))
    kmeans_all = {}
    kmeans_plot = {}
    for method, values in embeddings.items():
        labels = MiniBatchKMeans(n_clusters=k, random_state=SEED, batch_size=4096, n_init=10).fit_predict(values)
        kmeans_all[method] = labels
        kmeans_plot[method] = labels[plot_rows]
        rows.append(metric_row(
            method, "K-means", truth, labels, values,
            np.sort(rng.choice(len(common_ids), min(SILHOUETTE_CELLS, len(common_ids)), replace=False)),
            evaluation_scope=f"all_{len(common_ids)}_common_cells",
            leiden_neighbors=np.nan, leiden_resolution=np.nan,
        ))
    plot_row(embeddings, umaps, kmeans_plot, output / "04_umap_kmeans.png",
             f"{name}: intrinsic embeddings — K-means (K={k})")
    plot_row(embeddings, physical[plot_rows], kmeans_plot, output / "05_spatial_kmeans.png",
             f"{name}: K-means clusters in physical tissue", spatial=True)

    frame = pd.DataFrame(rows)
    frame["order"] = frame["clustering"].map({"Leiden": 0, "K-means": 1})
    frame = frame.sort_values(["order", "method"]).drop(columns="order")
    frame.to_csv(output / "03_ari_nmi_silhouette.csv", index=False)
    metrics_figure(frame, output / "03_ari_nmi_silhouette.png")
    np.savez_compressed(output / "plot_data.npz", source_rows=np.flatnonzero(source_keep)[plot_rows], **{
        f"umap_{m.lower().replace(' ', '_').replace('-', '_')}": xy for m, xy in umaps.items()
    })
    (output / "method.json").write_text(json.dumps({
        "cohort": name,
        "common_cells": int(len(common_ids)),
        "plot_and_leiden_cells": int(len(plot_rows)),
        "silhouette_cells": int(len(sil_rows)),
        "published_truth": "CosMx cell_type; Unknown excluded from ARI/NMI",
        "leiden": {"neighbors": 30, "resolution": 1.0},
        "kmeans_k": int(k),
        "niche_embedding_included": False,
        "methods": list(embeddings),
    }, indent=2), encoding="utf-8")
    print(frame.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["hvg2000", "shared5906", "all"], default="all")
    args = parser.parse_args()
    source = ROOT / "data" / "cosmx_frontal_cortex.h5ad"

    if args.cohort in {"hvg2000", "all"}:
        methods = {
            "CASTLE intrinsic (40 epoch)": load_castle(ROOT / "results" / "castle_refree_cosmx_frontal_cortex_hvg2000_beta2"),
            "scVI": load_anndata_embedding(ROOT / "results" / "scvi_cosmx_frontal_cortex_hvg2000" / "adata_scvi.h5ad", "X_scVI"),
        }
        tuned = ROOT / "results" / "embedding_cosmx_hvg2000" / "castle_i"
        if (tuned / "z_c.npy").exists() and (tuned / "cells.tsv").exists():
            methods["CASTLE intrinsic tuned i"] = load_castle(tuned)
        run_cohort("CosMx frontal cortex — fixed 2,000 HVGs", source, methods,
                   ROOT / "results" / "cosmx_identity_comparison_hvg2000")

    if args.cohort in {"shared5906", "all"}:
        methods = {
            "CASTLE intrinsic": load_castle(ROOT / "results" / "castle_refree_cosmx_frontal_cortex_beta2"),
            "scVI": load_anndata_embedding(ROOT / "results" / "scvi_cosmx_frontal_cortex" / "adata_scvi.h5ad", "X_scVI"),
            "scVIVA": load_anndata_embedding(ROOT / "results" / "scviva_cosmx_frontal_cortex" / "adata_scviva.h5ad", "X_scVIVA"),
        }
        run_cohort("CosMx frontal cortex — larger shared-gene input", source, methods,
                   ROOT / "results" / "cosmx_identity_comparison_shared5906")


if __name__ == "__main__":
    main()
