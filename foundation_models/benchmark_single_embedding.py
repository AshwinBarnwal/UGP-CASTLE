"""Apply the complete v2 benchmark to one embedding and make truth-paired UMAPs."""

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
import seaborn as sns
import umap
from sklearn.cluster import MiniBatchKMeans

from benchmark_embeddings import (
    FIXED_NEIGHBORS, FIXED_RESOLUTION, NEIGHBORS, RESOLUTIONS,
    cluster_scores, graph_label_metrics, leiden_sweep, make_graph,
    published_label_silhouette, run_leiden, score_silhouette,
)


def read_ids(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep="\t", header=None, dtype=str)[0].to_numpy()


def load_embedding(args) -> tuple[np.ndarray, np.ndarray]:
    if args.embedding_h5ad:
        obj = ad.read_h5ad(args.embedding_h5ad, backed="r")
        return (obj.obs_names.astype(str).to_numpy(),
                np.asarray(obj.obsm[args.obsm_key], dtype=np.float32))
    if args.cells is None:
        raise ValueError("--cells is required with --embedding-npy or --embedding-npz")
    ids = read_ids(args.cells)
    if args.embedding_npy:
        values = np.load(args.embedding_npy, mmap_mode="r")
    else:
        with np.load(args.embedding_npz) as archive:
            values = archive[args.embedding_key]
    values = np.asarray(values, dtype=np.float32)
    if len(ids) != len(values):
        raise ValueError(f"Cell IDs ({len(ids)}) and embeddings ({len(values)}) differ")
    return ids, values


def draw(ax, xy: np.ndarray, labels: np.ndarray, title: str) -> None:
    labels = labels.astype(str)
    categories = sorted(pd.unique(labels))
    palette = dict(zip(categories, sns.color_palette("husl", len(categories))))
    if "Unknown" in palette:
        palette["Unknown"] = (0.7, 0.7, 0.7)
    for label in categories:
        keep = labels == label
        ax.scatter(xy[keep, 0], xy[keep, 1], s=1.1, linewidths=0,
                   alpha=.75, color=palette[label], label=label, rasterized=True)
    ax.set_title(title, weight="bold"); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=7, markerscale=5, loc="upper left",
              bbox_to_anchor=(1.01, 1))


def paired_plot(path: Path, xy: np.ndarray, predicted: np.ndarray, truth: np.ndarray,
                method: str, clustering_title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    draw(axes[0], xy, predicted, clustering_title)
    draw(axes[1], xy, truth, "Published supervised labels")
    fig.suptitle(f"{method}: one embedding, two colourings", fontsize=15, weight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=240, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--embedding-npy", type=Path)
    group.add_argument("--embedding-npz", type=Path)
    group.add_argument("--embedding-h5ad", type=Path)
    parser.add_argument("--cells", type=Path)
    parser.add_argument("--embedding-key", default="z_intrinsic")
    parser.add_argument("--obsm-key", default="X_uce")
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-cells", type=int, default=30000)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    source = ad.read_h5ad(args.source, backed="r")
    source_ids = source.obs_names.astype(str).to_numpy()
    source_truth = source.obs["cell_type"].astype(str).to_numpy()
    embedding_ids, values = load_embedding(args)
    common = pd.Index(source_ids).intersection(embedding_ids, sort=False)
    srow = pd.Series(np.arange(len(source_ids)), index=source_ids).loc[common].to_numpy()
    erow = pd.Series(np.arange(len(embedding_ids)), index=embedding_ids).loc[common].to_numpy()
    truth = source_truth[srow]; values = np.asarray(values[erow], dtype=np.float32)
    known_classes = int(pd.Series(truth[truth != "Unknown"]).nunique())

    optimized, best, sweep = leiden_sweep(values, truth, args.seed)
    fixed_holder = make_graph(values, FIXED_NEIGHBORS, args.seed)
    fixed = run_leiden(fixed_holder, FIXED_RESOLUTION, args.seed)
    kmeans = MiniBatchKMeans(n_clusters=known_classes, random_state=args.seed,
                             batch_size=4096, n_init=10).fit_predict(values).astype(str)

    rows = [
        {"method": args.method, "clustering": "Leiden optimized", **best,
         "silhouette": score_silhouette(values, optimized, args.seed),
         "selection": "supervised: maximum ARI against non-Unknown published labels"},
        {"method": args.method, "clustering": "Leiden fixed",
         "neighbors": FIXED_NEIGHBORS, "resolution": FIXED_RESOLUTION,
         **cluster_scores(truth, fixed),
         "silhouette": score_silhouette(values, fixed, args.seed),
         "selection": "common fixed readout; no per-method tuning"},
        {"method": args.method, "clustering": "K-means", "neighbors": np.nan,
         "resolution": np.nan, **cluster_scores(truth, kmeans),
         "silhouette": score_silhouette(values, kmeans, args.seed),
         "selection": "K fixed to number of non-Unknown published classes"},
    ]
    pd.DataFrame(rows).to_csv(args.output_dir / "ari_nmi_silhouette.csv", index=False)
    pd.DataFrame([{"method": args.method,
                   "published_label_silhouette": published_label_silhouette(values, truth, args.seed),
                   "neighbors_for_direct_metrics": FIXED_NEIGHBORS,
                   **graph_label_metrics(fixed_holder, truth)}]).to_csv(
                       args.output_dir / "embedding_direct_metrics.csv", index=False)
    pd.DataFrame([{"method": args.method, **row} for row in sweep]).to_csv(
        args.output_dir / "leiden_parameter_sweep.csv", index=False)

    rng = np.random.default_rng(args.seed)
    plot_rows = np.sort(rng.choice(len(common), min(args.plot_cells, len(common)), replace=False))
    xy = umap.UMAP(n_neighbors=30, min_dist=.3, random_state=args.seed,
                   low_memory=True).fit_transform(values[plot_rows])
    paired_plot(args.output_dir / "umap_leiden_optimized_vs_truth.png", xy,
                optimized[plot_rows], truth[plot_rows], args.method,
                f"Optimized Leiden (k={int(best['neighbors'])}, r={best['resolution']})")
    paired_plot(args.output_dir / "umap_leiden_fixed_vs_truth.png", xy,
                fixed[plot_rows], truth[plot_rows], args.method,
                f"Fixed Leiden (k={FIXED_NEIGHBORS}, r={FIXED_RESOLUTION})")
    paired_plot(args.output_dir / "umap_kmeans_vs_truth.png", xy,
                kmeans[plot_rows], truth[plot_rows], args.method,
                f"K-means (K={known_classes})")
    pd.DataFrame({"cell_id": common, "published_label": truth,
                  "leiden_optimized": optimized, "leiden_fixed": fixed,
                  "kmeans": kmeans}).to_csv(
                      args.output_dir / "cell_assignments.csv.gz", index=False, compression="gzip")
    (args.output_dir / "benchmark_manifest.json").write_text(json.dumps({
        "method": args.method, "common_cells": int(len(common)),
        "embedding_dimensions": int(values.shape[1]), "known_classes": known_classes,
        "plot_cells": int(len(plot_rows)), "seed": args.seed,
        "leiden_neighbors": list(NEIGHBORS), "leiden_resolutions": list(RESOLUTIONS),
        "fixed_leiden": {"neighbors": FIXED_NEIGHBORS, "resolution": FIXED_RESOLUTION},
        "optimized_leiden_selection_is_supervised": True,
        "unknown_excluded_from_label_metrics": True, "pca_applied": False,
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
