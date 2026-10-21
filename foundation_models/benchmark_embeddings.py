"""Canonical benchmark for completed foundation-model embeddings.

Reports direct embedding metrics, one common fixed Leiden readout, and an
independently optimized Leiden readout (explicitly supervised). Complete
released embeddings are used directly; Scanpy is explicitly given use_rep=X
so it cannot silently insert PCA for wide representations.
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
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             balanced_accuracy_score,
                             normalized_mutual_info_score, silhouette_score)

# Expanded but still compact grid: 72 combinations per representation.
NEIGHBORS = (10, 15, 30, 50, 75, 100)
RESOLUTIONS = (0.1,) + tuple(np.round(np.arange(0.2, 2.01, 0.2), 1)) + (2.2,)
FIXED_NEIGHBORS = 30
FIXED_RESOLUTION = 1.0


def read_ids(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep="\t", header=None, dtype=str)[0].to_numpy()


def load_inputs(source_path: Path, niche_path: Path, niche_cells_path: Path,
                uce_path: Path):
    source = ad.read_h5ad(source_path, backed="r")
    source_ids = source.obs_names.astype(str).to_numpy()
    source_labels = source.obs["cell_type"].astype(str).to_numpy()
    niche_ids = read_ids(niche_cells_path)
    niche = np.load(niche_path, mmap_mode="r")
    uce_data = ad.read_h5ad(uce_path, backed="r")
    uce_ids = uce_data.obs_names.astype(str).to_numpy()
    common = pd.Index(source_ids).intersection(niche_ids, sort=False).intersection(uce_ids, sort=False)
    if len(common) == 0:
        raise ValueError("No common cell IDs across source, Nicheformer, and UCE")
    srow = pd.Series(np.arange(len(source_ids)), index=source_ids).loc[common].to_numpy()
    nrow = pd.Series(np.arange(len(niche_ids)), index=niche_ids).loc[common].to_numpy()
    urow = pd.Series(np.arange(len(uce_ids)), index=uce_ids).loc[common].to_numpy()
    truth = source_labels[srow]
    embeddings = {
        "Nicheformer": np.asarray(niche[nrow], dtype=np.float32),
        "UCE-33": np.asarray(uce_data.obsm["X_uce"][urow], dtype=np.float32),
    }
    return common.to_numpy(), truth, embeddings


def make_graph(values: np.ndarray, n_neighbors: int, seed: int) -> ad.AnnData:
    holder = ad.AnnData(X=np.asarray(values, dtype=np.float32))
    sc.pp.neighbors(holder, n_neighbors=n_neighbors, metric="euclidean",
                    use_rep="X", random_state=seed)
    return holder


def run_leiden(holder: ad.AnnData, resolution: float, seed: int) -> np.ndarray:
    sc.tl.leiden(holder, resolution=float(resolution), random_state=seed,
                 key_added="cluster", flavor="igraph", n_iterations=2, directed=False)
    return holder.obs["cluster"].astype(str).to_numpy()


def cluster_scores(truth: np.ndarray, labels: np.ndarray) -> dict:
    known = truth != "Unknown"
    return {
        "n_clusters": int(pd.Series(labels).nunique()),
        "ARI": float(adjusted_rand_score(truth[known], labels[known])),
        "NMI": float(normalized_mutual_info_score(truth[known], labels[known])),
    }


def leiden_sweep(values: np.ndarray, truth: np.ndarray, seed: int,
                 neighbors=NEIGHBORS, resolutions=RESOLUTIONS):
    sweep, best = [], None
    n_truth = pd.Series(truth[truth != "Unknown"]).nunique()
    for n_neighbors in neighbors:
        holder = make_graph(values, int(n_neighbors), seed)
        for resolution in resolutions:
            labels = run_leiden(holder, float(resolution), seed)
            row = {"neighbors": int(n_neighbors), "resolution": float(resolution),
                   **cluster_scores(truth, labels)}
            sweep.append(row)
            rank = (row["ARI"], row["NMI"], -abs(row["n_clusters"] - n_truth))
            if best is None or rank > best[0]:
                best = (rank, labels.copy(), row.copy())
    return best[1], best[2], sweep


def score_silhouette(values: np.ndarray, labels: np.ndarray, seed: int,
                     n: int = 5000) -> float:
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(len(values), min(n, len(values)), replace=False))
    if pd.Series(labels[take]).nunique() < 2:
        return float("nan")
    return float(silhouette_score(values[take], labels[take]))


def published_label_silhouette(values: np.ndarray, truth: np.ndarray, seed: int,
                               n: int = 5000) -> float:
    known_rows = np.flatnonzero(truth != "Unknown")
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(known_rows, min(n, len(known_rows)), replace=False))
    return float(silhouette_score(values[take], truth[take]))


def graph_label_metrics(holder: ad.AnnData, truth: np.ndarray) -> dict:
    """Leave-one-out label vote and same-label fraction; Unknown is omitted."""
    distances = sparse.csr_matrix(holder.obsp["distances"])
    known = truth != "Unknown"
    categories = pd.Index(sorted(pd.unique(truth[known])))
    codes = categories.get_indexer(truth)
    predictions, targets, purities = [], [], []
    for i in np.flatnonzero(known):
        js = distances.indices[distances.indptr[i]:distances.indptr[i + 1]]
        js = js[known[js]]
        if len(js) == 0:
            continue
        neighbor_codes = codes[js]
        predictions.append(int(np.bincount(neighbor_codes, minlength=len(categories)).argmax()))
        targets.append(int(codes[i]))
        purities.append(float(np.mean(neighbor_codes == codes[i])))
    return {
        "knn_label_accuracy": float(accuracy_score(targets, predictions)),
        "knn_label_balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "mean_neighborhood_purity": float(np.mean(purities)),
    }


def plot_panels(path: Path, coordinates: dict, colors: dict, title: str):
    methods = list(coordinates)
    fig, axes = plt.subplots(1, len(methods), figsize=(8 * len(methods), 7), squeeze=False)
    for ax, method in zip(axes[0], methods):
        labels = colors[method].astype(str)
        palette = dict(zip(sorted(np.unique(labels)), sns.color_palette("husl", len(np.unique(labels)))))
        for label in sorted(np.unique(labels)):
            keep = labels == label
            ax.scatter(coordinates[method][keep, 0], coordinates[method][keep, 1], s=1.1,
                       linewidths=0, alpha=.75, color=palette[label], label=label, rasterized=True)
        ax.set_title(method, weight="bold"); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(frameon=False, fontsize=7, markerscale=5, loc="upper left",
                  bbox_to_anchor=(1.01, 1))
    fig.suptitle(title, fontsize=15, weight="bold"); fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight"); plt.close(fig)


def write_plots(output_dir: Path, embeddings: dict, truth: np.ndarray,
                optimized: dict, fixed: dict, kmeans: dict,
                seed: int, plot_cells: int):
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(truth), min(plot_cells, len(truth)), replace=False))
    coordinates = {name: umap.UMAP(n_neighbors=30, min_dist=.3, random_state=seed,
                                    low_memory=True).fit_transform(values[rows])
                   for name, values in embeddings.items()}
    plot_panels(output_dir / "foundation_umap_leiden_optimized.png", coordinates,
                {n: x[rows] for n, x in optimized.items()},
                "Foundation models: independently optimized Leiden")
    plot_panels(output_dir / "foundation_umap_leiden_fixed.png", coordinates,
                {n: x[rows] for n, x in fixed.items()},
                f"Foundation models: fixed Leiden (k={FIXED_NEIGHBORS}, r={FIXED_RESOLUTION})")
    plot_panels(output_dir / "foundation_umap_kmeans.png", coordinates,
                {n: x[rows] for n, x in kmeans.items()}, "Foundation models: K-means")
    plot_panels(output_dir / "foundation_umap_published_labels.png", coordinates,
                {n: truth[rows] for n in embeddings}, "Foundation models: published labels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--nicheformer", type=Path, required=True)
    parser.add_argument("--nicheformer-cells", type=Path, required=True)
    parser.add_argument("--uce", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-cells", type=int, default=30000)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    common, truth, embeddings = load_inputs(args.source, args.nicheformer,
                                             args.nicheformer_cells, args.uce)
    known_classes = pd.Series(truth[truth != "Unknown"]).nunique()
    all_metrics, all_sweeps, direct = [], [], []
    optimized, fixed, kmeans = {}, {}, {}
    for method, values in embeddings.items():
        labels, best, sweep = leiden_sweep(values, truth, args.seed)
        optimized[method] = labels
        best.update({"method": method, "clustering": "Leiden optimized",
                     "silhouette": score_silhouette(values, labels, args.seed),
                     "selection": "supervised: maximum ARI against non-Unknown published labels"})
        all_metrics.append(best); all_sweeps.extend({"method": method, **r} for r in sweep)

        fixed_holder = make_graph(values, FIXED_NEIGHBORS, args.seed)
        fixed_labels = run_leiden(fixed_holder, FIXED_RESOLUTION, args.seed)
        fixed[method] = fixed_labels
        all_metrics.append({"method": method, "clustering": "Leiden fixed",
                            "neighbors": FIXED_NEIGHBORS, "resolution": FIXED_RESOLUTION,
                            **cluster_scores(truth, fixed_labels),
                            "silhouette": score_silhouette(values, fixed_labels, args.seed),
                            "selection": "common fixed readout; no per-method tuning"})

        km = MiniBatchKMeans(n_clusters=known_classes, random_state=args.seed,
                             batch_size=4096, n_init=10).fit_predict(values).astype(str)
        kmeans[method] = km; km_scores = cluster_scores(truth, km)
        all_metrics.append({"method": method, "clustering": "K-means", "neighbors": np.nan,
                            "resolution": np.nan, **km_scores,
                            "silhouette": score_silhouette(values, km, args.seed),
                            "selection": "K fixed to number of non-Unknown published classes"})
        direct.append({"method": method,
                       "published_label_silhouette": published_label_silhouette(values, truth, args.seed),
                       "neighbors_for_direct_metrics": FIXED_NEIGHBORS,
                       **graph_label_metrics(fixed_holder, truth)})

    pd.DataFrame(all_sweeps).to_csv(args.output_dir / "leiden_parameter_sweep.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(args.output_dir / "ari_nmi_silhouette.csv", index=False)
    pd.DataFrame(direct).to_csv(args.output_dir / "embedding_direct_metrics.csv", index=False)
    write_plots(args.output_dir, embeddings, truth, optimized, fixed, kmeans,
                args.seed, args.plot_cells)
    (args.output_dir / "benchmark_manifest.json").write_text(json.dumps({
        "protocol": "foundation benchmark v2", "common_cells": int(len(common)),
        "plot_cells": int(min(args.plot_cells, len(common))), "seed": args.seed,
        "leiden_neighbors": list(NEIGHBORS), "leiden_resolutions": list(RESOLUTIONS),
        "fixed_leiden": {"neighbors": FIXED_NEIGHBORS, "resolution": FIXED_RESOLUTION},
        "unknown_excluded_from_label_based_metrics": True,
        "optimized_leiden_selection_is_supervised": True,
        "pca_applied_to_foundation_embeddings": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
