"""Recreate saved Leiden solutions and add silhouettes and labelled PPT plots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scib_metrics.metrics._nmi_ari import _compute_clustering_leiden
from scib_metrics.nearest_neighbors import pynndescent
from sklearn.metrics import silhouette_score

METHODS = (
    "raw_pca50", "split_pca50", "castle_purified_pca50", "castle_intrinsic_z16"
)
DISPLAY = {
    "raw_pca50": "Raw\nPCA50",
    "split_pca50": "SPLIT\nPCA50",
    "castle_purified_pca50": "CASTLE purified\nPCA50",
    "castle_intrinsic_z16": "CASTLE intrinsic\nz16",
}


def label_bars(ax, bars) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                ha="center", va="bottom", fontsize=8.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()

    embeddings = np.load(result_dir / "benchmark_embeddings.npz")
    results = pd.read_csv(result_dir / "benchmark_results.csv", index_col="representation")
    sample = pd.read_csv(result_dir / "silhouette_sample_rows.csv")["evaluation_row"].to_numpy(int)

    leiden_silhouette = {}
    leiden_clusters = {}
    for method in METHODS:
        x = embeddings[method]
        resolution = float(results.loc[method, "leiden_best_resolution"])
        print(f"[{method}] building neighbour graph", flush=True)
        nn90 = pynndescent(x, n_neighbors=90, random_state=0, n_jobs=-1)
        graph15 = nn90.subset_neighbors(n=15).knn_graph_connectivities
        labels = np.asarray(_compute_clustering_leiden(graph15, resolution, seed=42))
        leiden_silhouette[method] = float(silhouette_score(x[sample], labels[sample]))
        leiden_clusters[method] = int(np.unique(labels).size)
        print(f"[{method}] silhouette={leiden_silhouette[method]:.6f}", flush=True)

    core = results.loc[list(METHODS), [
        "kmeans_ari", "kmeans_nmi", "leiden_ari", "leiden_nmi",
        "kmeans_silhouette_unscaled_sample",
    ]].copy()
    core["leiden_silhouette_unscaled_sample"] = pd.Series(leiden_silhouette)
    core["leiden_clusters"] = pd.Series(leiden_clusters)
    core.to_csv(result_dir / "professor_core_metrics.csv", index_label="representation")
    (result_dir / "leiden_silhouette.json").write_text(
        json.dumps({m: {"silhouette": leiden_silhouette[m],
                         "clusters": leiden_clusters[m],
                         "sample_cells": int(len(sample))} for m in METHODS}, indent=2),
        encoding="utf-8",
    )

    xpos = np.arange(len(METHODS))
    names = [DISPLAY[m] for m in METHODS]
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), constrained_layout=True)
    specs = [
        ("kmeans_ari", "kmeans_nmi", "K-means agreement with RCTD", "ARI", "NMI"),
        ("leiden_ari", "leiden_nmi", "Leiden agreement with RCTD", "ARI", "NMI"),
        ("kmeans_silhouette_unscaled_sample", "leiden_silhouette_unscaled_sample",
         "Cluster separation (silhouette)", "K-means", "Leiden"),
    ]
    for ax, (left, right, title, left_name, right_name) in zip(axes, specs):
        bars = ax.bar(xpos - width / 2, core[left], width, label=left_name, color="#457b9d")
        label_bars(ax, bars)
        bars = ax.bar(xpos + width / 2, core[right], width, label=right_name, color="#e9c46a")
        label_bars(ax, bars)
        ax.set_xticks(xpos, names, rotation=18, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Score (higher is better)")
    fig.suptitle("Raw, SPLIT and CASTLE clustering benchmark", fontsize=15)
    fig.savefig(result_dir / "professor_core_benchmark.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    bars = ax.bar(xpos, core["leiden_silhouette_unscaled_sample"],
                  color=["#8d99ae", "#e76f51", "#f4a261", "#2a9d8f"])
    label_bars(ax, bars)
    upper = max(0.3, float(core["leiden_silhouette_unscaled_sample"].max()) + 0.07)
    ax.set_xticks(xpos, names)
    ax.set_ylim(0, upper)
    ax.set_ylabel("Leiden silhouette (higher is better)")
    ax.set_title("Separation of Leiden clusters\nSame fixed 10,000-cell sample for every representation")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(result_dir / "leiden_silhouette.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
