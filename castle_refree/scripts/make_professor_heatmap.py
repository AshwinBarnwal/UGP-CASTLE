"""Create a precise-value heatmap including both cluster silhouettes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["raw_pca50", "split_pca50", "castle_purified_pca50", "castle_intrinsic_z16"]
DISPLAY = ["Raw\nPCA50", "SPLIT\nPCA50", "CASTLE purified\nPCA50", "CASTLE intrinsic\nz16"]
SPECS = [
    ("kmeans_ari", "K-means\nARI", True),
    ("kmeans_nmi", "K-means\nNMI", True),
    ("leiden_ari", "Leiden\nARI", True),
    ("leiden_nmi", "Leiden\nNMI", True),
    ("kmeans_silhouette_unscaled_sample", "K-means\nsilhouette", True),
    ("leiden_silhouette_unscaled_sample", "Leiden\nsilhouette", True),
    ("label_silhouette_rescaled", "RCTD-label\nsilhouette", True),
    ("clisi", "cLISI", True),
    ("isolated_labels", "Isolated\nlabels", True),
    ("label_calinski_harabasz", "Calinski-\nHarabasz", True),
    ("label_davies_bouldin", "Davies-\nBouldin", False),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    complete = pd.read_csv(result_dir / "benchmark_results.csv", index_col="representation")
    core = pd.read_csv(result_dir / "professor_core_metrics.csv", index_col="representation")
    complete["leiden_silhouette_unscaled_sample"] = core["leiden_silhouette_unscaled_sample"]
    selected = complete.loc[METHODS, [key for key, _, _ in SPECS]].copy()
    selected.to_csv(result_dir / "professor_all_metrics.csv", index_label="representation")

    raw = selected.to_numpy(float)
    goodness = np.zeros_like(raw)
    for column, (_, _, higher) in enumerate(SPECS):
        values = raw[:, column]
        low, high = values.min(), values.max()
        scaled = np.ones_like(values) if high == low else (values - low) / (high - low)
        goodness[:, column] = scaled if higher else 1 - scaled

    fig, ax = plt.subplots(figsize=(17, 5.8), constrained_layout=True)
    image = ax.imshow(goodness, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(SPECS)), [label for _, label, _ in SPECS])
    ax.set_yticks(np.arange(len(METHODS)), DISPLAY)
    for row in range(raw.shape[0]):
        for column in range(raw.shape[1]):
            value = raw[row, column]
            label = f"{value:.3f}" if abs(value) < 100 else f"{value:,.0f}"
            ax.text(column, row, label, ha="center", va="center", fontsize=8.5,
                    color="white" if goodness[row, column] > 0.62 else "black")
    ax.set_title("SPLIT–CASTLE benchmark: exact scores\nColour shows relative rank within each column")
    fig.colorbar(image, ax=ax, label="Relative performance within metric")
    fig.savefig(result_dir / "professor_full_metrics_heatmap.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
