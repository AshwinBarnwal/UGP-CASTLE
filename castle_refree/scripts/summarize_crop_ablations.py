"""Collect the Xenium crop ablation metrics and make a compact comparison plot."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = REPO_ROOT / "results"
VARIANTS = {
    "expression_beta4": ROOT / "castle_refree_xenium_breast_15000",
    "geometry_only": ROOT / "castle_refree_xenium_breast_15000_geometry_only",
    "expression_beta2": ROOT / "castle_refree_xenium_breast_15000_beta2",
    "expression_beta4_eta05": ROOT / "castle_refree_xenium_breast_15000_eta_reg",
}


rows = []
for name, directory in VARIANTS.items():
    diagnostics = json.loads((directory / "crop_diagnostics.json").read_text())
    labels = json.loads((directory / "rctd_label_embedding_metrics.json").read_text())
    history = pd.read_csv(directory / "training_history.csv")
    final = history.iloc[-1]
    castle = labels["castle_intrinsic"]
    raw = labels["raw_log_normalized_pca"]
    rows.append(
        {
            "variant": name,
            "cells": diagnostics["cells"],
            "genes": diagnostics["genes"],
            "fraction_counts_removed": diagnostics["fraction_removed"],
            "eta_median": diagnostics["eta"]["median"],
            "eta_p75": diagnostics["eta"]["p75"],
            "eta_p99": diagnostics["eta"]["p99"],
            "confidence_median": diagnostics["segmentation_confidence"]["median"],
            "final_reconstruction": final["reconstruction"],
            "final_kl_intrinsic": final["kl_intrinsic"],
            "final_kl_niche": final["kl_niche"],
            "niche_dimension_sd_mean": float(np.mean(diagnostics["niche_dimension_sd"])),
            "unsupervised_kmeans_silhouette": diagnostics["intrinsic_kmeans_silhouette_5000"],
            "rctd_ari": castle["adjusted_rand_index"],
            "rctd_nmi": castle["normalized_mutual_information"],
            "rctd_knn15_purity": castle["knn15_label_purity"],
            "rctd_label_silhouette": castle["label_silhouette_5000"],
            "raw_pca_ari": raw["adjusted_rand_index"],
            "raw_pca_nmi": raw["normalized_mutual_information"],
            "raw_pca_knn15_purity": raw["knn15_label_purity"],
            "raw_pca_label_silhouette": raw["label_silhouette_5000"],
        }
    )

summary = pd.DataFrame(rows)
csv_path = ROOT / "castle_refree_crop_ablation_summary.csv"
summary.to_csv(csv_path, index=False)

labels = ["expr b4", "geometry", "expr b2", "expr b4 + eta"]
colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
panels = [
    ("fraction_counts_removed", "Fraction of counts removed", None),
    ("final_reconstruction", "Final reconstruction loss (lower is better)", None),
    ("niche_dimension_sd_mean", "Mean niche dimension SD", None),
    ("rctd_ari", "RCTD-label ARI", "raw_pca_ari"),
    ("rctd_nmi", "RCTD-label NMI", "raw_pca_nmi"),
    ("rctd_knn15_purity", "15-NN RCTD-label purity", "raw_pca_knn15_purity"),
]
for ax, (column, title, raw_column) in zip(axes.flat, panels):
    values = summary[column].to_numpy()
    ax.bar(labels, values, color=colors)
    if raw_column:
        ax.axhline(summary[raw_column].iloc[0], color="black", ls="--", lw=1.3, label="raw PCA")
        ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
fig.suptitle("CASTLE reference-free ablations — Xenium breast crop (14,996 cells)")
fig.tight_layout()
png_path = ROOT / "castle_refree_crop_ablation_summary.png"
fig.savefig(png_path, dpi=180, bbox_inches="tight")
print(summary.to_string(index=False))
print(f"\nSaved {csv_path}\nSaved {png_path}")
