from __future__ import annotations

import argparse
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
from scipy.io import mmread
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def knn_label_purity(embedding: np.ndarray, labels: np.ndarray, k: int = 15) -> float:
    indices = NearestNeighbors(n_neighbors=k + 1).fit(embedding).kneighbors(return_distance=False)
    neighbors = indices[:, 1:]
    return float(np.mean(labels[neighbors] == labels[:, None]))


def embedding_metrics(
    embedding: np.ndarray, labels: np.ndarray, clusters: int, seed: int = 7
) -> dict[str, float]:
    predicted = KMeans(n_clusters=clusters, n_init=20, random_state=seed).fit_predict(embedding)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(labels), min(5000, len(labels)), replace=False)
    return {
        "adjusted_rand_index": float(adjusted_rand_score(labels, predicted)),
        "normalized_mutual_information": float(normalized_mutual_info_score(labels, predicted)),
        "knn15_label_purity": knn_label_purity(embedding, labels),
        "label_silhouette_5000": float(
            silhouette_score(embedding[sample], labels[sample])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input)
    result_dir = Path(args.results)

    coordinates = pd.read_csv(result_dir / "embedding_coordinates.csv.gz", dtype={"cell_id": str})
    labels = pd.read_csv(result_dir / "rctd_labels.csv", dtype={"cell_id": str})
    aligned = coordinates.merge(
        labels[["cell_id", "first_type", "spot_class"]], on="cell_id", how="inner",
        validate="one_to_one",
    )
    result_cells = pd.read_csv(result_dir / "cells.tsv", header=None, dtype=str)[0]
    row_lookup = pd.Series(np.arange(len(result_cells)), index=result_cells)
    rows = row_lookup.loc[aligned["cell_id"]].to_numpy()

    bundle = np.load(result_dir / "castle_embeddings_and_qc.npz")
    zc = bundle["z_intrinsic"][rows]
    raw = mmread(input_dir / "counts.mtx").T.tocsr()
    keep_cells = np.asarray(raw.sum(axis=1)).ravel() >= 5
    raw = raw[keep_cells]
    keep_genes = np.asarray((raw > 0).sum(axis=0)).ravel() >= 10
    raw = raw[:, keep_genes][rows].astype(np.float64)
    totals = np.asarray(raw.sum(axis=1)).ravel()
    raw = raw.multiply((1e4 / np.maximum(totals, 1.0))[:, None]).tocsr()
    raw.data = np.log1p(raw.data)
    raw_pca = TruncatedSVD(n_components=16, random_state=7).fit_transform(raw)

    type_labels = aligned["first_type"].to_numpy()
    n_types = int(pd.Series(type_labels).nunique())
    comparison = {
        "common_cells": int(len(aligned)),
        "reference_types": n_types,
        "castle_intrinsic": embedding_metrics(zc, type_labels, n_types),
        "raw_log_normalized_pca": embedding_metrics(raw_pca, type_labels, n_types),
        "note": "RCTD labels are evaluation-only and were not used for CASTLE training.",
    }
    (result_dir / "rctd_label_embedding_metrics.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    type_codes, type_names = pd.factorize(aligned["first_type"], sort=True)
    class_codes, class_names = pd.factorize(aligned["spot_class"], sort=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    point = dict(s=3, linewidths=0, rasterized=True)
    axes[0].scatter(
        aligned["intrinsic_umap_1"], aligned["intrinsic_umap_2"],
        c=type_codes, cmap="tab20", **point,
    )
    axes[0].set_title("CASTLE intrinsic UMAP: RCTD primary type (evaluation only)")
    axes[1].scatter(
        aligned["intrinsic_umap_1"], aligned["intrinsic_umap_2"],
        c=class_codes, cmap="Set2", **point,
    )
    axes[1].set_title("CASTLE intrinsic UMAP: RCTD confidence class")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=plt.cm.tab20(i / max(len(type_names)-1, 1)), label=name)
        for i, name in enumerate(type_names)
    ]
    axes[0].legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    class_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=plt.cm.Set2(i / max(len(class_names)-1, 1)), label=name)
        for i, name in enumerate(class_names)
    ]
    axes[1].legend(handles=class_handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.savefig(result_dir / "castle_crop_rctd_label_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

