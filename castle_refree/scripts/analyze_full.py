"""Validate and visualize a full CASTLE run without embedding every cell in UMAP."""

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
import umap
from scipy import sparse
from scipy.io import mmread
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def quantiles(values: np.ndarray) -> dict[str, float]:
    levels = [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
    names = ["min", "p01", "p25", "median", "p75", "p99", "max"]
    return dict(zip(names, map(float, np.quantile(values, levels)), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--clusters", type=int, default=17)
    parser.add_argument("--plot-cells", type=int, default=30000)
    args = parser.parse_args()

    input_dir = Path(args.input)
    result_dir = Path(args.results)
    bundle = np.load(result_dir / "castle_embeddings_and_qc.npz")
    zc = bundle["z_intrinsic"]
    zn = bundle["z_niche"]
    eta = bundle["eta"]
    confidence = bundle["segmentation_confidence"]
    niche_l1 = bundle["niche_effect_l1"]
    cells = pd.read_csv(result_dir / "cells.tsv", header=None, dtype=str)[0].to_numpy()

    if len(cells) != len(zc):
        raise ValueError("Cell identifiers and embedding rows differ")
    if not all(np.isfinite(v).all() for v in (zc, zn, eta, confidence, niche_l1)):
        raise ValueError("Non-finite values found in exported arrays")

    spatial = pd.read_csv(input_dir / "spatial.csv", dtype={"cell_id": str})
    spatial = spatial.set_index("cell_id").reindex(cells)
    if spatial[["x", "y"]].isna().any().any():
        raise ValueError("Result cells could not be aligned to spatial coordinates")

    raw = mmread(input_dir / "counts.mtx").T.tocsr()
    raw = raw[np.asarray(raw.sum(axis=1)).ravel() >= 5]
    raw = raw[:, np.asarray((raw > 0).sum(axis=0)).ravel() >= 10]
    purified = sparse.load_npz(result_dir / "purified_counts_cells_by_genes.npz")
    if raw.shape != purified.shape:
        raise ValueError(f"Raw {raw.shape} and purified {purified.shape} shapes differ")
    retained = np.asarray(purified.sum(axis=1)).ravel() / np.maximum(
        np.asarray(raw.sum(axis=1)).ravel(), 1.0
    )

    rng = np.random.default_rng(7)
    plot_idx = np.sort(rng.choice(len(cells), min(args.plot_cells, len(cells)), replace=False))
    score_idx = rng.choice(len(cells), min(5000, len(cells)), replace=False)
    intrinsic_model = MiniBatchKMeans(
        n_clusters=args.clusters, batch_size=4096, n_init=10, random_state=7
    ).fit(zc)
    intrinsic_clusters = intrinsic_model.labels_
    niche_model = MiniBatchKMeans(
        n_clusters=12, batch_size=4096, n_init=10, random_state=7
    ).fit(zn)
    niche_clusters = niche_model.labels_

    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.25, metric="cosine", random_state=7, low_memory=True
    )
    umap_intrinsic = reducer.fit_transform(zc[plot_idx])
    umap_niche = reducer.fit_transform(zn[plot_idx])

    diagnostics = {
        "cells": int(len(cells)),
        "genes": int(raw.shape[1]),
        "raw_total": float(raw.sum()),
        "purified_total": float(purified.sum()),
        "fraction_removed": float(1.0 - purified.sum() / raw.sum()),
        "eta": quantiles(eta),
        "segmentation_confidence": quantiles(confidence),
        "cell_count_retention": quantiles(retained),
        "niche_effect_l1": quantiles(niche_l1),
        "intrinsic_dimension_sd": list(map(float, zc.std(axis=0))),
        "niche_dimension_sd": list(map(float, zn.std(axis=0))),
        "intrinsic_kmeans_silhouette_5000": float(
            silhouette_score(zc[score_idx], intrinsic_clusters[score_idx])
        ),
        "intrinsic_kmeans_davies_bouldin_5000": float(
            davies_bouldin_score(zc[score_idx], intrinsic_clusters[score_idx])
        ),
        "intrinsic_kmeans_calinski_harabasz_5000": float(
            calinski_harabasz_score(zc[score_idx], intrinsic_clusters[score_idx])
        ),
        "all_finite": True,
        "umap_plot_cells": int(len(plot_idx)),
    }
    (result_dir / "full_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    frame = pd.DataFrame(
        {
            "cell_id": cells[plot_idx],
            "intrinsic_umap_1": umap_intrinsic[:, 0],
            "intrinsic_umap_2": umap_intrinsic[:, 1],
            "niche_umap_1": umap_niche[:, 0],
            "niche_umap_2": umap_niche[:, 1],
            "intrinsic_cluster": intrinsic_clusters[plot_idx],
            "niche_cluster": niche_clusters[plot_idx],
            "eta": eta[plot_idx],
            "segmentation_confidence": confidence[plot_idx],
            "cell_count_retention": retained[plot_idx],
            "x": spatial["x"].to_numpy()[plot_idx],
            "y": spatial["y"].to_numpy()[plot_idx],
        }
    )
    frame.to_csv(result_dir / "embedding_plot_sample_coordinates.csv.gz", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    point = dict(s=2, linewidths=0, rasterized=True)
    axes[0, 0].scatter(*umap_intrinsic.T, c=intrinsic_clusters[plot_idx], cmap="tab20", **point)
    axes[0, 0].set_title("Intrinsic embedding: exploratory clusters")
    im = axes[0, 1].scatter(*umap_intrinsic.T, c=eta[plot_idx], cmap="magma", **point)
    axes[0, 1].set_title("Intrinsic embedding: estimated leakage")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    im = axes[0, 2].scatter(*umap_intrinsic.T, c=confidence[plot_idx], cmap="viridis", **point)
    axes[0, 2].set_title("Intrinsic embedding: segmentation confidence")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)
    axes[1, 0].scatter(*umap_niche.T, c=niche_clusters[plot_idx], cmap="tab20", **point)
    axes[1, 0].set_title("Niche embedding (weak in beta=2 run)")
    im = axes[1, 1].scatter(frame["x"], frame["y"], c=frame["eta"], cmap="magma", **point)
    axes[1, 1].set_title("Spatial estimated leakage")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_aspect("equal")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    im = axes[1, 2].scatter(frame["x"], frame["y"], c=frame["segmentation_confidence"], cmap="viridis", **point)
    axes[1, 2].set_title("Spatial segmentation confidence")
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_aspect("equal")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(result_dir / "castle_full_embedding_overview.png", dpi=180)
    plt.close(fig)

    history = pd.read_csv(result_dir / "training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(history["epoch"], history["reconstruction"], marker="o", ms=2)
    axes[0].set(xlabel="Epoch", ylabel="NB reconstruction loss", title="Training reconstruction")
    axes[1].plot(history["epoch"], history["mean_eta"], label="Mean eta")
    axes[1].plot(history["epoch"], history["mean_own_responsibility"], label="Mean own responsibility")
    axes[1].set(xlabel="Epoch", title="Purification diagnostics")
    axes[1].legend(frameon=False)
    fig.savefig(result_dir / "castle_full_training_diagnostics.png", dpi=180)
    plt.close(fig)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
