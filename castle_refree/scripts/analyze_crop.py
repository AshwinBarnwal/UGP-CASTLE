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
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def quantiles(values: np.ndarray) -> dict[str, float]:
    levels = [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
    names = ["min", "p01", "p25", "median", "p75", "p99", "max"]
    return dict(zip(names, map(float, np.quantile(values, levels)), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CASTLE input bundle")
    parser.add_argument("--results", required=True, help="CASTLE result directory")
    parser.add_argument("--clusters", type=int, default=17)
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
    spatial = pd.read_csv(input_dir / "spatial.csv", dtype={"cell_id": str})
    spatial = spatial.set_index("cell_id").reindex(cells)
    if spatial[["x", "y"]].isna().any().any():
        raise ValueError("Result cells could not be aligned to spatial coordinates")

    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.25, metric="cosine", random_state=7,
        low_memory=True,
    )
    umap_intrinsic = reducer.fit_transform(zc)
    umap_niche = reducer.fit_transform(zn)
    intrinsic_clusters = KMeans(
        n_clusters=args.clusters, n_init=20, random_state=7
    ).fit_predict(zc)
    niche_clusters = KMeans(
        n_clusters=12, n_init=20, random_state=7
    ).fit_predict(zn)

    raw = mmread(input_dir / "counts.mtx").T.tocsr()
    keep_cells = np.asarray(raw.sum(axis=1)).ravel() >= 5
    raw = raw[keep_cells]
    keep_genes = np.asarray((raw > 0).sum(axis=0)).ravel() >= 10
    raw = raw[:, keep_genes]
    purified = sparse.load_npz(result_dir / "purified_counts_cells_by_genes.npz")
    raw_cell_total = np.asarray(raw.sum(axis=1)).ravel()
    purified_cell_total = np.asarray(purified.sum(axis=1)).ravel()
    retained = purified_cell_total / np.maximum(raw_cell_total, 1.0)

    rng = np.random.default_rng(7)
    sample = rng.choice(len(cells), min(5000, len(cells)), replace=False)
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
            silhouette_score(zc[sample], intrinsic_clusters[sample], metric="euclidean")
        ),
        "intrinsic_kmeans_davies_bouldin": float(
            davies_bouldin_score(zc, intrinsic_clusters)
        ),
        "intrinsic_kmeans_calinski_harabasz": float(
            calinski_harabasz_score(zc, intrinsic_clusters)
        ),
        "all_finite": bool(
            all(np.isfinite(v).all() for v in (zc, zn, eta, confidence, niche_l1))
        ),
    }
    (result_dir / "crop_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    frame = pd.DataFrame(
        {
            "cell_id": cells,
            "intrinsic_umap_1": umap_intrinsic[:, 0],
            "intrinsic_umap_2": umap_intrinsic[:, 1],
            "niche_umap_1": umap_niche[:, 0],
            "niche_umap_2": umap_niche[:, 1],
            "intrinsic_cluster": intrinsic_clusters,
            "niche_cluster": niche_clusters,
            "eta": eta,
            "segmentation_confidence": confidence,
            "cell_count_retention": retained,
            "x": spatial["x"].to_numpy(),
            "y": spatial["y"].to_numpy(),
        }
    )
    frame.to_csv(result_dir / "embedding_coordinates.csv.gz", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    point = dict(s=2, linewidths=0, rasterized=True)
    axes[0, 0].scatter(*umap_intrinsic.T, c=intrinsic_clusters, cmap="tab20", **point)
    axes[0, 0].set_title("Intrinsic embedding: exploratory k-means")
    im = axes[0, 1].scatter(*umap_intrinsic.T, c=eta, cmap="magma", **point)
    axes[0, 1].set_title("Intrinsic embedding: estimated leakage")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    im = axes[0, 2].scatter(*umap_intrinsic.T, c=confidence, cmap="viridis", **point)
    axes[0, 2].set_title("Intrinsic embedding: segmentation confidence")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)
    axes[1, 0].scatter(*umap_niche.T, c=niche_clusters, cmap="tab20", **point)
    axes[1, 0].set_title("Niche embedding: exploratory k-means")
    im = axes[1, 1].scatter(spatial["x"], spatial["y"], c=eta, cmap="magma", **point)
    axes[1, 1].set_title("Spatial estimated leakage")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_aspect("equal")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    im = axes[1, 2].scatter(
        spatial["x"], spatial["y"], c=confidence, cmap="viridis", **point
    )
    axes[1, 2].set_title("Spatial segmentation confidence")
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_aspect("equal")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(result_dir / "castle_crop_embedding_overview.png", dpi=180)
    plt.close(fig)

    history = pd.read_csv(result_dir / "training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(history["epoch"], history["reconstruction"], marker="o", ms=2)
    axes[0].set(xlabel="Epoch", ylabel="NB reconstruction loss", title="Training reconstruction")
    axes[1].plot(history["epoch"], history["mean_eta"], label="Mean eta")
    axes[1].plot(
        history["epoch"], history["mean_own_responsibility"],
        label="Mean gene-level own responsibility",
    )
    axes[1].set(xlabel="Epoch", title="Purification diagnostics")
    axes[1].legend(frameon=False)
    fig.savefig(result_dir / "castle_crop_training_diagnostics.png", dpi=180)
    plt.close(fig)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()

