"""Generate the six lung comparison figures using CASTLE's niche embedding."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import pandas as pd

from castle_refree.scripts.compare_lung_castle_split_rctd import (
    clusters_and_metrics,
    read_vector,
    save_pair,
    umap2,
)


def main() -> None:
    castle_dir = ROOT / "results" / "castle_refree_xenium_lung_L1_beta2_100epochs"
    intrinsic_dir = ROOT / "results" / "lung_L1_castle_split_comparison"
    spatial_dir = ROOT / "data" / "castle_xenium_lung_L1"
    out = ROOT / "results" / "lung_L1_castle_niche_split_comparison"
    out.mkdir(parents=True, exist_ok=True)

    # Reuse exactly the same common-cell ordering, RCTD labels, SPLIT clusters,
    # and SPLIT UMAP as the completed intrinsic comparison.
    prior = pd.read_csv(intrinsic_dir / "cell_assignments_and_umap.csv.gz", dtype={"cell_id": str})
    common = prior["cell_id"].to_numpy(str)
    truth = prior["rctd_primary"].to_numpy(str)
    split_km = prior["split_kmeans"].to_numpy(int)
    split_leiden = prior["split_leiden"].to_numpy(int)
    split_umap = prior[["split_umap_1", "split_umap_2"]].to_numpy(np.float32)

    castle_cells = read_vector(castle_dir / "cells.tsv")
    all_niche = np.load(castle_dir / "castle_embeddings_and_qc.npz")["z_niche"].astype(np.float32)
    rows = pd.Index(castle_cells).get_indexer(common)
    if np.any(rows < 0):
        raise ValueError("Common cells are missing from CASTLE niche output")
    niche = all_niche[rows]

    spatial = pd.read_csv(spatial_dir / "spatial.csv", dtype={"cell_id": str}).set_index("cell_id")
    physical_xy = spatial.loc[common, ["x", "y"]].to_numpy(np.float32)

    niche_km, niche_leiden, niche_res, niche_metrics = clusters_and_metrics(
        niche, truth, "CASTLE niche z16"
    )
    print("Computing CASTLE niche UMAP", flush=True)
    niche_umap = umap2(niche)

    save_pair(
        out / "01_castle_niche_kmeans_vs_rctd_primary_umap.png",
        niche_umap, niche_km, "CASTLE niche UMAP — K-means clusters",
        niche_umap, truth, "Same niche UMAP — RCTD primary type",
        right_legend=True,
    )
    save_pair(
        out / "02_castle_niche_leiden_vs_rctd_primary_umap.png",
        niche_umap, niche_leiden, "CASTLE niche UMAP — Leiden clusters",
        niche_umap, truth, "Same niche UMAP — RCTD primary type",
        right_legend=True,
    )
    save_pair(
        out / "03_castle_niche_vs_split_kmeans_umap.png",
        niche_umap, niche_km, "CASTLE niche UMAP — K-means",
        split_umap, split_km, "SPLIT PCA50 UMAP — K-means",
    )
    save_pair(
        out / "04_castle_niche_vs_split_leiden_umap.png",
        niche_umap, niche_leiden, "CASTLE niche UMAP — Leiden",
        split_umap, split_leiden, "SPLIT PCA50 UMAP — Leiden",
    )
    save_pair(
        out / "05_castle_niche_vs_split_kmeans_spatial.png",
        physical_xy, niche_km, "CASTLE niche — K-means in physical space",
        physical_xy, split_km, "SPLIT — K-means in physical space",
        spatial=True,
    )
    save_pair(
        out / "06_castle_niche_vs_split_leiden_spatial.png",
        physical_xy, niche_leiden, "CASTLE niche — Leiden in physical space",
        physical_xy, split_leiden, "SPLIT — Leiden in physical space",
        spatial=True,
    )

    old_metrics = pd.read_csv(intrinsic_dir / "metrics.csv")
    split_metrics = old_metrics[old_metrics["method"] == "SPLIT PCA50"]
    metrics = pd.concat([pd.DataFrame(niche_metrics), split_metrics], ignore_index=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({
        "cell_id": common,
        "rctd_primary": truth,
        "niche_kmeans": niche_km,
        "niche_leiden": niche_leiden,
        "split_kmeans": split_km,
        "split_leiden": split_leiden,
        "niche_umap_1": niche_umap[:, 0],
        "niche_umap_2": niche_umap[:, 1],
        "split_umap_1": split_umap[:, 0],
        "split_umap_2": split_umap[:, 1],
    }).to_csv(out / "cell_assignments_and_umap.csv.gz", index=False)
    manifest = {
        "dataset": "GSE311609 / GSM9509145 / NSCLC Lung L1",
        "cells": len(common),
        "rctd_primary_types": int(pd.Series(truth).nunique()),
        "castle_representation": "native 16D niche embedding; no PCA",
        "split_panels": "reused exactly from lung_L1_castle_split_comparison",
        "umap": {"neighbors": 15, "min_dist": 0.3, "random_seed": 42},
        "kmeans": {"k": int(pd.Series(truth).nunique()), "random_seed": 42, "n_init": 10},
        "leiden": {
            "neighbors": 15,
            "resolution_grid": "0.2 to 2.0 by 0.2; selected by maximum NMI to RCTD primary labels",
            "niche_resolution": niche_res,
            "random_seed": 42,
        },
        "silhouette": "Euclidean silhouette on the same fixed 10,000-cell sampling rule",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    print(f"Saved requested niche outputs to {out}", flush=True)


if __name__ == "__main__":
    main()
