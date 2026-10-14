"""Plot SPLIT cluster assignments against RCTD primary labels on one UMAP."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))

import pandas as pd
import matplotlib.pyplot as plt

from castle_refree.scripts.compare_lung_castle_split_rctd import (
    categorical_scatter,
    save_pair,
)


def main() -> None:
    result_dir = ROOT / "results" / "lung_L1_castle_split_comparison"
    assignments = pd.read_csv(
        result_dir / "cell_assignments_and_umap.csv.gz", dtype={"cell_id": str}
    )
    xy = assignments[["split_umap_1", "split_umap_2"]].to_numpy()
    truth = assignments["rctd_primary"].to_numpy(str)

    save_pair(
        result_dir / "07_split_kmeans_vs_rctd_primary_umap.png",
        xy,
        assignments["split_kmeans"].to_numpy(int),
        "SPLIT PCA50 UMAP — K-means clusters",
        xy,
        truth,
        "Same SPLIT UMAP — RCTD primary type",
        right_legend=True,
    )
    save_pair(
        result_dir / "08_split_leiden_vs_rctd_primary_umap.png",
        xy,
        assignments["split_leiden"].to_numpy(int),
        "SPLIT PCA50 UMAP — Leiden clusters",
        xy,
        truth,
        "Same SPLIT UMAP — RCTD primary type",
        right_legend=True,
    )

    spatial = pd.read_csv(
        ROOT / "data" / "castle_xenium_lung_L1" / "spatial.csv",
        dtype={"cell_id": str},
    ).set_index("cell_id")
    physical_xy = spatial.loc[assignments["cell_id"], ["x", "y"]].to_numpy()
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    categorical_scatter(
        ax,
        physical_xy,
        truth,
        "RCTD primary cell types in physical Xenium space",
        size=0.8,
        legend=True,
        invert_y=True,
    )
    fig.savefig(
        result_dir / "09_rctd_primary_types_spatial.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved SPLIT-vs-RCTD and standalone RCTD spatial figures to {result_dir}")


if __name__ == "__main__":
    main()
