"""Add transcript-derived geometry (CASTLE Assumption 1) and molecule tables.

Per cell, from Q >= 20 transcripts assigned to it:

- ``cell_z``: median transcript z (the cell's z position, for the kernel's
  z-plane term, spec eq. 15);
- ``z_extent``: 90th minus 10th percentile of transcript z;
- ``outer_fraction``: share of the cell's transcripts whose distance from the
  nucleus centroid, in units of the cell's equivalent radius sqrt(area / pi),
  exceeds the dataset-wide 90th percentile of that normalised distance.

All are expression-free. Columns are written to the raw bundle's geometry.csv and
merged into every bundle that copies it (crop and simulated datasets). For the
crop, a molecule table (``molecules.npz``) is saved for transcript-level
refinement (spec Sec. 4.6).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

QV_THRESHOLD = 20.0
COLUMNS = ["cell_z", "z_extent", "outer_fraction", "n_molecules"]


def polygon_centroids(path: Path) -> pd.DataFrame:
    rows = []
    for cell_id, group in pd.read_csv(path, dtype={"cell_id": str}).groupby("cell_id", sort=False):
        xy = group[["vertex_x", "vertex_y"]].to_numpy(dtype=np.float64)
        if len(xy) > 1 and np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        x, y = xy[:, 0], xy[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area = cross.sum() / 2.0
        if abs(area) < 1e-9:
            rows.append((cell_id, x.mean(), y.mean()))
            continue
        cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * area)
        cy = ((y + np.roll(y, -1)) * cross).sum() / (6.0 * area)
        rows.append((cell_id, cx, cy))
    return pd.DataFrame(rows, columns=["cell_id", "nx", "ny"]).set_index("cell_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Xenium output bundle directory")
    parser.add_argument("--raw", type=Path, default=Path("data/castle_xenium_breast_raw"))
    parser.add_argument("--crop", type=Path, default=Path("data/castle_xenium_breast_raw_crop"))
    parser.add_argument("--propagate", nargs="*", type=Path, default=[Path("data/castle_decontam_sim")])
    parser.add_argument("--chunk-rows", type=int, default=5_000_000)
    args = parser.parse_args()

    cells = pd.read_csv(args.bundle / "cells.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    nuclei = polygon_centroids(args.bundle / "nucleus_boundaries.csv.gz").reindex(cells.index)
    missing = nuclei["nx"].isna()
    nuclei.loc[missing, "nx"] = cells.loc[missing, "x_centroid"]
    nuclei.loc[missing, "ny"] = cells.loc[missing, "y_centroid"]
    radius = np.sqrt(cells["cell_area"].to_numpy() / np.pi)
    index = pd.Index(cells.index)

    crop_cells = pd.read_csv(args.crop / "cells.tsv", sep="\t", header=None, dtype=str)[0].to_numpy()
    crop_genes = pd.read_csv(args.crop / "genes.tsv", sep="\t", header=None, dtype=str)[0].to_numpy()
    crop_index = pd.Index(crop_cells)
    gene_index = pd.Index(crop_genes)

    cell_of, z_all, rho_all = [], [], []
    molecules = {k: [] for k in ("cell", "gene", "x", "y", "z", "overlaps_nucleus")}
    reader = pd.read_csv(
        args.bundle / "transcripts.csv.gz",
        usecols=["cell_id", "overlaps_nucleus", "feature_name", "x_location", "y_location", "z_location", "qv"],
        dtype={"cell_id": str, "feature_name": str},
        chunksize=args.chunk_rows,
    )
    seen = 0
    for chunk in reader:
        seen += len(chunk)
        chunk = chunk[chunk["qv"] >= QV_THRESHOLD]
        rows = index.get_indexer(chunk["cell_id"])
        keep = rows >= 0
        chunk, rows = chunk[keep], rows[keep]
        x = chunk["x_location"].to_numpy(np.float64)
        y = chunk["y_location"].to_numpy(np.float64)
        z = chunk["z_location"].to_numpy(np.float32)
        rho = np.hypot(x - nuclei["nx"].to_numpy()[rows], y - nuclei["ny"].to_numpy()[rows]) / np.maximum(radius[rows], 1e-6)
        cell_of.append(rows.astype(np.int32))
        z_all.append(z)
        rho_all.append(rho.astype(np.float32))

        crop_rows = crop_index.get_indexer(chunk["cell_id"])
        gene_cols = gene_index.get_indexer(chunk["feature_name"])
        sel = (crop_rows >= 0) & (gene_cols >= 0)
        molecules["cell"].append(crop_rows[sel].astype(np.int32))
        molecules["gene"].append(gene_cols[sel].astype(np.int16))
        molecules["x"].append(x[sel].astype(np.float32))
        molecules["y"].append(y[sel].astype(np.float32))
        molecules["z"].append(z[sel])
        molecules["overlaps_nucleus"].append(chunk["overlaps_nucleus"].to_numpy(np.int8)[sel])
        print(f"  transcripts read: {seen:,}", flush=True)

    cell_of = np.concatenate(cell_of)
    z_all = np.concatenate(z_all)
    rho_all = np.concatenate(rho_all)
    threshold = float(np.quantile(rho_all, 0.9))

    frame = pd.DataFrame({"cell": cell_of, "z": z_all, "outer": rho_all > threshold})
    grouped = frame.groupby("cell")
    geometry = pd.DataFrame(index=np.arange(len(index)))
    geometry["cell_z"] = grouped["z"].median()
    geometry["z_extent"] = grouped["z"].quantile(0.9) - grouped["z"].quantile(0.1)
    geometry["outer_fraction"] = grouped["outer"].mean()
    geometry["n_molecules"] = grouped.size()
    geometry = geometry.fillna({"z_extent": 0.0, "outer_fraction": 0.0, "n_molecules": 0})
    geometry["cell_z"] = geometry["cell_z"].fillna(float(np.median(z_all)))
    geometry.index = index

    def merge(path: Path) -> None:
        target = pd.read_csv(path, dtype={"cell_id": str})
        target = target.drop(columns=[c for c in COLUMNS if c in target.columns])
        target = target.join(geometry, on="cell_id")
        target.to_csv(path, index=False)

    merge(args.raw / "geometry.csv")
    merge(args.crop / "geometry.csv")
    for root in args.propagate:
        for path in sorted(root.glob("*/geometry.csv")):
            merge(path)

    np.savez_compressed(
        args.crop / "molecules.npz",
        **{k: np.concatenate(v) for k, v in molecules.items()},
        nucleus_x=nuclei["nx"].reindex(crop_cells).to_numpy(np.float32),
        nucleus_y=nuclei["ny"].reindex(crop_cells).to_numpy(np.float32),
        cell_radius=np.sqrt(cells["cell_area"].reindex(crop_cells).to_numpy() / np.pi).astype(np.float32),
    )
    info = {
        "molecules_used": int(len(cell_of)),
        "outer_distance_threshold_cell_radii": threshold,
        "median_cell_z": float(np.median(geometry["cell_z"])),
        "median_z_extent": float(np.median(geometry["z_extent"])),
        "median_outer_fraction": float(np.median(geometry["outer_fraction"])),
        "crop_molecules": int(sum(len(v) for v in molecules["cell"])),
    }
    (args.raw / "transcript_geometry.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
