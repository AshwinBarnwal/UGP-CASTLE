"""Prepare the official CosMx Human Frontal Cortex flat files for all methods.

The output is intentionally redundant: an H5AD serves scVI/scVIVA, while the
MatrixMarket bundle serves CASTLE and the RCTD/SPLIT R pipeline.  Molecule-level
and polygon files are not needed by any of those cell-level methods.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite


ROOT = Path(__file__).resolve().parents[2]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", type=Path,
        default=ROOT / "data" / "cosmx_frontal_cortex_raw" / "S3",
    )
    parser.add_argument(
        "--bundle-dir", type=Path,
        default=ROOT / "data" / "cosmx_frontal_cortex",
    )
    parser.add_argument(
        "--h5ad", type=Path,
        default=ROOT / "data" / "cosmx_frontal_cortex.h5ad",
    )
    parser.add_argument("--chunk-size", type=int, default=2048)
    return parser.parse_args()


def broad_label(label: str) -> str:
    if label.startswith("astro."):
        return "Astrocyte"
    if label == "endothelial":
        return "Endothelial"
    if label.startswith("Inh."):
        return "Inhibitory neuron"
    if label in {"L2_3", "L4", "L6"}:
        return "Cortical layer neuron"
    if label.startswith("microglia."):
        return "Microglia"
    if label == "oligodendrocyte":
        return "Oligodendrocyte"
    if label == "OPC":
        return "OPC"
    return "Unknown"


def main() -> None:
    args = arguments()
    expr = args.raw_dir / "S3_exprMat_file.csv.gz"
    meta_path = args.raw_dir / "S3_metadata_file.csv.gz"
    for path in (expr, meta_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with gzip.open(expr, "rt", newline="") as handle:
        header = next(csv.reader(handle))
    id_columns = header[:3]
    feature_columns = header[3:]
    control_columns = [
        name for name in feature_columns
        if name.startswith("NegPrb") or name.startswith("FalseCode")
    ]
    genes = [name for name in feature_columns if name not in set(control_columns)]
    print(f"Detected {len(genes):,} genes and {len(control_columns):,} controls")

    meta = pd.read_csv(meta_path, low_memory=False)
    if meta["cell"].duplicated().any():
        raise ValueError("CosMx metadata contains duplicate cell identifiers")
    meta = meta.set_index("cell", drop=False)

    blocks: list[sparse.csr_matrix] = []
    cell_ids: list[str] = []
    negative_counts: list[np.ndarray] = []
    usecols = id_columns + genes + control_columns
    for part_number, chunk in enumerate(
        pd.read_csv(expr, usecols=usecols, chunksize=args.chunk_size), start=1
    ):
        ids = chunk["cell"].astype(str).tolist()
        values = chunk[genes].to_numpy(dtype=np.int32, copy=False)
        if np.any(values < 0):
            raise ValueError("Counts contain negative values")
        blocks.append(sparse.csr_matrix(values))
        cell_ids.extend(ids)
        negative_counts.append(
            chunk[control_columns].to_numpy(dtype=np.int32, copy=False).sum(axis=1)
        )
        print(f"Read block {part_number}: {len(cell_ids):,} cells", flush=True)

    counts = sparse.vstack(blocks, format="csr", dtype=np.int32)
    del blocks
    cells = np.asarray(cell_ids, dtype=str)
    if len(cells) != len(meta) or set(cells) != set(meta.index.astype(str)):
        raise ValueError("Count-matrix and metadata cell identifiers do not match")
    meta = meta.loc[cells].copy()
    neg = np.concatenate(negative_counts).astype(np.float32)

    refined = meta["RefinedClusts_Final"].astype(str)
    broad = refined.map(broad_label)
    obs = pd.DataFrame(index=pd.Index(cells, name="cell_id"))
    obs["cell_type"] = broad.to_numpy()
    obs["cell_type_refined"] = refined.to_numpy()
    obs["sample"] = meta["Run_Tissue_name"].astype(str).to_numpy()
    obs["fov"] = meta["fov"].astype(str).to_numpy()
    obs["qc_pass"] = (~meta["qcCellsFlagged"].astype(bool)).to_numpy()
    obs["n_counts"] = np.asarray(counts.sum(axis=1)).ravel()
    obs["area_um2"] = meta["Area.um2"].astype(float).to_numpy()
    obs["negative_control_counts"] = neg
    spatial = meta[["CenterX_global_px", "CenterY_global_px"]].to_numpy(np.float32)

    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    args.h5ad.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.bundle_dir / "genes.tsv", np.asarray(genes), fmt="%s")
    np.savetxt(args.bundle_dir / "cells.tsv", cells, fmt="%s")
    pd.DataFrame(
        {
            "cell_id": cells,
            "x": spatial[:, 0],
            "y": spatial[:, 1],
            "area": obs["area_um2"].to_numpy(),
            "negative_control_counts": neg,
            "cell_type": broad.to_numpy(),
            "cell_type_refined": refined.to_numpy(),
            "fov": obs["fov"].to_numpy(),
        }
    ).to_csv(args.bundle_dir / "spatial.csv", index=False)
    # genes x cells keeps this compatible with the existing RCTD scripts.
    mmwrite(args.bundle_dir / "counts.mtx", counts.T.tocoo(), precision=8)

    adata = ad.AnnData(X=counts)
    adata.obs = obs
    adata.var_names = pd.Index(genes, name="gene")
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X
    adata.obsm["spatial"] = spatial
    adata.uns["dataset"] = {
        "name": "CosMx Human Frontal Cortex FFPE S3",
        "source": "Bruker/NanoString 6k release flatFiles.zip",
        "official_refined_label": "RefinedClusts_Final",
        "broad_label_mapping": "PASTA paper's seven named families plus Unknown",
    }
    adata.write_h5ad(args.h5ad, compression="gzip")

    manifest = {
        "cells": int(counts.shape[0]),
        "genes": int(counts.shape[1]),
        "nonzero_counts": int(counts.nnz),
        "controls": len(control_columns),
        "matrix_orientation": "genes_by_cells",
        "h5ad": str(args.h5ad.resolve()),
        "broad_label_counts": broad.value_counts().to_dict(),
        "refined_label_counts": refined.value_counts().to_dict(),
    }
    (args.bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
