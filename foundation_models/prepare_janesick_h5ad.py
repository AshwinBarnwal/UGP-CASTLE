"""Build a raw-count AnnData file for the Janesick Xenium breast dataset.

Inputs are the portable CASTLE MatrixMarket bundle and the official published
supervised labels.  The output is cells x genes, preserves integer counts, and
contains ``obsm['spatial']`` plus ``obs['cell_type']`` for foundation-model
inference and downstream evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


def read_vector(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep="\t", header=None, dtype=str)[0].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cells = read_vector(args.bundle / "cells.tsv")
    genes = read_vector(args.bundle / "genes.tsv")
    matrix = sparse.csr_matrix(mmread(args.bundle / "counts.mtx"), dtype=np.int32)
    if matrix.shape == (len(genes), len(cells)):
        matrix = matrix.T.tocsr()
    if matrix.shape != (len(cells), len(genes)):
        raise ValueError(f"Unexpected counts shape {matrix.shape}")

    spatial = pd.read_csv(args.bundle / "spatial.csv", dtype={"cell_id": str})
    spatial = spatial.set_index("cell_id").reindex(cells)
    if spatial[["x", "y"]].isna().any().any():
        raise ValueError("Some cells are missing spatial coordinates")

    labels = pd.read_csv(args.labels, dtype={"Barcode": str})
    if labels["Barcode"].duplicated().any():
        raise ValueError("Published labels contain duplicate barcodes")
    labels = labels.set_index("Barcode").reindex(cells)
    if labels["Cluster"].isna().any():
        raise ValueError("Some cells are absent from the published label file")
    original = labels["Cluster"].astype(str)
    evaluation = original.mask(original.str.casefold().eq("unlabeled"), "Unknown")

    obs = pd.DataFrame(index=pd.Index(cells, name="cell_id"))
    obs["cell_id"] = cells
    obs["cell_type"] = evaluation.to_numpy()
    obs["published_label_original"] = original.to_numpy()
    if "area" in spatial:
        obs["cell_area"] = spatial["area"].to_numpy(dtype=np.float32)
    var = pd.DataFrame(index=pd.Index(genes, name="gene_symbol"))
    result = ad.AnnData(X=matrix, obs=obs, var=var)
    result.obsm["spatial"] = spatial[["x", "y"]].to_numpy(dtype=np.float32)
    result.uns["dataset"] = "Janesick Xenium FFPE Human Breast Cancer Replicate 1"
    result.uns["label_provenance"] = (
        "Published supervised labels transferred from matched scFFPE-seq; "
        "Unlabeled mapped to Unknown for evaluation"
    )
    ad.settings.allow_write_nullable_strings = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(args.output, compression="gzip")

    summary = {
        "cells": int(result.n_obs), "genes": int(result.n_vars),
        "nonzero_counts": int(result.X.nnz),
        "label_counts": evaluation.value_counts().to_dict(),
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
