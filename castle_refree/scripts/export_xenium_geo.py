"""Convert official Xenium H5 + cells Parquet files into a CASTLE bundle."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-h5", required=True)
    parser.add_argument("--cells-parquet", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    matrix_path = Path(args.matrix_h5)
    parquet_path = Path(args.cells_parquet)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if parquet_path.suffix == ".gz":
        expanded = output / "cells_source.parquet"
        with gzip.open(parquet_path, "rb") as source, expanded.open("wb") as target:
            shutil.copyfileobj(source, target)
        parquet_path = expanded

    with h5py.File(matrix_path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(int(x) for x in group["shape"][:])
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=shape,
            dtype=np.float32,
        )
        barcodes = decode(group["barcodes"][:])
        genes = decode(group["features"]["name"][:])
        feature_types = decode(group["features"]["feature_type"][:])

    if matrix.shape != (len(genes), len(barcodes)):
        raise ValueError("H5 matrix dimensions do not match features/barcodes")

    gene_mask = feature_types == "Gene Expression"
    control_mask = np.isin(
        feature_types,
        ["Negative Control Codeword", "Negative Control Probe"],
    )
    expression = matrix[gene_mask, :].tocsr()
    control_counts = np.asarray(matrix[control_mask, :].sum(axis=0)).ravel()
    selected_genes = genes[gene_mask]

    cells = pd.read_parquet(parquet_path)
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells = cells.set_index("cell_id").reindex(barcodes)
    required = ["x_centroid", "y_centroid", "cell_area"]
    if cells[required].isna().any().any():
        raise ValueError("Cell metadata is missing coordinates/area for H5 barcodes")

    mmwrite(output / "counts.mtx", expression)
    pd.Series(selected_genes).to_csv(
        output / "genes.tsv", sep="\t", index=False, header=False
    )
    pd.Series(barcodes).to_csv(
        output / "cells.tsv", sep="\t", index=False, header=False
    )
    pd.DataFrame(
        {
            "cell_id": barcodes,
            "x": cells["x_centroid"].to_numpy(dtype=np.float32),
            "y": cells["y_centroid"].to_numpy(dtype=np.float32),
            "area": cells["cell_area"].to_numpy(dtype=np.float32),
            "nucleus_area": cells["nucleus_area"].to_numpy(dtype=np.float32),
            "negative_control_counts": control_counts.astype(np.float32),
        }
    ).to_csv(output / "spatial.csv", index=False)

    manifest = {
        "source_matrix": str(matrix_path.resolve()),
        "source_cells": str(Path(args.cells_parquet).resolve()),
        "cells": int(expression.shape[1]),
        "genes": int(expression.shape[0]),
        "matrix_orientation": "genes_by_cells",
        "gene_expression_counts": float(expression.sum()),
        "negative_control_counts": float(control_counts.sum()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(manifest)


if __name__ == "__main__":
    main()
