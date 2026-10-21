"""Export a count layer from an AnnData H5AD file to CASTLE's portable bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite


def _strings(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset.asstr()[:], dtype=str)


def _obs_column(obs: h5py.Group, name: str) -> np.ndarray:
    if name not in obs:
        raise ValueError(f"H5AD obs has no column {name!r}")
    value = obs[name]
    if isinstance(value, h5py.Group):
        encoding = value.attrs.get("encoding-type", "")
        if encoding != "categorical" or "codes" not in value or "categories" not in value:
            raise ValueError(f"Unsupported H5AD encoding for obs[{name!r}]")
        codes = value["codes"][:]
        categories = _strings(value["categories"])
        decoded = np.full(codes.shape, "", dtype=object)
        valid = codes >= 0
        decoded[valid] = categories[codes[valid].astype(np.int64)]
        return decoded.astype(str)
    if h5py.check_string_dtype(value.dtype) is not None:
        return _strings(value)
    return np.asarray(value[:])


def _read_count_layer(file: h5py.File, layer: str) -> sparse.csr_matrix:
    path = "X" if layer == "X" else f"layers/{layer}"
    if path not in file:
        raise ValueError(f"H5AD has no matrix at {path!r}")
    node = file[path]
    encoding = node.attrs.get("encoding-type", "")
    shape = (
        tuple(int(x) for x in node.attrs["shape"])
        if "shape" in node.attrs
        else tuple(int(x) for x in node.shape)
    )
    if encoding == "csr_matrix":
        matrix = sparse.csr_matrix(
            (node["data"][:], node["indices"][:], node["indptr"][:]), shape=shape
        )
    elif encoding == "csc_matrix":
        matrix = sparse.csc_matrix(
            (node["data"][:], node["indices"][:], node["indptr"][:]), shape=shape
        ).tocsr()
    elif isinstance(node, h5py.Dataset):
        matrix = sparse.csr_matrix(node[:])
    else:
        raise ValueError(
            f"Unsupported H5AD matrix encoding {encoding!r} at {path!r}; "
            "expected a sparse CSR/CSC matrix or dense array"
        )
    matrix = matrix.astype(np.float32, copy=False)
    if matrix.ndim != 2 or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("The selected expression matrix must contain finite nonnegative values")
    return matrix


def export_h5ad(
    input_path: Path,
    output_dir: Path,
    layer: str,
    x_column: str,
    y_column: str,
    label_column: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as handle:
        counts = _read_count_layer(handle, layer)
        obs = handle["obs"]
        var = handle["var"]
        obs_index = obs.attrs.get("_index", "_index")
        var_index = var.attrs.get("_index", "_index")
        cell_ids = _strings(obs[obs_index])
        gene_ids = _strings(var[var_index])
        x = np.asarray(_obs_column(obs, x_column), dtype=np.float64)
        y = np.asarray(_obs_column(obs, y_column), dtype=np.float64)
        labels = _obs_column(obs, label_column)
        n_counts = _obs_column(obs, "n_counts") if "n_counts" in obs else None

    if counts.shape != (len(cell_ids), len(gene_ids)):
        raise ValueError(
            f"Selected matrix shape {counts.shape} does not match obs/var "
            f"({len(cell_ids)}, {len(gene_ids)})"
        )
    if len(np.unique(cell_ids)) != len(cell_ids) or len(np.unique(gene_ids)) != len(gene_ids):
        raise ValueError("Cell and gene identifiers must be unique")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Selected spatial coordinates contain missing or non-finite values")
    if n_counts is not None:
        row_sums = np.asarray(counts.sum(axis=1)).ravel()
        if not np.allclose(row_sums, np.asarray(n_counts, dtype=float), rtol=0, atol=1e-5):
            raise ValueError(
                f"Layer {layer!r} row sums do not match obs['n_counts']; "
                "choose the raw count layer before exporting"
            )

    # Match the current CASTLE bundle format: MatrixMarket is genes by cells;
    # the loader detects and transposes this to cells by genes.
    mmwrite(output_dir / "counts.mtx", counts.T.tocoo())
    pd.Series(gene_ids).to_csv(output_dir / "genes.tsv", sep="\t", index=False, header=False)
    pd.Series(cell_ids).to_csv(output_dir / "cells.tsv", sep="\t", index=False, header=False)
    pd.DataFrame(
        {"cell_id": cell_ids, "x": x, "y": y, "area": np.ones(len(cell_ids), dtype=np.float32)}
    ).to_csv(output_dir / "spatial.csv", index=False)
    pd.DataFrame({"cell_id": cell_ids, label_column: labels}).to_csv(
        output_dir / "cell_annotations.csv", index=False
    )
    manifest = {
        "source_file": input_path.name,
        "format": "AnnData H5AD",
        "count_layer": layer,
        "cells": int(counts.shape[0]),
        "genes": int(counts.shape[1]),
        "coordinates": [x_column, y_column],
        "area": "not present in source; exported as 1 for every cell",
        "evaluation_annotation": label_column,
        "annotation_used_for_training": False,
        "negative_controls": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "output_dir": str(output_dir.resolve())}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input .h5ad file")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/castle_xenium_proseg_breast")
    )
    parser.add_argument(
        "--layer", default="counts", help="AnnData layer to export (default: counts)"
    )
    parser.add_argument("--x-column", default="x", help="Spatial x column in obs")
    parser.add_argument("--y-column", default="y", help="Spatial y column in obs")
    parser.add_argument(
        "--label-column", default="cell_type", help="Evaluation-only annotation column"
    )
    args = parser.parse_args()
    export_h5ad(
        args.input.resolve(),
        args.output_dir.resolve(),
        args.layer,
        args.x_column,
        args.y_column,
        args.label_column,
    )


if __name__ == "__main__":
    main()
