from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


@dataclass
class SpatialCountData:
    counts: sparse.csr_matrix  # cells x genes
    coordinates: np.ndarray  # cells x 2 (or 3)
    areas: np.ndarray
    cell_ids: np.ndarray
    gene_ids: np.ndarray
    negative_control_counts: np.ndarray | None = None

    @property
    def n_cells(self) -> int:
        return self.counts.shape[0]

    @property
    def n_genes(self) -> int:
        return self.counts.shape[1]


def _read_vector(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t", header=None, dtype=str)
    return frame.iloc[:, 0].to_numpy()


def load_spatial_data(
    input_dir: str | Path,
    min_counts: int = 5,
    min_cells_per_gene: int = 10,
) -> SpatialCountData:
    """Load the portable MatrixMarket bundle made by export_xenium_breast.R."""
    root = Path(input_dir)
    required = ["counts.mtx", "genes.tsv", "cells.tsv", "spatial.csv"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {root}: {', '.join(missing)}. "
            "Run R/export_xenium_breast.R first."
        )

    genes = _read_vector(root / "genes.tsv")
    cells = _read_vector(root / "cells.tsv")
    matrix = mmread(root / "counts.mtx")
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    if matrix.shape == (len(genes), len(cells)):
        matrix = matrix.T.tocsr()
    elif matrix.shape != (len(cells), len(genes)):
        raise ValueError(
            f"Count matrix shape {matrix.shape} matches neither cells x genes "
            f"({len(cells)}, {len(genes)}) nor genes x cells."
        )

    # Cell identifiers often look numeric in Xenium exports. Force strings so
    # pandas does not turn one side of the join into integers.
    spatial = pd.read_csv(root / "spatial.csv", dtype={"cell_id": str})
    if "cell_id" not in spatial:
        raise ValueError("spatial.csv must contain a cell_id column")
    spatial = spatial.set_index("cell_id").reindex(cells)
    if spatial.isnull().all(axis=1).any():
        raise ValueError("Some cells.tsv identifiers are absent from spatial.csv")
    coord_cols = [c for c in ("x", "y", "z") if c in spatial.columns]
    if len(coord_cols) < 2:
        raise ValueError("spatial.csv must contain x and y columns")
    coordinates = spatial[coord_cols].to_numpy(dtype=np.float32)
    areas = (
        spatial["area"].to_numpy(dtype=np.float32)
        if "area" in spatial
        else np.ones(len(cells), dtype=np.float32)
    )
    neg = (
        spatial["negative_control_counts"].to_numpy(dtype=np.float32)
        if "negative_control_counts" in spatial
        else None
    )

    cell_total = np.asarray(matrix.sum(axis=1)).ravel()
    keep_cells = cell_total >= min_counts
    matrix = matrix[keep_cells]
    coordinates = coordinates[keep_cells]
    areas = areas[keep_cells]
    cells = cells[keep_cells]
    if neg is not None:
        neg = neg[keep_cells]

    detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
    keep_genes = detected >= min_cells_per_gene
    matrix = matrix[:, keep_genes].tocsr()
    genes = genes[keep_genes]
    matrix.eliminate_zeros()

    if matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("Filtering removed all cells or left fewer than two genes")
    if not np.isfinite(coordinates).all() or not np.isfinite(areas).all():
        raise ValueError("Coordinates and areas must be finite")
    areas = np.maximum(areas, np.finfo(np.float32).eps)
    return SpatialCountData(matrix, coordinates, areas, cells, genes, neg)
