"""Export a raw Xenium output bundle to CASTLE's portable bundle with contact geometry.

Besides the usual counts/genes/cells/spatial files this writes:

- ``edges_contact.csv``: undirected cell pairs whose segmentation-mask pixels are
  4-connected neighbours (cells that physically touch), with the shared boundary
  length and centroid distance. Contact is read from the label mask that Xenium
  used for transcript assignment, not from the simplified display polygons.
- ``geometry.csv``: expression-free per-cell morphology features.
- ``controls.mtx`` / ``control_features.tsv``: per-cell negative-control counts.
- ``counts_nucleus.mtx``: counts from transcripts overlapping the nucleus
  (evaluation only; never used for training).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import zarr
from scipy import ndimage, sparse
from scipy.io import mmwrite
from scipy.spatial import ConvexHull
from zarr.storage import ZipStore

LABEL_BITS = 18  # pair key = (low label << 18) | high label
QV_THRESHOLD = 20.0  # Xenium only counts transcripts with Q-score >= 20
TISSUE_BLOCK_PX = 19  # ~4 um blocks for the coarse tissue mask


def _read_feature_matrix(path: Path) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        m = handle["matrix"]
        shape = tuple(int(x) for x in m["shape"][:])  # features x cells
        matrix = sparse.csc_matrix(
            (m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape
        )
        barcodes = m["barcodes"][:].astype(str)
        names = m["features/name"][:].astype(str)
        types = m["features/feature_type"][:].astype(str)
    return matrix.T.tocsr().astype(np.float32), barcodes, names, types


def _contacts_from_mask(
    mask: zarr.Array, n_cells: int, strip_rows: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Scan the label mask in row strips and count 4-connected label adjacencies."""
    height, width = mask.shape
    keys_all: list[np.ndarray] = []
    counts_all: list[np.ndarray] = []
    background_edges = np.zeros(n_cells + 1, dtype=np.int64)
    pixel_area = np.zeros(n_cells + 1, dtype=np.int64)
    coarse = np.zeros(
        (-(-height // TISSUE_BLOCK_PX), -(-width // TISSUE_BLOCK_PX)), dtype=bool
    )
    pad = (-width) % TISSUE_BLOCK_PX

    def add_pairs(a: np.ndarray, b: np.ndarray) -> None:
        touch = (a != b) & (a > 0) & (b > 0)
        if touch.any():
            lo = np.minimum(a[touch], b[touch]).astype(np.int64)
            hi = np.maximum(a[touch], b[touch]).astype(np.int64)
            key, cnt = np.unique((lo << LABEL_BITS) | hi, return_counts=True)
            keys_all.append(key)
            counts_all.append(cnt)
        edge = (a != b) & ((a == 0) | (b == 0))
        if edge.any():
            background_edges[:] += np.bincount(
                np.maximum(a[edge], b[edge]).astype(np.int64), minlength=n_cells + 1
            )

    previous_last_row: np.ndarray | None = None
    for r0 in range(0, height, strip_rows):
        strip = np.asarray(mask[r0 : min(r0 + strip_rows, height), :])
        pixel_area += np.bincount(strip.ravel().astype(np.int64), minlength=n_cells + 1)
        add_pairs(strip[:, :-1], strip[:, 1:])
        add_pairs(strip[:-1, :], strip[1:, :])
        if previous_last_row is not None:
            add_pairs(previous_last_row, strip[0])
        previous_last_row = strip[-1].copy()

        occupied = np.pad(strip > 0, ((0, 0), (0, pad)))
        col_any = occupied.reshape(strip.shape[0], -1, TISSUE_BLOCK_PX).any(axis=2)
        block_rows = (r0 + np.arange(strip.shape[0])) // TISSUE_BLOCK_PX
        for block in np.unique(block_rows):
            coarse[block] |= col_any[block_rows == block].any(axis=0)
        print(f"  mask rows {r0:>6}-{r0 + strip.shape[0]:<6} of {height}", flush=True)

    keys = np.concatenate(keys_all)
    counts = np.concatenate(counts_all)
    keys, inverse = np.unique(keys, return_inverse=True)
    shared_px = np.bincount(inverse, weights=counts).astype(np.int64)
    edges = pd.DataFrame(
        {
            "label_i": keys >> LABEL_BITS,
            "label_j": keys & ((1 << LABEL_BITS) - 1),
            "shared_px": shared_px,
        }
    )
    return edges, background_edges, pixel_area, coarse


def _polygon_shape(boundaries: pd.DataFrame, cell_ids: np.ndarray) -> pd.DataFrame:
    """Circularity and solidity from the (simplified) cell polygons."""
    rows = []
    for cell_id, group in boundaries.groupby("cell_id", sort=False):
        xy = group[["vertex_x", "vertex_y"]].to_numpy(dtype=np.float64)
        if len(xy) > 1 and np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        if len(xy) < 3:
            rows.append((cell_id, np.nan, np.nan, np.nan))
            continue
        x, y = xy[:, 0], xy[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        perimeter = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T).sum()
        try:
            hull_area = ConvexHull(xy).volume
        except Exception:  # degenerate polygon
            hull_area = np.nan
        rows.append((cell_id, perimeter, 4 * np.pi * area / perimeter**2, area / hull_area))
    shape = pd.DataFrame(rows, columns=["cell_id", "perimeter_um", "circularity", "solidity"])
    shape["cell_id"] = shape["cell_id"].astype(str)
    return shape.set_index("cell_id").reindex(cell_ids)


def _nucleus_counts(
    path: Path, cell_ids: np.ndarray, genes: np.ndarray, chunk_rows: int
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Gene counts from transcripts over the nucleus, plus whole-cell counts as a check."""
    cell_index = {c: i for i, c in enumerate(cell_ids)}
    gene_index = {g: i for i, g in enumerate(genes)}
    nucleus = sparse.csr_matrix((len(cell_ids), len(genes)), dtype=np.float32)
    whole = sparse.csr_matrix((len(cell_ids), len(genes)), dtype=np.float32)
    reader = pd.read_csv(
        path,
        usecols=["cell_id", "overlaps_nucleus", "feature_name", "qv"],
        dtype={"cell_id": str, "overlaps_nucleus": np.int8, "feature_name": str, "qv": np.float32},
        chunksize=chunk_rows,
    )
    seen = 0
    for chunk in reader:
        seen += len(chunk)
        chunk = chunk[chunk["qv"] >= QV_THRESHOLD]
        rows = chunk["cell_id"].map(cell_index)
        cols = chunk["feature_name"].map(gene_index)
        keep = rows.notna() & cols.notna()
        r = rows[keep].to_numpy(dtype=np.int64)
        c = cols[keep].to_numpy(dtype=np.int64)
        in_nucleus = chunk.loc[keep, "overlaps_nucleus"].to_numpy() == 1
        shape = (len(cell_ids), len(genes))
        ones = np.ones(len(r), dtype=np.float32)
        whole += sparse.csr_matrix((ones, (r, c)), shape=shape)
        nucleus += sparse.csr_matrix((ones[in_nucleus], (r[in_nucleus], c[in_nucleus])), shape=shape)
        print(f"  transcripts read: {seen:,}", flush=True)
    return nucleus, whole


def export_xenium_raw(
    bundle: Path, output_dir: Path, strip_rows: int, skip_transcripts: bool, chunk_rows: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading cell_feature_matrix.h5", flush=True)
    features, barcodes, names, types = _read_feature_matrix(bundle / "cell_feature_matrix.h5")
    is_gene = types == "Gene Expression"
    counts = features[:, is_gene].tocsr()
    controls = features[:, ~is_gene].tocsr()
    genes = names[is_gene]

    cells = pd.read_csv(bundle / "cells.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    if not np.array_equal(cells.index.to_numpy(), barcodes):
        raise ValueError("cells.csv.gz and cell_feature_matrix.h5 list cells in different orders")
    n_cells = len(barcodes)

    print("Scanning the cell segmentation mask for contacts", flush=True)
    with ZipStore(bundle / "cells.zarr.zip", mode="r") as store:
        root = zarr.open_group(store, mode="r", zarr_format=2)
        um_to_px = float(np.asarray(root["masks/homogeneous_transform"][:])[0, 0])
        edges, background_edges, pixel_area, coarse = _contacts_from_mask(
            root["masks/1"], n_cells, strip_rows
        )
    px_um = 1.0 / um_to_px

    # The mask label of a cell is its index + 1 (10x XOA zarr specification).
    label_to_id = np.concatenate([[""], barcodes])
    xy = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)
    i = edges["label_i"].to_numpy() - 1
    j = edges["label_j"].to_numpy() - 1
    edges_out = pd.DataFrame(
        {
            "cell_i": label_to_id[edges["label_i"]],
            "cell_j": label_to_id[edges["label_j"]],
            "shared_boundary_um": edges["shared_px"].to_numpy() * px_um,
            "centroid_distance_um": np.hypot(*(xy[i] - xy[j]).T),
        }
    )
    edges_out.to_csv(output_dir / "edges_contact.csv", index=False)

    mask_area_um2 = pixel_area[1:] * px_um**2
    area_check = np.corrcoef(mask_area_um2, cells["cell_area"].to_numpy())[0, 1]

    contact_degree = np.bincount(np.concatenate([i, j]), minlength=n_cells)
    contact_um = np.bincount(
        np.concatenate([i, j]),
        weights=np.tile(edges_out["shared_boundary_um"].to_numpy(), 2),
        minlength=n_cells,
    )
    background_um = background_edges[1:] * px_um
    dist = np.tile(edges_out["centroid_distance_um"].to_numpy(), 2)
    endpoint = np.concatenate([i, j])
    mean_contact = np.bincount(endpoint, weights=dist, minlength=n_cells) / np.maximum(contact_degree, 1)
    min_contact = np.full(n_cells, np.nan)
    np.fmin.at(min_contact, endpoint, dist)
    mean_contact[contact_degree == 0] = np.nan

    tissue = ndimage.binary_fill_holes(ndimage.binary_closing(coarse, iterations=5))
    edge_distance = ndimage.distance_transform_edt(tissue) * TISSUE_BLOCK_PX * px_um
    block = np.clip(
        (xy * um_to_px // TISSUE_BLOCK_PX).astype(np.int64),
        0,
        np.array(coarse.shape[::-1]) - 1,
    )
    dist_to_edge = edge_distance[block[:, 1], block[:, 0]]

    print("Computing polygon shape features", flush=True)
    boundaries = pd.read_csv(bundle / "cell_boundaries.csv.gz", dtype={"cell_id": str})
    shape = _polygon_shape(boundaries, barcodes)

    geometry = pd.DataFrame(
        {
            "cell_id": barcodes,
            "area_um2": cells["cell_area"].to_numpy(),
            "nucleus_area_um2": cells["nucleus_area"].to_numpy(),
            "nucleus_ratio": cells["nucleus_area"].to_numpy() / cells["cell_area"].to_numpy(),
            "perimeter_um": shape["perimeter_um"].to_numpy(),
            "circularity": shape["circularity"].to_numpy(),
            "solidity": shape["solidity"].to_numpy(),
            "contact_degree": contact_degree,
            "contact_boundary_um": contact_um,
            "background_boundary_um": background_um,
            "contact_boundary_fraction": contact_um / np.maximum(contact_um + background_um, 1e-9),
            "mean_contact_distance_um": mean_contact,
            "min_contact_distance_um": min_contact,
            "distance_to_tissue_edge_um": dist_to_edge,
        }
    )
    geometry.to_csv(output_dir / "geometry.csv", index=False)

    # Same layout as the other CASTLE bundles: MatrixMarket genes x cells.
    mmwrite(output_dir / "counts.mtx", counts.T.tocoo())
    mmwrite(output_dir / "controls.mtx", controls.T.tocoo())
    pd.Series(genes).to_csv(output_dir / "genes.tsv", sep="\t", index=False, header=False)
    pd.Series(barcodes).to_csv(output_dir / "cells.tsv", sep="\t", index=False, header=False)
    pd.DataFrame({"feature": names[~is_gene], "feature_type": types[~is_gene]}).to_csv(
        output_dir / "control_features.tsv", sep="\t", index=False
    )
    # Control totals are deliberately not named negative_control_counts: the
    # current loader would treat that column as an already-calibrated background.
    pd.DataFrame(
        {
            "cell_id": barcodes,
            "x": xy[:, 0],
            "y": xy[:, 1],
            "area": cells["cell_area"].to_numpy(),
            "nucleus_area": cells["nucleus_area"].to_numpy(),
            "control_probe_counts": cells["control_probe_counts"].to_numpy(),
            "control_codeword_counts": cells["control_codeword_counts"].to_numpy(),
        }
    ).to_csv(output_dir / "spatial.csv", index=False)

    transcript_check: dict[str, float] = {}
    if not skip_transcripts:
        print("Reading transcripts for nucleus-only counts", flush=True)
        nucleus, whole = _nucleus_counts(
            bundle / "transcripts.csv.gz", barcodes, genes, chunk_rows
        )
        mmwrite(output_dir / "counts_nucleus.mtx", nucleus.T.tocoo())
        difference = abs(whole - counts)
        transcript_check = {
            "transcript_whole_cell_total": float(whole.sum()),
            "feature_matrix_total": float(counts.sum()),
            "cells_with_any_mismatch": int((np.asarray(difference.sum(axis=1)).ravel() > 0).sum()),
            "nucleus_fraction_of_counts": float(nucleus.sum() / max(counts.sum(), 1.0)),
        }

    manifest = {
        "source_bundle": str(bundle),
        "cells": int(n_cells),
        "genes": int(counts.shape[1]),
        "control_features": int(controls.shape[1]),
        "segmentation": "Xenium nucleus expansion (label mask masks/1 in cells.zarr.zip)",
        "contact_definition": "4-connected adjacency of label-mask pixels",
        "pixel_size_um": px_um,
        "contact_edges": int(len(edges_out)),
        "median_contact_degree": float(np.median(contact_degree)),
        "cells_without_contacts": int((contact_degree == 0).sum()),
        "mask_area_vs_cell_area_correlation": float(area_check),
        "qv_threshold_for_nucleus_counts": QV_THRESHOLD,
        **transcript_check,
        "counts_nucleus_used_for_training": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "output_dir": str(output_dir.resolve())}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Xenium output bundle directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/castle_xenium_breast_raw"))
    parser.add_argument("--strip-rows", type=int, default=806, help="Mask rows per scan strip")
    parser.add_argument("--skip-transcripts", action="store_true", help="Do not build nucleus counts")
    parser.add_argument("--chunk-rows", type=int, default=5_000_000, help="Transcript rows per chunk")
    args = parser.parse_args()
    export_xenium_raw(
        args.bundle.resolve(),
        args.output_dir.resolve(),
        args.strip_rows,
        args.skip_transcripts,
        args.chunk_rows,
    )


if __name__ == "__main__":
    main()
