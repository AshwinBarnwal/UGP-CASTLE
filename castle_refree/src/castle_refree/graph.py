from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SpatialGraph:
    neighbors: np.ndarray  # donor x K recipient indices
    distances: np.ndarray
    source: np.ndarray
    target: np.ndarray
    incoming_order: np.ndarray
    incoming_indptr: np.ndarray

    @property
    def n_cells(self) -> int:
        return self.neighbors.shape[0]

    @property
    def degree(self) -> int:
        return self.neighbors.shape[1]

    def incoming_edges(self, target_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chunks: list[np.ndarray] = []
        local: list[np.ndarray] = []
        for local_id, target_id in enumerate(target_ids):
            lo = self.incoming_indptr[target_id]
            hi = self.incoming_indptr[target_id + 1]
            edge_ids = self.incoming_order[lo:hi]
            chunks.append(edge_ids)
            local.append(np.full(edge_ids.size, local_id, dtype=np.int64))
        if not chunks:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.concatenate(chunks), np.concatenate(local)


def build_spatial_graph(
    coordinates: np.ndarray,
    n_neighbors: int = 12,
    distance_quantile: float = 0.99,
) -> SpatialGraph:
    n_cells = coordinates.shape[0]
    if n_cells < 3:
        raise ValueError("At least three cells are required to build a spatial graph")
    k = min(max(2, n_neighbors), n_cells - 1)
    tree = cKDTree(coordinates)
    distances, neighbors = tree.query(coordinates, k=k + 1, workers=-1)
    distances = distances[:, 1:].astype(np.float32)
    neighbors = neighbors[:, 1:].astype(np.int64)

    cutoff = np.quantile(distances, distance_quantile)
    # Retain a fixed-size tensor for efficient caching. Long edges receive a
    # distance equal to the cutoff rather than disappearing and creating isolates.
    distances = np.minimum(distances, cutoff).astype(np.float32)
    source = np.repeat(np.arange(n_cells, dtype=np.int64), k)
    target = neighbors.reshape(-1)
    order = np.argsort(target, kind="stable")
    counts = np.bincount(target, minlength=n_cells)
    indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return SpatialGraph(neighbors, distances, source, target, order, indptr)


@dataclass
class SpilloverGraph:
    """Directed donor -> recipient edges along which transcripts can leak.

    Edges are sorted by donor so a donor's out-edges are contiguous
    (``out_indptr``), which is what the mass-conserving softmax normalises over.
    """

    source: np.ndarray  # donor index per edge
    target: np.ndarray  # recipient index per edge
    distance: np.ndarray
    boundary_fraction: np.ndarray  # c_ij / sum_k c_jk over the donor's contacts
    size_share: np.ndarray  # a_i^(1/2) / (a_i^(1/2) + a_j^(1/2)), i = recipient
    out_indptr: np.ndarray
    incoming_order: np.ndarray
    incoming_indptr: np.ndarray

    @property
    def has_out_edges(self) -> np.ndarray:
        return np.diff(self.out_indptr) > 0

    def incoming_edges(self, target_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = self.incoming_indptr[target_ids]
        hi = self.incoming_indptr[target_ids + 1]
        sizes = hi - lo
        if sizes.sum() == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        starts = np.repeat(lo - np.cumsum(sizes) + sizes, sizes)
        edge_ids = self.incoming_order[np.arange(sizes.sum()) + starts]
        local = np.repeat(np.arange(len(target_ids), dtype=np.int64), sizes)
        return edge_ids, local

    def out_edges(self, donor_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = self.out_indptr[donor_ids]
        hi = self.out_indptr[donor_ids + 1]
        sizes = hi - lo
        starts = np.repeat(lo - np.cumsum(sizes) + sizes, sizes)
        edge_ids = np.arange(sizes.sum(), dtype=np.int64) + starts
        local = np.repeat(np.arange(len(donor_ids), dtype=np.int64), sizes)
        return edge_ids, local


def _spillover_graph(
    source: np.ndarray, target: np.ndarray, distance: np.ndarray,
    boundary: np.ndarray, areas: np.ndarray, n_cells: int,
) -> SpilloverGraph:
    order = np.lexsort((target, source))
    source, target = source[order], target[order]
    distance, boundary = distance[order], boundary[order]
    out_counts = np.bincount(source, minlength=n_cells)
    out_indptr = np.concatenate(([0], np.cumsum(out_counts))).astype(np.int64)
    donor_total = np.bincount(source, weights=boundary, minlength=n_cells)
    boundary_fraction = boundary / np.maximum(donor_total[source], 1e-12)
    root = np.sqrt(areas)
    size_share = root[target] / (root[target] + root[source])
    incoming_order = np.argsort(target, kind="stable")
    in_counts = np.bincount(target, minlength=n_cells)
    incoming_indptr = np.concatenate(([0], np.cumsum(in_counts))).astype(np.int64)
    return SpilloverGraph(
        source.astype(np.int64), target.astype(np.int64),
        distance.astype(np.float32), boundary_fraction.astype(np.float32),
        size_share.astype(np.float32), out_indptr, incoming_order, incoming_indptr,
    )


def knn_spillover_graph(graph: SpatialGraph, areas: np.ndarray) -> SpilloverGraph:
    """The original kNN leakage graph (each donor leaks to its k nearest cells)."""
    return _spillover_graph(
        graph.source, graph.target, graph.distances.reshape(-1),
        np.zeros(graph.source.size), areas, graph.n_cells,
    )


def contact_spillover_graph(edges, areas: np.ndarray, n_cells: int) -> SpilloverGraph:
    """Leakage only between cells whose segmentation masks touch, in both directions."""
    i = edges["i"].to_numpy(dtype=np.int64)
    j = edges["j"].to_numpy(dtype=np.int64)
    c = edges["shared_boundary_um"].to_numpy(dtype=np.float64)
    d = edges["distance_um"].to_numpy(dtype=np.float64)
    return _spillover_graph(
        np.concatenate([i, j]), np.concatenate([j, i]), np.concatenate([d, d]),
        np.concatenate([c, c]), areas, n_cells,
    )


def _standardize(features: np.ndarray) -> np.ndarray:
    means = np.nanmean(features, axis=0, keepdims=True)
    scales = np.nanstd(features, axis=0, keepdims=True) + 1e-6
    return np.nan_to_num((features - means) / scales, nan=0.0).astype(np.float32)


def geometry_features(
    graph: SpatialGraph, coordinates: np.ndarray, areas: np.ndarray
) -> np.ndarray:
    mean_dist = graph.distances.mean(axis=1)
    min_dist = graph.distances.min(axis=1)
    density = 1.0 / np.maximum(mean_dist, 1e-6)
    features = np.column_stack(
        [np.log1p(areas), np.log1p(mean_dist), np.log1p(min_dist), np.log1p(density)]
    ).astype(np.float32)
    means = features.mean(axis=0, keepdims=True)
    scales = features.std(axis=0, keepdims=True) + 1e-6
    return (features - means) / scales


@dataclass
class StepNeighbours:
    """Cells within ``step`` contact steps (self excluded), padded per cell.

    Neighbours are reached only through touching cells. ``index`` is -1 where
    padded. ``share`` is the shared-boundary fraction for direct contacts (0 for
    farther cells); ``steps`` is the contact-step count; ``angle`` and
    ``distance`` give the direction and distance between centroids.
    """

    index: np.ndarray
    share: np.ndarray
    steps: np.ndarray
    distance: np.ndarray
    angle: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        return self.index >= 0


def contact_step_neighbours(
    edges, coordinates: np.ndarray, n_cells: int, max_steps: int
) -> list[StepNeighbours]:
    """One nested neighbour set per contact step 1..max_steps."""
    from scipy import sparse

    i = edges["i"].to_numpy(dtype=np.int64)
    j = edges["j"].to_numpy(dtype=np.int64)
    c = edges["shared_boundary_um"].to_numpy(dtype=np.float64)
    boundary = sparse.coo_matrix(
        (np.concatenate([c, c]), (np.concatenate([i, j]), np.concatenate([j, i]))),
        shape=(n_cells, n_cells),
    ).tocsr()
    adjacency = (boundary > 0).astype(np.int32).tocsr()
    total_boundary = np.asarray(boundary.sum(axis=1)).ravel()

    # Breadth-first rings: step[k] marks cells first reached at step k.
    reached = sparse.identity(n_cells, dtype=np.int32, format="csr")
    frontier = reached
    step_of = sparse.csr_matrix((n_cells, n_cells), dtype=np.int32)
    scales = []
    for step in range(1, max_steps + 1):
        expanded = (frontier @ adjacency > 0).astype(np.int32)
        new = (expanded - expanded.multiply(reached > 0)).tocsr()
        new.eliminate_zeros()
        step_of = step_of + new * step
        reached = ((reached + new) > 0).astype(np.int32).tocsr()
        frontier = new
        coo = step_of.tocoo()
        rows, cols, steps = coo.row, coo.col, coo.data
        order = np.lexsort((steps, rows))
        rows, cols, steps = rows[order], cols[order], steps[order]
        degree = np.bincount(rows, minlength=n_cells)
        width = max(int(degree.max()) if degree.size else 0, 1)
        position = np.arange(rows.size) - np.repeat(np.cumsum(degree) - degree, degree)
        index = np.full((n_cells, width), -1, dtype=np.int64)
        share = np.zeros((n_cells, width), dtype=np.float32)
        step_arr = np.zeros((n_cells, width), dtype=np.float32)
        distance = np.zeros((n_cells, width), dtype=np.float32)
        angle = np.zeros((n_cells, width), dtype=np.float32)
        delta = coordinates[cols, :2] - coordinates[rows, :2]
        direct = steps == 1
        shared = np.zeros(rows.size)
        shared[direct] = np.asarray(boundary[rows[direct], cols[direct]]).ravel() / np.maximum(
            total_boundary[rows[direct]], 1e-12
        )
        index[rows, position] = cols
        share[rows, position] = shared
        step_arr[rows, position] = steps
        distance[rows, position] = np.hypot(delta[:, 0], delta[:, 1])
        angle[rows, position] = np.arctan2(delta[:, 1], delta[:, 0])
        scales.append(StepNeighbours(index, share, step_arr, distance, angle))
    return scales


def full_geometry_features(geometry, transcript_features: bool = False) -> np.ndarray:
    """Expression-free morphology from geometry.csv (CASTLE Assumption 1).

    With ``transcript_features`` the cell's z-extent and outer-transcript fraction
    (add_transcript_geometry.py) are included as well.
    """
    columns = [
        np.log(geometry["area_um2"].to_numpy()),
        geometry["nucleus_ratio"].to_numpy(),
        geometry["circularity"].to_numpy(),
        geometry["solidity"].to_numpy(),
        np.log1p(geometry["contact_degree"].to_numpy()),
        np.log1p(geometry["contact_boundary_um"].to_numpy()),
        geometry["contact_boundary_fraction"].to_numpy(),
        np.log1p(geometry["mean_contact_distance_um"].to_numpy()),
        np.log1p(geometry["min_contact_distance_um"].to_numpy()),
        np.log1p(geometry["distance_to_tissue_edge_um"].to_numpy()),
    ]
    if transcript_features:
        missing = {"z_extent", "outer_fraction"} - set(geometry.columns)
        if missing:
            raise ValueError(f"geometry.csv lacks {sorted(missing)}; run add_transcript_geometry.py")
        columns += [
            np.log1p(geometry["z_extent"].to_numpy()),
            geometry["outer_fraction"].to_numpy(),
        ]
    return _standardize(np.column_stack(columns).astype(np.float64))

