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

