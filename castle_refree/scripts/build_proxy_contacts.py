"""Create a documented centroid-based proxy contact graph.

Use only when segmentation masks are unavailable (currently CosMx). Delaunay
edges are retained when centroids are within a generous sum-of-cell-radii
threshold. Isolated cells receive their nearest Delaunay edge. The resulting
file has CASTLE's edges_contact.csv schema but is explicitly marked as a proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--radius-factor", type=float, default=0.0,
                        help="Optional radius filter; 0 keeps the complete Delaunay graph")
    args = parser.parse_args()

    spatial = pd.read_csv(args.bundle / "spatial.csv", dtype={"cell_id": str})
    cells = pd.read_csv(args.bundle / "cells.tsv", sep="\t", header=None, dtype=str)[0]
    spatial = spatial.set_index("cell_id").reindex(cells)
    xy = spatial[["x", "y"]].to_numpy(dtype=np.float64)
    area = spatial["area"].to_numpy(dtype=np.float64) if "area" in spatial else np.ones(len(xy))
    radii = np.sqrt(np.maximum(area, 1e-6) / np.pi)

    triangles = Delaunay(xy).simplices
    pairs = np.vstack((triangles[:, [0, 1]], triangles[:, [0, 2]], triangles[:, [1, 2]]))
    pairs.sort(axis=1)
    pairs = np.unique(pairs, axis=0)
    distances = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    reach = args.radius_factor * (radii[pairs[:, 0]] + radii[pairs[:, 1]])
    keep = np.ones(len(pairs), dtype=bool) if args.radius_factor <= 0 else distances <= reach

    # Guarantee at least one topological neighbour for every non-degenerate cell.
    degree = np.bincount(pairs[keep].ravel(), minlength=len(cells))
    add = []
    for cell in np.flatnonzero(degree == 0):
        candidate = np.flatnonzero((pairs[:, 0] == cell) | (pairs[:, 1] == cell))
        if len(candidate):
            add.append(candidate[np.argmin(distances[candidate])])
    if add:
        keep[np.asarray(add, dtype=int)] = True

    pairs, distances, reach = pairs[keep], distances[keep], reach[keep]
    # This is a positive affinity proxy, not a measured shared boundary length.
    affinity = 1.0 / np.maximum(distances, 1e-6)
    output = pd.DataFrame({
        "cell_i": cells.iloc[pairs[:, 0]].to_numpy(),
        "cell_j": cells.iloc[pairs[:, 1]].to_numpy(),
        "shared_boundary_um": affinity,
        "centroid_distance_um": distances,
    })
    output.to_csv(args.bundle / "edges_contact.csv", index=False)
    manifest = {
        "kind": "proxy_contact_graph",
        "not_segmentation_mask_contacts": True,
        "construction": (
            "complete centroid Delaunay adjacency; inverse-distance proxy affinity"
            if args.radius_factor <= 0 else
            "Delaunay edges filtered by centroid distance <= radius_factor*(r_i+r_j)"
        ),
        "radius_factor": args.radius_factor,
        "cells": len(cells), "edges": len(output),
        "cells_without_edges": int((np.bincount(pairs.ravel(), minlength=len(cells)) == 0).sum()),
    }
    (args.bundle / "proxy_contacts_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
