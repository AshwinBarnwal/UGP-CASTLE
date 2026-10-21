"""Cut a contiguous square crop from a raw-export CASTLE bundle for decontamination work.

The window is chosen to be densely filled with tissue and to mix cell types, so
that many cells touch a cell of a different type (the only place contamination
is identifiable). Labels are used for choosing the window and for evaluation
only; they are never written into the training inputs.

Cells within ``--margin`` of the crop border lose contacts that lie outside the
crop, so ``crop_cells.csv`` marks them ``score=False`` for evaluation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread, mmwrite

COARSE_TYPES = {
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Myoepithelial",
    "Myoepi_KRT15+": "Myoepithelial",
    "Stromal": "Stromal",
    "Perivascular-Like": "Stromal",
    "Endothelial": "Endothelial",
    "CD4+_T_Cells": "T_cell",
    "CD8+_T_Cells": "T_cell",
    "B_Cells": "B_cell",
    "Macrophages_1": "Myeloid",
    "Macrophages_2": "Myeloid",
    "IRF7+_DCs": "Myeloid",
    "LAMP3+_DCs": "Myeloid",
    "Mast_Cells": "Myeloid",
}


def load_coarse_labels(path: Path) -> pd.Series:
    labels = pd.read_csv(path, encoding="utf-8-sig", dtype={"Barcode": str})
    return labels.set_index("Barcode")["Cluster"].map(COARSE_TYPES)


def choose_window(
    xy: np.ndarray, coarse: np.ndarray, side: float, step: float
) -> tuple[float, float, dict[str, float]]:
    best: tuple[float, float, float, dict[str, float]] | None = None
    expected = len(xy) / np.prod(np.ptp(xy, axis=0)) * side**2
    for x0 in np.arange(xy[:, 0].min(), xy[:, 0].max() - side, step):
        for y0 in np.arange(xy[:, 1].min(), xy[:, 1].max() - side, step):
            inside = (
                (xy[:, 0] >= x0) & (xy[:, 0] < x0 + side)
                & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + side)
            )
            n = int(inside.sum())
            if n < 0.9 * expected:  # mostly empty space or tissue edge
                continue
            share = pd.Series(coarse[inside]).value_counts(normalize=True)
            entropy = float(-(share * np.log(share)).sum())
            if best is None or entropy > best[2]:
                best = (float(x0), float(y0), entropy, {"cells": n, **share.round(3).to_dict()})
    if best is None:
        raise ValueError("No sufficiently filled window found; lower --side")
    return best[0], best[1], {"type_entropy": best[2], **best[3]}


def make_crop(
    input_dir: Path, output_dir: Path, labels_path: Path, side: float, step: float,
    margin: float, min_counts: int,
) -> None:
    spatial = pd.read_csv(input_dir / "spatial.csv", dtype={"cell_id": str})
    cells = pd.read_csv(input_dir / "cells.tsv", sep="\t", header=None, dtype=str)[0].to_numpy()
    if not np.array_equal(cells, spatial["cell_id"].to_numpy()):
        raise ValueError("cells.tsv and spatial.csv are not in the same order")
    coarse = load_coarse_labels(labels_path).reindex(cells).fillna("Other").to_numpy()
    xy = spatial[["x", "y"]].to_numpy()
    x0, y0, summary = choose_window(xy, coarse, side, step)

    inside = (xy[:, 0] >= x0) & (xy[:, 0] < x0 + side) & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + side)
    counts = mmread(input_dir / "counts.mtx").tocsc()  # genes x cells
    totals = np.asarray(counts.sum(axis=0)).ravel()
    # Filter once here so every simulated variant keeps the same cells; the
    # decontamination configs then disable the trainer's own filtering.
    keep = np.flatnonzero(inside & (totals >= min_counts))
    kept_ids = cells[keep]
    output_dir.mkdir(parents=True, exist_ok=True)

    mmwrite(output_dir / "counts.mtx", counts[:, keep].tocoo())
    for name in ("controls.mtx", "counts_nucleus.mtx"):
        if not (input_dir / name).exists():
            continue
        matrix = mmread(input_dir / name).tocsc()  # features x cells
        mmwrite(output_dir / name, matrix[:, keep].tocoo())
    for name in ("genes.tsv", "control_features.tsv"):
        if (input_dir / name).exists():
            shutil.copy(input_dir / name, output_dir / name)
    pd.Series(kept_ids).to_csv(output_dir / "cells.tsv", sep="\t", index=False, header=False)
    spatial.iloc[keep].to_csv(output_dir / "spatial.csv", index=False)
    geometry = pd.read_csv(input_dir / "geometry.csv", dtype={"cell_id": str})
    geometry.set_index("cell_id").loc[kept_ids].reset_index().to_csv(
        output_dir / "geometry.csv", index=False
    )
    edges = pd.read_csv(input_dir / "edges_contact.csv", dtype={"cell_i": str, "cell_j": str})
    kept = set(kept_ids)
    edges[edges["cell_i"].isin(kept) & edges["cell_j"].isin(kept)].to_csv(
        output_dir / "edges_contact.csv", index=False
    )

    local = xy[keep]
    border = np.minimum.reduce(
        [local[:, 0] - x0, x0 + side - local[:, 0], local[:, 1] - y0, y0 + side - local[:, 1]]
    )
    pd.DataFrame(
        {"cell_id": kept_ids, "coarse_type": coarse[keep], "score": border >= margin}
    ).to_csv(output_dir / "crop_cells.csv", index=False)

    manifest = {
        "source_bundle": str(input_dir),
        "window_um": {"x0": x0, "y0": y0, "side": side},
        "border_margin_um": margin,
        "min_counts": min_counts,
        "cells_kept": int(len(keep)),
        "scored_cells": int((border >= margin).sum()),
        "labels_used_for_training": False,
        **summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/castle_xenium_breast_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/castle_xenium_breast_raw_crop"))
    parser.add_argument("--labels", type=Path, default=Path("data/xenium_r1_supervised_labels.csv"))
    parser.add_argument("--side", type=float, default=2200.0, help="Crop side length in um")
    parser.add_argument("--step", type=float, default=100.0, help="Window search step in um")
    parser.add_argument("--margin", type=float, default=30.0, help="Unscored border width in um")
    parser.add_argument("--min-counts", type=int, default=5, help="Drop cells with fewer counts")
    args = parser.parse_args()
    make_crop(
        args.input_dir.resolve(), args.output_dir.resolve(), args.labels.resolve(),
        args.side, args.step, args.margin, args.min_counts,
    )


if __name__ == "__main__":
    main()
