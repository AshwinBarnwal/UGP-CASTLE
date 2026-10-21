"""Convert run_split_decontam.R outputs into the result layout scored by the suite.

SPLIT only returns cells that RCTD annotated (>= 10 counts, not rejected) and the
genes shared with its reference. To score every method on identical inputs, the
purified matrix is expanded back to the bundle's cells and genes; anything SPLIT
did not correct is passed through unchanged (what a user would keep). How many
cells and genes were passed through is recorded in ``run_metadata.json``.
SPLIT has no leakage fraction or latent space, so ``eta`` is NaN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


def read_list(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep="\t", header=None, dtype=str)[0].to_numpy()


def convert(split_dir: Path, bundle: Path) -> dict[str, object]:
    cells = read_list(bundle / "cells.tsv")
    genes = read_list(bundle / "genes.tsv")
    raw = mmread(bundle / "counts.mtx").tocsr()
    if raw.shape == (len(genes), len(cells)):
        raw = raw.T.tocsr()
    raw = raw.toarray().astype(np.float32)

    purified_part = mmread(split_dir / "purified.mtx").tocsc().T.toarray()  # cells x genes
    split_cells = read_list(split_dir / "purified_cells.tsv")
    split_genes = read_list(split_dir / "purified_genes.tsv")
    rows = pd.Index(cells).get_indexer(split_cells)
    cols = pd.Index(genes).get_indexer(split_genes)
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError(f"{split_dir} contains cells or genes absent from {bundle}")

    purified = raw.copy()
    purified[np.ix_(rows, cols)] = purified_part
    responsibility = np.where(raw > 0, purified / np.maximum(raw, 1e-9), 1.0)

    split_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(split_dir / "purified_counts_cells_by_genes.npz", sparse.csr_matrix(purified))
    sparse.save_npz(split_dir / "own_responsibility_cells_by_genes.npz", sparse.csr_matrix(responsibility))
    np.savez_compressed(
        split_dir / "castle_embeddings_and_qc.npz",
        eta=np.full(len(cells), np.nan, dtype=np.float32),
        segmentation_confidence=purified.sum(axis=1) / np.maximum(raw.sum(axis=1), 1.0),
    )
    np.savetxt(split_dir / "cells.tsv", cells, fmt="%s", delimiter="\t")
    np.savetxt(split_dir / "genes.tsv", genes, fmt="%s", delimiter="\t")
    info = {
        "cells_in_bundle": int(len(cells)),
        "cells_corrected_by_split": int(len(split_cells)),
        "cells_passed_through": int(len(cells) - len(split_cells)),
        "genes_passed_through": sorted(set(genes) - set(split_genes)),
        "removed_fraction": float(1 - purified.sum() / raw.sum()),
    }
    (split_dir / "run_metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, default=Path("results/decontamination/seed7"))
    parser.add_argument("--sim-root", type=Path, default=Path("data/castle_decontam_sim"))
    parser.add_argument("--modes", nargs="+", default=["SPLIT_default", "SPLIT_spatial"])
    parser.add_argument("--variants", nargs="+", default=["null", "spill", "niche"])
    args = parser.parse_args()
    for mode in args.modes:
        for variant in args.variants:
            split_dir = args.fits / mode / variant
            if not (split_dir / "purified.mtx").exists():
                print(f"skip {split_dir}: no SPLIT output yet")
                continue
            info = convert(split_dir, args.sim_root / variant)
            print(json.dumps({"mode": mode, "variant": variant, **info}))


if __name__ == "__main__":
    main()
