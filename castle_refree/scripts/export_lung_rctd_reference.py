"""Export the matched L1 lung Chromium reference for spacexr/RCTD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--xenium-genes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--donor", default="5", help="Chromium donor matching Xenium L1")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    xenium_genes = set(
        pd.read_csv(args.xenium_genes, sep="\t", header=None, dtype=str)[0]
    )

    adata = ad.read_h5ad(args.h5ad, backed="r")
    feature_names = adata.var["feature_name"].astype(str).to_numpy()
    feature_mask = np.asarray([name in xenium_genes for name in feature_names])
    labels = adata.obs["cell_type"].astype(str)
    donors = adata.obs["donor_id"].astype(str)
    keep_cells = (labels != "malignant cell") | (donors == str(args.donor))
    cell_indices = np.flatnonzero(keep_cells.to_numpy())
    gene_indices = np.flatnonzero(feature_mask)

    counts = adata.X[cell_indices, :][:, gene_indices]
    counts = sparse.csr_matrix(counts, dtype=np.float64)
    counts.data[counts.data < 0] = 0
    counts.eliminate_zeros()
    cell_ids = adata.obs_names[cell_indices].astype(str).to_numpy()
    genes = feature_names[gene_indices]
    selected_labels = labels.iloc[cell_indices].to_numpy()
    selected_donors = donors.iloc[cell_indices].to_numpy()
    full_library = adata.obs.iloc[cell_indices]["nCount_SoupX"].to_numpy(dtype=float)

    mmwrite(output / "reference_counts.mtx", counts.T.tocsr())
    pd.Series(genes).to_csv(output / "genes.tsv", sep="\t", header=False, index=False)
    pd.Series(cell_ids).to_csv(output / "cells.tsv", sep="\t", header=False, index=False)
    pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_type": selected_labels,
            "donor_id": selected_donors,
            "nUMI_full": full_library,
        }
    ).to_csv(output / "reference_metadata.csv", index=False)

    summary = {
        "matching_xenium_donor": "L1",
        "matching_chromium_donor_id": str(args.donor),
        "cells": int(counts.shape[0]),
        "genes_shared_with_xenium": int(counts.shape[1]),
        "cell_types": sorted(np.unique(selected_labels).tolist()),
        "malignant_donors_retained": sorted(
            np.unique(selected_donors[selected_labels == "malignant cell"]).tolist()
        ),
        "non_malignant_reference_uses_all_donors": True,
        "matrix_orientation": "genes_by_cells",
        "counts_are_soupx_corrected": True,
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
