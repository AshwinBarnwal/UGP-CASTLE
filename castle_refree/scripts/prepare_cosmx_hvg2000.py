"""Create one fixed 2,000-HVG CosMx input shared by GPU methods."""

from pathlib import Path
import json
import shutil

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "cosmx_frontal_cortex.h5ad"
REFERENCE = ROOT / "data" / "gse104276_reference" / "GSE104276_all_pfc_2394_UMI_count_NOERCC.xls"
OUTPUT = ROOT / "data" / "cosmx_frontal_cortex_hvg2000.h5ad"
BUNDLE = ROOT / "data" / "cosmx_frontal_cortex_hvg2000"


def main() -> None:
    reference_genes = set(
        pd.read_csv(REFERENCE, sep="\t", header=None, skiprows=1, usecols=[0], dtype=str)[0]
    )
    adata = ad.read_h5ad(SOURCE)
    shared = np.asarray([gene in reference_genes for gene in adata.var_names])
    adata = adata[:, shared].copy()
    sc.pp.highly_variable_genes(
        adata,
        layer="counts",
        flavor="seurat_v3",
        n_top_genes=2_000,
        subset=False,
        span=0.3,
    )
    selected = adata.var["highly_variable"].to_numpy()
    adata = adata[:, selected].copy()
    adata.X = adata.layers["counts"].copy()
    adata.uns["hvg_selection"] = {
        "method": "scanpy highly_variable_genes",
        "flavor": "seurat_v3",
        "n_top_genes": 2_000,
        "candidate_genes": int(shared.sum()),
        "candidate_rule": "CosMx biological genes intersected with GSE104276 reference genes",
        "counts_used": "raw counts layer",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(OUTPUT, compression="gzip")

    BUNDLE.mkdir(parents=True, exist_ok=True)
    counts = sparse.csr_matrix(adata.layers["counts"], dtype=np.float32)
    sparse.save_npz(BUNDLE / "counts.npz", counts, compressed=True)
    np.savetxt(BUNDLE / "genes.tsv", adata.var_names.to_numpy(), fmt="%s")
    np.savetxt(BUNDLE / "cells.tsv", adata.obs_names.to_numpy(), fmt="%s")
    shutil.copy2(ROOT / "data" / "cosmx_frontal_cortex" / "spatial.csv", BUNDLE / "spatial.csv")
    manifest = {
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "shared_gene_candidates": int(shared.sum()),
        "hvg_flavor": "seurat_v3",
        "source": str(SOURCE.resolve()),
        "h5ad": str(OUTPUT.resolve()),
    }
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
