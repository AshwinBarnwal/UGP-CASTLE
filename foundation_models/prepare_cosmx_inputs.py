"""Validate a spatial raw-count H5AD and build model-input manifests.

The raw H5AD is already suitable for UCE: X is raw cells-by-genes counts and
var_names are gene symbols. Nicheformer instead uses a fixed Ensembl-ID gene
order. This script creates the symbol-to-vocabulary mapping without expanding
the 188k x 20,310 matrix on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def sha256(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def build_hgnc_lookup(table: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    approved: dict[str, str] = {}
    aliases: dict[str, set[str]] = {}
    for row in table.itertuples(index=False):
        symbol = str(getattr(row, "symbol", ""))
        ensembl = str(getattr(row, "ensembl_gene_id", ""))
        if not symbol or not ensembl or ensembl == "nan":
            continue
        ensembl = ensembl.split(".")[0]
        approved[symbol.upper()] = ensembl
        for field in ("alias_symbol", "prev_symbol"):
            value = getattr(row, field, None)
            if pd.isna(value):
                continue
            for alias in str(value).split("|"):
                alias = alias.strip().upper()
                if alias:
                    aliases.setdefault(alias, set()).add(ensembl)
    unique_aliases = {key: next(iter(values)) for key, values in aliases.items() if len(values) == 1}
    return approved, unique_aliases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--model-vocabulary", type=Path, required=True,
                        help="Nicheformer model.h5ad (1 x 20,310)")
    parser.add_argument("--hgnc", type=Path, required=True)
    parser.add_argument("--technology-mean", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--technology", choices=("CosMx", "Xenium"), default="CosMx")
    parser.add_argument("--smoke-cells", type=int, default=256)
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = ad.read_h5ad(args.adata, backed="r")
    vocabulary = ad.read_h5ad(args.model_vocabulary, backed="r")
    model_genes = pd.Index(vocabulary.var_names.astype(str))
    technology_mean = np.load(args.technology_mean)
    if len(technology_mean) != len(model_genes):
        raise ValueError(f"Technology mean has {len(technology_mean)} entries; vocabulary has {len(model_genes)}")
    if not source.obs_names.is_unique or not source.var_names.is_unique:
        raise ValueError("Input cell IDs and gene symbols must be unique")
    if "spatial" not in source.obsm:
        raise ValueError("Input is missing obsm['spatial']")
    if "cell_type" not in source.obs:
        raise ValueError("Input is missing obs['cell_type']")
    sample_x = source[: min(1024, source.n_obs), :].X
    sample = np.asarray(sample_x.toarray() if sparse.issparse(sample_x) else sample_x)
    if np.any(sample < 0) or not np.allclose(sample, np.rint(sample)):
        raise ValueError("UCE and Nicheformer require non-negative raw counts in .X")

    hgnc = pd.read_csv(args.hgnc, sep="\t", low_memory=False)
    approved, aliases = build_hgnc_lookup(hgnc)
    model_position = {gene.split(".")[0]: i for i, gene in enumerate(model_genes)}
    seen_targets: set[str] = set()
    rows: list[dict[str, object]] = []
    for source_index, gene in enumerate(source.var_names.astype(str)):
        lookup = gene.upper()
        ensembl = approved.get(lookup)
        match_kind = "approved_symbol"
        if ensembl is None:
            ensembl = aliases.get(lookup)
            match_kind = "unique_alias"
        target_index = model_position.get(ensembl) if ensembl else None
        status = "matched"
        if ensembl is None:
            status = "no_hgnc_mapping"
        elif target_index is None:
            status = "not_in_nicheformer_vocabulary"
        elif ensembl in seen_targets:
            status = "duplicate_target_dropped"
        else:
            seen_targets.add(ensembl)
        rows.append({
            "source_index": source_index,
            "input_gene_symbol": gene,
            "ensembl_gene_id": ensembl or "",
            "model_index": target_index if target_index is not None else "",
            "match_kind": match_kind if ensembl else "",
            "status": status,
        })
    mapping = pd.DataFrame(rows)
    mapping.to_csv(args.output_dir / "nicheformer_gene_mapping.tsv", sep="\t", index=False)

    matched = mapping["status"].eq("matched")
    technology_token = {"CosMx": 8, "Xenium": 9}[args.technology]
    manifest = {
        "input_h5ad": str(args.adata.resolve()),
        "input_sha256": None if args.skip_sha256 else sha256(args.adata),
        "cells": int(source.n_obs),
        "input_genes": int(source.n_vars),
        "x_dtype": str(source.X.dtype),
        "raw_nonnegative_integer_counts": True,
        "cell_id_unique": bool(source.obs_names.is_unique),
        "gene_symbol_unique": bool(source.var_names.is_unique),
        "spatial_coordinates": list(source.obsm["spatial"].shape),
        "published_label_column": "cell_type",
        "published_label_counts": source.obs["cell_type"].astype(str).value_counts().to_dict(),
        "uce": {
            "input": "the original H5AD; UCE filters to human genes with protein embeddings",
            "gene_selection": "none (all raw panel genes)",
        },
        "nicheformer": {
            "model_vocabulary_genes": int(len(model_genes)),
            "matched_input_genes": int(matched.sum()),
            "unmatched_or_dropped_input_genes": int((~matched).sum()),
            "mapping_file": "nicheformer_gene_mapping.tsv",
            "species_token": 5,
            "technology": args.technology,
            "technology_token": technology_token,
            "modality_token_spatial": 4,
            "technology_mean": str(args.technology_mean.resolve()),
        },
    }
    (args.output_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    smoke_n = min(args.smoke_cells, source.n_obs)
    # The source uses pandas' nullable string dtype. AnnData 0.11 requires an
    # explicit opt-in before serializing it, even though the same release can
    # read the resulting H5AD.
    ad.settings.allow_write_nullable_strings = True
    smoke = source[:smoke_n, :].to_memory()
    prefix = args.technology.lower()
    smoke.write_h5ad(args.output_dir / f"{prefix}_smoke_{smoke_n}.h5ad", compression="gzip")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

