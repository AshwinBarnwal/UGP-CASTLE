"""Train scVIVA on the refined ProSeg breast-cancer H5AD from its tutorial setup."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "adata_for_tuto_s1.h5ad"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "results" / "scviva_xenium_proseg_breast"


def history_frame(history: object) -> pd.DataFrame:
    """Normalize scvi-tools history containers across package versions."""
    if isinstance(history, pd.DataFrame):
        return history.reset_index(drop=False)
    if hasattr(history, "items"):
        columns: dict[str, pd.Series] = {}
        for name, value in history.items():
            if isinstance(value, pd.DataFrame):
                series = value[name] if name in value else value.iloc[:, 0]
            elif isinstance(value, pd.Series):
                series = value
            else:
                array = np.asarray(value)
                array = array.reshape(1) if array.ndim == 0 else array.reshape(-1)
                series = pd.Series(array)
            columns[str(name)] = series.reset_index(drop=True)
        return pd.DataFrame(columns)
    return pd.DataFrame(np.asarray(history))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train scVIVA on the refined ProSeg breast H5AD using tutorial settings."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--k-nn", type=int, default=20)
    parser.add_argument(
        "--expression-embedding-key",
        default="X_scANVI",
        help="Existing expression embedding used by scVIVA (for example X_scVI).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_input(adata: object, expression_embedding_key: str) -> None:
    required_obs = {"sample", "cell_type"}
    missing_obs = required_obs - set(adata.obs.columns)
    missing_obsm = {"spatial", expression_embedding_key} - set(adata.obsm.keys())
    if missing_obs or missing_obsm or "counts" not in adata.layers:
        raise ValueError(
            f"H5AD must contain obs[sample, cell_type], obsm[spatial, {expression_embedding_key}], "
            f"and layers[counts]; missing obs={sorted(missing_obs)}, "
            f"missing obsm={sorted(missing_obsm)}, counts_layer={('counts' in adata.layers)}"
        )
    if adata.obsm["spatial"].shape[0] != adata.n_obs:
        raise ValueError("obsm['spatial'] rows do not match the observation count.")
    if adata.obsm[expression_embedding_key].shape[0] != adata.n_obs:
        raise ValueError(f"obsm[{expression_embedding_key!r}] rows do not match the observation count.")
    n_labels = adata.obs["cell_type"].nunique(dropna=True)
    if n_labels < 2:
        raise ValueError(f"scVIVA requires at least two cell_type values; found {n_labels}.")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input H5AD does not exist: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import scanpy as sc
    import torch
    import scvi
    from scviva.model._scviva import SCVIVA

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    scvi.settings.seed = args.seed

    adata = sc.read_h5ad(args.input)
    validate_input(adata, args.expression_embedding_key)
    setup_kwargs = {
        "sample_key": "sample",
        "labels_key": "cell_type",
        "cell_coordinates_key": "spatial",
        "expression_embedding_key": args.expression_embedding_key,
    }

    print(f"Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes from {args.input}")
    print("Preparing scVIVA neighborhood features (k_nn=20)...")
    SCVIVA.preprocessing_anndata(adata, k_nn=args.k_nn, **setup_kwargs)
    SCVIVA.setup_anndata(adata, layer="counts", batch_key="sample", **setup_kwargs)
    model = SCVIVA(adata)
    model.train(
        max_epochs=args.max_epochs,
        early_stopping=True,
        check_val_every_n_epoch=1,
        batch_size=args.batch_size,
        plan_kwargs={"lr": args.learning_rate},
    )

    adata.obsm["X_scVIVA"] = model.get_latent_representation()
    adata.uns["scviva_run_metadata"] = {
        "input_h5ad": str(args.input.resolve()),
        "sample_key": "sample",
        "labels_key": "cell_type",
        "cell_coordinates_key": "spatial",
        "expression_embedding_key": args.expression_embedding_key,
        "counts_layer": "counts",
        "k_nn": args.k_nn,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "cell_type_used_to_construct_scVIVA_neighborhood_features": True,
        "evaluation_note": (
            "ARI/NMI against cell_type are label-informed descriptive metrics; "
            "cell_type is also used in scVIVA neighborhood preprocessing."
        ),
    }

    model_path = args.output / "model"
    model.save(str(model_path), overwrite=True, save_anndata=False)
    history = getattr(model, "history", None)
    if history is not None:
        history_frame(history).to_csv(args.output / "training_history.csv", index=False)

    print("Computing UMAP from X_scVIVA for visualization...")
    sc.pp.neighbors(adata, use_rep="X_scVIVA", n_neighbors=30, random_state=args.seed)
    sc.tl.umap(adata, random_state=args.seed)
    sc.pl.umap(
        adata,
        color="cell_type",
        show=False,
        title="scVIVA embedding by cell type",
        save=False,
    )
    import matplotlib.pyplot as plt

    plt.gcf().savefig(args.output / "umap_scviva_by_cell_type.png", dpi=220, bbox_inches="tight")
    plt.close("all")
    adata.write_h5ad(args.output / "adata_scviva.h5ad", compression="gzip")
    metadata_path = args.output / "run_metadata.json"
    metadata_path.write_text(json.dumps(adata.uns["scviva_run_metadata"], indent=2), encoding="utf-8")
    print(f"Saved trained model, latent H5AD, history, and UMAP to {args.output.resolve()}")


if __name__ == "__main__":
    main()
