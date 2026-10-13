"""Train unsupervised scVI on the refined ProSeg breast-cancer H5AD."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "adata_for_tuto_s1.h5ad"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "results" / "scvi_xenium_proseg_breast"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train unsupervised scVI on raw counts from the refined ProSeg breast H5AD."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-key", default="sample", help="Use an empty string to disable batch correction.")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-latent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_input(adata: object, batch_key: str | None) -> str | None:
    if "counts" not in adata.layers:
        raise ValueError("H5AD must contain raw counts in layers['counts'].")
    if "cell_type" not in adata.obs:
        raise ValueError("H5AD must contain obs['cell_type'] for downstream evaluation.")
    if batch_key:
        if batch_key not in adata.obs:
            raise KeyError(f"Batch column {batch_key!r} is absent from obs.")
        if adata.obs[batch_key].isna().any():
            raise ValueError(f"Batch column {batch_key!r} contains missing values.")
        n_batches = int(adata.obs[batch_key].nunique())
        if n_batches < 2:
            print(f"Batch column {batch_key!r} has one category; training without batch correction.")
            return None
        print(f"Registering {batch_key!r} as batch_key with {n_batches} categories.")

    counts = adata.layers["counts"]
    values = counts.data if hasattr(counts, "data") and hasattr(counts, "indices") else np.asarray(counts)
    if values.size and (not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("layers['counts'] must contain finite, nonnegative values.")
    if counts.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(f"Counts layer has shape {counts.shape}, expected {(adata.n_obs, adata.n_vars)}.")
    return batch_key or None


def main() -> None:
    args = parse_args()
    if args.max_epochs < 1 or args.batch_size < 1 or args.n_latent < 1:
        raise ValueError("max-epochs, batch-size, and n-latent must all be positive.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input H5AD does not exist: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import scanpy as sc
    import torch
    import scvi

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    scvi.settings.seed = args.seed

    adata = sc.read_h5ad(args.input)
    batch_key = validate_input(adata, args.batch_key.strip() or None)
    print(f"Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes from {args.input}")
    print("Using raw layers['counts']; cell_type, spatial, and X_scANVI are not model inputs.")

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
    model = scvi.model.SCVI(
        adata,
        n_hidden=128,
        n_latent=args.n_latent,
        n_layers=1,
        dropout_rate=0.1,
        gene_likelihood="zinb",
    )
    model.train(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        train_size=0.9,
        validation_size=0.1,
        early_stopping=True,
        check_val_every_n_epoch=1,
    )

    adata.obsm["X_scVI"] = model.get_latent_representation(batch_size=args.batch_size, give_mean=True)
    metadata = {
        "input_h5ad": str(args.input.resolve()),
        "output_h5ad": str((args.output / "adata_scvi.h5ad").resolve()),
        "count_layer": "counts",
        "batch_key": batch_key,
        "labels_used_for_training": False,
        "label_column_reserved_for_evaluation": "cell_type",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_latent": args.n_latent,
        "n_hidden": 128,
        "n_layers": 1,
        "dropout_rate": 0.1,
        "gene_likelihood": "zinb",
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "train_fraction": 0.9,
        "validation_fraction": 0.1,
        "early_stopping": True,
        "seed": args.seed,
        "latent_key": "X_scVI",
    }
    adata.uns["scvi_run_metadata"] = metadata

    model.save(str(args.output / "model"), overwrite=True, save_anndata=False)
    history = model.history
    try:
        if isinstance(history, pd.DataFrame):
            history.to_csv(args.output / "training_history.csv")
        elif history is not None:
            pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    except (TypeError, ValueError) as exc:
        # Do not lose the latent/H5AD/UMAP outputs if scvi-tools changes the
        # shape/type of its in-memory history object between versions.
        note = (
            "scVI training finished and the model was saved, but training_history.csv "
            f"could not be serialized: {type(exc).__name__}: {exc}\n"
        )
        (args.output / "training_history_unavailable.txt").write_text(note, encoding="utf-8")
        print(note.strip())

    print("Computing UMAP from X_scVI for visualization...")
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=30, random_state=args.seed)
    sc.tl.umap(adata, random_state=args.seed)
    sc.pl.umap(adata, color="cell_type", show=False, title="scVI latent by cell type")
    import matplotlib.pyplot as plt

    plt.gcf().savefig(args.output / "umap_scvi_by_cell_type.png", dpi=220, bbox_inches="tight")
    plt.close("all")
    if batch_key:
        sc.pl.umap(adata, color=batch_key, show=False, title=f"scVI latent by {batch_key}")
        plt.gcf().savefig(args.output / "umap_scvi_by_batch.png", dpi=220, bbox_inches="tight")
        plt.close("all")

    adata.write_h5ad(args.output / "adata_scvi.h5ad", compression="gzip")
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model, X_scVI H5AD, history, and UMAPs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
