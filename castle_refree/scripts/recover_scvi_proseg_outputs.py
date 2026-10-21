"""Recover post-training scVI artifacts from a saved model checkpoint.

This script does not fit or update the model. It loads the existing checkpoint,
recomputes the deterministic scVI latent from the original H5AD, then writes
the H5AD, UMAP, and run metadata that the interrupted training script did not
reach.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "adata_for_tuto_s1.h5ad"
DEFAULT_OUTPUT = ROOT / "results" / "scvi_xenium_proseg_breast"
DEFAULT_MODEL = DEFAULT_OUTPUT / "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover scVI outputs from an existing trained model without retraining."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input H5AD does not exist: {args.input}")
    if not args.model.is_dir() or not (args.model / "model.pt").is_file():
        raise FileNotFoundError(
            f"Saved scVI checkpoint was not found at {args.model}. "
            "This recovery requires the model saved before the failed history export."
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scanpy as sc
    import scvi

    args.output.mkdir(parents=True, exist_ok=True)
    scvi.settings.seed = args.seed
    adata = sc.read_h5ad(args.input)
    if "counts" not in adata.layers:
        raise ValueError("H5AD must contain raw counts in layers['counts'].")
    if "cell_type" not in adata.obs:
        raise ValueError("H5AD must contain obs['cell_type'] for the requested UMAP.")

    print(f"Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes from {args.input}")
    print(f"Loading saved checkpoint from {args.model}; no training will be run.")
    model = scvi.model.SCVI.load(str(args.model), adata=adata, accelerator="cpu")
    latent = model.get_latent_representation(batch_size=args.batch_size, give_mean=True)
    if latent.shape[0] != adata.n_obs or not np.isfinite(latent).all():
        raise ValueError(f"Loaded model returned an invalid latent matrix with shape {latent.shape}.")
    adata.obsm["X_scVI"] = latent

    metadata = {
        "input_h5ad": str(args.input.resolve()),
        "model_dir": str(args.model.resolve()),
        "output_h5ad": str((args.output / "adata_scvi.h5ad").resolve()),
        "count_layer": "counts",
        "batch_key": None,
        "batch_correction": False,
        "labels_used_for_training": False,
        "label_column_reserved_for_evaluation": "cell_type",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_latent": int(latent.shape[1]),
        "latent_key": "X_scVI",
        "latent_recomputed_from_saved_model": True,
        "training_rerun": False,
        "umap_neighbors": 30,
        "seed": args.seed,
    }
    adata.uns["scvi_run_metadata"] = metadata

    print("Computing UMAP from recovered X_scVI...")
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=30, random_state=args.seed)
    sc.tl.umap(adata, random_state=args.seed)
    sc.pl.umap(adata, color="cell_type", show=False, title="scVI latent by cell type")
    plt.gcf().savefig(args.output / "umap_scvi_by_cell_type.png", dpi=220, bbox_inches="tight")
    plt.close("all")

    h5ad_path = args.output / "adata_scvi.h5ad"
    print(f"Writing {h5ad_path} (this can take a few minutes)...")
    adata.write_h5ad(h5ad_path, compression="gzip")
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history_note = (
        "training_history.csv could not be recovered. The failed run saved the scVI model checkpoint, "
        "but scvi-tools model.save() does not persist the trainer's epoch history, and no history/log "
        "file was present in the output directory. The latent representation and downstream outputs "
        "were recomputed from the saved checkpoint without retraining.\n"
    )
    (args.output / "training_history_unavailable.txt").write_text(history_note, encoding="utf-8")
    print(f"Recovered H5AD, cell-type UMAP, metadata, and history note in {args.output.resolve()}")


if __name__ == "__main__":
    main()
