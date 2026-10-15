"""Fit resolVI (scvi-tools) on the decontamination bundles as a comparator to CASTLE.

Run with the ``.venv-scvi`` environment. resolVI models each cell's expected
counts as its own expression plus diffusion from its spatial neighbours plus
background. The own-source share of every count is

    pi_own = true_proportion * px_rate
             / (true_proportion * px_rate + sum_n v_n * px_rate_n + background)

and the purified counts are ``x * pi_own``, the same definition CASTLE exports,
so ``evaluate_decontamination.py`` scores both identically. resolVI has no
donor leakage fraction, so ``eta`` is written as NaN and the eta tests are
skipped for it.

The same fit also writes resolVI's documented corrected output to
``<results>_native``: the median of ``px_rate`` sampled from
``module.model_corrected`` (neighbour diffusion and background switched off),
as in the scvi-tools resolVI tutorial (``layers["generated_expression"]``).
These are regenerated expected counts, not a reweighting of the observed ones.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

SITES = [
    "true_mixture_proportion",
    "px_rate",
    "diffusion_proportion_per_neighbor",
    "px_rate_n",
    "background",
]


def load_bundle(bundle: Path) -> ad.AnnData:
    cells = pd.read_csv(bundle / "cells.tsv", sep="\t", header=None, dtype=str)[0].to_numpy()
    genes = pd.read_csv(bundle / "genes.tsv", sep="\t", header=None, dtype=str)[0].to_numpy()
    counts = mmread(bundle / "counts.mtx").tocsr()
    if counts.shape == (len(genes), len(cells)):
        counts = counts.T.tocsr()
    spatial = pd.read_csv(bundle / "spatial.csv", dtype={"cell_id": str}).set_index("cell_id").loc[cells]
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts, dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(cells, name="cell_id")),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    adata.obsm["X_spatial"] = spatial[["x", "y"]].to_numpy(dtype=np.float32)
    return adata


def own_responsibility(model, adata: ad.AnnData, num_samples: int, batch_size: int) -> np.ndarray:
    import scvi  # noqa: F401  (registers the model classes)

    responsibility = np.empty((adata.n_obs, adata.n_vars), dtype=np.float32)
    # One minibatch per call: scvi-tools' own minibatch concatenation fails for
    # resolVI's per-neighbour sites when the last batch is smaller.
    for start in range(0, adata.n_obs, batch_size):
        ids = np.arange(start, min(start + batch_size, adata.n_obs))
        posterior = model.sample_posterior(
            adata=adata,
            indices=ids,
            num_samples=num_samples,
            return_sites=SITES,
            batch_size=len(ids),
            summary_fun={"means": np.mean},
        )["means"]
        n = len(ids)
        true_prop = np.asarray(posterior["true_mixture_proportion"]).reshape(n, 1)
        own = true_prop * np.asarray(posterior["px_rate"]).reshape(n, adata.n_vars)
        v = np.asarray(posterior["diffusion_proportion_per_neighbor"]).reshape(n, -1, 1)
        rate_n = np.asarray(posterior["px_rate_n"]).reshape(n, -1, adata.n_vars)
        diffusion = (v * rate_n).sum(axis=1)
        background = np.asarray(posterior["background"]).reshape(n, adata.n_vars)
        responsibility[ids] = own / np.maximum(own + diffusion + background, 1e-12)
    return responsibility


def corrected_expression(model, adata: ad.AnnData, num_samples: int, batch_size: int) -> np.ndarray:
    """resolVI's documented corrected output (tutorial: layers['generated_expression'])."""
    generated = np.empty((adata.n_obs, adata.n_vars), dtype=np.float32)
    for start in range(0, adata.n_obs, batch_size):
        ids = np.arange(start, min(start + batch_size, adata.n_obs))
        samples = model.sample_posterior(
            adata=adata,
            indices=ids,
            model=model.module.model_corrected,
            return_sites=["px_rate"],
            summary_fun={"post_sample_q50": np.median},
            num_samples=num_samples,
            batch_size=len(ids),
        )
        generated[ids] = np.asarray(samples["post_sample_q50"]["px_rate"]).reshape(len(ids), adata.n_vars)
    return generated


def write_result(output: Path, adata: ad.AnnData, x: np.ndarray, purified: np.ndarray,
                 responsibility: np.ndarray, latent: np.ndarray, info: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output / "purified_counts_cells_by_genes.npz", sparse.csr_matrix(purified))
    sparse.save_npz(output / "own_responsibility_cells_by_genes.npz", sparse.csr_matrix(responsibility))
    np.savez_compressed(
        output / "castle_embeddings_and_qc.npz",
        z_intrinsic=latent,
        eta=np.full(adata.n_obs, np.nan, dtype=np.float32),
        segmentation_confidence=purified.sum(axis=1) / np.maximum(x.sum(axis=1), 1.0),
    )
    np.savetxt(output / "cells.tsv", adata.obs_names.to_numpy(), fmt="%s", delimiter="\t")
    np.savetxt(output / "genes.tsv", adata.var_names.to_numpy(), fmt="%s", delimiter="\t")
    info = {**info, "removed_fraction": float(1 - purified.sum() / x.sum())}
    (output / "run_metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def fit_variant(bundle: Path, output: Path, native_output: Path, args: argparse.Namespace) -> list[dict]:
    import scvi

    scvi.settings.seed = args.seed
    adata = load_bundle(bundle)
    scvi.external.RESOLVI.setup_anndata(adata, prepare_data_kwargs={"n_neighbors": args.n_neighbors})
    model_dir = output / "model"
    started = time.perf_counter()
    if (model_dir / "model.pt").exists():
        model = scvi.external.RESOLVI.load(str(model_dir), adata=adata)
    else:
        model = scvi.external.RESOLVI(adata)
        model.train(max_epochs=args.max_epochs, batch_size=args.batch_size)
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir), overwrite=True)
    seconds = time.perf_counter() - started

    x = adata.X.toarray()
    latent = model.get_latent_representation()
    base = {
        "variant": bundle.name, "train_seconds": round(seconds, 1), "max_epochs": args.max_epochs,
        "n_neighbors": args.n_neighbors, "scvi_tools": scvi.__version__,
    }
    responsibility = own_responsibility(model, adata, args.num_samples, args.batch_size)
    reweighted = write_result(
        output, adata, x, x * np.clip(responsibility, 0.0, 1.0), responsibility, latent,
        {**base, "output": "observed counts x own-source share"},
    )
    generated = corrected_expression(model, adata, args.native_samples, args.batch_size)
    native = write_result(
        native_output, adata, x, generated, generated / np.maximum(x, 1e-9), latent,
        {**base, "output": "model_corrected px_rate median (tutorial generated_expression)",
         "native_samples": args.native_samples},
    )
    return [reweighted, native]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, default=Path("data/castle_decontam_sim"))
    parser.add_argument("--results", type=Path, default=Path("results/decontamination/seed7/resolVI"))
    parser.add_argument("--variants", nargs="+", default=["null", "spill", "niche"])
    parser.add_argument("--max-epochs", type=int, default=50, help="resolVI default")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-neighbors", type=int, default=10, help="resolVI default")
    parser.add_argument("--num-samples", type=int, default=10, help="Posterior samples")
    parser.add_argument("--native-samples", type=int, default=3, help="Tutorial default")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    native_root = args.results.parent / f"{args.results.name}_native"
    for variant in args.variants:
        for info in fit_variant(
            (args.sim_root / variant).resolve(), (args.results / variant).resolve(),
            (native_root / variant).resolve(), args,
        ):
            print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
