"""Full-slide embedding benchmark: train scVI baselines and score embeddings.

Run with ``.venv-scviva`` (scvi-tools 1.5.1 + scanpy/igraph).

``scvi``: train scVI with the ProSeg settings (10 latent dims, 128 hidden, ZINB,
early stopping within 200 epochs) on either the bundle's raw counts or a CASTLE
fit's purified counts (rounded to integers), restricted to the CASTLE fit's
cells and genes so every method sees the same data.

``score``: score embeddings (``name=path.npy``, rows in the CASTLE fit's cell
order) and/or PCA of count matrices against the Janesick labels: labelled cells
only (Unlabeled excluded), the best Leiden ARI over resolutions 0.2-2.0 (with NMI
at that resolution).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


def read_list(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep="\t", header=None, dtype=str)[0].to_numpy()


def raw_counts(bundle: Path, cells: np.ndarray, genes: np.ndarray) -> sparse.csr_matrix:
    bundle_cells, bundle_genes = read_list(bundle / "cells.tsv"), read_list(bundle / "genes.tsv")
    matrix = mmread(bundle / "counts.mtx").tocsr()
    if matrix.shape == (len(bundle_genes), len(bundle_cells)):
        matrix = matrix.T.tocsr()
    rows = pd.Index(bundle_cells).get_indexer(cells)
    cols = pd.Index(bundle_genes).get_indexer(genes)
    return matrix[rows][:, cols].tocsr().astype(np.float32)


def labels_for(cells: np.ndarray, path: Path) -> np.ndarray:
    table = pd.read_csv(path, encoding="utf-8-sig", dtype={"Barcode": str}).set_index("Barcode")["Cluster"]
    return table.reindex(cells).fillna("Unlabeled").to_numpy()


def train_scvi(args: argparse.Namespace) -> None:
    import anndata as ad
    import scvi
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    scvi.settings.seed = args.seed
    cells = read_list(args.castle_fit / "cells.tsv")
    genes = read_list(args.castle_fit / "genes.tsv")
    if args.counts == "raw":
        counts = raw_counts(args.bundle, cells, genes)
    else:
        purified = sparse.load_npz(args.castle_fit / "purified_counts_cells_by_genes.npz")
        counts = sparse.csr_matrix(np.rint(purified.toarray()).astype(np.float32))
    keep = np.asarray(counts.sum(axis=1)).ravel() > 0  # scVI needs non-empty cells
    adata = ad.AnnData(X=counts[keep], obs=pd.DataFrame(index=cells[keep]), var=pd.DataFrame(index=genes))
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1, gene_likelihood="zinb")
    model.train(max_epochs=200, batch_size=512, train_size=0.9, validation_size=0.1,
                early_stopping=True, check_val_every_n_epoch=1)
    latent = np.full((len(cells), 10), np.nan, dtype=np.float32)
    latent[keep] = model.get_latent_representation(batch_size=1024, give_mean=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, latent)
    epochs = len(model.history["elbo_train"]) if "elbo_train" in model.history else None
    info = {"counts": args.counts, "cells": int(len(cells)), "cells_used": int(keep.sum()), "epochs": epochs, "seed": args.seed}
    args.output.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


def train_scvi_scviva(args: argparse.Namespace) -> None:
    """scVI (label-free), then scANVI from it and scVIVA (both use the labels).

    Settings follow the ProSeg runs; cells and genes are the CASTLE fit's.
    """
    import anndata as ad
    import scvi
    import torch
    from scviva.model._scviva import SCVIVA

    if (args.output / "scviva.npy").exists():
        print(f"{args.output / 'scviva.npy'} exists; skipping training", flush=True)
        return
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    scvi.settings.seed = args.seed
    cells = read_list(args.castle_fit / "cells.tsv")
    genes = read_list(args.castle_fit / "genes.tsv")
    counts = raw_counts(args.bundle, cells, genes)
    spatial = pd.read_csv(args.bundle / "spatial.csv", dtype={"cell_id": str}).set_index("cell_id").loc[cells]
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame({"cell_type": labels_for(cells, args.labels), "sample": "slide"}, index=cells),
        var=pd.DataFrame(index=genes),
    )
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["spatial"] = spatial[["x", "y"]].to_numpy(dtype=np.float32)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    adata.obs["sample"] = adata.obs["sample"].astype("category")
    args.output.mkdir(parents=True, exist_ok=True)

    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    vae = scvi.model.SCVI(adata, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1, gene_likelihood="zinb")
    vae.train(max_epochs=200, batch_size=512, train_size=0.9, validation_size=0.1,
              early_stopping=True, check_val_every_n_epoch=1)
    adata.obsm["X_scVI"] = vae.get_latent_representation(batch_size=1024, give_mean=True)
    np.save(args.output / "scvi.npy", adata.obsm["X_scVI"])

    scanvi = scvi.model.SCANVI.from_scvi_model(vae, labels_key="cell_type", unlabeled_category="Unlabeled")
    scanvi.train(max_epochs=20, n_samples_per_label=100, batch_size=512)
    adata.obsm["X_scANVI"] = scanvi.get_latent_representation(batch_size=1024)
    np.save(args.output / "scanvi.npy", adata.obsm["X_scANVI"])

    setup = {"sample_key": "sample", "labels_key": "cell_type",
             "cell_coordinates_key": "spatial", "expression_embedding_key": "X_scANVI"}
    SCVIVA.preprocessing_anndata(adata, k_nn=20, **setup)
    SCVIVA.setup_anndata(adata, layer="counts", batch_key="sample", **setup)
    model = SCVIVA(adata)
    model.train(max_epochs=600, early_stopping=True, check_val_every_n_epoch=1,
                batch_size=512, plan_kwargs={"lr": 5e-4})
    np.save(args.output / "scviva.npy", model.get_latent_representation())
    info = {"cells": int(len(cells)), "genes": int(len(genes)), "seed": args.seed,
            "label_informed": {"scvi": False, "scanvi": True, "scviva": True}}
    (args.output / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


def score_embedding(embedding: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    import anndata as ad
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    holder = ad.AnnData(np.zeros((len(embedding), 1), dtype=np.float32))
    holder.obsm["E"] = embedding.astype(np.float32)
    sc.pp.neighbors(holder, use_rep="E", n_neighbors=15, random_state=seed)
    best = (-1.0, 0.0, 0.0)
    for resolution in np.round(np.arange(0.2, 2.01, 0.2), 1):
        clusters = sc.tl.leiden(holder, resolution=resolution, random_state=seed, flavor="igraph",
                                n_iterations=2, directed=False, copy=True).obs["leiden"]
        ari = adjusted_rand_score(labels, clusters)
        if ari > best[0]:
            best = (ari, normalized_mutual_info_score(labels, clusters), float(resolution))
    return {
        "best_leiden_ari": float(best[0]),
        "leiden_nmi_at_best": float(best[1]),
        "best_resolution": best[2],
    }


def pca_of(counts: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    lognorm = np.log1p(1e4 * counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-9))
    return PCA(n_components=30, random_state=0).fit_transform(lognorm)


def score(args: argparse.Namespace) -> None:
    cells = read_list(args.castle_fit / "cells.tsv")
    genes = read_list(args.castle_fit / "genes.tsv")
    labels = labels_for(cells, args.labels)
    embeddings: dict[str, np.ndarray] = {}
    for item in args.embedding:
        name, path = item.split("=", 1)
        embeddings[name] = np.load(path)
    if args.pca_raw:
        embeddings["PCA raw counts"] = pca_of(raw_counts(args.bundle, cells, genes).toarray())
    for item in args.pca_purified:
        name, fit = item.split("=", 1)
        embeddings[name] = pca_of(sparse.load_npz(Path(fit) / "purified_counts_cells_by_genes.npz").toarray())
    finite = np.all([np.isfinite(e).all(axis=1) for e in embeddings.values()], axis=0)
    keep = finite & (labels != "Unlabeled")
    rows = []
    for name, embedding in embeddings.items():
        row = {"embedding": name, "cells": int(keep.sum()), **score_embedding(embedding[keep], labels[keep], args.seed)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    table = pd.DataFrame(rows)
    if args.output.exists():
        previous = pd.read_csv(args.output)
        table = pd.concat([previous[~previous["embedding"].isin(table["embedding"])], table], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--castle-fit", type=Path, required=True, help="CASTLE fit defining cells/genes")
    common.add_argument("--bundle", type=Path, default=Path("data/castle_xenium_breast_raw"))
    common.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("scvi", parents=[common])
    p.add_argument("--counts", choices=["raw", "purified"], required=True)
    p.add_argument("--output", type=Path, required=True, help=".npy for the latent")
    p.set_defaults(func=train_scvi)
    p = sub.add_parser("scviva", parents=[common])
    p.add_argument("--labels", type=Path, default=Path("data/xenium_r1_supervised_labels.csv"))
    p.add_argument("--output", type=Path, required=True, help="Directory for the latents")
    p.add_argument("--threads", type=int, default=0, help="Torch threads (0 = default)")
    p.set_defaults(func=train_scvi_scviva)
    p = sub.add_parser("score", parents=[common])
    p.add_argument("--labels", type=Path, default=Path("data/xenium_r1_supervised_labels.csv"))
    p.add_argument("--embedding", nargs="*", default=[], help="name=path.npy")
    p.add_argument("--pca-raw", action="store_true")
    p.add_argument("--pca-purified", nargs="*", default=[], help="name=castle_fit_dir")
    p.add_argument("--output", type=Path, default=Path("results/embedding_full/scores.csv"))
    p.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
