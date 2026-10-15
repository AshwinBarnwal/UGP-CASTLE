"""Train scVI, scANVI and scVIVA on the decontamination crop and score their latents.

Run with ``.venv-scviva`` (scvi-tools 1.5.1 + scviva). Settings mirror the ProSeg
runs: scVI as in run_scvi_proseg.py (10 latent dims, 128 hidden, ZINB, early
stopping within 200 epochs); scVIVA as in run_scviva_proseg.py (k_nn = 20,
600 epochs with early stopping, lr 5e-4, batch 512). The crop has no precomputed
X_scANVI, so scANVI is trained from the scVI model with the Janesick labels to
provide scVIVA's expression embedding, as in the scVIVA tutorial.

scANVI and scVIVA use the same labels they are scored against, so their ARIs are
label-informed (as in the ProSeg evaluation); scVI is label-free.

Scoring uses the protocol of the CASTLE comparison: scored crop cells with a
label, K-means (K = number of classes, 3 seeds) and the best Leiden ARI over
resolutions 0.2-2.0.
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


def load_crop(bundle: Path, labels_path: Path):
    import anndata as ad

    cells = pd.read_csv(bundle / "cells.tsv", sep="\t", header=None, dtype=str)[0].to_numpy()
    genes = pd.read_csv(bundle / "genes.tsv", sep="\t", header=None, dtype=str)[0].to_numpy()
    counts = mmread(bundle / "counts.mtx").tocsr()
    if counts.shape == (len(genes), len(cells)):
        counts = counts.T.tocsr()
    spatial = pd.read_csv(bundle / "spatial.csv", dtype={"cell_id": str}).set_index("cell_id").loc[cells]
    crop = pd.read_csv(bundle / "crop_cells.csv", dtype={"cell_id": str}).set_index("cell_id").loc[cells]
    labels = (
        pd.read_csv(labels_path, encoding="utf-8-sig", dtype={"Barcode": str})
        .set_index("Barcode")["Cluster"].reindex(cells).fillna("Unlabeled")
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts, dtype=np.float32),
        obs=pd.DataFrame(
            {"cell_type": labels.to_numpy(), "sample": "crop", "score": crop["score"].to_numpy()},
            index=pd.Index(cells, name="cell_id"),
        ),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["spatial"] = spatial[["x", "y"]].to_numpy(dtype=np.float32)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    adata.obs["sample"] = adata.obs["sample"].astype("category")
    return adata


def score(embedding: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    import anndata as ad
    import scanpy as sc
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    k = len(np.unique(labels))
    kmeans = [
        KMeans(n_clusters=k, n_init=3, random_state=s).fit_predict(embedding) for s in range(3)
    ]
    holder = ad.AnnData(np.zeros((len(embedding), 1), dtype=np.float32))
    holder.obsm["E"] = embedding.astype(np.float32)
    sc.pp.neighbors(holder, use_rep="E", n_neighbors=15, random_state=seed)
    best = (-1.0, 0.0, 0.0)
    for resolution in np.round(np.arange(0.2, 2.01, 0.2), 1):
        clusters = sc.tl.leiden(
            holder, resolution=resolution, random_state=seed, flavor="igraph",
            n_iterations=2, directed=False, copy=True,
        ).obs["leiden"]
        ari = adjusted_rand_score(labels, clusters)
        if ari > best[0]:
            best = (ari, normalized_mutual_info_score(labels, clusters), float(resolution))
    return {
        "kmeans_ari": float(np.mean([adjusted_rand_score(labels, p) for p in kmeans])),
        "kmeans_nmi": float(np.mean([normalized_mutual_info_score(labels, p) for p in kmeans])),
        "best_leiden_ari": float(best[0]),
        "leiden_nmi_at_best_ari": float(best[1]),
        "best_leiden_resolution": best[2],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("data/castle_decontam_sim/null"))
    parser.add_argument("--labels", type=Path, default=Path("data/xenium_r1_supervised_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/decontamination/crop_embeddings"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import scvi
    import torch
    from scviva.model._scviva import SCVIVA

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    scvi.settings.seed = args.seed
    args.output.mkdir(parents=True, exist_ok=True)
    adata = load_crop(args.bundle, args.labels)

    # scVI: identical to run_scvi_proseg.py (single sample, so no batch key).
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    vae = scvi.model.SCVI(adata, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1, gene_likelihood="zinb")
    vae.train(max_epochs=200, batch_size=512, train_size=0.9, validation_size=0.1,
              early_stopping=True, check_val_every_n_epoch=1)
    adata.obsm["X_scVI"] = vae.get_latent_representation(batch_size=512, give_mean=True)

    # scANVI from the scVI model, as the scVIVA tutorial's expression embedding.
    scanvi = scvi.model.SCANVI.from_scvi_model(vae, labels_key="cell_type", unlabeled_category="Unlabeled")
    scanvi.train(max_epochs=20, n_samples_per_label=100, batch_size=512)
    adata.obsm["X_scANVI"] = scanvi.get_latent_representation(batch_size=512)

    # scVIVA: identical settings to run_scviva_proseg.py.
    setup = {
        "sample_key": "sample", "labels_key": "cell_type",
        "cell_coordinates_key": "spatial", "expression_embedding_key": "X_scANVI",
    }
    SCVIVA.preprocessing_anndata(adata, k_nn=20, **setup)
    SCVIVA.setup_anndata(adata, layer="counts", batch_key="sample", **setup)
    scviva_model = SCVIVA(adata)
    scviva_model.train(max_epochs=600, early_stopping=True, check_val_every_n_epoch=1,
                       batch_size=512, plan_kwargs={"lr": 5e-4})
    adata.obsm["X_scVIVA"] = scviva_model.get_latent_representation()

    keep = adata.obs["score"].to_numpy(dtype=bool) & (adata.obs["cell_type"].astype(str) != "Unlabeled").to_numpy()
    labels = adata.obs["cell_type"].astype(str).to_numpy()[keep]
    rows = []
    for key, label_informed in (("X_scVI", False), ("X_scANVI", True), ("X_scVIVA", True)):
        row = {"embedding": key, "label_informed": label_informed, "cells": int(keep.sum()),
               **score(adata.obsm[key][keep], labels, args.seed)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        np.save(args.output / f"{key}.npy", adata.obsm[key])
    pd.DataFrame(rows).to_csv(args.output / "embedding_scores.csv", index=False)
    pd.Series(adata.obs_names).to_csv(args.output / "cells.tsv", sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()
