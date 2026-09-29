"""Evaluate the full intrinsic embedding on a reproducible RCTD-labelled sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


def metrics(embedding: np.ndarray, labels: np.ndarray, clusters: int) -> dict[str, float]:
    predicted = MiniBatchKMeans(
        n_clusters=clusters, batch_size=4096, n_init=10, random_state=7
    ).fit_predict(embedding)
    neighbors = NearestNeighbors(n_neighbors=16).fit(embedding).kneighbors(return_distance=False)[:, 1:]
    rng = np.random.default_rng(7)
    score = rng.choice(len(labels), min(5000, len(labels)), replace=False)
    return {
        "adjusted_rand_index": float(adjusted_rand_score(labels, predicted)),
        "normalized_mutual_information": float(normalized_mutual_info_score(labels, predicted)),
        "knn15_label_purity": float(np.mean(labels[neighbors] == labels[:, None])),
        "label_silhouette_5000": float(silhouette_score(embedding[score], labels[score])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--sample-cells", type=int, default=30000)
    args = parser.parse_args()
    input_dir = Path(args.input)
    result_dir = Path(args.results)

    cells = pd.read_csv(result_dir / "cells.tsv", header=None, dtype=str)[0].to_numpy()
    labels = pd.read_csv(result_dir / "rctd_labels.csv", dtype={"cell_id": str})
    lookup = pd.Series(np.arange(len(cells)), index=cells)
    labels = labels[labels["cell_id"].isin(lookup.index)].copy()
    labels["row"] = lookup.loc[labels["cell_id"]].to_numpy()
    rng = np.random.default_rng(7)
    if len(labels) > args.sample_cells:
        labels = labels.iloc[np.sort(rng.choice(len(labels), args.sample_cells, replace=False))]
    rows = labels["row"].to_numpy()
    type_labels = labels["primary"].to_numpy()
    n_types = int(pd.Series(type_labels).nunique())

    zc = np.load(result_dir / "castle_embeddings_and_qc.npz")["z_intrinsic"][rows]
    raw = mmread(input_dir / "counts.mtx").T.tocsr()
    raw = raw[np.asarray(raw.sum(axis=1)).ravel() >= 5]
    raw = raw[:, np.asarray((raw > 0).sum(axis=0)).ravel() >= 10][rows].astype(np.float64)
    totals = np.asarray(raw.sum(axis=1)).ravel()
    raw = raw.multiply((1e4 / np.maximum(totals, 1.0))[:, None]).tocsr()
    raw.data = np.log1p(raw.data)
    raw_pca = TruncatedSVD(n_components=16, random_state=7).fit_transform(raw)

    comparison = {
        "common_labelled_cells_before_sampling": int(
            pd.read_csv(result_dir / "rctd_labels.csv", dtype={"cell_id": str})["cell_id"].isin(lookup.index).sum()
        ),
        "evaluated_sample_cells": int(len(labels)),
        "reference_types": n_types,
        "castle_intrinsic": metrics(zc, type_labels, n_types),
        "raw_log_normalized_pca": metrics(raw_pca, type_labels, n_types),
        "note": "RCTD labels are evaluation-only and were not used for CASTLE training.",
    }
    path = result_dir / "rctd_label_embedding_metrics.json"
    path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
