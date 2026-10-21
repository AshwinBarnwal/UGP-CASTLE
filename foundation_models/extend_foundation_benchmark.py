"""Extend the completed v1 CosMx sweep without recomputing old combinations.

The old grid was neighbors=(10,15,30,50), resolution=0.2..2.0. This computes
exactly the 32 absent combinations per method needed by protocol v2, merges
them with the old CSV, chooses the new optimum, and writes all v2 outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from benchmark_embeddings import (
    FIXED_NEIGHBORS, FIXED_RESOLUTION, NEIGHBORS, RESOLUTIONS,
    cluster_scores, graph_label_metrics, leiden_sweep, load_inputs, make_graph,
    published_label_silhouette, run_leiden, score_silhouette, write_plots,
)

OLD_NEIGHBORS = (10, 15, 30, 50)
OLD_RESOLUTIONS = tuple(np.round(np.arange(0.2, 2.01, 0.2), 1))


def key(neighbors, resolution):
    return int(neighbors), round(float(resolution), 6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--nicheformer", type=Path, required=True)
    parser.add_argument("--nicheformer-cells", type=Path, required=True)
    parser.add_argument("--uce", type=Path, required=True)
    parser.add_argument("--existing-dir", type=Path, required=True,
                        help="Completed v1 benchmark directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; v1 remains untouched")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-cells", type=int, default=30000)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    old = pd.read_csv(args.existing_dir / "leiden_parameter_sweep.csv")
    expected_old = {key(n, r) for n in OLD_NEIGHBORS for r in OLD_RESOLUTIONS}
    methods = sorted(old["method"].unique())
    for method in methods:
        got = {key(r.neighbors, r.resolution)
               for r in old.loc[old.method == method].itertuples()}
        if expected_old - got:
            raise RuntimeError(f"{method}: v1 sweep is missing {len(expected_old - got)} rows")

    common, truth, embeddings = load_inputs(args.source, args.nicheformer,
                                             args.nicheformer_cells, args.uce)
    if set(methods) != set(embeddings):
        raise RuntimeError(f"Methods differ: old={methods}, current={sorted(embeddings)}")

    target = {key(n, r) for n in NEIGHBORS for r in RESOLUTIONS}
    new_rows = []
    for method, values in embeddings.items():
        have = {key(r.neighbors, r.resolution)
                for r in old.loc[old.method == method].itertuples()}
        missing = target - have
        if len(missing) != 32:
            raise RuntimeError(f"{method}: expected exactly 32 new combinations, found {len(missing)}")
        for n in NEIGHBORS:
            missing_res = tuple(r for r in RESOLUTIONS if key(n, r) in missing)
            if missing_res:
                _, _, rows = leiden_sweep(values, truth, args.seed,
                                          neighbors=(n,), resolutions=missing_res)
                new_rows.extend({"method": method, **row} for row in rows)

    merged = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
    merged = merged.drop_duplicates(["method", "neighbors", "resolution"], keep="last")
    merged = merged.sort_values(["method", "neighbors", "resolution"]).reset_index(drop=True)
    merged.to_csv(args.output_dir / "leiden_parameter_sweep.csv", index=False)
    pd.DataFrame(new_rows).to_csv(args.output_dir / "new_32_combinations_per_method.csv", index=False)

    known_classes = pd.Series(truth[truth != "Unknown"]).nunique()
    metrics, direct, selected = [], [], {}
    optimized, fixed, kmeans = {}, {}, {}
    for method, values in embeddings.items():
        table = merged.loc[merged.method == method].copy()
        table["cluster_distance"] = (table.n_clusters - known_classes).abs()
        best = table.sort_values(["ARI", "NMI", "cluster_distance"],
                                 ascending=[False, False, True]).iloc[0]
        holder = make_graph(values, int(best.neighbors), args.seed)
        labels = run_leiden(holder, float(best.resolution), args.seed)
        optimized[method] = labels
        selected[method] = {"neighbors": int(best.neighbors),
                            "resolution": float(best.resolution)}
        metrics.append({"method": method, "clustering": "Leiden optimized",
                        "neighbors": int(best.neighbors), "resolution": float(best.resolution),
                        **cluster_scores(truth, labels),
                        "silhouette": score_silhouette(values, labels, args.seed),
                        "selection": "supervised: maximum ARI against non-Unknown published labels"})

        fixed_holder = make_graph(values, FIXED_NEIGHBORS, args.seed)
        fixed_labels = run_leiden(fixed_holder, FIXED_RESOLUTION, args.seed)
        fixed[method] = fixed_labels
        metrics.append({"method": method, "clustering": "Leiden fixed",
                        "neighbors": FIXED_NEIGHBORS, "resolution": FIXED_RESOLUTION,
                        **cluster_scores(truth, fixed_labels),
                        "silhouette": score_silhouette(values, fixed_labels, args.seed),
                        "selection": "common fixed readout; no per-method tuning"})

        km = MiniBatchKMeans(n_clusters=known_classes, random_state=args.seed,
                             batch_size=4096, n_init=10).fit_predict(values).astype(str)
        kmeans[method] = km; km_scores = cluster_scores(truth, km)
        metrics.append({"method": method, "clustering": "K-means", "neighbors": np.nan,
                        "resolution": np.nan, **km_scores,
                        "silhouette": score_silhouette(values, km, args.seed),
                        "selection": "K fixed to number of non-Unknown published classes"})
        direct.append({"method": method,
                       "published_label_silhouette": published_label_silhouette(values, truth, args.seed),
                       "neighbors_for_direct_metrics": FIXED_NEIGHBORS,
                       **graph_label_metrics(fixed_holder, truth)})

    pd.DataFrame(metrics).to_csv(args.output_dir / "ari_nmi_silhouette.csv", index=False)
    pd.DataFrame(direct).to_csv(args.output_dir / "embedding_direct_metrics.csv", index=False)
    write_plots(args.output_dir, embeddings, truth, optimized, fixed, kmeans,
                args.seed, args.plot_cells)
    (args.output_dir / "benchmark_manifest.json").write_text(json.dumps({
        "protocol": "foundation benchmark v2 incremental extension",
        "old_rows_reused": int(len(old)), "new_rows_computed": int(len(new_rows)),
        "new_combinations_per_method": 32, "common_cells": int(len(common)),
        "seed": args.seed, "leiden_neighbors": list(NEIGHBORS),
        "leiden_resolutions": list(RESOLUTIONS),
        "fixed_leiden": {"neighbors": FIXED_NEIGHBORS, "resolution": FIXED_RESOLUTION},
        "selected_optimized_parameters": selected,
        "unknown_excluded_from_label_based_metrics": True,
        "optimized_leiden_selection_is_supervised": True,
        "pca_applied_to_foundation_embeddings": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
