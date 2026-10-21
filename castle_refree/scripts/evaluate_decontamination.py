"""Score one CASTLE model version on the decontamination test suite.

Expects three CASTLE result directories fitted with identical settings on the
bundles written by ``simulate_contamination.py``:
``<results>/null``, ``<results>/spill`` and ``<results>/niche``.

Tests (all restricted to cells away from the crop border):

1. spill: does the model find the injected transfers? (eta and removal vs truth)
2. null: how much does it remove from real data? (over-correction guard)
3. niche: does it wrongly remove a neighbour-driven effect that is not transfer?
4. nucleus: do purified profiles move towards nucleus-only profiles?
5. off-type markers: do markers of other cell types drop, mostly in cells that
   touch a different type?

One row per model is appended to ``<summary>``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.stats import pearsonr, spearmanr


def read_list(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep="\t", header=None, dtype=str)[0].to_numpy()


def read_bundle_matrix(bundle: Path, name: str, cells: np.ndarray, genes: np.ndarray) -> np.ndarray:
    matrix = mmread(bundle / name).tocsr()
    if matrix.shape == (len(genes), len(cells)):
        matrix = matrix.T.tocsr()
    return matrix.toarray().astype(np.float64)


class Fit:
    """A CASTLE result aligned to the cells/genes it was trained on."""

    def __init__(self, result: Path, bundle: Path) -> None:
        self.cells = read_list(result / "cells.tsv")
        self.genes = read_list(result / "genes.tsv")
        self.purified = sparse.load_npz(result / "purified_counts_cells_by_genes.npz").toarray()
        qc = np.load(result / "castle_embeddings_and_qc.npz")
        self.eta = qc["eta"]
        bundle_cells = read_list(bundle / "cells.tsv")
        bundle_genes = read_list(bundle / "genes.tsv")
        raw = read_bundle_matrix(bundle, "counts.mtx", bundle_cells, bundle_genes)
        rows = pd.Index(bundle_cells).get_indexer(self.cells)
        cols = pd.Index(bundle_genes).get_indexer(self.genes)
        self.raw = raw[np.ix_(rows, cols)]
        self.removed = self.raw - self.purified


def align(values: np.ndarray, from_cells: np.ndarray, to_cells: np.ndarray) -> np.ndarray:
    return values[pd.Index(from_cells).get_indexer(to_cells)]


def lognorm(x: np.ndarray) -> np.ndarray:
    return np.log1p(1000.0 * x / np.maximum(x.sum(axis=1, keepdims=True), 1e-9))


def rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.maximum(
        np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1)), 1e-12
    )


def marker_sets(nucleus: np.ndarray, types: np.ndarray, genes: np.ndarray, per_type: int) -> dict[str, np.ndarray]:
    """Type-specific genes, defined on nucleus-only counts (the low-contamination reference)."""
    norm = nucleus / np.maximum(nucleus.sum(axis=1, keepdims=True), 1.0)
    labels = [t for t in np.unique(types) if t != "Other"]
    means = np.vstack([norm[types == t].mean(axis=0) for t in labels])
    markers = {}
    for k, t in enumerate(labels):
        other = np.delete(means, k, axis=0).max(axis=0)
        specificity = means[k] / np.maximum(other, 1e-6)
        ranked = np.argsort(-specificity)
        chosen = [g for g in ranked[:per_type] if specificity[g] >= 3.0]
        markers[t] = np.array(chosen, dtype=np.int64)
    return markers


def off_type_fraction(x: np.ndarray, types: np.ndarray, markers: dict[str, np.ndarray]) -> np.ndarray:
    total = np.maximum(x.sum(axis=1), 1e-9)
    out = np.full(len(x), np.nan)
    for t in markers:
        rows = types == t
        others = np.concatenate([m for u, m in markers.items() if u != t])
        out[rows] = x[np.ix_(rows, others)].sum(axis=1) / total[rows]
    return out


def evaluate(results: Path, sim_root: Path, name: str, marker_genes: int) -> dict[str, float | str]:
    null = Fit(results / "null", sim_root / "null")
    spill = Fit(results / "spill", sim_root / "spill")
    niche = Fit(results / "niche", sim_root / "niche")
    for fit in (spill, niche):
        if not (np.array_equal(fit.cells, null.cells) and np.array_equal(fit.genes, null.genes)):
            raise ValueError("null/spill/niche fits were trained on different cells or genes")
    cells, genes = null.cells, null.genes

    bundle_cells = read_list(sim_root / "null" / "cells.tsv")
    bundle_genes = read_list(sim_root / "null" / "genes.tsv")
    gene_cols = pd.Index(bundle_genes).get_indexer(genes)
    crop = pd.read_csv(sim_root / "null" / "crop_cells.csv", dtype={"cell_id": str}).set_index("cell_id").loc[cells]
    scored = crop["score"].to_numpy(dtype=bool)
    types = crop["coarse_type"].to_numpy()
    out: dict[str, float | str] = {"model": name, "cells": int(len(cells)), "scored_cells": int(scored.sum())}

    # 2. Over-correction guard on real data.
    frac_removed = null.removed.sum(axis=1) / np.maximum(null.raw.sum(axis=1), 1e-9)
    out["null_median_cell_removed"] = float(np.median(frac_removed[scored]))
    out["null_total_removed"] = float(null.removed[scored].sum() / null.raw[scored].sum())
    out["null_mean_eta"] = float(null.eta[scored].mean())

    # 1. Injected spillover.
    truth = np.load(sim_root / "truth_spill.npz", allow_pickle=True)
    eta_true = align(truth["eta_true"], truth["cells"], cells)
    received = align(truth["received"], truth["cells"], cells)[:, gene_cols].astype(np.float64)
    has_contact = eta_true > 0
    use = scored & has_contact
    extra_removed = spill.removed - null.removed
    out["spill_eta_spearman"] = float(spearmanr(spill.eta[use], eta_true[use])[0])
    out["spill_eta_delta_spearman"] = float(spearmanr(spill.eta[use] - null.eta[use], eta_true[use])[0])
    out["spill_injected_removed_fraction"] = float(extra_removed[scored].sum() / received[scored].sum())
    out["spill_cell_pearson"] = float(
        pearsonr(extra_removed[scored].sum(axis=1), received[scored].sum(axis=1))[0]
    )
    out["spill_entry_pearson"] = float(
        pearsonr(extra_removed[scored].ravel(), received[scored].ravel())[0]
    )

    # 3. Niche effect that must not be removed.
    ntruth = np.load(sim_root / "truth_niche.npz", allow_pickle=True)
    added = align(ntruth["added"], ntruth["cells"], cells)[:, gene_cols].astype(np.float64)
    responders = align(ntruth["responders"], ntruth["cells"], cells) & scored
    niche_cols = pd.Index(genes).get_indexer(ntruth["niche_gene_names"])
    niche_cols = niche_cols[niche_cols >= 0]
    block = np.ix_(responders, niche_cols)
    extra_niche = (niche.removed - null.removed)[block].sum()
    out["niche_extra_removed_fraction"] = float(extra_niche / max(added[block].sum(), 1e-9))
    out["niche_baseline_removed_rate"] = float(
        null.removed[block].sum() / max(null.raw[block].sum(), 1e-9)
    )

    # 4. Nucleus-only reference.
    nucleus = read_bundle_matrix(sim_root / "null", "counts_nucleus.mtx", bundle_cells, bundle_genes)
    nucleus = nucleus[np.ix_(pd.Index(bundle_cells).get_indexer(cells), gene_cols)]
    enough = scored & (nucleus.sum(axis=1) >= 10) & (null.raw.sum(axis=1) >= 20)
    r_raw = rowwise_pearson(lognorm(null.raw[enough]), lognorm(nucleus[enough]))
    r_pur = rowwise_pearson(lognorm(null.purified[enough]), lognorm(nucleus[enough]))
    out["nucleus_cells"] = int(enough.sum())
    out["nucleus_r_raw"] = float(r_raw.mean())
    out["nucleus_r_purified"] = float(r_pur.mean())
    out["nucleus_r_gain"] = float((r_pur - r_raw).mean())
    out["nucleus_fraction_cells_improved"] = float(np.mean(r_pur > r_raw))

    # 5. Off-type markers, split by whether a cell touches a different coarse type.
    labelled = scored & (types != "Other")
    markers = marker_sets(nucleus[labelled], types[labelled], genes, marker_genes)
    edges = pd.read_csv(sim_root / "null" / "edges_contact.csv", dtype={"cell_i": str, "cell_j": str})
    type_of = pd.Series(types, index=cells)
    ti = edges["cell_i"].map(type_of)
    tj = edges["cell_j"].map(type_of)
    hetero_edges = edges[(ti != tj) & ti.notna() & tj.notna() & (ti != "Other") & (tj != "Other")]
    hetero = pd.Index(cells).isin(pd.concat([hetero_edges["cell_i"], hetero_edges["cell_j"]]).unique())
    raw_off = off_type_fraction(null.raw, types, markers)
    pur_off = off_type_fraction(null.purified, types, markers)
    nuc_off = off_type_fraction(nucleus, types, markers)
    for label, mask in (("hetero", labelled & hetero), ("homo", labelled & ~hetero)):
        out[f"offtype_{label}_cells"] = int(mask.sum())
        out[f"offtype_{label}_raw"] = float(np.nanmean(raw_off[mask]))
        out[f"offtype_{label}_purified"] = float(np.nanmean(pur_off[mask]))
        out[f"offtype_{label}_nucleus"] = float(np.nanmean(nuc_off[mask]))
        out[f"offtype_{label}_reduction"] = float(1 - out[f"offtype_{label}_purified"] / out[f"offtype_{label}_raw"])
    out["marker_genes"] = json.dumps({t: genes[m].tolist() for t, m in markers.items()})
    return out


def evaluate_clean(results: Path, sim_root: Path) -> dict[str, float]:
    """Absolute over-correction tests on the contamination-free synthetic tissue.

    clean0: nothing to remove, so every removed count is over-correction.
    cleanspill: removal should match the known transfers, cell by cell.
    """
    clean = Fit(results / "clean0", sim_root / "clean0")
    spill = Fit(results / "cleanspill", sim_root / "cleanspill")
    crop = pd.read_csv(sim_root / "clean0" / "crop_cells.csv", dtype={"cell_id": str}).set_index("cell_id")
    scored = crop.loc[clean.cells, "score"].to_numpy(dtype=bool)
    truth = np.load(sim_root / "truth_clean.npz", allow_pickle=True)
    gene_cols = pd.Index(read_list(sim_root / "clean0" / "genes.tsv")).get_indexer(clean.genes)
    received = align(truth["received"], truth["cells"], clean.cells)[:, gene_cols].astype(np.float64)
    eta_true = align(truth["eta_true"], truth["cells"], clean.cells)

    out: dict[str, float] = {}
    clean_cell = clean.removed.sum(axis=1) / np.maximum(clean.raw.sum(axis=1), 1e-9)
    out["clean0_total_removed"] = float(clean.removed[scored].sum() / clean.raw[scored].sum())
    out["clean0_median_cell_removed"] = float(np.median(clean_cell[scored]))
    out["clean0_mean_eta"] = float(np.nanmean(clean.eta[scored]))

    removed = np.clip(spill.removed[scored], 0.0, None)
    true_rec = received[scored]
    out["cleanspill_true_moved_fraction"] = float(true_rec.sum() / spill.raw[scored].sum())
    out["cleanspill_total_removed"] = float(spill.removed[scored].sum() / spill.raw[scored].sum())
    # Removal beyond the clean0 baseline, as a share of what was truly received.
    out["cleanspill_recovery"] = float(
        (spill.removed[scored].sum() - clean.removed[scored].sum()) / true_rec.sum()
    )
    out["cleanspill_precision"] = float(np.minimum(removed, true_rec).sum() / max(removed.sum(), 1e-9))
    out["cleanspill_recall"] = float(np.minimum(removed, true_rec).sum() / max(true_rec.sum(), 1e-9))
    out["cleanspill_cell_pearson"] = float(pearsonr(removed.sum(axis=1), true_rec.sum(axis=1))[0])
    out["cleanspill_entry_pearson"] = float(pearsonr(removed.ravel(), true_rec.ravel())[0])
    contact = scored & (eta_true > 0)
    if np.isfinite(spill.eta[contact]).any():
        out["cleanspill_eta_spearman"] = float(spearmanr(spill.eta[contact], eta_true[contact])[0])
        out["cleanspill_mean_eta"] = float(np.mean(spill.eta[contact]))
    out["cleanspill_mean_eta_true"] = float(np.mean(eta_true[contact]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Directory with null/spill/niche fits")
    parser.add_argument("--sim-root", type=Path, default=Path("data/castle_decontam_sim"))
    parser.add_argument("--name", required=True, help="Model version label, e.g. M0")
    parser.add_argument("--summary", type=Path, default=Path("results/decontamination/summary.csv"))
    parser.add_argument("--marker-genes", type=int, default=5, help="Markers per coarse type")
    args = parser.parse_args()
    row = evaluate(args.results.resolve(), args.sim_root.resolve(), args.name, args.marker_genes)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if args.summary.exists():
        previous = pd.read_csv(args.summary)
        frame = pd.concat([previous[previous["model"] != args.name], frame], ignore_index=True)
    frame.to_csv(args.summary, index=False)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
