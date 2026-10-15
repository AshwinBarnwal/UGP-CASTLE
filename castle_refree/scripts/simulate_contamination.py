"""Write semi-synthetic variants of a crop bundle with known contamination or niche effects.

Three bundles are written next to each other, each readable by ``castle-refree``:

- ``null``:  the crop unchanged (reference fit).
- ``spill``: transcripts moved between *touching* cells. Donor j loses
  Binomial(x_jg, eta_j) of each gene, split over its contacts in proportion to the
  shared boundary length. ``eta_j`` depends on geometry only.
- ``niche``: a known neighbour-driven effect with no transcript movement. Selected
  genes are doubled (Poisson-thinned) in cells of one type that touch cells of
  another type. Genes are chosen to be low in the neighbouring type, so spillover
  cannot explain them; a correct model should leave them in place.

Ground truth (``truth_*.npz``) is saved beside each bundle and is never read by
training.

With ``--only-clean`` two fully synthetic bundles are written instead, for
measuring over-correction against an absolute truth:

- ``clean0``: the real geometry (positions, contacts, areas, total counts) with
  every cell's counts regenerated from its own label's pooled profile with
  Gamma-Poisson (NB) noise and no contamination. Correct removal is zero.
- ``cleanspill``: ``clean0`` plus the same geometric spillover as ``spill``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread, mmwrite

COPIED = (
    "genes.tsv", "cells.tsv", "spatial.csv", "geometry.csv", "edges_contact.csv",
    "controls.mtx", "control_features.tsv", "counts_nucleus.mtx", "crop_cells.csv",
)


def read_counts(bundle: Path, cells: np.ndarray, genes: np.ndarray) -> sparse.csr_matrix:
    matrix = mmread(bundle / "counts.mtx").tocsr()
    if matrix.shape == (len(genes), len(cells)):
        matrix = matrix.T.tocsr()
    return matrix.astype(np.int64)


def write_bundle(source: Path, target: Path, counts: sparse.csr_matrix, note: dict) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in COPIED:
        if (source / name).exists():
            shutil.copy(source / name, target / name)
    mmwrite(target / "counts.mtx", counts.T.tocoo().astype(np.float64))
    (target / "simulation.json").write_text(json.dumps(note, indent=2), encoding="utf-8")


def contact_arrays(
    edges: pd.DataFrame, index: dict[str, int], n_cells: int
) -> tuple[np.ndarray, np.ndarray]:
    """Padded per-cell contact lists (index -1 = none) and shared-boundary weights."""
    i = edges["cell_i"].map(index).to_numpy()
    j = edges["cell_j"].map(index).to_numpy()
    c = edges["shared_boundary_um"].to_numpy()
    src = np.concatenate([i, j])
    dst = np.concatenate([j, i])
    wt = np.concatenate([c, c])
    order = np.argsort(src, kind="stable")
    src, dst, wt = src[order], dst[order], wt[order]
    degree = np.bincount(src, minlength=n_cells)
    slots = np.zeros((n_cells, max(int(degree.max()), 1)), dtype=np.int64) - 1
    weights = np.zeros(slots.shape)
    position = np.arange(len(src)) - np.repeat(np.cumsum(degree) - degree, degree)
    slots[src, position] = dst
    weights[src, position] = wt
    return slots, weights


def simulate_spill(
    x: np.ndarray, slots: np.ndarray, weights: np.ndarray, geometry: pd.DataFrame,
    eta_mean: float, rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    has_contact = slots[:, 0] >= 0
    # Leakage rises with the share of a cell's outline that touches other cells,
    # relative to its size: a purely geometric rule, as CASTLE assumes.
    score = np.log1p(geometry["contact_boundary_um"].to_numpy()) - 0.5 * np.log(
        geometry["area_um2"].to_numpy()
    )
    score = (score - score[has_contact].mean()) / (score[has_contact].std() + 1e-9)
    eta = 1.0 / (1.0 + np.exp(-(score + np.log(eta_mean / (1 - eta_mean)))))
    eta = np.clip(eta, 0.0, 0.4) * has_contact

    leaked = rng.binomial(x, eta[:, None])
    received = np.zeros_like(x)
    w = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    remaining = leaked.copy()
    remaining_weight = np.ones(len(x))
    degree = (slots >= 0).sum(axis=1)
    for k in range(slots.shape[1]):
        target = slots[:, k]
        valid = target >= 0
        p = np.where(valid, w[:, k] / np.maximum(remaining_weight, 1e-12), 0.0)
        # The last contact takes whatever is left, so rounding cannot lose transcripts.
        p = np.where(k == degree - 1, 1.0, np.clip(p, 0.0, 1.0))
        sent = rng.binomial(remaining, p[:, None])
        sent[~valid] = 0
        np.add.at(received, target[valid], sent[valid])
        remaining -= sent
        remaining_weight -= np.where(valid, w[:, k], 0.0)
    if remaining.sum() != 0:
        raise RuntimeError("Spillover simulation did not conserve transcripts")
    return x - leaked + received, {"eta_true": eta, "lost": leaked, "received": received}


def simulate_niche(
    x: np.ndarray, slots: np.ndarray, coarse: np.ndarray, genes: np.ndarray,
    target_type: str, neighbour_type: str, n_genes: int, fold: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    neighbour_is = np.zeros(len(x), dtype=bool)
    neighbour_is[coarse == neighbour_type] = True
    touches = np.array(
        [bool(np.any(neighbour_is[s[s >= 0]])) for s in slots]
    )
    responders = (coarse == target_type) & touches
    norm = x / np.maximum(x.sum(axis=1, keepdims=True), 1)
    in_target = norm[coarse == target_type].mean(axis=0)
    in_neighbour = norm[coarse == neighbour_type].mean(axis=0)
    # Expressed by the responding type but nearly absent from the neighbour.
    eligible = (in_target > np.quantile(in_target, 0.5)) & (in_neighbour < 0.25 * in_target)
    candidates = np.flatnonzero(eligible)
    chosen = candidates[np.argsort(-in_target[candidates])[:n_genes]]
    added = np.zeros_like(x)
    added[np.ix_(responders, chosen)] = rng.poisson(
        (fold - 1.0) * x[np.ix_(responders, chosen)]
    )
    return x + added, {
        "added": added,
        "responders": responders,
        "niche_genes": chosen,
        "niche_gene_names": genes[chosen],
    }


def simulate_clean(
    x: np.ndarray, labels: np.ndarray, dispersion: float, rng: np.random.Generator
) -> np.ndarray:
    """Contamination-free counts: each cell drawn from its label's pooled profile.

    x_ig ~ Poisson(N_i * p_label(i),g * G_ig), G_ig ~ Gamma(theta, 1/theta), with N_i
    the cell's real total, so size, geometry and type layout stay realistic.
    """
    totals = x.sum(axis=1).astype(np.float64)
    clean = np.zeros_like(x)
    for label in np.unique(labels):
        rows = labels == label
        profile = x[rows].sum(axis=0).astype(np.float64)
        profile /= max(profile.sum(), 1.0)
        noise = rng.gamma(dispersion, 1.0 / dispersion, size=(rows.sum(), x.shape[1]))
        clean[rows] = rng.poisson(totals[rows, None] * profile[None, :] * noise)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/castle_xenium_breast_raw_crop"))
    parser.add_argument("--output-root", type=Path, default=Path("data/castle_decontam_sim"))
    parser.add_argument("--eta-mean", type=float, default=0.1, help="Mean injected leak fraction")
    parser.add_argument("--niche-target", default="Stromal")
    parser.add_argument("--niche-neighbour", default="Epithelial")
    parser.add_argument("--niche-genes", type=int, default=6)
    parser.add_argument("--niche-fold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--only-clean", action="store_true", help="Write clean0/cleanspill only")
    parser.add_argument("--labels", type=Path, default=Path("data/xenium_r1_supervised_labels.csv"))
    parser.add_argument("--clean-dispersion", type=float, default=4.0, help="Gamma shape (NB theta)")
    args = parser.parse_args()

    source = args.input_dir.resolve()
    root = args.output_root.resolve()
    rng = np.random.default_rng(args.seed)
    cells = pd.read_csv(source / "cells.tsv", sep="\t", header=None, dtype=str)[0].to_numpy()
    genes = pd.read_csv(source / "genes.tsv", sep="\t", header=None, dtype=str)[0].to_numpy()
    x = read_counts(source, cells, genes).toarray()
    index = {c: k for k, c in enumerate(cells)}
    geometry = pd.read_csv(source / "geometry.csv", dtype={"cell_id": str}).set_index("cell_id").loc[cells]
    edges = pd.read_csv(source / "edges_contact.csv", dtype={"cell_i": str, "cell_j": str})
    crop_cells = pd.read_csv(source / "crop_cells.csv", dtype={"cell_id": str}).set_index("cell_id").loc[cells]
    coarse = crop_cells["coarse_type"].to_numpy()
    slots, weights = contact_arrays(edges, index, len(cells))

    if args.only_clean:
        # Separate stream so the null/spill/niche bundles are unaffected.
        clean_rng = np.random.default_rng(args.seed + 1000)
        fine = (
            pd.read_csv(args.labels, encoding="utf-8-sig", dtype={"Barcode": str})
            .set_index("Barcode")["Cluster"].reindex(cells).fillna("Unlabeled").to_numpy()
        )
        clean = simulate_clean(x, fine, args.clean_dispersion, clean_rng)
        write_bundle(source, root / "clean0", sparse.csr_matrix(clean),
                     {"variant": "clean0", "dispersion": args.clean_dispersion, "seed": args.seed})
        spilled, truth = simulate_spill(clean, slots, weights, geometry, args.eta_mean, clean_rng)
        write_bundle(source, root / "cleanspill", sparse.csr_matrix(spilled),
                     {"variant": "cleanspill", "eta_mean": args.eta_mean, "seed": args.seed})
        np.savez_compressed(
            root / "truth_clean.npz", cells=cells, eta_true=truth["eta_true"],
            lost=truth["lost"].astype(np.int32), received=truth["received"].astype(np.int32),
        )
        summary = {
            "clean_total": int(clean.sum()), "real_total": int(x.sum()),
            "cleanspill_moved_fraction": float(truth["lost"].sum() / clean.sum()),
            "cleanspill_mean_eta_true": float(truth["eta_true"][slots[:, 0] >= 0].mean()),
        }
        (root / "summary_clean.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    write_bundle(source, root / "null", sparse.csr_matrix(x), {"variant": "null"})

    spilled, truth = simulate_spill(x, slots, weights, geometry, args.eta_mean, rng)
    write_bundle(source, root / "spill", sparse.csr_matrix(spilled),
                 {"variant": "spill", "eta_mean": args.eta_mean, "seed": args.seed})
    np.savez_compressed(
        root / "truth_spill.npz", cells=cells, eta_true=truth["eta_true"],
        lost=truth["lost"].astype(np.int32), received=truth["received"].astype(np.int32),
    )

    niche, ntruth = simulate_niche(
        x, slots, coarse, genes, args.niche_target, args.niche_neighbour,
        args.niche_genes, args.niche_fold, rng,
    )
    write_bundle(source, root / "niche", sparse.csr_matrix(niche),
                 {"variant": "niche", "target": args.niche_target,
                  "neighbour": args.niche_neighbour, "fold": args.niche_fold,
                  "genes": ntruth["niche_gene_names"].tolist(), "seed": args.seed})
    np.savez_compressed(root / "truth_niche.npz", cells=cells, **ntruth)

    summary = {
        "cells": len(cells),
        "spill_mean_eta_true": float(truth["eta_true"][slots[:, 0] >= 0].mean()),
        "spill_moved_fraction": float(truth["lost"].sum() / x.sum()),
        "niche_responders": int(ntruth["responders"].sum()),
        "niche_genes": ntruth["niche_gene_names"].tolist(),
        "niche_added_fraction_of_their_counts": float(
            ntruth["added"].sum() / max(x[np.ix_(ntruth["responders"], ntruth["niche_genes"])].sum(), 1)
        ),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
