"""Transcript-level refinement of a fitted CASTLE version (spec Sec. 4.6).

The fitted model gives, for every cell i and gene g, the expected counts from
each source: own (1 - eta_i) l_i mu_ig, spillover from each touching donor j
(eta_j l_j w_ji mu_jg) and background. Each molecule's position then reweights
those sources:

    own:        exp(-r^2 / 2 sigma_own^2),   r = distance to the nucleus centre / cell radius
    donor j:    exp(-b^2 / 2 sigma_sp^2),    b = distance to cell j's outline (the shared border)
    background: uniform over the cell

Each density is normalised over the cell's polygon (Monte Carlo). A molecule's
posterior own-probability is summed back into counts. sigma_own and sigma_sp are
chosen by maximum marginal likelihood of molecule positions over a grid, on the
real-data fit, and reused for the other datasets.

Molecule positions exist only for real transcripts. For the ``spill`` dataset,
moved molecules are given positions inside the recipient within 2 um of the
donor's outline, and for ``niche`` the added molecules reuse positions of the
cell's own molecules. The spill placement follows the same assumption as the
refinement, so its spill scores are optimistic.

Outputs ``<version>_refined/<dataset>`` and appends a scored row to summary.csv.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.path import Path as Polygon
from scipy import sparse
from scipy.io import mmread

from castle_refree.config import load_config
from castle_refree.data import load_spatial_data
from castle_refree.trainer import CastleTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
SPILL_BAND_UM = 2.0


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_list(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep="\t", header=None, dtype=str)[0].to_numpy()


def load_polygons(path: Path, cells: np.ndarray) -> list[np.ndarray]:
    wanted = set(cells)
    frame = pd.read_csv(path, dtype={"cell_id": str})
    frame = frame[frame["cell_id"].isin(wanted)]
    groups = {k: g[["vertex_x", "vertex_y"]].to_numpy(np.float64) for k, g in frame.groupby("cell_id")}
    return [groups[c] for c in cells]


def segment_distance(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    a = polygon[:-1]
    d = polygon[1:] - a
    length = np.maximum((d**2).sum(axis=1), 1e-12)
    t = np.clip(((points[:, None, :] - a[None]) * d[None]).sum(axis=2) / length[None], 0.0, 1.0)
    closest = a[None] + t[..., None] * d[None]
    return np.sqrt(((points[:, None, :] - closest) ** 2).sum(axis=2)).min(axis=1)


def sample_inside(polygon: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    shape = Polygon(polygon)
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    points = []
    while sum(len(p) for p in points) < n:
        trial = rng.uniform(lo, hi, size=(4 * n, 2))
        points.append(trial[shape.contains_points(trial)])
        if len(points) > 20:
            break
    out = np.concatenate(points) if points else np.empty((0, 2))
    return out[:n] if len(out) >= n else np.vstack([out, np.repeat(polygon[:1], n - len(out), axis=0)])


class Molecules:
    """Per-cell molecule positions and genes for one dataset."""

    def __init__(self, cell: np.ndarray, gene: np.ndarray, xy: np.ndarray, n_cells: int) -> None:
        order = np.argsort(cell, kind="stable")
        self.cell, self.gene, self.xy = cell[order], gene[order], xy[order]
        counts = np.bincount(self.cell, minlength=n_cells)
        self.indptr = np.concatenate(([0], np.cumsum(counts)))

    def of(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.indptr[i], self.indptr[i + 1]
        return self.gene[lo:hi], self.xy[lo:hi]


def build_molecules(variant: str, base: dict, sim_root: Path, polygons, contacts, rng) -> Molecules:
    cell, gene, xy = base["cell"].astype(np.int64), base["gene"].astype(np.int64), np.column_stack([base["x"], base["y"]])
    n_cells = len(polygons)
    if variant == "null":
        return Molecules(cell, gene, xy, n_cells)
    if variant == "spill":
        truth = np.load(sim_root / "truth_spill.npz", allow_pickle=True)
        lost, received = truth["lost"], truth["received"]
        keep = np.ones(len(cell), dtype=bool)
        members_of = pd.Series(np.arange(len(cell))).groupby(cell * 1000 + gene).indices
        for i, g in zip(*np.nonzero(lost)):
            members = members_of.get(i * 1000 + g, np.empty(0, dtype=np.int64))
            drop = rng.choice(members, size=min(int(lost[i, g]), len(members)), replace=False)
            keep[drop] = False
        new_cell, new_gene, new_xy = [], [], []
        for i in np.nonzero(received.sum(axis=1))[0]:
            neighbours = contacts.get(i, [])
            if not neighbours:
                continue
            donors, weights = zip(*neighbours)
            weights = np.asarray(weights) / np.sum(weights)
            candidates = sample_inside(polygons[i], 200, rng)
            near = {j: candidates[segment_distance(candidates, polygons[j]) <= SPILL_BAND_UM] for j in donors}
            for g in np.nonzero(received[i])[0]:
                for _ in range(int(received[i, g])):
                    j = donors[rng.choice(len(donors), p=weights)]
                    pool = near[j] if len(near[j]) else candidates
                    new_cell.append(i)
                    new_gene.append(g)
                    new_xy.append(pool[rng.integers(len(pool))])
        cell = np.concatenate([cell[keep], np.asarray(new_cell, dtype=np.int64)])
        gene = np.concatenate([gene[keep], np.asarray(new_gene, dtype=np.int64)])
        xy = np.vstack([xy[keep], np.asarray(new_xy).reshape(-1, 2)])
        return Molecules(cell, gene, xy, n_cells)
    if variant == "niche":
        truth = np.load(sim_root / "truth_niche.npz", allow_pickle=True)
        added = truth["added"]
        mol = Molecules(cell, gene, xy, n_cells)
        new_cell, new_gene, new_xy = [], [], []
        for i, g in zip(*np.nonzero(added)):
            _, own_xy = mol.of(i)
            picks = own_xy[rng.integers(len(own_xy), size=int(added[i, g]))] if len(own_xy) else np.repeat(polygons[i][:1], int(added[i, g]), axis=0)
            new_cell.extend([i] * len(picks))
            new_gene.extend([g] * len(picks))
            new_xy.append(picks)
        return Molecules(
            np.concatenate([cell, np.asarray(new_cell, dtype=np.int64)]),
            np.concatenate([gene, np.asarray(new_gene, dtype=np.int64)]),
            np.vstack([xy, *new_xy]) if new_xy else xy, n_cells,
        )
    raise ValueError(variant)


def source_rates(trainer: CastleTrainer):
    """Own and background rates (cells x genes) and per-edge donor rates."""
    cache = trainer.refresh_cache()
    own = ((1.0 - cache.eta) * cache.library)[:, None] * cache.mu
    # trainer.background holds the uniform per-gene share nu_i / G per cell.
    share = trainer.background[:, :1].astype(np.float64)
    if hasattr(trainer.model, "background_logits"):
        beta = torch.softmax(trainer.model.background_logits.detach(), dim=0).cpu().numpy()
        background = share * own.shape[1] * beta[None]
    else:
        background = np.repeat(share, own.shape[1], axis=1)
    spill = trainer.spill
    edge_scale = cache.eta[spill.source] * cache.library[spill.source] * cache.edge_weights
    return own, background, spill, edge_scale, cache


def refine(trainer: CastleTrainer, molecules: Molecules, nucleus: np.ndarray, radius: np.ndarray,
           polygons, sigma_own: float, sigma_sp: float, cells: np.ndarray | None = None,
           rng: np.random.Generator | None = None) -> tuple[np.ndarray, float]:
    """Refined own counts (cells x genes) and the molecules' marginal log-likelihood."""
    rng = rng or np.random.default_rng(0)
    own, background, spill, edge_scale, cache = source_rates(trainer)
    n, g = own.shape
    refined = np.zeros((n, g), dtype=np.float32)
    loglik = 0.0
    for i in (range(n) if cells is None else cells):
        genes, xy = molecules.of(i)
        if len(genes) == 0:
            continue
        lo, hi = spill.incoming_indptr[i], spill.incoming_indptr[i + 1]
        edges = spill.incoming_order[lo:hi]
        donors = spill.source[edges]
        samples = sample_inside(polygons[i], 64, rng)
        area = abs(np.dot(polygons[i][:-1, 0], polygons[i][1:, 1]) - np.dot(polygons[i][:-1, 1], polygons[i][1:, 0])) / 2

        def own_kernel(points):
            r = np.hypot(*(points - nucleus[i]).T) / max(radius[i], 1e-6)
            return np.exp(-(r**2) / (2 * sigma_own**2))

        dens = [own_kernel(xy) / max(area * own_kernel(samples).mean(), 1e-12)]
        rates = [own[i, genes]]
        for e, j in zip(edges, donors):
            kernel = lambda p, j=j: np.exp(-(segment_distance(p, polygons[j]) ** 2) / (2 * sigma_sp**2))
            dens.append(kernel(xy) / max(area * kernel(samples).mean(), 1e-12))
            rates.append(edge_scale[e] * cache.mu[j, genes])
        dens.append(np.full(len(genes), 1.0 / max(area, 1e-12)))
        rates.append(background[i, genes])
        weighted = np.vstack(rates) * np.vstack(dens)  # sources x molecules
        total = np.maximum(weighted.sum(axis=0), 1e-30)
        loglik += float(np.log(total / np.maximum(np.vstack(rates).sum(axis=0), 1e-30)).sum())
        np.add.at(refined[i], genes, weighted[0] / total)
    return refined, loglik


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Fitted version to refine, e.g. S2_P3")
    parser.add_argument("--fits", type=Path, default=Path("results/decontamination/seed7"))
    parser.add_argument("--sim-root", type=Path, default=Path("data/castle_decontam_sim"))
    parser.add_argument("--crop", type=Path, default=Path("data/castle_xenium_breast_raw_crop"))
    parser.add_argument("--boundaries", type=Path, default=Path("../Xenium_breast_cancer/cell_boundaries.csv.gz"))
    parser.add_argument("--base-config", type=Path, default=Path("castle_refree/config/decontam_base.yaml"))
    parser.add_argument("--summary", type=Path, default=Path("results/decontamination/summary.csv"))
    parser.add_argument("--sigma-own", nargs="+", type=float, default=[0.5, 0.75, 1.0, 1.5])
    parser.add_argument("--sigma-sp", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--grid-cells", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    suite, evaluator = _load("run_decontamination_suite"), _load("evaluate_decontamination")
    base = load_config(args.base_config)
    base.train.seed = args.seed
    cfg = suite.apply_overrides(base, suite.VERSIONS[args.version])
    rng = np.random.default_rng(args.seed)

    cells = read_list(args.crop / "cells.tsv")
    molecule_table = dict(np.load(args.crop / "molecules.npz"))
    nucleus = np.column_stack([molecule_table["nucleus_x"], molecule_table["nucleus_y"]]).astype(np.float64)
    radius = molecule_table["cell_radius"].astype(np.float64)
    polygons = load_polygons(args.boundaries, cells)
    edges = pd.read_csv(args.crop / "edges_contact.csv", dtype={"cell_i": str, "cell_j": str})
    index = pd.Index(cells)
    contacts: dict[int, list[tuple[int, float]]] = {}
    for a, b, c in zip(index.get_indexer(edges["cell_i"]), index.get_indexer(edges["cell_j"]), edges["shared_boundary_um"]):
        contacts.setdefault(a, []).append((b, c))
        contacts.setdefault(b, []).append((a, c))

    def trainer_for(variant: str) -> CastleTrainer:
        run = copy.deepcopy(cfg)
        run.data.input_dir = str((args.sim_root / variant).resolve())
        run.data.output_dir = str((args.fits / f"{args.version}_refined" / variant).resolve())
        data = load_spatial_data(run.data.input_dir, run.data.min_counts, run.data.min_cells_per_gene)
        if not np.array_equal(data.cell_ids, cells):
            raise ValueError("Bundle cells differ from the crop molecule table")
        trainer = CastleTrainer(data, run)
        return trainer.load_checkpoint(args.fits / args.version / variant / "castle_refree_model.pt")

    # Choose sigmas on a subset of real-data cells by marginal likelihood.
    null_trainer = trainer_for("null")
    null_molecules = build_molecules("null", molecule_table, args.sim_root, polygons, contacts, rng)
    subset = rng.choice(len(cells), size=min(args.grid_cells, len(cells)), replace=False)
    grid = []
    for so in args.sigma_own:
        for ss in args.sigma_sp:
            _, ll = refine(null_trainer, null_molecules, nucleus, radius, polygons, so, ss, cells=subset)
            grid.append({"sigma_own": so, "sigma_sp": ss, "loglik": ll})
            print(json.dumps(grid[-1]), flush=True)
    best = max(grid, key=lambda row: row["loglik"])

    out_root = args.fits / f"{args.version}_refined"
    for variant in ("null", "spill", "niche"):
        trainer = null_trainer if variant == "null" else trainer_for(variant)
        molecules = null_molecules if variant == "null" else build_molecules(variant, molecule_table, args.sim_root, polygons, contacts, rng)
        counts = mmread(args.sim_root / variant / "counts.mtx").tocsr().T.toarray()
        check = np.zeros_like(counts)
        np.add.at(check, (molecules.cell, molecules.gene), 1)
        if not np.array_equal(check, counts):
            raise ValueError(f"{variant}: molecule table does not reproduce the count matrix")
        refined, _ = refine(trainer, molecules, nucleus, radius, polygons, best["sigma_own"], best["sigma_sp"])
        out = out_root / variant
        out.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(out / "purified_counts_cells_by_genes.npz", sparse.csr_matrix(refined))
        np.savez_compressed(out / "castle_embeddings_and_qc.npz", eta=trainer.cache.eta, z_intrinsic=trainer.cache.zc)
        np.savetxt(out / "cells.tsv", cells, fmt="%s", delimiter="\t")
        np.savetxt(out / "genes.tsv", trainer.data.gene_ids, fmt="%s", delimiter="\t")
        print(json.dumps({"variant": variant, "removed_fraction": float(1 - refined.sum() / counts.sum())}), flush=True)

    pd.DataFrame(grid).to_csv(out_root / "sigma_grid.csv", index=False)
    (out_root / "refinement.json").write_text(json.dumps({"best": best, "spill_band_um": SPILL_BAND_UM}, indent=2), encoding="utf-8")
    name = f"{args.version}_refined"
    row = evaluator.evaluate(out_root.resolve(), args.sim_root.resolve(), name, marker_genes=5)
    row.update(seed=args.seed, sigma_own=best["sigma_own"], sigma_sp=best["sigma_sp"])
    frame = pd.DataFrame([row])
    if args.summary.exists():
        previous = pd.read_csv(args.summary)
        keep = ~((previous["model"] == name) & (previous.get("seed", -1) == args.seed))
        frame = pd.concat([previous[keep], frame], ignore_index=True)
    frame.to_csv(args.summary, index=False)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
