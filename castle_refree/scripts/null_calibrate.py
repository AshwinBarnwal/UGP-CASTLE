"""Null-calibrate a fitted CASTLE version's leakage fractions and score the result.

For each bundle the version was fitted on:

1. Reload the fitted model and its decoded own-expression profiles mu_i.
2. Write a contamination-free "null twin" of the bundle: identical cells,
   positions, contacts and geometry, with counts x_i ~ NB(T_i * mu_i, phi), where
   T_i is the cell's observed total and phi the fitted gene dispersion.
3. Fit the same version on the twin. Its leakage fractions eta_0 are pure bias,
   since the twin contains no spillover.
4. Calibrate: eta_cal = clip(eta_hat - eta_0, 0, eta_max), then re-export the
   real fit's purified counts with eta fixed to eta_cal (no retraining).

The calibrated outputs are written as ``<version>_nullcal`` and scored with the
same tests as every other version (summary.csv).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.io import mmwrite
from torch.nn import functional as F

from castle_refree.config import load_config
from castle_refree.data import load_spatial_data
from castle_refree.trainer import CastleTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
AUXILIARY = (
    "genes.tsv", "cells.tsv", "spatial.csv", "geometry.csv", "edges_contact.csv",
    "controls.mtx", "control_features.tsv", "counts_nucleus.mtx", "crop_cells.csv",
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trainer_for(cfg, bundle: Path, output: Path) -> CastleTrainer:
    cfg = copy.deepcopy(cfg)
    cfg.data.input_dir = str(bundle.resolve())
    cfg.data.output_dir = str(output.resolve())
    data = load_spatial_data(cfg.data.input_dir, cfg.data.min_counts, cfg.data.min_cells_per_gene)
    return CastleTrainer(data, cfg)


def write_twin(trainer: CastleTrainer, bundle: Path, twin: Path, seed: int) -> dict[str, float]:
    cache = trainer.refresh_cache()
    totals = np.asarray(trainer.data.counts.sum(axis=1)).ravel()
    with torch.no_grad():
        theta = (F.softplus(trainer.model.log_dispersion) + 1e-4).cpu().numpy()
    rng = np.random.default_rng(seed)
    rate = totals[:, None] * cache.mu
    gamma = rng.gamma(theta[None, :], 1.0 / theta[None, :], size=rate.shape)
    counts = rng.poisson(rate * gamma).astype(np.float64)
    twin.mkdir(parents=True, exist_ok=True)
    for name in AUXILIARY:
        if (bundle / name).exists():
            shutil.copy(bundle / name, twin / name)
    # The loader keeps trainer cells/genes; the bundle files list the same order
    # because the decontamination configs disable filtering.
    mmwrite(twin / "counts.mtx", sparse.csr_matrix(counts).T.tocoo())
    info = {"real_total": float(totals.sum()), "twin_total": float(counts.sum()), "seed": seed}
    (twin / "simulation.json").write_text(json.dumps({"variant": "null_twin", **info}, indent=2), encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="M6")
    parser.add_argument("--fits", type=Path, default=Path("results/decontamination/seed7"))
    parser.add_argument("--sim-root", type=Path, default=Path("data/castle_decontam_sim"))
    parser.add_argument("--twin-root", type=Path, default=Path("data/castle_decontam_twins"))
    parser.add_argument("--base-config", type=Path, default=Path("castle_refree/config/decontam_base.yaml"))
    parser.add_argument("--variants", nargs="+", default=["null", "spill", "niche", "clean0", "cleanspill"])
    parser.add_argument("--summary", type=Path, default=Path("results/decontamination/summary.csv"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    suite = _load("run_decontamination_suite")
    evaluator = _load("evaluate_decontamination")
    base = load_config(args.base_config)
    base.train.seed = args.seed
    cfg = suite.apply_overrides(base, suite.VERSIONS[args.version])
    calibrated_name = f"{args.version}_nullcal"
    diagnostics = []

    for variant in args.variants:
        bundle = args.sim_root / variant
        real_dir = args.fits / args.version / variant
        twin_bundle = args.twin_root / args.version / variant
        twin_fit = args.fits / f"{args.version}_twin" / variant
        out_dir = args.fits / calibrated_name / variant

        real = trainer_for(cfg, bundle, real_dir).load_checkpoint(real_dir / "castle_refree_model.pt")
        twin_info = write_twin(real, bundle, twin_bundle, seed=args.seed + 100)

        if not (twin_fit / "castle_embeddings_and_qc.npz").exists():
            trainer_for(cfg, twin_bundle, twin_fit).fit().export()
        eta_hat = np.load(real_dir / "castle_embeddings_and_qc.npz")["eta"]
        eta_null = np.load(twin_fit / "castle_embeddings_and_qc.npz")["eta"]
        eta_cal = np.clip(eta_hat - eta_null, 0.0, cfg.model.contamination.eta_max)

        real.cfg.data.output_dir = str(out_dir.resolve())
        real.eta_override = eta_cal.astype(np.float32)
        real.refresh_cache()
        real.export()
        row = {
            "variant": variant, **twin_info,
            "mean_eta_hat": float(eta_hat.mean()), "mean_eta_null": float(eta_null.mean()),
            "mean_eta_calibrated": float(eta_cal.mean()),
            "share_cells_eta_zero": float(np.mean(eta_cal == 0)),
        }
        diagnostics.append(row)
        print(json.dumps(row), flush=True)

    pd.DataFrame(diagnostics).to_csv(args.fits / calibrated_name / "calibration.csv", index=False)
    run_root = (args.fits / calibrated_name).resolve()
    if all((run_root / v).exists() for v in ("null", "spill", "niche")):
        row = evaluator.evaluate(run_root, args.sim_root.resolve(), calibrated_name, marker_genes=5)
        if all((run_root / v).exists() for v in ("clean0", "cleanspill")):
            row.update(evaluator.evaluate_clean(run_root, args.sim_root.resolve()))
        row["seed"] = args.seed
        frame = pd.DataFrame([row])
        if args.summary.exists():
            previous = pd.read_csv(args.summary)
            keep = ~((previous["model"] == calibrated_name) & (previous.get("seed", -1) == args.seed))
            frame = pd.concat([previous[keep], frame], ignore_index=True)
        frame.to_csv(args.summary, index=False)
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
