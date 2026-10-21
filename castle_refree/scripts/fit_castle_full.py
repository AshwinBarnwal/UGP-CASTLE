"""Fit one CASTLE version from the decontamination suite on a full-slide bundle.

Uses the same version definitions as ``run_decontamination_suite.py`` but on a
real bundle (no simulated variants), with the loader's usual cell/gene filters.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

from castle_refree.config import load_config
from castle_refree.data import load_spatial_data
from castle_refree.trainer import CastleTrainer

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="P3_eta_prior")
    parser.add_argument("--input", type=Path, default=Path("data/castle_xenium_breast_raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=Path("castle_refree/config/decontam_base.yaml"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-counts", type=int, default=5)
    parser.add_argument("--min-cells-per-gene", type=int, default=10)
    parser.add_argument("--epochs", nargs=3, type=int, metavar=("WARMUP", "CONTAM", "JOINT"))
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("suite", SCRIPT_DIR / "run_decontamination_suite.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)

    base = load_config(args.base_config)
    base.train.seed = args.seed
    if args.epochs:
        base.train.warmup_epochs, base.train.contamination_epochs, base.train.joint_epochs = args.epochs
    cfg = suite.apply_overrides(base, suite.VERSIONS[args.version])
    cfg.data.input_dir = str(args.input.resolve())
    cfg.data.output_dir = str(args.output.resolve())
    cfg.data.min_counts = args.min_counts
    cfg.data.min_cells_per_gene = args.min_cells_per_gene

    data = load_spatial_data(cfg.data.input_dir, cfg.data.min_counts, cfg.data.min_cells_per_gene)
    started = time.perf_counter()
    CastleTrainer(data, cfg).fit().export()
    info = {"version": args.version, "cells": data.n_cells, "genes": data.n_genes,
            "seed": args.seed, "seconds": round(time.perf_counter() - started, 1)}
    (args.output / "fit_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
