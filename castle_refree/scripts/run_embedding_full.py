"""Full-slide CASTLE embedding experiments.

Decontamination: M6 + eta prior (lambda = 3). Identity-model settings: the
label-free winner of tune_embedding.py (results/embedding_tuning/best.json),
trained on all genes.

- ``i``:  identity model, no niche.
- ``ii``: identity model + contact-step niche model: neighbours described by
  their z_c (no gradient into it), z_c-only reconstruction and no niche-effect
  gradient into z_c, eta network and kernel frozen during the niche phase, niche
  loss on touching neighbours' purified profiles and mean z_c, group sparsity.

z_c is saved as ``z_c.npy`` for scoring with embedding_benchmark.py.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from castle_refree.config import load_config
from castle_refree.data import load_spatial_data
from castle_refree.trainer import CastleTrainer

SCRIPT_DIR = Path(__file__).resolve().parent

NICHE = {
    "train.joint_epochs": 40,
    "model.niche_encoder": "contact_steps",
    "model.niche_steps": 3,
    "model.niche_input": "zc",
    "loss.identity_recon_weight": 0.5,
    "loss.block_delta_grad": True,
    "loss.lambda_niche_zc": 1.0,
    "loss.niche_target": "purified",
    "loss.niche_group_weight": 1.0,
    "contamination.freeze_in_joint": True,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["i", "ii"])
    parser.add_argument("--bundle", type=Path, default=Path("data/castle_xenium_breast_raw"))
    parser.add_argument("--best", type=Path, default=Path("results/embedding_tuning/best.json"))
    parser.add_argument("--output-root", type=Path, default=Path("results/embedding_full"))
    parser.add_argument("--base-config", type=Path, default=Path("castle_refree/config/decontam_base.yaml"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("tune", SCRIPT_DIR / "tune_embedding.py")
    tune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tune)
    suite = tune.load_suite()

    best = json.loads(args.best.read_text(encoding="utf-8"))
    overrides = {
        "loss.kl_max": float(best["kl_max"]),
        "model.intrinsic_dim": int(best["latent_dims"]),
        "model.prior_components": int(best["mixture_components"]),
        "model.encoder_input": best["encoder_input"],
        "train.num_threads": args.threads,
    }
    if args.experiment == "ii":
        overrides.update(NICHE)
    base = load_config(args.base_config)
    base.train.seed = args.seed
    cfg = tune.identity_config(base, suite, overrides)
    out = args.output_root / f"castle_{args.experiment}"
    cfg = copy.deepcopy(cfg)
    cfg.data.input_dir = str(args.bundle.resolve())
    cfg.data.output_dir = str(out.resolve())
    cfg.data.min_counts, cfg.data.min_cells_per_gene = 5, 10

    data = load_spatial_data(cfg.data.input_dir, cfg.data.min_counts, cfg.data.min_cells_per_gene)
    started = time.perf_counter()
    trainer = CastleTrainer(data, cfg).fit()
    trainer.export()
    np.save(out / "z_c.npy", trainer.cache.zc)
    info = {"experiment": args.experiment, "cells": data.n_cells, "genes": data.n_genes,
            "settings": overrides, "hours": round((time.perf_counter() - started) / 3600, 2)}
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
