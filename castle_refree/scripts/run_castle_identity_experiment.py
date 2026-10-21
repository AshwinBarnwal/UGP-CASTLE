"""Run recoverable CASTLE identity experiment i or ii on a prepared bundle.

Experiment ii trains the niche branch, but exports/evaluates only z_c. This
obeys the project rule against presenting CASTLE niche embeddings.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from castle_refree.config import load_config
from castle_refree.data import load_spatial_data
from castle_refree.trainer import CastleTrainer

ROOT = Path(__file__).resolve().parents[2]
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("i", "ii"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--cosmx-adaptations", action="store_true",
                        help="Use the same kNN/basic/auto decontamination adaptations as CosMx experiment i")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    tune = load_module("tune_embedding", SCRIPT_DIR / "tune_embedding.py")
    suite = tune.load_suite()
    base = load_config(ROOT / "castle_refree" / "config" / "decontam_base.yaml")
    base.train.seed = args.seed
    overrides = {
        "loss.kl_max": 0.5, "model.intrinsic_dim": 16,
        "model.prior_components": 30, "model.encoder_input": "dual",
        "train.num_threads": args.threads,
    }
    if args.experiment == "ii":
        overrides.update(NICHE)
    cfg = copy.deepcopy(tune.identity_config(base, suite, overrides))
    cfg.data.input_dir = str(args.bundle.resolve())
    cfg.data.output_dir = str(args.output.resolve())
    cfg.data.min_counts, cfg.data.min_cells_per_gene = 5, 10
    cfg.train.device = args.device
    cfg.train.checkpoint_every = 5
    if args.cosmx_adaptations:
        cfg.model.contamination.spillover_graph = "knn"
        cfg.model.contamination.geometry_features = "basic"
        cfg.model.contamination.background = "auto"

    data = load_spatial_data(cfg.data.input_dir, cfg.data.min_counts, cfg.data.min_cells_per_gene)
    args.output.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment, "dataset": args.dataset_name,
        "cells": data.n_cells, "genes": data.n_genes,
        "maximum_epochs": cfg.train.warmup_epochs + cfg.train.contamination_epochs + cfg.train.joint_epochs,
        "checkpoint_every": cfg.train.checkpoint_every,
        "evaluated_embedding": "z_c/intrinsic only",
        "cosmx_adaptations": bool(args.cosmx_adaptations),
    }
    (args.output / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (args.output / "resolved_config_at_launch.json").write_text(
        json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    trainer = CastleTrainer(data, cfg)
    checkpoint = args.output / "checkpoint.pt"
    if checkpoint.exists():
        print(f"Resuming from {checkpoint}", flush=True)
        trainer.load_checkpoint(checkpoint)
    started = time.perf_counter()
    try:
        trainer.fit()
    except KeyboardInterrupt:
        trainer.save_checkpoint(checkpoint)
        trainer.write_history(args.output / "training_history_progress.csv")
        status.update(status="interrupted_checkpoint_saved",
                      interrupted_utc=datetime.now(timezone.utc).isoformat(),
                      completed_history_rows=len(trainer.history))
        (args.output / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise

    trainer.export()
    np.save(args.output / "z_c.npy", trainer.cache.zc)
    info = {
        **status, "status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "hours_this_session": round((time.perf_counter() - started) / 3600, 2),
        "settings": overrides,
        "contact_edges_kind": (
            "proxy; see bundle/proxy_contacts_manifest.json"
            if (args.bundle / "proxy_contacts_manifest.json").exists()
            else "segmentation-mask contacts"
        ),
    }
    (args.output / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    status.update(status="completed", completed_utc=info["completed_utc"],
                  completed_history_rows=len(trainer.history))
    (args.output / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
