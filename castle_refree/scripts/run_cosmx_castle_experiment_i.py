"""Run tuned CASTLE experiment i on the 2,000-HVG CosMx cortex bundle."""

from __future__ import annotations

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
BUNDLE = ROOT / "data" / "cosmx_frontal_cortex_hvg2000"
OUTPUT = ROOT / "results" / "embedding_cosmx_hvg2000" / "castle_i"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    tune = load_module("tune_embedding", SCRIPT_DIR / "tune_embedding.py")
    suite = tune.load_suite()
    base = load_config(ROOT / "castle_refree" / "config" / "decontam_base.yaml")
    base.train.seed = 7
    overrides = {
        "loss.kl_max": 0.5,
        "model.intrinsic_dim": 16,
        "model.prior_components": 30,
        "model.encoder_input": "dual",
        "train.num_threads": 4,
    }
    cfg = tune.identity_config(base, suite, overrides)
    cfg = copy.deepcopy(cfg)
    cfg.data.input_dir = str(BUNDLE.resolve())
    cfg.data.output_dir = str(OUTPUT.resolve())
    cfg.data.min_counts = 5
    cfg.data.min_cells_per_gene = 10
    cfg.train.device = "cuda"
    # The experiment-suite base uses 1000 (effectively final-only).  This full
    # laptop run needs recoverable snapshots in case it must be interrupted.
    cfg.train.checkpoint_every = 5

    # CosMx-valid replacements for physical channels that only exist in the
    # teammate's Xenium raw bundle. All tuned identity settings remain exact.
    cfg.model.contamination.spillover_graph = "knn"
    cfg.model.contamination.geometry_features = "basic"
    cfg.model.contamination.background = "auto"

    data = load_spatial_data(cfg.data.input_dir, cfg.data.min_counts, cfg.data.min_cells_per_gene)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    launch = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "i",
        "dataset": "CosMx Human Frontal Cortex, fixed 2000 HVGs",
        "cells": data.n_cells,
        "genes": data.n_genes,
        "maximum_epochs": cfg.train.warmup_epochs + cfg.train.contamination_epochs + cfg.train.joint_epochs,
        "checkpoint_every": cfg.train.checkpoint_every,
        "niche_phase": False,
    }
    (OUTPUT / "run_status.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    (OUTPUT / "resolved_config_at_launch.json").write_text(
        json.dumps(cfg.to_dict(), indent=2), encoding="utf-8"
    )
    started = time.perf_counter()
    trainer = CastleTrainer(data, cfg)
    checkpoint = OUTPUT / "checkpoint.pt"
    if checkpoint.exists():
        print(f"Resuming from {checkpoint}", flush=True)
        trainer.load_checkpoint(checkpoint)
    try:
        trainer.fit()
    except KeyboardInterrupt:
        trainer.save_checkpoint(checkpoint)
        trainer.write_history(OUTPUT / "training_history_progress.csv")
        launch["status"] = "interrupted_checkpoint_saved"
        launch["interrupted_utc"] = datetime.now(timezone.utc).isoformat()
        launch["completed_history_rows"] = len(trainer.history)
        (OUTPUT / "run_status.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
        raise
    trainer.export()
    np.save(OUTPUT / "z_c.npy", trainer.cache.zc)
    info = {
        "experiment": "i",
        "dataset": "CosMx Human Frontal Cortex, fixed 2000 HVGs",
        "cells": data.n_cells,
        "genes": data.n_genes,
        "settings": overrides,
        "niche_phase": False,
        "cosmx_adaptations": {
            "spillover_graph": "knn (true Xenium-style contact edges unavailable in prepared CosMx bundle)",
            "geometry_features": "basic coordinates and area",
            "background": "auto from CosMx aggregate negative-control counts",
        },
        "hours": round((time.perf_counter() - started) / 3600, 2),
    }
    (OUTPUT / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    launch["status"] = "completed"
    launch["completed_utc"] = datetime.now(timezone.utc).isoformat()
    launch["completed_history_rows"] = len(trainer.history)
    (OUTPUT / "run_status.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
