from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import load_spatial_data
from .trainer import CastleTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference-free CASTLE training")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load/filter data and build graph, but do not train",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Resume from a checkpoint.pt written by an interrupted run",
    )
    parser.add_argument(
        "--geometry-only", action="store_true",
        help="Disable expression-guided edge correction but retain denoising",
    )
    parser.add_argument(
        "--no-contamination", action="store_true",
        help="Disable the complete contamination/purification module",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override the result directory",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg.data.output_dir = str(Path(args.output_dir).resolve())
    if args.geometry_only:
        cfg.model.contamination.expression_guided.enabled = False
        if not args.output_dir:
            cfg.data.output_dir += "_geometry_only"
    if args.no_contamination:
        cfg.model.contamination.enabled = False
        if not args.output_dir:
            cfg.data.output_dir += "_no_contamination"
    data = load_spatial_data(
        cfg.data.input_dir, cfg.data.min_counts, cfg.data.min_cells_per_gene
    )
    trainer = CastleTrainer(data, cfg)
    if args.resume:
        if args.geometry_only or args.no_contamination:
            raise ValueError(
                "Do not change ablation switches while resuming an existing model."
            )
        trainer.load_checkpoint(args.resume)
    print(
        json.dumps(
            {
                "cells": data.n_cells,
                "genes": data.n_genes,
                "edges": int(trainer.graph.source.size),
                "device": str(trainer.device),
                "expression_guided": cfg.model.contamination.expression_guided.enabled,
                "contamination": cfg.model.contamination.enabled,
            }
        ),
        flush=True,
    )
    if not args.dry_run:
        trainer.fit().export()


if __name__ == "__main__":
    main()
