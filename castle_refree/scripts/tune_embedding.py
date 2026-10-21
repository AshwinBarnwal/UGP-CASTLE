"""Label-free tuning of the CASTLE identity (z_c) model on the crop.

Decontamination is fixed to M6 + eta prior (lambda = 3). Settings are compared by
held-out gene likelihood (spec Sec. 10): 10% of genes are hidden from the encoder
(they stay in the likelihood), 10% of cells are held out of training, and each
setting is scored by the NB log-likelihood of the hidden genes in the held-out
cells. Cell-type labels are never used.

Grid: maximum KL weight x latent dimensions x mixture prior, then the best
setting re-run with purified-only encoder input.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from castle_refree.config import CastleConfig, load_config
from castle_refree.data import load_spatial_data
from castle_refree.model import _nb_nll
from castle_refree.trainer import CastleTrainer

SCRIPT_DIR = Path(__file__).resolve().parent

# Fixed identity-model settings shared by tuning and the full-slide experiments.
IDENTITY_BASE = {
    "model.n_layers": 2,
    "model.hidden_dim": 128,
    "model.norm": "batch",
    "model.dropout": 0.1,
    "model.encoder_input": "dual",
    "loss.kl_warmup_epochs": 20,
    "train.validation_fraction": 0.1,
    "train.early_stop_phase": "contamination",
    "train.early_stop_patience": 15,
    "train.early_stop_min_epochs": 20,
    "train.warmup_epochs": 5,
    "train.contamination_epochs": 200,
    "train.joint_epochs": 0,  # no niche
}


def load_suite():
    spec = importlib.util.spec_from_file_location("suite", SCRIPT_DIR / "run_decontamination_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity_config(base: CastleConfig, suite, overrides: dict) -> CastleConfig:
    """M6 + lambda = 3 decontamination with the identity-model settings applied."""
    cfg = suite.apply_overrides(base, suite.VERSIONS["P3_eta_prior"])
    return suite.apply_overrides(cfg, {**IDENTITY_BASE, **overrides})


def held_out_loglik(trainer: CastleTrainer, genes: list[int]) -> float:
    """Mean per-cell NB log-likelihood of the hidden genes in the held-out cells."""
    trainer.model.eval()
    phase = str(trainer.history[-1].get("phase", "contamination"))
    total, count = 0.0, 0
    with torch.no_grad():
        dispersion = F.softplus(trainer.model.log_dispersion).unsqueeze(0) + 1e-4
        for start in range(0, trainer.validation_ids.size, trainer.cfg.train.cache_batch_size):
            ids = trainer.validation_ids[start : start + trainer.cfg.train.cache_batch_size]
            tensors = trainer._batch_tensors(ids)
            result = trainer.model.forward_batch(**tensors, phase=phase, beta_intrinsic=1.0, stochastic=False)
            nll = _nb_nll(tensors["x"][:, genes], result.expected[:, genes], dispersion[:, genes])
            total += -float(nll) * len(ids)
            count += len(ids)
    return total / max(count, 1)


def run(name: str, cfg: CastleConfig, bundle: Path, out_root: Path, genes: list[int]) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.data.input_dir = str(bundle.resolve())
    cfg.data.output_dir = str((out_root / name).resolve())
    data = load_spatial_data(cfg.data.input_dir, cfg.data.min_counts, cfg.data.min_cells_per_gene)
    started = time.perf_counter()
    trainer = CastleTrainer(data, cfg).fit()
    row = {
        "name": name,
        "kl_max": cfg.loss.kl_max,
        "latent_dims": cfg.model.intrinsic_dim,
        "mixture_components": cfg.model.prior_components,
        "encoder_input": cfg.model.encoder_input,
        "heldout_gene_loglik": held_out_loglik(trainer, genes),
        "epochs": sum(1 for r in trainer.history if "loss" in r),
        "minutes": round((time.perf_counter() - started) / 60, 1),
    }
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trainer.history).to_csv(out / "training_history.csv", index=False)
    (out / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("data/castle_decontam_sim/null"))
    parser.add_argument("--output", type=Path, default=Path("results/embedding_tuning"))
    parser.add_argument("--base-config", type=Path, default=Path("castle_refree/config/decontam_base.yaml"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-fraction", type=float, default=0.1)
    args = parser.parse_args()

    suite = load_suite()
    base = load_config(args.base_config)
    base.train.seed = args.seed
    n_genes = len(pd.read_csv(args.bundle / "genes.tsv", sep="\t", header=None))
    hidden = sorted(np.random.default_rng(0).choice(n_genes, size=int(round(args.hidden_fraction * n_genes)), replace=False).tolist())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "hidden_genes.json").write_text(json.dumps(hidden), encoding="utf-8")

    rows = []
    for kl_max, dims, components in itertools.product([0.5, 1.0], [10, 16], [0, 30]):
        name = f"kl{kl_max}_z{dims}_mog{components}"
        overrides = {"loss.kl_max": kl_max, "model.intrinsic_dim": dims,
                     "model.prior_components": components, "model.hidden_genes": hidden}
        rows.append(run(name, identity_config(base, suite, overrides), args.bundle, args.output, hidden))
        pd.DataFrame(rows).to_csv(args.output / "tuning.csv", index=False)

    best = max(rows, key=lambda r: r["heldout_gene_loglik"])
    overrides = {"loss.kl_max": best["kl_max"], "model.intrinsic_dim": best["latent_dims"],
                 "model.prior_components": best["mixture_components"], "model.hidden_genes": hidden,
                 "model.encoder_input": "purified"}
    rows.append(run(best["name"] + "_purified_input", identity_config(base, suite, overrides), args.bundle, args.output, hidden))
    table = pd.DataFrame(rows).sort_values("heldout_gene_loglik", ascending=False)
    table.to_csv(args.output / "tuning.csv", index=False)
    winner = table.iloc[0].to_dict()
    (args.output / "best.json").write_text(json.dumps(winner, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
