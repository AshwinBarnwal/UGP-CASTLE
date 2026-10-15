"""Fit CASTLE model versions on the null/spill/niche bundles and score them.

Each version adds one decontamination fix to the previous one (M0 = original
code, M6 = all fixes); A1-A4 are ablations of M6. Every version is fitted on the
three simulated bundles with the same seed, then ``evaluate_decontamination.py``
appends its row to the summary table.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path

import pandas as pd

from castle_refree.config import CastleConfig, load_config
from castle_refree.data import load_spatial_data
from castle_refree.trainer import CastleTrainer

M1 = {"contamination.spillover_graph": "contact"}
M2 = {**M1, "contamination.kernel": "learned", "contamination.geometry_features": "full"}
M3 = {**M2, "contamination.donor_gradient": True}
M4 = {**M3, "contamination.library": "amortized"}
M5 = {**M4, "contamination.background": "controls"}
M6 = {
    **M5,
    "contamination.unroll_fix": True,
    "contamination.expression_guided.enabled": False,
    "loss.lambda_eta": 0.0,
}
VERSIONS: dict[str, dict[str, object]] = {
    "M0": {},
    "M1": M1,
    "M2": M2,
    "M3": M3,
    "M4": M4,
    "M5": M5,
    "M6": M6,
    "A1_expression_guided": {**M6, "contamination.expression_guided.enabled": True},
    "A2_niche_off": {**M6, "train.niche_off": True},
    "A3_no_contamination": {**M6, "contamination.enabled": False},
    "A4_zinb": {**M6, "model.likelihood": "zinb"},
    # Over-correction controls on M6: an exponential prior on eta (penalty
    # lambda_eta * eta per cell, i.e. MAP under eta ~ Exponential(lambda)) or a
    # lower hard cap on the leakage fraction.
    "P2_eta_prior": {**M6, "loss.lambda_eta": 2.0},
    "P3_eta_prior": {**M6, "loss.lambda_eta": 3.0},
    "P5_eta_prior": {**M6, "loss.lambda_eta": 5.0},
    "P10_eta_prior": {**M6, "loss.lambda_eta": 10.0},
    "P30_eta_prior": {**M6, "loss.lambda_eta": 30.0},
    "P100_eta_prior": {**M6, "loss.lambda_eta": 100.0},
    "C20_eta_max": {**M6, "contamination.eta_max": 0.20},
    "C10_eta_max": {**M6, "contamination.eta_max": 0.10},
}

# Spec-faithful stages (CASTLE text), each also run with the eta prior (lambda = 3).
S0 = {**M6, "contamination.background": "controls_learned"}  # learned beta_g, Dirichlet prior
S1 = {**S0, "contamination.geometry_features": "spec"}  # + z-extent, outer-transcript fraction
S2 = {**S1, "contamination.donor_live_profiles": True}  # + live donor profiles (Sec. 7.1)
S4 = {
    **S2,
    "model.niche_encoder": "contact_steps",  # attention + azimuthal over 1/2/3 contact steps
    "model.niche_steps": 3,
    "loss.niche_target": "purified",  # spec eq. 31
    "loss.niche_group_weight": 1.0,  # spec eq. 29
}
# M6 + lambda = 3 with only S4's contact-step niche (isolates the niche encoder).
VERSIONS["P3_contact_niche"] = {
    **M6,
    "loss.lambda_eta": 3.0,
    **{k: v for k, v in S4.items() if k.startswith("model.niche") or k.startswith("loss.niche")},
    "train.num_threads": 2,  # speed only; shares the CPU with full-slide runs
}
for _name, _overrides in {"S0": S0, "S1": S1, "S2": S2, "S4": S4}.items():
    VERSIONS[_name] = _overrides
    VERSIONS[f"{_name}_P3"] = {**_overrides, "loss.lambda_eta": 3.0}


def apply_overrides(cfg: CastleConfig, overrides: dict[str, object]) -> CastleConfig:
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if key == "train.niche_off":
            # Same total epochs, all spent before the niche phase.
            cfg.train.contamination_epochs += cfg.train.joint_epochs
            cfg.train.joint_epochs = 0
            continue
        path = key.split(".")
        target: object = cfg
        if path[0] == "contamination":
            target = cfg.model.contamination
            path = path[1:]
        for part in path[:-1]:
            target = getattr(target, part)
        if not hasattr(target, path[-1]):
            raise ValueError(f"Unknown override {key}")
        setattr(target, path[-1], value)
    return cfg


def load_evaluator(script_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "evaluate_decontamination", script_dir / "evaluate_decontamination.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(VERSIONS))
    parser.add_argument("--variants", nargs="+", default=["null", "spill", "niche"])
    parser.add_argument("--base-config", type=Path, default=Path("castle_refree/config/decontam_base.yaml"))
    parser.add_argument("--sim-root", type=Path, default=Path("data/castle_decontam_sim"))
    parser.add_argument("--results-root", type=Path, default=Path("results/decontamination"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", nargs=3, type=int, metavar=("WARMUP", "CONTAM", "JOINT"))
    parser.add_argument("--name-suffix", default="", help="Appended to version names, e.g. _ep80")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse finished fits")
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args()

    base = load_config(args.base_config)
    base.train.seed = args.seed
    if args.epochs:
        base.train.warmup_epochs, base.train.contamination_epochs, base.train.joint_epochs = args.epochs
    evaluator = load_evaluator(Path(__file__).resolve().parent)
    timings = []
    for name in args.models:
        if name not in VERSIONS:
            raise ValueError(f"Unknown model version {name}; choose from {list(VERSIONS)}")
        cfg_version = apply_overrides(base, VERSIONS[name])
        label = f"{name}{args.name_suffix}"
        run_root = (args.results_root / f"seed{args.seed}" / label).resolve()
        for variant in args.variants:
            out = run_root / variant
            if args.skip_existing and (out / "castle_embeddings_and_qc.npz").exists():
                continue
            cfg = copy.deepcopy(cfg_version)
            cfg.data.input_dir = str((args.sim_root / variant).resolve())
            cfg.data.output_dir = str(out)
            data = load_spatial_data(cfg.data.input_dir, cfg.data.min_counts, cfg.data.min_cells_per_gene)
            started = time.perf_counter()
            CastleTrainer(data, cfg).fit().export()
            seconds = time.perf_counter() - started
            timings.append({"model": label, "variant": variant, "seconds": round(seconds, 1)})
            print(json.dumps(timings[-1]), flush=True)
        if not args.no_evaluate and all((run_root / v).exists() for v in ("null", "spill", "niche")):
            row = evaluator.evaluate(run_root, args.sim_root.resolve(), label, marker_genes=5)
            if all((run_root / v / "castle_embeddings_and_qc.npz").exists() for v in ("clean0", "cleanspill")):
                row.update(evaluator.evaluate_clean(run_root, args.sim_root.resolve()))
            row["seed"] = args.seed
            row["epochs"] = (
                cfg_version.train.warmup_epochs + cfg_version.train.contamination_epochs
                + cfg_version.train.joint_epochs
            )
            summary = args.results_root / "summary.csv"
            summary.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame([row])
            if summary.exists():
                previous = pd.read_csv(summary)
                keep = ~((previous["model"] == label) & (previous.get("seed", -1) == args.seed))
                frame = pd.concat([previous[keep], frame], ignore_index=True)
            frame.to_csv(summary, index=False)
            print(json.dumps(row, indent=2), flush=True)
    if timings:
        timing_path = args.results_root / "timings.csv"
        frame = pd.DataFrame(timings)
        if timing_path.exists():
            frame = pd.concat([pd.read_csv(timing_path), frame], ignore_index=True)
        frame.to_csv(timing_path, index=False)


if __name__ == "__main__":
    main()
