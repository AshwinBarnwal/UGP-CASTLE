from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    input_dir: str = "../data/castle_xenium_breast"
    output_dir: str = "../results/castle_refree_xenium_breast"
    min_counts: int = 5
    min_cells_per_gene: int = 10


@dataclass
class GraphConfig:
    n_neighbors: int = 12
    distance_quantile: float = 0.99
    distance_scale: float = 0.0  # <= 0 means median graph distance


@dataclass
class ExpressionGuidedConfig:
    enabled: bool = True
    strength: float = 1.25
    max_logit_shift: float = 1.5
    projection_dim: int = 32
    seed: int = 17


@dataclass
class ContaminationConfig:
    enabled: bool = True
    eta_max: float = 0.35
    purify_steps: int = 2
    expression_guided: ExpressionGuidedConfig = field(
        default_factory=ExpressionGuidedConfig
    )
    # Defaults reproduce the original model (M0); the decontamination configs
    # switch these on one at a time.
    spillover_graph: str = "knn"  # "knn" | "contact" (segmentation-mask contacts)
    kernel: str = "fixed"  # "fixed" distance softmax | "learned" (distance, boundary, size)
    geometry_features: str = "basic"  # "basic" | "full" (geometry.csv morphology)
    donor_gradient: bool = False  # donor eta/kernel computed live with gradients
    library: str = "free"  # "free" per-cell parameter | "amortized" q(l|x) with area prior
    library_prior_sd: float = 0.5  # sd of log library prior around log(kappa * area)
    background: str = "auto"  # "auto" (legacy) | "none" | "controls" (eq. 13) | "controls_learned"
    background_dirichlet_alpha: float = 1000.0  # beta ~ Dirichlet(alpha * 1) for "controls_learned"
    unroll_fix: bool = False  # K real re-encodings; export the trained posterior
    # Spec-faithful option (CASTLE text, Sec. 7.1): donors' mu_j re-decoded inside
    # each batch with gradients (first hop live, second hop cached).
    donor_live_profiles: bool = False
    freeze_in_joint: bool = False  # eta network and transfer kernel fixed during the niche phase


@dataclass
class ModelConfig:
    hidden_dim: int = 96
    intrinsic_dim: int = 16
    niche_dim: int = 16
    dropout: float = 0.05
    likelihood: str = "nb"  # "nb" | "zinb" (zero inflation; ablation only)
    # "mean": averages over the 4/8/12 nearest cells. "contact_steps": one scale per
    # contact step (cells within 1, 2, ... steps of touching cells), each with
    # attention plus rotation-invariant azimuthal features (spec Sec. 5.2).
    niche_encoder: str = "mean"
    niche_steps: int = 3
    niche_input: str = "hidden"  # neighbours described by encoder "hidden" states or by their "zc"
    # Identity (z_c) model options; defaults reproduce the original network.
    n_layers: int = 1  # hidden layers in the expression encoder and decoders
    norm: str = "layer"  # "layer" | "batch"
    encoder_input: str = "purified"  # "purified" (re-encode purified counts) | "dual" (raw + purified)
    prior_components: int = 0  # 0 = standard normal prior on z_c; K > 0 = learned Gaussian mixture
    hidden_genes: list[int] = field(default_factory=list)  # genes masked from the encoder (tuning)
    contamination: ContaminationConfig = field(default_factory=ContaminationConfig)


@dataclass
class LossConfig:
    beta_intrinsic_start: float = 1.0
    beta_intrinsic_end: float = 4.0
    beta_niche: float = 0.5
    lambda_niche_sparse: float = 0.05
    lambda_niche_reconstruction: float = 0.5
    lambda_hsic: float = 0.01
    lambda_eta: float = 0.01
    lambda_library: float = 0.1
    kl_warmup_epochs: int = 0  # > 0: z_c KL weight rises 0 -> kl_max over these epochs (all phases)
    kl_max: float = 1.0
    prior_weight_alpha: float = 0.5  # Dirichlet concentration on mixture weights (< 1 = sparse)
    identity_recon_weight: float = 0.0  # extra reconstruction from z_c alone (no niche effect)
    block_delta_grad: bool = False  # the niche effect cannot send gradients into z_c
    lambda_niche_zc: float = 0.0  # niche head predicts the touching neighbours' mean z_c
    niche_target: str = "decoded"  # "decoded" mu_j | "purified" counts of touching cells (spec eq. 31)
    niche_group_weight: float = 0.0  # gamma_grp in spec eq. 29


@dataclass
class TrainConfig:
    warmup_epochs: int = 5
    contamination_epochs: int = 15
    joint_epochs: int = 20
    batch_size: int = 512
    cache_batch_size: int = 2048
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 5.0
    seed: int = 7
    device: str = "auto"
    num_threads: int = 0
    checkpoint_every: int = 5
    validation_fraction: float = 0.0  # cells held out of training (early stopping, tuning)
    early_stop_phase: str = ""  # phase whose length is set by early stopping ("" = off)
    early_stop_patience: int = 15
    early_stop_min_epochs: int = 10


@dataclass
class CastleConfig:
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown configuration key: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> CastleConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    cfg = _merge_dataclass(CastleConfig(), values)
    base = path.resolve().parent
    for attr in ("input_dir", "output_dir"):
        raw = Path(getattr(cfg.data, attr))
        if not raw.is_absolute():
            setattr(cfg.data, attr, str((base / raw).resolve()))
    return cfg
