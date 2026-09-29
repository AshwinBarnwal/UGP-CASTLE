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


@dataclass
class ModelConfig:
    hidden_dim: int = 96
    intrinsic_dim: int = 16
    niche_dim: int = 16
    dropout: float = 0.05
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
