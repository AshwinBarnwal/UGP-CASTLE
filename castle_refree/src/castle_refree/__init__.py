"""CASTLE-RefFree: joint reference-free denoising and embedding."""

from .config import CastleConfig, load_config
from .data import SpatialCountData, load_spatial_data
from .trainer import CastleTrainer

__all__ = [
    "CastleConfig",
    "CastleTrainer",
    "SpatialCountData",
    "load_config",
    "load_spatial_data",
]

__version__ = "0.1.0"

