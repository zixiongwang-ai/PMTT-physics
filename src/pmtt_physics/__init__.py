"""PMTT physics: patch-level local feature spaces for physical representation."""

from .config import PaperConfig, load_config
from .geometry import construct_geometries, factor_geometry

__all__ = [
    "PaperConfig",
    "construct_geometries",
    "factor_geometry",
    "load_config",
]

__version__ = "0.1.0"
