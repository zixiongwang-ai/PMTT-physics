"""Configuration loading for the paper protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class PaperConfig:
    seed: int = 20261103
    holdout_seed: int = 87261103
    observation_seed: int = 20261101
    source_seed: int = 20261102
    output_size: int = 224
    crop_side: int = 128
    synthetic_crop_side: int = 96
    minimum_common_points: int = 8
    points_per_state: int = 8
    future_offsets: tuple[int, ...] = (-6, -3, 3, 6)
    ranks: tuple[int, ...] = (1, 2, 4, 8)
    bootstrap_draws: int = 2000
    scalar_ridge: float = 0.001
    geometry_chunk: int = 64
    extraction_batch: int = 68
    fixed_forward_batch: bool = True

    def validate(self) -> None:
        if self.output_size <= 0 or self.output_size % 16:
            raise ValueError("output_size must be a positive multiple of 16")
        if self.crop_side <= 0 or self.synthetic_crop_side <= 0:
            raise ValueError("crop sizes must be positive")
        if tuple(sorted(self.ranks)) != self.ranks or self.ranks[-1] > 8:
            raise ValueError("ranks must be sorted and at most 8")
        if self.future_offsets != (-6, -3, 3, 6):
            raise ValueError("the released paper protocol uses future offsets -6, -3, 3, 6")
        if self.bootstrap_draws <= 0:
            raise ValueError("bootstrap_draws must be positive")


def _as_tuple(value: Any, cast: type) -> tuple:
    if not isinstance(value, list):
        raise TypeError(f"expected a TOML array, received {type(value).__name__}")
    return tuple(cast(item) for item in value)


def load_config(path: str | Path) -> PaperConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    protocol = data.get("protocol", data)
    config = PaperConfig(
        seed=int(protocol.get("seed", 20261103)),
        holdout_seed=int(protocol.get("holdout_seed", 87261103)),
        observation_seed=int(protocol.get("observation_seed", 20261101)),
        source_seed=int(protocol.get("source_seed", 20261102)),
        output_size=int(protocol.get("output_size", 224)),
        crop_side=int(protocol.get("crop_side", 128)),
        synthetic_crop_side=int(protocol.get("synthetic_crop_side", 96)),
        minimum_common_points=int(protocol.get("minimum_common_points", 8)),
        points_per_state=int(protocol.get("points_per_state", 8)),
        future_offsets=_as_tuple(protocol.get("future_offsets", [-6, -3, 3, 6]), int),
        ranks=_as_tuple(protocol.get("ranks", [1, 2, 4, 8]), int),
        bootstrap_draws=int(protocol.get("bootstrap_draws", 2000)),
        scalar_ridge=float(protocol.get("scalar_ridge", 0.001)),
        geometry_chunk=int(protocol.get("geometry_chunk", 64)),
        extraction_batch=int(protocol.get("extraction_batch", 68)),
        fixed_forward_batch=bool(protocol.get("fixed_forward_batch", True)),
    )
    config.validate()
    return config
