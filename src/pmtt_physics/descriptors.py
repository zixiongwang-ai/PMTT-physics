"""Source-only appearance descriptors used for matched controls."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import math

import numpy as np
from PIL import Image


def rgb_descriptor(
    image: str | Path | Image.Image | np.ndarray,
    point_xy: Sequence[float],
    side: float,
) -> np.ndarray:
    if isinstance(image, Image.Image):
        opened = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        opened = Image.fromarray(image).convert("RGB")
    else:
        with Image.open(image) as source:
            opened = source.convert("RGB")
    x_coordinate, y_coordinate = map(float, point_xy)
    crop = np.asarray(
        opened.resize(
            (32, 32),
            Image.Resampling.BICUBIC,
            box=(
                x_coordinate - side / 2,
                y_coordinate - side / 2,
                x_coordinate + side / 2,
                y_coordinate + side / 2,
            ),
        ),
        dtype="float32",
    ) / 255
    return np.concatenate(
        [
            crop.ravel() / math.sqrt(crop.size),
            0.5 * np.diff(crop, axis=0).ravel() / math.sqrt(31 * 32 * 3),
            0.5 * np.diff(crop, axis=1).ravel() / math.sqrt(32 * 31 * 3),
        ]
    )
