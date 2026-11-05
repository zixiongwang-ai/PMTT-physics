"""TAP-Vid-DAVIS loading and exact experiment-identity helpers."""

from __future__ import annotations

from pathlib import Path
import csv
import pickle
import numpy as np


def load_tapvid(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or len(data) != 30:
        raise ValueError("expected the 30-sequence TAP-Vid-DAVIS mapping")
    for name, record in data.items():
        video = record["video"]
        points = record["points"]
        occluded = record["occluded"]
        if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8:
            raise ValueError(f"invalid video array for {name}")
        if points.shape != (len(occluded), len(video), 2):
            raise ValueError(f"invalid point array for {name}")
        if occluded.dtype != bool or not np.isfinite(points).all():
            raise ValueError(f"invalid visibility data for {name}")
        if points.min() < 0 or points.max() > 1:
            raise ValueError(f"normalized coordinates out of range for {name}")
    return data


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_interfaces(
    data: dict[str, dict[str, np.ndarray]],
    rows: list[dict[str, str]],
) -> None:
    for row in rows:
        sequence = row["sequence"]
        track = int(row["track"])
        frame = int(row["frame"])
        if sequence not in data:
            raise ValueError(f"unknown sequence in interface manifest: {sequence}")
        record = data[sequence]
        if not (0 <= track < len(record["points"])) or not (0 <= frame < len(record["video"])):
            raise ValueError(f"interface index out of range: {row['interface_id']}")
        height, width = record["video"].shape[1:3]
        expected = record["points"][track, frame] * np.asarray([width, height])
        actual = np.asarray([float(row["x"]), float(row["y"])])
        if not np.allclose(actual, expected, atol=1e-5):
            raise ValueError(f"interface coordinate mismatch: {row['interface_id']}")
