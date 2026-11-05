"""Portable feature stores for PMTT physics reproduction runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch


ARRAY_NAMES = ("center", "states", "future", "rgb", "projected_geometry")


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if not materialized:
        destination.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _shape_text(shape: Sequence[int]) -> str:
    return "[" + ", ".join(str(int(value)) for value in shape) + "]"


def create_store(
    root: str | Path,
    *,
    kind: str,
    scale: str,
    rows: int,
    width: int,
    future_branches: int = 0,
    rgb_width: int = 0,
    geometry_width: int = 0,
) -> dict[str, np.memmap]:
    """Create an empty safe NumPy feature store.

    Arrays use the ``.npy`` format and can be memory-mapped without loading the
    full experiment into RAM. No pickle-backed array is used.
    """
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.memmap] = {
        "center": np.lib.format.open_memmap(
            destination / "center.npy", mode="w+", dtype=np.float32, shape=(rows, width)
        ),
        "states": np.lib.format.open_memmap(
            destination / "states.npy",
            mode="w+",
            dtype=np.float32,
            shape=(rows, 5, 16, width),
        ),
    }
    if future_branches:
        arrays["future"] = np.lib.format.open_memmap(
            destination / "future.npy",
            mode="w+",
            dtype=np.float32,
            shape=(rows, future_branches, width),
        )
    if rgb_width:
        arrays["rgb"] = np.lib.format.open_memmap(
            destination / "rgb.npy", mode="w+", dtype=np.float32, shape=(rows, rgb_width)
        )
    if geometry_width:
        arrays["projected_geometry"] = np.lib.format.open_memmap(
            destination / "projected_geometry.npy",
            mode="w+",
            dtype=np.float32,
            shape=(rows, geometry_width),
        )
    lines = [
        f'kind = "{kind}"',
        f'scale = "{scale}"',
        'dtype = "float32"',
        f"rows = {rows}",
        f"feature_width = {width}",
        "",
        "[arrays]",
    ]
    for name, array in arrays.items():
        lines.append(f'{name} = "{name}.npy"')
        lines.append(f'{name}_shape = {_shape_text(array.shape)}')
    (destination / "store.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return arrays


@dataclass
class FeatureStore:
    root: Path
    kind: str
    scale: str
    metadata: list[dict[str, str]]
    arrays: dict[str, np.ndarray]

    @classmethod
    def open(cls, root: str | Path, *, mode: str = "r") -> "FeatureStore":
        location = Path(root)
        config = tomllib.loads((location / "store.toml").read_text(encoding="utf-8"))
        arrays = {
            name: np.load(location / relative, mmap_mode=mode, allow_pickle=False)
            for name, relative in config["arrays"].items()
            if not name.endswith("_shape")
        }
        metadata = read_rows(location / "metadata.csv")
        rows = int(config["rows"])
        if len(metadata) != rows:
            raise ValueError(f"metadata row count mismatch: expected {rows}, received {len(metadata)}")
        for name, array in arrays.items():
            if len(array) != rows:
                raise ValueError(f"array row count mismatch for {name}")
            if array.dtype != np.float32:
                raise ValueError(f"unexpected dtype for {name}: {array.dtype}")
        return cls(
            root=location,
            kind=str(config["kind"]),
            scale=str(config["scale"]),
            metadata=metadata,
            arrays=arrays,
        )

    @property
    def index(self) -> dict[str, int]:
        field = "interface_id" if self.kind == "tapvid" else "sample_id"
        return {row[field]: index for index, row in enumerate(self.metadata)}

    def take(self, indices: Sequence[int], *, device: str | torch.device = "cpu") -> dict[str, Any]:
        selected = list(map(int, indices))
        result: dict[str, Any] = {
            name: torch.from_numpy(np.array(array[selected], copy=True)).to(device)
            for name, array in self.arrays.items()
        }
        result["metadata"] = [self.metadata[index] for index in selected]
        result["seeds"] = [int(self.metadata[index]["source_seed"]) for index in selected]
        result["units"] = [
            self.metadata[index].get("sequence", self.metadata[index].get("base_id", ""))
            for index in selected
        ]
        return result

    def center(self, indices: Sequence[int], *, device: str | torch.device = "cpu") -> torch.Tensor:
        selected = list(map(int, indices))
        return torch.from_numpy(np.array(self.arrays["center"][selected], copy=True)).to(device)

    def flush(self) -> None:
        for array in self.arrays.values():
            flush = getattr(array, "flush", None)
            if callable(flush):
                flush()
