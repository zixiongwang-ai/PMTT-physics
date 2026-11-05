"""Deterministic synthetic articulated-object substrate used by PMTT physics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import csv
import itertools
import math
from collections.abc import Iterable, Sequence

import numpy as np
from PIL import Image

from .constants import PHYSICAL_BLOCKS
from .randomness import source_observation_seed


@dataclass(frozen=True)
class SyntheticBase:
    base_id: str
    cohort: str
    identity_index: int
    texture: int
    width: float
    height: float
    depth: float
    yaw: float
    pitch: float
    hinge: float
    camera_yaw: float
    camera_pitch: float
    background: tuple[int, int, int] = (42, 46, 53)
    lighting: tuple[float, float, float] = (1.0, 1.0, 1.0)
    jitter_points: bool = False


@dataclass(frozen=True)
class SyntheticCondition:
    base: SyntheticBase
    block: str
    state_index: int
    value: float
    point_ids: tuple[int, ...]


def rotation(axis: str, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    if axis == "yaw":
        return np.asarray([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    if axis == "pitch":
        return np.asarray([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
    raise ValueError(f"unknown rotation axis: {axis}")


def rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1.0]])


def physical_pose(
    base: SyntheticBase,
    block: str,
    value: float,
    future_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if block not in PHYSICAL_BLOCKS:
        raise ValueError(f"unknown physical block: {block}")
    configuration = value if block == "configuration" else 0.0
    axis = value if block == "axis" else 0.0
    joint = value if block == "joint" else 0.0
    prismatic = block == "motion" and value == 1.0

    root = (
        rotation("yaw", base.camera_yaw) @ rotation("pitch", base.camera_pitch)
    ).T @ rotation("yaw", base.yaw) @ rotation("pitch", base.pitch)
    root = root @ rotation_z(axis)

    width, height, depth = base.width, base.height, base.depth
    pivot = np.asarray([joint * width, 0.0, 0.0])
    offset = np.asarray([width / 2, 0.0, 0.0])
    moving_rotation = rotation(
        "yaw",
        base.hinge + configuration + (0.0 if prismatic else future_offset),
    )
    centers = [np.asarray([-width / 2, 0.0, 0.0]), pivot + moving_rotation @ (offset - pivot)]
    if prismatic:
        centers[1] = centers[1] + np.asarray(
            [math.radians(future_offset) * width * 0.65, 0.0, 0.0]
        )
    rotations = [np.eye(3), moving_rotation]
    half_extents = [[width / 2, height / 2, depth / 2]] * 2

    if prismatic:
        for y_position in (-height * 0.40, height * 0.40):
            rotations.append(np.eye(3))
            centers.append(np.asarray([width * 0.25, y_position, -depth / 2 - 0.045]))
            half_extents.append([width * 0.72, 0.018, 0.018])
    else:
        rotations.append(np.eye(3))
        centers.append(pivot + np.asarray([0.0, 0.0, -depth / 2 - 0.045]))
        half_extents.append([0.022, height * 0.52, 0.022])

    rotations_out = np.stack([root @ value for value in rotations])
    translations = np.stack([root @ value + [0.0, 0.0, 4.0] for value in centers])
    return rotations_out, translations, np.asarray(half_extents)


def render(
    base: SyntheticBase,
    rotations: np.ndarray,
    translations: np.ndarray,
    half_extents: np.ndarray,
    size: int = 320,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    rays = np.stack(
        [(xx + 0.5 - size / 2) / 300, (yy + 0.5 - size / 2) / 300, np.ones_like(xx)],
        axis=-1,
    ).reshape(-1, 3)
    depth = np.full(len(rays), 1e6)
    surface = np.full(len(rays), -1, np.int16)
    rgb = np.tile(np.asarray(base.background, np.uint8), (len(rays), 1))

    for part, (part_rotation, translation, half) in enumerate(
        zip(rotations, translations, half_extents, strict=True)
    ):
        direction = rays @ part_rotation
        origin = -translation @ part_rotation
        for coordinate in range(3):
            for sign in (-1, 1):
                face = part * 6 + coordinate * 2 + int(sign > 0)
                denominator = direction[:, coordinate]
                distance = (sign * half[coordinate] - origin[coordinate]) / np.where(
                    np.abs(denominator) > 1e-9, denominator, 1e-9
                )
                local = origin + distance[:, None] * direction
                inside = (
                    (np.abs(local) <= half + 1e-7).all(axis=1)
                    & (distance > 0)
                    & (distance < depth)
                )
                if not inside.any():
                    continue
                if part < 2:
                    coordinates = local[inside][
                        :, [index for index in range(3) if index != coordinate]
                    ]
                    rng = np.random.default_rng(base.texture + int(face) * 131)
                    phases = rng.uniform(0, 2 * np.pi, 3)
                    waves = np.stack(
                        [
                            np.sin(coordinates[:, 0] * frequency + phases[index])
                            * np.cos(
                                coordinates[:, 1] * (frequency * 0.79)
                                + phases[(index + 1) % 3]
                            )
                            for index, frequency in enumerate((21, 33, 47))
                        ],
                        axis=1,
                    )
                    checks = (
                        (
                            np.floor(coordinates[:, 0] * 18)
                            + np.floor(coordinates[:, 1] * 18)
                        )
                        % 2
                    )[:, None]
                    color = np.clip(120 + 70 * waves + 35 * checks, 20, 245).astype(np.uint8)
                else:
                    color = np.tile(
                        np.asarray([172, 178, 185], dtype=np.uint8),
                        (inside.sum(), 1),
                    )
                # Rasterize the base uint8 surface color before applying the
                # New16 ambient multiplier. The operation order is numerically
                # significant and is preserved by the public implementation.
                if base.lighting != (1.0, 1.0, 1.0):
                    color = np.rint(
                        np.clip(
                            color.astype(np.float64) * np.asarray(base.lighting)[None],
                            0,
                            255,
                        )
                    ).astype(np.uint8)
                rgb[inside] = color
                depth[inside] = distance[inside]
                surface[inside] = face

    return (
        Image.fromarray(rgb.reshape(size, size, 3)),
        depth.reshape(size, size).astype("float32"),
        surface.reshape(size, size),
    )


def material_points(base: SyntheticBase) -> list[dict[str, object]]:
    points = [
        {
            "id": part * 16 + x_index * 4 + y_index,
            "part": part,
            "face": part * 6 + 4,
            "xyz": [
                float(x_position * base.width),
                float(y_position * base.height),
                -base.depth / 2,
            ],
        }
        for part in (0, 1)
        for x_index, x_position in enumerate(np.linspace(-0.32, 0.32, 4))
        for y_index, y_position in enumerate(np.linspace(-0.32, 0.32, 4))
    ]
    if base.jitter_points:
        rng = np.random.default_rng(base.texture + 337)
        for point in points:
            xyz = point["xyz"]
            assert isinstance(xyz, list)
            xyz[0] += float(rng.uniform(-0.018, 0.018) * base.width)
            xyz[1] += float(rng.uniform(-0.018, 0.018) * base.height)
    return points


def project_points(
    points: Sequence[dict[str, object]],
    rotations: np.ndarray,
    translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world = np.stack(
        [
            rotations[int(point["part"])] @ np.asarray(point["xyz"])
            + translations[int(point["part"])]
            for point in points
        ]
    )
    return world[:, :2] / world[:, 2:] * 300 + 160, world[:, 2]


def visibility(
    points: Sequence[dict[str, object]],
    xy: np.ndarray,
    z: np.ndarray,
    depth: np.ndarray,
    surface: np.ndarray,
    crop_side: int = 96,
) -> np.ndarray:
    values = []
    for point, (x_coordinate, y_coordinate), depth_value in zip(
        points, xy, z, strict=True
    ):
        x_index, y_index = int(x_coordinate), int(y_coordinate)
        visible = (
            crop_side * 0.76 < x_coordinate < 320 - crop_side * 0.76
            and crop_side * 0.76 < y_coordinate < 320 - crop_side * 0.76
        )
        if visible:
            region = surface[
                max(0, y_index - 2) : y_index + 3,
                max(0, x_index - 2) : x_index + 3,
            ]
            visible = bool(
                (region == int(point["face"])).all()
                and abs(depth[y_index, x_index] - depth_value) < 0.035
            )
        values.append(visible)
    return np.asarray(values)


def projected_geometry_descriptor(
    rotations: np.ndarray,
    translations: np.ndarray,
    half_extents: np.ndarray,
) -> np.ndarray:
    values: list[float] = []
    for part in range(2):
        for signs in itertools.product((-1, 1), repeat=3):
            point = rotations[part] @ (half_extents[part] * signs) + translations[part]
            values.extend((point[:2] / point[2] * 300 / 320).tolist())
    for part in range(2, len(rotations)):
        for sign in (-1, 1):
            local = half_extents[part] * (
                [sign, 0, 0]
                if half_extents[part, 0] > half_extents[part, 1]
                else [0, sign, 0]
            )
            point = rotations[part] @ local + translations[part]
            values.extend((point[:2] / point[2] * 300 / 320).tolist())
    while len(values) < 40:
        values.extend(values[-4:])
    return np.asarray(values[:40], dtype="float32")


def _parse_tuple(value: str, cast: type) -> tuple:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"expected list-like value, received {value!r}")
    return tuple(cast(item) for item in parsed)


def load_conditions(path: str | Path) -> list[SyntheticCondition]:
    output: list[SyntheticCondition] = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            base = SyntheticBase(
                base_id=row["base_id"],
                cohort=row["cohort"],
                identity_index=int(row["identity_index"]),
                texture=int(row["texture"]),
                width=float(row["width"]),
                height=float(row["height"]),
                depth=float(row["depth"]),
                yaw=float(row["yaw"]),
                pitch=float(row["pitch"]),
                hinge=float(row["hinge"]),
                camera_yaw=float(row["camera_yaw"]),
                camera_pitch=float(row["camera_pitch"]),
                background=_parse_tuple(row["background"], int),
                lighting=_parse_tuple(row["lighting"], float),
                jitter_points=row["jitter_points"].lower() == "true",
            )
            output.append(
                SyntheticCondition(
                    base=base,
                    block=row["block"],
                    state_index=int(row["state_index"]),
                    value=float(row["value"]),
                    point_ids=_parse_tuple(row["point_ids"], int),
                )
            )
    return output


def render_conditions(
    conditions: Iterable[SyntheticCondition],
    output_root: str | Path,
    future_offsets: Sequence[int] = (-6, -3, 3, 6),
    crop_side: int = 96,
) -> list[dict[str, object]]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for condition in conditions:
        points_by_id = {int(point["id"]): point for point in material_points(condition.base)}
        points = [points_by_id[point_id] for point_id in condition.point_ids]
        for branch_index, offset in enumerate((0, *future_offsets)):
            rotations, translations, half_extents = physical_pose(
                condition.base,
                condition.block,
                condition.value,
                float(offset),
            )
            image, depth, surface = render(
                condition.base, rotations, translations, half_extents
            )
            xy, z = project_points(points, rotations, translations)
            visible = visibility(points, xy, z, depth, surface, crop_side)
            if not visible.all():
                raise ValueError(
                    f"saved material-point identity is not visible: {condition.base.base_id}, "
                    f"{condition.block}, state {condition.state_index}, offset {offset}"
                )
            folder = root / condition.base.base_id / condition.block / str(condition.state_index)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{branch_index}.png"
            image.save(path)
            descriptor = projected_geometry_descriptor(rotations, translations, half_extents)
            for point, coordinates in zip(points, xy, strict=True):
                manifest.append(
                    {
                        "base_id": condition.base.base_id,
                        "cohort": condition.base.cohort,
                        "block": condition.block,
                        "state_index": condition.state_index,
                        "value": condition.value,
                        "point_id": int(point["id"]),
                        "part": int(point["part"]),
                        "branch": branch_index,
                        "future_offset": offset,
                        "x": float(coordinates[0]),
                        "y": float(coordinates[1]),
                        "source_seed": source_observation_seed(
                            condition.base.identity_index, int(point["id"])
                        ),
                        "image": str(path.relative_to(root)),
                        "geometry_descriptor": descriptor.tolist(),
                    }
                )
    return manifest
