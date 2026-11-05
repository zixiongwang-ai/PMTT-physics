"""Fresh feature extraction from official datasets and checkpoints."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import torch

from .backbone import DinoV3Backbone
from .config import PaperConfig
from .constants import MODEL_SPECS
from .descriptors import rgb_descriptor
from .features import encode_views, split_state_family
from .observations import (
    canonical_crop,
    generated_observation_families,
    prepare_source_views,
)
from .storage import create_store, read_rows, write_rows
from .synthetic import (
    load_conditions,
    material_points,
    physical_pose,
    project_points,
    projected_geometry_descriptor,
    render,
    visibility,
)
from .tapvid import load_tapvid, validate_interfaces


def _synthetic_scales(path: str | Path) -> dict[str, float]:
    result = {row["family"]: float(row["scale"]) for row in read_rows(path)}
    expected = {"SamplingMatched", "Geometric", "Context", "Sensitivity"}
    if set(result) != expected:
        raise ValueError("synthetic observation-scale table is incomplete")
    return result


def _real_scales(path: str | Path, fold: int) -> dict[str, float]:
    result = {
        row["family"]: float(row["scale"])
        for row in read_rows(path)
        if int(row["fold"]) == int(fold)
    }
    expected = {"SamplingMatched", "Geometric", "Context", "Sensitivity"}
    if set(result) != expected:
        raise ValueError(f"real observation-scale table is incomplete for fold {fold}")
    return result


def _configure_numerics() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def extract_synthetic(
    *,
    conditions_path: str | Path,
    observation_scales_path: str | Path,
    checkpoint: str | Path,
    scale: str,
    output: str | Path,
    config: PaperConfig,
    device: str = "cuda",
) -> None:
    """Render and encode the 32-base synthetic paper substrate from public identities."""
    _configure_numerics()
    conditions = load_conditions(conditions_path)
    total = sum(len(condition.point_ids) for condition in conditions)
    width = int(MODEL_SPECS[scale]["width"])
    arrays = create_store(
        output,
        kind="synthetic",
        scale=scale,
        rows=total,
        width=width,
        future_branches=4,
        rgb_width=9024,
        geometry_width=40,
    )
    scales = _synthetic_scales(observation_scales_path)
    backbone = DinoV3Backbone(scale, checkpoint, device=device)
    metadata: list[dict[str, object]] = []
    row_index = 0

    for condition in conditions:
        all_points = {int(point["id"]): point for point in material_points(condition.base)}
        points = [all_points[point_id] for point_id in condition.point_ids]
        frames: list[tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for offset in (0, *config.future_offsets):
            rotations, translations, extents = physical_pose(
                condition.base, condition.block, condition.value, float(offset)
            )
            image, depth, surface = render(condition.base, rotations, translations, extents)
            coordinates, z = project_points(points, rotations, translations)
            if not visibility(points, coordinates, z, depth, surface, config.synthetic_crop_side).all():
                raise ValueError(
                    f"material-point visibility changed for {condition.base.base_id}/"
                    f"{condition.block}/{condition.state_index}"
                )
            frames.append((image, coordinates, rotations, translations, extents))

        source_image, source_xy, rotations, translations, extents = frames[0]
        pixel_groups = []
        grid_groups = []
        flag_groups = []
        seeds = []
        for point, coordinates in zip(points, source_xy, strict=True):
            seed = int(
                config.source_seed
                + (condition.base.identity_index * 32 + int(point["id"])) * 7919
            )
            pixels, grids, flags = prepare_source_views(
                source_image,
                coordinates,
                config.synthetic_crop_side,
                seed,
                geometric_scale=scales["Geometric"],
                context_scale=scales["Context"],
                sensitivity_scale=scales["Sensitivity"],
                output_size=config.output_size,
            )
            pixel_groups.append(pixels)
            grid_groups.append(grids)
            flag_groups.append(flags)
            seeds.append(seed)
        encoded = encode_views(
            backbone,
            torch.cat(pixel_groups),
            torch.cat(grid_groups),
            torch.cat(flag_groups),
            batch_size=config.extraction_batch,
            fixed_forward_batch=config.fixed_forward_batch,
        ).reshape(len(points), 81, width)
        center, states = split_state_family(encoded, seeds)

        future_pixels = []
        for point_index, _point in enumerate(points):
            for future_image, future_xy, *_ in frames[1:]:
                future_pixels.append(
                    canonical_crop(
                        future_image,
                        future_xy[point_index],
                        config.synthetic_crop_side,
                        config.output_size,
                    )
                )
        count = len(future_pixels)
        future = encode_views(
            backbone,
            torch.from_numpy(np.stack(future_pixels)),
            torch.zeros(count, 2),
            torch.ones(count, dtype=torch.bool),
            batch_size=config.extraction_batch,
            fixed_forward_batch=config.fixed_forward_batch,
        ).reshape(len(points), 4, width)

        descriptor = projected_geometry_descriptor(rotations, translations, extents)
        destination = slice(row_index, row_index + len(points))
        arrays["center"][destination] = center.numpy()
        arrays["states"][destination] = states.numpy()
        arrays["future"][destination] = future.numpy()
        arrays["projected_geometry"][destination] = np.repeat(
            descriptor[None], len(points), axis=0
        )
        arrays["rgb"][destination] = np.stack(
            [
                rgb_descriptor(source_image, coordinates, config.synthetic_crop_side)
                for coordinates in source_xy
            ]
        )
        for point, coordinates, seed in zip(points, source_xy, seeds, strict=True):
            metadata.append(
                {
                    "sample_id": (
                        f"{condition.base.base_id}:{condition.block}:"
                        f"{condition.state_index}:{int(point['id'])}"
                    ),
                    "base_id": condition.base.base_id,
                    "cohort": condition.base.cohort,
                    "block": condition.block,
                    "state_index": condition.state_index,
                    "state_value": condition.value,
                    "point_id": int(point["id"]),
                    "part": int(point["part"]),
                    "x": float(coordinates[0]),
                    "y": float(coordinates[1]),
                    "source_seed": seed,
                }
            )
        row_index += len(points)
        for array in arrays.values():
            array.flush()

    if row_index != total:
        raise RuntimeError(f"synthetic extraction wrote {row_index} rows, expected {total}")
    write_rows(Path(output) / "metadata.csv", metadata)


def _real_manifest_rows(manifest: str | Path) -> list[dict[str, str]]:
    rows = read_rows(manifest)
    for row in rows:
        row["has_observations"] = str(row["has_observations"]).lower()
    return rows


def extract_tapvid(
    *,
    dataset: str | Path,
    interfaces_path: str | Path,
    observation_scales_path: str | Path,
    checkpoint: str | Path,
    scale: str,
    output: str | Path,
    config: PaperConfig,
    device: str = "cuda",
    records_per_group: int = 4,
) -> None:
    """Encode source and target interfaces for the natural-video experiments."""
    if records_per_group <= 0:
        raise ValueError("records_per_group must be positive")
    _configure_numerics()
    data = load_tapvid(dataset)
    rows = _real_manifest_rows(interfaces_path)
    validate_interfaces(data, rows)
    width = int(MODEL_SPECS[scale]["width"])
    arrays = create_store(output, kind="tapvid", scale=scale, rows=len(rows), width=width)
    backbone = DinoV3Backbone(scale, checkpoint, device=device)

    for start in range(0, len(rows), records_per_group):
        group = rows[start : start + records_per_group]
        pixels_all = []
        grids_all = []
        flags_all = []
        seeds = []
        for row in group:
            video = data[row["sequence"]]["video"]
            frame = video[int(row["frame"])]
            seed = int(row["source_seed"])
            fold = int(row["fold"])
            scales = _real_scales(observation_scales_path, fold)
            pixels, grids, flags = prepare_source_views(
                frame,
                (float(row["x"]), float(row["y"])),
                config.crop_side,
                seed,
                geometric_scale=scales["Geometric"],
                context_scale=scales["Context"],
                sensitivity_scale=scales["Sensitivity"],
                output_size=config.output_size,
            )
            pixels_all.append(pixels)
            grids_all.append(grids)
            flags_all.append(flags)
            seeds.append(seed)
        encoded = encode_views(
            backbone,
            torch.cat(pixels_all),
            torch.cat(grids_all),
            torch.cat(flags_all),
            batch_size=config.extraction_batch,
            fixed_forward_batch=config.fixed_forward_batch,
        ).reshape(len(group), 81, width)
        center, states = split_state_family(encoded, seeds)
        arrays["center"][start : start + len(group)] = center.numpy()
        arrays["states"][start : start + len(group)] = states.numpy()
        if start % 128 == 0:
            arrays["center"].flush()
            arrays["states"].flush()

    write_rows(Path(output) / "metadata.csv", rows)
    for array in arrays.values():
        array.flush()


def extract_reader_past(
    *,
    dataset: str | Path,
    anchors_path: str | Path,
    checkpoint: str | Path,
    scale: str,
    output: str | Path,
    config: PaperConfig,
    device: str = "cuda",
) -> None:
    """Extract canonical previous-frame features aligned with fixed reader anchors."""
    _configure_numerics()
    data = load_tapvid(dataset)
    anchors = sorted(
        (row for row in read_rows(anchors_path) if int(row["fixed_support_index"]) >= 0),
        key=lambda row: int(row["fixed_support_index"]),
    )
    width = int(MODEL_SPECS[scale]["width"])
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    past = np.lib.format.open_memmap(
        destination / "reader_past.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(anchors), width),
    )
    valid = np.zeros(len(anchors), dtype=bool)
    backbone = DinoV3Backbone(scale, checkpoint, device=device)
    metadata = []
    pending_pixels = []
    pending_rows = []

    def flush_pending() -> None:
        if not pending_pixels:
            return
        count = len(pending_pixels)
        encoded = encode_views(
            backbone,
            torch.from_numpy(np.stack(pending_pixels)),
            torch.zeros(count, 2),
            torch.ones(count, dtype=torch.bool),
            batch_size=config.extraction_batch,
            fixed_forward_batch=config.fixed_forward_batch,
        ).numpy()
        for target_row, feature in zip(pending_rows, encoded, strict=True):
            past[target_row] = feature
            valid[target_row] = True
        pending_pixels.clear()
        pending_rows.clear()

    for row_index, row in enumerate(anchors):
        metadata.append(
            {
                "reader_row": row_index,
                "source_interface": row["interface_id"],
                "sequence": row["sequence"],
                "fold": int(row["fold"]),
                "past_available": str(row["past"]).lower() == "true",
            }
        )
        if str(row["past"]).lower() != "true":
            continue
        frame_index = int(row["frame"]) - 1
        record = data[row["sequence"]]
        frame = record["video"][frame_index]
        height, width_pixels = frame.shape[:2]
        coordinates = record["points"][int(row["track"]), frame_index] * np.asarray(
            [width_pixels, height]
        )
        pending_pixels.append(
            canonical_crop(
                frame,
                coordinates,
                config.crop_side,
                config.output_size,
            )
        )
        pending_rows.append(row_index)
        if len(pending_pixels) >= config.extraction_batch:
            flush_pending()
    flush_pending()
    past.flush()
    np.save(destination / "reader_past_valid.npy", valid, allow_pickle=False)
    write_rows(destination / "reader_anchors.csv", metadata)


def extract_reader_context_variant(
    *,
    dataset: str | Path,
    anchors_path: str | Path,
    checkpoint: str | Path,
    scale: str,
    output: str | Path,
    config: PaperConfig,
    device: str = "cuda",
) -> None:
    """Extract the Context scale-1 family used by two inner reader folds.

    The main TAP-Vid feature store contains the source-only scale selected for
    the attribution experiments. The nested reader protocol uses one additional
    source-only Context setting for a small number of validation folds. This
    function materializes only that 16-state variant for the fixed reader
    anchors.
    """
    _configure_numerics()
    data = load_tapvid(dataset)
    anchors = sorted(
        (row for row in read_rows(anchors_path) if int(row["fixed_support_index"]) >= 0),
        key=lambda row: int(row["fixed_support_index"]),
    )
    width = int(MODEL_SPECS[scale]["width"])
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    states = np.lib.format.open_memmap(
        destination / "reader_context_scale1.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(anchors), 16, width),
    )
    backbone = DinoV3Backbone(scale, checkpoint, device=device)
    metadata: list[dict[str, object]] = []
    for start in range(0, len(anchors), 4):
        group = anchors[start : start + 4]
        pixels: list[torch.Tensor] = []
        for row in group:
            frame = data[row["sequence"]]["video"][int(row["frame"])]
            center = canonical_crop(
                frame,
                (float(row["x"]), float(row["y"])),
                config.crop_side,
                config.output_size,
            )
            generated = generated_observation_families(
                torch.from_numpy(center),
                geometric_scale=2.0,
                context_scale=1.0,
                sensitivity_scale=2.0,
                seed=int(row["source_seed"]),
            )["Context"]
            pixels.append(
                (generated * 255)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
            )
        batch = torch.cat(pixels)
        encoded = encode_views(
            backbone,
            batch,
            torch.zeros(len(batch), 2),
            torch.zeros(len(batch), dtype=torch.bool),
            batch_size=config.extraction_batch,
            fixed_forward_batch=config.fixed_forward_batch,
        ).reshape(len(group), 16, width)
        states[start : start + len(group)] = encoded.numpy()
        for row_index, row in enumerate(group, start=start):
            metadata.append(
                {
                    "reader_row": row_index,
                    "source_interface": row["interface_id"],
                    "sequence": row["sequence"],
                    "fold": int(row["fold"]),
                }
            )
        if start % 128 == 0:
            states.flush()
    states.flush()
    write_rows(destination / "reader_context_scale1.csv", metadata)
