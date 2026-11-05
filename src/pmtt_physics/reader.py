"""Nested sequence reader used as the signed-flow diagnostic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import PaperConfig
from .geometry import (
    construct_geometries,
    factor_geometry,
    project,
    residual,
    tangent,
)
from .observations import acquisition_pools
from .records import MetricRecorder
from .reporting import write_metric_tables
from .storage import FeatureStore, read_rows, write_rows


READER_METHODS = (
    "Orbital16",
    "SamplingMatched16",
    "Lawful32",
    "Lawful64",
    "Sensitivity16",
    "GenericGeometry",
)
_POOL_NAMES = {
    "Orbital16": "Orbital",
    "SamplingMatched16": "SamplingMatched",
    "Lawful32": "Lawful32",
    "Lawful64": "Lawful64",
    "Sensitivity16": "Sensitivity",
}


def fit_center_map(design: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    """Fit the shared linear Center proposal with the released ridge objective."""
    if design.ndim != 2 or target.ndim != 2 or len(design) != len(target):
        raise ValueError("design and target must be aligned matrices")
    penalty = float(ridge) * len(design)
    if len(design) < design.shape[1]:
        identity = torch.eye(len(design), device=design.device, dtype=design.dtype)
        return design.T @ torch.linalg.solve(design @ design.T + penalty * identity, target)
    identity = torch.eye(design.shape[1], device=design.device, dtype=design.dtype)
    return torch.linalg.solve(design.T @ design + penalty * identity, design.T @ target)


def fit_scalar_channels(
    channels: torch.Tensor,
    target: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Fit at most three scalar mixture coefficients."""
    if channels.ndim != 3 or target.ndim != 2 or len(channels) != len(target):
        raise ValueError("channels and target must be aligned")
    value = channels.double()
    response = target.double()
    gram = torch.einsum("nkd,njd->kj", value, value)
    rhs = torch.einsum("nkd,nd->k", value, response)
    regularizer = float(ridge) * gram.diag().mean().clamp_min(1e-12)
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram + regularizer * identity, rhs).float()


def center_design(center: torch.Tensor, past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    past_direction = tangent(center, center - past)
    design = torch.cat(
        [center, past_direction, torch.ones(len(center), 1, device=center.device)],
        dim=1,
    )
    return design, past_direction


def _support_directions(
    center: torch.Tensor,
    states: torch.Tensor,
    seeds: Sequence[int],
) -> dict[str, torch.Tensor]:
    pools = acquisition_pools()
    selected = {method: pools[pool] for method, pool in _POOL_NAMES.items()}
    geometries = construct_geometries(center, states, seeds, pools=selected)
    output = {method: geometry["q"][..., :4] for method, geometry in geometries.items()}

    rows = residual(
        center,
        torch.cat(
            [states[:, 2], states[:, 4], states[:, 3], states[:, 1]],
            dim=1,
        ),
    )
    grouped = rows.reshape(len(center), 4, 16, -1)
    norm = grouped.square().sum(dim=(-1, -2), keepdim=True).sqrt()
    generic = factor_geometry((grouped / norm.clamp_min(1e-6)).reshape(len(center), 64, -1))
    output["GenericGeometry"] = generic["q"][..., :4]
    return output


def reader_channels(
    center_proposal: torch.Tensor,
    past_direction: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    inside = project(directions, center_proposal)
    outside = center_proposal - inside
    past_inside = project(directions, past_direction)
    return torch.stack([inside, outside, past_inside], dim=1)


def _bool(value: str | bool) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _sequence_set(value: str) -> set[str]:
    return {item for item in str(value).split("|") if item}


def _load_past(root: str | Path, anchors: Sequence[Mapping[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    location = Path(root)
    metadata = read_rows(location / "reader_anchors.csv")
    if len(metadata) != len(anchors):
        raise ValueError("reader-past metadata does not match the fixed anchor set")
    for expected, actual in zip(anchors, metadata, strict=True):
        if expected["interface_id"] != actual["source_interface"]:
            raise ValueError("reader-past ordering does not match fixed_support_index")
    past = np.load(location / "reader_past.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(location / "reader_past_valid.npy", allow_pickle=False)
    if past.ndim != 2 or len(past) != len(anchors) or valid.shape != (len(anchors),):
        raise ValueError("invalid reader-past arrays")
    return past, valid.astype(bool)


def _load_context_variant(
    root: str | Path | None,
    anchors: Sequence[Mapping[str, str]],
) -> np.ndarray | None:
    if root is None:
        return None
    location = Path(root)
    metadata = read_rows(location / "reader_context_scale1.csv")
    if len(metadata) != len(anchors):
        raise ValueError("Context variant metadata does not match the fixed anchor set")
    for expected, actual in zip(anchors, metadata, strict=True):
        if expected["interface_id"] != actual["source_interface"]:
            raise ValueError("Context variant ordering does not match fixed_support_index")
    value = np.load(location / "reader_context_scale1.npy", mmap_mode="r", allow_pickle=False)
    if value.ndim != 3 or value.shape[:2] != (len(anchors), 16):
        raise ValueError("invalid Context scale-1 feature array")
    return value


def _scale_lookup(path: str | Path) -> dict[tuple[int, int, str, str], float]:
    output: dict[tuple[int, int, str, str], float] = {}
    for row in read_rows(path):
        fit_role = row["fit_role"]
        inner = row["inner_fold"] if fit_role == "inner" else ""
        key = (
            int(row["outer_fold"]),
            int(row["horizon"]),
            fit_role,
            inner + ":" + row["family"],
        )
        output[key] = float(row["scale"])
    return output


def _states_for_role(
    store: FeatureStore,
    source_store_indices: Sequence[int],
    reader_rows: Sequence[int],
    *,
    fold: int,
    horizon: int,
    fit_role: str,
    inner_fold: int | None,
    scales: Mapping[tuple[int, int, str, str], float],
    context_scale1: np.ndarray | None,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    states = torch.from_numpy(
        np.array(store.arrays["states"][list(source_store_indices)], copy=True)
    ).to(device)
    suffix = "" if fit_role == "refit" else str(int(inner_fold))
    context_scale = scales[(fold, horizon, fit_role, suffix + ":Context")]
    if context_scale == 1.0:
        if context_scale1 is None:
            raise ValueError(
                "full scalar refitting requires the Context scale-1 reader variant; "
                "run extract-reader-context first"
            )
        states[:, 3] = torch.from_numpy(
            np.array(context_scale1[list(reader_rows)], copy=True)
        ).to(device)
    elif context_scale != 2.0:
        raise ValueError(f"unsupported Context scale in reader protocol: {context_scale}")
    seeds = [int(store.metadata[index]["source_seed"]) for index in source_store_indices]
    return states, seeds


def _published_coefficients(path: str | Path) -> dict[tuple[int, int, str], torch.Tensor]:
    grouped: dict[tuple[int, int, str], list[tuple[int, float]]] = {}
    for row in read_rows(path):
        key = (int(row["fold"]), int(row["horizon"]), row["method"])
        grouped.setdefault(key, []).append((int(row["coefficient"]), float(row["value"])))
    result = {}
    for key, values in grouped.items():
        values.sort()
        if [index for index, _ in values] != [0, 1, 2]:
            raise ValueError(f"reader coefficients are incomplete for {key}")
        result[key] = torch.tensor([value for _, value in values], dtype=torch.float32)
    return result


def analyze_reader(
    *,
    store_root: str | Path,
    anchors_path: str | Path,
    past_root: str | Path,
    coefficients_path: str | Path,
    cross_validation_path: str | Path,
    observation_scales_path: str | Path,
    output: str | Path,
    config: PaperConfig,
    mode: str = "published_scalars",
    context_variant_root: str | Path | None = None,
    device: str = "cuda",
) -> None:
    """Run the sequence-disjoint fixed-subset reader diagnostic.

    ``published_scalars`` uses the released three-scalar coefficients and
    refits only the Center proposal from the public training folds.
    ``refit_scalars`` reconstructs the complete nested out-of-fold scalar fit
    and requires the Context scale-1 variant.
    """
    if mode not in {"published_scalars", "refit_scalars"}:
        raise ValueError("mode must be 'published_scalars' or 'refit_scalars'")
    store = FeatureStore.open(store_root)
    if store.kind != "tapvid" or store.scale != "B":
        raise ValueError("the reader diagnostic uses the B-scale TAP-Vid store")
    device_value = torch.device(device)
    anchors = sorted(
        (row for row in read_rows(anchors_path) if int(row["fixed_support_index"]) >= 0),
        key=lambda row: int(row["fixed_support_index"]),
    )
    if len(anchors) != 3840:
        raise ValueError(f"expected 3,840 fixed reader anchors, received {len(anchors)}")
    source_store_indices = [store.index[row["interface_id"]] for row in anchors]
    centers = torch.from_numpy(
        np.array(store.arrays["center"][source_store_indices], copy=True)
    ).to(device_value)
    past_array, past_valid = _load_past(past_root, anchors)
    past = torch.from_numpy(np.array(past_array, copy=True)).to(device_value)
    context_scale1 = _load_context_variant(context_variant_root, anchors)
    scales = _scale_lookup(observation_scales_path)
    cross_rows = read_rows(cross_validation_path)
    published = _published_coefficients(coefficients_path)

    sequences = np.asarray([row["sequence"] for row in anchors])
    folds = np.asarray([int(row["fold"]) for row in anchors])
    source_ids = np.asarray(source_store_indices)
    recorder = MetricRecorder()
    coefficient_rows: list[dict[str, Any]] = []

    for fold in range(5):
        for horizon in (1, 4):
            horizon_valid = np.asarray([_bool(row[f"h{horizon}"]) for row in anchors])
            valid = past_valid & horizon_valid
            train_rows = np.flatnonzero((folds != fold) & valid)
            test_rows = np.flatnonzero((folds == fold) & valid)
            if not len(train_rows) or not len(test_rows):
                raise ValueError(f"empty reader split for fold={fold}, horizon={horizon}")
            if set(sequences[train_rows]) & set(sequences[test_rows]):
                raise ValueError("reader folds are not sequence-disjoint")

            train_center = centers[train_rows]
            train_past = past[train_rows]
            train_design, train_past_direction = center_design(train_center, train_past)
            train_target_ids = [
                store.index[
                    f"{anchors[row]['sequence']}:{int(anchors[row]['track'])}:"
                    f"{int(anchors[row]['frame']) + horizon}"
                ]
                for row in train_rows
            ]
            train_future = store.center(train_target_ids, device=device_value)
            train_target = tangent(train_center, train_future - train_center)
            out_of_fold = torch.empty_like(train_target)
            out_of_fold_channels = {
                method: torch.empty(
                    len(train_rows), 3, train_target.shape[1], device=device_value
                )
                for method in READER_METHODS
            }
            local_position = {int(row): position for position, row in enumerate(train_rows)}

            split_rows = [
                row
                for row in cross_rows
                if int(row["fold"]) == fold and int(row["horizon"]) == horizon
            ]
            if len(split_rows) != 4:
                raise ValueError(f"expected four inner folds for fold={fold}, horizon={horizon}")
            for split in sorted(split_rows, key=lambda row: int(row["inner_fold"])):
                inner = int(split["inner_fold"])
                fit_sequences = _sequence_set(split["train_sequences"])
                validation_sequences = _sequence_set(split["validation_sequences"])
                test_sequences = _sequence_set(split["test_sequences"])
                if test_sequences != set(sequences[test_rows]):
                    raise ValueError("reader test-sequence identity mismatch")
                fit_rows = np.asarray(
                    [row for row in train_rows if sequences[row] in fit_sequences], dtype=int
                )
                validation_rows = np.asarray(
                    [row for row in train_rows if sequences[row] in validation_sequences],
                    dtype=int,
                )
                if not len(fit_rows) or not len(validation_rows):
                    raise ValueError("empty nested reader split")
                fit_positions = [local_position[int(row)] for row in fit_rows]
                validation_positions = [local_position[int(row)] for row in validation_rows]
                center_map = fit_center_map(
                    train_design[fit_positions], train_target[fit_positions], config.scalar_ridge
                )
                proposal = train_design[validation_positions] @ center_map
                out_of_fold[validation_positions] = proposal
                if mode == "refit_scalars":
                    role_states, role_seeds = _states_for_role(
                        store,
                        source_ids[validation_rows],
                        validation_rows,
                        fold=fold,
                        horizon=horizon,
                        fit_role="inner",
                        inner_fold=inner,
                        scales=scales,
                        context_scale1=context_scale1,
                        device=device_value,
                    )
                    directions = _support_directions(
                        centers[validation_rows], role_states, role_seeds
                    )
                    for method in READER_METHODS:
                        out_of_fold_channels[method][validation_positions] = reader_channels(
                            proposal,
                            train_past_direction[validation_positions],
                            directions[method],
                        )

            center_map = fit_center_map(train_design, train_target, config.scalar_ridge)
            test_center = centers[test_rows]
            test_past = past[test_rows]
            test_design, test_past_direction = center_design(test_center, test_past)
            test_proposal = test_design @ center_map
            test_target_ids = [
                store.index[
                    f"{anchors[row]['sequence']}:{int(anchors[row]['track'])}:"
                    f"{int(anchors[row]['frame']) + horizon}"
                ]
                for row in test_rows
            ]
            test_future = store.center(test_target_ids, device=device_value)
            test_target = tangent(test_center, test_future - test_center)
            energy = test_target.square().sum(dim=1)
            units = sequences[test_rows].tolist()

            damped = fit_scalar_channels(train_past_direction[:, None], train_target, config.scalar_ridge)
            center_scalar = fit_scalar_channels(out_of_fold[:, None], train_target, config.scalar_ridge)
            for method, prediction in (
                ("Zero", torch.zeros_like(test_target)),
                ("DampedPast", damped[0] * test_past_direction),
                ("CenterScalar", center_scalar[0] * test_proposal),
            ):
                recorder.add(
                    "prediction_diagnostic",
                    f"real_reader_h{horizon}",
                    "FrozenReader",
                    method,
                    "normalized_error",
                    units,
                    (prediction - test_target).square().sum(dim=1),
                    energy,
                )

            test_states, test_seeds = _states_for_role(
                store,
                source_ids[test_rows],
                test_rows,
                fold=fold,
                horizon=horizon,
                fit_role="refit",
                inner_fold=None,
                scales=scales,
                context_scale1=context_scale1,
                device=device_value,
            )
            test_directions = _support_directions(test_center, test_states, test_seeds)
            for method in READER_METHODS:
                if mode == "refit_scalars":
                    weights = fit_scalar_channels(
                        out_of_fold_channels[method], train_target, config.scalar_ridge
                    )
                else:
                    weights = published[(fold, horizon, method)].to(device_value)
                for index, value in enumerate(weights):
                    coefficient_rows.append(
                        {
                            "fold": fold,
                            "horizon": horizon,
                            "method": method,
                            "coefficient": index,
                            "value": float(value),
                            "mode": mode,
                        }
                    )
                channels = reader_channels(
                    test_proposal, test_past_direction, test_directions[method]
                )
                prediction = torch.einsum("nkd,k->nd", channels, weights)
                target_inside = project(test_directions[method], test_target)
                prediction_inside = project(test_directions[method], prediction)
                total = (prediction - test_target).square().sum(dim=1)
                support_miss = (test_target - target_inside).square().sum(dim=1)
                inside_error = (prediction_inside - target_inside).square().sum(dim=1)
                outside_error = (
                    (prediction - prediction_inside) - (test_target - target_inside)
                ).square().sum(dim=1)
                if not torch.allclose(total, inside_error + outside_error, atol=2e-5, rtol=2e-4):
                    raise ValueError("reader error decomposition failed")
                for metric, numerator in (
                    ("normalized_error", total),
                    ("support_miss", support_miss),
                    ("flow_error_inside", inside_error),
                    ("outside_prediction_error", outside_error),
                    ("projected_prediction_error", support_miss + inside_error),
                ):
                    recorder.add(
                        "prediction_diagnostic",
                        f"real_reader_h{horizon}",
                        "FrozenReader",
                        method,
                        metric,
                        units,
                        numerator,
                        energy,
                    )

    output_path = Path(output)
    write_metric_tables(
        recorder,
        scale="B",
        substrate="real_reader_subset",
        output=output_path,
        config=config,
    )
    write_rows(output_path / "coefficients.csv", coefficient_rows)
