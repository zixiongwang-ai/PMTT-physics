"""Physical-state matrices and attribution measurements."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from .geometry import (
    EPS,
    bounded_error,
    construct_geometries,
    explained_energy,
    factor_geometry,
    orthonormalize,
    project,
    random_basis,
    subspace_overlap,
    take_geometry,
    tangent,
    transport_directions,
)
from .observations import acquisition_pools
from .records import MetricRecorder

PRIMARY_METHODS = ("Orbital", "SamplingMatched", "Lawful32", "Lawful64", "Sensitivity")


def nearest_wrong_states(
    canonical: torch.Tensor,
    rgb: torch.Tensor,
    projected_geometry: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    states = canonical.shape[0]
    distances = {
        "DINO": -(canonical.permute(1, 0, 2) @ canonical.permute(1, 2, 0)),
        "RGB": torch.cdist(rgb.permute(1, 0, 2), rgb.permute(1, 0, 2)).square(),
        "Geometry": torch.cdist(
            projected_geometry.permute(1, 0, 2),
            projected_geometry.permute(1, 0, 2),
        ).square(),
    }
    selectors: dict[str, torch.Tensor] = {}
    for name, distance in distances.items():
        value = distance.clone()
        indices = torch.arange(states, device=value.device)
        value[:, indices, indices] = torch.inf
        selectors[name] = value.argmin(dim=-1).T
    return selectors, distances


def exact_retrieval_permutation(matrix: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    states = len(matrix)
    prediction = matrix.argmax(axis=1)
    accuracy = float(np.mean(prediction == np.arange(states)))
    permutations = np.asarray(list(itertools.permutations(range(states))))
    null = np.mean(permutations[:, prediction] == np.arange(states), axis=1)
    return accuracy, float(np.mean(null >= accuracy - 1e-12)), float(null.mean()), prediction


def _matrix_batch(
    geometries: Sequence[dict[str, torch.Tensor]],
    canonical: torch.Tensor,
    future: torch.Tensor,
    future_residual: torch.Tensor,
    oracles: Sequence[Sequence[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states, points, width = canonical.shape
    source_directions = torch.stack([geometry["q"][..., :4] for geometry in geometries])
    target_center = canonical[:, None].expand(states, states, points, width).reshape(-1, width)
    source_directions = (
        source_directions[None]
        .expand(states, states, points, width, 4)
        .reshape(-1, width, 4)
    )
    directions = transport_directions(source_directions, target_center).reshape(
        states, states, points, width, 4
    )
    rank = (directions.square().sum(dim=-2) > 0.5).sum(dim=-1).min(dim=1).values
    mask = (
        torch.arange(4, device=canonical.device)[None, None] < rank[:, :, None]
    )[:, None, :, None]
    directions = directions * mask
    numerators = (future_residual[:, None] @ directions).square().sum(dim=(-1, -2))
    denominators = future_residual.square().sum(dim=(-1, -2))

    stacked = {key: torch.stack([geometry[key] for geometry in geometries]) for key in ("q", "s")}
    source_geometry = {
        key: value[None]
        .expand(states, *value.shape)
        .reshape(states * states * points, *value.shape[2:])
        for key, value in stacked.items()
    }
    delta = (future[:, None] - canonical[None, :, :, None]).reshape(
        states * states * points, 4, width
    )
    error, _ = bounded_error(source_geometry, delta)
    center_error = (future - canonical[:, :, None]).square().sum(dim=(-1, -2))
    bounded = center_error[:, None] - error.reshape(states, states, points, 4).sum(dim=-1)

    oracle_directions = torch.stack(
        [torch.stack(branches, dim=1) for branches in oracles]
    )
    cross = directions.transpose(-1, -2)[:, :, :, None] @ oracle_directions[:, None]
    valid = torch.minimum(
        directions.square().sum(dim=(-1, -2))[:, :, :, None],
        oracle_directions.square().sum(dim=(-1, -2))[:, None],
    )
    oracle_values = (
        cross.square().sum(dim=(-1, -2)) / valid.clamp_min(1) * (valid > 0)
    ).sum(dim=-1)
    oracle_valid = (valid > 0).sum(dim=-1).float()
    ranks = rank[:, None].expand(states, states, points).float()
    return numerators, denominators, bounded, oracle_values, oracle_valid, ranks


def evaluate_physical_block(
    canonical: torch.Tensor,
    observation_states: torch.Tensor,
    future: torch.Tensor,
    rgb: torch.Tensor,
    projected_geometry: torch.Tensor,
    seeds: Sequence[int],
    point_ids: Sequence[int],
    state_values: Sequence[float],
    recorder: MetricRecorder,
    *,
    base_id: str,
    block: str,
    constructions: Sequence[str] = ("Original", "PairDifference"),
    seed: int = 20261103,
) -> dict[str, object]:
    """Evaluate one base and one physical block.

    Shapes are ``canonical[S,P,D]``, ``observation_states[S,P,5,16,D]``,
    ``future[S,P,4,D]``, ``rgb[S,P,R]``, and ``projected_geometry[S,P,G]``.
    """
    states, points, _ = canonical.shape
    if len(point_ids) != points or len(state_values) != states:
        raise ValueError("point and state metadata do not match feature tensors")
    selectors, distances = nearest_wrong_states(canonical, rgb, projected_geometry)
    future_residual = tangent(canonical[:, :, None], future - canonical[:, :, None])
    future_energy = future_residual.square().sum(dim=-1)

    oracles: list[list[torch.Tensor]] = []
    for state_index in range(states):
        branches = []
        for held_out in range(4):
            rows = future_residual[
                state_index, :, [index for index in range(4) if index != held_out]
            ].transpose(1, 2)
            directions, _ = orthonormalize(rows)
            branches.append(directions)
        oracles.append(branches)

    public_pools = acquisition_pools()
    matrix_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    collected: dict[tuple[str, str], dict[str, float]] = {}

    for criterion, selector in selectors.items():
        for target in range(states):
            for point in range(points):
                wrong = int(selector[target, point])
                selector_rows.append(
                    {
                        "base_id": base_id,
                        "block": block,
                        "target_state": target,
                        "point_id": int(point_ids[point]),
                        "criterion": criterion,
                        "wrong_state": wrong,
                        "distance": float(distances[criterion][point, target, wrong]),
                    }
                )

    for construction in constructions:
        pools = (
            public_pools
            if construction == "Original"
            else {name: public_pools[name] for name in PRIMARY_METHODS}
        )
        geometry_all = construct_geometries(
            canonical.reshape(states * points, -1),
            observation_states.reshape(states * points, 5, 16, -1),
            seeds,
            construction,
            pools,
        )
        for method, complete_geometry in geometry_all.items():
            if construction != "Original" and method not in PRIMARY_METHODS:
                continue
            geometries = [
                take_geometry(complete_geometry, slice(index * points, (index + 1) * points))
                for index in range(states)
            ]
            numerators, denominators, bounded, oracle_values, oracle_valid, ranks = _matrix_batch(
                geometries, canonical, future, future_residual, oracles
            )
            matrix = (
                numerators.sum(dim=-1)
                / denominators.sum(dim=-1)[:, None].clamp_min(EPS)
            ).cpu().numpy()
            full_error = (future - canonical[:, :, None]).square().sum(dim=(-1, -2))
            units = [base_id]
            for target in range(states):
                for source in range(states):
                    entries = (
                        ("TEE", numerators[target, source].sum(), denominators[target].sum(), 4),
                        ("oracle_overlap", oracle_values[target, source].sum(), oracle_valid[target, source].sum(), 4),
                        ("bounded_coverage", bounded[target, source].sum(), full_error[target].sum(), 8),
                    )
                    for metric, numerator, denominator, rank_value in entries:
                        denominator_float = float(denominator)
                        matrix_rows.append(
                            {
                                "base_id": base_id,
                                "block": block,
                                "construction": construction,
                                "method": method,
                                "metric": metric,
                                "target_state": target,
                                "source_state": source,
                                "target_value": float(state_values[target]),
                                "source_value": float(state_values[source]),
                                "numerator": float(numerator),
                                "denominator": denominator_float,
                                "value": float(numerator) / denominator_float
                                if denominator_float > EPS
                                else "",
                                "actual_rank": float(ranks[target, source].mean()),
                            }
                        )
                        if denominator_float > EPS:
                            recorder.add(
                                "physical_matrix",
                                block,
                                construction,
                                method,
                                metric,
                                units,
                                [float(numerator)],
                                [denominator_float],
                                rank_value,
                                target,
                                source,
                            )

            off_diagonal = matrix.copy()
            np.fill_diagonal(off_diagonal, -np.inf)
            diagonal = matrix.diagonal()
            best_wrong = off_diagonal.max(axis=1)
            mean_wrong = (matrix.sum(axis=1) - diagonal) / (states - 1)
            accuracy, exact_p, chance, predictions = exact_retrieval_permutation(matrix)
            rank_position = 1 + (matrix > diagonal[:, None]).sum(axis=1)
            for metric, value in (
                ("diagonal_margin", np.mean(diagonal - best_wrong)),
                ("mean_wrong_margin", np.mean(diagonal - mean_wrong)),
                ("retrieval_top1", accuracy),
                ("retrieval_top2", np.mean(rank_position <= 2)),
                ("retrieval_mrr", np.mean(1 / rank_position)),
                ("retrieval_above_chance", accuracy - chance),
            ):
                recorder.add(
                    "physical_state", block, construction, method, metric, units, [value]
                )
            retrieval_rows.append(
                {
                    "base_id": base_id,
                    "block": block,
                    "construction": construction,
                    "method": method,
                    "accuracy": accuracy,
                    "chance": chance,
                    "within_unit_exact_p": exact_p,
                    "predictions": "|".join(map(str, predictions)),
                    "states": states,
                }
            )

            physical_distance = np.abs(
                np.asarray(state_values)[:, None] - np.asarray(state_values)[None]
            )
            if block == "axis":
                physical_distance = np.minimum(physical_distance, 180 - physical_distance)
            if block != "motion":
                mask = ~np.eye(states, dtype=bool)
                correlation = spearmanr(physical_distance[mask], matrix[mask]).statistic
                if np.isfinite(correlation):
                    recorder.add(
                        "physical_state",
                        block,
                        construction,
                        method,
                        "distance_spearman",
                        units,
                        [correlation],
                    )

            matched_index = torch.arange(states, device=canonical.device)
            for criterion, selector in selectors.items():
                matched = numerators[matched_index, matched_index]
                wrong = torch.stack(
                    [
                        numerators[target, selector[target], torch.arange(points, device=canonical.device)]
                        for target in range(states)
                    ]
                )
                recorder.add(
                    "physical_state",
                    block,
                    construction,
                    method,
                    f"{criterion}_gap",
                    units,
                    [float((matched - wrong).sum())],
                    [float(denominators.sum())],
                )
                matched_oracle = oracle_values[matched_index, matched_index]
                wrong_oracle = torch.stack(
                    [
                        oracle_values[target, selector[target], torch.arange(points, device=canonical.device)]
                        for target in range(states)
                    ]
                )
                oracle_denominator = oracle_valid[matched_index, matched_index]
                recorder.add(
                    "physical_state",
                    block,
                    construction,
                    method,
                    f"{criterion}_oracle_gap",
                    units,
                    [float((matched_oracle - wrong_oracle).sum())],
                    [float(oracle_denominator.sum())],
                )
                matched_bounded = bounded[matched_index, matched_index]
                wrong_bounded = torch.stack(
                    [
                        bounded[target, selector[target], torch.arange(points, device=canonical.device)]
                        for target in range(states)
                    ]
                )
                recorder.add(
                    "physical_state",
                    block,
                    construction,
                    method,
                    f"{criterion}_bounded_gap",
                    units,
                    [float((matched_bounded - wrong_bounded).sum())],
                    [float(full_error.sum())],
                    rank=8,
                )

            dino_selector = selectors["DINO"]
            dino_wrong = torch.stack(
                [
                    numerators[target, dino_selector[target], torch.arange(points, device=canonical.device)]
                    for target in range(states)
                ]
            )
            collected[construction, method] = {
                "diagonal": float(np.mean(diagonal - best_wrong)),
                "dino": float(
                    (
                        numerators[matched_index, matched_index].sum()
                        - dino_wrong.sum()
                    )
                    / denominators.sum().clamp_min(EPS)
                ),
            }
            if method == "Lawful32" and construction in ("Original", "PairDifference"):
                residual_rows.extend(
                    evaluate_state_specific_residual(
                        canonical,
                        geometries,
                        future_residual,
                        dino_selector,
                        oracles,
                        recorder,
                        base_id=base_id,
                        block=block,
                        construction=construction,
                        seed=seed,
                    )
                )

        if (construction, "Lawful32") in collected and (construction, "Sensitivity") in collected:
            for metric in ("diagonal", "dino"):
                recorder.add(
                    "physical_state",
                    block,
                    construction,
                    "Lawful32-Sensitivity",
                    f"specificity_{metric}",
                    [base_id],
                    [
                        collected[construction, "Lawful32"][metric]
                        - collected[construction, "Sensitivity"][metric]
                    ],
                )

    return {
        "matrix_rows": matrix_rows,
        "selector_rows": selector_rows,
        "retrieval_rows": retrieval_rows,
        "residual_rows": residual_rows,
        "future_energy": future_energy,
    }


def evaluate_state_specific_residual(
    canonical: torch.Tensor,
    geometries: Sequence[dict[str, torch.Tensor]],
    future_residual: torch.Tensor,
    closest_wrong: torch.Tensor,
    oracles: Sequence[Sequence[torch.Tensor]],
    recorder: MetricRecorder,
    *,
    base_id: str,
    block: str,
    construction: str,
    seed: int,
) -> list[dict[str, object]]:
    states, points, _ = canonical.shape
    records: list[dict[str, object]] = []
    for target in range(states):
        other_rows = []
        for source in range(states):
            if source == target:
                continue
            rows = tangent(canonical[target, :, None], geometries[source]["r"])
            rows = rows / rows.square().sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
            other_rows.append(rows)
        shared = factor_geometry(torch.cat(other_rows, dim=1))["q"]
        shared8 = shared[..., :8]
        matched = transport_directions(geometries[target]["q"][..., :4], canonical[target])
        wrong_index = closest_wrong[target]
        raw_wrong = torch.stack(
            [geometries[int(wrong_index[point])]["q"][point, :, :4] for point in range(points)]
        )
        wrong = transport_directions(raw_wrong, canonical[target])
        matched_unique, matched_scale = orthonormalize(
            matched - shared8 @ (shared8.transpose(1, 2) @ matched), absolute_threshold=1e-4
        )
        wrong_unique, wrong_scale = orthonormalize(
            wrong - shared8 @ (shared8.transpose(1, 2) @ wrong), absolute_threshold=1e-4
        )
        rank = torch.minimum(
            (matched_scale > 0).sum(dim=1), (wrong_scale > 0).sum(dim=1)
        ).clamp(max=4)
        mask = (
            torch.arange(4, device=canonical.device)[None] < rank[:, None]
        )[:, None]
        matched_unique = matched_unique * mask
        wrong_unique = wrong_unique * mask
        random = random_basis(canonical[target], 4, seed + target, shared8) * mask
        leftover = future_residual[target] - project(shared8, future_residual[target])
        denominator = leftover.square().sum(dim=(1, 2))
        total = future_residual[target].square().sum(dim=(1, 2))
        units = [base_id] * points
        recorder.add(
            "state_specific",
            block,
            construction,
            "Lawful32",
            "remaining_energy",
            units,
            denominator,
            total,
        )
        recorder.add(
            "state_specific", block, construction, "Lawful32", "unique_rank", units, rank
        )
        for label, directions in (
            ("Matched", matched_unique),
            ("Wrong", wrong_unique),
            ("Random", random),
        ):
            value = explained_energy(directions, leftover).sum(dim=1)
            valid = (rank > 0).float()
            recorder.add(
                "state_specific",
                block,
                construction,
                label,
                "TEE",
                units,
                value * valid,
                denominator * valid,
            )
            overlap_values = []
            overlap_valid = []
            for oracle in oracles[target]:
                oracle_unique, _ = orthonormalize(
                    oracle - shared8 @ (shared8.transpose(1, 2) @ oracle),
                    absolute_threshold=1e-4,
                )
                overlap, valid_denominator = subspace_overlap(directions, oracle_unique)
                overlap_values.append(overlap * (valid_denominator > 0))
                overlap_valid.append((valid_denominator > 0).float())
            recorder.add(
                "state_specific",
                block,
                construction,
                label,
                "oracle_overlap",
                units,
                torch.stack(overlap_values).sum(dim=0),
                torch.stack(overlap_valid).sum(dim=0),
            )
        valid = rank > 0
        recorder.add(
            "state_specific",
            block,
            construction,
            "Lawful32",
            "matched_minus_wrong",
            units,
            (
                explained_energy(matched_unique, leftover)
                - explained_energy(wrong_unique, leftover)
            ).sum(dim=1)
            * valid,
            denominator * valid,
        )
        recorder.add(
            "state_specific",
            block,
            construction,
            "Lawful32",
            "matched_minus_random",
            units,
            (
                explained_energy(matched_unique, leftover)
                - explained_energy(random, leftover)
            ).sum(dim=1)
            * valid,
            denominator * valid,
        )
        for point in range(points):
            records.append(
                {
                    "base_id": base_id,
                    "block": block,
                    "construction": construction,
                    "state_index": target,
                    "point_index": point,
                    "wrong_state": int(wrong_index[point]),
                    "unique_rank": int(rank[point]),
                    "remaining_energy": float(denominator[point]),
                    "total_energy": float(total[point]),
                }
            )
    return records
