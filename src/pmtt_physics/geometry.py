"""Patch-level local geometry used by PMTT physics.

All operations are expressed through feature vectors and small Gram matrices. The
implementation never materializes a feature-dimension by feature-dimension
operator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import torch
import torch.nn.functional as F

from .observations import acquisition_pools, pair_indices

EPS = 1e-12
Geometry: TypeAlias = dict[str, torch.Tensor]


def finite(value: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(value).all():
        raise FloatingPointError("non-finite tensor")
    return value


def unit(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(EPS)


def tangent(center: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return value - center * (center * value).sum(dim=-1, keepdim=True)


def residual(center: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    return tangent(center[:, None], states - center[:, None])


def orthonormalize(
    factor: torch.Tensor,
    relative_threshold: float = 1e-4,
    absolute_threshold: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormalize columns using an FP64 Gram eigendecomposition."""
    value = factor.double()
    eigenvalues, eigenvectors = torch.linalg.eigh(value.transpose(-2, -1) @ value)
    eigenvalues = eigenvalues.flip(-1).clamp_min(0)
    eigenvectors = eigenvectors.flip(-1)
    singular = eigenvalues.sqrt()
    threshold = torch.maximum(
        singular[..., :1] * relative_threshold,
        torch.full_like(singular[..., :1], absolute_threshold),
    )
    active = singular > threshold
    directions = (value @ eigenvectors) / singular.clamp_min(EPS).unsqueeze(-2)
    directions = directions * active.unsqueeze(-2)
    return directions.float(), (singular * active).float()


def factor_geometry(rows: torch.Tensor, maximum_rank: int = 8) -> Geometry:
    """Summarize observation residuals as a centered bounded low-rank factor."""
    if rows.ndim != 3:
        raise ValueError("rows must have shape [batch, observations, features]")
    value = rows.double()
    gram = value @ value.transpose(-2, -1)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    singular = eigenvalues.flip(-1).clamp_min(0).sqrt()
    eigenvectors = eigenvectors.flip(-1)

    rank = min(maximum_rank, rows.shape[-2])
    singular = singular[..., :rank]
    eigenvectors = eigenvectors[..., :rank]
    active = singular > torch.maximum(
        singular[..., :1] * 1e-4,
        torch.full_like(singular[..., :1], 1e-6),
    )
    directions = (value.transpose(-2, -1) @ eigenvectors) / singular.clamp_min(EPS).unsqueeze(-2)
    directions = directions * active.unsqueeze(-2)

    radii = singular / (rows.shape[-2] ** 0.5)
    coordinates = (value @ directions) / radii.clamp_min(EPS).unsqueeze(-2)
    scale = coordinates.norm(dim=-1).amax(dim=-1, keepdim=True).clamp_min(1.0)
    radii = radii * scale * active

    directions = F.pad(directions.float(), (0, maximum_rank - rank))
    radii = F.pad(radii.float(), (0, maximum_rank - rank))
    spectrum = F.pad(
        (singular.square() / rows.shape[-2]).float(),
        (0, maximum_rank - rank),
    )
    return {
        "q": finite(directions),
        "s": finite(radii),
        "a": finite(directions * radii.unsqueeze(-2)),
        "spectrum": finite(spectrum),
        "rank": active.sum(dim=-1),
        "energy": rows.square().sum(dim=(-1, -2)) / rows.shape[-2],
        "r": rows,
    }


def construct_geometries(
    center: torch.Tensor,
    states: torch.Tensor,
    seeds: Sequence[int],
    construction: str = "Original",
    pools: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Geometry]:
    """Construct one geometry for each requested observation pool.

    Args:
        center: Unit canonical states with shape ``[batch, features]``.
        states: Observation states with shape ``[batch, 5, 16, features]``.
        seeds: Frozen source-view seeds, one per batch item.
        construction: ``Original``, ``PairDifference``, ``VisibleMean``, or
            ``SplitReference``.
        pools: Optional explicit mapping from public pool name to flattened
            observation indices.
    """
    if states.ndim != 4 or states.shape[1:3] != (5, 16):
        raise ValueError("states must have shape [batch, 5, 16, features]")
    if len(seeds) != len(center):
        raise ValueError("one seed is required for each center")

    flattened = states.flatten(1, 2)
    requested = acquisition_pools() if pools is None else dict(pools)
    output: dict[str, Geometry] = {}

    for name, indices_raw in requested.items():
        indices = list(map(int, indices_raw))
        visible = flattened[:, indices]

        if construction == "Original":
            rows = residual(center, visible)
            geometry = factor_geometry(rows)
        elif construction == "VisibleMean":
            rows = tangent(center[:, None], visible - visible.mean(dim=1, keepdim=True))
            geometry = factor_geometry(rows)
        elif construction == "PairDifference":
            differences = []
            for batch_index, seed in enumerate(seeds):
                pairs: list[tuple[int, int]] = []
                for family_index, family in enumerate(
                    ("Orbital", "SamplingMatched", "Geometric", "Context", "Sensitivity")
                ):
                    offset = family_index * 16
                    pairs.extend(
                        (offset + first, offset + second)
                        for first, second in pair_indices(family, int(seed))
                        if offset + first in indices and offset + second in indices
                    )
                if not pairs:
                    raise ValueError(f"no matched observation pairs for pool {name}")
                differences.append(
                    torch.stack(
                        [flattened[batch_index, first] - flattened[batch_index, second] for first, second in pairs]
                    )
                )
            rows = tangent(center[:, None], torch.stack(differences))
            geometry = factor_geometry(rows)
        elif construction == "SplitReference":
            split_a = [index for index in indices if index % 16 in (0, 2, 4, 6, 9, 11, 13, 15)]
            split_b = [index for index in indices if index not in split_a]
            if not split_a or not split_b:
                raise ValueError(f"pool {name} has an empty split")
            reference_a = unit(flattened[:, split_a].mean(dim=1))
            reference_b = unit(flattened[:, split_b].mean(dim=1))
            source = factor_geometry(residual(reference_a, flattened[:, split_a]))
            factor = tangent(
                reference_b[:, :, None].transpose(1, 2),
                source["a"].transpose(1, 2),
            ).transpose(1, 2)
            factor = tangent(center[:, None], factor.transpose(1, 2)).transpose(1, 2)
            directions, radii = orthonormalize(factor)
            geometry = {
                "q": directions,
                "s": radii,
                "a": factor,
                "rank": (radii > 0).sum(dim=1),
                "energy": source["energy"],
                "spectrum": source["spectrum"],
                "r": residual(center, flattened[:, split_a]),
            }
        else:
            raise ValueError(f"unknown construction: {construction}")
        output[name] = geometry

    return output


def project(directions: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return (values @ directions) @ directions.transpose(-2, -1)


def explained_energy(directions: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    return (residuals @ directions).square().sum(dim=-1)


def transport_directions(directions: torch.Tensor, target_center: torch.Tensor) -> torch.Tensor:
    transported = directions - target_center[:, :, None] * (
        target_center[:, :, None] * directions
    ).sum(dim=1, keepdim=True)
    return orthonormalize(transported)[0]


def subspace_overlap(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = torch.minimum(
        first.square().sum(dim=(-1, -2)),
        second.square().sum(dim=(-1, -2)),
    )
    value = (first.transpose(-2, -1) @ second).square().sum(dim=(-1, -2))
    value = value / denominator.clamp_min(1)
    return value, denominator


def principal_angle_median(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    singular = torch.linalg.svdvals(first.transpose(-2, -1) @ second).clamp(0, 1)
    return torch.rad2deg(torch.acos(singular)).median(dim=-1).values


def bounded_error(geometry: Geometry, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared error to the bounded local factor and coordinate norm."""
    directions = geometry["q"]
    radii = geometry["s"]
    projected = delta @ directions
    while radii.ndim < projected.ndim:
        radii = radii.unsqueeze(-2)
    radii_squared = radii.square()
    coordinates = radii * projected / radii_squared.clamp_min(EPS)
    inside = coordinates.square().sum(dim=-1) <= 1

    lower = torch.zeros_like(inside, dtype=delta.dtype)
    upper = (radii * projected).norm(dim=-1).clamp_min(EPS)
    for _ in range(36):
        midpoint = (lower + upper) * 0.5
        candidate = radii * projected / (radii_squared + midpoint.unsqueeze(-1)).clamp_min(EPS)
        outside = candidate.square().sum(dim=-1) > 1
        lower = torch.where(outside, midpoint, lower)
        upper = torch.where(outside, upper, midpoint)

    boundary = radii * projected / (radii_squared + upper.unsqueeze(-1)).clamp_min(EPS)
    coordinates = torch.where(inside.unsqueeze(-1), coordinates, boundary)
    prediction = (coordinates * radii) @ directions.transpose(-2, -1)
    return (delta - prediction).square().sum(dim=-1), coordinates.norm(dim=-1)


def random_basis(
    center: torch.Tensor,
    rank: int,
    seed: int,
    excluded: torch.Tensor | None = None,
) -> torch.Tensor:
    generator = torch.Generator(device=center.device).manual_seed(int(seed))
    factor = torch.randn(
        len(center),
        center.shape[1],
        rank,
        device=center.device,
        dtype=center.dtype,
        generator=generator,
    )
    factor -= center[:, :, None] * (center[:, :, None] * factor).sum(dim=1, keepdim=True)
    if excluded is not None:
        factor -= excluded @ (excluded.transpose(1, 2) @ factor)
    return orthonormalize(factor)[0]


def take_geometry(geometry: Geometry, index: object) -> Geometry:
    return {key: value[index] for key, value in geometry.items()}


def verify_geometry(geometry: Geometry, center: torch.Tensor) -> None:
    directions = geometry["q"]
    active = (directions.square().sum(dim=1) > 0.5).float()
    gram = directions.transpose(1, 2) @ directions
    if not torch.allclose(gram, torch.diag_embed(active), atol=3e-4):
        raise ValueError("geometry directions are not orthonormal")
    if (directions * center[:, :, None]).sum(dim=1).abs().max() >= 3e-4:
        raise ValueError("geometry directions are not tangent to the center")
