"""Persistent-identity continuity measurements for natural video."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .geometry import (
    construct_geometries,
    explained_energy,
    random_basis,
    subspace_overlap,
    transport_directions,
)
from .observations import acquisition_pools
from .records import MetricRecorder


def evaluate_continuity(
    source_center: torch.Tensor,
    source_states: torch.Tensor,
    source_seeds: Sequence[int],
    next_center: torch.Tensor,
    next_states: torch.Tensor,
    next_seeds: Sequence[int],
    units: Sequence[str],
    recorder: MetricRecorder,
    *,
    scope: str,
    controls: Mapping[str, tuple[torch.Tensor, torch.Tensor, Sequence[int], torch.Tensor]],
    ranks: Sequence[int] = (1, 2, 4, 8),
    seed: int = 20261103,
) -> None:
    pools = {
        name: acquisition_pools()[name]
        for name in ("Orbital", "SamplingMatched", "Lawful32", "Sensitivity", "Lawful64")
    }
    source_geometry = construct_geometries(
        source_center, source_states, source_seeds, pools=pools
    )
    next_geometry = construct_geometries(next_center, next_states, next_seeds, pools=pools)
    control_geometry = {
        label: (
            construct_geometries(center, states, seeds, pools=pools),
            valid.to(source_center.device),
        )
        for label, (center, states, seeds, valid) in controls.items()
    }
    random = random_basis(source_center, 8, seed)

    for method, geometry in source_geometry.items():
        held = next_states.flatten(1, 2)[:, pools[method]] - source_center[:, None]
        held_energy = held.square().sum(dim=(1, 2))
        for rank in ranks:
            own, actual_rank = subspace_overlap(
                geometry["q"][..., :rank],
                transport_directions(next_geometry[method]["q"][..., :rank], source_center),
            )
            valid_own = actual_rank.gt(0)
            recorder.add(
                "continuity", scope, "Original", method, "same_track_overlap", units, own, valid_own, rank
            )
            recorder.add(
                "continuity",
                scope,
                "Original",
                method,
                "source_next_coverage",
                units,
                explained_energy(geometry["q"][..., :rank], held).sum(dim=1),
                held_energy,
                rank,
            )
            comparisons: list[tuple[str, torch.Tensor, torch.Tensor]] = [
                ("random", random[..., :rank], valid_own)
            ]
            for label, (geometries, valid) in control_geometry.items():
                comparisons.append(
                    (
                        label,
                        transport_directions(geometries[method]["q"][..., :rank], source_center),
                        valid & valid_own,
                    )
                )
            for label, directions, valid in comparisons:
                value, _ = subspace_overlap(geometry["q"][..., :rank], directions)
                public_label = "cross_video_similar" if label == "similar" else label
                recorder.add(
                    "continuity",
                    scope,
                    "Original",
                    method,
                    f"{public_label}_overlap",
                    units,
                    value * valid,
                    valid,
                    rank,
                )
                recorder.add(
                    "continuity",
                    scope,
                    "Original",
                    method,
                    f"same_minus_{public_label}",
                    units,
                    (own - value) * valid,
                    valid,
                    rank,
                )
