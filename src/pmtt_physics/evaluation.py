"""Static-to-future support, organization, and acquisition measurements."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .geometry import (
    EPS,
    bounded_error,
    construct_geometries,
    explained_energy,
    factor_geometry,
    orthonormalize,
    random_basis,
    residual,
    subspace_overlap,
    transport_directions,
    verify_geometry,
)
from .observations import acquisition_pools
from .records import MetricRecorder


DEFAULT_METHODS = (
    "Orbital",
    "SamplingMatched",
    "Geometric",
    "Context",
    "Sensitivity",
    "Lawful32",
    "Lawful64",
)


def evaluate_support(
    center: torch.Tensor,
    states: torch.Tensor,
    future: torch.Tensor,
    seeds: Sequence[int],
    units: Sequence[str],
    recorder: MetricRecorder,
    *,
    scope: str,
    ranks: Sequence[int] = (1, 2, 4, 8),
    seed: int = 20261103,
    donor_center: torch.Tensor | None = None,
    donor_states: torch.Tensor | None = None,
    donor_seeds: Sequence[int] | None = None,
    similar_center: torch.Tensor | None = None,
    similar_states: torch.Tensor | None = None,
    similar_seeds: Sequence[int] | None = None,
    include_acquisition: bool = True,
) -> None:
    if future.ndim == 2:
        future = future[:, None]
    pools = acquisition_pools()
    geometries = construct_geometries(center, states, seeds)
    flattened = states.flatten(1, 2)
    future_residual = residual(center, future)
    tangent_energy = future_residual.square().sum(dim=(1, 2))
    full_delta = future - center[:, None]
    center_error = full_delta.square().sum(dim=(1, 2))

    donor_geometries = None
    similar_geometries = None
    if donor_center is not None and donor_states is not None and donor_seeds is not None:
        selected = {name: pools[name] for name in ("Lawful32", "L_A32", "L_B32", "Orbital", "SamplingMatched")}
        donor_geometries = construct_geometries(
            donor_center, donor_states, donor_seeds, pools=selected
        )
    if similar_center is not None and similar_states is not None and similar_seeds is not None:
        selected = {name: pools[name] for name in ("L_A32", "L_B32", "Orbital", "SamplingMatched")}
        similar_geometries = construct_geometries(
            similar_center, similar_states, similar_seeds, pools=selected
        )

    random = random_basis(center, 8, seed)
    for name, geometry in geometries.items():
        verify_geometry(geometry, center)
        for rank in ranks:
            numerator = explained_energy(
                geometry["q"][..., :rank], future_residual
            ).sum(dim=1)
            recorder.add(
                "support", scope, "Original", name, "TEE", units, numerator, tangent_energy, rank
            )
            recorder.add(
                "support",
                scope,
                "Original",
                name,
                "absolute_residual",
                units,
                tangent_energy - numerator,
                rank=rank,
            )

        bounded, _ = bounded_error(geometry, full_delta)
        observed_error = (
            future[:, :, None] - flattened[:, None, pools[name]]
        ).square().sum(dim=-1).min(dim=-1).values
        observed_error = torch.minimum(observed_error, full_delta.square().sum(dim=-1))
        for metric, numerator in (
            ("bounded_coverage", center_error - bounded.sum(dim=1)),
            ("observed_coverage", center_error - observed_error.sum(dim=1)),
            ("organization_value", observed_error.sum(dim=1) - bounded.sum(dim=1)),
            ("acquisition_value", center_error - observed_error.sum(dim=1)),
        ):
            recorder.add(
                "support", scope, "Original", name, metric, units, numerator, center_error, 8
            )
        spectrum = geometry["spectrum"]
        recorder.add(
            "support",
            scope,
            "Original",
            name,
            "effective_rank",
            units,
            spectrum.sum(dim=1).square() / spectrum.square().sum(dim=1).clamp_min(EPS),
            rank=8,
        )
        for rank in (1, 4):
            recorder.add(
                "support",
                scope,
                "Original",
                name,
                "singular_energy_fraction",
                units,
                spectrum[:, :rank].sum(dim=1),
                spectrum.sum(dim=1),
                rank,
            )

    recorder.add(
        "support",
        scope,
        "Original",
        "Center",
        "coverage",
        units,
        torch.zeros_like(center_error),
        center_error,
        8,
    )
    recorder.add(
        "support",
        scope,
        "Original",
        "Future",
        "magnitude",
        units,
        future_residual.square().sum(dim=-1).sqrt().mean(dim=1),
    )

    if donor_geometries is not None:
        for label, directions in (
            ("Donor", donor_geometries["Lawful32"]["q"]),
            ("Random", random),
        ):
            for rank in ranks:
                comparison = (
                    transport_directions(directions[..., :rank], center)
                    if label == "Donor"
                    else directions[..., :rank]
                )
                own = geometries["Lawful32"]["q"][..., :rank]
                actual_rank = torch.minimum(
                    (comparison.square().sum(dim=1) > 0.5).sum(dim=1),
                    (own.square().sum(dim=1) > 0.5).sum(dim=1),
                )
                mask = (
                    torch.arange(rank, device=center.device)[None] < actual_rank[:, None]
                )[:, None]
                comparison_value = explained_energy(comparison * mask, future_residual).sum(dim=1)
                own_value = explained_energy(own * mask, future_residual).sum(dim=1)
                recorder.add(
                    "support",
                    scope,
                    "Original",
                    label,
                    "TEE",
                    units,
                    comparison_value,
                    tangent_energy,
                    rank,
                )
                recorder.add(
                    "support",
                    scope,
                    "Original",
                    f"Lawful32-{label}",
                    "TEE_gap",
                    units,
                    own_value - comparison_value,
                    tangent_energy,
                    rank,
                )

    if future.shape[1] > 1:
        for held_out in range(future.shape[1]):
            oracle = factor_geometry(
                future_residual[:, [index for index in range(future.shape[1]) if index != held_out]]
            )
            for rank in ranks:
                numerator = explained_energy(
                    oracle["q"][..., :rank], future_residual[:, held_out : held_out + 1]
                ).sum(dim=1)
                recorder.add(
                    "support",
                    scope,
                    "LeaveOneFutureOut",
                    "Oracle",
                    "TEE",
                    units,
                    numerator,
                    future_residual[:, held_out].square().sum(dim=1),
                    rank,
                )
                recorder.add(
                    "support",
                    scope,
                    "LeaveOneFutureOut",
                    "Oracle",
                    "actual_rank",
                    units,
                    oracle["rank"].clamp(max=rank),
                    rank=rank,
                )

    if not include_acquisition:
        return
    pairs = [
        ("Orbital", "SamplingMatched"),
        ("SamplingMatched", "Orbital"),
        ("L_A32", "L_B32"),
        ("L_B32", "L_A32"),
        ("WithoutOrbital", "Orbital"),
        ("WithoutSamplingMatched", "SamplingMatched"),
        ("WithoutGeometric", "Geometric"),
        ("WithoutContext", "Context"),
    ]
    for source_name, held_name in pairs:
        geometry = geometries[source_name]
        held = flattened[:, pools[held_name]]
        delta = held - center[:, None]
        denominator = delta.square().sum(dim=(1, 2))
        unbounded = explained_energy(geometry["q"][..., :4], delta).sum(dim=1)
        bounded, _ = bounded_error(geometry, delta)
        overlap, valid = subspace_overlap(
            geometry["q"][..., :4], geometries[held_name]["q"][..., :4]
        )
        method = f"{source_name}->{held_name}"
        recorder.add(
            "acquisition",
            scope,
            "Original",
            method,
            "overlap",
            units,
            overlap,
            valid.gt(0),
            4,
        )
        recorder.add(
            "acquisition",
            scope,
            "Original",
            method,
            "unbounded_coverage",
            units,
            unbounded,
            denominator,
            4,
        )
        recorder.add(
            "acquisition",
            scope,
            "Original",
            method,
            "bounded_coverage",
            units,
            denominator - bounded.sum(dim=1),
            denominator,
            8,
        )

        controls: list[tuple[str, dict[str, torch.Tensor]]] = []
        if donor_geometries is not None and source_name in donor_geometries:
            controls.append(("Donor", donor_geometries[source_name]))
        if similar_geometries is not None and source_name in similar_geometries:
            controls.append(("SimilarDonor", similar_geometries[source_name]))
        controls.append(("Random", {"q": random, "s": geometry["s"]}))
        for label, other in controls:
            other_q = transport_directions(other["q"][..., :4], center)
            other_unbounded = explained_energy(other_q, delta).sum(dim=1)
            factor = other["q"] * other["s"][:, None]
            factor = factor - center[:, :, None] * (
                center[:, :, None] * factor
            ).sum(dim=1, keepdim=True)
            bounded_q, bounded_s = orthonormalize(factor)
            other_error, _ = bounded_error({"q": bounded_q, "s": bounded_s}, delta)
            recorder.add(
                "acquisition",
                scope,
                "Original",
                method,
                f"unbounded_minus_{label}",
                units,
                unbounded - other_unbounded,
                denominator,
            )
            recorder.add(
                "acquisition",
                scope,
                "Original",
                method,
                f"bounded_minus_{label}",
                units,
                other_error.sum(dim=1) - bounded.sum(dim=1),
                denominator,
                8,
            )
            other_overlap, _ = subspace_overlap(other_q, geometries[held_name]["q"][..., :4])
            recorder.add(
                "acquisition",
                scope,
                "Original",
                method,
                f"overlap_minus_{label}",
                units,
                overlap - other_overlap,
                valid.gt(0),
            )

    ceiling = explained_energy(geometries["L64"]["q"][..., :4], future_residual).sum(dim=1)
    for budget in (8, 16, 32, 64):
        name = f"L{budget}"
        geometry = geometries[name]
        numerator = explained_energy(geometry["q"][..., :4], future_residual).sum(dim=1)
        recorder.add(
            "acquisition",
            scope,
            "Original",
            name,
            "ceiling_recovery",
            units,
            numerator,
            ceiling,
        )
        overlap, valid = subspace_overlap(
            geometry["q"][..., :4], geometries["L64"]["q"][..., :4]
        )
        recorder.add(
            "acquisition",
            scope,
            "Original",
            name,
            "overlap_ceiling",
            units,
            overlap,
            valid.gt(0),
        )
        for family_index, family in enumerate(("Orbital", "SamplingMatched", "Geometric", "Context")):
            indices = [index for index in pools[name] if index // 16 != family_index]
            directions = factor_geometry(residual(center, flattened[:, indices]))["q"][..., :4]
            held_residual = residual(center, states[:, family_index])
            recorder.add(
                "acquisition",
                scope,
                "Original",
                name,
                "heldout_family_coverage",
                units,
                explained_energy(directions, held_residual).sum(dim=1),
                held_residual.square().sum(dim=(1, 2)),
                target=family,
            )
    for lower, upper in ((8, 16), (16, 32), (32, 64)):
        difference = explained_energy(
            geometries[f"L{upper}"]["q"][..., :4], future_residual
        ).sum(dim=1) - explained_energy(
            geometries[f"L{lower}"]["q"][..., :4], future_residual
        ).sum(dim=1)
        recorder.add(
            "acquisition",
            scope,
            "Original",
            f"L{lower}->L{upper}",
            "future_increment",
            units,
            difference,
            tangent_energy,
        )
