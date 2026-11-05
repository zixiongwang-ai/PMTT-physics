import torch

from pmtt_physics.geometry import (
    bounded_error,
    construct_geometries,
    explained_energy,
    factor_geometry,
    unit,
    verify_geometry,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    generator = torch.Generator().manual_seed(7)
    center = unit(torch.randn(4, 32, generator=generator))
    states = unit(center[:, None, None] + 0.08 * torch.randn(4, 5, 16, 32, generator=generator))
    return center, states, [11, 12, 13, 14]


def test_original_and_pair_difference_geometry() -> None:
    center, states, seeds = _inputs()
    for construction in ("Original", "PairDifference", "VisibleMean", "SplitReference"):
        geometry = construct_geometries(
            center,
            states,
            seeds,
            construction=construction,
            pools={"Lawful32": list(range(32))},
        )["Lawful32"]
        verify_geometry(geometry, center)
        assert geometry["q"].shape == (4, 32, 8)
        assert geometry["s"].shape == (4, 8)
        assert torch.isfinite(geometry["a"]).all()


def test_factor_and_bounded_projection() -> None:
    rows = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]])
    geometry = factor_geometry(rows)
    delta_inside = torch.tensor([[0.2, 0.1, 0.0]])
    error, coordinate_norm = bounded_error(geometry, delta_inside)
    assert error.item() < 1e-8
    assert coordinate_norm.item() <= 1.0 + 1e-6
    energy = explained_energy(geometry["q"][:, :, :2], delta_inside)
    assert energy.item() > 0
