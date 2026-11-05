import torch

from pmtt_physics.reader import fit_center_map, fit_scalar_channels


def test_center_map_recovers_linear_target() -> None:
    generator = torch.Generator().manual_seed(3)
    design = torch.randn(40, 6, generator=generator)
    weight = torch.randn(6, 5, generator=generator)
    target = design @ weight
    fitted = fit_center_map(design, target, ridge=0.0)
    assert torch.allclose(fitted, weight, atol=2e-5, rtol=2e-5)


def test_scalar_channel_fit() -> None:
    generator = torch.Generator().manual_seed(4)
    channels = torch.randn(60, 3, 7, generator=generator)
    expected = torch.tensor([0.25, -0.5, 1.2])
    target = torch.einsum("nkd,k->nd", channels, expected)
    fitted = fit_scalar_channels(channels, target, ridge=0.0)
    assert torch.allclose(fitted, expected, atol=2e-5, rtol=2e-5)
