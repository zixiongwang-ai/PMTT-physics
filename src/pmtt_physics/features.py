"""Pixel-to-feature extraction for canonical and observation-conditioned states."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .constants import IMAGE_MEAN, IMAGE_STD
from .geometry import finite, unit
from .observations import canonicalize_sampling


def normalize_uint8(pixels: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Normalize uint8 NHWC images through a deterministic lookup table."""
    if pixels.dtype != torch.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError("pixels must be uint8 NHWC images")
    device = torch.device(device)
    value = pixels.to(device, non_blocking=device.type == "cuda").permute(0, 3, 1, 2)
    table = torch.arange(256, dtype=torch.float32).expand(3, -1).clone().div_(255)
    table.sub_(torch.tensor(IMAGE_MEAN)[:, None]).div_(torch.tensor(IMAGE_STD)[:, None])
    table = table.to(device)
    channels = torch.arange(3, device=device)[None, :, None, None]
    return table[channels, value.long()]


def sample_patch_tokens(
    tokens: torch.Tensor,
    grids: torch.Tensor,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    batch, count, width = tokens.shape
    height, columns = grid_shape
    if count != height * columns:
        raise ValueError("token count does not match grid_shape")
    image = tokens.transpose(1, 2).reshape(batch, width, height, columns)
    sampled = F.grid_sample(
        image,
        grids[:, None, None],
        align_corners=False,
        padding_mode="zeros",
    )
    return sampled.squeeze(-1).squeeze(-1)


@torch.inference_mode()
def encode_views(
    backbone: object,
    pixels: torch.Tensor,
    grids: torch.Tensor,
    center_flags: torch.Tensor,
    *,
    batch_size: int = 68,
    fixed_forward_batch: bool = True,
) -> torch.Tensor:
    """Encode a set of center and observation views.

    Canonical states are sampled from raw patch tokens and normalized after
    interpolation. Observation-conditioned states normalize each patch token
    before interpolation, then normalize the interpolated feature.
    """
    if len(pixels) != len(grids) or len(pixels) != len(center_flags):
        raise ValueError("pixels, grids, and center_flags must have the same length")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = getattr(backbone, "device", torch.device("cuda"))
    outputs = []
    for start in range(0, len(pixels), batch_size):
        pixel_batch = pixels[start : start + batch_size]
        grid_batch = grids[start : start + batch_size].to(device)
        flag_batch = center_flags[start : start + batch_size].to(device)
        count = len(pixel_batch)
        normalized = normalize_uint8(pixel_batch, device)
        if fixed_forward_batch and count < batch_size:
            normalized = torch.cat(
                [normalized, normalized[-1:].expand(batch_size - count, -1, -1, -1)]
            )
        encoded = backbone(normalized)
        tokens = encoded["tokens"][:count]
        shape = encoded["grid_shape"]
        canonical = unit(sample_patch_tokens(tokens, grid_batch, shape))
        observed = unit(sample_patch_tokens(unit(tokens), grid_batch, shape))
        outputs.append(torch.where(flag_batch[:, None], canonical, observed).cpu())
    return finite(torch.cat(outputs))


def split_state_family(
    encoded: torch.Tensor,
    seeds: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split 81 encoded views into centers and five 16-view families."""
    if encoded.ndim != 3 or encoded.shape[1] != 81:
        raise ValueError("encoded must have shape [batch, 81, features]")
    center = encoded[:, 0]
    states = encoded[:, 1:].reshape(len(encoded), 5, 16, -1)
    return center, canonicalize_sampling(states, seeds)
