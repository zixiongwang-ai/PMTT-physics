"""Identity-preserving observation families used by PMTT physics."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .constants import OBSERVATION_FAMILIES

# Each row is ``left_offset, top_offset, width, height`` in units of the
# canonical crop side. The first row is the canonical crop, followed by the
# sixteen frozen Orbital observations.
ORBITAL_RELATIVE_CROPS = np.asarray(
    [
        [-0.5, -0.5, 1.0, 1.0],
        [-0.46840643226543799, -0.39269981972521273, 0.85582475614158471, 0.78128626960682013],
        [-0.59659582246486309, -0.45852096486837679, 1.1547234066866006, 0.88454665024149126],
        [-0.4970924678959634, -0.44818593188771461, 0.98926362125960587, 0.83031124365728481],
        [-0.35043489972284059, -0.37997552380848743, 0.75533657762853057, 0.70718059020024604],
        [-0.33720233544684186, -0.33832112935649561, 0.7413378469224754, 0.67964907940261976],
        [-0.45053608258767608, -0.40104115623011327, 0.9773464034097834, 0.87483628878230835],
        [-0.58748525566196541, -0.44069610184677654, 1.1766893620569339, 0.97349626117608479],
        [-0.59830609113618549, -0.4354544511574574, 1.1491584937518373, 0.90753532244440716],
        [-0.46678884240321628, -0.35134225707231304, 0.89757112128825678, 0.70527202480219275],
        [-0.5117101265850843, -0.44813668271786278, 0.97045640188042737, 0.85257375864911134],
        [-0.48416232010004423, -0.70475935790968958, 0.95992554471027336, 1.2779501835840803],
        [-0.4487924775497214, -0.54178959120266157, 1.0095347395028991, 0.98048051282240922],
        [-0.49390767003897529, -0.54323014428059113, 1.0491810938827442, 1.0849995466195663],
        [-0.5484730870245873, -0.47135300650311862, 1.2409122479061296, 1.0573878719718308],
        [-0.39770810670950296, -0.41093398655467039, 0.79633600042407193, 0.91227946423958295],
        [-0.3641768357802313, -0.33847354565161525, 0.66841605163964823, 0.74953872975508895],
    ],
    dtype=np.float64,
)

ORBITAL_PAIRS = ((0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15))
SIGNED_PAIRS = tuple((index, index + 1) for index in range(0, 16, 2))
CONTEXT_PAIRS = tuple((index, index + 8) for index in range(8))

_HALF_A = {
    "Orbital": (0, 4, 2, 6, 9, 13, 11, 15),
    "SamplingMatched": (0, 4, 2, 6, 9, 13, 11, 15),
    "Geometric": (0, 1, 4, 5, 8, 9, 12, 13),
    "Context": (0, 8, 2, 10, 5, 13, 7, 15),
}


def acquisition_pools() -> dict[str, list[int]]:
    pools = {
        family: [16 * family_index + view for view in range(16)]
        for family_index, family in enumerate(OBSERVATION_FAMILIES)
    }
    pools["Lawful32"] = list(range(32))
    pools["Lawful64"] = list(range(64))
    pools["L_A32"] = [
        16 * family_index + view
        for family_index, family in enumerate(OBSERVATION_FAMILIES[:4])
        for view in _HALF_A[family]
    ]
    pools["L_B32"] = [index for index in range(64) if index not in pools["L_A32"]]
    for budget in (8, 16, 32):
        pools[f"L{budget}"] = [
            16 * family_index + view
            for family_index, family in enumerate(OBSERVATION_FAMILIES[:4])
            for view in _HALF_A[family][: budget // 4]
        ]
    pools["L64"] = pools["Lawful64"]
    for family_index, family in enumerate(OBSERVATION_FAMILIES[:4]):
        pools[f"Without{family}"] = [
            index for index in range(64) if index // 16 != family_index
        ]
    return pools


def pair_indices(family: str, seed: int) -> tuple[tuple[int, int], ...]:
    del seed  # The saved source seed fixes SamplingMatched ordering before construction.
    if family == "Orbital":
        return ORBITAL_PAIRS
    if family in ("Geometric", "Sensitivity"):
        return SIGNED_PAIRS
    if family == "Context":
        return CONTEXT_PAIRS
    if family == "SamplingMatched":
        return ORBITAL_PAIRS
    raise ValueError(f"unknown observation family: {family}")


def _as_rgb(image: Image.Image | np.ndarray | str) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    with Image.open(image) as opened:
        return opened.convert("RGB")


def canonical_crop(
    image: Image.Image | np.ndarray | str,
    point_xy: Sequence[float],
    side: float,
    output_size: int = 224,
) -> np.ndarray:
    image_rgb = _as_rgb(image)
    x, y = map(float, point_xy)
    crop = image_rgb.resize(
        (output_size, output_size),
        Image.Resampling.BICUBIC,
        box=(x - side / 2, y - side / 2, x + side / 2, y + side / 2),
    )
    return np.asarray(crop)


def orbital_views(
    image: Image.Image | np.ndarray | str,
    point_xy: Sequence[float],
    side: float,
    output_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    image_rgb = _as_rgb(image)
    x, y = map(float, point_xy)
    crops = ORBITAL_RELATIVE_CROPS.copy() * float(side)
    crops[:, :2] += np.asarray([x, y])
    size = np.asarray(image_rgb.size, dtype=np.float64)
    if np.any(crops[:, :2] < 0) or np.any(crops[:, :2] + crops[:, 2:] > size):
        raise ValueError("the Orbital observation stencil exceeds the image boundary")

    views: list[np.ndarray] = []
    grids: list[list[float]] = []
    for left, top, width, height in crops[1:]:
        crop = image_rgb.resize(
            (output_size, output_size),
            Image.Resampling.BICUBIC,
            box=(left, top, left + width, top + height),
        )
        views.append(np.asarray(crop))
        grids.append([(x - left) / width * 2 - 1, (y - top) / height * 2 - 1])
    return np.stack(views), np.asarray(grids, dtype=np.float32)


def sampling_matched_views(
    image: Image.Image | np.ndarray | str,
    point_xy: Sequence[float],
    side: float,
    seed: int,
    output_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    image_rgb = _as_rgb(image)
    x, y = map(float, point_xy)
    relative = ORBITAL_RELATIVE_CROPS[1:]
    widths_heights = relative[:, 2:]
    scale = np.sqrt(widths_heights.prod(axis=1))
    aspect = widths_heights[:, 0] / widths_heights[:, 1]
    offsets = -relative[:, :2] / widths_heights

    rng = np.random.default_rng(int(seed))
    scales = scale[rng.permutation(16)]
    aspects = aspect[rng.permutation(16)]
    offsets = np.stack(
        [offsets[rng.permutation(16), 0], offsets[rng.permutation(16), 1]],
        axis=1,
    )
    widths = scales * np.sqrt(aspects) * side
    heights = scales / np.sqrt(aspects) * side
    lefts = x - offsets[:, 0] * widths
    tops = y - offsets[:, 1] * heights

    pad_left = max(0, int(math.ceil(-float(lefts.min()))))
    pad_top = max(0, int(math.ceil(-float(tops.min()))))
    pad_right = max(0, int(math.ceil(float((lefts + widths).max()) - image_rgb.width)))
    pad_bottom = max(0, int(math.ceil(float((tops + heights).max()) - image_rgb.height)))
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        padded = np.pad(
            np.asarray(image_rgb),
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )
        image_rgb = Image.fromarray(padded)
        x += pad_left
        y += pad_top
        lefts += pad_left
        tops += pad_top

    views = []
    for left, top, width, height in zip(lefts, tops, widths, heights, strict=True):
        if left < 0 or top < 0 or left + width > image_rgb.width or top + height > image_rgb.height:
            raise ValueError("invalid SamplingMatched crop")
        crop = image_rgb.resize(
            (output_size, output_size),
            Image.Resampling.BICUBIC,
            box=(left, top, left + width, top + height),
        )
        views.append(np.asarray(crop))
    return np.stack(views), (offsets * 2 - 1).astype(np.float32)


def generated_observation_families(
    center: torch.Tensor,
    *,
    geometric_scale: float,
    context_scale: float,
    sensitivity_scale: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Generate Geometric, Context, and Sensitivity views from a center crop.

    Args:
        center: ``uint8`` image with shape ``[H, W, 3]`` or float image in
            ``[0, 1]`` with shape ``[3, H, W]``.
    """
    if center.ndim != 3:
        raise ValueError("center must be a three-dimensional image tensor")
    if center.shape[-1] == 3:
        image = center.permute(2, 0, 1).float()[None]
    elif center.shape[0] == 3:
        image = center.float()[None]
    else:
        raise ValueError("center must have three channels")
    if center.dtype == torch.uint8 or image.max() > 1.5:
        image = image / 255

    height, width = image.shape[-2:]
    if height != width:
        raise ValueError("the paper protocol uses square center crops")
    coordinate = torch.linspace(
        -1 + 1 / width,
        1 - 1 / width,
        width,
        device=image.device,
        dtype=image.dtype,
    )
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")

    geometric_grids = []
    for axis in range(8):
        for sign in (-1, 1):
            amount = sign * geometric_scale
            gx, gy = xx, yy
            if axis == 0:
                gx = xx * math.exp(0.07 * amount)
            elif axis == 1:
                gy = yy * math.exp(0.07 * amount)
            elif axis == 2:
                gx = xx + 0.08 * amount * yy
            elif axis == 3:
                gy = yy + 0.08 * amount * xx
            elif axis == 4:
                gx = xx / (1 + 0.045 * amount * xx)
                gy = yy / (1 + 0.045 * amount * xx)
            elif axis == 5:
                gx = xx / (1 + 0.045 * amount * yy)
                gy = yy / (1 + 0.045 * amount * yy)
            elif axis == 6:
                gx = xx + 0.045 * amount * torch.sin(math.pi * yy)
            else:
                gy = yy + 0.045 * amount * torch.sin(math.pi * xx)
            geometric_grids.append(torch.stack([gx, gy], dim=-1))
    geometric = F.grid_sample(
        image.expand(16, -1, -1, -1),
        torch.stack(geometric_grids),
        mode="bicubic",
        padding_mode="border",
        align_corners=False,
    ).clamp(0, 1)

    blurred = F.avg_pool2d(F.pad(image, (4, 4, 4, 4), mode="reflect"), 9, 1)
    gray = image.mean(dim=1, keepdim=True)
    mean = image.mean(dim=(2, 3), keepdim=True)
    field = torch.sin(math.pi * xx) * torch.cos(math.pi * yy)
    phase = (int(seed) % 997) / 997 * 2 * math.pi
    probe = torch.sin(math.pi * xx + phase) * torch.cos(2 * math.pi * yy - phase)

    sensitivity = []
    for axis in range(8):
        for sign in (-1, 1):
            amount = sign * sensitivity_scale
            if axis == 0:
                transformed = image + 0.035 * amount
            elif axis == 1:
                transformed = mean + (image - mean) * math.exp(0.12 * amount)
            elif axis == 2:
                transformed = image.clamp_min(1e-5).pow(math.exp(0.12 * amount))
            elif axis == 3:
                transformed = gray + (image - gray) * math.exp(0.18 * amount)
            elif axis == 4:
                transformed = image + 0.6 * amount * (image - blurred)
            elif axis == 5:
                transformed = image + 0.04 * amount * field
            elif axis == 6:
                transformed = image * (
                    1 + 0.07 * amount * field * torch.exp(-2 * (xx.square() + yy.square()))
                )
            else:
                transformed = image + 0.035 * amount * probe
            sensitivity.append(transformed[0].clamp(0, 1))

    context = []
    radius = torch.maximum(xx.abs(), yy.abs())
    for kind in range(2):
        for index in range(8):
            inner = 0.24 + 0.025 * index
            mask = ((radius - inner) / 0.12).clamp(0, 1)
            mask = mask.square() * (3 - 2 * mask)
            strength = min(1.0, context_scale * (0.25 + 0.09 * index))
            target = blurred if kind == 0 else (0.5 * blurred + 0.5 * mean)
            context.append((image * (1 - mask * strength) + target * mask * strength)[0])

    return {
        "Geometric": geometric,
        "Context": torch.stack(context),
        "Sensitivity": torch.stack(sensitivity),
    }


def prepare_source_views(
    image: Image.Image | np.ndarray | str,
    point_xy: Sequence[float],
    side: float,
    seed: int,
    *,
    geometric_scale: float,
    context_scale: float,
    sensitivity_scale: float,
    output_size: int = 224,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the center plus all five 16-view observation families.

    The returned images are uint8 NHWC tensors with shape ``[81, H, W, 3]``.
    The second output contains point-sampling grids in ``[-1, 1]``. The third
    marks the canonical center.
    """
    center = canonical_crop(image, point_xy, side, output_size)
    orbital, orbital_grid = orbital_views(image, point_xy, side, output_size)
    sampling, sampling_grid = sampling_matched_views(
        image,
        point_xy,
        side,
        seed,
        output_size,
    )
    generated = generated_observation_families(
        torch.from_numpy(center).to(device),
        geometric_scale=geometric_scale,
        context_scale=context_scale,
        sensitivity_scale=sensitivity_scale,
        seed=seed,
    )
    extra = torch.cat(
        [generated["Geometric"], generated["Context"], generated["Sensitivity"]],
        dim=0,
    )
    extra_uint8 = (extra * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()

    pixels = torch.cat(
        [
            torch.from_numpy(center.copy())[None],
            torch.from_numpy(orbital.copy()),
            torch.from_numpy(sampling.copy()),
            extra_uint8,
        ],
        dim=0,
    )
    grids = torch.cat(
        [
            torch.zeros(1, 2),
            torch.from_numpy(orbital_grid),
            torch.from_numpy(sampling_grid),
            torch.zeros(48, 2),
        ],
        dim=0,
    ).float()
    center_flags = torch.zeros(81, dtype=torch.bool)
    center_flags[0] = True
    return pixels, grids, center_flags


def canonicalize_sampling(states: torch.Tensor, seeds: Sequence[int]) -> torch.Tensor:
    """Undo the source-specific SamplingMatched scale permutation."""
    output = states.clone()
    for batch_index, seed in enumerate(seeds):
        permutation = np.random.default_rng(int(seed)).permutation(16)
        inverse = np.argsort(permutation)
        output[batch_index, 1] = output[batch_index, 1, inverse]
    return output
