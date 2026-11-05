"""Scientific constants used by the released PMTT physics implementation."""

from __future__ import annotations

from typing import Final

IMAGE_MEAN: Final = (0.485, 0.456, 0.406)
IMAGE_STD: Final = (0.229, 0.224, 0.225)

OBSERVATION_FAMILIES: Final = (
    "Orbital",
    "SamplingMatched",
    "Geometric",
    "Context",
    "Sensitivity",
)
LAWFUL_FAMILIES: Final = OBSERVATION_FAMILIES[:4]

PHYSICAL_BLOCKS: Final = {
    "configuration": (-20.0, -10.0, 0.0, 10.0, 20.0),
    "axis": (0.0, 45.0, 90.0, 135.0),
    "joint": (-0.25, 0.0, 0.25),
    "motion": (0.0, 1.0),
}

MODEL_SPECS: Final = {
    "S": {
        "constructor": "vits16",
        "width": 384,
        "checkpoint": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "bytes": 86_531_063,
    },
    "B": {
        "constructor": "vitb16",
        "width": 768,
        "checkpoint": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "bytes": 342_860_279,
    },
    "L": {
        "constructor": "vitl16",
        "width": 1024,
        "checkpoint": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "bytes": 1_213_050_671,
    },
}
