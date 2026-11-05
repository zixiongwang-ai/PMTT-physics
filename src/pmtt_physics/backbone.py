"""DINOv3 patch-token wrapper used by PMTT physics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .constants import MODEL_SPECS


class DinoV3Backbone:
    """Load an official DINOv3 S, B, or L checkpoint and return patch tokens."""

    def __init__(
        self,
        scale: str,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cuda",
    ) -> None:
        if scale not in MODEL_SPECS:
            raise ValueError(f"unsupported DINOv3 scale: {scale}")
        self.scale = scale
        self.spec = MODEL_SPECS[scale]
        self.device = torch.device(device)
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.name != self.spec["checkpoint"]:
            raise ValueError(
                f"expected checkpoint {self.spec['checkpoint']}, received {checkpoint_path.name}"
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        expected_bytes = int(self.spec["bytes"])
        if checkpoint_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"checkpoint size mismatch for {scale}: expected {expected_bytes}, "
                f"received {checkpoint_path.stat().st_size}"
            )

        try:
            from dinov3.hub import backbones
        except ImportError as error:
            raise ImportError(
                "The official DINOv3 source package is required. Follow INSTALL.md."
            ) from error

        constructor = getattr(backbones, f"dinov3_{self.spec['constructor']}")
        model = constructor(pretrained=False)
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        model.load_state_dict(state, strict=True)
        del state
        self.model = model.eval().requires_grad_(False).to(self.device)

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> dict[str, Any]:
        images = images.to(self.device)
        use_autocast = self.device.type == "cuda"
        context = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else _NullContext()
        with context:
            tokens, (height, width) = self.model.prepare_tokens_with_masks(images)
            rotary = self.model.rope_embed(H=height, W=width)
            for block in self.model.blocks:
                tokens = block(tokens, rotary)
            final = self.model.norm(tokens)
            final = final[:, self.model.n_storage_tokens + 1 :].float()
        expected = (height * width, int(self.spec["width"]))
        if tuple(final.shape[1:]) != expected:
            raise ValueError(f"patch grid mismatch: expected {expected}, received {tuple(final.shape[1:])}")
        if not torch.isfinite(final).all():
            raise FloatingPointError("DINOv3 returned non-finite patch tokens")
        return {"tokens": final, "grid_shape": (height, width)}


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False
