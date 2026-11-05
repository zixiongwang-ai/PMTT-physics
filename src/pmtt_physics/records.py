"""Per-independent-unit accumulation for PMTT physics metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


KEY_FIELDS = (
    "analysis",
    "scope",
    "construction",
    "method",
    "metric",
    "rank",
    "target",
    "source",
)


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().double().cpu().numpy()
    return np.asarray(value, dtype=float)


@dataclass
class MetricRecorder:
    values: dict[tuple[Any, ...], list[float]] = field(default_factory=dict)
    excluded: dict[tuple[Any, ...], int] = field(default_factory=dict)

    def add(
        self,
        analysis: str,
        scope: str,
        construction: str,
        method: str,
        metric: str,
        units: Sequence[str],
        numerator: Any,
        denominator: Any | None = None,
        rank: int = 4,
        target: str | int = "",
        source: str | int = "",
    ) -> None:
        numerator_array = _array(numerator)
        denominator_array = (
            np.ones_like(numerator_array)
            if denominator is None
            else _array(denominator)
        )
        if numerator_array.shape != denominator_array.shape or numerator_array.shape != (len(units),):
            raise ValueError(
                "numerator and denominator must have one value per independent unit"
            )
        if not np.isfinite(numerator_array).all() or not np.isfinite(denominator_array).all():
            raise FloatingPointError(f"non-finite metric input: {analysis}/{metric}")
        for unit, first, second in zip(
            units, numerator_array, denominator_array, strict=True
        ):
            key = (
                analysis,
                scope,
                construction,
                method,
                metric,
                int(rank),
                str(target),
                str(source),
                str(unit),
            )
            if second <= 1e-12:
                self.excluded[key] = self.excluded.get(key, 0) + 1
                continue
            value = self.values.setdefault(key, [0.0, 0.0, 0.0])
            value[0] += float(first)
            value[1] += float(second)
            value[2] += 1

    def rows(self) -> list[dict[str, Any]]:
        output = []
        for key, value in self.values.items():
            numerator, denominator, count = value
            output.append(
                {
                    **dict(zip((*KEY_FIELDS, "unit"), key, strict=True)),
                    "numerator": numerator,
                    "denominator": denominator,
                    "value": numerator / denominator,
                    "count": int(count),
                    "excluded": self.excluded.get(key, 0),
                }
            )
        return output
