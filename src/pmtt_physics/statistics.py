"""Independent-unit aggregation and deterministic paired bootstrap intervals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    mean: float
    lower: float
    upper: float
    units: int


def bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    *,
    seed: int = 20261103,
    draws: int = 2000,
) -> Interval:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("non-finite statistical input")
    indices = np.random.default_rng(seed).integers(0, len(array), (draws, len(array)))
    distribution = array[indices].mean(axis=1)
    return Interval(
        mean=float(array.mean()),
        lower=float(np.quantile(distribution, 0.025)),
        upper=float(np.quantile(distribution, 0.975)),
        units=len(array),
    )


def pool_ratios(
    rows: Iterable[Mapping[str, object]],
    group_fields: Sequence[str],
    unit_field: str = "unit",
) -> list[dict[str, object]]:
    pooled: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        key = tuple(row[field] for field in (*group_fields, unit_field))
        value = pooled.setdefault(key, [0.0, 0.0, 0.0])
        value[0] += float(row["numerator"])
        value[1] += float(row["denominator"])
        value[2] += float(row.get("count", 1))
    output = []
    for key, (numerator, denominator, count) in pooled.items():
        if denominator <= 1e-12:
            continue
        output.append(
            {
                **dict(zip((*group_fields, unit_field), key, strict=True)),
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator,
                "count": count,
            }
        )
    return output


def summarize_units(
    rows: Iterable[Mapping[str, object]],
    group_fields: Sequence[str],
    *,
    seed: int = 20261103,
    draws: int = 2000,
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(float(row["value"]))
    output = []
    for key, values in groups.items():
        interval = bootstrap_mean(values, seed=seed, draws=draws)
        output.append(
            {
                **dict(zip(group_fields, key, strict=True)),
                "mean": interval.mean,
                "lower": interval.lower,
                "upper": interval.upper,
                "units": interval.units,
                "positive_fraction": float(np.mean(np.asarray(values) > 0)),
            }
        )
    return output


def paired_difference(
    first: Mapping[str, float],
    second: Mapping[str, float],
    *,
    lower_is_better: bool = False,
    seed: int = 20261103,
    draws: int = 2000,
) -> Interval:
    units = sorted(set(first) & set(second))
    if not units:
        raise ValueError("the paired comparison has no shared units")
    sign = -1.0 if lower_is_better else 1.0
    values = [sign * (float(first[unit]) - float(second[unit])) for unit in units]
    return bootstrap_mean(values, seed=seed, draws=draws)
