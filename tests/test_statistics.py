import numpy as np

from pmtt_physics.statistics import bootstrap_mean, paired_difference, pool_ratios


def test_deterministic_bootstrap() -> None:
    interval = bootstrap_mean([1.0, 2.0, 3.0, 4.0], seed=20261103, draws=2000)
    assert interval.mean == 2.5
    assert interval.lower == 1.5
    assert interval.upper == 3.5
    assert interval.units == 4


def test_ratio_pooling_and_paired_direction() -> None:
    rows = [
        {"method": "A", "unit": "u1", "numerator": 1.0, "denominator": 2.0},
        {"method": "A", "unit": "u1", "numerator": 2.0, "denominator": 2.0},
        {"method": "A", "unit": "u2", "numerator": 1.0, "denominator": 4.0},
    ]
    pooled = pool_ratios(rows, ["method"])
    values = {row["unit"]: row["value"] for row in pooled}
    assert values == {"u1": 0.75, "u2": 0.25}
    interval = paired_difference({"u1": 0.5, "u2": 0.4}, {"u1": 0.2, "u2": 0.1})
    assert np.isclose(interval.mean, 0.3)
