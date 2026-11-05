"""Comparison of reproduced summaries with released reference tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .storage import read_rows, write_rows


KEYS = (
    "scale", "substrate", "cohort", "analysis", "scope", "construction",
    "method", "metric", "rank", "target", "source",
)


def compare_reference(
    generated: str | Path,
    reference: str | Path,
    output: str | Path,
    *,
    absolute_tolerance: float | None = None,
) -> dict[str, Any]:
    generated_rows = read_rows(generated)
    reference_rows = read_rows(reference)
    generated_index = {
        tuple(row.get(key, "") for key in KEYS): row for row in generated_rows
    }
    report = []
    for reference_row in reference_rows:
        key = tuple(reference_row.get(name, "") for name in KEYS)
        if key not in generated_index:
            continue
        generated_row = generated_index[key]
        delta = float(generated_row["mean"]) - float(reference_row["mean"])
        report.append(
            {
                **{name: reference_row.get(name, "") for name in KEYS},
                "reference_mean": reference_row["mean"],
                "generated_mean": generated_row["mean"],
                "absolute_difference": abs(delta),
                "within_tolerance": (
                    "" if absolute_tolerance is None else abs(delta) <= absolute_tolerance
                ),
            }
        )
    if not report:
        raise ValueError("the generated and reference tables have no matching result keys")
    write_rows(output, report)
    maximum = max(float(row["absolute_difference"]) for row in report)
    return {
        "matched_rows": len(report),
        "maximum_absolute_difference": maximum,
        "tolerance_checked": absolute_tolerance is not None,
        "all_within_tolerance": (
            None
            if absolute_tolerance is None
            else all(bool(row["within_tolerance"]) for row in report)
        ),
    }
