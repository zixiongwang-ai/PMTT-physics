"""Public result naming and independent-unit summaries."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import PaperConfig
from .records import KEY_FIELDS, MetricRecorder
from .statistics import bootstrap_mean
from .storage import write_rows


_METHOD_NAMES = {
    "Orbital": "Orbital16",
    "SamplingMatched": "SamplingMatched16",
    "Geometric": "Geometric16",
    "Context": "Context16",
    "Sensitivity": "Sensitivity16",
    "G": "GenericGeometry",
    "ScalarCenter": "CenterScalar",
}
_ANALYSIS_NAMES = {
    "support": "static_future",
    "acquisition": "acquisition",
    "physical_state": "physical_state",
    "physical_matrix": "physical_matrix",
    "state_specific": "state_specific",
    "continuity": "continuity",
    "prediction_diagnostic": "prediction_diagnostic",
}


def public_method_name(name: str) -> str:
    """Return the stable reader-facing name for a method or comparison."""
    value = str(name)
    for source in sorted(_METHOD_NAMES, key=len, reverse=True):
        replacement = _METHOD_NAMES[source]
        value = value.replace(source, replacement)
    value = value.replace("WithoutSamplingMatched16", "WithoutSamplingMatched")
    value = value.replace("->", "_to_")
    return value


def public_analysis_name(name: str) -> str:
    return _ANALYSIS_NAMES.get(str(name), str(name))


def cohort_from_unit(unit: str) -> str:
    if unit.startswith("existing-"):
        return "Existing16"
    if unit.startswith("holdout-"):
        return "New16"
    if unit.startswith("real:") or ":" in unit:
        return "Real"
    raise ValueError(f"unsupported independent-unit identity: {unit}")


def write_metric_tables(
    recorder: MetricRecorder,
    *,
    scale: str,
    substrate: str,
    output: str | Path,
    config: PaperConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Write per-independent-unit values and deterministic bootstrap summaries."""
    per_unit: list[dict[str, Any]] = []
    for row in recorder.rows():
        enriched = dict(row)
        enriched["analysis"] = public_analysis_name(str(enriched["analysis"]))
        enriched["method"] = public_method_name(str(enriched["method"]))
        enriched["scale"] = scale
        enriched["substrate"] = substrate
        enriched["cohort"] = cohort_from_unit(str(row["unit"]))
        per_unit.append(enriched)

    combined = [
        dict(row, cohort="Combined32")
        for row in per_unit
        if row["cohort"] in {"Existing16", "New16"}
    ]
    summary_input = per_unit + combined
    group_fields = (*KEY_FIELDS, "scale", "substrate", "cohort")
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in summary_input:
        grouped[tuple(row[field] for field in group_fields)].append(float(row["value"]))

    summary: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(map(str, item))):
        values = grouped[key]
        interval = bootstrap_mean(values, seed=config.seed, draws=config.bootstrap_draws)
        summary.append(
            {
                **dict(zip(group_fields, key, strict=True)),
                "mean": interval.mean,
                "lo": interval.lower,
                "hi": interval.upper,
                "n": interval.units,
                "win_rate": float(np.mean(np.asarray(values) > 0)),
            }
        )

    per_unit.sort(
        key=lambda row: tuple(str(row.get(field, "")) for field in (*group_fields, "unit"))
    )
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    write_rows(output_path / "per_unit.csv", per_unit)
    write_rows(output_path / "summary.csv", summary)
    return per_unit, summary
