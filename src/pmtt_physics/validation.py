"""Validation of released experiment identities and small reference assets."""

from __future__ import annotations

from pathlib import Path
import ast
import csv
import tomllib

from .config import load_config
from .storage import read_rows


def _require_columns(path: Path, required: set[str]) -> list[dict[str, str]]:
    rows = read_rows(path)
    with path.open(newline="", encoding="utf-8") as stream:
        header = set(next(csv.reader(stream), []))
    missing = required - header
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return rows


def validate_assets(root: str | Path) -> dict[str, int]:
    """Validate exact public identities without loading datasets or model weights."""
    project = Path(root).resolve()
    config = load_config(project / "config" / "paper.toml")
    config.validate()
    manifests = project / "assets" / "manifests"
    parameters = project / "assets" / "parameters"
    references = project / "reference_results"

    conditions = _require_columns(
        manifests / "synthetic_conditions.csv",
        {"base_id", "cohort", "identity_index", "block", "state_index", "point_ids"},
    )
    if len(conditions) != 448:
        raise ValueError(f"expected 448 synthetic conditions, received {len(conditions)}")
    bases = {row["base_id"] for row in conditions}
    if len(bases) != 32:
        raise ValueError(f"expected 32 synthetic identities, received {len(bases)}")
    for base_id in bases:
        rows = [row for row in conditions if row["base_id"] == base_id]
        if len(rows) != 14:
            raise ValueError(f"{base_id} does not have 14 one-factor-at-a-time states")
        for row in rows:
            points = ast.literal_eval(row["point_ids"])
            if len(points) != 8 or len(set(points)) != 8:
                raise ValueError(f"{base_id}/{row['block']}/{row['state_index']} has invalid points")

    anchors = _require_columns(
        manifests / "tapvid_anchors.csv",
        {
            "sequence", "track", "frame", "interface_id", "fold", "past", "h1", "h4",
            "fixed_support_index",
        },
    )
    if len(anchors) != 23_686 or len({row["sequence"] for row in anchors}) != 30:
        raise ValueError("TAP-Vid anchor identities do not match the released protocol")
    if sum(str(row["h1"]).lower() == "true" for row in anchors) != 23_500:
        raise ValueError("unexpected +1 anchor count")
    if sum(str(row["h4"]).lower() == "true" for row in anchors) != 20_886:
        raise ValueError("unexpected +4 anchor count")
    fixed = [row for row in anchors if int(row["fixed_support_index"]) >= 0]
    if len(fixed) != 3_840 or sorted(int(row["fixed_support_index"]) for row in fixed) != list(range(3_840)):
        raise ValueError("fixed support subset is incomplete or unordered")

    transitions = _require_columns(
        manifests / "tapvid_transitions.csv",
        {"source_interface", "target_interface", "horizon", "sequence", "fixed_continuity_subset"},
    )
    if len(transitions) != 44_386:
        raise ValueError("unexpected transition count")
    by_horizon = {
        horizon: sum(int(row["horizon"]) == horizon for row in transitions)
        for horizon in (1, 4)
    }
    if by_horizon != {1: 23_500, 4: 20_886}:
        raise ValueError(f"unexpected transition counts: {by_horizon}")
    fixed_by_horizon = {
        horizon: sum(
            int(row["horizon"]) == horizon and str(row["fixed_continuity_subset"]).lower() == "true"
            for row in transitions
        )
        for horizon in (1, 4)
    }
    if fixed_by_horizon != {1: 960, 4: 890}:
        raise ValueError(f"unexpected fixed continuity-transition counts: {fixed_by_horizon}")

    interfaces = _require_columns(
        manifests / "tapvid_interfaces.csv",
        {"sequence", "track", "frame", "interface_id", "source_seed", "has_observations"},
    )
    if len(interfaces) != 24_565 or len({row["interface_id"] for row in interfaces}) != len(interfaces):
        raise ValueError("TAP-Vid interface identities are incomplete or duplicated")
    fixed_interfaces = read_rows(manifests / "tapvid_fixed_interfaces.csv")
    if len(fixed_interfaces) != 9_307:
        raise ValueError("unexpected fixed-interface count")

    folds = read_rows(manifests / "tapvid_folds.csv")
    if len(folds) != 30 or {int(row["fold"]) for row in folds} != set(range(5)):
        raise ValueError("five-fold sequence partition is incomplete")
    donors = read_rows(manifests / "tapvid_donor_controls.csv")
    if len(donors) != 11_520:
        raise ValueError("saved fixed-subset donor controls are incomplete")
    selectors = read_rows(manifests / "wrong_state_selectors.csv")
    if len(selectors) != 32_256:
        raise ValueError("physical-state selector identities are incomplete")

    acquisition = read_rows(manifests / "acquisition_membership.csv")
    pools = {row["pool"] for row in acquisition}
    required_pools = {"Lawful32", "Lawful64", "L8", "L16", "L32", "L64", "L_A32", "L_B32"}
    if not required_pools.issubset(pools):
        raise ValueError("acquisition membership is incomplete")

    registry = tomllib.loads((parameters / "model_registry.toml").read_text(encoding="utf-8"))
    if set(registry.get("models", {})) != {"S", "B", "L"}:
        raise ValueError("model registry must define S, B, and L")
    coefficient_rows = read_rows(parameters / "reader_coefficients.csv")
    if len(coefficient_rows) != 180:
        raise ValueError("reader coefficient table is incomplete")
    cross_validation = _require_columns(
        parameters / "reader_cross_validation.csv",
        {
            "fold",
            "horizon",
            "inner_fold",
            "train_sequences",
            "validation_sequences",
            "test_sequences",
            "proposal_source",
        },
    )
    if len(cross_validation) != 40:
        raise ValueError("reader cross-validation table is incomplete")

    reader_scales = _require_columns(
        parameters / "reader_observation_scales.csv",
        {"outer_fold", "horizon", "fit_role", "inner_fold", "family", "scale", "uses_future"},
    )
    if len(reader_scales) != 200 or {row["fit_role"] for row in reader_scales} != {"inner", "refit"}:
        raise ValueError("reader observation-scale table is incomplete")
    if any(str(row["uses_future"]).lower() != "false" for row in reader_scales):
        raise ValueError("reader observation scales must be source-only")

    real_scales = _require_columns(
        parameters / "real_observation_scales.csv",
        {"fold", "family", "scale", "uses_future"},
    )
    if len(real_scales) != 20 or any(str(row["uses_future"]).lower() != "false" for row in real_scales):
        raise ValueError("real observation-scale table is incomplete")

    synthetic_scales = _require_columns(
        parameters / "synthetic_observation_scales.csv",
        {"family", "scale", "uses_future"},
    )
    if len(synthetic_scales) != 4 or any(str(row["uses_future"]).lower() != "false" for row in synthetic_scales):
        raise ValueError("synthetic observation-scale table is incomplete")

    oracle = _require_columns(
        manifests / "future_oracle_membership.csv",
        {"block", "target_state", "held_out_branch", "oracle_branches", "base_id"},
    )
    if len(oracle) != 1_792:
        raise ValueError("future-oracle membership is incomplete")

    reference_files = sorted(references.glob("*.csv"))
    if not reference_files:
        raise ValueError("reference results are missing")
    for path in reference_files:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            if not next(reader, []):
                raise ValueError(f"empty reference table: {path.name}")
            for row_number, row in enumerate(reader, start=2):
                if not row:
                    raise ValueError(f"blank row in {path.name}:{row_number}")

    return {
        "synthetic_conditions": len(conditions),
        "synthetic_identities": len(bases),
        "tapvid_anchors": len(anchors),
        "tapvid_interfaces": len(interfaces),
        "tapvid_transitions": len(transitions),
        "reference_tables": len(reference_files),
    }
