"""Evaluation on freshly extracted PMTT physics feature stores."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import PaperConfig
from .continuity import evaluate_continuity
from .evaluation import evaluate_support
from .physical import evaluate_physical_block
from .records import MetricRecorder
from .reporting import write_metric_tables
from .storage import FeatureStore, read_rows, write_rows


BLOCK_ORDER = {"configuration": 0, "axis": 1, "joint": 2, "motion": 3}


def analyze_synthetic(
    *,
    store_root: str | Path,
    output: str | Path,
    config: PaperConfig,
    device: str = "cuda",
) -> None:
    store = FeatureStore.open(store_root)
    if store.kind != "synthetic":
        raise ValueError("analyze_synthetic requires a synthetic feature store")
    recorder = MetricRecorder()
    device_value = torch.device(device)

    # Core 1 support and acquisition, with donor choices defined from source-only centers.
    for block in sorted(BLOCK_ORDER, key=BLOCK_ORDER.get):
        indices = [index for index, row in enumerate(store.metadata) if row["block"] == block]
        units = [store.metadata[index]["base_id"] for index in indices]
        centers = store.center(indices, device=device_value)
        same_base = torch.tensor(
            np.equal.outer(units, units), device=device_value, dtype=torch.bool
        )
        similarity = (centers @ centers.T).masked_fill(same_base, -torch.inf)
        similar = similarity.argmax(dim=1).cpu().tolist()
        fixed = [next(position for position, unit in enumerate(units) if unit != current) for current in units]

        for offset in range(0, len(indices), config.geometry_chunk):
            local_positions = list(
                range(offset, min(offset + config.geometry_chunk, len(indices)))
            )
            source_indices = [indices[position] for position in local_positions]
            current = store.take(source_indices, device=device_value)
            donor = store.take(
                [indices[fixed[position]] for position in local_positions],
                device=device_value,
            )
            close = store.take(
                [indices[similar[position]] for position in local_positions],
                device=device_value,
            )
            evaluate_support(
                current["center"],
                current["states"],
                current["future"],
                current["seeds"],
                current["units"],
                recorder,
                scope=block,
                ranks=config.ranks,
                seed=config.seed + offset,
                donor_center=donor["center"],
                donor_states=donor["states"],
                donor_seeds=donor["seeds"],
                similar_center=close["center"],
                similar_states=close["states"],
                similar_seeds=close["seeds"],
            )

    # Core 2 physical-state matrices and attribution.
    matrix_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    bases = sorted({row["base_id"] for row in store.metadata})
    constructions = (
        ("Original", "PairDifference", "VisibleMean", "SplitReference")
        if store.scale == "B"
        else ("Original", "PairDifference")
    )
    for base_id in bases:
        for block in sorted(BLOCK_ORDER, key=BLOCK_ORDER.get):
            indices = [
                index
                for index, row in enumerate(store.metadata)
                if row["base_id"] == base_id and row["block"] == block
            ]
            indices.sort(
                key=lambda index: (
                    int(store.metadata[index]["state_index"]),
                    int(store.metadata[index]["point_id"]),
                )
            )
            states = len({store.metadata[index]["state_index"] for index in indices})
            points = len(indices) // states
            current = store.take(indices, device=device_value)
            point_ids = [int(store.metadata[index]["point_id"]) for index in indices[:points]]
            state_values = [float(store.metadata[indices[state * points]]["state_value"]) for state in range(states)]
            result = evaluate_physical_block(
                current["center"].reshape(states, points, -1),
                current["states"].reshape(states, points, 5, 16, -1),
                current["future"].reshape(states, points, 4, -1),
                current["rgb"].reshape(states, points, -1),
                current["projected_geometry"].reshape(states, points, -1),
                current["seeds"],
                point_ids,
                state_values,
                recorder,
                base_id=base_id,
                block=block,
                constructions=constructions,
                seed=config.seed,
            )
            if base_id.startswith("existing-"):
                cohort = "Existing16"
            elif base_id.startswith("holdout-"):
                cohort = "New16"
            else:
                raise ValueError(f"unsupported synthetic identity: {base_id}")
            for name, destination in (
                ("matrix_rows", matrix_rows),
                ("selector_rows", selector_rows),
                ("retrieval_rows", retrieval_rows),
                ("residual_rows", residual_rows),
            ):
                destination.extend(
                    dict(row, scale=store.scale, cohort=cohort) for row in result[name]
                )

    output_path = Path(output)
    write_metric_tables(
        recorder,
        scale=store.scale,
        substrate="synthetic",
        output=output_path,
        config=config,
    )
    write_rows(output_path / "physical_matrices.csv", matrix_rows)
    write_rows(output_path / "wrong_state_selectors.csv", selector_rows)
    write_rows(output_path / "retrieval.csv", retrieval_rows)
    write_rows(output_path / "state_specific.csv", residual_rows)


def _bool(value: str | bool) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _compute_real_controls(
    store: FeatureStore,
    source_indices: Sequence[int],
    *,
    device: torch.device,
) -> dict[int, dict[str, int]]:
    source_indices = list(map(int, source_indices))
    bank = store.center(source_indices, device=device)
    source_metadata = [store.metadata[index] for index in source_indices]
    by_sequence: dict[str, list[int]] = defaultdict(list)
    by_frame: dict[tuple[str, int], list[int]] = defaultdict(list)
    for position, row in enumerate(source_metadata):
        by_sequence[row["sequence"]].append(position)
        by_frame[row["sequence"], int(row["frame"])].append(position)
    cross = {
        sequence: [position for position, row in enumerate(source_metadata) if row["sequence"] != sequence]
        for sequence in by_sequence
    }
    output: dict[int, dict[str, int]] = {}
    for start in range(0, len(source_indices), 128):
        query_indices = source_indices[start : start + 128]
        query = store.center(query_indices, device=device)
        similarity = query @ bank.T
        for local, source_index in enumerate(query_indices):
            row = store.metadata[source_index]
            sequence = row["sequence"]
            same_sequence = by_sequence[sequence]
            candidates = {
                "cross_video_similar": cross[sequence],
                "same_video_similar": [
                    position
                    for position in same_sequence
                    if source_metadata[position]["track"] != row["track"]
                ],
                "same_frame_other_track": [
                    position
                    for position in by_frame[sequence, int(row["frame"])]
                    if source_metadata[position]["track"] != row["track"]
                ],
            }
            if not cross[sequence]:
                raise ValueError("a cross-video donor is required")
            chosen = {"fixed_cross_video": source_indices[cross[sequence][0]]}
            for label, positions in candidates.items():
                chosen[label] = (
                    source_indices[positions[int(similarity[local, positions].argmax())]]
                    if positions
                    else -1
                )
            output[source_index] = chosen
    return output


def _saved_real_controls(
    store: FeatureStore,
    path: str | Path,
    scale: str,
) -> dict[int, dict[str, int]]:
    index = store.index
    output = {}
    for row in read_rows(path):
        if row["scale"] != scale or row["source_interface"] not in index:
            continue
        source_index = index[row["source_interface"]]
        output[source_index] = {}
        for field in (
            "fixed_cross_video",
            "cross_video_similar",
            "same_video_similar",
            "same_frame_other_track",
        ):
            identity = row[field]
            output[source_index][field] = index[identity] if identity and identity in index else -1
    return output


def analyze_tapvid(
    *,
    store_root: str | Path,
    anchors_path: str | Path,
    transitions_path: str | Path,
    output: str | Path,
    config: PaperConfig,
    scope: str,
    saved_controls_path: str | Path | None = None,
    device: str = "cuda",
) -> None:
    store = FeatureStore.open(store_root)
    if store.kind != "tapvid":
        raise ValueError("analyze_tapvid requires a TAP-Vid feature store")
    if scope not in {"full", "fixed"}:
        raise ValueError("scope must be 'full' or 'fixed'")
    device_value = torch.device(device)
    anchors = read_rows(anchors_path)
    if scope == "fixed":
        anchors = [row for row in anchors if int(row["fixed_support_index"]) >= 0]
    anchors = [row for row in anchors if row["interface_id"] in store.index]
    source_indices = [store.index[row["interface_id"]] for row in anchors]
    if saved_controls_path is not None and scope == "fixed":
        controls = _saved_real_controls(store, saved_controls_path, store.scale)
        if set(source_indices) - set(controls):
            raise ValueError("saved donor controls do not cover the fixed source set")
    else:
        controls = _compute_real_controls(store, source_indices, device=device_value)

    recorder = MetricRecorder()
    for horizon in (1, 4):
        eligible = [row for row in anchors if _bool(row[f"h{horizon}"])]
        result_scope = f"real_{scope}_h{horizon}"
        for start in range(0, len(eligible), config.geometry_chunk):
            group = eligible[start : start + config.geometry_chunk]
            ids = [store.index[row["interface_id"]] for row in group]
            target_ids = [
                store.index[f"{row['sequence']}:{int(row['track'])}:{int(row['frame']) + horizon}"]
                for row in group
            ]
            current = store.take(ids, device=device_value)
            current["future"] = store.center(target_ids, device=device_value)
            donor = store.take(
                [controls[index]["fixed_cross_video"] for index in ids],
                device=device_value,
            )
            similar = store.take(
                [controls[index]["cross_video_similar"] for index in ids],
                device=device_value,
            )
            evaluate_support(
                current["center"],
                current["states"],
                current["future"],
                current["seeds"],
                current["units"],
                recorder,
                scope=result_scope,
                ranks=config.ranks,
                seed=config.seed + start,
                donor_center=donor["center"],
                donor_states=donor["states"],
                donor_seeds=donor["seeds"],
                similar_center=similar["center"],
                similar_states=similar["states"],
                similar_seeds=similar["seeds"],
            )

    transitions = read_rows(transitions_path)
    if scope == "fixed":
        transitions = [row for row in transitions if _bool(row["fixed_continuity_subset"])]
    transitions = [
        row
        for row in transitions
        if row["source_interface"] in store.index and row["target_interface"] in store.index
    ]
    for horizon in (1, 4):
        selected = [row for row in transitions if int(row["horizon"]) == horizon]
        result_scope = f"real_{scope}_h{horizon}"
        for start in range(0, len(selected), config.geometry_chunk):
            group = selected[start : start + config.geometry_chunk]
            source_ids = [store.index[row["source_interface"]] for row in group]
            target_ids = [store.index[row["target_interface"]] for row in group]
            source = store.take(source_ids, device=device_value)
            target = store.take(target_ids, device=device_value)
            comparison_data = {}
            for label in (
                "same_frame_other_track",
                "same_video_similar",
                "cross_video_similar",
            ):
                chosen = [controls[index][label] for index in source_ids]
                valid = torch.tensor([index >= 0 for index in chosen], device=device_value)
                fallback = source_ids[0]
                resolved = [index if index >= 0 else fallback for index in chosen]
                value = store.take(resolved, device=device_value)
                public_label = "similar" if label == "cross_video_similar" else label
                comparison_data[public_label] = (
                    value["center"],
                    value["states"],
                    value["seeds"],
                    valid,
                )
            evaluate_continuity(
                source["center"],
                source["states"],
                source["seeds"],
                target["center"],
                target["states"],
                target["seeds"],
                source["units"],
                recorder,
                scope=result_scope,
                controls=comparison_data,
                ranks=config.ranks,
                seed=config.seed + start,
            )

    output_path = Path(output)
    write_metric_tables(
        recorder,
        scale=store.scale,
        substrate="real",
        output=output_path,
        config=config,
    )
