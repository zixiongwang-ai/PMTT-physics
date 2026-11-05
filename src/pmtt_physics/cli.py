"""Command-line interface for PMTT physics reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_synthetic, analyze_tapvid
from .comparison import compare_reference
from .config import load_config
from .extraction import (
    extract_reader_context_variant,
    extract_reader_past,
    extract_synthetic,
    extract_tapvid,
)
from .reader import analyze_reader
from .validation import validate_assets


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _config(args: argparse.Namespace):
    return load_config(_path(args.config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pmtt-physics",
        description="Reproduce PMTT physics feature extraction and evaluation.",
    )
    parser.add_argument("--config", default="config/paper.toml")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-assets", help="validate released identities")
    validate.add_argument("--root", default=".")

    synthetic = commands.add_parser("extract-synthetic", help="extract synthetic feature store")
    synthetic.add_argument("--conditions", default="assets/manifests/synthetic_conditions.csv")
    synthetic.add_argument("--observation-scales", default="assets/parameters/synthetic_observation_scales.csv")
    synthetic.add_argument("--checkpoint", required=True)
    synthetic.add_argument("--scale", choices=("S", "B", "L"), required=True)
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--device", default="cuda")

    tapvid = commands.add_parser("extract-tapvid", help="extract TAP-Vid feature store")
    tapvid.add_argument("--dataset", required=True)
    tapvid.add_argument("--interfaces", default="assets/manifests/tapvid_interfaces.csv")
    tapvid.add_argument("--observation-scales", default="assets/parameters/real_observation_scales.csv")
    tapvid.add_argument("--checkpoint", required=True)
    tapvid.add_argument("--scale", choices=("S", "B", "L"), required=True)
    tapvid.add_argument("--output", required=True)
    tapvid.add_argument("--device", default="cuda")
    tapvid.add_argument("--records-per-group", type=int, default=4)

    past = commands.add_parser("extract-reader-past", help="extract previous-frame reader features")
    past.add_argument("--dataset", required=True)
    past.add_argument("--anchors", default="assets/manifests/tapvid_anchors.csv")
    past.add_argument("--checkpoint", required=True)
    past.add_argument("--scale", choices=("B",), default="B")
    past.add_argument("--output", required=True)
    past.add_argument("--device", default="cuda")

    context = commands.add_parser(
        "extract-reader-context", help="extract the optional Context scale-1 reader variant"
    )
    context.add_argument("--dataset", required=True)
    context.add_argument("--anchors", default="assets/manifests/tapvid_anchors.csv")
    context.add_argument("--checkpoint", required=True)
    context.add_argument("--scale", choices=("B",), default="B")
    context.add_argument("--output", required=True)
    context.add_argument("--device", default="cuda")

    analyze_syn = commands.add_parser("analyze-synthetic", help="evaluate a synthetic store")
    analyze_syn.add_argument("--store", required=True)
    analyze_syn.add_argument("--output", required=True)
    analyze_syn.add_argument("--device", default="cuda")

    analyze_real = commands.add_parser("analyze-tapvid", help="evaluate a TAP-Vid store")
    analyze_real.add_argument("--store", required=True)
    analyze_real.add_argument("--anchors", default="assets/manifests/tapvid_anchors.csv")
    analyze_real.add_argument("--transitions", default="assets/manifests/tapvid_transitions.csv")
    analyze_real.add_argument("--saved-controls", default="assets/manifests/tapvid_donor_controls.csv")
    analyze_real.add_argument("--scope", choices=("fixed", "full"), required=True)
    analyze_real.add_argument("--output", required=True)
    analyze_real.add_argument("--device", default="cuda")

    reader = commands.add_parser("analyze-reader", help="run the fixed-subset reader diagnostic")
    reader.add_argument("--store", required=True)
    reader.add_argument("--past-root", required=True)
    reader.add_argument("--context-variant-root")
    reader.add_argument("--anchors", default="assets/manifests/tapvid_anchors.csv")
    reader.add_argument("--coefficients", default="assets/parameters/reader_coefficients.csv")
    reader.add_argument("--cross-validation", default="assets/parameters/reader_cross_validation.csv")
    reader.add_argument("--observation-scales", default="assets/parameters/reader_observation_scales.csv")
    reader.add_argument("--mode", choices=("published_scalars", "refit_scalars"), default="published_scalars")
    reader.add_argument("--output", required=True)
    reader.add_argument("--device", default="cuda")

    compare = commands.add_parser("compare-reference", help="compare reproduced and reference summaries")
    compare.add_argument("--generated", required=True)
    compare.add_argument("--reference", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--atol", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-assets":
        print(json.dumps(validate_assets(_path(args.root)), indent=2, sort_keys=True))
    elif args.command == "extract-synthetic":
        extract_synthetic(
            conditions_path=_path(args.conditions),
            observation_scales_path=_path(args.observation_scales),
            checkpoint=_path(args.checkpoint),
            scale=args.scale,
            output=_path(args.output),
            config=_config(args),
            device=args.device,
        )
    elif args.command == "extract-tapvid":
        extract_tapvid(
            dataset=_path(args.dataset),
            interfaces_path=_path(args.interfaces),
            observation_scales_path=_path(args.observation_scales),
            checkpoint=_path(args.checkpoint),
            scale=args.scale,
            output=_path(args.output),
            config=_config(args),
            device=args.device,
            records_per_group=args.records_per_group,
        )
    elif args.command == "extract-reader-past":
        extract_reader_past(
            dataset=_path(args.dataset),
            anchors_path=_path(args.anchors),
            checkpoint=_path(args.checkpoint),
            scale=args.scale,
            output=_path(args.output),
            config=_config(args),
            device=args.device,
        )
    elif args.command == "extract-reader-context":
        extract_reader_context_variant(
            dataset=_path(args.dataset),
            anchors_path=_path(args.anchors),
            checkpoint=_path(args.checkpoint),
            scale=args.scale,
            output=_path(args.output),
            config=_config(args),
            device=args.device,
        )
    elif args.command == "analyze-synthetic":
        analyze_synthetic(
            store_root=_path(args.store),
            output=_path(args.output),
            config=_config(args),
            device=args.device,
        )
    elif args.command == "analyze-tapvid":
        analyze_tapvid(
            store_root=_path(args.store),
            anchors_path=_path(args.anchors),
            transitions_path=_path(args.transitions),
            saved_controls_path=(
                _path(args.saved_controls) if args.scope == "fixed" else None
            ),
            scope=args.scope,
            output=_path(args.output),
            config=_config(args),
            device=args.device,
        )
    elif args.command == "analyze-reader":
        analyze_reader(
            store_root=_path(args.store),
            anchors_path=_path(args.anchors),
            past_root=_path(args.past_root),
            context_variant_root=(
                _path(args.context_variant_root) if args.context_variant_root else None
            ),
            coefficients_path=_path(args.coefficients),
            cross_validation_path=_path(args.cross_validation),
            observation_scales_path=_path(args.observation_scales),
            mode=args.mode,
            output=_path(args.output),
            config=_config(args),
            device=args.device,
        )
    elif args.command == "compare-reference":
        print(
            json.dumps(
                compare_reference(
                    _path(args.generated),
                    _path(args.reference),
                    _path(args.output),
                    absolute_tolerance=args.atol,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        parser.error(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
