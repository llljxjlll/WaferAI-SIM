#!/usr/bin/env python3
"""Run one reproducible optimized intra-die split-K timing workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

from llm.frontend.wafer_frontend import (
    NaiveRunCase,
    NaiveRunRequest,
    NaiveRunResult,
    NaiveRunValidation,
    run_naive,
)
from llm.frontend.wafer_frontend.schema.intra_die_refine import (
    SplitKRefineOptions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one optimized intra-die split-K E1/E2 timing workload."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("E1", "E2"), default="E1")
    parser.add_argument("--profile", "--profile-id", dest="profile_id")
    parser.add_argument("--split-k-parts", type=int, default=2)
    parser.add_argument("--enable-reduce", action="store_true")
    parser.add_argument("--enable-double-buffer", action="store_true")
    parser.add_argument("--repeat", type=int, default=2)
    return parser


def require_optimized_intra_die(spec_path: Path) -> None:
    """Fail before compilation when the v2 refine policy cannot be selected."""
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("spec YAML must contain a mapping")
    policy = raw.get("policy")
    if not isinstance(policy, Mapping) or policy.get("intra_die") != "optimized":
        raise ValueError("split-K run requires spec policy.intra_die=optimized")


def build_request(args: argparse.Namespace) -> NaiveRunRequest:
    options = SplitKRefineOptions(
        split_k_parts=args.split_k_parts,
        enable_reduce=args.enable_reduce,
        enable_double_buffer=args.enable_double_buffer,
    )
    options.validate("cli.intra_die_refine_options")
    return NaiveRunRequest(
        case=NaiveRunCase(args.case),
        validation=NaiveRunValidation.TIMING,
        spec_path=args.spec,
        hardware_config_path=args.hardware,
        simulation_config_path=args.simulation,
        mapping_config_path=args.mapping,
        output_dir=args.output,
        npusim_path=args.npusim,
        finalizer_path=args.finalizer,
        profile_id=args.profile_id,
        repeat=args.repeat,
        intra_die_refine_options=options,
    )


def report_summary(result: NaiveRunResult) -> dict[str, object]:
    report = result.report
    runtime = report.runtime
    artifact = report.artifact
    if not isinstance(runtime, Mapping) or not isinstance(artifact, Mapping):
        raise TypeError("run report lacks runtime/artifact evidence")
    return {
        "report_id": result.report_id,
        "report_path": str(result.report_path),
        "output_dir": str(result.output_dir),
        "makespan_cycles": runtime.get("makespan_cycles"),
        "repeat_signature_stable": runtime.get("repeat_signature_stable"),
        "artifact_sha256": artifact.get("artifact_sha256"),
        "record_count": artifact.get("record_count"),
    }


def run_once(
    args: argparse.Namespace,
    *,
    runner: Callable[[NaiveRunRequest], NaiveRunResult] = run_naive,
) -> dict[str, object]:
    require_optimized_intra_die(args.spec)
    result = runner(build_request(args))
    return report_summary(result)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_once(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
