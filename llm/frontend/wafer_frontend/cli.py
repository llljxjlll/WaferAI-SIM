"""Command-line entry points for N1 contract validation."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .errors import FrontendError, SchemaError
from .passes.pass_manager import pipeline_description
from .schema.common import stable_artifact_id
from .schema.experiment import ExperimentSpec
from .schema.serde import canonical_digest, canonical_json, from_data


VALIDATION_REPORT_SCHEMA_VERSION = "wafer_frontend.validation_report/v1alpha1"


@dataclass(frozen=True, slots=True)
class SpecValidationReport:
    schema_version: str
    producer_pass: str
    id: str
    status: str
    spec_id: str
    spec_digest: str


def _load_yaml(path: Path) -> object:
    try:
        import yaml
    except ImportError as error:
        raise SchemaError(
            "PyYAML is required for validate-spec; install the 'PyYAML' package",
            path="spec",
        ) from error

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SchemaError(str(error), path="spec") from error

    class StrictSafeLoader(yaml.SafeLoader):
        def __init__(self, stream: str) -> None:
            super().__init__(stream)
            self.strict_path = "spec"

    def child_path(parent: str, child: object) -> str:
        return f"{parent}.{child}" if isinstance(child, str) and child else parent

    def construct_mapping(loader: StrictSafeLoader, node: object) -> object:
        if not isinstance(node, yaml.MappingNode):
            raise SchemaError("expected a YAML mapping", path=loader.strict_path)
        loader.flatten_mapping(node)
        result: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            key_path = child_path(loader.strict_path, key)
            try:
                duplicate = key in result
            except TypeError as error:
                raise SchemaError("YAML mapping keys must be hashable", path=key_path) from error
            if duplicate:
                mark = key_node.start_mark
                raise SchemaError(
                    f"duplicate YAML mapping key {key!r} "
                    f"at line {mark.line + 1}, column {mark.column + 1}",
                    path=key_path,
                )
            previous_path = loader.strict_path
            loader.strict_path = key_path
            try:
                result[key] = loader.construct_object(value_node, deep=True)
            finally:
                loader.strict_path = previous_path
        return result

    def construct_sequence(loader: StrictSafeLoader, node: object) -> object:
        if not isinstance(node, yaml.SequenceNode):
            raise SchemaError("expected a YAML sequence", path=loader.strict_path)
        result = []
        for index, child_node in enumerate(node.value):
            previous_path = loader.strict_path
            loader.strict_path = f"{previous_path}[{index}]"
            try:
                result.append(loader.construct_object(child_node, deep=True))
            finally:
                loader.strict_path = previous_path
        return result

    StrictSafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    StrictSafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG,
        construct_sequence,
    )

    def reject_nonfinite(value: object, value_path: str) -> None:
        if type(value) is float and not math.isfinite(value):
            raise SchemaError("non-finite YAML floats are not allowed", path=value_path)
        if isinstance(value, dict):
            for key, child in value.items():
                reject_nonfinite(child, child_path(value_path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                reject_nonfinite(child, f"{value_path}[{index}]")

    try:
        result = yaml.load(text, Loader=StrictSafeLoader)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        problem = getattr(error, "problem", None) or str(error)
        if mark is not None:
            problem = (
                f"{problem} at line {mark.line + 1}, column {mark.column + 1}"
            )
        raise SchemaError(problem, path="spec") from error
    except RecursionError as error:
        raise SchemaError("YAML nesting is too deep or recursive", path="spec") from error
    reject_nonfinite(result, "spec")
    return result


def load_experiment_spec(path: Path) -> ExperimentSpec:
    """Strictly load one YAML/JSON experiment document as an ExperimentSpec."""

    if not isinstance(path, Path):
        raise SchemaError("must be a pathlib.Path", path="spec")
    return from_data(ExperimentSpec, _load_yaml(path), path="spec")


def validate_spec(path: Path) -> SpecValidationReport:
    spec = load_experiment_spec(path)
    spec_digest = canonical_digest(spec)
    spec_id = stable_artifact_id(
        "spec", spec, schema_version=spec.schema_version
    )
    report_id = stable_artifact_id(
        "validation",
        {"spec_id": spec_id, "spec_digest": spec_digest, "status": "valid"},
        schema_version=VALIDATION_REPORT_SCHEMA_VERSION,
    )
    return SpecValidationReport(
        schema_version=VALIDATION_REPORT_SCHEMA_VERSION,
        producer_pass="validate_spec",
        id=report_id,
        status="valid",
        spec_id=spec_id,
        spec_digest=spec_digest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wafer_frontend")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate-spec", help="strictly validate one MVP experiment YAML"
    )
    validate.add_argument("spec", type=Path)
    subparsers.add_parser(
        "dump-empty-pipeline", help="print the fixed pass state machine"
    )
    run = subparsers.add_parser(
        "run", help="compile, finalize and execute one N7 E1/E2 timing case"
    )
    run.add_argument("--case", choices=("E1", "E2"), required=True)
    run.add_argument("--validation", choices=("timing",), required=True)
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--hardware", type=Path, required=True)
    run.add_argument("--simulation", type=Path, required=True)
    run.add_argument("--mapping", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--npusim", type=Path, required=True)
    run.add_argument("--finalizer", type=Path, required=True)
    run.add_argument("--profile-id")
    run.add_argument("--trace-window", type=int, default=1_000_000)
    run.add_argument("--timeout-seconds", type=int, default=300)
    run.add_argument("--keep-failed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-spec":
            output: object = validate_spec(args.spec)
        elif args.command == "dump-empty-pipeline":
            output = pipeline_description()
        elif args.command == "run":
            from .runner import (
                NaiveRunCase,
                NaiveRunRequest,
                NaiveRunValidation,
                run_naive,
            )

            result = run_naive(
                NaiveRunRequest(
                    case=NaiveRunCase(args.case),
                    validation=NaiveRunValidation(args.validation),
                    spec_path=args.spec,
                    hardware_config_path=args.hardware,
                    simulation_config_path=args.simulation,
                    mapping_config_path=args.mapping,
                    output_dir=args.out,
                    npusim_path=args.npusim,
                    finalizer_path=args.finalizer,
                    profile_id=args.profile_id,
                    trace_window=args.trace_window,
                    timeout_seconds=args.timeout_seconds,
                    keep_failed=args.keep_failed,
                )
            )
            output = result.report
        else:  # pragma: no cover - argparse enforces the closed command set.
            raise AssertionError(f"unhandled command {args.command!r}")
        sys.stdout.write(canonical_json(output) + "\n")
        return 0
    except FrontendError as error:
        sys.stderr.write(str(error) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
