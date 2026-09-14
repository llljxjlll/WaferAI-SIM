#!/usr/bin/env python3
"""Run the bounded rectangular-mesh acceptance evidence set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "cases.json"
SCHEMA_VERSION = "rect_mesh_e2e.acceptance_cases/v1alpha1"
RUN_SCHEMA_VERSION = "rect_mesh_e2e.acceptance_run/v1alpha1"
CLASSIFICATIONS = {
    "preflight_pass",
    "runtime_pass",
    "compile_only",
    "known_blocker",
}
ALLOWED_MODULES = {
    "unittest",
    "llm.test.frontend.integration.run_dense_sequence_runtime_canary",
    "llm.test.frontend.integration.run_dense_training_sequence_runtime_canary",
    "llm.test.frontend.integration.run_dense_external_offload_runtime_canary",
    "llm.test.frontend.integration.run_moe_compile_sequence_canary",
}
ALLOWED_UNITTEST_TARGETS = {
    "test_workload_shape_matrix",
    "llm.test.frontend.unit.test_moe_full_model_compile_sequence",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields mismatch: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("acceptance config must be one JSON object")
    _require_exact_keys(
        value,
        {"schema_version", "description", "default_case_ids", "cases"},
        "config",
    )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported config schema: {value['schema_version']!r}")
    if not isinstance(value["description"], str) or not value["description"]:
        raise ValueError("config description must be a non-empty string")
    if not isinstance(value["cases"], list) or not value["cases"]:
        raise ValueError("config cases must be a non-empty list")

    case_ids: set[str] = set()
    for index, case in enumerate(value["cases"]):
        if not isinstance(case, dict):
            raise ValueError(f"case[{index}] must be one JSON object")
        _validate_case(case, index=index)
        case_id = case["case_id"]
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        case_ids.add(case_id)

    defaults = value["default_case_ids"]
    if not isinstance(defaults, list) or not defaults:
        raise ValueError("default_case_ids must be a non-empty list")
    if len(defaults) != len(set(defaults)):
        raise ValueError("default_case_ids contains duplicates")
    unknown = set(defaults) - case_ids
    if unknown:
        raise ValueError(f"default_case_ids names unknown cases: {sorted(unknown)}")
    case_by_id = {case["case_id"]: case for case in value["cases"]}
    blocked_defaults = [
        case_id
        for case_id in defaults
        if case_by_id[case_id]["classification"] == "known_blocker"
    ]
    if blocked_defaults:
        raise ValueError(
            "known blockers must be selected explicitly: "
            f"{sorted(blocked_defaults)}"
        )
    return value


def _validate_case(case: Mapping[str, Any], *, index: int) -> None:
    _require_exact_keys(
        case,
        {
            "case_id",
            "classification",
            "scope",
            "artifact_output",
            "entrypoint",
            "arguments",
            "python_paths",
            "expected",
        },
        f"case[{index}]",
    )
    case_id = case["case_id"]
    if (
        not isinstance(case_id, str)
        or not case_id
        or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in case_id)
    ):
        raise ValueError(f"case[{index}].case_id is not canonical")
    if case["classification"] not in CLASSIFICATIONS:
        raise ValueError(f"case[{index}] has unknown classification")
    if not isinstance(case["scope"], str) or not case["scope"]:
        raise ValueError(f"case[{index}].scope must be non-empty")
    artifact_output = Path(case["artifact_output"])
    if not artifact_output.is_absolute() or artifact_output.parent != Path("/tmp"):
        raise ValueError(f"case[{index}].artifact_output must be directly under /tmp")

    entrypoint = case["entrypoint"]
    if entrypoint not in ALLOWED_MODULES:
        raise ValueError(f"case[{index}] entrypoint is not allowlisted: {entrypoint}")
    arguments = case["arguments"]
    if not isinstance(arguments, list) or not all(
        isinstance(item, str) for item in arguments
    ):
        raise ValueError(f"case[{index}].arguments must be a string list")
    if entrypoint == "unittest":
        targets = [item for item in arguments if not item.startswith("-")]
        if len(targets) != 1 or targets[0] not in ALLOWED_UNITTEST_TARGETS:
            raise ValueError(f"case[{index}] has a non-allowlisted unittest target")
        if any(item.startswith("-") and item != "-v" for item in arguments):
            raise ValueError(f"case[{index}] has a non-allowlisted unittest flag")

    python_paths = case["python_paths"]
    if not isinstance(python_paths, list) or not all(
        isinstance(item, str) and item for item in python_paths
    ):
        raise ValueError(f"case[{index}].python_paths must be a string list")

    expected = case["expected"]
    if not isinstance(expected, dict):
        raise ValueError(f"case[{index}].expected must be one JSON object")
    _require_exact_keys(
        expected,
        {"return_codes", "required_output_substrings", "forbidden_output_substrings"},
        f"case[{index}].expected",
    )
    codes = expected["return_codes"]
    if not isinstance(codes, list) or not codes or not all(
        isinstance(item, int) for item in codes
    ):
        raise ValueError(f"case[{index}] return_codes must be a non-empty int list")
    for key in ("required_output_substrings", "forbidden_output_substrings"):
        items = expected[key]
        if not isinstance(items, list) or not all(
            isinstance(item, str) and item for item in items
        ):
            raise ValueError(f"case[{index}] {key} must be a string list")
    if case["classification"] == "known_blocker":
        if 0 in codes or not expected["required_output_substrings"]:
            raise ValueError("known blockers require a nonzero code and diagnostic text")
    elif codes != [0]:
        raise ValueError("non-blocker cases must require exactly return code 0")


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def source_binding() -> dict[str, Any]:
    commit = _git_value("rev-parse", "HEAD")
    tree = _git_value("rev-parse", "HEAD^{tree}")
    tracked_changes = _git_value("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "tree": tree,
        "tracked_worktree_clean": not bool(tracked_changes),
        "tracked_change_digest": _sha256_bytes(tracked_changes.encode("utf-8")),
        "binding_digest": _canonical_digest(
            {
                "commit": commit,
                "tree": tree,
                "tracked_changes": tracked_changes,
            }
        ),
    }


def _placeholders(
    args: argparse.Namespace, case: Mapping[str, Any]
) -> dict[str, str]:
    simulation = os.path.relpath(Path(args.simulation).resolve(), ROOT)
    return {
        "repo": str(ROOT),
        "python": str(Path(args.python).resolve()),
        "finalizer": str(Path(args.finalizer).resolve()),
        "npusim": str(Path(args.npusim).resolve()),
        # NpuSim resolves the DRAM config nested in this JSON from its process
        # working directory.  Keep the outer path relative to the same ROOT.
        "simulation": simulation,
        "timeout": str(args.timeout),
        "case_output": str(Path(case["artifact_output"])),
    }


def _expand(value: str, placeholders: Mapping[str, str]) -> str:
    try:
        return value.format_map(placeholders)
    except KeyError as error:
        raise ValueError(f"unknown command placeholder: {error.args[0]}") from error


def build_command(
    case: Mapping[str, Any], args: argparse.Namespace
) -> tuple[list[str], dict[str, str]]:
    placeholders = _placeholders(args, case)
    command = [
        placeholders["python"],
        "-B",
        "-m",
        case["entrypoint"],
        *[_expand(item, placeholders) for item in case["arguments"]],
    ]
    python_paths = [_expand(item, placeholders) for item in case["python_paths"]]
    environment = os.environ.copy()
    inherited = environment.get("PYTHONPATH")
    if inherited:
        python_paths.append(inherited)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return command, environment


def classify_observation(
    case: Mapping[str, Any], *, return_code: int, combined_output: str
) -> tuple[str, list[str]]:
    expected = case["expected"]
    problems: list[str] = []
    if return_code not in expected["return_codes"]:
        problems.append(
            f"return_code={return_code} expected={expected['return_codes']}"
        )
    for marker in expected["required_output_substrings"]:
        if marker not in combined_output:
            problems.append(f"missing required output marker: {marker}")
    for marker in expected["forbidden_output_substrings"]:
        if marker in combined_output:
            problems.append(f"observed forbidden output marker: {marker}")
    if problems:
        return "unexpected_result", problems
    if case["classification"] == "known_blocker":
        return "known_blocker_reproduced", []
    return str(case["classification"]), []


def _tool_bindings(args: argparse.Namespace, config_path: Path) -> dict[str, Any]:
    paths = {
        "python": Path(args.python).resolve(),
        "finalizer": Path(args.finalizer).resolve(),
        "npusim": Path(args.npusim).resolve(),
        "simulation": Path(args.simulation).resolve(),
        "config": config_path.resolve(),
        "runner": Path(__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"tool binding files are missing: {sorted(missing)}")
    return {
        name: {"path": str(path), "sha256": _file_digest(path)}
        for name, path in paths.items()
    }


def run_case(
    case: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    command, environment = build_command(case, args)
    output_root = Path(args.output).resolve()
    log_output = output_root / "logs" / case["case_id"]
    log_output.mkdir(parents=True, exist_ok=True)
    artifact_output = Path(_placeholders(args, case)["case_output"])
    artifact_output.mkdir(parents=True, exist_ok=True)
    started_ns = time.monotonic_ns()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=args.timeout,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        return_code = 124
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
    wall_seconds = (time.monotonic_ns() - started_ns) / 1_000_000_000

    (log_output / "stdout.log").write_text(stdout, encoding="utf-8")
    (log_output / "stderr.log").write_text(stderr, encoding="utf-8")
    combined = stdout + "\n" + stderr
    observed, problems = classify_observation(
        case, return_code=return_code, combined_output=combined
    )
    if timed_out:
        observed = "unexpected_result"
        problems.append(f"case exceeded timeout_seconds={args.timeout}")
    return {
        "case_id": case["case_id"],
        "declared_classification": case["classification"],
        "scope": case["scope"],
        "observed_outcome": observed,
        "problems": problems,
        "command": command,
        "artifact_output": str(artifact_output),
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_seconds": round(wall_seconds, 6),
        "stdout": {
            "path": str((log_output / "stdout.log").relative_to(output_root)),
            "bytes": len(stdout.encode("utf-8")),
            "sha256": _sha256_bytes(stdout.encode("utf-8")),
        },
        "stderr": {
            "path": str((log_output / "stderr.log").relative_to(output_root)),
            "bytes": len(stderr.encode("utf-8")),
            "sha256": _sha256_bytes(stderr.encode("utf-8")),
        },
    }


def _select_cases(
    config: Mapping[str, Any], requested: Sequence[str]
) -> list[dict[str, Any]]:
    by_id = {case["case_id"]: case for case in config["cases"]}
    selected_ids = list(requested) if requested else list(config["default_case_ids"])
    unknown = set(selected_ids) - set(by_id)
    if unknown:
        raise ValueError(f"unknown requested cases: {sorted(unknown)}")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("requested case IDs contain duplicates")
    return [by_id[case_id] for case_id in selected_ids]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    build = ROOT / "build-debug-final"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--finalizer", type=Path, default=build / "npusim_program_finalizer")
    parser.add_argument("--npusim", type=Path, default=build / "npusim")
    parser.add_argument(
        "--simulation",
        type=Path,
        default=ROOT / "llm/test/program/p5_behavioral_simulation.json",
    )
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not (args.list or args.dry_run) and args.output is None:
        parser.error("--output is required for an actual run")
    if args.output is None:
        args.output = Path("/tmp/rect_mesh_e2e_acceptance")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    cases = _select_cases(config, args.case)

    if args.list:
        for case in config["cases"]:
            default = case["case_id"] in config["default_case_ids"]
            print(
                f"{case['case_id']} classification={case['classification']} "
                f"default={int(default)} scope={case['scope']}"
            )
        return 0

    if args.dry_run:
        commands = []
        for case in cases:
            command, _ = build_command(case, args)
            commands.append(
                {
                    "case_id": case["case_id"],
                    "classification": case["classification"],
                    "command": command,
                }
            )
        print(json.dumps(commands, ensure_ascii=False, indent=2))
        return 0

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bindings = _tool_bindings(args, config_path)
    source = source_binding()
    observations = []
    for case in cases:
        print(
            f"RUN case={case['case_id']} classification={case['classification']}",
            flush=True,
        )
        observation = run_case(case, args)
        observations.append(observation)
        print(
            f"OBSERVED case={case['case_id']} "
            f"outcome={observation['observed_outcome']} "
            f"wall_seconds={observation['wall_seconds']}",
            flush=True,
        )

    counts: dict[str, int] = {}
    for observation in observations:
        outcome = observation["observed_outcome"]
        counts[outcome] = counts.get(outcome, 0) + 1
    report = {
        "schema_version": RUN_SCHEMA_VERSION,
        "source_binding": source,
        "tool_bindings": bindings,
        "config_digest": _file_digest(config_path),
        "selected_case_ids": [case["case_id"] for case in cases],
        "artifact_outputs": {
            case["case_id"]: case["artifact_output"] for case in cases
        },
        "outcome_counts": dict(sorted(counts.items())),
        "unexpected_result_count": counts.get("unexpected_result", 0),
        "cases": observations,
    }
    report["report_digest"] = _canonical_digest(report)
    report_path = output / "run_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"REPORT path={report_path} digest={report['report_digest']}")
    return 1 if report["unexpected_result_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
