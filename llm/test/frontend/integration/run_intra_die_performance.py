#!/usr/bin/env python3
"""Run a fail-closed OFF/AUTO intra-die performance comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import inspect
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

from intra_die_resource_evidence import write_resource_evidence

from llm.frontend.wafer_frontend import (
    NaiveRunCase,
    NaiveRunRequest,
    NaiveRunResult,
    NaiveRunValidation,
    run_naive,
)


SCHEMA_VERSION = "wafer_frontend.intra_die_performance_compare/v1alpha2"
MODES = ("off", "auto")
CORE16_MODES = ("naive", "auto")
REPEAT = 3
PRE_REFINE_PASS = "intra_die_refine"
SOURCE_IR1_PASS = "placement"
REQUIRED_VALIDATION = ("timing", "address_lifecycle", "transport_control")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one same-input intra-die OFF/AUTO performance comparison."
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
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--expect", choices=("gain", "identity", "no-regression"),
        default="no-regression",
    )
    parser.add_argument("--minimum-gain", type=float, default=0.05)
    parser.add_argument(
        "--cores-per-die", type=int, choices=(2, 16), default=16,
        help="Use 16 for a same-work naive-barrier/AUTO-streaming comparison.",
    )
    return parser


def require_performance_spec(spec_path: Path) -> None:
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("spec YAML must contain a mapping")
    policy = raw.get("policy")
    if not isinstance(policy, Mapping) or policy.get("intra_die") != "optimized":
        raise ValueError("performance comparison requires policy.intra_die=optimized")


def build_mode_options(
    mode: str, *, hardware_digest: str | None = None,
    simulation_digest: str | None = None,
    compute_groups_per_die: int = 2,
) -> object:
    """Construct the unified API while keeping this CLI isolated from schema churn."""
    if mode not in (*MODES, "naive"):
        raise ValueError(f"unsupported performance mode: {mode!r}")
    if compute_groups_per_die not in (2, 16):
        raise ValueError("compute_groups_per_die must be 2 or 16")
    from llm.frontend.wafer_frontend.schema import intra_die_refine as schema

    options_type = getattr(schema, "IntraDieOptimizationOptions", None)
    mode_type = getattr(schema, "IntraDieOptimizationMode", None)
    if options_type is None or mode_type is None:
        raise RuntimeError(
            "OFF/AUTO API is unavailable; expected IntraDieOptimizationOptions "
            "and IntraDieOptimizationMode in schema.intra_die_refine"
        )
    parameters = inspect.signature(options_type).parameters
    core16 = compute_groups_per_die == 16
    naive = mode == "naive"
    kwargs: dict[str, object] = {
        "mode": mode_type("force" if naive else mode),
        "allowed_candidates": (
            ("identity", "split_k_barrier")
            if naive else (
                ("identity", "split_k_barrier", "split_k", "split_k_tree_direct_dma")
                if core16 else ("identity", "split_k")
            )
        ),
        "max_candidates": 8,
        "split_k_parts": (16,) if core16 else (2, 4),
        "temporal_chunks": (1,),
        "compute_groups_per_die": compute_groups_per_die,
        "require_full_compute_groups": core16 and not naive,
        "force_candidate": "split_k_barrier" if naive else None,
        "timing_hardware_digest": hardware_digest,
        "timing_simulation_digest": simulation_digest,
    }
    if "schema_version" in parameters and parameters["schema_version"].default is inspect.Parameter.empty:
        version = getattr(schema, "INTRA_DIE_OPTIMIZATION_OPTIONS_SCHEMA_VERSION", None)
        if not isinstance(version, str):
            raise RuntimeError("unified options schema version constant is unavailable")
        kwargs["schema_version"] = version
    options = options_type(**{key: value for key, value in kwargs.items() if key in parameters})
    validate = getattr(options, "validate", None)
    if callable(validate):
        validate("cli.intra_die_optimization_options")
    return options


def build_request(
    args: argparse.Namespace,
    mode: str,
    *,
    option_factory: Callable[[str], object] = build_mode_options,
) -> NaiveRunRequest:
    parameters = inspect.signature(option_factory).parameters
    compute_groups_per_die = getattr(args, "cores_per_die", 2)
    options = (
        option_factory(
            mode, hardware_digest=_sha256(args.hardware),
            simulation_digest=_sha256(args.simulation),
            compute_groups_per_die=compute_groups_per_die,
        )
        if {
            "hardware_digest", "simulation_digest", "compute_groups_per_die",
        }.issubset(parameters)
        else option_factory(
            mode, hardware_digest=_sha256(args.hardware),
            simulation_digest=_sha256(args.simulation),
        )
        if "hardware_digest" in parameters and "simulation_digest" in parameters
        else option_factory(mode)
    )
    return NaiveRunRequest(
        case=NaiveRunCase(args.case),
        validation=NaiveRunValidation.TIMING,
        spec_path=args.spec,
        hardware_config_path=args.hardware,
        simulation_config_path=args.simulation,
        mapping_config_path=args.mapping,
        output_dir=args.output / mode,
        npusim_path=args.npusim,
        finalizer_path=args.finalizer,
        profile_id=args.profile_id,
        timeout_seconds=args.timeout_seconds,
        repeat=REPEAT,
        intra_die_refine_options=options,
    )


def _report_dict(value: object) -> dict[str, object]:
    report = getattr(value, "report", value)
    if isinstance(report, dict):
        return report
    if is_dataclass(report):
        from llm.frontend.wafer_frontend.schema.serde import to_primitive

        primitive = to_primitive(report)
        if isinstance(primitive, dict):
            return primitive
    raise TypeError("runner must return a report or result containing one")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipts(report: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("run report lacks provenance")
    raw = provenance.get("pass_receipts")
    if not isinstance(raw, (list, tuple)) or not all(isinstance(row, Mapping) for row in raw):
        raise ValueError("run report lacks pass receipts")
    return tuple(raw)  # type: ignore[arg-type]


def _pass_output_digest(report: Mapping[str, object], pass_name: str) -> str:
    matches = [row for row in _receipts(report) if row.get("pass_name") == pass_name]
    if len(matches) != 1 or not isinstance(matches[0].get("output_digest"), str):
        raise ValueError(f"run report must contain exactly one {pass_name!r} receipt")
    return str(matches[0]["output_digest"])


def _pass_input_digest(report: Mapping[str, object], pass_name: str) -> str:
    matches = [row for row in _receipts(report) if row.get("pass_name") == pass_name]
    if len(matches) != 1 or not isinstance(matches[0].get("input_digest"), str):
        raise ValueError(f"run report must contain exactly one {pass_name!r} receipt")
    return str(matches[0]["input_digest"])


def _inter_die_selection(report: Mapping[str, object]) -> Mapping[str, object]:
    provenance = report.get("provenance")
    selections = provenance.get("policy_selections") if isinstance(provenance, Mapping) else None
    matches = [row for row in selections or () if isinstance(row, Mapping) and row.get("kind") == "inter_die"]
    if len(matches) != 1:
        raise ValueError("run report must contain exactly one inter-die policy selection")
    return matches[0]


def _run_evidence(report: Mapping[str, object]) -> dict[str, object]:
    runtime = report.get("runtime")
    artifact = report.get("artifact")
    validation = report.get("validation")
    if not all(isinstance(value, Mapping) for value in (runtime, artifact, validation)):
        raise ValueError("run report lacks runtime/artifact/validation evidence")
    assert isinstance(runtime, Mapping) and isinstance(artifact, Mapping)
    assert isinstance(validation, Mapping)
    cycles = runtime.get("makespan_cycles")
    if not isinstance(cycles, int) or cycles <= 0:
        raise ValueError("makespan_cycles must be a positive integer")
    if runtime.get("repeat") != REPEAT or runtime.get("repeat_signature_stable") is not True:
        raise ValueError(f"performance evidence requires exactly {REPEAT} stable simulator runs")
    for field in REQUIRED_VALIDATION:
        if validation.get(field) != "pass":
            raise ValueError(f"performance evidence requires validation.{field}=pass")
    artifact_sha = artifact.get("artifact_sha256")
    if not isinstance(artifact_sha, str) or len(artifact_sha) != 64:
        raise ValueError("run report lacks artifact_sha256")
    return {
        "report_id": report.get("id"),
        "makespan_cycles": cycles,
        "repeat": REPEAT,
        "repeat_signature_stable": True,
        "artifact_sha256": artifact_sha,
        "record_count": artifact.get("record_count"),
        "core_count": artifact.get("core_count"),
        "done_by_core": runtime.get("done_by_core"),
        "ack_by_core": runtime.get("ack_by_core"),
        "validation": {field: "pass" for field in REQUIRED_VALIDATION},
    }


def _equivalence_value(report: Mapping[str, object], workload_sha256: str) -> dict[str, object]:
    inputs = report.get("inputs")
    tools = report.get("tools")
    provenance = report.get("provenance")
    if not all(isinstance(value, Mapping) for value in (inputs, tools, provenance)):
        raise ValueError("run report lacks input/tool/provenance equivalence evidence")
    assert isinstance(inputs, Mapping) and isinstance(tools, Mapping)
    assert isinstance(provenance, Mapping)
    return {
        "workload_yaml_sha256": workload_sha256,
        "spec_digest": inputs.get("spec_digest"),
        "fabric_digest": inputs.get("fabric_digest"),
        "hardware_sha256": inputs.get("hardware_sha256"),
        "simulation_sha256": inputs.get("simulation_sha256"),
        "mapping_sha256": inputs.get("mapping_sha256"),
        "finalizer_sha256": tools.get("finalizer_sha256"),
        "npusim_sha256": tools.get("npusim_sha256"),
        "profile_id": provenance.get("profile_id"),
        "inter_die_policy": dict(_inter_die_selection(report)),
        "source_ir1_digest": _pass_output_digest(report, SOURCE_IR1_PASS),
        "pre_refine_projection_digest": _pass_input_digest(report, PRE_REFINE_PASS),
    }


def _load_search_evidence(output: Path) -> object:
    path = output / "compile" / "intra_die_v2_search_decisions.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _load_calibration_evidence(output: Path) -> object:
    path = output / "compile" / "intra_die_v2_calibration_evidence.json"
    if not path.is_file():
        raise ValueError(f"missing timing calibration evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _calibration_summary(value: object, measured_cycles: int) -> dict[str, object]:
    if not isinstance(value, list) or not value:
        raise ValueError("calibration evidence must be a non-empty list")
    rows = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("calibration evidence row must be an object")
        if raw.get("simulator_measured_makespan_cycles") != measured_cycles:
            raise ValueError("calibration measured cycles disagree with run report")
        if raw.get("simulator_calls_used") != REPEAT or raw.get("reserved_simulator_calls_for_final_evidence") != REPEAT:
            raise ValueError("calibration must close the three-call final evidence budget")
        if raw.get("repeat_signature_stable") is not True:
            raise ValueError("calibration repeat signature must be stable")
        if raw.get("calibrated") is not True or not isinstance(raw.get("relative_error"), (int, float)) or raw["relative_error"] > 0.20:
            raise ValueError("selected candidate prediction error exceeds 20 percent")
        rows.append({
            "id": raw.get("id"), "selected_candidate_ref": raw.get("selected_candidate_ref"),
            "predicted_makespan_cycles": raw.get("predicted_makespan_cycles"),
            "simulator_measured_makespan_cycles": measured_cycles,
            "relative_error": raw.get("relative_error"), "calibrated": True,
        })
    return {"status": "pass", "rows": rows, "maximum_relative_error": max(row["relative_error"] for row in rows)}


def _selection_summary(search: object) -> dict[str, object]:
    rows = search if isinstance(search, list) else []
    selected: list[dict[str, object]] = []
    simulator_calls = 0
    maximum_generated_candidates = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        simulator_calls += int(row.get("simulator_calls_during_search", 0))
        generated = row.get("generated_candidate_count", 0)
        if isinstance(generated, int):
            maximum_generated_candidates = max(maximum_generated_candidates, generated)
        ref = row.get("selected_candidate_ref")
        kind = None
        for candidate in row.get("candidates", ()) if isinstance(row.get("candidates"), list) else ():
            if isinstance(candidate, Mapping) and candidate.get("id") == ref:
                kind = candidate.get("kind")
        selected.append({
            "decision_id": row.get("id"),
            "selected_candidate_ref": ref,
            "selected_candidate_kind": kind,
            "selection_reason": row.get("selection_reason"),
        })
    return {
        "decisions": selected,
        "simulator_calls_during_search": simulator_calls,
        "maximum_generated_candidates": maximum_generated_candidates,
    }


def build_comparison(
    reports: Mapping[str, Mapping[str, object]],
    *,
    workload_sha256: str,
    searches: Mapping[str, object] | None = None,
    expectation: str = "no-regression",
    minimum_gain: float = 0.05,
    resource_evidence: Mapping[str, object] | None = None,
    calibrations: Mapping[str, object] | None = None,
    modes: tuple[str, str] = MODES,
    cores_per_die: int = 2,
    expected_die_count: int | None = None,
) -> dict[str, object]:
    if modes not in (MODES, CORE16_MODES):
        raise ValueError("unsupported comparison modes")
    if set(reports) != set(modes):
        raise ValueError(
            "comparison requires exactly " + " and ".join(modes) + " reports"
        )
    if not 0.0 <= minimum_gain < 1.0:
        raise ValueError("minimum_gain must be in [0, 1)")
    equivalence = {
        mode: _equivalence_value(reports[mode], workload_sha256) for mode in modes
    }
    baseline_mode = modes[0]
    if equivalence[baseline_mode] != equivalence["auto"]:
        mismatches = sorted(
            key for key in equivalence[baseline_mode]
            if equivalence[baseline_mode][key] != equivalence["auto"][key]
        )
        raise ValueError(
            f"{baseline_mode}/AUTO input equivalence failed: "
            + ", ".join(mismatches)
        )
    runs = {mode: _run_evidence(reports[mode]) for mode in modes}
    if cores_per_die == 16:
        if expected_die_count is None or expected_die_count < 1:
            raise ValueError("16-core comparison requires expected_die_count")
        expected_ids = list(range(expected_die_count * 16))
        for mode in modes:
            run = runs[mode]
            done = run.get("done_by_core")
            ack = run.get("ack_by_core")
            done_ids = sorted(
                int(row[0]) for row in done or ()
                if isinstance(row, (list, tuple)) and len(row) == 2
            )
            ack_ids = sorted(
                int(row[0]) for row in ack or ()
                if isinstance(row, (list, tuple)) and len(row) == 2
            )
            if (
                run.get("core_count") != len(expected_ids)
                or done_ids != expected_ids
                or ack_ids != expected_ids
            ):
                raise ValueError(
                    f"{mode} must activate exactly 16 cores on every die"
                )
    baseline = int(runs[baseline_mode]["makespan_cycles"])
    optimized = int(runs["auto"]["makespan_cycles"])
    speedup = baseline / optimized
    gain = 1.0 - optimized / baseline
    no_regression = optimized <= baseline
    meets_gain = gain >= minimum_gain
    search_summaries = {
        mode: _selection_summary((searches or {}).get(mode, [])) for mode in modes
    }
    calibration_summaries = (
        {mode: _calibration_summary(calibrations[mode], int(runs[mode]["makespan_cycles"])) for mode in modes}
        if calibrations is not None else {}
    )
    if any(row["simulator_calls_during_search"] != 0 for row in search_summaries.values()):
        raise ValueError("product AUTO search must not call the simulator")
    if any(row["maximum_generated_candidates"] > 8 for row in search_summaries.values()):
        raise ValueError("product candidate count must not exceed 8")
    if any(not row["decisions"] for row in search_summaries.values()):
        raise ValueError("OFF/AUTO comparison requires persisted search decisions")
    baseline_kinds = {
        row.get("selected_candidate_kind")
        for row in search_summaries[baseline_mode]["decisions"]
        if isinstance(row, Mapping)
    }
    expected_baseline_kind = (
        "split_k_fallback" if baseline_mode == "naive" else "identity"
    )
    if baseline_kinds != {expected_baseline_kind}:
        raise ValueError(
            f"{baseline_mode} must select {expected_baseline_kind} for every profile"
        )
    auto_kinds = {
        row.get("selected_candidate_kind")
        for row in search_summaries["auto"]["decisions"]
        if isinstance(row, Mapping)
    }
    if expectation == "gain" and not meets_gain:
        raise ValueError(f"AUTO gain {gain:.6f} is below required {minimum_gain:.6f}")
    if expectation == "identity" and auto_kinds != {"identity"}:
        raise ValueError("sync-bound AUTO must select identity")
    if expectation in ("identity", "no-regression") and not no_regression:
        raise ValueError("AUTO regressed against the same-input OFF baseline")
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "producer_pass": "intra_die_performance_runner",
        "status": "pass",
        "expectation": expectation,
        "minimum_gain": minimum_gain,
        "equivalence": {"status": "pass", **equivalence[baseline_mode]},
        "runs": runs,
        "selection": search_summaries,
        "resource_evidence": dict(resource_evidence or {}),
        "calibration": calibration_summaries,
        "metrics": {
            "baseline_mode": baseline_mode,
            "baseline_makespan_cycles": baseline,
            "identity_makespan_cycles": (
                baseline if baseline_mode == "off" else None
            ),
            "auto_makespan_cycles": optimized,
            "speedup": speedup,
            "gain": gain,
            "no_regression": no_regression,
            "meets_minimum_gain": meets_gain,
        },
        "budget": {
            "finalizer_calls": 2,
            "final_simulator_calls": REPEAT * 2,
            "product_search_simulator_calls": 0,
        },
    }
    digest = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**semantic, "id": f"intra_die_performance_compare_{digest[:16]}"}


def run_comparison(
    args: argparse.Namespace,
    *,
    runner: Callable[[NaiveRunRequest], NaiveRunResult] = run_naive,
    option_factory: Callable[[str], object] = build_mode_options,
) -> dict[str, object]:
    require_performance_spec(args.spec)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("output directory must not already exist")
    args.output.mkdir(parents=True)
    reports: dict[str, dict[str, object]] = {}
    searches: dict[str, object] = {}
    resources: dict[str, object] = {}
    calibrations: dict[str, object] = {}
    cores_per_die = getattr(args, "cores_per_die", 2)
    modes = CORE16_MODES if cores_per_die == 16 else MODES
    for mode in modes:
        result = runner(build_request(args, mode, option_factory=option_factory))
        reports[mode] = _report_dict(result)
        searches[mode] = _load_search_evidence(args.output / mode)
        resources[mode] = write_resource_evidence(args.output / mode, REPEAT)
        calibrations[mode] = _load_calibration_evidence(args.output / mode)
    hardware = json.loads(args.hardware.read_text(encoding="utf-8"))
    die = hardware.get("die", {}) if isinstance(hardware, Mapping) else {}
    expected_die_count = (
        int(die.get("x", 1)) * int(die.get("y", die.get("x", 1)))
        if isinstance(die, Mapping) else 1
    )
    comparison = build_comparison(
        reports,
        workload_sha256=_sha256(args.spec),
        searches=searches,
        expectation=args.expect,
        minimum_gain=args.minimum_gain,
        resource_evidence=resources,
        calibrations=calibrations,
        modes=modes,
        cores_per_die=cores_per_die,
        expected_die_count=expected_die_count,
    )
    path = args.output / "comparison.json"
    path.write_text(json.dumps(comparison, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (args.output / "SUCCESS").write_text(comparison["id"] + "\n", encoding="utf-8")
    return comparison


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_comparison(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
