#!/usr/bin/env python3
"""Run the C2 inter/intra-die 2x2 ablation and emit stable evidence."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Callable, Mapping

import yaml


CASES = {
    "A00": ("naive", "off"),
    "A10": ("swizzle_topo", "off"),
    "A01": ("naive", "auto"),
    "A11": ("swizzle_topo", "auto"),
}
PROJECTION_STAGE_INDEX = 6
_CONFIG_DIGEST_FIELDS = (
    "spec_digest",
    "fabric_digest",
    "hardware_sha256",
    "simulation_sha256",
    "mapping_sha256",
)


def derive_spec(base: Mapping[str, object], inter: str, intra_mode: str) -> dict[str, object]:
    result = deepcopy(dict(base))
    policy = result.setdefault("policy", {})
    if not isinstance(policy, dict):
        raise ValueError("base YAML policy must be a mapping")
    policy["inter_die"] = inter
    policy["intra_die"] = "optimized"
    return result


def _report_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    report = getattr(value, "report", value)
    if isinstance(report, dict):
        return report
    from llm.frontend.wafer_frontend.schema.serde import to_primitive
    primitive = to_primitive(report)
    if not isinstance(primitive, dict):
        raise TypeError("runner must return a report or result containing one")
    return primitive


def _available_case_evidence(
    report: Mapping[str, object], provenance: Mapping[str, object],
) -> dict[str, object]:
    """Return runner evidence when it is present without constraining dry runs."""
    evidence: dict[str, object] = {}
    inputs = report.get("inputs")
    if isinstance(inputs, Mapping):
        config_digests = {
            name: inputs[name] for name in _CONFIG_DIGEST_FIELDS if name in inputs
        }
        if config_digests:
            evidence["config_digests"] = config_digests
    for name in ("profile_id", "policy_selections", "pass_receipts", "stage_digests"):
        if name in provenance:
            evidence[name] = provenance[name]
    return evidence


def build_comparison(reports: Mapping[str, Mapping[str, object]], *, scope: str) -> dict[str, object]:
    normalized: dict[str, dict[str, object]] = {}
    for name in CASES:
        report = reports[name]
        runtime = report.get("runtime")
        provenance = report.get("provenance")
        if not isinstance(runtime, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError(f"{name} report lacks runtime/provenance")
        stages = provenance.get("stage_digests")
        cycles = runtime.get("makespan_cycles")
        if not isinstance(stages, (list, tuple)) or len(stages) <= PROJECTION_STAGE_INDEX:
            raise ValueError(f"{name} lacks projection stage digest")
        if not isinstance(cycles, (int, float)) or cycles <= 0:
            raise ValueError(f"{name} makespan_cycles must be positive")
        calibration = report.get("_intra_calibration_evidence", [])
        calibrated = (
            isinstance(calibration, list) and bool(calibration)
            and all(isinstance(row, Mapping) and row.get("calibrated") is True for row in calibration)
        )
        normalized[name] = {
            "inter_die": CASES[name][0], "intra_die": CASES[name][1],
            "calibrated": calibrated,
            "calibration_evidence": calibration,
            "makespan_cycles": cycles,
            "projection_stage_digest": stages[PROJECTION_STAGE_INDEX],
            "report_id": report.get("id"),
            **_available_case_evidence(report, provenance),
        }
    for left, right in (("A00", "A01"), ("A10", "A11")):
        if normalized[left]["projection_stage_digest"] != normalized[right]["projection_stage_digest"]:
            raise ValueError(f"fixed-inter projection digest changed: {left}/{right}")
    c = {name: float(normalized[name]["makespan_cycles"]) for name in CASES}
    metrics = {
        "intra_speedup_naive_inter": c["A00"] / c["A01"],
        "intra_speedup_swizzle_inter": c["A10"] / c["A11"],
        "inter_speedup_naive_intra": c["A00"] / c["A10"],
        "inter_speedup_optimized_intra": c["A01"] / c["A11"],
        "combined_speedup": c["A00"] / c["A11"],
        "interaction": c["A11"] - c["A10"] - c["A01"] + c["A00"],
    }
    calibrated = all(bool(row["calibrated"]) for row in normalized.values())
    return {"schema_version": "wafer_frontend.intra_die_ablation/v3", "scope": scope,
            "status": "pass" if calibrated else "fail_uncalibrated",
            "calibrated": calibrated,
            "cases": normalized, "metrics": metrics}


def markdown(matrix: Mapping[str, object]) -> str:
    cases = matrix["cases"]
    metrics = matrix["metrics"]
    assert isinstance(cases, Mapping) and isinstance(metrics, Mapping)
    lines = ["# Intra-die 2x2 ablation", "", f"Scope: `{matrix['scope']}`; status: `{matrix['status']}`; calibrated: `{str(matrix['calibrated']).lower()}`.", "",
             "| Case | inter_die | intra_die | cycles | projection digest |", "|---|---|---|---:|---|"]
    for name in CASES:
        row = cases[name]; assert isinstance(row, Mapping)
        lines.append(f"| {name} | {row['inter_die']} | {row['intra_die']} | {row['makespan_cycles']} | `{row['projection_stage_digest']}` |")
    lines.extend(["", "## Comparison (interaction = A11 - A10 - A01 + A00 cycles)", ""] + [f"- {key}: `{value:.9g}`" for key, value in metrics.items()])
    return "\n".join(lines) + "\n"


def run_matrix(base: Mapping[str, object], output: Path,
               runner: Callable[[str, dict[str, object], Path], object], *,
               scope: str = "dense_tp2_gemm_rs_common_ir2") -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, object]] = {}
    for name, (inter, intra) in CASES.items():
        spec = derive_spec(base, inter, intra)
        case_dir = output / name
        existing_report = case_dir / "run" / "run_report.json"
        if existing_report.is_file():
            loaded = json.loads(existing_report.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"{name} existing run report must be an object")
            calibration_path = case_dir / "run" / "compile" / "intra_die_v2_calibration_evidence.json"
            loaded["_intra_calibration_evidence"] = (
                json.loads(calibration_path.read_text(encoding="utf-8"))
                if calibration_path.is_file() else []
            )
            reports[name] = loaded
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        spec_path = case_dir / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=True), encoding="utf-8")
        reports[name] = _report_dict(runner(name, spec, spec_path))
    matrix = build_comparison(reports, scope=scope)
    (output / "matrix.json").write_text(json.dumps(matrix, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "comparison.md").write_text(markdown(matrix), encoding="utf-8")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", default="dense_tp2_gemm_rs_common_ir2")
    parser.add_argument("--dry-compile", action="store_true")
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--npusim", type=Path)
    parser.add_argument("--finalizer", type=Path)
    parser.add_argument("--case", choices=("E1", "E2"), default="E1")
    parser.add_argument("--profile-id")
    args = parser.parse_args()
    base = yaml.safe_load(args.base_spec.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("base YAML must contain a mapping")
    if args.dry_compile:
        def dry(name: str, _spec: dict[str, object], _path: Path) -> object:
            inter = CASES[name][0]
            cycles = {"A00": 100, "A10": 80, "A01": 75, "A11": 55}[name]
            return {"id": f"dry-{name}", "runtime": {"makespan_cycles": cycles},
                    "provenance": {"stage_digests": ["0"] * 6 + [inter]}}
        run_matrix(base, args.output, dry, scope=f"{args.scope}:dry_compile")
        return 0
    required = (args.hardware, args.simulation, args.mapping, args.npusim, args.finalizer)
    if any(value is None for value in required):
        parser.error("production mode requires --hardware --simulation --mapping --npusim --finalizer")
    from llm.frontend.wafer_frontend import (
        NaiveRunCase, NaiveRunRequest, NaiveRunValidation, run_naive,
    )

    def production(name: str, _spec: dict[str, object], spec_path: Path) -> object:
        assert args.hardware is not None and args.simulation is not None
        assert args.mapping is not None and args.npusim is not None
        assert args.finalizer is not None
        from run_intra_die_performance import build_mode_options
        import hashlib
        hardware_digest = hashlib.sha256(args.hardware.read_bytes()).hexdigest()
        simulation_digest = hashlib.sha256(args.simulation.read_bytes()).hexdigest()
        result = run_naive(NaiveRunRequest(
            case=NaiveRunCase(args.case),
            validation=NaiveRunValidation.TIMING,
            spec_path=spec_path,
            hardware_config_path=args.hardware,
            simulation_config_path=args.simulation,
            mapping_config_path=args.mapping,
            output_dir=args.output / name / "run",
            npusim_path=args.npusim,
            finalizer_path=args.finalizer,
            profile_id=args.profile_id,
            repeat=3,
            intra_die_refine_options=build_mode_options(
                CASES[name][1], hardware_digest=hardware_digest,
                simulation_digest=simulation_digest,
            ),
        ))
        report = _report_dict(result)
        calibration_path = result.output_dir / "compile" / "intra_die_v2_calibration_evidence.json"
        report["_intra_calibration_evidence"] = json.loads(
            calibration_path.read_text(encoding="utf-8")
        )
        return report

    run_matrix(base, args.output, production, scope=args.scope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
