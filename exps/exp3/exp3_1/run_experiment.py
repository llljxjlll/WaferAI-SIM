#!/usr/bin/env python3
"""Run all 48 Exp3.1 cases and emit the three requested paired metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

from case_matrix import build_logical_cases, emit_artifacts
from gpu_lut import load_gpu_lut
from legacy_alignment import (
    calibration_evidence_audit, compatibility_audit, dense_native, moe_d4_anchor, scale_moe_anchor_to_d,
)
from resource_replay import replay_dense_candidates, replay_moe_candidate
from result_schema import build_case_states, validate_state_rows


HERE = Path(__file__).resolve().parent
CSV_FIELDS = (
    "case_id", "operator_family", "stage", "model_or_moe_config", "D", "Px", "Py", "S",
    "state", "comparison", "inter_enabled", "intra_enabled", "compute_source", "algorithm",
    "compute_time_ns", "communication_time_ns", "total_time_ns", "baseline_state", "speedup",
    "inter_port_time_ns",
    "ideal_speedup", "attainment", "evidence", "production_expert_placement_closed",
    "valid_flops", "padded_flops", "config_sha256", "gpu_yaml_sha256",
    "required_shapes_sha256", "git_commit",
)


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HERE, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _native(case: object) -> dict[str, object]:
    if getattr(case, "operator_family") == "gemm_rs":
        return dense_native(case)
    return scale_moe_anchor_to_d(moe_d4_anchor(case), int(getattr(case, "D")))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(*, gpu_yaml: Path, output_dir: Path, generated_dir: Path, allow_placeholder: bool) -> dict[str, object]:
    generated_paths = emit_artifacts(generated_dir)
    required_path = generated_paths["required_gpu_shapes"]
    lut = load_gpu_lut(required_path, gpu_yaml, allow_placeholder=allow_placeholder)
    history = compatibility_audit(raise_on_error=True)
    calibration = calibration_evidence_audit()
    config = {
        "schema_version": 6,
        "die_counts": [6, 9, 36],
        "meshes": {"6": [2, 3], "9": [3, 3], "36": [6, 6]},
        "sequence_lengths": [2304, 36864],
        "state_order": ["W00", "W11", "C00", "C10", "G00", "G10"],
        "moe_extrapolation": "D4 Exp1.2 compact/H128/architecture anchor; compute*4/D; communication*sqrt(4/D)",
        "gpu_placeholder_allowed": allow_placeholder,
        "native_full": "single-die 16-core comparison: W00 uses canonical Pm=4,Pn=4,Pk=1; W11 uses the Exp1-derived adaptive 16-core intra schedule plus inter streaming; both retain the same base HBM workload",
        "native_inter_only": "fixed canonical 16-core Pm=4,Pn=4,Pk=1 mapping; C00/C10 retain identical base HBM workload and explicit core-to-two-D2D-port routing; only inter-die serial versus streaming scheduling differs",
    }
    provenance = {
        "config_sha256": _digest(config),
        "gpu_yaml_sha256": lut.measurement_sha256,
        "required_shapes_sha256": lut.required_sha256,
        "normalized_gpu_lut_sha256": lut.normalized_sha256,
        "git_commit": _git_commit(),
    }
    rows: list[dict[str, object]] = []
    case_audit: list[dict[str, object]] = []
    for case in build_logical_cases():
        native = _native(case)
        replay = (
            replay_dense_candidates(case, lut, native)
            if case.operator_family == "gemm_rs"
            else replay_moe_candidate(case, lut, native)
        )
        selected = replay["selected"]
        case_rows = build_case_states(case, native, selected, provenance)
        rows.extend(case_rows)
        case_audit.append({
            "case_id": case.case_id,
            "intra_ablation": native.get("intra_ablation"),
            "native_evidence": native["evidence"],
            "controlled_inter_port": {
                "C00": case_rows[2]["inter_port"],
                "C10": case_rows[3]["inter_port"],
            },
            "native_analytical_extrapolation": bool(native.get("analytical_extrapolation", False)),
            "selected_algorithm": selected.algorithm,
            "selected_pair": selected.to_dict(),
            "coarse_diagnostic": replay["coarse_diagnostic"],
            "candidates": [item.to_dict() for item in replay["candidates"]],
            "production_expert_placement_closed": case_rows[-1]["production_expert_placement_closed"],
        })
    validate_state_rows(rows)
    if len(rows) != 288:
        raise AssertionError(f"expected 288 state rows, got {len(rows)}")

    comparisons = []
    by_case: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), {})[str(row["state"])] = row
    for case_id, states in by_case.items():
        for comparison, off, on in (
            ("native_full", "W00", "W11"),
            ("native_inter_only", "C00", "C10"),
            ("gpu_inter", "G00", "G10"),
        ):
            baseline_time = float(states[off]["total_time_ns"])
            optimized_time = float(states[on]["total_time_ns"])
            comparisons.append({
                "case_id": case_id, "comparison": comparison,
                "baseline_state": off, "optimized_state": on,
                "baseline_time_ns": baseline_time,
                "optimized_time_ns": optimized_time,
                "speedup": baseline_time / optimized_time, "ideal_speedup": states[on]["ideal_speedup"],
                "attainment": states[on]["attainment"], "D": states[on]["D"], "S": states[on]["S"],
                "operator_family": states[on]["operator_family"], "stage": states[on]["stage"],
                "model_or_moe_config": states[on]["model_or_moe_config"], "evidence": states[on]["evidence"],
            })
    if len(comparisons) != 144:
        raise AssertionError("expected 144 paired comparisons")

    result = {
        "schema_version": 6,
        "experiment": "exp3_1_inter_die_ablation",
        "status": "placeholder_smoke_test" if lut.uses_placeholders else "measured_gpu_run",
        "summary": {
            "logical_cases": len(by_case), "state_rows": len(rows),
            "paired_comparisons": len(comparisons), "gpu_lut_semantic_entries": 120,
            "gpu_lut_unique_shapes": len(lut.latencies_ns),
            "placeholder_shapes": len(lut.placeholder_shapes),
            "history_compatibility_passed": history["passed"],
        },
        "config": config, "provenance": provenance, "records": rows, "comparisons": comparisons,
    }
    audit = {
        "schema_version": 6, "status": result["status"], "history_compatibility": history,
        "calibration_evidence": calibration,
        "gpu_lut": {
            "evidence": lut.evidence, "unique_shapes": len(lut.latencies_ns),
            "placeholder_shapes": len(lut.placeholder_shapes), "warnings": list(lut.warnings), **provenance,
        },
        "pairing_invariant": "W00/W11 both use 16 cores and the same base HBM workload; C00/C10 keep identical canonical 16-core mapping, two-D2D-port placement, core-to-port routes, and base HBM workload, changing only inter serial versus streaming scheduling; G00/G10 keep identical GPU decomposition",
        "cases": case_audit,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "exp3_1_results.json", result)
    _write_csv(output_dir / "exp3_1_results.csv", rows)
    _write_json(output_dir / "audit.json", audit)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--generated-dir", type=Path, default=HERE / "generated")
    parser.add_argument("--allow-placeholder", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run(gpu_yaml=args.gpu_yaml, output_dir=args.output_dir, generated_dir=args.generated_dir, allow_placeholder=args.allow_placeholder)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
