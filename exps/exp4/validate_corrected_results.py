#!/usr/bin/env python3
"""Audit corrected Exp-4 artifacts and write a machine-readable gate summary."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from candidate_loader import load_candidates


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> Any:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> int:
    primitive = load("primitive_results.json")
    request = load("request_results.json")
    sw_only = load("sw_opt_only.json")
    pareto = load("pareto_summary.json")
    manifest = load("run_manifest.json")
    calibration = json.loads((ROOT / "calibration/calibration_summary.json").read_text(encoding="utf-8"))
    candidates = load_candidates()

    checks: dict[str, Any] = {}
    checks["candidate_count"] = len(candidates) == 383
    checks["primitive_result_count"] = len(primitive) == 27_576
    checks["request_result_count"] = len(request) == 9_192
    checks["software_control_count"] = len(sw_only) == 48
    checks["corrected_model_selected"] = (
        manifest.get("schema_version") == "exp4.corrected_action_replay_run.v2"
        and manifest.get("analytical_model") == "action_flops_bytes_dependency_resource_lower_bound"
        and manifest.get("invalidated_model") == "duration-ledger-scaling-v1"
    )
    checks["reference_reproduction_72_of_72"] = (
        manifest.get("reference_reproduction_audit", {}).get("row_count") == 72
        and manifest.get("reference_reproduction_audit", {}).get("failure_count") == 0
    )
    checks["finite_positive_and_above_lower_bound"] = all(
        math.isfinite(float(row["estimate_cycles"]))
        and float(row["estimate_cycles"]) >= float(row["theory_lower_cycles"]) > 0
        for row in primitive
    )
    checks["hardware_semantics"] = all(
        c.f_Hz == 500_000_000
        and math.isclose(c.P_core_FLOPs, 2 * c.N_PE * c.f_Hz, rel_tol=0, abs_tol=1e-6)
        and c.DTE_channel == math.ceil(c.B_GBs / 128)
        and math.isclose(c.D2D_edge_one_dir_GBs, min(c.d_d2d * c.B_GBs, 512), rel_tol=0, abs_tol=1e-12)
        for c in candidates
    )
    exp2_result_dir = ROOT.parent / "exp2/exp2_1/results"
    exp2_digests = {
        row["result_digest"]
        for name in ("training_e2e.json", "inference_prefill_pd_breakdown.json",
                     "inference_decode_e2e.json", "inference_request_e2e.json")
        for row in json.loads((exp2_result_dir / name).read_text(encoding="utf-8"))
    }
    checks["control_rows_match_exp2_digest"] = all(
        row["source_result_digest"] in exp2_digests for row in sw_only
    )
    checks["projection_evidence_not_feasible"] = all(
        not point["evidence"].get("all_feasible", True)
        for group in pareto.get("projection_groups", []) + pareto.get("projection_average_groups", [])
        for point in group.get("points", [])
    )
    checks["prefill_sw_hw_exceeds_sw_only"] = all(
        float(row["sw_hw_opt"]) > float(row["sw_opt_only"])
        for row in pareto["speedup_figure"]["rows"]
        if row["workload_type"] == "prefill"
    )
    checks["svg_white_background"] = all(
        '<rect width="' in path.read_text(encoding="utf-8") and 'fill="white"' in path.read_text(encoding="utf-8")
        for path in (ROOT / "figures/optimization_speedup.svg", ROOT / "figures/hardware_pareto.svg")
    )

    paired = {(r["candidate_id"], r["workload_id"], r["software_state"]): r for r in primitive}
    slowdowns = []
    for row in primitive:
        if row["software_state"] != "sw_opt":
            continue
        naive = paired[(row["candidate_id"], row["workload_id"], "naive")]
        speedup = float(naive["estimate_cycles"]) / float(row["estimate_cycles"])
        if speedup < 1.0:
            slowdowns.append(speedup)

    analytical_gate = all(checks.values())
    anchors_complete = (
        calibration.get("completed_run_count") == calibration.get("planned_run_count") == 168
        and calibration.get("current_build_direct_gate_passed") is True
    )
    summary = {
        "schema_version": "exp4.corrected_validation.v1",
        "checks": checks,
        "analytical_sweep_gate_passed": analytical_gate,
        "cycle_anchor_gate_passed": anchors_complete,
        "release_gate_passed": analytical_gate and anchors_complete,
        "evidence_level": "resource_explicit_analytical_extrapolation",
        "strict_pareto_group_count": len(pareto.get("groups", [])),
        "projection_pareto_group_count": len(pareto.get("projection_groups", [])),
        "software_slowdown_pair_count": len(slowdowns),
        "minimum_software_speedup": min(slowdowns, default=1.0),
        "software_slowdown_policy": "preserved_without_positive-benefit-clamp_per_development_plan",
        "artifact_sha256": {
            name: digest(RESULTS / name)
            for name in ("primitive_results.json", "request_results.json", "sw_opt_only.json",
                         "pareto_summary.json", "sensitivity_summary.json", "run_manifest.json")
        },
        "figure_sha256": {
            name: digest(ROOT / "figures" / name)
            for name in ("optimization_speedup.svg", "hardware_pareto.svg")
        },
    }
    summary["validation_digest"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = RESULTS / "validation_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"analytical_gate={analytical_gate} cycle_anchor_gate={anchors_complete} release_gate={summary['release_gate_passed']}")
    print(f"wrote {output}")
    return 0 if analytical_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
