#!/usr/bin/env python3
"""Build a fast calibration prior entirely from checked-in simulator evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load(path: str):
    return json.loads((ROOT / path).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_cost(path: str) -> dict[str, int]:
    decision = load(path)[0]
    selected = next(
        row for row in decision["candidates"]
        if row["id"] == decision["selected_candidate_ref"]
    )
    return selected["analytic_cost"]


def timing_defaults() -> dict[str, int]:
    path = ROOT / "llm/frontend/wafer_frontend/schema/intra_die_timing_model.py"
    tree = ast.parse(path.read_text())
    create = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "create"
    )
    names = [arg.arg for arg in create.args.kwonlyargs]
    values = [ast.literal_eval(node) if node is not None else None for node in create.args.kw_defaults]
    wanted = {
        "compute_setup_cycles",
        "effective_gemm_ops_per_cycle",
        "hbm_load_setup_cycles",
        "hbm_load_bytes_per_cycle",
        "local_transport_setup_cycles",
        "local_transport_bytes_per_cycle",
        "local_reduce_setup_cycles",
        "local_reduce_bytes_per_cycle",
        "sync_issue_cycles",
        "fixed_pipeline_cycles",
    }
    return {name: value for name, value in zip(names, values, strict=True) if name in wanted}


def run_anchor(report_path: str, search_path: str, calibration_path: str) -> dict:
    report = load(report_path)
    calibration = load(calibration_path)[0]
    return {
        "measured_makespan_cycles": report["runtime"]["makespan_cycles"],
        "predicted_makespan_cycles": calibration["predicted_makespan_cycles"],
        "relative_error": calibration["relative_error"],
        "repeat_count": report["runtime"]["repeat"],
        "repeat_signature_stable": report["runtime"]["repeat_signature_stable"],
        "analytic_components_cycles": selected_cost(search_path),
        "action_counts": report["static_metrics"]["action_counts"],
        "source": report_path,
        "source_sha256": sha256(ROOT / report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "existing_calibration.json",
    )
    args = parser.parse_args()

    hardware = ROOT / "exps/exp1/exp1_1/configs/hardware.json"
    simulation = ROOT / "exps/exp1/exp1_1/configs/simulation.json"
    tree_auto = run_anchor(
        "notes/frontend/intra_die/reports/performance/model_v5_tree_direct_16core/auto/run_report.json",
        "notes/frontend/intra_die/reports/performance/model_v5_tree_direct_16core/auto/search_decisions.json",
        "notes/frontend/intra_die/reports/performance/model_v5_tree_direct_16core/auto/calibration.json",
    )
    barrier = run_anchor(
        "notes/frontend/intra_die/reports/performance/model_v5_tree_direct_16core/naive/run_report.json",
        "notes/frontend/intra_die/reports/performance/model_v5_tree_direct_16core/naive/search_decisions.json",
        "notes/frontend/intra_die/reports/performance/model_v5_tree_direct_16core/naive/calibration.json",
    )
    compute = run_anchor(
        "notes/frontend/intra_die/reports/performance/model_v3_compute/off/run_report.json",
        "notes/frontend/intra_die/reports/performance/model_v3_compute/off/search_decisions.json",
        "notes/frontend/intra_die/reports/performance/model_v3_compute/off/calibration.json",
    )
    sync = run_anchor(
        "notes/frontend/intra_die/reports/performance/model_v3_sync/off/run_report.json",
        "notes/frontend/intra_die/reports/performance/model_v3_sync/off/search_decisions.json",
        "notes/frontend/intra_die/reports/performance/model_v3_sync/off/calibration.json",
    )
    reference_hardware = load(
        "notes/frontend/intra_die/reports/performance/model_v5_tree_direct_16core/auto/run_report.json"
    )["inputs"]["hardware_sha256"]
    reference_simulation = load(
        "notes/frontend/intra_die/reports/performance/model_v5_tree_direct_16core/auto/run_report.json"
    )["inputs"]["simulation_sha256"]

    noc_text = (ROOT / "llm/test/noc_congestion/noc_congestion_summary.txt").read_text()
    no_congestion, congestion = [
        tuple(map(int, match))
        for match in re.findall(r"\|\s+(?:no_congestion|congestion)\s+\|\s+(\d+) ns\s+\|\s+(\d+) ns", noc_text)
    ]
    sram = load("notes/extensions/SRAM/exp/results/summary.json")
    result = {
        "schema_version": "exp1.existing_cycle_calibration/v1",
        "method": "checked_in_evidence_only_no_compile_no_simulator",
        "experiment_input_binding": {
            "hardware_sha256": sha256(hardware),
            "simulation_sha256": sha256(simulation),
            "reference_hardware_sha256": reference_hardware,
            "reference_simulation_sha256": reference_simulation,
            "reference_digest_match": (
                sha256(hardware) == reference_hardware
                and sha256(simulation) == reference_simulation
            ),
            "status": "directly_applicable" if (
                sha256(hardware) == reference_hardware
                and sha256(simulation) == reference_simulation
            ) else "prior_only_requires_current_config_smoke_run",
        },
        "frozen_timing_model_prior": {
            **timing_defaults(),
            "source": "llm/frontend/wafer_frontend/schema/intra_die_timing_model.py",
            "units": "cycles or bytes/cycle as named",
        },
        "end_to_end_cycle_anchors": {
            "compute_bound_identity": compute,
            "sync_bound_identity": sync,
            "split_k_16_linear_barrier": barrier,
            "split_k_16_tree_direct_dma": tree_auto,
            "tree_vs_linear_measured_speedup": (
                barrier["measured_makespan_cycles"]
                / tree_auto["measured_makespan_cycles"]
            ),
        },
        "operation_level_anchor": {
            "source": "notes/extensions/SRAM/exp/results/summary.json",
            "source_sha256": sha256(ROOT / "notes/extensions/SRAM/exp/results/summary.json"),
            "time_unit": "ns",
            "tile_bytes": 4096,
            "collective_chunk_bytes": 512,
            "operation_breakdown": sram["operation_breakdown_ns"],
            "communication_breakdown": sram["communication_breakdown_ns"],
            "warning": "legacy two-core handcrafted workload; useful for setup-scale priors, not direct current-config calibration",
        },
        "congestion_anchors": {
            "single_die_4x4_distributed_gemm": {
                "behavioral_no_congestion_ns": no_congestion[0],
                "cycle_no_congestion_ns": no_congestion[1],
                "behavioral_congestion_ns": congestion[0],
                "cycle_congestion_ns": congestion[1],
                "cycle_congestion_factor": congestion[1] / no_congestion[1],
                "isolated_cycle_penalty_ns": (congestion[1] - congestion[0]) - (no_congestion[1] - no_congestion[0]),
                "source": "llm/test/noc_congestion/noc_congestion_summary.txt",
            },
            "mixed_d2d_local_noc_v5": {
                "disjoint_flow_completion_cycle": 319,
                "shared_flow_completion_cycle": 347,
                "measured_penalty_cycles": 28,
                "static_bottleneck_penalty_cycles": 32,
                "measured_to_static_factor": 28 / 32,
                "shared_over_disjoint_flow_factor": 347 / 319,
                "blocked_output_events": 11,
                "source": "llm/test/d2d_link/mixed_noc_congestion_v5/mixed_noc_congestion_v5_report.md",
            },
        },
        "recommended_fast_use": [
            "Use frozen timing constants to rank candidates without simulator calls.",
            "Apply congestion factors only to matching topology/route-sharing classes.",
            "Run one tiny current-config compute+collective smoke case and one repeat before accepting absolute cycles.",
            "Do not fit separate GEMM/HBM/reduce constants from whole-program makespan alone because overlap makes the system underdetermined.",
        ],
        "limitations": [
            "The strongest v5 calibration report is bound to different hardware/simulation digests than exp1 configs.",
            "Most run_report files expose makespan and counters, not per-record begin/end timestamps.",
            "The operation-level SRAM report uses a handcrafted two-core workload and reports ns, not the production manifest path.",
            "No checked-in isolated current-config HBM sweep was found; HBM constants remain priors until a small byte sweep is run.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
