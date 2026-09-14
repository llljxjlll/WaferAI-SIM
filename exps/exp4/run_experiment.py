#!/usr/bin/env python3
"""Run the complete 383-candidate exp4 analytical hardware sweep."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import action_replay
from analytical_replay import CLOCK_HZ, canonical_digest
from candidate_loader import load_candidates
from exp2_workload_adapter import (DEFAULT_EXP2_ROOT, load_primitive_workloads,
                                   load_sw_opt_only_rows, source_inventory)
from placement_mapper import TopologyCapacityError, map_logical_ranks


ROOT = Path(__file__).resolve().parent


def _load_source_rows(_exp2_root: Path) -> dict[str, dict[str, Any]]:
    """Load the frozen Exp-2 rows, never the invalid duration ledgers."""
    result_dir = ROOT.parent / "exp2/exp2_1/results"
    rows: dict[str, dict[str, Any]] = {}
    for name in ("training_e2e.json", "inference_prefill_pd_breakdown.json",
                 "inference_decode_e2e.json"):
        for row in json.loads((result_dir / name).read_text(encoding="utf-8")):
            rows[row["case_id"]] = row
    if len(rows) != 36:
        raise ValueError("expected 36 frozen Exp-2 primitive rows")
    return rows


def _estimate(source: dict[str, Any], kind: str, state: str, candidate: Any) -> SimpleNamespace:
    replay = action_replay.estimate(source, kind, state, candidate)
    return SimpleNamespace(
        estimate_cycles=replay.estimate_cycles,
        theory_lower_cycles=replay.theory_lower_cycles,
        attainment=replay.attainment,
        critical_resource=replay.critical_resource,
        class_service_cycles=replay.class_service_cycles,
        resource_service_cycles=replay.resource_service_cycles,
        calibration_factor=1.0 / replay.attainment,
        result_digest=replay.result_digest,
    )


def _write_json_csv(rows: list[dict[str, Any]], stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = list(rows[0]) if rows else []
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _throughput(kind: str, seq_len: int, batch: int, cycles: float) -> float:
    tokens = batch if kind == "decode" else batch * seq_len
    return tokens * CLOCK_HZ / cycles


def _capacity(required: int, candidate: Any) -> tuple[str, int]:
    available = int(36 * candidate.HBM_stack_count * candidate.HBM_stack_capacity_GB * 1e9)
    return ("capacity_feasible" if required <= available else "capacity_infeasible_projection",
            available - required)


def _blockers(candidate: Any) -> dict[str, Any]:
    return {
        "dte": f"{candidate.DTE_channel} independent 128 GB/s, 2048-bit channels",
        "d2d": f"directed edge shared cap min({candidate.d_d2d}*{candidate.B_GBs},512)={candidate.D2D_edge_one_dir_GBs:g} GB/s",
        "stripe": f"arbitrary deterministic stripe across {candidate.d_d2d} usable physical ports",
        "hbm3": f"{candidate.HBM_stack_count}x16GB physical stacks/module, 737.28 GB/s sustained each; noncontiguous edge ports",
    }


def _row(candidate: Any, workload: Any, source: dict[str, Any], state: str,
         mapping: Any, calibration: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    replay = _estimate(source, workload.kind, state, candidate)
    required = int(source.get("total_resident_bytes", 0))
    capacity_status, headroom = _capacity(required, candidate)
    topology_status = mapping.topology_status
    valid = candidate.status == "valid" and capacity_status == "capacity_feasible" and "infeasible" not in topology_status
    status = "analytical_feasible" if valid else "analytical_infeasible_projection"
    uncertainty = 0.35 + (0.10 if not valid else 0.0)
    classes = replay.class_service_cycles
    row = {
        "candidate_id": candidate.candidate_id,
        "candidate_digest": candidate.candidate_digest,
        "model_id": workload.model_id,
        "model": workload.model,
        "workload_id": workload.workload_id,
        "workload_type": workload.kind,
        "seq_len": workload.seq_len,
        "batch_size": workload.batch_size,
        "kv_len": workload.kv_len,
        "software_state": state,
        "status": status,
        "evidence_level": calibration["evidence_level"],
        "estimate_cycles": replay.estimate_cycles,
        "throughput": _throughput(workload.kind, workload.seq_len, workload.batch_size, replay.estimate_cycles),
        "latency": replay.estimate_cycles / CLOCK_HZ,
        "capacity_status": capacity_status,
        "capacity_headroom_bytes": headroom,
        "topology_status": topology_status,
        "mapping_digest": mapping.mapping_digest,
        "compute_service_cycles": classes["compute"],
        "sram_read_cycles": classes["sram_read"],
        "sram_write_cycles": classes["sram_write"],
        "dte_service_cycles": classes["dte"],
        "noc_service_cycles": classes["noc"],
        "d2d_service_cycles": classes["d2d"],
        "hbm_service_cycles": classes["hbm"],
        "control_service_cycles": classes["control"],
        "critical_resource": replay.critical_resource,
        "theory_lower_cycles": replay.theory_lower_cycles,
        "attainment": replay.attainment,
        "calibration_factor": replay.calibration_factor,
        "four_blocker_assumptions": _blockers(candidate),
        "calibration_digest": calibration["calibration_digest"],
        "uncertainty_low": replay.estimate_cycles * (1.0 - uncertainty),
        "uncertainty_high": replay.estimate_cycles * (1.0 + uncertainty),
        "source_result_digest": workload.source_result_digest,
        "source_file_digest": workload.source_file_digest,
        "source_calibration_status": workload.calibration_status,
        "limitation_tags": list(workload.limitation_tags) + ["current_build_cycle_anchor_pending"],
        "f_Hz": candidate.f_Hz,
        "N_PE": candidate.N_PE,
        "cores_per_die": candidate.N * candidate.N,
        "core_TFLOPs": candidate.P_core_TFLOPs,
        "B_GBs": candidate.B_GBs,
        "B_s_GBs_per_core_direction": candidate.B_s_GBs,
        "K_MiB_per_core": candidate.K_MiB,
        "DTE_channel": candidate.DTE_channel,
        "D2D_edge_one_dir_GBs": candidate.D2D_edge_one_dir_GBs,
        "HBM_stacks_per_module": candidate.HBM_stack_count,
        "modules_per_wafer": candidate.modules_per_wafer,
        "replica_count": mapping.replica_count,
    }
    row["result_digest"] = canonical_digest(row)
    ledger = [{"result_digest": row["result_digest"], "candidate_id": candidate.candidate_id,
               "workload_id": workload.workload_id, "software_state": state,
               "resource_class": name, "service_cycles": value,
               "admitted_equals_completed": True}
              for name, value in sorted(classes.items())]
    return row, ledger


def _requests(primitive: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(r["candidate_id"], r["model_id"], r["software_state"], r["workload_type"],
              int(r["seq_len"]), int(r["batch_size"])): r for r in primitive}
    output: list[dict[str, Any]] = []
    candidates = sorted({r["candidate_id"] for r in primitive})
    models = sorted({r["model_id"] for r in primitive})
    for cid in candidates:
        for model in models:
            for state in ("naive", "sw_opt"):
                prefill = index[(cid, model, state, "prefill", 2304, 1)]
                for batch in (64, 512):
                    decode = index[(cid, model, state, "decode", 1, batch)]
                    cycles = float(prefill["estimate_cycles"]) + 512 * float(decode["estimate_cycles"])
                    feasible = "infeasible" not in (prefill["status"] + decode["status"])
                    row = {
                        "candidate_id": cid, "candidate_digest": prefill["candidate_digest"],
                        "model_id": model, "workload_id": f"request__{model}__b{batch}__g512",
                        "workload_type": "request", "software_state": state,
                        "seq_len": 2304, "batch_size": batch, "output_tokens": 512,
                        "status": "analytical_feasible" if feasible else "analytical_infeasible_projection",
                        "evidence_level": prefill["evidence_level"],
                        "estimate_cycles": cycles, "latency": cycles / CLOCK_HZ,
                        "throughput": batch * 512 * CLOCK_HZ / cycles,
                        "capacity_status": ("capacity_feasible" if feasible else "capacity_infeasible_projection"),
                        "topology_status": prefill["topology_status"],
                        "prefill_result_digest": prefill["result_digest"],
                        "decode_result_digest": decode["result_digest"],
                        "request_formula": "prefill_cycles + 512 * steady_decode_cycles",
                        "uncertainty_low": prefill["uncertainty_low"] + 512 * decode["uncertainty_low"],
                        "uncertainty_high": prefill["uncertainty_high"] + 512 * decode["uncertainty_high"],
                    }
                    row["result_digest"] = canonical_digest(row)
                    output.append(row)
    return output


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=ROOT / "results")
    ap.add_argument("--limit-candidates", type=int)
    args = ap.parse_args(argv)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    calibration_path = ROOT / "calibration/calibration_summary.json"
    if not calibration_path.is_file():
        raise SystemExit("run run_calibration.py first")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    reference_audit = action_replay.audit_reference_reproduction()
    if reference_audit["failure_count"]:
        raise SystemExit(f"reference reproduction gate failed: {reference_audit['failures']}")
    candidates = load_candidates()
    if args.limit_candidates:
        candidates = candidates[:args.limit_candidates]
    workloads = load_primitive_workloads()
    raw = _load_source_rows(Path(DEFAULT_EXP2_ROOT))
    mapping_cache: dict[tuple[int, int], Any] = {}
    primitive: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    for number, candidate in enumerate(candidates, 1):
        dims = (candidate.wafer_nx, candidate.wafer_ny)
        if dims not in mapping_cache:
            try:
                mapping_cache[dims] = map_logical_ranks(*dims)
            except TopologyCapacityError:
                class InfeasibleMapping:
                    topology_status = "topology_capacity_infeasible"
                    replica_count = 0
                    mapping_digest = canonical_digest(
                        {"wafer_nx": dims[0], "wafer_ny": dims[1], "status": topology_status}
                    )
                mapping_cache[dims] = InfeasibleMapping()
        mapping = mapping_cache[dims]
        for workload in workloads:
            for state in ("naive", "sw_opt"):
                row, ledger = _row(candidate, workload, raw[workload.workload_id], state, mapping, calibration)
                primitive.append(row); ledgers.extend(ledger)
        if number % 50 == 0:
            print(f"processed {number}/{len(candidates)} candidates")
    requests = _requests(primitive)
    expected_primitive = len(candidates) * 36 * 2
    expected_request = len(candidates) * 6 * 2 * 2
    assert len(primitive) == expected_primitive and len(requests) == expected_request
    assert all(r["estimate_cycles"] >= r["theory_lower_cycles"] > 0 for r in primitive)
    _write_json_csv(primitive, output / "primitive_results")
    _write_json_csv(requests, output / "request_results")
    _write_json_csv(ledgers, output / "resource_ledger")
    sw_only = [row.manifest_dict() for row in load_sw_opt_only_rows()]
    _write_json_csv(sw_only, output / "sw_opt_only")
    inventory = source_inventory()
    profiles = json.loads((ROOT / "inputs/action_profiles.json").read_text(encoding="utf-8"))
    manifest = {"schema_version": "exp4.corrected_action_replay_run.v2",
                "analytical_model": "action_flops_bytes_dependency_resource_lower_bound",
                "invalidated_model": "duration-ledger-scaling-v1",
                "action_profile_digest": profiles["document_digest"],
                "reference_reproduction_audit": reference_audit,
                "candidate_count": len(candidates),
                "primitive_workload_count": 36, "software_state_count": 2,
                "primary_result_count": len(primitive), "request_derived_count": len(requests),
                "planned_cycle_anchor_run_count": calibration["planned_run_count"],
                "completed_cycle_anchor_run_count": calibration["completed_run_count"],
                "output_file_sha256": {
                    name: _file_digest(output / name) for name in
                    ("primitive_results.json", "primitive_results.csv", "request_results.json",
                     "request_results.csv", "resource_ledger.json", "resource_ledger.csv",
                     "sw_opt_only.json", "sw_opt_only.csv")},
                "source_inventory": inventory, "calibration_digest": calibration["calibration_digest"]}
    manifest["run_digest"] = canonical_digest(manifest)
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(primitive)} primitive and {len(requests)} request rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
