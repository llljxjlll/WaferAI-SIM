#!/usr/bin/env python3
"""Run the fixed-2x3 Dispatch+GEMM intra-die preflight sweep.

The generated data are explicitly an analytical timing-model preflight, not
NpuSim measurements.  It mirrors Exp3.2's I0/I1 contract without overwriting
the GEMM+ReduceScatter result set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

HERE = Path(__file__).resolve().parent
M_RANK_VALUES = (64, 256, 1024, 4096, 16384)
I_VALUES = (512, 2048, 8192, 32768, 131072)
H_PRIMARY, H_ROBUSTNESS = 7168, 4096
D, CORES_PER_DIE, DTYPE_BYTES = 6, 16, 2
AVERAGE_REMOTE_HOPS = D * D / (4 * (D - 1))  # 1.8: remote-only, shortest Ring hops
PEAK_FLOPS_PER_CORE, COMPUTE_UTILIZATION = 8_000.0, 0.70
LOCAL_BW_BYTES_PER_NS, LOCAL_STARTUP_NS = 256.0, 4_000.0
RING_BW_BYTES_PER_NS, RING_STARTUP_NS = 256.0, 8_000.0
OVERLAP_RESIDUAL_FRACTION = 0.50
EVIDENCE = "analytical_preflight_model_not_npusim"

CSV_FIELDS = (
    "case_id", "subset", "operator_stage", "state", "M_rank", "H", "I", "D",
    "logical_shape", "valid_flops", "dispatch_activation_bytes", "T_comp_ns",
    "T_move_ns", "T_intra_ns", "T_ring_ns", "T_total_ns",
    "selected_candidate", "evidence",
)


def _digest(value: object) -> str:
    source = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _components(m_rank: int, hidden: int, intermediate: int) -> dict[str, float | int]:
    valid_flops = 2 * m_rank * intermediate * hidden
    die_peak = CORES_PER_DIE * PEAK_FLOPS_PER_CORE * COMPUTE_UTILIZATION
    compute = valid_flops / die_peak
    activation_bytes = m_rank * hidden * DTYPE_BYTES
    move = LOCAL_STARTUP_NS + activation_bytes / LOCAL_BW_BYTES_PER_NS
    ring = AVERAGE_REMOTE_HOPS * (
        RING_STARTUP_NS + activation_bytes / RING_BW_BYTES_PER_NS
    )
    return {
        "valid_flops": valid_flops,
        "dispatch_activation_bytes": activation_bytes,
        "T_comp_ns": compute,
        "T_move_ns": move,
        "T_ring_ns": ring,
    }


def _pair(
    subset: str, operator_stage: str, m_rank: int, hidden: int, intermediate: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    component = _components(m_rank, hidden, intermediate)
    compute = float(component["T_comp_ns"])
    move = float(component["T_move_ns"])
    ring = float(component["T_ring_ns"])
    serial = compute + move
    overlapped = max(compute, move) + OVERLAP_RESIDUAL_FRACTION * min(compute, move)
    if not 0 < overlapped <= serial:
        raise AssertionError("invalid intra stage timing")
    case_id = f"dispatch_{operator_stage}_h{hidden}_m{m_rank}_i{intermediate}"
    common: dict[str, object] = {
        "case_id": case_id,
        "subset": subset,
        "operator_stage": operator_stage,
        "M_rank": m_rank,
        "H": hidden,
        "I": intermediate,
        "D": D,
        "active_mesh": [2, 3],
        "logical_shape": [m_rank, intermediate, hidden],
        "runtime_shape": [m_rank, intermediate, hidden],
        "valid_flops": component["valid_flops"],
        "padded_flops": component["valid_flops"],
        "dispatch_activation_bytes": component["dispatch_activation_bytes"],
        "T_comp_ns": compute,
        "T_move_ns": move,
        "T_ring_ns": ring,
        "ring_algorithm": "fixed_1d_personalized_a2a_blocking_model",
        "ring_chunk_count": D,
        "ring_unroll": 1,
        "ring_order": [0, 1, 2, 5, 4, 3],
        "average_remote_hops": AVERAGE_REMOTE_HOPS,
        "intra_dependency": (
            "dispatch_staging_then_grouped_gemm"
            if operator_stage == "gate_up"
            else "grouped_gemm_then_combine_staging"
        ),
        "evidence": EVIDENCE,
    }
    records = [
        {
            **common,
            "state": "I0",
            "selected_candidate": "split_k_barrier",
            "T_intra_ns": serial,
            "T_total_ns": serial + ring,
        },
        {
            **common,
            "state": "I1",
            "selected_candidate": "split_k_streaming_model",
            "T_intra_ns": overlapped,
            "T_total_ns": overlapped + ring,
        },
    ]
    ideal = serial / max(compute, move)
    return records, {
        "case_id": case_id,
        "subset": subset,
        "operator_stage": operator_stage,
        "M_rank": m_rank,
        "H": hidden,
        "I": intermediate,
        "D": D,
        "baseline_state": "I0",
        "optimized_state": "I1",
        "r_local": compute / move,
        "r_effective": compute / (AVERAGE_REMOTE_HOPS * move),
        "stage_speedup": serial / overlapped,
        "e2e_speedup": (serial + ring) / (overlapped + ring),
        "ideal_speedup": ideal,
        "attainment": (serial / overlapped) / ideal,
        "overlap_efficiency": (serial - overlapped) / min(compute, move),
        "no_regression": True,
        "evidence": EVIDENCE,
    }


def _cases() -> Iterable[tuple[str, str, int, int, int]]:
    for intermediate in I_VALUES:
        for m_rank in M_RANK_VALUES:
            yield "main", "gate_up", m_rank, H_PRIMARY, intermediate
    for m_rank in (112, 28672):  # preserve M_rank×H when H changes 7168 -> 4096
        for intermediate in (512, 131072):
            yield "h_robustness", "gate_up", m_rank, H_ROBUSTNESS, intermediate
    for m_rank, intermediate in (
        (64, 512),
        (64, 131072),
        (1024, 8192),
        (16384, 512),
        (16384, 131072),
    ):
        yield "down_direction", "down_combine", m_rank, H_PRIMARY, intermediate


def run(output_dir: Path) -> dict[str, object]:
    config = {
        "experiment": "exp3_2_dispatch_gemm_intra_die_ablation",
        "status": "analytical_preflight_model",
        "evidence": EVIDENCE,
        "fixed_topology": {
            "D": D,
            "active_mesh": [2, 3],
            "ring_order": [0, 1, 2, 5, 4, 3],
            "average_remote_hops": AVERAGE_REMOTE_HOPS,
        },
        "main_matrix": {"H": H_PRIMARY, "M_rank": list(M_RANK_VALUES), "I": list(I_VALUES)},
        "h_robustness": {"H": H_ROBUSTNESS, "M_rank": [112, 28672], "I": [512, 131072], "preserve_m_rank_times_h": True},
        "down_direction": {"H": H_PRIMARY, "points": [[64, 512], [64, 131072], [1024, 8192], [16384, 512], [16384, 131072]]},
        "timing_model": {
            "cores_per_die": CORES_PER_DIE,
            "peak_flops_per_core": PEAK_FLOPS_PER_CORE,
            "compute_utilization": COMPUTE_UTILIZATION,
            "local_bw_bytes_per_ns": LOCAL_BW_BYTES_PER_NS,
            "local_startup_ns": LOCAL_STARTUP_NS,
            "ring_bw_bytes_per_ns": RING_BW_BYTES_PER_NS,
            "ring_startup_ns": RING_STARTUP_NS,
            "overlap_residual_fraction": OVERLAP_RESIDUAL_FRACTION,
        },
    }
    records: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for case in _cases():
        state_rows, pair = _pair(*case)
        records.extend(state_rows)
        pairs.append(pair)
    main_pairs = [pair for pair in pairs if pair["subset"] == "main"]
    if len(pairs) != 34 or len(records) != 68 or len(main_pairs) != 25:
        raise AssertionError("unexpected Dispatch+GEMM matrix size")
    document = {
        "schema_version": "exp3_2_dispatch_results/v1alpha1",
        "experiment": config["experiment"],
        "status": config["status"],
        "evidence": EVIDENCE,
        "summary": {
            "logical_cases": 34,
            "main_pairs": 25,
            "h_robustness_pairs": 4,
            "down_direction_pairs": 5,
            "state_rows": 68,
            "metric": "stage_speedup",
        },
        "config": config,
        "config_sha256": _digest(config),
        "records": records,
        "pairs": pairs,
        "main_pairs": main_pairs,
    }
    audit = {
        "schema_version": "exp3_2_dispatch_audit/v1alpha1",
        "status": config["status"],
        "evidence": EVIDENCE,
        "limitations": [
            "No current runner composes exact 2x3 personalized A2A Ring, blocking inter schedule, arbitrary grouped-GEMM shape, and 16-core intra-die ON/OFF.",
            "Rows are deterministic timing-model preflight results, not NpuSim measurements.",
            "Dispatch+GEMM values must not be compared numerically with GEMM+ReduceScatter because the fixed Ring has personalized-A2A hop amplification.",
        ],
        "pair_invariants": {
            "same_logical_work": True,
            "same_fixed_personalized_ring": True,
            "same_ring_order": [0, 1, 2, 5, 4, 3],
            "same_average_remote_hops": AVERAGE_REMOTE_HOPS,
            "only_intra_composition_changes": True,
        },
        "config_sha256": document["config_sha256"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "exp3_2_dispatch_gemm_results.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "exp3_2_dispatch_gemm_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "dispatch_gemm")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output_dir)["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
