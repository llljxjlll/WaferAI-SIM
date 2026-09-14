#!/usr/bin/env python3
"""Run the reproducible Exp3.2 2x3 intra-die timing-model sweep.

This is an analytical preflight model, not an NpuSim measurement. It freezes
the Exp3.2 35-point logical matrix and labels every result accordingly.
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
M_VALUES = (256, 512, 1024, 2048, 4096, 8192, 16384)
K_VALUES = (256, 1024, 4096, 16384, 65536)
N, D, CORES_PER_DIE, DTYPE_BYTES = 12288, 6, 16, 2

# Frozen preflight assumptions. A future NpuSim adapter can replace only this
# provider while preserving the case/result/plot contracts.
PEAK_FLOPS_PER_CORE = 8_000.0  # 8 TFLOP/s = 8,000 FLOP/ns
COMPUTE_UTILIZATION = 0.70
LOCAL_BW_BYTES_PER_NS, LOCAL_STARTUP_NS = 256.0, 4_000.0
RING_BW_BYTES_PER_NS, RING_STARTUP_NS = 256.0, 8_000.0
OVERLAP_RESIDUAL_FRACTION = 0.50
EVIDENCE = "analytical_preflight_model_not_npusim"

CSV_FIELDS = (
    "case_id", "state", "M", "N", "K", "D", "logical_shape",
    "valid_flops", "local_output_bytes", "T_comp_ns", "T_move_ns",
    "T_intra_ns", "T_ring_ns", "T_total_ns", "selected_candidate", "evidence",
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _components(m: int, k: int) -> dict[str, float | int]:
    valid_flops = 2 * m * N * k
    die_peak = CORES_PER_DIE * PEAK_FLOPS_PER_CORE * COMPUTE_UTILIZATION
    compute = valid_flops / D / die_peak
    local_output_bytes = m * N // D * DTYPE_BYTES
    move = LOCAL_STARTUP_NS + local_output_bytes / LOCAL_BW_BYTES_PER_NS
    ring_payload = 2.0 * (D - 1) / D * local_output_bytes
    ring = RING_STARTUP_NS + ring_payload / RING_BW_BYTES_PER_NS
    return {
        "valid_flops": valid_flops,
        "local_output_bytes": local_output_bytes,
        "T_comp_ns": _positive(compute, "T_comp_ns"),
        "T_move_ns": _positive(move, "T_move_ns"),
        "T_ring_ns": _positive(ring, "T_ring_ns"),
    }


def _pair(m: int, k: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    component = _components(m, k)
    compute, move, ring = (float(component[key]) for key in ("T_comp_ns", "T_move_ns", "T_ring_ns"))
    serial = compute + move
    overlapped = max(compute, move) + OVERLAP_RESIDUAL_FRACTION * min(compute, move)
    if overlapped > serial:
        raise AssertionError("intra ON regressed")
    case_id = f"d6_m{m}_n{N}_k{k}"
    common: dict[str, object] = {
        "case_id": case_id, "M": m, "N": N, "K": k, "D": D,
        "active_mesh": [2, 3], "logical_shape": [m, N, k], "runtime_shape": [m, N, k],
        "valid_flops": component["valid_flops"], "padded_flops": component["valid_flops"],
        "local_output_bytes": component["local_output_bytes"], "T_comp_ns": compute,
        "T_move_ns": move, "T_ring_ns": ring, "ring_algorithm": "fixed_1d_ring_blocking_model",
        "ring_chunk_count": D, "ring_unroll": 1, "ring_order": [0, 1, 2, 5, 4, 3],
        "evidence": EVIDENCE,
    }
    records = [
        {**common, "state": "I0", "selected_candidate": "split_k_barrier", "T_intra_ns": serial, "T_total_ns": serial + ring},
        {**common, "state": "I1", "selected_candidate": "split_k_streaming_model", "T_intra_ns": overlapped, "T_total_ns": overlapped + ring},
    ]
    ideal = serial / max(compute, move)
    speedup = serial / overlapped
    return records, {
        "case_id": case_id, "M": m, "N": N, "K": k, "D": D,
        "baseline_state": "I0", "optimized_state": "I1", "r": compute / move,
        "stage_speedup": speedup, "e2e_speedup": (serial + ring) / (overlapped + ring),
        "ideal_speedup": ideal, "attainment": speedup / ideal,
        "overlap_efficiency": (serial - overlapped) / min(compute, move),
        "no_regression": True, "evidence": EVIDENCE,
    }


def _matrix() -> Iterable[tuple[int, int]]:
    for k in K_VALUES:
        for m in M_VALUES:
            yield m, k


def run(output_dir: Path) -> dict[str, object]:
    config = {
        "experiment": "exp3_2_intra_die_ablation", "status": "analytical_preflight_model",
        "evidence": EVIDENCE, "matrix": {"M": list(M_VALUES), "N": N, "K": list(K_VALUES), "D": D},
        "hardware": {
            "cores_per_die": CORES_PER_DIE, "peak_flops_per_core": PEAK_FLOPS_PER_CORE,
            "compute_utilization": COMPUTE_UTILIZATION, "local_bw_bytes_per_ns": LOCAL_BW_BYTES_PER_NS,
            "local_startup_ns": LOCAL_STARTUP_NS, "ring_bw_bytes_per_ns": RING_BW_BYTES_PER_NS,
            "ring_startup_ns": RING_STARTUP_NS, "overlap_residual_fraction": OVERLAP_RESIDUAL_FRACTION,
        },
    }
    records: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for m, k in _matrix():
        case_records, pair = _pair(m, k)
        records.extend(case_records)
        pairs.append(pair)
    if len(records) != 70 or len(pairs) != 35:
        raise AssertionError("unexpected matrix size")
    document = {
        "schema_version": "exp3_2_results/v1alpha1", "experiment": config["experiment"],
        "status": config["status"], "evidence": EVIDENCE,
        "summary": {"logical_cases": 35, "state_rows": 70, "pairs": 35, "metric": "stage_speedup"},
        "config": config, "config_sha256": _digest(config), "records": records, "pairs": pairs,
    }
    audit = {
        "schema_version": "exp3_2_audit/v1alpha1", "status": config["status"], "evidence": EVIDENCE,
        "limitations": [
            "No current runner composes exact 2x3 Wang Ring, blocking inter schedule, arbitrary GEMM shape, and 16-core intra-die ON/OFF.",
            "These rows are deterministic timing-model preflight results, not NpuSim measurements.",
        ],
        "pair_invariants": {"same_logical_work": True, "same_fixed_ring": True, "same_ring_order": [0, 1, 2, 5, 4, 3], "only_intra_composition_changes": True},
        "config_sha256": document["config_sha256"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "exp3_2_results.json").write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "exp3_2_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)
    result = run(args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
