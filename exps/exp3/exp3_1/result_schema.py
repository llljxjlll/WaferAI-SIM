#!/usr/bin/env python3
"""Result construction and invariants for the six Exp3.1 states."""

from __future__ import annotations

from typing import Mapping, Sequence

from case_matrix import LogicalCase
from resource_replay import ReplayPair


STATE_META = {
    "W00": (False, False, "wafer_model", "native_full", "W00"),
    "W11": (True, True, "wafer_model", "native_full", "W00"),
    "C00": (False, False, "wafer_model", "native_inter_only", "C00"),
    "C10": (True, False, "wafer_model", "native_inter_only", "C00"),
    "G00": (False, None, "gpu_lut", "gpu_inter", "G00"),
    "G10": (True, None, "gpu_lut", "gpu_inter", "G00"),
}


def _native_state(
    case: LogicalCase,
    native: Mapping[str, object],
    state: str,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    inter, intra, source, comparison, baseline = STATE_META[state]
    stages = native.get("stages", {})
    is_controlled = state.startswith("C")
    stage_key = "controlled_state_stages" if is_controlled else "state_stages"
    cycle_key = "controlled_state_cycles" if is_controlled else "state_cycles"
    state_stages = native.get(stage_key)
    stage = (
        state_stages.get(state, {}) if isinstance(state_stages, Mapping)
        else (stages.get(state, {}) if isinstance(stages, Mapping) else {})
    )
    cycles = native.get(cycle_key)
    if not isinstance(cycles, Mapping):
        raise ValueError(f"native result lacks {cycle_key}")
    total = float(cycles[state])
    base_total = float(cycles[baseline])
    compute = stage.get("compute_cycles") if isinstance(stage, Mapping) else None
    communication = (
        stage.get("communication_cycles") if isinstance(stage, Mapping) else None
    )
    if compute is None or communication is None:
        # Exp1.1 exposes one shared dense-stage mapping rather than a mapping per
        # W state. Recover the exact resource components from that mapping.
        if not isinstance(stages, Mapping):
            raise ValueError("native stage decomposition must be a mapping")
        record = native.get("legacy_record")
        if not isinstance(record, Mapping):
            raise ValueError("dense native result lacks its legacy record")
        unfused_comm = float(stages["communication"])
        fused_comm = unfused_comm / (
            float(record["segmented_comm_efficiency"])
            * float(record["topology_efficiency"])
        )
        naive_compute = (
            float(stages["compute_naive"])
            + float(stages["hbm"])
            + float(stages["local_transport"])
        )
        optimized_compute = (
            float(native["state_cycles"]["W01"]) - unfused_comm
        )
        compute = naive_compute if state == "W00" else optimized_compute
        communication = unfused_comm if state == "W00" else fused_comm
    return {
        **_common(case, provenance),
        "state": state,
        "comparison": comparison,
        "inter_enabled": inter,
        "intra_enabled": intra,
        "compute_source": source,
        "algorithm": "fixed_16core_inter_control" if is_controlled else "fixed_16core_full_control",
        "lookup_keys": [],
        "inter_port_time_ns": (
            stage.get("inter_port_cycles") if isinstance(stage, Mapping) else None
        ),
        "inter_port": stage.get("inter_port") if isinstance(stage, Mapping) else None,
        "lookup_latency_ns": None,
        "local_compute_time_ns": (
            stage.get("local_compute_cycles") if isinstance(stage, Mapping) else None
        ),
        "hbm_time_ns": stage.get("hbm_cycles") if isinstance(stage, Mapping) else None,
        "local_noc_time_ns": (
            stage.get("local_transport_cycles") if isinstance(stage, Mapping) else None
        ),
        "compute_time_ns": compute,
        "intra_schedule": stage.get("intra_schedule") if isinstance(stage, Mapping) else None,
        "communication_time_ns": communication,
        "critical_path_ns": total,
        "total_time_ns": total,
        "baseline_state": baseline,
        "speedup": base_total / total,
        "ideal_speedup": None,
        "attainment": None,
        "evidence": native.get("controlled_evidence", native["evidence"]) if is_controlled else native["evidence"],
    }


def _gpu_state(
    case: LogicalCase,
    pair: ReplayPair,
    state: str,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    inter, intra, source, comparison, baseline = STATE_META[state]
    total = pair.on_time_ns if inter else pair.off_time_ns
    communication = pair.communication_on_ns if inter else pair.communication_off_ns
    return {
        **_common(case, provenance),
        "state": state,
        "comparison": comparison,
        "inter_enabled": inter,
        "intra_enabled": intra,
        "compute_source": source,
        "algorithm": pair.algorithm,
        "lookup_keys": [list(pair.lookup_key)],
        "lookup_latency_ns": pair.lookup_latency_ns,
        "lookup_execution_count": pair.execution_count,
        "compute_time_ns": pair.compute_time_ns,
        "communication_time_ns": communication,
        "critical_path_ns": total,
        "total_time_ns": total,
        "baseline_state": baseline,
        "speedup": 1.0 if state == "G00" else pair.speedup,
        "ideal_speedup": pair.ideal_speedup,
        "attainment": (1.0 / pair.ideal_speedup if state == "G00" else pair.attainment),
        "evidence": pair.evidence,
    }


def _common(case: LogicalCase, provenance: Mapping[str, object]) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "operator_family": case.operator_family,
        "stage": case.stage,
        "model_or_moe_config": case.model,
        "D": case.D,
        "Px": case.Px,
        "Py": case.Py,
        "S": case.S,
        "logical_shape": list(case.logical_shape),
        "runtime_shape": list(case.runtime_shape),
        "valid_flops": case.valid_flops,
        "padded_flops": case.padded_flops,
        "gemm_execution_count": case.gemm_execution_count,
        "production_expert_placement_closed": not (
            case.operator_family == "dispatch_gemm"
            and case.model == "Mixtral-8x7B"
            and case.D > int(case.experts or 0)
        ),
        **dict(provenance),
    }


def build_case_states(
    case: LogicalCase,
    native: Mapping[str, object],
    gpu_pair: ReplayPair,
    provenance: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = [_native_state(case, native, state, provenance)
            for state in ("W00", "W11", "C00", "C10")]
    rows.extend(
        _gpu_state(case, gpu_pair, state, provenance)
        for state in ("G00", "G10")
    )
    by_state = {str(row["state"]): row for row in rows}
    for off, on in (("W00", "W11"), ("C00", "C10")):
        baseline = float(by_state[off]["total_time_ns"])
        by_state[off]["ideal_speedup"] = 1.0
        by_state[off]["attainment"] = 1.0
        on_row = by_state[on]
        compute = float(on_row["compute_time_ns"])
        communication = float(on_row["communication_time_ns"])
        ideal_time = max(compute, communication, 1.0)
        ideal = baseline / ideal_time
        on_row["ideal_speedup"] = ideal
        on_row["attainment"] = float(on_row["speedup"]) / ideal
    validate_state_rows(rows)
    return rows


def validate_state_rows(rows: Sequence[Mapping[str, object]]) -> None:
    if len(rows) % 6:
        raise ValueError("state row count must be divisible by six")
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    for case_id, members in grouped.items():
        states = {str(row["state"]) for row in members}
        if states != set(STATE_META):
            raise ValueError(f"{case_id} has invalid state set: {sorted(states)}")
        by_state = {str(row["state"]): row for row in members}
        for off, on in (("W00", "W11"), ("C00", "C10"), ("G00", "G10")):
            if by_state[on]["baseline_state"] != off:
                raise ValueError(f"{case_id} {on} has wrong baseline")
        # The GPU pair must differ only in inter scheduling, never decomposition.
        for field in ("algorithm", "lookup_keys", "lookup_latency_ns", "compute_time_ns"):
            if by_state["G00"][field] != by_state["G10"][field]:
                raise ValueError(f"{case_id} GPU pair changes {field}")


__all__ = ["STATE_META", "build_case_states", "validate_state_rows"]
