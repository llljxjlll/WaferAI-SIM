#!/usr/bin/env python3
"""Paired inter-die replay for Exp3.1.

The off/on members of a pair always share the same decomposition, local GEMM
duration and communication work.  Only the dependency/composition rule is
changed.  This makes the reported ratio an inter-die scheduling ablation,
rather than a comparison between different GEMM kernels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

from case_matrix import LogicalCase
from gpu_lut import GpuLut


@dataclass(frozen=True, slots=True)
class ReplayPair:
    algorithm: str
    lookup_key: tuple[int, int, int]
    lookup_latency_ns: float
    execution_count: int
    compute_time_ns: float
    communication_off_ns: float
    communication_on_ns: float
    off_time_ns: float
    on_time_ns: float
    ideal_speedup: float
    speedup: float
    attainment: float
    overlap_factor: float
    evidence: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["lookup_key"] = list(self.lookup_key)
        return value


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def paired_replay(
    *,
    algorithm: str,
    lookup_key: tuple[int, int, int],
    lookup_latency_ns: float,
    execution_count: int,
    communication_off_ns: float,
    communication_on_ns: float,
    overlap_factor: float,
    evidence: str,
) -> ReplayPair:
    """Build one same-decomposition serial/overlapped pair.

    ``overlap_factor`` multiplies the shorter resource on top of the critical
    resource.  It includes the boundary wave and contention tail retained from
    the corresponding Exp1 schedule.  A zero communication fixture is handled
    explicitly so its speedup is exactly one.
    """

    if not algorithm:
        raise ValueError("algorithm must be non-empty")
    if type(execution_count) is not int or execution_count <= 0:
        raise ValueError("execution_count must be a positive integer")
    latency = _positive(lookup_latency_ns, "lookup_latency_ns")
    compute = latency * execution_count
    comm_off = _positive(communication_off_ns, "communication_off_ns")
    comm_on = _positive(communication_on_ns, "communication_on_ns")
    factor = _positive(overlap_factor, "overlap_factor")
    if factor > 1.0:
        raise ValueError("overlap_factor must not exceed one")
    off = compute + comm_off
    if comm_off == 0.0 and comm_on == 0.0:
        on = compute
    else:
        on = max(compute, comm_on) + factor * min(compute, comm_on)
    ideal = off / max(compute, comm_off, 1.0)
    speedup = off / max(on, 1.0)
    return ReplayPair(
        algorithm=algorithm,
        lookup_key=lookup_key,
        lookup_latency_ns=latency,
        execution_count=execution_count,
        compute_time_ns=compute,
        communication_off_ns=comm_off,
        communication_on_ns=comm_on,
        off_time_ns=off,
        on_time_ns=on,
        ideal_speedup=ideal,
        speedup=speedup,
        attainment=speedup / ideal,
        overlap_factor=factor,
        evidence=evidence,
    )


def _dense_parameters(native: Mapping[str, object]) -> tuple[float, float, float]:
    stages = native["stages"]
    record = native["legacy_record"]
    if not isinstance(stages, Mapping) or not isinstance(record, Mapping):
        raise ValueError("dense native record lacks stage details")
    communication = float(stages["communication"])
    efficiency = (
        float(record["segmented_comm_efficiency"])
        * float(record["topology_efficiency"])
    )
    fused = communication / efficiency
    factor = 1.0 / float(record["Qfull"]) + float(record["t10_contention_tail"])
    return communication, fused, factor


def replay_dense_candidates(
    case: LogicalCase, lut: GpuLut, native: Mapping[str, object]
) -> dict[str, object]:
    """Replay Ring and RC, then select the lowest inter-on makespan."""

    if case.ring_key is None or case.rc_key is None:
        raise ValueError("dense case lacks Ring/RC lookup keys")
    comm_off, comm_on, factor = _dense_parameters(native)
    evidence = (
        "analytical_gpu_placeholder_resource_replay"
        if lut.uses_placeholders
        else "measured_gpu_lut_resource_replay"
    )
    candidates = [
        paired_replay(
            algorithm="1d_ring_c_eq_d",
            lookup_key=case.ring_key,
            lookup_latency_ns=lut.lookup(case.ring_key),
            execution_count=case.D,
            communication_off_ns=comm_off,
            communication_on_ns=comm_on,
            overlap_factor=factor,
            evidence=evidence,
        ),
        paired_replay(
            algorithm="2d_row_column",
            lookup_key=case.rc_key,
            lookup_latency_ns=lut.lookup(case.rc_key),
            execution_count=1,
            communication_off_ns=comm_off,
            communication_on_ns=comm_on,
            overlap_factor=factor,
            evidence=evidence,
        ),
    ]
    selected = min(candidates, key=lambda item: (item.on_time_ns, item.algorithm))
    coarse_latency = lut.lookup(case.coarse_key)
    return {
        "selected": selected,
        "candidates": tuple(candidates),
        "coarse_diagnostic": {
            "lookup_key": list(case.coarse_key),
            "lookup_latency_ns": coarse_latency,
            "execution_count": 1,
            "compute_time_ns": coarse_latency,
            "not_a_paired_baseline": True,
        },
    }


def replay_moe_candidate(
    case: LogicalCase, lut: GpuLut, native: Mapping[str, object]
) -> dict[str, object]:
    """Replay the balanced source-expert chunk decomposition."""

    if case.chunk_key is None:
        raise ValueError("MoE case lacks a source-expert chunk lookup key")
    stages = native["stages"]
    if not isinstance(stages, Mapping):
        raise ValueError("MoE native record lacks stage details")
    off_stage = stages["W00"]
    on_stage = stages["W10"]
    if not isinstance(off_stage, Mapping) or not isinstance(on_stage, Mapping):
        raise ValueError("MoE native W00/W10 stage details are invalid")
    execution_count = case.gemm_execution_count * int(case.experts or 0)
    evidence = (
        "analytical_gpu_placeholder_resource_replay"
        if lut.uses_placeholders
        else "measured_gpu_lut_resource_replay"
    )
    on_compute = float(on_stage["compute_cycles"])
    on_communication = float(on_stage["communication_cycles"])
    on_total = float(
        on_stage.get("scaled_total_cycles", on_stage["anchor_total_cycles"])
    )
    overlap_factor = (
        max(0.0, on_total - max(on_compute, on_communication))
        / max(min(on_compute, on_communication), 1.0)
    )
    pair = paired_replay(
        algorithm="balanced_source_expert_chunk",
        lookup_key=case.chunk_key,
        lookup_latency_ns=lut.lookup(case.chunk_key),
        execution_count=execution_count,
        communication_off_ns=float(off_stage["communication_cycles"]),
        communication_on_ns=on_communication,
        overlap_factor=min(1.0, overlap_factor),
        evidence=evidence,
    )
    coarse_latency = lut.lookup(case.coarse_key)
    return {
        "selected": pair,
        "candidates": (pair,),
        "coarse_diagnostic": {
            "lookup_key": list(case.coarse_key),
            "lookup_latency_ns": coarse_latency,
            "execution_count": case.gemm_execution_count,
            "compute_time_ns": coarse_latency * case.gemm_execution_count,
            "not_a_paired_baseline": True,
        },
    }


__all__ = [
    "ReplayPair",
    "paired_replay",
    "replay_dense_candidates",
    "replay_moe_candidate",
]
