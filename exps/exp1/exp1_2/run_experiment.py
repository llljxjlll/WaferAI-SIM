#!/usr/bin/env python3
"""Run the exp1-2 per-expert analytical MoE resource-replay experiment.

Logical assignments, per-expert padding, physical traffic and runtime work are
kept separate.  H128 and H2000 use the same workload, placement, memory and
network model; only tensor throughput changes.  Results remain analytical
extrapolations until the cycle-accurate calibration gates in the development
plan are met.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

from physical_model import (
    PhysicalModel, build_physical_model, default_hbm_stacks,
    die_id_to_coordinate,
)


CLOCK_HZ = 1.0e9
DTYPE_BYTES = 2
FP32_BYTES = 4
DIE_CORES = 16

# Backwards-compatible aliases for the former single H2000 implementation.
DIE_FLOPS = 2.0e15
CORE_FLOPS = DIE_FLOPS / DIE_CORES
DIE_VECTOR_FLOPS = 60.0e12
CORE_VECTOR_FLOPS = DIE_VECTOR_FLOPS / DIE_CORES

D2D_LINK_BPS = 1.0e12
NOC_LINK_BPS = 256.0e9
D2D_ATTACHED_NOC_LINKS = 4
D2D_INJECTION_BPS = min(D2D_LINK_BPS, D2D_ATTACHED_NOC_LINKS * NOC_LINK_BPS)
HBM_STACK_BPS = 256.0e9
HBM_BPS = HBM_STACK_BPS  # legacy import name
HBM_STACK_CAPACITY_BYTES = 16 * 1024**3
LOCAL_NOC_BPS = 256.0e9
SRAM_CAPACITY_BYTES = 3 * 1024 * 1024
RUNTIME_RESERVE_BYTES = 512 * 1024
TILE_M, TILE_N, TILE_K = 128, 512, 256
EP_ROWS = 2
EP_COLUMNS = 2
EP_SIZE = EP_ROWS * EP_COLUMNS
SEQ_LENS = (2304, 36864)
OPERATORS = ("DISPATCH_GEMM", "GEMM_COMBINE")
HBM_TRAFFIC_MODEL = "per_expert_four_physical_stack_replay_v2"
NETWORK_SCENARIO = "isolated_group"

# These are explicitly provisional priors, not measured calibration values.
TENSOR_EFFICIENCY_PRIOR = 0.80
DTE_LAUNCH_CYCLES = 8.0
DTE_SYNC_CYCLES = 2.0
DTE_HOP_CYCLES = 1.0
SESSION_OPEN_CYCLES = 1.0
SESSION_RETIRE_CYCLES = 1.0


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    hidden_size: int
    expert_intermediate_size: int
    top_k: int
    expert_count: int


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    name: str
    die_tensor_flops: float
    die_vector_flops: float
    tensor_rate_status: str = "target_profile"
    vector_rate_status: str = "provisional_vector_rate"


@dataclass(frozen=True, slots=True)
class Placement:
    name: str
    label: str
    physical_mesh: str
    member_coordinates: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    operator: str
    placement: Placement
    model: ModelProfile
    seq_len: int

    @property
    def case_id(self) -> str:
        return (
            f"{self.operator.lower()}__{self.placement.name}__"
            f"{self.model.name.lower()}__s{self.seq_len}"
        ).replace("-", "_")


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    event_id: str
    kind: str
    deps: tuple[str, ...]
    resource: str
    duration: float


@dataclass(frozen=True, slots=True)
class InterCandidate:
    name: str
    stages: int
    unroll_degree: int
    double_buffer: bool
    physical_slots: int


MODELS = (
    ModelProfile("Mixtral-8x7B", 4096, 14336, 2, 8),
    ModelProfile("DeepSeek-V3", 7168, 2048, 8, 256),
)
HARDWARE_PROFILES = {
    "H128": HardwareProfile("H128", 128.0e12, DIE_VECTOR_FLOPS),
    "H2000": HardwareProfile("H2000", 2000.0e12, DIE_VECTOR_FLOPS),
}
PLACEMENTS = (
    Placement(
        "compact", "Compact / adjacent", "6x6:absolute:center-2x2",
        ((2, 2), (3, 2), (2, 3), (3, 3)),
    ),
    Placement(
        "noncompact", "Non-compact / corners", "6x6:absolute:4x4-corners",
        ((1, 1), (4, 1), (1, 4), (4, 4)),
    ),
)
HBM_STACKS = tuple(default_hbm_stacks())


def align_up(value: int, multiple: int) -> int:
    if value < 0 or multiple <= 0:
        raise ValueError("value must be non-negative and multiple positive")
    return ((value + multiple - 1) // multiple) * multiple


def tile_live_bytes(
    mt: int = TILE_M, nt: int = TILE_N, kt: int = TILE_K,
    operator: str = "DISPATCH_GEMM",
) -> int:
    """Conservative tile live set including runtime reserve.

    Dispatch reserves a second double-buffered B stream for distinct gate/up
    weights.  Combine needs only one weight stream.
    """
    ordinary = (
        2 * DTYPE_BYTES * (mt * kt + kt * nt)
        + 2 * FP32_BYTES * mt * nt
        + RUNTIME_RESERVE_BYTES
    )
    if operator == "DISPATCH_GEMM":
        return ordinary + 2 * DTYPE_BYTES * kt * nt
    if operator == "GEMM_COMBINE":
        return ordinary
    raise ValueError(f"unsupported operator: {operator}")


def iter_cases() -> Iterable[ExperimentCase]:
    """Return the 16 workload/placement cases; profiles are a separate axis."""
    for operator in OPERATORS:
        for placement in PLACEMENTS:
            for model in MODELS:
                for seq_len in SEQ_LENS:
                    yield ExperimentCase(operator, placement, model, seq_len)


def _assignment_matrix(case: ExperimentCase) -> list[list[int]]:
    """Deterministically balance integer assignments across sources/experts."""
    base_tokens, token_remainder = divmod(case.seq_len, EP_SIZE)
    matrix: list[list[int]] = []
    for source_rank in range(EP_SIZE):
        source_tokens = base_tokens + int(source_rank < token_remainder)
        source_assignments = source_tokens * case.model.top_k
        quotient, remainder = divmod(source_assignments, case.model.expert_count)
        row = [quotient] * case.model.expert_count
        start = (source_rank * max(1, remainder)) % case.model.expert_count
        for offset in range(remainder):
            row[(start + offset) % case.model.expert_count] += 1
        matrix.append(row)
    return matrix


def normalize(case: ExperimentCase) -> dict[str, object]:
    """Normalize with per-expert, never rank-flattened, M padding."""
    matrix = _assignment_matrix(case)
    expert_logical_m = [
        sum(matrix[source][expert] for source in range(EP_SIZE))
        for expert in range(case.model.expert_count)
    ]
    expert_runtime_m = [align_up(value, TILE_M) for value in expert_logical_m]
    expert_home_rank = [expert % EP_SIZE for expert in range(case.model.expert_count)]
    per_rank_expert_ids = [
        [expert for expert, home in enumerate(expert_home_rank) if home == rank]
        for rank in range(EP_SIZE)
    ]
    per_rank_logical_m = [
        sum(expert_logical_m[expert] for expert in expert_ids)
        for expert_ids in per_rank_expert_ids
    ]
    per_rank_runtime_m = [
        sum(expert_runtime_m[expert] for expert in expert_ids)
        for expert_ids in per_rank_expert_ids
    ]
    if case.operator == "DISPATCH_GEMM":
        logical_n = case.model.expert_intermediate_size
        logical_k = case.model.hidden_size
    elif case.operator == "GEMM_COMBINE":
        logical_n = case.model.hidden_size
        logical_k = case.model.expert_intermediate_size
    else:
        raise ValueError(f"unsupported operator: {case.operator}")
    runtime_n = align_up(logical_n, TILE_N)
    runtime_k = align_up(logical_k, TILE_K)
    bottleneck_rank = max(range(EP_SIZE), key=per_rank_runtime_m.__getitem__)
    runtime_m = per_rank_runtime_m[bottleneck_rank]
    logical_assignments = case.seq_len * case.model.top_k
    total_runtime_m = sum(expert_runtime_m)
    return {
        "logical_assignment_count": logical_assignments,
        "assignment_matrix": matrix,
        "expert_home_rank": expert_home_rank,
        "per_expert_logical_M": expert_logical_m,
        "per_expert_runtime_M": expert_runtime_m,
        "per_expert_padding_ratio": [
            runtime / logical if logical else 1.0
            for logical, runtime in zip(
                expert_logical_m, expert_runtime_m, strict=True,
            )
        ],
        "per_expert_Tm": [value // TILE_M for value in expert_runtime_m],
        "per_rank_expert_ids": per_rank_expert_ids,
        "per_rank_logical_M": per_rank_logical_m,
        "per_rank_runtime_M": per_rank_runtime_m,
        "rank_logical_M": per_rank_logical_m[bottleneck_rank],
        "runtime_M": runtime_m,
        "total_runtime_M": total_runtime_m,
        "bottleneck_rank": bottleneck_rank,
        "logical_N": logical_n,
        "logical_K": logical_k,
        "runtime_N": runtime_n,
        "runtime_K": runtime_k,
        "Tm": runtime_m // TILE_M,
        "Tm_group": runtime_m // TILE_M,
        "Tn": runtime_n // TILE_N,
        "Tk": runtime_k // TILE_K,
        "padding_ratio": total_runtime_m / logical_assignments,
        "expert_load_min": min(expert_logical_m),
        "expert_load_max": max(expert_logical_m),
        "expert_load_mean": logical_assignments / case.model.expert_count,
    }


def _xy_directed_edges(
    source: tuple[int, int], destination: tuple[int, int],
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Return deterministic X-first routes on the 6x6 physical mesh."""
    x, y = source
    dst_x, dst_y = destination
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while x != dst_x:
        next_x = x + (1 if dst_x > x else -1)
        edges.append(((x, y), (next_x, y)))
        x = next_x
    while y != dst_y:
        next_y = y + (1 if dst_y > y else -1)
        edges.append(((x, y), (x, next_y)))
        y = next_y
    return tuple(edges)


def _add_route_bytes(
    loads: dict[tuple[tuple[int, int], tuple[int, int]], float],
    source: tuple[int, int], destination: tuple[int, int], payload_bytes: float,
) -> int:
    route = _xy_directed_edges(source, destination)
    for edge in route:
        loads[edge] = loads.get(edge, 0.0) + payload_bytes
    return len(route)


def _serialized_link_loads(
    loads: dict[tuple[tuple[int, int], tuple[int, int]], float],
) -> list[dict[str, object]]:
    return [
        {"source": list(source), "destination": list(destination), "bytes": value}
        for (source, destination), value in sorted(loads.items())
    ]


def _d2d_metrics(
    case: ExperimentCase, shape: dict[str, object], physical: PhysicalModel,
) -> dict[str, object]:
    """Aggregate every explicit physical-model group on directed links."""
    loads: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    outbound = [0.0] * EP_SIZE
    inbound = [0.0] * EP_SIZE
    per_die_outbound = [0.0] * 36
    per_die_inbound = [0.0] * 36
    max_hops = 0
    for flow in physical.flows:
        if case.operator == "DISPATCH_GEMM":
            source_rank, destination_rank = flow.source_rank, flow.destination_rank
            source_die, destination_die = flow.source_die, flow.destination_die
        else:
            source_rank, destination_rank = flow.destination_rank, flow.source_rank
            source_die, destination_die = flow.destination_die, flow.source_die
        source_coord = die_id_to_coordinate(source_die)
        destination_coord = die_id_to_coordinate(destination_die)
        payload = flow.payload_bytes
        outbound[source_rank] += payload
        inbound[destination_rank] += payload
        per_die_outbound[source_die] += payload
        per_die_inbound[destination_die] += payload
        max_hops = max(max_hops, _add_route_bytes(
            loads,
            (source_coord.x, source_coord.y),
            (destination_coord.x, destination_coord.y),
            payload,
        ))
    scenario_logical_assignments = physical.logical_assignments
    remote_assignments = physical.remote_assignments
    local_assignments = scenario_logical_assignments - remote_assignments
    group_count = len(physical.groups)
    per_group_local = local_assignments // group_count
    per_group_remote = remote_assignments // group_count
    max_link_bytes = max(loads.values(), default=0.0)
    max_outbound = max(per_die_outbound, default=0.0)
    max_inbound = max(per_die_inbound, default=0.0)
    link_cycles = max_link_bytes / D2D_LINK_BPS * CLOCK_HZ
    injection_cycles = max_outbound / D2D_INJECTION_BPS * CLOCK_HZ
    ejection_cycles = max_inbound / D2D_INJECTION_BPS * CLOCK_HZ
    return {
        "local_assignment_count": local_assignments,
        "remote_assignment_count": remote_assignments,
        "per_group_local_assignment_count": per_group_local,
        "per_group_remote_assignment_count": per_group_remote,
        "scenario_logical_assignment_count": scenario_logical_assignments,
        "local_assignment_fraction": local_assignments / scenario_logical_assignments,
        "remote_assignment_fraction": remote_assignments / scenario_logical_assignments,
        "remote_payload_bytes": remote_assignments * case.model.hidden_size * DTYPE_BYTES,
        "packet_metadata_bytes": len(physical.flows) * 16,
        "packet_gate_weight_bytes": remote_assignments * DTYPE_BYTES,
        "remote_packet_count": len(physical.flows),
        "physical_flow_count": len(physical.flows),
        "scenario_group_count": group_count,
        "per_rank_outbound_bytes": outbound,
        "per_rank_inbound_bytes": inbound,
        "per_die_outbound_bytes": per_die_outbound,
        "per_die_inbound_bytes": per_die_inbound,
        "runtime_comm_bytes_per_die": max_outbound,
        "max_directed_link_bytes": max_link_bytes,
        "total_directed_link_bytes": sum(loads.values()),
        "runtime_comm_byte_hops_per_die": sum(loads.values()) / (EP_SIZE * group_count),
        "max_path_hops": max_hops,
        "d2d_directed_link_loads": _serialized_link_loads(loads),
        "d2d_link_cycles": link_cycles,
        "noc_injection_cycles": injection_cycles,
        "noc_ejection_cycles": ejection_cycles,
        "comm_ideal_cycles": max(link_cycles, injection_cycles, ejection_cycles),
        "d2d_load_source": "physical_model_explicit_group_flow_aggregation",
        "_raw_d2d_link_loads": loads,
    }


def _hbm_metrics(
    case: ExperimentCase, shape: dict[str, object], physical: PhysicalModel,
) -> dict[str, object]:
    runtime_m = shape["per_expert_runtime_M"]
    expert_tm = shape["per_expert_Tm"]
    homes = shape["expert_home_rank"]
    assert isinstance(runtime_m, list) and isinstance(expert_tm, list)
    assert isinstance(homes, list)
    runtime_k = int(shape["runtime_K"])
    runtime_n = int(shape["runtime_N"])
    tn = int(shape["Tn"])
    stack_reads = [0.0] * len(HBM_STACKS)
    per_rank_reads = [0.0] * EP_SIZE
    per_die_reads = [0.0] * 36
    allocations: list[dict[str, object]] = []
    per_expert_bytes: list[float] = []
    route_loads: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    binding_lookup = {
        (binding.expert_id, binding.matrix): binding
        for binding in physical.weight_allocation.bindings
    }
    audit_by_stack = {
        audit.stack_id: audit for audit in physical.weight_allocation.stack_audits
    }
    for expert, (logical_m, tm) in enumerate(zip(runtime_m, expert_tm, strict=True)):
        home = homes[expert]
        if case.operator == "DISPATCH_GEMM":
            matrices = ("gate", "up")
            replay_bytes = 2 * tm * runtime_k * runtime_n * DTYPE_BYTES
            activation_replay_bytes = 0
        else:
            matrices = ("down",)
            activation_replay_bytes = tn * logical_m * runtime_k * DTYPE_BYTES
            replay_bytes = (
                tm * runtime_k * runtime_n * DTYPE_BYTES
                + activation_replay_bytes
            )
        bindings = [binding_lookup[(expert, matrix)] for matrix in matrices]
        stack_id = bindings[0].stack_id
        if any(binding.stack_id != stack_id for binding in bindings):
            raise AssertionError("one expert's selected roots must share an owner stack")
        stack_reads[stack_id] += replay_bytes * len(physical.groups)
        per_rank_reads[home] += replay_bytes
        per_expert_bytes.append(replay_bytes)
        allocations.append({
            "expert_id": expert,
            "home_rank": home,
            "stack_id": stack_id,
            "matrices": list(matrices),
            "matrix_bindings": [binding.manifest_dict() for binding in bindings],
            "address_begin": min(binding.address for binding in bindings),
            "address_end": max(
                binding.address + binding.size_bytes for binding in bindings
            ),
            "weight_bytes": sum(binding.size_bytes for binding in bindings),
            "activation_replay_bytes": activation_replay_bytes,
            "read_bytes": replay_bytes,
        })
        stack_coord = physical.stacks[stack_id].coordinate
        for group in physical.groups:
            destination_die = group.rank_to_die[home]
            destination_coord = die_id_to_coordinate(destination_die)
            per_die_reads[destination_die] += replay_bytes
            _add_route_bytes(
                route_loads, (stack_coord.x, stack_coord.y),
                (destination_coord.x, destination_coord.y), replay_bytes,
            )
    stack_records: list[dict[str, object]] = []
    for stack, reads in zip(physical.stacks, stack_reads, strict=True):
        audit = audit_by_stack[stack.stack_id]
        stack_records.append({
            **stack.manifest_dict(),
            "attachment": stack.coordinate.as_list(),
            "capacity_used_bytes": audit.allocated_span_bytes,
            "capacity_utilization": (
                audit.allocated_span_bytes / stack.capacity_bytes
            ),
            "read_bytes": reads,
            "write_bytes": 0,
            "service_cycles": reads / stack.bandwidth_Bps * CLOCK_HZ,
            "queueing_cycles": 0.0,
        })
    max_service = max(item["service_cycles"] for item in stack_records)
    max_route_bytes = max(route_loads.values(), default=0.0)
    route_cycles = max_route_bytes / D2D_LINK_BPS * CLOCK_HZ
    return {
        "hbm_bytes_per_die": max(per_rank_reads),
        "per_rank_hbm_read_bytes": per_rank_reads,
        "per_die_hbm_read_bytes": per_die_reads,
        "per_expert_hbm_read_bytes": per_expert_bytes,
        "per_expert_hbm_read_bytes_scope": "per_physical_group",
        "hbm_total_read_bytes": sum(stack_reads),
        "hbm_group_count": len(physical.groups),
        "hbm_stack_count": len(HBM_STACKS),
        "hbm_stack_bandwidth_Bps": HBM_STACK_BPS,
        "hbm_stacks": stack_records,
        "hbm_weight_allocations": allocations,
        "hbm_capacity_scope": "complete_gate_up_down_moe_layer_weight_set",
        "hbm_address_owner_unique": True,
        "hbm_capacity_feasible": physical.weight_allocation.feasible,
        "hbm_max_stack_service_cycles": max_service,
        "hbm_route_max_directed_link_bytes": max_route_bytes,
        "hbm_route_cycles": route_cycles,
        "hbm_directed_link_loads": _serialized_link_loads(route_loads),
        "hbm_cycles": max(max_service, route_cycles),
        "_raw_hbm_link_loads": route_loads,
    }


def _layout_coord(
    ie: int, im: int, inn: int, ik: int,
    pe: int, pm: int, pn: int, pk: int,
    order: tuple[str, str, str, str],
) -> tuple[int, int]:
    indices = {"e": ie, "m": im, "n": inn, "k": ik}
    extents = {"e": pe, "m": pm, "n": pn, "k": pk}
    linear = indices[order[0]]
    for dimension in order[1:]:
        linear = linear * extents[dimension] + indices[dimension]
    return divmod(linear, 4)


def _local_noc_route_metrics(
    pe: int, pm: int, pn: int, pk: int,
    a_bytes: float, b_bytes: float, reduction_bytes: float,
    *, optimize_mapping: bool,
) -> dict[str, object]:
    orders = (
        tuple(itertools.permutations(("e", "m", "n", "k")))
        if optimize_mapping else (("e", "m", "n", "k"),)
    )
    candidates: list[tuple[tuple[float, float, str], dict[str, object]]] = []
    for order in orders:
        coord = lambda ie, im, inn, ik: _layout_coord(
            ie, im, inn, ik, pe, pm, pn, pk, order,
        )
        a_messages = [
            (coord(ie, im, 0, ik), coord(ie, im, inn, ik))
            for ie in range(pe) for im in range(pm) for ik in range(pk)
            for inn in range(1, pn)
        ]
        b_messages = [
            (coord(ie, 0, inn, ik), coord(ie, im, inn, ik))
            for ie in range(pe) for inn in range(pn) for ik in range(pk)
            for im in range(1, pm)
        ]
        reduction_messages = [
            (coord(ie, im, inn, ik), coord(ie, im, inn, 0))
            for ie in range(pe) for im in range(pm) for inn in range(pn)
            for ik in range(1, pk)
        ]
        loads: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
        for messages, total_bytes in (
            (a_messages, a_bytes), (b_messages, b_bytes),
            (reduction_messages, reduction_bytes),
        ):
            if not messages or total_bytes <= 0:
                continue
            bytes_per_message = total_bytes / len(messages)
            for source, destination in messages:
                _add_route_bytes(loads, source, destination, bytes_per_message)
        maximum = max(loads.values(), default=0.0)
        total_hops = sum(loads.values())
        order_name = "".join(order)
        metrics = {
            "intra_core_mapping_order": order_name,
            "local_noc_max_directed_link_bytes": maximum,
            "local_noc_total_byte_hops": total_hops,
            "local_noc_physical_links_used": len(loads),
            "local_noc_directed_link_loads": _serialized_link_loads(loads),
        }
        candidates.append(((maximum, total_hops, order_name), metrics))
    return min(candidates, key=lambda item: item[0])[1]


def _evaluate_intra_schedule(
    shape: dict[str, object], flops_per_die: float, profile: HardwareProfile,
    pe: int, pm: int, pn: int, pk: int, *, optimize_mapping: bool,
) -> dict[str, object]:
    """Evaluate a legal PE x PM x PN x PK grouped-GEMM schedule."""
    active_cores = pe * pm * pn * pk
    if active_cores > DIE_CORES or active_cores <= 0:
        raise ValueError("grouped schedule exceeds the 16 compute cores")
    local_ids = shape["per_rank_expert_ids"][int(shape["bottleneck_rank"])]
    expert_tm_all = shape["per_expert_Tm"]
    assert isinstance(local_ids, list) and isinstance(expert_tm_all, list)
    expert_tm = [expert_tm_all[expert] for expert in local_ids]
    tn, tk = int(shape["Tn"]), int(shape["Tk"])
    if pe > len(expert_tm) or pm > max(expert_tm) or pn > tn or pk > tk:
        raise ValueError("grouped schedule extent exceeds the workload")
    a_broadcast = 0
    b_broadcast = 0
    reduction = 0
    for tm in expert_tm:
        output_tiles = tm * tn
        active_pm = min(pm, tm)
        active_pn = min(pn, tn)
        active_pk = min(pk, tk)
        # Tk is intentionally present: every output tile traverses all K tiles.
        a_broadcast += (
            output_tiles * tk * (active_pn - 1)
            * TILE_M * TILE_K * DTYPE_BYTES
        )
        b_broadcast += (
            output_tiles * tk * (active_pm - 1)
            * TILE_K * TILE_N * DTYPE_BYTES
        )
        reduction += (
            output_tiles * (active_pk - 1)
            * TILE_M * TILE_N * FP32_BYTES
        )
    noc = _local_noc_route_metrics(
        pe, pm, pn, pk, a_broadcast, b_broadcast, reduction,
        optimize_mapping=optimize_mapping,
    )
    expert_util = len(expert_tm) / (math.ceil(len(expert_tm) / pe) * pe)
    m_util = sum(expert_tm) / sum(math.ceil(tm / pm) * pm for tm in expert_tm)
    n_util = tn / (math.ceil(tn / pn) * pn)
    k_util = tk / (math.ceil(tk / pk) * pk)
    schedule_utilization = expert_util * m_util * n_util * k_util
    tensor_efficiency = TENSOR_EFFICIENCY_PRIOR * math.sqrt(schedule_utilization)
    core_tensor_flops = profile.die_tensor_flops / DIE_CORES
    compute_cycles = (
        flops_per_die / (active_cores * core_tensor_flops * tensor_efficiency)
        * CLOCK_HZ
    )
    transport_bytes = a_broadcast + b_broadcast + reduction
    serial_cycles = transport_bytes / LOCAL_NOC_BPS * CLOCK_HZ
    transport_cycles = (
        float(noc["local_noc_max_directed_link_bytes"])
        / LOCAL_NOC_BPS * CLOCK_HZ
    )
    return {
        "intra_pe": pe,
        "intra_pm": pm,
        "intra_pn": pn,
        "intra_pk": pk,
        "active_cores": active_cores,
        "underfilled_cores": DIE_CORES - active_cores,
        "intra_spatial_utilization": schedule_utilization,
        "intra_compute_efficiency": tensor_efficiency,
        "compute_cycles": compute_cycles,
        "a_broadcast_bytes": a_broadcast,
        "b_broadcast_bytes": b_broadcast,
        "reduction_bytes": reduction,
        "local_transport_bytes": transport_bytes,
        "local_transport_serial_cycles": serial_cycles,
        "local_transport_cycles": transport_cycles,
        **noc,
    }


def _baseline_intra_schedule(
    shape: dict[str, object], flops_per_die: float, profile: HardwareProfile,
) -> dict[str, object]:
    """Deterministically project the old 4x4x1 mapping onto grouped GEMM."""
    local_ids = shape["per_rank_expert_ids"][int(shape["bottleneck_rank"])]
    expert_tm_all = shape["per_expert_Tm"]
    assert isinstance(local_ids, list) and isinstance(expert_tm_all, list)
    max_tm = max(expert_tm_all[expert] for expert in local_ids)
    pm = min(4, max_tm)
    pn = min(4, int(shape["Tn"]), DIE_CORES // pm)
    pk = 1
    remaining = DIE_CORES // (pm * pn * pk)
    pe = min(len(local_ids), max(1, remaining))
    return _evaluate_intra_schedule(
        shape, flops_per_die, profile, pe, pm, pn, pk,
        optimize_mapping=False,
    )


def _select_intra_schedule(
    shape: dict[str, object], flops_per_die: float, profile: HardwareProfile,
    modeled_hbm_cycles: float,
) -> dict[str, object]:
    local_ids = shape["per_rank_expert_ids"][int(shape["bottleneck_rank"])]
    expert_tm_all = shape["per_expert_Tm"]
    assert isinstance(local_ids, list) and isinstance(expert_tm_all, list)
    max_tm = max(expert_tm_all[expert] for expert in local_ids)
    factors = (1, 2, 4, 8, 16)
    candidates: list[tuple[tuple[float, ...], dict[str, object]]] = []
    for pe in factors:
        if pe > len(local_ids):
            continue
        for pm in factors:
            if pm > max_tm:
                continue
            for pn in factors:
                if pn > int(shape["Tn"]):
                    continue
                for pk in factors:
                    if pk > int(shape["Tk"]) or pe * pm * pn * pk > DIE_CORES:
                        continue
                    schedule = _evaluate_intra_schedule(
                        shape, flops_per_die, profile, pe, pm, pn, pk,
                        optimize_mapping=True,
                    )
                    stage_cycles = max(
                        float(schedule["compute_cycles"]), modeled_hbm_cycles,
                        float(schedule["local_transport_cycles"]),
                    )
                    schedule["intra_stage_cycles"] = stage_cycles
                    candidates.append(((
                        stage_cycles,
                        -float(schedule["intra_spatial_utilization"]),
                        float(schedule["local_transport_cycles"]),
                        -int(schedule["active_cores"]), pe, pm, pn, pk,
                    ), schedule))
    if not candidates:
        raise ValueError(f"no legal grouped schedule for {shape}")
    return min(candidates, key=lambda item: item[0])[1]


def _combine_reduction_noc_metrics(
    total_bytes: float, top_k: int,
) -> tuple[float, float, int]:
    if total_bytes <= 0 or top_k <= 1:
        return 0.0, 0.0, 0
    loads: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    messages: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for owner in range(DIE_CORES):
        for offset in range(1, min(top_k, DIE_CORES)):
            messages.append((divmod((owner + offset) % DIE_CORES, 4), divmod(owner, 4)))
    bytes_per_message = total_bytes / len(messages)
    for source, destination in messages:
        _add_route_bytes(loads, source, destination, bytes_per_message)
    return max(loads.values(), default=0.0), sum(loads.values()), len(loads)


def _earliest_schedule(events: list[ReplayEvent]) -> dict[str, tuple[float, float]]:
    finish: dict[str, float] = {}
    resource_ready: dict[str, float] = {}
    timings: dict[str, tuple[float, float]] = {}
    for event in events:
        if any(ref not in finish for ref in event.deps):
            raise ValueError(f"event {event.event_id} has a dangling dependency")
        start = max(
            max((finish[ref] for ref in event.deps), default=0.0),
            resource_ready.get(event.resource, 0.0),
        )
        end = start + event.duration
        finish[event.event_id] = end
        resource_ready[event.resource] = end
        timings[event.event_id] = (start, end)
    return timings


def _candidate_grid(case: ExperimentCase) -> tuple[InterCandidate, ...]:
    owner_slots = max(1, round(DIE_CORES * 0.75))
    direct = InterCandidate("direct_xy_personalized_a2a", 1, 1, False, owner_slots)
    if case.placement.name != "compact":
        return (direct,)
    return (
        direct,
        InterCandidate("comet_mesh_personalized_a2a_u1", 2, 1, False, owner_slots),
        InterCandidate("comet_mesh_personalized_a2a_u2_db", 2, 2, True, 2 * owner_slots),
    )


def _replay_fused_candidate(
    case: ExperimentCase, shape: dict[str, object], candidate: InterCandidate,
    intra_cycles: float, comm_stream_cycles: float,
    combine_reduce_cycles: float, fusion_setup_cycles: float,
) -> dict[str, object]:
    transport_blocks = 1 if case.operator == "DISPATCH_GEMM" else 2
    waves = max(1, int(shape["Tm_group"]) * transport_blocks)
    physical_hops = max(
        len(_xy_directed_edges(source, destination))
        for source in case.placement.member_coordinates
        for destination in case.placement.member_coordinates
    )
    packet_overhead = (
        SESSION_OPEN_CYCLES + DTE_LAUNCH_CYCLES
        + physical_hops * DTE_HOP_CYCLES
        + DTE_SYNC_CYCLES + SESSION_RETIRE_CYCLES
    )
    network_per_wave = comm_stream_cycles / waves + packet_overhead * candidate.stages
    compute_only_cycles = max(0.0, intra_cycles - combine_reduce_cycles)
    compute_per_wave = compute_only_cycles / waves
    reduce_per_wave = (
        combine_reduce_cycles / waves if case.operator == "GEMM_COMBINE" else 0.0
    )
    events: list[ReplayEvent] = []
    terminals: list[str] = []
    for wave in range(waves):
        reuse_dep = (terminals[wave - candidate.physical_slots],) \
            if wave >= candidate.physical_slots else ()
        network_ids = tuple(
            f"network.{stage}.{wave}" for stage in range(candidate.stages)
        )
        comp_id = f"comp.{wave}"
        if case.operator == "DISPATCH_GEMM":
            previous = reuse_dep
            for stage, event_id in enumerate(network_ids):
                events.append(ReplayEvent(
                    event_id, "dispatch_send_recv_wait", previous,
                    f"d2d.mesh_stage_{stage}", network_per_wave / candidate.stages,
                ))
                previous = (event_id,)
            events.append(ReplayEvent(
                comp_id, "gate_up_gemm_swiglu", previous,
                "compute.tensor_vector_hbm_noc",
                compute_per_wave,
            ))
            terminal = comp_id
        else:
            events.append(ReplayEvent(
                comp_id, "down_gemm", reuse_dep,
                "compute.tensor_hbm_noc",
                compute_per_wave,
            ))
            previous = (comp_id,)
            for stage, event_id in enumerate(network_ids):
                events.append(ReplayEvent(
                    event_id, "combine_send_recv_wait", previous,
                    f"d2d.mesh_stage_{stage}", network_per_wave / candidate.stages,
                ))
                previous = (event_id,)
            reduce_id = f"topk_join_reduce.{wave}"
            events.append(ReplayEvent(
                reduce_id, "topk_join_weighted_combine", previous,
                "vector.combine_reducer", reduce_per_wave,
            ))
            terminal = reduce_id
        terminals.append(terminal)
    timings = _earliest_schedule(events)
    steady_cycles = max((end for _, end in timings.values()), default=0.0)
    return {
        "algorithm": candidate.name,
        "dependency_order": (
            "COMM->WAIT->GATE_GEMM->UP_GEMM->SWIGLU"
            if case.operator == "DISPATCH_GEMM"
            else "DOWN_GEMM->COMM->TOPK_JOIN->WEIGHTED_COMBINE"
        ),
        "transport_output_blocks": transport_blocks,
        "pipeline_waves": waves,
        "packet_count": waves * candidate.stages,
        "action_count": len(events),
        "physical_slots": candidate.physical_slots,
        "unroll_degree": candidate.unroll_degree,
        "double_buffer": candidate.double_buffer,
        "physical_hops": physical_hops,
        "packet_overhead_cycles": packet_overhead,
        "network_per_wave_cycles": network_per_wave,
        "compute_per_wave_cycles": compute_per_wave,
        "reduce_per_wave_cycles": reduce_per_wave,
        "steady_cycles": steady_cycles,
        "makespan_cycles": fusion_setup_cycles + steady_cycles,
    }


def _select_inter_schedule(
    case: ExperimentCase, shape: dict[str, object], intra_cycles: float,
    comm_stream_cycles: float, combine_reduce_cycles: float,
    fusion_setup_cycles: float,
) -> dict[str, object]:
    candidates = tuple(
        _replay_fused_candidate(
            case, shape, candidate, intra_cycles, comm_stream_cycles,
            combine_reduce_cycles, fusion_setup_cycles,
        )
        for candidate in _candidate_grid(case)
    )
    selected = min(candidates, key=lambda item: (
        item["makespan_cycles"], item["packet_count"], item["algorithm"],
    ))
    return {
        **selected,
        "candidate_count": len(candidates),
        "candidate_cycles": {
            item["algorithm"]: item["makespan_cycles"] for item in candidates
        },
    }


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_profile(profile: HardwareProfile | str | None) -> HardwareProfile:
    if profile is None:
        return HARDWARE_PROFILES["H2000"]
    if isinstance(profile, HardwareProfile):
        return profile
    key = profile.upper()
    if key not in HARDWARE_PROFILES:
        raise ValueError(f"unknown hardware profile: {profile}")
    return HARDWARE_PROFILES[key]


def estimate_case(
    case: ExperimentCase, *, include_hbm: bool = True,
    hardware_profile: HardwareProfile | str | None = None,
    network_scenario: str = NETWORK_SCENARIO,
) -> dict[str, object]:
    if network_scenario not in ("isolated_group", "loaded_groups"):
        raise ValueError(f"unsupported network scenario: {network_scenario}")
    profile = _resolve_profile(hardware_profile)
    shape = normalize(case)
    model = case.model
    runtime_k, runtime_n = int(shape["runtime_K"]), int(shape["runtime_N"])
    per_rank_ids = shape["per_rank_expert_ids"]
    expert_runtime_m = shape["per_expert_runtime_M"]
    assert isinstance(per_rank_ids, list) and isinstance(expert_runtime_m, list)

    flop_factor = 4 if case.operator == "DISPATCH_GEMM" else 2
    per_rank_runtime_flops = [
        sum(
            flop_factor * expert_runtime_m[expert] * runtime_k * runtime_n
            for expert in expert_ids
        )
        for expert_ids in per_rank_ids
    ]
    runtime_flops_per_die = max(per_rank_runtime_flops)
    runtime_total_flops = sum(per_rank_runtime_flops)
    logical_flops = (
        flop_factor * case.seq_len * model.top_k
        * model.hidden_size * model.expert_intermediate_size
    )
    logical_gate_flops = logical_flops // 2 if case.operator == "DISPATCH_GEMM" else 0
    logical_up_flops = logical_gate_flops
    logical_down_flops = logical_flops if case.operator == "GEMM_COMBINE" else 0
    compute_ideal_cycles = runtime_flops_per_die / profile.die_tensor_flops * CLOCK_HZ
    compute_single_core_cycles = (
        runtime_flops_per_die / (profile.die_tensor_flops / DIE_CORES) * CLOCK_HZ
    )

    physical = build_physical_model(
        placement=case.placement.name,
        network_scenario=network_scenario,
        assignments=shape["assignment_matrix"],
        expert_home_ranks=shape["expert_home_rank"],
        hidden_size=model.hidden_size,
        intermediate_size=model.expert_intermediate_size,
        dtype_bytes=DTYPE_BYTES,
        strict_capacity=False,
    )
    physical_manifest = physical.manifest_dict()
    physical_coordinates = tuple(
        (coordinate.x, coordinate.y)
        for coordinate in physical.groups[0].rank_coordinates
    )
    if (
        network_scenario == "isolated_group"
        and physical_coordinates != case.placement.member_coordinates
    ):
        raise AssertionError("runner placement drifted from physical_model.py")
    d2d = _d2d_metrics(case, shape, physical)
    hbm = _hbm_metrics(case, shape, physical)
    d2d_raw = d2d.pop("_raw_d2d_link_loads")
    hbm_raw = hbm.pop("_raw_hbm_link_loads")
    shared_link_loads = dict(d2d_raw)
    if include_hbm:
        for edge, value in hbm_raw.items():
            shared_link_loads[edge] = shared_link_loads.get(edge, 0.0) + value
    shared_max_bytes = max(shared_link_loads.values(), default=0.0)
    shared_fabric_cycles = shared_max_bytes / D2D_LINK_BPS * CLOCK_HZ
    modeled_hbm_cycles = float(hbm["hbm_cycles"]) if include_hbm else 0.0

    optimized_schedule = _select_intra_schedule(
        shape, runtime_flops_per_die, profile, modeled_hbm_cycles,
    )
    baseline_schedule = _baseline_intra_schedule(
        shape, runtime_flops_per_die, profile,
    )

    rank_tokens = math.ceil(case.seq_len / EP_SIZE)
    if case.operator == "DISPATCH_GEMM":
        per_rank_vector_ops = [
            sum(4 * expert_runtime_m[e] * runtime_n for e in ids)
            for ids in per_rank_ids
        ]
        vector_ops = max(per_rank_vector_ops)
        post_gemm_vector_cycles = vector_ops / profile.die_vector_flops * CLOCK_HZ
        combine_reduce_bytes = 0
        combine_reduce_noc_max = 0.0
        combine_reduce_noc_hops = 0.0
        combine_reduce_noc_links = 0
        combine_reduce_noc_cycles = 0.0
        combine_reduce_noc_serial_cycles = 0.0
        combine_reduce_vector_cycles = 0.0
        combine_reduce_cycles = 0.0
    else:
        vector_ops = rank_tokens * model.hidden_size * (2 * model.top_k - 1)
        post_gemm_vector_cycles = 0.0
        combine_reduce_bytes = (
            rank_tokens * max(0, model.top_k - 1)
            * model.hidden_size * FP32_BYTES
        )
        (
            combine_reduce_noc_max,
            combine_reduce_noc_hops,
            combine_reduce_noc_links,
        ) = _combine_reduction_noc_metrics(combine_reduce_bytes, model.top_k)
        combine_reduce_noc_serial_cycles = (
            combine_reduce_bytes / LOCAL_NOC_BPS * CLOCK_HZ
        )
        combine_reduce_noc_cycles = combine_reduce_noc_max / LOCAL_NOC_BPS * CLOCK_HZ
        combine_reduce_vector_cycles = vector_ops / profile.die_vector_flops * CLOCK_HZ
        combine_reduce_cycles = max(
            combine_reduce_noc_cycles, combine_reduce_vector_cycles,
        )

    baseline_compute = float(baseline_schedule["compute_cycles"])
    baseline_noc = float(baseline_schedule["local_transport_cycles"])
    optimized_compute = float(optimized_schedule["compute_cycles"])
    optimized_noc = float(optimized_schedule["local_transport_cycles"])
    # Baseline serializes resources. Optimized may overlap them, but shared HBM
    # and all-to-all route occupancy remains an explicit physical floor.
    intra_naive_cycles = (
        baseline_compute + modeled_hbm_cycles + baseline_noc
        + post_gemm_vector_cycles + combine_reduce_cycles
    )
    intra_optimized_cycles = (
        max(optimized_compute, modeled_hbm_cycles, optimized_noc,
            shared_fabric_cycles if include_hbm else 0.0)
        + post_gemm_vector_cycles + combine_reduce_cycles
    )

    physical_hops = int(d2d["max_path_hops"])
    lifecycle_cycles = (
        SESSION_OPEN_CYCLES + DTE_LAUNCH_CYCLES
        + physical_hops * DTE_HOP_CYCLES
        + DTE_SYNC_CYCLES + SESSION_RETIRE_CYCLES
    )
    comm_ideal_cycles = float(d2d["comm_ideal_cycles"])
    comm_unfused_cycles = comm_ideal_cycles + 2 * lifecycle_cycles
    comm_fused_stream_cycles = comm_ideal_cycles
    fusion_setup_cycles = lifecycle_cycles
    t00_cycles = comm_unfused_cycles + intra_naive_cycles
    t01_cycles = comm_unfused_cycles + intra_optimized_cycles
    t10_schedule = _select_inter_schedule(
        case, shape, intra_naive_cycles, comm_fused_stream_cycles,
        combine_reduce_cycles, fusion_setup_cycles,
    )
    t11_schedule = _select_inter_schedule(
        case, shape, intra_optimized_cycles, comm_fused_stream_cycles,
        combine_reduce_cycles, fusion_setup_cycles,
    )
    t10_cycles = float(t10_schedule["makespan_cycles"])
    t11_cycles = float(t11_schedule["makespan_cycles"])

    resource_floor_cycles = max(
        compute_ideal_cycles,
        modeled_hbm_cycles,
        optimized_noc,
        post_gemm_vector_cycles,
        combine_reduce_cycles,
        comm_ideal_cycles,
        shared_fabric_cycles,
    )
    precedence_floor_cycles = max(
        (comm_ideal_cycles + compute_ideal_cycles)
        / max(1, int(shape["Tm_group"])),
        combine_reduce_cycles,
    )
    architecture_lower_bound = max(resource_floor_cycles, precedence_floor_cycles)

    t00_i = max(1, round(t00_cycles))
    t10_i = max(1, round(t10_cycles))
    t01_i = max(1, round(t01_cycles))
    t11_i = max(1, round(t11_cycles))
    theory_i = max(1, round(architecture_lower_bound))
    actual_speedup = t00_i / t11_i
    theory_speedup = t00_i / theory_i
    attainment = actual_speedup / theory_speedup

    memory_mode = "architecture" if include_hbm else "hbm_free_compute_comm"
    hardware_manifest = {
        "wafer": physical_manifest["wafer"],
        "compute_cores_per_die": DIE_CORES,
        "tensor_profile": profile.name,
        "die_tensor_flops": profile.die_tensor_flops,
        "die_vector_flops": profile.die_vector_flops,
        "d2d_link_Bps_per_direction": D2D_LINK_BPS,
        "local_noc_link_Bps": LOCAL_NOC_BPS,
        "hbm_stacks": physical_manifest["hbm_stacks"],
        "simulator_unit_closure": physical_manifest["simulator_unit_closure"],
        "simulator_unit_closure_detail": physical_manifest[
            "simulator_unit_closure_detail"
        ],
    }
    mapping_manifest = {
        "placement": case.placement.name,
        "members": [list(coord) for coord in case.placement.member_coordinates],
        "physical_groups": physical_manifest["groups"],
        "expert_home_rank": shape["expert_home_rank"],
        "optimized": {
            key: optimized_schedule[f"intra_{key.lower()}"]
            for key in ("PE", "PM", "PN", "PK")
        },
    }
    workload_manifest = {
        "operator": case.operator, "model": model.name,
        "seq_len": case.seq_len, "top_k": model.top_k,
        "expert_count": model.expert_count,
        "assignment_matrix": shape["assignment_matrix"],
    }
    simulation_manifest = {
        "clock_Hz": CLOCK_HZ,
        "tile": [TILE_M, TILE_N, TILE_K],
        "network_scenario": network_scenario,
        "memory_mode": memory_mode,
        "tensor_efficiency_prior": TENSOR_EFFICIENCY_PRIOR,
        "simulator_unit_closure": False,
        "simulator_unit_closure_reason": physical_manifest[
            "simulator_unit_closure_detail"
        ]["reasons"],
    }
    return {
        "case_id": case.case_id,
        "profile_case_id": f"{profile.name.lower()}__{case.case_id}",
        "scenario_case_id": (
            f"{network_scenario}__{profile.name.lower()}__{case.case_id}"
        ),
        "operator": case.operator,
        "placement": case.placement.name,
        "placement_label": case.placement.label,
        "physical_mesh": case.placement.physical_mesh,
        "member_coordinates": [list(item) for item in case.placement.member_coordinates],
        "placement_coordinate_space": "absolute_6x6_wafer",
        "model": model.name,
        "seq_len": case.seq_len,
        "hidden_size": model.hidden_size,
        "expert_intermediate_size": model.expert_intermediate_size,
        "top_k": model.top_k,
        "expert_count": model.expert_count,
        "local_experts_per_rank": model.expert_count // EP_SIZE,
        "deepseek_scope": "routed_expert_only" if model.name == "DeepSeek-V3" else "all_routed_experts",
        "ep_rows": EP_ROWS,
        "ep_columns": EP_COLUMNS,
        "ep_size": EP_SIZE,
        "balanced_routing_assumption": True,
        **shape,
        "tile_M": TILE_M,
        "tile_N": TILE_N,
        "tile_K": TILE_K,
        "tile_live_bytes": tile_live_bytes(operator=case.operator),
        "sram_capacity_bytes": SRAM_CAPACITY_BYTES,
        "sram_runtime_reserve_bytes": RUNTIME_RESERVE_BYTES,
        "sram_capacity_feasible": tile_live_bytes(operator=case.operator) <= SRAM_CAPACITY_BYTES,
        "logical_flops": logical_flops,
        "scenario_logical_flops": logical_flops * len(physical.groups),
        "logical_gate_flops": logical_gate_flops,
        "logical_up_flops": logical_up_flops,
        "logical_down_flops": logical_down_flops,
        "runtime_flops": runtime_total_flops,
        "scenario_runtime_flops": runtime_total_flops * len(physical.groups),
        "runtime_flops_per_die": runtime_flops_per_die,
        "per_rank_runtime_flops": per_rank_runtime_flops,
        "runtime_flops_over_logical_flops": runtime_total_flops / logical_flops,
        **hbm,
        "modeled_hbm_cycles": modeled_hbm_cycles,
        "hbm_included": include_hbm,
        "memory_mode": memory_mode,
        "hbm_traffic_model": HBM_TRAFFIC_MODEL,
        "hbm_all_to_all_shared_max_directed_link_bytes": shared_max_bytes,
        "hbm_all_to_all_shared_link_cycles": shared_fabric_cycles,
        "shared_directed_link_loads": _serialized_link_loads(shared_link_loads),
        **optimized_schedule,
        "logical_hops": EP_ROWS + EP_COLUMNS - 2,
        "d2d_link_bandwidth_Bps_per_direction": D2D_LINK_BPS,
        "noc_link_bandwidth_Bps": NOC_LINK_BPS,
        "d2d_attached_noc_links": D2D_ATTACHED_NOC_LINKS,
        "d2d_injection_bandwidth_Bps": D2D_INJECTION_BPS,
        "contention_model": "explicit_directed_edge_flow_aggregation",
        "external_ep_group_flow_count": max(0, len(physical.groups) - 1),
        **d2d,
        "segment_bytes": float(d2d["runtime_comm_bytes_per_die"]) / max(1, int(shape["Tm_group"])),
        "segmented_comm_efficiency": 1.0,
        "route_efficiency": 1.0,
        "comm_actual_cycles": comm_fused_stream_cycles,
        "comm_fused_stream_cycles": comm_fused_stream_cycles,
        "comm_unfused_cycles": comm_unfused_cycles,
        "unfused_comm_efficiency": 1.0,
        "unfused_setup_cycles": 2 * lifecycle_cycles,
        "inter_pipeline_waves": t11_schedule["pipeline_waves"],
        "tensor_profile": profile.name,
        "hardware_profile": profile.name,
        "profile": profile.name,
        "die_tensor_flops": profile.die_tensor_flops,
        "core_tensor_flops": profile.die_tensor_flops / DIE_CORES,
        "die_vector_flops": profile.die_vector_flops,
        "core_vector_flops": profile.die_vector_flops / DIE_CORES,
        "tensor_rate_status": profile.tensor_rate_status,
        "vector_rate_status": profile.vector_rate_status,
        "tensor_efficiency_status": "provisional_uncalibrated_prior",
        "vector_ops": vector_ops,
        "post_gemm_vector_cycles": post_gemm_vector_cycles,
        "combine_reduce_bytes": combine_reduce_bytes,
        "combine_reduce_noc_max_directed_link_bytes": combine_reduce_noc_max,
        "combine_reduce_noc_total_byte_hops": combine_reduce_noc_hops,
        "combine_reduce_noc_physical_links_used": combine_reduce_noc_links,
        "combine_reduce_noc_serial_cycles": combine_reduce_noc_serial_cycles,
        "combine_reduce_noc_cycles": combine_reduce_noc_cycles,
        "combine_reduce_vector_cycles": combine_reduce_vector_cycles,
        "combine_reduce_cycles": combine_reduce_cycles,
        "compute_ideal_cycles": compute_ideal_cycles,
        "compute_single_core_cycles": compute_single_core_cycles,
        "baseline_intra_pe": baseline_schedule["intra_pe"],
        "baseline_intra_pm": baseline_schedule["intra_pm"],
        "baseline_intra_pn": baseline_schedule["intra_pn"],
        "baseline_intra_pk": baseline_schedule["intra_pk"],
        "baseline_active_cores": baseline_schedule["active_cores"],
        "baseline_compute_cycles": baseline_schedule["compute_cycles"],
        "baseline_local_transport_bytes": baseline_schedule["local_transport_bytes"],
        "baseline_local_transport_serial_cycles": baseline_schedule["local_transport_serial_cycles"],
        "baseline_local_transport_cycles": baseline_schedule["local_transport_cycles"],
        "baseline_local_noc_max_directed_link_bytes": baseline_schedule["local_noc_max_directed_link_bytes"],
        "baseline_local_noc_total_byte_hops": baseline_schedule["local_noc_total_byte_hops"],
        "baseline_local_noc_physical_links_used": baseline_schedule["local_noc_physical_links_used"],
        "baseline_intra_core_mapping_order": baseline_schedule["intra_core_mapping_order"],
        "baseline_core_schedule_efficiency": baseline_schedule["intra_compute_efficiency"],
        "core_schedule_efficiency": optimized_schedule["intra_compute_efficiency"],
        "intra_naive_cycles": intra_naive_cycles,
        "intra_optimized_cycles": intra_optimized_cycles,
        "intra_actual_cycles": intra_optimized_cycles,
        "t10_stage_balance": min(comm_fused_stream_cycles, intra_naive_cycles) / max(comm_fused_stream_cycles, intra_naive_cycles),
        "t11_stage_balance": min(comm_fused_stream_cycles, intra_optimized_cycles) / max(comm_fused_stream_cycles, intra_optimized_cycles),
        "stage_balance": min(comm_fused_stream_cycles, intra_optimized_cycles) / max(comm_fused_stream_cycles, intra_optimized_cycles),
        "fusion_setup_cycles": fusion_setup_cycles,
        "T10_inter_algorithm": t10_schedule["algorithm"],
        "T11_inter_algorithm": t11_schedule["algorithm"],
        "inter_dependency_order": t11_schedule["dependency_order"],
        "inter_transport_output_blocks": t11_schedule["transport_output_blocks"],
        "inter_packet_count": t11_schedule["packet_count"],
        "inter_action_count": t11_schedule["action_count"],
        "inter_physical_slots": t11_schedule["physical_slots"],
        "inter_unroll_degree": t11_schedule["unroll_degree"],
        "inter_double_buffer": t11_schedule["double_buffer"],
        "inter_physical_hops": t11_schedule["physical_hops"],
        "inter_packet_overhead_cycles": t11_schedule["packet_overhead_cycles"],
        "inter_network_per_wave_cycles": t11_schedule["network_per_wave_cycles"],
        "inter_compute_per_wave_cycles": t11_schedule["compute_per_wave_cycles"],
        "inter_reduce_per_wave_cycles": t11_schedule["reduce_per_wave_cycles"],
        "inter_steady_cycles": t11_schedule["steady_cycles"],
        "inter_candidate_count": t11_schedule["candidate_count"],
        "T10_candidate_cycles": t10_schedule["candidate_cycles"],
        "T11_candidate_cycles": t11_schedule["candidate_cycles"],
        "T00_cycles": t00_i,
        "T10_cycles": t10_i,
        "T01_cycles": t01_i,
        "T11_cycles": t11_i,
        "naive_cycles": t00_i,
        "optimized_cycles": t11_i,
        "theory_optimized_cycles": theory_i,
        "architecture_upper_cycles": theory_i,
        "architecture_lower_bound_cycles": theory_i,
        "theory_naive_cycles": t00_i,
        "theory_intra_floor_cycles": max(compute_ideal_cycles, modeled_hbm_cycles, optimized_noc),
        "theory_core_floor_cycles": max(compute_ideal_cycles, modeled_hbm_cycles, optimized_noc, post_gemm_vector_cycles),
        "theory_resource_floor_cycles": resource_floor_cycles,
        "theory_precedence_floor_cycles": precedence_floor_cycles,
        "naive_time": t00_i / CLOCK_HZ,
        "optimized_time": t11_i / CLOCK_HZ,
        "theory_time": theory_i / CLOCK_HZ,
        "throughput_scope": "focus_ep_group",
        "naive_tflops": logical_flops / (t00_i / CLOCK_HZ) / 1e12,
        "optimized_tflops": logical_flops / (t11_i / CLOCK_HZ) / 1e12,
        "scenario_naive_tflops": (
            logical_flops * len(physical.groups) / (t00_i / CLOCK_HZ) / 1e12
        ),
        "scenario_optimized_tflops": (
            logical_flops * len(physical.groups) / (t11_i / CLOCK_HZ) / 1e12
        ),
        "inter_speedup_without_intra": t00_i / t10_i,
        "inter_speedup_with_intra": t01_i / t11_i,
        "intra_speedup_without_inter": t00_i / t01_i,
        "intra_speedup_with_inter": t10_i / t11_i,
        "total_speedup": actual_speedup,
        "actual_speedup": actual_speedup,
        "synergy": t10_i * t01_i / (t00_i * t11_i),
        "theory_speedup": theory_speedup,
        "theory_attainment_rate": attainment,
        "algorithmic_comp_cycles": compute_ideal_cycles,
        "algorithmic_comm_cycles": comm_ideal_cycles,
        "algorithmic_naive_cycles": compute_ideal_cycles + comm_ideal_cycles,
        "algorithmic_floor_cycles": max(compute_ideal_cycles, comm_ideal_cycles),
        "algorithmic_theory_speedup": (
            compute_ideal_cycles + comm_ideal_cycles
        ) / max(compute_ideal_cycles, comm_ideal_cycles),
        "theory_floor_correction": "resource_and_precedence_max",
        "network_scenario": network_scenario,
        "physical_model_manifest": physical_manifest,
        "hardware_manifest": hardware_manifest,
        "mapping_manifest": mapping_manifest,
        "workload_manifest": workload_manifest,
        "simulation_manifest": simulation_manifest,
        "hardware_digest": _digest(hardware_manifest),
        "mapping_digest": _digest(mapping_manifest),
        "workload_digest": _digest(workload_manifest),
        "simulation_digest": _digest(simulation_manifest),
        "physical_model_digest": _digest(physical_manifest),
        "tool_digest": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "physical_model_tool_digest": hashlib.sha256(
            (Path(__file__).resolve().parent / "physical_model.py").read_bytes()
        ).hexdigest(),
        "inter_schedule_source": "production_moe_swizzle_template",
        "inter_duration_source": "provisional_cost_constants_analytical_scaling",
        "intra_schedule_source": "per_expert_grouped_gemm_pe_pm_pn_pk_search",
        "estimate_source": (
            "analytical_physical_resource_replay"
            if include_hbm else "analytical_physical_resource_replay_hbm_free"
        ),
        "simulator_unit_closure": False,
        "simulator_unit_closure_reason": physical_manifest[
            "simulator_unit_closure_detail"
        ]["reasons"],
        "status": (
            "capacity_infeasible_projection"
            if not bool(hbm["hbm_capacity_feasible"])
            else "estimated_via_operator_specific_dag_replay"
        ),
        "calibration_status": "cycle_accurate_calibration_pending",
        "error": "",
    }


def write_results(records: list[dict[str, object]], output_dir: Path) -> None:
    if not records:
        raise ValueError("cannot write an empty result set")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = list(records[0])
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, tuple, dict)) else value
                for key, value in record.items()
            })


def _output_path(
    base: Path, requested: Path | None, profile: HardwareProfile,
    memory_mode: str, network_scenario: str, selection_count: int,
    multiple_scenarios: bool,
) -> Path:
    mode_dir = "architecture" if memory_mode == "architecture" else "hbm_free_compute_comm"
    if requested is None:
        if network_scenario == "loaded_groups":
            return base / "results" / "loaded_groups" / profile.name.lower() / mode_dir
        return base / "results" / profile.name.lower() / mode_dir
    if selection_count == 1:
        return requested
    if network_scenario == "loaded_groups":
        return requested / "loaded_groups" / profile.name.lower() / mode_dir
    if multiple_scenarios:
        return requested / "isolated_group" / profile.name.lower() / mode_dir
    return requested / profile.name.lower() / mode_dir


def _cases_for_scenario(network_scenario: str) -> tuple[ExperimentCase, ...]:
    cases = tuple(iter_cases())
    if network_scenario == "isolated_group":
        return cases
    if network_scenario == "loaded_groups":
        return tuple(
            case for case in cases
            if case.model.name == "DeepSeek-V3" and case.seq_len == 36864
        )
    raise ValueError(f"unsupported network scenario: {network_scenario}")


def main() -> int:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-mode", choices=("architecture", "hbm-free", "all"),
        default="architecture",
    )
    parser.add_argument(
        "--profile", choices=("h128", "h2000", "all"), default="all",
    )
    parser.add_argument(
        "--network-scenario",
        choices=("isolated_group", "loaded_groups", "all"),
        default="isolated_group",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if max(tile_live_bytes(operator=operator) for operator in OPERATORS) > SRAM_CAPACITY_BYTES:
        raise SystemExit("fixed tile exceeds 3 MiB SRAM")
    profiles = (
        tuple(HARDWARE_PROFILES.values())
        if args.profile == "all" else (HARDWARE_PROFILES[args.profile.upper()],)
    )
    requested_memory_modes = (
        ("architecture", "hbm-free") if args.memory_mode == "all"
        else (args.memory_mode,)
    )
    scenarios = (
        ("isolated_group", "loaded_groups")
        if args.network_scenario == "all" else (args.network_scenario,)
    )
    if scenarios == ("loaded_groups",) and args.memory_mode == "hbm-free":
        parser.error("loaded_groups is an architecture-only sensitivity")
    jobs: list[tuple[str, HardwareProfile, str]] = []
    for network_scenario in scenarios:
        memory_modes = (
            tuple(mode for mode in requested_memory_modes if mode == "architecture")
            if network_scenario == "loaded_groups" else requested_memory_modes
        )
        for profile in profiles:
            for memory_mode in memory_modes:
                jobs.append((network_scenario, profile, memory_mode))
    selection_count = len(jobs)
    outputs: list[dict[str, object]] = []
    for network_scenario, profile, memory_mode in jobs:
        include_hbm = memory_mode == "architecture"
        cases = _cases_for_scenario(network_scenario)
        records = [
            estimate_case(
                case, include_hbm=include_hbm, hardware_profile=profile,
                network_scenario=network_scenario,
            )
            for case in cases
        ]
        expected_cases = 16 if network_scenario == "isolated_group" else 4
        if len(records) != expected_cases:
            raise AssertionError(
                f"expected {expected_cases} {network_scenario} cases, "
                f"got {len(records)}"
            )
        output_dir = _output_path(
            base, args.output_dir, profile, memory_mode, network_scenario,
            selection_count, len(scenarios) > 1,
        )
        write_results(records, output_dir)
        outputs.append({
            "network_scenario": network_scenario,
            "profile": profile.name,
            "memory_mode": records[0]["memory_mode"],
            "cases": len(records),
            "successful": sum(
                record["status"].startswith("estimated") for record in records
            ),
            "source": records[0]["estimate_source"],
            "results": str(output_dir / "results.csv"),
        })
    print(json.dumps({
        "total_cases": sum(int(item["cases"]) for item in outputs),
        "outputs": outputs,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
