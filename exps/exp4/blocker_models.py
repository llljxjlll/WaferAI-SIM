"""Analytical replacements for the four exp4 simulator blockers.

These functions deliberately do not mutate or extend the cycle simulator.
They expose byte-conserving resource ledgers for analytical replay.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

try:
    from .hardware_resources import FREQUENCY_HZ, bandwidth_cycles
except ImportError:  # Allows direct imports when exps/exp4 is put on sys.path.
    from hardware_resources import FREQUENCY_HZ, bandwidth_cycles


DTE_CHANNEL_GBS = 128.0
DTE_CHANNEL_WIDTH_BITS = 2048
D2D_PHY_EDGE_ONE_DIR_GBS = 512.0
HBM_STACK_CAPACITY_BYTES = 16_000_000_000
HBM_STACK_PEAK_GBS = 819.2
HBM_ACTIVITY = 0.90
HBM_STACK_SUSTAINED_GBS = 737.28
HBM_INTERLEAVE_BYTES = 256
DEFAULT_PAYLOAD_BYTES = 16


@dataclass(frozen=True)
class DTETransfer:
    action_id: str
    byte_count: int
    direction: str
    launch_cycles: int = 0
    ready_cycle: int = 0


@dataclass(frozen=True)
class DTEChannelEntry:
    action_id: str
    channel_id: int
    direction: str
    byte_count: int
    start_cycle: int
    data_cycles: int
    launch_cycles: int
    finish_cycle: int


@dataclass(frozen=True)
class DTESchedule:
    channel_count: int
    channel_width_bits: int
    makespan_cycles: int
    channel_finish_cycles: tuple[int, ...]
    ledger: tuple[DTEChannelEntry, ...]
    admitted_bytes: int
    analytical_parallel_channels: bool = True


def schedule_dte(transfers: Iterable[DTETransfer], channel_count: int) -> DTESchedule:
    """Earliest-finish list scheduling over independent physical channels."""

    if channel_count <= 0:
        raise ValueError("channel_count must be positive")
    available = [0] * channel_count
    ledger: list[DTEChannelEntry] = []
    total = 0
    for transfer in transfers:
        if transfer.byte_count < 0 or transfer.launch_cycles < 0 or transfer.ready_cycle < 0:
            raise ValueError("DTE bytes and cycles must be non-negative")
        if transfer.direction not in ("read", "write"):
            raise ValueError("DTE direction must be 'read' or 'write'")
        channel = min(range(channel_count), key=lambda idx: (max(available[idx], transfer.ready_cycle), idx))
        start = max(available[channel], transfer.ready_cycle)
        data_cycles = bandwidth_cycles(transfer.byte_count, DTE_CHANNEL_GBS)
        finish = start + transfer.launch_cycles + data_cycles
        available[channel] = finish
        total += transfer.byte_count
        ledger.append(DTEChannelEntry(
            transfer.action_id, channel, transfer.direction, transfer.byte_count,
            start, data_cycles, transfer.launch_cycles, finish,
        ))
    return DTESchedule(
        channel_count, DTE_CHANNEL_WIDTH_BITS, max(available, default=0),
        tuple(available), tuple(ledger), total,
    )


def d2d_edge_capacity_GBs(d_d2d: int, B_GBs: float) -> float:
    if d_d2d <= 0 or B_GBs <= 0:
        raise ValueError("D2D port count and NoC bandwidth must be positive")
    return min(d_d2d * B_GBs, D2D_PHY_EDGE_ONE_DIR_GBS)


@dataclass(frozen=True)
class D2DEdgeService:
    edge_id: str
    byte_count: int
    logical_port_count: int
    capacity_GBs: float
    bulk_cycles: int
    hop_latency_cycles: int
    busy_cycles: int
    cut_utilization: float


def service_d2d_edge(edge_id: str, byte_count: int, d_d2d: int, B_GBs: float,
                     *, hop_latency_cycles: int = 0,
                     observation_cycles: int | None = None) -> D2DEdgeService:
    """Service one directed physical edge shared by all logical edge ports."""

    if byte_count < 0 or hop_latency_cycles < 0:
        raise ValueError("D2D bytes and latency must be non-negative")
    capacity = d2d_edge_capacity_GBs(d_d2d, B_GBs)
    bulk = bandwidth_cycles(byte_count, capacity)
    busy = (hop_latency_cycles + bulk) if byte_count else 0
    denominator = observation_cycles if observation_cycles is not None else busy
    utilization = 0.0 if denominator == 0 else bulk / denominator
    return D2DEdgeService(
        edge_id, byte_count, d_d2d, capacity, bulk,
        hop_latency_cycles if byte_count else 0, busy, utilization,
    )


@dataclass(frozen=True)
class StripeAllocation:
    flow_id: str
    byte_count: int
    stripe_count: int
    port_bytes: tuple[tuple[int, int], ...]
    payload_bytes: int
    imbalance: float


def stripe_flow(flow_id: str, byte_count: int, port_ids: Sequence[int], *,
                payload_bytes: int = DEFAULT_PAYLOAD_BYTES,
                max_stripes: int | None = None) -> StripeAllocation:
    """Deterministically stripe a flow over any number of nonempty ports.

    A stable SHA-256 rotation chooses the first port.  Bytes are then split into
    near-equal stripe streams (at most one byte apart), each of which is
    packetized independently with the frozen payload size.  This is equivalent
    to round-robin fragment issue while permitting a short tail on every stripe,
    and satisfies the plan's strict imbalance bound.
    """

    if byte_count < 0 or payload_bytes <= 0 or not port_ids:
        raise ValueError("invalid byte count, payload size, or empty port set")
    if len(set(port_ids)) != len(port_ids):
        raise ValueError("port_ids must be unique")
    limit = len(port_ids) if max_stripes is None else min(len(port_ids), max_stripes)
    if limit <= 0:
        raise ValueError("max_stripes must be positive")
    fragment_count = math.ceil(byte_count / payload_bytes) if byte_count else 0
    stripe_count = min(limit, fragment_count)
    if stripe_count == 0:
        return StripeAllocation(flow_id, 0, 0, (), payload_bytes, 0.0)

    rotation = int.from_bytes(hashlib.sha256(flow_id.encode()).digest()[:8], "big") % len(port_ids)
    rotated = tuple(port_ids[rotation:]) + tuple(port_ids[:rotation])
    selected_ports = rotated[:stripe_count]
    quotient, remainder = divmod(byte_count, stripe_count)
    totals = {
        port: quotient + (1 if index < remainder else 0)
        for index, port in enumerate(selected_ports)
    }
    port_bytes = tuple((port, totals[port]) for port in selected_ports)
    imbalance = max(totals.values()) / (byte_count / stripe_count)
    return StripeAllocation(flow_id, byte_count, stripe_count, port_bytes, payload_bytes, imbalance)


@dataclass(frozen=True)
class HBMStack:
    stack_id: int
    edge: str
    capacity_bytes: int
    peak_GBs: float
    sustained_GBs: float
    port_ids: tuple[int, ...]
    port_positions: tuple[int, ...]
    address_interleave: tuple[int, int, int]  # stripe bytes, stack count, residue


@dataclass(frozen=True)
class HBM3Topology:
    stacks: tuple[HBMStack, ...]
    edge_capacity_GBs: tuple[tuple[str, float], ...]
    total_capacity_bytes: int
    noncontiguous_port_binding: bool = True


def build_hbm3_topology(candidate: object) -> HBM3Topology:
    """Build HBM3 resources and explicit clustered port-position lists."""

    e_h = int(getattr(candidate, "e_H"))
    m = int(getattr(candidate, "m"))
    t_hbm = int(getattr(candidate, "t_hbm"))
    b_gbs = float(getattr(candidate, "B_GBs"))
    positions = tuple(int(p) for p in getattr(candidate, "hbm_port_positions"))
    if len(positions) != t_hbm:
        raise ValueError("candidate HBM position count does not equal t_hbm")
    edges = tuple(getattr(candidate, "hbm_edges"))
    if len(edges) != e_h:
        raise ValueError("candidate HBM edge count does not equal e_H")
    stack_count = e_h * m
    stacks: list[HBMStack] = []
    edge_caps: list[tuple[str, float]] = []
    for edge_index, edge in enumerate(edges):
        port_ids = tuple(edge_index * t_hbm + index for index in range(t_hbm))
        edge_caps.append((edge, min(m * HBM_STACK_SUSTAINED_GBS, t_hbm * b_gbs)))
        for local_stack in range(m):
            stack_id = edge_index * m + local_stack
            stacks.append(HBMStack(
                stack_id, edge, HBM_STACK_CAPACITY_BYTES, HBM_STACK_PEAK_GBS,
                HBM_STACK_SUSTAINED_GBS, port_ids, positions,
                (HBM_INTERLEAVE_BYTES, stack_count, stack_id),
            ))
    return HBM3Topology(tuple(stacks), tuple(edge_caps), stack_count * HBM_STACK_CAPACITY_BYTES)


def _prefix_interleaved_bytes(end_address: int, stack_count: int,
                              stripe_bytes: int = HBM_INTERLEAVE_BYTES) -> tuple[int, ...]:
    """Bytes in ``[0,end_address)`` assigned to every interleaved stack."""

    if end_address < 0 or stack_count <= 0 or stripe_bytes <= 0:
        raise ValueError("invalid address/interleave parameters")
    full_stripes, tail = divmod(end_address, stripe_bytes)
    rounds, extra = divmod(full_stripes, stack_count)
    values = [rounds * stripe_bytes for _ in range(stack_count)]
    for stack in range(extra):
        values[stack] += stripe_bytes
    if tail:
        values[extra] += tail
    return tuple(values)


def interleaved_stack_bytes(start_address: int, byte_count: int, stack_count: int,
                            *, stripe_bytes: int = HBM_INTERLEAVE_BYTES) -> tuple[int, ...]:
    """Exact 256 B stack-interleaved byte distribution for a contiguous range."""

    if start_address < 0 or byte_count < 0:
        raise ValueError("address and byte_count must be non-negative")
    before = _prefix_interleaved_bytes(start_address, stack_count, stripe_bytes)
    after = _prefix_interleaved_bytes(start_address + byte_count, stack_count, stripe_bytes)
    result = tuple(high - low for low, high in zip(before, after))
    if sum(result) != byte_count:
        raise AssertionError("HBM interleave failed byte conservation")
    return result


@dataclass(frozen=True)
class HBMService:
    byte_count: int
    stack_bytes: tuple[int, ...]
    edge_bytes: tuple[tuple[str, int], ...]
    stack_service_cycles: tuple[int, ...]
    edge_service_cycles: tuple[tuple[str, int], ...]
    noc_path_service_cycles: int
    first_byte_latency_cycles: int
    service_cycles: int


def service_hbm3(candidate: object, start_address: int, byte_count: int, *,
                 first_byte_latency_cycles: int = 0,
                 noc_path_service_cycles: int = 0) -> HBMService:
    topology = build_hbm3_topology(candidate)
    if first_byte_latency_cycles < 0 or noc_path_service_cycles < 0:
        raise ValueError("HBM latency and NoC service must be non-negative")
    stack_bytes = interleaved_stack_bytes(start_address, byte_count, len(topology.stacks))
    stack_cycles = tuple(bandwidth_cycles(count, HBM_STACK_SUSTAINED_GBS) for count in stack_bytes)
    edge_totals = {edge: 0 for edge, _ in topology.edge_capacity_GBs}
    for stack, count in zip(topology.stacks, stack_bytes):
        edge_totals[stack.edge] += count
    cap_by_edge = dict(topology.edge_capacity_GBs)
    edge_cycles = tuple(
        (edge, bandwidth_cycles(edge_totals[edge], cap_by_edge[edge]))
        for edge in edge_totals
    )
    bulk = max((*stack_cycles, *(cycles for _, cycles in edge_cycles), noc_path_service_cycles), default=0)
    total = 0 if byte_count == 0 else bulk + first_byte_latency_cycles
    return HBMService(
        byte_count, stack_bytes, tuple(edge_totals.items()), stack_cycles,
        edge_cycles, noc_path_service_cycles, first_byte_latency_cycles if byte_count else 0, total,
    )


__all__ = [
    "DEFAULT_PAYLOAD_BYTES", "DTEChannelEntry", "DTESchedule", "DTETransfer",
    "D2DEdgeService", "HBM3Topology", "HBMService", "HBMStack", "StripeAllocation",
    "build_hbm3_topology", "d2d_edge_capacity_GBs", "interleaved_stack_bytes",
    "schedule_dte", "service_d2d_edge", "service_hbm3", "stripe_flow",
]
