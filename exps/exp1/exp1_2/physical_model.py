#!/usr/bin/env python3
"""Physical 6x6 wafer model for the exp1-2 analytical replay.

This module deliberately does not claim target-hardware cycle accuracy.  It
provides the physical facts that the analytical runner needs: deterministic
placement, X-first routes, remote-only MoE traffic, four edge HBM stacks, and
collision-free expert-weight addresses.  In particular, congestion is derived
by summing flows on directed links; there is no contention multiplier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias


WAFER_COLUMNS = 6
WAFER_ROWS = 6
WAFER_DIE_COUNT = WAFER_COLUMNS * WAFER_ROWS
EP_SIZE = 4

DTYPE_BYTES = 2
ADDRESS_ALIGNMENT_BYTES = 256
STACK_CAPACITY_BYTES = 16 * 1024**3
STACK_BANDWIDTH_BPS = 256.0e9
D2D_TARGET_BPS = 1.0e12

PLACEMENTS = ("compact", "noncompact")
NETWORK_SCENARIOS = ("isolated_group", "loaded_groups")
MATRIX_KINDS = ("gate", "up", "down")

AssignmentMatrix: TypeAlias = Sequence[Sequence[int]]
AssignmentsByGroup: TypeAlias = AssignmentMatrix | Mapping[str, AssignmentMatrix]


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def align_up(value: int, alignment: int) -> int:
    _require_int(value, "value")
    _require_int(alignment, "alignment", minimum=1)
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True, order=True, slots=True)
class Coordinate:
    """Canonical wafer coordinate, always ordered as ``(x, y)``."""

    x: int
    y: int

    def __post_init__(self) -> None:
        _require_int(self.x, "x")
        _require_int(self.y, "y")
        if self.x >= WAFER_COLUMNS or self.y >= WAFER_ROWS:
            raise ValueError("coordinate lies outside the 6x6 wafer")

    @property
    def die_id(self) -> int:
        return self.y * WAFER_COLUMNS + self.x

    def as_list(self) -> list[int]:
        return [self.x, self.y]


def coordinate_to_die_id(x: int, y: int) -> int:
    return Coordinate(x, y).die_id


def die_id_to_coordinate(die_id: int) -> Coordinate:
    _require_int(die_id, "die_id")
    if die_id >= WAFER_DIE_COUNT:
        raise ValueError("die_id lies outside the 6x6 wafer")
    y, x = divmod(die_id, WAFER_COLUMNS)
    return Coordinate(x, y)


@dataclass(frozen=True, order=True, slots=True)
class DirectedEdge:
    source_die: int
    destination_die: int

    def __post_init__(self) -> None:
        source = die_id_to_coordinate(self.source_die)
        destination = die_id_to_coordinate(self.destination_die)
        if abs(source.x - destination.x) + abs(source.y - destination.y) != 1:
            raise ValueError("a directed edge must join adjacent wafer dies")

    @property
    def edge_id(self) -> str:
        return f"d2d.die{self.source_die}.to.die{self.destination_die}"

    def manifest_dict(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "source_die": self.source_die,
            "destination_die": self.destination_die,
            "source_coordinate": die_id_to_coordinate(self.source_die).as_list(),
            "destination_coordinate": die_id_to_coordinate(
                self.destination_die
            ).as_list(),
        }


def x_first_directed_edges(
    source_die: int, destination_die: int
) -> tuple[DirectedEdge, ...]:
    """Return the exact X-then-Y route used by backend-v1."""

    source = die_id_to_coordinate(source_die)
    destination = die_id_to_coordinate(destination_die)
    x, y = source.x, source.y
    edges: list[DirectedEdge] = []
    while x != destination.x:
        next_x = x + (1 if destination.x > x else -1)
        edges.append(
            DirectedEdge(
                coordinate_to_die_id(x, y),
                coordinate_to_die_id(next_x, y),
            )
        )
        x = next_x
    while y != destination.y:
        next_y = y + (1 if destination.y > y else -1)
        edges.append(
            DirectedEdge(
                coordinate_to_die_id(x, y),
                coordinate_to_die_id(x, next_y),
            )
        )
        y = next_y
    return tuple(edges)


@dataclass(frozen=True, slots=True)
class PhysicalGroup:
    group_id: str
    placement: str
    grid_x: int
    grid_y: int
    rank_to_die: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.placement not in PLACEMENTS:
            raise ValueError(f"unsupported placement: {self.placement}")
        _require_int(self.grid_x, "grid_x")
        _require_int(self.grid_y, "grid_y")
        if len(self.rank_to_die) != EP_SIZE:
            raise ValueError("an EP group must contain exactly four ranks")
        if len(set(self.rank_to_die)) != EP_SIZE:
            raise ValueError("an EP group cannot place two ranks on one die")
        for die_id in self.rank_to_die:
            die_id_to_coordinate(die_id)

    @property
    def rank_coordinates(self) -> tuple[Coordinate, ...]:
        return tuple(die_id_to_coordinate(die_id) for die_id in self.rank_to_die)

    def manifest_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "placement": self.placement,
            "grid_index": [self.grid_x, self.grid_y],
            "rank_to_die": list(self.rank_to_die),
            "rank_coordinates_xy": [item.as_list() for item in self.rank_coordinates],
        }


def _placement_group(placement: str, grid_x: int, grid_y: int) -> PhysicalGroup:
    if placement not in PLACEMENTS:
        raise ValueError(f"unsupported placement: {placement}")
    for value, name in ((grid_x, "grid_x"), (grid_y, "grid_y")):
        _require_int(value, name)
        if value >= 3:
            raise ValueError(f"{name} must lie in [0,2]")

    if placement == "compact":
        x0, y0 = 2 * grid_x, 2 * grid_y
        coordinates = (
            Coordinate(x0, y0),
            Coordinate(x0 + 1, y0),
            Coordinate(x0, y0 + 1),
            Coordinate(x0 + 1, y0 + 1),
        )
    else:
        x0, y0 = grid_x, grid_y
        coordinates = (
            Coordinate(x0, y0),
            Coordinate(x0 + 3, y0),
            Coordinate(x0, y0 + 3),
            Coordinate(x0 + 3, y0 + 3),
        )
    return PhysicalGroup(
        group_id=f"{placement}.x{grid_x}.y{grid_y}",
        placement=placement,
        grid_x=grid_x,
        grid_y=grid_y,
        rank_to_die=tuple(item.die_id for item in coordinates),
    )


def placement_groups(
    placement: str, network_scenario: str
) -> tuple[PhysicalGroup, ...]:
    """Build the center isolated group or nine explicit loaded groups.

    Both loaded placements are disjoint partitions of all 36 dies.  Compact
    uses nine adjacent 2x2 blocks.  Noncompact uses nine four-corner groups
    with offsets ``(grid_x, grid_y)`` in the lower-left 3x3 sub-grid.
    """

    if network_scenario not in NETWORK_SCENARIOS:
        raise ValueError(f"unsupported network scenario: {network_scenario}")
    if network_scenario == "isolated_group":
        return (_placement_group(placement, 1, 1),)
    groups = tuple(
        _placement_group(placement, grid_x, grid_y)
        for grid_y in range(3)
        for grid_x in range(3)
    )
    occupied = [die_id for group in groups for die_id in group.rank_to_die]
    if len(occupied) != WAFER_DIE_COUNT or set(occupied) != set(
        range(WAFER_DIE_COUNT)
    ):
        raise AssertionError("loaded placement must partition the complete wafer")
    return groups


@dataclass(frozen=True, slots=True)
class HbmStack:
    stack_id: int
    home_die: int
    side: str
    capacity_bytes: int
    address_base: int
    address_size: int
    bandwidth_Bps: float
    alignment_bytes: int = ADDRESS_ALIGNMENT_BYTES

    def __post_init__(self) -> None:
        _require_int(self.stack_id, "stack_id")
        home = die_id_to_coordinate(self.home_die)
        if self.side not in ("N", "S"):
            raise ValueError("the exp1 wafer places stacks only on N/S edges")
        if self.side == "S" and home.y != 0:
            raise ValueError("a south stack must be homed on the south wafer row")
        if self.side == "N" and home.y != WAFER_ROWS - 1:
            raise ValueError("a north stack must be homed on the north wafer row")
        _require_int(self.capacity_bytes, "capacity_bytes", minimum=1)
        _require_int(self.address_base, "address_base")
        _require_int(self.address_size, "address_size", minimum=1)
        _require_int(self.alignment_bytes, "alignment_bytes", minimum=1)
        if self.address_size != self.capacity_bytes:
            raise ValueError("stack address range must equal exposed capacity")
        if self.address_base % self.alignment_bytes:
            raise ValueError("stack address base is not aligned")
        if self.bandwidth_Bps <= 0:
            raise ValueError("stack bandwidth must be positive")

    @property
    def coordinate(self) -> Coordinate:
        return die_id_to_coordinate(self.home_die)

    @property
    def address_end(self) -> int:
        return self.address_base + self.address_size

    def manifest_dict(self) -> dict[str, object]:
        return {
            "stack_id": self.stack_id,
            "home_die": self.home_die,
            "home_coordinate_xy": self.coordinate.as_list(),
            "side": self.side,
            "capacity_bytes": self.capacity_bytes,
            "address_range": [self.address_base, self.address_end],
            "address_end_exclusive": True,
            "alignment_bytes": self.alignment_bytes,
            "bandwidth_Bps": self.bandwidth_Bps,
        }


def default_hbm_stacks() -> tuple[HbmStack, ...]:
    homes = (
        (Coordinate(1, 0), "S"),
        (Coordinate(4, 0), "S"),
        (Coordinate(1, 5), "N"),
        (Coordinate(4, 5), "N"),
    )
    stacks = tuple(
        HbmStack(
            stack_id=stack_id,
            home_die=coordinate.die_id,
            side=side,
            capacity_bytes=STACK_CAPACITY_BYTES,
            address_base=stack_id * STACK_CAPACITY_BYTES,
            address_size=STACK_CAPACITY_BYTES,
            bandwidth_Bps=STACK_BANDWIDTH_BPS,
        )
        for stack_id, (coordinate, side) in enumerate(homes)
    )
    ranges = sorted((stack.address_base, stack.address_end) for stack in stacks)
    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        raise AssertionError("default HBM address ranges overlap")
    return stacks


@dataclass(frozen=True, slots=True)
class WeightBinding:
    expert_id: int
    home_rank: int
    matrix: str
    stack_id: int
    address: int
    size_bytes: int

    def manifest_dict(self) -> dict[str, object]:
        return {
            "expert_id": self.expert_id,
            "home_rank": self.home_rank,
            "matrix": self.matrix,
            "stack_id": self.stack_id,
            "address": self.address,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class StackCapacityAudit:
    stack_id: int
    capacity_bytes: int
    payload_bytes: int
    allocated_span_bytes: int
    overflow_bytes: int

    @property
    def feasible(self) -> bool:
        return self.overflow_bytes == 0

    def manifest_dict(self) -> dict[str, object]:
        return {
            "stack_id": self.stack_id,
            "capacity_bytes": self.capacity_bytes,
            "payload_bytes": self.payload_bytes,
            "allocated_span_bytes": self.allocated_span_bytes,
            "remaining_bytes": max(0, self.capacity_bytes - self.allocated_span_bytes),
            "overflow_bytes": self.overflow_bytes,
            "utilization": self.allocated_span_bytes / self.capacity_bytes,
            "feasible": self.feasible,
        }


@dataclass(frozen=True, slots=True)
class WeightAllocation:
    bindings: tuple[WeightBinding, ...]
    stack_audits: tuple[StackCapacityAudit, ...]

    @property
    def feasible(self) -> bool:
        return all(item.feasible for item in self.stack_audits)

    @property
    def total_weight_bytes(self) -> int:
        return sum(item.size_bytes for item in self.bindings)

    def manifest_dict(self) -> dict[str, object]:
        return {
            "matrix_kinds": list(MATRIX_KINDS),
            "total_weight_bytes": self.total_weight_bytes,
            "capacity_feasible": self.feasible,
            "status": (
                "capacity_feasible"
                if self.feasible
                else "capacity_infeasible_projection"
            ),
            "per_stack": [item.manifest_dict() for item in self.stack_audits],
            "bindings": [item.manifest_dict() for item in self.bindings],
        }


class CapacityError(ValueError):
    def __init__(self, allocation: WeightAllocation):
        self.allocation = allocation
        super().__init__("expert gate/up/down weights exceed the four-stack capacity")


def _validate_expert_homes(expert_home_ranks: Sequence[int]) -> tuple[int, ...]:
    if not expert_home_ranks:
        raise ValueError("expert_home_ranks must not be empty")
    homes = tuple(expert_home_ranks)
    for expert, home in enumerate(homes):
        _require_int(home, f"expert_home_ranks[{expert}]")
        if home >= EP_SIZE:
            raise ValueError("expert home rank must lie in [0,3]")
    return homes


def allocate_expert_weights(
    hidden_size: int,
    intermediate_size: int,
    expert_home_ranks: Sequence[int],
    *,
    dtype_bytes: int = DTYPE_BYTES,
    stacks: Sequence[HbmStack] | None = None,
    strict_capacity: bool = True,
) -> WeightAllocation:
    """Allocate each expert's gate/up/down roots on its home-rank stack."""

    _require_int(hidden_size, "hidden_size", minimum=1)
    _require_int(intermediate_size, "intermediate_size", minimum=1)
    _require_int(dtype_bytes, "dtype_bytes", minimum=1)
    homes = _validate_expert_homes(expert_home_ranks)
    stack_list = tuple(default_hbm_stacks() if stacks is None else stacks)
    stack_by_id = {stack.stack_id: stack for stack in stack_list}
    if set(stack_by_id) != set(range(EP_SIZE)) or len(stack_list) != EP_SIZE:
        raise ValueError("weight allocation requires exactly stacks 0..3")

    matrix_bytes = hidden_size * intermediate_size * dtype_bytes
    bindings: list[WeightBinding] = []
    audits: list[StackCapacityAudit] = []
    for stack_id in range(EP_SIZE):
        stack = stack_by_id[stack_id]
        cursor = stack.address_base
        payload_bytes = 0
        for expert, home_rank in enumerate(homes):
            if home_rank != stack_id:
                continue
            for matrix in MATRIX_KINDS:
                cursor = align_up(cursor, stack.alignment_bytes)
                bindings.append(
                    WeightBinding(
                        expert_id=expert,
                        home_rank=home_rank,
                        matrix=matrix,
                        stack_id=stack_id,
                        address=cursor,
                        size_bytes=matrix_bytes,
                    )
                )
                cursor += matrix_bytes
                payload_bytes += matrix_bytes
        span = cursor - stack.address_base
        audits.append(
            StackCapacityAudit(
                stack_id=stack_id,
                capacity_bytes=stack.capacity_bytes,
                payload_bytes=payload_bytes,
                allocated_span_bytes=span,
                overflow_bytes=max(0, span - stack.capacity_bytes),
            )
        )
    allocation = WeightAllocation(tuple(bindings), tuple(audits))
    if strict_capacity and not allocation.feasible:
        raise CapacityError(allocation)
    return allocation


def balanced_expert_home_ranks(expert_count: int) -> tuple[int, ...]:
    _require_int(expert_count, "expert_count", minimum=1)
    return tuple(expert % EP_SIZE for expert in range(expert_count))


def _normalize_assignment_matrix(
    assignments: AssignmentMatrix,
    expert_count: int,
    *,
    path: str,
) -> tuple[tuple[int, ...], ...]:
    if len(assignments) != EP_SIZE:
        raise ValueError(f"{path} must contain four source-rank rows")
    rows: list[tuple[int, ...]] = []
    for source_rank, row in enumerate(assignments):
        if len(row) != expert_count:
            raise ValueError(
                f"{path}[{source_rank}] must contain {expert_count} experts"
            )
        values = tuple(row)
        for expert, count in enumerate(values):
            _require_int(count, f"{path}[{source_rank}][{expert}]")
        rows.append(values)
    return tuple(rows)


def _assignments_for_groups(
    assignments: AssignmentsByGroup,
    groups: Sequence[PhysicalGroup],
    expert_count: int,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    if isinstance(assignments, Mapping):
        group_ids = {group.group_id for group in groups}
        if set(assignments) != group_ids:
            raise ValueError("assignment mapping must exactly cover the physical groups")
        return {
            group.group_id: _normalize_assignment_matrix(
                assignments[group.group_id],
                expert_count,
                path=f"assignments[{group.group_id!r}]",
            )
            for group in groups
        }
    common = _normalize_assignment_matrix(assignments, expert_count, path="assignments")
    return {group.group_id: common for group in groups}


@dataclass(frozen=True, slots=True)
class MoeFlow:
    flow_id: str
    group_id: str
    source_rank: int
    destination_rank: int
    source_die: int
    destination_die: int
    expert_ids: tuple[int, ...]
    remote_assignments: int
    payload_bytes: int
    metadata_bytes: int
    route: tuple[DirectedEdge, ...]

    @property
    def total_bytes(self) -> int:
        return self.payload_bytes + self.metadata_bytes

    def manifest_dict(self) -> dict[str, object]:
        return {
            "flow_id": self.flow_id,
            "group_id": self.group_id,
            "source_rank": self.source_rank,
            "destination_rank": self.destination_rank,
            "source_die": self.source_die,
            "destination_die": self.destination_die,
            "expert_ids": list(self.expert_ids),
            "remote_assignments": self.remote_assignments,
            "payload_bytes": self.payload_bytes,
            "metadata_bytes": self.metadata_bytes,
            "total_bytes": self.total_bytes,
            "route": [edge.edge_id for edge in self.route],
        }


def build_remote_moe_flows(
    groups: Sequence[PhysicalGroup],
    assignments: AssignmentsByGroup,
    expert_home_ranks: Sequence[int],
    hidden_size: int,
    *,
    dtype_bytes: int = DTYPE_BYTES,
    metadata_bytes_per_assignment: int = 0,
) -> tuple[MoeFlow, ...]:
    """Aggregate ``A[source_rank, expert]`` into remote src/dst flows only."""

    _require_int(hidden_size, "hidden_size", minimum=1)
    _require_int(dtype_bytes, "dtype_bytes", minimum=1)
    _require_int(
        metadata_bytes_per_assignment,
        "metadata_bytes_per_assignment",
    )
    homes = _validate_expert_homes(expert_home_ranks)
    matrices = _assignments_for_groups(assignments, groups, len(homes))
    flows: list[MoeFlow] = []
    for group in groups:
        matrix = matrices[group.group_id]
        for source_rank in range(EP_SIZE):
            for destination_rank in range(EP_SIZE):
                if source_rank == destination_rank:
                    continue
                expert_ids = tuple(
                    expert
                    for expert, home in enumerate(homes)
                    if home == destination_rank and matrix[source_rank][expert] > 0
                )
                remote_assignments = sum(
                    matrix[source_rank][expert] for expert in expert_ids
                )
                if remote_assignments == 0:
                    continue
                source_die = group.rank_to_die[source_rank]
                destination_die = group.rank_to_die[destination_rank]
                flows.append(
                    MoeFlow(
                        flow_id=(
                            f"moe.{group.group_id}.r{source_rank}.to.r"
                            f"{destination_rank}"
                        ),
                        group_id=group.group_id,
                        source_rank=source_rank,
                        destination_rank=destination_rank,
                        source_die=source_die,
                        destination_die=destination_die,
                        expert_ids=expert_ids,
                        remote_assignments=remote_assignments,
                        payload_bytes=remote_assignments * hidden_size * dtype_bytes,
                        metadata_bytes=(
                            remote_assignments * metadata_bytes_per_assignment
                        ),
                        route=x_first_directed_edges(source_die, destination_die),
                    )
                )
    return tuple(flows)


@dataclass(frozen=True, slots=True)
class DirectedEdgeLoad:
    edge: DirectedEdge
    moe_payload_bytes: int
    moe_metadata_bytes: int
    flow_count: int

    @property
    def total_bytes(self) -> int:
        return self.moe_payload_bytes + self.moe_metadata_bytes

    def manifest_dict(self) -> dict[str, object]:
        return {
            **self.edge.manifest_dict(),
            "moe_payload_bytes": self.moe_payload_bytes,
            "moe_metadata_bytes": self.moe_metadata_bytes,
            "total_bytes": self.total_bytes,
            "flow_count": self.flow_count,
            "capacity_Bps": D2D_TARGET_BPS,
        }


def aggregate_directed_edge_loads(
    flows: Sequence[MoeFlow],
) -> tuple[DirectedEdgeLoad, ...]:
    """Sum actual flow incidence on every directed edge."""

    payload: dict[DirectedEdge, int] = {}
    metadata: dict[DirectedEdge, int] = {}
    counts: dict[DirectedEdge, int] = {}
    for flow in flows:
        for edge in flow.route:
            payload[edge] = payload.get(edge, 0) + flow.payload_bytes
            metadata[edge] = metadata.get(edge, 0) + flow.metadata_bytes
            counts[edge] = counts.get(edge, 0) + 1
    return tuple(
        DirectedEdgeLoad(edge, payload[edge], metadata[edge], counts[edge])
        for edge in sorted(payload)
    )


@dataclass(frozen=True, slots=True)
class PhysicalModel:
    placement: str
    network_scenario: str
    groups: tuple[PhysicalGroup, ...]
    stacks: tuple[HbmStack, ...]
    expert_home_ranks: tuple[int, ...]
    flows: tuple[MoeFlow, ...]
    directed_edge_loads: tuple[DirectedEdgeLoad, ...]
    weight_allocation: WeightAllocation
    logical_assignments: int
    remote_assignments: int

    @property
    def remote_payload_bytes(self) -> int:
        return sum(flow.payload_bytes for flow in self.flows)

    def manifest_dict(self) -> dict[str, object]:
        return {
            "schema_version": "exp1_2_physical_model_v1",
            "source": "analytical_physical_resource_replay",
            "wafer": {
                "shape_xy": [WAFER_COLUMNS, WAFER_ROWS],
                "coordinate_order": "x_y",
                "die_id_formula": "die_id=y*6+x",
                "die_count": WAFER_DIE_COUNT,
            },
            "placement": self.placement,
            "network_scenario": self.network_scenario,
            "route_policy": "x_first",
            "groups": [group.manifest_dict() for group in self.groups],
            "hbm_stacks": [stack.manifest_dict() for stack in self.stacks],
            "expert_home_ranks": list(self.expert_home_ranks),
            "weight_allocation": self.weight_allocation.manifest_dict(),
            "moe_flows": [flow.manifest_dict() for flow in self.flows],
            "directed_edge_loads": [
                item.manifest_dict() for item in self.directed_edge_loads
            ],
            "conservation": {
                "logical_assignments": self.logical_assignments,
                "remote_assignments": self.remote_assignments,
                "remote_flow_assignments": sum(
                    flow.remote_assignments for flow in self.flows
                ),
                "remote_payload_bytes": self.remote_payload_bytes,
            },
            "contention_model": (
                "derived_from_explicit_group_flows_and_directed_edge_incidence"
            ),
            "simulator_unit_closure": False,
            "simulator_unit_closure_detail": {
                "target_d2d_Bps_per_direction": D2D_TARGET_BPS,
                "simulator_cycle_ns": 2,
                "simulator_payload_bytes_per_flit": 16,
                "single_lane_representable_Bps": 8.0e9,
                "reasons": [
                    "cycle D2D rate is capped at one flit per lane per cycle",
                    "frontend PhysicalFabric v1 requires local HBM on every die",
                    "the four-edge-stack 6x6 target therefore lacks a closed current runtime binding",
                ],
                "allowed_use": (
                    "cycle-accurate motifs calibrate structure/setup only; target "
                    "absolute cycles remain analytical extrapolation"
                ),
            },
        }


def build_physical_model(
    *,
    placement: str,
    network_scenario: str,
    assignments: AssignmentsByGroup,
    expert_home_ranks: Sequence[int],
    hidden_size: int,
    intermediate_size: int,
    dtype_bytes: int = DTYPE_BYTES,
    metadata_bytes_per_assignment: int = 0,
    strict_capacity: bool = True,
) -> PhysicalModel:
    groups = placement_groups(placement, network_scenario)
    stacks = default_hbm_stacks()
    homes = _validate_expert_homes(expert_home_ranks)
    matrices = _assignments_for_groups(assignments, groups, len(homes))
    flows = build_remote_moe_flows(
        groups,
        matrices,
        homes,
        hidden_size,
        dtype_bytes=dtype_bytes,
        metadata_bytes_per_assignment=metadata_bytes_per_assignment,
    )
    logical_assignments = sum(
        count
        for group in groups
        for row in matrices[group.group_id]
        for count in row
    )
    remote_assignments = sum(
        matrices[group.group_id][source_rank][expert]
        for group in groups
        for source_rank in range(EP_SIZE)
        for expert, home in enumerate(homes)
        if source_rank != home
    )
    model = PhysicalModel(
        placement=placement,
        network_scenario=network_scenario,
        groups=groups,
        stacks=stacks,
        expert_home_ranks=homes,
        flows=flows,
        directed_edge_loads=aggregate_directed_edge_loads(flows),
        weight_allocation=allocate_expert_weights(
            hidden_size,
            intermediate_size,
            homes,
            dtype_bytes=dtype_bytes,
            stacks=stacks,
            strict_capacity=strict_capacity,
        ),
        logical_assignments=logical_assignments,
        remote_assignments=remote_assignments,
    )
    if sum(flow.remote_assignments for flow in flows) != remote_assignments:
        raise AssertionError("remote assignment conservation failed")
    return model


__all__ = [
    "ADDRESS_ALIGNMENT_BYTES",
    "CapacityError",
    "Coordinate",
    "D2D_TARGET_BPS",
    "DirectedEdge",
    "DirectedEdgeLoad",
    "DTYPE_BYTES",
    "EP_SIZE",
    "HbmStack",
    "MATRIX_KINDS",
    "MoeFlow",
    "NETWORK_SCENARIOS",
    "PLACEMENTS",
    "PhysicalGroup",
    "PhysicalModel",
    "STACK_BANDWIDTH_BPS",
    "STACK_CAPACITY_BYTES",
    "StackCapacityAudit",
    "WAFER_COLUMNS",
    "WAFER_DIE_COUNT",
    "WAFER_ROWS",
    "WeightAllocation",
    "WeightBinding",
    "aggregate_directed_edge_loads",
    "align_up",
    "allocate_expert_weights",
    "balanced_expert_home_ranks",
    "build_physical_model",
    "build_remote_moe_flows",
    "coordinate_to_die_id",
    "default_hbm_stacks",
    "die_id_to_coordinate",
    "placement_groups",
    "x_first_directed_edges",
]
