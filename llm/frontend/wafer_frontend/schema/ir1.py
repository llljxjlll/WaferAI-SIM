"""Versioned physical graph (IR-1) with a self-contained immutable value table."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import (
    DType,
    MeshAxisName,
    ProfileKey,
    TensorValue,
    stable_artifact_id,
    validate_nonempty,
    validate_uint64,
    validate_unique_ids,
)
from .ir0 import (
    AttentionWorkload,
    CollectiveWorkload,
    CrossEntropyBackwardWorkload,
    CrossEntropyForwardWorkload,
    EdgeKind,
    FusionCandidate,
    FusionSemanticContract,
    FusionImpl,
    GemmWorkload,
    GreedySampleWorkload,
    GraphEdge,
    InstanceProfileBinding,
    NodeProfileBinding,
    LogicalRole,
    NodeEffects,
    NodeMath,
    NodeWorkload,
    OpKind,
    OpPhase,
    P2PByteWorkload,
    EmbeddingWorkload,
    ResidualWorkload,
    RopeQkWorkload,
    RmsNormWorkload,
    SgdUpdateWorkload,
    StateAccess,
    StateAccessMode,
    SwiGluWorkload,
    state_access_tensor_view,
    validate_value_graph,
)
from .persistent_state import (
    PersistentStateAccess,
    PersistentStateManifest,
)


IR1_SCHEMA_VERSION = "wafer_frontend.ir1/v1alpha14"
CROSS_GROUP_ROUTE_SCHEMA_VERSION = (
    "wafer_frontend.cross_group_route/v1alpha1"
)
PROGRAM_CORE_ID_MAX = (1 << 16) - 1


class Direction(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


class RoutingMode(str, Enum):
    BACKEND_XY_V1 = "backend_xy_v1"


class MemoryInitiator(str, Enum):
    COMPUTE = "compute"
    DTE = "dte"
    LSU = "lsu"
    NOC_RX = "noc_rx"
    LEGACY = "legacy"


class SramAllocator(str, Enum):
    FIXED = "fixed"
    BLOCK = "block"


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _opposite(direction: Direction) -> Direction:
    return {
        Direction.NORTH: Direction.SOUTH,
        Direction.SOUTH: Direction.NORTH,
        Direction.EAST: Direction.WEST,
        Direction.WEST: Direction.EAST,
    }[direction]


def _direction_between(source: tuple[int, int], destination: tuple[int, int]) -> Direction:
    delta = (destination[0] - source[0], destination[1] - source[1])
    directions = {
        (0, 1): Direction.NORTH,
        (0, -1): Direction.SOUTH,
        (1, 0): Direction.EAST,
        (-1, 0): Direction.WEST,
    }
    try:
        return directions[delta]
    except KeyError as error:
        raise SchemaError("link endpoints must be adjacent dies") from error


def _backend_xy_die_path(
    source: tuple[int, int], destination: tuple[int, int], die_grid: tuple[int, int]
) -> tuple[int, ...]:
    x, y = source
    destination_x, destination_y = destination
    coords = [(x, y)]
    while x != destination_x:
        x += 1 if destination_x > x else -1
        coords.append((x, y))
    while y != destination_y:
        y += 1 if destination_y > y else -1
        coords.append((x, y))
    return tuple(coord_y * die_grid[0] + coord_x for coord_x, coord_y in coords)


@dataclass(frozen=True, slots=True)
class SramRegionSpec:
    id: str
    name: str
    base_bytes: int
    size_bytes: int
    allocator: SramAllocator
    spillable: bool
    access: tuple[MemoryInitiator, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.name, f"{path}.name")
        if len(self.name.encode("utf-8")) > 64:
            raise SchemaError("UTF-8 encoding must not exceed 64 bytes", path=f"{path}.name")
        validate_uint64(self.base_bytes, f"{path}.base_bytes")
        validate_uint64(self.size_bytes, f"{path}.size_bytes")
        if self.size_bytes == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.size_bytes")
        if not self.access:
            raise SchemaError("must contain at least one initiator", path=f"{path}.access")
        if len(set(self.access)) != len(self.access):
            raise SchemaError("contains a duplicate initiator", path=f"{path}.access")


@dataclass(frozen=True, slots=True)
class SramProfile:
    id: str
    capacity_bytes: int
    allocation_alignment_bytes: int
    bank_count: int
    bank_interleave_bytes: int
    real_data_path: bool
    manual_regions: bool
    manual_memory_schedule: bool
    regions: tuple[SramRegionSpec, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        for field_name in (
            "capacity_bytes",
            "allocation_alignment_bytes",
            "bank_count",
            "bank_interleave_bytes",
        ):
            value = getattr(self, field_name)
            validate_uint64(value, f"{path}.{field_name}")
            if value == 0:
                raise SchemaError("must be greater than zero", path=f"{path}.{field_name}")
        if not _is_power_of_two(self.allocation_alignment_bytes):
            raise SchemaError("must be a power of two", path=f"{path}.allocation_alignment_bytes")
        if not _is_power_of_two(self.bank_count):
            raise SchemaError("must be a power of two", path=f"{path}.bank_count")
        if not _is_power_of_two(self.bank_interleave_bytes):
            raise SchemaError("must be a power of two", path=f"{path}.bank_interleave_bytes")
        if not self.regions:
            raise SchemaError("must contain at least one named region", path=f"{path}.regions")
        for capability in ("real_data_path", "manual_regions", "manual_memory_schedule"):
            if not getattr(self, capability):
                raise SchemaError(
                    "must be true for strict Program backend-v1",
                    path=f"{path}.{capability}",
                )
        validate_unique_ids(self.regions, f"{path}.regions")
        names: set[str] = set()
        intervals: list[tuple[int, int, int]] = []
        for index, region in enumerate(self.regions):
            region_path = f"{path}.regions[{index}]"
            region.validate(region_path)
            if region.name in names:
                raise SchemaError(f"duplicate region name {region.name!r}", path=f"{region_path}.name")
            names.add(region.name)
            if region.base_bytes % self.allocation_alignment_bytes:
                raise SchemaError("must satisfy allocation alignment", path=f"{region_path}.base_bytes")
            if region.size_bytes % self.allocation_alignment_bytes:
                raise SchemaError("must satisfy allocation alignment", path=f"{region_path}.size_bytes")
            end = region.base_bytes + region.size_bytes
            if end > self.capacity_bytes:
                raise SchemaError("region exceeds SRAM capacity", path=region_path)
            intervals.append((region.base_bytes, end, index))
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise SchemaError(
                    f"overlaps regions[{previous[2]}]",
                    path=f"{path}.regions[{current[2]}]",
                )


@dataclass(frozen=True, slots=True)
class CoreSpec:
    id: str
    local_core_id: int
    runtime_core_id: int
    noc_coord: tuple[int, int]
    sram_profile_ref: str

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_uint64(self.local_core_id, f"{path}.local_core_id")
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        if self.runtime_core_id > PROGRAM_CORE_ID_MAX:
            raise SchemaError("must fit backend-v1 uint16 core id", path=f"{path}.runtime_core_id")
        for index, coordinate in enumerate(self.noc_coord):
            validate_uint64(coordinate, f"{path}.noc_coord[{index}]")
        validate_nonempty(self.sram_profile_ref, f"{path}.sram_profile_ref")


@dataclass(frozen=True, slots=True)
class C2CPort:
    id: str
    runtime_port_id: int
    side: Direction
    direction: Direction
    noc_coord: tuple[int, int]
    egress_resource_id: str
    bytes_per_cycle: int
    buffer_packets: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_uint64(self.runtime_port_id, f"{path}.runtime_port_id")
        for index, coordinate in enumerate(self.noc_coord):
            validate_uint64(coordinate, f"{path}.noc_coord[{index}]")
        validate_nonempty(self.egress_resource_id, f"{path}.egress_resource_id")
        for field_name in ("bytes_per_cycle", "buffer_packets"):
            value = getattr(self, field_name)
            validate_uint64(value, f"{path}.{field_name}")
            if value == 0:
                raise SchemaError("must be greater than zero", path=f"{path}.{field_name}")


@dataclass(frozen=True, slots=True)
class DieSpec:
    id: int
    coord: tuple[int, int]
    noc_grid: tuple[int, int]
    noc_bytes_per_cycle: int
    hbm_bytes_per_cycle: int
    cores: tuple[CoreSpec, ...]
    ports: tuple[C2CPort, ...]

    def validate(self, path: str) -> None:
        validate_uint64(self.id, f"{path}.id")
        for field_name in ("coord", "noc_grid"):
            pair = getattr(self, field_name)
            for index, value in enumerate(pair):
                validate_uint64(value, f"{path}.{field_name}[{index}]")
                if field_name == "noc_grid" and value == 0:
                    raise SchemaError("must be greater than zero", path=f"{path}.{field_name}[{index}]")
        for field_name in ("noc_bytes_per_cycle", "hbm_bytes_per_cycle"):
            value = getattr(self, field_name)
            validate_uint64(value, f"{path}.{field_name}")
            if value == 0:
                raise SchemaError("must be greater than zero", path=f"{path}.{field_name}")
        expected_core_count = self.noc_grid[0] * self.noc_grid[1]
        if len(self.cores) != expected_core_count:
            raise SchemaError(
                f"must describe all {expected_core_count} cores in noc_grid",
                path=f"{path}.cores",
            )
        validate_unique_ids(self.cores, f"{path}.cores")
        local_ids: set[int] = set()
        runtime_ids: set[int] = set()
        core_coords: set[tuple[int, int]] = set()
        for index, core in enumerate(self.cores):
            core_path = f"{path}.cores[{index}]"
            core.validate(core_path)
            if core.local_core_id in local_ids:
                raise SchemaError("duplicate local_core_id", path=f"{core_path}.local_core_id")
            if core.runtime_core_id in runtime_ids:
                raise SchemaError("duplicate runtime_core_id", path=f"{core_path}.runtime_core_id")
            if core.noc_coord in core_coords:
                raise SchemaError("duplicate core noc_coord", path=f"{core_path}.noc_coord")
            if any(core.noc_coord[axis] >= self.noc_grid[axis] for axis in (0, 1)):
                raise SchemaError("core coordinate lies outside noc_grid", path=f"{core_path}.noc_coord")
            expected_local_id = core.noc_coord[1] * self.noc_grid[0] + core.noc_coord[0]
            if core.local_core_id != expected_local_id:
                raise SchemaError(
                    f"backend-v1 row-major local_core_id must be {expected_local_id}",
                    path=f"{core_path}.local_core_id",
                )
            local_ids.add(core.local_core_id)
            runtime_ids.add(core.runtime_core_id)
            core_coords.add(core.noc_coord)
        if local_ids != set(range(expected_core_count)):
            raise SchemaError("local_core_id set must be contiguous", path=f"{path}.cores")
        validate_unique_ids(self.ports, f"{path}.ports")
        runtime_port_ids: set[int] = set()
        directions: set[Direction] = set()
        for index, port in enumerate(self.ports):
            port_path = f"{path}.ports[{index}]"
            port.validate(port_path)
            if port.runtime_port_id in runtime_port_ids:
                raise SchemaError("duplicate runtime_port_id", path=f"{port_path}.runtime_port_id")
            if port.direction in directions:
                raise SchemaError("backend-v1 permits one C2C port per direction", path=f"{port_path}.direction")
            if port.side != port.direction:
                raise SchemaError("backend-v1 requires side to match direction", path=port_path)
            if any(port.noc_coord[axis] >= self.noc_grid[axis] for axis in (0, 1)):
                raise SchemaError("port coordinate lies outside noc_grid", path=f"{port_path}.noc_coord")
            x, y = port.noc_coord
            on_edge = {
                Direction.WEST: x == 0,
                Direction.EAST: x == self.noc_grid[0] - 1,
                Direction.SOUTH: y == 0,
                Direction.NORTH: y == self.noc_grid[1] - 1,
            }[port.side]
            if not on_edge:
                raise SchemaError("port coordinate does not lie on declared side", path=f"{port_path}.noc_coord")
            runtime_port_ids.add(port.runtime_port_id)
            directions.add(port.direction)


@dataclass(frozen=True, slots=True)
class D2DLink:
    id: str
    source_die: int
    source_port_ref: str
    destination_die: int
    destination_port_ref: str
    bytes_per_cycle: int
    latency_cycles: int
    resource_id: str
    link_group_ref: str | None

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_uint64(self.source_die, f"{path}.source_die")
        validate_uint64(self.destination_die, f"{path}.destination_die")
        if self.source_die == self.destination_die:
            raise SchemaError("link endpoints must differ", path=path)
        validate_nonempty(self.source_port_ref, f"{path}.source_port_ref")
        validate_nonempty(self.destination_port_ref, f"{path}.destination_port_ref")
        for field_name in ("bytes_per_cycle", "latency_cycles"):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.bytes_per_cycle == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.bytes_per_cycle")
        validate_nonempty(self.resource_id, f"{path}.resource_id")
        if self.link_group_ref is not None:
            validate_nonempty(self.link_group_ref, f"{path}.link_group_ref")


@dataclass(frozen=True, slots=True)
class PhysicalFabric:
    routing_mode: RoutingMode
    die_grid: tuple[int, int]
    sram_profiles: tuple[SramProfile, ...]
    dies: tuple[DieSpec, ...]
    links: tuple[D2DLink, ...]

    def validate(self, path: str) -> None:
        if self.routing_mode is not RoutingMode.BACKEND_XY_V1:
            raise SchemaError("unsupported routing mode", path=f"{path}.routing_mode")
        for index, value in enumerate(self.die_grid):
            validate_uint64(value, f"{path}.die_grid[{index}]")
            if value == 0:
                raise SchemaError("must be greater than zero", path=f"{path}.die_grid[{index}]")
        profiles = validate_unique_ids(self.sram_profiles, f"{path}.sram_profiles")
        if not profiles:
            raise SchemaError("must contain at least one per-core SRAM profile", path=f"{path}.sram_profiles")
        for index, profile in enumerate(self.sram_profiles):
            profile.validate(f"{path}.sram_profiles[{index}]")
        expected_die_count = self.die_grid[0] * self.die_grid[1]
        if len(self.dies) != expected_die_count:
            raise SchemaError(
                f"must describe all {expected_die_count} dies in die_grid",
                path=f"{path}.dies",
            )
        die_ids: set[int] = set()
        coords: set[tuple[int, int]] = set()
        runtime_core_ids: set[int] = set()
        common_noc_grid: tuple[int, int] | None = None
        for index, die in enumerate(self.dies):
            die_path = f"{path}.dies[{index}]"
            die.validate(die_path)
            if die.id in die_ids:
                raise SchemaError(f"duplicate die id {die.id}", path=f"{path}.dies[{index}].id")
            if die.coord in coords:
                raise SchemaError(f"duplicate die coordinate {die.coord!r}", path=f"{path}.dies[{index}].coord")
            if any(die.coord[axis] >= self.die_grid[axis] for axis in (0, 1)):
                raise SchemaError("die coordinate lies outside die_grid", path=f"{path}.dies[{index}].coord")
            expected_die_id = die.coord[1] * self.die_grid[0] + die.coord[0]
            if die.id != expected_die_id:
                raise SchemaError(
                    f"backend-v1 row-major die id must be {expected_die_id}",
                    path=f"{die_path}.id",
                )
            if common_noc_grid is None:
                common_noc_grid = die.noc_grid
            elif die.noc_grid != common_noc_grid:
                raise SchemaError("backend-v1 requires one uniform noc_grid", path=f"{die_path}.noc_grid")
            cores_per_die = die.noc_grid[0] * die.noc_grid[1]
            for core_index, core in enumerate(die.cores):
                if core.sram_profile_ref not in profiles:
                    raise SchemaError(
                        "unknown per-core SRAM profile",
                        path=f"{die_path}.cores[{core_index}].sram_profile_ref",
                    )
                expected_runtime_id = die.id * cores_per_die + core.local_core_id
                if core.runtime_core_id != expected_runtime_id:
                    raise SchemaError(
                        f"backend-v1 runtime_core_id must be {expected_runtime_id}",
                        path=f"{die_path}.cores[{core_index}].runtime_core_id",
                    )
                if core.runtime_core_id in runtime_core_ids:
                    raise SchemaError("duplicate global runtime_core_id", path=f"{die_path}.cores[{core_index}].runtime_core_id")
                runtime_core_ids.add(core.runtime_core_id)
            die_ids.add(die.id)
            coords.add(die.coord)
        if die_ids != set(range(expected_die_count)):
            raise SchemaError("die ids must be contiguous row-major ids", path=f"{path}.dies")
        die_index = {die.id: die for die in self.dies}
        port_indexes = {die.id: {port.id: port for port in die.ports} for die in self.dies}
        port_resource_ids: set[str] = set()
        for die_index_value, die in enumerate(self.dies):
            for port_index, port in enumerate(die.ports):
                if port.egress_resource_id in port_resource_ids:
                    raise SchemaError(
                        "duplicate global port egress_resource_id",
                        path=f"{path}.dies[{die_index_value}].ports[{port_index}].egress_resource_id",
                    )
                port_resource_ids.add(port.egress_resource_id)
        validate_unique_ids(self.links, f"{path}.links")
        outgoing_endpoints: set[tuple[int, str]] = set()
        link_resource_ids: set[str] = set()
        cut_resource_ids: set[str] = set()
        for index, link in enumerate(self.links):
            link_path = f"{path}.links[{index}]"
            link.validate(link_path)
            if link.source_die not in die_ids or link.destination_die not in die_ids:
                raise SchemaError("link contains a dangling die reference", path=link_path)
            source_port = port_indexes[link.source_die].get(link.source_port_ref)
            destination_port = port_indexes[link.destination_die].get(link.destination_port_ref)
            if source_port is None or destination_port is None:
                raise SchemaError("link contains a dangling port reference", path=link_path)
            direction = _direction_between(
                die_index[link.source_die].coord, die_index[link.destination_die].coord
            )
            if source_port.direction != direction or destination_port.direction != _opposite(direction):
                raise SchemaError("link ports do not face reciprocal adjacent dies", path=link_path)
            endpoint = (link.source_die, link.source_port_ref)
            if endpoint in outgoing_endpoints:
                raise SchemaError("backend-v1 port has multiple outgoing links", path=f"{link_path}.source_port_ref")
            outgoing_endpoints.add(endpoint)
            if (
                link.resource_id in link_resource_ids
                or link.resource_id in port_resource_ids
                or link.resource_id in cut_resource_ids
            ):
                raise SchemaError("duplicate physical resource_id", path=f"{link_path}.resource_id")
            link_resource_ids.add(link.resource_id)
            if link.link_group_ref is not None and (
                link.link_group_ref in port_resource_ids
                or link.link_group_ref in link_resource_ids
            ):
                raise SchemaError(
                    "link_group_ref must identify a distinct directed cut resource",
                    path=f"{link_path}.link_group_ref",
                )
            if link.link_group_ref is not None:
                cut_resource_ids.add(link.link_group_ref)
        links_by_tuple = {
            (link.source_die, link.source_port_ref, link.destination_die, link.destination_port_ref): link
            for link in self.links
        }
        for index, link in enumerate(self.links):
            reverse_key = (
                link.destination_die,
                link.destination_port_ref,
                link.source_die,
                link.source_port_ref,
            )
            reverse = links_by_tuple.get(reverse_key)
            if reverse is None:
                raise SchemaError("missing reciprocal directed link", path=f"{path}.links[{index}]")
            if (
                reverse.bytes_per_cycle != link.bytes_per_cycle
                or reverse.latency_cycles != link.latency_cycles
            ):
                raise SchemaError("reciprocal link properties must match", path=f"{path}.links[{index}]")


@dataclass(frozen=True, slots=True)
class RouteHop:
    index: int
    link_ref: str
    source_die: int
    source_port_ref: str
    destination_die: int
    destination_port_ref: str
    resource_ids: tuple[str, ...]

    def validate(self, path: str) -> None:
        validate_uint64(self.index, f"{path}.index")
        validate_nonempty(self.link_ref, f"{path}.link_ref")
        validate_uint64(self.source_die, f"{path}.source_die")
        validate_uint64(self.destination_die, f"{path}.destination_die")
        validate_nonempty(self.source_port_ref, f"{path}.source_port_ref")
        validate_nonempty(self.destination_port_ref, f"{path}.destination_port_ref")
        if self.source_die == self.destination_die:
            raise SchemaError("hop endpoints must differ", path=path)
        if not self.resource_ids:
            raise SchemaError("must contain egress/link resources", path=f"{path}.resource_ids")
        for index, resource_id in enumerate(self.resource_ids):
            validate_nonempty(resource_id, f"{path}.resource_ids[{index}]")
        if len(set(self.resource_ids)) != len(self.resource_ids):
            raise SchemaError("contains a duplicate resource", path=f"{path}.resource_ids")


@dataclass(frozen=True, slots=True)
class RankPlacement:
    rank: int
    die_id: int
    logical_coord: tuple[int, ...]

    def validate(self, path: str) -> None:
        validate_uint64(self.rank, f"{path}.rank")
        validate_uint64(self.die_id, f"{path}.die_id")
        for index, coordinate in enumerate(self.logical_coord):
            validate_uint64(coordinate, f"{path}.logical_coord[{index}]")


@dataclass(frozen=True, slots=True)
class PairRoute:
    id: str
    source_rank: int
    destination_rank: int
    die_path: tuple[int, ...]
    hops: tuple[RouteHop, ...]
    resource_ids: tuple[str, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_uint64(self.source_rank, f"{path}.source_rank")
        validate_uint64(self.destination_rank, f"{path}.destination_rank")
        if self.source_rank == self.destination_rank:
            raise SchemaError("route ranks must differ", path=path)
        if len(self.die_path) < 2:
            raise SchemaError("must contain source and destination die", path=f"{path}.die_path")
        for index, die_id in enumerate(self.die_path):
            validate_uint64(die_id, f"{path}.die_path[{index}]")
        if len(self.hops) != len(self.die_path) - 1:
            raise SchemaError("hop count must equal len(die_path) - 1", path=f"{path}.hops")
        flattened_resources: list[str] = []
        for index, hop in enumerate(self.hops):
            hop_path = f"{path}.hops[{index}]"
            hop.validate(hop_path)
            if hop.index != index:
                raise SchemaError("hop indices must be contiguous from zero", path=f"{hop_path}.index")
            if (hop.source_die, hop.destination_die) != (
                self.die_path[index],
                self.die_path[index + 1],
            ):
                raise SchemaError("hop endpoints disagree with die_path", path=hop_path)
            for resource_id in hop.resource_ids:
                if resource_id not in flattened_resources:
                    flattened_resources.append(resource_id)
        if not self.resource_ids:
            raise SchemaError("must contain at least one resource", path=f"{path}.resource_ids")
        for index, resource_id in enumerate(self.resource_ids):
            validate_nonempty(resource_id, f"{path}.resource_ids[{index}]")
        if len(set(self.resource_ids)) != len(self.resource_ids):
            raise SchemaError("contains a duplicate resource", path=f"{path}.resource_ids")
        if self.resource_ids != tuple(flattened_resources):
            raise SchemaError("must be the canonical first-seen union of hop resources", path=f"{path}.resource_ids")

    def validate_against(
        self,
        fabric: PhysicalFabric,
        rank_to_die: dict[int, int],
        path: str,
    ) -> None:
        source_die = rank_to_die.get(self.source_rank)
        destination_die = rank_to_die.get(self.destination_rank)
        if source_die is None or destination_die is None:
            raise SchemaError("route references an unknown rank", path=path)
        if self.die_path[0] != source_die or self.die_path[-1] != destination_die:
            raise SchemaError("route endpoints disagree with rank placement", path=f"{path}.die_path")
        dies = {die.id: die for die in fabric.dies}
        expected_path = _backend_xy_die_path(
            dies[source_die].coord,
            dies[destination_die].coord,
            fabric.die_grid,
        )
        if self.die_path != expected_path:
            raise SchemaError("route must be the exact backend-v1 X-then-Y path", path=f"{path}.die_path")
        ports = {die.id: {port.id: port for port in die.ports} for die in fabric.dies}
        links = {link.id: link for link in fabric.links}
        for index, hop in enumerate(self.hops):
            hop_path = f"{path}.hops[{index}]"
            link = links.get(hop.link_ref)
            if link is None:
                raise SchemaError("hop references an unknown directed link", path=f"{hop_path}.link_ref")
            link_identity = (
                link.source_die,
                link.source_port_ref,
                link.destination_die,
                link.destination_port_ref,
            )
            hop_identity = (
                hop.source_die,
                hop.source_port_ref,
                hop.destination_die,
                hop.destination_port_ref,
            )
            if hop_identity != link_identity:
                raise SchemaError("hop endpoints/ports disagree with directed link", path=hop_path)
            source_port = ports[hop.source_die][hop.source_port_ref]
            expected_resources = (
                source_port.egress_resource_id,
                link.resource_id,
            )
            if link.link_group_ref is not None:
                expected_resources += (link.link_group_ref,)
            if hop.resource_ids != expected_resources:
                raise SchemaError(
                    "must contain source egress port, directed link, and optional directed shared-cut resource in canonical order",
                    path=f"{hop_path}.resource_ids",
                )


@dataclass(frozen=True, slots=True)
class CrossGroupRoute:
    """One payload-independent physical route between group-local ranks."""

    id: str
    source_group_ref: str
    source_rank: int
    destination_group_ref: str
    destination_rank: int
    die_path: tuple[int, ...]
    hops: tuple[RouteHop, ...]
    resource_ids: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        source_group_ref: str,
        source_rank: int,
        destination_group_ref: str,
        destination_rank: int,
        die_path: tuple[int, ...],
        hops: tuple[RouteHop, ...],
        resource_ids: tuple[str, ...],
    ) -> "CrossGroupRoute":
        semantic_key = {
            "source_group_ref": source_group_ref,
            "source_rank": source_rank,
            "destination_group_ref": destination_group_ref,
            "destination_rank": destination_rank,
            "die_path": die_path,
            "hops": hops,
            "resource_ids": resource_ids,
        }
        result = cls(
            id=stable_artifact_id(
                "cross_group_route",
                semantic_key,
                schema_version=CROSS_GROUP_ROUTE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate("cross_group_route")
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_group_ref": self.source_group_ref,
            "source_rank": self.source_rank,
            "destination_group_ref": self.destination_group_ref,
            "destination_rank": self.destination_rank,
            "die_path": self.die_path,
            "hops": self.hops,
            "resource_ids": self.resource_ids,
        }

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.source_group_ref, f"{path}.source_group_ref")
        validate_nonempty(
            self.destination_group_ref, f"{path}.destination_group_ref"
        )
        validate_uint64(self.source_rank, f"{path}.source_rank")
        validate_uint64(self.destination_rank, f"{path}.destination_rank")
        if self.source_group_ref == self.destination_group_ref:
            raise SchemaError(
                "route endpoints must belong to distinct groups", path=path
            )
        if len(self.die_path) < 2:
            raise SchemaError(
                "must contain source and destination die",
                path=f"{path}.die_path",
            )
        if len(set(self.die_path)) != len(self.die_path):
            raise SchemaError("cannot contain a cycle", path=f"{path}.die_path")
        for index, die_id in enumerate(self.die_path):
            validate_uint64(die_id, f"{path}.die_path[{index}]")
        if len(self.hops) != len(self.die_path) - 1:
            raise SchemaError(
                "hop count must equal len(die_path) - 1",
                path=f"{path}.hops",
            )
        flattened_resources: list[str] = []
        for index, hop in enumerate(self.hops):
            hop_path = f"{path}.hops[{index}]"
            hop.validate(hop_path)
            if hop.index != index:
                raise SchemaError(
                    "hop indices must be contiguous from zero",
                    path=f"{hop_path}.index",
                )
            if (hop.source_die, hop.destination_die) != (
                self.die_path[index],
                self.die_path[index + 1],
            ):
                raise SchemaError(
                    "hop endpoints disagree with die_path", path=hop_path
                )
            for resource_id in hop.resource_ids:
                if resource_id not in flattened_resources:
                    flattened_resources.append(resource_id)
        if not self.resource_ids:
            raise SchemaError(
                "must contain at least one resource",
                path=f"{path}.resource_ids",
            )
        for index, resource_id in enumerate(self.resource_ids):
            validate_nonempty(resource_id, f"{path}.resource_ids[{index}]")
        if len(set(self.resource_ids)) != len(self.resource_ids):
            raise SchemaError(
                "contains a duplicate resource", path=f"{path}.resource_ids"
            )
        if self.resource_ids != tuple(flattened_resources):
            raise SchemaError(
                "must be the canonical first-seen union of hop resources",
                path=f"{path}.resource_ids",
            )
        expected_id = stable_artifact_id(
            "cross_group_route",
            self._semantic_key(),
            schema_version=CROSS_GROUP_ROUTE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        fabric: PhysicalFabric,
        groups: dict[str, "PhysicalGroup"],
        path: str,
    ) -> None:
        self.validate(path)
        source_group = groups.get(self.source_group_ref)
        destination_group = groups.get(self.destination_group_ref)
        if source_group is None:
            raise SchemaError(
                "references an unknown source group",
                path=f"{path}.source_group_ref",
            )
        if destination_group is None:
            raise SchemaError(
                "references an unknown destination group",
                path=f"{path}.destination_group_ref",
            )
        if source_group.instance_id == destination_group.instance_id:
            raise SchemaError(
                "route endpoints must belong to distinct instances", path=path
            )
        source = next(
            (
                placement
                for placement in source_group.placements
                if placement.rank == self.source_rank
            ),
            None,
        )
        destination = next(
            (
                placement
                for placement in destination_group.placements
                if placement.rank == self.destination_rank
            ),
            None,
        )
        if source is None:
            raise SchemaError(
                "references an unknown group-local rank",
                path=f"{path}.source_rank",
            )
        if destination is None:
            raise SchemaError(
                "references an unknown group-local rank",
                path=f"{path}.destination_rank",
            )
        if source.die_id == destination.die_id:
            raise SchemaError(
                "route endpoint dies must differ", path=f"{path}.die_path"
            )
        if (
            self.die_path[0] != source.die_id
            or self.die_path[-1] != destination.die_id
        ):
            raise SchemaError(
                "route endpoints disagree with group-local rank placement",
                path=f"{path}.die_path",
            )
        dies = {die.id: die for die in fabric.dies}
        if not set(self.die_path).issubset(dies):
            raise SchemaError(
                "route references an unknown die", path=f"{path}.die_path"
            )
        expected_path = _backend_xy_die_path(
            dies[source.die_id].coord,
            dies[destination.die_id].coord,
            fabric.die_grid,
        )
        if self.die_path != expected_path:
            raise SchemaError(
                "route must be the exact backend-v1 X-then-Y path",
                path=f"{path}.die_path",
            )
        ports = {
            die.id: {port.id: port for port in die.ports}
            for die in fabric.dies
        }
        links = {link.id: link for link in fabric.links}
        for index, hop in enumerate(self.hops):
            hop_path = f"{path}.hops[{index}]"
            link = links.get(hop.link_ref)
            if link is None:
                raise SchemaError(
                    "hop references an unknown directed link",
                    path=f"{hop_path}.link_ref",
                )
            if (
                hop.source_die,
                hop.source_port_ref,
                hop.destination_die,
                hop.destination_port_ref,
            ) != (
                link.source_die,
                link.source_port_ref,
                link.destination_die,
                link.destination_port_ref,
            ):
                raise SchemaError(
                    "hop endpoints/ports disagree with directed link",
                    path=hop_path,
                )
            source_port = ports[hop.source_die][hop.source_port_ref]
            expected_resources = (
                source_port.egress_resource_id,
                link.resource_id,
            )
            if link.link_group_ref is not None:
                expected_resources += (link.link_group_ref,)
            if hop.resource_ids != expected_resources:
                raise SchemaError(
                    "must contain source egress port, directed link, and optional directed shared-cut resource in canonical order",
                    path=f"{hop_path}.resource_ids",
                )


@dataclass(frozen=True, slots=True)
class ResourceCapacity:
    id: str
    bytes_per_cycle: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_uint64(self.bytes_per_cycle, f"{path}.bytes_per_cycle")
        if self.bytes_per_cycle == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.bytes_per_cycle")


@dataclass(frozen=True, slots=True)
class FlowWeight:
    source_rank: int
    destination_rank: int
    normalized_bytes: float

    def validate(self, path: str) -> None:
        validate_uint64(self.source_rank, f"{path}.source_rank")
        validate_uint64(self.destination_rank, f"{path}.destination_rank")
        if not math.isfinite(self.normalized_bytes) or self.normalized_bytes < 0.0:
            raise SchemaError("must be a finite non-negative float", path=f"{path}.normalized_bytes")


@dataclass(frozen=True, slots=True)
class ResourceWork:
    resource_id: str
    normalized_bytes: float

    def validate(self, path: str) -> None:
        validate_nonempty(self.resource_id, f"{path}.resource_id")
        if not math.isfinite(self.normalized_bytes) or self.normalized_bytes < 0.0:
            raise SchemaError("must be a finite non-negative float", path=f"{path}.normalized_bytes")


@dataclass(frozen=True, slots=True)
class CanonicalBandwidthProfile:
    id: str
    traffic_template: str
    flow_weights: tuple[FlowWeight, ...]
    resource_work: tuple[ResourceWork, ...]
    bottleneck_resource: str
    lane_eq_bandwidth: float

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.traffic_template, f"{path}.traffic_template")
        validate_nonempty(self.bottleneck_resource, f"{path}.bottleneck_resource")
        if not math.isfinite(self.lane_eq_bandwidth) or self.lane_eq_bandwidth <= 0.0:
            raise SchemaError("must be a finite positive float", path=f"{path}.lane_eq_bandwidth")
        for index, weight in enumerate(self.flow_weights):
            weight.validate(f"{path}.flow_weights[{index}]")
        for index, work in enumerate(self.resource_work):
            work.validate(f"{path}.resource_work[{index}]")


@dataclass(frozen=True, slots=True)
class GroupEmbedding:
    routes: tuple[PairRoute, ...]
    resource_capacities: tuple[ResourceCapacity, ...]
    canonical_profiles: tuple[CanonicalBandwidthProfile, ...]

    def validate(self, path: str) -> None:
        validate_unique_ids(self.routes, f"{path}.routes")
        capacities = validate_unique_ids(self.resource_capacities, f"{path}.resource_capacities")
        validate_unique_ids(self.canonical_profiles, f"{path}.canonical_profiles")
        for index, route in enumerate(self.routes):
            route.validate(f"{path}.routes[{index}]")
            if not set(route.resource_ids).issubset(capacities):
                raise SchemaError("route references an unknown resource", path=f"{path}.routes[{index}].resource_ids")
        for index, capacity in enumerate(self.resource_capacities):
            capacity.validate(f"{path}.resource_capacities[{index}]")
        for index, profile in enumerate(self.canonical_profiles):
            profile.validate(f"{path}.canonical_profiles[{index}]")
            if profile.bottleneck_resource not in capacities:
                raise SchemaError("unknown bottleneck resource", path=f"{path}.canonical_profiles[{index}].bottleneck_resource")
            for work in profile.resource_work:
                if work.resource_id not in capacities:
                    raise SchemaError("profile references an unknown resource", path=f"{path}.canonical_profiles[{index}].resource_work")


@dataclass(frozen=True, slots=True)
class PhysicalGroup:
    id: str
    instance_id: str
    mesh_ref: str
    axis: MeshAxisName
    logical_shape: tuple[int, ...]
    placements: tuple[RankPlacement, ...]
    embedding: GroupEmbedding

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.instance_id, f"{path}.instance_id")
        validate_nonempty(self.mesh_ref, f"{path}.mesh_ref")
        if not self.logical_shape:
            raise SchemaError("must be non-empty", path=f"{path}.logical_shape")
        for index, size in enumerate(self.logical_shape):
            validate_uint64(size, f"{path}.logical_shape[{index}]")
            if size == 0:
                raise SchemaError("must be greater than zero", path=f"{path}.logical_shape[{index}]")
        if not self.placements:
            raise SchemaError("must contain rank placements", path=f"{path}.placements")
        if len(self.placements) != math.prod(self.logical_shape):
            raise SchemaError(
                "placement count must equal the product of logical_shape",
                path=f"{path}.placements",
            )
        ranks: set[int] = set()
        dies: set[int] = set()
        for index, placement in enumerate(self.placements):
            placement.validate(f"{path}.placements[{index}]")
            if placement.rank in ranks:
                raise SchemaError("duplicate rank", path=f"{path}.placements[{index}].rank")
            if placement.die_id in dies:
                raise SchemaError("die appears more than once", path=f"{path}.placements[{index}].die_id")
            if len(placement.logical_coord) != len(self.logical_shape):
                raise SchemaError("coordinate rank does not match logical_shape", path=f"{path}.placements[{index}].logical_coord")
            if any(placement.logical_coord[axis] >= self.logical_shape[axis] for axis in range(len(self.logical_shape))):
                raise SchemaError("logical coordinate lies outside logical_shape", path=f"{path}.placements[{index}].logical_coord")
            ranks.add(placement.rank)
            dies.add(placement.die_id)
        if ranks != set(range(len(self.placements))):
            raise SchemaError("ranks must be contiguous from zero", path=f"{path}.placements")
        self.embedding.validate(f"{path}.embedding")
        by_rank = {placement.rank: placement.die_id for placement in self.placements}
        for index, route in enumerate(self.embedding.routes):
            if route.source_rank not in by_rank or route.destination_rank not in by_rank:
                raise SchemaError("route references an unknown rank", path=f"{path}.embedding.routes[{index}]")
            if route.die_path[0] != by_rank[route.source_rank] or route.die_path[-1] != by_rank[route.destination_rank]:
                raise SchemaError("route endpoints disagree with rank placement", path=f"{path}.embedding.routes[{index}].die_path")


@dataclass(frozen=True, slots=True)
class PhysicalInstance:
    id: str
    origin_instance_id: str
    role: LogicalRole
    die_region: tuple[int, ...]
    group_ids: tuple[str, ...]
    node_ids: tuple[str, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.origin_instance_id, f"{path}.origin_instance_id")
        if not self.die_region or len(set(self.die_region)) != len(self.die_region):
            raise SchemaError("must contain unique die ids", path=f"{path}.die_region")
        for index, die_id in enumerate(self.die_region):
            validate_uint64(die_id, f"{path}.die_region[{index}]")
        for field_name in ("group_ids", "node_ids"):
            refs = getattr(self, field_name)
            if len(set(refs)) != len(refs):
                raise SchemaError("contains duplicate ids", path=f"{path}.{field_name}")
            for index, ref in enumerate(refs):
                validate_nonempty(ref, f"{path}.{field_name}[{index}]")


@dataclass(frozen=True, slots=True)
class PhysicalNode:
    id: str
    origin_node_id: str
    instance_id: str
    kind: OpKind
    phase: OpPhase
    stage: int
    mesh_ref: str
    execution_group_ref: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    workload: NodeWorkload
    math: NodeMath
    effects: NodeEffects
    impl_ref: str

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.origin_node_id, f"{path}.origin_node_id")
        validate_nonempty(self.instance_id, f"{path}.instance_id")
        if type(self.kind) is not OpKind:
            raise SchemaError("must be an OpKind", path=f"{path}.kind")
        validate_uint64(self.stage, f"{path}.stage")
        validate_nonempty(self.mesh_ref, f"{path}.mesh_ref")
        validate_nonempty(self.execution_group_ref, f"{path}.execution_group_ref")
        validate_nonempty(self.impl_ref, f"{path}.impl_ref")
        for field_name in ("inputs", "outputs"):
            refs = getattr(self, field_name)
            if len(set(refs)) != len(refs):
                raise SchemaError("contains duplicate value ids", path=f"{path}.{field_name}")
        expected_types = {
            OpKind.GEMM: GemmWorkload,
            OpKind.ELEMENTWISE: (SwiGluWorkload, ResidualWorkload),
            OpKind.NORM: RmsNormWorkload,
            OpKind.ATTENTION: AttentionWorkload,
            OpKind.COLLECTIVE: CollectiveWorkload,
            OpKind.P2P: P2PByteWorkload,
            OpKind.EMBEDDING: EmbeddingWorkload,
            OpKind.ROPE: RopeQkWorkload,
            OpKind.SAMPLING: GreedySampleWorkload,
            OpKind.CE_FORWARD: CrossEntropyForwardWorkload,
            OpKind.CE_BACKWARD: CrossEntropyBackwardWorkload,
            OpKind.OPTIMIZER_UPDATE: SgdUpdateWorkload,
        }[self.kind]
        expected_types = expected_types if type(expected_types) is tuple else (expected_types,)
        if type(self.workload) not in expected_types:
            raise SchemaError(
                f"{self.kind.value!r} requires one of {[item.__name__ for item in expected_types]!r}",
                path=f"{path}.workload",
            )
        self.workload.validate(f"{path}.workload")
        self.math.validate(f"{path}.math")
        self.effects.validate(f"{path}.effects")


@dataclass(frozen=True, slots=True)
class FusedOpSkeleton:
    id: str
    fusion_ref: str
    instance_id: str
    member_node_ids: tuple[str, ...]
    boundary_inputs: tuple[str, ...]
    boundary_outputs: tuple[str, ...]
    semantic_contract: FusionSemanticContract
    impl: FusionImpl

    def validate(self, path: str) -> None:
        for field_name in ("id", "fusion_ref", "instance_id"):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if not self.member_node_ids or len(set(self.member_node_ids)) != len(self.member_node_ids):
            raise SchemaError("must contain unique member ids", path=f"{path}.member_node_ids")
        for field_name in ("boundary_inputs", "boundary_outputs"):
            refs = getattr(self, field_name)
            if len(set(refs)) != len(refs):
                raise SchemaError("contains duplicate value ids", path=f"{path}.{field_name}")
        self.semantic_contract.validate(f"{path}.semantic_contract")


@dataclass(frozen=True, slots=True)
class KvRoute:
    id: str
    source_instance_id: str
    destination_instance_id: str
    value_id: str
    bytes: int
    die_path: tuple[int, ...]

    def validate(self, path: str) -> None:
        for field_name in ("id", "source_instance_id", "destination_instance_id", "value_id"):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        validate_uint64(self.bytes, f"{path}.bytes")
        if len(self.die_path) < 2:
            raise SchemaError("must contain source and destination die", path=f"{path}.die_path")
        for index, die_id in enumerate(self.die_path):
            validate_uint64(die_id, f"{path}.die_path[{index}]")


@dataclass(frozen=True, slots=True)
class IR1:
    schema_version: str
    producer_pass: str
    id: str
    source_ir0_id: str
    profile: ProfileKey
    fabric: PhysicalFabric
    instances: tuple[PhysicalInstance, ...]
    groups: tuple[PhysicalGroup, ...]
    nodes: tuple[PhysicalNode, ...]
    values: tuple[TensorValue, ...]
    edges: tuple[GraphEdge, ...]
    fusion_candidates: tuple[FusionCandidate, ...]
    fused_op_skeletons: tuple[FusedOpSkeleton, ...]
    cross_routes: tuple[CrossGroupRoute, ...]
    state_accesses: tuple[StateAccess, ...]
    persistent_state_manifest: PersistentStateManifest | None
    instance_profiles: tuple[InstanceProfileBinding, ...] = ()
    node_profiles: tuple[NodeProfileBinding, ...] = ()
    pd_plan_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        source_ir0_id: str,
        profile: ProfileKey,
        fabric: PhysicalFabric,
        instances: tuple[PhysicalInstance, ...],
        groups: tuple[PhysicalGroup, ...],
        nodes: tuple[PhysicalNode, ...],
        values: tuple[TensorValue, ...],
        edges: tuple[GraphEdge, ...],
        fusion_candidates: tuple[FusionCandidate, ...] = (),
        fused_op_skeletons: tuple[FusedOpSkeleton, ...] = (),
        cross_routes: tuple[CrossGroupRoute, ...] = (),
        state_accesses: tuple[StateAccess, ...] = (),
        persistent_state_manifest: PersistentStateManifest | None = None,
        instance_profiles: tuple[InstanceProfileBinding, ...] = (),
        node_profiles: tuple[NodeProfileBinding, ...] = (),
        pd_plan_id: str | None = None,
    ) -> "IR1":
        semantic_key = {
            "source_ir0_id": source_ir0_id,
            "profile": profile,
            "fabric": fabric,
            "instances": instances,
            "groups": groups,
            "nodes": nodes,
            "values": values,
            "edges": edges,
            "fusion_candidates": fusion_candidates,
            "fused_op_skeletons": fused_op_skeletons,
            "cross_routes": cross_routes,
            "state_accesses": state_accesses,
            "persistent_state_manifest": persistent_state_manifest,
            "instance_profiles": instance_profiles,
            "node_profiles": node_profiles,
            "pd_plan_id": pd_plan_id,
        }
        return cls(
            schema_version=IR1_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id("ir1", semantic_key, schema_version=IR1_SCHEMA_VERSION),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_ir0_id": self.source_ir0_id,
            "profile": self.profile,
            "fabric": self.fabric,
            "instances": self.instances,
            "groups": self.groups,
            "nodes": self.nodes,
            "values": self.values,
            "edges": self.edges,
            "fusion_candidates": self.fusion_candidates,
            "fused_op_skeletons": self.fused_op_skeletons,
            "cross_routes": self.cross_routes,
            "state_accesses": self.state_accesses,
            "persistent_state_manifest": self.persistent_state_manifest,
            "instance_profiles": self.instance_profiles,
            "node_profiles": self.node_profiles,
            "pd_plan_id": self.pd_plan_id,
        }

    def validate(self, path: str = "ir1") -> None:
        if self.schema_version != IR1_SCHEMA_VERSION:
            raise SchemaError(f"unsupported schema version {self.schema_version!r}", path=f"{path}.schema_version")
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        validate_nonempty(self.source_ir0_id, f"{path}.source_ir0_id")
        self.profile.validate(f"{path}.profile")
        self.fabric.validate(f"{path}.fabric")
        die_ids = {die.id for die in self.fabric.dies}
        instance_index = validate_unique_ids(self.instances, f"{path}.instances")
        group_index = validate_unique_ids(self.groups, f"{path}.groups")
        node_index, value_index = validate_value_graph(self.nodes, self.values, self.edges, path=path)
        if type(self.instance_profiles) is not tuple:
            raise SchemaError(
                "must be an immutable tuple",
                path=f"{path}.instance_profiles",
            )
        if self.instance_profiles:
            if self.pd_plan_id is None:
                raise SchemaError(
                    "is required with instance_profiles",
                    path=f"{path}.pd_plan_id",
                )
            validate_nonempty(self.pd_plan_id, f"{path}.pd_plan_id")
            if self.instance_profiles != tuple(
                sorted(
                    self.instance_profiles,
                    key=lambda item: (
                        item.instance_ref,
                        item.profile.stable_id(),
                    ),
                )
            ):
                raise SchemaError(
                    "must use canonical instance_ref/profile order",
                    path=f"{path}.instance_profiles",
                )
            binding_keys: set[tuple[str, str]] = set()
            bound_instances: set[str] = set()
            profiles_by_instance: dict[str, set[ProfileKey]] = {}
            for index, binding in enumerate(self.instance_profiles):
                binding_path = f"{path}.instance_profiles[{index}]"
                if type(binding) is not InstanceProfileBinding:
                    raise SchemaError(
                        "must be an InstanceProfileBinding", path=binding_path
                    )
                binding.validate(binding_path)
                if binding.instance_ref not in instance_index:
                    raise SchemaError(
                        "references a dangling instance",
                        path=f"{binding_path}.instance_ref",
                    )
                key = (binding.instance_ref, binding.profile.stable_id())
                if key in binding_keys:
                    raise SchemaError(
                        "duplicate instance/profile binding", path=binding_path
                    )
                binding_keys.add(key)
                bound_instances.add(binding.instance_ref)
                profiles_by_instance.setdefault(binding.instance_ref, set()).add(
                    binding.profile
                )
            if bound_instances != set(instance_index):
                raise SchemaError(
                    "must bind every instance at least once",
                    path=f"{path}.instance_profiles",
                )
            if self.profile != self.instance_profiles[0].profile:
                raise SchemaError(
                    "must equal the first canonical instance profile",
                    path=f"{path}.profile",
                )
        else:
            if self.pd_plan_id is not None:
                raise SchemaError(
                    "must be null without instance_profiles",
                    path=f"{path}.pd_plan_id",
                )
            if len(self.instances) != 1:
                raise SchemaError(
                    "multi-instance IR1 requires instance_profiles",
                    path=f"{path}.instance_profiles",
                )
            profiles_by_instance = {self.instances[0].id: {self.profile}}
        if type(self.node_profiles) is not tuple:
            raise SchemaError(
                "must be an immutable tuple", path=f"{path}.node_profiles"
            )
        multiple_profiles = any(
            len(profiles) > 1 for profiles in profiles_by_instance.values()
        )
        if multiple_profiles:
            if tuple(binding.node_ref for binding in self.node_profiles) != tuple(
                node.id for node in self.nodes
            ):
                raise SchemaError(
                    "multi-profile instances require one binding per node in node order",
                    path=f"{path}.node_profiles",
                )
        elif self.node_profiles:
            raise SchemaError(
                "must be empty when every instance has one profile",
                path=f"{path}.node_profiles",
            )
        for index, binding in enumerate(self.node_profiles):
            binding_path = f"{path}.node_profiles[{index}]"
            if type(binding) is not NodeProfileBinding:
                raise SchemaError("must be a NodeProfileBinding", path=binding_path)
            binding.validate(binding_path)
            node = node_index.get(binding.node_ref)
            if node is None:
                raise SchemaError(
                    "references a dangling node", path=f"{binding_path}.node_ref"
                )
            if binding.profile not in profiles_by_instance[node.instance_id]:
                raise SchemaError(
                    "node profile must belong to its instance",
                    path=f"{binding_path}.profile",
                )
            workload_profile = getattr(node.workload, "profile", None)
            if workload_profile is not None and workload_profile != binding.profile:
                raise SchemaError(
                    "workload profile must equal the node profile",
                    path=f"{binding_path}.profile",
                )
        for index, instance in enumerate(self.instances):
            instance.validate(f"{path}.instances[{index}]")
            if not set(instance.die_region).issubset(die_ids):
                raise SchemaError("contains a dangling die", path=f"{path}.instances[{index}].die_region")
            if not set(instance.group_ids).issubset(group_index):
                raise SchemaError("contains a dangling group", path=f"{path}.instances[{index}].group_ids")
            if not set(instance.node_ids).issubset(node_index):
                raise SchemaError("contains a dangling node", path=f"{path}.instances[{index}].node_ids")
        mesh_refs_by_instance: dict[str, set[str]] = {instance.id: set() for instance in self.instances}
        route_ids: set[str] = set()
        for index, group in enumerate(self.groups):
            group.validate(f"{path}.groups[{index}]")
            instance = instance_index.get(group.instance_id)
            if instance is None:
                raise SchemaError("dangling instance", path=f"{path}.groups[{index}].instance_id")
            if group.id not in instance.group_ids:
                raise SchemaError("owning instance does not reference this group", path=f"{path}.groups[{index}].id")
            placement_dies = {placement.die_id for placement in group.placements}
            if not placement_dies.issubset(instance.die_region):
                raise SchemaError("rank placement lies outside instance die_region", path=f"{path}.groups[{index}].placements")
            if not placement_dies.issubset(die_ids):
                raise SchemaError("rank placement references an unknown die", path=f"{path}.groups[{index}].placements")
            rank_to_die = {placement.rank: placement.die_id for placement in group.placements}
            for route_index, route in enumerate(group.embedding.routes):
                if route.id in route_ids:
                    raise SchemaError(
                        "PairRoute ids must be globally unique in IR-1",
                        path=f"{path}.groups[{index}].embedding.routes[{route_index}].id",
                    )
                route_ids.add(route.id)
                if not set(route.die_path).issubset(die_ids):
                    raise SchemaError("route references an unknown die", path=f"{path}.groups[{index}].embedding.routes")
                route.validate_against(
                    self.fabric,
                    rank_to_die,
                    f"{path}.groups[{index}].embedding.routes[{route_index}]",
                )
            mesh_refs_by_instance[group.instance_id].add(group.mesh_ref)
        for index, node in enumerate(self.nodes):
            instance = instance_index.get(node.instance_id)
            if instance is None:
                raise SchemaError("dangling instance", path=f"{path}.nodes[{index}].instance_id")
            if node.id not in instance.node_ids:
                raise SchemaError("owning instance does not reference this node", path=f"{path}.nodes[{index}].id")
            if node.mesh_ref not in mesh_refs_by_instance[node.instance_id]:
                raise SchemaError("dangling physical mesh", path=f"{path}.nodes[{index}].mesh_ref")
            execution_group = group_index.get(node.execution_group_ref)
            if execution_group is None:
                raise SchemaError(
                    "dangling execution group",
                    path=f"{path}.nodes[{index}].execution_group_ref",
                )
            if (
                execution_group.instance_id != node.instance_id
                or execution_group.mesh_ref != node.mesh_ref
            ):
                raise SchemaError(
                    "execution group must match node instance and mesh",
                    path=f"{path}.nodes[{index}].execution_group_ref",
                )
        if type(self.state_accesses) is not tuple:
            raise SchemaError(
                "must be an immutable tuple", path=f"{path}.state_accesses"
            )
        if self.state_accesses != tuple(
            sorted(
                self.state_accesses,
                key=lambda item: (
                    item.node_ref, item.state_ref, item.rank, item.id
                ),
            )
        ):
            raise SchemaError(
                "must use canonical node/state/rank/id order",
                path=f"{path}.state_accesses",
            )
        if self.persistent_state_manifest is None:
            if self.state_accesses:
                raise SchemaError(
                    "state accesses require a persistent state manifest",
                    path=f"{path}.state_accesses",
                )
        else:
            if type(self.persistent_state_manifest) is not PersistentStateManifest:
                raise SchemaError(
                    "must be a PersistentStateManifest or null",
                    path=f"{path}.persistent_state_manifest",
                )
            manifest = self.persistent_state_manifest
            manifest.validate(f"{path}.persistent_state_manifest")
            for space_index, space in enumerate(manifest.address_spaces):
                if space.die_id not in die_ids:
                    raise SchemaError(
                        "HBM address space references an unknown physical die",
                        path=(
                            f"{path}.persistent_state_manifest"
                            f".address_spaces[{space_index}].die_id"
                        ),
                    )
            groups_by_owner: dict[tuple[str, str], PhysicalGroup] = {}
            for group in self.groups:
                owner = (group.instance_id, group.mesh_ref)
                if owner in groups_by_owner:
                    raise SchemaError(
                        "persistent state owner must resolve to one physical group",
                        path=f"{path}.groups",
                    )
                groups_by_owner[owner] = group
            declarations = {
                declaration.id: declaration
                for declaration in manifest.declarations
            }
            bindings = {
                binding.state_ref: binding for binding in manifest.bindings
            }
            for declaration_index, declaration in enumerate(manifest.declarations):
                identity = declaration.identity
                declaration_path = (
                    f"{path}.persistent_state_manifest"
                    f".declarations[{declaration_index}]"
                )
                group = groups_by_owner.get(
                    (identity.instance_ref, identity.mesh_ref)
                )
                if group is None:
                    raise SchemaError(
                        "state identity has no physical owner group",
                        path=f"{declaration_path}.identity",
                    )
                placement = next(
                    (
                        item
                        for item in group.placements
                        if item.rank == identity.shard_index
                    ),
                    None,
                )
                if placement is None:
                    raise SchemaError(
                        "state shard has no rank placement",
                        path=f"{declaration_path}.identity.shard_index",
                    )
                if bindings[declaration.id].die_id != placement.die_id:
                    raise SchemaError(
                        "HBM binding die must equal the state shard home die",
                        path=(
                            f"{path}.persistent_state_manifest"
                            f".bindings[{declaration.id}]"
                        ),
                    )

            validate_unique_ids(self.state_accesses, f"{path}.state_accesses")
            allowed_modes = {
                PersistentStateAccess.READ_ONLY: {StateAccessMode.READ},
                PersistentStateAccess.READ_WRITE: {
                    StateAccessMode.READ,
                    StateAccessMode.WRITE,
                    StateAccessMode.READ_WRITE,
                },
                PersistentStateAccess.RESERVED: set(),
            }
            for access_index, access in enumerate(self.state_accesses):
                access_path = f"{path}.state_accesses[{access_index}]"
                if type(access) is not StateAccess:
                    raise SchemaError("must be a StateAccess", path=access_path)
                access.validate(access_path)
                node = node_index.get(access.node_ref)
                if node is None:
                    raise SchemaError(
                        "state access references a dangling node",
                        path=f"{access_path}.node_ref",
                    )
                declaration = declarations.get(access.state_ref)
                if declaration is None:
                    raise SchemaError(
                        "state access references a dangling state declaration",
                        path=f"{access_path}.state_ref",
                    )
                identity = declaration.identity
                if (
                    node.instance_id != identity.instance_ref
                    or node.mesh_ref != identity.mesh_ref
                ):
                    raise SchemaError(
                        "state access node must match the state owner",
                        path=access_path,
                    )
                if access.rank != identity.shard_index:
                    raise SchemaError(
                        "state access rank must equal the state shard",
                        path=f"{access_path}.rank",
                    )
                if access.mode not in allowed_modes[declaration.access]:
                    raise SchemaError(
                        "state access mode exceeds declaration permission",
                        path=f"{access_path}.mode",
                    )
                if access.mode in (
                    StateAccessMode.READ,
                    StateAccessMode.READ_WRITE,
                ):
                    state_access_tensor_view(
                        access,
                        declaration,
                        "read",
                        path=access_path,
                    )
                if access.mode in (
                    StateAccessMode.WRITE,
                    StateAccessMode.READ_WRITE,
                ):
                    state_access_tensor_view(
                        access,
                        declaration,
                        "write",
                        path=access_path,
                    )
        all_mesh_refs = {group.mesh_ref for group in self.groups}
        for index, value in enumerate(self.values):
            if value.sharding.mesh_ref not in all_mesh_refs:
                raise SchemaError("dangling physical mesh", path=f"{path}.values[{index}].sharding.mesh_ref")
        validate_unique_ids(self.fusion_candidates, f"{path}.fusion_candidates")
        for index, candidate in enumerate(self.fusion_candidates):
            candidate_path = f"{path}.fusion_candidates[{index}]"
            candidate.validate(candidate_path)
            member_set = set(candidate.members)
            if not member_set.issubset(node_index):
                raise SchemaError(
                    "contains a dangling member",
                    path=f"{candidate_path}.members",
                )
            if not set(candidate.boundary_inputs).issubset(value_index):
                raise SchemaError(
                    "contains a dangling boundary input",
                    path=f"{candidate_path}.boundary_inputs",
                )
            if not set(candidate.boundary_outputs).issubset(value_index):
                raise SchemaError(
                    "contains a dangling boundary output",
                    path=f"{candidate_path}.boundary_outputs",
                )
            expected_inputs = {
                value.id
                for value in self.values
                if any(consumer in member_set for consumer in value.consumers)
                and value.producer not in member_set
            }
            expected_outputs = {
                value.id
                for value in self.values
                if value.producer in member_set
                and (
                    not value.consumers
                    or any(consumer not in member_set for consumer in value.consumers)
                )
            }
            if set(candidate.boundary_inputs) != expected_inputs:
                raise SchemaError(
                    "does not match graph-derived boundary inputs",
                    path=f"{candidate_path}.boundary_inputs",
                )
            if set(candidate.boundary_outputs) != expected_outputs:
                raise SchemaError(
                    "does not match graph-derived boundary outputs",
                    path=f"{candidate_path}.boundary_outputs",
                )
        validate_unique_ids(self.fused_op_skeletons, f"{path}.fused_op_skeletons")
        for index, skeleton in enumerate(self.fused_op_skeletons):
            skeleton.validate(f"{path}.fused_op_skeletons[{index}]")
            if skeleton.instance_id not in instance_index:
                raise SchemaError("dangling instance", path=f"{path}.fused_op_skeletons[{index}].instance_id")
            if not set(skeleton.member_node_ids).issubset(node_index):
                raise SchemaError("contains a dangling member", path=f"{path}.fused_op_skeletons[{index}].member_node_ids")
            if not set(skeleton.boundary_inputs).issubset(value_index) or not set(skeleton.boundary_outputs).issubset(value_index):
                raise SchemaError("contains a dangling boundary value", path=f"{path}.fused_op_skeletons[{index}]")
            if any(node_index[node_id].instance_id != skeleton.instance_id for node_id in skeleton.member_node_ids):
                raise SchemaError("members do not belong to skeleton instance", path=f"{path}.fused_op_skeletons[{index}].member_node_ids")
        if type(self.cross_routes) is not tuple:
            raise SchemaError(
                "must be an immutable tuple", path=f"{path}.cross_routes"
            )
        for index, route in enumerate(self.cross_routes):
            if type(route) is not CrossGroupRoute:
                raise SchemaError(
                    "must be a CrossGroupRoute",
                    path=f"{path}.cross_routes[{index}]",
                )
        validate_unique_ids(self.cross_routes, f"{path}.cross_routes")
        if self.cross_routes != tuple(
            sorted(
                self.cross_routes,
                key=lambda route: (
                    route.source_group_ref,
                    route.source_rank,
                    route.destination_group_ref,
                    route.destination_rank,
                    route.id,
                ),
            )
        ):
            raise SchemaError(
                "must use canonical endpoint/id order",
                path=f"{path}.cross_routes",
            )
        for index, route in enumerate(self.cross_routes):
            route_path = f"{path}.cross_routes[{index}]"
            if route.id in route_ids:
                raise SchemaError(
                    "route ids must be globally unique in IR-1",
                    path=f"{route_path}.id",
                )
            route_ids.add(route.id)
            route.validate_against(self.fabric, group_index, route_path)
        expected_id = stable_artifact_id("ir1", self._semantic_key(), schema_version=IR1_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")
