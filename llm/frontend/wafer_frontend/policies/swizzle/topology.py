"""Immutable physical-topology views used by Swizzle candidate generators.

The helpers in this module deliberately consume the routes frozen in IR-1.  A
candidate may choose an order in which those routes are used, but it must not
invent a wrap-around link or silently replace backend-v1 X-then-Y routing.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...errors import SchemaError
from ...schema.ir1 import PairRoute, PhysicalFabric, PhysicalGroup


@dataclass(frozen=True, slots=True)
class TopologyRank:
    rank: int
    die_id: int
    logical_coord: tuple[int, ...]
    physical_coord: tuple[int, int]


@dataclass(frozen=True, slots=True)
class TopologyRoute:
    id: str
    source_rank: int
    destination_rank: int
    die_path: tuple[int, ...]
    resource_ids: tuple[str, ...]
    hop_count: int
    leaves_group: bool


@dataclass(frozen=True, slots=True)
class TopologyResource:
    id: str
    bytes_per_cycle: int
    route_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResourceLoad:
    resource_id: str
    logical_bytes: int


@dataclass(frozen=True, slots=True)
class TopologyFlow:
    source_rank: int
    destination_rank: int
    logical_bytes: int


@dataclass(frozen=True, slots=True)
class SwizzleTopologyView:
    """Canonical, immutable projection of one PhysicalGroup onto the fabric."""

    group_ref: str
    logical_shape: tuple[int, ...]
    ranks: tuple[TopologyRank, ...]
    routes: tuple[TopologyRoute, ...]
    resources: tuple[TopologyResource, ...]
    physical_origin: tuple[int, int]
    physical_shape: tuple[int, int]
    is_complete_rectangle: bool
    row_rank_orders: tuple[tuple[int, ...], ...]
    column_rank_orders: tuple[tuple[int, ...], ...]
    snake_rank_order: tuple[int, ...]
    hamiltonian_cycle_rank_order: tuple[int, ...]
    cross_group_route_refs: tuple[str, ...]

    @property
    def rank_count(self) -> int:
        return len(self.ranks)

    @property
    def has_hamiltonian_cycle(self) -> bool:
        return bool(self.hamiltonian_cycle_rank_order)

    def route(self, source_rank: int, destination_rank: int) -> TopologyRoute:
        for route in self.routes:
            if (
                route.source_rank == source_rank
                and route.destination_rank == destination_rank
            ):
                return route
        raise SchemaError(
            f"missing route {source_rank}->{destination_rank}",
            path="swizzle_topology.routes",
        )


def _route_view(route: PairRoute, group_dies: frozenset[int]) -> TopologyRoute:
    return TopologyRoute(
        id=route.id,
        source_rank=route.source_rank,
        destination_rank=route.destination_rank,
        die_path=route.die_path,
        resource_ids=route.resource_ids,
        hop_count=len(route.hops),
        leaves_group=not set(route.die_path).issubset(group_dies),
    )


def _rank_lines(
    rank_by_coord: dict[tuple[int, int], int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    xs = sorted({coord[0] for coord in rank_by_coord})
    ys = sorted({coord[1] for coord in rank_by_coord})
    rows = tuple(
        tuple(rank_by_coord[(x, y)] for x in xs if (x, y) in rank_by_coord)
        for y in ys
    )
    columns = tuple(
        tuple(rank_by_coord[(x, y)] for y in ys if (x, y) in rank_by_coord)
        for x in xs
    )
    return rows, columns


def _snake(rows: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(
        rank
        for row_index, row in enumerate(rows)
        for rank in (row if row_index % 2 == 0 else tuple(reversed(row)))
    )


def _hamiltonian_cycle(
    ranks: tuple[int, ...],
    direct_edges: frozenset[tuple[int, int]],
) -> tuple[int, ...]:
    """Return the lexicographically first real directed cycle, if one exists.

    The search is intentionally anchored at the smallest rank so rotations of
    the same cycle cannot perturb candidate identity.  It operates only on
    one-hop routes in the frozen embedding; multi-hop XY routes cannot be used
    to pretend that a physical wrap-around link exists.
    """

    if len(ranks) < 3:
        return ()
    start = min(ranks)
    target_length = len(ranks)

    def visit(path: tuple[int, ...], remaining: frozenset[int]) -> tuple[int, ...]:
        if not remaining:
            return path if (path[-1], start) in direct_edges else ()
        current = path[-1]
        for candidate in sorted(remaining):
            if (current, candidate) not in direct_edges:
                continue
            found = visit(path + (candidate,), remaining - {candidate})
            if found:
                return found
        return ()

    result = visit((start,), frozenset(ranks) - {start})
    if len(result) != target_length:
        return ()
    return result


def build_topology_view(
    group: PhysicalGroup,
    fabric: PhysicalFabric,
) -> SwizzleTopologyView:
    """Build and independently close one group's physical topology.

    All ordered rank pairs must have exactly one route.  Routes that transit a
    Die outside the group remain visible through ``cross_group_route_refs`` so
    Level-0 feasibility can reject the candidate with typed evidence.
    """

    group.validate("physical_group")
    fabric.validate("physical_fabric")
    die_by_id = {die.id: die for die in fabric.dies}
    placement_dies = frozenset(item.die_id for item in group.placements)
    if not placement_dies.issubset(die_by_id):
        raise SchemaError(
            "group placement references a Die outside the fabric",
            path="physical_group.placements",
        )

    ranks = tuple(
        TopologyRank(
            rank=placement.rank,
            die_id=placement.die_id,
            logical_coord=placement.logical_coord,
            physical_coord=die_by_id[placement.die_id].coord,
        )
        for placement in sorted(group.placements, key=lambda item: item.rank)
    )
    rank_numbers = tuple(item.rank for item in ranks)
    expected_pairs = tuple(
        (source, destination)
        for source in rank_numbers
        for destination in rank_numbers
        if source != destination
    )
    route_by_pair: dict[tuple[int, int], PairRoute] = {}
    for route in group.embedding.routes:
        key = (route.source_rank, route.destination_rank)
        if key in route_by_pair:
            raise SchemaError(
                f"duplicate route {key[0]}->{key[1]}",
                path="physical_group.embedding.routes",
            )
        route.validate_against(
            fabric,
            {item.rank: item.die_id for item in group.placements},
            "physical_group.embedding.routes",
        )
        route_by_pair[key] = route
    if set(route_by_pair) != set(expected_pairs):
        missing = sorted(set(expected_pairs) - set(route_by_pair))
        extra = sorted(set(route_by_pair) - set(expected_pairs))
        raise SchemaError(
            f"must contain every ordered rank-pair route; missing={missing!r}, extra={extra!r}",
            path="physical_group.embedding.routes",
        )
    routes = tuple(
        _route_view(route_by_pair[pair], placement_dies) for pair in expected_pairs
    )

    capacity_by_id = {
        item.id: item.bytes_per_cycle
        for item in group.embedding.resource_capacities
    }
    used_resources = {resource for route in routes for resource in route.resource_ids}
    if used_resources != set(capacity_by_id):
        raise SchemaError(
            "resource capacities must exactly cover route resources",
            path="physical_group.embedding.resource_capacities",
        )
    resources = tuple(
        TopologyResource(
            id=resource_id,
            bytes_per_cycle=capacity_by_id[resource_id],
            route_refs=tuple(
                route.id for route in routes if resource_id in route.resource_ids
            ),
        )
        for resource_id in sorted(capacity_by_id)
    )

    rank_by_coord = {item.physical_coord: item.rank for item in ranks}
    xs = sorted({coord[0] for coord in rank_by_coord})
    ys = sorted({coord[1] for coord in rank_by_coord})
    origin = (xs[0], ys[0])
    shape = (xs[-1] - xs[0] + 1, ys[-1] - ys[0] + 1)
    rectangle_coords = {
        (x, y)
        for x in range(xs[0], xs[-1] + 1)
        for y in range(ys[0], ys[-1] + 1)
    }
    is_rectangle = set(rank_by_coord) == rectangle_coords
    rows, columns = _rank_lines(rank_by_coord)
    snake = _snake(rows)
    direct_edges = frozenset(
        (route.source_rank, route.destination_rank)
        for route in routes
        if route.hop_count == 1 and not route.leaves_group
    )
    cycle = _hamiltonian_cycle(rank_numbers, direct_edges)
    return SwizzleTopologyView(
        group_ref=group.id,
        logical_shape=group.logical_shape,
        ranks=ranks,
        routes=routes,
        resources=resources,
        physical_origin=origin,
        physical_shape=shape,
        is_complete_rectangle=is_rectangle,
        row_rank_orders=rows,
        column_rank_orders=columns,
        snake_rank_order=snake,
        hamiltonian_cycle_rank_order=cycle,
        cross_group_route_refs=tuple(route.id for route in routes if route.leaves_group),
    )


def reconstruct_resource_load(
    topology: SwizzleTopologyView,
    flows: tuple[TopologyFlow, ...],
) -> tuple[ResourceLoad, ...]:
    """Rebuild exact candidate-specific resource work from frozen PairRoutes."""

    rank_set = {item.rank for item in topology.ranks}
    load = {resource.id: 0 for resource in topology.resources}
    for index, flow in enumerate(flows):
        if flow.source_rank not in rank_set or flow.destination_rank not in rank_set:
            raise SchemaError(
                "flow references a rank outside the topology",
                path=f"swizzle_flows[{index}]",
            )
        if flow.source_rank == flow.destination_rank:
            raise SchemaError(
                "flow endpoints must differ",
                path=f"swizzle_flows[{index}]",
            )
        if type(flow.logical_bytes) is not int or flow.logical_bytes <= 0:
            raise SchemaError(
                "logical_bytes must be a positive integer",
                path=f"swizzle_flows[{index}].logical_bytes",
            )
        route = topology.route(flow.source_rank, flow.destination_rank)
        for resource_id in route.resource_ids:
            load[resource_id] += flow.logical_bytes
    return tuple(
        ResourceLoad(resource_id, load[resource_id])
        for resource_id in sorted(load)
    )


def canonical_bidirectional_pairs(
    order: tuple[int, ...],
    step: int,
) -> tuple[tuple[int, int], ...]:
    """Return real line transfers for one outward bidirectional wave.

    Unlike a logical ring this helper never wraps the two endpoints.  ``step``
    expands from both ends of the line and is primarily useful for canonical
    Wang witnesses and analytical resource accounting.
    """

    if type(step) is not int or step < 0:
        raise SchemaError("step must be a non-negative integer", path="step")
    if len(order) < 2 or step >= len(order) - 1:
        return ()
    pairs: list[tuple[int, int]] = []
    left = step
    right = len(order) - 1 - step
    if left + 1 < len(order):
        pairs.append((order[left], order[left + 1]))
    if right - 1 >= 0:
        reverse_pair = (order[right], order[right - 1])
        if reverse_pair not in pairs:
            pairs.append(reverse_pair)
    return tuple(pairs)


__all__ = [
    "ResourceLoad",
    "SwizzleTopologyView",
    "TopologyFlow",
    "TopologyRank",
    "TopologyResource",
    "TopologyRoute",
    "build_topology_view",
    "canonical_bidirectional_pairs",
    "reconstruct_resource_load",
]
