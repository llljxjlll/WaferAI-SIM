"""N3b deterministic physical group placement and static embedding.

The current frontend scope is deliberately small: one logical instance, one
one-dimensional TP mesh, and the versioned ``direct_a2a_unit_chunk/v1``
traffic template. Every field in the resulting :class:`PhysicalGroup` is
derived from the IR-0 graph and :class:`PlacementContext`.
"""

from __future__ import annotations

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.common import MeshAxisName
from ..schema.experiment import PlacementStrategy
from ..schema.ir0 import IR0, DeviceMesh, LogicalInstance
from ..schema.ir1 import (
    CanonicalBandwidthProfile,
    CrossGroupRoute,
    D2DLink,
    FlowWeight,
    GroupEmbedding,
    PairRoute,
    PhysicalFabric,
    PhysicalGroup,
    RankPlacement,
    ResourceCapacity,
    ResourceWork,
    RouteHop,
)
from ..schema.placement import PlacementContext, TrafficTemplate
from ..schema.stage4_pd import Stage4PdMode, Stage4PdPlan


def _unsupported(message: str, *, path: str) -> None:
    raise UnsupportedFeatureError(message, path=path)


def _scopes(graph: IR0) -> tuple[tuple[LogicalInstance, DeviceMesh], ...]:
    """Return every canonical instance/TP-mesh scope in source order."""

    scopes: list[tuple[LogicalInstance, DeviceMesh]] = []
    for index, instance in enumerate(graph.instances):
        instance_path = f"ir0.instances[{index}]"
        if instance.replicas != 1:
            _unsupported(
                "N3b requires a single replica per logical instance",
                path=f"{instance_path}.replicas",
            )
        if len(instance.meshes) != 1:
            _unsupported(
                "N3b requires exactly one device mesh per logical instance",
                path=f"{instance_path}.meshes",
            )
        mesh = instance.meshes[0]
        if len(mesh.axes) != 1 or mesh.axes[0].name is not MeshAxisName.TP:
            _unsupported(
                "N3b requires a one-dimensional TP mesh",
                path=f"{instance_path}.meshes[0].axes",
            )
        if mesh.axes[0].size != instance.parallel.tp:
            raise SchemaError(
                "TP mesh size must equal instance.parallel.tp",
                path=f"{instance_path}.meshes[0].axes[0].size",
            )
        scopes.append((instance, mesh))
    return tuple(scopes)


def _placement_die_ids(
    instance: LogicalInstance,
    mesh: DeviceMesh,
    context: PlacementContext,
) -> tuple[int, ...]:
    rank_count = mesh.axes[0].size
    placement = context.placement
    if placement.strategy is PlacementStrategy.COMPACT:
        row_major_dies = sorted(context.fabric.dies, key=lambda die: die.id)
        die_ids = tuple(die.id for die in row_major_dies[:rank_count])
        if len(die_ids) != rank_count:
            raise SchemaError(
                f"TP group requires {rank_count} dies but fabric has {len(context.fabric.dies)}",
                path="placement.strategy",
            )
        return die_ids

    key = (instance.id, mesh.id)
    matches = tuple(
        item
        for item in placement.groups
        if (item.instance_id, item.mesh_ref) == key
    )
    if len(matches) != 1:
        raise SchemaError(
            "explicit placement must contain the exact IR-0 instance/mesh group",
            path="placement.groups",
        )
    die_ids = matches[0].die_ids
    if len(die_ids) != rank_count:
        raise SchemaError(
            f"explicit die list length must equal TP mesh size {rank_count}",
            path="placement.groups[0].die_ids",
        )
    return die_ids


def _xy_path(
    fabric: PhysicalFabric,
    source_die: int,
    destination_die: int,
) -> tuple[int, ...]:
    dies = {die.id: die for die in fabric.dies}
    x, y = dies[source_die].coord
    destination_x, destination_y = dies[destination_die].coord
    path = [source_die]
    while x != destination_x:
        x += 1 if destination_x > x else -1
        path.append(y * fabric.die_grid[0] + x)
    while y != destination_y:
        y += 1 if destination_y > y else -1
        path.append(y * fabric.die_grid[0] + x)
    return tuple(path)


def _link_index(fabric: PhysicalFabric) -> dict[tuple[int, int], D2DLink]:
    result: dict[tuple[int, int], D2DLink] = {}
    for index, link in enumerate(fabric.links):
        key = (link.source_die, link.destination_die)
        if key in result:
            raise SchemaError(
                "backend-v1 requires one directed link per adjacent die pair",
                path=f"placement_context.fabric.links[{index}]",
            )
        result[key] = link
    return result


def _capacity_catalog(fabric: PhysicalFabric) -> dict[str, int]:
    """Return exact capacities and reject inconsistent shared-cut aliases."""

    result: dict[str, int] = {}

    def add(resource_id: str, capacity: int, path: str) -> None:
        previous = result.get(resource_id)
        if previous is not None and previous != capacity:
            raise SchemaError(
                "one physical resource must have one exact capacity",
                path=path,
            )
        result[resource_id] = capacity

    for die_index, die in enumerate(fabric.dies):
        for port_index, port in enumerate(die.ports):
            add(
                port.egress_resource_id,
                port.bytes_per_cycle,
                f"placement_context.fabric.dies[{die_index}].ports[{port_index}]",
            )
    for link_index, link in enumerate(fabric.links):
        path = f"placement_context.fabric.links[{link_index}]"
        add(link.resource_id, link.bytes_per_cycle, path)
        if link.link_group_ref is not None:
            add(link.link_group_ref, link.bytes_per_cycle, f"{path}.link_group_ref")
    return result


def _route(
    *,
    group_id: str,
    source_rank: int,
    destination_rank: int,
    rank_to_die: tuple[int, ...],
    fabric: PhysicalFabric,
    links: dict[tuple[int, int], D2DLink],
) -> PairRoute:
    die_path = _xy_path(
        fabric,
        rank_to_die[source_rank],
        rank_to_die[destination_rank],
    )
    port_indexes = {
        die.id: {port.id: port for port in die.ports} for die in fabric.dies
    }
    hops: list[RouteHop] = []
    route_resources: list[str] = []
    seen_route_resources: set[str] = set()
    for hop_index, (source_die, destination_die) in enumerate(
        zip(die_path, die_path[1:])
    ):
        link = links.get((source_die, destination_die))
        if link is None:
            raise SchemaError(
                f"missing directed link {source_die}->{destination_die} on X-then-Y path",
                path="placement_context.fabric.links",
            )
        source_port = port_indexes[source_die][link.source_port_ref]
        resources = (source_port.egress_resource_id, link.resource_id)
        if link.link_group_ref is not None:
            resources += (link.link_group_ref,)
        hops.append(
            RouteHop(
                index=hop_index,
                link_ref=link.id,
                source_die=source_die,
                source_port_ref=link.source_port_ref,
                destination_die=destination_die,
                destination_port_ref=link.destination_port_ref,
                resource_ids=resources,
            )
        )
        for resource_id in resources:
            if resource_id not in seen_route_resources:
                seen_route_resources.add(resource_id)
                route_resources.append(resource_id)
    return PairRoute(
        id=f"route__{group_id}__r{source_rank}__r{destination_rank}",
        source_rank=source_rank,
        destination_rank=destination_rank,
        die_path=die_path,
        hops=tuple(hops),
        resource_ids=tuple(route_resources),
    )


def _expected_group(
    graph: IR0,
    context: PlacementContext,
    instance: LogicalInstance,
    mesh: DeviceMesh,
    *,
    rank_to_die_override: tuple[int, ...] | None = None,
    group_id_suffix: str = "",
) -> PhysicalGroup:
    rank_to_die = (
        _placement_die_ids(instance, mesh, context)
        if rank_to_die_override is None
        else rank_to_die_override
    )
    if len(rank_to_die) != mesh.axes[0].size:
        raise SchemaError(
            "rank_to_die override must exactly cover the TP mesh",
            path="placement.train_replica",
        )
    group_id = f"group__{instance.id}__{mesh.id}{group_id_suffix}"
    placements = tuple(
        RankPlacement(rank=rank, die_id=die_id, logical_coord=(rank,))
        for rank, die_id in enumerate(rank_to_die)
    )
    rank_count = len(rank_to_die)
    if rank_count == 1:
        return PhysicalGroup(
            id=group_id,
            instance_id=instance.id,
            mesh_ref=mesh.id,
            axis=MeshAxisName.TP,
            logical_shape=(rank_count,),
            placements=placements,
            embedding=GroupEmbedding((), (), ()),
        )

    link_index = _link_index(context.fabric)
    capacity_catalog = _capacity_catalog(context.fabric)
    routes = tuple(
        _route(
            group_id=group_id,
            source_rank=source_rank,
            destination_rank=destination_rank,
            rank_to_die=rank_to_die,
            fabric=context.fabric,
            links=link_index,
        )
        for source_rank in range(rank_count)
        for destination_rank in range(rank_count)
        if source_rank != destination_rank
    )

    used_resource_ids: list[str] = []
    used_resource_set: set[str] = set()
    for route in routes:
        for resource_id in route.resource_ids:
            if resource_id not in used_resource_set:
                used_resource_set.add(resource_id)
                used_resource_ids.append(resource_id)
    capacities = tuple(
        ResourceCapacity(resource_id, capacity_catalog[resource_id])
        for resource_id in used_resource_ids
    )
    flow_weights = tuple(
        FlowWeight(route.source_rank, route.destination_rank, 1.0)
        for route in routes
    )
    work_by_resource = {resource_id: 0.0 for resource_id in used_resource_ids}
    for route, weight in zip(routes, flow_weights):
        for resource_id in route.resource_ids:
            work_by_resource[resource_id] += weight.normalized_bytes
    resource_work = tuple(
        ResourceWork(resource_id, work_by_resource[resource_id])
        for resource_id in used_resource_ids
    )
    ratios = tuple(
        capacity.bytes_per_cycle / work_by_resource[capacity.id]
        for capacity in capacities
    )
    bottleneck_index = min(range(len(capacities)), key=ratios.__getitem__)
    template = TrafficTemplate.DIRECT_A2A_UNIT_CHUNK_V1.value
    profile = CanonicalBandwidthProfile(
        id=f"canonical__{group_id}__direct_a2a_unit_chunk_v1",
        traffic_template=template,
        flow_weights=flow_weights,
        resource_work=resource_work,
        bottleneck_resource=capacities[bottleneck_index].id,
        lane_eq_bandwidth=ratios[bottleneck_index],
    )
    return PhysicalGroup(
        id=group_id,
        instance_id=instance.id,
        mesh_ref=mesh.id,
        axis=MeshAxisName.TP,
        logical_shape=(rank_count,),
        placements=placements,
        embedding=GroupEmbedding(routes, capacities, (profile,)),
    )


def build_train_replica_groups(
    graph: IR0,
    context: PlacementContext,
) -> tuple[PhysicalGroup, ...]:
    """Build one disjoint TP group per DP replica for N6.2 TRAIN placement."""

    graph.validate("ir0")
    context.validate("placement_context")
    if context.placement.strategy is not PlacementStrategy.COMPACT:
        _unsupported(
            "N6.2 DP replica placement supports compact row-major placement only",
            path="placement.strategy",
        )
    if len(graph.instances) != 1:
        raise SchemaError(
            "N6.2 TRAIN placement requires exactly one logical instance",
            path="ir0.instances",
        )
    instance = graph.instances[0]
    if len(instance.meshes) != 1:
        raise SchemaError(
            "N6.2 TRAIN placement requires one TP mesh",
            path="ir0.instances[0].meshes",
        )
    mesh = instance.meshes[0]
    tp = instance.parallel.tp
    dp = instance.parallel.dp
    ordered_dies = tuple(
        die.id for die in sorted(context.fabric.dies, key=lambda item: item.id)
    )
    required = tp * dp
    if len(ordered_dies) < required:
        raise SchemaError(
            f"DPxTP placement requires {required} dies but fabric has {len(ordered_dies)}",
            path="placement.strategy",
        )
    return tuple(
        _expected_group(
            graph,
            context,
            instance,
            mesh,
            rank_to_die_override=ordered_dies[
                replica_index * tp : (replica_index + 1) * tp
            ],
            group_id_suffix=f"__dp{replica_index}",
        )
        for replica_index in range(dp)
    )


def validate_group_against(
    graph: IR0,
    context: PlacementContext,
    group: PhysicalGroup,
    *,
    path: str = "physical_group",
) -> None:
    """Strictly validate a group by recomputing every derived field."""

    if type(graph) is not IR0:
        raise SchemaError("must be an IR0", path="ir0")
    if type(context) is not PlacementContext:
        raise SchemaError("must be a PlacementContext", path="placement_context")
    if type(group) is not PhysicalGroup:
        raise SchemaError("must be a PhysicalGroup", path=path)
    graph.validate("ir0")
    context.validate("placement_context")
    group.validate(path)
    scopes = {
        (instance.id, mesh.id): (instance, mesh)
        for instance, mesh in _scopes(graph)
    }
    scope = scopes.get((group.instance_id, group.mesh_ref))
    if scope is None:
        raise SchemaError(
            "does not identify an IR-0 instance/mesh scope",
            path=path,
        )
    expected = _expected_group(graph, context, *scope)
    for field_name in (
        "id",
        "instance_id",
        "mesh_ref",
        "axis",
        "logical_shape",
    ):
        if getattr(group, field_name) != getattr(expected, field_name):
            raise SchemaError(
                "does not match the IR-0 group identity",
                path=f"{path}.{field_name}",
            )
    if group.placements != expected.placements:
        raise SchemaError(
            "must exactly match deterministic compact/explicit rank placement",
            path=f"{path}.placements",
        )
    if group.embedding.routes != expected.embedding.routes:
        raise SchemaError(
            "must contain every ordered pair route in canonical order",
            path=f"{path}.embedding.routes",
        )
    if group.embedding.resource_capacities != expected.embedding.resource_capacities:
        raise SchemaError(
            "must contain exactly the used resource capacities in canonical order",
            path=f"{path}.embedding.resource_capacities",
        )
    if group.embedding.canonical_profiles != expected.embedding.canonical_profiles:
        raise SchemaError(
            "canonical profile weights, work, bottleneck, or lane bandwidth disagree with recomputation",
            path=f"{path}.embedding.canonical_profiles",
        )


def build_group_registry(
    graph: IR0,
    context: PlacementContext,
) -> tuple[PhysicalGroup, ...]:
    """Build one exact physical group for every logical instance TP mesh."""

    if type(graph) is not IR0:
        raise SchemaError("must be an IR0", path="ir0")
    if type(context) is not PlacementContext:
        raise SchemaError("must be a PlacementContext", path="placement_context")
    graph.validate("ir0")
    context.validate("placement_context")
    scopes = _scopes(graph)
    expected_keys = {(instance.id, mesh.id) for instance, mesh in scopes}
    if (
        len(scopes) > 1
        and context.placement.strategy is PlacementStrategy.COMPACT
    ):
        _unsupported(
            "multi-instance placement requires explicit disjoint die groups",
            path="placement.strategy",
        )
    if context.placement.strategy is PlacementStrategy.EXPLICIT:
        actual_keys = {
            (item.instance_id, item.mesh_ref)
            for item in context.placement.groups
        }
        if actual_keys != expected_keys:
            raise SchemaError(
                "explicit placement must contain exactly the IR-0 instance/mesh groups",
                path="placement.groups",
            )
        die_ids = tuple(
            die_id
            for item in context.placement.groups
            for die_id in item.die_ids
        )
        if len(set(die_ids)) != len(die_ids):
            raise SchemaError(
                "multi-instance physical groups must use disjoint dies",
                path="placement.groups",
            )
    groups = tuple(
        _expected_group(graph, context, instance, mesh)
        for instance, mesh in scopes
    )
    for index, group in enumerate(groups):
        validate_group_against(
            graph,
            context,
            group,
            path=f"physical_groups[{index}]",
        )
    return groups


def stage4_route_endpoint_pairs(
    plan: Stage4PdPlan,
) -> tuple[tuple[int, int], ...]:
    """Return the exact distinct rank endpoints requested by Stage 4 flows."""

    if type(plan) is not Stage4PdPlan:
        raise SchemaError("must be a Stage4PdPlan", path="stage4_pd_plan")
    plan.validate("stage4_pd_plan")
    if plan.mode is not Stage4PdMode.SEPARATED:
        _unsupported(
            "cross-group routes require a separated Stage 4 plan",
            path="stage4_pd_plan.mode",
        )
    return tuple(
        sorted(
            {
                (flow.source_rank, flow.destination_rank)
                for handoff in plan.handoffs
                for flow in handoff.flows
            }
        )
    )


def _cross_group_route(
    *,
    source_group: PhysicalGroup,
    source_rank: int,
    destination_group: PhysicalGroup,
    destination_rank: int,
    fabric: PhysicalFabric,
    links: dict[tuple[int, int], D2DLink],
) -> CrossGroupRoute:
    source_die = next(
        placement.die_id
        for placement in source_group.placements
        if placement.rank == source_rank
    )
    destination_die = next(
        placement.die_id
        for placement in destination_group.placements
        if placement.rank == destination_rank
    )
    die_path = _xy_path(fabric, source_die, destination_die)
    port_indexes = {
        die.id: {port.id: port for port in die.ports}
        for die in fabric.dies
    }
    hops: list[RouteHop] = []
    route_resources: list[str] = []
    for hop_index, (hop_source, hop_destination) in enumerate(
        zip(die_path, die_path[1:])
    ):
        link = links.get((hop_source, hop_destination))
        if link is None:
            raise SchemaError(
                f"missing directed link {hop_source}->{hop_destination} on X-then-Y path",
                path="placement_context.fabric.links",
            )
        source_port = port_indexes[hop_source][link.source_port_ref]
        resources = (source_port.egress_resource_id, link.resource_id)
        if link.link_group_ref is not None:
            resources += (link.link_group_ref,)
        hops.append(
            RouteHop(
                index=hop_index,
                link_ref=link.id,
                source_die=hop_source,
                source_port_ref=link.source_port_ref,
                destination_die=hop_destination,
                destination_port_ref=link.destination_port_ref,
                resource_ids=resources,
            )
        )
        for resource_id in resources:
            if resource_id not in route_resources:
                route_resources.append(resource_id)
    return CrossGroupRoute.create(
        source_group_ref=source_group.id,
        source_rank=source_rank,
        destination_group_ref=destination_group.id,
        destination_rank=destination_rank,
        die_path=die_path,
        hops=tuple(hops),
        resource_ids=tuple(route_resources),
    )


def build_stage4_cross_group_routes(
    plan: Stage4PdPlan,
    context: PlacementContext,
    groups: tuple[PhysicalGroup, ...],
) -> tuple[CrossGroupRoute, ...]:
    """Build one physical route per distinct Stage 4 flow endpoint pair."""

    if type(context) is not PlacementContext:
        raise SchemaError("must be a PlacementContext", path="placement_context")
    if type(groups) is not tuple:
        raise SchemaError("must be an immutable tuple", path="groups")
    context.validate("placement_context")
    endpoint_pairs = stage4_route_endpoint_pairs(plan)
    by_instance: dict[str, list[PhysicalGroup]] = {}
    for index, group in enumerate(groups):
        if type(group) is not PhysicalGroup:
            raise SchemaError("must be a PhysicalGroup", path=f"groups[{index}]")
        group.validate(f"groups[{index}]")
        by_instance.setdefault(group.instance_id, []).append(group)
    source_matches = by_instance.get(plan.prefill_instance_ref, [])
    destination_matches = by_instance.get(plan.decode_instance_ref, [])
    if len(source_matches) != 1 or len(destination_matches) != 1:
        raise SchemaError(
            "selected Stage 4 endpoints must each own exactly one physical group",
            path="groups",
        )
    source_group = source_matches[0]
    destination_group = destination_matches[0]
    if source_group.logical_shape != (plan.prefill_tp,):
        raise SchemaError(
            "prefill group size must equal plan.prefill_tp",
            path="groups",
        )
    if destination_group.logical_shape != (plan.decode_tp,):
        raise SchemaError(
            "decode group size must equal plan.decode_tp",
            path="groups",
        )
    links = _link_index(context.fabric)
    routes = tuple(
        _cross_group_route(
            source_group=source_group,
            source_rank=source_rank,
            destination_group=destination_group,
            destination_rank=destination_rank,
            fabric=context.fabric,
            links=links,
        )
        for source_rank, destination_rank in endpoint_pairs
    )
    group_index = {group.id: group for group in groups}
    for index, route in enumerate(routes):
        route.validate_against(
            context.fabric,
            group_index,
            f"cross_group_routes[{index}]",
        )
    return routes


__all__ = [
    "build_group_registry",
    "build_stage4_cross_group_routes",
    "stage4_route_endpoint_pairs",
    "validate_group_against",
]
