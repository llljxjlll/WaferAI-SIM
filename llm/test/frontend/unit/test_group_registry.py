from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes import (
    build_group_registry as public_build_group_registry,
    validate_group_against as public_validate_group_against,
)
from llm.frontend.wafer_frontend.passes.group_registry import (
    build_group_registry,
    validate_group_against,
)
from llm.frontend.wafer_frontend.schema.common import MeshAxisName
from llm.frontend.wafer_frontend.schema.experiment import (
    ExplicitGroupPlacement,
    PlacementSpec,
    PlacementStrategy,
)
from llm.frontend.wafer_frontend.schema.ir0 import (
    DeviceMesh,
    IR0,
    JobKind,
    LogicalInstance,
    LogicalRole,
    MeshAxis,
    ParallelAxes,
)
from llm.frontend.wafer_frontend.schema.ir1 import (
    C2CPort,
    CoreSpec,
    D2DLink,
    DieSpec,
    Direction,
    PhysicalFabric,
    ResourceCapacity,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext

from _fixtures import static_profile, valid_ir1


def mesh_graph(size: int, *, axis: MeshAxisName = MeshAxisName.TP) -> IR0:
    mesh = DeviceMesh("mesh_tp", (MeshAxis(axis, size),))
    instance = LogicalInstance(
        id="P0",
        role=LogicalRole.PREFILL,
        replicas=1,
        parallel=ParallelAxes(tp=size, sp=size > 1, dp=1, pp=1, ep=1),
        meshes=(mesh,),
    )
    result = IR0.create(
        producer_pass="n3b_fixture",
        job=JobKind.INFER,
        instances=(instance,),
        nodes=(),
        values=(),
        edges=(),
        fusion_candidates=(),
        profile=static_profile(),
    )
    result.validate()
    return result


def mesh_fabric(width: int, height: int, *, capacity: int = 16) -> PhysicalFabric:
    base = valid_ir1().fabric
    noc_grid = base.dies[0].noc_grid
    cores_per_die = noc_grid[0] * noc_grid[1]

    def cores(die_id: int) -> tuple[CoreSpec, ...]:
        return tuple(
            replace(
                core,
                id=f"core_{die_id}_{core.local_core_id}",
                runtime_core_id=die_id * cores_per_die + core.local_core_id,
            )
            for core in base.dies[0].cores
        )

    def port(die_id: int, direction: Direction, runtime_id: int) -> C2CPort:
        x = {
            Direction.WEST: 0,
            Direction.EAST: noc_grid[0] - 1,
            Direction.NORTH: 1,
            Direction.SOUTH: 1,
        }[direction]
        y = {
            Direction.SOUTH: 0,
            Direction.NORTH: noc_grid[1] - 1,
            Direction.WEST: 1,
            Direction.EAST: 1,
        }[direction]
        return C2CPort(
            id=f"{direction.value}_{die_id}",
            runtime_port_id=runtime_id,
            side=direction,
            direction=direction,
            noc_coord=(x, y),
            egress_resource_id=f"port_{die_id}_{direction.value}",
            bytes_per_cycle=capacity,
            buffer_packets=8,
        )

    dies: list[DieSpec] = []
    for y in range(height):
        for x in range(width):
            die_id = y * width + x
            directions: list[Direction] = []
            if x > 0:
                directions.append(Direction.WEST)
            if x + 1 < width:
                directions.append(Direction.EAST)
            if y > 0:
                directions.append(Direction.SOUTH)
            if y + 1 < height:
                directions.append(Direction.NORTH)
            dies.append(
                DieSpec(
                    id=die_id,
                    coord=(x, y),
                    noc_grid=noc_grid,
                    noc_bytes_per_cycle=128,
                    hbm_bytes_per_cycle=128,
                    cores=cores(die_id),
                    ports=tuple(
                        port(die_id, direction, index)
                        for index, direction in enumerate(directions)
                    ),
                )
            )

    direction_by_delta = {
        (1, 0): (Direction.EAST, Direction.WEST),
        (-1, 0): (Direction.WEST, Direction.EAST),
        (0, 1): (Direction.NORTH, Direction.SOUTH),
        (0, -1): (Direction.SOUTH, Direction.NORTH),
    }
    links: list[D2DLink] = []
    for source in dies:
        for destination in dies:
            delta = (
                destination.coord[0] - source.coord[0],
                destination.coord[1] - source.coord[1],
            )
            if delta not in direction_by_delta:
                continue
            source_direction, destination_direction = direction_by_delta[delta]
            links.append(
                D2DLink(
                    id=f"link_{source.id}_{destination.id}",
                    source_die=source.id,
                    source_port_ref=f"{source_direction.value}_{source.id}",
                    destination_die=destination.id,
                    destination_port_ref=(
                        f"{destination_direction.value}_{destination.id}"
                    ),
                    bytes_per_cycle=capacity,
                    latency_cycles=2,
                    resource_id=f"d2d_{source.id}_{destination.id}",
                    link_group_ref=f"cut_{source.id}_{destination.id}",
                )
            )
    result = PhysicalFabric(
        routing_mode=base.routing_mode,
        die_grid=(width, height),
        sram_profiles=base.sram_profiles,
        dies=tuple(dies),
        links=tuple(links),
    )
    result.validate("fabric")
    return result


def context(
    fabric: PhysicalFabric,
    placement: PlacementSpec | None = None,
) -> PlacementContext:
    result = PlacementContext.create(
        producer_pass="n3b_fixture",
        fabric=fabric,
        placement=placement or PlacementSpec(PlacementStrategy.COMPACT, ()),
    )
    result.validate()
    return result


class GroupRegistryTest(unittest.TestCase):
    def test_public_exports_and_tp1_empty_embedding(self) -> None:
        self.assertIs(public_build_group_registry, build_group_registry)
        self.assertIs(public_validate_group_against, validate_group_against)
        group = build_group_registry(mesh_graph(1), context(mesh_fabric(2, 1)))[0]
        self.assertEqual(tuple(item.die_id for item in group.placements), (0,))
        self.assertEqual(group.embedding.routes, ())
        self.assertEqual(group.embedding.resource_capacities, ())
        self.assertEqual(group.embedding.canonical_profiles, ())

    def test_compact_2x1_routes_resources_and_lane_are_exact(self) -> None:
        graph = mesh_graph(2)
        placement_context = context(mesh_fabric(2, 1))
        group = build_group_registry(graph, placement_context)[0]
        validate_group_against(graph, placement_context, group)

        self.assertEqual(
            tuple((item.rank, item.die_id, item.logical_coord) for item in group.placements),
            ((0, 0, (0,)), (1, 1, (1,))),
        )
        self.assertEqual(
            tuple(route.die_path for route in group.embedding.routes),
            ((0, 1), (1, 0)),
        )
        self.assertEqual(
            group.embedding.routes[0].hops[0].resource_ids,
            ("port_0_east", "d2d_0_1", "cut_0_1"),
        )
        profile = group.embedding.canonical_profiles[0]
        self.assertEqual(profile.traffic_template, "direct_a2a_unit_chunk/v1")
        self.assertEqual(
            tuple((weight.source_rank, weight.destination_rank, weight.normalized_bytes)
                  for weight in profile.flow_weights),
            ((0, 1, 1.0), (1, 0, 1.0)),
        )
        self.assertTrue(all(work.normalized_bytes == 1.0 for work in profile.resource_work))
        self.assertEqual(profile.bottleneck_resource, "port_0_east")
        self.assertEqual(profile.lane_eq_bandwidth, 16.0)

    def test_compact_2x2_path_table_and_lane_numeric_golden(self) -> None:
        graph = mesh_graph(4)
        placement_context = context(mesh_fabric(2, 2))
        group = build_group_registry(graph, placement_context)[0]
        path_table = {
            (route.source_rank, route.destination_rank): route.die_path
            for route in group.embedding.routes
        }
        self.assertEqual(
            path_table,
            {
                (0, 1): (0, 1),
                (0, 2): (0, 2),
                (0, 3): (0, 1, 3),
                (1, 0): (1, 0),
                (1, 2): (1, 0, 2),
                (1, 3): (1, 3),
                (2, 0): (2, 0),
                (2, 1): (2, 3, 1),
                (2, 3): (2, 3),
                (3, 0): (3, 2, 0),
                (3, 1): (3, 1),
                (3, 2): (3, 2),
            },
        )
        profile = group.embedding.canonical_profiles[0]
        work = {item.resource_id: item.normalized_bytes for item in profile.resource_work}
        self.assertEqual(work["port_0_east"], 2.0)
        self.assertEqual(work["d2d_0_1"], 2.0)
        self.assertEqual(profile.bottleneck_resource, "port_0_east")
        self.assertEqual(profile.lane_eq_bandwidth, 8.0)

    def test_explicit_rank_order_is_preserved_before_xy_routing(self) -> None:
        graph = mesh_graph(2)
        placement = PlacementSpec(
            PlacementStrategy.EXPLICIT,
            (ExplicitGroupPlacement("P0", "mesh_tp", (3, 0)),),
        )
        placement_context = context(mesh_fabric(2, 2), placement)
        group = build_group_registry(graph, placement_context)[0]
        self.assertEqual(tuple(item.die_id for item in group.placements), (3, 0))
        self.assertEqual(
            tuple(route.die_path for route in group.embedding.routes),
            ((3, 2, 0), (0, 1, 3)),
        )

    def test_compact_is_row_major_even_if_fabric_tuple_is_shuffled(self) -> None:
        fabric = mesh_fabric(2, 2)
        shuffled = replace(fabric, dies=tuple(reversed(fabric.dies)))
        shuffled.validate("fabric")
        group = build_group_registry(mesh_graph(2), context(shuffled))[0]
        self.assertEqual(tuple(item.die_id for item in group.placements), (0, 1))

    def test_recomputation_rejects_missing_extra_and_reordered_routes(self) -> None:
        graph = mesh_graph(4)
        placement_context = context(mesh_fabric(2, 2))
        group = build_group_registry(graph, placement_context)[0]
        routes = group.embedding.routes
        extra = replace(routes[0], id="extra_route")
        mutations = (
            routes[:-1],
            routes + (extra,),
            (routes[1], routes[0]) + routes[2:],
        )
        for mutation in mutations:
            with self.subTest(count=len(mutation)), self.assertRaisesRegex(
                SchemaError, "ordered pair route"
            ):
                validate_group_against(
                    graph,
                    placement_context,
                    replace(
                        group,
                        embedding=replace(group.embedding, routes=mutation),
                    ),
                )

    def test_recomputation_rejects_capacity_and_profile_mutations(self) -> None:
        graph = mesh_graph(4)
        placement_context = context(mesh_fabric(2, 2))
        group = build_group_registry(graph, placement_context)[0]
        capacities = group.embedding.resource_capacities
        reordered = (capacities[1], capacities[0]) + capacities[2:]
        extra = capacities + (ResourceCapacity("unused_capacity", 16),)
        for mutation in (reordered, extra):
            with self.subTest(capacity_count=len(mutation)), self.assertRaisesRegex(
                SchemaError, "used resource capacities"
            ):
                validate_group_against(
                    graph,
                    placement_context,
                    replace(
                        group,
                        embedding=replace(
                            group.embedding,
                            resource_capacities=mutation,
                        ),
                    ),
                )

        profile = group.embedding.canonical_profiles[0]
        work = profile.resource_work
        profiles = (
            replace(profile, lane_eq_bandwidth=9.0),
            replace(profile, bottleneck_resource=capacities[-1].id),
            replace(
                profile,
                resource_work=(
                    replace(work[0], normalized_bytes=work[0].normalized_bytes + 1.0),
                ) + work[1:],
            ),
        )
        for mutation in profiles:
            with self.subTest(profile=mutation), self.assertRaisesRegex(
                SchemaError, "canonical profile"
            ):
                validate_group_against(
                    graph,
                    placement_context,
                    replace(
                        group,
                        embedding=replace(
                            group.embedding,
                            canonical_profiles=(mutation,),
                        ),
                    ),
                )

    def test_recomputation_rejects_wrong_placement(self) -> None:
        graph = mesh_graph(2)
        placement_context = context(mesh_fabric(2, 2))
        group = build_group_registry(graph, placement_context)[0]
        with self.assertRaises(SchemaError):
            validate_group_against(
                graph,
                placement_context,
                replace(
                    group,
                    placements=(
                        replace(group.placements[0], die_id=2),
                        group.placements[1],
                    ),
                ),
            )

    def test_wrong_explicit_key_size_and_non_tp_mesh_fail_closed(self) -> None:
        fabric = mesh_fabric(2, 2)
        graph = mesh_graph(2)
        wrong_key = PlacementSpec(
            PlacementStrategy.EXPLICIT,
            (ExplicitGroupPlacement("Q0", "mesh_tp", (0, 1)),),
        )
        with self.assertRaisesRegex(SchemaError, "exactly the IR-0"):
            build_group_registry(graph, context(fabric, wrong_key))

        wrong_size = PlacementSpec(
            PlacementStrategy.EXPLICIT,
            (ExplicitGroupPlacement("P0", "mesh_tp", (0, 1, 2)),),
        )
        with self.assertRaisesRegex(SchemaError, "length.*TP mesh size"):
            build_group_registry(graph, context(fabric, wrong_size))

        with self.assertRaisesRegex(UnsupportedFeatureError, "one-dimensional TP"):
            build_group_registry(mesh_graph(2, axis=MeshAxisName.EP), context(fabric))


if __name__ == "__main__":
    unittest.main()
