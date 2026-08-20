from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.action import SyncContract
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.ir1 import (
    C2CPort,
    CoreSpec,
    D2DLink,
    DieSpec,
    Direction,
    GroupEmbedding,
    IR1,
    PairRoute,
    PhysicalFabric,
    RankPlacement,
    ResourceCapacity,
    RouteHop,
    RoutingMode,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferAccess,
    BufferBinding,
    BufferOwnership,
    BufferUseRole,
    CoreOrder,
    FlowRouteBinding,
    FlowRouteRole,
    FusedNodeOrigin,
    IntraDieDAG,
    IntraDieRegion,
    IntraDieSchedule,
    IntraDieScheduleSet,
    IntraDieValue,
    IR2ProjectionResult,
    LogicalRuntimeBinding,
    OriginKind,
    PortLeg,
    RegionLowering,
    SemanticFlow,
    SemanticTask,
    SemanticTaskKind,
    TaskBufferUse,
    TaskPlacement,
    TensorSlice,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass

from _fixtures import valid_ir1


def _xy_path(
    source: tuple[int, int], destination: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    x, y = source
    destination_x, destination_y = destination
    result = [(x, y)]
    while x != destination_x:
        x += 1 if destination_x > x else -1
        result.append((x, y))
    while y != destination_y:
        y += 1 if destination_y > y else -1
        result.append((x, y))
    return tuple(result)


def _recreate_schedule(
    schedule: IntraDieSchedule, **changes: object
) -> IntraDieSchedule:
    fields = schedule._semantic_key()
    fields.update(changes)
    return IntraDieSchedule.create(producer_pass=schedule.producer_pass, **fields)


def _recreate_dag(dag: IntraDieDAG, **changes: object) -> IntraDieDAG:
    fields = dag._semantic_key()
    fields.update(changes)
    return IntraDieDAG.create(producer_pass=dag.producer_pass, **fields)


def _projection(ir1: IR1, dags: tuple[IntraDieDAG, ...]) -> IR2ProjectionResult:
    return IR2ProjectionResult.create(
        producer_pass="route_projection_fixture",
        source_ir1_id=ir1.id,
        fusion_plan_ids=("fp_route",),
        standalone_collective_plan_ids=(),
        dags=dags,
    )


def _schedule_set(
    ir1: IR1,
    projection: IR2ProjectionResult,
    schedules: tuple[IntraDieSchedule, ...],
) -> IntraDieScheduleSet:
    return IntraDieScheduleSet.create(
        producer_pass="route_schedule_fixture",
        source_projection_id=projection.id,
        source_ir1_id=ir1.id,
        schedules=schedules,
    )


def _transport_dag(
    ir1: IR1,
    *,
    die_id: int,
    kind: SemanticTaskKind,
    route: PairRoute,
) -> IntraDieDAG:
    task_id = {
        SemanticTaskKind.SEND: "task_send",
        SemanticTaskKind.RECV: "task_recv",
        SemanticTaskKind.TRANSIT: f"task_transit_{die_id}",
    }[kind]
    action_id = "send_action" if kind is not SemanticTaskKind.RECV else "recv_action"
    rank = 0 if kind is not SemanticTaskKind.RECV else 1
    tensor_slice = TensorSlice("v_flow", (0,), (16,))
    task = SemanticTask(
        id=task_id,
        kind=kind,
        origin_ref=FusedNodeOrigin(OriginKind.FUSED, "fp_route", rank, action_id),
        region_id=f"region_{die_id}",
        op_kind=OpKind.COLLECTIVE,
        member_id="p_rs_0",
        flow_id="flow_route",
        chunk_id=0,
        collective_step=0,
        source_rank=0,
        destination_rank=1,
        tensor_slice=tensor_slice,
        bytes=32,
        dtype=DType.FP16,
        shape=(16,),
        read_values=("v_source",) if kind is SemanticTaskKind.SEND else (),
        write_values=("v_destination",) if kind is SemanticTaskKind.RECV else (),
        compute=None,
        reduction=None,
        sync=SyncContract(f"event_{task_id}", None, None),
        deps=(),
    )
    return IntraDieDAG.create(
        producer_pass="route_projection_fixture",
        source_ir1_id=ir1.id,
        die_id=die_id,
        fusion_plan_ids=("fp_route",),
        standalone_collective_plan_ids=(),
        ordinary_node_ids=(),
        tasks=(task,),
        values=(
            (
                IntraDieValue(
                    "v_source",
                    "route_fixture_source",
                    (16,),
                    DType.FP16,
                    "flat",
                    ir1.values[0].sharding,
                    None,
                    (),
                    (task_id,),
                )
                if kind is SemanticTaskKind.SEND
                else IntraDieValue(
                    "v_destination",
                    "route_fixture_destination",
                    (16,),
                    DType.FP16,
                    "flat",
                    ir1.values[0].sharding,
                    None,
                    (task_id,),
                    (),
                )
            ),
        ) if kind is not SemanticTaskKind.TRANSIT else (),
        flows=(
            SemanticFlow(
                "flow_route",
                "channel_route",
                route.id,
                0,
                1,
                route.die_path[0],
                route.die_path[-1],
                route.die_path,
                tensor_slice,
                32,
                DType.FP16,
                (task_id,),
            ),
        ),
        regions=(
            IntraDieRegion(
                f"region_{die_id}",
                "fp_route",
                None,
                RegionLowering.ISA_REGION,
                (task_id,),
            ),
        ),
    )


def _empty_dag(ir1: IR1, die_id: int) -> IntraDieDAG:
    return IntraDieDAG.create(
        producer_pass="route_projection_fixture",
        source_ir1_id=ir1.id,
        die_id=die_id,
        fusion_plan_ids=(),
        standalone_collective_plan_ids=(),
        ordinary_node_ids=(),
        tasks=(),
        values=(),
        flows=(),
        regions=(),
    )


def _route_schedule(
    ir1: IR1, dag: IntraDieDAG, route: PairRoute
) -> IntraDieSchedule:
    die = next(item for item in ir1.fabric.dies if item.id == dag.die_id)
    if not dag.tasks:
        return IntraDieSchedule.create(
            producer_pass="route_schedule_fixture",
            dag_id=dag.id,
            die_id=dag.die_id,
            placements=(),
            buffer_bindings=(),
            task_buffer_uses=(),
            task_state_uses=(),
            flow_routes=(),
            runtime_bindings=(),
            core_orders=(),
        )
    position = route.die_path.index(dag.die_id)
    task = dag.tasks[0]
    if position == 0:
        role = FlowRouteRole.SOURCE
        ingress = None
        egress = PortLeg(route.hops[0].link_ref, route.hops[0].source_port_ref)
        start = die.cores[0].noc_coord
        end = next(port.noc_coord for port in die.ports if port.id == egress.port_ref)
    elif position == len(route.die_path) - 1:
        role = FlowRouteRole.DESTINATION
        ingress = PortLeg(route.hops[-1].link_ref, route.hops[-1].destination_port_ref)
        egress = None
        start = next(port.noc_coord for port in die.ports if port.id == ingress.port_ref)
        end = die.cores[0].noc_coord
    else:
        role = FlowRouteRole.TRANSIT
        ingress_hop = route.hops[position - 1]
        egress_hop = route.hops[position]
        ingress = PortLeg(ingress_hop.link_ref, ingress_hop.destination_port_ref)
        egress = PortLeg(egress_hop.link_ref, egress_hop.source_port_ref)
        start = next(port.noc_coord for port in die.ports if port.id == ingress.port_ref)
        end = next(port.noc_coord for port in die.ports if port.id == egress.port_ref)
    executable = task.kind is not SemanticTaskKind.TRANSIT
    core_id = die.cores[0].runtime_core_id
    if task.kind is SemanticTaskKind.SEND:
        value_id = "v_source"
        buffer_role = BufferUseRole.SEND_SOURCE
        buffer_access = BufferAccess.READ
    elif task.kind is SemanticTaskKind.RECV:
        value_id = "v_destination"
        buffer_role = BufferUseRole.RECV_DESTINATION
        buffer_access = BufferAccess.WRITE
    else:
        value_id = None
        buffer_role = None
        buffer_access = None
    bindings = (
        (
            BufferBinding(
                f"buffer_{task.id}",
                value_id,
                TensorSlice(value_id, (0,), (16,)),
                core_id,
                "sram_main",
                0,
                32,
                2,
                (0,),
                f"storage_{task.id}",
                None,
                (
                    BufferOwnership.OWNED
                    if buffer_access is BufferAccess.WRITE
                    else BufferOwnership.BORROWED
                ),
                0,
                1,
                DType.FP16,
                "flat",
            ),
        )
        if executable
        else ()
    )
    uses = (
        (
            TaskBufferUse(
                task.id,
                bindings[0].id,
                buffer_access,
                buffer_role,
                0,
                None,
                bindings[0].tensor_slice,
            ),
        )
        if executable
        else ()
    )
    runtime = (
        (
            LogicalRuntimeBinding(
                task.id,
                "flow_route",
                "channel_route",
                f"event_{task.id}",
                f"token_{task.id}",
            ),
        )
        if executable
        else ()
    )
    return IntraDieSchedule.create(
        producer_pass="route_schedule_fixture",
        dag_id=dag.id,
        die_id=dag.die_id,
        placements=(TaskPlacement(task.id, core_id),) if executable else (),
        buffer_bindings=bindings,
        task_buffer_uses=uses,
        task_state_uses=(),
        flow_routes=(
            FlowRouteBinding(
                "flow_route", route.id, role, ingress, egress, _xy_path(start, end)
            ),
        ),
        runtime_bindings=runtime,
        core_orders=(CoreOrder(core_id, (task.id,)),) if executable else (),
    )


def _two_die_case() -> tuple[
    IR1, IR2ProjectionResult, IntraDieScheduleSet
]:
    ir1 = valid_ir1()
    route = ir1.groups[0].embedding.routes[0]
    dags = (
        _transport_dag(ir1, die_id=0, kind=SemanticTaskKind.SEND, route=route),
        _transport_dag(ir1, die_id=1, kind=SemanticTaskKind.RECV, route=route),
    )
    projection = _projection(ir1, dags)
    schedules = tuple(_route_schedule(ir1, dag, route) for dag in dags)
    return ir1, projection, _schedule_set(ir1, projection, schedules)


def _port(
    port_id: str,
    runtime_port_id: int,
    direction: Direction,
    coord: tuple[int, int],
    resource_id: str,
) -> C2CPort:
    return C2CPort(
        port_id,
        runtime_port_id,
        direction,
        direction,
        coord,
        resource_id,
        64,
        8,
    )


def _two_by_two_ir1() -> IR1:
    template = valid_ir1()
    profile = template.fabric.sram_profiles[0]

    def cores(die_id: int) -> tuple[CoreSpec, ...]:
        return tuple(
            CoreSpec(
                f"core_{die_id}_{local_id}",
                local_id,
                die_id * 16 + local_id,
                (local_id % 4, local_id // 4),
                profile.id,
            )
            for local_id in range(16)
        )

    dies = (
        DieSpec(
            0,
            (0, 0),
            (4, 4),
            128,
            128,
            cores(0),
            (_port("east_0", 0, Direction.EAST, (3, 1), "port_0_e"),),
        ),
        DieSpec(
            1,
            (1, 0),
            (4, 4),
            128,
            128,
            cores(1),
            (
                _port("west_1", 0, Direction.WEST, (0, 1), "port_1_w"),
                _port("north_1", 1, Direction.NORTH, (2, 3), "port_1_n"),
            ),
        ),
        DieSpec(2, (0, 1), (4, 4), 128, 128, cores(2), ()),
        DieSpec(
            3,
            (1, 1),
            (4, 4),
            128,
            128,
            cores(3),
            (_port("south_3", 0, Direction.SOUTH, (2, 0), "port_3_s"),),
        ),
    )
    links = (
        D2DLink("link_0_1", 0, "east_0", 1, "west_1", 64, 2, "d2d_0_1", "cut_01"),
        D2DLink("link_1_0", 1, "west_1", 0, "east_0", 64, 2, "d2d_1_0", "cut_10"),
        D2DLink("link_1_3", 1, "north_1", 3, "south_3", 64, 2, "d2d_1_3", "cut_13"),
        D2DLink("link_3_1", 3, "south_3", 1, "north_1", 64, 2, "d2d_3_1", "cut_31"),
    )
    fabric = PhysicalFabric(
        RoutingMode.BACKEND_XY_V1,
        (2, 2),
        (profile,),
        dies,
        links,
    )
    hops = (
        RouteHop(
            0,
            "link_0_1",
            0,
            "east_0",
            1,
            "west_1",
            ("port_0_e", "d2d_0_1", "cut_01"),
        ),
        RouteHop(
            1,
            "link_1_3",
            1,
            "north_1",
            3,
            "south_3",
            ("port_1_n", "d2d_1_3", "cut_13"),
        ),
    )
    route = PairRoute(
        "route_0_3",
        0,
        1,
        (0, 1, 3),
        hops,
        (
            "port_0_e",
            "d2d_0_1",
            "cut_01",
            "port_1_n",
            "d2d_1_3",
            "cut_13",
        ),
    )
    embedding = GroupEmbedding(
        routes=(route,),
        resource_capacities=tuple(
            ResourceCapacity(resource_id, 64) for resource_id in route.resource_ids
        ),
        canonical_profiles=(),
    )
    group = replace(
        template.groups[0],
        placements=(RankPlacement(0, 0, (0,)), RankPlacement(1, 3, (1,))),
        embedding=embedding,
    )
    instance = replace(template.instances[0], die_region=(0, 1, 2, 3))
    fields = template._semantic_key()
    fields.update(fabric=fabric, groups=(group,), instances=(instance,))
    result = IR1.create(producer_pass=template.producer_pass, **fields)
    result.validate()
    return result


def _two_by_two_case() -> tuple[
    IR1, IR2ProjectionResult, IntraDieScheduleSet
]:
    ir1 = _two_by_two_ir1()
    route = ir1.groups[0].embedding.routes[0]
    dags = (
        _transport_dag(ir1, die_id=0, kind=SemanticTaskKind.SEND, route=route),
        _transport_dag(ir1, die_id=1, kind=SemanticTaskKind.TRANSIT, route=route),
        _empty_dag(ir1, 2),
        _transport_dag(ir1, die_id=3, kind=SemanticTaskKind.RECV, route=route),
    )
    projection = _projection(ir1, dags)
    schedules = tuple(_route_schedule(ir1, dag, route) for dag in dags)
    return ir1, projection, _schedule_set(ir1, projection, schedules)


class IR2RouteScheduleTest(unittest.TestCase):
    def test_two_die_schedule_set_is_complete_and_round_trips(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        schedule_set.validate_against(projection, ir1)
        decoded = loads_dataclass(
            IntraDieScheduleSet, canonical_json(schedule_set)
        )
        self.assertEqual(decoded, schedule_set)

    def test_transport_runtime_symbols_are_exact_and_complete(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        source = schedule_set.schedules[0]
        runtime = source.runtime_bindings[0]
        cases = (
            replace(runtime, event_symbol=None),
            replace(runtime, token_symbol=None),
            replace(runtime, channel_symbol="forged_channel"),
            replace(runtime, event_symbol="forged_event"),
        )
        for changed in cases:
            with self.subTest(binding=changed), self.assertRaisesRegex(
                SchemaError, "runtime binding|transport runtime"
            ):
                _recreate_schedule(
                    source, runtime_bindings=(changed,)
                ).validate_against(projection.dags[0], ir1)

    def test_two_die_role_leg_port_link_and_local_path_fail_closed(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        source = schedule_set.schedules[0]
        binding = source.flow_routes[0]
        invalid_bindings = (
            replace(
                binding,
                role=FlowRouteRole.DESTINATION,
                ingress=PortLeg("link_0_1", "east_0"),
                egress=None,
            ),
            replace(binding, egress=PortLeg("link_1_0", "east_0")),
            replace(binding, egress=PortLeg("link_0_1", "missing_port")),
        )
        for invalid in invalid_bindings:
            changed = _recreate_schedule(source, flow_routes=(invalid,))
            with self.subTest(binding=invalid), self.assertRaisesRegex(
                SchemaError, "role or port legs"
            ):
                changed.validate_against(projection.dags[0], ir1)

        non_xy = replace(
            binding,
            local_noc_path=((0, 0), (0, 1), (1, 1), (2, 1), (3, 1)),
        )
        with self.assertRaisesRegex(SchemaError, "X-then-Y"):
            _recreate_schedule(source, flow_routes=(non_xy,)).validate_against(
                projection.dags[0], ir1
            )

    def test_two_by_two_route_requires_exact_transit_coverage(self) -> None:
        ir1, projection, schedule_set = _two_by_two_case()
        schedule_set.validate_against(projection, ir1)
        self.assertEqual(
            tuple(
                schedule.flow_routes[0].role
                for schedule in schedule_set.schedules
                if schedule.flow_routes
            ),
            (
                FlowRouteRole.SOURCE,
                FlowRouteRole.TRANSIT,
                FlowRouteRole.DESTINATION,
            ),
        )

        missing_dag = _empty_dag(ir1, 1)
        missing_projection = _projection(
            ir1,
            (
                projection.dags[0],
                missing_dag,
                projection.dags[2],
                projection.dags[3],
            ),
        )
        missing_schedules = (
            schedule_set.schedules[0],
            _route_schedule(
                ir1,
                missing_dag,
                ir1.groups[0].embedding.routes[0],
            ),
            schedule_set.schedules[2],
            schedule_set.schedules[3],
        )
        missing = _schedule_set(ir1, missing_projection, missing_schedules)
        with self.assertRaisesRegex(SchemaError, "one TRANSIT"):
            missing.validate_against(missing_projection, ir1)

        transit = schedule_set.schedules[1]
        duplicate = _recreate_schedule(
            transit, flow_routes=transit.flow_routes + transit.flow_routes
        )
        with self.assertRaisesRegex(SchemaError, "duplicate flow route"):
            _schedule_set(
                ir1,
                projection,
                (schedule_set.schedules[0], duplicate) + schedule_set.schedules[2:],
            ).validate_against(projection, ir1)

    def test_transit_rejects_all_executable_physical_bindings(self) -> None:
        ir1, projection, schedule_set = _two_by_two_case()
        transit_dag = projection.dags[1]
        transit = schedule_set.schedules[1]
        task_id = transit_dag.tasks[0].id
        core_id = ir1.fabric.dies[1].cores[0].runtime_core_id
        bad_binding = BufferBinding(
            "transit_buffer",
            "missing_value",
            TensorSlice("missing_value", (0,), (16,)),
            core_id,
            "sram_main",
            0,
            32,
            32,
            (0,),
            "transit_storage",
            None,
            BufferOwnership.BORROWED,
            0,
            1,
            DType.FP16,
            "flat",
        )
        cases = (
            (
                {"placements": (TaskPlacement(task_id, core_id),)},
                "exclude TRANSIT",
            ),
            (
                {"core_orders": (CoreOrder(core_id, (task_id,)),)},
                "exclude TRANSIT",
            ),
            (
                {
                    "runtime_bindings": (
                        LogicalRuntimeBinding(
                            task_id,
                            "flow_route",
                            "channel_route",
                            "event_transit",
                            "token_transit",
                        ),
                    )
                },
                "runtime binding",
            ),
            (
                {
                    "buffer_bindings": (bad_binding,),
                    "task_buffer_uses": (
                        TaskBufferUse(
                            task_id,
                            bad_binding.id,
                            BufferAccess.READ,
                            BufferUseRole.SEND_SOURCE,
                            0,
                            None,
                            bad_binding.tensor_slice,
                        ),
                    ),
                },
                "buffer use",
            ),
        )
        for changes, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError, message
            ):
                _recreate_schedule(transit, **changes).validate_against(
                    transit_dag, ir1
                )

    def test_schedule_set_rejects_missing_or_extra_dag_schedule(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        with self.assertRaisesRegex(SchemaError, "exactly one schedule"):
            _schedule_set(
                ir1, projection, schedule_set.schedules[:1]
            ).validate_against(projection, ir1)

    def test_schedule_set_must_follow_projection_dag_order(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        reordered = _schedule_set(
            ir1, projection, tuple(reversed(schedule_set.schedules))
        )
        with self.assertRaisesRegex(SchemaError, "DAG tuple order"):
            reordered.validate_against(projection, ir1)


if __name__ == "__main__":
    unittest.main()
