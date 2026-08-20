from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.action import (
    BarrierContract,
    BarrierScope,
    SyncContract,
)
from llm.frontend.wafer_frontend.schema.global_action import (
    ActionBufferUse,
    ActionStateUse,
    GlobalAction,
    GlobalActionDAG,
    LogicalCoreRef,
    ScheduledDagRef,
    ScheduledSourceRef,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.ir2 import (
    CoreOrder,
    IR2ProjectionResult,
    IntraDieDAG,
    IntraDieRegion,
    IntraDieSchedule,
    IntraDieScheduleSet,
    LogicalRuntimeBinding,
    SemanticTask,
    SemanticTaskKind,
    TaskPlacement,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json, loads_dataclass

from test_ir2_route_schedule import _two_by_two_case, _two_die_case


def _create_global(
    ir1: object,
    projection: IR2ProjectionResult,
    schedule_set: IntraDieScheduleSet,
) -> GlobalActionDAG:
    dag_by_id = {dag.id: dag for dag in projection.dags}
    action_ids = {
        (schedule.dag_id, schedule.id, task.id): GlobalAction.stable_id_for_source(
            ScheduledSourceRef(schedule.dag_id, schedule.id, task.id)
        )
        for schedule in schedule_set.schedules
        for task in dag_by_id[schedule.dag_id].tasks
    }
    actions: list[GlobalAction] = []
    for schedule in schedule_set.schedules:
        dag = dag_by_id[schedule.dag_id]
        die = next(item for item in ir1.fabric.dies if item.id == dag.die_id)
        cores = {core.runtime_core_id: core for core in die.cores}
        placements = {placement.task_id: placement.core_id for placement in schedule.placements}
        orders = {order.core_id: order.task_ids for order in schedule.core_orders}
        regions = {region.id: region for region in dag.regions}
        flows = {flow.id: flow for flow in dag.flows}
        routes = {route.flow_id: route for route in schedule.flow_routes}
        runtimes = {binding.task_id: binding for binding in schedule.runtime_bindings}
        for task in dag.tasks:
            action_id = action_ids[(dag.id, schedule.id, task.id)]
            if task.kind is SemanticTaskKind.TRANSIT:
                logical_core = None
                position = None
                predecessor = ()
            else:
                runtime_core = placements[task.id]
                logical_core = LogicalCoreRef(dag.die_id, cores[runtime_core].local_core_id)
                order = orders[runtime_core]
                position = order.index(task.id)
                predecessor = (
                    (action_ids[(dag.id, schedule.id, order[position - 1])],)
                    if position
                    else ()
                )
            deps: list[str] = []
            for dependency in tuple(
                action_ids[(dag.id, schedule.id, dep)] for dep in task.deps
            ) + predecessor:
                if dependency != action_id and dependency not in deps:
                    deps.append(dependency)
            actions.append(GlobalAction.create(
                source=ScheduledSourceRef(dag.id, schedule.id, task.id),
                task_kind=task.kind,
                origin_ref=task.origin_ref,
                lowering=regions[task.region_id].lowering,
                region_id=task.region_id,
                op_kind=task.op_kind,
                member_id=task.member_id,
                flow_id=task.flow_id,
                chunk_id=task.chunk_id,
                collective_step=task.collective_step,
                source_rank=task.source_rank,
                destination_rank=task.destination_rank,
                tensor_slice=task.tensor_slice,
                bytes=task.bytes,
                dtype=task.dtype,
                shape=task.shape,
                read_values=task.read_values,
                write_values=task.write_values,
                compute=task.compute,
                reduction=task.reduction,
                sync=task.sync,
                dma=task.dma,
                logical_core=logical_core,
                core_order_index=position,
                flow=flows.get(task.flow_id),
                flow_route=routes.get(task.flow_id),
                runtime_binding=runtimes.get(task.id),
                buffer_uses=tuple(
                    ActionBufferUse(
                        use.binding_id,
                        use.access,
                        use.role,
                        use.operand_index,
                        use.contribution_rank,
                        use.tensor_slice,
                    )
                    for use in schedule.task_buffer_uses if use.task_id == task.id
                ),
                state_uses=tuple(
                    ActionStateUse(
                        use.hbm_binding_ref,
                        use.access,
                    )
                    for use in schedule.task_state_uses
                    if use.task_id == task.id
                ),
                deps=tuple(deps),
            ))
    return GlobalActionDAG.create(
        producer_pass="global_action_fixture",
        source_ir1_id=ir1.id,
        source_state_manifest_id=(
            ir1.persistent_state_manifest.id
            if ir1.persistent_state_manifest is not None else None
        ),
        source_projection_id=projection.id,
        source_schedule_set_id=schedule_set.id,
        scheduled_dags=tuple(
            ScheduledDagRef(schedule.dag_id, schedule.id, schedule.die_id)
            for schedule in schedule_set.schedules
        ),
        actions=tuple(actions),
    )


def _recreate(dag: GlobalActionDAG, **changes: object) -> GlobalActionDAG:
    fields = dag._semantic_key()
    fields.update(changes)
    return GlobalActionDAG.create(producer_pass=dag.producer_pass, **fields)


def _with_same_core_predecessor():
    ir1, projection, schedule_set = _two_die_case()
    dag0, dag1 = projection.dags
    schedule0, schedule1 = schedule_set.schedules
    send = dag0.tasks[0]
    predecessor = SemanticTask(
        id="task_barrier",
        kind=SemanticTaskKind.BARRIER,
        origin_ref=replace(send.origin_ref, action_id="barrier_action"),
        region_id=send.region_id,
        op_kind=OpKind.COLLECTIVE,
        member_id=send.member_id,
        flow_id=None,
        chunk_id=send.chunk_id,
        collective_step=send.collective_step,
        source_rank=None,
        destination_rank=None,
        tensor_slice=None,
        bytes=0,
        dtype=None,
        shape=(),
        read_values=(),
        write_values=(),
        compute=None,
        reduction=None,
        sync=SyncContract(
            "event_barrier_done",
            None,
            BarrierContract(
                "barrier_predecessor",
                (send.origin_ref.rank,),
                1,
                BarrierScope.PLAN,
            ),
        ),
        deps=(),
    )
    send = replace(send, deps=(predecessor.id,))
    region = dag0.regions[0]
    dag0 = IntraDieDAG.create(
        producer_pass=dag0.producer_pass,
        **{
            **dag0._semantic_key(),
            "tasks": (predecessor, send),
            "regions": (
                replace(region, task_ids=(predecessor.id, send.id)),
            ),
        },
    )
    binding = replace(
        schedule0.buffer_bindings[0],
        lifetime_start=1,
        lifetime_end_exclusive=2,
    )
    core_id = schedule0.placements[0].core_id
    schedule0 = IntraDieSchedule.create(
        producer_pass=schedule0.producer_pass,
        **{
            **schedule0._semantic_key(),
            "dag_id": dag0.id,
            "placements": (
                TaskPlacement(predecessor.id, core_id),
                TaskPlacement(send.id, core_id),
            ),
            "buffer_bindings": (binding,),
            "runtime_bindings": (
                LogicalRuntimeBinding(
                    predecessor.id,
                    None,
                    None,
                    "barrier_predecessor",
                    "token_barrier",
                ),
            ) + schedule0.runtime_bindings,
            "core_orders": (
                CoreOrder(core_id, (predecessor.id, send.id)),
            ),
        },
    )
    projection = IR2ProjectionResult.create(
        producer_pass=projection.producer_pass,
        **{**projection._semantic_key(), "dags": (dag0, dag1)},
    )
    schedule_set = IntraDieScheduleSet.create(
        producer_pass=schedule_set.producer_pass,
        source_projection_id=projection.id,
        source_ir1_id=ir1.id,
        schedules=(schedule0, schedule1),
    )
    schedule_set.validate_against(projection, ir1)
    return ir1, projection, schedule_set, _create_global(ir1, projection, schedule_set)


def _with_recv_wait():
    ir1, projection, schedule_set = _two_die_case()
    source_dag, destination_dag = projection.dags
    source_schedule, destination_schedule = schedule_set.schedules
    recv = destination_dag.tasks[0]
    assert recv.sync is not None
    wait = SemanticTask(
        id="task_wait_for_recv",
        kind=SemanticTaskKind.WAIT,
        origin_ref=replace(recv.origin_ref, action_id="wait_action"),
        region_id=recv.region_id,
        op_kind=OpKind.COLLECTIVE,
        member_id=recv.member_id,
        flow_id=None,
        chunk_id=recv.chunk_id,
        collective_step=recv.collective_step,
        source_rank=None,
        destination_rank=None,
        tensor_slice=None,
        bytes=0,
        dtype=None,
        shape=(),
        read_values=(),
        write_values=(),
        compute=None,
        reduction=None,
        sync=SyncContract(
            "event_wait_done",
            recv.sync.completion_event,
            None,
        ),
        deps=(recv.id,),
    )
    region = destination_dag.regions[0]
    destination_dag = IntraDieDAG.create(
        producer_pass=destination_dag.producer_pass,
        **{
            **destination_dag._semantic_key(),
            "tasks": (recv, wait),
            "regions": (
                replace(region, task_ids=(recv.id, wait.id)),
            ),
        },
    )
    core_id = destination_schedule.placements[0].core_id
    recv_runtime = destination_schedule.runtime_bindings[0]
    destination_schedule = IntraDieSchedule.create(
        producer_pass=destination_schedule.producer_pass,
        **{
            **destination_schedule._semantic_key(),
            "dag_id": destination_dag.id,
            "placements": destination_schedule.placements
            + (TaskPlacement(wait.id, core_id),),
            "runtime_bindings": destination_schedule.runtime_bindings
            + (
                LogicalRuntimeBinding(
                    wait.id,
                    None,
                    None,
                    recv.sync.completion_event,
                    recv_runtime.token_symbol,
                ),
            ),
            "core_orders": (CoreOrder(core_id, (recv.id, wait.id)),),
        },
    )
    projection = IR2ProjectionResult.create(
        producer_pass=projection.producer_pass,
        **{
            **projection._semantic_key(),
            "dags": (source_dag, destination_dag),
        },
    )
    schedule_set = IntraDieScheduleSet.create(
        producer_pass=schedule_set.producer_pass,
        source_projection_id=projection.id,
        source_ir1_id=ir1.id,
        schedules=(source_schedule, destination_schedule),
    )
    schedule_set.validate_against(projection, ir1)
    return ir1, projection, schedule_set


class GlobalActionSchemaTest(unittest.TestCase):
    def test_two_rank_asymmetric_exact_round_trip(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        dag = _create_global(ir1, projection, schedule_set)
        dag.validate_against(ir1, projection, schedule_set)
        decoded = loads_dataclass(GlobalActionDAG, canonical_json(dag))
        self.assertEqual(decoded, dag)
        self.assertEqual(canonical_digest(decoded), canonical_digest(dag))
        send, recv = dag.actions
        self.assertNotEqual(send.logical_core, recv.logical_core)
        self.assertNotEqual(send.buffer_uses, recv.buffer_uses)
        self.assertNotEqual(send.flow_route, recv.flow_route)

    def test_bijection_and_schedule_pairing_are_strict(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        dag = _create_global(ir1, projection, schedule_set)
        with self.assertRaisesRegex(SchemaError, "exactly preserve"):
            _recreate(dag, actions=dag.actions[:-1]).validate_against(ir1, projection, schedule_set)
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(dag.actions[0], id="arbitrary_action").validate("action")
        forged = replace(dag.actions[0], source=replace(dag.actions[0].source, schedule_id=dag.scheduled_dags[1].schedule_id))
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            _recreate(dag, actions=(forged, dag.actions[1])).validate()

    def test_action_and_schedule_order_are_canonical(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        dag = _create_global(ir1, projection, schedule_set)
        with self.assertRaisesRegex(SchemaError, "exactly preserve"):
            _recreate(dag, actions=tuple(reversed(dag.actions))).validate_against(
                ir1, projection, schedule_set
            )
        with self.assertRaisesRegex(SchemaError, "pairing/order"):
            _recreate(dag, scheduled_dags=tuple(reversed(dag.scheduled_dags))).validate_against(
                ir1, projection, schedule_set
            )

    def test_semantic_route_runtime_and_core_tampering_fail(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        dag = _create_global(ir1, projection, schedule_set)
        send, recv = dag.actions
        mutations = (
            (replace(send, bytes=send.bytes + 2), "semantics"),
            (replace(send, flow_route=replace(send.flow_route, local_noc_path=tuple(reversed(send.flow_route.local_noc_path)))), "route"),
            (replace(send, runtime_binding=replace(send.runtime_binding, event_symbol="forged_event")), "runtime"),
            (replace(send, logical_core=LogicalCoreRef(0, 1)), "logical core"),
            (replace(send, core_order_index=1), "order index"),
        )
        for changed, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(SchemaError, message):
                _recreate(dag, actions=(changed, recv)).validate_against(ir1, projection, schedule_set)

    def test_buffer_use_subset_access_and_role_fail(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        dag = _create_global(ir1, projection, schedule_set)
        send, recv = dag.actions
        original = send.buffer_uses[0]
        for uses in (
            (),
            (replace(original, access=recv.buffer_uses[0].access),),
            (replace(original, role=recv.buffer_uses[0].role),),
            (
                replace(
                    original,
                    tensor_slice=replace(
                        original.tensor_slice,
                        offset=(original.tensor_slice.offset[0] + 1,),
                    ),
                ),
            ),
        ):
            with self.assertRaisesRegex(SchemaError, "buffer uses"):
                _recreate(dag, actions=(replace(send, buffer_uses=uses), recv)).validate_against(ir1, projection, schedule_set)

    def test_exact_semantic_and_same_core_predecessor_deps(self) -> None:
        ir1, projection, schedule_set, dag = _with_same_core_predecessor()
        dag.validate_against(ir1, projection, schedule_set)
        predecessor, send, recv = dag.actions
        self.assertEqual(send.deps, (predecessor.id,))
        for deps in ((), (predecessor.id, recv.id)):
            with self.assertRaisesRegex(SchemaError, "deps must exactly"):
                _recreate(
                    dag,
                    actions=(predecessor, replace(send, deps=deps), recv),
                ).validate_against(ir1, projection, schedule_set)

    def test_wait_runtime_uses_waited_event_not_its_own_completion(self) -> None:
        ir1, projection, schedule_set = _with_recv_wait()
        source, destination = schedule_set.schedules
        recv_task, wait_task = projection.dags[1].tasks
        recv_binding, wait_binding = destination.runtime_bindings
        self.assertEqual(wait_binding.event_symbol, wait_task.sync.wait_event)
        self.assertEqual(wait_binding.event_symbol, recv_task.sync.completion_event)
        self.assertEqual(wait_binding.token_symbol, recv_binding.token_symbol)
        forged_destination = IntraDieSchedule.create(
            producer_pass=destination.producer_pass,
            **{
                **destination._semantic_key(),
                "runtime_bindings": (
                    recv_binding,
                    replace(
                        wait_binding,
                        token_symbol="forged_wait_token",
                    ),
                ),
            },
        )
        forged_set = IntraDieScheduleSet.create(
            producer_pass=schedule_set.producer_pass,
            source_projection_id=projection.id,
            source_ir1_id=ir1.id,
            schedules=(source, forged_destination),
        )
        with self.assertRaisesRegex(SchemaError, "reuse.*RECV.*token"):
            forged_set.validate_against(projection, ir1)

    def test_transit_is_the_only_coreless_action(self) -> None:
        ir1, projection, schedule_set = _two_by_two_case()
        dag = _create_global(ir1, projection, schedule_set)
        dag.validate_against(ir1, projection, schedule_set)
        transit = next(action for action in dag.actions if action.task_kind is SemanticTaskKind.TRANSIT)
        self.assertIsNone(transit.logical_core)
        self.assertIsNotNone(transit.flow_route)
        with self.assertRaisesRegex(SchemaError, "TRANSIT cannot carry"):
            replace(transit, logical_core=LogicalCoreRef(1, 0), core_order_index=0).validate("action")


if __name__ == "__main__":
    unittest.main()
