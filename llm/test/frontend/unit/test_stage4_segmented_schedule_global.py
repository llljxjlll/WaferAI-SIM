from __future__ import annotations

from collections import Counter
from dataclasses import replace
from functools import lru_cache
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.global_action import (
    build_global_action_dag,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import (
    NAIVE_INTRADIE_POLICY_SCHEMA_VERSION,
    NaiveIntraDiePolicy,
)
from llm.frontend.wafer_frontend.schema.global_action import (
    GLOBAL_ACTION_DAG_SCHEMA_VERSION,
    GLOBAL_ACTION_SCHEMA_VERSION,
    GlobalActionDAG,
    state_transfer_wave_task_dependencies,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    INTRA_DIE_SCHEDULE_SCHEMA_VERSION,
    INTRA_DIE_SCHEDULE_SET_SCHEMA_VERSION,
    BufferUseRole,
    CoreOrder,
    FlowRouteRole,
    IntraDieSchedule,
    SemanticTaskKind,
    StateIoOrigin,
    StateTransferOrigin,
    TensorSlice,
)

from test_stage4_project_segmented_state_transfer import _case


@lru_cache(maxsize=1)
def _artifacts():
    planned, contracts, projection = _case()
    schedules = NaiveIntraDiePolicy().schedule(projection, planned.graph)
    global_dag = build_global_action_dag(
        planned.graph,
        projection,
        schedules,
    )
    return planned, contracts, projection, schedules, global_dag


def _rebuild_schedule(
    schedule: IntraDieSchedule,
    **changes: object,
) -> IntraDieSchedule:
    fields = schedule._semantic_key()
    fields.update(changes)
    return IntraDieSchedule.create(
        producer_pass=schedule.producer_pass,
        **fields,
    )


def _rebuild_global(
    global_dag: GlobalActionDAG,
    **changes: object,
) -> GlobalActionDAG:
    fields = global_dag._semantic_key()
    fields.update(changes)
    return GlobalActionDAG.create(
        producer_pass=global_dag.producer_pass,
        **fields,
    )


class Stage4SegmentedScheduleGlobalTest(unittest.TestCase):
    def test_tp2_to_tp1_real_capacity_versions_and_counts(self) -> None:
        planned, contracts, projection, schedules, global_dag = _artifacts()
        self.assertEqual(
            (
                NAIVE_INTRADIE_POLICY_SCHEMA_VERSION,
                INTRA_DIE_SCHEDULE_SCHEMA_VERSION,
                INTRA_DIE_SCHEDULE_SET_SCHEMA_VERSION,
                GLOBAL_ACTION_SCHEMA_VERSION,
                GLOBAL_ACTION_DAG_SCHEMA_VERSION,
            ),
            (
                "wafer_frontend.naive_intra_die_policy/v8",
                "wafer_frontend.intra_die_schedule/v1alpha14",
                "wafer_frontend.intra_die_schedule_set/v1alpha9",
                "wafer_frontend.global_action/v1alpha8",
                "wafer_frontend.global_action_dag/v1alpha11",
            ),
        )
        schedules.validate_against(projection, planned.graph)
        global_dag.validate_against(planned.graph, projection, schedules)
        self.assertEqual(len(contracts), 8)
        self.assertEqual(sum(len(contract.segments) for contract in contracts), 64)
        self.assertEqual(sum(contract.bytes for contract in contracts), 1024)
        self.assertEqual(
            len({contract.cross_group_route_ref for contract in contracts}),
            2,
        )
        self.assertEqual(
            tuple(
                (
                    schedule.die_id,
                    len(schedule.placements),
                    len(schedule.buffer_bindings),
                    len(schedule.task_buffer_uses),
                    len(schedule.task_state_uses),
                    len(schedule.flow_routes),
                )
                for schedule in schedules.schedules
            ),
            (
                (0, 112, 69, 168, 19, 48),
                (1, 112, 69, 168, 19, 80),
                (2, 172, 45, 152, 19, 64),
            ),
        )
        max_end_by_die = {
            schedule.die_id: max(
                binding.region_offset_bytes + binding.size_bytes
                for binding in schedule.buffer_bindings
            )
            for schedule in schedules.schedules
        }
        self.assertEqual(max_end_by_die, {0: 17536, 1: 17536, 2: 15904})
        self.assertEqual(
            len(global_dag.actions),
            sum(len(dag.tasks) for dag in projection.dags),
        )
        self.assertEqual(len(global_dag.actions), 428)
        self.assertEqual(
            len(state_transfer_wave_task_dependencies(projection, schedules)),
            55,
        )

    def test_segment_lineage_shared_staging_waits_and_transit_quotient(self) -> None:
        planned, contracts, projection, schedules, global_dag = _artifacts()
        dag_by_die = {dag.die_id: dag for dag in projection.dags}
        schedule_by_die = {
            schedule.die_id: schedule for schedule in schedules.schedules
        }
        route_by_id = {route.id: route for route in planned.graph.cross_routes}
        action_by_task = {
            (next(
                ref.die_id
                for ref in global_dag.scheduled_dags
                if ref.dag_id == action.source.dag_id
            ), action.source.task_id): action
            for action in global_dag.actions
        }
        for contract in contracts:
            route = route_by_id[contract.cross_group_route_ref]
            source_die = route.die_path[0]
            destination_die = route.die_path[-1]
            source_dag = dag_by_die[source_die]
            destination_dag = dag_by_die[destination_die]
            source_schedule = schedule_by_die[source_die]
            destination_schedule = schedule_by_die[destination_die]
            source_dma = next(
                task
                for task in source_dag.tasks
                if isinstance(task.origin_ref, StateIoOrigin)
                and task.origin_ref.state_access_ref
                == contract.source_state_access_ref
            )
            destination_dma = next(
                task
                for task in destination_dag.tasks
                if isinstance(task.origin_ref, StateIoOrigin)
                and task.origin_ref.state_access_ref
                == contract.destination_state_access_ref
            )
            source_dma_use = next(
                use
                for use in source_schedule.task_buffer_uses
                if use.task_id == source_dma.id
                and use.role is BufferUseRole.DMA_SOURCE
            )
            destination_dma_use = next(
                use
                for use in destination_schedule.task_buffer_uses
                if use.task_id == destination_dma.id
                and use.role is BufferUseRole.DMA_SOURCE
            )
            waits = []
            for segment_index, segment in enumerate(contract.segments):
                segment_actions = tuple(
                    action
                    for action in global_dag.actions
                    if isinstance(action.origin_ref, StateTransferOrigin)
                    and action.origin_ref.state_transfer_ref == contract.id
                    and action.origin_ref.segment_index == segment_index
                )
                self.assertEqual(
                    Counter(action.task_kind for action in segment_actions),
                    Counter(
                        {
                            SemanticTaskKind.SEND: 1,
                            SemanticTaskKind.RECV: 1,
                            SemanticTaskKind.WAIT: 1,
                            SemanticTaskKind.TRANSIT: len(route.die_path) - 2,
                        }
                    ),
                )
                send = next(
                    action
                    for action in segment_actions
                    if action.task_kind is SemanticTaskKind.SEND
                )
                recv = next(
                    action
                    for action in segment_actions
                    if action.task_kind is SemanticTaskKind.RECV
                )
                wait = next(
                    action
                    for action in segment_actions
                    if action.task_kind is SemanticTaskKind.WAIT
                )
                waits.append(wait.id)
                self.assertEqual(
                    (send.bytes, recv.bytes),
                    (segment.bytes, segment.bytes),
                )
                self.assertEqual(
                    next(
                        use.binding_id
                        for use in send.buffer_uses
                        if use.role is BufferUseRole.SEND_SOURCE
                    ),
                    source_dma_use.binding_id,
                )
                self.assertEqual(
                    next(
                        use.binding_id
                        for use in recv.buffer_uses
                        if use.role is BufferUseRole.RECV_DESTINATION
                    ),
                    destination_dma_use.binding_id,
                )
                self.assertEqual(send.flow_route.pair_route_ref, route.id)
                self.assertEqual(recv.flow_route.pair_route_ref, route.id)
            assert destination_dma.dma is not None
            for target_id in destination_dma.dma.access_task_refs:
                target = action_by_task[(destination_die, target_id)]
                self.assertTrue(set(waits).issubset(target.deps))

        transit = tuple(
            action
            for action in global_dag.actions
            if isinstance(action.origin_ref, StateTransferOrigin)
            and action.task_kind is SemanticTaskKind.TRANSIT
        )
        self.assertEqual(len(transit), 32)
        self.assertTrue(
            all(
                action.logical_core is None
                and action.core_order_index is None
                and action.runtime_binding is None
                and not action.buffer_uses
                and not action.state_uses
                and action.flow_route.role is FlowRouteRole.TRANSIT
                and action.flow_route.ingress is not None
                and action.flow_route.egress is not None
                for action in transit
            )
        )

    def test_schedule_segment_view_route_and_order_tamper_fail_closed(self) -> None:
        planned, contracts, projection, schedules, _global_dag = _artifacts()
        first = contracts[0]
        route = next(
            item
            for item in planned.graph.cross_routes
            if item.id == first.cross_group_route_ref
        )
        source_dag = next(
            dag for dag in projection.dags if dag.die_id == route.die_path[0]
        )
        source_schedule = next(
            schedule
            for schedule in schedules.schedules
            if schedule.die_id == route.die_path[0]
        )
        send = next(
            task
            for task in source_dag.tasks
            if isinstance(task.origin_ref, StateTransferOrigin)
            and task.origin_ref.state_transfer_ref == first.id
            and task.origin_ref.segment_index == 0
            and task.kind is SemanticTaskKind.SEND
        )
        send_use = next(
            use for use in source_schedule.task_buffer_uses
            if use.task_id == send.id
        )
        wrong_view = replace(
            send_use,
            tensor_slice=TensorSlice(
                send_use.tensor_slice.value_id,
                first.segments[1].source_local_offset,
                first.segments[1].source_local_shape,
            ),
        )
        bad_view = _rebuild_schedule(
            source_schedule,
            task_buffer_uses=tuple(
                wrong_view if use == send_use else use
                for use in source_schedule.task_buffer_uses
            ),
        )
        with self.assertRaisesRegex(SchemaError, "task.tensor_slice"):
            bad_view.validate_against(source_dag, planned.graph)

        send_route = next(
            binding
            for binding in source_schedule.flow_routes
            if binding.flow_id == send.flow_id
        )
        bad_route = _rebuild_schedule(
            source_schedule,
            flow_routes=tuple(
                replace(binding, egress=None)
                if binding == send_route
                else binding
                for binding in source_schedule.flow_routes
            ),
        )
        with self.assertRaises(SchemaError):
            bad_route.validate_against(source_dag, planned.graph)

        order = next(
            item
            for item in source_schedule.core_orders
            if send.id in item.task_ids
        )
        dependency = send.deps[0]
        tampered_ids = list(order.task_ids)
        send_position = tampered_ids.index(send.id)
        dependency_position = tampered_ids.index(dependency)
        tampered_ids[send_position], tampered_ids[dependency_position] = (
            tampered_ids[dependency_position],
            tampered_ids[send_position],
        )
        bad_order = _rebuild_schedule(
            source_schedule,
            core_orders=tuple(
                CoreOrder(item.core_id, tuple(tampered_ids))
                if item == order
                else item
                for item in source_schedule.core_orders
            ),
        )
        with self.assertRaisesRegex(SchemaError, "dependency"):
            bad_order.validate_against(source_dag, planned.graph)

    def test_global_segment_origin_dependency_and_transit_tamper_fail_closed(
        self,
    ) -> None:
        planned, contracts, projection, schedules, global_dag = _artifacts()
        transfer_actions = tuple(
            action
            for action in global_dag.actions
            if isinstance(action.origin_ref, StateTransferOrigin)
        )
        send = next(
            action
            for action in transfer_actions
            if action.task_kind is SemanticTaskKind.SEND
        )
        assert isinstance(send.origin_ref, StateTransferOrigin)
        bad_origin_action = replace(
            send,
            origin_ref=replace(send.origin_ref, segment_index=99),
        )
        bad_origin = _rebuild_global(
            global_dag,
            actions=tuple(
                bad_origin_action if action.id == send.id else action
                for action in global_dag.actions
            ),
        )
        with self.assertRaises(SchemaError):
            bad_origin.validate_against(planned.graph, projection, schedules)

        first = contracts[0]
        route = next(
            item
            for item in planned.graph.cross_routes
            if item.id == first.cross_group_route_ref
        )
        destination_dag = next(
            dag for dag in projection.dags if dag.die_id == route.die_path[-1]
        )
        destination_dma = next(
            task
            for task in destination_dag.tasks
            if isinstance(task.origin_ref, StateIoOrigin)
            and task.origin_ref.state_access_ref
            == first.destination_state_access_ref
        )
        assert destination_dma.dma is not None
        target_task_id = destination_dma.dma.access_task_refs[0]
        scheduled_ref = next(
            ref for ref in global_dag.scheduled_dags
            if ref.die_id == route.die_path[-1]
        )
        target = next(
            action
            for action in global_dag.actions
            if action.source.dag_id == scheduled_ref.dag_id
            and action.source.task_id == target_task_id
        )
        wait_id = next(
            action.id
            for action in transfer_actions
            if action.task_kind is SemanticTaskKind.WAIT
            and action.origin_ref.state_transfer_ref == first.id
            and action.origin_ref.segment_index == 0
        )
        bad_target = replace(
            target,
            deps=tuple(dep for dep in target.deps if dep != wait_id),
        )
        bad_dependency = _rebuild_global(
            global_dag,
            actions=tuple(
                bad_target if action.id == target.id else action
                for action in global_dag.actions
            ),
        )
        with self.assertRaisesRegex(SchemaError, "deps"):
            bad_dependency.validate_against(planned.graph, projection, schedules)

        transit = next(
            action
            for action in transfer_actions
            if action.task_kind is SemanticTaskKind.TRANSIT
        )
        bad_transit_action = replace(
            transit,
            buffer_uses=(send.buffer_uses[0],),
        )
        bad_transit = _rebuild_global(
            global_dag,
            actions=tuple(
                bad_transit_action if action.id == transit.id else action
                for action in global_dag.actions
            ),
        )
        with self.assertRaises(SchemaError):
            bad_transit.validate_against(planned.graph, projection, schedules)


if __name__ == "__main__":
    unittest.main()
