from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir2 import TensorSlice
from llm.frontend.wafer_frontend.schema.local_transport import (
    LocalEvent,
    LocalEventPhase,
    LocalFlow,
    LocalNocRoute,
    LocalTransportPlan,
)


def _materialization_plan(source_dag_id: str) -> LocalTransportPlan:
    flow = LocalFlow(
        id="materialized", source_task_id="z_task_a", destination_task_id="a_task_c",
        source_core_id=0, destination_core_id=1, value_id="v_mid",
        tensor_slice=TensorSlice("v_mid", (0, 0), (32, 128)), bytes=8192,
        dtype=DType.FP16, route_id="route_materialized",
        send_event_id="send_materialized", recv_event_id="recv_materialized",
    )
    return LocalTransportPlan.create(
        producer_pass="test_local_transport", source_dag_id=source_dag_id,
        flows=(flow,),
        routes=(LocalNocRoute("route_materialized", "materialized", ((0, 0), (1, 0))),),
        events=(
            LocalEvent("recv_materialized", "materialized", "a_task_c", LocalEventPhase.RECV_READY),
            LocalEvent("send_materialized", "materialized", "z_task_a", LocalEventPhase.SEND_COMPLETE),
        ),
    )


def _plan() -> LocalTransportPlan:
    flow = LocalFlow(
        id="local_0",
        source_task_id="producer",
        destination_task_id="consumer",
        source_core_id=3,
        destination_core_id=7,
        value_id="value_0",
        tensor_slice=TensorSlice("value_0", (0,), (16,)),
        bytes=32,
        dtype=DType.FP16,
        route_id="route_0",
        send_event_id="event_send_0",
        recv_event_id="event_recv_0",
    )
    return LocalTransportPlan.create(
        producer_pass="intra_die_refine",
        source_dag_id="dag_0",
        flows=(flow,),
        routes=(LocalNocRoute("route_0", "local_0", ((0, 0), (1, 0), (1, 1))),),
        events=(
            LocalEvent("event_recv_0", "local_0", "consumer", LocalEventPhase.RECV_READY),
            LocalEvent("event_send_0", "local_0", "producer", LocalEventPhase.SEND_COMPLETE),
        ),
    )


class LocalTransportSchemaTest(unittest.TestCase):
    def test_valid_plan_and_exact_placement_validation(self) -> None:
        plan = _plan()
        plan.validate()
        plan.validate_against_placement(
            task_ids={"producer", "consumer"},
            placements={"producer": 3, "consumer": 7},
            core_noc_coords={3: (0, 0), 7: (1, 1)},
        )

    def test_rejects_non_xy_route_even_when_hops_are_adjacent(self) -> None:
        plan = _plan()
        route = replace(plan.routes[0], noc_path=((0, 0), (0, 1), (1, 1)))
        changed = LocalTransportPlan.create(
            producer_pass=plan.producer_pass,
            source_dag_id=plan.source_dag_id,
            flows=plan.flows,
            routes=(route,),
            events=plan.events,
        )
        with self.assertRaisesRegex(SchemaError, "exact backend-v1 X-then-Y"):
            changed.validate_against_placement(
                task_ids={"producer", "consumer"},
                placements={"producer": 3, "consumer": 7},
                core_noc_coords={3: (0, 0), 7: (1, 1)},
            )

    def test_rejects_event_with_wrong_owner_or_phase(self) -> None:
        plan = _plan()
        events = (replace(plan.events[0], task_id="producer"), plan.events[1])
        changed = LocalTransportPlan.create(
            producer_pass=plan.producer_pass,
            source_dag_id=plan.source_dag_id,
            flows=plan.flows,
            routes=plan.routes,
            events=events,
        )
        with self.assertRaisesRegex(SchemaError, "receive event"):
            changed.validate()

    def test_rejects_same_core_flow(self) -> None:
        plan = _plan()
        flow = replace(plan.flows[0], destination_core_id=3)
        changed = LocalTransportPlan.create(
            producer_pass=plan.producer_pass,
            source_dag_id=plan.source_dag_id,
            flows=(flow,),
            routes=plan.routes,
            events=plan.events,
        )
        with self.assertRaisesRegex(SchemaError, "distinct cores"):
                changed.validate()

    def test_materializes_explicit_local_task_chain(self) -> None:
        from test_naive_intra_die import _component_projection
        from llm.frontend.wafer_frontend.schema.ir2 import SemanticTaskKind

        _ir1, projection = _component_projection()
        dag = projection.dags[0]
        materialized = _materialization_plan(dag.id).materialize_into_dag(dag)
        materialized.validate()
        _materialization_plan(dag.id).validate_materialized_dag(materialized)
        tasks = {task.id: task for task in materialized.tasks}
        self.assertEqual(tasks["local_send_materialized"].kind, SemanticTaskKind.LOCAL_SEND)
        self.assertEqual(tasks["local_recv_materialized"].kind, SemanticTaskKind.LOCAL_RECV)
        self.assertEqual(tasks["local_wait_materialized"].kind, SemanticTaskKind.LOCAL_WAIT)
        self.assertEqual(tasks["local_recv_materialized"].deps, ("local_send_materialized",))
        self.assertIn("local_wait_materialized", tasks["a_task_c"].deps)

    def test_naive_scheduler_places_materialized_handoff_on_two_cores(self) -> None:
        from test_naive_intra_die import _component_projection
        from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
        from llm.frontend.wafer_frontend.schema.ir2 import IR2ProjectionResult

        ir1, projection = _component_projection()
        dag = projection.dags[0]
        materialized = _materialization_plan(dag.id).materialize_into_dag(dag)
        local_projection = IR2ProjectionResult.create(
            producer_pass="test_local_transport",
            source_ir1_id=ir1.id,
            fusion_plan_ids=(), standalone_collective_plan_ids=(),
            dags=(materialized,),
        )
        schedule = NaiveIntraDiePolicy().schedule(local_projection, ir1).schedules[0]
        placements = {item.task_id: item.core_id for item in schedule.placements}
        self.assertNotEqual(
            placements["local_send_materialized"],
            placements["local_recv_materialized"],
        )

    def test_global_action_preserves_local_handoff_without_d2d_flow(self) -> None:
        from test_naive_intra_die import _component_projection
        from llm.frontend.wafer_frontend.passes.global_action import build_global_action_dag
        from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
        from llm.frontend.wafer_frontend.schema.ir2 import IR2ProjectionResult, SemanticTaskKind

        ir1, projection = _component_projection()
        dag = projection.dags[0]
        materialized = _materialization_plan(dag.id).materialize_into_dag(dag)
        local_projection = IR2ProjectionResult.create(
            producer_pass="test_local_transport", source_ir1_id=ir1.id,
            fusion_plan_ids=(), standalone_collective_plan_ids=(), dags=(materialized,),
        )
        schedules = NaiveIntraDiePolicy().schedule(local_projection, ir1)
        actions = build_global_action_dag(ir1, local_projection, schedules)
        by_task = {action.source.task_id: action for action in actions.actions}
        send = by_task["local_send_materialized"]
        recv = by_task["local_recv_materialized"]
        self.assertEqual(send.task_kind, SemanticTaskKind.LOCAL_SEND)
        self.assertEqual(recv.task_kind, SemanticTaskKind.LOCAL_RECV)
        self.assertEqual(send.flow_id, "local.materialized")
        self.assertIsNone(send.flow)
        self.assertIsNone(send.flow_route)


    def test_local_handoff_lowers_to_native_noc_records(self) -> None:
        from test_naive_intra_die import _component_projection
        from llm.frontend.wafer_frontend.lowering.coarse import NaiveCoarseLowering
        from llm.frontend.wafer_frontend.lowering.context import LoweringContext
        from llm.frontend.wafer_frontend.passes.global_action import build_global_action_dag
        from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
        from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
        from llm.frontend.wafer_frontend.schema.ir2 import IR2ProjectionResult

        ir1, projection = _component_projection()
        dag = projection.dags[0]
        materialized = _materialization_plan(dag.id).materialize_into_dag(dag)
        local_projection = IR2ProjectionResult.create(
            producer_pass="test_local_transport", source_ir1_id=ir1.id,
            fusion_plan_ids=(), standalone_collective_plan_ids=(), dags=(materialized,),
        )
        schedules = NaiveIntraDiePolicy().schedule(local_projection, ir1)
        actions = build_global_action_dag(ir1, local_projection, schedules)
        context = LoweringContext(ir1, (), (), local_projection, schedules, actions)
        lowerer = NaiveCoarseLowering(validate_output=False)
        lowerer._validated_contexts.append(context)
        expected = {
            "local_send_materialized": RecordOpcode.LOCAL_NOC_SEND,
            "local_recv_materialized": RecordOpcode.LOCAL_NOC_RECV,
            "local_wait_materialized": RecordOpcode.LOCAL_NOC_WAIT,
        }
        event_ids = []
        for task_id, opcode in expected.items():
            action = next(item for item in actions.actions if item.source.task_id == task_id)
            fragment = lowerer.lower(action, context)
            fragment.validate_against(actions)
            self.assertEqual(fragment.core_streams[0].records[0].opcode, opcode)
            self.assertEqual(len(fragment.buffer_abi), 0 if opcode is RecordOpcode.LOCAL_NOC_WAIT else 1)
            event_ids.extend(
                operand.literal_value
                for operand in fragment.core_streams[0].records[0].operands
                if operand.name == "event_id"
            )
        self.assertEqual(len(set(event_ids)), 1)
        self.assertGreaterEqual(event_ids[0], 0x80000000)


if __name__ == "__main__":
    unittest.main()
