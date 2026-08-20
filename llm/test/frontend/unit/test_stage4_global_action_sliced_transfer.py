from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.global_action import (
    build_global_action_dag,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import (
    NaiveIntraDiePolicy,
)
from llm.frontend.wafer_frontend.schema.global_action import (
    GLOBAL_ACTION_DAG_SCHEMA_VERSION,
    GLOBAL_ACTION_SCHEMA_VERSION,
    GlobalActionDAG,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferUseRole,
    FlowRouteRole,
    SemanticTaskKind,
    StateIoOrigin,
    StateTransferOrigin,
    StateUseAccess,
)

from test_stage4_project_sliced_state_transfer import _project_tp1


def _build():
    ir1, fusion, standalone, contracts, projection = _project_tp1(
        large_sram=True
    )
    schedules = NaiveIntraDiePolicy().schedule(projection, ir1)
    actions = build_global_action_dag(ir1, projection, schedules)
    return ir1, fusion, standalone, contracts, projection, schedules, actions


def _recreate(dag: GlobalActionDAG, *, actions):
    fields = dag._semantic_key()
    fields["actions"] = tuple(actions)
    return GlobalActionDAG.create(
        producer_pass=dag.producer_pass,
        **fields,
    )


class Stage4GlobalActionSlicedTransferTest(unittest.TestCase):
    def test_tp1_cross_group_transfer_quotient_is_contract_exact(self) -> None:
        ir1, _fusion, _standalone, contracts, projection, schedules, dag = (
            _build()
        )
        self.assertEqual(
            GLOBAL_ACTION_SCHEMA_VERSION,
            "wafer_frontend.global_action/v1alpha8",
        )
        self.assertEqual(
            GLOBAL_ACTION_DAG_SCHEMA_VERSION,
            "wafer_frontend.global_action_dag/v1alpha11",
        )
        dag.validate_against(ir1, projection, schedules)
        self.assertEqual(len(dag.actions), 100)
        transfer_actions = tuple(
            action
            for action in dag.actions
            if isinstance(action.origin_ref, StateTransferOrigin)
        )
        self.assertEqual(
            {
                kind: sum(action.task_kind is kind for action in transfer_actions)
                for kind in (
                    SemanticTaskKind.SEND,
                    SemanticTaskKind.RECV,
                    SemanticTaskKind.WAIT,
                )
            },
            {
                SemanticTaskKind.SEND: 4,
                SemanticTaskKind.RECV: 4,
                SemanticTaskKind.WAIT: 4,
            },
        )
        action_by_task = {
            action.source.task_id: action for action in dag.actions
        }
        route_index = {route.id: route for route in ir1.cross_routes}
        for contract in contracts:
            route = route_index[contract.cross_group_route_ref]
            send = next(
                action
                for action in transfer_actions
                if action.origin_ref.state_transfer_ref == contract.id
                and action.task_kind is SemanticTaskKind.SEND
            )
            recv = next(
                action
                for action in transfer_actions
                if action.origin_ref.state_transfer_ref == contract.id
                and action.task_kind is SemanticTaskKind.RECV
            )
            wait = next(
                action
                for action in transfer_actions
                if action.origin_ref.state_transfer_ref == contract.id
                and action.task_kind is SemanticTaskKind.WAIT
            )
            source_dma = next(
                action
                for action in dag.actions
                if isinstance(action.origin_ref, StateIoOrigin)
                and action.origin_ref.state_access_ref
                == contract.source_state_access_ref
            )
            destination_dma = next(
                action
                for action in dag.actions
                if isinstance(action.origin_ref, StateIoOrigin)
                and action.origin_ref.state_access_ref
                == contract.destination_state_access_ref
            )
            self.assertEqual(source_dma.task_kind, SemanticTaskKind.DMA_OUT)
            self.assertEqual(destination_dma.task_kind, SemanticTaskKind.DMA_OUT)
            self.assertEqual(source_dma.state_uses[0].access, StateUseAccess.WRITE)
            self.assertEqual(
                destination_dma.state_uses[0].access,
                StateUseAccess.WRITE,
            )
            assert source_dma.dma is not None
            assert destination_dma.dma is not None
            source_targets = tuple(
                action_by_task[task_id].id
                for task_id in source_dma.dma.access_task_refs
            )
            destination_targets = tuple(
                action_by_task[task_id].id
                for task_id in destination_dma.dma.access_task_refs
            )
            self.assertTrue(all(target in send.deps for target in source_targets))
            self.assertTrue(
                all(target in source_dma.deps for target in source_targets)
            )
            self.assertIn(recv.id, wait.deps)
            self.assertTrue(
                all(
                    wait.id in action_by_task[task_id].deps
                    for task_id in destination_dma.dma.access_task_refs
                )
            )
            self.assertTrue(
                all(target in destination_dma.deps for target in destination_targets)
            )
            self.assertEqual(send.bytes, 256)
            self.assertEqual(recv.bytes, 256)
            self.assertEqual(send.tensor_slice.offset, contract.source_local_offset)
            self.assertEqual(send.tensor_slice.shape, contract.source_local_shape)
            self.assertEqual(
                recv.tensor_slice.offset,
                contract.destination_local_offset,
            )
            self.assertEqual(
                recv.tensor_slice.shape,
                contract.destination_local_shape,
            )
            self.assertEqual(send.flow.pair_route_ref, route.id)
            self.assertEqual(recv.flow.pair_route_ref, route.id)
            self.assertEqual(send.flow_route.pair_route_ref, route.id)
            self.assertEqual(recv.flow_route.pair_route_ref, route.id)
            self.assertEqual(send.flow_route.role, FlowRouteRole.SOURCE)
            self.assertEqual(recv.flow_route.role, FlowRouteRole.DESTINATION)
            send_use = next(
                use
                for use in send.buffer_uses
                if use.role is BufferUseRole.SEND_SOURCE
            )
            source_dma_use = next(
                use
                for use in source_dma.buffer_uses
                if use.role is BufferUseRole.DMA_SOURCE
            )
            recv_use = next(
                use
                for use in recv.buffer_uses
                if use.role is BufferUseRole.RECV_DESTINATION
            )
            destination_dma_use = next(
                use
                for use in destination_dma.buffer_uses
                if use.role is BufferUseRole.DMA_SOURCE
            )
            self.assertEqual(send_use.binding_id, source_dma_use.binding_id)
            self.assertEqual(recv_use.binding_id, destination_dma_use.binding_id)

        self.assertEqual(_build()[-1], dag)

    def test_route_dependency_and_staging_tampering_fail_closed(self) -> None:
        ir1, _fusion, _standalone, _contracts, projection, schedules, dag = (
            _build()
        )
        send_index = next(
            index
            for index, action in enumerate(dag.actions)
            if isinstance(action.origin_ref, StateTransferOrigin)
            and action.task_kind is SemanticTaskKind.SEND
        )
        send = dag.actions[send_index]
        assert send.flow_route is not None
        changed = list(dag.actions)
        changed[send_index] = replace(
            send,
            flow_route=replace(
                send.flow_route,
                pair_route_ref="cross_group_route_forged",
            ),
        )
        with self.assertRaisesRegex(SchemaError, "flow route is not exact"):
            _recreate(dag, actions=changed).validate_against(
                ir1, projection, schedules
            )

        changed[send_index] = replace(send, deps=())
        with self.assertRaisesRegex(SchemaError, "deps must exactly equal"):
            _recreate(dag, actions=changed).validate_against(
                ir1, projection, schedules
            )

        recv = next(
            action
            for action in dag.actions
            if isinstance(action.origin_ref, StateTransferOrigin)
            and action.task_kind is SemanticTaskKind.RECV
        )
        assert recv.buffer_uses
        changed = list(dag.actions)
        changed[send_index] = replace(
            send,
            buffer_uses=(
                replace(
                    send.buffer_uses[0],
                    binding_id=recv.buffer_uses[0].binding_id,
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "buffer uses are not exact"):
            _recreate(dag, actions=changed).validate_against(
                ir1, projection, schedules
            )


if __name__ == "__main__":
    unittest.main()
