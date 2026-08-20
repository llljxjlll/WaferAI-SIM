from __future__ import annotations

from dataclasses import replace
import json
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    build_stage4_global_action as public_build_stage4_global_action,
)
from llm.frontend.wafer_frontend.passes.global_action_dag import (
    build_stage4_global_action,
)
from llm.frontend.wafer_frontend.passes.intra_die_schedule import (
    schedule_stage4,
)
from llm.frontend.wafer_frontend.schema import (
    Stage4GlobalAction as PublicStage4GlobalAction,
)
from llm.frontend.wafer_frontend.schema.global_action import GlobalActionDAG
from llm.frontend.wafer_frontend.schema.ir0 import EdgeKind
from llm.frontend.wafer_frontend.schema.ir2 import (
    FlowRouteRole,
    SemanticTaskKind,
    StateIoOrigin,
    StateTransferOrigin,
    StateUseAccess,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    STAGE4_GLOBAL_ACTION_SCHEMA_VERSION,
    Stage4GlobalAction,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass

from test_stage4_scheduled_carrier import (
    _projected,
    _projected_pdr,
    _scheduling_context,
)


def _scheduled(*, fused: bool):
    return schedule_stage4(_projected(fused=fused), _scheduling_context())


def _scheduled_pdr():
    return schedule_stage4(_projected_pdr(), _scheduling_context())


def _recreate_global(dag: GlobalActionDAG, *, actions):
    fields = dag._semantic_key()
    fields["actions"] = tuple(actions)
    return GlobalActionDAG.create(
        producer_pass=dag.producer_pass,
        **fields,
    )


class Stage4GlobalActionCarrierTest(unittest.TestCase):
    def test_public_api_fused_tp1_actions_provenance_and_strict_serde(self) -> None:
        self.assertIs(
            public_build_stage4_global_action,
            build_stage4_global_action,
        )
        self.assertIs(PublicStage4GlobalAction, Stage4GlobalAction)
        source = _scheduled(fused=True)
        result = build_stage4_global_action(source)
        result.validate_against(source)
        self.assertEqual(
            STAGE4_GLOBAL_ACTION_SCHEMA_VERSION,
            "wafer_frontend.stage4_global_action/v1alpha2",
        )
        self.assertEqual(len(result.global_dag.actions), 92)
        self.assertEqual(
            {
                kind: sum(
                    action.task_kind is kind
                    for action in result.global_dag.actions
                )
                for kind in (
                    SemanticTaskKind.DMA_IN,
                    SemanticTaskKind.COMP,
                    SemanticTaskKind.DMA_OUT,
                )
            },
            {
                SemanticTaskKind.DMA_IN: 34,
                SemanticTaskKind.COMP: 50,
                SemanticTaskKind.DMA_OUT: 8,
            },
        )
        self.assertFalse(
            any(
                isinstance(action.origin_ref, StateTransferOrigin)
                for action in result.global_dag.actions
            )
        )
        self.assertEqual(result.source_scheduled_carrier_id, source.id)
        self.assertEqual(
            result.source_projected_carrier_id,
            source.source_projected_carrier_id,
        )
        self.assertEqual(result.projection, source.projection)
        self.assertEqual(result.schedule_set, source.schedule_set)
        self.assertEqual(
            loads_dataclass(
                Stage4GlobalAction,
                canonical_json(result),
                path="result",
            ),
            result,
        )
        self.assertEqual(build_stage4_global_action(source), result)

        raw = json.loads(canonical_json(result))
        raw["unexpected"] = True
        with self.assertRaises(SchemaError):
            loads_dataclass(
                Stage4GlobalAction,
                canonical_json(raw),
                path="result",
            )
        del raw["unexpected"]
        del raw["global_dag"]
        with self.assertRaises(SchemaError):
            loads_dataclass(
                Stage4GlobalAction,
                canonical_json(raw),
                path="result",
            )

    def test_pds_tp1_cross_instance_control_and_four_completions_are_exact(self) -> None:
        source = _scheduled(fused=False)
        result = build_stage4_global_action(source)
        result.validate_against(source)
        self.assertEqual(len(result.global_dag.actions), 100)
        node_index = {node.id: node for node in result.graph.nodes}
        cross_instance_control = tuple(
            edge
            for edge in result.graph.edges
            if edge.kind is EdgeKind.CONTROL
            and node_index[edge.source_node].instance_id
            != node_index[edge.destination_node].instance_id
        )
        self.assertEqual(len(cross_instance_control), 1)
        control = cross_instance_control[0]
        self.assertEqual(
            (
                node_index[control.source_node].instance_id,
                node_index[control.destination_node].instance_id,
            ),
            ("P0", "D0"),
        )

        actions = result.global_dag.actions
        transfer_actions = tuple(
            action
            for action in actions
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
        action_by_task = {action.source.task_id: action for action in actions}
        for contract in result.projection.state_transfers:
            matching = tuple(
                action
                for action in transfer_actions
                if action.origin_ref.state_transfer_ref == contract.id
            )
            send = next(
                action
                for action in matching
                if action.task_kind is SemanticTaskKind.SEND
            )
            recv = next(
                action
                for action in matching
                if action.task_kind is SemanticTaskKind.RECV
            )
            wait = next(
                action
                for action in matching
                if action.task_kind is SemanticTaskKind.WAIT
            )
            source_dma = next(
                action
                for action in actions
                if isinstance(action.origin_ref, StateIoOrigin)
                and action.origin_ref.state_access_ref
                == contract.source_state_access_ref
            )
            destination_dma = next(
                action
                for action in actions
                if isinstance(action.origin_ref, StateIoOrigin)
                and action.origin_ref.state_access_ref
                == contract.destination_state_access_ref
            )
            self.assertIs(source_dma.task_kind, SemanticTaskKind.DMA_OUT)
            self.assertIs(destination_dma.task_kind, SemanticTaskKind.DMA_OUT)
            self.assertEqual(
                tuple(use.access for use in source_dma.state_uses),
                (StateUseAccess.WRITE,),
            )
            self.assertEqual(
                tuple(use.access for use in destination_dma.state_uses),
                (StateUseAccess.WRITE,),
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
            self.assertEqual((send.bytes, recv.bytes), (256, 256))
            self.assertEqual(
                (send.flow.pair_route_ref, recv.flow.pair_route_ref),
                (
                    contract.cross_group_route_ref,
                    contract.cross_group_route_ref,
                ),
            )
            self.assertEqual(
                (
                    send.flow_route.pair_route_ref,
                    recv.flow_route.pair_route_ref,
                ),
                (
                    contract.cross_group_route_ref,
                    contract.cross_group_route_ref,
                ),
            )
            self.assertIs(send.flow_route.role, FlowRouteRole.SOURCE)
            self.assertIs(recv.flow_route.role, FlowRouteRole.DESTINATION)

    def test_pdr_tp2_to_tp1_segmented_quotient_is_exact(self) -> None:
        source = _scheduled_pdr()
        result = build_stage4_global_action(source)
        result.validate_against(source)
        contracts = result.projection.state_transfers
        actions = result.global_dag.actions
        self.assertEqual(len(contracts), 8)
        self.assertEqual(
            sum(len(contract.segments) for contract in contracts),
            64,
        )
        self.assertEqual(sum(contract.bytes for contract in contracts), 1024)
        self.assertEqual(len(result.graph.cross_routes), 2)
        self.assertEqual(len(actions), 428)
        transfer_actions = tuple(
            action
            for action in actions
            if isinstance(action.origin_ref, StateTransferOrigin)
        )
        self.assertEqual(
            {
                kind: sum(
                    action.task_kind is kind for action in transfer_actions
                )
                for kind in (
                    SemanticTaskKind.SEND,
                    SemanticTaskKind.RECV,
                    SemanticTaskKind.WAIT,
                    SemanticTaskKind.TRANSIT,
                )
            },
            {
                SemanticTaskKind.SEND: 64,
                SemanticTaskKind.RECV: 64,
                SemanticTaskKind.WAIT: 64,
                SemanticTaskKind.TRANSIT: 32,
            },
        )
        self.assertEqual(result.source_scheduled_carrier_id, source.id)
        self.assertEqual(result.schedule_set, source.schedule_set)
        self.assertEqual(
            loads_dataclass(
                Stage4GlobalAction,
                canonical_json(result),
                path="result",
            ),
            result,
        )

    def test_version_source_global_route_and_hbm_direction_tamper_reject(self) -> None:
        source = _scheduled(fused=False)
        result = build_stage4_global_action(source)
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                result,
                schema_version="wafer_frontend.stage4_global_action/v1alpha1",
            ).validate()
        with self.assertRaisesRegex(SchemaError, "source scheduled carrier"):
            result.validate_against(_scheduled(fused=True))
        for field_name in (
            "source_projected_carrier_id",
            "projection_context_id",
            "scheduling_context_id",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(SchemaError):
                    replace(result, **{field_name: "wrong"}).validate_against(
                        source
                    )
        with self.assertRaisesRegex(SchemaError, "global_action_dag"):
            replace(
                result,
                global_dag=replace(
                    result.global_dag,
                    producer_pass="wrong",
                ),
            ).validate()

        actions = list(result.global_dag.actions)
        dma_index = next(
            index
            for index, action in enumerate(actions)
            if isinstance(action.origin_ref, StateIoOrigin)
            and action.task_kind is SemanticTaskKind.DMA_OUT
        )
        dma = actions[dma_index]
        actions[dma_index] = replace(
            dma,
            state_uses=(
                replace(dma.state_uses[0], access=StateUseAccess.READ),
            ),
        )
        with self.assertRaises(SchemaError):
            replace(
                result,
                global_dag=_recreate_global(result.global_dag, actions=actions),
            ).validate()

        actions = list(result.global_dag.actions)
        send_index = next(
            index
            for index, action in enumerate(actions)
            if isinstance(action.origin_ref, StateTransferOrigin)
            and action.task_kind is SemanticTaskKind.SEND
        )
        send = actions[send_index]
        assert send.flow_route is not None
        actions[send_index] = replace(
            send,
            flow_route=replace(
                send.flow_route,
                pair_route_ref="cross_group_route_forged",
            ),
        )
        with self.assertRaisesRegex(SchemaError, "flow route is not exact"):
            replace(
                result,
                global_dag=_recreate_global(result.global_dag, actions=actions),
            ).validate()


if __name__ == "__main__":
    unittest.main()
