from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import unittest

from test_train_forward_schedule import _schedule

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import build_train_global_action
from llm.frontend.wafer_frontend.schema.common import DType, stable_artifact_id
from llm.frontend.wafer_frontend.schema.global_action import (
    ActionStateUse,
    GlobalActionDAG,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.ir2 import (
    SemanticTaskKind,
    StateUseAccess,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.train_global_action import (
    TRAIN_GLOBAL_ACTION_SCHEMA_VERSION,
    TrainGlobalAction,
    TrainGlobalActionReplica,
)


def _global_action():
    _, _, scheduled = _schedule()
    return scheduled, build_train_global_action(scheduled)


def _rebuild_dag(dag: GlobalActionDAG, *, actions) -> GlobalActionDAG:
    return GlobalActionDAG.create(
        producer_pass=dag.producer_pass,
        source_ir1_id=dag.source_ir1_id,
        source_state_manifest_id=dag.source_state_manifest_id,
        source_projection_id=dag.source_projection_id,
        source_schedule_set_id=dag.source_schedule_set_id,
        scheduled_dags=dag.scheduled_dags,
        actions=tuple(actions),
    )


class TrainForwardGlobalActionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source, cls.result = _global_action()

    def test_dp2_tp2_exact_action_hbm_collective_and_ce_goldens(self) -> None:
        result = self.result
        self.assertEqual(result.schema_version, TRAIN_GLOBAL_ACTION_SCHEMA_VERSION)
        self.assertEqual(len(result.replicas), 2)
        for replica_index, replica in enumerate(result.replicas):
            actions = replica.global_dag.actions
            self.assertEqual(len(actions), 154)
            self.assertEqual(sum(len(action.deps) for action in actions), 232)
            self.assertEqual(sum(bool(action.deps) for action in actions), 152)
            self.assertEqual(max(len(action.deps) for action in actions), 3)
            self.assertEqual(sum(len(action.buffer_uses) for action in actions), 270)
            self.assertEqual(
                Counter(action.task_kind for action in actions),
                Counter(
                    {
                        SemanticTaskKind.COMP: 60,
                        SemanticTaskKind.DMA_IN: 30,
                        SemanticTaskKind.SEND: 16,
                        SemanticTaskKind.RECV: 16,
                        SemanticTaskKind.LOCAL_COPY: 8,
                        SemanticTaskKind.REDUCE: 8,
                        SemanticTaskKind.WAIT: 8,
                        SemanticTaskKind.BARRIER: 8,
                    }
                ),
            )

            state_actions = tuple(action for action in actions if action.state_uses)
            self.assertEqual(len(state_actions), 30)
            self.assertTrue(
                all(
                    action.task_kind is SemanticTaskKind.DMA_IN
                    and action.state_uses[0].access is StateUseAccess.READ
                    for action in state_actions
                )
            )
            self.assertEqual(sum(action.bytes for action in state_actions), 14656)

            flows = {
                action.flow.id: action.flow
                for action in actions
                if action.flow is not None
            }
            self.assertEqual(len(flows), 16)
            self.assertEqual(sum(flow.bytes for flow in flows.values()), 2048)
            self.assertEqual(
                Counter(
                    action.task_kind for action in actions
                    if action.flow is not None
                ),
                Counter(
                    {
                        SemanticTaskKind.SEND: 16,
                        SemanticTaskKind.RECV: 16,
                    }
                ),
            )
            self.assertEqual(
                len(
                    {
                        action.flow_route.pair_route_ref
                        for action in actions
                        if action.flow_route is not None
                    }
                ),
                2,
            )

            ce_actions = tuple(
                action for action in actions
                if action.op_kind is OpKind.CE_FORWARD
            )
            self.assertEqual(len(ce_actions), 2)
            binding_index = {
                binding.id: binding
                for schedule in replica.scheduled.schedule_set.schedules
                for binding in schedule.buffer_bindings
            }
            for action in ce_actions:
                self.assertEqual(len(action.deps), 1)
                self.assertEqual(action.core_order_index, 76)
                self.assertEqual(
                    tuple(binding_index[use.binding_id].dtype for use in action.buffer_uses),
                    (DType.FP16, DType.INT32, DType.FP32),
                )
                self.assertEqual(
                    tuple(
                        binding_index[use.binding_id].size_bytes
                        for use in action.buffer_uses
                    ),
                    (256, 16, 16),
                )
                predecessor = next(
                    item for item in actions if item.id == action.deps[0]
                )
                self.assertEqual(predecessor.core_order_index, 75)
                self.assertEqual(predecessor.logical_core, action.logical_core)

            local_dies = {
                action.logical_core.die_id
                for action in actions
                if action.logical_core is not None
            }
            self.assertEqual(
                local_dies,
                ({0, 1} if replica_index == 0 else {2, 3}),
            )

        self.assertEqual(
            sum(len(replica.global_dag.actions) for replica in result.replicas),
            308,
        )
        self.assertEqual(
            sum(
                action.bytes
                for replica in result.replicas
                for action in replica.global_dag.actions
                if action.state_uses
            ),
            29312,
        )
        result.validate_against(self.source)

    def test_strict_roundtrip_lineage_and_replica_isolation(self) -> None:
        result = self.result
        self.assertEqual(
            loads_dataclass(TrainGlobalAction, canonical_json(result)),
            result,
        )
        result.validate_against(self.source)

        payload = json.loads(canonical_json(result))
        del payload["source_planned_carrier_id"]
        with self.assertRaisesRegex(SchemaError, "missing required field"):
            loads_dataclass(TrainGlobalAction, json.dumps(payload))
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                result,
                schema_version="wafer_frontend.train_global_action/v0",
            ).validate()
        forged_lineage = replace(
            result,
            source_planned_carrier_id="forged.plan",
        )
        forged_lineage = replace(
            forged_lineage,
            id=stable_artifact_id(
                "train_global_action",
                forged_lineage._semantic_key(),
                schema_version=TRAIN_GLOBAL_ACTION_SCHEMA_VERSION,
            ),
        )
        with self.assertRaisesRegex(
            SchemaError, "complete train scheduling provenance"
        ):
            forged_lineage.validate_against(self.source)
        with self.assertRaisesRegex(SchemaError, "canonical DP order"):
            replace(result, replicas=tuple(reversed(result.replicas))).validate()

        imported = replace(
            result.replicas[1],
            global_dag=result.replicas[0].global_dag,
        )
        with self.assertRaisesRegex(SchemaError, "different IR-1"):
            replace(result, replicas=(result.replicas[0], imported)).validate()

    def test_ce_hbm_and_route_tampering_fail_closed(self) -> None:
        replica = self.result.replicas[0]
        actions = list(replica.global_dag.actions)

        ce_index = next(
            index
            for index, action in enumerate(actions)
            if action.op_kind is OpKind.CE_FORWARD
        )
        ce_actions = list(actions)
        ce_actions[ce_index] = replace(ce_actions[ce_index], deps=())
        ce_dag = _rebuild_dag(replica.global_dag, actions=ce_actions)
        with self.assertRaisesRegex(SchemaError, "deps must exactly equal"):
            TrainGlobalActionReplica.create(source=replica.scheduled, global_dag=ce_dag)

        state_index = next(
            index for index, action in enumerate(actions) if action.state_uses
        )
        state_actions = list(actions)
        use = state_actions[state_index].state_uses[0]
        state_actions[state_index] = replace(
            state_actions[state_index],
            state_uses=(
                ActionStateUse(use.hbm_binding_ref, StateUseAccess.WRITE),
            ),
        )
        state_dag = _rebuild_dag(replica.global_dag, actions=state_actions)
        with self.assertRaisesRegex(SchemaError, "direction-matching HBM use"):
            TrainGlobalActionReplica.create(
                source=replica.scheduled,
                global_dag=state_dag,
            )

        route_index = next(
            index
            for index, action in enumerate(actions)
            if action.flow_route is not None
        )
        foreign_route = next(
            action.flow_route.pair_route_ref
            for action in self.result.replicas[1].global_dag.actions
            if action.flow_route is not None
        )
        route_actions = list(actions)
        route_actions[route_index] = replace(
            route_actions[route_index],
            flow_route=replace(
                route_actions[route_index].flow_route,
                pair_route_ref=foreign_route,
            ),
        )
        route_dag = _rebuild_dag(replica.global_dag, actions=route_actions)
        with self.assertRaisesRegex(SchemaError, "flow route is not exact"):
            TrainGlobalActionReplica.create(
                source=replica.scheduled,
                global_dag=route_dag,
            )


if __name__ == "__main__":
    unittest.main()
