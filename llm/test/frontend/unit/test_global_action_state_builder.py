from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.global_action import (
    build_global_action_dag,
)
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    DirectAllGatherPolicy,
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import (
    NaiveIntraDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.schema.global_action import (
    ActionStateUse,
    GlobalActionDAG,
)
from llm.frontend.wafer_frontend.schema.ir0 import (
    CollectiveKind,
    OpKind,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    SemanticTaskKind,
    StateUseAccess,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from test_naive_project_to_ir2 import _partitioned_graph
from test_naive_project_state_transfer import (
    _with_sram_capacity,
)


def _stateful_tp2_pipeline():
    graph = _with_sram_capacity(
        _partitioned_graph(tp=2), 2 * 1024 * 1024
    )
    fusion_plans = tuple(
        NaiveInterDiePolicy().plan(graph, skeleton, graph.profile)
        for skeleton in graph.fused_op_skeletons
    )
    fused_members = {
        member_id
        for skeleton in graph.fused_op_skeletons
        for member_id in skeleton.member_node_ids
    }
    standalone_nodes = tuple(
        node
        for node in graph.nodes
        if node.id not in fused_members
        and node.kind is OpKind.COLLECTIVE
        and getattr(node.workload, "collective", None)
        is CollectiveKind.ALL_GATHER
    )
    standalone_plans = tuple(
        DirectAllGatherPolicy().plan(graph, node, graph.profile)
        for node in standalone_nodes
    )
    projection = NaiveProjectToIR2().run(
        graph,
        fusion_plans,
        standalone_plans,
        state_transfers=(),
    )
    schedule_set = NaiveIntraDiePolicy().schedule(projection, graph)
    actions = build_global_action_dag(graph, projection, schedule_set)
    return graph, projection, schedule_set, actions


def _recreate(
    dag: GlobalActionDAG,
    **changes: object,
) -> GlobalActionDAG:
    fields = dag._semantic_key()
    fields.update(changes)
    return GlobalActionDAG.create(
        producer_pass=dag.producer_pass,
        **fields,
    )


class GlobalActionStateBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.graph,
            cls.projection,
            cls.schedule_set,
            cls.global_dag,
        ) = _stateful_tp2_pipeline()

    def test_tp2_l1_parameter_and_kv_numeric_golden(self) -> None:
        graph = self.graph
        global_dag = self.global_dag
        manifest = graph.persistent_state_manifest
        assert manifest is not None
        self.assertEqual(global_dag.source_state_manifest_id, manifest.id)
        self.assertEqual(
            canonical_digest(
                build_global_action_dag(
                    graph,
                    self.projection,
                    self.schedule_set,
                )
            ),
            canonical_digest(global_dag),
        )
        bindings = {binding.id: binding for binding in manifest.bindings}
        declarations = {
            declaration.id: declaration
            for declaration in manifest.declarations
        }
        hbm_binding_ids = set(bindings)
        node_index = {node.id: node for node in graph.nodes}

        for scheduled_ref in global_dag.scheduled_dags:
            actions = tuple(
                action
                for action in global_dag.actions
                if action.source.dag_id == scheduled_ref.dag_id
            )
            state_actions = tuple(
                action for action in actions if action.state_uses
            )
            self.assertEqual(len(actions), 43)
            self.assertEqual(len(state_actions), 11)
            self.assertEqual(
                Counter(
                    use.access
                    for action in state_actions
                    for use in action.state_uses
                ),
                Counter(
                    {
                        StateUseAccess.READ: 9,
                        StateUseAccess.WRITE: 2,
                    }
                ),
            )
            self.assertEqual(
                len(
                    {
                        use.hbm_binding_ref
                        for action in state_actions
                        for use in action.state_uses
                    }
                ),
                11,
            )
            self.assertEqual(
                sum(
                    action.bytes
                    for action in state_actions
                    if action.state_uses[0].access is StateUseAccess.READ
                ),
                1_115_648,
            )
            self.assertEqual(
                sum(
                    action.bytes
                    for action in state_actions
                    if action.state_uses[0].access is StateUseAccess.WRITE
                ),
                8_192,
            )
            parameter_actions = tuple(
                action
                for action in state_actions
                if declarations[
                    bindings[
                        action.state_uses[0].hbm_binding_ref
                    ].state_ref
                ].identity.kind
                is StateKind.PARAMETER
            )
            kv_actions = tuple(
                action
                for action in state_actions
                if declarations[
                    bindings[
                        action.state_uses[0].hbm_binding_ref
                    ].state_ref
                ].identity.kind
                in (StateKind.KV_KEY, StateKind.KV_VALUE)
            )
            self.assertEqual(len(parameter_actions), 9)
            self.assertEqual(
                sum(action.bytes for action in parameter_actions),
                1_115_648,
            )
            self.assertEqual(len(kv_actions), 2)
            self.assertEqual(
                sum(
                    action.bytes
                    for action in kv_actions
                    if action.state_uses[0].access is StateUseAccess.READ
                ),
                0,
            )
            self.assertEqual(
                sum(
                    action.bytes
                    for action in kv_actions
                    if action.state_uses[0].access is StateUseAccess.WRITE
                ),
                8_192,
            )
            self.assertEqual(
                sum(
                    action.dma is not None
                    and len(action.dma.access_task_refs) > 1
                    for action in parameter_actions
                ),
                2,
            )
            self.assertFalse(
                any(
                    use.binding_id in hbm_binding_ids
                    for action in state_actions
                    for use in action.buffer_uses
                )
            )
            for action in actions:
                if (
                    action.task_kind is SemanticTaskKind.COMP
                    and action.member_id is not None
                    and node_index[action.member_id].kind is OpKind.ATTENTION
                ):
                    self.assertEqual(action.state_uses, ())
                    assert action.compute is not None
                    self.assertEqual(
                        tuple(
                            operand.value_id
                            for operand in action.compute.inputs
                        ),
                        node_index[action.member_id].inputs,
                    )

    def test_schedule_state_use_quotient_is_exact(self) -> None:
        dag = self.global_dag
        state_index = next(
            index
            for index, action in enumerate(dag.actions)
            if action.state_uses
        )
        action = dag.actions[state_index]
        actions = list(dag.actions)
        actions[state_index] = replace(action, state_uses=())
        with self.assertRaisesRegex(SchemaError, "direction-matching HBM"):
            _recreate(dag, actions=tuple(actions)).validate_against(
                self.graph,
                self.projection,
                self.schedule_set,
            )

        same_direction = next(
            candidate
            for candidate in dag.actions
            if candidate.id != action.id
            and candidate.state_uses
            and candidate.state_uses[0].access
            is action.state_uses[0].access
        )
        actions[state_index] = replace(
            action,
            state_uses=(
                ActionStateUse(
                    same_direction.state_uses[0].hbm_binding_ref,
                    action.state_uses[0].access,
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "exact schedule quotient"):
            _recreate(dag, actions=tuple(actions)).validate_against(
                self.graph,
                self.projection,
                self.schedule_set,
            )

    def test_manifest_and_dma_tampering_fail_closed(self) -> None:
        dag = self.global_dag
        with self.assertRaisesRegex(SchemaError, "manifest provenance"):
            _recreate(
                dag,
                source_state_manifest_id="wrong-manifest",
            ).validate_against(
                self.graph,
                self.projection,
                self.schedule_set,
            )
        index = next(
            index
            for index, action in enumerate(dag.actions)
            if action.dma is not None
        )
        action = dag.actions[index]
        assert action.dma is not None
        actions = list(dag.actions)
        actions[index] = replace(
            action,
            dma=replace(action.dma, state_ref="forged-state"),
        )
        with self.assertRaisesRegex(SchemaError, "source task semantics"):
            _recreate(dag, actions=tuple(actions)).validate_against(
                self.graph,
                self.projection,
                self.schedule_set,
            )


if __name__ == "__main__":
    unittest.main()
