"""Two full AdamW source steps must preserve every persistent state version."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_adamw_ir0 import (
    build_full_dense_training_two_step_adamw_ir0,
)
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.ir0 import (
    EdgeKind, OpKind, StateAccessMode,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class FullDenseTwoStepAdamwIR0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        plan = build_flexible_dense_train_plan(_spec(1, 1), RectMeshSpec(1, 1))
        cls.graph = build_full_dense_training_two_step_adamw_ir0(plan)

    def test_two_full_layers_and_all_seventy_five_state_versions(self) -> None:
        graph = self.graph
        DenseIR0Validator.validate(graph, "two_step_adamw")
        self.assertEqual((len(graph.nodes), len(graph.persistent_states),
                          len(graph.state_accesses)), (170, 75, 200))
        updates = tuple(node for node in graph.nodes
                        if node.kind is OpKind.OPTIMIZER_UPDATE)
        self.assertEqual(len(updates), 30)
        self.assertEqual({node.workload.step for node in updates}, {1, 2})
        self.assertEqual(sum(state.identity.kind is StateKind.OPTIMIZER_STEP
                             for state in graph.persistent_states), 15)
        version_edges = tuple(edge for edge in graph.edges
                              if edge.kind is EdgeKind.CONTROL
                              and edge.source_node.startswith("adamw_update::"))
        self.assertEqual(len(version_edges), 40)
        self.assertEqual(sum(edge.destination_node.startswith("adamw_update::")
                             for edge in version_edges), 15)
        self.assertEqual(sum(access.mode is StateAccessMode.READ_WRITE
                             for access in graph.state_accesses), 150)

    def test_missing_optimizer_version_edge_fails_closed(self) -> None:
        graph = self.graph
        missing = replace(graph, edges=tuple(edge for edge in graph.edges
                                             if not (edge.kind is EdgeKind.CONTROL
                                                     and edge.source_node.startswith("adamw_update::")
                                                     and edge.destination_node.startswith("adamw_update::"))
                                             or not edge.source_node.endswith("::step0")
                                             or "layer0" not in edge.source_node))
        with self.assertRaisesRegex(SchemaError, "AdamW STORE must control"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                missing, "missing_optimizer_version",
            )

    def test_wrong_second_step_or_missing_moment_read_fails_closed(self) -> None:
        graph = self.graph
        victim = next(node for node in graph.nodes
                      if node.kind is OpKind.OPTIMIZER_UPDATE
                      and node.id.endswith("::step1"))
        wrong_step = replace(graph, nodes=tuple(
            replace(node, workload=replace(node.workload, step=1))
            if node.id == victim.id else node for node in graph.nodes
        ))
        with self.assertRaisesRegex(SchemaError, "own real FP32 WGRAD"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                wrong_step, "wrong_step",
            )
        moment = next(access for access in graph.state_accesses
                      if access.node_ref == victim.id
                      and next(state.identity.kind for state in graph.persistent_states
                               if state.id == access.state_ref)
                         is StateKind.OPTIMIZER_MOMENT1)
        missing = replace(graph, state_accesses=tuple(
            access for access in graph.state_accesses if access.id != moment.id
        ))
        with self.assertRaisesRegex(SchemaError, "exact master/m/v/step"):
            DenseIR0Validator._validate_full_dense_adamw_persistent_states(
                missing, {value.id: value for value in missing.values},
                "missing_moment_read",
            )


if __name__ == "__main__":
    unittest.main()
