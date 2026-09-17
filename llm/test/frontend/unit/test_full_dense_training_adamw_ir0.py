"""A full Dense AdamW source must bind every true derivative and optimizer state."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_training_adamw_ir0 import build_full_dense_training_adamw_ir0
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import AdamwUpdateWorkload, OpKind, OpPhase, StateAccessMode
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class FullDenseTrainingAdamwIr0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_flexible_dense_train_plan(_spec(1, 1), RectMeshSpec(1, 1))
        cls.graph = build_full_dense_training_adamw_ir0(cls.plan)

    def test_complete_two_layer_source_has_fifteen_real_adamw_updates(self) -> None:
        graph = self.graph
        DenseIR0Validator.validate(graph, "full_dense_adamw")
        updates = tuple(node for node in graph.nodes
                        if node.kind is OpKind.OPTIMIZER_UPDATE)
        self.assertEqual((len(graph.nodes), len(updates)), (85, 15))
        self.assertEqual(len(graph.persistent_states), 75)
        self.assertEqual(len(graph.state_accesses), 100)
        self.assertEqual(sum(state.identity.kind is StateKind.TRAINABLE_PARAMETER
                             for state in graph.persistent_states), 15)
        self.assertEqual(sum(state.identity.kind is StateKind.OPTIMIZER_STEP
                             for state in graph.persistent_states), 15)
        self.assertEqual(sum(len(state.shape) == 1 and
                             state.identity.kind is StateKind.TRAINABLE_PARAMETER
                             for state in graph.persistent_states), 5)
        values = {value.id: value for value in graph.values}
        for update in updates:
            self.assertIs(type(update.workload), AdamwUpdateWorkload)
            self.assertEqual(update.workload.step, 1)
            self.assertEqual(update.inputs[1],
                             f"wgrad::{update.inputs[0]}::tp0.output")
            self.assertIs(values[update.inputs[1]].dtype, DType.FP32)
            accesses = tuple(access for access in graph.state_accesses
                             if access.node_ref == update.id)
            self.assertEqual(len(accesses), 5)
            self.assertTrue(all(access.mode is StateAccessMode.READ_WRITE
                                for access in accesses))

    def test_missing_optimizer_state_or_wrong_derivative_fails_closed(self) -> None:
        graph = self.graph
        missing = replace(graph, state_accesses=tuple(
            access for access in graph.state_accesses
            if not (access.mode is StateAccessMode.READ_WRITE
                    and access.state_ref == next(
                        state.id for state in graph.persistent_states
                        if state.identity.kind is StateKind.OPTIMIZER_MOMENT2))
        ))
        with self.assertRaisesRegex(SchemaError, "15 parameters and exact|not exact"):
            DenseIR0Validator._validate_full_dense_adamw_persistent_states(
                missing, {value.id: value for value in missing.values}, "missing_moment",
            )
        updates = [node for node in graph.nodes
                   if node.kind is OpKind.OPTIMIZER_UPDATE]
        victim = updates[0]
        wrong_gradient = replace(graph, nodes=tuple(
            replace(node, inputs=(node.inputs[0], updates[1].inputs[1],
                                  *node.inputs[2:])) if node.id == victim.id else node
            for node in graph.nodes
        ))
        with self.assertRaisesRegex(SchemaError, "own real FP32 WGRAD"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                wrong_gradient, "wrong_gradient",
            )

    def test_missing_backbone_reverse_is_rejected(self) -> None:
        graph = self.graph
        victim = next(node for node in graph.nodes
                      if node.phase is OpPhase.DGRAD
                      and node.kind is not OpKind.CE_BACKWARD
                      and node.id.startswith("backward::"))
        missing = replace(graph, nodes=tuple(node for node in graph.nodes
                                             if node.id != victim.id))
        with self.assertRaisesRegex(SchemaError, "omits or fabricates a backbone reverse"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                missing, "missing_reverse",
            )


if __name__ == "__main__":
    unittest.main()
