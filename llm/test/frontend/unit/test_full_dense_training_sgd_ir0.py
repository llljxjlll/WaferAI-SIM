"""Production Dense source SGD must consume all 15 real derivatives and states."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import (
    build_flexible_dense_train_plan,
)
from llm.frontend.wafer_frontend.passes.full_dense_training_sgd_ir0 import (
    build_full_dense_training_sgd_ir0,
)
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import OpKind, StateAccessMode
from llm.frontend.wafer_frontend.schema.common import MeshAxisName
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateAccess, StateKind,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class FullDenseTrainingSgdIr0Test(unittest.TestCase):
    def _build(self):
        plan = build_flexible_dense_train_plan(_spec(1, 1), RectMeshSpec(1, 1))
        return plan, build_full_dense_training_sgd_ir0(plan)

    def test_every_real_wgrad_updates_exactly_one_trainable_state(self) -> None:
        plan, graph = self._build()
        DenseIR0Validator.validate(graph, "full_dense_sgd")
        updates = tuple(node for node in graph.nodes
                        if node.kind is OpKind.OPTIMIZER_UPDATE)
        self.assertEqual(len(updates), len(plan.parameter_templates))
        self.assertEqual(len(graph.persistent_states), len(updates))
        self.assertTrue(all(state.identity.kind is StateKind.TRAINABLE_PARAMETER
                            and state.access is PersistentStateAccess.READ_WRITE
                            for state in graph.persistent_states))
        values = {value.id: value for value in graph.values}
        accesses = {access.node_ref: access for access in graph.state_accesses
                    if access.mode is StateAccessMode.READ_WRITE}
        self.assertEqual(set(accesses), {node.id for node in updates})
        states = {state.id: state for state in graph.persistent_states}
        for node in updates:
            weight, gradient = (values[ref] for ref in node.inputs)
            output = values[node.outputs[0]]
            self.assertIs(gradient.dtype, DType.FP32)
            self.assertIs(weight.dtype, DType.FP16)
            self.assertIs(output.dtype, DType.FP16)
            self.assertEqual(gradient.producer,
                             f"wgrad::{weight.id}::tp0")
            self.assertEqual(states[accesses[node.id].state_ref].identity.tensor_ref,
                             weight.id)
            self.assertEqual(output.alias_set, f"trainable:{weight.id}")
        self.assertEqual(sum(len(state.shape) == 1
                             for state in graph.persistent_states), 5)

    def test_wrong_sgd_wgrad_or_state_is_rejected(self) -> None:
        _, graph = self._build()
        update = next(node for node in graph.nodes
                      if node.kind is OpKind.OPTIMIZER_UPDATE)
        swapped = replace(graph, nodes=tuple(
            replace(node, inputs=(node.inputs[0],
                    next(other.inputs[1] for other in graph.nodes
                         if other.kind is OpKind.OPTIMIZER_UPDATE
                         and other.id != update.id)))
            if node.id == update.id else node for node in graph.nodes
        ))
        with self.assertRaisesRegex(SchemaError, "own FP32 WGRAD"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                swapped, "swapped_wgrad")
        missing = replace(graph, state_accesses=tuple(
            access for access in graph.state_accesses
            if access.node_ref != update.id))
        with self.assertRaisesRegex(SchemaError, "not exact"):
            DenseIR0Validator._validate_full_dense_backward_persistent_states(
                missing, {value.id: value for value in missing.values},
                {update.mesh_ref: {MeshAxisName.TP: 1}},
                "missing_update_state",
            )


if __name__ == "__main__":
    unittest.main()
