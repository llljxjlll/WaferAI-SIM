from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.full_dense_training_ce_ir0 import (
    append_dense_training_ce_backward_source,
    build_dense_training_ce_backward_source,
)
from llm.frontend.wafer_frontend.passes.train_forward import build_train_forward_ir0
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import EdgeKind, OpKind, OpPhase
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class FullDenseTrainingCeIR0Test(unittest.TestCase):
    def test_complete_two_layer_forward_and_independent_seeded_ce_source(self) -> None:
        source = build_train_forward_ir0(_spec(1, 1))
        graph = build_dense_training_ce_backward_source(_spec(1, 1))
        self.assertEqual(graph, append_dense_training_ce_backward_source(source))
        self.assertEqual(graph.nodes[:-1], source.nodes)
        self.assertEqual(graph.persistent_states, source.persistent_states)
        self.assertEqual(len(graph.persistent_states), 15)
        ce = graph.nodes[-1]
        self.assertIs(ce.kind, OpKind.CE_BACKWARD)
        self.assertIs(ce.phase, OpPhase.DGRAD)
        original_ce = next(node for node in source.nodes if node.kind is OpKind.CE_FORWARD)
        values = {value.id: value for value in graph.values}
        loss = values[original_ce.outputs[0]]
        incoming = values[ce.inputs[2]]
        self.assertEqual(loss.producer, original_ce.id)
        self.assertIsNone(incoming.producer)
        self.assertIs(incoming.dtype, DType.FP32)
        self.assertNotEqual(incoming.id, loss.id)
        self.assertNotIn(loss.id, ce.inputs)
        self.assertEqual(ce.inputs[:2], original_ce.inputs)
        self.assertEqual(
            len(tuple(edge for edge in graph.edges if edge.kind is EdgeKind.CONTROL
                and edge.source_node == original_ce.id
                and edge.destination_node == ce.id)),
            1,
        )
        graph.validate()

    def test_current_lite_validator_explicitly_rejects_multilayer_ce_only(self) -> None:
        graph = build_dense_training_ce_backward_source(_spec(1, 1))
        with self.assertRaises(SchemaError):
            DenseIR0Validator.validate(graph)

    def test_tp_sharded_ce_requires_new_real_backward_contract(self) -> None:
        with self.assertRaisesRegex(SchemaError, "TP1"):
            build_dense_training_ce_backward_source(_spec(1, 4))

    def test_forward_loss_cannot_be_repurposed_as_input_gradient(self) -> None:
        forward = build_train_forward_ir0(_spec(1, 1))
        ce = next(node for node in forward.nodes if node.kind is OpKind.CE_FORWARD)
        bad_values = tuple(
            replace(value, producer=None) if value.id == ce.outputs[0] else value
            for value in forward.values
        )
        with self.assertRaises(SchemaError):
            append_dense_training_ce_backward_source(replace(forward, values=bad_values))


if __name__ == "__main__":
    unittest.main()
