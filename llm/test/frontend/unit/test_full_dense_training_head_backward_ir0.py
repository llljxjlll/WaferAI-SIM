from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.full_dense_training_ce_ir0 import (
    build_dense_training_ce_backward_source,
)
from llm.frontend.wafer_frontend.passes.full_dense_training_head_backward_ir0 import (
    append_dense_training_head_backward_source,
)
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import EdgeKind, OpKind, OpPhase
from llm.frontend.wafer_frontend.schema.gemm_weight_wgrad_workload import (
    GemmWeightWgradWorkload,
)
from llm.frontend.wafer_frontend.schema.gemm_input_dx_workload import GemmInputDxWorkload
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class DenseTrainingHeadBackwardIR0Test(unittest.TestCase):
    def test_two_layer_ce_derivative_has_two_distinct_source_and_fp32_wgrad(self) -> None:
        source = build_dense_training_ce_backward_source(_spec(1, 1))
        graph = append_dense_training_head_backward_source(source)
        graph.validate()
        self.assertEqual(graph.nodes[:-2], source.nodes)
        self.assertEqual(graph.persistent_states, source.persistent_states)
        self.assertEqual(len(graph.persistent_states), 15)
        self.assertEqual(len(graph.nodes), 29)
        head = next(node for node in source.nodes if node.id == "T0.lm_head")
        ce_backward, wgrad, dgrad = graph.nodes[-3:]
        values = {value.id: value for value in graph.values}
        self.assertIs(ce_backward.kind, OpKind.CE_BACKWARD)
        self.assertEqual(wgrad.id, f"backward::{head.id}::{graph.state_accesses[-1].state_ref}")
        self.assertEqual(dgrad.id, f"backward::{head.id}")
        self.assertIs(wgrad.phase, OpPhase.WGRAD)
        self.assertIs(wgrad.kind, OpKind.GEMM_WEIGHT_WGRAD)
        self.assertIsInstance(wgrad.workload, GemmWeightWgradWorkload)
        self.assertEqual(wgrad.impl_ref, "gemm_weight_wgrad_timing")
        self.assertEqual((wgrad.workload.m, wgrad.workload.n, wgrad.workload.k),
                         (head.workload.rank_shape[2],
                          head.workload.rank_shape[1],
                          head.workload.rank_shape[0]))
        self.assertEqual(wgrad.workload.source_forward_op_ref, head.id)
        self.assertEqual(wgrad.workload.source_parameter_state_ref,
                         graph.state_accesses[-1].state_ref)
        self.assertEqual(wgrad.workload.gradient_bytes,
                         values[head.inputs[1]].shape[0] *
                         values[head.inputs[1]].shape[1] * 4)
        self.assertIs(dgrad.phase, OpPhase.DGRAD)
        self.assertEqual(wgrad.inputs, (head.inputs[0], ce_backward.outputs[0]))
        self.assertEqual(dgrad.inputs, (head.inputs[1], ce_backward.outputs[0]))
        self.assertIs(dgrad.kind, OpKind.GEMM_INPUT_DX)
        self.assertIsInstance(dgrad.workload, GemmInputDxWorkload)
        self.assertEqual(dgrad.impl_ref, "gemm_input_dx_timing")
        self.assertEqual(dgrad.workload.source_forward_op_ref, head.id)
        self.assertEqual(dgrad.workload.source_parameter_state_ref,
                         graph.state_accesses[-1].state_ref)
        self.assertIs(values[wgrad.outputs[0]].dtype, DType.FP32)
        self.assertEqual(values[wgrad.outputs[0]].shape,
                         values[head.inputs[1]].shape)
        self.assertIs(values[dgrad.outputs[0]].dtype, DType.FP16)
        self.assertEqual(values[dgrad.outputs[0]].shape,
                         values[head.inputs[0]].shape)
        self.assertIs(values[ce_backward.inputs[2]].dtype, DType.FP32)
        self.assertIsNone(values[ce_backward.inputs[2]].producer)
        self.assertEqual(graph.state_accesses[-1].node_ref, dgrad.id)
        self.assertEqual(graph.state_accesses[-1].rank, 0)
        self.assertEqual(
            {(edge.source_node, edge.destination_node) for edge in graph.edges
             if edge.kind is EdgeKind.DATA and edge.value_id == ce_backward.outputs[0]},
            {(ce_backward.id, wgrad.id), (ce_backward.id, dgrad.id)},
        )

    def test_missing_ce_backward_or_wrong_dlogits_dtype_fails_closed(self) -> None:
        source = build_dense_training_ce_backward_source(_spec(1, 1))
        with self.assertRaises(SchemaError):
            append_dense_training_head_backward_source(replace(
                source, nodes=source.nodes[:-1],
            ))
        dlogits = source.nodes[-1].outputs[0]
        with self.assertRaises(SchemaError):
            append_dense_training_head_backward_source(replace(
                source, values=tuple(
                    replace(value, dtype=DType.FP32) if value.id == dlogits else value
                    for value in source.values
                ),
            ))

    def test_head_source_remains_incomplete_for_full_training_validator(self) -> None:
        graph = append_dense_training_head_backward_source(
            build_dense_training_ce_backward_source(_spec(1, 1))
        )
        with self.assertRaises(SchemaError):
            DenseIR0Validator.validate(graph)


if __name__ == "__main__":
    unittest.main()
