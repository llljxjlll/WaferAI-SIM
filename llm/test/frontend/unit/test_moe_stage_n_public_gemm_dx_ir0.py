"""Public FP16 input-gradient source and loss-upstream fault injection."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.full_dense_training_ce_ir0 import (
    build_dense_training_ce_backward_source,
)
from llm.frontend.wafer_frontend.passes.full_dense_training_head_backward_ir0 import (
    append_dense_training_head_backward_source,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.gemm_input_dx_workload import (
    GemmInputDxWorkload,
)
from llm.frontend.wafer_frontend.schema.ir0 import IR0, OpKind
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class PublicGemmDxSourceIR0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = append_dense_training_head_backward_source(
            build_dense_training_ce_backward_source(_spec(1, 1)))
        cls.head = next(node for node in cls.original.nodes
                        if node.id == "T0.lm_head")
        cls.dgrad = cls.original.nodes[-1]
        cls.parameter = next(item for item in cls.original.persistent_states
                             if item.identity.tensor_ref == cls.head.inputs[1])
        cls.workload = GemmInputDxWorkload(
            k=cls.head.workload.rank_shape[0],
            m=cls.head.workload.rank_shape[2],
            n=cls.head.workload.rank_shape[1],
            source_forward_op_ref=cls.head.id,
            source_parameter_state_ref=cls.parameter.id,
        )

    def _resign(self, *, workload=None, weight_ref=None, upstream_ref=None,
                output_dtype=DType.FP16):
        old = self.dgrad
        node = replace(old, kind=OpKind.GEMM_INPUT_DX,
                       impl_ref="gemm_input_dx_timing",
                       phase=old.phase,
                       workload=self.workload if workload is None else workload,
                       inputs=(self.head.inputs[1] if weight_ref is None
                               else weight_ref,
                               old.inputs[1] if upstream_ref is None
                               else upstream_ref),
                       math=replace(old.math, accumulation_dtype=DType.FP32))
        return IR0.create(
            producer_pass=self.original.producer_pass,
            **{**self.original._semantic_key(),
               "nodes": (*self.original.nodes[:-1], node),
               "values": tuple(replace(value, dtype=output_dtype)
                               if value.id == old.outputs[0] else value
                               for value in self.original.values)},
        )

    def test_real_lm_head_weight_read_and_ce_backward_fp16_dy_to_fp16_dx(self):
        graph = self._resign()
        graph.validate()
        self.assertEqual(graph.nodes[-1].kind, OpKind.GEMM_INPUT_DX)
        self.assertEqual(graph.nodes[-1].impl_ref, "gemm_input_dx_timing")
        self.assertEqual(graph.nodes[-1].workload.output_bytes,
                         2 * self.workload.k * self.workload.m)

    def test_forward_state_ref_cannot_name_a_different_real_parameter(self):
        other = next(item.id for item in self.original.persistent_states
                     if item.id != self.parameter.id)
        with self.assertRaisesRegex(SchemaError, "real forward X/W StateDecl"):
            self._resign(workload=replace(
                self.workload, source_parameter_state_ref=other)).validate()

    def test_a_different_real_forward_gemm_cannot_be_substituted(self):
        other = next(node.id for node in self.original.nodes
                     if node.kind is OpKind.GEMM and node.id != self.head.id)
        with self.assertRaisesRegex(SchemaError, "real forward X/W StateDecl"):
            self._resign(workload=replace(
                self.workload, source_forward_op_ref=other)).validate()

    def test_upstream_and_fp16_output_are_real_physical_operands(self):
        with self.assertRaisesRegex(SchemaError, "consumer node|true upstream"):
            self._resign(upstream_ref=self.head.outputs[0]).validate()
        with self.assertRaisesRegex(SchemaError, "owned FP16 dX"):
            self._resign(output_dtype=DType.FP32).validate()


if __name__ == "__main__":
    unittest.main()
