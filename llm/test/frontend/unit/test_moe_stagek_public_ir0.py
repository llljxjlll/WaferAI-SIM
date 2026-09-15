"""Source-grounded public FP32 GEMM WGRAD identity fault tests."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.full_dense_training_ce_ir0 import (
    build_dense_training_ce_backward_source,
)
from llm.frontend.wafer_frontend.passes.full_dense_training_head_backward_ir0 import (
    append_dense_training_head_backward_source,
)
from llm.frontend.wafer_frontend.schema.ir0 import IR0, OpKind
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class PublicGemmWgradSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = append_dense_training_head_backward_source(
            build_dense_training_ce_backward_source(_spec(1, 1)))
        cls.wgrad = next(node for node in cls.graph.nodes
                         if node.kind is OpKind.GEMM_WEIGHT_WGRAD)

    def _resign_workload(self, **kwargs):
        altered = replace(self.wgrad, workload=replace(self.wgrad.workload,
                                                      **kwargs))
        return IR0.create(
            producer_pass=self.graph.producer_pass,
            **{**self.graph._semantic_key(),
               "nodes": tuple(altered if node.id == self.wgrad.id else node
                              for node in self.graph.nodes)},
        )

    def test_genuine_forward_state_and_parameter_read(self):
        self.graph.validate()
        state = next(state for state in self.graph.persistent_states
                     if state.id == self.wgrad.workload.source_parameter_state_ref)
        self.assertEqual(state.identity.tensor_ref,
                         next(node for node in self.graph.nodes
                              if node.id == self.wgrad.workload.source_forward_op_ref).
                         inputs[1])

    def test_other_real_state_cannot_be_substituted_by_label(self):
        other = next(state.id for state in self.graph.persistent_states
                     if state.id != self.wgrad.workload.source_parameter_state_ref)
        with self.assertRaisesRegex(SchemaError, "real forward parameter read"):
            self._resign_workload(source_parameter_state_ref=other).validate()

    def test_other_real_forward_op_cannot_own_head_weight_gradient(self):
        other = next(node.id for node in self.graph.nodes
                     if node.kind is OpKind.GEMM
                     and node.id != self.wgrad.workload.source_forward_op_ref)
        with self.assertRaisesRegex(SchemaError, "real forward parameter read"):
            self._resign_workload(source_forward_op_ref=other).validate()

    def test_rank_rows_cannot_shift_independently_of_forward(self):
        with self.assertRaisesRegex(SchemaError, "exact FP32 parameter shape|rank rows differ"):
            self._resign_workload(k=self.wgrad.workload.k + 1).validate()


if __name__ == "__main__":
    unittest.main()
