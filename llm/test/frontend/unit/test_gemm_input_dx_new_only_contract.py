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
from llm.frontend.wafer_frontend.passes.gemm_input_dx_source_contract import (
    derive_gemm_input_dx_source_provenance,
    require_public_gemm_input_dx_physical_opcode,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.gemm_input_dx_workload import GemmInputDxWorkload
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class GemmInputDxNewOnlyContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = append_dense_training_head_backward_source(
            build_dense_training_ce_backward_source(_spec(1, 1))
        )
        cls.head = next(node for node in cls.graph.nodes if node.id == "T0.lm_head")
        cls.state_ref = next(access.state_ref for access in cls.graph.state_accesses
                             if access.node_ref == cls.head.id)
        cls.dlogits = next(node.outputs[0] for node in cls.graph.nodes
                           if node.id == "T0.cross_entropy_backward")
        k, n, m = cls.head.workload.rank_shape
        cls.workload = GemmInputDxWorkload(
            k=k, m=m, n=n, source_forward_op_ref=cls.head.id,
            source_parameter_state_ref=cls.state_ref,
        )

    def test_real_two_layer_dense_head_forward_state_and_ce_upstream(self) -> None:
        workload = self.workload
        provenance = derive_gemm_input_dx_source_provenance(
            self.graph, workload, upstream_value_ref=self.dlogits,
        )
        self.assertEqual((provenance.k, provenance.m, provenance.n),
                         self.head.workload.rank_shape[::2] +
                         (self.head.workload.rank_shape[1],))
        self.assertEqual(provenance.weight_state_ref, self.state_ref)
        self.assertEqual(provenance.upstream_producer_ref,
                         "T0.cross_entropy_backward")
        self.assertEqual((workload.activation_bytes, workload.weight_bytes,
                          workload.upstream_bytes, workload.output_bytes,
                          workload.fma_ops), (8, 64, 16, 16, 32))
        with self.assertRaisesRegex(SchemaError, "public GEMM_DX_TIMING"):
            require_public_gemm_input_dx_physical_opcode(provenance)

    def test_false_source_geometry_state_substitution_and_output_dtype(self) -> None:
        with self.assertRaisesRegex(SchemaError, "geometry differs"):
            derive_gemm_input_dx_source_provenance(
                self.graph, replace(self.workload, m=8),
                upstream_value_ref=self.dlogits,
            )
        with self.assertRaisesRegex(SchemaError, "forward GEMM/StateDecl/dY"):
            derive_gemm_input_dx_source_provenance(
                self.graph, replace(self.workload,
                                    source_parameter_state_ref="wrong_state"),
                upstream_value_ref=self.dlogits,
            )
        with self.assertRaisesRegex(SchemaError, "source FP16 X/W/dY and FP32 dX"):
            replace(self.workload, output_dtype=DType.FP16).validate()
        with self.assertRaisesRegex(SchemaError, "one GEMM dX tile"):
            replace(self.workload, k=10000).validate()


if __name__ == "__main__":
    unittest.main()
