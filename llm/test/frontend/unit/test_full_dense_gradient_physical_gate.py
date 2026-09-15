"""Real L2 training source must reject a motif masquerading as full gradients."""

from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.full_dense_gradient_physical_gate import (
    require_exact_dense_parameter_state_inventory,
    require_full_dense_physical_gradient_paths,
    require_source_gemm_wgrad_geometry,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.full_dense_gradient_requirements import (
    build_dense_full_train_requirements,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.test.frontend.unit.test_full_training_timeline_linker import (
    FullTrainingTimelineLinkerTest,
)


class FullDenseGradientPhysicalGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        FullTrainingTimelineLinkerTest.setUpClass()
        cls.fixture = FullTrainingTimelineLinkerTest
        cls.plan = cls.fixture.forward.plan
        cls.requirements = build_dense_full_train_requirements(cls.plan, steps=2)

    def test_real_motif_inventory_only_does_not_certify_gradient_producer(self) -> None:
        mapping = require_exact_dense_parameter_state_inventory(
            self.fixture.backward.manifest, self.plan, self.requirements,
        )
        self.assertEqual(len(mapping), 15)
        self.assertEqual(sum(abi.size_bytes for abi in mapping.values()), 936)
        self.assertEqual(len(self.requirements.required_gradient_producers), 30)

    def test_forward_read_only_parameter_cannot_pass_trainable_state_gate(self) -> None:
        with self.assertRaisesRegex(SchemaError, "TRAINABLE_PARAMETER"):
            require_exact_dense_parameter_state_inventory(
                self.fixture.forward.linked_forward.manifest,
                self.plan, self.requirements,
            )

    def test_true_gemm_gradient_geometry_mn_parameter_and_k_rows(self) -> None:
        nodes = {node.id: node for node in self.plan.forward_graph.nodes}
        shapes = {decl.id: decl.shape for decl in
                  self.plan.forward_graph.persistent_states}
        path = next(path for path in self.requirements.paths
                    if nodes[path.forward_op_refs[0]].kind is OpKind.GEMM
                    and len(shapes[path.parameter_state_ref]) == 2
                    and shapes[path.parameter_state_ref][0] !=
                        nodes[path.forward_op_refs[0]].workload.rank_shape[0])
        m, n = shapes[path.parameter_state_ref]
        k = nodes[path.forward_op_refs[0]].workload.rank_shape[0]
        require_source_gemm_wgrad_geometry(self.plan, path, m=m, n=n, k=k)
        with self.assertRaisesRegex(SchemaError, "M×N and K"):
            require_source_gemm_wgrad_geometry(self.plan, path,
                                               m=m, n=n, k=m)
        with self.assertRaisesRegex(SchemaError, "M×N and K"):
            require_source_gemm_wgrad_geometry(self.plan, path,
                                               m=k, n=n, k=k)

    def test_old_all_matmul_backbone_reverse_rejects_real_norm_and_attention(self) -> None:
        reverse = {ref: RecordOpcode.MATMUL for ref in
                   self.requirements.required_backbone_backward_refs}
        derivatives = {path.named_wgrad_op_ref: RecordOpcode.MATMUL
                       for path in self.requirements.paths}
        with self.assertRaisesRegex(SchemaError, "real source derivative family"):
            require_full_dense_physical_gradient_paths(
                self.fixture.backward.manifest, self.plan, self.requirements,
                None, required_backward_opcodes=reverse,
                required_wgrad_opcodes=derivatives,
            )

    def test_missing_named_parameter_derivative_cannot_be_inferred(self) -> None:
        reverse = {ref: RecordOpcode.MATMUL for ref in
                   self.requirements.required_backbone_backward_refs}
        with self.assertRaisesRegex(SchemaError, "all named parameter derivatives"):
            require_full_dense_physical_gradient_paths(
                self.fixture.backward.manifest, self.plan, self.requirements,
                None, required_backward_opcodes=reverse,
                required_wgrad_opcodes={},
            )


if __name__ == "__main__":
    unittest.main()
