from __future__ import annotations

from dataclasses import replace
from collections import Counter
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import (
    build_flexible_dense_train_plan,
)
from llm.frontend.wafer_frontend.schema.dense_backbone_reverse_source_admission import (
    build_dense_two_step_backbone_source_admission,
)
from llm.frontend.wafer_frontend.schema.dense_rope_residual_reverse_requirements import (
    DenseReverseSourceFamily,
)
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class DenseBackboneReverseSourceAdmissionTest(unittest.TestCase):
    def _plan(self, rows: int, columns: int):
        return build_flexible_dense_train_plan(
            _spec(rows, columns), RectMeshSpec(rows, columns)
        )

    def test_tp1_and_tp3_cover_every_reverse_leaf_for_two_steps(self) -> None:
        expected = {
            (1, 1): {
                DenseReverseSourceFamily.GEMM_DX: 9,
                DenseReverseSourceFamily.NORM_DX: 5,
                DenseReverseSourceFamily.ATTENTION_DX: 2,
                DenseReverseSourceFamily.ROPE_QK_DX: 2,
                DenseReverseSourceFamily.RESIDUAL_DUAL_DX: 4,
                DenseReverseSourceFamily.SWIGLU_DX: 2,
            },
            (2, 3): {
                DenseReverseSourceFamily.GEMM_DX: 9,
                DenseReverseSourceFamily.NORM_DX: 5,
                DenseReverseSourceFamily.COLLECTIVE_DX: 8,
                DenseReverseSourceFamily.ATTENTION_DX: 2,
                DenseReverseSourceFamily.ROPE_QK_DX: 2,
                DenseReverseSourceFamily.RESIDUAL_DUAL_DX: 4,
                DenseReverseSourceFamily.SWIGLU_DX: 2,
            },
        }
        for mesh, counts in expected.items():
            with self.subTest(mesh=mesh):
                plan = self._plan(*mesh)
                result = build_dense_two_step_backbone_source_admission(plan)
                result.validate_against(plan)
                self.assertEqual(Counter(item.family for item in result.leaves),
                                 counts)
                self.assertEqual(len(result.step_leaves),
                                 2 * sum(counts.values()))
                self.assertEqual(
                    {(item.step, item.backward_ref) for item in result.step_leaves},
                    {(step, leaf.backward_ref)
                     for step in range(2) for leaf in result.leaves},
                )
                self.assertTrue(result.source_gap_gate.is_complete)
                self.assertFalse(result.physical_program_admitted)

    def test_tp3_geometry_uses_rank_source_and_exact_inverse_collectives(self) -> None:
        result = build_dense_two_step_backbone_source_admission(
            self._plan(2, 3)
        )
        by_ref = {item.forward_ref: item for item in result.leaves}
        head = by_ref["T0.lm_head"]
        self.assertEqual(
            (head.saved_forward_bytes, head.rank_upstream_bytes,
             head.rank_output_bytes, len(head.parameter_state_refs)),
            (24, 48, 48, 3),
        )
        attention = by_ref["T0.layer1.attention"]
        self.assertEqual(
            (attention.saved_forward_bytes, attention.rank_upstream_bytes,
             attention.rank_output_bytes),
            (72, 24, 72),
        )
        swiglu = by_ref["T0.layer1.swiglu"]
        self.assertEqual(
            (swiglu.saved_forward_bytes, swiglu.rank_upstream_bytes,
             swiglu.rank_output_bytes),
            (96, 48, 96),
        )
        ag = by_ref["T0.layer1.ag1"]
        rs = by_ref["T0.layer1.rs1"]
        self.assertEqual(
            (ag.rank_upstream_bytes, ag.rank_output_bytes,
             ag.reverse_collective),
            (72, 24, CollectiveKind.REDUCE_SCATTER),
        )
        self.assertEqual(
            (rs.rank_upstream_bytes, rs.rank_output_bytes,
             rs.reverse_collective),
            (24, 72, CollectiveKind.ALL_GATHER),
        )

    def test_admission_cannot_be_promoted_to_physical_runtime_evidence(self) -> None:
        plan = self._plan(2, 3)
        result = build_dense_two_step_backbone_source_admission(plan)
        forged = replace(result, physical_program_admitted=True)
        with self.assertRaisesRegex(SchemaError, "source admission drifted"):
            forged.validate_against(plan)

    def test_source_identity_tamper_fails_closed(self) -> None:
        plan = self._plan(2, 3)
        graph = plan.forward_graph
        nodes = tuple(
            replace(node, outputs=("T0.layer1.attention_out",))
            if node.id == "T0.layer0.attention" else node
            for node in graph.nodes
        )
        fake = replace(plan, forward_graph=replace(graph, nodes=nodes))
        with self.assertRaises(SchemaError):
            build_dense_two_step_backbone_source_admission(fake)


if __name__ == "__main__":
    unittest.main()
