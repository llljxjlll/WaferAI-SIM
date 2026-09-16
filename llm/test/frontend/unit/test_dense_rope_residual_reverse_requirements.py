from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import (
    build_flexible_dense_train_plan,
)
from llm.frontend.wafer_frontend.schema.dense_rope_residual_reverse_requirements import (
    DenseReverseSourceFamily,
    build_dense_full_reverse_source_gap_gate,
    build_dense_rope_residual_reverse_requirements,
    build_dense_rope_residual_source_gap_gate,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class DenseRopeResidualReverseRequirementsTest(unittest.TestCase):
    def _source(self, rows: int, columns: int):
        return build_flexible_dense_train_plan(
            _spec(rows, columns), RectMeshSpec(rows, columns)
        )

    def test_exact_two_layer_source_inventory_on_tp1_and_tp3(self) -> None:
        for mesh in ((1, 1), (2, 3)):
            with self.subTest(mesh=mesh):
                plan = self._source(*mesh)
                result = build_dense_rope_residual_reverse_requirements(plan)
                result.validate_against(plan)
                self.assertEqual((len(result.rope), len(result.residual)), (2, 4))
                self.assertEqual(result.rope[0].forward_ref, "T0.layer1.rope")
                self.assertEqual(result.residual[0].forward_ref,
                                 "T0.layer1.residual2")
                self.assertTrue(all(item.forward_ref in
                                    {node.id for node in plan.forward_graph.nodes}
                                    for item in (*result.rope, *result.residual)))
                self.assertTrue(all(item.left_forward_value_ref !=
                                    item.right_forward_value_ref
                                    for item in result.residual))
                self.assertFalse(plan.full_model_backward_materialized)

    def test_tp3_rank_geometry_comes_from_production_qkv_and_residual(self) -> None:
        requirement = build_dense_rope_residual_reverse_requirements(
            self._source(2, 3)
        )
        rope = requirement.rope[0]
        self.assertEqual((rope.logical_query_heads, rope.logical_kv_heads,
                          rope.rank_query_heads, rope.rank_kv_heads,
                          rope.tp_degree, rope.head_dim), (3, 3, 1, 1, 3, 4))
        self.assertEqual((rope.position_bytes, rope.fp16_upstream_bytes,
                          rope.fp16_output_bytes, rope.inverse_rotary_pairs,
                          rope.pass_through_v_elements), (12, 72, 72, 12, 12))
        self.assertEqual(rope.required_position_trace_ref,
                         "T0.layer1.rope.position_ids")
        self.assertNotIn(rope.required_position_trace_ref,
                         {value.id for value in self._source(2, 3).forward_graph.values})
        residual = requirement.residual[0]
        self.assertEqual((residual.logical_rows, residual.rank_rows,
                          residual.tp_degree, residual.hidden_size),
                         (3, 1, 3, 12))
        self.assertEqual((residual.fp16_upstream_bytes,
                          residual.fp16_left_output_bytes,
                          residual.fp16_right_output_bytes), (24, 24, 24))

    def test_tampered_reverse_byte_extent_cannot_validate_as_source(self) -> None:
        plan = self._source(2, 3)
        requirement = build_dense_rope_residual_reverse_requirements(plan)
        wrong_residual = replace(requirement.residual[0],
                                 fp16_right_output_bytes=12)
        fake = replace(requirement,
                       residual=(wrong_residual, *requirement.residual[1:]))
        with self.assertRaisesRegex(SchemaError, "source rank geometry drifted"):
            fake.validate_against(plan)

    def test_aliasing_two_forward_residual_branches_rejected(self) -> None:
        plan = self._source(2, 3)
        graph = plan.forward_graph
        nodes = tuple(
            replace(node, inputs=(node.inputs[0], node.inputs[0]))
            if node.id == "T0.layer1.residual2" else node
            for node in graph.nodes
        )
        fake_plan = replace(plan, forward_graph=replace(graph, nodes=nodes))
        with self.assertRaises(SchemaError):
            build_dense_rope_residual_reverse_requirements(fake_plan)

    def test_full_source_gap_gate_is_exact_and_remains_closed(self) -> None:
        for mesh, expected in (((1, 1), (24, 6, 18)),
                               ((2, 3), (32, 6, 26))):
            with self.subTest(mesh=mesh):
                plan = self._source(*mesh)
                gate = build_dense_rope_residual_source_gap_gate(plan)
                self.assertEqual(
                    (len(gate.required), len(gate.contracted), len(gate.missing)),
                    expected,
                )
                self.assertFalse(gate.is_complete)
                self.assertEqual(
                    {item.family for item in gate.contracted},
                    {
                        DenseReverseSourceFamily.ROPE_QK_DX,
                        DenseReverseSourceFamily.RESIDUAL_DUAL_DX,
                    },
                )
                with self.assertRaisesRegex(
                    SchemaError, f"source contracts missing: {expected[2]}"
                ):
                    gate.require_complete_source_contracts()

    def test_source_gap_gate_rejects_name_or_wrong_family_admission(self) -> None:
        plan = self._source(2, 3)
        with self.assertRaisesRegex(SchemaError, "not a required reverse leaf"):
            build_dense_full_reverse_source_gap_gate(
                plan,
                contracted_families={
                    "backward::invented": DenseReverseSourceFamily.ROPE_QK_DX,
                },
            )
        with self.assertRaisesRegex(SchemaError, "differs from forward source"):
            build_dense_full_reverse_source_gap_gate(
                plan,
                contracted_families={
                    "backward::T0.layer1.rope":
                        DenseReverseSourceFamily.RESIDUAL_DUAL_DX,
                },
            )

    def test_only_explicit_all_family_source_contracts_open_source_gate(self) -> None:
        plan = self._source(2, 3)
        initial = build_dense_full_reverse_source_gap_gate(
            plan, contracted_families={}
        )
        admitted = {item.backward_ref: item.family for item in initial.required}
        complete = build_dense_full_reverse_source_gap_gate(
            plan, contracted_families=admitted
        )
        complete.validate_against(plan, admitted)
        self.assertTrue(complete.is_complete)
        complete.require_complete_source_contracts()
        self.assertEqual(len(complete.contracted_by_ref), 32)


if __name__ == "__main__":
    unittest.main()
