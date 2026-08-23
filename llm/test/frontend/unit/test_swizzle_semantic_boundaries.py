from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.swizzle.semantics import (
    analyze_ag_gemm,
    analyze_gemm_ar,
    analyze_gemm_rs,
)
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleTensorAxisRole

from test_swizzle_schema import _ag_case, _post_collective


class SwizzleSemanticBoundaryTest(unittest.TestCase):
    def test_post_gemm_local_weight_is_not_a_fusion_boundary(self) -> None:
        for kind, analyzer in (
            (CollectiveKind.REDUCE_SCATTER, analyze_gemm_rs),
            (CollectiveKind.ALL_REDUCE, analyze_gemm_ar),
        ):
            with self.subTest(kind=kind):
                gemm, collective = _post_collective(kind)
                local_weight_gemm = replace(
                    gemm,
                    boundary_input_refs=(gemm.lhs.value_ref,),
                    local_operand_refs=(gemm.rhs.value_ref,),
                )
                witness = analyzer(
                    local_weight_gemm,
                    collective,
                    boundary_input_refs=(gemm.lhs.value_ref,),
                    boundary_output_refs=(collective.output.value_ref,),
                )
                self.assertEqual(
                    witness.boundary_input_refs,
                    (gemm.lhs.value_ref,),
                )

    def test_ag_local_weight_keeps_only_collective_source_boundary(self) -> None:
        gemm, collective, _ = _ag_case(SwizzleTensorAxisRole.CONTRACT)
        other = gemm.rhs
        local_weight_gemm = replace(
            gemm,
            boundary_input_refs=(collective.input.value_ref,),
            local_operand_refs=(other.value_ref,),
        )
        witness = analyze_ag_gemm(
            local_weight_gemm,
            collective,
            boundary_input_refs=(collective.input.value_ref,),
            boundary_output_refs=(gemm.output.value_ref,),
        )
        self.assertEqual(witness.boundary_input_refs, (collective.input.value_ref,))

    def test_collective_layout_transition_preserves_exact_provenance(self) -> None:
        gemm, collective = _post_collective(CollectiveKind.REDUCE_SCATTER)
        transitioned = replace(
            collective,
            output=replace(collective.output, layout="MN_shard_tp"),
        )
        witness = analyze_gemm_rs(
            gemm,
            transitioned,
            boundary_input_refs=gemm.boundary_input_refs,
            boundary_output_refs=(transitioned.output.value_ref,),
        )
        self.assertTrue(witness.input_layout_closed)
        self.assertTrue(witness.output_layout_closed)
        self.assertNotEqual(transitioned.input.layout, transitioned.output.layout)

    def test_boundary_local_overlap_and_omission_fail_closed(self) -> None:
        gemm, collective = _post_collective(CollectiveKind.REDUCE_SCATTER)
        with self.assertRaisesRegex(SchemaError, "disjoint"):
            replace(
                gemm,
                local_operand_refs=(gemm.rhs.value_ref,),
            ).validate()
        omitted = replace(gemm, boundary_input_refs=(gemm.lhs.value_ref,))
        with self.assertRaisesRegex(SchemaError, "do not close"):
            analyze_gemm_rs(
                omitted,
                collective,
                boundary_input_refs=omitted.boundary_input_refs,
                boundary_output_refs=(collective.output.value_ref,),
            )


if __name__ == "__main__":
    unittest.main()
