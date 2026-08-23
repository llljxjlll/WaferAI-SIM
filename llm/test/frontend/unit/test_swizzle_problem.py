from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.policies.swizzle.problem import (
    build_swizzle_problem,
)
from llm.frontend.wafer_frontend.policies.swizzle.semantics import (
    analyze_gemm_rs,
    validate_semantic_witness,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleAlgorithm,
    SwizzleConstraints,
)

from _fixtures import valid_ir1
from test_swizzle_schema import _profile


class SwizzleProblemBuilderTest(unittest.TestCase):
    def test_valid_ir1_local_weight_closes_problem_and_rs_semantics(self) -> None:
        ir1 = valid_ir1()
        skeleton = ir1.fused_op_skeletons[0]
        problem = build_swizzle_problem(
            ir1,
            skeleton,
            _profile(),
            SwizzleConstraints(
                allowed_algorithms=(SwizzleAlgorithm.UNFUSED,),
                max_candidates=32,
                max_actions=512,
                max_buffers=64,
                max_chunk_count=32,
                allow_unroll_two=True,
            ),
        )
        self.assertEqual(problem.pattern, FusionPattern.GEMM_RS)
        self.assertEqual(problem.gemm.boundary_input_refs, skeleton.boundary_inputs)
        self.assertEqual(
            problem.gemm.local_operand_refs,
            (f"{problem.gemm.node_ref}::implicit_rhs",),
        )
        self.assertNotEqual(
            problem.collective.input.layout,
            problem.collective.output.layout,
        )
        witness = analyze_gemm_rs(
            problem.gemm,
            problem.collective,
            boundary_input_refs=skeleton.boundary_inputs,
            boundary_output_refs=skeleton.boundary_outputs,
        )
        validate_semantic_witness(
            witness,
            problem.gemm,
            problem.collective,
        )
        self.assertEqual(witness.boundary_input_refs, ("p_v_in",))


if __name__ == "__main__":
    unittest.main()
