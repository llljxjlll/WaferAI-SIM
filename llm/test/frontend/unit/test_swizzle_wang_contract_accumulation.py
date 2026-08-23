from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.policies.swizzle.semantics import analyze_ag_gemm
from llm.frontend.wafer_frontend.policies.swizzle.wang_1d import (
    generate_wang_1d_drafts,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleActionKind,
    SwizzleProblem,
    SwizzleTensorAxisRole,
)

from test_swizzle_schema import _ag_case, _profile
from test_swizzle_wang_1d import _constraints, _group4


def _problem_for_role(role: SwizzleTensorAxisRole):
    gemm, collective, boundaries = _ag_case(role)
    axis = collective.gather_tensor_axis
    assert axis is not None
    local_shape = list(collective.output.shape)
    local_shape[axis] //= 4
    collective = replace(
        collective,
        participant_ranks=(0, 1, 2, 3),
        rank_input_bytes=16,
        input=replace(collective.input, shape=tuple(local_shape)),
    )
    witness = analyze_ag_gemm(
        gemm,
        collective,
        boundary_input_refs=boundaries,
        boundary_output_refs=(gemm.output.value_ref,),
    )
    problem = SwizzleProblem.create(
        source_ir1_id="ir1",
        fused_op_id=f"fused.ag.{role.value}",
        pattern=FusionPattern.AG_GEMM,
        gemm=gemm,
        collective=collective,
        group=_group4(),
        hardware_profile=_profile(),
        constraints=_constraints(),
    )
    return problem, witness


class WangContractAccumulationTest(unittest.TestCase):
    def test_contract_split_has_one_loop_carried_accumulator_per_rank(self) -> None:
        problem, witness = _problem_for_role(SwizzleTensorAxisRole.CONTRACT)
        draft = generate_wang_1d_drafts(problem, witness)[0]

        for program in draft.rank_programs:
            reductions = tuple(
                action
                for action in program.actions
                if action.kind is SwizzleActionKind.REDUCE
            )
            self.assertEqual(len(reductions), 3)
            for index, action in enumerate(reductions):
                self.assertEqual(len(action.input_refs), 2)
                self.assertEqual(len(action.output_refs), 1)
                self.assertEqual(len(action.deps), 2)
                if index:
                    self.assertIn(reductions[index - 1].id, action.deps)
            self.assertTrue(reductions[-1].output_refs[0].endswith("::boundary"))
            barrier = next(
                action
                for action in program.actions
                if action.kind is SwizzleActionKind.BARRIER
            )
            self.assertEqual(barrier.deps, (reductions[-1].id,))

    def test_free_axis_split_retains_disjoint_output_slices_without_reduce(self) -> None:
        problem, witness = _problem_for_role(SwizzleTensorAxisRole.FREE_LHS)
        draft = generate_wang_1d_drafts(problem, witness)[0]

        for program in draft.rank_programs:
            reductions = tuple(
                action
                for action in program.actions
                if action.kind is SwizzleActionKind.REDUCE
            )
            self.assertEqual(reductions, ())
            barrier = next(
                action
                for action in program.actions
                if action.kind is SwizzleActionKind.BARRIER
            )
            self.assertEqual(len(barrier.deps), 4)


if __name__ == "__main__":
    unittest.main()
