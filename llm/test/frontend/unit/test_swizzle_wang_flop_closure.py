from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.policies.swizzle.wang_1d import (
    generate_wang_1d_drafts,
)
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleActionKind

from test_swizzle_wang_1d import _ag_problem, _post_problem


class WangFlopClosureTest(unittest.TestCase):
    def test_distributed_comp_actions_exactly_partition_logical_gemm_flops(self) -> None:
        fixtures = (
            _ag_problem(),
            _post_problem(CollectiveKind.REDUCE_SCATTER),
            _post_problem(CollectiveKind.ALL_REDUCE),
        )
        for problem, witness in fixtures:
            with self.subTest(pattern=problem.pattern.value):
                draft = generate_wang_1d_drafts(problem, witness)[0]
                comp_flops = sum(
                    action.flops
                    for program in draft.rank_programs
                    for action in program.actions
                    if action.kind is SwizzleActionKind.COMP
                )
                self.assertEqual(comp_flops, problem.gemm.flops)


if __name__ == "__main__":
    unittest.main()
