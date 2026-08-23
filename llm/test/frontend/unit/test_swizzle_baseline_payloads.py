from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.policies.swizzle.cost import (
    build_unfused_baseline,
)
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind

from test_swizzle_wang_1d import _ag_problem, _post_problem


def _one_phase_cycles(problem, step_bytes: int) -> float:
    profile = problem.hardware_profile
    ranks = len(problem.group.placements)
    return float(
        profile.dte_launch_cycles
        + (ranks - 1)
        * (
            profile.dte_sync_cycles
            + step_bytes / profile.lane_bytes_per_cycle
            + profile.hop_latency_cycles
        )
    )


class SwizzleBaselinePayloadTest(unittest.TestCase):
    def test_ag_uses_per_rank_input_payload(self) -> None:
        problem, witness = _ag_problem()
        cost = build_unfused_baseline(problem, witness).cost
        self.assertEqual(
            cost.prologue_cycles,
            _one_phase_cycles(problem, problem.collective.rank_input_bytes),
        )

    def test_rs_uses_per_rank_output_payload(self) -> None:
        problem, witness = _post_problem(CollectiveKind.REDUCE_SCATTER)
        cost = build_unfused_baseline(problem, witness).cost
        self.assertEqual(
            cost.epilogue_cycles,
            _one_phase_cycles(problem, problem.collective.rank_output_bytes),
        )

    def test_ar_uses_logical_payload_per_rank_for_each_phase(self) -> None:
        problem, witness = _post_problem(CollectiveKind.ALL_REDUCE)
        cost = build_unfused_baseline(problem, witness).cost
        ranks = len(problem.group.placements)
        self.assertEqual(
            cost.epilogue_cycles,
            2
            * _one_phase_cycles(
                problem,
                problem.collective.logical_bytes // ranks,
            ),
        )


if __name__ == "__main__":
    unittest.main()
