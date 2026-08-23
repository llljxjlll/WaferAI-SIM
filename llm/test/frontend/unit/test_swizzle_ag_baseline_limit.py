from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.policies.swizzle.cost import build_unfused_baseline
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleHardwareProfile

from test_swizzle_wang_1d import _ag_problem


class SwizzleAgBaselineLimitTest(unittest.TestCase):
    def test_ag_baseline_uses_one_local_shard_per_ring_step(self) -> None:
        """Each of the ranks-1 ring steps transfers one rank-local shard."""

        problem, witness = _ag_problem()
        source = problem.hardware_profile
        profile = SwizzleHardwareProfile.create(
            peak_flops_per_cycle=source.peak_flops_per_cycle,
            confidence_fraction=source.confidence_fraction,
            efficiency_points=source.efficiency_points,
            dte_launch_cycles=0,
            dte_sync_cycles=0,
            hop_latency_cycles=0,
            lane_bytes_per_cycle=source.lane_bytes_per_cycle,
            max_inflight_dte=source.max_inflight_dte,
            min_transfer_bytes=source.min_transfer_bytes,
            efficient_tile_floor=source.efficient_tile_floor,
            sram_budget_bytes=source.sram_budget_bytes,
            double_buffer_supported=source.double_buffer_supported,
        )
        problem = type(problem).create(
            source_ir1_id=problem.source_ir1_id,
            fused_op_id=problem.fused_op_id,
            pattern=problem.pattern,
            gemm=problem.gemm,
            collective=problem.collective,
            group=problem.group,
            hardware_profile=profile,
            constraints=problem.constraints,
        )
        baseline = build_unfused_baseline(problem, witness)
        ranks = len(problem.collective.participant_ranks)
        current_step_cycles = (
            (ranks - 1)
            * problem.collective.rank_input_bytes
            / profile.lane_bytes_per_cycle
        )

        self.assertEqual(baseline.cost.prologue_cycles, current_step_cycles)


if __name__ == "__main__":
    unittest.main()
