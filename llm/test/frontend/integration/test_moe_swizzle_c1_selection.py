"""Production C1 whole-workload MoE joint-selection golden."""

from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.passes.build_moe_swizzle_execution import (
    build_moe_swizzle_execution,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_pair_feasibility import (
    build_moe_swizzle_pair_feasibility_witness,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_workload_state_abi import (
    build_moe_swizzle_workload_state_abi,
)
from llm.frontend.wafer_frontend.passes.discover_moe_swizzle import (
    discover_moe_swizzle_regions,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_comet_mesh import (
    build_comet_mesh_moe_candidate_grid,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_cost import (
    build_moe_swizzle_pair_cost_context,
    decide_moe_swizzle,
    estimate_moe_swizzle_materialized_pair_cycles,
    select_moe_swizzle_workload_deployment,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_direct_xy import (
    build_direct_xy_moe_candidates,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_problem import (
    build_moe_swizzle_problem,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_unfused import (
    build_executable_moe_unfused_baseline,
)
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleAlgorithm
from llm.frontend.wafer_frontend.schema.swizzle_moe_execution import (
    MoeScaleExecutionMode,
)
from moe_swizzle_scale_cases import build_moe_swizzle_scale_cases
from test_swizzle_moe_cost import _measured_profile


class MoeSwizzleC1SelectionTest(unittest.TestCase):
    def test_c1_bounded_corrected_frontier_selects_m2_pair(self) -> None:
        cases = build_moe_swizzle_scale_cases()
        case = cases[1]
        profile = _measured_profile()
        execution = build_moe_swizzle_execution(
            case.spec, case.oracle, MoeScaleExecutionMode.INFER_FORWARD,
        )
        decisions = []
        for region in discover_moe_swizzle_regions(
            case.spec, case.oracle, execution,
        ):
            problem = build_moe_swizzle_problem(
                region, case.spec, case.oracle, execution,
                hardware_facts=case.hardware_facts,
                endpoint_session_contract=case.endpoint_session_contract,
            )
            baseline = build_executable_moe_unfused_baseline(
                problem, case.spec, case.oracle, execution,
                calibration_profile=profile,
            )
            fused = build_direct_xy_moe_candidates(
                problem, case.spec, case.oracle, execution,
                calibration_profile=profile,
            ) + build_comet_mesh_moe_candidate_grid(
                problem, case.spec, case.oracle, execution,
                calibration_profile=profile,
            )
            decisions.append(decide_moe_swizzle(problem, baseline, fused))
        dispatch, combine = decisions
        context = build_moe_swizzle_pair_cost_context(dispatch, combine)
        argmin = lambda decision: min(
            decision.ranked_candidates,
            key=lambda candidate: (
                candidate.cost.estimated_cycles, candidate.id,
            ),
        ).id
        baseline_pair = (dispatch.baseline.id, combine.baseline.id)
        fused_pair = (argmin(dispatch), argmin(combine))
        mode_pairs = tuple(sorted({
            (left, right)
            for left in (baseline_pair[0], fused_pair[0])
            for right in (baseline_pair[1], fused_pair[1])
        }))
        ir1 = cases[0].c0_production_case.forward.n4.graph
        state_abi = build_moe_swizzle_workload_state_abi(
            ir1, execution, case.spec, case.oracle,
        )
        witnesses = tuple(
            build_moe_swizzle_pair_feasibility_witness(
                ir1, execution, case.spec, tuple(decisions), state_abi, pair,
            )
            for pair in mode_pairs
        )
        selection = select_moe_swizzle_workload_deployment(
            dispatch, combine, witnesses,
        )
        selected_pair = (
            selection.selected_dispatch_candidate_ref,
            selection.selected_combine_candidate_ref,
        )
        self.assertEqual(
            selection.id,
            "moe_swizzle_workload_selection_43f9528b59770ab6",
        )
        self.assertEqual(
            selected_pair,
            ("moe_swizzle_candidate_7fe346ff6813c4ed",
             "moe_swizzle_candidate_8b7c05539a859626"),
        )
        self.assertEqual(selection.frontier_pair_refs, (selected_pair,))
        self.assertEqual(selection.frontier_stop_lower_bound, 254.0)
        self.assertEqual(
            (selection.baseline_estimated_cycles,
             selection.selected_estimated_cycles),
            (352.0, 250.0),
        )
        self.assertGreater(
            selection.baseline_estimated_cycles
            / selection.selected_estimated_cycles,
            1.10,
        )
        candidates = {
            candidate.id: candidate
            for decision in decisions
            for candidate in decision.ranked_candidates
        }
        self.assertEqual(
            tuple(
                (
                    candidates[ref].algorithm,
                    candidates[ref].token_block_size,
                    candidates[ref].unroll_degree,
                    candidates[ref].compute_output_block_count,
                    candidates[ref].transport_output_block_count,
                )
                for ref in selected_pair
            ),
            (
                (SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A, 2, 1, 1, 1),
                (SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A, 2, 1, 1, 1),
            ),
        )
        witness = next(
            item for item in witnesses if item.candidate_refs == selected_pair
        )
        self.assertEqual(
            witness.id, "moe_whole_pair_feasibility_847f45175ada55f1",
        )
        self.assertEqual(
            (witness.whole_physical_root_count,
             witness.whole_alloc_count, witness.whole_free_count),
            (108, 100, 92),
        )
        self.assertEqual(
            tuple(
                (item.runtime_core_id, item.alloc_count,
                 item.bind_count, item.free_count)
                for item in witness.core_lifecycle_counts
            ),
            (
                (0, 6, 1, 5), (1, 6, 1, 5),
                (2, 6, 1, 6), (3, 6, 1, 6),
                (4, 6, 1, 6), (5, 6, 1, 6),
                (6, 7, 1, 6), (7, 7, 1, 6),
                (8, 6, 1, 6), (9, 6, 1, 6),
                (10, 7, 1, 6), (11, 7, 1, 6),
                (12, 6, 1, 5), (13, 6, 1, 5),
                (14, 6, 1, 6), (15, 6, 1, 6),
            ),
        )
        self.assertEqual(
            estimate_moe_swizzle_materialized_pair_cycles(
                dispatch, combine, witness, context=context,
            ),
            250.0,
        )
        self.assertEqual(context.bounds[0], (selected_pair, 250.0))
        self.assertEqual(context.bounds[1][1], 254.0)


if __name__ == "__main__":
    unittest.main()
