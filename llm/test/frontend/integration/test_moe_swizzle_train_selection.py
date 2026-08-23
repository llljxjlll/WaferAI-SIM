"""Production C1/C2 TRAIN_FORWARD corrected-cost joint selections."""

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


def _build_train_selection(case_index: int):
    cases = build_moe_swizzle_scale_cases()
    case = cases[case_index]
    execution = build_moe_swizzle_execution(
        case.spec, case.oracle, MoeScaleExecutionMode.TRAIN_FORWARD,
    )
    profile = _measured_profile()
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
        key=lambda candidate: (candidate.cost.estimated_cycles, candidate.id),
    ).id
    baseline_pair = (dispatch.baseline.id, combine.baseline.id)
    fused_pair = (argmin(dispatch), argmin(combine))
    mode_pairs = {
        (left, right)
        for left in (baseline_pair[0], fused_pair[0])
        for right in (baseline_pair[1], fused_pair[1])
    }
    ir1 = cases[0].c0_production_case.forward.n4.graph
    state_abi = build_moe_swizzle_workload_state_abi(
        ir1, execution, case.spec, case.oracle,
    )
    witnesses = {
        pair: build_moe_swizzle_pair_feasibility_witness(
            ir1, execution, case.spec, tuple(decisions), state_abi, pair,
        )
        for pair in sorted(mode_pairs)
    }
    best = min(
        estimate_moe_swizzle_materialized_pair_cycles(
            dispatch, combine, witness, context=context,
        )
        for witness in witnesses.values()
        if witness.feasible
    )
    for pair, lower in context.bounds:
        if lower >= best:
            break
        witness = witnesses.get(pair)
        if witness is None:
            witness = build_moe_swizzle_pair_feasibility_witness(
                ir1, execution, case.spec, tuple(decisions), state_abi, pair,
            )
            witnesses[pair] = witness
        if witness.feasible:
            best = min(
                best,
                estimate_moe_swizzle_materialized_pair_cycles(
                    dispatch, combine, witness, context=context,
                ),
            )
    selection = select_moe_swizzle_workload_deployment(
        dispatch, combine, tuple(witnesses[pair] for pair in sorted(witnesses)),
    )
    return tuple(decisions), context, witnesses, selection


_CORE = (
    (0, 7, 1, 5), (1, 7, 1, 5),
    (2, 7, 1, 6), (3, 7, 1, 6),
    (4, 7, 1, 6), (5, 7, 1, 6),
    (6, 8, 1, 6), (7, 8, 1, 6),
    (8, 7, 1, 6), (9, 7, 1, 6),
    (10, 8, 1, 6), (11, 8, 1, 6),
    (12, 7, 1, 5), (13, 7, 1, 5),
    (14, 7, 1, 6), (15, 7, 1, 6),
)


class MoeSwizzleTrainSelectionTest(unittest.TestCase):
    def _assert_selection(
        self, case_index, *, selection_id, pair, algorithms, cycles,
        frontier, stop, witness_id,
    ) -> None:
        decisions, context, witnesses, selection = _build_train_selection(
            case_index,
        )
        self.assertEqual(selection.id, selection_id)
        self.assertEqual(
            (selection.selected_dispatch_candidate_ref,
             selection.selected_combine_candidate_ref),
            pair,
        )
        self.assertEqual(selection.frontier_pair_refs, frontier)
        self.assertEqual(selection.frontier_stop_lower_bound, stop)
        self.assertEqual(
            (selection.baseline_estimated_cycles,
             selection.selected_estimated_cycles),
            cycles,
        )
        self.assertGreater(cycles[0] / cycles[1], 1.10)
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
                for ref in pair
            ),
            algorithms,
        )
        witness = witnesses[pair]
        self.assertEqual(witness.id, witness_id)
        self.assertEqual(
            (witness.whole_physical_root_count,
             witness.whole_alloc_count, witness.whole_free_count),
            (124, 116, 92),
        )
        self.assertEqual(
            tuple(
                (item.runtime_core_id, item.alloc_count,
                 item.bind_count, item.free_count)
                for item in witness.core_lifecycle_counts
            ),
            _CORE,
        )
        self.assertEqual(
            estimate_moe_swizzle_materialized_pair_cycles(
                decisions[0], decisions[1], witness, context=context,
            ),
            cycles[1],
        )

    def test_c1_train_bounded_corrected_frontier(self) -> None:
        selected = (
            "moe_swizzle_candidate_d05c7b19114d6e16",
            "moe_swizzle_candidate_d9985218e397aa45",
        )
        self._assert_selection(
            1,
            selection_id="moe_swizzle_workload_selection_68ad344b2d3714c2",
            pair=selected,
            algorithms=(
                (SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A, 2, 1, 1, 1),
                (SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A, 2, 1, 1, 1),
            ),
            cycles=(359.0, 257.0),
            frontier=(
                selected,
                ("moe_swizzle_candidate_2d24bd07e4d69ff9",
                 "moe_swizzle_candidate_d9985218e397aa45"),
            ),
            stop=259.0,
            witness_id="moe_whole_pair_feasibility_b244e14f1db1b35c",
        )

    def test_c2_train_bounded_corrected_frontier(self) -> None:
        selected = (
            "moe_swizzle_candidate_5a990b1fdcb7e740",
            "moe_swizzle_candidate_93276d879f73dfd6",
        )
        self._assert_selection(
            2,
            selection_id="moe_swizzle_workload_selection_7cf8b0f1f5bf379c",
            pair=selected,
            algorithms=(
                (SwizzleAlgorithm.COMET_MESH_PERSONALIZED_A2A, 4, 1, 1, 1),
                (SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A, 4, 1, 1, 1),
            ),
            cycles=(564.0, 304.0),
            frontier=(selected,),
            stop=308.0,
            witness_id="moe_whole_pair_feasibility_cec0593327441c4d",
        )


if __name__ == "__main__":
    unittest.main()
