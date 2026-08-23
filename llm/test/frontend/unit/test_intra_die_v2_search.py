from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.policies.intra_die_v2_search import (
    evaluate_intra_die_v2_candidates,
)
from llm.frontend.wafer_frontend.policies.intra_die_timing_model import (
    DEFAULT_INTRA_DIE_HARDWARE_DIGEST,
    DEFAULT_INTRA_DIE_SIMULATION_DIGEST,
)
from llm.frontend.wafer_frontend.policies.split_k_intra_die_refine import refine_split_k_projection
from llm.frontend.wafer_frontend.schema.intra_die_timing_model import IntraDieTimingModel
from llm.frontend.wafer_frontend.schema.intra_die_refine import (
    IntraDieOptimizationMode,
    IntraDieOptimizationOptions,
    SplitKRefineOptions,
)
from llm.frontend.wafer_frontend.schema.intra_die_v2_search import (
    IntraDieV2CandidateKind,
    IntraDieV2CandidateRejection,
    IntraDieV2SearchBudget,
    IntraDieV2SearchDecision,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    from_data,
    to_primitive,
)

from test_split_k_intra_die_refine import _source_projection


def _timing_model(**overrides: int) -> IntraDieTimingModel:
    return IntraDieTimingModel.create(
        hardware_digest=DEFAULT_INTRA_DIE_HARDWARE_DIGEST,
        simulation_digest=DEFAULT_INTRA_DIE_SIMULATION_DIGEST,
        **overrides,
    )


class IntraDieV2SearchTest(unittest.TestCase):
    def test_identity_and_explicit_split_fallback_are_bounded_deterministic(self) -> None:
        graph, projection = _source_projection()
        options = SplitKRefineOptions(
            split_k_parts=2,
            enable_reduce=True,
            enable_double_buffer=True,
        )

        first = evaluate_intra_die_v2_candidates(projection, graph, options)
        second = evaluate_intra_die_v2_candidates(projection, graph, options)

        self.assertEqual(first, second)
        self.assertEqual(canonical_digest(first), canonical_digest(second))
        self.assertEqual(
            from_data(
                IntraDieV2SearchDecision,
                to_primitive(first),
                path="decision",
            ),
            first,
        )
        self.assertEqual(first.generated_candidate_count, 2)
        self.assertEqual(first.full_analytic_evaluation_count, 2)
        self.assertEqual(first.simulator_calls_during_search, 0)
        self.assertEqual(
            first.reserved_simulator_calls_for_final_evidence, 2
        )
        self.assertLessEqual(
            first.simulator_calls_during_search
            + first.reserved_simulator_calls_for_final_evidence,
            first.budget.simulator_call_budget,
        )
        self.assertLessEqual(
            first.generated_candidate_count, first.budget.max_candidates
        )
        self.assertLessEqual(first.budget.simulator_call_budget, 10)
        self.assertEqual(
            {candidate.kind for candidate in first.candidates},
            {
                IntraDieV2CandidateKind.IDENTITY,
                IntraDieV2CandidateKind.SPLIT_K_FALLBACK,
            },
        )
        selected = next(
            item
            for item in first.candidates
            if item.id == first.selected_candidate_ref
        )
        self.assertIs(
            selected.kind, IntraDieV2CandidateKind.SPLIT_K_FALLBACK
        )
        self.assertEqual(first.selection_reason, "explicit_split_k_request")

    def test_budget_and_counter_tampering_fail_closed(self) -> None:
        with self.assertRaisesRegex(Exception, "max_candidates"):
            IntraDieV2SearchBudget(max_candidates=17).validate()
        with self.assertRaisesRegex(Exception, "simulator_call_budget"):
            IntraDieV2SearchBudget(simulator_call_budget=11).validate()

        graph, projection = _source_projection()
        decision = evaluate_intra_die_v2_candidates(
            projection,
            graph,
            SplitKRefineOptions(split_k_parts=2),
        )
        with self.assertRaisesRegex(Exception, "search counters"):
            replace(decision, simulator_calls_during_search=1).validate()
        with self.assertRaisesRegex(Exception, "explicit split-K"):
            replace(
                decision,
                selected_candidate_ref=next(
                    item.id
                    for item in decision.candidates
                    if item.kind is IntraDieV2CandidateKind.IDENTITY
                ),
            ).validate()


    def test_off_selects_identity_only_and_reserves_three_runs(self) -> None:
        graph, projection = _source_projection()
        decision = evaluate_intra_die_v2_candidates(
            projection, graph,
            IntraDieOptimizationOptions(mode=IntraDieOptimizationMode.OFF),
        )
        self.assertIs(decision.mode, IntraDieOptimizationMode.OFF)
        self.assertEqual(len(decision.candidates), 1)
        self.assertIs(decision.candidates[0].kind, IntraDieV2CandidateKind.IDENTITY)
        self.assertEqual(decision.selected_candidate_ref, decision.candidates[0].id)
        self.assertEqual(decision.selection_reason, "off_identity_only")
        self.assertEqual(decision.reserved_simulator_calls_for_final_evidence, 3)

    def test_off_keeps_exact_projection_with_decision(self) -> None:
        graph, projection = _source_projection()
        carrier = refine_split_k_projection(
            projection,
            IntraDieOptimizationOptions(mode=IntraDieOptimizationMode.OFF),
            graph,
        )
        self.assertIs(carrier.projection, projection)
        self.assertEqual(carrier.rewrites, ())
        selected = next(
            item for item in carrier.search_decision.candidates
            if item.id == carrier.search_decision.selected_candidate_ref
        )
        self.assertIs(selected.kind, IntraDieV2CandidateKind.IDENTITY)
        carrier.validate_against(
            projection, graph, split_k_parts=1,
            enable_reduce=False, enable_double_buffer=False,
        )

    def test_auto_eliminates_non_profitable_split_and_records_break_even(self) -> None:
        graph, projection = _source_projection()
        options = IntraDieOptimizationOptions(
            mode=IntraDieOptimizationMode.AUTO,
            allowed_candidates=("identity", "split_k"),
            split_k_parts=(2,),
        )
        decision = evaluate_intra_die_v2_candidates(
            projection, graph, options,
            timing_model=_timing_model(effective_gemm_ops_per_cycle=1_000_000),
        )
        self.assertIs(decision.mode, IntraDieOptimizationMode.AUTO)
        self.assertEqual(decision.selection_reason, "auto_identity_no_profitable_candidate")
        self.assertEqual(len(decision.candidates), 1)
        self.assertEqual(len(decision.rejected_candidates), 1)
        rejection = decision.rejected_candidates[0]
        self.assertEqual(rejection.reason, "break_even_not_met")
        self.assertLessEqual(rejection.compute_savings_cycles, rejection.overhead_cycles)
        self.assertGreaterEqual(
            rejection.candidate_predicted_makespan_cycles,
            rejection.identity_predicted_makespan_cycles,
        )
        self.assertEqual(decision.simulator_calls_during_search, 0)
        self.assertEqual(decision.reserved_simulator_calls_for_final_evidence, 3)

    def test_break_even_rejection_uses_de_morgan_of_both_profit_conditions(self) -> None:
        common = {
            "candidate_name": "split_k", "split_k_parts": 2,
            "enable_double_buffer": False, "reason": "break_even_not_met",
            "identity_predicted_makespan_cycles": 100,
        }
        IntraDieV2CandidateRejection.create(
            **common, candidate_predicted_makespan_cycles=100,
            compute_savings_cycles=11, overhead_cycles=10,
        ).validate()
        IntraDieV2CandidateRejection.create(
            **common, candidate_predicted_makespan_cycles=99,
            compute_savings_cycles=10, overhead_cycles=10,
        ).validate()
        with self.assertRaisesRegex(Exception, "break-even rejection evidence"):
            IntraDieV2CandidateRejection.create(
                **common, candidate_predicted_makespan_cycles=99,
                compute_savings_cycles=11, overhead_cycles=10,
            ).validate()

    def test_auto_selects_strictly_profitable_split_and_identity_wins_tie(self) -> None:
        graph, projection = _source_projection()
        options = IntraDieOptimizationOptions(
            allowed_candidates=("identity", "split_k"), split_k_parts=(2,),
        )
        profitable = evaluate_intra_die_v2_candidates(
            projection, graph, options,
            timing_model=_timing_model(
                effective_gemm_ops_per_cycle=1_000,
                local_transport_setup_cycles=12,
            ),
        )
        selected = next(
            candidate for candidate in profitable.candidates
            if candidate.id == profitable.selected_candidate_ref
        )
        self.assertIs(selected.kind, IntraDieV2CandidateKind.SPLIT_K_FALLBACK)
        self.assertEqual(profitable.selection_reason, "auto_minimum_predicted_makespan")
        self.assertLess(
            selected.analytic_cost.predicted_makespan_cycles,
            next(
                candidate.analytic_cost.predicted_makespan_cycles
                for candidate in profitable.candidates
                if candidate.kind is IntraDieV2CandidateKind.IDENTITY
            ),
        )

        tied = evaluate_intra_die_v2_candidates(
            projection, graph, options,
            timing_model=_timing_model(
                effective_gemm_ops_per_cycle=10_000,
                local_transport_setup_cycles=0,
                local_transport_bytes_per_cycle=1_000_000_000,
                local_reduce_setup_cycles=7,
                local_reduce_bytes_per_cycle=163,
            ),
        )
        self.assertIs(tied.candidates[0].kind, IntraDieV2CandidateKind.IDENTITY)
        self.assertEqual(tied.selection_reason, "auto_identity_no_profitable_candidate")
        self.assertEqual(
            tied.rejected_candidates[0].candidate_predicted_makespan_cycles,
            tied.rejected_candidates[0].identity_predicted_makespan_cycles,
        )

    def test_versioned_options_and_candidate_cap_fail_closed(self) -> None:
        options = IntraDieOptimizationOptions(
            allowed_candidates=("identity", "split_k", "split_k_double_buffer"),
            split_k_parts=(2, 4),
        )
        options.validate()
        self.assertEqual(
            from_data(
                IntraDieOptimizationOptions,
                to_primitive(options),
                path="options",
            ),
            options,
        )
        with self.assertRaisesRegex(Exception, "max_candidates"):
            IntraDieOptimizationOptions(max_candidates=9).validate()
        with self.assertRaisesRegex(Exception, "candidate product"):
            IntraDieOptimizationOptions(
                allowed_candidates=("identity", "split_k", "split_k_double_buffer"),
                split_k_parts=(2, 4, 8, 16),
            ).validate()
        with self.assertRaisesRegex(Exception, "identity"):
            IntraDieOptimizationOptions(
                allowed_candidates=("split_k", "identity"),
            ).validate()

    def test_unified_force_names_exactly_one_candidate(self) -> None:
        graph, projection = _source_projection()
        decision = evaluate_intra_die_v2_candidates(
            projection, graph,
            IntraDieOptimizationOptions(
                mode=IntraDieOptimizationMode.FORCE,
                allowed_candidates=("identity", "split_k"),
                split_k_parts=(2,),
                force_candidate="split_k",
            ),
        )
        self.assertIs(decision.mode, IntraDieOptimizationMode.FORCE)
        self.assertEqual(len(decision.candidates), 2)
        self.assertEqual(decision.selection_reason, "explicit_split_k_request")
        self.assertEqual(decision.reserved_simulator_calls_for_final_evidence, 3)


if __name__ == "__main__":
    unittest.main()
