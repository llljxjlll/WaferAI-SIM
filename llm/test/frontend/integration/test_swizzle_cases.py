from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes.placement import place_ir0
from llm.frontend.wafer_frontend.schema.ir0 import FusionOrigin, FusionPattern
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
    SwizzleDecisionReason,
)

from swizzle_cases import (
    build_dense_tp_swizzle_cases,
    build_swizzle_integration_cases,
    build_synthetic_gemm_ar_swizzle_case,
)
from swizzle_forced import ForcedSwizzleReason


class SwizzleIntegrationCasesTest(unittest.TestCase):
    def test_dense_ag_and_rs_share_one_real_discovered_and_placed_graph(self) -> None:
        ag, rs = build_dense_tp_swizzle_cases()

        self.assertEqual((ag.pattern, rs.pattern), (FusionPattern.AG_GEMM, FusionPattern.GEMM_RS))
        self.assertEqual(ag.source_graph, rs.source_graph)
        self.assertEqual(ag.placed_graph, rs.placed_graph)
        self.assertEqual(ag.partitioned_graph, rs.partitioned_graph)
        self.assertEqual(ag.placed_graph.producer_pass, "placement")
        self.assertEqual(
            tuple(
                item.semantic_contract.pattern
                for item in ag.source_graph.fusion_candidates
            ),
            (
                FusionPattern.AG_GEMM,
                FusionPattern.GEMM_RS,
                FusionPattern.AG_GEMM,
                FusionPattern.GEMM_RS,
            ),
        )
        self.assertTrue(
            all(
                item.origin is FusionOrigin.DISCOVERED
                for item in ag.source_graph.fusion_candidates
            )
        )

    def test_naive_and_swizzle_partition_selection_remain_independent(self) -> None:
        ag, rs = build_dense_tp_swizzle_cases()
        candidates = {
            item.id: item.semantic_contract.pattern
            for item in ag.source_graph.fusion_candidates
        }

        self.assertEqual(
            {candidates[ref] for ref in ag.naive_fusion_refs},
            {FusionPattern.GEMM_RS},
        )
        self.assertEqual(set(ag.swizzle_fusion_refs), set(candidates))
        ag_skeleton = next(
            item
            for item in ag.partitioned_graph.fused_op_skeletons
            if item.id == ag.skeleton_ref
        )
        rs_skeleton = next(
            item
            for item in rs.partitioned_graph.fused_op_skeletons
            if item.id == rs.skeleton_ref
        )
        self.assertNotIn(ag_skeleton.fusion_ref, ag.naive_fusion_refs)
        self.assertIn(rs_skeleton.fusion_ref, rs.naive_fusion_refs)

    def test_synthetic_ar_is_discovered_then_exactly_placed_without_relaxing_dense(self) -> None:
        case = build_synthetic_gemm_ar_swizzle_case()

        self.assertTrue(case.synthetic_placement)
        self.assertEqual(case.placed_graph.producer_pass, "synthetic_swizzle_placement")
        self.assertEqual(len(case.source_graph.fusion_candidates), 1)
        candidate = case.source_graph.fusion_candidates[0]
        self.assertIs(candidate.origin, FusionOrigin.DISCOVERED)
        self.assertIs(candidate.semantic_contract.pattern, FusionPattern.GEMM_AR)
        self.assertEqual(case.naive_fusion_refs, ())
        self.assertEqual(case.swizzle_fusion_refs, (candidate.id,))
        with self.assertRaisesRegex(
            (SchemaError, UnsupportedFeatureError),
            "ReduceScatter|AllGather",
        ):
            place_ir0(case.source_graph, case.placement_context)

    def test_all_cases_close_stable_decision_route_action_and_adapter_provenance(self) -> None:
        first = build_swizzle_integration_cases()
        second = build_swizzle_integration_cases()
        self.assertEqual(first, second)
        self.assertEqual(
            tuple(item.pattern for item in first),
            (FusionPattern.AG_GEMM, FusionPattern.GEMM_RS, FusionPattern.GEMM_AR),
        )
        for case in first:
            with self.subTest(case=case.name):
                case.validate()
                self.assertIs(
                    case.decision.decision_reason,
                    SwizzleDecisionReason.NO_PROFITABLE_FUSION,
                )
                self.assertIs(
                    case.decision.ranked_candidates[0].algorithm,
                    SwizzleAlgorithm.UNFUSED,
                )
                self.assertEqual(case.adapter.selection.economic_decision, case.decision)
                self.assertIs(
                    case.adapter.selection.reason,
                    ForcedSwizzleReason.EXPLICIT_INTEGRATION_COVERAGE,
                )
                self.assertIsNot(
                    case.adapter.algorithm,
                    SwizzleAlgorithm.UNFUSED,
                )
                routes = {
                    route.id: route
                    for route in case.decision.problem.group.routes
                }
                candidate_actions = tuple(
                    action
                    for program in case.adapter.candidate.rank_programs
                    for action in program.actions
                )
                adapted_actions = tuple(
                    action
                    for program in case.adapter.rank_programs
                    for action in program.actions
                )
                self.assertEqual(
                    tuple(item.source_action for item in adapted_actions),
                    candidate_actions,
                )
                for action in adapted_actions:
                    source = action.source_action
                    if source.kind not in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
                        self.assertEqual(action.expected_route, ())
                        continue
                    route = routes[source.route_ref]
                    self.assertEqual(action.expected_route, route.die_path)
                    endpoints = (
                        (source.rank, source.peer_rank)
                        if source.kind is SwizzleActionKind.SEND
                        else (source.peer_rank, source.rank)
                    )
                    self.assertEqual(
                        (route.source_rank, route.destination_rank),
                        endpoints,
                    )


if __name__ == "__main__":
    unittest.main()
