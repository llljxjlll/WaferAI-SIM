from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes.project_swizzle_plan import (
    SwizzlePlanProjection,
    project_swizzle_plan,
)
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize import (
    force_swizzle_deployment,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize_ir1 import (
    materialize_swizzle_plan,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.ir2 import (
    OrdinaryNodeOrigin,
    SemanticTaskKind,
    SwizzleNodeOrigin,
)

from swizzle_cases import build_swizzle_integration_cases


def _production_plans():
    result = []
    for case in build_swizzle_integration_cases():
        selection = force_swizzle_deployment(
            case.decision,
        ).deployment_selection
        plan = materialize_swizzle_plan(
            case.partitioned_graph,
            case.decision,
            case.partitioned_graph.profile,
            deployment_selection=selection,
        )
        result.append((case, plan))
    return tuple(result)


class SwizzlePlanProjectionTest(unittest.TestCase):
    def test_three_production_plans_project_with_exact_plan_provenance(self) -> None:
        entries = _production_plans()
        self.assertEqual(
            tuple(case.pattern for case, _plan in entries),
            (
                FusionPattern.AG_GEMM,
                FusionPattern.GEMM_RS,
                FusionPattern.GEMM_AR,
            ),
        )
        for case, plan in entries:
            with self.subTest(pattern=case.pattern.value):
                result = project_swizzle_plan(case.partitioned_graph, plan)
                self.assertIs(type(result), SwizzlePlanProjection)
                result.validate_against(case.partitioned_graph, plan)
                self.assertEqual(result.source_plan_ref, plan.id)
                self.assertEqual(result.adapter.deployment_selection, plan.deployment_selection)
                self.assertEqual(result.adapter.buffer_requirements, plan.buffer_requirements)
                self.assertEqual(result.projection.source_decision_ref, plan.decision.id)
                self.assertEqual(result.projection.source_candidate_ref, plan.candidate.id)
                self.assertFalse(
                    result.projection.downstream_gate.current_ir2_compatible
                )
                adapted = tuple(
                    action
                    for program in result.adapter.rank_programs
                    for action in program.actions
                )
                bound = tuple(
                    action
                    for program in plan.rank_programs
                    for action in program.actions
                )
                self.assertEqual(
                    tuple(
                        (
                            action.source_action,
                            action.fusion_kind,
                            action.member_ref,
                            action.expected_route,
                        )
                        for action in adapted
                    ),
                    tuple(
                        (
                            action.source_action,
                            action.fusion_kind,
                            action.member_ref,
                            action.expected_route,
                        )
                        for action in bound
                    ),
                )

    def test_projection_is_stable_and_plan_tamper_is_rejected(self) -> None:
        first = tuple(
            project_swizzle_plan(case.partitioned_graph, plan)
            for case, plan in _production_plans()
        )
        second = tuple(
            project_swizzle_plan(case.partitioned_graph, plan)
            for case, plan in _production_plans()
        )
        self.assertEqual(first, second)
        case, plan = _production_plans()[0]
        result = first[0]
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(result, schema_version="wafer_frontend.swizzle_plan_projection/v0").validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(result, source_plan_ref="swizzle_fusion_plan_tampered").validate()
        other_plan = _production_plans()[1][1]
        with self.assertRaisesRegex(SchemaError, "different production plan"):
            result.validate_against(case.partitioned_graph, other_plan)

    def test_common_ir2_projector_supports_only_gemm_rs(self) -> None:
        entries = _production_plans()
        for case, plan in (entries[0], entries[2]):
            with self.subTest(pattern=case.pattern.value):
                with self.assertRaisesRegex(UnsupportedFeatureError, "supports GEMM_RS only"):
                    NaiveProjectToIR2().run(
                        case.partitioned_graph,
                        (plan,),
                        (),
                        state_transfers=(),
                    )

        case, plan = entries[1]
        projection = NaiveProjectToIR2().run(
            case.partitioned_graph,
            (plan,),
            (),
            state_transfers=(),
        )
        projection.validate_against(case.partitioned_graph, (plan,), ())
        self.assertEqual(projection.fusion_plan_ids, (plan.id,))
        self.assertTrue(
            all(
                any(isinstance(task.origin_ref, SwizzleNodeOrigin) for task in dag.tasks)
                and any(isinstance(task.origin_ref, OrdinaryNodeOrigin) for task in dag.tasks)
                and len(dag.ordinary_node_ids) == 13
                for dag in projection.dags
            )
        )
        self.assertEqual(
            tuple(len(dag.tasks) for dag in projection.dags),
            (20, 20),
        )
        self.assertEqual(
            tuple(len(dag.flows) for dag in projection.dags),
            (2, 2),
        )


    def test_gemm_rs_common_ir2_reaches_production_isa_regions(self) -> None:
        from llm.frontend.wafer_frontend.lowering.context import LoweringContext
        from llm.frontend.wafer_frontend.lowering.isa_region import NaiveIsaRegionLowering
        from llm.frontend.wafer_frontend.passes.global_action import build_global_action_dag
        from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
        from llm.frontend.wafer_frontend.policies.optimized_intra_die import OptimizedIntraDiePolicy
        case, plan = _production_plans()[1]
        projection = NaiveProjectToIR2().run(
            case.partitioned_graph, (plan,), (), state_transfers=()
        )
        for policy in (NaiveIntraDiePolicy(), OptimizedIntraDiePolicy()):
            schedules = policy.schedule(projection, case.partitioned_graph)
            actions = build_global_action_dag(
                case.partitioned_graph, projection, schedules
            )
            context = LoweringContext(
                case.partitioned_graph, (plan,), (), projection, schedules, actions
            )
            plan_actions = tuple(
                action
                for action in actions.actions
                if isinstance(action.origin_ref, SwizzleNodeOrigin)
                and action.origin_ref.plan_id == plan.id
                and action.task_kind is not SemanticTaskKind.TRANSIT
            )
            regions = NaiveIsaRegionLowering().lower(
                plan, plan_actions, context
            )
            self.assertEqual(tuple(region.fusion_plan_id for region in regions), (plan.id, plan.id))
            self.assertTrue(all(region.fragment.core_streams for region in regions))

if __name__ == "__main__":
    unittest.main()
