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
    IntraDieDAG,
    IR2ProjectionResult,
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

    def test_common_ir2_projector_supports_canonical_ag_and_gemm_rs(self) -> None:
        entries = _production_plans()

        ar_case, ar_plan = entries[2]
        with self.assertRaisesRegex(
            UnsupportedFeatureError, "supports GEMM_RS and AG_GEMM only"
        ):
            NaiveProjectToIR2().run(
                ar_case.partitioned_graph,
                (ar_plan,),
                (),
                state_transfers=(),
            )

        ag_case, ag_plan = entries[0]
        ag_projection = NaiveProjectToIR2().run(
            ag_case.partitioned_graph,
            (ag_plan,),
            (),
            state_transfers=(),
        )
        ag_projection.validate_against(
            ag_case.partitioned_graph, (ag_plan,), ()
        )
        for dag in ag_projection.dags:
            comp = tuple(
                task for task in dag.tasks
                if task.kind is SemanticTaskKind.COMP
                and isinstance(task.origin_ref, SwizzleNodeOrigin)
            )
            self.assertEqual(len(comp), 2)
            self.assertTrue(all(task.compute is not None for task in comp))
            self.assertTrue(
                all(task.member_id == ag_plan.decision.problem.gemm.node_ref for task in comp)
            )
            self.assertEqual(
                tuple(task.tensor_slice.shape for task in comp),
                ((4, 24), (4, 24)),
            )
            self.assertEqual(
                tuple(task.tensor_slice.offset for task in comp),
                ((0, dag.die_id * 24), (4, dag.die_id * 24)),
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
        self.assertEqual(tuple(len(dag.tasks) for dag in projection.dags), (20, 20))
        self.assertEqual(tuple(len(dag.flows) for dag in projection.dags), (2, 2))


    def test_ag_common_ir2_compute_provenance_tamper_fails_closed(self) -> None:
        case, plan = _production_plans()[0]
        projection = NaiveProjectToIR2().run(
            case.partitioned_graph, (plan,), (), state_transfers=()
        )
        dag = projection.dags[0]
        source = next(
            task for task in dag.tasks
            if task.kind is SemanticTaskKind.COMP
            and isinstance(task.origin_ref, SwizzleNodeOrigin)
        )
        assert source.compute is not None
        forged_task = replace(
            source,
            compute=replace(source.compute, impl_ref="forged.impl"),
        )
        dag_semantic = dag._semantic_key()
        dag_semantic["tasks"] = tuple(
            forged_task if task.id == source.id else task for task in dag.tasks
        )
        forged_dag = IntraDieDAG.create(
            producer_pass=dag.producer_pass, **dag_semantic
        )
        projection_semantic = projection._semantic_key()
        projection_semantic["dags"] = (forged_dag,) + projection.dags[1:]
        forged = IR2ProjectionResult.create(
            producer_pass=projection.producer_pass, **projection_semantic
        )
        with self.assertRaisesRegex(
            SchemaError, "compute/output slice disagrees with exact provenance"
        ):
            forged.validate_against(case.partitioned_graph, (plan,), ())


    def test_gemm_rs_common_ir2_swizzle_comp_uses_all_16_cores(self) -> None:
        from unittest.mock import patch
        import swizzle_cases
        from llm.frontend.wafer_frontend.policies.naive_intra_die import (
            NaiveIntraDiePolicy,
        )
        from llm.frontend.wafer_frontend.policies.split_k_intra_die_refine import (
            refine_split_k_projection,
        )
        from llm.frontend.wafer_frontend.schema.intra_die_refine import (
            SplitKRefineOptions,
        )

        base = swizzle_cases._dense_spec()
        spec = replace(
            base,
            model=replace(base.model, H=64, I=128, NH=16, KVH=16),
        )
        with patch.object(swizzle_cases, "_dense_spec", return_value=spec):
            case = swizzle_cases.build_dense_tp_swizzle_cases()[1]
        selection = force_swizzle_deployment(
            case.decision
        ).deployment_selection
        plan = materialize_swizzle_plan(
            case.partitioned_graph,
            case.decision,
            case.partitioned_graph.profile,
            deployment_selection=selection,
        )
        source = NaiveProjectToIR2().run(
            case.partitioned_graph, (plan,), (), state_transfers=()
        )
        refined = refine_split_k_projection(
            source,
            SplitKRefineOptions(
                split_k_parts=16,
                enable_reduce=True,
                compute_groups_per_die=16,
                enable_tree_reduce=True,
                enable_direct_dma=True,
            ),
            case.partitioned_graph,
        )

        schedules = NaiveIntraDiePolicy().schedule(
            refined.projection, case.partitioned_graph
        )
        placements = {
            placement.task_id: placement.core_id
            for schedule in schedules.schedules
            for placement in schedule.placements
        }
        source_dags = {dag.id: dag for dag in source.dags}
        refined_dags = {dag.die_id: dag for dag in refined.projection.dags}
        swizzle_rewrites = []
        for rewrite in refined.rewrites:
            source_dag = source_dags[rewrite.source_dag_id]
            source_task = next(
                task for task in source_dag.tasks
                if task.id == rewrite.source_task_id
            )
            if not isinstance(source_task.origin_ref, SwizzleNodeOrigin):
                continue
            swizzle_rewrites.append(rewrite)
            self.assertEqual(rewrite.compute_group_count, 16)
            self.assertEqual(len(rewrite.part_task_ids), 16)
            self.assertEqual(
                len({placements[task_id] for task_id in rewrite.part_task_ids}),
                16,
            )
            self.assertEqual(len(rewrite.local_handoffs), 15)
            # Swizzle operands are compute-produced, so direct-DMA is enabled
            # as a candidate capability but has no legal source DMA to clone.
            self.assertEqual(rewrite.direct_dma_task_ids, ())

            refined_dag = refined_dags[source_dag.die_id]
            output = next(
                value for value in refined_dag.swizzle_values
                if value.id == rewrite.source_output_value_id
            )
            final_reduce = rewrite.reduction_task_ids[-1]
            self.assertEqual(output.producer_tasks, (final_reduce,))
            original_output = next(
                value for value in source_dag.swizzle_values
                if value.id == rewrite.source_output_value_id
            )
            self.assertEqual(output.consumer_tasks, original_output.consumer_tasks)
            refined_tasks = {task.id: task for task in refined_dag.tasks}
            source_tasks = {task.id: task for task in source_dag.tasks}
            for consumer_id in output.consumer_tasks:
                self.assertEqual(
                    refined_tasks[consumer_id].deps,
                    tuple(
                        final_reduce if dep == rewrite.source_task_id else dep
                        for dep in source_tasks[consumer_id].deps
                    ),
                )
        self.assertEqual(len(swizzle_rewrites), 4)

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
