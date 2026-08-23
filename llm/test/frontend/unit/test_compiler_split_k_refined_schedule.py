from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.compiler import (
    _refined_projected_schedule_view,
    _schedule_refined_bundle,
)
from llm.frontend.wafer_frontend.passes.intra_die_refine import refine_bundle
from llm.frontend.wafer_frontend.passes import (
    build_global_bundle, lower_bundle, schedule_bundle,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.ir2 import SemanticTaskKind
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.policies.registry import (
    RegistryKind,
    production_registry,
)
from llm.frontend.wafer_frontend.schema.intra_die_refine import (
    IntraDieRefineContext,
    IntraDieRefineContract,
    SplitKRefineOptions,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.n4 import (
    InterDiePlanBundle,
    InterDiePlannedProfile,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    IntraDieSchedulingContext,
    ProjectToIR2Context,
    ProjectedIR2Bundle,
    ProjectedProfileIR2,
)

from _fixtures import naive_inter_die_planning_context
from test_n4_schema import _single_profile_partition
from test_naive_inter_die import _partitioned_graph



def _isolated_gemm_graph() -> IR1:
    """Build one state-free, ordinary GEMM that is eligible for split-K."""

    source = _partitioned_graph(tp=1)
    node = next(candidate for candidate in source.nodes if candidate.kind is OpKind.GEMM)
    local_value_ids = set(node.inputs + node.outputs)
    values = tuple(
        replace(
            value,
            producer=node.id if value.id in node.outputs else None,
            consumers=(node.id,) if value.id in node.inputs else (),
        )
        for value in source.values
        if value.id in local_value_ids
    )
    fields = source._semantic_key()
    fields.update(
        instances=(replace(source.instances[0], node_ids=(node.id,)),),
        nodes=(node,),
        values=values,
        edges=(),
        fusion_candidates=(),
        fused_op_skeletons=(),
        cross_routes=(),
        state_accesses=(),
        persistent_state_manifest=None,
        node_profiles=(),
    )
    graph = IR1.create(producer_pass="compiler_refined_schedule_fixture", **fields)
    graph.validate("compiler_refined_schedule_fixture")
    return graph


def _projected_gemm_bundle() -> ProjectedIR2Bundle:
    graph = _isolated_gemm_graph()
    _partition_context, partitioned = _single_profile_partition(graph)
    planning_context = naive_inter_die_planning_context(
        "compiler_refined_schedule_fixture"
    )
    planned_entry = InterDiePlannedProfile.create(
        source=partitioned.entries[0],
        context=planning_context,
        fusion_plans=(),
        standalone_plans=(),
    )
    planned = InterDiePlanBundle.create(
        source=partitioned,
        context=planning_context,
        entries=(planned_entry,),
    )
    planned.validate_against(partitioned, planning_context)

    projection_context = ProjectToIR2Context.create(
        producer_pass="compiler_refined_schedule_fixture",
        state_transfers=(),
    )
    projection = NaiveProjectToIR2().run(
        planned_entry.graph,
        (),
        (),
        state_transfers=(),
    )
    projected_entry = ProjectedProfileIR2.create(
        source=planned_entry,
        context=projection_context,
        projection=projection,
    )
    projected = ProjectedIR2Bundle.create(
        source=planned,
        context=projection_context,
        entries=(projected_entry,),
    )
    projected.validate_against(planned, projection_context)
    return projected


class CompilerSplitKRefinedScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selection = production_registry().instantiate(
            RegistryKind.INTRA_DIE,
            "naive",
        ).selection
        cls.projected = _projected_gemm_bundle()
        cls.enabled_refine_context = IntraDieRefineContext.create(
            producer_pass="compiler_refined_schedule_fixture",
            policy=cls.selection,
            contract=(
                IntraDieRefineContract.SPLIT_K_REDUCE_DOUBLE_BUFFER_V2
            ),
            options=SplitKRefineOptions(
                split_k_parts=2,
                enable_reduce=True,
                enable_double_buffer=True,
            ),
        )
        cls.refined = refine_bundle(cls.projected, cls.enabled_refine_context)
        cls.scheduling_context = IntraDieSchedulingContext.create(
            producer_pass="compiler_refined_schedule_fixture",
            policy=cls.selection,
        )

    def test_schedule_view_prefers_and_restables_refined_projection(self) -> None:
        refined_entry = self.refined.entries[0]
        carrier = refined_entry.split_k_refinement
        self.assertIsNotNone(carrier)
        assert carrier is not None

        first = _refined_projected_schedule_view(self.refined)
        second = _refined_projected_schedule_view(self.refined)

        self.assertEqual(first, second)
        self.assertIs(first.entries[0].projection, carrier.projection)
        self.assertNotEqual(
            first.entries[0].projection.id,
            refined_entry.projection.id,
        )
        self.assertNotEqual(first.entries[0].id, refined_entry.source.id)
        self.assertNotEqual(first.id, self.refined.source.id)
        first.validate("compiler_refined_schedule_view")
        self.refined.validate("compiler_refined_source_unchanged")

    def test_schedule_adapter_forwards_refined_projection_context_and_policy(
        self,
    ) -> None:
        carrier = self.refined.entries[0].split_k_refinement
        self.assertIsNotNone(carrier)
        assert carrier is not None
        policy = object()
        sentinel = object()

        with patch(
            "llm.frontend.wafer_frontend.compiler.schedule_bundle",
            return_value=sentinel,
        ) as schedule_bundle_mock:
            result = _schedule_refined_bundle(
                self.refined,
                self.scheduling_context,
                policy,
            )

        self.assertIs(result, sentinel)
        schedule_bundle_mock.assert_called_once()
        view, forwarded_context, forwarded_policy = (
            schedule_bundle_mock.call_args.args
        )
        self.assertIs(forwarded_context, self.scheduling_context)
        self.assertIs(forwarded_policy, policy)
        self.assertIs(view.entries[0].projection, carrier.projection)
        view.validate("compiler_refined_schedule_adapter.view")

        refined_task_ids = {
            task.id
            for dag in view.entries[0].projection.dags
            for task in dag.tasks
        }
        rewrite = carrier.rewrites[0]
        required_rewrite_tasks = set(rewrite.part_task_ids)
        required_rewrite_tasks.add(rewrite.reduce_task_id)
        required_rewrite_tasks.update(
            task_id
            for version in rewrite.double_buffer_versions
            for task_id in version.stage_task_ids
        )
        self.assertNotIn(None, required_rewrite_tasks)
        self.assertTrue(required_rewrite_tasks.issubset(refined_task_ids))

    def test_refined_schedule_global_and_lowering_are_cross_core(self) -> None:
        carrier = self.refined.entries[0].split_k_refinement
        assert carrier is not None
        view = _refined_projected_schedule_view(self.refined)
        scheduled = schedule_bundle(view, self.scheduling_context)
        schedule = scheduled.entries[0].schedule_set.schedules[0]
        placements = {item.task_id: item.core_id for item in schedule.placements}
        rewrite = carrier.rewrites[0]
        self.assertEqual(rewrite.compute_group_count, 2)
        self.assertTrue(rewrite.local_handoffs)
        for handoff in rewrite.local_handoffs:
            self.assertNotEqual(
                placements[handoff.send_task_id],
                placements[handoff.recv_task_id],
            )
            self.assertNotEqual(
                placements[handoff.part_task_id],
                placements[rewrite.reduce_task_id],
            )

        global_bundle = build_global_bundle(scheduled)
        kinds = {
            action.task_kind
            for action in global_bundle.entries[0].global_dag.actions
        }
        self.assertTrue(
            {
                SemanticTaskKind.LOCAL_SEND,
                SemanticTaskKind.LOCAL_RECV,
                SemanticTaskKind.LOCAL_WAIT,
            }.issubset(kinds)
        )
        lowered = lower_bundle(global_bundle)
        opcodes = {
            record.opcode
            for fragment in lowered.entries[0].fragments
            if hasattr(fragment, "core_streams")
            for stream in fragment.core_streams
            for record in stream.records
        }
        self.assertTrue(
            {
                RecordOpcode.LOCAL_NOC_SEND,
                RecordOpcode.LOCAL_NOC_RECV,
                RecordOpcode.LOCAL_NOC_WAIT,
            }.issubset(opcodes)
        )

    def test_disabled_v2_schedules_canonical_projection(self) -> None:
        disabled_context = IntraDieRefineContext.create(
            producer_pass="compiler_refined_schedule_fixture",
            policy=self.selection,
            contract=(
                IntraDieRefineContract.SPLIT_K_REDUCE_DOUBLE_BUFFER_V2
            ),
        )
        disabled = refine_bundle(self.projected, disabled_context)
        self.assertIsNone(disabled.entries[0].split_k_refinement)
        policy = object()
        sentinel = object()

        with patch(
            "llm.frontend.wafer_frontend.compiler.schedule_bundle",
            return_value=sentinel,
        ) as schedule_bundle_mock:
            result = _schedule_refined_bundle(
                disabled,
                self.scheduling_context,
                policy,
            )

        canonical = self.projected.entries[0].projection
        self.assertIs(result, sentinel)
        schedule_bundle_mock.assert_called_once()
        view, forwarded_context, forwarded_policy = (
            schedule_bundle_mock.call_args.args
        )
        self.assertIs(forwarded_context, self.scheduling_context)
        self.assertIs(forwarded_policy, policy)
        self.assertIs(view.entries[0].projection, canonical)
        view.validate("compiler_disabled_refined_schedule_adapter.view")


if __name__ == "__main__":
    unittest.main()
