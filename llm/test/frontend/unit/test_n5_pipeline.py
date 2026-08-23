from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.compiler import _schedule_refined_bundle
from llm.frontend.wafer_frontend.passes.intra_die_refine import refine_bundle
from llm.frontend.wafer_frontend.passes import (
    PassManager,
    PipelinePhase,
    build_global_bundle,
    build_ir0,
    load_physical_fabric,
    logical_expand,
    partition_bundle,
    place_bundle,
    plan_bundle,
    project_bundle,
    schedule_bundle,
)
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.intra_die_refine import IntraDieRefineContext
from llm.frontend.wafer_frontend.policies.registry import production_registry, RegistryKind
from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
from llm.frontend.wafer_frontend.schema.n4 import (
    FusionPartitionContext,
    InterDiePlanningContext,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    GlobalActionBundle,
    IntraDieSchedulingContext,
    ProjectToIR2Context,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)

from _fixtures import (
    naive_inter_die_planning_context,
    naive_intra_die_scheduling_context,
    valid_hbm_address_spaces,
    valid_spec,
)
from test_n4_pipeline import _compile_through_n4


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "llm/test/sram/hardware_numa.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_LARGE_SRAM_BYTES = 64 * 1024 * 1024


def _large_sram_fabric():
    fabric = load_physical_fabric(_HARDWARE, _MAPPING)
    profiles = tuple(
        replace(
            profile,
            capacity_bytes=_LARGE_SRAM_BYTES,
            regions=(
                replace(
                    next(
                        region
                        for region in profile.regions
                        if region.name == "comm"
                    ),
                    base_bytes=0,
                    size_bytes=_LARGE_SRAM_BYTES,
                ),
            ),
        )
        for profile in fabric.sram_profiles
    )
    result = replace(fabric, sram_profiles=profiles)
    result.validate("large_sram_fabric")
    return result


def _compile_through_n5():
    spec = from_data(ExperimentSpec, valid_spec(), path="spec")
    manager = PassManager()
    template = manager.run_pass("build_ir0", spec, build_ir0)
    expanded = manager.run_pass("logical_expand", template, logical_expand)
    fabric = _large_sram_fabric()
    placement_context = PlacementContext.create(
        producer_pass="n5_pipeline_fixture",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    placed = manager.run_pass(
        "placement", expanded, place_bundle, context=placement_context
    )
    partition_context = FusionPartitionContext.create(
        producer_pass="n5_pipeline_fixture"
    )
    partitioned = manager.run_pass(
        "fusion_partition",
        placed,
        partition_bundle,
        context=partition_context,
    )
    planning_context = naive_inter_die_planning_context("n5_pipeline_fixture")
    planned = manager.run_pass(
        "inter_die_plan",
        partitioned,
        plan_bundle,
        context=planning_context,
        policy_selections=(
            planning_context.fused_policy,
            planning_context.standalone_policy,
        ),
    )
    projection_context = ProjectToIR2Context.create(
        producer_pass="n5_pipeline_fixture",
        state_transfers=(),
    )
    projected = manager.run_pass(
        "project_to_ir2",
        planned,
        project_bundle,
        context=projection_context,
    )
    intra_selection = production_registry().instantiate(RegistryKind.INTRA_DIE, "naive").selection
    refine_context = IntraDieRefineContext.create(
        producer_pass="n5_pipeline_fixture", policy=intra_selection
    )
    refined = manager.run_pass(
        "intra_die_refine", projected, refine_bundle, context=refine_context
    )
    scheduling_context = naive_intra_die_scheduling_context("n5_pipeline_fixture")
    scheduled = manager.run_pass(
        "intra_die_schedule",
        refined,
        lambda source, context: _schedule_refined_bundle(source, context, NaiveIntraDiePolicy()),
        context=scheduling_context,
        policy_selections=(scheduling_context.policy,),
    )
    global_bundle = manager.run_pass(
        "global_action_dag", scheduled, build_global_bundle
    )
    return (
        manager,
        projected,
        projection_context,
        scheduled,
        scheduling_context,
        global_bundle,
    )


class N5PipelineTest(unittest.TestCase):
    def test_real_naive_tp2_pipeline_is_exact_reproducible_and_linkable(self) -> None:
        (
            manager,
            projected,
            projection_context,
            scheduled,
            scheduling_context,
            global_bundle,
        ) = _compile_through_n5()

        self.assertEqual(manager.snapshot.phase, PipelinePhase.GLOBAL_DAG_BUILT)
        self.assertEqual(len(manager.snapshot.receipts), 9)
        self.assertEqual(
            tuple(receipt.pass_name for receipt in manager.snapshot.receipts[-4:]),
            ("project_to_ir2", "intra_die_refine", "intra_die_schedule", "global_action_dag"),
        )
        self.assertEqual(
            manager.snapshot.receipts[-4].context_digest,
            canonical_digest(projection_context),
        )
        self.assertIsNotNone(
            manager.snapshot.receipts[-3].context_digest
        )
        self.assertEqual(
            manager.snapshot.receipts[-2].context_digest,
            canonical_digest(scheduling_context),
        )
        self.assertIsNone(manager.snapshot.receipts[-1].context_digest)

        projected_entry = projected.entries[0]
        scheduled_entry = scheduled.entries[0]
        global_entry = global_bundle.entries[0]
        self.assertEqual(
            tuple(len(dag.tasks) for dag in projected_entry.projection.dags),
            (43, 43),
        )
        self.assertEqual(
            sum(len(dag.flows) for dag in projected_entry.projection.dags),
            16,
        )
        self.assertEqual(
            tuple(
                len(schedule.buffer_bindings)
                for schedule in scheduled_entry.schedule_set.schedules
            ),
            (38, 38),
        )
        self.assertEqual(
            tuple(
                len(schedule.task_state_uses)
                for schedule in scheduled_entry.schedule_set.schedules
            ),
            (11, 11),
        )
        self.assertEqual(len(global_entry.global_dag.actions), 86)
        self.assertEqual(
            sum(
                len(action.state_uses)
                for action in global_entry.global_dag.actions
            ),
            22,
        )
        context = global_entry.lowering_context()
        context.validate()

        decoded = loads_dataclass(
            GlobalActionBundle,
            canonical_json(global_bundle),
            path="global_action_bundle",
        )
        self.assertEqual(decoded, global_bundle)
        decoded.validate_against(scheduled)

        second = _compile_through_n5()
        self.assertEqual(second[0].snapshot, manager.snapshot)
        self.assertEqual(second[-1], global_bundle)

    def test_real_hardware_comm_capacity_failure_rolls_back_schedule_pass(self) -> None:
        (
            manager,
            _placed,
            _placement_context,
            _partitioned,
            _partition_context,
            planned,
            _planning_context,
        ) = _compile_through_n4()
        projection_context = ProjectToIR2Context.create(
            producer_pass="n5_pipeline_fixture",
            state_transfers=(),
        )
        projected = manager.run_pass(
            "project_to_ir2",
            planned,
            project_bundle,
            context=projection_context,
        )
        intra_selection = production_registry().instantiate(RegistryKind.INTRA_DIE, "naive").selection
        refine_context = IntraDieRefineContext.create(
            producer_pass="n5_pipeline_fixture", policy=intra_selection
        )
        refined = manager.run_pass(
            "intra_die_refine", projected, refine_bundle, context=refine_context
        )
        before = manager.snapshot
        scheduling_context = naive_intra_die_scheduling_context("n5_pipeline_fixture")
        with self.assertRaisesRegex(SchemaError, "SRAM capacity.*comm"):
            manager.run_pass(
                "intra_die_schedule",
                refined,
                lambda source, context: _schedule_refined_bundle(source, context, NaiveIntraDiePolicy()),
                context=scheduling_context,
                policy_selections=(scheduling_context.policy,),
            )
        self.assertEqual(manager.snapshot, before)
        self.assertEqual(manager.snapshot.phase, PipelinePhase.INTRADIE_REFINED)
        self.assertEqual(len(manager.snapshot.receipts), 7)


if __name__ == "__main__":
    unittest.main()
