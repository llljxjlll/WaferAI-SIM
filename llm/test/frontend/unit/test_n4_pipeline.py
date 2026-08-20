from __future__ import annotations

import math
from pathlib import Path
import unittest

from llm.frontend.wafer_frontend.passes import (
    PassManager,
    PipelinePhase,
    build_ir0,
    load_physical_fabric,
    logical_expand,
    partition_bundle,
    place_bundle,
    plan_bundle,
)
from llm.frontend.wafer_frontend.schema.action import FusionActionKind
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.n4 import (
    FusionPartitionContext,
    InterDiePlanBundle,
    InterDiePlanningContext,
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
    valid_hbm_address_spaces,
    valid_spec,
)


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "llm/test/sram/hardware_numa.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"


def _compile_through_n4():
    spec = from_data(ExperimentSpec, valid_spec(), path="spec")
    manager = PassManager()
    template = manager.run_pass("build_ir0", spec, build_ir0)
    expanded = manager.run_pass("logical_expand", template, logical_expand)
    fabric = load_physical_fabric(_HARDWARE, _MAPPING)
    placement_context = PlacementContext.create(
        producer_pass="load_physical_fabric",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    placed = manager.run_pass(
        "placement",
        expanded,
        place_bundle,
        context=placement_context,
    )
    partition_context = FusionPartitionContext.create(producer_pass="unit")
    partitioned = manager.run_pass(
        "fusion_partition",
        placed,
        partition_bundle,
        context=partition_context,
    )
    planning_context = naive_inter_die_planning_context("unit")
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
    return (
        manager,
        placed,
        placement_context,
        partitioned,
        partition_context,
        planned,
        planning_context,
    )


class N4PipelineTest(unittest.TestCase):
    def test_real_pipeline_round_trip_provenance_and_numeric_gate(self) -> None:
        (
            manager,
            placed,
            placement_context,
            partitioned,
            partition_context,
            planned,
            planning_context,
        ) = _compile_through_n4()

        self.assertEqual(manager.snapshot.phase, PipelinePhase.INTERDIE_PLANNED)
        self.assertEqual(len(manager.snapshot.receipts), 5)
        partition_receipt, planning_receipt = manager.snapshot.receipts[-2:]
        self.assertEqual(partition_receipt.pass_name, "fusion_partition")
        self.assertEqual(planning_receipt.pass_name, "inter_die_plan")
        self.assertEqual(partition_receipt.context_digest, canonical_digest(partition_context))
        self.assertEqual(planning_receipt.context_digest, canonical_digest(planning_context))
        self.assertNotEqual(partition_receipt.context_digest, planning_receipt.context_digest)
        self.assertEqual(partition_receipt.input_digest, canonical_digest(placed))
        self.assertEqual(planning_receipt.input_digest, canonical_digest(partitioned))
        self.assertEqual(planning_receipt.output_digest, canonical_digest(planned))

        planned.validate_against(partitioned, planning_context)
        decoded = loads_dataclass(
            InterDiePlanBundle,
            canonical_json(planned),
            path="inter_die_plan_bundle",
        )
        self.assertEqual(decoded, planned)
        decoded.validate_against(partitioned, planning_context)

        entry = planned.entries[0]
        self.assertEqual(len(entry.fusion_plans), 2)
        self.assertEqual(len(entry.standalone_plans), 2)
        all_plans = (*entry.fusion_plans, *entry.standalone_plans)
        actions = tuple(
            action
            for plan in all_plans
            for program in plan.rank_programs
            for action in program.actions
        )
        self.assertEqual(len(actions), 40)
        self.assertEqual(
            sum(action.kind is FusionActionKind.SEND for action in actions),
            8,
        )
        self.assertEqual(
            sum(
                action.bytes
                for action in actions
                if action.kind is FusionActionKind.SEND
            ),
            65536,
        )
        chunk_gemm_flops = sum(
            2 * math.prod(action.compute.workload.rank_shape)
            for action in actions
            if action.kind is FusionActionKind.COMP and action.compute is not None
        )
        self.assertEqual(chunk_gemm_flops, 12582912)
        for plan in entry.fusion_plans:
            self.assertEqual(plan.logical_output_layout, plan.physical_output_layout)
            self.assertTrue(
                all(
                    item.logical_owner_rank == item.physical_owner_rank == item.chunk_id
                    for item in plan.output_permutation
                )
            )
        self.assertFalse(hasattr(planned, "per_die_dags"))
        self.assertEqual(placement_context.fabric, entry.graph.fabric)

    def test_repeat_is_deterministic_and_inputs_and_contexts_are_immutable(self) -> None:
        first = _compile_through_n4()
        first_digests = tuple(canonical_digest(item) for item in first[1:])
        second = _compile_through_n4()
        self.assertEqual(second[0].snapshot, first[0].snapshot)
        self.assertEqual(second[5], first[5])
        self.assertEqual(
            tuple(canonical_digest(item) for item in first[1:]),
            first_digests,
        )


if __name__ == "__main__":
    unittest.main()
