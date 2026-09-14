from __future__ import annotations

import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import swizzle_cases

from llm.frontend.wafer_frontend.passes import physical_fabric_from_data
from llm.frontend.wafer_frontend.lowering.context import LoweringContext
from llm.frontend.wafer_frontend.lowering.isa_region import NaiveIsaRegionLowering
from llm.frontend.wafer_frontend.lowering.linker import NaiveManifestLinker
from llm.frontend.wafer_frontend.passes.global_action import build_global_action_dag
from llm.frontend.wafer_frontend.passes.lower_program import (
    _lower_fragments,
    _resolve_dependencies,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import NaiveProjectToIR2
from llm.frontend.wafer_frontend.policies.split_k_intra_die_refine import (
    refine_split_k_projection,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize import (
    force_swizzle_deployment,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize_ir1 import (
    materialize_swizzle_plan,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode,
    RegionManifest,
)
from llm.frontend.wafer_frontend.schema.intra_die_refine import SplitKRefineOptions
from llm.frontend.wafer_frontend.schema.ir2 import (
    SemanticTaskKind,
    SwizzleNodeOrigin,
)


def _context(*, parts: int, hidden: int, intermediate: int, case_index: int = 1):
    base = swizzle_cases._dense_spec()
    spec = replace(
        base,
        model=replace(
            base.model,
            H=hidden,
            I=intermediate,
            NH=hidden // 4,
            KVH=hidden // 4,
        ),
    )
    hardware_raw, _ = swizzle_cases._hardware()
    hardware_raw = copy.deepcopy(hardware_raw)
    hardware_raw["memory"]["sram_size"] = 3 << 20
    hardware_raw["memory"]["sram"]["capacity_bytes"] = 3 << 20
    hardware_raw["memory"]["sram"]["regions"][0]["size_bytes"] = 3 << 20
    hardware = (
        hardware_raw,
        physical_fabric_from_data(hardware_raw, path="swizzle.hardware.3mib"),
    )
    with (
        patch.object(swizzle_cases, "_dense_spec", return_value=spec),
        patch.object(swizzle_cases, "_hardware", return_value=hardware),
    ):
        case = swizzle_cases.build_dense_tp_swizzle_cases()[case_index]
    selection = force_swizzle_deployment(case.decision).deployment_selection
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
            split_k_parts=parts,
            enable_reduce=True,
            compute_groups_per_die=parts,
        ),
        case.partitioned_graph,
    )
    schedules = NaiveIntraDiePolicy().schedule(
        refined.projection, case.partitioned_graph
    )
    global_dag = build_global_action_dag(
        case.partitioned_graph, refined.projection, schedules
    )
    return LoweringContext(
        case.partitioned_graph,
        (plan,),
        (),
        refined.projection,
        schedules,
        global_dag,
    ), plan


class SwizzleSplitKIsaLoweringTest(unittest.TestCase):
    def test_fused_ag_gemm_16_core_linear_reduce_lowers(self) -> None:
        context, plan = _context(
            parts=16, hidden=64, intermediate=128, case_index=0
        )
        plan_actions = tuple(
            action
            for action in context.global_dag.actions
            if isinstance(action.origin_ref, SwizzleNodeOrigin)
            and action.origin_ref.plan_id == plan.id
            and action.task_kind is not SemanticTaskKind.TRANSIT
        )
        regions = NaiveIsaRegionLowering().lower(plan, plan_actions, context)
        active_cores = {
            (stream.logical_core.die_id, stream.logical_core.local_core_id)
            for region in regions
            for stream in region.fragment.core_streams
        }
        opcodes = {
            record.opcode
            for region in regions
            for stream in region.fragment.core_streams
            for record in stream.records
        }
        self.assertEqual(len(regions), 2)
        self.assertEqual(len(active_cores), 32)
        self.assertTrue(
            {
                RecordOpcode.LOCAL_NOC_SEND,
                RecordOpcode.LOCAL_NOC_RECV,
                RecordOpcode.LOCAL_NOC_WAIT,
            }.issubset(opcodes)
        )

    def test_fused_gemm_rs_16_core_local_noc_lowers(self) -> None:
        context, plan = _context(parts=16, hidden=32, intermediate=64)
        plan_actions = tuple(
            action
            for action in context.global_dag.actions
            if isinstance(action.origin_ref, SwizzleNodeOrigin)
            and action.origin_ref.plan_id == plan.id
            and action.task_kind is not SemanticTaskKind.TRANSIT
        )
        regions = NaiveIsaRegionLowering().lower(
            plan, plan_actions, context
        )
        active_cores = {
            (stream.logical_core.die_id, stream.logical_core.local_core_id)
            for region in regions
            for stream in region.fragment.core_streams
        }
        opcodes = {
            record.opcode
            for region in regions
            for stream in region.fragment.core_streams
            for record in stream.records
        }
        self.assertEqual(len(regions), 2)
        self.assertEqual(len(active_cores), 32)
        self.assertTrue(
            {
                RecordOpcode.LOCAL_NOC_SEND,
                RecordOpcode.LOCAL_NOC_RECV,
                RecordOpcode.LOCAL_NOC_WAIT,
            }.issubset(opcodes)
        )

    def test_fused_local_noc_complete_program_links(self) -> None:
        context, _plan = _context(parts=2, hidden=16, intermediate=32)
        fragments = _lower_fragments(
            context, _resolve_dependencies(None, None, None, None, None)
        )
        linked = NaiveManifestLinker().link(context, fragments)
        opcodes = {
            record.opcode
            for linked_fragment in linked.fragments
            for fragment in (
                linked_fragment.fragment
                if isinstance(linked_fragment, RegionManifest)
                else linked_fragment,
            )
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


if __name__ == "__main__":
    unittest.main()
