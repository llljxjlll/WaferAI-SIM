from __future__ import annotations

from dataclasses import replace
import unittest
from llm.frontend.wafer_frontend.lowering.moe_swizzle_workload_abi import (
    build_moe_swizzle_workload_abi,
    validate_moe_swizzle_workload_abi_against,
)

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_moe_scale_swizzle_overlay import (
    build_moe_scale_swizzle_overlay,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_execution import (
    build_moe_swizzle_execution,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_workload_value_bridge import (
    build_moe_swizzle_workload_value_bridge,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_workload_state_abi import (
    build_moe_swizzle_workload_state_abi,
)
from llm.frontend.wafer_frontend.passes.project_moe_swizzle_whole_workload import (
    project_moe_swizzle_whole_workload,
)
from llm.frontend.wafer_frontend.passes.project_moe_scale_swizzle_ir2 import (
    project_moe_scale_swizzle_ir2,
)
from llm.frontend.wafer_frontend.passes.schedule_moe_swizzle_workload_endpoints import (
    schedule_moe_swizzle_workload_endpoints,
)
from llm.frontend.wafer_frontend.passes.schedule_moe_swizzle_workload_storage_reuse import (
    schedule_moe_swizzle_workload_storage_reuse,
)
from llm.frontend.wafer_frontend.passes.discover_moe_swizzle import (
    discover_moe_swizzle_regions,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_comet_mesh import (
    build_comet_mesh_moe_candidates,
    build_comet_mesh_moe_candidate_grid,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_direct_xy import (
    build_direct_xy_moe_candidate,
    build_direct_xy_moe_candidates,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_cost import decide_moe_swizzle
from llm.frontend.wafer_frontend.policies.swizzle.moe_problem import (
    build_moe_swizzle_problem,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_unfused import (
    build_executable_moe_unfused_baseline,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleDecisionReason
from llm.frontend.wafer_frontend.schema.swizzle_moe import MoeSwizzleDecision
from llm.frontend.wafer_frontend.schema.swizzle_moe_execution import (
    MoeScaleExecutionActionKind,
    MoeScaleExecutionMode,
    MoeScaleExecutionTerminalKind,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_plan import MoeSwizzleOverlay
from llm.frontend.wafer_frontend.schema.swizzle_moe_operand_abi import (
    build_moe_swizzle_operand_abi,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_workload import MoeSwizzleWorkloadProjection
from llm.frontend.wafer_frontend.schema.swizzle_moe_placement import (
    build_moe_candidate_dynamic_root_keys,
    build_moe_projection_dynamic_root_keys,
    build_moe_swizzle_task_placement,
    build_moe_swizzle_workload_endpoint_widths,
    build_moe_swizzle_workload_placement,
)
from llm.test.frontend.integration.moe_swizzle_scale_cases import (
    build_moe_swizzle_scale_cases,
)


def _build(case: object, mode: MoeScaleExecutionMode) -> tuple[object, tuple[object, ...], MoeSwizzleOverlay]:
    execution = build_moe_swizzle_execution(case.spec, case.oracle, mode)
    problems = tuple(
        build_moe_swizzle_problem(
            region,
            case.spec,
            case.oracle,
            execution,
            hardware_facts=case.hardware_facts,
            endpoint_session_contract=case.endpoint_session_contract,
        )
        for region in discover_moe_swizzle_regions(case.spec, case.oracle, execution)
    )
    baselines = tuple(
        build_executable_moe_unfused_baseline(
            problem, case.spec, case.oracle, execution,
        )
        for problem in problems
    )
    decisions = tuple(
        decide_moe_swizzle(problem, baseline, ())
        for problem, baseline in zip(problems, baselines, strict=True)
    )
    return execution, decisions, build_moe_scale_swizzle_overlay(execution, decisions)


class MoeScaleSwizzleOverlayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = tuple(
            (case, mode, *_build(case, mode))
            for case in build_moe_swizzle_scale_cases()[1:4]
            for mode in (
                MoeScaleExecutionMode.INFER_FORWARD,
                MoeScaleExecutionMode.TRAIN_FORWARD,
            )
        )

    def test_c1_c3_infer_and_train_formulas_and_partition(self) -> None:
        for case, mode, execution, _, overlay in self.rows:
            token_count = case.spec.tokens
            remote_count = len(case.oracle.remote_token_indices)
            train = mode is MoeScaleExecutionMode.TRAIN_FORWARD
            self.assertEqual(len(overlay.replaced_action_refs), 4 * token_count + 6 * remote_count)
            self.assertEqual(len(overlay.preserved_action_refs), 3 * token_count + (token_count if train else 0))
            self.assertEqual(len(overlay.linked_actions), 7 * token_count + 6 * remote_count + (token_count if train else 0))
            source = {item.id for item in execution.actions}
            replaced = set(overlay.replaced_action_refs)
            preserved = set(overlay.preserved_action_refs)
            self.assertFalse(replaced & preserved)
            self.assertEqual(replaced | preserved, source)
            provenance = tuple(
                ref for action in overlay.linked_actions for ref in action.source_action_refs
            )
            self.assertEqual(len(provenance), len(set(provenance)))
            self.assertEqual(set(provenance), source)
            linked_ids = {item.id for item in overlay.linked_actions}
            self.assertTrue(all(set(item.deps) <= linked_ids for item in overlay.linked_actions))
            self.assertEqual(
                set(overlay.terminal_value_refs),
                {item.value_ref for item in execution.terminals},
            )

    def test_train_tape_actions_and_terminals_are_preserved_exactly(self) -> None:
        for case, mode, execution, _, overlay in self.rows:
            if mode is not MoeScaleExecutionMode.TRAIN_FORWARD:
                continue
            tape_sources = {
                item.id for item in execution.actions
                if item.kind is MoeScaleExecutionActionKind.TAPE_COPY
            }
            tape_linked = tuple(
                item for item in overlay.linked_actions
                if item.source_action_refs and item.source_action_refs[0] in tape_sources
            )
            self.assertEqual(len(tape_linked), case.spec.tokens)
            self.assertTrue(all(item.preserved and item.kind == "preserved.tape_copy" for item in tape_linked))
            self.assertEqual({item.source_action_refs[0] for item in tape_linked}, tape_sources)
            tape_values = {
                ref for item in execution.actions
                if item.id in tape_sources for ref in item.write_values
            }
            self.assertTrue(tape_values <= set(overlay.terminal_value_refs))

    def test_generalized_overlay_serde_determinism_and_tamper(self) -> None:
        _, _, execution, decisions, overlay = self.rows[0]
        self.assertEqual(
            loads_dataclass(MoeSwizzleOverlay, canonical_json(overlay), path="overlay"),
            overlay,
        )
        self.assertEqual(build_moe_scale_swizzle_overlay(execution, decisions), overlay)
        with self.assertRaises(SchemaError):
            replace(overlay, linked_actions=overlay.linked_actions[:-1]).validate()

    def test_fused_direct_and_comet_u2_empty_subactions_are_not_orphans(self) -> None:
        case = build_moe_swizzle_scale_cases()[1]
        execution = build_moe_swizzle_execution(
            case.spec, case.oracle, MoeScaleExecutionMode.INFER_FORWARD,
        )
        problems = tuple(
            build_moe_swizzle_problem(
                region, case.spec, case.oracle, execution,
                hardware_facts=case.hardware_facts,
                endpoint_session_contract=case.endpoint_session_contract,
            )
            for region in discover_moe_swizzle_regions(case.spec, case.oracle, execution)
        )
        baselines = tuple(
            build_executable_moe_unfused_baseline(
                problem, case.spec, case.oracle, execution,
            )
            for problem in problems
        )
        direct = tuple(
            build_direct_xy_moe_candidate(problem, case.spec, case.oracle, execution)
            for problem in problems
        )
        comet_u2 = tuple(
            build_comet_mesh_moe_candidates(problem, case.spec, case.oracle, execution)[1]
            for problem in problems
        )
        for candidates in (direct, comet_u2):
            decisions = tuple(
                MoeSwizzleDecision.create(
                    problem=problem, baseline=baseline,
                    ranked_candidates=(candidate, baseline),
                    selected_candidate_ref=candidate.id,
                    decision_reason=SwizzleDecisionReason.NO_PROFITABLE_FUSION,
                    performance_complete=False,
                )
                for problem, baseline, candidate in zip(
                    problems, baselines, candidates, strict=True,
                )
            )
            overlay = build_moe_scale_swizzle_overlay(execution, decisions)
            empty = tuple(
                item for item in overlay.linked_actions
                if not item.preserved and not item.source_action_refs
            )
            self.assertTrue(empty)
            covered = tuple(
                ref for item in overlay.linked_actions for ref in item.source_action_refs
            )
            self.assertEqual(len(covered), len(set(covered)))
            self.assertEqual(set(covered), {item.id for item in execution.actions})
            self.assertTrue(all(item.assignment_refs for item in empty))

    def test_grouped_m_block_projects_exact_terminal_slices(self) -> None:
        cases = build_moe_swizzle_scale_cases()
        rows = (
            (1, "direct", MoeScaleExecutionMode.INFER_FORWARD),
            (1, "comet", MoeScaleExecutionMode.INFER_FORWARD),
            (2, "comet", MoeScaleExecutionMode.INFER_FORWARD),
            (1, "comet", MoeScaleExecutionMode.TRAIN_FORWARD),
        )
        for case_index, algorithm, mode in rows:
            case = cases[case_index]
            execution = build_moe_swizzle_execution(
                case.spec, case.oracle, mode,
            )
            problems = tuple(
                build_moe_swizzle_problem(
                    region, case.spec, case.oracle, execution,
                    hardware_facts=case.hardware_facts,
                    endpoint_session_contract=case.endpoint_session_contract,
                )
                for region in discover_moe_swizzle_regions(case.spec, case.oracle, execution)
            )
            baselines = tuple(
                build_executable_moe_unfused_baseline(
                    problem, case.spec, case.oracle, execution,
                )
                for problem in problems
            )
            candidates = tuple(
                (
                    build_direct_xy_moe_candidate(
                        problem, case.spec, case.oracle, execution,
                    )
                    if algorithm == "direct"
                    else build_comet_mesh_moe_candidates(
                        problem, case.spec, case.oracle, execution,
                    )[1]
                )
                for problem in problems
            )
            decisions = tuple(
                MoeSwizzleDecision.create(
                    problem=problem, baseline=baseline,
                    ranked_candidates=(candidate, baseline),
                    selected_candidate_ref=candidate.id,
                    decision_reason=SwizzleDecisionReason.NO_PROFITABLE_FUSION,
                    performance_complete=False,
                )
                for problem, baseline, candidate in zip(
                    problems, baselines, candidates, strict=True,
                )
            )
            overlay = build_moe_scale_swizzle_overlay(execution, decisions)
            projection = project_moe_scale_swizzle_ir2(
                overlay, execution, case.spec, decisions,
                endpoint_session_capacity=case.endpoint_session_contract.capacity_per_core,
            )
            expected_m = {
                len(action.assignment_refs)
                for candidate in candidates
                for program in candidate.rank_programs
                for action in program.actions
                if action.kind.value == "comp"
            }
            self.assertGreater(max(expected_m), 1)
            self.assertEqual(
                {item.matmul_m for item in projection.tasks if item.matmul_m is not None},
                expected_m,
            )
            combined = tuple(
                item for item in execution.terminals
                if item.kind is MoeScaleExecutionTerminalKind.COMBINED
            )
            combine_candidate = next(
                item for item in candidates
                if item.pattern.value == "moe_gemm_combine"
            )
            remote_count = len(case.oracle.remote_token_indices)
            local_count = case.spec.tokens - remote_count
            self.assertEqual(
                len(projection.terminal_refs),
                local_count * combine_candidate.compute_output_block_count
                + remote_count * combine_candidate.transport_output_block_count,
            )
            self.assertEqual(
                sum(
                    value.size_bytes if value.terminal_ref is not None
                    else sum(item.size_bytes for item in value.terminal_slices)
                    for value in projection.values
                ),
                sum(item.bytes for item in combined),
            )
            ir1 = cases[0].c0_production_case.forward.n4.graph
            state_abi = build_moe_swizzle_workload_state_abi(
                ir1, execution, case.spec, case.oracle,
            )
            workload = project_moe_swizzle_whole_workload(
                overlay, execution, projection, state_abi,
            )
            operand_abi = build_moe_swizzle_operand_abi(projection)
            self.assertEqual(operand_abi.source_projection_id, projection.id)
            placement = dict(build_moe_swizzle_task_placement(
                ir1, projection, case.hardware_facts,
            ))
            workload_placement = {
                item.action_ref: item for item in build_moe_swizzle_workload_placement(
                    ir1, workload, projection, case.hardware_facts,
                )
            }
            comp_groups = {}
            for task in projection.tasks:
                if task.kind.value != "comp":
                    continue
                comp_groups.setdefault(
                    (task.rank, task.expert_index, task.pipeline_index), set(),
                ).add(placement[task.id].runtime_core_id)
            self.assertTrue(all(len(cores) == 1 for cores in comp_groups.values()))
            for rank in range(4):
                rank_cores = {
                    next(iter(cores))
                    for (task_rank, _, _), cores in comp_groups.items()
                    if task_rank == rank
                }
                self.assertGreaterEqual(len(rank_cores), 2)
            tasks = {item.id: item for item in projection.tasks}
            for value in projection.values:
                if value.producer_task_ref is None:
                    continue
                producer_core = placement[value.producer_task_ref].runtime_core_id
                consumer_refs = set(value.consumer_task_refs)
                consumer_refs.update(
                    ref for alias in projection.values
                    if value.id in alias.alias_source_refs
                    for ref in alias.consumer_task_refs
                )
                self.assertTrue(all(
                    placement[ref].runtime_core_id == producer_core
                    for ref in consumer_refs
                ))
            for action in workload.actions:
                if not action.preserved:
                    continue
                owner = workload_placement[action.id]
                self.assertTrue(all(
                    workload_placement[ref].runtime_core_id == owner.runtime_core_id
                    for ref in owner.owner_witness_refs
                ))
            self.assertEqual({item.id for item in workload.actions}, {item.id for item in overlay.linked_actions})
            tape = case.spec.tokens if mode is MoeScaleExecutionMode.TRAIN_FORWARD else 0
            self.assertEqual(
                len([item for item in workload.actions if item.preserved]),
                3 * case.spec.tokens + tape,
            )
            self.assertEqual({item.value_ref for item in workload.terminals}, {item.value_ref for item in execution.terminals})
            if algorithm == "direct":
                self.assertEqual(
                    loads_dataclass(
                        MoeSwizzleWorkloadProjection,
                        canonical_json(workload),
                        path="workload",
                    ),
                    workload,
                )
                with self.assertRaises(SchemaError):
                    replace(workload, actions=workload.actions[:-1]).validate()
            self.assertTrue(any(value.alias_source_refs for value in projection.values))
            self.assertEqual(

                max(item.m for item in operand_abi.matmuls),
                max(expected_m),
            )
    def test_forced_representative_c1_c2_whole_endpoint_and_root_quotient(self) -> None:
        cases = build_moe_swizzle_scale_cases()
        expected_m = {"C1": 2, "C2": 4, "C3": 4}
        root_keys_by_scale = {}
        for case in (cases[1], cases[2], cases[3]):
            execution = build_moe_swizzle_execution(
                case.spec, case.oracle, MoeScaleExecutionMode.INFER_FORWARD,
            )
            problems = tuple(
                build_moe_swizzle_problem(
                    region, case.spec, case.oracle, execution,
                    hardware_facts=case.hardware_facts,
                    endpoint_session_contract=case.endpoint_session_contract,
                )
                for region in discover_moe_swizzle_regions(
                    case.spec, case.oracle, execution,
                )
            )
            baselines = tuple(
                build_executable_moe_unfused_baseline(
                    problem, case.spec, case.oracle, execution,
                )
                for problem in problems
            )
            selected = tuple(
                next(
                    item for item in build_direct_xy_moe_candidates(
                        problem, case.spec, case.oracle, execution,
                    )
                    if item.token_block_size == expected_m[case.spec.name]
                    and item.unroll_degree == 1
                    and item.transport_output_block_count == 1
                )
                for problem in problems
            )
            decisions = tuple(
                MoeSwizzleDecision.create(
                    problem=problem,
                    baseline=baseline,
                    ranked_candidates=(candidate, baseline),
                    selected_candidate_ref=candidate.id,
                    decision_reason=SwizzleDecisionReason.NO_PROFITABLE_FUSION,
                    performance_complete=False,
                )
                for problem, baseline, candidate in zip(
                    problems, baselines, selected, strict=True,
                )
            )
            self.assertEqual(
                {item.token_block_size for item in selected},
                {expected_m[case.spec.name]},
            )
            self.assertEqual(
                tuple(item.transport_output_block_count for item in selected),
                (1, 1),
            )
            overlay = build_moe_scale_swizzle_overlay(execution, decisions)
            projection = project_moe_scale_swizzle_ir2(
                overlay, execution, case.spec, decisions,
                endpoint_session_capacity=case.endpoint_session_contract.capacity_per_core,
            )
            ir1 = cases[0].c0_production_case.forward.n4.graph
            state_abi = build_moe_swizzle_workload_state_abi(
                ir1, execution, case.spec, case.oracle,
            )
            workload = project_moe_swizzle_whole_workload(
                overlay, execution, projection, state_abi,
            )
            workload_placement = build_moe_swizzle_workload_placement(
                ir1, workload, projection, case.hardware_facts,
            )
            workload = schedule_moe_swizzle_workload_endpoints(
                workload, projection, workload_placement,
                capacity_per_core=case.endpoint_session_contract.capacity_per_core,
            )
            workload_placement = build_moe_swizzle_workload_placement(
                ir1, workload, projection, case.hardware_facts,
            )
            value_bridge = build_moe_swizzle_workload_value_bridge(
                execution, workload, projection,
            )
            workload = schedule_moe_swizzle_workload_storage_reuse(
                workload, projection, value_bridge, workload_placement,
            )
            workload_placement = build_moe_swizzle_workload_placement(
                ir1, workload, projection, case.hardware_facts,
            )
            endpoint_widths = build_moe_swizzle_workload_endpoint_widths(
                workload, projection, workload_placement,
                capacity_per_core=case.endpoint_session_contract.capacity_per_core,
            )
            value_bridge = build_moe_swizzle_workload_value_bridge(
                execution, workload, projection,
            )
            workload_abi = build_moe_swizzle_workload_abi(
                ir1, workload, projection, state_abi, value_bridge, case.hardware_facts,
            )
            root_keys_by_scale[case.spec.name] = {
                (item.runtime_core_id, item.family, item.slot)
                for item in workload_abi.roots
            }
            self.assertLessEqual(len(workload_abi.roots), 16 * 8)
            self.assertEqual(
                (len(workload_abi.roots), workload_abi.alloc_count, workload_abi.free_count),
                (128, 112, 96) if case.spec.name == "C3" else (108, 100, 92),
            )
            terminal_root_count = sum(
                item.family.startswith("terminal_") for item in workload_abi.roots
            )
            self.assertEqual(
                workload_abi.alloc_count,
                workload_abi.free_count + terminal_root_count,
            )
            self.assertLessEqual(
                max(item.max_inflight for item in endpoint_widths),
                case.endpoint_session_contract.capacity_per_core,
            )
            selected_root_keys = tuple(
                key
                for problem, candidate in zip(problems, selected, strict=True)
                for key in build_moe_candidate_dynamic_root_keys(problem, candidate)
            )
            projected_root_keys = build_moe_projection_dynamic_root_keys(
                ir1, projection, case.hardware_facts,
            )
            self.assertEqual(
                set(selected_root_keys), set(projected_root_keys),
                (case.spec.name, selected_root_keys, projected_root_keys),
            )
            whole_dynamic_keys = {
                (item.runtime_core_id, item.family, item.slot)
                for item in workload_abi.roots
                if item.family in ("dispatch_operand", "combine_output")
            }
            self.assertEqual(whole_dynamic_keys, set(projected_root_keys))
            self.assertEqual(
                build_moe_swizzle_workload_abi(
                    ir1, workload, projection, state_abi, value_bridge, case.hardware_facts,
                ),
                workload_abi,
            )
            validate_moe_swizzle_workload_abi_against(
                workload_abi, ir1, workload, projection, state_abi,
                value_bridge, case.hardware_facts,
            )
            lifetime_root = next(
                item for item in workload_abi.roots
                if item.family == "dispatch_operand"
                and item.lifetime_end_exclusive - item.lifetime_start > 1
            )
            early = replace(
                lifetime_root,
                lifetime_end_exclusive=lifetime_root.lifetime_end_exclusive - 1,
            )
            tampered_abi = replace(
                workload_abi,
                roots=tuple(
                    early if item == lifetime_root else item
                    for item in workload_abi.roots
                ),
            )
            with self.assertRaisesRegex(SchemaError, "lifetime/actions escape its root"):
                validate_moe_swizzle_workload_abi_against(
                    tampered_abi, ir1, workload, projection, state_abi,
                    value_bridge, case.hardware_facts,
                )

            if case.spec.name == "C1":
                naive_decisions = tuple(
                    decide_moe_swizzle(problem, baseline, ())
                    for problem, baseline in zip(problems, baselines, strict=True)
                )
                naive_overlay = build_moe_scale_swizzle_overlay(execution, naive_decisions)
                naive_projection = project_moe_scale_swizzle_ir2(
                    naive_overlay, execution, case.spec, naive_decisions,
                    endpoint_session_capacity=case.endpoint_session_contract.capacity_per_core,
                )
                naive_workload = project_moe_swizzle_whole_workload(
                    naive_overlay, execution, naive_projection, state_abi,
                )
                naive_placement = build_moe_swizzle_workload_placement(
                    ir1, naive_workload, naive_projection, case.hardware_facts,
                )
                naive_workload = schedule_moe_swizzle_workload_endpoints(
                    naive_workload, naive_projection, naive_placement,
                    capacity_per_core=case.endpoint_session_contract.capacity_per_core,
                )
                naive_placement = build_moe_swizzle_workload_placement(
                    ir1, naive_workload, naive_projection, case.hardware_facts,
                )
                naive_bridge = build_moe_swizzle_workload_value_bridge(
                    execution, naive_workload, naive_projection,
                )
                naive_workload = schedule_moe_swizzle_workload_storage_reuse(
                    naive_workload, naive_projection, naive_bridge, naive_placement,
                )
                naive_bridge = build_moe_swizzle_workload_value_bridge(
                    execution, naive_workload, naive_projection,
                )
                naive_abi = build_moe_swizzle_workload_abi(
                    ir1, naive_workload, naive_projection, state_abi, naive_bridge,
                    case.hardware_facts,
                )
                naive_fixed_counts = {
                    family: sum(item.family == family for item in naive_abi.roots)
                    for family in {
                        item.family for item in naive_abi.roots
                        if item.family not in ("dispatch_operand", "combine_output")
                    }
                }
                fused_fixed_counts = {
                    family: sum(item.family == family for item in workload_abi.roots)
                    for family in {
                        item.family for item in workload_abi.roots
                        if item.family not in ("dispatch_operand", "combine_output")
                    }
                }
                self.assertEqual(
                    naive_fixed_counts,
                    {
                        "boundary_input": 4,
                        "state_stage.gate": 8,
                        "state_stage.up": 8,
                        "state_stage.down": 8,
                        "swiglu_output": 8,
                        "terminal_combined": 4,
                    },
                )
                self.assertEqual(
                    fused_fixed_counts,
                    {
                        "boundary_input": 8,
                        "state_stage.gate": 16,
                        "state_stage.up": 16,
                        "state_stage.down": 16,
                        "swiglu_output": 16,
                        "terminal_combined": 8,
                    },
                )
                self.assertEqual(len(naive_abi.roots), 54)
                self.assertEqual(sum(item.allocate for item in naive_abi.roots), 50)
                self.assertEqual(sum(item.free for item in naive_abi.roots), 46)
                self.assertEqual(len(workload_abi.roots), 108)
                self.assertEqual(sum(item.allocate for item in workload_abi.roots), 100)
                self.assertEqual(sum(item.free for item in workload_abi.roots), 92)

        self.assertEqual(root_keys_by_scale["C1"], root_keys_by_scale["C2"])
        self.assertLess(root_keys_by_scale["C2"], root_keys_by_scale["C3"])
        self.assertEqual(len(root_keys_by_scale["C3"]), 16 * 8)


if __name__ == "__main__":
    unittest.main()
