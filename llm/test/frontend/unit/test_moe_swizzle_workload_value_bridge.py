from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_moe_scale_swizzle_overlay import (
    build_moe_scale_swizzle_overlay,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_execution import (
    build_moe_swizzle_execution,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_pair_feasibility import (
    build_moe_swizzle_pair_feasibility_witnesses,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_program_io import (
    build_moe_swizzle_program_io,
    validate_moe_swizzle_program_io_against,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_workload_state_abi import (
    build_moe_swizzle_workload_state_abi,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_workload_value_bridge import (
    build_moe_swizzle_workload_value_bridge,
    validate_moe_swizzle_workload_value_bridge_against,
)
from llm.frontend.wafer_frontend.passes.discover_moe_swizzle import (
    discover_moe_swizzle_regions,
)
from llm.frontend.wafer_frontend.passes.project_moe_scale_swizzle_ir2 import (
    project_moe_scale_swizzle_ir2,
)
from llm.frontend.wafer_frontend.passes.project_moe_swizzle_whole_workload import (
    project_moe_swizzle_whole_workload,
)
from llm.frontend.wafer_frontend.passes.schedule_moe_swizzle_workload_endpoints import (
    schedule_moe_swizzle_workload_endpoints,
)
from llm.frontend.wafer_frontend.passes.schedule_moe_swizzle_workload_storage_reuse import (
    schedule_moe_swizzle_workload_storage_reuse,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_direct_xy import (
    build_direct_xy_moe_candidate,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_cost import (
    select_moe_swizzle_workload_deployment,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_problem import (
    build_moe_swizzle_problem,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_unfused import (
    build_executable_moe_unfused_baseline,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleActionKind, SwizzleDecisionReason
from llm.frontend.wafer_frontend.schema.swizzle_moe import MoeSwizzleDecision
from llm.frontend.wafer_frontend.schema.swizzle_moe_ir2 import MoeSwizzleIr2Projection
from llm.frontend.wafer_frontend.schema.swizzle_moe_placement import (
    MoeSemanticWorkItem,
    MoeSemanticWorkKey,
    MoeWholePairPlacementReason,
    build_moe_candidate_action_owner_map,
    build_moe_candidate_core_lifecycle_floor,
    build_moe_candidate_core_fixed_lifecycle_floor,
    build_moe_candidate_dynamic_root_keys,
    build_moe_work_owner_map,
    build_moe_projection_dynamic_root_keys,
    build_moe_swizzle_task_placement,
    build_moe_swizzle_workload_placement,
    measure_moe_swizzle_workload_endpoint_widths,
)
from llm.frontend.wafer_frontend.lowering.moe_swizzle_abi import (
    allocate_moe_swizzle_core_address_abi,
)
from llm.frontend.wafer_frontend.lowering.moe_swizzle_workload_abi import (
    build_moe_swizzle_workload_abi,
)
from llm.frontend.wafer_frontend.lowering.moe_swizzle_workload_standard import (
    lower_moe_swizzle_workload_fragment,
)
from llm.frontend.wafer_frontend.lowering.moe_swizzle_workload_linker import (
    build_moe_swizzle_standard_linked_program,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoPurpose
from llm.frontend.wafer_frontend.schema.swizzle_moe_operand_abi import (
    build_moe_swizzle_operand_abi,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_execution import (
    MoeScaleExecutionMode,
    MoeScaleExecutionTerminalKind,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_workload_bridge import (
    MoeSwizzleWorkloadValueBridge,
    MoeSwizzleWorkloadValueKind,
)
from llm.test.frontend.integration.moe_swizzle_scale_cases import (
    build_moe_swizzle_scale_cases,
)


class MoeSwizzleWorkloadValueBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cases = build_moe_swizzle_scale_cases()
        case = cases[1]
        execution = build_moe_swizzle_execution(
            case.spec, case.oracle, MoeScaleExecutionMode.TRAIN_FORWARD,
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
        candidates = tuple(
            build_direct_xy_moe_candidate(
                problem, case.spec, case.oracle, execution,
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
                problems, baselines, candidates, strict=True,
            )
        )
        overlay = build_moe_scale_swizzle_overlay(execution, decisions)
        projection = project_moe_scale_swizzle_ir2(
            overlay, execution, case.spec, decisions,
            endpoint_session_capacity=case.endpoint_session_contract.capacity_per_core,
        )
        ir1 = cases[0].c0_production_case.forward.n4.graph
        state_abi = build_moe_swizzle_workload_state_abi(
            ir1,
            execution, case.spec, case.oracle,
        )
        workload = project_moe_swizzle_whole_workload(
            overlay, execution, projection, state_abi,
        )
        placement = build_moe_swizzle_workload_placement(
            ir1, workload, projection, case.hardware_facts,
        )
        workload = schedule_moe_swizzle_workload_endpoints(
            workload, projection, placement,
            capacity_per_core=case.endpoint_session_contract.capacity_per_core,
        )
        placement = build_moe_swizzle_workload_placement(
            ir1, workload, projection, case.hardware_facts,
        )
        bridge = build_moe_swizzle_workload_value_bridge(
            execution, workload, projection,
        )
        workload = schedule_moe_swizzle_workload_storage_reuse(
            workload, projection, bridge, placement,
        )
        cls.case = case
        cls.ir1 = ir1
        cls.decisions = decisions
        cls.state_abi = state_abi
        cls.execution = execution
        cls.projection = projection
        cls.workload = workload
        cls.bridge = build_moe_swizzle_workload_value_bridge(
            execution, workload, projection,
        )

    def test_whole_standard_fragment_is_exact_and_deterministic(self) -> None:
        core_abi = allocate_moe_swizzle_core_address_abi(
            self.ir1, self.projection, hardware_facts=self.case.hardware_facts,
            workload_projection=self.workload,
        )
        operand_abi = build_moe_swizzle_operand_abi(self.projection)
        workload_abi = build_moe_swizzle_workload_abi(
            self.ir1, self.workload, self.projection, self.state_abi,
            self.bridge, self.case.hardware_facts,
        )
        fragment = lower_moe_swizzle_workload_fragment(
            self.ir1, self.execution, self.workload, self.projection,
            core_abi, operand_abi, self.state_abi, self.bridge,
            workload_abi, self.case.hardware_facts,
        )
        self.assertEqual(
            fragment,
            lower_moe_swizzle_workload_fragment(
                self.ir1, self.execution, self.workload, self.projection,
                core_abi, operand_abi, self.state_abi, self.bridge,
                workload_abi, self.case.hardware_facts,
            ),
        )
        self.assertEqual(
            fragment.claimed_action_ids,
            tuple(sorted(item.id for item in self.workload.actions)),
        )
        self.assertEqual(
            {ref for item in self.workload.actions for ref in item.source_action_refs},
            {item.id for item in self.execution.actions},
        )
        roots = tuple(item for item in fragment.buffer_abi if item.alias_of is None)
        aliases = tuple(item for item in fragment.buffer_abi if item.ownership is BufferOwnership.ALIASED)
        self.assertEqual(len(roots), len(workload_abi.roots))
        self.assertEqual(len(roots) + len(aliases), len(fragment.buffer_abi))
        opcodes = {
            record.opcode for stream in fragment.core_streams for record in stream.records
        }
        self.assertTrue({
            RecordOpcode.LSU_LOAD, RecordOpcode.MATMUL, RecordOpcode.SRAM_BIND,
            RecordOpcode.SWIGLU, RecordOpcode.DTE_SEND, RecordOpcode.DTE_RECV,
            RecordOpcode.DTE_WAIT, RecordOpcode.DTE_ISSUE,
            RecordOpcode.SRAM_ALLOC_AT, RecordOpcode.SRAM_FREE,
        }.issubset(opcodes))
        self.assertEqual(len(fragment.state_abi), 12)

    def test_grouped_rows_weights_swiglu_and_terminals_are_exact(self) -> None:
        binding_by_ref = {
            item.semantic_value_ref: item for item in self.bridge.bindings
        }
        actions = {item.id: item for item in self.execution.actions}
        grouped = tuple(
            item for item in self.projection.tasks
            if item.matmul_m is not None and item.matmul_m > 1
        )
        self.assertTrue(grouped)
        for task in grouped:
            for row, original_ref in enumerate(task.original_action_refs):
                output_ref = actions[original_ref].write_values[0]
                physical = next(item for item in binding_by_ref[output_ref].physical_slices
                    if item.anchor_execution_action_ref == original_ref)
                physical_task = next(
                    item for item in self.projection.tasks
                    if item.id == physical.physical_task_ref
                )
                self.assertEqual(physical_task.id, task.id)
                self.assertEqual(
                    physical.physical_byte_offset, row * task.matmul_n * 2,
                )
                self.assertEqual(physical.shape, (1, task.matmul_n))

        by_kind = {}
        for item in self.bridge.bindings:
            by_kind.setdefault(item.kind, []).append(item)
        self.assertEqual(
            len(by_kind[MoeSwizzleWorkloadValueKind.WEIGHT_STAGING]),
            3 * self.case.spec.tokens,
        )
        self.assertEqual(
            len(by_kind[MoeSwizzleWorkloadValueKind.SWIGLU_OUTPUT]),
            self.case.spec.tokens,
        )
        combined = tuple(
            item for item in self.execution.terminals
            if item.kind is MoeScaleExecutionTerminalKind.COMBINED
        )
        for terminal in combined:
            binding = binding_by_ref[terminal.value_ref]
            self.assertIs(binding.terminal_kind, MoeScaleExecutionTerminalKind.COMBINED)
            exact = tuple(
                item for item in binding.physical_slices
                if item.ir2_terminal_ref is not None
            )
            self.assertTrue(exact)
            self.assertEqual(sum(item.size_bytes for item in exact), terminal.bytes)
        self.assertEqual(
            len(self.bridge.preserved_terminal_refs), self.case.spec.tokens,
        )

    def test_full_n_t2_transport_pipeline_tamper_is_rejected(self) -> None:
        target = next(
            item for item in self.projection.tasks
            if item.kind is SwizzleActionKind.SEND
            and item.work_role == "moe_gemm_combine.transport"
            and item.n_block == 1
        )
        tasks = tuple(
            replace(item, pipeline_index=1 - item.pipeline_index)
            if item.id == target.id else item
            for item in self.projection.tasks
        )
        tampered = MoeSwizzleIr2Projection.create(
            source_execution_id=self.projection.source_execution_id,
            source_overlay_id=self.projection.source_overlay_id,
            tasks=tasks,
            values=self.projection.values,
            buffers=self.projection.buffers,
            flows=self.projection.flows,
            terminal_refs=self.projection.terminal_refs,
            endpoint_session_capacity=self.projection.endpoint_session_capacity,
        )
        with self.assertRaisesRegex(
            SchemaError,
            "transport pipeline does not match its exact placement group",
        ):
            build_moe_swizzle_task_placement(
                self.ir1, tampered, self.case.hardware_facts,
            )

    def test_candidate_lifecycle_floor_and_owner_map_are_whole_stable(self) -> None:
        baseline_decisions = tuple(
            MoeSwizzleDecision.create(
                problem=decision.problem,
                baseline=decision.baseline,
                ranked_candidates=(decision.baseline,) + tuple(
                    item for item in decision.ranked_candidates
                    if item.id != decision.baseline.id
                ),
                selected_candidate_ref=decision.baseline.id,
                decision_reason=decision.decision_reason,
                performance_complete=decision.performance_complete,
            )
            for decision in self.decisions
        )
        overlay = build_moe_scale_swizzle_overlay(
            self.execution, baseline_decisions,
        )
        projection = project_moe_scale_swizzle_ir2(
            overlay, self.execution, self.case.spec, baseline_decisions,
            endpoint_session_capacity=(
                self.case.endpoint_session_contract.capacity_per_core
            ),
        )
        projected_owners = dict(build_moe_swizzle_task_placement(
            self.ir1, projection, self.case.hardware_facts,
        ))
        candidate_dynamic_roots = set()
        for decision in baseline_decisions:
            candidate = decision.baseline
            actions = tuple(
                action for program in candidate.rank_programs
                for action in program.actions
            )
            candidate_owners = build_moe_candidate_action_owner_map(
                decision.problem, actions,
            )
            self.assertEqual(
                {
                    ref: owner.runtime_core_id
                    for ref, owner in candidate_owners.items()
                },
                {
                    action.id: projected_owners[action.id].runtime_core_id
                    for action in actions
                },
            )
            floor = build_moe_candidate_core_lifecycle_floor(
                decision.problem, candidate,
            )
            fixed_floor = build_moe_candidate_core_fixed_lifecycle_floor(
                decision.problem, candidate,
            )
            total_by_core = {
                item.runtime_core_id: item for item in floor
            }
            fixed_by_core = {
                item.runtime_core_id: item for item in fixed_floor
            }
            dynamic_by_core = {}
            candidate_dynamic_roots.update(
                build_moe_candidate_dynamic_root_keys(
                    decision.problem, candidate,
                )
            )
            for runtime_core_id, _family, _slot in (
                build_moe_candidate_dynamic_root_keys(
                    decision.problem, candidate,
                )
            ):
                dynamic_by_core[runtime_core_id] = (
                    dynamic_by_core.get(runtime_core_id, 0) + 1
                )
            for runtime_core_id, item in total_by_core.items():
                fixed = fixed_by_core.get(runtime_core_id)
                dynamic = dynamic_by_core.get(runtime_core_id, 0)
                self.assertEqual(
                    (item.alloc_count, item.bind_count, item.free_count),
                    (
                        (0 if fixed is None else fixed.alloc_count) + dynamic,
                        0 if fixed is None else fixed.bind_count,
                        (0 if fixed is None else fixed.free_count) + dynamic,
                    ),
                )
            self.assertTrue(floor)
            self.assertTrue(all(item.candidate_ref == candidate.id for item in floor))
            with self.assertRaisesRegex(SchemaError, "lifecycle floor is invalid"):
                replace(
                    floor[0], free_count=floor[0].alloc_count + 1,
                ).validate("floor")
        self.assertEqual(
            candidate_dynamic_roots,
            set(build_moe_projection_dynamic_root_keys(
                self.ir1, projection, self.case.hardware_facts,
            )),
        )

        # Pipeline provenance cannot turn source-token work into expert-tile
        # placement affinity.  Both typed keys must stay on one real core.
        source_items = (
            MoeSemanticWorkItem(0, MoeSemanticWorkKey(
                "dispatch_source_token", 7, 1, 9, None,
                "moe_dispatch_gemm.transport", 0,
            )),
            MoeSemanticWorkItem(0, MoeSemanticWorkKey(
                "dispatch_source_token", 7, 1, 9, None,
                "moe_dispatch_gemm.transport", 1,
            )),
        )
        source_owners = build_moe_work_owner_map(
            self.case.hardware_facts, source_items,
        )
        self.assertEqual(
            len({item.runtime_core_id for item in source_owners}), 1,
        )

    def test_real_complete_pair_witnesses_feed_joint_selector(self) -> None:
        witnesses = build_moe_swizzle_pair_feasibility_witnesses(
            self.ir1, self.execution, self.case.spec,
            self.decisions, self.state_abi,
        )
        expected = {
            (dispatch.id, combine.id)
            for dispatch in self.decisions[0].ranked_candidates
            for combine in self.decisions[1].ranked_candidates
        }
        self.assertEqual(
            tuple(item.candidate_refs for item in witnesses),
            tuple(sorted(expected)),
        )
        placed = tuple(item for item in witnesses if item.placement.feasible)
        unplaced = tuple(item for item in witnesses if not item.placement.feasible)
        self.assertTrue(placed)
        self.assertTrue(unplaced)
        self.assertTrue(all(item.endpoint.widths for item in placed))
        self.assertTrue(all(item.dynamic_root_keys for item in placed))
        self.assertTrue(all(item.core_lifecycle_counts for item in placed))
        self.assertTrue(all(
            sum(core.alloc_count for core in item.core_lifecycle_counts)
            == item.whole_alloc_count
            and sum(core.free_count for core in item.core_lifecycle_counts)
            == item.whole_free_count
            for item in placed
        ))
        for item in unplaced:
            self.assertIn(
                item.placement.reason,
                (
                    MoeWholePairPlacementReason.ORDINARY_VALUE_CROSS_CORE,
                    MoeWholePairPlacementReason.PRESERVED_OWNER_MISMATCH,
                ),
            )
            self.assertIsNone(item.endpoint)
            self.assertIsNone(item.workload_abi_id)
            self.assertEqual(item.dynamic_root_keys, ())
            self.assertEqual(item.dynamic_sram_high_water_bytes, 0)
            self.assertEqual(item.core_lifecycle_counts, ())
            self.assertFalse(item.feasible)
        with self.assertRaisesRegex(SchemaError, "cannot fabricate"):
            replace(
                unplaced[0],
                dynamic_root_keys=((0, "forged", 0),),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "disagrees with reason"):
            replace(
                unplaced[0].placement,
                feasible=True,
            ).validate()
        selection = select_moe_swizzle_workload_deployment(
            self.decisions[0], self.decisions[1], witnesses,
        )
        self.assertEqual(
            (selection.selected_dispatch_candidate_ref,
             selection.selected_combine_candidate_ref),
            (self.decisions[0].baseline.id, self.decisions[1].baseline.id),
        )
        self.assertEqual(selection.pair_feasibilities, witnesses)
        linked = build_moe_swizzle_standard_linked_program(
            self.ir1, self.case.spec, self.case.oracle, self.execution,
            self.decisions, selection, self.case.hardware_facts,
            "0" * 64,
        )
        alloc_locations = []
        for stream in linked.fragment.core_streams:
            for index, record in enumerate(stream.records):
                if record.opcode is RecordOpcode.SRAM_ALLOC_AT:
                    label = next(
                        symbol for symbol in linked.fragment.program_symbols
                        if symbol.id == record.operands[1].symbol_ref
                    )
                    root = next(
                        binding for binding in linked.fragment.buffer_abi
                        if binding.alias_of is None
                        and binding.storage_id == label.source_ref
                    )
                    self.assertEqual(
                        record.operands[5].literal_value,
                        2 if root.layout.startswith("moe_swizzle_terminal_") else 0,
                    )
                    alloc_locations.append((stream, index, record))
                if record.opcode is not RecordOpcode.MATMUL:
                    continue
                self.assertGreater(index, 0)
                bind = stream.records[index - 1]
                self.assertIs(bind.opcode, RecordOpcode.SRAM_BIND)
                self.assertEqual(bind.source_global_action_id, record.source_global_action_id)
                self.assertEqual(bind.operands[0].literal_value, 2)
        for original_lifetime, forged_lifetime in ((2, 0), (0, 2)):
            stream, record_index, record = next(
                item for item in alloc_locations
                if item[2].operands[5].literal_value == original_lifetime
            )
            operands = list(record.operands)
            operands[5] = replace(operands[5], literal_value=forged_lifetime)
            records = list(stream.records)
            records[record_index] = replace(record, operands=tuple(operands))
            streams = tuple(
                replace(item, records=tuple(records)) if item is stream else item
                for item in linked.fragment.core_streams
            )
            with self.assertRaisesRegex(
                SchemaError,
                "PERSISTENT allocations must exactly cover terminal roots",
            ):
                replace(linked.fragment, core_streams=streams).validate()
        program_io = linked.program_io
        self.assertIsNotNone(program_io)
        assert program_io is not None
        validate_moe_swizzle_program_io_against(program_io, linked)
        self.assertEqual(program_io.program_artifact_sha256, "0" * 64)
        timing_initializations = tuple(
            item for item in program_io.initializations
            if item.purpose is ProgramIoPurpose.TIMING_PARTIAL
        )
        self.assertEqual(
            len(program_io.initializations),
            len([item for item in linked.fragment.buffer_abi
                 if item.alias_of is None
                 and item.ownership is BufferOwnership.BORROWED])
            + len(linked.fragment.state_abi)
            + len(timing_initializations),
        )
        self.assertTrue(timing_initializations)
        self.assertTrue(
            {item.target.layout for item in timing_initializations}.issubset({
                "moe_swizzle_combine_output_root/v1",
                "moe_swizzle_swiglu_output_root/v1",
                "moe_swizzle_terminal_combined_root/v1",
            }),
        )
        removed = timing_initializations[0]
        forged_initializations = tuple(
            item for item in program_io.initializations if item is not removed
        )
        used_blobs = {
            item.blob_ref
            for item in (*forged_initializations, *program_io.output_probes)
        }
        forged_timing_io = type(program_io).create(
            producer_pass=program_io.producer_pass,
            mode=program_io.mode,
            source_manifest=linked.manifest,
            program_artifact_sha256=program_io.program_artifact_sha256,
            blobs=tuple(
                item for item in program_io.blobs if item.id in used_blobs
            ),
            initializations=forged_initializations,
            output_probes=program_io.output_probes,
        )
        with self.assertRaisesRegex(
            SchemaError, "deterministic whole-workload rebuild",
        ):
            validate_moe_swizzle_program_io_against(
                forged_timing_io, linked,
            )
        self.assertEqual(
            len(program_io.output_probes), len(linked.workload.terminals),
        )
        self.assertEqual(
            sum(item.length_bytes for item in program_io.output_probes),
            sum(item.bytes for item in linked.workload.terminals),
        )
        forged_io = type(program_io).create(
            producer_pass=program_io.producer_pass,
            mode=program_io.mode,
            source_manifest=linked.manifest,
            program_artifact_sha256=program_io.program_artifact_sha256,
            blobs=program_io.blobs,
            initializations=program_io.initializations,
            output_probes=program_io.output_probes[:-1],
        )
        with self.assertRaisesRegex(SchemaError, "deterministic whole-workload rebuild"):
            validate_moe_swizzle_program_io_against(forged_io, linked)
        self.assertEqual(linked.selection, selection)
        self.assertEqual(
            linked.manifest.source_global_dag_id, linked.workload.id,
        )

        self.assertEqual(len(linked.manifest.input_digests), 17)
        self.assertEqual(
            linked.fragment.claimed_action_ids,
            tuple(sorted(item.id for item in linked.workload.actions)),
        )
        selected_witness = next(
            item for item in witnesses
            if item.candidate_refs == (
                selection.selected_dispatch_candidate_ref,
                selection.selected_combine_candidate_ref,
            )
        )
        self.assertTrue(selected_witness.placement.feasible)
        joint_overlay = build_moe_scale_swizzle_overlay(
            self.execution, self.decisions, selection,
        )
        self.assertEqual(joint_overlay.source_workload_selection_id, selection.id)
        self.assertEqual(
            tuple(item.candidate_ref for item in joint_overlay.deployment_selections),
            (
                selection.selected_dispatch_candidate_ref,
                selection.selected_combine_candidate_ref,
            ),
        )
        with self.assertRaisesRegex(SchemaError, "lineage"):
            project_moe_scale_swizzle_ir2(
                joint_overlay, self.execution, self.case.spec, self.decisions,
                endpoint_session_capacity=self.case.endpoint_session_contract.capacity_per_core,
            )
        joint_projection = project_moe_scale_swizzle_ir2(
            joint_overlay, self.execution, self.case.spec, self.decisions,
            selection,
            endpoint_session_capacity=self.case.endpoint_session_contract.capacity_per_core,
        )
        self.assertEqual(joint_projection.source_overlay_id, joint_overlay.id)
        with self.assertRaises(SchemaError):
            replace(
                joint_overlay,
                source_workload_selection_id=self.decisions[0].id,
            ).validate()
        with self.assertRaises(SchemaError):
            build_moe_scale_swizzle_overlay(
                self.execution,
                self.decisions,
                replace(
                    selection,
                    source_dispatch_decision_id=self.decisions[1].id,
                ),
            )

    def test_c2_train_relanes_endpoint_sessions_across_regions(self) -> None:
        case = build_moe_swizzle_scale_cases()[2]
        execution = build_moe_swizzle_execution(
            case.spec, case.oracle, MoeScaleExecutionMode.TRAIN_FORWARD,
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
        candidates = tuple(
            build_direct_xy_moe_candidate(
                problem, case.spec, case.oracle, execution,
            )
            for problem in problems
        )
        baselines = tuple(
            build_executable_moe_unfused_baseline(
                problem, case.spec, case.oracle, execution,
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
                problems, baselines, candidates, strict=True,
            )
        )
        overlay = build_moe_scale_swizzle_overlay(execution, decisions)
        actions = {
            action.id: action
            for program in overlay.replacement_rank_programs
            for action in program.actions
        }
        cross_region_retire_edges = tuple(
            (dependency, action.id)
            for action in actions.values()
            if action.kind is SwizzleActionKind.RECV
            for dependency in action.deps
            if actions[dependency].kind is SwizzleActionKind.WAIT
            and actions[dependency].work_role != action.work_role
        )
        self.assertTrue(cross_region_retire_edges)
        projection = project_moe_scale_swizzle_ir2(
            overlay, execution, case.spec, decisions,
            endpoint_session_capacity=(
                case.endpoint_session_contract.capacity_per_core
            ),
        )
        state_abi = build_moe_swizzle_workload_state_abi(
            self.ir1, execution, case.spec, case.oracle,
        )
        workload = project_moe_swizzle_whole_workload(
            overlay, execution, projection, state_abi,
        )
        placement = build_moe_swizzle_workload_placement(
            self.ir1, workload, projection, case.hardware_facts,
        )
        workload = schedule_moe_swizzle_workload_endpoints(
            workload, projection, placement,
            capacity_per_core=(
                case.endpoint_session_contract.capacity_per_core
            ),
        )
        placement = build_moe_swizzle_workload_placement(
            self.ir1, workload, projection, case.hardware_facts,
        )
        widths = measure_moe_swizzle_workload_endpoint_widths(
            workload, projection, placement,
            capacity_per_core=(
                case.endpoint_session_contract.capacity_per_core
            ),
        )
        self.assertLessEqual(
            max(item.max_inflight for item in widths),
            case.endpoint_session_contract.capacity_per_core,
        )


    def test_serde_determinism_rebuild_and_tamper(self) -> None:
        self.assertEqual(
            loads_dataclass(
                MoeSwizzleWorkloadValueBridge,
                canonical_json(self.bridge),
                path="bridge",
            ),
            self.bridge,
        )
        self.assertEqual(
            build_moe_swizzle_workload_value_bridge(
                self.execution, self.workload, self.projection,
            ),
            self.bridge,
        )
        validate_moe_swizzle_workload_value_bridge_against(
            self.bridge, self.execution, self.workload, self.projection,
        )
        first = self.bridge.bindings[0]
        forged_binding = replace(first, producer_action_refs=())
        forged = type(self.bridge).create(
            source_execution_id=self.bridge.source_execution_id,
            source_overlay_id=self.bridge.source_overlay_id,
            source_workload_projection_id=self.bridge.source_workload_projection_id,
            source_replacement_projection_id=self.bridge.source_replacement_projection_id,
            bindings=(forged_binding, *self.bridge.bindings[1:]),
            preserved_terminal_refs=self.bridge.preserved_terminal_refs,
        )
        with self.assertRaisesRegex(SchemaError, "deterministic typed rebuild"):
            validate_moe_swizzle_workload_value_bridge_against(
                forged, self.execution, self.workload, self.projection,
            )
        with self.assertRaises(SchemaError):
            replace(
                self.bridge,
                bindings=(replace(first, physical_slices=()), *self.bridge.bindings[1:]),
            ).validate()


if __name__ == "__main__":
    unittest.main()
