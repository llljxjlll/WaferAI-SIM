from __future__ import annotations

import json

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_execution import (
    build_moe_swizzle_execution,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_overlay import (
    build_moe_swizzle_overlay,
)
from llm.frontend.wafer_frontend.passes.project_moe_swizzle_ir2 import (
    project_moe_swizzle_ir2,
    validate_moe_swizzle_ir2_against_overlay,
)
from llm.frontend.wafer_frontend.lowering.moe_swizzle_abi import (
    allocate_moe_swizzle_core_address_abi,
)
from llm.frontend.wafer_frontend.lowering.moe_swizzle_standard import (
    lower_moe_swizzle_standard_fragment,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    AddressRelocation, CommandFragment, CoreFragmentStream, FragmentKind,
    ProgramSymbol, ProgramSymbolKind, RecordOpcode, RecordOperand,
    RelocatableRecord, SemanticOperandId, StateABI,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.global_action import LogicalCoreRef
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateAccess, PersistentStateLifetime, StateKind,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_operand_abi import (
    build_moe_swizzle_operand_abi,
)
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleActionKind
from llm.frontend.wafer_frontend.schema.swizzle_moe import (
    MoeEndpointSessionContract, MoeHardwareFacts,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_scale import (
    build_moe_swizzle_scale_truth,
)
from llm.frontend.wafer_frontend.passes.discover_moe_swizzle import (
    discover_moe_swizzle_regions,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_cost import (
    decide_moe_swizzle,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_problem import (
    build_moe_swizzle_problem,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_unfused import (
    build_executable_moe_unfused_baseline,
)
from llm.frontend.wafer_frontend.schema.lite_moe_dp4_execution import (
    LiteMoeDp4TaskKind,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_execution import (
    MoeScaleExecutionMode,
)
from llm.test.frontend.integration.lite_moe_dp4_cases import (
    LiteMoeDp4Mode,
    build_lite_moe_dp4_case,
)


class MoeSwizzleOverlayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.infer_case = build_lite_moe_dp4_case(LiteMoeDp4Mode.INFER)
        cls.train_case = build_lite_moe_dp4_case(LiteMoeDp4Mode.TRAIN_FORWARD)
        truth = build_moe_swizzle_scale_truth(
            cls.infer_case.spec,
            cls.infer_case.topology,
        )
        cls.spec, cls.oracle = truth[0]
        cls.execution = build_moe_swizzle_execution(
            cls.spec,
            cls.oracle,
            MoeScaleExecutionMode.INFER_FORWARD,
        )
        hardware_facts = MoeHardwareFacts.from_fabric(
            physical_fabric_from_data(json.loads(cls.infer_case.hardware_json), path="moe_overlay.hardware")
        )
        endpoint_contract = MoeEndpointSessionContract.production()
        regions = discover_moe_swizzle_regions(
            cls.spec,
            cls.oracle,
            cls.execution,
        )
        problems = tuple(
            build_moe_swizzle_problem(
                region,
                cls.spec,
                cls.oracle,
                cls.execution,
                hardware_facts=hardware_facts,
                endpoint_session_contract=endpoint_contract,
            )
            for region in regions
        )
        baselines = tuple(
            build_executable_moe_unfused_baseline(
                problem,
                cls.spec,
                cls.oracle,
                cls.execution,
            )
            for problem in problems
        )
        cls.decisions = tuple(
            decide_moe_swizzle(problem, baseline, ())
            for problem, baseline in zip(problems, baselines, strict=True)
        )
        cls.overlay = build_moe_swizzle_overlay(
            cls.infer_case.forward,
            cls.decisions,
            source_execution=cls.execution,
        )

    def test_exact_single_replacement_partition(self) -> None:
        overlay = self.overlay
        source = {item.id for item in self.infer_case.forward.global_dag.actions}
        replaced = set(overlay.replaced_action_refs)
        preserved = set(overlay.preserved_action_refs)
        provenance = tuple(
            ref for action in overlay.linked_actions for ref in action.source_action_refs
        )
        self.assertEqual((len(replaced), len(preserved)), (68, 24))
        self.assertFalse(replaced & preserved)
        self.assertEqual(replaced | preserved, source)
        self.assertEqual(len(provenance), len(set(provenance)))
        self.assertEqual(set(provenance), source)
        self.assertEqual(len(overlay.linked_actions), 92)
        self.assertEqual(
            Counter(item.kind for item in overlay.linked_actions if item.preserved),
            Counter({"preserved.dma_in": 24}),
        )

    def test_state_dispatch_swiglu_down_and_combine_are_rewired(self) -> None:
        forward = self.infer_case.forward
        tasks = {
            task.id: task for die in forward.projection.dies for task in die.tasks
        }
        source_actions = {item.id: item for item in forward.global_dag.actions}
        owner = {
            ref: action.id
            for action in self.overlay.linked_actions
            for ref in action.source_action_refs
        }
        linked = {item.id: item for item in self.overlay.linked_actions}
        for source in forward.global_dag.actions:
            task = tasks[source.task_ref]
            action = linked[owner[source.id]]
            if task.kind is LiteMoeDp4TaskKind.GEMM:
                dma_deps = {
                    owner[ref]
                    for ref in source.deps
                    if source_actions[ref].kind is LiteMoeDp4TaskKind.DMA_IN
                }
                self.assertTrue(dma_deps.issubset(set(action.deps)))
                if task.node_ref.endswith(".down"):
                    swiglu_deps = {
                        owner[ref]
                        for ref in source.deps
                        if source_actions[ref].kind is LiteMoeDp4TaskKind.SWIGLU
                    }
                    self.assertEqual(len(swiglu_deps), 1)
                    self.assertTrue(swiglu_deps.issubset(set(action.deps)))
            elif task.kind is LiteMoeDp4TaskKind.SWIGLU:
                self.assertFalse(action.preserved)
                self.assertEqual(action.kind, "replacement.swiglu")
                self.assertEqual(
                    {linked[ref].kind for ref in action.deps},
                    {"replacement.comp"},
                )
            elif task.kind is LiteMoeDp4TaskKind.SEND and task.node_ref.endswith(".combine"):
                down_deps = {
                    owner[ref]
                    for ref in source.deps
                    if tasks[source_actions[ref].task_ref].node_ref.endswith(".down")
                }
                self.assertTrue(down_deps.issubset(set(action.deps)))
        producer = {
            value: action
            for action in self.overlay.linked_actions
            for value in action.write_value_refs
        }
        self.assertEqual(
            Counter(producer[ref].kind for ref in self.overlay.terminal_value_refs),
            Counter({"replacement.recv": 6, "replacement.comp": 2}),
        )

    def test_train_forward_preserves_tape_fork_without_double_compute(self) -> None:
        assert self.train_case.train_forward is not None
        overlay = build_moe_swizzle_overlay(
            self.train_case.forward,
            self.decisions,
            train_forward=self.train_case.train_forward,
            source_execution=self.execution,
        )
        self.assertEqual(
            (len(overlay.replaced_action_refs), len(overlay.preserved_action_refs), len(overlay.linked_actions)),
            (68, 32, 100),
        )
        tape = tuple(item for item in overlay.linked_actions if item.kind == "preserved.local_copy")
        self.assertEqual(len(tape), 8)
        self.assertEqual(len(overlay.terminal_value_refs), 16)
        linked = {item.id: item for item in overlay.linked_actions}
        for item in tape:
            self.assertEqual(len(item.metadata_action_refs), 1)
            self.assertEqual(linked[item.metadata_action_refs[0]].kind, "replacement.comp")
            self.assertEqual(linked[item.deps[0]].kind, "replacement.swiglu")

    def test_partition_and_source_bridge_tampering_fail_closed(self) -> None:
        with self.assertRaises(SchemaError):
            build_moe_swizzle_overlay(
                self.infer_case.forward,
                self.decisions,
            )
    def test_replacement_only_projection_and_multicore_abis(self) -> None:
        projection = project_moe_swizzle_ir2(
            self.overlay,
            self.infer_case.forward,
            endpoint_session_capacity=4,
        )
        self.assertEqual(
            (len(projection.tasks), len(projection.values), len(projection.buffers), len(projection.flows), len(projection.terminal_refs)),
            (68, 84, 12, 12, 8),
        )
        originals = tuple(
            ref for task in projection.tasks for ref in task.original_action_refs
        )
        self.assertEqual(set(originals), set(self.overlay.replaced_action_refs))
        self.assertFalse(set(originals) & set(self.overlay.preserved_action_refs))
        abi = allocate_moe_swizzle_core_address_abi(
            self.infer_case.forward.n4.graph,
            projection,
        )
        operand = build_moe_swizzle_operand_abi(projection)
        self.assertEqual(len(abi.task_bindings), 68)
        self.assertEqual((len(operand.matmuls), len(operand.dtes)), (24, 24))
        task_by_id = {item.id: item for item in projection.tasks}
        binding = {item.task_ref: item for item in abi.task_bindings}
        for rank in range(4):
            comp_cores = {
                binding[item.id].logical_core.local_core_id
                for item in projection.tasks
                if item.rank == rank and item.kind is SwizzleActionKind.COMP
            }
            self.assertGreaterEqual(len(comp_cores), 2)
        validate_moe_swizzle_ir2_against_overlay(projection, self.overlay)

    def test_moe_standard_state_and_opcode_boundary_is_producer_scoped(self) -> None:
        state = StateABI.create(
            state_ref="moe.test.parameter",
            hbm_binding_ref="moe.test.parameter.hbm",
            kind=StateKind.PARAMETER,
            lifetime=PersistentStateLifetime.PERSISTENT,
            access=PersistentStateAccess.READ_ONLY,
            shape=(1,), dtype=DType.FP16, layout="flat_fp16",
            die_id=0, address=0, size_bytes=2, alignment_bytes=1,
        )
        hbm = ProgramSymbol("moe.test.hbm.symbol", ProgramSymbolKind.ABSOLUTE_ADDRESS, state.hbm_binding_ref)
        destination = ProgramSymbol("moe.test.destination.symbol", ProgramSymbolKind.ABSOLUTE_ADDRESS, "moe.test.destination")
        record = RelocatableRecord("moe.test.dma", RecordOpcode.LSU_LOAD, (
            RecordOperand.address("hbm_address", SemanticOperandId.HBM_ADDRESS, hbm.id),
            RecordOperand.literal("size_bytes", state.size_bytes),
            RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, destination.id),
        ))
        stream = CoreFragmentStream(
            LogicalCoreRef(0, 0), (record,), (), tuple(sorted((
                AddressRelocation(0, SemanticOperandId.HBM_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, hbm.id, 0),
                AddressRelocation(0, SemanticOperandId.DESTINATION_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, destination.id, 0),
            ), key=lambda item: int(item.operand_id))),
        )
        with_state = CommandFragment.create(
            producer_pass="moe_swizzle_standard_lowering",
            source_global_dag_id="moe.test.workload", kind=FragmentKind.MOE_SWIZZLE,
            claimed_action_ids=(record.source_global_action_id,), core_streams=(stream,),
            runtime_symbols=(), program_symbols=tuple(sorted((hbm, destination), key=lambda item: item.id)),
            buffer_abi=(), state_abi=(state,),
        )
        with_state.validate()
        with self.assertRaisesRegex(SchemaError, "only STATE_IO or the exact MOE_SWIZZLE producer"):
            replace(with_state, producer_pass="foreign_moe_lowering").validate()
        with self.assertRaisesRegex(SchemaError, "only STATE_IO or the exact MOE_SWIZZLE producer"):
            replace(
                with_state, kind=FragmentKind.SWIZZLE,
                producer_pass="swizzle_standard_lowering",
            ).validate()
        bad_record = replace(record, opcode=RecordOpcode.LSU_STORE)
        tampered = replace(with_state, core_streams=(replace(stream, records=(bad_record,)),))
        with self.assertRaises(SchemaError):
            tampered.validate()

    def test_projection_cannot_materialize_preserved_action(self) -> None:
        projection = project_moe_swizzle_ir2(
            self.overlay,
            self.infer_case.forward,
            endpoint_session_capacity=4,
        )
        tasks = list(projection.tasks)
        tasks[0] = replace(
            tasks[0],
            original_action_refs=(self.overlay.preserved_action_refs[0],),
        )
        tampered = replace(projection, tasks=tuple(tasks))
        with self.assertRaises(SchemaError):
            validate_moe_swizzle_ir2_against_overlay(tampered, self.overlay)

        overlap = replace(
            self.overlay,
            preserved_action_refs=(
                self.overlay.preserved_action_refs
                + (self.overlay.replaced_action_refs[0],)
            ),
        )
        with self.assertRaises(SchemaError):
            overlap.validate()
        missing = replace(
            self.overlay,
            linked_actions=self.overlay.linked_actions[:-1],
        )
        with self.assertRaises(SchemaError):
            missing.validate()

    def test_overlay_is_deterministic(self) -> None:
        rebuilt = build_moe_swizzle_overlay(
            self.infer_case.forward,
            self.decisions,
            source_execution=self.execution,
        )
        self.assertEqual(self.overlay, rebuilt)


if __name__ == "__main__":
    unittest.main()
