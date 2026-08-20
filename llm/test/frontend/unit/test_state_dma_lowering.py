from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import (
    LoweringContext,
    NaiveStateDmaLowering,
)
from llm.frontend.wafer_frontend.passes.global_action import (
    build_global_action_dag,
)
from llm.frontend.wafer_frontend.passes import (
    build_ir0,
    logical_expand,
    place_ir0,
)
from llm.frontend.wafer_frontend.policies.naive_fusion_partition import (
    NaiveFusionPartition,
)
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    DirectAllGatherPolicy,
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import (
    NaiveIntraDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    FragmentKind,
    ProgramSymbolKind,
    RecordOpcode,
    SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind, OpKind
from llm.frontend.wafer_frontend.schema.ir2 import SemanticTaskKind
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.placement import PlacementContext

from _fixtures import valid_hbm_address_spaces, valid_ir1
from test_stage3_decode_pipeline import _spec as _decode_spec


_LOWERING_SRAM_BYTES = 64 * 1024 * 1024


def _stateful_context(tp: int) -> LoweringContext:
    spec = _decode_spec(tp)
    logical = logical_expand(build_ir0(spec)).entries[0].graph
    fabric = valid_ir1().fabric
    placement_context = PlacementContext.create(
        producer_pass="state_dma_lowering",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    placed = place_ir0(logical, placement_context)
    placed_fields = placed._semantic_key()
    placed_fields["fused_op_skeletons"] = NaiveFusionPartition().run(placed)
    graph = type(placed).create(
        producer_pass="fusion_partition",
        **placed_fields,
    )
    fields = graph._semantic_key()
    fields["fabric"] = replace(
        graph.fabric,
        sram_profiles=tuple(
            replace(
                profile,
                capacity_bytes=_LOWERING_SRAM_BYTES,
                regions=tuple(
                    replace(region, size_bytes=_LOWERING_SRAM_BYTES)
                    for region in profile.regions
                ),
            )
            for profile in graph.fabric.sram_profiles
        ),
    )
    graph = type(graph).create(
        producer_pass=graph.producer_pass,
        **fields,
    )
    graph.validate()
    fusion_plans = tuple(
        NaiveInterDiePolicy().plan(graph, skeleton, graph.profile)
        for skeleton in graph.fused_op_skeletons
    )
    fused_members = {
        member_id
        for skeleton in graph.fused_op_skeletons
        for member_id in skeleton.member_node_ids
    }
    standalone_plans = tuple(
        DirectAllGatherPolicy().plan(graph, node, graph.profile)
        for node in graph.nodes
        if node.id not in fused_members
        and node.kind is OpKind.COLLECTIVE
        and getattr(node.workload, "collective", None)
        is CollectiveKind.ALL_GATHER
    )
    projection = NaiveProjectToIR2().run(
        graph,
        fusion_plans,
        standalone_plans,
        state_transfers=(),
    )
    schedule_set = NaiveIntraDiePolicy().schedule(projection, graph)
    global_dag = build_global_action_dag(graph, projection, schedule_set)
    context = LoweringContext(
        graph,
        fusion_plans,
        standalone_plans,
        projection,
        schedule_set,
        global_dag,
    )
    context.validate()
    return context


def _state_kind(context, action):
    manifest = context.ir1.persistent_state_manifest
    assert manifest is not None
    binding = next(
        binding
        for binding in manifest.bindings
        if binding.id == action.state_uses[0].hbm_binding_ref
    )
    declaration = next(
        declaration
        for declaration in manifest.declarations
        if declaration.id == binding.state_ref
    )
    return declaration.identity.kind


def _recreate(fragment: CommandFragment, **changes: object) -> CommandFragment:
    fields = fragment._semantic_key()
    fields.update(changes)
    return CommandFragment.create(producer_pass=fragment.producer_pass, **fields)


class NaiveStateDmaLoweringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parameter_context = _stateful_context(1)
        cls.context = _stateful_context(2)
        parameter_actions = tuple(
            action
            for action in cls.parameter_context.global_dag.actions
            if action.state_uses
        )
        state_actions = tuple(
            action for action in cls.context.global_dag.actions if action.state_uses
        )
        cls.parameter_load = next(
            action
            for action in parameter_actions
            if action.task_kind is SemanticTaskKind.DMA_IN
            and _state_kind(cls.parameter_context, action) is StateKind.PARAMETER
        )
        cls.kv_load = next(
            action
            for action in state_actions
            if action.task_kind is SemanticTaskKind.DMA_IN
            and _state_kind(cls.context, action)
            in (StateKind.KV_KEY, StateKind.KV_VALUE)
        )
        cls.kv_store = next(
            action
            for action in state_actions
            if action.task_kind is SemanticTaskKind.DMA_OUT
            and _state_kind(cls.context, action)
            in (StateKind.KV_KEY, StateKind.KV_VALUE)
        )

    def test_parameter_and_kv_lower_to_exact_blocking_lsu(self) -> None:
        lowerer = NaiveStateDmaLowering()
        for context, action, opcode, local_operand, hbm_addend in (
            (
                self.parameter_context,
                self.parameter_load,
                RecordOpcode.LSU_LOAD,
                SemanticOperandId.DESTINATION_ADDRESS,
                0,
            ),
            (
                self.context,
                self.kv_load,
                RecordOpcode.LSU_LOAD,
                SemanticOperandId.DESTINATION_ADDRESS,
                0,
            ),
            (
                self.context,
                self.kv_store,
                RecordOpcode.LSU_STORE,
                SemanticOperandId.SOURCE_ADDRESS,
                2176,
            ),
        ):
            with self.subTest(action=action.id):
                fragment = lowerer.lower(action, context)
                self.assertEqual(fragment, lowerer.lower(action, context))
                self.assertEqual(fragment.kind, FragmentKind.STATE_IO)
                self.assertEqual(fragment.claimed_action_ids, (action.id,))
                self.assertEqual(len(fragment.core_streams), 1)
                record = fragment.core_streams[0].records[0]
                self.assertEqual(record.opcode, opcode)
                self.assertEqual(record.operands[1].literal_value, action.bytes)
                relocations = fragment.core_streams[0].address_relocations
                self.assertEqual(
                    tuple(relocation.operand_id for relocation in relocations),
                    (local_operand, SemanticOperandId.HBM_ADDRESS),
                )
                self.assertEqual(relocations[1].addend, hbm_addend)
                self.assertEqual(len(fragment.buffer_abi), 1)
                self.assertEqual(len(fragment.state_abi), 1)
                self.assertEqual(
                    fragment.state_abi[0].hbm_binding_ref,
                    action.state_uses[0].hbm_binding_ref,
                )
                hbm_symbol = next(
                    symbol
                    for symbol in fragment.program_symbols
                    if symbol.id == relocations[1].symbol_ref
                )
                self.assertEqual(
                    (hbm_symbol.kind, hbm_symbol.source_ref),
                    (
                        ProgramSymbolKind.ABSOLUTE_ADDRESS,
                        action.state_uses[0].hbm_binding_ref,
                    ),
                )
                fragment.validate_against(context.global_dag)

    def test_direction_bytes_and_hbm_ref_tampering_fail_closed(self) -> None:
        fragment = NaiveStateDmaLowering().lower(
            self.parameter_load,
            self.parameter_context,
        )
        stream = fragment.core_streams[0]
        record = stream.records[0]
        with self.subTest("direction"):
            tampered_record = replace(record, opcode=RecordOpcode.LSU_STORE)
            tampered = _recreate(
                fragment,
                core_streams=(replace(stream, records=(tampered_record,)),),
            )
            with self.assertRaises(SchemaError):
                tampered.validate_against(self.parameter_context.global_dag)

        with self.subTest("bytes"):
            operands = list(record.operands)
            operands[1] = replace(
                operands[1], literal_value=self.parameter_load.bytes + 2
            )
            tampered = _recreate(
                fragment,
                core_streams=(
                    replace(
                        stream,
                        records=(replace(record, operands=tuple(operands)),),
                    ),
                ),
            )
            with self.assertRaisesRegex(SchemaError, "byte range"):
                tampered.validate_against(self.parameter_context.global_dag)

        with self.subTest("hbm_ref"):
            hbm_relocation = next(
                relocation
                for relocation in stream.address_relocations
                if relocation.operand_id is SemanticOperandId.HBM_ADDRESS
            )
            symbols = tuple(
                replace(symbol, source_ref="forged-hbm-binding")
                if symbol.id == hbm_relocation.symbol_ref
                else symbol
                for symbol in fragment.program_symbols
            )
            tampered = _recreate(fragment, program_symbols=symbols)
            with self.assertRaisesRegex(SchemaError, "matching StateABI"):
                tampered.validate_against(self.parameter_context.global_dag)

        with self.subTest("hbm_addend"):
            relocations = tuple(
                replace(relocation, addend=2)
                if relocation.operand_id is SemanticOperandId.HBM_ADDRESS
                else relocation
                for relocation in stream.address_relocations
            )
            tampered = _recreate(
                fragment,
                core_streams=(
                    replace(stream, address_relocations=relocations),
                ),
            )
            with self.assertRaisesRegex(SchemaError, "byte range"):
                tampered.validate_against(self.parameter_context.global_dag)


if __name__ == "__main__":
    unittest.main()
