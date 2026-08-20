from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import (
    NaiveCoarseLowering,
    NaiveIsaRegionLowering,
    NaiveManifestLinker,
    NaiveStateDmaLowering,
    NaiveStandaloneCollectiveLowering,
    add_fixed_sram_lifecycle,
)
from llm.frontend.wafer_frontend.passes import lower_profile
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    FragmentKind,
    LinkedProgramManifest,
    ProgramSymbolKind,
    RecordOpcode,
    RegionManifest,
    RuntimeSymbolKind,
    SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    FusedNodeOrigin,
    OrdinaryNodeOrigin,
    SemanticTaskKind,
    StateIoOrigin,
    StandaloneNodeOrigin,
)

from test_linked_program_manifest_schema import valid_exact_linked_manifest
from test_isa_region_lowering import _context as _isa_context
from test_n5_pipeline import _compile_through_n5


def _ordinary_case():
    context, _fixture_manifest = valid_exact_linked_manifest()
    lowerer = NaiveCoarseLowering()
    fragments = tuple(
        lowerer.lower(action, context) for action in context.global_dag.actions
    )
    return context, fragments


def _recreate_fragment(
    fragment: CommandFragment,
    **changes: object,
) -> CommandFragment:
    fields = fragment._semantic_key()
    fields.update(changes)
    return CommandFragment.create(
        producer_pass=fragment.producer_pass,
        **fields,
    )


def _recreate_linked(
    manifest: LinkedProgramManifest,
    **changes: object,
) -> LinkedProgramManifest:
    fields = manifest._semantic_key()
    fields.update(changes)
    return LinkedProgramManifest.create(
        producer_pass=manifest.producer_pass,
        **fields,
    )


def _validate_linked(manifest: LinkedProgramManifest, context) -> None:
    manifest.validate_against(
        context.ir1,
        context.fusion_plans,
        context.standalone_plans,
        context.projection,
        context.schedule_set,
        context.global_dag,
        manifest.fragments,
    )


class NaiveManifestLinkerTest(unittest.TestCase):
    def test_links_all_ordinary_records_into_one_exact_manifest(self) -> None:
        context, fragments = _ordinary_case()
        linker = NaiveManifestLinker()
        manifest = linker.link(context, fragments)
        manifest.validate_against(
            context.ir1,
            context.fusion_plans,
            context.standalone_plans,
            context.projection,
            context.schedule_set,
            context.global_dag,
            manifest.fragments,
        )
        self.assertEqual(manifest, linker.link(context, tuple(reversed(fragments))))
        self.assertEqual(len(manifest.fragments), 2)
        self.assertEqual(len(manifest.core_streams), 2)
        self.assertEqual(sum(len(stream.records) for stream in manifest.core_streams), 4)
        self.assertEqual(len(manifest.address_operand_bindings), 10)
        self.assertEqual(len(manifest.program_symbol_definitions), 10)
        self.assertEqual(
            tuple(
                record.opcode
                for fragment in manifest.fragments
                for stream in fragment.core_streams
                for record in stream.records
            ).count(RecordOpcode.MATMUL),
            2,
        )
        self.assertEqual(
            manifest.envelope.active_cores,
            manifest.envelope.expected_done_cores,
        )

    def test_links_exact_fixed_sram_lifecycle(self) -> None:
        context, fragments = _ordinary_case()
        fragments = tuple(
            add_fixed_sram_lifecycle(fragment, context)
            for fragment in fragments
        )
        linker = NaiveManifestLinker()
        manifest = linker.link(context, fragments)
        manifest.validate_against(
            context.ir1,
            context.fusion_plans,
            context.standalone_plans,
            context.projection,
            context.schedule_set,
            context.global_dag,
            manifest.fragments,
        )
        opcodes = tuple(
            record.opcode
            for fragment in manifest.fragments
            for stream in fragment.core_streams
            for record in stream.records
        )
        self.assertEqual(opcodes.count(RecordOpcode.SRAM_ALLOC_AT), 6)
        self.assertEqual(opcodes.count(RecordOpcode.SRAM_FREE), 6)
        self.assertEqual(opcodes.count(RecordOpcode.SRAM_BIND), 2)
        self.assertEqual(opcodes.count(RecordOpcode.MATMUL), 2)
        self.assertEqual(manifest, linker.link(context, tuple(reversed(fragments))))

    def test_links_fused_tp2_runtime_and_lifecycle(self) -> None:
        context, plan, actions = _isa_context(2)
        regions = tuple(
            RegionManifest.create(
                producer_pass=region.producer_pass,
                region_id=region.region_id,
                fusion_plan_id=region.fusion_plan_id,
                target_dies=region.target_dies,
                fragment=add_fixed_sram_lifecycle(region.fragment, context),
            )
            for region in NaiveIsaRegionLowering().lower(plan, actions, context)
        )
        linker = NaiveManifestLinker()
        manifest = linker.link(context, regions)
        self.assertEqual(manifest, linker.link(context, tuple(reversed(regions))))
        self.assertEqual(len(manifest.fragments), 2)
        self.assertEqual(
            Counter(
                definition.symbol.kind
                for definition in manifest.runtime_symbol_definitions
            ),
            Counter(
                {
                    RuntimeSymbolKind.RUNTIME_CORE: 4,
                    RuntimeSymbolKind.DTE_FSM: 2,
                    RuntimeSymbolKind.DTE_TOKEN: 2,
                    RuntimeSymbolKind.START_TAG: 2,
                }
            ),
        )
        self.assertEqual(
            Counter(
                definition.symbol.kind
                for definition in manifest.program_symbol_definitions
            ),
            Counter(
                {
                    ProgramSymbolKind.ABSOLUTE_ADDRESS: 16,
                    ProgramSymbolKind.SRAM_LABEL: 14,
                    ProgramSymbolKind.SRAM_REGION: 1,
                }
            ),
        )
        self.assertEqual(
            Counter(
                record.opcode
                for region in manifest.fragments
                if isinstance(region, RegionManifest)
                for stream in region.fragment.core_streams
                for record in stream.records
            ),
            Counter(
                {
                    RecordOpcode.SRAM_ALLOC_AT: 14,
                    RecordOpcode.SRAM_FREE: 14,
                    RecordOpcode.SRAM_BIND: 4,
                    RecordOpcode.MATMUL: 4,
                    RecordOpcode.DTE_SEND: 2,
                    RecordOpcode.DTE_RECV: 2,
                    RecordOpcode.DTE_WAIT: 2,
                    RecordOpcode.LOCAL_REDUCE: 2,
                }
            ),
        )

    def test_real_tp2_links_all_dense_compute_operand_roles(self) -> None:
        context = _compile_through_n5()[-1].entries[0].lowering_context()
        fragments = [
            add_fixed_sram_lifecycle(
                NaiveStateDmaLowering().lower(action, context),
                context,
            )
            for action in context.global_dag.actions
            if isinstance(action.origin_ref, StateIoOrigin)
        ]
        fragments.extend(
            [
                add_fixed_sram_lifecycle(
                    NaiveCoarseLowering().lower(action, context),
                    context,
                )
                for action in context.global_dag.actions
                if isinstance(action.origin_ref, OrdinaryNodeOrigin)
            ]
        )
        for plan in context.fusion_plans:
            actions = tuple(
                action
                for action in context.global_dag.actions
                if isinstance(action.origin_ref, FusedNodeOrigin)
                and action.origin_ref.plan_id == plan.id
                and action.task_kind is not SemanticTaskKind.TRANSIT
            )
            fragments.extend(
                RegionManifest.create(
                    producer_pass=region.producer_pass,
                    region_id=region.region_id,
                    fusion_plan_id=region.fusion_plan_id,
                    target_dies=region.target_dies,
                    fragment=add_fixed_sram_lifecycle(
                        region.fragment,
                        context,
                    ),
                )
                for region in NaiveIsaRegionLowering().lower(
                    plan,
                    actions,
                    context,
                )
            )
        for plan in context.standalone_plans:
            actions = tuple(
                action
                for action in context.global_dag.actions
                if isinstance(action.origin_ref, StandaloneNodeOrigin)
                and action.origin_ref.collective_plan_id == plan.id
            )
            fragments.append(
                add_fixed_sram_lifecycle(
                    NaiveStandaloneCollectiveLowering().lower(
                        actions,
                        context,
                    ),
                    context,
                )
            )

        manifest = NaiveManifestLinker().link(context, tuple(fragments))
        manifest.validate_against(
            context.ir1,
            context.fusion_plans,
            context.standalone_plans,
            context.projection,
            context.schedule_set,
            context.global_dag,
            manifest.fragments,
        )
        coarse_ids = {
            linked.id
            for linked in manifest.fragments
            if not isinstance(linked, RegionManifest)
            and linked.kind is FragmentKind.COARSE
        }
        leaves = {
            linked.id: linked
            for linked in manifest.fragments
            if not isinstance(linked, RegionManifest)
            and linked.kind is FragmentKind.COARSE
        }

        def _bound_record(binding):
            fragment = leaves[binding.fragment_id]
            stream = next(
                item
                for item in fragment.core_streams
                if item.logical_core == binding.logical_core
            )
            return stream.records[binding.fragment_record_index]

        coarse_bindings = tuple(
            binding
            for binding in manifest.address_operand_bindings
            if binding.fragment_id in coarse_ids
            and _bound_record(binding).opcode
            not in (RecordOpcode.SRAM_ALLOC_AT, RecordOpcode.SRAM_FREE)
        )
        self.assertEqual(len(coarse_bindings), 120)

        compute_operands = {}
        for binding in coarse_bindings:
            record = _bound_record(binding)
            if record.opcode is not RecordOpcode.SRAM_BIND:
                compute_operands.setdefault(record.opcode, set()).add(
                    binding.operand_id
                )
        self.assertEqual(
            compute_operands,
            {
                RecordOpcode.MATMUL: {
                    SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                    SemanticOperandId.COMPUTE_DATA_ADDRESS,
                    SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                },
                RecordOpcode.EMBEDDING_LOOKUP: {
                    SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                    SemanticOperandId.COMPUTE_DATA_ADDRESS,
                    SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                },
                RecordOpcode.ROPE_QK_EXACT: {
                    SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                    SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                },
                RecordOpcode.ATTENTION_EXACT: {
                    SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                    SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                },
                RecordOpcode.RMSNORM: {
                    SemanticOperandId.COMPUTE_DATA_ADDRESS,
                    SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                    SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                },
                RecordOpcode.SWIGLU: {
                    SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                    SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                },
                RecordOpcode.RESIDUAL: {
                    SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                    SemanticOperandId.COMPUTE_DATA_ADDRESS,
                    SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                },
            },
        )

    def test_real_tp2_lower_program_links_exact_state_operands(self) -> None:
        entry = _compile_through_n5()[-1].entries[0]
        context = entry.lowering_context()
        lowered = lower_profile(entry)
        linker = NaiveManifestLinker()
        manifest = linker.link(context, lowered.fragments)
        _validate_linked(manifest, context)
        self.assertEqual(
            manifest,
            linker.link(context, tuple(reversed(lowered.fragments))),
        )
        self.assertEqual(len(manifest.state_operand_bindings), 22)
        self.assertFalse(
            any(
                binding.operand_id is SemanticOperandId.HBM_ADDRESS
                for binding in manifest.address_operand_bindings
            )
        )
        leaves = {
            (
                linked.fragment.id
                if isinstance(linked, RegionManifest)
                else linked.id
            ): (
                linked.fragment
                if isinstance(linked, RegionManifest)
                else linked
            )
            for linked in manifest.fragments
        }
        state_records = tuple(
            leaves[binding.fragment_id]
            .core_streams[0]
            .records[binding.fragment_record_index]
            for binding in manifest.state_operand_bindings
        )
        self.assertEqual(
            sum(
                record.opcode is RecordOpcode.LSU_LOAD
                for record in state_records
            ),
            18,
        )
        self.assertEqual(
            sum(
                record.opcode is RecordOpcode.LSU_STORE
                for record in state_records
            ),
            4,
        )

    def test_state_linking_tampering_fails_closed(self) -> None:
        entry = _compile_through_n5()[-1].entries[0]
        context = entry.lowering_context()
        lowered = lower_profile(entry)
        fragments = lowered.fragments
        state_index = next(
            index
            for index, fragment in enumerate(fragments)
            if type(fragment) is CommandFragment
            and fragment.kind is FragmentKind.STATE_IO
        )
        state_fragment = fragments[state_index]
        assert type(state_fragment) is CommandFragment

        def rejected_fragment(replacement: CommandFragment) -> None:
            values = list(fragments)
            values[state_index] = replacement
            with self.assertRaises(SchemaError):
                NaiveManifestLinker().link(context, tuple(values))

        with self.subTest("missing_state_abi"):
            rejected_fragment(
                _recreate_fragment(state_fragment, state_abi=())
            )
        with self.subTest("double_state_abi"):
            rejected_fragment(
                _recreate_fragment(
                    state_fragment,
                    state_abi=state_fragment.state_abi * 2,
                )
            )
        stream = state_fragment.core_streams[0]
        hbm_relocation = next(
            relocation
            for relocation in stream.address_relocations
            if relocation.operand_id is SemanticOperandId.HBM_ADDRESS
        )
        with self.subTest("source"):
            symbols = tuple(
                replace(symbol, source_ref="forged-hbm-binding")
                if symbol.id == hbm_relocation.symbol_ref
                else symbol
                for symbol in state_fragment.program_symbols
            )
            rejected_fragment(
                _recreate_fragment(state_fragment, program_symbols=symbols)
            )
        with self.subTest("addend"):
            relocations = tuple(
                replace(relocation, addend=2)
                if relocation is hbm_relocation
                else relocation
                for relocation in stream.address_relocations
            )
            rejected_fragment(
                _recreate_fragment(
                    state_fragment,
                    core_streams=(
                        replace(stream, address_relocations=relocations),
                    ),
                )
            )
        with self.subTest("direction"):
            records = list(stream.records)
            source_record = records[hbm_relocation.record_index]
            records[hbm_relocation.record_index] = replace(
                source_record,
                opcode=(
                    RecordOpcode.LSU_STORE
                    if source_record.opcode is RecordOpcode.LSU_LOAD
                    else RecordOpcode.LSU_LOAD
                ),
            )
            rejected_fragment(
                _recreate_fragment(
                    state_fragment,
                    core_streams=(
                        replace(stream, records=tuple(records)),
                    ),
                )
            )

        manifest = NaiveManifestLinker().link(context, fragments)
        binding = manifest.state_operand_bindings[0]
        with self.subTest("missing_state_binding"):
            tampered = _recreate_linked(
                manifest,
                state_operand_bindings=manifest.state_operand_bindings[1:],
            )
            with self.assertRaisesRegex(SchemaError, "StateABI closure"):
                _validate_linked(tampered, context)
        with self.subTest("double_state_binding"):
            tampered = _recreate_linked(
                manifest,
                state_operand_bindings=(binding, *manifest.state_operand_bindings),
            )
            with self.assertRaisesRegex(SchemaError, "duplicate state operand"):
                _validate_linked(tampered, context)
        with self.subTest("wrong_state_abi"):
            bindings = (
                replace(binding, state_abi_id="forged-state-abi"),
                *manifest.state_operand_bindings[1:],
            )
            tampered = _recreate_linked(
                manifest,
                state_operand_bindings=bindings,
            )
            with self.assertRaisesRegex(SchemaError, "unknown StateABI"):
                _validate_linked(tampered, context)

        leaf = next(
            linked
            for linked in manifest.fragments
            if type(linked) is CommandFragment
            and linked.id == binding.fragment_id
        )
        abi = next(item for item in leaf.state_abi if item.id == binding.state_abi_id)
        definition_index = next(
            index
            for index, definition in enumerate(
                manifest.program_symbol_definitions
            )
            if definition.symbol.source_ref == abi.hbm_binding_ref
        )
        for field_name, value in (
            ("value", abi.address + abi.alignment_bytes),
            ("size_bytes", abi.size_bytes + abi.alignment_bytes),
        ):
            with self.subTest(field_name):
                definitions = list(manifest.program_symbol_definitions)
                definitions[definition_index] = replace(
                    definitions[definition_index],
                    **{field_name: value},
                )
                tampered = _recreate_linked(
                    manifest,
                    program_symbol_definitions=tuple(definitions),
                )
                with self.assertRaisesRegex(
                    SchemaError,
                    "exactly preserve StateABI",
                ):
                    _validate_linked(tampered, context)

    def test_rejects_missing_or_duplicate_action_coverage(self) -> None:
        context, fragments = _ordinary_case()
        linker = NaiveManifestLinker()
        with self.assertRaisesRegex(SchemaError, "cover every executable action"):
            linker.link(context, fragments[:1])
        with self.assertRaisesRegex(SchemaError, "duplicate fragment"):
            linker.link(context, (fragments[0], fragments[0]))

    def test_rejects_partial_lifecycle_coverage(self) -> None:
        context, fragments = _ordinary_case()
        mixed = (
            add_fixed_sram_lifecycle(fragments[0], context),
            fragments[1],
        )
        with self.assertRaisesRegex(SchemaError, "exactly cover every scheduled storage"):
            NaiveManifestLinker().link(context, mixed)

    def test_rejects_wrong_input_types(self) -> None:
        context, fragments = _ordinary_case()
        linker = NaiveManifestLinker()
        with self.assertRaisesRegex(SchemaError, "LoweringContext"):
            linker.link(object(), fragments)
        with self.assertRaisesRegex(SchemaError, "must contain"):
            linker.link(context, ())


if __name__ == "__main__":
    unittest.main()
