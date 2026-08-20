from __future__ import annotations

import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import (
    NaiveCoarseLowering,
    NaiveStateDmaLowering,
    add_fixed_sram_lifecycle,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    CoreFragmentStream,
    RecordOpcode,
)

from test_linked_program_manifest_schema import valid_exact_linked_manifest
from test_state_dma_lowering import _stateful_context


def _case():
    context, _manifest = valid_exact_linked_manifest()
    action = context.global_dag.actions[0]
    leaf = NaiveCoarseLowering().lower(action, context)
    return context, action, leaf, add_fixed_sram_lifecycle(leaf, context)


class FixedSramLifecycleLoweringTest(unittest.TestCase):
    def test_exact_offsets_storage_labels_order_and_determinism(self) -> None:
        context, action, leaf, result = _case()
        self.assertEqual(result, add_fixed_sram_lifecycle(leaf, context))
        self.assertEqual(len(leaf.core_streams[0].records), 2)
        records = result.core_streams[0].records
        self.assertEqual(
            tuple(record.opcode for record in records),
            (
                RecordOpcode.SRAM_ALLOC_AT,
                RecordOpcode.SRAM_ALLOC_AT,
                RecordOpcode.SRAM_ALLOC_AT,
                RecordOpcode.SRAM_BIND,
                RecordOpcode.MATMUL,
                RecordOpcode.SRAM_FREE,
                RecordOpcode.SRAM_FREE,
                RecordOpcode.SRAM_FREE,
            ),
        )
        expected = tuple(
            sorted(
                leaf.buffer_abi,
                key=lambda abi: (
                    abi.region_ref,
                    abi.region_offset_bytes,
                    abi.storage_id,
                    abi.id,
                ),
            )
        )
        self.assertEqual(
            tuple(
                (
                    record.operands[2].literal_value,
                    record.operands[3].literal_value,
                    record.operands[4].literal_value,
                )
                for record in records[:3]
            ),
            tuple(
                (abi.region_offset_bytes, abi.size_bytes, abi.alignment_bytes)
                for abi in expected
            ),
        )
        symbols = {symbol.id: symbol for symbol in result.program_symbols}
        self.assertEqual(
            tuple(
                symbols[record.operands[1].symbol_ref].source_ref
                for record in records[:3]
            ),
            tuple(abi.storage_id for abi in expected),
        )
        result.validate_against(context.global_dag)
        self.assertEqual(action.id, records[0].source_global_action_id)

    def test_state_abi_and_hbm_relocation_survive_lifecycle_decoration(self) -> None:
        context = _stateful_context(1)
        action = next(
            action for action in context.global_dag.actions if action.state_uses
        )
        leaf = NaiveStateDmaLowering().lower(action, context)
        result = add_fixed_sram_lifecycle(leaf, context)

        self.assertEqual(result.state_abi, leaf.state_abi)
        self.assertEqual(
            tuple(
                record.opcode
                for record in result.core_streams[0].records
                if record.opcode
                in (RecordOpcode.LSU_LOAD, RecordOpcode.LSU_STORE)
            ),
            (leaf.core_streams[0].records[0].opcode,),
        )
        hbm = next(
            relocation
            for relocation in result.core_streams[0].address_relocations
            if relocation.operand_id.name == "HBM_ADDRESS"
        )
        self.assertEqual(hbm.addend, 0)
        self.assertEqual(hbm.record_index, 1)
        self.assertEqual(
            result.core_streams[0].records[hbm.record_index].opcode,
            leaf.core_streams[0].records[0].opcode,
        )
        result.validate_against(context.global_dag)

    def test_rejects_tampered_offset_missing_free_and_double_decoration(self) -> None:
        context, _action, _leaf, result = _case()
        stream = result.core_streams[0]
        records = list(stream.records)
        alloc = records[0]
        operands = list(alloc.operands)
        operands[2] = replace(
            operands[2], literal_value=operands[2].literal_value + 1
        )
        records[0] = replace(alloc, operands=tuple(operands))
        tampered_stream = CoreFragmentStream(
            stream.logical_core,
            tuple(records),
            stream.runtime_relocations,
            stream.address_relocations,
        )
        tampered = CommandFragment.create(
            producer_pass=result.producer_pass,
            **{
                **result._semantic_key(),
                "core_streams": (tampered_stream,),
            },
        )
        with self.assertRaisesRegex(SchemaError, "exactly preserve BufferABI"):
            tampered.validate_against(context.global_dag)

        missing_records = stream.records[:-1]
        missing_relocations = tuple(
            relocation
            for relocation in stream.address_relocations
            if relocation.record_index < len(missing_records)
        )
        missing_stream = CoreFragmentStream(
            stream.logical_core,
            missing_records,
            stream.runtime_relocations,
            missing_relocations,
        )
        missing = CommandFragment.create(
            producer_pass=result.producer_pass,
            **{
                **result._semantic_key(),
                "core_streams": (missing_stream,),
            },
        )
        with self.assertRaisesRegex(SchemaError, "first/last uses"):
            missing.validate_against(context.global_dag)
        with self.assertRaisesRegex(SchemaError, "already contains"):
            add_fixed_sram_lifecycle(result, context)


if __name__ == "__main__":
    unittest.main()
