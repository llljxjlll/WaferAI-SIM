from __future__ import annotations

from collections import Counter
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.lite_moe_backward import (
    lower_lite_moe_backward,
    validate_lite_moe_backward_fragments,
)
from llm.frontend.wafer_frontend.passes.lite_moe_backward import (
    build_lite_moe_backward_overlay,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    FragmentKind,
    ProgramSymbolKind,
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.test.frontend.integration.lite_moe_cases import (
    build_lite_moe_execution_case,
)


class LiteMoeBackwardLoweringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.forward = build_lite_moe_execution_case()
        cls.overlay = build_lite_moe_backward_overlay(
            cls.forward.n4,
            cls.forward.projection,
            cls.forward.schedule,
            cls.forward.global_dag,
            cls.forward.n6_intent,
            cls.forward.source.moe_spec.trace,
        )
        cls.fragments = lower_lite_moe_backward(
            cls.overlay,
            cls.forward.n4,
            cls.forward.projection,
            cls.forward.schedule,
            cls.forward.global_dag,
            cls.forward.n6_intent,
            cls.forward.source.moe_spec.trace,
        )

    def _validate(self, fragments: tuple[CommandFragment, ...]) -> None:
        validate_lite_moe_backward_fragments(
            fragments,
            self.overlay,
            self.forward.n4,
            self.forward.projection,
            self.forward.schedule,
            self.forward.global_dag,
            self.forward.n6_intent,
            self.forward.source.moe_spec.trace,
        )

    @staticmethod
    def _records(fragment: CommandFragment):
        return fragment.core_streams[0].records

    def test_exact_leaf_record_relocation_and_abi_counts(self) -> None:
        records = tuple(
            record
            for fragment in self.fragments
            for stream in fragment.core_streams
            for record in stream.records
        )
        self.assertEqual(len(self.fragments), 28)
        self.assertEqual(len(records), 104)
        self.assertEqual(
            Counter(fragment.kind for fragment in self.fragments),
            Counter({
                FragmentKind.COARSE: 12,
                FragmentKind.MOE_TRANSFER: 8,
                FragmentKind.STATE_IO: 8,
            }),
        )
        self.assertEqual(
            Counter(record.opcode for record in records),
            Counter({
                RecordOpcode.SRAM_ALLOC_AT: 28,
                RecordOpcode.SRAM_FREE: 28,
                RecordOpcode.SRAM_BIND: 12,
                RecordOpcode.MATMUL: 8,
                RecordOpcode.DTE_SEND: 4,
                RecordOpcode.DTE_RECV: 4,
                RecordOpcode.DTE_WAIT: 4,
                RecordOpcode.LOCAL_REDUCE: 4,
                RecordOpcode.LSU_LOAD: 4,
                RecordOpcode.SGD_UPDATE: 4,
                RecordOpcode.LSU_STORE: 4,
            }),
        )
        self.assertEqual(
            sum(
                len(stream.address_relocations)
                for fragment in self.fragments
                for stream in fragment.core_streams
            ),
            188,
        )
        self.assertEqual(
            sum(
                len(stream.runtime_relocations)
                for fragment in self.fragments
                for stream in fragment.core_streams
            ),
            24,
        )
        self.assertEqual(sum(len(item.buffer_abi) for item in self.fragments), 68)
        self.assertEqual(sum(len(item.state_abi) for item in self.fragments), 8)
        unique_abis = {
            item.id: item
            for fragment in self.fragments
            for item in fragment.buffer_abi
        }
        borrowed = tuple(
            item
            for item in unique_abis.values()
            if item.ownership is BufferOwnership.BORROWED
        )
        self.assertEqual(
            Counter(item.size_bytes for item in borrowed),
            Counter({64: 8, 32: 8}),
        )
        self.assertEqual(
            sum(
                item.ownership is BufferOwnership.OWNED
                and item.size_bytes == 32
                and item.value_id.endswith(".received")
                for item in unique_abis.values()
            ),
            4,
        )
        label_storages = {
            symbol.source_ref
            for fragment in self.fragments
            for symbol in fragment.program_symbols
            if symbol.kind is ProgramSymbolKind.SRAM_LABEL
        }
        for abi in borrowed:
            self.assertIn(abi.storage_id, label_storages)
            users = tuple(
                fragment
                for fragment in self.fragments
                if any(item.id == abi.id for item in fragment.buffer_abi)
                and any(
                    record.opcode in (RecordOpcode.DTE_SEND, RecordOpcode.MATMUL)
                    for record in self._records(fragment)
                )
            )
            self.assertEqual(len(users), 1)
            opcodes = tuple(record.opcode for record in self._records(users[0]))
            read_index = next(
                index
                for index, opcode in enumerate(opcodes)
                if opcode in (RecordOpcode.DTE_SEND, RecordOpcode.MATMUL)
            )
            self.assertLess(opcodes.index(RecordOpcode.SRAM_ALLOC_AT), read_index)
            self.assertGreater(
                len(opcodes) - 1 - opcodes[::-1].index(RecordOpcode.SRAM_FREE),
                read_index,
            )
        self._validate(self.fragments)

    def test_wgrad_reduce_and_sgd_exact_abi(self) -> None:
        wgrads = tuple(
            fragment
            for fragment in self.fragments
            if any(record.opcode is RecordOpcode.MATMUL for record in self._records(fragment))
        )
        self.assertEqual(len(wgrads), 8)
        for fragment in wgrads:
            records = self._records(fragment)
            matmul = next(record for record in records if record.opcode is RecordOpcode.MATMUL)
            literals = {
                operand.name: operand.literal_value
                for operand in matmul.operands
                if operand.literal_value is not None
            }
            self.assertEqual(literals, {"datatype": 1, "parameters": (1, 32, 1, 16)})
            abis = {item.value_id: item for item in fragment.buffer_abi}
            self.assertEqual(
                Counter((item.dtype, item.size_bytes) for item in abis.values()),
                Counter({
                    (DType.FP16, 64): 1,
                    (DType.FP16, 32): 1,
                    (DType.FP32, 2048): 1,
                    (DType.FP32, 4096): 1,
                }),
            )
            activation = next(item for item in abis.values() if item.size_bytes == 64)
            gradient = next(item for item in abis.values() if item.size_bytes == 32)
            self.assertIs(activation.ownership, BufferOwnership.BORROWED)
            self.assertIn(
                gradient.ownership,
                (BufferOwnership.BORROWED, BufferOwnership.OWNED),
            )
            self.assertEqual(
                sum(record.opcode is RecordOpcode.SRAM_FREE for record in records),
                2,
            )

        reduces = tuple(
            self._records(fragment)[0]
            for fragment in self.fragments
            if len(self._records(fragment)) == 1
            and self._records(fragment)[0].opcode is RecordOpcode.LOCAL_REDUCE
        )
        self.assertEqual(len(reduces), 4)
        for record in reduces:
            literals = tuple(
                operand.literal_value
                for operand in record.operands
                if operand.literal_value is not None
            )
            self.assertEqual(literals, (1, 1, 1, 1, 0, 0, 2, 512, 2048))

        sgds = tuple(
            fragment
            for fragment in self.fragments
            if any(record.opcode is RecordOpcode.SGD_UPDATE for record in self._records(fragment))
        )
        self.assertEqual(len(sgds), 4)
        for fragment in sgds:
            records = self._records(fragment)
            self.assertEqual(
                tuple(record.opcode for record in records),
                (
                    RecordOpcode.SRAM_BIND,
                    RecordOpcode.SGD_UPDATE,
                    RecordOpcode.LSU_STORE,
                    RecordOpcode.SRAM_FREE,
                    RecordOpcode.SRAM_FREE,
                ),
            )
            state = fragment.state_abi[0]
            self.assertEqual(state.kind.value, "trainable_parameter")
            self.assertEqual(state.access.value, "read_write")
            weight = next(
                item
                for item in fragment.buffer_abi
                if item.ownership is BufferOwnership.OWNED and item.size_bytes == 1024
            )
            updated = next(
                item
                for item in fragment.buffer_abi
                if item.ownership is BufferOwnership.ALIASED and item.size_bytes == 1024
            )
            self.assertEqual(updated.alias_of, weight.binding_id)
            self.assertEqual(
                (
                    updated.logical_core,
                    updated.region_ref,
                    updated.region_offset_bytes,
                    updated.size_bytes,
                    updated.storage_id,
                    updated.dtype,
                    updated.layout,
                ),
                (
                    weight.logical_core,
                    weight.region_ref,
                    weight.region_offset_bytes,
                    weight.size_bytes,
                    weight.storage_id,
                    weight.dtype,
                    weight.layout,
                ),
            )

    def test_determinism_and_restable_fragment_tamper_fail_closed(self) -> None:
        rebuilt = lower_lite_moe_backward(
            self.overlay,
            self.forward.n4,
            self.forward.projection,
            self.forward.schedule,
            self.forward.global_dag,
            self.forward.n6_intent,
            self.forward.source.moe_spec.trace,
        )
        self.assertEqual(rebuilt, self.fragments)

        original = self.fragments[0]
        semantic = original._semantic_key()
        semantic["source_global_dag_id"] = f"{original.source_global_dag_id}.forged"
        forged = CommandFragment.create(
            producer_pass=original.producer_pass,
            **semantic,
        )
        forged.validate("forged")
        tampered = tuple(
            forged if fragment is original else fragment
            for fragment in self.fragments
        )
        with self.assertRaisesRegex(SchemaError, "exact typed lowering quotient"):
            self._validate(tampered)

        with self.assertRaisesRegex(SchemaError, "must be a tuple"):
            self._validate(list(self.fragments))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
