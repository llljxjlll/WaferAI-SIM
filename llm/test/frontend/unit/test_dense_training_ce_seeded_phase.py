"""Check independent dLoss seed and true native CE on a linked L2 forward."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.dense_training_ce_seeded_phase import (
    build_dense_training_ce_seeded_phase,
)
from llm.frontend.wafer_frontend.schema.action import ComputeOperand
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    ProgramSymbol, ProgramSymbolDefinition, ProgramSymbolKind, RecordOpcode,
    _lifecycle_payload_indices,
)
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.test.frontend.unit.test_dense_training_ce_phase_carrier import (
    DenseTrainingCePhaseCarrierTest as OldPhase,
)


class DenseSeededCePhaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        OldPhase.setUpClass()
        cls.forward = OldPhase.manifest
        cls.tape = OldPhase.tape
        cls.region = OldPhase.region
        cls.loss_gradient = replace(
            cls.tape.per_row_loss,
            id="dense_true_per_row_dloss_abi",
            binding_id="dense_true_per_row_dloss_binding",
            storage_id="dense_true_per_row_dloss_storage",
            value_id="T0.loss_gradient",
            tensor_slice=replace(cls.tape.per_row_loss.tensor_slice,
                                 value_id="T0.loss_gradient"),
            ownership=BufferOwnership.BORROWED,
            region_offset_bytes=3584,
            lifetime_start=41,
            lifetime_end_exclusive=42,
        )
        cls.logits_gradient = replace(
            OldPhase.grad, region_offset_bytes=3648,
        )
        cls.loss_address = ProgramSymbolDefinition(
            ProgramSymbol("dense_true_dloss_address", ProgramSymbolKind.ABSOLUTE_ADDRESS,
                          cls.loss_gradient.binding_id),
            "dense.loss_gradient.absolute", 3584, 16, (cls.tape.logical_core,),
        )
        cls.loss_label = ProgramSymbolDefinition(
            ProgramSymbol("dense_true_dloss_label", ProgramSymbolKind.SRAM_LABEL,
                          cls.loss_gradient.storage_id),
            "dense.loss_gradient.label", 0, 0, (cls.tape.logical_core,),
        )
        cls.grad_address = replace(OldPhase.address, value=3648)
        cls.grad_label = OldPhase.label
        previous = OldPhase.backward
        cls.backward = replace(
            previous,
            read_values=(cls.tape.logits.value_id, cls.tape.labels.value_id,
                         cls.loss_gradient.value_id),
            compute=replace(previous.compute,
                            inputs=(previous.compute.inputs[0],
                                    previous.compute.inputs[1],
                                    ComputeOperand(cls.loss_gradient.value_id, "loss_gradient"))),
            buffer_uses=(previous.buffer_uses[0], previous.buffer_uses[1],
                         replace(previous.buffer_uses[2],
                                 binding_id=cls.loss_gradient.binding_id,
                                 tensor_slice=cls.loss_gradient.tensor_slice),
                         replace(previous.buffer_uses[3],
                                 binding_id=cls.logits_gradient.binding_id,
                                 tensor_slice=cls.logits_gradient.tensor_slice)),
        )
        cls.backward.validate("genuine_ce_backward_action")

    def carrier(self, *, seed=None, seed_address=None, seed_label=None,
                output=None, output_address=None, backward=None):
        return build_dense_training_ce_seeded_phase(
            self.forward, self.tape,
            backward_action=self.backward if backward is None else backward,
            source_global_dag_id=self.forward.source_global_dag_id,
            loss_gradient=self.loss_gradient if seed is None else seed,
            loss_gradient_address=self.loss_address if seed_address is None else seed_address,
            loss_gradient_label=self.loss_label if seed_label is None else seed_label,
            logits_gradient=self.logits_gradient if output is None else output,
            logits_gradient_address=self.grad_address if output_address is None else output_address,
            logits_gradient_label=self.grad_label,
            region=self.region,
        )

    def test_real_l2_ce_forward_loss_is_not_used_as_dloss(self) -> None:
        source = self.carrier()
        stream = source.fragment.core_streams[0]
        self.assertEqual(tuple(record.opcode for record in stream.records), (
            RecordOpcode.SRAM_ALLOC_AT, RecordOpcode.SRAM_ALLOC_AT,
            RecordOpcode.SRAM_BIND, RecordOpcode.CROSS_ENTROPY_BACKWARD,
            *(RecordOpcode.SRAM_FREE,) * 4,
        ))
        self.assertEqual(source.retained_forward_loss_free, self.tape.terminal_frees[2])
        self.assertEqual([new.fragment_record_index for _, new in
                          source.terminal_free_replacements], [7, 6])
        self.assertEqual(len(source.old_to_extended_abi_ids), 2)
        self.assertEqual(stream.records[3].operands[8].symbol_ref,
                         self.loss_address.symbol.id)
        self.assertNotEqual(stream.records[3].operands[8].symbol_ref,
                            self.tape.original_operand_definitions[2].symbol.id)
        self.assertEqual(source.loss_gradient_seed, b"\x00\x00\x80\x3f" * 4)
        self.assertIs(source.loss_gradient.ownership, BufferOwnership.BORROWED)
        self.assertEqual(_lifecycle_payload_indices(
            self.backward, stream.records, list(range(8)),
            {symbol.id: symbol for symbol in source.fragment.program_symbols},
            source.fragment.buffer_abi,
            lifecycle_required=True, path="real_dense_ce_seeded_action",
        ), [2, 3])

    def test_sram_seed_must_not_overlap_forward_loss_write(self) -> None:
        with self.assertRaisesRegex(SchemaError, "overlaps physical forward SRAM"):
            self.carrier(seed=replace(self.loss_gradient, region_offset_bytes=3520),
                         seed_address=replace(self.loss_address, value=3520))

    def test_reusing_physical_forward_loss_symbol_is_rejected(self) -> None:
        reused = replace(self.loss_address,
                         symbol=self.tape.original_operand_definitions[2].symbol,
                         value=self.tape.original_operand_definitions[2].value)
        with self.assertRaises(SchemaError):
            self.carrier(seed_address=reused)

    def test_dloss_cannot_be_owned_output(self) -> None:
        with self.assertRaisesRegex(SchemaError, "BORROWED FP32"):
            self.carrier(seed=replace(self.loss_gradient,
                                      ownership=BufferOwnership.OWNED))

    def test_wrong_per_row_abi_bytes_are_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "BORROWED FP32"):
            self.carrier(seed=replace(self.loss_gradient, size_bytes=20),
                         seed_address=replace(self.loss_address, size_bytes=20))


if __name__ == "__main__":
    unittest.main()
