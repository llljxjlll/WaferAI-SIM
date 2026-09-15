"""Test native CE-backward allocation against an actually linked Dense forward."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.full_training_timeline_linker import (
    locate_forward_cross_entropy_tape,
)
from llm.frontend.wafer_frontend.passes.dense_training_ce_phase_carrier import (
    build_dense_training_ce_phase_carrier,
)
from llm.frontend.wafer_frontend.schema.action import ComputeOperand
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    ProgramSymbol, ProgramSymbolDefinition, ProgramSymbolKind, RecordOpcode,
    _lifecycle_payload_indices,
)
from llm.frontend.wafer_frontend.schema.global_action import GlobalAction
from llm.frontend.wafer_frontend.schema.ir0 import CrossEntropyBackwardWorkload, OpKind
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferAccess, BufferOwnership, BufferUseRole,
)
from llm.test.frontend.unit.test_full_training_timeline_linker import (
    FullTrainingTimelineLinkerTest,
)


class DenseTrainingCePhaseCarrierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        FullTrainingTimelineLinkerTest.setUpClass()
        source = FullTrainingTimelineLinkerTest
        cls.manifest = source.forward.linked_forward.manifest
        cls.tape = locate_forward_cross_entropy_tape(cls.manifest, action_id=source.ce_action)
        forward_action = next(action for action in
                              source.forward.linked_forward.source.replicas[0]
                              .lowering_context.global_dag.actions
                              if action.id == source.ce_action)
        workload = forward_action.compute.workload
        backward_workload = CrossEntropyBackwardWorkload(
            profile=workload.profile,
            reduction=workload.reduction,
            logical_logits_shape=workload.logical_logits_shape,
            rank_logits_shape=workload.rank_logits_shape,
            logical_label_shape=workload.logical_label_shape,
            rank_label_shape=workload.rank_label_shape,
            logical_loss_gradient_shape=workload.logical_loss_shape,
            rank_loss_gradient_shape=workload.rank_loss_shape,
            logical_logits_gradient_shape=workload.logical_logits_shape,
            rank_logits_gradient_shape=workload.rank_logits_shape,
            logits_dtype=workload.logits_dtype,
            label_dtype=workload.label_dtype,
            loss_gradient_dtype=workload.loss_dtype,
            logits_gradient_dtype=workload.logits_dtype,
        )
        cls.grad = replace(
            cls.tape.logits,
            id="actual_dense_ce_grad_buffer",
            binding_id="actual_dense_ce_grad_binding",
            storage_id="actual_dense_ce_grad_storage",
            value_id="T0.logits_gradient",
            tensor_slice=replace(cls.tape.logits.tensor_slice, value_id="T0.logits_gradient"),
            region_offset_bytes=3584,
            lifetime_start=41,
            lifetime_end_exclusive=42,
        )
        source_ref = replace(forward_action.source,
                             task_id=f"{forward_action.source.task_id}.native_backward")
        compute = replace(
            forward_action.compute,
            op_kind=OpKind.CE_BACKWARD,
            workload=backward_workload,
            impl_ref="cross_entropy_backward",
            inputs=(ComputeOperand(cls.tape.logits.value_id, "logits"),
                    ComputeOperand(cls.tape.labels.value_id, "labels"),
                    ComputeOperand(cls.tape.per_row_loss.value_id, "loss_gradient")),
            outputs=(ComputeOperand(cls.grad.value_id, "logits_gradient"),),
        )
        cls.backward = replace(
            forward_action,
            id=GlobalAction.stable_id_for_source(source_ref),
            source=source_ref,
            origin_ref=replace(forward_action.origin_ref,
                               op_id="T0.cross_entropy_backward__dp0"),
            op_kind=OpKind.CE_BACKWARD,
            region_id="region.ordinary.T0.cross_entropy_backward__dp0.rank.0",
            member_id="T0.cross_entropy_backward__dp0",
            read_values=(cls.tape.logits.value_id, cls.tape.labels.value_id,
                         cls.tape.per_row_loss.value_id),
            write_values=(cls.grad.value_id,),
            compute=compute,
            core_order_index=41,
            deps=(forward_action.id,),
            buffer_uses=(forward_action.buffer_uses[0], forward_action.buffer_uses[1],
                         replace(forward_action.buffer_uses[2],
                                 access=BufferAccess.READ,
                                 role=BufferUseRole.COMP_INPUT,
                                 operand_index=2),
                         replace(forward_action.buffer_uses[0],
                                 binding_id=cls.grad.binding_id,
                                 tensor_slice=cls.grad.tensor_slice,
                                 access=BufferAccess.WRITE,
                                 role=BufferUseRole.COMP_OUTPUT,
                                 operand_index=0)),
        )
        cls.backward.validate("real_ce_backward_action")
        region = next(definition for definition in cls.manifest.program_symbol_definitions
                      if definition.symbol.kind is ProgramSymbolKind.SRAM_REGION
                      and definition.symbol.source_ref == cls.grad.region_ref)
        core = cls.tape.logical_core
        address = ProgramSymbol("actual_dense_ce_grad_address", ProgramSymbolKind.ABSOLUTE_ADDRESS,
                                cls.grad.binding_id)
        label = ProgramSymbol("actual_dense_ce_grad_label", ProgramSymbolKind.SRAM_LABEL,
                              cls.grad.storage_id)
        cls.region = region
        cls.address = ProgramSymbolDefinition(address, "dense.ce.grad.absolute", 3584, 128, (core,))
        cls.label = ProgramSymbolDefinition(label, "dense.ce.grad.label", 0, 0, (core,))

    def carrier(self, *, backward=None, gradient=None, address=None, label=None, region=None):
        return build_dense_training_ce_phase_carrier(
            self.manifest, self.tape,
            backward_action=self.backward if backward is None else backward,
            gradient=self.grad if gradient is None else gradient,
            gradient_address=self.address if address is None else address,
            gradient_label=self.label if label is None else label,
            region=self.region if region is None else region,
            source_global_dag_id=self.manifest.source_global_dag_id,
        )

    def test_real_native_backward_has_exact_alloc_compute_four_frees(self) -> None:
        carrier = self.carrier()
        stream = carrier.fragment.core_streams[0]
        self.assertEqual(tuple(record.opcode for record in stream.records), (
            RecordOpcode.SRAM_ALLOC_AT, RecordOpcode.SRAM_BIND,
            RecordOpcode.CROSS_ENTROPY_BACKWARD,
            *(RecordOpcode.SRAM_FREE,) * 4,
        ))
        self.assertEqual(tuple(record.source_global_action_id for record in stream.records),
                         (self.backward.id,) * 7)
        self.assertEqual(stream.records[1].operands[0].literal_value, 3)
        self.assertEqual(carrier.fragment.source_global_dag_id,
                         self.manifest.source_global_dag_id)
        self.assertNotEqual(carrier.fragment.source_global_dag_id,
                            self.backward.source.dag_id)
        self.assertEqual(_lifecycle_payload_indices(
            self.backward, stream.records, list(range(len(stream.records))),
            {symbol.id: symbol for symbol in carrier.fragment.program_symbols},
            carrier.fragment.buffer_abi,
            lifecycle_required=True,
            path="real_dense_ce_backward_action",
        ), [1, 2])
        self.assertEqual(sum(item.symbol_ref is not None for item in
                             stream.records[1].operands[1:17]), 3)
        self.assertEqual(
            tuple(binding.buffer_abi_ids for binding in carrier.address_bindings
                  if binding.fragment_record_index == 2),
            tuple((abi.id,) for abi in (*carrier.extended_forward_buffers, self.grad)),
        )
        self.assertEqual(tuple(after for _, after in carrier.old_to_extended_abi_ids),
                         tuple(abi.id for abi in carrier.extended_forward_buffers))
        self.assertTrue(all(old != new for old, new in carrier.old_to_extended_abi_ids))
        self.assertEqual(tuple(old for old, _ in carrier.terminal_free_replacements),
                         self.tape.terminal_frees)
        self.assertEqual(tuple(new.fragment_record_index for _, new in
                               carrier.terminal_free_replacements), (6, 5, 4))
        self.assertTrue(all(new.source_global_action_id == self.backward.id
                            for _, new in carrier.terminal_free_replacements))
        self.assertEqual(self.tape.terminal_free_core_positions, (152, 151, 150))

    def test_live_forward_sram_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "overlaps forward physical SRAM"):
            self.carrier(gradient=replace(self.grad, region_offset_bytes=3456),
                         address=replace(self.address, value=3456))

    def test_too_small_physical_region_is_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "exceeds its physical SRAM region"):
            self.carrier(region=replace(self.region, size_bytes=3648))

    def test_stale_forward_dependency_is_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "depend on original CE forward"):
            self.carrier(backward=replace(self.backward, deps=()))

    def test_wrong_grad_address_declaration_is_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "declarations differ"):
            self.carrier(address=replace(self.address, value=3648))

    def test_borrowed_gradient_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "owned FP16"):
            self.carrier(gradient=replace(self.grad, ownership=BufferOwnership.BORROWED))

    def test_scheduled_dag_cannot_be_claimed_as_global_dag(self) -> None:
        with self.assertRaisesRegex(SchemaError, "not ScheduledDagRef"):
            build_dense_training_ce_phase_carrier(
                self.manifest, self.tape,
                backward_action=self.backward, gradient=self.grad,
                gradient_address=self.address, gradient_label=self.label,
                region=self.region,
                source_global_dag_id=self.backward.source.dag_id,
            )


if __name__ == "__main__":
    unittest.main()
