"""A physical CE-backward action carrier for an existing Dense forward tape.

The caller owns the new GlobalAction, schedule, and SRAM placement.  This
factory preserves the forward operand symbols while producing the actual
native 0x1F record, an output allocation, and four terminal frees.  Its result
is a CommandFragment for the final training timeline linker, not a standalone
LinkedProgramManifest: the existing forward allocations must be retained and
their three BufferABI lifetimes extended in that final manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..errors import SchemaError
from ..lowering.full_training_timeline_linker import ForwardCrossEntropyTape
from ..schema.artifact_manifest import (
    AddressOperandBinding, AddressRelocation, BufferABI, CommandFragment,
    CoreFragmentStream, FragmentKind, LinkedProgramManifest, LinkedRecordRef,
    ProgramSymbolDefinition, ProgramSymbolKind, RecordOpcode, RecordOperand,
    RelocatableRecord, SemanticOperandId,
)
from ..schema.global_action import GlobalAction
from ..schema.ir0 import OpKind
from ..schema.ir2 import BufferOwnership
from ..schema.common import stable_artifact_id
from .dense_training_ce_backward import build_dense_training_ce_backward_record


@dataclass(frozen=True, slots=True)
class DenseCeBackwardPhaseCarrier:
    fragment: CommandFragment
    program_definitions: tuple[ProgramSymbolDefinition, ...]
    address_bindings: tuple[AddressOperandBinding, ...]
    extended_forward_buffers: tuple[BufferABI, BufferABI, BufferABI]
    old_to_extended_abi_ids: tuple[tuple[str, str], ...]
    terminal_free_replacements: tuple[tuple[LinkedRecordRef, LinkedRecordRef], ...]
    gradient: BufferABI


def build_dense_training_ce_phase_carrier(
    forward: LinkedProgramManifest,
    tape: ForwardCrossEntropyTape,
    *,
    backward_action: GlobalAction,
    gradient: BufferABI,
    gradient_address: ProgramSymbolDefinition,
    gradient_label: ProgramSymbolDefinition,
    region: ProgramSymbolDefinition,
    source_global_dag_id: str,
) -> DenseCeBackwardPhaseCarrier:
    """Build a typed native phase, rejecting stale tape and overlapping SRAM.

    The four backward BufferABI closures reference one real scheduled action.
    The final timeline must also re-sign the original three forward BufferABI
    declarations/bindings, and remove their three earlier FREE record refs.
    """
    forward.validate("ce_phase_forward")
    backward_action.validate("ce_phase_backward_action")
    if not source_global_dag_id or source_global_dag_id == backward_action.source.dag_id:
        raise SchemaError("requires the separately signed production GlobalActionDAG ID, not ScheduledDagRef", path="source_global_dag_id")
    if backward_action.op_kind is not OpKind.CE_BACKWARD or backward_action.compute is None:
        raise SchemaError("requires a native CE_BACKWARD GlobalAction", path="backward_action")
    originals = (tape.logits, tape.labels, tape.per_row_loss)
    core = tape.logical_core
    if (backward_action.logical_core != core
            or backward_action.core_order_index is None
            or backward_action.source.schedule_id != originals[0].schedule_id):
        raise SchemaError("CE backward must share the original scheduled core", path="backward_action")
    if (gradient.logical_core != core
            or gradient.schedule_id != originals[0].schedule_id
            or gradient.dtype != originals[0].dtype
            or gradient.tensor_slice.shape != originals[0].tensor_slice.shape
            or gradient.size_bytes != originals[0].size_bytes
            or gradient.region_ref != originals[0].region_ref
            or gradient.ownership is not BufferOwnership.OWNED
            or gradient.alias_of is not None
            or gradient.lifetime_start != backward_action.core_order_index
            or gradient.lifetime_end_exclusive != backward_action.core_order_index + 1):
        raise SchemaError("gradient must have one owned FP16 action-local physical span", path="gradient")
    if (region.symbol.kind is not ProgramSymbolKind.SRAM_REGION
            or region.symbol.source_ref != gradient.region_ref
            or gradient.region_offset_bytes + gradient.size_bytes > region.size_bytes):
        raise SchemaError("CE gradient exceeds its physical SRAM region", path="gradient")
    if (gradient.region_offset_bytes % gradient.alignment_bytes
            or region.value % gradient.alignment_bytes):
        raise SchemaError("CE gradient address is misaligned", path="gradient")
    grad_lo = region.value + gradient.region_offset_bytes
    grad_hi = grad_lo + gradient.size_bytes
    original_ids = {abi.id for abi in originals}
    for abi in (b for fragment in forward.fragments for b in fragment.buffer_abi
                if b.logical_core == core and b.region_ref == gradient.region_ref
                and b.lifetime_start <= backward_action.core_order_index
                and (b.lifetime_end_exclusive > backward_action.core_order_index
                     or b.id in original_ids)):
        if region.value + abi.region_offset_bytes < grad_hi and grad_lo < region.value + abi.region_offset_bytes + abi.size_bytes:
            raise SchemaError("CE gradient overlaps forward physical SRAM", path="gradient")
    if (gradient_address.symbol.kind is not ProgramSymbolKind.ABSOLUTE_ADDRESS
            or gradient_address.symbol.source_ref != gradient.binding_id
            or gradient_address.value != grad_lo
            or gradient_address.size_bytes != gradient.size_bytes
            or gradient_label.symbol.kind is not ProgramSymbolKind.SRAM_LABEL
            or gradient_label.symbol.source_ref != gradient.storage_id
            or gradient_label.value != 0
            or gradient_label.size_bytes != 0):
        raise SchemaError("gradient program declarations differ from physical ABI", path="gradient_address")
    if any(definition.logical_cores != (core,) for definition in (region, gradient_address, gradient_label)):
        raise SchemaError("CE gradient symbols must have exact core scope", path="program_definitions")

    fragments = {fragment.id: fragment for fragment in forward.fragments}
    ref = tape.forward_compute_ref
    source_fragment = fragments[ref.fragment_id]
    source_stream = next(stream for stream in source_fragment.core_streams if stream.logical_core == core)
    forward_record = source_stream.records[ref.fragment_record_index]
    if forward_record.source_global_action_id not in backward_action.deps:
        raise SchemaError("CE backward must depend on original CE forward", path="backward_action.deps")
    forward_action_ids = {record.source_global_action_id: record for fragment in forward.fragments
                          for stream in fragment.core_streams for record in stream.records}
    if (forward_record.opcode is not RecordOpcode.CROSS_ENTROPY_FORWARD
            or forward_record.source_global_action_id not in forward_action_ids):
        raise SchemaError("requires the original native forward CE record", path="tape")
    definitions = {definition.symbol.id: definition for definition in forward.program_symbol_definitions}
    original_labels = []
    for free_ref, abi in zip(tape.terminal_frees, originals):
        free_stream = next(stream for stream in fragments[free_ref.fragment_id].core_streams
                           if stream.logical_core == core)
        free_record = free_stream.records[free_ref.fragment_record_index]
        if free_record.opcode is not RecordOpcode.SRAM_FREE:
            raise SchemaError("forward CE tape lacks its original FREE", path="tape")
        label = definitions[free_record.operands[0].symbol_ref]
        if label.symbol.kind is not ProgramSymbolKind.SRAM_LABEL or label.symbol.source_ref != abi.storage_id:
            raise SchemaError("forward CE FREE label differs from tape", path="tape")
        original_labels.append((free_record, label))

    next_order = backward_action.core_order_index
    extended = tuple(replace(
        abi,
        id=stable_artifact_id("dense_training_extended_ce_tape_abi", {
            "original_abi": abi.id,
            "backward_action": backward_action.id,
            "lifetime_end_exclusive": next_order + 1,
        }, schema_version="wafer_frontend.dense_training_ce_phase_carrier/v1alpha1"),
        lifetime_end_exclusive=next_order + 1,
    ) for abi in originals)
    if any(abi.lifetime_end_exclusive != next_order for abi in originals):
        raise SchemaError("native CE forward tape must end immediately before backward", path="tape")
    output = gradient_address.symbol
    native = build_dense_training_ce_backward_record(
        forward_record,
        logits=tape.original_operand_definitions[0].symbol,
        labels=tape.original_operand_definitions[1].symbol,
        upstream=tape.original_operand_definitions[2].symbol,
        logits_gradient=output,
    )
    native = replace(native, source_global_action_id=backward_action.id)
    alloc = RelocatableRecord(backward_action.id, RecordOpcode.SRAM_ALLOC_AT, (
        RecordOperand.address("region_name", SemanticOperandId.REGION_NAME, region.symbol.id),
        RecordOperand.address("label_symbol", SemanticOperandId.LABEL_SYMBOL, gradient_label.symbol.id),
        RecordOperand.literal("region_offset_bytes", gradient.region_offset_bytes),
        RecordOperand.literal("size_bytes", gradient.size_bytes),
        RecordOperand.literal("alignment_bytes", gradient.alignment_bytes),
        RecordOperand.literal("lifetime", 0),
        RecordOperand.literal("spillable", False),
    ))
    bind = RelocatableRecord(backward_action.id, RecordOpcode.SRAM_BIND, (
        RecordOperand.literal("input_count", 3),
        *(RecordOperand.address(f"input_label_{index}",
                                 SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0) + index),
                                 original_labels[index][1].symbol.id)
          if index < 3 else RecordOperand.literal(f"input_label_{index}", 0)
          for index in range(16)),
        RecordOperand.address("output_label", SemanticOperandId.SRAM_BIND_OUTPUT,
                              gradient_label.symbol.id),
    ))
    records = (alloc, bind, native) + tuple(
        RelocatableRecord(backward_action.id, RecordOpcode.SRAM_FREE, (
            RecordOperand.address("symbol", SemanticOperandId.SYMBOL, label.symbol.id),
        )) for label in (gradient_label, *(item[1] for item in original_labels[::-1])))
    relocations = []
    bindings = []
    for index, operand, definition, abi in (
        (0, SemanticOperandId.REGION_NAME, region, gradient),
        (0, SemanticOperandId.LABEL_SYMBOL, gradient_label, gradient),
        *((1, SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0) + i),
           item[1], extended[i]) for i, item in enumerate(original_labels)),
        (1, SemanticOperandId.SRAM_BIND_OUTPUT, gradient_label, gradient),
        (2, SemanticOperandId.COMPUTE_INPUT_ADDRESS, tape.original_operand_definitions[0], extended[0]),
        (2, SemanticOperandId.COMPUTE_DATA_ADDRESS, tape.original_operand_definitions[1], extended[1]),
        (2, SemanticOperandId.COMPUTE_AUX_ADDRESS, tape.original_operand_definitions[2], extended[2]),
        (2, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, gradient_address, gradient),
        (3, SemanticOperandId.SYMBOL, gradient_label, gradient),
        *((4 + i, SemanticOperandId.SYMBOL, item[1], extended[2 - i])
          for i, item in enumerate(original_labels[::-1])),
    ):
        relocations.append(AddressRelocation(index, operand, definition.symbol.kind,
                                             definition.symbol.id, 0))
        bindings.append(AddressOperandBinding("__ce_phase__", core, index, operand,
                                               (abi.id,), (abi.tensor_slice,)))
    program_definitions = tuple(sorted(
        {definition.symbol.id: definition for definition in
         (*tape.original_operand_definitions, region, gradient_address, gradient_label,
          *(item[1] for item in original_labels))}.values(),
        key=lambda definition: definition.symbol.id,
    ))
    fragment = CommandFragment.create(
        producer_pass="dense_training_ce_phase_carrier",
        source_global_dag_id=source_global_dag_id,
        kind=FragmentKind.COARSE,
        claimed_action_ids=(backward_action.id,),
        core_streams=(CoreFragmentStream(core, records, (), tuple(sorted(
            relocations, key=lambda item: (item.record_index, int(item.operand_id)))),),),
        runtime_symbols=(),
        program_symbols=tuple(definition.symbol for definition in program_definitions),
        buffer_abi=tuple(sorted((*extended, gradient), key=lambda abi: abi.id)),
        state_abi=(),
    )
    fragment.validate("dense_training_ce_phase_carrier")
    free_replacements = tuple((original, LinkedRecordRef(fragment.id, 6 - index,
                                                           backward_action.id))
                              for index, original in enumerate(tape.terminal_frees))
    return DenseCeBackwardPhaseCarrier(
        fragment, program_definitions,
        tuple(replace(binding, fragment_id=fragment.id) for binding in bindings),
        extended,
        tuple((before.id, after.id) for before, after in zip(originals, extended)),
        free_replacements,
        gradient,
    )


__all__ = ["DenseCeBackwardPhaseCarrier", "build_dense_training_ce_phase_carrier"]
