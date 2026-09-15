"""Source-backed native CE backward with a separate, seeded dLoss buffer.

The two forward inputs remain live until this backward action.  The FP32
forward loss is a reported result, not the gradient of that result: it retains
its original forward FREE.  A full training timeline must link this fragment,
its new scheduled task, and an actual ProgramIO initialization for dLoss.
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
from ..schema.common import DType, stable_artifact_id
from ..schema.global_action import GlobalAction
from ..schema.ir0 import OpKind
from ..schema.ir2 import BufferOwnership
from .dense_training_ce_backward import (
    build_dense_training_ce_backward_from_loss_gradient,
    dense_training_per_row_loss_gradient_seed,
)


_SCHEMA = "wafer_frontend.dense_training_ce_seeded_phase/v1alpha1"


@dataclass(frozen=True, slots=True)
class DenseCeSeededPhase:
    fragment: CommandFragment
    program_definitions: tuple[ProgramSymbolDefinition, ...]
    address_bindings: tuple[AddressOperandBinding, ...]
    extended_forward_buffers: tuple[BufferABI, BufferABI]
    old_to_extended_abi_ids: tuple[tuple[str, str], ...]
    terminal_free_replacements: tuple[tuple[LinkedRecordRef, LinkedRecordRef], ...]
    retained_forward_loss_free: LinkedRecordRef
    loss_gradient: BufferABI
    loss_gradient_address: ProgramSymbolDefinition
    loss_gradient_seed: bytes
    logits_gradient: BufferABI


def _declaration_matches(abi: BufferABI, address: ProgramSymbolDefinition,
                         label: ProgramSymbolDefinition, region: ProgramSymbolDefinition) -> bool:
    return (
        address.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
        and address.symbol.source_ref == abi.binding_id
        and address.value == region.value + abi.region_offset_bytes
        and address.size_bytes == abi.size_bytes
        and label.symbol.kind is ProgramSymbolKind.SRAM_LABEL
        and label.symbol.source_ref == abi.storage_id
        and label.value == label.size_bytes == 0
        and address.logical_cores == label.logical_cores == region.logical_cores
        == (abi.logical_core,)
    )


def build_dense_training_ce_seeded_phase(
    forward: LinkedProgramManifest,
    tape: ForwardCrossEntropyTape,
    *,
    backward_action: GlobalAction,
    source_global_dag_id: str,
    loss_gradient: BufferABI,
    loss_gradient_address: ProgramSymbolDefinition,
    loss_gradient_label: ProgramSymbolDefinition,
    logits_gradient: BufferABI,
    logits_gradient_address: ProgramSymbolDefinition,
    logits_gradient_label: ProgramSymbolDefinition,
    region: ProgramSymbolDefinition,
) -> DenseCeSeededPhase:
    """Emit action-local ALLOC/BIND/native CE/FREE with explicit dLoss seed.

    The final source-backed linker must preserve the original forward CE loss
    FREE, replace only the logits/labels FREE, and record the returned seed as
    a nonzero ProgramIO SRAM initialization before execution.
    """
    forward.validate("seeded_ce_forward")
    backward_action.validate("seeded_ce_backward_action")
    if (backward_action.op_kind is not OpKind.CE_BACKWARD
            or backward_action.compute is None
            or not source_global_dag_id
            or source_global_dag_id == backward_action.source.dag_id):
        raise SchemaError("requires CE_BACKWARD and separately signed GlobalActionDAG ID",
                          path="backward_action")
    logits, labels, forward_loss = tape.logits, tape.labels, tape.per_row_loss
    core = tape.logical_core
    order = backward_action.core_order_index
    if (backward_action.logical_core != core
            or order is None or logits.lifetime_end_exclusive != order
            or labels.lifetime_end_exclusive != order
            or backward_action.source.schedule_id != logits.schedule_id):
        raise SchemaError("backward must follow the physical logits/labels forward tape",
                          path="backward_action")
    if (loss_gradient.ownership is not BufferOwnership.BORROWED
            or loss_gradient.dtype is not DType.FP32
            or loss_gradient.size_bytes != forward_loss.size_bytes
            or loss_gradient.tensor_slice.shape != forward_loss.tensor_slice.shape
            or logits_gradient.ownership is not BufferOwnership.OWNED
            or logits_gradient.dtype is not DType.FP16
            or logits_gradient.size_bytes != logits.size_bytes
            or logits_gradient.tensor_slice.shape != logits.tensor_slice.shape):
        raise SchemaError("dLoss must be BORROWED FP32; logits gradient OWNED FP16",
                          path="loss_gradient")
    newcomers = (loss_gradient, logits_gradient)
    if any(abi.logical_core != core or abi.schedule_id != logits.schedule_id
           or abi.region_ref != logits.region_ref or abi.alias_of is not None
           or abi.lifetime_start != order or abi.lifetime_end_exclusive != order + 1
           or abi.region_offset_bytes % abi.alignment_bytes
           for abi in newcomers):
        raise SchemaError("CE seed and output require fresh action-local SRAM placement",
                          path="new_buffers")
    if (region.symbol.kind is not ProgramSymbolKind.SRAM_REGION
            or region.symbol.source_ref != logits.region_ref
            or any(abi.region_offset_bytes + abi.size_bytes > region.size_bytes
                   for abi in newcomers)):
        raise SchemaError("loss seed/output exceed the physical SRAM region", path="region")
    for abi, address, label in (
        (loss_gradient, loss_gradient_address, loss_gradient_label),
        (logits_gradient, logits_gradient_address, logits_gradient_label),
    ):
        if not _declaration_matches(abi, address, label, region):
            raise SchemaError("CE seed/output symbol declarations differ from physical ABI",
                              path="program_definitions")
    original_refs = {abi.id for abi in (logits, labels, forward_loss)}
    occupied = tuple(abi for fragment in forward.fragments for abi in fragment.buffer_abi
                     if abi.logical_core == core and abi.region_ref == logits.region_ref)
    for i, new in enumerate(newcomers):
        for old in (*occupied, *newcomers[i + 1:]):
            # dLoss is seeded before the entire program starts; no earlier
            # forward write may overwrite its bytes.  The owned output must
            # avoid all forward tape data kept alive for backward.
            if (new is logits_gradient and old.id not in original_refs
                    and old.lifetime_end_exclusive <= order):
                continue
            lo, hi = new.region_offset_bytes, new.region_offset_bytes + new.size_bytes
            if lo < old.region_offset_bytes + old.size_bytes and old.region_offset_bytes < hi:
                raise SchemaError("CE seed/output overlaps physical forward SRAM",
                                  path="new_buffers")

    fragments = {fragment.id: fragment for fragment in forward.fragments}
    ce_ref = tape.forward_compute_ref
    source_stream = next(s for s in fragments[ce_ref.fragment_id].core_streams
                         if s.logical_core == core)
    ce_record = source_stream.records[ce_ref.fragment_record_index]
    if (ce_record.opcode is not RecordOpcode.CROSS_ENTROPY_FORWARD
            or ce_record.source_global_action_id not in backward_action.deps):
        raise SchemaError("backward lacks the original native CE dependency", path="tape")
    original_defs = {d.symbol.id: d for d in forward.program_symbol_definitions}
    labels_by_tape = []
    for abi, free_ref in zip((logits, labels, forward_loss), tape.terminal_frees):
        stream = next(s for s in fragments[free_ref.fragment_id].core_streams
                      if s.logical_core == core)
        free = stream.records[free_ref.fragment_record_index]
        definition = original_defs[free.operands[0].symbol_ref]
        if (free.opcode is not RecordOpcode.SRAM_FREE
                or definition.symbol.kind is not ProgramSymbolKind.SRAM_LABEL
                or definition.symbol.source_ref != abi.storage_id):
            raise SchemaError("original CE loss/logits/labels FREE has stale label", path="tape")
        labels_by_tape.append(definition)

    extended = tuple(replace(
        abi,
        id=stable_artifact_id("dense_seeded_ce_extended_tape", {
            "original_abi": abi.id, "backward_action": backward_action.id,
            "lifetime_end": order + 1,
        }, schema_version=_SCHEMA),
        lifetime_end_exclusive=order + 1,
    ) for abi in (logits, labels))
    if any(abi.id in {old.id for old in (logits, labels)} for abi in extended):
        raise SchemaError("CE tape must receive new physical ABI IDs", path="tape")
    seed_bytes = dense_training_per_row_loss_gradient_seed(
        next(item.literal_value for item in ce_record.operands if item.name == "rank_rows")
    )
    if len(seed_bytes) != loss_gradient.size_bytes:
        raise SchemaError("loss-gradient ProgramIO seed size differs from ABI", path="loss_gradient")
    native = build_dense_training_ce_backward_from_loss_gradient(
        ce_record,
        logits=tape.original_operand_definitions[0].symbol,
        labels=tape.original_operand_definitions[1].symbol,
        loss_gradient=loss_gradient_address.symbol,
        logits_gradient=logits_gradient_address.symbol,
    )
    native = replace(native, source_global_action_id=backward_action.id)

    allocation = sorted(((loss_gradient, loss_gradient_label),
                         (logits_gradient, logits_gradient_label)),
                        key=lambda item: (item[0].region_ref, item[0].region_offset_bytes,
                                          item[0].storage_id))
    alloc_records = [RelocatableRecord(backward_action.id, RecordOpcode.SRAM_ALLOC_AT, (
        RecordOperand.address("region_name", SemanticOperandId.REGION_NAME, region.symbol.id),
        RecordOperand.address("label_symbol", SemanticOperandId.LABEL_SYMBOL, label.symbol.id),
        RecordOperand.literal("region_offset_bytes", abi.region_offset_bytes),
        RecordOperand.literal("size_bytes", abi.size_bytes),
        RecordOperand.literal("alignment_bytes", abi.alignment_bytes),
        RecordOperand.literal("lifetime", 0),
        RecordOperand.literal("spillable", False),
    )) for abi, label in allocation]
    bind_index, native_index = len(alloc_records), len(alloc_records) + 1
    bind = RelocatableRecord(backward_action.id, RecordOpcode.SRAM_BIND, (
        RecordOperand.literal("input_count", 3),
        *(RecordOperand.address(f"input_label_{i}",
                                 SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0) + i),
                                 label.symbol.id)
          if i < 3 else RecordOperand.literal(f"input_label_{i}", 0)
          for i, label in enumerate((*labels_by_tape[:2], loss_gradient_label,
                                    *(None,) * 13))),
        RecordOperand.address("output_label", SemanticOperandId.SRAM_BIND_OUTPUT,
                              logits_gradient_label.symbol.id),
    ))
    final_frees = tuple(sorted(
        ((abi, label) for abi, label in
         ((extended[0], labels_by_tape[0]), (extended[1], labels_by_tape[1]),
          (loss_gradient, loss_gradient_label),
          (logits_gradient, logits_gradient_label))),
        key=lambda item: (item[0].region_ref, item[0].region_offset_bytes,
                          item[0].storage_id), reverse=True,
    ))
    free_records = [RelocatableRecord(backward_action.id, RecordOpcode.SRAM_FREE, (
        RecordOperand.address("symbol", SemanticOperandId.SYMBOL, label.symbol.id),
    )) for _, label in final_frees]
    records = (*alloc_records, bind, native, *free_records)

    addresses = (
        tape.original_operand_definitions[0], tape.original_operand_definitions[1],
        loss_gradient_address, logits_gradient_address,
    )
    abis = (*extended, loss_gradient, logits_gradient)
    closures = []
    for i, (abi, label) in enumerate(allocation):
        closures.extend(((i, SemanticOperandId.REGION_NAME, region, abi),
                         (i, SemanticOperandId.LABEL_SYMBOL, label, abi)))
    for i, (label, abi) in enumerate(zip((*labels_by_tape[:2], loss_gradient_label),
                                          (*extended, loss_gradient))):
        closures.append((bind_index,
                         SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0) + i),
                         label, abi))
    closures.append((bind_index, SemanticOperandId.SRAM_BIND_OUTPUT,
                     logits_gradient_label, logits_gradient))
    closures.extend((native_index, operand, definition, abi) for operand, definition, abi in zip(
        (SemanticOperandId.COMPUTE_INPUT_ADDRESS, SemanticOperandId.COMPUTE_DATA_ADDRESS,
         SemanticOperandId.COMPUTE_AUX_ADDRESS, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS),
        addresses, abis,
    ))
    closures.extend((native_index + 1 + i, SemanticOperandId.SYMBOL, label, abi)
                    for i, (abi, label) in enumerate(final_frees))
    relocations = tuple(sorted((AddressRelocation(index, operand, definition.symbol.kind,
                                                   definition.symbol.id, 0)
                                for index, operand, definition, _ in closures),
                               key=lambda item: (item.record_index, int(item.operand_id))))
    definitions = tuple(sorted({definition.symbol.id: definition for definition in
                                (*addresses, region, *labels_by_tape[:2],
                                 loss_gradient_label, logits_gradient_label)}.values(),
                               key=lambda item: item.symbol.id))
    fragment = CommandFragment.create(
        producer_pass="dense_training_ce_seeded_phase",
        source_global_dag_id=source_global_dag_id,
        kind=FragmentKind.COARSE,
        claimed_action_ids=(backward_action.id,),
        core_streams=(CoreFragmentStream(core, tuple(records), (), relocations),),
        runtime_symbols=(),
        program_symbols=tuple(definition.symbol for definition in definitions),
        buffer_abi=tuple(sorted(abis, key=lambda item: item.id)),
        state_abi=(),
    )
    fragment.validate("seeded_dense_training_ce_phase")
    bindings = tuple(AddressOperandBinding(fragment.id, core, index, operand,
                                           (abi.id,), (abi.tensor_slice,))
                     for index, operand, _, abi in closures)
    replaced = tuple((free_ref, LinkedRecordRef(
        fragment.id,
        native_index + 1 + next(index for index, (free_abi, _) in enumerate(final_frees)
                                if free_abi.id == abi.id),
        backward_action.id,
    )) for free_ref, abi in zip(tape.terminal_frees[:2], extended))
    return DenseCeSeededPhase(
        fragment, definitions, bindings, extended,
        tuple((old.id, new.id) for old, new in zip((logits, labels), extended)),
        replaced, tape.terminal_frees[2], loss_gradient,
        loss_gradient_address, seed_bytes, logits_gradient,
    )


__all__ = ["DenseCeSeededPhase", "build_dense_training_ce_seeded_phase"]
