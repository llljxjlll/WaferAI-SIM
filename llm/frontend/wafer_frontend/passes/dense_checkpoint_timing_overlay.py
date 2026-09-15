"""Scoped signed activation HBM LSU/replay overlay on a real L2 Dense carrier.

The forward/CE backward source IR1 still lacks checkpoint actions, so
validate_against production IR1 must reject this bounded timing overlay.  A
successful native run proves only the physical traffic and recompute timing
reported by its probes, not full-model gradient equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressOperandBinding, AddressRelocation, CommandFragment,
    CoreFragmentStream, FragmentInterface, FragmentKind,
    LinkedCoreStream, LinkedProgramManifest, LinkedRecordRef,
    ManifestInputDigest, ManifestInputKind, ProgramSymbol,
    ProgramSymbolDefinition, ProgramSymbolKind, RecordOpcode,
    RecordOperand, RelocatableRecord, SemanticOperandId, StateABI,
    StateOperandBinding,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.persistent_state import (
    PersistentStateAccess, PersistentStateLifetime, StateKind,
)
from ..schema.serde import canonical_digest
from .dense_checkpoint_physical_source import (
    DenseCheckpointActivationTape, DenseCheckpointPhysicalCut,
    derive_dense_checkpoint_replay_template,
)


@dataclass(frozen=True, slots=True)
class DenseCheckpointTimingOverlay:
    manifest: LinkedProgramManifest
    activation_state_abi: StateABI
    activation_fragment_id: str
    save_record_index: int
    restore_record_index: int
    replay_record_index: int | None
    checkpoint_enabled: bool


def build_dense_checkpoint_timing_overlay(
    source: LinkedProgramManifest,
    cut: DenseCheckpointPhysicalCut,
    tape: DenseCheckpointActivationTape,
) -> DenseCheckpointTimingOverlay:
    """Materialize one new typed LSU activation tape and optional native replay."""
    source.validate("checkpoint_overlay_source")
    tape.validate(cut)
    template = derive_dense_checkpoint_replay_template(source, cut)
    checkpoint = tape.checkpoint_enabled
    abi_source = cut.replay_input.buffer if checkpoint else cut.replay_output.buffer
    if abi_source.id != tape.saved_buffer_abi_id:
        raise SchemaError("activation tape no longer binds actual source BufferABI",
                          path="checkpoint_overlay_tape")
    state = StateABI.create(
        state_ref=stable_artifact_id(
            "dense_checkpoint_activation_state",
            {"cut_id": cut.id, "saved_abi_id": abi_source.id},
            schema_version="wafer_frontend.dense_checkpoint_activation_state/v1alpha1"),
        hbm_binding_ref=tape.hbm_binding_ref,
        kind=StateKind.ACTIVATION,
        lifetime=PersistentStateLifetime.STEP,
        access=PersistentStateAccess.READ_WRITE,
        shape=tape.shape, dtype=DType.FP16, layout=tape.layout,
        die_id=tape.die_id, address=tape.activation_hbm_address,
        size_bytes=tape.size_bytes, alignment_bytes=64)
    state.validate("checkpoint_activation_state")
    original_symbols = {d.symbol.id: d.symbol
                        for d in source.program_symbol_definitions}
    original_definitions = {d.symbol.id: d
                            for d in source.program_symbol_definitions}
    chosen = {
        SemanticOperandId.COMPUTE_INPUT_ADDRESS if checkpoint
        else SemanticOperandId.COMPUTE_OUTPUT_ADDRESS
    }
    staging_relocations = [r for r in template.address_relocations
                           if r.operand_id in chosen]
    if len(staging_relocations) != 1:
        raise SchemaError("signed activation SRAM address missing from LM-head",
                          path="checkpoint_overlay_address")
    staging_symbol_ref = staging_relocations[0].symbol_ref
    staging_symbol = original_symbols[staging_symbol_ref]
    if staging_symbol.kind is not ProgramSymbolKind.ABSOLUTE_ADDRESS:
        raise SchemaError("activation staging requires absolute SRAM address",
                          path="checkpoint_overlay_address")
    new_hbm_id = stable_artifact_id(
        "dense_checkpoint_hbm_symbol",
        {"activation_tape_id": tape.id},
        schema_version="wafer_frontend.dense_checkpoint_hbm_symbol/v1alpha1")
    hbm_symbol = ProgramSymbol(
        new_hbm_id, ProgramSymbolKind.ABSOLUTE_ADDRESS, state.hbm_binding_ref)
    hbm_definition = ProgramSymbolDefinition(
        hbm_symbol, "dense.checkpoint.activation.hbm",
        state.address, state.size_bytes, (cut.replay_output.core,))
    save_action = stable_artifact_id(
        "dense_checkpoint_save_action",
        {"source_cut_id": cut.id, "tape_id": tape.id},
        schema_version="wafer_frontend.dense_checkpoint_save_action/v1alpha1")
    restore_action = stable_artifact_id(
        "dense_checkpoint_restore_action",
        {"source_cut_id": cut.id, "tape_id": tape.id},
        schema_version="wafer_frontend.dense_checkpoint_restore_action/v1alpha1")
    replay_action = stable_artifact_id(
        "dense_checkpoint_replay_action",
        {"source_cut_id": cut.id, "tape_id": tape.id,
         "source_record_digest": template.source_record_digest},
        schema_version="wafer_frontend.dense_checkpoint_replay_action/v1alpha1")
    save = RelocatableRecord(
        save_action,
        RecordOpcode.LSU_STORE, (
            RecordOperand.address("hbm_address",
                                  SemanticOperandId.HBM_ADDRESS, new_hbm_id),
            RecordOperand.literal("size_bytes", state.size_bytes),
            RecordOperand.address("source_address",
                                  SemanticOperandId.SOURCE_ADDRESS,
                                  staging_symbol_ref),
        ))
    restore = RelocatableRecord(
        restore_action,
        RecordOpcode.LSU_LOAD, (
            RecordOperand.address("hbm_address",
                                  SemanticOperandId.HBM_ADDRESS, new_hbm_id),
            RecordOperand.literal("size_bytes", state.size_bytes),
            RecordOperand.address("destination_address",
                                  SemanticOperandId.DESTINATION_ADDRESS,
                                  staging_symbol_ref),
        ))
    producer_fragment = next(f for f in source.fragments
                             if f.id == template.source_fragment_id)
    producer_local = next(stream for stream in producer_fragment.core_streams
                          if stream.logical_core == cut.replay_output.core)
    original_bind_index = template.source_fragment_record_index - 1
    if checkpoint and (original_bind_index < 0 or
                       producer_local.records[original_bind_index].opcode
                       is not RecordOpcode.SRAM_BIND):
        raise SchemaError("replay requires original native SRAM_BIND+MATMUL pair",
                          path="checkpoint_replay_bind")
    original_bind = (producer_local.records[original_bind_index]
                     if checkpoint else None)
    bind_relocations = tuple(
        r for r in producer_local.address_relocations
        if checkpoint and r.record_index == original_bind_index)
    records = ((save, restore,
                replace(original_bind, source_global_action_id=replay_action),
                replace(template.record, source_global_action_id=replay_action))
               if checkpoint else (save, restore))
    relocations = [
        AddressRelocation(0, SemanticOperandId.HBM_ADDRESS,
                          ProgramSymbolKind.ABSOLUTE_ADDRESS, new_hbm_id, 0),
        AddressRelocation(0, SemanticOperandId.SOURCE_ADDRESS,
                          ProgramSymbolKind.ABSOLUTE_ADDRESS,
                          staging_symbol_ref, staging_relocations[0].addend),
        AddressRelocation(1, SemanticOperandId.HBM_ADDRESS,
                          ProgramSymbolKind.ABSOLUTE_ADDRESS, new_hbm_id, 0),
        AddressRelocation(1, SemanticOperandId.DESTINATION_ADDRESS,
                          ProgramSymbolKind.ABSOLUTE_ADDRESS,
                          staging_symbol_ref, staging_relocations[0].addend),
    ]
    if checkpoint:
        relocations.extend(
            AddressRelocation(2, r.operand_id, r.symbol_kind, r.symbol_ref,
                              r.addend)
            for r in bind_relocations)
        relocations.extend(
            AddressRelocation(3, r.operand_id, r.symbol_kind, r.symbol_ref,
                              r.addend)
            for r in template.address_relocations)
    relocations = tuple(sorted(relocations,
        key=lambda r: (r.record_index, int(r.operand_id))))
    imported_symbol_ids = {
        staging_symbol_ref,
        *(r.symbol_ref for r in template.address_relocations if checkpoint),
        *(r.symbol_ref for r in bind_relocations),
    }
    program_symbols = tuple(sorted(
        (hbm_symbol, *(original_symbols[sid] for sid in imported_symbol_ids)),
        key=lambda symbol: symbol.id))
    buffer_abis = {abi_source.id: abi_source}
    if checkpoint:
        for item in (cut.replay_input, cut.replay_weight, cut.replay_output):
            buffer_abis[item.buffer.id] = item.buffer
        source_buffers = {abi.id: abi for leaf in source.fragments
                          for abi in leaf.buffer_abi}
        for binding in source.address_operand_bindings:
            if (binding.fragment_id == producer_fragment.id
                    and binding.logical_core == cut.replay_output.core
                    and binding.fragment_record_index == original_bind_index):
                for abi_id in binding.buffer_abi_ids:
                    buffer_abis[abi_id] = source_buffers[abi_id]
    fragment = CommandFragment.create(
        producer_pass="dense_checkpoint_timing_overlay",
        source_global_dag_id=source.source_global_dag_id,
        kind=FragmentKind.STATE_IO,
        claimed_action_ids=tuple(sorted((
            save_action, restore_action, *((replay_action,) if checkpoint else ())))),
        core_streams=(CoreFragmentStream(
            cut.replay_output.core, records, (), relocations),),
        runtime_symbols=(), program_symbols=program_symbols,
        buffer_abi=tuple(sorted(buffer_abis.values(), key=lambda b: b.id)),
        state_abi=(state,))
    fragment.validate("checkpoint_native_overlay_fragment")
    interface = FragmentInterface(
        fragment.id, (), (), tuple(sorted(imported_symbol_ids)),
        (new_hbm_id,), (), ())
    interface.validate("checkpoint_native_overlay_interface")
    original_core = next(stream for stream in source.core_streams
                         if stream.logical_core == cut.replay_output.core)
    old = original_core.records
    save_ref = LinkedRecordRef(
        fragment.id, 0, save_action)
    restore_ref = LinkedRecordRef(
        fragment.id, 1, restore_action)
    replay_bind_ref = LinkedRecordRef(fragment.id, 2, replay_action)
    replay_ref = LinkedRecordRef(fragment.id, 3, replay_action)
    save_position = cut.replay_output.linked_position + 1
    if not checkpoint:
        while save_position < len(old) and (
                old[save_position].source_global_action_id ==
                cut.replay_output.record.source_global_action_id):
            save_position += 1
    restore_position = cut.backward_input.linked_position
    while (restore_position > 0 and
           old[restore_position - 1].source_global_action_id ==
           cut.backward_input.record.source_global_action_id):
        restore_position -= 1
    new_refs = (
        old[:save_position] + (save_ref,) +
        old[save_position:restore_position] +
        (restore_ref, replay_bind_ref, replay_ref) + old[restore_position:]
        if checkpoint else
        old[:save_position] + (save_ref,) +
        old[save_position:restore_position] +
        (restore_ref,) + old[restore_position:])
    updated_core = LinkedCoreStream(original_core.logical_core,
                                    original_core.runtime_core_id, new_refs)
    new_bindings = [
        AddressOperandBinding(
            fragment.id, cut.replay_output.core, 0,
            SemanticOperandId.SOURCE_ADDRESS,
            (abi_source.id,), (abi_source.tensor_slice,)),
        AddressOperandBinding(
            fragment.id, cut.replay_output.core, 1,
            SemanticOperandId.DESTINATION_ADDRESS,
            (abi_source.id,), (abi_source.tensor_slice,)),
    ]
    if checkpoint:
        new_bindings.extend(
            replace(binding, fragment_id=fragment.id,
                    fragment_record_index=2)
            for binding in source.address_operand_bindings
            if binding.fragment_id == producer_fragment.id
            and binding.logical_core == cut.replay_output.core
            and binding.fragment_record_index == original_bind_index)
        new_bindings.extend(
            AddressOperandBinding(
                fragment.id, cut.replay_output.core, 3, operand,
                (item.buffer.id,), (item.buffer.tensor_slice,))
            for operand, item in (
                (SemanticOperandId.COMPUTE_INPUT_ADDRESS, cut.replay_input),
                (SemanticOperandId.COMPUTE_DATA_ADDRESS, cut.replay_weight),
                (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, cut.replay_output)))
    new_state_bindings = (
        StateOperandBinding(fragment.id, cut.replay_output.core, 0,
                            SemanticOperandId.HBM_ADDRESS, state.id),
        StateOperandBinding(fragment.id, cut.replay_output.core, 1,
                            SemanticOperandId.HBM_ADDRESS, state.id),
    )
    digest = ManifestInputDigest(
        ManifestInputKind.COMMAND_FRAGMENT, fragment.id,
        fragment.schema_version, canonical_digest(fragment))
    args = source._semantic_key()
    args.update({
        "fragments": tuple(sorted((*source.fragments, fragment),
                                  key=lambda f: f.id)),
        "fragment_interfaces": tuple(sorted(
            (*source.fragment_interfaces, interface),
            key=lambda i: i.fragment_id)),
        "core_streams": tuple(updated_core if stream.logical_core ==
                              updated_core.logical_core else stream
                              for stream in source.core_streams),
        "program_symbol_definitions": tuple(sorted(
            (*source.program_symbol_definitions, hbm_definition),
            key=lambda definition: definition.symbol.id)),
        "address_operand_bindings": tuple(sorted(
            (*source.address_operand_bindings, *new_bindings),
            key=lambda b: (b.logical_core.die_id, b.logical_core.local_core_id,
                           b.fragment_id, b.fragment_record_index,
                           int(b.operand_id)))),
        "state_operand_bindings": tuple(sorted(
            (*source.state_operand_bindings, *new_state_bindings),
            key=lambda b: (b.logical_core.die_id, b.logical_core.local_core_id,
                           b.fragment_id, b.fragment_record_index,
                           int(b.operand_id)))),
        "input_digests": tuple(sorted(
            (*source.input_digests, digest),
            key=lambda d: (d.kind.value, d.artifact_id))),
    })
    result = LinkedProgramManifest.create(
        producer_pass="dense_checkpoint_timing_overlay", **args)
    result.validate("dense_checkpoint_timing_overlay")
    return DenseCheckpointTimingOverlay(
        result, state, fragment.id, 0, 1, 3 if checkpoint else None,
        checkpoint)


__all__ = ["DenseCheckpointTimingOverlay",
           "build_dense_checkpoint_timing_overlay"]
