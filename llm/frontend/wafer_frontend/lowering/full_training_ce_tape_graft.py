"""Retain the original native CE tape through its real backward physical action.

This is a typed fragment/closure graft for the final full TRAIN linker, not
an independently verified full-model run.  The caller must provide a newly
validated GlobalActionDAG and schedule with the native backward action before
it constructs its sole production LinkedProgramManifest.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..errors import SchemaError
from ..passes.dense_training_ce_phase_carrier import DenseCeBackwardPhaseCarrier
from ..passes.dense_training_ce_seeded_phase import DenseCeSeededPhase
from ..schema.artifact_manifest import (
    AddressOperandBinding, CommandFragment, CoreFragmentStream,
    LinkedCoreStream, LinkedProgramManifest, LinkedRecordRef,
    ProgramSymbolDefinition, RecordOpcode, RuntimeSymbolDefinition,
    RuntimeSymbolKind, StateOperandBinding,
)
from .full_training_timeline_linker import ForwardCrossEntropyTape


@dataclass(frozen=True, slots=True)
class CeGraftedPhysicalForward:
    """Physical forward with the exact CE inputs kept live for native backward."""

    source_forward_manifest_id: str
    global_dag_id: str
    fragments: tuple[CommandFragment, ...]
    core_streams: tuple[LinkedCoreStream, ...]
    runtime_definitions: tuple[RuntimeSymbolDefinition, ...]
    program_definitions: tuple[ProgramSymbolDefinition, ...]
    address_bindings: tuple[AddressOperandBinding, ...]
    state_bindings: tuple[StateOperandBinding, ...]
    original_to_backward_frees: tuple[tuple[LinkedRecordRef, LinkedRecordRef], ...]
    old_to_extended_tape_abi_ids: tuple[tuple[str, str], ...]
    loss_gradient_seed: bytes | None = None
    loss_gradient_abi_id: str | None = None
    retained_forward_loss_free: LinkedRecordRef | None = None


def _graft_ce_backward_onto_forward_tape(
    forward: LinkedProgramManifest,
    tape: ForwardCrossEntropyTape,
    carrier: DenseCeBackwardPhaseCarrier | DenseCeSeededPhase,
    *,
    source_global_dag_id: str,
    seeded: bool,
) -> CeGraftedPhysicalForward:
    """Move only the consumed source frees and update exact ABI lifetimes.

    Source actions remain immutable (the CE factory already re-signs the
    original FREE origins as the new native backward action).  Only the exact
    consumed terminal-free refs disappear; each extended CE BufferABI ID
    propagates to every original fragment and address closure before the
    native ALLOC/BIND/COMPUTE/FREE phase is appended.  Schedule/GlobalActionDAG
    validity and backward gradient producer coverage remain the final caller's
    required proof.
    """
    forward.validate("ce_graft_forward_source")
    carrier.fragment.validate("ce_graft_native_source")
    if seeded and not isinstance(carrier, DenseCeSeededPhase):
        raise SchemaError("full training CE requires typed independent seeded phase",
                          path="carrier")
    if not seeded and not isinstance(carrier, DenseCeBackwardPhaseCarrier):
        raise SchemaError("legacy timing CE carrier has wrong physical source",
                          path="carrier")
    if (not source_global_dag_id or
            source_global_dag_id == forward.source_global_dag_id or
            carrier.fragment.source_global_dag_id != source_global_dag_id):
        raise SchemaError("native CE phase needs the newly re-signed real Global DAG",
                          path="source_global_dag_id")
    mapping = dict(carrier.old_to_extended_abi_ids)
    extended = {abi.id: abi for abi in carrier.extended_forward_buffers}
    expected_count = 2 if seeded else 3
    consumed = (tape.logits, tape.labels) if seeded else (
        tape.logits, tape.labels, tape.per_row_loss)
    consumed_frees = tape.terminal_frees[:expected_count]
    if (len(mapping) != expected_count or len(extended) != expected_count
            or set(mapping.values()) != set(extended)
            or set(mapping) != {abi.id for abi in consumed}):
        raise SchemaError("native CE tape must re-sign exactly its consumed BufferABI",
                          path="carrier.old_to_extended_abi_ids")
    original_frees = {(ref.fragment_id, tape.logical_core,
                       ref.fragment_record_index): ref
                      for ref in consumed_frees}
    if (len(original_frees) != expected_count or
            tuple(old for old, _ in carrier.terminal_free_replacements)
            != consumed_frees):
        raise SchemaError("CE native replacement must cover exactly consumed original frees",
                          path="carrier.terminal_free_replacements")
    if seeded:
        if (carrier.retained_forward_loss_free != tape.terminal_frees[2]
                or carrier.loss_gradient.id in mapping
                or len(carrier.loss_gradient_seed) !=
                   carrier.loss_gradient.size_bytes
                or not any(carrier.loss_gradient_seed)):
            raise SchemaError("seeded CE must retain forward loss FREE and a separate nonzero dLoss",
                              path="carrier.loss_gradient")
        native = next((record for record in carrier.fragment.core_streams[0].records
                       if record.opcode is RecordOpcode.CROSS_ENTROPY_BACKWARD), None)
        upstream = next((operand.symbol_ref for operand in native.operands
                         if operand.name == "upstream_address"), None) if native else None
        if (upstream != carrier.loss_gradient_address.symbol.id
                or upstream == tape.original_operand_definitions[2].symbol.id
                or not any(binding.buffer_abi_ids ==
                           (carrier.loss_gradient.id,)
                           and binding.operand_id.name == "COMPUTE_AUX_ADDRESS"
                           for binding in carrier.address_bindings)):
            raise SchemaError("native CE upstream does not bind a distinct dLoss seed",
                              path="carrier.loss_gradient")
    target = next((stream for stream in forward.core_streams
                   if stream.logical_core == tape.logical_core), None)
    if target is None or set(tape.terminal_free_core_positions) != set(
            range(len(target.records) - 3, len(target.records))):
        raise SchemaError("original logits/labels/loss frees must close forward timeline",
                          path="tape.terminal_free_core_positions")
    if seeded and tape.terminal_frees[2] not in target.records:
        raise SchemaError("seeded native CE must retain real forward loss FREE",
                          path="tape.terminal_frees")
    for old_ref in consumed_frees:
        fragment = next(f for f in forward.fragments if f.id == old_ref.fragment_id)
        local = next(s for s in fragment.core_streams
                     if s.logical_core == tape.logical_core)
        if local.records[old_ref.fragment_record_index].opcode is not RecordOpcode.SRAM_FREE:
            raise SchemaError("terminal tape replacement references non-FREE record",
                              path="tape.terminal_frees")

    fragments, record_maps = [], {}
    for original in forward.fragments:
        core_streams = []
        for stream in original.core_streams:
            retained = []
            index_map = {}
            for old_index, record in enumerate(stream.records):
                if (original.id, stream.logical_core, old_index) in original_frees:
                    continue
                index_map[old_index] = len(retained)
                retained.append(record)
            if not retained:
                raise SchemaError("CE graft would erase an entire source physical leaf",
                                  path=original.id)
            record_maps[(original.id, stream.logical_core)] = index_map
            core_streams.append(CoreFragmentStream(
                stream.logical_core,
                tuple(retained),
                tuple(replace(item, record_index=index_map[item.record_index])
                      for item in stream.runtime_relocations
                      if item.record_index in index_map),
                tuple(replace(item, record_index=index_map[item.record_index])
                      for item in stream.address_relocations
                      if item.record_index in index_map),
            ))
        clone = CommandFragment.create(
            producer_pass=original.producer_pass,
            source_global_dag_id=source_global_dag_id,
            kind=original.kind,
            claimed_action_ids=tuple(sorted({record.source_global_action_id
                for stream in core_streams for record in stream.records})),
            core_streams=tuple(core_streams),
            runtime_symbols=original.runtime_symbols,
            program_symbols=original.program_symbols,
            buffer_abi=tuple(sorted((extended[mapping[abi.id]] if abi.id in mapping
                                     else abi for abi in original.buffer_abi),
                                    key=lambda abi: abi.id)),
            state_abi=original.state_abi,
        )
        clone.validate("ce_graft_retained_source_fragment")
        fragments.append(clone)
        for stream in original.core_streams:
            record_maps[(original.id, stream.logical_core)] = (
                clone.id, record_maps[(original.id, stream.logical_core)]
            )

    def mapped_ref(ref: LinkedRecordRef, core):
        mapped = record_maps[(ref.fragment_id, core)]
        return replace(ref, fragment_id=mapped[0],
                       fragment_record_index=mapped[1][ref.fragment_record_index])

    streams = []
    for stream in forward.core_streams:
        filtered = [mapped_ref(ref, stream.logical_core)
                    for ref in stream.records
                    if (ref.fragment_id, stream.logical_core,
                        ref.fragment_record_index) not in original_frees]
        if stream.logical_core == tape.logical_core:
            if carrier.fragment.core_streams[0].logical_core != stream.logical_core:
                raise SchemaError("CE native gradient runs on wrong physical core",
                                  path="carrier.fragment.core_streams")
            filtered.extend(LinkedRecordRef(carrier.fragment.id, index,
                        record.source_global_action_id)
                            for index, record in enumerate(
                                carrier.fragment.core_streams[0].records))
        streams.append(replace(stream, records=tuple(filtered)))

    address = [replace(binding,
        fragment_id=record_maps[(binding.fragment_id, binding.logical_core)][0],
        fragment_record_index=record_maps[(binding.fragment_id,
             binding.logical_core)][1][binding.fragment_record_index],
        buffer_abi_ids=tuple(mapping.get(item, item) for item
                             in binding.buffer_abi_ids))
               for binding in forward.address_operand_bindings
               if (binding.fragment_id, binding.logical_core,
                   binding.fragment_record_index) not in original_frees]
    address.extend(carrier.address_bindings)
    state = [replace(binding,
        fragment_id=record_maps[(binding.fragment_id, binding.logical_core)][0],
        fragment_record_index=record_maps[(binding.fragment_id,
             binding.logical_core)][1][binding.fragment_record_index])
             for binding in forward.state_operand_bindings]
    declared_program = {symbol.id for fragment in (*fragments, carrier.fragment)
                        for symbol in fragment.program_symbols}
    program = {}
    for definition in (*forward.program_symbol_definitions,
                       *carrier.program_definitions):
        if definition.symbol.id not in declared_program:
            continue
        prior = program.setdefault(definition.symbol.id, definition)
        if prior != definition:
            raise SchemaError("CE source and native declarations collide",
                              path=definition.symbol.id)
    source_runtime = tuple(definition for definition in
                           forward.runtime_symbol_definitions
                           if definition.symbol.kind is not RuntimeSymbolKind.START_TAG)
    return CeGraftedPhysicalForward(
        forward.id, source_global_dag_id,
        tuple(sorted((*fragments, carrier.fragment), key=lambda f: f.id)),
        tuple(streams), source_runtime,
        tuple(sorted(program.values(), key=lambda definition: definition.symbol.id)),
        tuple(address), tuple(state), carrier.terminal_free_replacements,
        carrier.old_to_extended_abi_ids,
        carrier.loss_gradient_seed if seeded else None,
        carrier.loss_gradient.id if seeded else None,
        mapped_ref(carrier.retained_forward_loss_free, tape.logical_core)
        if seeded else None,
    )


def graft_seeded_ce_backward_onto_forward_tape(
    forward: LinkedProgramManifest,
    tape: ForwardCrossEntropyTape,
    carrier: DenseCeSeededPhase,
    *,
    source_global_dag_id: str,
) -> CeGraftedPhysicalForward:
    """Only the independent, nonzero dLoss source is eligible for full TRAIN."""
    return _graft_ce_backward_onto_forward_tape(
        forward, tape, carrier, source_global_dag_id=source_global_dag_id,
        seeded=True,
    )


def graft_native_ce_backward_onto_forward_tape(
    forward: LinkedProgramManifest,
    tape: ForwardCrossEntropyTape,
    carrier: DenseCeBackwardPhaseCarrier,
    *,
    source_global_dag_id: str,
) -> CeGraftedPhysicalForward:
    """Preserve legacy physical CE timing surrogate; this is not a trainer."""
    return _graft_ce_backward_onto_forward_tape(
        forward, tape, carrier, source_global_dag_id=source_global_dag_id,
        seeded=False,
    )


__all__ = ["CeGraftedPhysicalForward", "graft_native_ce_backward_onto_forward_tape",
           "graft_seeded_ce_backward_onto_forward_tape"]
