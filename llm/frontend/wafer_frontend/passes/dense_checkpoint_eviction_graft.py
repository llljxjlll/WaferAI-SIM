"""Signed bounded L2 checkpoint eviction and physical replay graft.

This is a scoped timing carrier.  Its added actions have no source IR1/schedule
counterparts, so validate_against must not be reported as a full-model pass.
"""
from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressOperandBinding, AddressRelocation, BufferABI, CommandFragment,
    CoreFragmentStream, FragmentInterface, LinkedCoreStream,
    LinkedProgramManifest, LinkedRecordRef, ManifestInputDigest,
    ManifestInputKind, ProgramSymbol, ProgramSymbolDefinition,
    ProgramSymbolKind, RecordOpcode, RecordOperand, RelocatableRecord,
    SemanticOperandId, StateOperandBinding,
)
from ..schema.common import stable_artifact_id
from ..schema.serde import canonical_digest
from .dense_checkpoint_physical_source import (
    DenseCheckpointActivationTape, DenseCheckpointPhysicalCut,
)
from .dense_checkpoint_timing_overlay import build_dense_checkpoint_timing_overlay


def _id(prefix: str, cut: DenseCheckpointPhysicalCut, field: str) -> str:
    return stable_artifact_id(
        prefix, {"cut_id": cut.id, "field": field},
        schema_version="wafer_frontend.dense_checkpoint_eviction_graft/v1alpha1")


def _swap_symbol(record: RelocatableRecord, old: str, new: str) -> RelocatableRecord:
    return replace(record, operands=tuple(
        replace(operand, symbol_ref=new) if operand.symbol_ref == old else operand
        for operand in record.operands))


def graft_dense_checkpoint_eviction(
    source: LinkedProgramManifest,
    cut: DenseCheckpointPhysicalCut,
    tape: DenseCheckpointActivationTape,
) -> LinkedProgramManifest:
    """Store hidden32, free it, allocate a distinct staging slot, load and replay."""
    if not tape.checkpoint_enabled:
        raise SchemaError("eviction graft requires checkpoint tape", path="checkpoint_eviction")
    old_overlay = build_dense_checkpoint_timing_overlay(source, cut, tape)
    signed = old_overlay.manifest
    core = cut.replay_output.core
    producer = next(f for f in source.fragments
                    if f.id == cut.replay_output.record.fragment_id)
    local = next(s for s in producer.core_streams if s.logical_core == core)
    if (len(local.records) != 5 or
            tuple(r.opcode for r in local.records[3:]) !=
            (RecordOpcode.SRAM_FREE, RecordOpcode.SRAM_FREE)):
        raise SchemaError("actual LM-head producer free suffix changed", path="checkpoint_eviction.producer")
    producer_clone = CommandFragment.create(
        producer_pass="dense_checkpoint_evicted_producer",
        **{**producer._semantic_key(), "core_streams": (
            CoreFragmentStream(core, local.records[:3],
                tuple(r for r in local.runtime_relocations if r.record_index < 3),
                tuple(r for r in local.address_relocations if r.record_index < 3)),)})
    producer_clone.validate("checkpoint_evicted_producer")
    previous = next(f for f in signed.fragments if f.id == old_overlay.activation_fragment_id)
    prior_local = previous.core_streams[0]
    if len(prior_local.records) != 4:
        raise SchemaError("activation overlay record shape changed", path="checkpoint_eviction.overlay")
    original_bind = local.records[1]
    original_matmul = local.records[2]
    save_action = prior_local.records[0].source_global_action_id
    restore_action = prior_local.records[1].source_global_action_id
    replay_action = prior_local.records[2].source_global_action_id
    source_label = local.records[4].operands[0].symbol_ref
    weight_label = local.records[3].operands[0].symbol_ref
    source_abs = original_matmul.operands[1].symbol_ref
    region_symbol = local.records[0].operands[0].symbol_ref
    if (source_label is None or weight_label is None or
            source_abs is None or region_symbol is None):
        raise SchemaError("producer loses physical SRAM symbols", path="checkpoint_eviction.symbols")
    source_abi = cut.replay_input.buffer
    storage = _id("dense_checkpoint_restored_storage", cut, "hidden")
    binding = _id("dense_checkpoint_restored_binding", cut, "hidden")
    value = _id("dense_checkpoint_restored_value", cut, "hidden")
    restored = replace(source_abi,
        id=_id("dense_checkpoint_restored_buffer_abi", cut, "hidden"),
        binding_id=binding, value_id=value,
        tensor_slice=replace(source_abi.tensor_slice, value_id=value),
        storage_id=storage, region_offset_bytes=4096,
        lifetime_start=40, lifetime_end_exclusive=43)
    restored.validate("checkpoint_restored_hidden")
    label_id = _id("dense_checkpoint_restored_label_symbol", cut, "hidden")
    abs_id = _id("dense_checkpoint_restored_abs_symbol", cut, "hidden")
    label = ProgramSymbol(label_id, ProgramSymbolKind.SRAM_LABEL, storage)
    address = ProgramSymbol(abs_id, ProgramSymbolKind.ABSOLUTE_ADDRESS, binding)
    defs = (
        ProgramSymbolDefinition(label, "dense.checkpoint.restored.hidden", 0, 0, (core,)),
        ProgramSymbolDefinition(address, "dense.checkpoint.restored.hidden.addr",
                                restored.region_offset_bytes, restored.size_bytes, (core,)),
    )
    allocate = replace(local.records[0], source_global_action_id=restore_action,
        operands=tuple(
            replace(operand, symbol_ref=label_id)
            if operand.symbol_ref == local.records[0].operands[1].symbol_ref else
            replace(operand, literal_value=restored.region_offset_bytes)
            if operand.name == "region_offset_bytes" else
            replace(operand, literal_value=source_abi.size_bytes)
            if operand.name == "size_bytes" else operand
            for operand in local.records[0].operands))
    restore = _swap_symbol(prior_local.records[1], source_abs, abs_id)
    bind = _swap_symbol(prior_local.records[2], source_label, label_id)
    matmul = _swap_symbol(prior_local.records[3], source_abs, abs_id)
    records = (
        prior_local.records[0],
        replace(local.records[4], source_global_action_id=save_action),
        allocate, restore, bind, matmul,
        replace(local.records[3], source_global_action_id=replay_action),
        _swap_symbol(replace(local.records[4], source_global_action_id=replay_action),
                     source_label, label_id),
    )
    # Every address operand gets one relocation, with the input label/address
    # rebound only in the restored replay.  Original forward bits remain signed.
    reloc = []
    reloc.extend(replace(r, record_index=0) for r in prior_local.address_relocations
                 if r.record_index == 0)
    reloc.extend(replace(r, record_index=1) for r in local.address_relocations
                 if r.record_index == 4)
    reloc.extend(
        replace(r, record_index=2, symbol_ref=label_id
                if r.operand_id is SemanticOperandId.LABEL_SYMBOL else r.symbol_ref)
        for r in local.address_relocations if r.record_index == 0)
    reloc.extend(replace(r, record_index=3, symbol_ref=abs_id
                         if r.operand_id is SemanticOperandId.DESTINATION_ADDRESS else r.symbol_ref)
                 for r in prior_local.address_relocations if r.record_index == 1)
    reloc.extend(replace(r, record_index=4, symbol_ref=label_id
                         if r.operand_id is SemanticOperandId.SRAM_BIND_INPUT_0 else r.symbol_ref)
                 for r in prior_local.address_relocations if r.record_index == 2)
    reloc.extend(replace(r, record_index=5, symbol_ref=abs_id
                         if r.operand_id is SemanticOperandId.COMPUTE_INPUT_ADDRESS else r.symbol_ref)
                 for r in prior_local.address_relocations if r.record_index == 3)
    reloc.extend(replace(r, record_index=6) for r in local.address_relocations
                 if r.record_index == 3)
    reloc.append(AddressRelocation(7, SemanticOperandId.SYMBOL,
                                   ProgramSymbolKind.SRAM_LABEL, label_id, 0))
    reloc = tuple(sorted(reloc, key=lambda r: (r.record_index, int(r.operand_id))))
    known = {s.id: s for f in signed.fragments for s in f.program_symbols}
    known[label_id], known[abs_id] = label, address
    used = {r.symbol_ref for r in reloc}
    fragment = CommandFragment.create(
        producer_pass="dense_checkpoint_physical_eviction",
        source_global_dag_id=source.source_global_dag_id,
        kind=previous.kind, claimed_action_ids=previous.claimed_action_ids,
        core_streams=(CoreFragmentStream(core, records, (), reloc),),
        runtime_symbols=(),
        program_symbols=tuple(known[s] for s in sorted(used)),
        buffer_abi=tuple(sorted((*previous.buffer_abi, restored), key=lambda a: a.id)),
        state_abi=previous.state_abi)
    fragment.validate("checkpoint_physical_eviction_fragment")
    original_interface = next(i for i in signed.fragment_interfaces
                              if i.fragment_id == producer.id)
    producer_interface = replace(original_interface, fragment_id=producer_clone.id)
    previous_interface = next(i for i in signed.fragment_interfaces
                              if i.fragment_id == previous.id)
    exported = set(previous_interface.program_exports) | {s.id for s in (label, address)}
    imported = used - exported
    activation_interface = FragmentInterface(
        fragment.id, (), (), tuple(sorted(imported)), tuple(sorted(exported)), (), ())
    activation_interface.validate("checkpoint_eviction_interface")
    producer_digest = ManifestInputDigest(
        ManifestInputKind.COMMAND_FRAGMENT, producer_clone.id,
        producer_clone.schema_version, canonical_digest(producer_clone))
    activation_digest = ManifestInputDigest(
        ManifestInputKind.COMMAND_FRAGMENT, fragment.id,
        fragment.schema_version, canonical_digest(fragment))
    old_core = next(s for s in signed.core_streams if s.logical_core == core)
    prior_refs = old_core.records
    producer_refs = [i for i, ref in enumerate(prior_refs)
                     if ref.fragment_id == producer.id]
    if len(producer_refs) != 5 or [prior_refs[i].fragment_record_index for i in producer_refs] != list(range(5)):
        raise SchemaError("producer refs no longer a five-record action", path="checkpoint_eviction.refs")
    backward_pos = cut.backward_input.linked_position
    # New source overlay adds four records before backward; find its first action ref.
    backward_ref = cut.backward_input.record
    backwards = [i for i, ref in enumerate(prior_refs) if ref == backward_ref]
    if len(backwards) != 1:
        raise SchemaError("backward consumer ref changed", path="checkpoint_eviction.backward")
    backward_start = backwards[0]
    while backward_start and (prior_refs[backward_start - 1].source_global_action_id ==
                              backward_ref.source_global_action_id):
        backward_start -= 1
    retained = tuple(ref for ref in prior_refs
                     if ref.fragment_id not in (producer.id, previous.id))
    p = producer_refs[0]
    ce = next(i for i, ref in enumerate(retained)
              if ref == prior_refs[backward_start])
    # Derive splice positions from actual retained source refs rather than a
    # guessed source record count; CE action itself remains whole and ordered.
    before_ce = next(i for i, ref in enumerate(retained)
                     if ref == prior_refs[backward_start])
    core_refs = (
        retained[:p] + tuple(LinkedRecordRef(producer_clone.id, i,
                    local.records[i].source_global_action_id) for i in range(3)) +
        tuple(LinkedRecordRef(fragment.id, i, records[i].source_global_action_id)
              for i in range(2)) + retained[p:before_ce] +
        tuple(LinkedRecordRef(fragment.id, i, records[i].source_global_action_id)
              for i in range(2, 8)) + retained[before_ce:])
    updated_core = LinkedCoreStream(core, old_core.runtime_core_id, core_refs)
    source_bindings = [replace(b, fragment_id=producer_clone.id)
                       for b in signed.address_operand_bindings
                       if b.fragment_id == producer.id and b.fragment_record_index < 3]
    source_bindings.extend(b for b in signed.address_operand_bindings
                           if b.fragment_id not in (producer.id, previous.id))
    by_old = {(b.fragment_record_index, b.operand_id): b
              for b in signed.address_operand_bindings
              if b.fragment_id == previous.id}
    by_producer = {(b.fragment_record_index, b.operand_id): b
                   for b in signed.address_operand_bindings
                   if b.fragment_id == producer.id}
    mapped = ((0, 0, False), (4, 1, True), (0, 2, True),
              (1, 3, False), (2, 4, False), (3, 5, False),
              (3, 6, True))
    for src_index, new_index, from_producer in mapped:
        entries = by_producer if from_producer else by_old
        for (record_index, _operand), binding_item in entries.items():
            if record_index != src_index:
                continue
            abi_ids = binding_item.buffer_abi_ids
            tensor = binding_item.tensor_slices
            if new_index == 2:
                abi_ids = (restored.id,)
                tensor = (restored.tensor_slice,)
            if new_index in (3, 4, 5) and source_abi.id in abi_ids:
                abi_ids = tuple(restored.id if a == source_abi.id else a for a in abi_ids)
                tensor = tuple(restored.tensor_slice if a == source_abi.id else t
                               for a, t in zip(binding_item.buffer_abi_ids, tensor))
            source_bindings.append(replace(binding_item, fragment_id=fragment.id,
                fragment_record_index=new_index, buffer_abi_ids=abi_ids,
                tensor_slices=tensor))
    source_bindings.append(AddressOperandBinding(
        fragment.id, core, 7, SemanticOperandId.SYMBOL,
        (restored.id,), (restored.tensor_slice,)))
    state_bindings = [replace(b, fragment_id=producer_clone.id)
                      if b.fragment_id == producer.id else
                      replace(b, fragment_id=fragment.id,
                              fragment_record_index=0 if b.fragment_record_index == 0 else 3)
                      if b.fragment_id == previous.id else b
                      for b in signed.state_operand_bindings]
    args = signed._semantic_key()
    args.update({
        "fragments": tuple(sorted((f for f in signed.fragments
             if f.id not in (producer.id, previous.id)), key=lambda f: f.id)) +
             tuple(sorted((producer_clone, fragment), key=lambda f: f.id)),
        "fragment_interfaces": tuple(sorted((i for i in signed.fragment_interfaces
             if i.fragment_id not in (producer.id, previous.id)), key=lambda i: i.fragment_id)) +
             tuple(sorted((producer_interface, activation_interface), key=lambda i: i.fragment_id)),
        "core_streams": tuple(updated_core if s.logical_core == core else s
                              for s in signed.core_streams),
        "program_symbol_definitions": tuple(sorted(
            (*signed.program_symbol_definitions, *defs),
            key=lambda d: d.symbol.id)),
        "address_operand_bindings": tuple(sorted(source_bindings,
            key=lambda b: (b.logical_core.die_id, b.logical_core.local_core_id,
                           b.fragment_id, b.fragment_record_index, int(b.operand_id)))),
        "state_operand_bindings": tuple(sorted(state_bindings,
            key=lambda b: (b.logical_core.die_id, b.logical_core.local_core_id,
                           b.fragment_id, b.fragment_record_index, int(b.operand_id)))),
        "input_digests": tuple(sorted((d for d in signed.input_digests
            if not (d.kind is ManifestInputKind.COMMAND_FRAGMENT and
                    d.artifact_id in (producer.id, previous.id))),
            key=lambda d: (d.kind.value, d.artifact_id))) +
            tuple(sorted((producer_digest, activation_digest),
                         key=lambda d: (d.kind.value, d.artifact_id))),
    })
    args["fragments"] = tuple(sorted(args["fragments"], key=lambda f: f.id))
    args["fragment_interfaces"] = tuple(sorted(args["fragment_interfaces"], key=lambda i: i.fragment_id))
    args["input_digests"] = tuple(sorted(args["input_digests"], key=lambda d: (d.kind.value, d.artifact_id)))
    result = LinkedProgramManifest.create(
        producer_pass="dense_checkpoint_physical_eviction", **args)
    result.validate("dense_checkpoint_physical_eviction")
    return result


__all__ = ["graft_dense_checkpoint_eviction"]
