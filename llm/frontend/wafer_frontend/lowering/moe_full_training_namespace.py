"""Typed physical namespace for each real MoE TRAIN layer and SGD step.

This pass clones production CommandFragments and their closures.  It never
adds fake CE/backbone work; the common timeline merger must arrange the exact
forward/backward streams around the real shared loss and reject any missing
native layer operation before constructing its one linked program.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressOperandBinding, BufferABI, CommandFragment, LinkedCoreStream,
    LinkedRecordRef, ProgramSymbolDefinition, ProgramSymbolKind, RecordOpcode,
    RuntimeSymbolDefinition, RuntimeSymbolKind, StateABI, StateOperandBinding,
)
from ..schema.common import stable_artifact_id
from ..schema.moe_compile_sequence import MoeCompileUnit
from ..schema.flexible_moe import MoeRectActionKind
from ..schema.global_action import LogicalCoreRef
from ..schema.persistent_state import HbmAddressSpace
from .full_training_timeline_linker import cut_moe_training_unit
from .moe_full_model_linker import _clone_moe_fragment


_SCHEMA = "wafer_frontend.moe_full_training_namespace/v1alpha1"
_DENSE_SRAM_PARTITION = 8192
_SRAM_REGION_SHIFT = 4096
_SRAM_MOE_LAYER_STRIDE = 8192
_HBM_REGION_SHIFT = 1 << 24
_HBM_MOE_LAYER_STRIDE = 1 << 20


@dataclass(frozen=True, slots=True)
class NamespacedMoeTrainingUnit:
    step: int
    layer: int
    source_unit_id: str
    source_manifest_id: str
    fragments: tuple[CommandFragment, ...]
    forward: tuple[LinkedCoreStream, ...]
    backward: tuple[LinkedCoreStream, ...]
    runtime_definitions: tuple[RuntimeSymbolDefinition, ...]
    program_definitions: tuple[ProgramSymbolDefinition, ...]
    address_bindings: tuple[AddressOperandBinding, ...]
    state_bindings: tuple[StateOperandBinding, ...]
    action_ids: tuple[tuple[str, str], ...]


def derive_moe_training_transport_edges(
    unit: MoeCompileUnit,
    named: NamespacedMoeTrainingUnit,
) -> tuple[tuple[str, str], ...]:
    """Require every original P2 flow to depend on its real opposite-die DTE."""
    if (unit.id != named.source_unit_id
            or (unit.step, unit.layer) != (named.step, named.layer)):
        raise SchemaError("named transport lacks its source unit step/layer",
                          path="training_moe_transport")
    actions = tuple(unit.plan.actions)
    mapping = dict(named.action_ids)
    records = {record.source_global_action_id: record.opcode
               for fragment in named.fragments for stream in fragment.core_streams
               for record in stream.records if record.opcode in (
                   RecordOpcode.DTE_SEND, RecordOpcode.DTE_RECV)}
    edges = []
    for flow in unit.plan.flows:
        sends = [action for action in actions
                 if action.kind is MoeRectActionKind.SEND
                 and action.flow_ref == flow.id and action.rank == flow.source_rank]
        recvs = [action for action in actions
                 if action.kind is MoeRectActionKind.RECV
                 and action.flow_ref == flow.id and action.rank == flow.destination_rank]
        if len(sends) != 1 or len(recvs) != 1:
            raise SchemaError("P2 source transport lacks exact SEND→RECV actions",
                              path=f"training_moe_transport[{flow.id}]")
        source, target = mapping.get(sends[0].id), mapping.get(recvs[0].id)
        if (source is None or target is None
                or records.get(source) is not RecordOpcode.DTE_SEND
                or records.get(target) is not RecordOpcode.DTE_RECV):
            raise SchemaError("P2 SEND→RECV has no actual cloned DTE carrier",
                              path=f"training_moe_transport[{flow.id}]")
        edges.append((source, target))
    return tuple(sorted(set(edges)))


def derive_moe_training_state_version_edges(
    before: NamespacedMoeTrainingUnit,
    after: NamespacedMoeTrainingUnit,
) -> tuple[tuple[str, str], ...]:
    """Tie each step0 physical parameter STORE to step1 LOAD at the same home."""
    if (before.step, after.step) != (0, 1) or before.layer != after.layer:
        raise SchemaError("version edge needs SGD step0→1 of the same MoE layer",
                          path="training_moe_state_version")

    def operands(named: NamespacedMoeTrainingUnit, wanted: RecordOpcode):
        fragment_map = {fragment.id: fragment for fragment in named.fragments}
        witnessed = {}
        for binding in named.state_bindings:
            fragment = fragment_map[binding.fragment_id]
            local = next(stream for stream in fragment.core_streams
                         if stream.logical_core == binding.logical_core)
            record = local.records[binding.fragment_record_index]
            if record.opcode is not wanted:
                continue
            abi = next(state for state in fragment.state_abi
                       if state.id == binding.state_abi_id)
            key = (abi.die_id, abi.state_ref)
            if key in witnessed:
                raise SchemaError("one parameter needs one source HBM closure per step",
                                  path=f"training_moe_state_version[{key}]")
            witnessed[key] = (abi, record.source_global_action_id)
        return witnessed

    old = operands(before, RecordOpcode.LSU_STORE)
    new = operands(after, RecordOpcode.LSU_LOAD)
    if len(old) != 4 or set(old) != set(new):
        raise SchemaError("every Die/layer expert+gate parameter needs STORE→LOAD",
                          path="training_moe_state_version")
    edges = []
    for key in sorted(old):
        old_abi, source = old[key]
        new_abi, target = new[key]
        if (old_abi.id != new_abi.id
                or (old_abi.address, old_abi.size_bytes, old_abi.dtype,
                    old_abi.shape, old_abi.access)
                != (new_abi.address, new_abi.size_bytes, new_abi.dtype,
                    new_abi.shape, new_abi.access)):
            raise SchemaError("SGD parameter home or typed shape changed between steps",
                              path=f"training_moe_state_version[{key}]")
        edges.append((source, target))
    return tuple(sorted(edges))


def namespace_moe_training_unit(
    unit: MoeCompileUnit,
    *,
    timeline_source_id: str,
    hbm_homes: tuple[HbmAddressSpace, HbmAddressSpace],
    sram_capacity_bytes: int,
) -> NamespacedMoeTrainingUnit:
    """Clone the 4 original units with unique actions and stable SGD homes.

    Source SRAM input/comm region bases 4096/40960 move to 8192/45056 so
    full Dense training tape occupies 0..8192.  Layer1 adds 8192 to BufferABI
    offsets/actual ALLOC literals/relocations; step1 reuses physical SRAM only
    after step0 frees while every action/symbol/BufferABI identity is unique.
    A canonical StateABI is shared between steps at the same layer/die home,
    with a later versioned read/write dependency checked by the main merger.
    """
    unit.validate("training_namespace_source_unit")
    manifest = unit.linked_manifest
    if not timeline_source_id or unit.layer not in (0, 1) or unit.step not in (0, 1):
        raise SchemaError("MoE namespace requires one valid two-step timeline",
                          path="timeline_source_id")
    if len(hbm_homes) != 2 or {home.die_id for home in hbm_homes} != {0, 1}:
        raise SchemaError("MoE namespace requires both finite die HBM homes",
                          path="hbm_homes")
    for home in hbm_homes:
        home.validate("training_namespace_hbm_home")
    if sram_capacity_bytes <= _DENSE_SRAM_PARTITION:
        raise SchemaError("SRAM cannot place shared tape plus two MoE layers",
                          path="sram_capacity_bytes")
    cut = cut_moe_training_unit(unit.plan, manifest)
    physical_id = lambda kind, old, *, layer=True, step=True: stable_artifact_id(
        kind,
        {"timeline": timeline_source_id,
         "layer": unit.layer if layer else None,
         "step": unit.step if step else None,
         "old": old},
        schema_version=_SCHEMA,
    )
    old_buffers = {abi.id: abi for fragment in manifest.fragments
                   for abi in fragment.buffer_abi}
    buffer_map, buffer_sources = {}, {}
    for abi in old_buffers.values():
        value = f"moe_full_training.step{unit.step}.layer{unit.layer}.{abi.value_id}"
        replacement = replace(
            abi,
            id=physical_id("moe_full_training_buffer", abi.id),
            schedule_id=physical_id("moe_full_training_schedule", abi.schedule_id),
            binding_id=physical_id("moe_full_training_binding", abi.binding_id),
            value_id=value,
            tensor_slice=replace(abi.tensor_slice, value_id=value),
            region_offset_bytes=abi.region_offset_bytes
                                + unit.layer * _SRAM_MOE_LAYER_STRIDE,
            storage_id=physical_id("moe_full_training_storage", abi.storage_id),
            alias_of=(physical_id("moe_full_training_binding", abi.alias_of)
                      if abi.alias_of else None),
        )
        replacement.validate("moe_full_training_buffer")
        buffer_map[abi.id] = replacement
        buffer_sources[abi.binding_id] = replacement.binding_id
        buffer_sources[abi.storage_id] = replacement.storage_id

    actions = {record.source_global_action_id
               for fragment in manifest.fragments for stream in fragment.core_streams
               for record in stream.records}
    action_map = {action: physical_id("moe_full_training_action", action)
                  for action in actions}
    runtime_map = {symbol.id: physical_id("moe_full_training_runtime", symbol.id)
                   for fragment in manifest.fragments for symbol in fragment.runtime_symbols}
    program_map = {symbol.id: physical_id(
        "moe_full_training_program", symbol.id,
        layer=(symbol.kind is not ProgramSymbolKind.SRAM_REGION),
        step=(symbol.kind is not ProgramSymbolKind.SRAM_REGION),
    ) for fragment in manifest.fragments for symbol in fragment.program_symbols}
    clones, fragment_ids, state_ids, states_by_hbm = [], {}, {}, {}
    for fragment in manifest.fragments:
        clone, indices, source_states, shifted = _clone_moe_fragment(
            fragment, timeline_source_id, unit.layer,
            action_map, runtime_map, program_map, buffer_map, buffer_sources,
        )
        # Keep trainable weight StateABI identity and address stable across
        # steps; the ordinary inference helper already namespaces by layer.
        homes = {home.die_id: home for home in hbm_homes}
        replacement_states = []
        candidate_to_canonical = {}
        for source_abi in fragment.state_abi:
            candidate = shifted[source_abi.hbm_binding_ref]
            home = homes[source_abi.die_id]
            address = (home.base_address + _HBM_REGION_SHIFT
                       + unit.layer * _HBM_MOE_LAYER_STRIDE + source_abi.address)
            if (source_abi.address + source_abi.size_bytes > _HBM_MOE_LAYER_STRIDE
                    or address + source_abi.size_bytes >
                    home.base_address + home.size_bytes):
                raise SchemaError("actual MoE tensor exceeds finite global die HBM",
                                  path=f"hbm_homes[{source_abi.id}]")
            abi = StateABI.create(
                state_ref=physical_id("moe_full_training_state", source_abi.state_ref,
                                      step=False),
                hbm_binding_ref=physical_id("moe_full_training_hbm_binding",
                                             source_abi.hbm_binding_ref, step=False),
                kind=source_abi.kind, lifetime=source_abi.lifetime,
                access=source_abi.access, shape=source_abi.shape,
                dtype=source_abi.dtype, layout=source_abi.layout,
                die_id=source_abi.die_id, address=address,
                size_bytes=source_abi.size_bytes,
                alignment_bytes=source_abi.alignment_bytes,
            )
            replacement_states.append(abi)
            source_states[source_abi.id] = abi.id
            candidate_to_canonical[candidate.hbm_binding_ref] = abi.hbm_binding_ref
            shifted[source_abi.hbm_binding_ref] = abi
        # The source clone has already rebound its HBM symbols to the old
        # candidate; give every symbol the canonical finite-home binding.
        clone = CommandFragment.create(
            producer_pass=clone.producer_pass,
            source_global_dag_id=clone.source_global_dag_id,
            kind=clone.kind, claimed_action_ids=clone.claimed_action_ids,
            core_streams=clone.core_streams,
            runtime_symbols=clone.runtime_symbols,
            program_symbols=tuple(sorted((
                replace(symbol,
                        source_ref=candidate_to_canonical.get(symbol.source_ref,
                                                              symbol.source_ref))
                for symbol in clone.program_symbols
            ), key=lambda symbol: symbol.id)),
            buffer_abi=clone.buffer_abi,
            state_abi=tuple(sorted(replacement_states, key=lambda state: state.id)),
        )
        clone.validate("moe_training_layer_clone")
        clones.append(clone)
        fragment_ids[fragment.id] = (clone.id, indices)
        state_ids.update(source_states)
        states_by_hbm.update(shifted)

    def mapped_stream(stream: LinkedCoreStream) -> LinkedCoreStream:
        return replace(stream, records=tuple(replace(
            ref,
            fragment_id=fragment_ids[ref.fragment_id][0],
            fragment_record_index=fragment_ids[ref.fragment_id][1][ref.fragment_record_index],
            source_global_action_id=action_map[ref.source_global_action_id],
        ) for ref in stream.records))

    runtime_defs, program_defs = [], []
    for definition in manifest.runtime_symbol_definitions:
        if definition.symbol.kind is RuntimeSymbolKind.START_TAG:
            continue  # The eventual single timeline issues one fresh start/core.
        runtime_defs.append(replace(
            definition,
            symbol=replace(definition.symbol, id=runtime_map[definition.symbol.id]),
            source_action_id=action_map.get(definition.source_action_id,
                                            definition.source_action_id),
            destination_action_id=action_map.get(definition.destination_action_id,
                                                 definition.destination_action_id),
        ))
    for definition in manifest.program_symbol_definitions:
        symbol = replace(definition.symbol,
                         id=program_map[definition.symbol.id],
                         source_ref=buffer_sources.get(definition.symbol.source_ref,
                                                       definition.symbol.source_ref))
        adjusted = replace(definition, symbol=symbol)
        if symbol.kind is ProgramSymbolKind.SRAM_REGION:
            adjusted = replace(adjusted, value=adjusted.value + _SRAM_REGION_SHIFT)
            if adjusted.value + adjusted.size_bytes > sram_capacity_bytes:
                raise SchemaError("MoE physical SRAM region exceeds actual hardware",
                                  path=definition.symbol.id)
        elif symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS:
            state = states_by_hbm.get(definition.symbol.source_ref)
            if state is not None:
                adjusted = replace(adjusted,
                                   symbol=replace(symbol, source_ref=state.hbm_binding_ref),
                                   value=state.address, size_bytes=state.size_bytes)
            elif definition.symbol.source_ref in buffer_sources:
                adjusted = replace(adjusted, value=adjusted.value + _SRAM_REGION_SHIFT
                                   + unit.layer * _SRAM_MOE_LAYER_STRIDE)
        program_defs.append(adjusted)
    address_bindings = tuple(replace(
        binding,
        fragment_id=fragment_ids[binding.fragment_id][0],
        fragment_record_index=fragment_ids[binding.fragment_id][1]
            [binding.fragment_record_index],
        buffer_abi_ids=tuple(buffer_map[item].id for item in binding.buffer_abi_ids),
        tensor_slices=tuple(replace(item, value_id=buffer_map[abi_id].value_id)
                            for item, abi_id in zip(binding.tensor_slices,
                                                    binding.buffer_abi_ids)),
    ) for binding in manifest.address_operand_bindings)
    state_bindings = tuple(replace(
        binding,
        fragment_id=fragment_ids[binding.fragment_id][0],
        fragment_record_index=fragment_ids[binding.fragment_id][1]
            [binding.fragment_record_index],
        state_abi_id=state_ids[binding.state_abi_id],
    ) for binding in manifest.state_operand_bindings)
    return NamespacedMoeTrainingUnit(
        unit.step, unit.layer, unit.id, manifest.id,
        tuple(sorted(clones, key=lambda fragment: fragment.id)),
        tuple(mapped_stream(stream) for stream in cut.forward),
        tuple(mapped_stream(stream) for stream in cut.backward),
        tuple(runtime_defs), tuple(program_defs), address_bindings,
        state_bindings, tuple(sorted(action_map.items())),
    )


__all__ = ["NamespacedMoeTrainingUnit", "namespace_moe_training_unit",
           "derive_moe_training_transport_edges",
           "derive_moe_training_state_version_edges"]
