"""Link a Dense spine and per-layer Flexible-MoE programs on one timeline."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressOperandBinding,
    AddressRelocation,
    BufferABI,
    CommandFragment,
    CoreFragmentStream,
    CoreRuntimeBinding,
    EmptyCoreAckPolicy,
    EventCredit,
    FragmentInterface,
    FragmentKind,
    LinkedCoreStream,
    LinkedProgramManifest,
    LinkedRecordRef,
    LogicalStartEvent,
    ManifestInputDigest,
    ManifestInputKind,
    OperandKind,
    ProgramControlEnvelope,
    ProgramFailurePolicy,
    ProgramSymbol,
    ProgramSymbolDefinition,
    ProgramSymbolKind,
    RecordOpcode,
    RecordOperand,
    RelocatableRecord,
    RuntimeOperandField,
    RuntimeRelocation,
    RuntimeSymbol,
    RuntimeSymbolDefinition,
    RuntimeSymbolKind,
    SemanticOperandId,
    StateABI,
    StateOperandBinding,
)
from ..schema.common import stable_artifact_id
from ..schema.global_action import LogicalCoreRef
from ..schema.serde import canonical_digest
from ..schema.global_action import GLOBAL_ACTION_DAG_SCHEMA_VERSION
from ..schema.ir1 import IR1_SCHEMA_VERSION
from ..schema.ir2 import (
    IR2_PROJECTION_RESULT_SCHEMA_VERSION,
    INTRA_DIE_SCHEDULE_SET_SCHEMA_VERSION,
)


_PASS = "moe_full_model_region_linker"
_SCHEMA = "wafer_frontend.moe_full_model_region_linker/v1alpha1"
_HBM_LAYER_STRIDE = 1 << 20
_HBM_MOE_BASE = 1 << 24
_HBM_DIE_STRIDE = 1 << 30
_SRAM_MOE_LAYER_STRIDE = 8192
_DENSE_SRAM_PARTITION_BYTES = 4096


def _origin_node_ref(action) -> str | None:
    origin = action.origin_ref
    return getattr(origin, "op_id", getattr(origin, "node_ref", None))


def _filter_fragment(
    fragment: CommandFragment,
    source_id: str,
    removed_actions: set[str],
) -> tuple[CommandFragment | None, dict[int, int], dict[str, str]]:
    streams = []
    record_map: dict[int, int] = {}
    used_runtime: set[str] = set()
    used_program: set[str] = set()
    used_state_hbm: set[str] = set()
    for stream in fragment.core_streams:
        kept = []
        index_map = {}
        for old_index, record in enumerate(stream.records):
            if record.source_global_action_id in removed_actions:
                continue
            index_map[old_index] = len(kept)
            kept.append(record)
            for operand in record.operands:
                if operand.kind is OperandKind.RUNTIME_SYMBOL:
                    assert operand.symbol_ref is not None
                    used_runtime.add(operand.symbol_ref)
                elif operand.kind is OperandKind.ADDRESS_SYMBOL:
                    assert operand.symbol_ref is not None
                    used_program.add(operand.symbol_ref)
        if not kept:
            continue
        record_map.update(index_map)
        runtime = tuple(
            replace(item, record_index=index_map[item.record_index])
            for item in stream.runtime_relocations
            if item.record_index in index_map
        )
        address = tuple(
            replace(item, record_index=index_map[item.record_index])
            for item in stream.address_relocations
            if item.record_index in index_map
        )
        for item in address:
            if item.operand_id is SemanticOperandId.HBM_ADDRESS:
                symbol = next(s for s in fragment.program_symbols if s.id == item.symbol_ref)
                used_state_hbm.add(symbol.source_ref)
        streams.append(replace(stream, records=tuple(kept), runtime_relocations=runtime, address_relocations=address))
    if not streams:
        return None, {}, {}
    state_abi = tuple(item for item in fragment.state_abi if item.hbm_binding_ref in used_state_hbm)
    result = CommandFragment.create(
        producer_pass=fragment.producer_pass,
        source_global_dag_id=source_id,
        kind=fragment.kind,
        claimed_action_ids=tuple(sorted({record.source_global_action_id for stream in streams for record in stream.records})),
        core_streams=tuple(streams),
        runtime_symbols=tuple(item for item in fragment.runtime_symbols if item.id in used_runtime),
        program_symbols=tuple(item for item in fragment.program_symbols if item.id in used_program),
        buffer_abi=fragment.buffer_abi,
        state_abi=state_abi,
    )
    result.validate("filtered_dense_fragment")
    return result, record_map, {item.id: item.id for item in state_abi}


def _clone_moe_fragment(
    fragment: CommandFragment,
    source_id: str,
    layer: int,
    action_map: dict[str, str],
    runtime_map: dict[str, str],
    program_map: dict[str, str],
    buffer_map: dict[str, BufferABI],
    buffer_source_map: dict[str, str],
) -> tuple[CommandFragment, dict[int, int], dict[str, str], dict[str, StateABI]]:
    state_map: dict[str, str] = {}
    states: dict[str, StateABI] = {}
    shifted = []
    for abi in fragment.state_abi:
        hbm_binding_ref = stable_artifact_id(
            "moe_full_model_hbm_binding",
            {"source": source_id, "layer": layer, "binding": abi.hbm_binding_ref},
            schema_version=_SCHEMA,
        )
        replacement = StateABI.create(
            state_ref=stable_artifact_id("moe_full_model_state", {"source": source_id, "layer": layer, "state": abi.state_ref}, schema_version=_SCHEMA),
            hbm_binding_ref=hbm_binding_ref,
            kind=abi.kind,
            lifetime=abi.lifetime,
            access=abi.access,
            shape=abi.shape,
            dtype=abi.dtype,
            layout=abi.layout,
            die_id=abi.die_id,
            address=abi.die_id * _HBM_DIE_STRIDE + _HBM_MOE_BASE + layer * _HBM_LAYER_STRIDE + abi.address,
            size_bytes=abi.size_bytes,
            alignment_bytes=abi.alignment_bytes,
        )
        state_map[abi.id] = replacement.id
        states[abi.hbm_binding_ref] = replacement
        shifted.append(replacement)
    streams = tuple(CoreFragmentStream(
        stream.logical_core,
        tuple(RelocatableRecord(
            action_map[record.source_global_action_id],
            record.opcode,
            tuple(
                replace(operand, literal_value=operand.literal_value + layer * _SRAM_MOE_LAYER_STRIDE)
                if record.opcode is RecordOpcode.SRAM_ALLOC_AT and operand.name == "region_offset_bytes"
                else replace(
                    operand,
                    symbol_ref=(
                        runtime_map[operand.symbol_ref]
                        if operand.kind is OperandKind.RUNTIME_SYMBOL
                        else program_map[operand.symbol_ref]
                    ),
                ) if operand.kind is not OperandKind.LITERAL else operand
                for operand in record.operands
            ),
        ) for record in stream.records),
        tuple(replace(item, symbol_ref=runtime_map[item.symbol_ref]) for item in stream.runtime_relocations),
        tuple(replace(item, symbol_ref=program_map[item.symbol_ref]) for item in stream.address_relocations),
    ) for stream in fragment.core_streams)
    result = CommandFragment.create(
        producer_pass=fragment.producer_pass,
        source_global_dag_id=source_id,
        kind=fragment.kind,
        claimed_action_ids=tuple(sorted(action_map[item] for item in fragment.claimed_action_ids)),
        core_streams=streams,
        runtime_symbols=tuple(sorted((replace(item, id=runtime_map[item.id]) for item in fragment.runtime_symbols), key=lambda item: item.id)),
        program_symbols=tuple(sorted((replace(
            item,
            id=program_map[item.id],
            source_ref=(
                states[item.source_ref].hbm_binding_ref if item.source_ref in states
                else buffer_source_map.get(item.source_ref, item.source_ref)
            ),
        ) for item in fragment.program_symbols), key=lambda item: item.id)),
        buffer_abi=tuple(sorted((buffer_map[item.id] for item in fragment.buffer_abi), key=lambda item: item.id)),
        state_abi=tuple(sorted(shifted, key=lambda item: item.id)),
    )
    result.validate("rebased_moe_fragment")
    indices = {
        index: index for stream in fragment.core_streams for index in range(len(stream.records))
    }
    return result, indices, state_map, states


def _clone_moe_buffers(source_id, layer, manifest):
    unique = {abi.id: abi for fragment in manifest.fragments for abi in fragment.buffer_abi}
    buffer_map = {}
    source_map = {}
    for abi in unique.values():
        value_id = f"moe_full_model.layer{layer}.{abi.value_id}"
        identity = lambda kind, item: stable_artifact_id(
            kind, {"source": source_id, "layer": layer, "old": item}, schema_version=_SCHEMA,
        )
        replacement = replace(
            abi,
            id=identity("moe_full_model_buffer_abi", abi.id),
            schedule_id=identity("moe_full_model_schedule", abi.schedule_id),
            binding_id=identity("moe_full_model_buffer_binding", abi.binding_id),
            value_id=value_id,
            tensor_slice=replace(abi.tensor_slice, value_id=value_id),
            region_offset_bytes=abi.region_offset_bytes + layer * _SRAM_MOE_LAYER_STRIDE,
            storage_id=identity("moe_full_model_buffer_storage", abi.storage_id),
            alias_of=(identity("moe_full_model_buffer_binding", abi.alias_of) if abi.alias_of else None),
        )
        replacement.validate("moe_full_model_buffer_abi")
        buffer_map[abi.id] = replacement
        source_map[abi.binding_id] = replacement.binding_id
        source_map[abi.storage_id] = replacement.storage_id
    return buffer_map, source_map


def _bridge_fragment(source_id: str, layer_buffers, lifecycles):
    core = LogicalCoreRef(0, 0)
    records = []
    runtime_symbols = []
    program_symbols = []
    runtime_relocations = []
    address_relocations = []
    address_bindings = []
    runtime_definitions = []
    program_definitions = []
    buffers = {}
    refs = {}
    for layer, dense_input, moe_input, moe_output, dense_output, absolute in layer_buffers:
        refs[layer] = {"input": [], "output": []}
        for role, source, destination in (
            ("input", dense_input, moe_input),
            ("output", moe_output, dense_output),
        ):
            action_id = stable_artifact_id("moe_full_model_bridge_action", {"source": source_id, "layer": layer, "role": role}, schema_version=_SCHEMA)
            token = RuntimeSymbol(
                stable_artifact_id("moe_full_model_bridge_token", {"action": action_id}, schema_version=_SCHEMA),
                RuntimeSymbolKind.DTE_TOKEN,
                action_id,
            )
            runtime_symbols.append(token)
            runtime_definitions.append(RuntimeSymbolDefinition(token, (core,), action_id, None))
            symbols = []
            for endpoint, abi in (("source", source), ("destination", destination)):
                symbol = ProgramSymbol(
                    stable_artifact_id("moe_full_model_bridge_address", {"action": action_id, "endpoint": endpoint, "binding": abi.binding_id}, schema_version=_SCHEMA),
                    ProgramSymbolKind.ABSOLUTE_ADDRESS,
                    abi.binding_id,
                )
                symbols.append(symbol)
                program_symbols.append(symbol)
                program_definitions.append(ProgramSymbolDefinition(
                    symbol,
                    f"moe_full_model.layer{layer}.{role}.{endpoint}",
                    absolute(abi),
                    abi.size_bytes,
                    (core,),
                ))
                buffers[abi.id] = abi
            size = min(source.size_bytes, destination.size_bytes)
            lifecycle_record, lifecycle_relocs, lifecycle_symbols = lifecycles[layer][role]
            for symbol in lifecycle_symbols:
                program_symbols.append(symbol)
            if role == "output":
                start = len(records)
                records.append(replace(lifecycle_record, source_global_action_id=action_id))
                address_relocations.extend(replace(item, record_index=start) for item in lifecycle_relocs)
                address_bindings.extend(AddressOperandBinding(
                    "__bridge__", core, start, item.operand_id,
                    (destination.id,), (destination.tensor_slice,),
                ) for item in lifecycle_relocs)
            index = len(records)
            records.extend((
                RelocatableRecord(action_id, RecordOpcode.DTE_ISSUE, (
                    RecordOperand.literal("direction", 0),
                    RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),
                    RecordOperand.literal("payload_bits", size * 8),
                    RecordOperand.literal("size_bytes", size),
                    RecordOperand.literal("hbm_address", 0),
                    RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, symbols[0].id),
                    RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, symbols[1].id),
                )),
                RelocatableRecord(action_id, RecordOpcode.DTE_WAIT, (
                    RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),
                )),
            ))
            runtime_relocations.extend((
                RuntimeRelocation(index, RuntimeOperandField.DTE_TOKEN, token.id),
                RuntimeRelocation(index + 1, RuntimeOperandField.DTE_TOKEN, token.id),
            ))
            for operand_id, symbol, abi in (
                (SemanticOperandId.SOURCE_ADDRESS, symbols[0], source),
                (SemanticOperandId.DESTINATION_ADDRESS, symbols[1], destination),
            ):
                address_relocations.append(AddressRelocation(index, operand_id, ProgramSymbolKind.ABSOLUTE_ADDRESS, symbol.id, 0))
                address_bindings.append(AddressOperandBinding(
                    "__bridge__", core, index, operand_id, (abi.id,), (abi.tensor_slice,),
                ))
            if role == "input":
                end = len(records)
                records.append(replace(lifecycle_record, source_global_action_id=action_id))
                address_relocations.extend(replace(item, record_index=end) for item in lifecycle_relocs)
                address_bindings.extend(AddressOperandBinding(
                    "__bridge__", core, end, item.operand_id,
                    (source.id,), (source.tensor_slice,),
                ) for item in lifecycle_relocs)
            refs[layer][role] = (
                [index, index + 1, end] if role == "input"
                else [start, index, index + 1]
            )
    fragment = CommandFragment.create(
        producer_pass=_PASS,
        source_global_dag_id=source_id,
        kind=FragmentKind.COARSE,
        claimed_action_ids=tuple(sorted({record.source_global_action_id for record in records})),
        core_streams=(CoreFragmentStream(core, tuple(records), tuple(runtime_relocations), tuple(address_relocations)),),
        runtime_symbols=tuple(sorted(runtime_symbols, key=lambda item: item.id)),
        program_symbols=tuple(sorted(set(program_symbols), key=lambda item: item.id)),
        buffer_abi=tuple(sorted(buffers.values(), key=lambda item: item.id)),
        state_abi=(),
    )
    bindings = tuple(replace(item, fragment_id=fragment.id) for item in address_bindings)
    fragment.validate("moe_full_model_bridge_fragment")
    return fragment, refs, tuple(runtime_definitions), tuple(program_definitions), bindings


def _interfaces(fragments: tuple[CommandFragment, ...]) -> tuple[FragmentInterface, ...]:
    program_owners = defaultdict(list)
    runtime_owners = defaultdict(list)
    for fragment in fragments:
        for symbol in fragment.program_symbols:
            program_owners[symbol.id].append(fragment.id)
        for symbol in fragment.runtime_symbols:
            runtime_owners[symbol.id].append(fragment.id)
    result = []
    for fragment in fragments:
        program = tuple(item.id for item in fragment.program_symbols)
        runtime = tuple(item.id for item in fragment.runtime_symbols)
        entries = Counter()
        exits = Counter()
        for stream in fragment.core_streams:
            for record in stream.records:
                operands = {item.name: item for item in record.operands}
                if record.opcode is RecordOpcode.EVENT_SET:
                    exits[operands["tag"].symbol_ref] += 1
                elif record.opcode is RecordOpcode.EVENT_WAIT:
                    entries[operands["tag"].symbol_ref] += operands["count"].literal_value
        p_exports = tuple(sorted(item for item in program if fragment.id == min(program_owners[item])))
        r_exports = tuple(sorted(item for item in runtime if fragment.id == min(runtime_owners[item])))
        result.append(FragmentInterface(
            fragment.id,
            tuple(sorted(set(runtime) - set(r_exports))), r_exports,
            tuple(sorted(set(program) - set(p_exports))), p_exports,
            tuple(EventCredit(key, entries[key]) for key in sorted(entries)),
            tuple(EventCredit(key, exits[key]) for key in sorted(exits)),
        ))
    return tuple(sorted(result, key=lambda item: item.fragment_id))


def link_moe_full_model_segment(profile, units, replaced_node_refs: tuple[str, ...]) -> LinkedProgramManifest:
    """Replace Dense MLP records with layer MoE records and bridge their SRAM values."""
    dense = profile.manifest
    actions = {item.id: item for item in profile.lowering_context.global_dag.actions}
    removed = {item.id for item in actions.values() if _origin_node_ref(item) in set(replaced_node_refs)}
    if not removed:
        raise SchemaError("Dense MLP replacement selected no executable actions", path="replaced_node_refs")
    source_id = stable_artifact_id("moe_full_model_timeline", {
        "dense": dense.id,
        "moe": tuple(unit.linked_manifest.id for unit in units),
        "removed": tuple(sorted(removed)),
    }, schema_version=_SCHEMA)
    fragments = []
    clone_maps = {}
    state_maps = {}
    source_manifests = [("dense", dense, None)] + [
        (f"moe:{unit.layer}", unit.linked_manifest, unit.layer) for unit in units
    ]
    namespaces = {}
    for key, manifest, layer in source_manifests:
        if layer is not None:
            source_actions = {
                record.source_global_action_id
                for fragment in manifest.fragments for stream in fragment.core_streams
                for record in stream.records
            }
            action_map = {item: stable_artifact_id("moe_full_model_action", {"source": source_id, "layer": layer, "action": item}, schema_version=_SCHEMA) for item in source_actions}
            runtime_map = {item.id: stable_artifact_id("moe_full_model_runtime_symbol", {"source": source_id, "layer": layer, "symbol": item.id}, schema_version=_SCHEMA) for fragment in manifest.fragments for item in fragment.runtime_symbols}
            program_map = {
                item.id: (
                    stable_artifact_id("moe_full_model_program_symbol", {"source": source_id, "symbol": item.id}, schema_version=_SCHEMA)
                    if item.kind is ProgramSymbolKind.SRAM_REGION
                    else stable_artifact_id("moe_full_model_program_symbol", {"source": source_id, "layer": layer, "symbol": item.id}, schema_version=_SCHEMA)
                )
                for fragment in manifest.fragments for item in fragment.program_symbols
            }
            buffer_map, buffer_source_map = _clone_moe_buffers(source_id, layer, manifest)
            namespaces[key] = (action_map, runtime_map, program_map, buffer_map, buffer_source_map)
        for fragment in manifest.fragments:
            if type(fragment) is not CommandFragment:
                raise SchemaError("full-model linker requires leaf CommandFragments", path="fragments")
            if layer is None:
                clone, indices, states = _filter_fragment(fragment, source_id, removed)
                shifted_by_hbm = {}
            else:
                clone, indices, states, shifted_by_hbm = _clone_moe_fragment(
                    fragment, source_id, layer, *namespaces[key]
                )
            if clone is None:
                continue
            fragments.append(clone)
            clone_maps[(key, fragment.id)] = (clone.id, indices)
            state_maps[(key, fragment.id)] = (states, shifted_by_hbm)

    dense_abis = {abi.value_id: abi for fragment in profile.leaf_fragments for abi in fragment.buffer_abi}
    unit_abis = {}
    for unit in units:
        unit_abis[unit.layer] = {
            (abi.logical_core.die_id, abi.value_id.rsplit(".", 1)[-1]): namespaces[f"moe:{unit.layer}"][3][abi.id]
            for fragment in unit.linked_manifest.fragments for abi in fragment.buffer_abi
        }
    definitions_by_symbol = {}
    for _key, manifest, _layer in source_manifests:
        for definition in manifest.program_symbol_definitions:
            definitions_by_symbol.setdefault(definition.symbol.id, definition)
    def absolute(abi):
        candidates = [
            definition for definition in definitions_by_symbol.values()
            if definition.symbol.kind is ProgramSymbolKind.SRAM_REGION
            and definition.symbol.source_ref == abi.region_ref
            and abi.logical_core in definition.logical_cores
        ]
        if len(candidates) != 1:
            raise SchemaError("buffer has no exact SRAM region definition", path=abi.id)
        return candidates[0].value + abi.region_offset_bytes
    layer_buffers = []
    origin = profile.lowering_context.ir1.instances[0].origin_instance_id
    for unit in units:
        try:
            layer_buffers.append((
                unit.layer,
                dense_abis[f"{origin}.layer{unit.layer}.norm2_out"],
                unit_abis[unit.layer][(0, "activation")],
                unit_abis[unit.layer][(0, "output")],
                dense_abis[f"{origin}.layer{unit.layer}.down_out"],
                absolute,
            ))
        except KeyError as error:
            raise SchemaError("MoE bridge buffer is absent", path=f"layer[{unit.layer}]") from error
    # Replacing an MLP also removes the norm2 input FREE and the down output
    # ALLOC.  Carry their original records and relocations onto the physical
    # SRAM bridge, preserving the Dense allocation/free lifecycle exactly.
    lifecycle_targets = {
        unit.layer: {
            "input": (dense_abis[f"{origin}.layer{unit.layer}.norm2_out"].storage_id, RecordOpcode.SRAM_FREE),
            "output": (dense_abis[f"{origin}.layer{unit.layer}.down_out"].storage_id, RecordOpcode.SRAM_ALLOC_AT),
        } for unit in units
    }
    lifecycles = {layer: {} for layer in lifecycle_targets}
    for fragment in dense.fragments:
        symbols = {item.id: item for item in fragment.program_symbols}
        for stream in fragment.core_streams:
            for index, record in enumerate(stream.records):
                if record.source_global_action_id not in removed:
                    continue
                for layer, targets in lifecycle_targets.items():
                    for role, (storage, opcode) in targets.items():
                        if record.opcode is not opcode or not any(
                            item.symbol_ref in symbols and symbols[item.symbol_ref].source_ref == storage
                            for item in record.operands if item.symbol_ref is not None
                        ):
                            continue
                        if role in lifecycles[layer]:
                            raise SchemaError("Dense MLP boundary has multiple SRAM lifecycle records", path=f"layer[{layer}].{role}")
                        relocs = tuple(item for item in stream.address_relocations if item.record_index == index)
                        lifecycles[layer][role] = (
                            record, relocs, tuple(symbols[item.symbol_ref] for item in relocs),
                        )
    for layer, targets in lifecycles.items():
        if set(targets) != {"input", "output"}:
            raise SchemaError("Dense MLP boundary lacks exact SRAM lifecycle records", path=f"layer[{layer}]")
    bridge, bridge_refs, bridge_runtime, bridge_program, bridge_bindings = _bridge_fragment(source_id, layer_buffers, lifecycles)
    fragments.append(bridge)
    fragments = tuple(sorted(fragments, key=lambda item: item.id))

    address_bindings = list(bridge_bindings)
    state_bindings = []
    runtime_definitions = {item.symbol.id: item for item in bridge_runtime}
    program_definitions = {item.symbol.id: item for item in bridge_program}
    for key, manifest, layer in source_manifests:
        action_map, runtime_map, program_map, buffer_map, buffer_source_map = namespaces.get(key, ({}, {}, {}, {}, {}))
        for definition in manifest.runtime_symbol_definitions:
            if definition.symbol.kind is not RuntimeSymbolKind.START_TAG:
                adjusted = definition
                if layer is not None:
                    adjusted = replace(
                        definition,
                        symbol=replace(definition.symbol, id=runtime_map[definition.symbol.id]),
                        source_action_id=action_map.get(definition.source_action_id, definition.source_action_id),
                        destination_action_id=action_map.get(definition.destination_action_id, definition.destination_action_id),
                    )
                previous = runtime_definitions.setdefault(adjusted.symbol.id, adjusted)
                if previous != adjusted:
                    raise SchemaError("runtime symbol collision", path=definition.symbol.id)
        for definition in manifest.program_symbol_definitions:
            adjusted = (
                replace(definition, symbol=replace(
                    definition.symbol, id=program_map[definition.symbol.id],
                    source_ref=buffer_source_map.get(definition.symbol.source_ref, definition.symbol.source_ref),
                ))
                if layer is not None else definition
            )
            if (
                layer is None
                and definition.symbol.kind is ProgramSymbolKind.SRAM_REGION
                and definition.name == "sram"
            ):
                dense_buffers = (
                    abi for fragment in dense.fragments for abi in fragment.buffer_abi
                    if abi.region_ref == definition.symbol.source_ref
                )
                if any(abi.region_offset_bytes + abi.size_bytes > _DENSE_SRAM_PARTITION_BYTES for abi in dense_buffers):
                    raise SchemaError("Dense spine exceeds its disjoint SRAM partition", path="shared_spine_profile.buffer_abi")
                adjusted = replace(adjusted, size_bytes=_DENSE_SRAM_PARTITION_BYTES)
            if layer is not None:
                if (
                    definition.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
                    and definition.symbol.source_ref in buffer_source_map
                ):
                    adjusted = replace(adjusted, value=adjusted.value + layer * _SRAM_MOE_LAYER_STRIDE)
                shifted = {}
                for (state_key, _fragment_id), (_state_map, states) in state_maps.items():
                    if state_key == key:
                        shifted.update(states)
                abi = shifted.get(definition.symbol.source_ref)
                if abi is not None:
                    adjusted = replace(
                        adjusted,
                        symbol=replace(adjusted.symbol, source_ref=abi.hbm_binding_ref),
                        value=abi.address,
                        size_bytes=abi.size_bytes,
                    )
            previous = program_definitions.setdefault(adjusted.symbol.id, adjusted)
            if previous != adjusted:
                raise SchemaError("program symbol collision", path=adjusted.symbol.id)
        for binding in manifest.address_operand_bindings:
            mapped = clone_maps.get((key, binding.fragment_id))
            if mapped is None or binding.fragment_record_index not in mapped[1]:
                continue
            address_bindings.append(replace(
                binding,
                fragment_id=mapped[0],
                fragment_record_index=mapped[1][binding.fragment_record_index],
                buffer_abi_ids=(
                    tuple(buffer_map[item].id for item in binding.buffer_abi_ids)
                    if layer is not None else binding.buffer_abi_ids
                ),
                tensor_slices=(
                    tuple(replace(item, value_id=buffer_map[abi_id].value_id)
                          for item, abi_id in zip(binding.tensor_slices, binding.buffer_abi_ids))
                    if layer is not None else binding.tensor_slices
                ),
            ))
        for binding in manifest.state_operand_bindings:
            mapped = clone_maps.get((key, binding.fragment_id))
            if mapped is None or binding.fragment_record_index not in mapped[1]:
                continue
            state_map, _shifted = state_maps[(key, binding.fragment_id)]
            state_bindings.append(replace(
                binding,
                fragment_id=mapped[0],
                fragment_record_index=mapped[1][binding.fragment_record_index],
                state_abi_id=state_map[binding.state_abi_id],
            ))

    declared_program = {
        symbol.id for fragment in fragments for symbol in fragment.program_symbols
    }
    declared_runtime = {
        symbol.id for fragment in fragments for symbol in fragment.runtime_symbols
    }
    program_definitions = {
        key: value for key, value in program_definitions.items()
        if key in declared_program
    }
    runtime_definitions = {
        key: value for key, value in runtime_definitions.items()
        if key in declared_runtime
    }
    # Program names are an artifact-wide namespace even when symbol ids differ.
    name_counts = Counter(item.name for item in program_definitions.values())
    program_definitions = {
        key: (replace(value, name=f"{value.name}.{key[-8:]}") if name_counts[value.name] > 1 else value)
        for key, value in program_definitions.items()
    }
    unit_cores = tuple(
        sorted(
            {stream.logical_core for stream in units[0].linked_manifest.core_streams},
            key=lambda item: (item.die_id, item.local_core_id),
        )
    )
    if not unit_cores or unit_cores[0] != LogicalCoreRef(0, 0):
        raise SchemaError("MoE units lack rank-zero shared-spine core", path="units")
    if any(
        tuple(sorted(
            {stream.logical_core for stream in unit.linked_manifest.core_streams},
            key=lambda item: (item.die_id, item.local_core_id),
        )) != unit_cores
        for unit in units
    ):
        raise SchemaError("MoE units disagree on active cores", path="units")
    by_core = {core: [] for core in unit_cores}
    dense_stream = dense.core_streams[0]
    inserted = set()
    unit_by_layer = {unit.layer: unit for unit in units}
    for ref in dense_stream.records:
        action = actions[ref.source_global_action_id]
        node_ref = _origin_node_ref(action) or ""
        removed_layer = next((layer for layer in unit_by_layer if f".layer{layer}." in node_ref), None)
        if ref.source_global_action_id in removed:
            if removed_layer not in inserted:
                unit = unit_by_layer[removed_layer]
                by_core[LogicalCoreRef(0, 0)].extend(
                    LinkedRecordRef(bridge.id, index, bridge.core_streams[0].records[index].source_global_action_id)
                    for index in bridge_refs[removed_layer]["input"]
                )
                source_stream = next(item for item in unit.linked_manifest.core_streams if item.logical_core.die_id == 0)
                for item in source_stream.records:
                    clone_id, indices = clone_maps[(f"moe:{removed_layer}", item.fragment_id)]
                    by_core[LogicalCoreRef(0, 0)].append(replace(item, fragment_id=clone_id, fragment_record_index=indices[item.fragment_record_index], source_global_action_id=namespaces[f"moe:{removed_layer}"][0][item.source_global_action_id]))
                by_core[LogicalCoreRef(0, 0)].extend(
                    LinkedRecordRef(bridge.id, index, bridge.core_streams[0].records[index].source_global_action_id)
                    for index in bridge_refs[removed_layer]["output"]
                )
                inserted.add(removed_layer)
            continue
        mapped = clone_maps.get(("dense", ref.fragment_id))
        if mapped is not None:
            by_core[LogicalCoreRef(0, 0)].append(replace(ref, fragment_id=mapped[0], fragment_record_index=mapped[1][ref.fragment_record_index]))
    for unit in units:
        for core in unit_cores[1:]:
            source_stream = next(
                item for item in unit.linked_manifest.core_streams
                if item.logical_core == core
            )
            for item in source_stream.records:
                clone_id, indices = clone_maps[(f"moe:{unit.layer}", item.fragment_id)]
                by_core[core].append(replace(
                    item,
                    fragment_id=clone_id,
                    fragment_record_index=indices[item.fragment_record_index],
                    source_global_action_id=namespaces[f"moe:{unit.layer}"][0][item.source_global_action_id],
                ))

    cores = tuple(by_core)
    unit_bindings = {
        item.logical_core: item
        for item in units[0].linked_manifest.core_bindings
    }
    core_bindings = (
        dense.core_bindings[0],
        *(unit_bindings[core] for core in unit_cores[1:]),
    )
    starts = []
    for core in cores:
        first = by_core[core][0].source_global_action_id
        symbol = RuntimeSymbol(stable_artifact_id("moe_full_model_start", {"source": source_id, "core": core, "first": first}, schema_version=_SCHEMA), RuntimeSymbolKind.START_TAG, first)
        runtime_definitions[symbol.id] = RuntimeSymbolDefinition(symbol, (core,), None, None)
        starts.append(LogicalStartEvent(core, symbol.id, 1))
    upstream = {
        ManifestInputKind.IR1: (profile.lowering_context.ir1, IR1_SCHEMA_VERSION),
    }
    standalone = tuple(
        ManifestInputDigest(
            ManifestInputKind.STANDALONE_PLAN,
            item.id, item.schema_version, canonical_digest(item),
        )
        for item in profile.lowering_context.standalone_plans
        if any(fragment.kind is FragmentKind.STANDALONE_COLLECTIVE for fragment in fragments)
    )
    inputs = tuple(sorted((
        *(ManifestInputDigest(kind, artifact.id, schema, canonical_digest(artifact)) for kind, (artifact, schema) in upstream.items()),
        *(ManifestInputDigest(kind, source_id, schema, canonical_digest({"source": source_id, "dense": dense.id, "moe": tuple(unit.id for unit in units)})) for kind, schema in (
            (ManifestInputKind.IR2_PROJECTION, IR2_PROJECTION_RESULT_SCHEMA_VERSION),
            (ManifestInputKind.SCHEDULE_SET, INTRA_DIE_SCHEDULE_SET_SCHEMA_VERSION),
            (ManifestInputKind.GLOBAL_ACTION_DAG, GLOBAL_ACTION_DAG_SCHEMA_VERSION),
        )),
        *standalone,
        *(ManifestInputDigest(ManifestInputKind.COMMAND_FRAGMENT, fragment.id, fragment.schema_version, canonical_digest(fragment)) for fragment in fragments),
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    result = LinkedProgramManifest.create(
        producer_pass=_PASS,
        capabilities=0,
        source_ir1_id=profile.source_ir1_id,
        source_projection_id=source_id,
        source_schedule_set_id=source_id,
        source_global_dag_id=source_id,
        input_digests=inputs,
        fragments=fragments,
        fragment_interfaces=_interfaces(fragments),
        core_bindings=core_bindings,
        core_streams=tuple(LinkedCoreStream(core, core_bindings[index].runtime_core_id, tuple(by_core[core])) for index, core in enumerate(cores)),
        runtime_symbol_definitions=tuple(sorted(runtime_definitions.values(), key=lambda item: item.symbol.id)),
        program_symbol_definitions=tuple(sorted(program_definitions.values(), key=lambda item: item.symbol.id)),
        address_operand_bindings=tuple(sorted(address_bindings, key=lambda item: (item.logical_core.die_id, item.logical_core.local_core_id, item.fragment_id, item.fragment_record_index, int(item.operand_id)))),
        state_operand_bindings=tuple(sorted(state_bindings, key=lambda item: (item.logical_core.die_id, item.logical_core.local_core_id, item.fragment_id, item.fragment_record_index, int(item.operand_id)))),
        core_groups=(),
        envelope=ProgramControlEnvelope(cores, tuple(starts), cores, cores, cores, EmptyCoreAckPolicy.INCLUDE_EMPTY, ProgramFailurePolicy.ABORT_ALL),
    )
    definition_symbols = {item.symbol.id: item.symbol for item in result.program_symbol_definitions}
    for fragment in fragments:
        for symbol in fragment.program_symbols:
            if definition_symbols.get(symbol.id) != symbol:
                raise SchemaError(
                    f"local program symbol differs from definition: {symbol!r} vs {definition_symbols.get(symbol.id)!r}",
                    path="moe_full_model_runtime_manifest.program_symbol_definitions",
                )
    result.validate("moe_full_model_runtime_manifest")
    return result


__all__ = ["link_moe_full_model_segment"]
