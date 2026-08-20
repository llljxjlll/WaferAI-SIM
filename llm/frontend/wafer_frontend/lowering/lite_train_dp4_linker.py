"""One-manifest linker for the fixed S2-Lite DP4 tree AllReduce."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressOperandBinding,
    BufferABI,
    EmptyCoreAckPolicy,
    LinkedCoreStream,
    LinkedProgramManifest,
    LinkedRecordRef,
    ProgramControlEnvelope,
    ProgramFailurePolicy,
    ProgramSymbolDefinition,
    ProgramSymbolKind,
    RecordOpcode,
    RuntimeSymbolDefinition,
    RuntimeSymbolKind,
    SemanticOperandId,
)
from ..schema.lite_train_dp4_n6 import S2LiteDp4TreeArLoweredProgram
from .linker import NaiveManifestLinker
from .lite_train_rooted_ar_linker import _input_digests, _interfaces, _name


def _overlay_parts(source: S2LiteDp4TreeArLoweredProgram):
    fragments = source.overlay_fragments
    units = {unit.id: unit for unit in source.intent.units}
    abis = {
        abi.id: abi
        for abi in source.intent.gradient_buffer_abis
        + source.intent.scratch_buffer_abis
    }
    symbols = {
        symbol.id: symbol
        for fragment in fragments
        for symbol in fragment.program_symbols
    }
    runtime_symbols = {
        symbol.id: symbol
        for fragment in fragments
        for symbol in fragment.runtime_symbols
    }
    address_bindings = []
    symbol_abis: dict[str, dict[str, BufferABI]] = defaultdict(dict)
    symbol_cores = defaultdict(set)
    runtime_uses = defaultdict(list)
    scratch = source.intent.scratch_buffer_abis
    for fragment in fragments:
        for stream in fragment.core_streams:
            for relocation in stream.runtime_relocations:
                record = stream.records[relocation.record_index]
                runtime_uses[relocation.symbol_ref].append(
                    (record, relocation.field, stream.logical_core)
                )
            for relocation in stream.address_relocations:
                record = stream.records[relocation.record_index]
                symbol = symbols[relocation.symbol_ref]
                if (
                    record.opcode is RecordOpcode.LOCAL_REDUCE
                    and relocation.operand_id is SemanticOperandId.SOURCE_ADDRESS
                ):
                    unit = units[record.source_global_action_id]
                    matched = tuple(abis[item] for item in unit.input_buffer_abi_refs)
                elif relocation.operand_id is SemanticOperandId.REGION_NAME:
                    literals = {
                        operand.name: operand.literal_value
                        for operand in record.operands
                    }
                    matched = tuple(
                        abi
                        for abi in scratch
                        if abi.logical_core == stream.logical_core
                        and abi.region_ref == symbol.source_ref
                        and abi.region_offset_bytes == literals["region_offset_bytes"]
                        and abi.size_bytes == literals["size_bytes"]
                    )
                elif relocation.operand_id in (
                    SemanticOperandId.LABEL_SYMBOL,
                    SemanticOperandId.SYMBOL,
                ):
                    matched = tuple(
                        abi for abi in scratch if abi.storage_id == symbol.source_ref
                    )
                else:
                    matched = tuple(
                        abi for abi in abis.values() if abi.binding_id == symbol.source_ref
                    )
                if not matched:
                    raise SchemaError(
                        "DP4 rooted relocation lacks exact BufferABI closure",
                        path="overlay_fragments",
                    )
                for abi in matched:
                    symbol_abis[symbol.id][abi.id] = abi
                symbol_cores[symbol.id].add(stream.logical_core)
                address_bindings.append(
                    AddressOperandBinding(
                        fragment.id,
                        stream.logical_core,
                        relocation.record_index,
                        relocation.operand_id,
                        tuple(abi.id for abi in matched),
                        tuple(abi.tensor_slice for abi in matched),
                    )
                )

    region_specs = {}
    for context in source.intent.lowering_contexts:
        for die in context.ir1.fabric.dies:
            for core in die.cores:
                profile = next(
                    item
                    for item in context.ir1.fabric.sram_profiles
                    if item.id == core.sram_profile_ref
                )
                region_specs[(die.id, core.local_core_id)] = profile
    definitions = []
    for symbol_id in sorted(symbols):
        symbol = symbols[symbol_id]
        local_abis = tuple(symbol_abis[symbol_id].values())
        cores = tuple(
            sorted(
                symbol_cores[symbol_id],
                key=lambda item: (item.die_id, item.local_core_id),
            )
        )
        if symbol.kind is ProgramSymbolKind.SRAM_LABEL:
            value, size = 0, 0
        elif symbol.kind is ProgramSymbolKind.SRAM_REGION:
            shapes = set()
            for core in cores:
                profile = region_specs[(core.die_id, core.local_core_id)]
                region = next(
                    item for item in profile.regions if item.id == symbol.source_ref
                )
                shapes.add((region.base_bytes, region.size_bytes, region.name))
            if len(shapes) != 1:
                raise SchemaError(
                    "DP4 rooted SRAM region symbol is inconsistent",
                    path="overlay_fragments",
                )
            value, size, region_name = next(iter(shapes))
        else:
            spans = set()
            for abi in local_abis:
                profile = region_specs[
                    (abi.logical_core.die_id, abi.logical_core.local_core_id)
                ]
                region = next(
                    item for item in profile.regions if item.id == abi.region_ref
                )
                spans.add((region.base_bytes + abi.region_offset_bytes, abi.size_bytes))
            if len(spans) == 2:
                ordered = sorted(spans)
                if ordered[0][0] + ordered[0][1] != ordered[1][0]:
                    raise SchemaError(
                        "DP4 reduce source ABI span is not contiguous",
                        path="overlay_fragments",
                    )
                value, size = ordered[0][0], ordered[0][1] + ordered[1][1]
            elif len(spans) == 1:
                value, size = next(iter(spans))
            else:
                raise SchemaError(
                    "DP4 absolute rooted symbol has invalid ABI closure",
                    path="overlay_fragments",
                )
        definitions.append(
            ProgramSymbolDefinition(
                symbol,
                region_name
                if symbol.kind is ProgramSymbolKind.SRAM_REGION
                else _name(symbol),
                value,
                size,
                cores,
            )
        )

    runtime_definitions = []
    for symbol_id in sorted(runtime_symbols):
        symbol = runtime_symbols[symbol_id]
        uses = runtime_uses[symbol_id]
        if symbol.kind is RuntimeSymbolKind.DTE_FSM:
            endpoints = tuple(
                sorted(
                    {core for _record, _field, core in uses},
                    key=lambda item: (item.die_id, item.local_core_id),
                )
            )
            send = next(
                record
                for record, _field, _core in uses
                if record.opcode is RecordOpcode.DTE_SEND
            )
            recv = next(
                record
                for record, _field, _core in uses
                if record.opcode is RecordOpcode.DTE_RECV
            )
            definition = RuntimeSymbolDefinition(
                symbol,
                endpoints,
                send.source_global_action_id,
                recv.source_global_action_id,
            )
        elif symbol.kind is RuntimeSymbolKind.DTE_TOKEN:
            cores = tuple(
                sorted(
                    {core for _record, _field, core in uses},
                    key=lambda item: (item.die_id, item.local_core_id),
                )
            )
            producer = next(
                (
                    record.source_global_action_id
                    for record, _field, _core in uses
                    if record.opcode in (RecordOpcode.DTE_ISSUE, RecordOpcode.DTE_RECV)
                ),
                None,
            )
            consumer = next(
                (
                    record.source_global_action_id
                    for record, _field, _core in uses
                    if record.opcode is RecordOpcode.DTE_WAIT
                ),
                None,
            )
            definition = RuntimeSymbolDefinition(symbol, cores, producer, consumer)
        elif symbol.kind is RuntimeSymbolKind.RUNTIME_CORE:
            peer = units.get(symbol.source_ref)
            if peer is None:
                raise SchemaError(
                    "peer symbol lacks DP4 executable unit",
                    path="overlay_fragments",
                )
            definition = RuntimeSymbolDefinition(
                symbol, (peer.logical_core,), None, None
            )
        else:
            raise SchemaError(
                "unsupported DP4 rooted runtime symbol", path="overlay_fragments"
            )
        definition.validate("dp4_rooted_runtime_definition")
        runtime_definitions.append(definition)
    return (
        tuple(runtime_definitions),
        tuple(definitions),
        tuple(address_bindings),
    )


def link_s2_lite_dp4_tree_ar_manifest(
    source: S2LiteDp4TreeArLoweredProgram,
) -> LinkedProgramManifest:
    if type(source) is not S2LiteDp4TreeArLoweredProgram:
        raise SchemaError("must be an S2LiteDp4TreeArLoweredProgram", path="source")
    source.validate("source")
    linker = NaiveManifestLinker()
    local_manifests = tuple(
        linker.link(
            context,
            tuple(
                fragment
                for fragment in source.local_fragments
                if fragment.source_global_dag_id == context.global_dag.id
            ),
        )
        for context in source.intent.lowering_contexts
    )
    overlay_runtime, overlay_program, overlay_addresses = _overlay_parts(source)
    fragments = tuple(
        sorted(
            source.local_fragments + source.overlay_fragments,
            key=lambda item: item.id,
        )
    )
    interfaces = _interfaces(fragments)
    core_bindings = tuple(
        sorted(
            (
                binding
                for manifest in local_manifests
                for binding in manifest.core_bindings
            ),
            key=lambda item: (
                item.logical_core.die_id,
                item.logical_core.local_core_id,
            ),
        )
    )
    overlay_by_core = defaultdict(list)
    fragment_by_claim = {
        claim: fragment
        for fragment in source.overlay_fragments
        for claim in fragment.claimed_action_ids
    }
    for unit in source.intent.units:
        fragment = fragment_by_claim[unit.id]
        stream = next(
            item for item in fragment.core_streams if item.logical_core == unit.logical_core
        )
        overlay_by_core[unit.logical_core].extend(
            LinkedRecordRef(fragment.id, index, unit.id)
            for index, record in enumerate(stream.records)
            if record.source_global_action_id == unit.id
        )
    core_streams = []
    for manifest, context in zip(local_manifests, source.intent.lowering_contexts):
        for stream in manifest.core_streams:
            dag = context.global_dag
            sgd = next(
                action
                for action in dag.actions
                if ".sgd_update" in getattr(action.source, "task_id", "")
            )
            wgrad = next(
                action
                for action in dag.actions
                if ".lm_head_wgrad" in getattr(action.source, "task_id", "")
            )
            stream_action_ids = {ref.source_global_action_id for ref in stream.records}
            if sgd.id not in stream_action_ids and wgrad.id not in stream_action_ids:
                core_streams.append(stream)
                continue
            if not {sgd.id, wgrad.id}.issubset(stream_action_ids):
                raise SchemaError(
                    "WGRAD and SGD must share one executable core stream",
                    path="source.local_fragments",
                )
            split = next(
                index
                for index, ref in enumerate(stream.records)
                if ref.source_global_action_id == sgd.id
            )
            wgrad_last = max(
                index
                for index, ref in enumerate(stream.records)
                if ref.source_global_action_id == wgrad.id
            )
            if wgrad_last >= split:
                raise SchemaError(
                    "DP4 overlay requires WGRAD before SGD",
                    path="source.local_fragments",
                )
            injected = tuple(overlay_by_core[stream.logical_core])
            core_streams.append(
                LinkedCoreStream(
                    stream.logical_core,
                    stream.runtime_core_id,
                    (*stream.records[:split], *injected, *stream.records[split:]),
                )
            )

    def flatten(name):
        return tuple(
            item for manifest in local_manifests for item in getattr(manifest, name)
        )

    runtime_defs = tuple(
        sorted(
            (*flatten("runtime_symbol_definitions"), *overlay_runtime),
            key=lambda item: item.symbol.id,
        )
    )
    program_by_id = {}
    for definition in (*flatten("program_symbol_definitions"), *overlay_program):
        previous = program_by_id.get(definition.symbol.id)
        if previous is None:
            program_by_id[definition.symbol.id] = definition
        elif previous != definition:
            merged_cores = tuple(
                sorted(
                    set(previous.logical_cores + definition.logical_cores),
                    key=lambda item: (item.die_id, item.local_core_id),
                )
            )
            if (
                previous.symbol,
                previous.name,
                previous.value,
                previous.size_bytes,
            ) != (
                definition.symbol,
                definition.name,
                definition.value,
                definition.size_bytes,
            ):
                raise SchemaError("conflicting DP4 program definitions", path="source")
            program_by_id[definition.symbol.id] = ProgramSymbolDefinition(
                previous.symbol,
                previous.name,
                previous.value,
                previous.size_bytes,
                merged_cores,
            )
    program_defs = tuple(program_by_id[key] for key in sorted(program_by_id))
    address_bindings = tuple(
        sorted(
            (*flatten("address_operand_bindings"), *overlay_addresses),
            key=lambda item: (
                item.logical_core.die_id,
                item.logical_core.local_core_id,
                item.fragment_id,
                item.fragment_record_index,
                int(item.operand_id),
            ),
        )
    )
    state_bindings = tuple(
        sorted(
            flatten("state_operand_bindings"),
            key=lambda item: (
                item.logical_core.die_id,
                item.logical_core.local_core_id,
                item.fragment_id,
                item.fragment_record_index,
                int(item.operand_id),
            ),
        )
    )
    active = tuple(binding.logical_core for binding in core_bindings)
    starts = tuple(
        sorted(
            (
                event
                for manifest in local_manifests
                for event in manifest.envelope.start_events
            ),
            key=lambda item: (
                item.target_core.die_id,
                item.target_core.local_core_id,
                item.tag_symbol_ref,
            ),
        )
    )
    scheduled = source.intent.source.scheduled
    manifest = LinkedProgramManifest.create(
        producer_pass="s2_lite_dp4_tree_ar_manifest_linker",
        capabilities=0,
        source_ir1_id=scheduled.source_planned_carrier_id,
        source_projection_id=scheduled.source_projected_carrier_id,
        source_schedule_set_id=scheduled.id,
        source_global_dag_id=source.intent.source.id,
        input_digests=_input_digests(source, fragments),
        fragments=fragments,
        fragment_interfaces=interfaces,
        core_bindings=core_bindings,
        core_streams=tuple(
            sorted(
                core_streams,
                key=lambda item: (
                    item.logical_core.die_id,
                    item.logical_core.local_core_id,
                ),
            )
        ),
        runtime_symbol_definitions=runtime_defs,
        program_symbol_definitions=program_defs,
        address_operand_bindings=address_bindings,
        state_operand_bindings=state_bindings,
        core_groups=tuple(
            sorted(flatten("core_groups"), key=lambda item: item.symbol_ref)
        ),
        envelope=ProgramControlEnvelope(
            active,
            starts,
            active,
            active,
            active,
            EmptyCoreAckPolicy.INCLUDE_EMPTY,
            ProgramFailurePolicy.ABORT_ALL,
        ),
    )
    manifest.validate("s2_lite_dp4_tree_ar_manifest")
    return manifest


__all__ = ["link_s2_lite_dp4_tree_ar_manifest"]
