"""Strict single-manifest linker for the isolated S3-Lite MoE backward overlay."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
    AddressOperandBinding,
    BufferABI,
    BufferOwnership,
    CommandFragment,
    CoreRuntimeBinding,
    EmptyCoreAckPolicy,
    FragmentInterface,
    LinkedCoreStream,
    LinkedProgramManifest,
    LinkedRecordRef,
    LogicalStartEvent,
    ManifestInputDigest,
    ManifestInputKind,
    ProgramControlEnvelope,
    ProgramFailurePolicy,
    ProgramSymbolDefinition,
    ProgramSymbolKind,
    RecordOpcode,
    RuntimeOperandField,
    RuntimeSymbol,
    RuntimeSymbolDefinition,
    RuntimeSymbolKind,
    SemanticOperandId,
    StateOperandBinding,
)
from ..schema.common import stable_artifact_id
from ..schema.global_action import LogicalCoreRef
from ..schema.lite_moe_dp4_n6 import LiteMoeDp4BackwardLoweredProgram
from ..schema.serde import canonical_digest


def _program_name(symbol_id: str, kind: ProgramSymbolKind) -> str:
    prefix = {
        ProgramSymbolKind.ABSOLUTE_ADDRESS: "abs",
        ProgramSymbolKind.SRAM_REGION: "region",
        ProgramSymbolKind.SRAM_LABEL: "label",
    }[kind]
    return f"frontend_{prefix}_{symbol_id}"


def _literal(record, name: str) -> object:
    values = tuple(
        operand.literal_value
        for operand in record.operands
        if operand.name == name
    )
    if len(values) != 1:
        raise SchemaError(f"record lacks one {name} literal", path="fragments")
    return values[0]


def _runtime_cores(source: LiteMoeDp4BackwardLoweredProgram) -> dict[int, tuple[LogicalCoreRef, int, str]]:
    logical_cores = {
        stream.logical_core
        for fragment in source.fragments
        for stream in fragment.core_streams
    }
    if {core.die_id for core in logical_cores} != {0, 1, 2, 3} or len(logical_cores) != 4:
        raise SchemaError("backward manifest requires one active core on each die", path="fragments")
    result: dict[int, tuple[LogicalCoreRef, int, str]] = {}
    for logical_core in logical_cores:
        die = next(item for item in source.source.train_forward.forward.n4.graph.fabric.dies if item.id == logical_core.die_id)
        core = next(
            item for item in die.cores
            if item.local_core_id == logical_core.local_core_id
        )
        result[logical_core.die_id] = (
            logical_core,
            core.runtime_core_id,
            core.sram_profile_ref,
        )
    return result


def _action_contract(source: LiteMoeDp4BackwardLoweredProgram):
    overlay = source.source
    actions: dict[str, tuple[LogicalCoreRef, tuple[int, ...]]] = {}
    cores = _runtime_cores(source)

    def add(action_id: str, die_id: int, order: tuple[int, ...]) -> None:
        if action_id in actions:
            raise SchemaError("backward action id is duplicated", path="overlay")
        actions[action_id] = (cores[die_id][0], order)

    for state in overlay.trainable_down_states:
        add(
            f"{overlay.id}.expert{state.expert_index}.weight_load",
            state.home_die_id,
            (0, state.expert_index),
        )
    for unit in overlay.remote_gradients:
        add(unit.send_ref, unit.source_die_id, (1, unit.token_index, 0))
        add(unit.recv_ref, unit.destination_die_id, (1, unit.token_index, 1))
        add(unit.wait_ref, unit.destination_die_id, (1, unit.token_index, 2))
    for unit in overlay.token_wgrads:
        add(unit.id, unit.home_die_id, (2, unit.expert_index, unit.token_index))
    for unit in overlay.expert_reduces:
        add(unit.id, unit.home_die_id, (3, unit.expert_index))
    for unit in overlay.sgd_stores:
        add(unit.id, unit.home_die_id, (4, unit.expert_index))
    if len(actions) != 38:
        raise SchemaError("backward action contract must contain 38 claims", path="overlay")
    return actions


def _runtime_definitions(
    source: LiteMoeDp4BackwardLoweredProgram,
    fragments: tuple[CommandFragment, ...],
    actions,
) -> tuple[RuntimeSymbolDefinition, ...]:
    declarations = {}
    uses: dict[str, list[tuple[str, RuntimeOperandField, LogicalCoreRef]]] = defaultdict(list)
    for fragment in fragments:
        for symbol in fragment.runtime_symbols:
            previous = declarations.setdefault(symbol.id, symbol)
            if previous != symbol:
                raise SchemaError("conflicting runtime symbol declarations", path="fragments")
        for stream in fragment.core_streams:
            for relocation in stream.runtime_relocations:
                action_id = stream.records[relocation.record_index].source_global_action_id
                uses[relocation.symbol_ref].append((action_id, relocation.field, stream.logical_core))

    unit_by_action = {
        action_id: unit
        for unit in source.source.remote_gradients
        for action_id in (unit.send_ref, unit.recv_ref, unit.wait_ref)
    }
    definitions = []
    for symbol_id in sorted(declarations):
        symbol = declarations[symbol_id]
        occurrences = uses.get(symbol_id, ())
        if not occurrences:
            raise SchemaError("runtime declaration has no relocation", path="fragments")
        units = {unit_by_action.get(action_id) for action_id, _field, _core in occurrences}
        if None in units or len(units) != 1:
            raise SchemaError("runtime symbol must belong to one remote-gradient flow", path="fragments")
        unit = next(iter(units))
        source_core = actions[unit.send_ref][0]
        destination_core = actions[unit.recv_ref][0]
        if symbol.kind is RuntimeSymbolKind.DTE_FSM:
            definition = RuntimeSymbolDefinition(
                symbol,
                tuple(sorted((source_core, destination_core), key=lambda item: (item.die_id, item.local_core_id))),
                unit.send_ref,
                unit.recv_ref,
            )
        elif symbol.kind is RuntimeSymbolKind.DTE_TOKEN:
            definition = RuntimeSymbolDefinition(
                symbol,
                (destination_core,),
                unit.recv_ref,
                unit.wait_ref,
            )
        elif symbol.kind is RuntimeSymbolKind.RUNTIME_CORE:
            fields = {field for _action, field, _core in occurrences}
            if fields != {RuntimeOperandField.PEER_CORE}:
                raise SchemaError("runtime-core symbol only supports PEER_CORE", path="fragments")
            occurrence_actions = {action for action, _field, _core in occurrences}
            if occurrence_actions == {unit.send_ref}:
                peer = destination_core
            elif occurrence_actions == {unit.recv_ref}:
                peer = source_core
            else:
                raise SchemaError("runtime-core symbol must identify one endpoint peer", path="fragments")
            definition = RuntimeSymbolDefinition(symbol, (peer,), None, None)
        else:
            raise SchemaError("unsupported backward runtime symbol", path="fragments")
        definition.validate("runtime_symbol_definition")
        definitions.append(definition)
    return tuple(definitions)


def _select_buffer(
    source: LiteMoeDp4BackwardLoweredProgram,
    fragment: CommandFragment,
    record,
    operand_id: SemanticOperandId,
    symbol,
) -> BufferABI:
    candidates = tuple(
        abi for abi in fragment.buffer_abi
        if (
            (symbol.kind is ProgramSymbolKind.SRAM_REGION and abi.region_ref == symbol.source_ref)
            or (symbol.kind is ProgramSymbolKind.SRAM_LABEL and abi.storage_id == symbol.source_ref)
            or (symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS and abi.binding_id == symbol.source_ref)
        )
    )
    if record.opcode is RecordOpcode.SRAM_ALLOC_AT:
        offset = _literal(record, "region_offset_bytes")
        size = _literal(record, "size_bytes")
        candidates = tuple(
            abi for abi in candidates
            if abi.region_offset_bytes == offset and abi.size_bytes == size
        )
    elif record.opcode is RecordOpcode.SRAM_FREE:
        owned = tuple(abi for abi in candidates if abi.ownership is BufferOwnership.OWNED)
        if owned:
            candidates = owned
    elif record.opcode is RecordOpcode.SRAM_BIND:
        wgrad = next((unit for unit in source.source.token_wgrads if unit.id == record.source_global_action_id), None)
        sgd = next((unit for unit in source.source.sgd_stores if unit.id == record.source_global_action_id), None)
        if wgrad is not None:
            expected_value = {
                SemanticOperandId.SRAM_BIND_INPUT_0: wgrad.tape_value_ref,
                SemanticOperandId.SRAM_BIND_INPUT_1: wgrad.upstream_gradient_ref,
                SemanticOperandId.SRAM_BIND_OUTPUT: wgrad.contribution_ref,
            }.get(operand_id)
        elif sgd is not None:
            state = next(item for item in source.source.trainable_down_states if item.expert_index == sgd.expert_index)
            expected_value = {
                SemanticOperandId.SRAM_BIND_INPUT_0: state.declaration.id,
                SemanticOperandId.SRAM_BIND_INPUT_1: sgd.gradient_alias_ref,
                SemanticOperandId.SRAM_BIND_OUTPUT: f"{state.declaration.id}.updated",
            }.get(operand_id)
        else:
            expected_value = None
        if expected_value is not None:
            candidates = tuple(abi for abi in candidates if abi.value_id == expected_value)
    unique = {abi.id: abi for abi in candidates}
    if len(unique) != 1:
        raise SchemaError(
            "relocation does not identify one backward BufferABI "
            f"(action={record.source_global_action_id}, operand={operand_id.name}, matches={tuple(unique)})",
            path="fragments",
        )
    return next(iter(unique.values()))


def link_lite_moe_dp4_backward_manifest(
    source: LiteMoeDp4BackwardLoweredProgram,
) -> LinkedProgramManifest:
    """Link the exact 32-leaf backward quotient into one executable manifest."""

    if type(source) is not LiteMoeDp4BackwardLoweredProgram:
        raise SchemaError("must be a LiteMoeDp4BackwardLoweredProgram", path="source")
    source.validate("source")
    fragments = source.fragments
    actions = _action_contract(source)
    claimed: dict[str, CommandFragment] = {}
    for fragment in fragments:
        fragment.validate("fragments")
        if fragment.source_global_dag_id != source.source.id:
            raise SchemaError("backward fragment must reference overlay id", path="fragments")
        if len(fragment.core_streams) != 1:
            raise SchemaError("backward leaf must contain one core stream", path="fragments")
        stream = fragment.core_streams[0]
        for action_id in fragment.claimed_action_ids:
            if action_id in claimed:
                raise SchemaError("backward action is claimed twice", path="fragments")
            if action_id not in actions or actions[action_id][0] != stream.logical_core:
                raise SchemaError("backward claim has wrong typed core", path="fragments")
            claimed[action_id] = fragment
        if {record.source_global_action_id for record in stream.records} != set(fragment.claimed_action_ids):
            raise SchemaError("backward records and claims must cover each other", path="fragments")
    if set(claimed) != set(actions):
        raise SchemaError("fragments must cover all backward actions", path="fragments")

    abi_by_id = {}
    state_by_id = {}
    symbols = {}
    for fragment in fragments:
        for collection, item in (
            *((abi_by_id, abi) for abi in fragment.buffer_abi),
            *((state_by_id, abi) for abi in fragment.state_abi),
            *((symbols, symbol) for symbol in fragment.program_symbols),
        ):
            previous = collection.setdefault(item.id, item)
            if previous != item:
                raise SchemaError("conflicting fragment declaration", path="fragments")

    address_bindings = []
    state_bindings = []
    symbol_buffer_abis: dict[str, list[BufferABI]] = defaultdict(list)
    symbol_state_abis: dict[str, list[object]] = defaultdict(list)
    symbol_cores: dict[str, set[LogicalCoreRef]] = defaultdict(set)
    local_reduce_source_symbols: set[str] = set()
    for fragment in fragments:
        for stream in fragment.core_streams:
            for relocation in stream.address_relocations:
                record = stream.records[relocation.record_index]
                symbol = symbols[relocation.symbol_ref]
                symbol_cores[relocation.symbol_ref].add(stream.logical_core)
                if relocation.addend != 0:
                    raise SchemaError("backward relocation addend must be zero", path="fragments")
                if relocation.operand_id is SemanticOperandId.HBM_ADDRESS:
                    matches = tuple(
                        abi for abi in fragment.state_abi
                        if abi.hbm_binding_ref == symbol.source_ref
                    )
                    if len(matches) != 1:
                        raise SchemaError("HBM relocation lacks one StateABI", path="fragments")
                    abi = matches[0]
                    state_bindings.append(StateOperandBinding(
                        fragment.id, stream.logical_core, relocation.record_index,
                        relocation.operand_id, abi.id,
                    ))
                    symbol_state_abis[relocation.symbol_ref].append(abi)
                else:
                    abi = _select_buffer(source, fragment, record, relocation.operand_id, symbol)
                    binding_abis = (abi,)
                    if (
                        record.opcode is RecordOpcode.LOCAL_REDUCE
                        and relocation.operand_id is SemanticOperandId.SOURCE_ADDRESS
                    ):
                        units = tuple(
                            unit
                            for unit in source.source.expert_reduces
                            if unit.id == record.source_global_action_id
                        )
                        if len(units) != 1:
                            raise SchemaError(
                                "LOCAL_REDUCE lacks one typed expert contract",
                                path="fragments",
                            )
                        unit = units[0]
                        ordered = []
                        for contribution_ref in unit.contribution_refs:
                            matches = tuple(
                                item
                                for item in fragment.buffer_abi
                                if item.value_id == contribution_ref
                            )
                            if len(matches) != 1:
                                raise SchemaError(
                                    "LOCAL_REDUCE contribution lacks one BufferABI",
                                    path="fragments",
                                )
                            ordered.append(matches[0])
                        binding_abis = tuple(ordered)
                        if (
                            unit.input_offsets != (0, 2048)
                            or len(binding_abis) != 2
                            or binding_abis[0] != abi
                            or any(item.size_bytes != 2048 for item in binding_abis)
                            or binding_abis[0].logical_core != binding_abis[1].logical_core
                            or binding_abis[0].region_ref != binding_abis[1].region_ref
                            or binding_abis[0].storage_id != binding_abis[1].storage_id
                            or binding_abis[1].region_offset_bytes
                            != binding_abis[0].region_offset_bytes + 2048
                        ):
                            raise SchemaError(
                                "LOCAL_REDUCE source must bind two contiguous typed contributions",
                                path="fragments",
                            )
                        local_reduce_source_symbols.add(relocation.symbol_ref)
                    address_bindings.append(AddressOperandBinding(
                        fragment.id, stream.logical_core, relocation.record_index,
                        relocation.operand_id,
                        tuple(item.id for item in binding_abis),
                        tuple(item.tensor_slice for item in binding_abis),
                    ))
                    symbol_buffer_abis[relocation.symbol_ref].extend(binding_abis)

    region_specs = {
        (die.id, core.local_core_id): next(
            profile for profile in source.source.train_forward.forward.n4.graph.fabric.sram_profiles
            if profile.id == core.sram_profile_ref
        )
        for die in source.source.train_forward.forward.n4.graph.fabric.dies for core in die.cores
    }
    definitions = []
    for symbol_id in sorted(symbols):
        symbol = symbols[symbol_id]
        abis = {abi.id: abi for abi in symbol_buffer_abis.get(symbol_id, ())}
        state_abis = {abi.id: abi for abi in symbol_state_abis.get(symbol_id, ())}
        if bool(abis) == bool(state_abis):
            raise SchemaError("program symbol requires one ABI class", path="fragments")
        cores = tuple(sorted(symbol_cores[symbol_id], key=lambda item: (item.die_id, item.local_core_id)))
        if state_abis:
            if len(state_abis) != 1 or symbol.kind is not ProgramSymbolKind.ABSOLUTE_ADDRESS:
                raise SchemaError("HBM symbol requires one StateABI", path="fragments")
            state = next(iter(state_abis.values()))
            name, value, size = _program_name(symbol.id, symbol.kind), state.address, state.size_bytes
        elif symbol.kind is ProgramSymbolKind.SRAM_LABEL:
            name, value, size = _program_name(symbol.id, symbol.kind), 0, 0
        elif symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS:
            spans = sorted({
                (
                    next(
                        region for region in region_specs[(abi.logical_core.die_id, abi.logical_core.local_core_id)].regions
                        if region.id == abi.region_ref
                    ).base_bytes + abi.region_offset_bytes,
                    abi.size_bytes,
                )
                for abi in abis.values()
            })
            if symbol_id in local_reduce_source_symbols:
                if (
                    len(spans) != 2
                    or spans[0][1] != 2048
                    or spans[1] != (spans[0][0] + 2048, 2048)
                ):
                    raise SchemaError(
                        "LOCAL_REDUCE source symbol must span two contiguous contributions",
                        path="fragments",
                    )
                value, size = spans[0][0], 4096
            elif len(spans) != 1:
                raise SchemaError("absolute symbol has inconsistent SRAM spans", path="fragments")
            else:
                value, size = spans[0]
            name = _program_name(symbol.id, symbol.kind)
        else:
            shapes = {
                (region.name, region.base_bytes, region.size_bytes)
                for abi in abis.values()
                for region in region_specs[(abi.logical_core.die_id, abi.logical_core.local_core_id)].regions
                if region.id == abi.region_ref
            }
            if len(shapes) != 1:
                raise SchemaError("SRAM region symbol is inconsistent", path="fragments")
            name, value, size = next(iter(shapes))
        definitions.append(ProgramSymbolDefinition(symbol, name, value, size, cores))

    program_fragments: dict[str, list[str]] = defaultdict(list)
    runtime_fragments: dict[str, list[str]] = defaultdict(list)
    for fragment in fragments:
        for symbol in fragment.program_symbols:
            program_fragments[symbol.id].append(fragment.id)
        for symbol in fragment.runtime_symbols:
            runtime_fragments[symbol.id].append(fragment.id)
    interfaces = []
    for fragment in fragments:
        local_program = tuple(symbol.id for symbol in fragment.program_symbols)
        local_runtime = tuple(symbol.id for symbol in fragment.runtime_symbols)
        program_exports = tuple(sorted(symbol for symbol in local_program if fragment.id == min(program_fragments[symbol])))
        runtime_exports = tuple(sorted(symbol for symbol in local_runtime if fragment.id == min(runtime_fragments[symbol])))
        interfaces.append(FragmentInterface(
            fragment.id,
            tuple(sorted(set(local_runtime).difference(runtime_exports))), runtime_exports,
            tuple(sorted(set(local_program).difference(program_exports))), program_exports,
            (), (),
        ))

    cores = _runtime_cores(source)
    core_bindings = []
    core_streams = []
    starts = []
    runtime_definitions = list(_runtime_definitions(source, fragments, actions))
    for die_id in range(4):
        logical_core, runtime_core_id, sram_profile_ref = cores[die_id]
        die = next(item for item in source.source.train_forward.forward.n4.graph.fabric.dies if item.id == die_id)
        core = next(item for item in die.cores if item.local_core_id == logical_core.local_core_id)
        core_bindings.append(CoreRuntimeBinding(logical_core, core.id, runtime_core_id, sram_profile_ref))
        local_fragments = sorted(
            (fragment for fragment in fragments if fragment.core_streams[0].logical_core == logical_core),
            key=lambda fragment: min(actions[action_id][1] for action_id in fragment.claimed_action_ids),
        )
        refs = tuple(
            LinkedRecordRef(fragment.id, index, record.source_global_action_id)
            for fragment in local_fragments
            for index, record in enumerate(fragment.core_streams[0].records)
        )
        if not refs:
            raise SchemaError("active backward core lacks records", path="fragments")
        core_streams.append(LinkedCoreStream(logical_core, runtime_core_id, refs))
        first_action = refs[0].source_global_action_id
        symbol = RuntimeSymbol(
            stable_artifact_id(
                "start_tag",
                {
                    "source_global_dag_id": source.source.id,
                    "logical_core": logical_core,
                    "first_action_id": first_action,
                },
                schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
            ),
            RuntimeSymbolKind.START_TAG,
            first_action,
        )
        runtime_definitions.append(RuntimeSymbolDefinition(symbol, (logical_core,), None, None))
        starts.append(LogicalStartEvent(logical_core, symbol.id, 1))

    artifacts = [
        (ManifestInputKind.S3_LITE_MOE, source),
        (ManifestInputKind.IR1, source.source.train_forward.forward.n4.graph),
        (ManifestInputKind.IR2_PROJECTION, source.source.train_forward.forward.projection),
        (ManifestInputKind.SCHEDULE_SET, source.source.train_forward.forward.schedule),
        (ManifestInputKind.GLOBAL_ACTION_DAG, source.source),
        *((ManifestInputKind.COMMAND_FRAGMENT, fragment) for fragment in fragments),
    ]
    input_digests = tuple(sorted((
        ManifestInputDigest(kind, artifact.id, artifact.schema_version, canonical_digest(artifact))
        for kind, artifact in artifacts
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    active_cores = tuple(cores[die][0] for die in range(4))
    envelope = ProgramControlEnvelope(
        active_cores, tuple(starts), active_cores, active_cores, active_cores,
        EmptyCoreAckPolicy.INCLUDE_EMPTY, ProgramFailurePolicy.ABORT_ALL,
    )
    manifest = LinkedProgramManifest.create(
        producer_pass="lite_moe_dp4_backward_manifest_linker",
        capabilities=0,
        source_ir1_id=source.source.train_forward.forward.n4.graph.id,
        source_projection_id=source.source.train_forward.forward.projection.id,
        source_schedule_set_id=source.source.train_forward.forward.schedule.id,
        source_global_dag_id=source.source.id,
        input_digests=input_digests,
        fragments=fragments,
        fragment_interfaces=tuple(sorted(interfaces, key=lambda item: item.fragment_id)),
        core_bindings=tuple(core_bindings),
        core_streams=tuple(core_streams),
        runtime_symbol_definitions=tuple(sorted(runtime_definitions, key=lambda item: item.symbol.id)),
        program_symbol_definitions=tuple(sorted(definitions, key=lambda item: item.symbol.id)),
        address_operand_bindings=tuple(sorted(address_bindings, key=lambda item: (
            item.logical_core.die_id, item.logical_core.local_core_id,
            item.fragment_id, item.fragment_record_index, int(item.operand_id),
        ))),
        state_operand_bindings=tuple(sorted(state_bindings, key=lambda item: (
            item.logical_core.die_id, item.logical_core.local_core_id,
            item.fragment_id, item.fragment_record_index, int(item.operand_id),
        ))),
        core_groups=(),
        envelope=envelope,
    )
    manifest.validate("lite_moe_backward_manifest")
    return manifest


__all__ = ["link_lite_moe_dp4_backward_manifest"]
