"""Strict single-manifest linker for isolated four-die S3-Lite MoE inference."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
    AddressOperandBinding,
    BufferABI,
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
from ..schema.common import DType, stable_artifact_id
from ..schema.global_action import LogicalCoreRef
from ..schema.ir2 import TensorSlice
from ..schema.lite_moe_dp4_execution import LiteMoeDp4TaskKind
from ..schema.lite_moe_n6 import LiteMoeBufferOperand
from ..schema.lite_moe_dp4_n6 import (
    LiteMoeDp4InferLoweredProgram,
    LiteMoeDp4TrainForwardLoweredProgram,
)
from ..schema.serde import canonical_digest
from .lite_moe_dp4 import lower_lite_moe_dp4_infer


def _program_name(symbol_id: str, kind: ProgramSymbolKind) -> str:
    prefix = {
        ProgramSymbolKind.ABSOLUTE_ADDRESS: "abs",
        ProgramSymbolKind.SRAM_REGION: "region",
        ProgramSymbolKind.SRAM_LABEL: "label",
    }[kind]
    return f"frontend_{prefix}_{symbol_id}"


def _view(abi: BufferABI, operand: LiteMoeBufferOperand) -> TensorSlice:
    root = abi.tensor_slice
    if operand.offset_bytes == 0 and operand.size_bytes == abi.size_bytes:
        return root
    if abi.dtype is not DType.FP16 or not root.shape:
        raise SchemaError("Lite-MoE subviews require a dense FP16 root", path="fragments")
    elements_before = operand.offset_bytes // 2
    elements = operand.size_bytes // 2
    if (
        operand.offset_bytes % 2
        or operand.size_bytes % 2
        or any(extent != 1 for extent in root.shape[:-1])
        or elements_before + elements > root.shape[-1]
    ):
        raise SchemaError("Lite-MoE packed view is not a contiguous row slice", path="fragments")
    return TensorSlice(
        root.value_id,
        (*root.offset[:-1], root.offset[-1] + elements_before),
        (*root.shape[:-1], elements),
    )


def _core_for_action(source: LiteMoeDp4InferLoweredProgram, action_id: str) -> LogicalCoreRef:
    action = next(item for item in source.source.global_dag.actions if item.id == action_id)
    die = next(item for item in source.source.n4.graph.fabric.dies if item.id == action.die_id)
    core = next(item for item in die.cores if item.id == action.core_ref)
    return LogicalCoreRef(action.die_id, core.local_core_id)


def _runtime_definitions(
    source: LiteMoeDp4InferLoweredProgram,
    fragments: tuple[CommandFragment, ...],
) -> tuple[RuntimeSymbolDefinition, ...]:
    declarations: dict[str, RuntimeSymbol] = {}
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

    actions = {action.id: action for action in source.source.global_dag.actions}
    flow_units = {unit.flow_ref: unit for unit in source.intent.dte_units}
    definitions: list[RuntimeSymbolDefinition] = []
    for symbol_id in sorted(declarations):
        symbol = declarations[symbol_id]
        occurrences = uses.get(symbol_id, ())
        if not occurrences:
            raise SchemaError("runtime declaration has no relocation", path="fragments")
        if symbol.kind is RuntimeSymbolKind.DTE_FSM:
            units = {actions[action_id].flow_ref for action_id, _field, _core in occurrences}
            if len(units) != 1 or None in units:
                raise SchemaError("DTE FSM must belong to one flow", path="fragments")
            unit = flow_units[next(iter(units))]
            source_core = _core_for_action(source, unit.send_action_ref)
            destination_core = _core_for_action(source, unit.recv_action_ref)
            definition = RuntimeSymbolDefinition(
                symbol,
                tuple(sorted((source_core, destination_core), key=lambda item: (item.die_id, item.local_core_id))),
                unit.send_action_ref,
                unit.recv_action_ref,
            )
        elif symbol.kind is RuntimeSymbolKind.DTE_TOKEN:
            units = {actions[action_id].flow_ref for action_id, _field, _core in occurrences}
            if len(units) != 1 or None in units:
                raise SchemaError("DTE token must belong to one flow", path="fragments")
            unit = flow_units[next(iter(units))]
            definition = RuntimeSymbolDefinition(
                symbol,
                (_core_for_action(source, unit.recv_action_ref),),
                unit.recv_action_ref,
                unit.wait_action_ref,
            )
        elif symbol.kind is RuntimeSymbolKind.RUNTIME_CORE:
            peers: set[LogicalCoreRef] = set()
            for action_id, field, _core in occurrences:
                if field is not RuntimeOperandField.PEER_CORE:
                    raise SchemaError("MoE runtime core only supports PEER_CORE", path="fragments")
                action = actions[action_id]
                unit = flow_units[action.flow_ref]
                peer_id = unit.recv_action_ref if action.kind is LiteMoeDp4TaskKind.SEND else unit.send_action_ref
                peers.add(_core_for_action(source, peer_id))
            if len(peers) != 1:
                raise SchemaError("runtime core must identify one peer", path="fragments")
            definition = RuntimeSymbolDefinition(symbol, (next(iter(peers)),), None, None)
        else:
            raise SchemaError("unsupported Lite-MoE runtime symbol", path="fragments")
        definition.validate("runtime_symbol_definition")
        definitions.append(definition)
    return tuple(definitions)


def link_lite_moe_dp4_infer_manifest(source: LiteMoeDp4InferLoweredProgram) -> LinkedProgramManifest:
    """Link one exact S3-Lite lowering into a single executable manifest."""

    if type(source) is not LiteMoeDp4InferLoweredProgram:
        raise SchemaError("must be a LiteMoeDp4InferLoweredProgram", path="source")
    source.validate("source")
    fragments = source.fragments
    actions = {action.id: action for action in source.source.global_dag.actions}
    placements = {item.task_ref: item for item in source.source.schedule.placements}
    claimed: dict[str, CommandFragment] = {}
    canonical = {
        fragment.claimed_action_ids: fragment
        for fragment in lower_lite_moe_dp4_infer(source.intent, source.source)
    }
    for index, fragment in enumerate(fragments):
        if canonical.get(fragment.claimed_action_ids) != fragment:
            raise SchemaError(
                "fragment is not exact DP4 infer lowering",
                path=f"fragments[{index}]",
            )
        for action_id in fragment.claimed_action_ids:
            if action_id in claimed:
                raise SchemaError("action is claimed twice", path="fragments")
            claimed[action_id] = fragment
    if set(claimed) != set(actions):
        raise SchemaError("fragments must cover all MoE actions", path="fragments")

    abi_by_id: dict[str, BufferABI] = {}
    state_by_id = {}
    symbols = {}
    runtime_symbols = {}
    for fragment in fragments:
        for abi in fragment.buffer_abi:
            previous = abi_by_id.setdefault(abi.id, abi)
            if previous != abi:
                raise SchemaError("conflicting BufferABI", path="fragments")
        for abi in fragment.state_abi:
            previous = state_by_id.setdefault(abi.id, abi)
            if previous != abi:
                raise SchemaError("conflicting StateABI", path="fragments")
        for symbol in fragment.program_symbols:
            previous = symbols.setdefault(symbol.id, symbol)
            if previous != symbol:
                raise SchemaError("conflicting program symbol", path="fragments")
        for symbol in fragment.runtime_symbols:
            previous = runtime_symbols.setdefault(symbol.id, symbol)
            if previous != symbol:
                raise SchemaError("conflicting runtime symbol", path="fragments")

    compute_by_action = {unit.action_ref: unit for unit in source.intent.compute_units}
    state_by_action = {unit.action_ref: unit for unit in source.intent.state_loads}
    dte_by_action = {
        action_id: unit
        for unit in source.intent.dte_units
        for action_id in (unit.send_action_ref, unit.recv_action_ref, unit.wait_action_ref)
    }
    address_bindings: list[AddressOperandBinding] = []
    state_bindings: list[StateOperandBinding] = []
    symbol_buffer_abis: dict[str, list[BufferABI]] = defaultdict(list)
    symbol_state_abis: dict[str, list[object]] = defaultdict(list)
    symbol_cores: dict[str, set[LogicalCoreRef]] = defaultdict(set)

    def operand_for(action_id: str, operand_id: SemanticOperandId, fragment: CommandFragment, symbol_ref: str, record):
        if action_id in compute_by_action:
            unit = compute_by_action[action_id]
            mapping = {
                SemanticOperandId.SRAM_BIND_INPUT_0: unit.inputs[0],
                SemanticOperandId.SRAM_BIND_OUTPUT: unit.output,
                SemanticOperandId.COMPUTE_INPUT_ADDRESS: unit.inputs[0],
                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS: unit.output,
            }
            if len(unit.inputs) == 2:
                mapping[SemanticOperandId.COMPUTE_DATA_ADDRESS] = unit.inputs[1]
            operand = mapping.get(operand_id)
            if operand is not None:
                return abi_by_id[operand.buffer_abi_ref], operand
        if action_id in state_by_action:
            unit = state_by_action[action_id]
            if operand_id is SemanticOperandId.DESTINATION_ADDRESS:
                abi = abi_by_id[unit.destination_buffer_abi_ref]
                return abi, LiteMoeBufferOperand(abi.id, 0, abi.size_bytes)
        if action_id in dte_by_action:
            unit = dte_by_action[action_id]
            ref = (
                unit.source_buffer_abi_ref
                if operand_id is SemanticOperandId.SOURCE_ADDRESS
                else unit.destination_buffer_abi_ref
            )
            if operand_id in (SemanticOperandId.SOURCE_ADDRESS, SemanticOperandId.DESTINATION_ADDRESS):
                abi = abi_by_id[ref]
                return abi, LiteMoeBufferOperand(abi.id, 0, abi.size_bytes)
        symbol = symbols[symbol_ref]
        matches = [
            abi for abi in fragment.buffer_abi
            if (
                (symbol.kind is ProgramSymbolKind.SRAM_REGION and abi.region_ref == symbol.source_ref)
                or (symbol.kind is ProgramSymbolKind.SRAM_LABEL and abi.storage_id == symbol.source_ref)
                or (symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS and abi.binding_id == symbol.source_ref)
            )
        ]
        if operand_id is SemanticOperandId.REGION_NAME:
            literals = {operand.name: operand.literal_value for operand in record.operands}
            ordinal = placements[actions[action_id].task_ref].ordinal
            matches = [
                abi for abi in matches
                if abi.region_offset_bytes == literals["region_offset_bytes"]
                and abi.size_bytes == literals["size_bytes"]
                and abi.lifetime_start == ordinal
            ]
        unique = {abi.id: abi for abi in matches}
        if len(unique) != 1:
            raise SchemaError(
                "relocation does not identify one BufferABI "
                f"(action={action_id}, operand={operand_id.name}, "
                f"symbol_kind={symbol.kind.name}, source={symbol.source_ref}, "
                f"matches={tuple(unique)})",
                path="fragments",
            )
        abi = next(iter(unique.values()))
        return abi, LiteMoeBufferOperand(abi.id, 0, abi.size_bytes)

    for fragment in fragments:
        for stream in fragment.core_streams:
            for relocation in stream.address_relocations:
                record = stream.records[relocation.record_index]
                symbol_cores[relocation.symbol_ref].add(stream.logical_core)
                if relocation.operand_id is SemanticOperandId.HBM_ADDRESS:
                    unit = state_by_action.get(record.source_global_action_id)
                    candidates = [abi for abi in fragment.state_abi if unit is not None and abi.hbm_binding_ref == unit.hbm_binding_ref]
                    if len(candidates) != 1:
                        raise SchemaError("HBM relocation lacks one StateABI", path="fragments")
                    abi = candidates[0]
                    state_bindings.append(StateOperandBinding(
                        fragment.id, stream.logical_core, relocation.record_index,
                        relocation.operand_id, abi.id,
                    ))
                    symbol_state_abis[relocation.symbol_ref].append(abi)
                else:
                    abi, operand = operand_for(
                        record.source_global_action_id, relocation.operand_id,
                        fragment, relocation.symbol_ref, record,
                    )
                    if relocation.addend != operand.offset_bytes and relocation.operand_id not in (
                        SemanticOperandId.REGION_NAME, SemanticOperandId.LABEL_SYMBOL,
                        SemanticOperandId.SYMBOL, SemanticOperandId.SRAM_BIND_INPUT_0,
                        SemanticOperandId.SRAM_BIND_OUTPUT,
                    ):
                        raise SchemaError("relocation addend disagrees with typed view", path="fragments")
                    address_bindings.append(AddressOperandBinding(
                        fragment.id, stream.logical_core, relocation.record_index,
                        relocation.operand_id, (abi.id,), (_view(abi, operand),),
                    ))
                    symbol_buffer_abis[relocation.symbol_ref].append(abi)

    region_specs = {
        (die.id, core.local_core_id): (
            core,
            next(profile for profile in source.source.n4.graph.fabric.sram_profiles if profile.id == core.sram_profile_ref),
        )
        for die in source.source.n4.graph.fabric.dies for core in die.cores
    }
    definitions: list[ProgramSymbolDefinition] = []
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
            abi = next(iter(state_abis.values()))
            value, size, name = abi.address, abi.size_bytes, _program_name(symbol.id, symbol.kind)
        elif symbol.kind is ProgramSymbolKind.SRAM_LABEL:
            value, size, name = 0, 0, _program_name(symbol.id, symbol.kind)
        elif symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS:
            spans = set()
            for abi in abis.values():
                _core, profile = region_specs[(abi.logical_core.die_id, abi.logical_core.local_core_id)]
                region = next(item for item in profile.regions if item.id == abi.region_ref)
                spans.add((region.base_bytes + abi.region_offset_bytes, abi.size_bytes))
            if len(spans) != 1:
                raise SchemaError("absolute symbol has inconsistent SRAM spans", path="fragments")
            value, size = next(iter(spans)); name = _program_name(symbol.id, symbol.kind)
        else:
            regions = []
            for abi in abis.values():
                _core, profile = region_specs[(abi.logical_core.die_id, abi.logical_core.local_core_id)]
                regions.append(next(item for item in profile.regions if item.id == abi.region_ref))
            shapes = {(region.name, region.base_bytes, region.size_bytes) for region in regions}
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

    actions_by_core: dict[LogicalCoreRef, list[object]] = defaultdict(list)
    for action in actions.values():
        actions_by_core[_core_for_action(source, action.id)].append(action)
    active_cores = tuple(sorted(actions_by_core, key=lambda item: (item.die_id, item.local_core_id)))
    core_bindings = []
    core_streams = []
    starts = []
    runtime_definitions = list(_runtime_definitions(source, fragments))
    for logical_core in active_cores:
        die = next(item for item in source.source.n4.graph.fabric.dies if item.id == logical_core.die_id)
        core = next(item for item in die.cores if item.local_core_id == logical_core.local_core_id)
        core_bindings.append(CoreRuntimeBinding(logical_core, core.id, core.runtime_core_id, core.sram_profile_ref))
        ordered_actions = sorted(actions_by_core[logical_core], key=lambda item: placements[item.task_ref].ordinal)
        refs = []
        for action in ordered_actions:
            fragment = claimed[action.id]
            stream = next(item for item in fragment.core_streams if item.logical_core == logical_core)
            refs.extend(
                LinkedRecordRef(fragment.id, index, action.id)
                for index, record in enumerate(stream.records)
                if record.source_global_action_id == action.id
            )
        core_streams.append(LinkedCoreStream(logical_core, core.runtime_core_id, tuple(refs)))
        first = ordered_actions[0]
        symbol = RuntimeSymbol(
            stable_artifact_id(
                "start_tag",
                {"source_global_dag_id": source.source.global_dag.id, "logical_core": logical_core, "first_action_id": first.id},
                schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
            ),
            RuntimeSymbolKind.START_TAG,
            first.id,
        )
        runtime_definitions.append(RuntimeSymbolDefinition(symbol, (logical_core,), None, None))
        starts.append(LogicalStartEvent(logical_core, symbol.id, 1))

    artifacts = [
        (ManifestInputKind.S3_LITE_MOE, source),
        (ManifestInputKind.IR1, source.source.n4.graph),
        (ManifestInputKind.IR2_PROJECTION, source.source.projection),
        (ManifestInputKind.SCHEDULE_SET, source.source.schedule),
        (ManifestInputKind.GLOBAL_ACTION_DAG, source.source.global_dag),
        *((ManifestInputKind.COMMAND_FRAGMENT, fragment) for fragment in fragments),
    ]
    input_digests = tuple(sorted((
        ManifestInputDigest(kind, artifact.id, artifact.schema_version, canonical_digest(artifact))
        for kind, artifact in artifacts
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    envelope = ProgramControlEnvelope(
        active_cores, tuple(starts), active_cores, active_cores, active_cores,
        EmptyCoreAckPolicy.INCLUDE_EMPTY, ProgramFailurePolicy.ABORT_ALL,
    )
    manifest = LinkedProgramManifest.create(
        producer_pass="lite_moe_dp4_infer_manifest_linker",
        capabilities=0,
        source_ir1_id=source.source.n4.graph.id,
        source_projection_id=source.source.projection.id,
        source_schedule_set_id=source.source.schedule.id,
        source_global_dag_id=source.source.global_dag.id,
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
    manifest.validate("lite_moe_manifest")
    return manifest


def link_lite_moe_dp4_train_forward_manifest(
    source: LiteMoeDp4TrainForwardLoweredProgram,
) -> LinkedProgramManifest:
    """Extend exact infer linkage with eight terminal local-copy tape leaves."""

    if type(source) is not LiteMoeDp4TrainForwardLoweredProgram:
        raise SchemaError("must be a LiteMoeDp4TrainForwardLoweredProgram", path="source")
    source.validate("source")
    base = link_lite_moe_dp4_infer_manifest(source.forward)
    rebased_by_claim = {
        fragment.claimed_action_ids: fragment
        for fragment in source.fragments
        if fragment.claimed_action_ids not in {
            (item.id,) for item in source.source.tape_copies
        }
    }
    fragment_id_map = {
        fragment.id: rebased_by_claim[fragment.claimed_action_ids].id
        for fragment in source.forward.fragments
    }
    tape_by_claim = {
        item.id: fragment
        for item in source.source.tape_copies
        for fragment in source.tape_fragments
        if fragment.claimed_action_ids == (item.id,)
    }
    if len(tape_by_claim) != 8:
        raise SchemaError("tape leaves do not cover eight typed copies", path="source.tape_fragments")

    fragments = source.fragments
    program_fragments: dict[str, list[str]] = defaultdict(list)
    runtime_fragments: dict[str, list[str]] = defaultdict(list)
    for fragment in fragments:
        for symbol in fragment.program_symbols:
            program_fragments[symbol.id].append(fragment.id)
        for symbol in fragment.runtime_symbols:
            runtime_fragments[symbol.id].append(fragment.id)
    interfaces = tuple(sorted((
        FragmentInterface(
            fragment.id,
            tuple(sorted(
                symbol.id for symbol in fragment.runtime_symbols
                if fragment.id != min(runtime_fragments[symbol.id])
            )),
            tuple(sorted(
                symbol.id for symbol in fragment.runtime_symbols
                if fragment.id == min(runtime_fragments[symbol.id])
            )),
            tuple(sorted(
                symbol.id for symbol in fragment.program_symbols
                if fragment.id != min(program_fragments[symbol.id])
            )),
            tuple(sorted(
                symbol.id for symbol in fragment.program_symbols
                if fragment.id == min(program_fragments[symbol.id])
            )),
            (),
            (),
        )
        for fragment in fragments
    ), key=lambda item: item.fragment_id))

    address_bindings = [
        AddressOperandBinding(
            fragment_id_map[item.fragment_id],
            item.logical_core,
            item.fragment_record_index,
            item.operand_id,
            item.buffer_abi_ids,
            item.tensor_slices,
        )
        for item in base.address_operand_bindings
    ]
    program_definitions = {item.symbol.id: item for item in base.program_symbol_definitions}
    graph = source.source.forward.n4.graph
    profile_by_id = {item.id: item for item in graph.fabric.sram_profiles}
    core_by_logical = {
        LogicalCoreRef(die.id, core.local_core_id): core
        for die in graph.fabric.dies for core in die.cores
    }
    for fragment in source.tape_fragments:
        stream = fragment.core_streams[0]
        abi_by_binding = {item.binding_id: item for item in fragment.buffer_abi}
        abi_by_storage = {item.storage_id: item for item in fragment.buffer_abi}
        for relocation in stream.address_relocations:
            symbol = next(item for item in fragment.program_symbols if item.id == relocation.symbol_ref)
            if relocation.operand_id in (
                SemanticOperandId.REGION_NAME,
                SemanticOperandId.LABEL_SYMBOL,
                SemanticOperandId.DESTINATION_ADDRESS,
            ):
                abi = next(item for item in fragment.buffer_abi if item.binding_id in {
                    tape.id for tape in source.source.tape_buffers
                })
            elif relocation.operand_id is SemanticOperandId.SOURCE_ADDRESS:
                abi = abi_by_binding[symbol.source_ref]
            else:
                raise SchemaError("unexpected tape relocation role", path="source.tape_fragments")
            address_bindings.append(AddressOperandBinding(
                fragment.id,
                stream.logical_core,
                relocation.record_index,
                relocation.operand_id,
                (abi.id,),
                (abi.tensor_slice,),
            ))
        for symbol in fragment.program_symbols:
            if symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS:
                abi = abi_by_binding[symbol.source_ref]
                core = core_by_logical[abi.logical_core]
                profile = profile_by_id[core.sram_profile_ref]
                region = next(item for item in profile.regions if item.id == abi.region_ref)
                definition = ProgramSymbolDefinition(
                    symbol,
                    _program_name(symbol.id, symbol.kind),
                    region.base_bytes + abi.region_offset_bytes,
                    abi.size_bytes,
                    (stream.logical_core,),
                )
            elif symbol.kind is ProgramSymbolKind.SRAM_LABEL:
                abi = abi_by_storage[symbol.source_ref]
                definition = ProgramSymbolDefinition(
                    symbol, _program_name(symbol.id, symbol.kind), 0, 0,
                    (abi.logical_core,),
                )
            else:
                abi = next(item for item in fragment.buffer_abi if item.region_ref == symbol.source_ref)
                core = core_by_logical[abi.logical_core]
                profile = profile_by_id[core.sram_profile_ref]
                region = next(item for item in profile.regions if item.id == abi.region_ref)
                definition = ProgramSymbolDefinition(
                    symbol, region.name, region.base_bytes, region.size_bytes,
                    (stream.logical_core,),
                )
            previous = program_definitions.get(symbol.id)
            if previous is None:
                program_definitions[symbol.id] = definition
            elif (
                previous.symbol != definition.symbol
                or (previous.name, previous.value, previous.size_bytes)
                != (definition.name, definition.value, definition.size_bytes)
            ):
                raise SchemaError("tape program symbol conflicts with infer definition", path="source.tape_fragments")
            else:
                cores = tuple(sorted(
                    set(previous.logical_cores).union(definition.logical_cores),
                    key=lambda item: (item.die_id, item.local_core_id),
                ))
                program_definitions[symbol.id] = ProgramSymbolDefinition(
                    previous.symbol, previous.name, previous.value,
                    previous.size_bytes, cores,
                )

    runtime_definitions = {
        item.symbol.id: item
        for item in base.runtime_symbol_definitions
        if item.symbol.kind is not RuntimeSymbolKind.START_TAG
    }
    for fragment in source.tape_fragments:
        stream = fragment.core_streams[0]
        action = fragment.claimed_action_ids[0]
        tokens = tuple(item for item in fragment.runtime_symbols if item.kind is RuntimeSymbolKind.DTE_TOKEN)
        if len(tokens) != 1:
            raise SchemaError("tape leaf requires one DTE token", path="source.tape_fragments")
        definition = RuntimeSymbolDefinition(tokens[0], (stream.logical_core,), action, action)
        definition.validate("tape_token_definition")
        runtime_definitions[tokens[0].id] = definition

    tape_by_core: dict[LogicalCoreRef, list[tuple[int, CommandFragment]]] = defaultdict(list)
    tape_index = {item.id: item for item in source.source.tape_copies}
    tape_buffers = {item.id: item for item in source.source.tape_buffers}
    for fragment in source.tape_fragments:
        action = fragment.claimed_action_ids[0]
        unit = tape_index[action]
        ordinal = tape_buffers[unit.destination_buffer_ref].ordinal
        tape_by_core[fragment.core_streams[0].logical_core].append((ordinal, fragment))
    core_streams = []
    starts = []
    for base_stream in base.core_streams:
        refs = [
            LinkedRecordRef(
                fragment_id_map[item.fragment_id],
                item.fragment_record_index,
                item.source_global_action_id,
            )
            for item in base_stream.records
        ]
        for _ordinal, fragment in sorted(tape_by_core[base_stream.logical_core], key=lambda item: item[0]):
            refs.extend(
                LinkedRecordRef(fragment.id, index, record.source_global_action_id)
                for index, record in enumerate(fragment.core_streams[0].records)
            )
        core_streams.append(LinkedCoreStream(
            base_stream.logical_core, base_stream.runtime_core_id, tuple(refs)
        ))
        first_action = refs[0].source_global_action_id
        start = RuntimeSymbol(
            stable_artifact_id(
                "start_tag",
                {
                    "source_global_dag_id": source.source.id,
                    "logical_core": base_stream.logical_core,
                    "first_action_id": first_action,
                },
                schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
            ),
            RuntimeSymbolKind.START_TAG,
            first_action,
        )
        runtime_definitions[start.id] = RuntimeSymbolDefinition(
            start, (base_stream.logical_core,), None, None
        )
        starts.append(LogicalStartEvent(base_stream.logical_core, start.id, 1))

    formal = source.source.forward
    artifacts = [
        (ManifestInputKind.S3_LITE_MOE, source),
        (ManifestInputKind.IR1, formal.n4.graph),
        (ManifestInputKind.IR2_PROJECTION, formal.projection),
        (ManifestInputKind.SCHEDULE_SET, formal.schedule),
        (ManifestInputKind.GLOBAL_ACTION_DAG, source.source),
        *((ManifestInputKind.COMMAND_FRAGMENT, fragment) for fragment in fragments),
    ]
    input_digests = tuple(sorted((
        ManifestInputDigest(kind, artifact.id, artifact.schema_version, canonical_digest(artifact))
        for kind, artifact in artifacts
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    active = tuple(item.logical_core for item in base.core_bindings)
    manifest = LinkedProgramManifest.create(
        producer_pass="lite_moe_dp4_train_forward_manifest_linker",
        capabilities=0,
        source_ir1_id=formal.n4.graph.id,
        source_projection_id=formal.projection.id,
        source_schedule_set_id=formal.schedule.id,
        source_global_dag_id=source.source.id,
        input_digests=input_digests,
        fragments=fragments,
        fragment_interfaces=interfaces,
        core_bindings=base.core_bindings,
        core_streams=tuple(core_streams),
        runtime_symbol_definitions=tuple(sorted(runtime_definitions.values(), key=lambda item: item.symbol.id)),
        program_symbol_definitions=tuple(sorted(program_definitions.values(), key=lambda item: item.symbol.id)),
        address_operand_bindings=tuple(sorted(address_bindings, key=lambda item: (
            item.logical_core.die_id, item.logical_core.local_core_id,
            item.fragment_id, item.fragment_record_index, int(item.operand_id),
        ))),
        state_operand_bindings=tuple(sorted((
            StateOperandBinding(
                fragment_id_map[item.fragment_id],
                item.logical_core,
                item.fragment_record_index,
                item.operand_id,
                item.state_abi_id,
            )
            for item in base.state_operand_bindings
        ), key=lambda item: (
            item.logical_core.die_id, item.logical_core.local_core_id,
            item.fragment_id, item.fragment_record_index, int(item.operand_id),
        ))),
        core_groups=(),
        envelope=ProgramControlEnvelope(
            active, tuple(starts), active, active, active,
            EmptyCoreAckPolicy.INCLUDE_EMPTY, ProgramFailurePolicy.ABORT_ALL,
        ),
    )
    manifest.validate("lite_moe_dp4_train_forward_manifest")
    return manifest


__all__ = [
    "link_lite_moe_dp4_infer_manifest",
    "link_lite_moe_dp4_train_forward_manifest",
]
