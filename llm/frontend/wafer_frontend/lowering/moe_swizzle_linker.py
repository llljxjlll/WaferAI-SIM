"""Canonical manifest linker for the replacement-only MoE fragment."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressOperandBinding,
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
    RuntimeSymbol,
    RuntimeSymbolDefinition,
    RuntimeSymbolKind,
    SemanticOperandId,
)
from ..schema.common import stable_artifact_id
from ..schema.ir1 import IR1
from ..schema.serde import canonical_digest
from ..schema.swizzle import SwizzleActionKind
from ..schema.swizzle_moe import MoeSwizzleDecision
from ..schema.swizzle_moe_abi import MoeSwizzleCoreAddressABI
from ..schema.swizzle_moe_execution import MoeScaleExecution
from ..schema.swizzle_moe_ir2 import MoeSwizzleIr2Projection
from ..schema.swizzle_moe_operand_abi import MoeSwizzleOperandABI
from ..schema.swizzle_moe_plan import MoeSwizzleOverlay
from ..schema.swizzle_moe_scale import MoeSwizzleScaleOracle, MoeSwizzleScaleSpec
from .moe_swizzle_standard import _id, _region, lower_moe_swizzle_standard_fragment


_SCHEMA = "wafer_frontend.moe_swizzle_standard_linker/v1alpha1"


def _start_id(projection_id: str, core: object, first: str) -> str:
    return stable_artifact_id(
        "moe_swizzle_start_tag",
        {"projection": projection_id, "core": core, "first": first},
        schema_version=_SCHEMA,
    )


def link_moe_swizzle_standard_manifest(
    ir1: IR1,
    spec: MoeSwizzleScaleSpec,
    oracle: MoeSwizzleScaleOracle,
    execution: MoeScaleExecution,
    decisions: tuple[MoeSwizzleDecision, MoeSwizzleDecision],
    overlay: MoeSwizzleOverlay,
    projection: MoeSwizzleIr2Projection,
    core_abi: MoeSwizzleCoreAddressABI,
    operand_abi: MoeSwizzleOperandABI,
    fragment: object,
) -> LinkedProgramManifest:
    """Link the exact canonical fragment through the standard symbolic ABI."""
    expected = lower_moe_swizzle_standard_fragment(ir1, projection, core_abi, operand_abi)
    if fragment != expected:
        raise SchemaError("fragment is not the exact MoE standard lowering", path="moe_swizzle_manifest.fragment")
    execution.validate_against(spec, oracle, "moe_swizzle_manifest.execution")
    if len(decisions) != 2 or len({item.id for item in decisions}) != 2:
        raise SchemaError("manifest requires exactly two decisions", path="moe_swizzle_manifest.decisions")
    for index, decision in enumerate(decisions):
        decision.validate(f"moe_swizzle_manifest.decisions[{index}]")
        if decision.problem.source_execution_id != execution.id:
            raise SchemaError("decision belongs to another execution", path="moe_swizzle_manifest.decisions")
    overlay.validate("moe_swizzle_manifest.overlay")
    if overlay.source_execution_id != execution.id or projection.source_overlay_id != overlay.id:
        raise SchemaError("overlay/projection lineage is not exact", path="moe_swizzle_manifest.overlay")
    tasks = {item.id: item for item in projection.tasks}
    task_core = {item.task_ref: item.logical_core for item in core_abi.task_bindings}
    binding_by_core = {}
    for item in core_abi.task_bindings:
        binding_by_core.setdefault(item.logical_core, item)
    buffers_by_id = {item.id: item for item in fragment.buffer_abi}
    buffers_by_binding = {item.binding_id: item for item in fragment.buffer_abi}
    buffers_by_storage = defaultdict(list)
    for item in fragment.buffer_abi:
        buffers_by_storage[item.storage_id].append(item)
    symbols = {item.id: item for item in fragment.program_symbols}
    symbol_uses = defaultdict(set)
    address_bindings = []
    for stream in fragment.core_streams:
        for relocation in stream.address_relocations:
            symbol_uses[relocation.symbol_ref].add(stream.logical_core)
            record = stream.records[relocation.record_index]
            symbol = symbols[relocation.symbol_ref]
            if symbol.kind is ProgramSymbolKind.SRAM_REGION:
                label_ref = next(
                    item.symbol_ref for item in record.operands
                    if item.operand_id is SemanticOperandId.LABEL_SYMBOL
                )
                storage_ref = symbols[label_ref].source_ref
                candidates = [item for item in buffers_by_storage[storage_ref] if item.alias_of is None]
            elif symbol.kind is ProgramSymbolKind.SRAM_LABEL:
                candidates = [item for item in buffers_by_storage[symbol.source_ref] if item.alias_of is None]
            elif symbol.source_ref in buffers_by_binding:
                candidates = [buffers_by_binding[symbol.source_ref]]
            else:
                raise SchemaError("program relocation lacks typed BufferABI witness", path="moe_swizzle_manifest.address")
            candidates.sort(key=lambda item: (item.region_offset_bytes, item.id))
            address_bindings.append(AddressOperandBinding(
                fragment.id, stream.logical_core, relocation.record_index,
                relocation.operand_id, tuple(item.id for item in candidates),
                tuple(item.tensor_slice for item in candidates),
            ))
    program_defs = []
    for symbol in fragment.program_symbols:
        cores = tuple(sorted(symbol_uses[symbol.id], key=lambda item: (item.die_id, item.local_core_id)))
        if symbol.kind is ProgramSymbolKind.SRAM_LABEL:
            name, value, size = f"moe_label_{symbol.id[-16:]}", 0, 0
        elif symbol.kind is ProgramSymbolKind.SRAM_REGION:
            abi = next(item for item in fragment.buffer_abi if item.region_ref == symbol.source_ref and item.logical_core in cores)
            binding = next(item for item in core_abi.value_bindings if item.logical_core == abi.logical_core and item.region_ref == abi.region_ref)
            region = _region(ir1, binding)
            name, value, size = region.name, region.base_bytes, region.size_bytes
        else:
            abi = buffers_by_binding.get(symbol.source_ref)
            if abi is None:
                raise SchemaError("absolute symbol lacks BufferABI", path="moe_swizzle_manifest.program_symbols")
            binding = next(item for item in core_abi.value_bindings if (
                item.value_ref == abi.value_id and item.logical_core == abi.logical_core and item.slot == next(
                    candidate.slot for candidate in core_abi.value_bindings
                    if candidate.value_ref == abi.value_id and candidate.logical_core == abi.logical_core
                )
            ))
            name, value, size = f"moe_abs_{symbol.id[-16:]}", binding.address, binding.size_bytes
        program_defs.append(ProgramSymbolDefinition(symbol, name, value, size, cores))

    runtime_defs = []
    wait_for_recv = {
        task.deps[0]: task.id for task in projection.tasks
        if task.kind is SwizzleActionKind.WAIT
    }
    for symbol in fragment.runtime_symbols:
        if symbol.kind is RuntimeSymbolKind.DTE_FSM:
            binding = next(item for item in core_abi.runtime_bindings if item.fsm_symbol_ref == symbol.id)
            flow = next(item for item in projection.flows if item.id == binding.flow_ref)
            cores = tuple(sorted((task_core[flow.send_task_ref], task_core[flow.recv_task_ref]), key=lambda item: (item.die_id, item.local_core_id)))
            source, destination = flow.send_task_ref, flow.recv_task_ref
        elif symbol.kind is RuntimeSymbolKind.DTE_TOKEN:
            binding = next(item for item in core_abi.runtime_bindings if item.token_symbol_ref == symbol.id and tasks[item.task_ref].kind is SwizzleActionKind.RECV)
            cores = (task_core[binding.task_ref],)
            source, destination = binding.task_ref, wait_for_recv[binding.task_ref]
        elif symbol.kind is RuntimeSymbolKind.RUNTIME_CORE:
            match = next(
                (item for item in core_abi.runtime_bindings if item.peer_core is not None and _id("peer_core", {"task": item.task_ref, "peer": item.peer_core}) == symbol.id),
                None,
            )
            if match is None:
                raise SchemaError("runtime core symbol lacks exact peer witness", path="moe_swizzle_manifest.runtime_symbols")
            cores, source, destination = (match.peer_core,), None, None
        else:
            raise SchemaError("unexpected fragment runtime symbol", path="moe_swizzle_manifest.runtime_symbols")
        runtime_defs.append(RuntimeSymbolDefinition(symbol, cores, source, destination))

    active = tuple(item.logical_core for item in fragment.core_streams)
    starts = []
    for core in active:
        first = min(
            (item for item in core_abi.task_bindings if item.logical_core == core),
            key=lambda item: item.core_order,
        )
        symbol = RuntimeSymbol(_start_id(projection.id, core, first.task_ref), RuntimeSymbolKind.START_TAG, first.task_ref)
        runtime_defs.append(RuntimeSymbolDefinition(symbol, (core,), None, None))
        starts.append(LogicalStartEvent(core, symbol.id, 1))
    terminal_tasks = {
        value.producer_task_ref for value in projection.values
        if (value.terminal_ref is not None or value.terminal_slices) and value.producer_task_ref is not None
    }
    terminal_cores = tuple(sorted({task_core[ref] for ref in terminal_tasks}, key=lambda item: (item.die_id, item.local_core_id)))
    core_bindings = []
    linked_streams = []
    for stream in fragment.core_streams:
        task_binding = binding_by_core[stream.logical_core]
        die = next(item for item in ir1.fabric.dies if item.id == stream.logical_core.die_id)
        core = next(item for item in die.cores if item.local_core_id == stream.logical_core.local_core_id)
        core_bindings.append(CoreRuntimeBinding(stream.logical_core, core.id, task_binding.runtime_core_id, core.sram_profile_ref))
        linked_streams.append(LinkedCoreStream(stream.logical_core, task_binding.runtime_core_id, tuple(
            LinkedRecordRef(fragment.id, index, record.source_global_action_id)
            for index, record in enumerate(stream.records)
        )))
    interface = FragmentInterface(
        fragment.id, (), tuple(sorted(item.id for item in fragment.runtime_symbols)),
        (), tuple(sorted(item.id for item in fragment.program_symbols)), (), (),
    )
    artifacts = (
        (ManifestInputKind.IR1, ir1),
        (ManifestInputKind.MOE_SWIZZLE_SCALE_SPEC, spec),
        (ManifestInputKind.MOE_SWIZZLE_SCALE_ORACLE, oracle),
        (ManifestInputKind.MOE_SWIZZLE_EXECUTION, execution),
        *((ManifestInputKind.MOE_SWIZZLE_DECISION, item) for item in decisions),
        (ManifestInputKind.MOE_SWIZZLE_OVERLAY, overlay),
        (ManifestInputKind.MOE_SWIZZLE_PROJECTION, projection),
        (ManifestInputKind.MOE_SWIZZLE_CORE_ADDRESS_ABI, core_abi),
        (ManifestInputKind.MOE_SWIZZLE_OPERAND_ABI, operand_abi),
        (ManifestInputKind.COMMAND_FRAGMENT, fragment),
    )
    digests = tuple(sorted((
        ManifestInputDigest(kind, item.id, item.schema_version, canonical_digest(item))
        for kind, item in artifacts
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    result = LinkedProgramManifest.create(
        producer_pass="moe_swizzle_standard_linker", capabilities=0,
        source_ir1_id=ir1.id, source_projection_id=projection.id,
        source_schedule_set_id=core_abi.id, source_global_dag_id=projection.id,
        input_digests=digests, fragments=(fragment,), fragment_interfaces=(interface,),
        core_bindings=tuple(core_bindings), core_streams=tuple(linked_streams),
        runtime_symbol_definitions=tuple(sorted(runtime_defs, key=lambda item: item.symbol.id)),
        program_symbol_definitions=tuple(sorted(program_defs, key=lambda item: item.symbol.id)),
        address_operand_bindings=tuple(sorted(address_bindings, key=lambda item: (
            item.logical_core.die_id, item.logical_core.local_core_id,
            item.fragment_record_index, int(item.operand_id),
        ))),
        state_operand_bindings=(), core_groups=(),
        envelope=ProgramControlEnvelope(
            active, tuple(starts), terminal_cores, active, terminal_cores,
            EmptyCoreAckPolicy.INCLUDE_EMPTY, ProgramFailurePolicy.ABORT_ALL,
        ),
    )
    result.validate()
    return result


__all__ = ["link_moe_swizzle_standard_manifest"]
