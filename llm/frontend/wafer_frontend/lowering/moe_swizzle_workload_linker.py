"""Exact standard linker for one complete MoE Swizzle workload leaf."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressOperandBinding, CommandFragment, CoreRuntimeBinding,
    EmptyCoreAckPolicy, FragmentInterface, LinkedCoreStream,
    LinkedProgramManifest, LinkedRecordRef, LogicalStartEvent,
    ManifestInputDigest, ManifestInputKind, ProgramControlEnvelope,
    ProgramFailurePolicy, ProgramSymbolDefinition, ProgramSymbolKind,
    RuntimeSymbol, RuntimeSymbolDefinition, RuntimeSymbolKind,
    SemanticOperandId, StateOperandBinding,
)
from ..schema.common import stable_artifact_id
from ..schema.ir1 import IR1
from ..schema.serde import canonical_digest
from ..schema.swizzle import SwizzleActionKind
from ..schema.swizzle_moe import (
    MoeHardwareFacts, MoeSwizzleDecision, MoeSwizzleWorkloadSelection,
)
from ..schema.swizzle_moe_abi import MoeSwizzleCoreAddressABI
from ..schema.swizzle_moe_execution import MoeScaleExecution
from ..schema.swizzle_moe_ir2 import MoeSwizzleIr2Projection
from ..schema.swizzle_moe_operand_abi import MoeSwizzleOperandABI
from ..schema.swizzle_moe_plan import MoeSwizzleOverlay
from ..schema.swizzle_moe_scale import MoeSwizzleScaleOracle, MoeSwizzleScaleSpec
from ..schema.swizzle_moe_state import MoeSwizzleWorkloadStateABI
from ..schema.swizzle_moe_workload import MoeSwizzleWorkloadProjection
from ..schema.swizzle_moe_workload_abi import MoeSwizzleWorkloadABI
from ..schema.swizzle_moe_workload_bridge import MoeSwizzleWorkloadValueBridge
from .moe_swizzle_workload_standard import lower_moe_swizzle_workload_fragment


_SCHEMA = "wafer_frontend.moe_swizzle_workload_linker/v1alpha1"


def _start_id(workload_id: str, core: object, first: str) -> str:
    return stable_artifact_id(
        "moe_swizzle_workload_start_tag",
        {"workload": workload_id, "core": core, "first": first},
        schema_version=_SCHEMA,
    )


def link_moe_swizzle_workload_manifest(
    ir1: IR1,
    spec: MoeSwizzleScaleSpec,
    oracle: MoeSwizzleScaleOracle,
    execution: MoeScaleExecution,
    decisions: tuple[MoeSwizzleDecision, MoeSwizzleDecision],
    selection: MoeSwizzleWorkloadSelection,
    overlay: MoeSwizzleOverlay,
    workload: MoeSwizzleWorkloadProjection,
    projection: MoeSwizzleIr2Projection,
    core_abi: MoeSwizzleCoreAddressABI,
    operand_abi: MoeSwizzleOperandABI,
    state_abi: MoeSwizzleWorkloadStateABI,
    value_bridge: MoeSwizzleWorkloadValueBridge,
    workload_abi: MoeSwizzleWorkloadABI,
    hardware_facts: MoeHardwareFacts,
    fragment: CommandFragment,
) -> LinkedProgramManifest:
    """Link the deterministic whole-workload lowering with closed provenance."""

    expected = lower_moe_swizzle_workload_fragment(
        ir1, execution, workload, projection, core_abi, operand_abi,
        state_abi, value_bridge, workload_abi, hardware_facts,
    )
    if fragment != expected:
        raise SchemaError("fragment is not the exact whole MoE lowering", path="moe_workload_manifest.fragment")
    execution.validate_against(spec, oracle, "moe_workload_manifest.execution")
    selection.validate("moe_workload_manifest.selection")
    if (
        len(decisions) != 2 or len({item.id for item in decisions}) != 2
        or {
            selection.source_dispatch_decision_id,
            selection.source_combine_decision_id,
        } != {item.id for item in decisions}
        or overlay.source_workload_selection_id != selection.id
        or workload.source_overlay_id != overlay.id
        or workload.replacement_projection_id != projection.id
    ):
        raise SchemaError("whole manifest lineage is not exact", path="moe_workload_manifest")

    buffers_by_binding = {item.binding_id: item for item in fragment.buffer_abi}
    roots_by_storage = {
        item.storage_id: item for item in fragment.buffer_abi if item.alias_of is None
    }
    states_by_hbm = {item.hbm_binding_ref: item for item in fragment.state_abi}
    symbols = {item.id: item for item in fragment.program_symbols}
    symbol_uses = defaultdict(set)
    address_bindings = []
    state_bindings = []
    for stream in fragment.core_streams:
        for relocation in stream.address_relocations:
            symbol_uses[relocation.symbol_ref].add(stream.logical_core)
            symbol = symbols[relocation.symbol_ref]
            record = stream.records[relocation.record_index]
            if relocation.operand_id is SemanticOperandId.HBM_ADDRESS:
                state = states_by_hbm.get(symbol.source_ref)
                if state is None:
                    raise SchemaError("HBM symbol lacks exact StateABI", path="moe_workload_manifest.state")
                state_bindings.append(StateOperandBinding(
                    fragment.id, stream.logical_core, relocation.record_index,
                    relocation.operand_id, state.id,
                ))
                continue
            if symbol.kind is ProgramSymbolKind.SRAM_REGION:
                label_ref = next(
                    item.symbol_ref for item in record.operands
                    if item.operand_id is SemanticOperandId.LABEL_SYMBOL
                )
                root = roots_by_storage[symbols[label_ref].source_ref]
                candidates = (root,)
            elif symbol.kind is ProgramSymbolKind.SRAM_LABEL:
                candidates = (roots_by_storage[symbol.source_ref],)
            else:
                abi = buffers_by_binding.get(symbol.source_ref)
                if abi is None:
                    raise SchemaError("absolute symbol lacks exact BufferABI", path="moe_workload_manifest.address")
                candidates = (abi,)
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
            core = cores[0]
            die = next(item for item in ir1.fabric.dies if item.id == core.die_id)
            logical = next(item for item in die.cores if item.local_core_id == core.local_core_id)
            profile = next(item for item in ir1.fabric.sram_profiles if item.id == logical.sram_profile_ref)
            region = next(item for item in profile.regions if item.id == symbol.source_ref)
            name, value, size = region.name, region.base_bytes, region.size_bytes
        elif symbol.source_ref in states_by_hbm:
            state = states_by_hbm[symbol.source_ref]
            name, value, size = f"moe_hbm_{symbol.id[-16:]}", state.address, state.size_bytes
        else:
            abi = buffers_by_binding.get(symbol.source_ref)
            if abi is None:
                raise SchemaError("absolute definition lacks BufferABI", path="moe_workload_manifest.program")
            root = roots_by_storage[abi.storage_id]
            name = f"moe_abs_{symbol.id[-16:]}"
            value = next(
                item.region_base_bytes for die in hardware_facts.ordered_cores_by_die
                for item in die if item.logical_core == abi.logical_core
            ) + abi.region_offset_bytes
            size = abi.size_bytes
            if root.logical_core != abi.logical_core:
                raise SchemaError("alias/root core mismatch", path="moe_workload_manifest.program")
        program_defs.append(ProgramSymbolDefinition(symbol, name, value, size, cores))

    tasks = {item.id: item for item in projection.tasks}
    core_by_task = {item.task_ref: item.logical_core for item in core_abi.task_bindings}
    runtime_defs = []
    waits = {
        task.deps[0]: task.id for task in projection.tasks
        if task.kind is SwizzleActionKind.WAIT
    }
    for symbol in fragment.runtime_symbols:
        if symbol.kind is RuntimeSymbolKind.DTE_FSM:
            binding = next(item for item in core_abi.runtime_bindings if item.fsm_symbol_ref == symbol.id)
            flow = next(item for item in projection.flows if item.id == binding.flow_ref)
            cores = tuple(sorted((core_by_task[flow.send_task_ref], core_by_task[flow.recv_task_ref]), key=lambda item: (item.die_id, item.local_core_id)))
            source, destination = flow.send_task_ref, flow.recv_task_ref
        elif symbol.kind is RuntimeSymbolKind.DTE_TOKEN:
            binding = next((item for item in core_abi.runtime_bindings if item.token_symbol_ref == symbol.id), None)
            if binding is None:
                action_ref = symbol.source_ref
                owner = next(item for item in workload.actions if item.id == action_ref)
                core = next(item.logical_core for item in hardware_facts.ordered_cores_by_die[owner.rank] if item.runtime_core_id == next(
                    placement.runtime_core_id for placement in __import__(
                        "llm.frontend.wafer_frontend.schema.swizzle_moe_placement", fromlist=["build_moe_swizzle_workload_placement"]
                    ).build_moe_swizzle_workload_placement(ir1, workload, projection, hardware_facts) if placement.action_ref == action_ref
                ))
                cores, source, destination = (core,), action_ref, action_ref
            else:
                cores = (core_by_task[binding.task_ref],)
                source, destination = binding.task_ref, waits[binding.task_ref]
        elif symbol.kind is RuntimeSymbolKind.RUNTIME_CORE:
            binding = next(item for item in core_abi.runtime_bindings if item.peer_core is not None and str(item.peer_core) == symbol.source_ref)
            cores, source, destination = (binding.peer_core,), None, None
        else:
            raise SchemaError("unexpected runtime symbol", path="moe_workload_manifest.runtime")
        runtime_defs.append(RuntimeSymbolDefinition(symbol, cores, source, destination))

    active = tuple(item.logical_core for item in fragment.core_streams)
    facts_by_core = {
        item.logical_core: item for die in hardware_facts.ordered_cores_by_die for item in die
    }
    first_action = {
        stream.logical_core: stream.records[0].source_global_action_id
        for stream in fragment.core_streams
    }
    starts = []
    for core in active:
        symbol = RuntimeSymbol(_start_id(workload.id, core, first_action[core]), RuntimeSymbolKind.START_TAG, first_action[core])
        runtime_defs.append(RuntimeSymbolDefinition(symbol, (core,), None, None))
        starts.append(LogicalStartEvent(core, symbol.id, 1))
    placement_by_action = {
        item.action_ref: item for item in __import__(
            "llm.frontend.wafer_frontend.schema.swizzle_moe_placement", fromlist=["build_moe_swizzle_workload_placement"]
        ).build_moe_swizzle_workload_placement(ir1, workload, projection, hardware_facts)
    }
    terminal_cores = tuple(sorted({
        placement_by_action[item.producer_action_ref].logical_core for item in workload.terminals
    }, key=lambda item: (item.die_id, item.local_core_id)))
    core_bindings = []
    linked_streams = []
    for stream in fragment.core_streams:
        fact = facts_by_core[stream.logical_core]
        die = next(item for item in ir1.fabric.dies if item.id == stream.logical_core.die_id)
        core = next(item for item in die.cores if item.local_core_id == stream.logical_core.local_core_id)
        core_bindings.append(CoreRuntimeBinding(stream.logical_core, core.id, fact.runtime_core_id, core.sram_profile_ref))
        linked_streams.append(LinkedCoreStream(stream.logical_core, fact.runtime_core_id, tuple(
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
        (ManifestInputKind.MOE_SWIZZLE_WORKLOAD_SELECTION, selection),
        (ManifestInputKind.MOE_SWIZZLE_WORKLOAD_PROJECTION, workload),
        (ManifestInputKind.MOE_SWIZZLE_WORKLOAD_STATE_ABI, state_abi),
        (ManifestInputKind.MOE_SWIZZLE_WORKLOAD_VALUE_BRIDGE, value_bridge),
        (ManifestInputKind.MOE_SWIZZLE_WORKLOAD_ABI, workload_abi),
        (ManifestInputKind.MOE_SWIZZLE_HARDWARE_FACTS, hardware_facts),
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
        source_schedule_set_id=core_abi.id, source_global_dag_id=workload.id,
        input_digests=digests, fragments=(fragment,), fragment_interfaces=(interface,),
        core_bindings=tuple(core_bindings), core_streams=tuple(linked_streams),
        runtime_symbol_definitions=tuple(sorted(runtime_defs, key=lambda item: item.symbol.id)),
        program_symbol_definitions=tuple(sorted(program_defs, key=lambda item: item.symbol.id)),
        address_operand_bindings=tuple(sorted(address_bindings, key=lambda item: (
            item.logical_core.die_id, item.logical_core.local_core_id,
            item.fragment_id, item.fragment_record_index, int(item.operand_id),
        ))),
        state_operand_bindings=tuple(sorted(state_bindings, key=lambda item: (
            item.logical_core.die_id, item.logical_core.local_core_id,
            item.fragment_id, item.fragment_record_index, int(item.operand_id),
        ))),
        core_groups=(),
        envelope=ProgramControlEnvelope(
            active, tuple(starts), terminal_cores, active, terminal_cores,
            EmptyCoreAckPolicy.INCLUDE_EMPTY, ProgramFailurePolicy.ABORT_ALL,
        ),
    )
    result.validate()
    return result


def build_moe_swizzle_standard_linked_program(
    ir1: IR1,
    spec: MoeSwizzleScaleSpec,
    oracle: MoeSwizzleScaleOracle,
    execution: MoeScaleExecution,
    decisions: tuple[MoeSwizzleDecision, MoeSwizzleDecision],
    selection: MoeSwizzleWorkloadSelection,
    hardware_facts: MoeHardwareFacts,
    program_artifact_sha256: str | None = None,
):
    """Build the exact selected whole-workload standard program."""

    selected_pair = (
        selection.selected_dispatch_candidate_ref,
        selection.selected_combine_candidate_ref,
    )
    selected_witness = next((
        item for item in selection.pair_feasibilities
        if item.candidate_refs == selected_pair
    ), None)
    if selected_witness is None or not selected_witness.feasible:
        raise SchemaError(
            "selected workload pair lacks an exact feasible whole witness",
            path="moe_workload_standard_linker.selection",
        )

    from .moe_swizzle_abi import allocate_moe_swizzle_core_address_abi
    from .moe_swizzle_workload_abi import build_moe_swizzle_workload_abi
    from ..passes.build_moe_scale_swizzle_overlay import build_moe_scale_swizzle_overlay
    from ..passes.build_moe_swizzle_workload_state_abi import build_moe_swizzle_workload_state_abi
    from ..passes.build_moe_swizzle_workload_value_bridge import build_moe_swizzle_workload_value_bridge
    from ..passes.project_moe_scale_swizzle_ir2 import project_moe_scale_swizzle_ir2
    from ..passes.project_moe_swizzle_whole_workload import project_moe_swizzle_whole_workload
    from ..passes.schedule_moe_swizzle_workload_endpoints import schedule_moe_swizzle_workload_endpoints
    from ..passes.schedule_moe_swizzle_workload_storage_reuse import schedule_moe_swizzle_workload_storage_reuse
    from ..schema.swizzle_moe_operand_abi import build_moe_swizzle_operand_abi
    from ..schema.swizzle_moe_placement import build_moe_swizzle_workload_placement
    from ..schema.swizzle_moe_standard import MoeSwizzleStandardLinkedProgram

    overlay = build_moe_scale_swizzle_overlay(
        execution, decisions, selection,
    )
    projection = project_moe_scale_swizzle_ir2(
        overlay, execution, spec, decisions, selection,
        endpoint_session_capacity=decisions[0].problem.endpoint_session_capacity,
    )
    state_abi = build_moe_swizzle_workload_state_abi(
        ir1, execution, spec, oracle,
    )
    workload = project_moe_swizzle_whole_workload(
        overlay, execution, projection, state_abi,
    )
    placement = build_moe_swizzle_workload_placement(
        ir1, workload, projection, hardware_facts,
    )
    workload = schedule_moe_swizzle_workload_endpoints(
        workload, projection, placement,
        capacity_per_core=projection.endpoint_session_capacity,
    )
    placement = build_moe_swizzle_workload_placement(
        ir1, workload, projection, hardware_facts,
    )
    value_bridge = build_moe_swizzle_workload_value_bridge(
        execution, workload, projection,
    )
    workload = schedule_moe_swizzle_workload_storage_reuse(
        workload, projection, value_bridge, placement,
    )
    value_bridge = build_moe_swizzle_workload_value_bridge(
        execution, workload, projection,
    )
    core_abi = allocate_moe_swizzle_core_address_abi(
        ir1, projection, hardware_facts=hardware_facts,
        workload_projection=workload,
    )
    operand_abi = build_moe_swizzle_operand_abi(projection)
    workload_abi = build_moe_swizzle_workload_abi(
        ir1, workload, projection, state_abi, value_bridge, hardware_facts,
    )
    fragment = lower_moe_swizzle_workload_fragment(
        ir1, execution, workload, projection, core_abi, operand_abi,
        state_abi, value_bridge, workload_abi, hardware_facts,
    )
    manifest = link_moe_swizzle_workload_manifest(
        ir1, spec, oracle, execution, decisions, selection, overlay, workload,
        projection, core_abi, operand_abi, state_abi, value_bridge,
        workload_abi, hardware_facts, fragment,
    )
    semantic = {
        "ir1": ir1, "spec": spec, "oracle": oracle,
        "execution": execution, "decisions": decisions,
        "selection": selection, "overlay": overlay,
        "workload": workload, "projection": projection,
        "core_abi": core_abi, "operand_abi": operand_abi,
        "state_abi": state_abi, "value_bridge": value_bridge,
        "workload_abi": workload_abi, "hardware_facts": hardware_facts,
        "fragment": fragment, "manifest": manifest,
    }
    result = MoeSwizzleStandardLinkedProgram.create(
        **semantic, program_io=None,
    )
    if program_artifact_sha256 is None:
        return result
    from ..passes.build_moe_swizzle_program_io import (
        build_moe_swizzle_program_io,
    )
    program_io = build_moe_swizzle_program_io(
        result, program_artifact_sha256,
    )
    return MoeSwizzleStandardLinkedProgram.create(
        **semantic, program_io=program_io,
    )


__all__ = [
    "build_moe_swizzle_standard_linked_program",
    "link_moe_swizzle_workload_manifest",
]
