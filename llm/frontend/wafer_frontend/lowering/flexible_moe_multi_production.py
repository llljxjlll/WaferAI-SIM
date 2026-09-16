"""Multi-die production lowering for the opt-in Flexible-MoE v2 carrier.

The 1x1 path stays in :mod:`flexible_moe_production` because it has frozen
runtime evidence.  This module maps every remote typed flow to one exact
SEND/RECV/WAIT triple while retaining the same two-fragment strict lineage.
"""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
    STATE_TRANSFER_WAVE_RUNTIME_SYMBOL_SCHEMA_VERSION,
    AddressOperandBinding,
    AddressRelocation,
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
    ProgramControlEnvelope,
    ProgramFailurePolicy,
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
    StateOperandBinding,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.flexible_moe import (
    FlexibleMoeExecutablePlan,
    FlexibleMoeMode,
    FlexibleMoeSpec,
    MoeRectActionKind,
    MoeRectFlowStage,
    MoeRectStateRole,
)
from ..schema.global_action import LogicalCoreRef
from ..schema.ir2 import BufferOwnership, TensorSlice
from ..schema.serde import canonical_digest, canonical_json
from .flexible_moe_standard import plan_flexible_moe_standard_mapping
from .flexible_moe_production import (
    FlexibleMoeProductionArtifacts,
    _LINKER_PASS,
    _LOWERING_PASS,
    _REGION_BASE_BYTES,
    _REGION_NAME,
    _REGION_REF,
    _REGION_SIZE_BYTES,
    _SCHEDULE_ID,
    _align,
    _buffer,
    _state_abi,
    _symbol,
)


def expert_projection_action_ids(plan_id: str, expert_action_id: str) -> tuple[str, str, str]:
    """Keep P2 EXPERT_FORWARD as gate; name the other physical stages exactly."""
    return tuple(stable_artifact_id(
        "flexible_moe_expert_projection_action",
        {"plan": plan_id, "expert": expert_action_id, "stage": stage},
        schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
    ) for stage in ("up", "swiglu", "down"))


def expert_wgrad_action_ids(plan_id: str, source_action_id: str) -> tuple[str, ...]:
    """Physically name three gradient GEMMs and their FP32 storage casts."""
    return tuple(stable_artifact_id(
        "flexible_moe_expert_wgrad_stage",
        {"plan": plan_id, "wgrad": source_action_id, "stage": stage},
        schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
    ) for stage in ("up", "down", "cast_gate", "cast_up", "cast_down"))


def gate_wgrad_cast_action_id(plan_id: str, source_action_id: str) -> str:
    return stable_artifact_id(
        "flexible_moe_gate_wgrad_cast_action",
        {"plan": plan_id, "wgrad": source_action_id},
        schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
    )


def expert_dgrad_action_ids(plan_id: str, source_action_id: str) -> tuple[str, ...]:
    """Name the native backward SwiGLU and two projection/dX join stages."""
    return tuple(stable_artifact_id(
        "flexible_moe_expert_dgrad_stage",
        {"plan": plan_id, "dgrad": source_action_id, "stage": stage},
        schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
    ) for stage in ("swiglu_backward", "gate", "up", "sum_dx"))


_STATE_KINDS = (MoeRectActionKind.STATE_LOAD, MoeRectActionKind.STATE_STORE)
_TRANSPORT_KINDS = (
    MoeRectActionKind.SEND,
    MoeRectActionKind.RECV,
    MoeRectActionKind.WAIT,
)
_COMM_REGION_REF = "flexible_moe.sram.comm"
_COMM_REGION_NAME = "comm"
_COMM_REGION_BASE_BYTES = 40960
_COMM_REGION_SIZE_BYTES = 36864


def _validate_fan_in_event_refs(
    set_refs: tuple[str, ...],
    wait_refs: tuple[str, ...],
) -> None:
    if len(set_refs) != len(set(set_refs)):
        raise SchemaError(
            "fan-in EVENT_SET tags must be unique",
            path="fan_in_events.set_refs",
        )
    if len(wait_refs) != len(set(wait_refs)):
        raise SchemaError(
            "fan-in EVENT_WAIT tags must be unique",
            path="fan_in_events.wait_refs",
        )
    if set(set_refs) != set(wait_refs):
        raise SchemaError(
            "fan-in EVENT_SET/EVENT_WAIT tags must close exactly",
            path="fan_in_events",
        )

_P5_RUNTIME_CORES_PER_DIE = 16


def _dte_send(action_id: str, flow, source, fsm, peer):
    return RelocatableRecord(action_id, RecordOpcode.DTE_SEND, (
        RecordOperand.literal("mode", 0),
        RecordOperand.literal("source_space", 0),
        RecordOperand.literal("completion", 1),
        RecordOperand.literal("datatype", 0),
        RecordOperand.literal("reduce_op", 0),
        RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, fsm.id),
        RecordOperand.literal("token", 0),
        RecordOperand.literal("length_bytes", flow.logical_bytes),
        RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, source.id),
        RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, peer.id),
        RecordOperand.literal("expected_sources", 0),
        RecordOperand.literal("tree_id", 0),
        RecordOperand.literal("group_id", 0),
        RecordOperand.literal("collective_id", 0),
        RecordOperand.literal("epoch", 0),
    ))


def _dte_recv(action_id: str, flow, destination, fsm, token, peer):
    return RelocatableRecord(action_id, RecordOpcode.DTE_RECV, (
        RecordOperand.literal("mode", 0),
        RecordOperand.literal("completion", 0),
        RecordOperand.literal("datatype", 0),
        RecordOperand.literal("reduce_op", 0),
        RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, fsm.id),
        RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),
        RecordOperand.literal("length_bytes", flow.logical_bytes),
        RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, destination.id),
        RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, peer.id),
        RecordOperand.literal("expected_sources", 0),
        RecordOperand.literal("tree_id", 0),
        RecordOperand.literal("group_id", 0),
        RecordOperand.literal("collective_id", 0),
        RecordOperand.literal("epoch", 0),
    ))


def lower_link_flexible_moe_multi(
    plan: FlexibleMoeExecutablePlan,
    spec: FlexibleMoeSpec,
    *,
    physical_region_name: str | None = None,
    full_model_dataflow: bool = False,
    runtime_core_ids: tuple[int, ...] | None = None,
) -> FlexibleMoeProductionArtifacts:
    """Materialize a deterministic multi-die timing manifest, fail closed."""

    plan.validate_against(spec)
    release_region = physical_region_name is not None
    shared_region_name = (
        _REGION_NAME if physical_region_name is None else physical_region_name
    )
    comm_region_name = (
        _COMM_REGION_NAME if physical_region_name is None else physical_region_name
    )
    if type(shared_region_name) is not str or not shared_region_name:
        raise SchemaError("physical SRAM region name is empty", path="physical_region_name")
    rank_count = spec.mesh.rank_count
    if not 1 <= rank_count <= 100:
        raise SchemaError("full-model production rank count must lie in [1, 100]", path="spec.mesh")
    if runtime_core_ids is None:
        runtime_core_ids = tuple(
            rank * _P5_RUNTIME_CORES_PER_DIE for rank in range(rank_count)
        )
    if (
        type(runtime_core_ids) is not tuple
        or len(runtime_core_ids) != rank_count
        or any(type(item) is not int or item < 0 for item in runtime_core_ids)
        or len(set(runtime_core_ids)) != rank_count
    ):
        raise SchemaError(
            "runtime_core_ids must bind one unique non-negative core per rank",
            path="runtime_core_ids",
        )
    standard_ir = plan_flexible_moe_standard_mapping(plan, spec)
    cores = tuple(LogicalCoreRef(rank, 0) for rank in range(rank_count))
    core_by_rank = dict(enumerate(cores))
    actions_by_core = {
        core: tuple(action for action in plan.actions if action.rank == rank)
        for rank, core in enumerate(cores)
    }
    if any(not actions for actions in actions_by_core.values()):
        raise SchemaError("every rank must own at least one action", path="plan.actions")
    first_by_core = {core: actions[0].id for core, actions in actions_by_core.items()}
    last_by_core = {core: actions[-1].id for core, actions in actions_by_core.items()}

    state_abis = _state_abi(plan, spec)
    state_by_ref = {item.state_ref: item for item in state_abis}
    states_by_rank = defaultdict(list)
    for state in plan.state_bindings:
        states_by_rank[state.owner_rank].append(state)

    buffers = []
    source_tokens = [0] * rank_count
    expert_tokens = [0] * rank_count
    for assignment in spec.trace.assignments:
        source_tokens[assignment.source_rank] += 1
        expert_tokens[assignment.expert_home_rank] += 1
    max_rank_tokens = max((*source_tokens, *expert_tokens), default=1)
    max_flow_bytes = max((item.logical_bytes for item in plan.flows), default=0)
    activation_bytes = max(
        64, max_rank_tokens * spec.hidden_size * 2, max_flow_bytes,
    )
    workspace_bytes = max(
        1024,
        activation_bytes,
        max_flow_bytes,
        max_rank_tokens * spec.intermediate_size * 2,
        max_rank_tokens * spec.expert_count * 2,
    )
    projection_concat_bytes = max(64, 4 * max_rank_tokens * spec.intermediate_size)
    projection_activated_bytes = max(64, 2 * max_rank_tokens * spec.intermediate_size)
    projection_concat_offset = _align(workspace_bytes)
    projection_activated_offset = _align(projection_concat_offset + projection_concat_bytes)
    gate_wgrad_stage_offset = _align(projection_activated_offset + projection_activated_bytes)
    gate_wgrad_stage_bytes = max(64, 2 * spec.hidden_size * spec.expert_count)
    gate_reduce_offset = _align(gate_wgrad_stage_offset + gate_wgrad_stage_bytes)
    gate_gradient_bytes = 4 * spec.hidden_size * spec.expert_count
    children_by_rank = {
        rank: tuple(child for child in (2 * rank + 1, 2 * rank + 2)
                    if child < rank_count)
        for rank in range(rank_count)
    }
    gate_reduce_region_bytes = max(64, 3 * gate_gradient_bytes)
    dgrad_activated_offset = _align(gate_reduce_offset + gate_reduce_region_bytes)
    dgrad_activated_bytes = max(64, 2 * max_rank_tokens * spec.intermediate_size)
    dgrad_gate_up_offset = _align(dgrad_activated_offset + dgrad_activated_bytes)
    dgrad_gate_up_bytes = max(64, 4 * max_rank_tokens * spec.intermediate_size)
    dgrad_dx_parts_offset = _align(dgrad_gate_up_offset + dgrad_gate_up_bytes)
    dgrad_dx_parts_bytes = max(64, 4 * max_rank_tokens * spec.hidden_size)
    strict_training = full_model_dataflow and spec.mode is FlexibleMoeMode.TRAIN
    empty_source_kinds = (
        MoeRectActionKind.GATE,
        MoeRectActionKind.PACK,
        MoeRectActionKind.WEIGHTED_COMBINE,
    ) + ((
        MoeRectActionKind.EXPERT_DGRAD,
        MoeRectActionKind.EXPERT_WGRAD,
        MoeRectActionKind.GATE_WGRAD,
        MoeRectActionKind.COMBINE_BACKWARD,
    ) if strict_training else ())
    for rank, core in enumerate(cores):
        buffers.append(_buffer(
            name=f"rank{rank}.activation", core=core,
            offset=_REGION_BASE_BYTES if release_region else 0,
            size_bytes=activation_bytes, dtype=DType.FP16,
            ownership=BufferOwnership.BORROWED,
            action_count=len(actions_by_core[core]), region_ref=_REGION_REF,
        ))
        buffers.append(_buffer(
            name=f"rank{rank}.output", core=core,
            offset=_COMM_REGION_BASE_BYTES if release_region else 0,
            size_bytes=workspace_bytes, dtype=DType.FP16,
            ownership=BufferOwnership.OWNED,
            action_count=len(actions_by_core[core]),
            region_ref=_REGION_REF if release_region else _COMM_REGION_REF,
        ))
        if full_model_dataflow:
            stage_buffers = (
                ("expert_gate_up", projection_concat_offset, projection_concat_bytes),
                ("expert_activated", projection_activated_offset, projection_activated_bytes),
            )
            if strict_training and source_tokens[rank]:
                stage_buffers += (("gate_wgrad_stage", gate_wgrad_stage_offset, gate_wgrad_stage_bytes),)
            if strict_training and children_by_rank[rank]:
                stage_buffers += (("gate_reduce_inputs", gate_reduce_offset,
                                   gate_reduce_region_bytes),)
            if strict_training and expert_tokens[rank]:
                stage_buffers += (
                    ("dgrad_activated", dgrad_activated_offset, dgrad_activated_bytes),
                    ("dgrad_gate_up", dgrad_gate_up_offset, dgrad_gate_up_bytes),
                    ("dgrad_dx_parts", dgrad_dx_parts_offset, dgrad_dx_parts_bytes),
                )
            for name, offset, size in stage_buffers:
                buffers.append(_buffer(
                    name=f"rank{rank}.{name}", core=core,
                    offset=offset + (_COMM_REGION_BASE_BYTES if release_region else 0),
                    size_bytes=size,
                    dtype=DType.FP32 if name == "gate_reduce_inputs" else DType.FP16,
                    ownership=BufferOwnership.OWNED,
                    action_count=len(actions_by_core[core]),
                    region_ref=_REGION_REF if release_region else _COMM_REGION_REF,
                ))
        offset = activation_bytes
        for state in sorted(states_by_rank[rank], key=lambda item: item.id):
            offset = _align(offset)
            buffers.append(_buffer(
                name=f"rank{rank}.state.{state.id}", core=core,
                offset=offset + (_REGION_BASE_BYTES if release_region else 0),
                size_bytes=state.size_bytes, dtype=state.dtype,
                ownership=BufferOwnership.OWNED,
                action_count=len(actions_by_core[core]), region_ref=_REGION_REF,
            ))
            offset += state.size_bytes
        if strict_training:
            offset = _align(offset)
            buffers.append(_buffer(
                name=f"rank{rank}.backward_gradient", core=core,
                offset=offset + (_REGION_BASE_BYTES if release_region else 0),
                size_bytes=activation_bytes, dtype=DType.FP16,
                ownership=BufferOwnership.BORROWED,
                action_count=len(actions_by_core[core]), region_ref=_REGION_REF,
            ))
            offset += activation_bytes
        if (
            _align(offset) > _REGION_SIZE_BYTES
            or _align(workspace_bytes) > _COMM_REGION_SIZE_BYTES
            or (full_model_dataflow and
                _align((dgrad_dx_parts_offset + dgrad_dx_parts_bytes) if strict_training
                       else (projection_activated_offset + projection_activated_bytes)) > _COMM_REGION_SIZE_BYTES)
        ):
            raise SchemaError("rank-local SRAM ABI exceeds the production region", path=f"rank[{rank}].buffers")
    buffers = tuple(sorted(buffers, key=lambda item: item.id))
    buffers_by_core = {
        core: tuple(item for item in buffers if item.logical_core == core) for core in cores
    }
    if release_region:
        for core, core_buffers in buffers_by_core.items():
            spans = sorted(
                (item.region_offset_bytes, item.region_offset_bytes + item.size_bytes)
                for item in core_buffers
            )
            if (
                spans[-1][1] > (1 << 20)
                or any(left[1] > right[0] for left, right in zip(spans, spans[1:]))
            ):
                raise SchemaError(
                    "release SRAM subspans overlap or exceed 1 MiB",
                    path=f"core[{core.die_id}].buffers",
                )
    activation_by_rank = {
        rank: next(item for item in buffers_by_core[core] if item.value_id.endswith(".activation"))
        for rank, core in enumerate(cores)
    }
    output_by_rank = {
        rank: next(item for item in buffers_by_core[core] if item.value_id.endswith(".output"))
        for rank, core in enumerate(cores)
    }
    projection_concat_by_rank = {
        rank: next(item for item in buffers_by_core[core] if item.value_id.endswith(".expert_gate_up"))
        for rank, core in enumerate(cores)
    } if full_model_dataflow else {}
    projection_activated_by_rank = {
        rank: next(item for item in buffers_by_core[core] if item.value_id.endswith(".expert_activated"))
        for rank, core in enumerate(cores)
    } if full_model_dataflow else {}
    backward_by_rank = {
        rank: next(item for item in buffers_by_core[core] if item.value_id.endswith(".backward_gradient"))
        for rank, core in enumerate(cores)
    } if strict_training else {}
    gate_wgrad_stage_by_rank = {
        rank: next(item for item in buffers_by_core[core]
                   if item.value_id.endswith(".gate_wgrad_stage"))
        for rank, core in enumerate(cores)
        if source_tokens[rank]
    } if strict_training else {}
    gate_reduce_by_rank = {
        rank: next(item for item in buffers_by_core[core]
                   if item.value_id.endswith(".gate_reduce_inputs"))
        for rank, core in enumerate(cores)
        if children_by_rank[rank]
    } if strict_training else {}
    dgrad_scratch_by_rank = {
        rank: tuple(next(item for item in buffers_by_core[core]
                         if item.value_id.endswith(f".{name}"))
                    for name in ("dgrad_activated", "dgrad_gate_up", "dgrad_dx_parts"))
        for rank, core in enumerate(cores)
        if expert_tokens[rank]
    } if strict_training else {}
    buffer_by_state = {
        state.id: next(
            item for item in buffers_by_core[core_by_rank[state.owner_rank]]
            if item.value_id.endswith(f".state.{state.id}")
        )
        for state in plan.state_bindings
    }
    gate_gradient_by_rank = {
        rank: buffer_by_state[next(state.id for state in states_by_rank[rank]
                                   if state.role is MoeRectStateRole.GATE_GRADIENT)]
        for rank in range(rank_count)
    } if strict_training else {}

    shared_region = _symbol(ProgramSymbolKind.SRAM_REGION, _REGION_REF, "region")
    comm_region = (
        shared_region if release_region
        else _symbol(ProgramSymbolKind.SRAM_REGION, _COMM_REGION_REF, "region")
    )
    region_by_ref = {_REGION_REF: shared_region, _COMM_REGION_REF: comm_region}
    label_by_buffer = {
        item.id: _symbol(ProgramSymbolKind.SRAM_LABEL, item.storage_id, "label") for item in buffers
    }
    absolute_by_buffer = {
        item.id: _symbol(ProgramSymbolKind.ABSOLUTE_ADDRESS, item.binding_id, "absolute") for item in buffers
    }
    hbm_by_state = {
        item.id: _symbol(ProgramSymbolKind.ABSOLUTE_ADDRESS, item.hbm_binding_ref, "hbm")
        for item in state_abis
    }

    records = {role: {core: [] for core in cores} for role in ("state", "compute")}
    address_relocs = {role: {core: [] for core in cores} for role in ("state", "compute")}
    runtime_relocs = {role: {core: [] for core in cores} for role in ("state", "compute")}
    refs_by_action = defaultdict(list)
    operand_views = {}
    physical_children_by_action = {}

    def add_record(role, core, record, address=(), runtime=()):
        index = len(records[role][core])
        records[role][core].append(record)
        refs_by_action[record.source_global_action_id].append((role, core, index))
        for operand_id, symbol, addend in address:
            address_relocs[role][core].append(
                AddressRelocation(index, operand_id, symbol.kind, symbol.id, addend)
            )
        for field, symbol in runtime:
            runtime_relocs[role][core].append(RuntimeRelocation(index, field, symbol.id))

    def view(role, core, operand_id, abi, offset_bytes, length_bytes):
        if offset_bytes < 0 or length_bytes <= 0 or offset_bytes + length_bytes > abi.size_bytes:
            raise SchemaError("expert projection view exceeds SRAM BufferABI", path="expert_projection")
        element_bytes = 4 if abi.dtype is DType.FP32 else 2
        if offset_bytes % element_bytes or length_bytes % element_bytes:
            raise SchemaError("projection view is not aligned to buffer dtype", path="expert_projection")
        operand_views[(role, core, len(records[role][core]), operand_id)] = TensorSlice(
            abi.value_id, (offset_bytes // element_bytes,), (length_bytes // element_bytes,),
        )

    def absolute(abi):
        return absolute_by_buffer[abi.id]

    for core in cores:
        for abi in buffers_by_core[core]:
            region = region_by_ref[abi.region_ref]
            label = label_by_buffer[abi.id]
            add_record("state", core, RelocatableRecord(
                first_by_core[core], RecordOpcode.SRAM_ALLOC_AT, (
                    RecordOperand.address("region_name", SemanticOperandId.REGION_NAME, region.id),
                    RecordOperand.address("label_symbol", SemanticOperandId.LABEL_SYMBOL, label.id),
                    RecordOperand.literal("region_offset_bytes", abi.region_offset_bytes),
                    RecordOperand.literal("size_bytes", abi.size_bytes),
                    RecordOperand.literal("alignment_bytes", abi.alignment_bytes),
                    RecordOperand.literal("lifetime", 0),
                    RecordOperand.literal(
                        "spillable",
                        False if release_region else abi.region_ref == _REGION_REF,
                    ),
                ),
            ), ((SemanticOperandId.REGION_NAME, region, 0), (SemanticOperandId.LABEL_SYMBOL, label, 0)))

    flow_by_id = {flow.id: flow for flow in plan.flows}
    flow_actions = defaultdict(dict)
    for action in plan.actions:
        if action.flow_ref is not None:
            flow_actions[action.flow_ref][action.kind] = action
    peer_symbols = {}
    runtime_symbols = {}
    fsm_by_flow = {}
    token_by_flow = {}
    runtime_definitions = []

    def peer(rank):
        symbol = peer_symbols.get(rank)
        if symbol is None:
            symbol = RuntimeSymbol(
                stable_artifact_id("flexible_moe_runtime_core", {"plan": plan.id, "rank": rank}, schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION),
                RuntimeSymbolKind.RUNTIME_CORE,
                f"rank.{rank}",
            )
            peer_symbols[rank] = symbol
            runtime_symbols[symbol.id] = symbol
            runtime_definitions.append(RuntimeSymbolDefinition(symbol, (core_by_rank[rank],), None, None))
        return symbol

    for flow in plan.flows:
        members = flow_actions[flow.id]
        send = members[MoeRectActionKind.SEND]
        recv = members[MoeRectActionKind.RECV]
        wait = members[MoeRectActionKind.WAIT]
        fsm = RuntimeSymbol(
            stable_artifact_id("flexible_moe_dte_fsm", {"plan": plan.id, "flow": flow.id}, schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION),
            RuntimeSymbolKind.DTE_FSM,
            flow.id,
        )
        token = RuntimeSymbol(
            stable_artifact_id("flexible_moe_dte_token", {"plan": plan.id, "flow": flow.id}, schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION),
            RuntimeSymbolKind.DTE_TOKEN,
            recv.id,
        )
        runtime_symbols[fsm.id] = fsm
        runtime_symbols[token.id] = token
        fsm_by_flow[flow.id] = fsm
        token_by_flow[flow.id] = token
        endpoints = tuple(sorted((core_by_rank[flow.source_rank], core_by_rank[flow.destination_rank]), key=lambda item: (item.die_id, item.local_core_id)))
        runtime_definitions.append(RuntimeSymbolDefinition(fsm, endpoints, send.id, recv.id))
        runtime_definitions.append(RuntimeSymbolDefinition(token, (core_by_rank[flow.destination_rank],), recv.id, wait.id))

    action_by_id = {action.id: action for action in plan.actions}
    fan_in_waits = defaultdict(list)
    fan_in_sets = defaultdict(list)
    for action in plan.actions:
        if action.kind is not MoeRectActionKind.SEND or action.flow_ref is None:
            continue
        for dependency_ref in action.deps:
            dependency = action_by_id[dependency_ref]
            if dependency.kind is not MoeRectActionKind.WAIT:
                continue
            previous_flow = flow_by_id[dependency.flow_ref]
            current_flow = flow_by_id[action.flow_ref]
            if (
                previous_flow.stage is not current_flow.stage
                or not (
                    previous_flow.destination_rank == current_flow.destination_rank
                    or previous_flow.source_rank == current_flow.source_rank
                )
                or previous_flow.wave_index >= current_flow.wave_index
            ):
                raise SchemaError(
                    "fan-in/fan-out fence dependency drifted",
                    path="plan.actions",
                )
            source_core = core_by_rank[previous_flow.destination_rank]
            destination_core = core_by_rank[current_flow.source_rank]
            binding_ref = stable_artifact_id(
                "state_transfer_wave_binding",
                {
                    "source_global_dag_id": plan.id,
                    "source_action_id": dependency.id,
                    "destination_action_id": action.id,
                    "capacity": 3,
                },
                schema_version=STATE_TRANSFER_WAVE_RUNTIME_SYMBOL_SCHEMA_VERSION,
            )
            source_symbol = RuntimeSymbol(
                stable_artifact_id(
                    "state_transfer_wave_core",
                    {
                        "binding_ref": binding_ref,
                        "role": "source",
                        "logical_core": source_core,
                    },
                    schema_version=STATE_TRANSFER_WAVE_RUNTIME_SYMBOL_SCHEMA_VERSION,
                ),
                RuntimeSymbolKind.RUNTIME_CORE,
                binding_ref,
            )
            destination_symbol = RuntimeSymbol(
                stable_artifact_id(
                    "state_transfer_wave_core",
                    {
                        "binding_ref": binding_ref,
                        "role": "destination",
                        "logical_core": destination_core,
                    },
                    schema_version=STATE_TRANSFER_WAVE_RUNTIME_SYMBOL_SCHEMA_VERSION,
                ),
                RuntimeSymbolKind.RUNTIME_CORE,
                binding_ref,
            )
            event = RuntimeSymbol(
                stable_artifact_id(
                    "state_transfer_wave_event",
                    {
                        "binding_ref": binding_ref,
                        "source_action_id": dependency.id,
                        "destination_action_id": action.id,
                    },
                    schema_version=STATE_TRANSFER_WAVE_RUNTIME_SYMBOL_SCHEMA_VERSION,
                ),
                RuntimeSymbolKind.EVENT_TAG,
                binding_ref,
            )
            runtime_symbols[source_symbol.id] = source_symbol
            runtime_symbols[destination_symbol.id] = destination_symbol
            runtime_symbols[event.id] = event
            runtime_definitions.append(RuntimeSymbolDefinition(
                source_symbol, (source_core,), None, None,
            ))
            runtime_definitions.append(RuntimeSymbolDefinition(
                destination_symbol, (destination_core,), None, None,
            ))
            runtime_definitions.append(RuntimeSymbolDefinition(
                event,
                tuple(sorted(
                    (
                        source_core,
                        destination_core,
                    ),
                    key=lambda core: (core.die_id, core.local_core_id),
                )),
                dependency.id,
                action.id,
            ))
            fence = (event, source_symbol, destination_symbol)
            fan_in_sets[dependency.id].append(fence)
            fan_in_waits[action.id].append(fence)

    _validate_fan_in_event_refs(
        tuple(
            fence[0].id
            for fences in fan_in_sets.values()
            for fence in fences
        ),
        tuple(
            fence[0].id
            for fences in fan_in_waits.values()
            for fence in fences
        ),
    )

    def add_fan_in_event(action, opcode, fence):
        event, source_symbol, destination_symbol = fence
        operands = (
            RecordOperand.runtime(
                "source_core", RuntimeOperandField.SOURCE_CORE, source_symbol.id,
            ),
            RecordOperand.runtime(
                "destination_core", RuntimeOperandField.DESTINATION_CORE,
                destination_symbol.id,
            ),
            RecordOperand.runtime(
                "tag", RuntimeOperandField.EVENT_TAG, event.id,
            ),
        )
        if opcode is RecordOpcode.EVENT_WAIT:
            operands = (*operands, RecordOperand.literal("count", 1))
        add_record(
            "compute",
            core_by_rank[action.rank],
            RelocatableRecord(action.id, opcode, operands),
            (),
            (
                (RuntimeOperandField.EVENT_TAG, event),
                (RuntimeOperandField.SOURCE_CORE, source_symbol),
                (RuntimeOperandField.DESTINATION_CORE, destination_symbol),
            ),
        )

    def add_bind(action, inputs, destination, *, source_action_id=None):
        operands = [RecordOperand.literal("input_count", len(inputs))]
        relocs = []
        for index in range(16):
            operand_id = SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0) + index)
            if index < len(inputs):
                label = label_by_buffer[inputs[index].id]
                operands.append(RecordOperand.address(f"input_label_{index}", operand_id, label.id))
                relocs.append((operand_id, label, 0))
            else:
                operands.append(RecordOperand.literal(f"input_label_{index}", 0))
        label = label_by_buffer[destination.id]
        operands.append(RecordOperand.address("output_label", SemanticOperandId.SRAM_BIND_OUTPUT, label.id))
        relocs.append((SemanticOperandId.SRAM_BIND_OUTPUT, label, 0))
        add_record("compute", core_by_rank[action.rank], RelocatableRecord(
            action.id if source_action_id is None else source_action_id,
            RecordOpcode.SRAM_BIND, tuple(operands)), tuple(relocs))

    for action in plan.actions:
        core = core_by_rank[action.rank]
        activation = activation_by_rank[action.rank]
        output = output_by_rank[action.rank]
        if full_model_dataflow and (
            action.kind in empty_source_kinds and not action.assignment_refs
        ):
            if action.flops != 0 or action.logical_bytes != 0:
                raise SchemaError("empty source action must declare zero logical work", path=f"actions[{action.id}]")
            # The P2 action/deps remain typed, while an empty source rank has
            # no gate rows, no activation to pack, and no result to combine.
            continue
        if full_model_dataflow and action.kind is MoeRectActionKind.EXPERT_FORWARD:
            m = len(action.assignment_refs)
            h, intermediate = spec.hidden_size, spec.intermediate_size
            if action.flops != m * 6 * h * intermediate:
                raise SchemaError("expert three-projection FLOPs disagree with P2", path=f"actions[{action.id}]")
            if m == 0:
                continue
            weight = buffer_by_state[action.state_refs[0]]
            weight_matrix_bytes = 2 * h * intermediate
            if weight.size_bytes != 3 * weight_matrix_bytes:
                raise SchemaError("expert gate/up/down tensor must contain all three FP16 matrices", path=f"actions[{action.id}]")
            concat = projection_concat_by_rank[action.rank]
            activated = projection_activated_by_rank[action.rank]
            projection_bytes = 2 * m * intermediate
            if concat.size_bytes < 2 * projection_bytes or activated.size_bytes < projection_bytes or output.size_bytes < 2 * m * h:
                raise SchemaError("expert intermediate SRAM does not fit all three projections", path=f"actions[{action.id}]")
            up_id, swiglu_id, down_id = expert_projection_action_ids(plan.id, action.id)
            physical_children_by_action[action.id] = (up_id, swiglu_id, down_id)
            for action_id, weight_offset, output_offset in (
                (action.id, 0, 0), (up_id, weight_matrix_bytes, projection_bytes),
            ):
                add_bind(action, (activation,), concat, source_action_id=action_id)
                view("compute", core, SemanticOperandId.COMPUTE_DATA_ADDRESS, weight, weight_offset, weight_matrix_bytes)
                view("compute", core, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, concat, output_offset, projection_bytes)
                add_record("compute", core, RelocatableRecord(action_id, RecordOpcode.MATMUL, (
                    RecordOperand.literal("datatype", 1),
                    RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(activation).id),
                    RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(weight).id),
                    RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(concat).id),
                    RecordOperand.literal("parameters", (1, m, h, intermediate)),
                )), ((SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(activation), 0),
                     (SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(weight), weight_offset),
                     (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(concat), output_offset)))
            add_bind(action, (concat,), activated, source_action_id=swiglu_id)
            view("compute", core, SemanticOperandId.COMPUTE_INPUT_ADDRESS, concat, 0, 2 * projection_bytes)
            view("compute", core, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, activated, 0, projection_bytes)
            add_record("compute", core, RelocatableRecord(swiglu_id, RecordOpcode.SWIGLU, (
                RecordOperand.literal("datatype", 1),
                RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(concat).id),
                RecordOperand.literal("data_address", 0),
                RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(activated).id),
                RecordOperand.literal("parameters", (m * intermediate,)),
            )), ((SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(concat), 0),
                 (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(activated), 0)))
            add_bind(action, (activated,), output, source_action_id=down_id)
            view("compute", core, SemanticOperandId.COMPUTE_INPUT_ADDRESS, activated, 0, projection_bytes)
            view("compute", core, SemanticOperandId.COMPUTE_DATA_ADDRESS, weight, 2 * weight_matrix_bytes, weight_matrix_bytes)
            view("compute", core, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, output, 0, 2 * m * h)
            add_record("compute", core, RelocatableRecord(down_id, RecordOpcode.MATMUL, (
                RecordOperand.literal("datatype", 1),
                RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(activated).id),
                RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(weight).id),
                RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(output).id),
                RecordOperand.literal("parameters", (1, m, intermediate, h)),
            )), ((SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(activated), 0),
                 (SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(weight), 2 * weight_matrix_bytes),
                 (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(output), 0)))
            continue
        if strict_training and action.kind is MoeRectActionKind.EXPERT_DGRAD:
            m, h, intermediate = len(action.assignment_refs), spec.hidden_size, spec.intermediate_size
            if action.flops != m * 6 * h * intermediate:
                raise SchemaError("expert three-projection DGRAD FLOPs disagree with P2", path=action.id)
            if m == 0:
                continue
            weight_ref = next(ref for ref in action.state_refs
                              if next(state for state in plan.state_bindings if state.id == ref).role
                              is MoeRectStateRole.EXPERT_PARAMETER)
            weight = buffer_by_state[weight_ref]
            weight_matrix_bytes = 2 * h * intermediate
            if weight.size_bytes != 3 * weight_matrix_bytes:
                raise SchemaError("DGRAD must read every forward expert weight projection", path=action.id)
            activated_grad, gate_up_grad, dx_parts = dgrad_scratch_by_rank[action.rank]
            forward_concat = projection_concat_by_rank[action.rank]
            upstream = backward_by_rank[action.rank]
            upstream_bytes, intermediate_bytes = 2 * m * h, 2 * m * intermediate
            if (activated_grad.size_bytes < intermediate_bytes
                    or gate_up_grad.size_bytes < 2 * intermediate_bytes
                    or dx_parts.size_bytes < 2 * upstream_bytes
                    or output.size_bytes < upstream_bytes):
                raise SchemaError("expert backward SRAM stages do not fit P2 assignment", path=action.id)
            swiglu_id, gate_id, up_id, sum_id = expert_dgrad_action_ids(plan.id, action.id)
            physical_children_by_action[action.id] = (swiglu_id, gate_id, up_id, sum_id)
            # dY (m×H) × Wdown (H×I) -> dActivated (m×I).
            add_bind(action, (upstream,), activated_grad)
            view("compute", core, SemanticOperandId.COMPUTE_DATA_ADDRESS,
                 weight, 2 * weight_matrix_bytes, weight_matrix_bytes)
            view("compute", core, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                 activated_grad, 0, intermediate_bytes)
            add_record("compute", core, RelocatableRecord(action.id, RecordOpcode.MATMUL, (
                RecordOperand.literal("datatype", 1),
                RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(upstream).id),
                RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(weight).id),
                RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(activated_grad).id),
                RecordOperand.literal("parameters", (1, m, h, intermediate)),
            )), ((SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(upstream), 0),
                 (SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(weight), 2 * weight_matrix_bytes),
                 (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(activated_grad), 0)))
            # Distinct derivative primitive: gate/up forward values and
            # upstream dActivated feed physical gate/up gradients, SFU/vec>0.
            add_bind(action, (forward_concat, activated_grad), gate_up_grad,
                     source_action_id=swiglu_id)
            view("compute", core, SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                 forward_concat, 0, 2 * intermediate_bytes)
            view("compute", core, SemanticOperandId.COMPUTE_DATA_ADDRESS,
                 activated_grad, 0, intermediate_bytes)
            view("compute", core, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                 gate_up_grad, 0, 2 * intermediate_bytes)
            add_record("compute", core, RelocatableRecord(swiglu_id,
                                                          RecordOpcode.SWIGLU_BACKWARD_TIMING, (
                RecordOperand.literal("datatype", 1),
                RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(forward_concat).id),
                RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(activated_grad).id),
                RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(gate_up_grad).id),
                RecordOperand.literal("parameters", (m * intermediate,)),
            )), ((SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(forward_concat), 0),
                 (SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(activated_grad), 0),
                 (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(gate_up_grad), 0)))
            # The two H×I gate/up transpose projections each charge 2mHI
            # FLOPs; their disjoint dX halves are joined by actual FP16 sum.
            for projection_index, projection_id in enumerate((gate_id, up_id)):
                add_bind(action, (gate_up_grad,), dx_parts,
                         source_action_id=projection_id)
                view("compute", core, SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                     gate_up_grad, projection_index * intermediate_bytes,
                     intermediate_bytes)
                view("compute", core, SemanticOperandId.COMPUTE_DATA_ADDRESS,
                     weight, projection_index * weight_matrix_bytes,
                     weight_matrix_bytes)
                view("compute", core, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                     dx_parts, projection_index * upstream_bytes,
                     upstream_bytes)
                add_record("compute", core, RelocatableRecord(projection_id,
                                                              RecordOpcode.MATMUL, (
                    RecordOperand.literal("datatype", 1),
                    RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(gate_up_grad).id),
                    RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(weight).id),
                    RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(dx_parts).id),
                    RecordOperand.literal("parameters", (1, m, intermediate, h)),
                )), ((SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(gate_up_grad),
                       projection_index * intermediate_bytes),
                     (SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(weight),
                      projection_index * weight_matrix_bytes),
                     (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(dx_parts),
                      projection_index * upstream_bytes)))
            view("compute", core, SemanticOperandId.SOURCE_ADDRESS,
                 dx_parts, 0, 2 * upstream_bytes)
            view("compute", core, SemanticOperandId.DESTINATION_ADDRESS,
                 output, 0, upstream_bytes)
            add_record("compute", core, RelocatableRecord(sum_id, RecordOpcode.LOCAL_REDUCE, (
                RecordOperand.literal("input_dtype", 0),
                RecordOperand.literal("accumulator_dtype", 1),
                RecordOperand.literal("output_dtype", 0),
                RecordOperand.literal("reduce_op", 1),
                RecordOperand.literal("rounding", 0),
                RecordOperand.literal("order", 0),
                RecordOperand.literal("input_count", 2),
                RecordOperand.literal("element_count", m * h),
                RecordOperand.literal("input_stride_bytes", upstream_bytes),
                RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, absolute(dx_parts).id),
                RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, absolute(output).id),
            )), ((SemanticOperandId.SOURCE_ADDRESS, absolute(dx_parts), 0),
                 (SemanticOperandId.DESTINATION_ADDRESS, absolute(output), 0)))
            continue
        if strict_training and action.kind is MoeRectActionKind.EXPERT_WGRAD:
            m, h, intermediate = len(action.assignment_refs), spec.hidden_size, spec.intermediate_size
            if action.flops != m * 6 * h * intermediate:
                raise SchemaError("expert three-projection WGRAD FLOPs disagree with P2", path=f"actions[{action.id}]")
            if m == 0:
                continue
            gradient_ref = next(ref for ref in action.state_refs
                                if next(state for state in plan.state_bindings if state.id == ref).role
                                is MoeRectStateRole.EXPERT_GRADIENT)
            gradient = buffer_by_state[gradient_ref]
            matrix_fp32 = 4 * h * intermediate
            if gradient.dtype is not DType.FP32 or gradient.size_bytes != 3 * matrix_fp32:
                raise SchemaError("expert WGRAD needs exactly three FP32 weight gradient matrices", path=f"actions[{action.id}]")
            up_id, down_id, *_ = expert_wgrad_action_ids(plan.id, action.id)
            if full_model_dataflow:
                # Named native dW consumes the genuine expert tape.  Down
                # takes backward dY[k,H]; gate/up take disjoint FP16[k,I]
                # outputs of the earlier SwiGLU-backward producer.  An FP16
                # MATMUL followed by an FP32 cast is not an FP32 WGRAD.
                _, gate_up_grad, _ = dgrad_scratch_by_rank[action.rank]
                activated = projection_activated_by_rank[action.rank]
                backward = backward_by_rank[action.rank]
                activation = activation_by_rank[action.rank]
                rank_activation_bytes = 2 * m * h
                rank_intermediate_bytes = 2 * m * intermediate
                if (activation.size_bytes < rank_activation_bytes
                        or activated.size_bytes < rank_intermediate_bytes
                        or backward.size_bytes < rank_activation_bytes
                        or gate_up_grad.size_bytes < 2 * rank_intermediate_bytes):
                    raise SchemaError(
                        "native expert WGRAD has no real forward/SwiGLU-backward FP16 tape",
                        path=f"actions[{action.id}]",
                    )
                physical_children_by_action[action.id] = (up_id, down_id)
                for (projection, source_ref, input_abi, input_size,
                     data_abi, data_offset, data_size, input_dim, output_dim) in (
                    ("gate", action.id, activation, rank_activation_bytes,
                     gate_up_grad, 0, rank_intermediate_bytes, h, intermediate),
                    ("up", up_id, activation, rank_activation_bytes,
                     gate_up_grad, rank_intermediate_bytes,
                     rank_intermediate_bytes, h, intermediate),
                    ("down", down_id, activated, rank_intermediate_bytes,
                     backward, 0, rank_activation_bytes, intermediate, h),
                ):
                    output_offset = (0 if projection == "gate" else
                                     matrix_fp32 if projection == "up" else
                                     2 * matrix_fp32)
                    add_bind(action, (input_abi, data_abi), gradient,
                             source_action_id=source_ref)
                    view("compute", core, SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                         input_abi, 0, input_size)
                    view("compute", core, SemanticOperandId.COMPUTE_DATA_ADDRESS,
                         data_abi, data_offset, data_size)
                    view("compute", core, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                         gradient, output_offset, matrix_fp32)
                    add_record("compute", core, RelocatableRecord(
                        source_ref, RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING, (
                            RecordOperand.literal("activation_datatype", 1),
                            RecordOperand.literal("upstream_datatype", 1),
                            RecordOperand.literal("gradient_datatype", 3),
                            RecordOperand.address("activation_address",
                                SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                                absolute(input_abi).id),
                            RecordOperand.address("upstream_address",
                                SemanticOperandId.COMPUTE_DATA_ADDRESS,
                                absolute(data_abi).id),
                            RecordOperand.address("gradient_address",
                                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                                absolute(gradient).id),
                            RecordOperand.literal("m", input_dim),
                            RecordOperand.literal("n", output_dim),
                            RecordOperand.literal("k", m),
                        )), (
                            (SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                             absolute(input_abi), 0),
                            (SemanticOperandId.COMPUTE_DATA_ADDRESS,
                             absolute(data_abi), data_offset),
                            (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                             absolute(gradient), output_offset),
                        ))
                continue
            raise SchemaError("explicit TRAIN expert WGRAD requires native FP32 projection records",
                              path=action.id)
        if strict_training and action.kind is MoeRectActionKind.GATE_WGRAD:
            m, h, e = len(action.assignment_refs), spec.hidden_size, spec.expert_count
            if action.flops != m * 2 * h * e:
                raise SchemaError("gate WGRAD FLOPs disagree with P2", path=f"actions[{action.id}]")
            if m == 0:
                continue
            gradient_ref = next(ref for ref in action.state_refs
                                if next(state for state in plan.state_bindings if state.id == ref).role
                                is MoeRectStateRole.GATE_GRADIENT)
            gradient = buffer_by_state[gradient_ref]
            stage = gate_wgrad_stage_by_rank[action.rank]
            local_destination = gate_reduce_by_rank.get(action.rank, gradient)
            stage_bytes, gradient_bytes = 2 * h * e, 4 * h * e
            if gradient.dtype is not DType.FP32 or gradient.size_bytes != gradient_bytes or stage.size_bytes < stage_bytes:
                raise SchemaError("gate WGRAD needs one full FP32 gradient matrix", path=f"actions[{action.id}]")
            cast_id = gate_wgrad_cast_action_id(plan.id, action.id)
            physical_children_by_action[action.id] = (cast_id,)
            add_bind(action, (activation,), stage)
            view("compute", core, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, stage, 0, stage_bytes)
            add_record("compute", core, RelocatableRecord(action.id, RecordOpcode.MATMUL, (
                RecordOperand.literal("datatype", 1),
                RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(activation).id),
                RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(backward_by_rank[action.rank]).id),
                RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(stage).id),
                RecordOperand.literal("parameters", (1, h, m, e)),
            )), ((SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(activation), 0),
                 (SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(backward_by_rank[action.rank]), 0),
                 (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(stage), 0)))
            view("compute", core, SemanticOperandId.SOURCE_ADDRESS, stage, 0, stage_bytes)
            view("compute", core, SemanticOperandId.DESTINATION_ADDRESS, local_destination, 0, gradient_bytes)
            add_record("compute", core, RelocatableRecord(cast_id, RecordOpcode.LOCAL_REDUCE, (
                RecordOperand.literal("input_dtype", 0),
                RecordOperand.literal("accumulator_dtype", 1),
                RecordOperand.literal("output_dtype", 1),
                RecordOperand.literal("reduce_op", 1),
                RecordOperand.literal("rounding", 0),
                RecordOperand.literal("order", 0),
                RecordOperand.literal("input_count", 1),
                RecordOperand.literal("element_count", h * e),
                RecordOperand.literal("input_stride_bytes", stage_bytes),
                RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, absolute(stage).id),
                RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, absolute(local_destination).id),
            )), ((SemanticOperandId.SOURCE_ADDRESS, absolute(stage), 0),
                 (SemanticOperandId.DESTINATION_ADDRESS, absolute(local_destination), 0)))
            continue
        if action.kind is MoeRectActionKind.STATE_LOAD:
            state = state_by_ref.get(action.state_refs[0]) if action.state_refs else None
            if state is None:
                raise SchemaError("STATE_LOAD must reference one persistent state", path="plan.actions")
            destination = buffer_by_state[state.state_ref]
            hbm = hbm_by_state[state.id]
            add_record("state", core, RelocatableRecord(action.id, RecordOpcode.LSU_LOAD, (
                RecordOperand.address("hbm_address", SemanticOperandId.HBM_ADDRESS, hbm.id),
                RecordOperand.literal("size_bytes", state.size_bytes),
                RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, absolute(destination).id),
            )), ((SemanticOperandId.HBM_ADDRESS, hbm, 0), (SemanticOperandId.DESTINATION_ADDRESS, absolute(destination), 0)))
            continue
        if action.kind is MoeRectActionKind.STATE_STORE:
            state = state_by_ref.get(action.state_refs[0]) if action.state_refs else None
            if state is None:
                raise SchemaError("STATE_STORE must reference one persistent state", path="plan.actions")
            source = buffer_by_state[state.state_ref]
            hbm = hbm_by_state[state.id]
            add_record("state", core, RelocatableRecord(action.id, RecordOpcode.LSU_STORE, (
                RecordOperand.address("hbm_address", SemanticOperandId.HBM_ADDRESS, hbm.id),
                RecordOperand.literal("size_bytes", state.size_bytes),
                RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, absolute(source).id),
            )), ((SemanticOperandId.HBM_ADDRESS, hbm, 0), (SemanticOperandId.SOURCE_ADDRESS, absolute(source), 0)))
            continue
        if action.kind in _TRANSPORT_KINDS:
            flow = flow_by_id[action.flow_ref]
            send = flow_actions[flow.id][MoeRectActionKind.SEND]
            recv = flow_actions[flow.id][MoeRectActionKind.RECV]
            wait = flow_actions[flow.id][MoeRectActionKind.WAIT]
            fsm = fsm_by_flow[flow.id]
            token = token_by_flow[flow.id]
            if action.kind is MoeRectActionKind.SEND:
                for fence in sorted(
                    fan_in_waits[action.id], key=lambda item: item[0].id,
                ):
                    add_fan_in_event(action, RecordOpcode.EVENT_WAIT, fence)
                if strict_training and flow.stage is MoeRectFlowStage.BACKWARD_GRADIENT:
                    source_abi = backward_by_rank[flow.source_rank]
                elif full_model_dataflow and flow.stage is MoeRectFlowStage.COMBINE:
                    source_abi = output_by_rank[flow.source_rank]
                elif strict_training and flow.stage is MoeRectFlowStage.BACKWARD_DX:
                    source_abi = output_by_rank[flow.source_rank]
                elif strict_training and flow.stage is MoeRectFlowStage.GATE_ALL_REDUCE:
                    source_abi = gate_gradient_by_rank[flow.source_rank]
                else:
                    source_abi = activation_by_rank[flow.source_rank]
                source = absolute(source_abi)
                remote = peer(flow.destination_rank)
                add_record("compute", core, _dte_send(action.id, flow, source, fsm, remote),
                    ((SemanticOperandId.SOURCE_ADDRESS, source, 0),),
                    ((RuntimeOperandField.DTE_FSM, fsm), (RuntimeOperandField.PEER_CORE, remote)))
            elif action.kind is MoeRectActionKind.RECV:
                if full_model_dataflow and flow.stage is MoeRectFlowStage.DISPATCH:
                    target_abi = activation_by_rank[flow.destination_rank]
                elif strict_training and flow.stage is MoeRectFlowStage.BACKWARD_GRADIENT:
                    target_abi = backward_by_rank[flow.destination_rank]
                elif strict_training and flow.stage is MoeRectFlowStage.GATE_ALL_REDUCE:
                    is_reduce = flow.assignment_refs[0].startswith("gate_gradient.reduce.")
                    target_abi = (gate_reduce_by_rank[flow.destination_rank] if is_reduce
                                  else gate_gradient_by_rank[flow.destination_rank])
                else:
                    target_abi = output_by_rank[flow.destination_rank]
                destination = absolute(target_abi)
                remote = peer(flow.source_rank)
                recv_offset = (
                    (1 + children_by_rank[flow.destination_rank].index(flow.source_rank))
                    * gate_gradient_bytes
                    if strict_training and flow.stage is MoeRectFlowStage.GATE_ALL_REDUCE
                    and flow.assignment_refs[0].startswith("gate_gradient.reduce.")
                    else 0
                )
                if recv_offset:
                    view("compute", core, SemanticOperandId.DESTINATION_ADDRESS,
                         target_abi, recv_offset, gate_gradient_bytes)
                add_record("compute", core, _dte_recv(action.id, flow, destination, fsm, token, remote),
                    ((SemanticOperandId.DESTINATION_ADDRESS, destination, recv_offset),),
                    ((RuntimeOperandField.DTE_TOKEN, token), (RuntimeOperandField.DTE_FSM, fsm), (RuntimeOperandField.PEER_CORE, remote)))
            else:
                add_record("compute", core, RelocatableRecord(action.id, RecordOpcode.DTE_WAIT, (
                    RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),
                )), (), ((RuntimeOperandField.DTE_TOKEN, token),))
                for fence in sorted(
                    fan_in_sets[action.id], key=lambda item: item[0].id,
                ):
                    add_fan_in_event(action, RecordOpcode.EVENT_SET, fence)
            continue

        state_buffers = tuple(buffer_by_state[ref] for ref in action.state_refs if ref in buffer_by_state)
        if action.kind in (MoeRectActionKind.EXPERT_SGD, MoeRectActionKind.GATE_SGD):
            if len(state_buffers) != 2:
                raise SchemaError("SGD requires parameter and gradient buffers", path="plan.actions")
            weight, gradient = state_buffers
            add_bind(action, (weight, gradient), weight)
            add_record("compute", core, RelocatableRecord(action.id, RecordOpcode.SGD_UPDATE, (
                RecordOperand.literal("weight_datatype", 1),
                RecordOperand.literal("gradient_datatype", 3),
                RecordOperand.literal("output_datatype", 1),
                RecordOperand.literal("rounding", 0),
                RecordOperand.address("weight_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(weight).id),
                RecordOperand.address("gradient_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(gradient).id),
                RecordOperand.address("updated_weight_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(weight).id),
                RecordOperand.literal("element_count", weight.size_bytes // 2),
                RecordOperand.literal("learning_rate_f64_bits", 4562254508917369340),
                RecordOperand.literal("momentum_f64_bits", 0),
            )), ((SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(weight), 0), (SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(gradient), 0), (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(weight), 0)))
        elif strict_training and action.kind in (
            MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE,
            MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE,
        ):
            gradient = gate_gradient_by_rank[action.rank]
            is_local = action.kind is MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE
            children = children_by_rank[action.rank] if is_local else ()
            source_abi = (gate_reduce_by_rank[action.rank] if children else gradient)
            input_count = len(children) + 1
            if source_abi.dtype is not DType.FP32 or gradient.dtype is not DType.FP32:
                raise SchemaError("gate gradient tree must reduce FP32 bytes", path=action.id)
            if source_abi.size_bytes < input_count * gate_gradient_bytes:
                raise SchemaError("gate rank-major FP32 SRAM staging is too short", path=action.id)
            if source_abi is not gradient:
                view("compute", core, SemanticOperandId.SOURCE_ADDRESS,
                     source_abi, 0, input_count * gate_gradient_bytes)
            add_record("compute", core, RelocatableRecord(action.id, RecordOpcode.LOCAL_REDUCE, (
                RecordOperand.literal("input_dtype", 1),
                RecordOperand.literal("accumulator_dtype", 1),
                RecordOperand.literal("output_dtype", 1),
                RecordOperand.literal("reduce_op", 1),
                RecordOperand.literal("rounding", 0),
                RecordOperand.literal("order", 0),
                RecordOperand.literal("input_count", input_count),
                RecordOperand.literal("element_count", spec.hidden_size * spec.expert_count),
                RecordOperand.literal("input_stride_bytes", gate_gradient_bytes),
                RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, absolute(source_abi).id),
                RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, absolute(gradient).id),
            )), ((SemanticOperandId.SOURCE_ADDRESS, absolute(source_abi), 0),
                 (SemanticOperandId.DESTINATION_ADDRESS, absolute(gradient), 0)))
        elif action.kind in (MoeRectActionKind.PACK, MoeRectActionKind.WEIGHTED_COMBINE, MoeRectActionKind.COMBINE_BACKWARD, MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE, MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE):
            reduction_source = (
                output if (
                    full_model_dataflow and action.kind is MoeRectActionKind.WEIGHTED_COMBINE
                ) or (
                    strict_training and action.kind is MoeRectActionKind.COMBINE_BACKWARD
                ) else activation
            )
            add_record("compute", core, RelocatableRecord(action.id, RecordOpcode.LOCAL_REDUCE, (
                RecordOperand.literal("input_dtype", 0),
                RecordOperand.literal("accumulator_dtype", 1),
                RecordOperand.literal("output_dtype", 0),
                RecordOperand.literal("reduce_op", 1),
                RecordOperand.literal("rounding", 0),
                RecordOperand.literal("order", 0),
                RecordOperand.literal("input_count", 1),
                RecordOperand.literal("element_count", activation.size_bytes // 2),
                RecordOperand.literal("input_stride_bytes", activation.size_bytes),
                RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, absolute(reduction_source).id),
                RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, absolute(output).id),
            )), ((SemanticOperandId.SOURCE_ADDRESS, absolute(reduction_source), 0), (SemanticOperandId.DESTINATION_ADDRESS, absolute(output), 0)))
        else:
            wgrad = action.kind in (
                MoeRectActionKind.EXPERT_WGRAD, MoeRectActionKind.GATE_WGRAD,
            )
            data = activation if wgrad else (
                state_buffers[0] if state_buffers else activation
            )
            input_abi = (
                backward_by_rank[action.rank]
                if strict_training and action.kind is MoeRectActionKind.EXPERT_DGRAD
                else activation
            )
            if full_model_dataflow and action.kind is MoeRectActionKind.GATE:
                # P2 declares one H×E FP16 gate tensor and exactly one gate
                # row per source assignment.  Empty gates were omitted above.
                if action.flops != len(action.assignment_refs) * 2 * spec.hidden_size * spec.expert_count:
                    raise SchemaError("gate MATMUL operation count differs from P2 plan", path=f"actions[{action.id}]")
                parameters = (
                    1, len(action.assignment_refs), spec.hidden_size,
                    spec.expert_count,
                )
            elif action.kind is MoeRectActionKind.GATE or wgrad:
                parameters = (1, 32, 1, 16)
            else:
                parameters = (
                    1, 1, max(1, spec.hidden_size), max(1, spec.intermediate_size),
                )
            add_bind(action, (input_abi,), output)
            add_record("compute", core, RelocatableRecord(action.id, RecordOpcode.MATMUL, (
                RecordOperand.literal("datatype", 1),
                RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(input_abi).id),
                RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(data).id),
                RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(output).id),
                RecordOperand.literal("parameters", parameters),
            )), ((SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(input_abi), 0), (SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(data), 0), (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(output), 0)))

    for core in cores:
        for abi in buffers_by_core[core]:
            label = label_by_buffer[abi.id]
            add_record("state", core, RelocatableRecord(last_by_core[core], RecordOpcode.SRAM_FREE, (
                RecordOperand.address("symbol", SemanticOperandId.SYMBOL, label.id),
            )), ((SemanticOperandId.SYMBOL, label, 0),))

    all_program_symbols = tuple(sorted({item.id: item for item in (
        *region_by_ref.values(), *label_by_buffer.values(), *absolute_by_buffer.values(), *hbm_by_state.values(),
    )}.values(), key=lambda item: item.id))
    state_symbol_ids = {
        relocation.symbol_ref
        for core in cores for relocation in address_relocs["state"][core]
    }
    region_symbol_ids = {item.id for item in region_by_ref.values()}
    state_program_symbols = tuple(
        item for item in all_program_symbols
        if item.id in state_symbol_ids or item.id in region_symbol_ids
    )
    compute_symbol_ids = {
        relocation.symbol_ref for core in cores for relocation in address_relocs["compute"][core]
    }
    compute_program_symbols = tuple(item for item in all_program_symbols if item.id in compute_symbol_ids)
    compute_cores = tuple(core for core in cores if records["compute"][core])
    compute_buffers = tuple(
        item for item in buffers if item.logical_core in compute_cores
    )

    state_fragment = CommandFragment.create(
        producer_pass=_LOWERING_PASS, source_global_dag_id=plan.id, kind=FragmentKind.STATE_IO,
        claimed_action_ids=tuple(sorted(action.id for action in plan.actions if action.kind in _STATE_KINDS)),
        core_streams=tuple(CoreFragmentStream(
            core, tuple(records["state"][core]), (),
            tuple(sorted(address_relocs["state"][core], key=lambda item: (item.record_index, int(item.operand_id)))),
        ) for core in cores),
        runtime_symbols=(), program_symbols=state_program_symbols,
        buffer_abi=buffers, state_abi=state_abis,
    )
    compute_fragment = CommandFragment.create(
        producer_pass=_LOWERING_PASS, source_global_dag_id=plan.id, kind=FragmentKind.COARSE,
        claimed_action_ids=tuple(sorted({record.source_global_action_id for core in cores for record in records["compute"][core]})),
        core_streams=tuple(CoreFragmentStream(
            core, tuple(records["compute"][core]),
            tuple(runtime_relocs["compute"][core]),
            tuple(sorted(address_relocs["compute"][core], key=lambda item: (item.record_index, int(item.operand_id)))),
        ) for core in compute_cores),
        runtime_symbols=tuple(sorted(runtime_symbols.values(), key=lambda item: item.id)),
        program_symbols=compute_program_symbols, buffer_abi=compute_buffers, state_abi=(),
    )
    state_fragment.validate("state_fragment")
    compute_fragment.validate("compute_fragment")
    fragments = tuple(sorted((state_fragment, compute_fragment), key=lambda item: item.id))

    buffer_by_id = {item.id: item for item in buffers}
    state_abi_by_id = {item.id: item for item in state_abis}
    buffer_by_symbol = {
        **{symbol.id: buffer_by_id[abi_id] for abi_id, symbol in label_by_buffer.items()},
        **{symbol.id: buffer_by_id[abi_id] for abi_id, symbol in absolute_by_buffer.items()},
    }
    state_by_symbol = {
        symbol.id: state_abi_by_id[state_id]
        for state_id, symbol in hbm_by_state.items()
    }
    address_bindings = []
    state_bindings = []
    for fragment in fragments:
        for stream in fragment.core_streams:
            for relocation in stream.address_relocations:
                if relocation.operand_id is SemanticOperandId.HBM_ADDRESS:
                    state_bindings.append(StateOperandBinding(fragment.id, stream.logical_core, relocation.record_index, relocation.operand_id, state_by_symbol[relocation.symbol_ref].id))
                    continue
                abi = buffer_by_symbol.get(relocation.symbol_ref)
                if relocation.operand_id is SemanticOperandId.REGION_NAME:
                    label_ref = stream.records[relocation.record_index].operands[1].symbol_ref
                    abi = buffer_by_symbol[label_ref]
                if abi is None:
                    raise SchemaError("address relocation has no rank-local BufferABI", path="fragments")
                tensor_slice = operand_views.get(
                    ("state" if fragment.id == state_fragment.id else "compute",
                     stream.logical_core, relocation.record_index, relocation.operand_id),
                    abi.tensor_slice,
                )
                address_bindings.append(AddressOperandBinding(fragment.id, stream.logical_core, relocation.record_index, relocation.operand_id, (abi.id,), (tensor_slice,)))

    definitions = [ProgramSymbolDefinition(
        shared_region, shared_region_name,
        0 if release_region else _REGION_BASE_BYTES,
        (1 << 20) if release_region else _REGION_SIZE_BYTES,
        cores,
    )]
    if not release_region:
        definitions.append(ProgramSymbolDefinition(
            comm_region, comm_region_name, _COMM_REGION_BASE_BYTES,
            _COMM_REGION_SIZE_BYTES, cores,
        ))
    for rank, core in enumerate(cores):
        for index, abi in enumerate(buffers_by_core[core]):
            definitions.append(ProgramSymbolDefinition(label_by_buffer[abi.id], f"flexible_moe_r{rank}_label_{index}", 0, 0, (core,)))
            base = 0 if release_region else (
                _REGION_BASE_BYTES
                if abi.region_ref == _REGION_REF else _COMM_REGION_BASE_BYTES
            )
            definitions.append(ProgramSymbolDefinition(absolute_by_buffer[abi.id], f"flexible_moe_r{rank}_abs_{index}", base + abi.region_offset_bytes, abi.size_bytes, (core,)))
    for index, state in enumerate(state_abis):
        definitions.append(ProgramSymbolDefinition(hbm_by_state[state.id], f"flexible_moe_hbm_{index}", state.address, state.size_bytes, (core_by_rank[state.die_id],)))
    declared_program_ids = {
        symbol.id for fragment in fragments for symbol in fragment.program_symbols
    }
    definitions = tuple(sorted(
        (item for item in definitions if item.symbol.id in declared_program_ids),
        key=lambda item: item.symbol.id,
    ))

    symbol_fragments = {
        symbol.id: tuple(fragment.id for fragment in fragments if symbol in fragment.program_symbols)
        for symbol in {
            symbol.id: symbol
            for fragment in fragments for symbol in fragment.program_symbols
        }.values()
    }
    interfaces = []
    event_credits = tuple(
        EventCredit(symbol.id, 1)
        for symbol in sorted(
            (
                item for item in runtime_symbols.values()
                if item.kind is RuntimeSymbolKind.EVENT_TAG
            ),
            key=lambda item: item.id,
        )
    )
    for fragment in fragments:
        local = tuple(symbol.id for symbol in fragment.program_symbols)
        exports = tuple(sorted(symbol for symbol in local if fragment.id == min(symbol_fragments[symbol])))
        runtime_exports = tuple(sorted(symbol.id for symbol in fragment.runtime_symbols))
        interfaces.append(FragmentInterface(
            fragment.id, (), runtime_exports,
            tuple(sorted(set(local).difference(exports))), exports,
            event_credits if fragment.id == compute_fragment.id else (),
            event_credits if fragment.id == compute_fragment.id else (),
        ))

    fragment_by_role = {"state": state_fragment, "compute": compute_fragment}
    linked_streams = []
    for core in cores:
        linked_refs = []
        for action in plan.actions:
            for physical_action_id in (action.id, *physical_children_by_action.get(action.id, ())):
                for role, record_core, record_index in refs_by_action[physical_action_id]:
                    if record_core == core:
                        linked_refs.append(LinkedRecordRef(fragment_by_role[role].id, record_index, physical_action_id))
        runtime_core_id = runtime_core_ids[core.die_id]
        linked_streams.append(LinkedCoreStream(core, runtime_core_id, tuple(linked_refs)))

    starts = []
    for core in cores:
        symbol = RuntimeSymbol(
            stable_artifact_id("flexible_moe_start_tag", {"plan_id": plan.id, "core": core}, schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION),
            RuntimeSymbolKind.START_TAG,
            first_by_core[core],
        )
        runtime_definitions.append(RuntimeSymbolDefinition(symbol, (core,), None, None))
        starts.append(LogicalStartEvent(core, symbol.id, 1))

    inputs = (
        ManifestInputDigest(ManifestInputKind.FLEXIBLE_MOE_PLAN, plan.id, plan.schema_version, canonical_digest(plan)),
        ManifestInputDigest(ManifestInputKind.FLEXIBLE_MOE_STANDARD_MAPPING, standard_ir.id, standard_ir.schema_version, canonical_digest(standard_ir)),
        *(ManifestInputDigest(ManifestInputKind.COMMAND_FRAGMENT, fragment.id, fragment.schema_version, canonical_digest(fragment)) for fragment in fragments),
    )
    manifest = LinkedProgramManifest.create(
        producer_pass=_LINKER_PASS, capabilities=0, source_ir1_id=spec.id,
        source_projection_id=standard_ir.id, source_schedule_set_id=_SCHEDULE_ID,
        source_global_dag_id=plan.id,
        input_digests=tuple(sorted(inputs, key=lambda item: (item.kind.value, item.artifact_id))),
        fragments=fragments,
        fragment_interfaces=tuple(sorted(interfaces, key=lambda item: item.fragment_id)),
        core_bindings=tuple(CoreRuntimeBinding(
            core, f"flexible_moe.die{core.die_id}.core0",
            runtime_core_ids[core.die_id],
            "flexible_moe.sram_profile0",
        ) for core in cores),
        core_streams=tuple(linked_streams),
        runtime_symbol_definitions=tuple(sorted(runtime_definitions, key=lambda item: item.symbol.id)),
        program_symbol_definitions=definitions,
        address_operand_bindings=tuple(sorted(address_bindings, key=lambda item: (item.logical_core.die_id, item.logical_core.local_core_id, item.fragment_id, item.fragment_record_index, int(item.operand_id)))),
        state_operand_bindings=tuple(sorted(state_bindings, key=lambda item: (item.logical_core.die_id, item.logical_core.local_core_id, item.fragment_id, item.fragment_record_index, int(item.operand_id)))),
        core_groups=(),
        envelope=ProgramControlEnvelope(cores, tuple(starts), cores, cores, cores, EmptyCoreAckPolicy.INCLUDE_EMPTY, ProgramFailurePolicy.ABORT_ALL),
    )
    manifest.validate("flexible_moe_multi_production_manifest")
    if len(canonical_json(manifest).encode("utf-8")) > spec.limits.max_artifact_file_bytes:
        raise SchemaError(
            "linked manifest file capacity exceeded",
            path="flexible_moe_multi_production_manifest",
        )
    result = FlexibleMoeProductionArtifacts(standard_ir, fragments, manifest, True, False)
    result.validate_against(plan, spec, allow_zero_work_omission=full_model_dataflow)
    return result


__all__ = ["expert_projection_action_ids", "lower_link_flexible_moe_multi"]
