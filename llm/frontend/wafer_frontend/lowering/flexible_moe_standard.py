"""Lower/link flexible MoE v2 into a strict standard timing manifest."""

from __future__ import annotations

from ..schema.artifact_manifest import RecordOpcode
from ..schema.flexible_moe import (
    FlexibleMoeExecutablePlan,
    FlexibleMoeMode,
    FlexibleMoeSpec,
    MoeRectActionKind,
    MoeRectStateRole,
)
from ..schema.flexible_moe_standard import (
    FlexibleMoeCoreStream,
    FlexibleMoeEndpointAbi,
    FlexibleMoeIoRole,
    FlexibleMoeStandardLoweringPlan,
    FlexibleMoeProgramIoEntry,
    FlexibleMoeStandardProgramIoPlan,
    FlexibleMoeStandardRecord,
    FlexibleMoeStandardStateAbi,
    FlexibleMoeStateAccess,
)
from ..schema.serde import canonical_digest


_OPCODE_BY_ACTION = {
    MoeRectActionKind.STATE_LOAD: RecordOpcode.LSU_LOAD,
    MoeRectActionKind.GATE: RecordOpcode.MATMUL,
    MoeRectActionKind.PACK: RecordOpcode.SRAM_BIND,
    MoeRectActionKind.SEND: RecordOpcode.DTE_SEND,
    MoeRectActionKind.RECV: RecordOpcode.DTE_RECV,
    MoeRectActionKind.WAIT: RecordOpcode.DTE_WAIT,
    MoeRectActionKind.EXPERT_FORWARD: RecordOpcode.MATMUL,
    MoeRectActionKind.WEIGHTED_COMBINE: RecordOpcode.LOCAL_REDUCE,
    MoeRectActionKind.EXPERT_DGRAD: RecordOpcode.MATMUL,
    MoeRectActionKind.EXPERT_WGRAD: RecordOpcode.MATMUL,
    MoeRectActionKind.COMBINE_BACKWARD: RecordOpcode.LOCAL_REDUCE,
    MoeRectActionKind.GATE_WGRAD: RecordOpcode.MATMUL,
    MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE: RecordOpcode.LOCAL_REDUCE,
    MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE: RecordOpcode.DTE_ISSUE,
    MoeRectActionKind.EXPERT_SGD: RecordOpcode.SGD_UPDATE,
    MoeRectActionKind.GATE_SGD: RecordOpcode.SGD_UPDATE,
    MoeRectActionKind.STATE_STORE: RecordOpcode.LSU_STORE,
}


def _align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def _state_abi(
    plan: FlexibleMoeExecutablePlan,
    spec: FlexibleMoeSpec,
) -> tuple[FlexibleMoeStandardStateAbi, ...]:
    by_rank = {rank: [] for rank in range(spec.mesh.rank_count)}
    for state in plan.state_bindings:
        by_rank[state.owner_rank].append(state)
    result = []
    for rank in range(spec.mesh.rank_count):
        address = 0x10000000 + rank * 0x01000000
        for state in by_rank[rank]:
            address = _align(address)
            access = (
                FlexibleMoeStateAccess.SCRATCH
                if state.role in (
                    MoeRectStateRole.EXPERT_GRADIENT,
                    MoeRectStateRole.GATE_GRADIENT,
                )
                else (
                    FlexibleMoeStateAccess.READ_WRITE
                    if spec.mode is FlexibleMoeMode.TRAIN
                    else FlexibleMoeStateAccess.READ_ONLY
                )
            )
            result.append(FlexibleMoeStandardStateAbi(
                state_ref=state.id,
                runtime_core_id=rank,
                address=address,
                size_bytes=state.size_bytes,
                dtype=state.dtype,
                access=access,
                persistent=state.persistent,
            ))
            address = _align(address + state.size_bytes)
    return tuple(result)


def plan_flexible_moe_standard_mapping(
    plan: FlexibleMoeExecutablePlan,
    spec: FlexibleMoeSpec,
) -> FlexibleMoeStandardLoweringPlan:
    """Build a deterministic opcode/ABI mapping plan, not a linked artifact."""

    plan.validate_against(spec)
    flow_index = {item.id: item for item in plan.flows}
    tag_by_flow = {item.id: index + 1 for index, item in enumerate(plan.flows)}
    record_by_action = {}
    records = []
    for action in plan.actions:
        flow = None if action.flow_ref is None else flow_index[action.flow_ref]
        peer = None
        tag = None
        wave = None
        if flow is not None:
            tag = tag_by_flow[flow.id]
            wave = flow.wave_index
            peer = (
                flow.destination_rank
                if action.kind is MoeRectActionKind.SEND
                else flow.source_rank
            )
        record = FlexibleMoeStandardRecord.create(
            source_action_ref=action.id,
            runtime_core_id=action.rank,
            opcode=_OPCODE_BY_ACTION[action.kind],
            dependency_record_refs=tuple(
                record_by_action[ref].id for ref in action.deps
            ),
            flow_ref=action.flow_ref,
            peer_runtime_core_id=peer,
            transport_tag=tag,
            wave_index=wave,
            logical_bytes=action.logical_bytes,
            flops=action.flops,
            state_refs=action.state_refs,
        )
        record_by_action[action.id] = record
        records.append(record)
    streams = tuple(
        FlexibleMoeCoreStream(
            rank,
            tuple(item for item in records if item.runtime_core_id == rank),
        )
        for rank in range(spec.mesh.rank_count)
    )
    endpoint_abi = []
    actions_by_flow = {}
    for action in plan.actions:
        if action.flow_ref is not None:
            actions_by_flow.setdefault(action.flow_ref, {})[action.kind] = action
    for flow in plan.flows:
        by_kind = {
            kind: record_by_action[action.id]
            for kind, action in actions_by_flow[flow.id].items()
        }
        endpoint_abi.append(FlexibleMoeEndpointAbi(
            flow_ref=flow.id,
            transport_tag=tag_by_flow[flow.id],
            stage=flow.stage.value,
            wave_index=flow.wave_index,
            source_runtime_core_id=flow.source_rank,
            destination_runtime_core_id=flow.destination_rank,
            logical_bytes=flow.logical_bytes,
            die_path=flow.die_path,
            send_record_ref=by_kind[MoeRectActionKind.SEND].id,
            recv_record_ref=by_kind[MoeRectActionKind.RECV].id,
            wait_record_ref=by_kind[MoeRectActionKind.WAIT].id,
        ))
    manifest = FlexibleMoeStandardLoweringPlan.create(
        source_plan_id=plan.id,
        source_plan_digest=canonical_digest(plan),
        source_spec_id=spec.id,
        source_spec_digest=spec.digest,
        core_streams=streams,
        state_abi=_state_abi(plan, spec),
        endpoint_abi=tuple(endpoint_abi),
        record_count=len(records),
        max_sessions_per_rank_wave=plan.max_sessions_per_rank_wave,
        standard_mapping_verified=True,
        lower_link_verified=False,
        runtime_verified=False,
    )
    manifest.validate_against(plan, spec)
    return manifest


def build_flexible_moe_standard_program_io_plan(
    manifest: FlexibleMoeStandardLoweringPlan,
    plan: FlexibleMoeExecutablePlan,
    spec: FlexibleMoeSpec,
    program_artifact_sha256: str,
) -> FlexibleMoeStandardProgramIoPlan:
    """Build planned timing initialization/probe coverage; no artifact claim."""

    manifest.validate_against(plan, spec)
    entries = []
    for state in manifest.state_abi:
        if state.persistent:
            entries.append(FlexibleMoeProgramIoEntry(
                role=FlexibleMoeIoRole.STATE_INITIALIZATION,
                runtime_core_id=state.runtime_core_id,
                size_bytes=state.size_bytes,
                state_ref=state.state_ref,
                terminal_action_ref=None,
                address=state.address,
                zero_fill=True,
            ))
    if spec.mode is FlexibleMoeMode.INFERENCE:
        action_index = {item.id: item for item in plan.actions}
        token_bytes = spec.hidden_size * 2
        for terminal_ref in plan.terminal_action_refs:
            action = action_index[terminal_ref]
            entries.append(FlexibleMoeProgramIoEntry(
                role=FlexibleMoeIoRole.OUTPUT_PROBE,
                runtime_core_id=action.rank,
                size_bytes=max(1, len(action.assignment_refs)) * token_bytes,
                state_ref=None,
                terminal_action_ref=terminal_ref,
                address=None,
                zero_fill=False,
            ))
    else:
        for state in manifest.state_abi:
            if state.persistent:
                entries.append(FlexibleMoeProgramIoEntry(
                    role=FlexibleMoeIoRole.UPDATED_STATE_PROBE,
                    runtime_core_id=state.runtime_core_id,
                    size_bytes=state.size_bytes,
                    state_ref=state.state_ref,
                    terminal_action_ref=None,
                    address=state.address,
                    zero_fill=False,
                ))
    program_io = FlexibleMoeStandardProgramIoPlan.create(
        source_manifest_id=manifest.id,
        source_manifest_digest=manifest.digest,
        program_artifact_sha256=program_artifact_sha256,
        entries=tuple(entries),
        timing_execution=True,
        runtime_verified=False,
    )
    program_io.validate_against(manifest, plan, spec)
    return program_io


__all__ = [
    "build_flexible_moe_standard_program_io_plan",
    "plan_flexible_moe_standard_mapping",
]
