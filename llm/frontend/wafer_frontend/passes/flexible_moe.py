"""Deterministic Direct-XY executable baseline for flexible-Mesh MoE."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.flexible_moe import (
    FlexibleMoeExecutablePlan,
    FlexibleMoeLimits,
    FlexibleMoeMode,
    FlexibleMoeSpec,
    MoeRectAction,
    MoeRectActionKind,
    MoeRectFlow,
    MoeRectFlowStage,
    MoeRectGateAllReduce,
    MoeRectStateBinding,
    MoeRectStateRole,
    MoeRectStaticTrace,
    MoeRectTraceAssignment,
)
from ..schema.rect_mesh import RectMeshSpec


_FLOW_STAGE_ORDER = {
    MoeRectFlowStage.DISPATCH: 0,
    MoeRectFlowStage.COMBINE: 1,
    MoeRectFlowStage.BACKWARD_GRADIENT: 2,
    MoeRectFlowStage.BACKWARD_DX: 3,
    MoeRectFlowStage.GATE_ALL_REDUCE: 4,
}


def _assignment_ref(assignment: MoeRectTraceAssignment) -> str:
    return f"assignment.{assignment.token_index}"


def _xy_path(mesh: RectMeshSpec, source: int, destination: int) -> tuple[int, ...]:
    source_x, source_y = mesh.coordinate(source)
    destination_x, destination_y = mesh.coordinate(destination)
    path = [source]
    while source_x != destination_x:
        source_x += 1 if destination_x > source_x else -1
        path.append(mesh.rank(source_y, source_x))
    while source_y != destination_y:
        source_y += 1 if destination_y > source_y else -1
        path.append(mesh.rank(source_y, source_x))
    return tuple(path)


def _flows(spec: FlexibleMoeSpec) -> tuple[MoeRectFlow, ...]:
    token_bytes = spec.hidden_size * 2
    stages = (
        (
            MoeRectFlowStage.DISPATCH,
            lambda item: (item.source_rank, item.expert_home_rank),
        ),
        (
            MoeRectFlowStage.COMBINE,
            lambda item: (item.expert_home_rank, item.source_rank),
        ),
    )
    if spec.mode is FlexibleMoeMode.TRAIN:
        stages += (
            (
                MoeRectFlowStage.BACKWARD_GRADIENT,
                lambda item: (item.source_rank, item.expert_home_rank),
            ),
            (
                MoeRectFlowStage.BACKWARD_DX,
                lambda item: (item.expert_home_rank, item.source_rank),
            ),
        )
    result = []
    for stage, endpoints in stages:
        buckets: dict[tuple[int, int], list[MoeRectTraceAssignment]] = defaultdict(list)
        for assignment in spec.trace.assignments:
            source, destination = endpoints(assignment)
            if source != destination:
                buckets[(source, destination)].append(assignment)
        for source, destination in sorted(
            buckets,
            key=lambda pair: ((pair[1] - pair[0]) % spec.mesh.rank_count, *pair),
        ):
            assignments = tuple(buckets[(source, destination)])
            result.append(
                MoeRectFlow.create(
                    stage=stage,
                    source_rank=source,
                    destination_rank=destination,
                    wave_index=(destination - source) % spec.mesh.rank_count,
                    assignment_refs=tuple(_assignment_ref(item) for item in assignments),
                    logical_bytes=len(assignments) * token_bytes,
                    die_path=_xy_path(spec.mesh, source, destination),
                )
            )
    if spec.mode is FlexibleMoeMode.TRAIN:
        gate_gradient_bytes = spec.hidden_size * spec.expert_count * 4
        for rank in range(1, spec.mesh.rank_count):
            parent = (rank - 1) // 2
            for source, destination, phase in (
                (rank, parent, "reduce"),
                (parent, rank, "broadcast"),
            ):
                result.append(MoeRectFlow.create(
                    stage=MoeRectFlowStage.GATE_ALL_REDUCE,
                    source_rank=source,
                    destination_rank=destination,
                    wave_index=(destination - source) % spec.mesh.rank_count,
                    assignment_refs=(f"gate_gradient.{phase}.rank.{rank}",),
                    logical_bytes=gate_gradient_bytes,
                    die_path=_xy_path(spec.mesh, source, destination),
                ))
    return tuple(
        sorted(
            result,
            key=lambda item: (
                _FLOW_STAGE_ORDER[item.stage], item.wave_index,
                item.source_rank, item.destination_rank,
            ),
        )
    )


def _states(spec: FlexibleMoeSpec) -> tuple[MoeRectStateBinding, ...]:
    expert_parameter_bytes = 3 * spec.hidden_size * spec.intermediate_size * 2
    expert_gradient_bytes = 3 * spec.hidden_size * spec.intermediate_size * 4
    gate_parameter_bytes = spec.hidden_size * spec.expert_count * 2
    gate_gradient_bytes = spec.hidden_size * spec.expert_count * 4
    result = []
    for rank in range(spec.mesh.rank_count):
        result.extend(
            (
                MoeRectStateBinding.create(
                    role=MoeRectStateRole.EXPERT_PARAMETER,
                    owner_rank=rank,
                    expert_index=rank,
                    dtype=DType.FP16,
                    size_bytes=expert_parameter_bytes,
                    persistent=True,
                ),
                MoeRectStateBinding.create(
                    role=MoeRectStateRole.GATE_PARAMETER,
                    owner_rank=rank,
                    expert_index=None,
                    dtype=DType.FP16,
                    size_bytes=gate_parameter_bytes,
                    persistent=True,
                ),
            )
        )
        if spec.mode is FlexibleMoeMode.TRAIN:
            result.extend(
                (
                    MoeRectStateBinding.create(
                        role=MoeRectStateRole.EXPERT_GRADIENT,
                        owner_rank=rank,
                        expert_index=rank,
                        dtype=DType.FP32,
                        size_bytes=expert_gradient_bytes,
                        persistent=False,
                    ),
                    MoeRectStateBinding.create(
                        role=MoeRectStateRole.GATE_GRADIENT,
                        owner_rank=rank,
                        expert_index=None,
                        dtype=DType.FP32,
                        size_bytes=gate_gradient_bytes,
                        persistent=False,
                    ),
                )
            )
    return tuple(result)


def _preflight(spec: FlexibleMoeSpec, flows: tuple[MoeRectFlow, ...]) -> None:
    ranks = spec.mesh.rank_count
    state_count = ranks * (4 if spec.mode is FlexibleMoeMode.TRAIN else 2)
    action_count = (
        (7 if spec.mode is FlexibleMoeMode.INFERENCE else 16) * ranks
        + 3 * len(flows)
    )
    record_count = action_count * 4 + len(flows) * 3 + state_count * 2
    file_bytes = record_count * 48 + state_count * 64
    if (
        len(flows) > spec.limits.max_flows
        or action_count > spec.limits.max_actions
        or state_count > spec.limits.max_state_bindings
        or record_count > spec.limits.max_records
        or file_bytes > spec.limits.max_artifact_file_bytes
    ):
        raise SchemaError("flexible MoE capacity preflight failed", path="flexible_moe_spec.limits")
    if flows and spec.limits.max_sessions_per_rank_wave < 2:
        raise SchemaError("remote traffic requires two sessions per rank/wave", path="flexible_moe_spec.limits.max_sessions_per_rank_wave")


class _Actions:
    def __init__(self) -> None:
        self.items: list[MoeRectAction] = []

    def add(
        self,
        rank: int,
        kind: MoeRectActionKind,
        deps: tuple[str, ...],
        *,
        assignments: tuple[str, ...] = (),
        flow: MoeRectFlow | None = None,
        logical_bytes: int = 0,
        flops: int = 0,
        states: tuple[str, ...] = (),
    ) -> MoeRectAction:
        action = MoeRectAction.create(
            rank=rank,
            kind=kind,
            deps=tuple(dict.fromkeys(deps)),
            assignment_refs=assignments,
            flow_ref=None if flow is None else flow.id,
            logical_bytes=logical_bytes,
            flops=flops,
            state_refs=states,
        )
        self.items.append(action)
        return action


def _transport_actions(
    actions: _Actions,
    flows: tuple[MoeRectFlow, ...],
    source_dep: dict[int, str],
) -> dict[str, str]:
    waits = {}
    incoming_counts: dict[int, int] = {}
    outgoing_counts: dict[int, int] = {}
    for flow in flows:
        incoming_counts[flow.destination_rank] = incoming_counts.get(flow.destination_rank, 0) + 1
        outgoing_counts[flow.source_rank] = outgoing_counts.get(flow.source_rank, 0) + 1
    segmented_fan_in = max(incoming_counts.values(), default=0) > 3
    segmented_fan_out = max(outgoing_counts.values(), default=0) > 3
    if segmented_fan_in and segmented_fan_out:
        raise SchemaError(
            "simultaneous high-degree fan-in and fan-out is not supported",
            path="flexible_moe_spec.trace",
        )
    if segmented_fan_in:
        ordered_flows = tuple(sorted(
            flows,
            key=lambda item: (
                item.destination_rank, item.wave_index, item.source_rank, item.id,
            ),
        ))
    elif segmented_fan_out:
        ordered_flows = tuple(sorted(
            flows,
            key=lambda item: (
                item.source_rank, item.wave_index, item.destination_rank, item.id,
            ),
        ))
    else:
        ordered_flows = flows
    previous_wait_by_endpoint: dict[int, str] = {}
    for flow in ordered_flows:
        deps = [source_dep[flow.source_rank]]
        endpoint = (
            flow.destination_rank if segmented_fan_in else flow.source_rank
        )
        previous_wait = previous_wait_by_endpoint.get(endpoint)
        if (segmented_fan_in or segmented_fan_out) and previous_wait is not None:
            deps.append(previous_wait)
        send = actions.add(
            flow.source_rank,
            MoeRectActionKind.SEND,
            tuple(deps),
            assignments=flow.assignment_refs,
            flow=flow,
            logical_bytes=flow.logical_bytes,
        )
        recv = actions.add(
            flow.destination_rank,
            MoeRectActionKind.RECV,
            (),
            assignments=flow.assignment_refs,
            flow=flow,
            logical_bytes=flow.logical_bytes,
        )
        wait = actions.add(
            flow.destination_rank,
            MoeRectActionKind.WAIT,
            (send.id, recv.id),
            assignments=flow.assignment_refs,
            flow=flow,
        )
        waits[flow.id] = wait.id
        if segmented_fan_in or segmented_fan_out:
            previous_wait_by_endpoint[endpoint] = wait.id
    return waits


def compile_flexible_moe_baseline(spec: FlexibleMoeSpec) -> FlexibleMoeExecutablePlan:
    """Compile a finite timing-only Direct-XY inference or SGD training step."""

    spec.validate()
    flows = _flows(spec)
    _preflight(spec, flows)
    states = _states(spec)
    by_rank_role = {(item.owner_rank, item.role): item for item in states}
    assignments_by_source: dict[int, tuple[str, ...]] = {}
    assignments_by_expert: dict[int, tuple[str, ...]] = {}
    for rank in range(spec.mesh.rank_count):
        assignments_by_source[rank] = tuple(
            _assignment_ref(item) for item in spec.trace.assignments
            if item.source_rank == rank
        )
        assignments_by_expert[rank] = tuple(
            _assignment_ref(item) for item in spec.trace.assignments
            if item.expert_home_rank == rank
        )
    by_stage = {
        stage: tuple(item for item in flows if item.stage is stage)
        for stage in MoeRectFlowStage
    }
    actions = _Actions()
    expert_load, gate_load, gate, pack = {}, {}, {}, {}
    for rank in range(spec.mesh.rank_count):
        parameter_refs = (
            by_rank_role[(rank, MoeRectStateRole.EXPERT_PARAMETER)].id,
            by_rank_role[(rank, MoeRectStateRole.GATE_PARAMETER)].id,
        )
        expert_load[rank] = actions.add(
            rank, MoeRectActionKind.STATE_LOAD, (), states=(parameter_refs[0],),
        )
        gate_load[rank] = actions.add(
            rank, MoeRectActionKind.STATE_LOAD, (), states=(parameter_refs[1],),
        )
        gate[rank] = actions.add(
            rank, MoeRectActionKind.GATE, (gate_load[rank].id,),
            assignments=assignments_by_source[rank],
            flops=len(assignments_by_source[rank]) * 2 * spec.hidden_size * spec.expert_count,
            states=(parameter_refs[1],),
        )
        pack[rank] = actions.add(
            rank, MoeRectActionKind.PACK, (gate[rank].id,),
            assignments=assignments_by_source[rank],
        )

    dispatch_waits = _transport_actions(
        actions, by_stage[MoeRectFlowStage.DISPATCH],
        {rank: pack[rank].id for rank in pack},
    )
    expert = {}
    for rank in range(spec.mesh.rank_count):
        incoming = tuple(
            dispatch_waits[item.id]
            for item in by_stage[MoeRectFlowStage.DISPATCH]
            if item.destination_rank == rank
        )
        local = (pack[rank].id,) if any(
            item.source_rank == rank and item.expert_home_rank == rank
            for item in spec.trace.assignments
        ) else ()
        expert[rank] = actions.add(
            rank, MoeRectActionKind.EXPERT_FORWARD,
            (expert_load[rank].id, *incoming, *local),
            assignments=assignments_by_expert[rank],
            flops=len(assignments_by_expert[rank]) * 6 * spec.hidden_size * spec.intermediate_size,
            states=(by_rank_role[(rank, MoeRectStateRole.EXPERT_PARAMETER)].id,),
        )
    combine_waits = _transport_actions(
        actions, by_stage[MoeRectFlowStage.COMBINE],
        {rank: expert[rank].id for rank in expert},
    )
    combine = {}
    for rank in range(spec.mesh.rank_count):
        incoming = tuple(
            combine_waits[item.id]
            for item in by_stage[MoeRectFlowStage.COMBINE]
            if item.destination_rank == rank
        )
        local = (expert[rank].id,) if any(
            item.source_rank == rank and item.expert_home_rank == rank
            for item in spec.trace.assignments
        ) else ()
        combine[rank] = actions.add(
            rank, MoeRectActionKind.WEIGHTED_COMBINE,
            (*incoming, *local, gate[rank].id),
            assignments=assignments_by_source[rank],
            flops=len(assignments_by_source[rank]) * 2 * spec.hidden_size,
        )

    gate_ar = None
    terminals = tuple(combine[rank].id for rank in sorted(combine))
    if spec.mode is FlexibleMoeMode.INFERENCE:
        retention_stores = tuple(
            actions.add(
                rank,
                MoeRectActionKind.STATE_STORE,
                (combine[rank].id,),
                states=(
                    by_rank_role[(rank, MoeRectStateRole.EXPERT_PARAMETER)].id,
                ),
            )
            for rank in range(spec.mesh.rank_count)
        )
        terminals = tuple(item.id for item in retention_stores)
    if spec.mode is FlexibleMoeMode.TRAIN:
        backward_waits = _transport_actions(
            actions, by_stage[MoeRectFlowStage.BACKWARD_GRADIENT],
            {rank: combine[rank].id for rank in combine},
        )
        dgrad, wgrad = {}, {}
        for rank in range(spec.mesh.rank_count):
            incoming = tuple(
                backward_waits[item.id]
                for item in by_stage[MoeRectFlowStage.BACKWARD_GRADIENT]
                if item.destination_rank == rank
            )
            deps = (expert[rank].id, *incoming)
            dgrad[rank] = actions.add(
                rank, MoeRectActionKind.EXPERT_DGRAD, deps,
                assignments=assignments_by_expert[rank],
                flops=len(assignments_by_expert[rank]) * 6 * spec.hidden_size * spec.intermediate_size,
                states=(by_rank_role[(rank, MoeRectStateRole.EXPERT_PARAMETER)].id,),
            )
            wgrad[rank] = actions.add(
                rank, MoeRectActionKind.EXPERT_WGRAD, deps,
                assignments=assignments_by_expert[rank],
                flops=len(assignments_by_expert[rank]) * 6 * spec.hidden_size * spec.intermediate_size,
                states=(by_rank_role[(rank, MoeRectStateRole.EXPERT_GRADIENT)].id,),
            )
        dx_waits = _transport_actions(
            actions, by_stage[MoeRectFlowStage.BACKWARD_DX],
            {rank: dgrad[rank].id for rank in dgrad},
        )
        combine_backward, gate_wgrad = {}, {}
        for rank in range(spec.mesh.rank_count):
            incoming = tuple(
                dx_waits[item.id]
                for item in by_stage[MoeRectFlowStage.BACKWARD_DX]
                if item.destination_rank == rank
            )
            local = (dgrad[rank].id,) if any(
                item.source_rank == rank and item.expert_home_rank == rank
                for item in spec.trace.assignments
            ) else ()
            combine_backward[rank] = actions.add(
                rank, MoeRectActionKind.COMBINE_BACKWARD,
                (*incoming, *local, combine[rank].id),
                assignments=assignments_by_source[rank],
            )
            gate_wgrad[rank] = actions.add(
                rank, MoeRectActionKind.GATE_WGRAD,
                (combine_backward[rank].id,),
                assignments=assignments_by_source[rank],
                flops=len(assignments_by_source[rank]) * 2 * spec.hidden_size * spec.expert_count,
                states=(by_rank_role[(rank, MoeRectStateRole.GATE_GRADIENT)].id,),
            )
        gate_gradient_bytes = spec.hidden_size * spec.expert_count * 4
        gate_ar = MoeRectGateAllReduce(
            participant_ranks=tuple(range(spec.mesh.rank_count)),
            logical_bytes_per_rank=gate_gradient_bytes,
            wave_count=max(0, spec.mesh.rank_count - 1),
            max_sessions_per_rank_wave=0 if spec.mesh.rank_count == 1 else 2,
        )
        gate_ar.validate()
        gate_flows = by_stage[MoeRectFlowStage.GATE_ALL_REDUCE]
        gate_reduce_flows = tuple(
            item for item in gate_flows
            if item.assignment_refs[0].startswith("gate_gradient.reduce.")
        )
        gate_broadcast_flows = tuple(
            item for item in gate_flows
            if item.assignment_refs[0].startswith("gate_gradient.broadcast.")
        )
        reduce_by_child = {item.source_rank: item for item in gate_reduce_flows}
        broadcast_by_child = {item.destination_rank: item for item in gate_broadcast_flows}
        children = {
            rank: tuple(
                child for child in (2 * rank + 1, 2 * rank + 2)
                if child < spec.mesh.rank_count
            )
            for rank in range(spec.mesh.rank_count)
        }
        depth = {}
        for rank in range(spec.mesh.rank_count):
            value, rank_depth = rank, 0
            while value > 0:
                value = (value - 1) // 2
                rank_depth += 1
            depth[rank] = rank_depth
        reduce_waits: dict[int, str] = {}
        local_reduce = {}
        for current_depth in range(max(depth.values(), default=0), -1, -1):
            for rank in range(spec.mesh.rank_count):
                if depth[rank] != current_depth:
                    continue
                local_reduce[rank] = actions.add(
                    rank,
                    MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE,
                    (gate_wgrad[rank].id, *(reduce_waits[child] for child in children[rank])),
                    logical_bytes=gate_gradient_bytes,
                    states=(by_rank_role[(rank, MoeRectStateRole.GATE_GRADIENT)].id,),
                )
                if rank == 0:
                    continue
                flow = reduce_by_child[rank]
                send = actions.add(
                    rank, MoeRectActionKind.SEND, (local_reduce[rank].id,),
                    assignments=flow.assignment_refs, flow=flow,
                    logical_bytes=flow.logical_bytes,
                )
                recv = actions.add(
                    flow.destination_rank, MoeRectActionKind.RECV, (),
                    assignments=flow.assignment_refs, flow=flow,
                    logical_bytes=flow.logical_bytes,
                )
                wait = actions.add(
                    flow.destination_rank, MoeRectActionKind.WAIT, (send.id, recv.id),
                    assignments=flow.assignment_refs, flow=flow,
                )
                reduce_waits[rank] = wait.id
        gate_ar_actions = {0: actions.add(
            0, MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE,
            (local_reduce[0].id,), logical_bytes=gate_gradient_bytes,
            states=(by_rank_role[(0, MoeRectStateRole.GATE_GRADIENT)].id,),
        )}
        for current_depth in range(max(depth.values(), default=0)):
            for parent in range(spec.mesh.rank_count):
                if depth[parent] != current_depth:
                    continue
                for rank in children[parent]:
                    flow = broadcast_by_child[rank]
                    send = actions.add(
                        parent, MoeRectActionKind.SEND, (gate_ar_actions[parent].id,),
                        assignments=flow.assignment_refs, flow=flow,
                        logical_bytes=flow.logical_bytes,
                    )
                    recv = actions.add(
                        rank, MoeRectActionKind.RECV, (),
                        assignments=flow.assignment_refs, flow=flow,
                        logical_bytes=flow.logical_bytes,
                    )
                    wait = actions.add(
                        rank, MoeRectActionKind.WAIT, (send.id, recv.id),
                        assignments=flow.assignment_refs, flow=flow,
                    )
                    gate_ar_actions[rank] = actions.add(
                        rank, MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE,
                        (local_reduce[rank].id, wait.id),
                        logical_bytes=gate_gradient_bytes,
                        states=(by_rank_role[(rank, MoeRectStateRole.GATE_GRADIENT)].id,),
                    )
        stores = []
        for rank in range(spec.mesh.rank_count):
            expert_sgd = actions.add(
                rank, MoeRectActionKind.EXPERT_SGD, (wgrad[rank].id,),
                flops=3 * spec.hidden_size * spec.intermediate_size,
                states=(
                    by_rank_role[(rank, MoeRectStateRole.EXPERT_PARAMETER)].id,
                    by_rank_role[(rank, MoeRectStateRole.EXPERT_GRADIENT)].id,
                ),
            )
            gate_sgd = actions.add(
                rank, MoeRectActionKind.GATE_SGD, (gate_ar_actions[rank].id,),
                flops=spec.hidden_size * spec.expert_count,
                states=(
                    by_rank_role[(rank, MoeRectStateRole.GATE_PARAMETER)].id,
                    by_rank_role[(rank, MoeRectStateRole.GATE_GRADIENT)].id,
                ),
            )
            expert_store = actions.add(
                rank, MoeRectActionKind.STATE_STORE,
                (expert_sgd.id,),
                states=(by_rank_role[(rank, MoeRectStateRole.EXPERT_PARAMETER)].id,),
            )
            gate_store = actions.add(
                rank, MoeRectActionKind.STATE_STORE,
                (gate_sgd.id,),
                states=(by_rank_role[(rank, MoeRectStateRole.GATE_PARAMETER)].id,),
            )
            stores.extend((expert_store, gate_store))
        terminals = tuple(item.id for item in stores)

    symbolic_record_count = len(actions.items) * 4 + len(flows) * 3 + len(states) * 2
    plan = FlexibleMoeExecutablePlan.create(
        source_spec_id=spec.id,
        source_spec_digest=spec.digest,
        mesh_digest=spec.mesh.digest,
        actions=tuple(actions.items),
        flows=flows,
        state_bindings=states,
        gate_all_reduce=gate_ar,
        terminal_action_refs=terminals,
        symbolic_record_count=symbolic_record_count,
        symbolic_file_bytes=symbolic_record_count * 48 + len(states) * 64,
        timing_execution=True,
        functional_execution=False,
    )
    plan.validate_against(spec)
    return plan


def build_round_robin_flexible_moe_spec(
    mesh: RectMeshSpec,
    mode: FlexibleMoeMode,
    *,
    tokens_per_rank: int = 1,
    routing_shift: int = 1,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    limits: FlexibleMoeLimits = FlexibleMoeLimits(),
) -> FlexibleMoeSpec:
    """Build a deterministic balanced trace; shift=0 is the all-local case."""

    mesh.validate()
    if type(tokens_per_rank) is not int or tokens_per_rank <= 0:
        raise SchemaError("must be positive", path="tokens_per_rank")
    if type(routing_shift) is not int or not 0 <= routing_shift < mesh.rank_count:
        raise SchemaError("must lie in [0, R)", path="routing_shift")
    slot_by_expert = [0] * mesh.rank_count
    assignments = []
    for token in range(mesh.rank_count * tokens_per_rank):
        source = token % mesh.rank_count
        expert = (source + routing_shift) % mesh.rank_count
        assignments.append(MoeRectTraceAssignment(
            token_index=token,
            source_rank=source,
            expert_index=expert,
            expert_home_rank=expert,
            slot_index=slot_by_expert[expert],
        ))
        slot_by_expert[expert] += 1
    trace = MoeRectStaticTrace.create(
        token_count=len(assignments),
        expert_count=mesh.rank_count,
        capacity_per_expert=max(slot_by_expert),
        assignments=tuple(assignments),
    )
    return FlexibleMoeSpec.create(
        mesh=mesh,
        mode=mode,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        expert_count=mesh.rank_count,
        expert_parallel_degree=mesh.rank_count,
        top_k=1,
        trace_mode="static",
        trace=trace,
        limits=limits,
        expert_dtype=DType.FP16,
        combine_dtype=DType.FP32,
        token_drop=False,
    )


def adapt_ep4_scale_spec(
    scale_spec: object,
    *,
    mode: FlexibleMoeMode = FlexibleMoeMode.INFERENCE,
    limits: FlexibleMoeLimits = FlexibleMoeLimits(),
) -> FlexibleMoeSpec:
    """Map a frozen 2x2/EP4 scale point into the v2 carrier without mutation."""

    from ..schema.swizzle_moe_scale import MoeSwizzleScaleSpec

    if type(scale_spec) is not MoeSwizzleScaleSpec:
        raise SchemaError("must be a frozen MoeSwizzleScaleSpec", path="scale_spec")
    scale_spec.validate()
    assignments = tuple(
        MoeRectTraceAssignment(
            token_index=item.token_index,
            source_rank=scale_spec.token_source_die_ids[item.token_index],
            expert_index=item.expert_index,
            expert_home_rank=scale_spec.expert_home_die_ids[item.expert_index],
            slot_index=item.slot_index,
        )
        for item in scale_spec.trace.assignments
    )
    trace = MoeRectStaticTrace.create(
        token_count=scale_spec.tokens,
        expert_count=4,
        capacity_per_expert=scale_spec.capacity_per_expert,
        assignments=assignments,
    )
    return FlexibleMoeSpec.create(
        mesh=RectMeshSpec(2, 2),
        mode=mode,
        hidden_size=scale_spec.hidden_size,
        intermediate_size=scale_spec.intermediate_size,
        expert_count=4,
        expert_parallel_degree=4,
        top_k=1,
        trace_mode="static",
        trace=trace,
        limits=limits,
        expert_dtype=DType.FP16,
        combine_dtype=DType.FP32,
        token_drop=False,
    )


def compile_flexible_moe_signed_top1_train_source(
    spec: FlexibleMoeSpec, signed_source,
) -> FlexibleMoeExecutablePlan:
    """Explicit opt-in signed router P2 source; old compiler remains unchanged.

    This still lacks real shared-backbone dCombined, route SRAM and a linked
    public 0x27/0x28 executor. Existing baseline consumers reject the producer.
    """
    from .moe_signed_router_train_source_plan import compile_moe_signed_top1_train_source_plan
    return compile_moe_signed_top1_train_source_plan(
        spec, signed_source=signed_source,
    )


__all__ = [
    "compile_flexible_moe_signed_top1_train_source",
    "adapt_ep4_scale_spec",
    "build_round_robin_flexible_moe_spec",
    "compile_flexible_moe_baseline",
]
