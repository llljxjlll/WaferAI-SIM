"""Project a W7 Swizzle adapter into the isolated executable W8 carrier."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..policies.swizzle.materialize import SwizzleFusionPlanAdapter
from ..schema.common import stable_artifact_id, validate_dependency_dag
from ..schema.ir0 import FusionPattern
from ..schema.swizzle import SwizzleActionKind, SwizzlePhase
from ..schema.swizzle_ir2 import (
    SWIZZLE_IR2_VALUE_SCHEMA_VERSION,
    SwizzleIr2ArStage,
    SwizzleIr2Buffer,
    SwizzleIr2BufferAccess,
    SwizzleIr2BufferRole,
    SwizzleIr2ConsumerContract,
    SwizzleIr2DownstreamGate,
    SwizzleIr2Flow,
    SwizzleIr2OutputOwnership,
    SwizzleIr2Projection,
    SwizzleIr2RankDag,
    SwizzleIr2Task,
    SwizzleIr2TaskBufferUse,
    SwizzleIr2Value,
    SwizzleIr2ValueOriginKind,
)


def _value_id(rank: int, symbolic_ref: str) -> str:
    return stable_artifact_id(
        "swizzle_ir2_value",
        {"rank": rank, "symbolic_ref": symbolic_ref},
        schema_version=SWIZZLE_IR2_VALUE_SCHEMA_VERSION,
    )


def _rank_die_map(adapter: SwizzleFusionPlanAdapter) -> dict[int, int]:
    result: dict[int, int] = {}
    for route in adapter.decision.problem.group.routes:
        for rank, die in (
            (route.source_rank, route.die_path[0]),
            (route.destination_rank, route.die_path[-1]),
        ):
            previous = result.get(rank)
            if previous is not None and previous != die:
                raise SchemaError(
                    "routes disagree on rank-to-Die placement",
                    path="swizzle_adapter.decision.problem.group.routes",
                )
            result[rank] = die
    ranks = tuple(program.rank for program in adapter.rank_programs)
    if set(result) != set(ranks):
        raise SchemaError(
            "routes must identify the Die for every projected rank",
            path="swizzle_adapter.decision.problem.group.routes",
        )
    return result


def _ar_stage(adapter: SwizzleFusionPlanAdapter, action: object) -> SwizzleIr2ArStage:
    if adapter.pattern is not FusionPattern.GEMM_AR:
        return SwizzleIr2ArStage.NONE
    kind = getattr(action, "kind")
    phase = getattr(action, "phase")
    if phase is SwizzlePhase.EPILOGUE and kind in (
        SwizzleActionKind.SEND,
        SwizzleActionKind.RECV,
        SwizzleActionKind.WAIT,
        SwizzleActionKind.BARRIER,
        SwizzleActionKind.LOCAL_COPY,
    ):
        return SwizzleIr2ArStage.REPLICATION
    return SwizzleIr2ArStage.REDUCTION


def _buffer_uses(
    adapter: SwizzleFusionPlanAdapter,
    action: object,
) -> tuple[SwizzleIr2TaskBufferUse, ...]:
    result = []
    inputs = set(getattr(action, "input_refs"))
    outputs = set(getattr(action, "output_refs"))
    chunk = getattr(action, "chunk_index")
    for requirement in adapter.buffer_requirements:
        if requirement.rank != getattr(action, "rank") or getattr(action, "id") not in requirement.lifetime_action_refs:
            continue
        read = requirement.buffer_ref in inputs
        write = requirement.buffer_ref in outputs
        if read and write:
            access = SwizzleIr2BufferAccess.READ_WRITE
        elif read:
            access = SwizzleIr2BufferAccess.READ
        elif write:
            access = SwizzleIr2BufferAccess.WRITE
        else:
            access = SwizzleIr2BufferAccess.LIFETIME
        slot = 0
        if requirement.double_buffered and chunk is not None:
            slot = chunk % 2
        result.append(SwizzleIr2TaskBufferUse(requirement.buffer_ref, access, slot))
    return tuple(result)


def _topological_tasks(
    adapter: SwizzleFusionPlanAdapter,
    rank_dies: dict[int, int],
) -> tuple[dict[str, SwizzleIr2Task], dict[int, tuple[str, ...]]]:
    action_adapters = tuple(
        action for program in adapter.rank_programs for action in program.actions
    )
    source_actions = tuple(action.source_action for action in action_adapters)
    action_index = validate_dependency_dag(source_actions, "swizzle_adapter.rank_programs.actions")
    adapter_by_action = {item.source_action.id: item for item in action_adapters}
    canonical_order = {action.id: index for index, action in enumerate(source_actions)}
    remaining = {action.id: len(action.deps) for action in source_actions}
    dependents: dict[str, list[str]] = {action.id: [] for action in source_actions}
    for action in source_actions:
        for dependency in action.deps:
            dependents[dependency].append(action.id)
    ready = sorted(
        (ref for ref, count in remaining.items() if count == 0),
        key=canonical_order.__getitem__,
    )
    task_by_action: dict[str, SwizzleIr2Task] = {}
    while ready:
        action_ref = ready.pop(0)
        action = action_index[action_ref]
        materialized = adapter_by_action[action_ref]
        task = SwizzleIr2Task.create(
            source_action_ref=action.id,
            rank=action.rank,
            die_id=rank_dies[action.rank],
            kind=action.kind,
            phase=action.phase,
            ar_stage=_ar_stage(adapter, action),
            member_ref=materialized.member_ref,
            chunk_index=action.chunk_index,
            deps=tuple(task_by_action[ref].id for ref in action.deps),
            read_value_refs=tuple(_value_id(action.rank, ref) for ref in action.input_refs),
            write_value_refs=tuple(_value_id(action.rank, ref) for ref in action.output_refs),
            buffer_uses=_buffer_uses(adapter, action),
            peer_rank=action.peer_rank,
            route_ref=action.route_ref,
            expected_route=materialized.expected_route,
            logical_bytes=action.logical_bytes,
            flops=action.flops,
        )
        task_by_action[action_ref] = task
        for dependent in dependents[action_ref]:
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                ready.append(dependent)
        ready.sort(key=canonical_order.__getitem__)
    if len(task_by_action) != len(source_actions):
        raise SchemaError("source action DAG did not close", path="swizzle_adapter.rank_programs.actions")
    per_rank_source_order = {
        program.rank: tuple(action.source_action.id for action in program.actions)
        for program in adapter.rank_programs
    }
    return task_by_action, per_rank_source_order


def _origin_base(symbolic_ref: str, candidates: tuple[str, ...]) -> str | None:
    exact = next((ref for ref in candidates if symbolic_ref == ref), None)
    if exact is not None:
        return exact
    return next(
        (
            ref
            for ref in candidates
            if symbolic_ref.startswith(f"{ref}::")
            or symbolic_ref.startswith(f"{ref}.")
        ),
        None,
    )


def _values_for_rank(
    adapter: SwizzleFusionPlanAdapter,
    rank: int,
    tasks: tuple[SwizzleIr2Task, ...],
    action_by_ref: dict[str, object],
) -> tuple[SwizzleIr2Value, ...]:
    symbols: set[str] = set()
    producers: dict[str, list[str]] = defaultdict(list)
    consumers: dict[str, list[str]] = defaultdict(list)
    task_by_source = {task.source_action_ref: task for task in tasks}
    for task in tasks:
        action = action_by_ref[task.source_action_ref]
        for ref in action.input_refs:
            symbols.add(ref)
            consumers[ref].append(task.id)
        for ref in action.output_refs:
            symbols.add(ref)
            producers[ref].append(task.id)
    requirements = {
        item.buffer_ref: item
        for item in adapter.buffer_requirements
        if item.rank == rank
    }
    symbols.update(requirements)
    problem = adapter.decision.problem
    boundary_refs = tuple(
        dict.fromkeys(
            problem.gemm.boundary_input_refs
            + adapter.candidate.semantic_witness.boundary_input_refs
            + (problem.collective.input.value_ref,)
        )
    )
    local_refs = problem.gemm.local_operand_refs
    values = []
    for symbolic_ref in sorted(symbols):
        producer_refs = tuple(dict.fromkeys(producers[symbolic_ref]))
        consumer_refs = tuple(dict.fromkeys(consumers[symbolic_ref]))
        buffer_ref = symbolic_ref if symbolic_ref in requirements else None
        producer_tasks = tuple(
            task for task in tasks if task.id in set(producer_refs)
        )
        read_modify_write = any(
            _value_id(rank, symbolic_ref) in task.read_value_refs
            and _value_id(rank, symbolic_ref) in task.write_value_refs
            for task in producer_tasks
        )
        loop_carried = len(producer_refs) > 1 or read_modify_write
        boundary = _origin_base(symbolic_ref, boundary_refs)
        local = _origin_base(symbolic_ref, local_refs)
        if loop_carried:
            if buffer_ref is None:
                matching = tuple(
                    requirement.buffer_ref
                    for requirement in requirements.values()
                    if set(action_by_ref[task.source_action_ref].id for task in producer_tasks).issubset(requirement.lifetime_action_refs)
                )
                buffer_ref = matching[0] if len(matching) == 1 else None
            if buffer_ref is None:
                raise SchemaError(
                    "loop-carried value requires an explicit buffer witness",
                    path=f"swizzle_projection.rank.{rank}.value.{symbolic_ref}",
                )
            origin_kind = SwizzleIr2ValueOriginKind.LOOP_ACCUMULATOR
            origin_ref = buffer_ref
        elif boundary is not None and not producer_refs:
            origin_kind = SwizzleIr2ValueOriginKind.BOUNDARY_INPUT
            origin_ref = boundary
        elif local is not None and not producer_refs:
            origin_kind = SwizzleIr2ValueOriginKind.LOCAL_OPERAND
            origin_ref = local
        elif buffer_ref is not None:
            origin_kind = SwizzleIr2ValueOriginKind.BUFFER
            origin_ref = buffer_ref
        elif producer_refs:
            if all(task.kind is SwizzleActionKind.RECV for task in producer_tasks):
                origin_kind = SwizzleIr2ValueOriginKind.RECEIVED_PAYLOAD
            else:
                origin_kind = SwizzleIr2ValueOriginKind.ACTION_OUTPUT
            origin_ref = action_by_ref[producer_tasks[0].source_action_ref].id
        else:
            raise SchemaError(
                "temporary input has no explicit boundary/local/buffer origin",
                path=f"swizzle_projection.rank.{rank}.value.{symbolic_ref}",
            )
        values.append(
            SwizzleIr2Value.create(
                rank=rank,
                symbolic_ref=symbolic_ref,
                origin_kind=origin_kind,
                origin_ref=origin_ref,
                producer_task_refs=producer_refs,
                consumer_task_refs=consumer_refs,
                buffer_ref=buffer_ref,
                loop_carried=loop_carried,
            )
        )
    return tuple(values)


def _buffer_role(
    adapter: SwizzleFusionPlanAdapter,
    requirement: object,
    tasks: tuple[SwizzleIr2Task, ...],
) -> SwizzleIr2BufferRole:
    if getattr(requirement, "double_buffered"):
        return SwizzleIr2BufferRole.DOUBLE_BUFFER
    relevant = tuple(
        task
        for task in tasks
        if task.source_action_ref in getattr(requirement, "lifetime_action_refs")
    )
    if any(
        use.buffer_ref == getattr(requirement, "buffer_ref")
        and use.access is SwizzleIr2BufferAccess.READ_WRITE
        for task in relevant
        for use in task.buffer_uses
    ):
        return SwizzleIr2BufferRole.LOOP_ACCUMULATOR
    has_reduction = any(task.kind is SwizzleActionKind.REDUCE for task in relevant)
    has_replication = any(task.ar_stage is SwizzleIr2ArStage.REPLICATION for task in relevant)
    if has_reduction and has_replication:
        return SwizzleIr2BufferRole.LOOP_ACCUMULATOR
    if has_reduction:
        return SwizzleIr2BufferRole.REDUCTION
    if adapter.pattern is FusionPattern.GEMM_AR and has_replication:
        return SwizzleIr2BufferRole.REPLICATION
    return SwizzleIr2BufferRole.TEMPORARY


def _buffers_for_rank(
    adapter: SwizzleFusionPlanAdapter,
    rank: int,
    tasks: tuple[SwizzleIr2Task, ...],
    values: tuple[SwizzleIr2Value, ...],
    task_by_action: dict[str, SwizzleIr2Task],
) -> tuple[SwizzleIr2Buffer, ...]:
    result = []
    for requirement in adapter.buffer_requirements:
        if requirement.rank != rank:
            continue
        linked = tuple(
            value.id for value in values if value.buffer_ref == requirement.buffer_ref
        )
        if not linked:
            raise SchemaError(
                "buffer requirement has no projected value",
                path=f"swizzle_projection.rank.{rank}.buffer.{requirement.buffer_ref}",
            )
        result.append(
            SwizzleIr2Buffer(
                rank=rank,
                buffer_ref=requirement.buffer_ref,
                role=_buffer_role(adapter, requirement, tasks),
                size_bytes=requirement.size_bytes,
                slot_count=2 if requirement.double_buffered else 1,
                lifetime_task_refs=tuple(
                    task_by_action[ref].id for ref in requirement.lifetime_action_refs
                ),
                value_refs=linked,
            )
        )
    return tuple(result)


def _flows(
    adapter: SwizzleFusionPlanAdapter,
    task_by_action: dict[str, SwizzleIr2Task],
    rank_dies: dict[int, int],
) -> tuple[SwizzleIr2Flow, ...]:
    action_by_id = {
        action.source_action.id: action.source_action
        for program in adapter.rank_programs
        for action in program.actions
    }
    result = []
    for action in action_by_id.values():
        if action.kind is not SwizzleActionKind.RECV:
            continue
        matching = tuple(
            action_by_id[dependency]
            for dependency in action.deps
            if action_by_id[dependency].kind is SwizzleActionKind.SEND
            and action_by_id[dependency].route_ref == action.route_ref
            and action_by_id[dependency].logical_bytes == action.logical_bytes
        )
        if len(matching) != 1:
            raise SchemaError(
                "RECV must depend on exactly one matching SEND",
                path=f"swizzle_adapter.action.{action.id}.deps",
            )
        send = matching[0]
        route = next(
            route
            for route in adapter.decision.problem.group.routes
            if route.id == action.route_ref
        )
        result.append(
            SwizzleIr2Flow.create(
                route_ref=route.id,
                source_rank=send.rank,
                destination_rank=action.rank,
                source_die=rank_dies[send.rank],
                destination_die=rank_dies[action.rank],
                die_path=route.die_path,
                send_task_ref=task_by_action[send.id].id,
                recv_task_ref=task_by_action[action.id].id,
                chunk_index=action.chunk_index,
                logical_bytes=action.logical_bytes,
            )
        )
    return tuple(sorted(result, key=lambda item: item.id))


def project_swizzle_adapter(
    adapter: SwizzleFusionPlanAdapter,
) -> SwizzleIr2Projection:
    """Produce an executable, isolated timing DAG without semantic inference."""

    if type(adapter) is not SwizzleFusionPlanAdapter:
        raise SchemaError("must be a SwizzleFusionPlanAdapter", path="adapter")
    adapter.validate("adapter")
    rank_dies = _rank_die_map(adapter)
    task_by_action, per_rank_order = _topological_tasks(adapter, rank_dies)
    action_by_ref = {
        action.source_action.id: action.source_action
        for program in adapter.rank_programs
        for action in program.actions
    }
    rank_dags = []
    for rank in sorted(per_rank_order):
        tasks = tuple(task_by_action[ref] for ref in per_rank_order[rank])
        values = _values_for_rank(adapter, rank, tasks, action_by_ref)
        buffers = _buffers_for_rank(
            adapter, rank, tasks, values, task_by_action
        )
        rank_dags.append(
            SwizzleIr2RankDag(rank, rank_dies[rank], tasks, values, buffers)
        )
    all_tasks = tuple(task for dag in rank_dags for task in dag.tasks)
    depended = {
        dependency
        for task in all_tasks
        for dependency in task.deps
    }
    ownership = tuple(
        SwizzleIr2OutputOwnership(
            rank=dag.rank,
            logical_owner_rank=dag.rank,
            physical_owner_rank=dag.rank,
            boundary_output_refs=adapter.candidate.semantic_witness.boundary_output_refs,
            terminal_task_refs=tuple(task.id for task in dag.tasks if task.id not in depended),
            replicated=adapter.pattern is FusionPattern.GEMM_AR,
        )
        for dag in rank_dags
    )
    projection = SwizzleIr2Projection.create(
        source_adapter_ref=adapter.id,
        source_decision_ref=adapter.decision.id,
        source_candidate_ref=adapter.candidate.id,
        source_ir1_id=adapter.source_ir1_id,
        fused_op_id=adapter.fused_op_id,
        group_ref=adapter.group_ref,
        pattern=adapter.pattern,
        algorithm=adapter.algorithm,
        split_axis=adapter.candidate.split_axis,
        chunk_count=adapter.candidate.chunk_count,
        unroll_degree=adapter.candidate.unroll_degree,
        rank_dags=tuple(rank_dags),
        flows=_flows(adapter, task_by_action, rank_dies),
        output_ownership=ownership,
        downstream_gate=SwizzleIr2DownstreamGate(
            current_ir2_compatible=False,
            required_consumer=SwizzleIr2ConsumerContract.STRICT_SWIZZLE_TIMING_V1,
            timing_execution=True,
            functional_execution=False,
            reason=(
                "current IR2 requires ComputeContract/ReductionContract; "
                "this carrier preserves W7 timing witnesses without inventing them"
            ),
        ),
    )
    validate_swizzle_projection_against_adapter(projection, adapter)
    return projection


def validate_swizzle_projection_against_adapter(
    projection: SwizzleIr2Projection,
    adapter: SwizzleFusionPlanAdapter,
) -> None:
    """Exact cross-carrier gate used before every downstream consumer."""

    projection.validate("projection")
    adapter.validate("adapter")
    if (
        projection.source_adapter_ref,
        projection.source_decision_ref,
        projection.source_candidate_ref,
        projection.source_ir1_id,
        projection.fused_op_id,
        projection.group_ref,
        projection.pattern,
        projection.algorithm,
        projection.split_axis,
        projection.chunk_count,
        projection.unroll_degree,
    ) != (
        adapter.id,
        adapter.decision.id,
        adapter.candidate.id,
        adapter.source_ir1_id,
        adapter.fused_op_id,
        adapter.group_ref,
        adapter.pattern,
        adapter.algorithm,
        adapter.candidate.split_axis,
        adapter.candidate.chunk_count,
        adapter.candidate.unroll_degree,
    ):
        raise SchemaError("projection provenance disagrees with adapter", path="projection")
    source_actions = {
        action.source_action.id: action.source_action
        for program in adapter.rank_programs
        for action in program.actions
    }
    projected = {
        task.source_action_ref: task
        for dag in projection.rank_dags
        for task in dag.tasks
    }
    if set(source_actions) != set(projected):
        raise SchemaError("projection must map every source action exactly once", path="projection.rank_dags.tasks")
    rank_dies = _rank_die_map(adapter)
    expected_tasks, _ = _topological_tasks(adapter, rank_dies)
    if projected != expected_tasks:
        raise SchemaError(
            "projected task objects drift from source adapter",
            path="projection.rank_dags.tasks",
        )
    expected_flows = _flows(adapter, expected_tasks, rank_dies)
    if projection.flows != expected_flows:
        raise SchemaError("projection flows drift from source adapter routes", path="projection.flows")
    for action_ref, action in source_actions.items():
        task = projected[action_ref]
        expected_deps = tuple(projected[ref].id for ref in action.deps)
        if (
            task.rank,
            task.kind,
            task.phase,
            task.chunk_index,
            task.deps,
            task.peer_rank,
            task.route_ref,
            task.logical_bytes,
            task.flops,
        ) != (
            action.rank,
            action.kind,
            action.phase,
            action.chunk_index,
            expected_deps,
            action.peer_rank,
            action.route_ref,
            action.logical_bytes,
            action.flops,
        ):
            raise SchemaError("projected task drifts from source action", path=f"projection.task.{task.id}")
    projection.require_consumer(SwizzleIr2ConsumerContract.STRICT_SWIZZLE_TIMING_V1)


__all__ = ["project_swizzle_adapter", "validate_swizzle_projection_against_adapter"]
