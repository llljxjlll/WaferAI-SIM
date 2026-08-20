"""Losslessly quotient a physical IR-2 schedule into global actions."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.global_action import (
    ActionBufferUse,
    ActionStateUse,
    GlobalAction,
    GlobalActionDAG,
    LogicalCoreRef,
    ScheduledDagRef,
    ScheduledSourceRef,
    state_transfer_wave_task_dependencies,
)
from ..schema.ir1 import IR1
from ..schema.ir2 import (
    IR2ProjectionResult,
    IntraDieScheduleSet,
    SemanticTaskKind,
)


def _ordered_unique_without_self(values: tuple[str, ...], self_id: str) -> tuple[str, ...]:
    result: list[str] = []
    seen = {self_id}
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)


def build_global_action_dag(
    ir1: IR1,
    projection: IR2ProjectionResult,
    schedule_set: IntraDieScheduleSet,
) -> GlobalActionDAG:
    """Build the canonical one-action-per-task quotient without policy choices."""

    if not isinstance(ir1, IR1):
        raise SchemaError("ir1 must be an IR1 artifact", path="ir1")
    if not isinstance(projection, IR2ProjectionResult):
        raise SchemaError(
            "projection must be an IR2ProjectionResult artifact",
            path="projection",
        )
    if not isinstance(schedule_set, IntraDieScheduleSet):
        raise SchemaError(
            "schedule_set must be an IntraDieScheduleSet artifact",
            path="schedule_set",
        )
    schedule_set.validate_against(projection, ir1)

    schedule_by_dag = {schedule.dag_id: schedule for schedule in schedule_set.schedules}
    action_ids = {
        (dag.id, schedule_by_dag[dag.id].id, task.id): GlobalAction.stable_id_for_source(
            ScheduledSourceRef(dag.id, schedule_by_dag[dag.id].id, task.id)
        )
        for dag in projection.dags
        for task in dag.tasks
    }
    wave_task_deps = state_transfer_wave_task_dependencies(
        projection, schedule_set
    )
    actions: list[GlobalAction] = []
    scheduled_dags: list[ScheduledDagRef] = []
    for dag in projection.dags:
        schedule = schedule_by_dag[dag.id]
        scheduled_dags.append(ScheduledDagRef(dag.id, schedule.id, dag.die_id))
        die = next(item for item in ir1.fabric.dies if item.id == dag.die_id)
        cores = {core.runtime_core_id: core for core in die.cores}
        regions = {region.id: region for region in dag.regions}
        flows = {flow.id: flow for flow in dag.flows}
        placements = {placement.task_id: placement.core_id for placement in schedule.placements}
        core_orders = {order.core_id: order.task_ids for order in schedule.core_orders}
        routes = {route.flow_id: route for route in schedule.flow_routes}
        runtimes = {binding.task_id: binding for binding in schedule.runtime_bindings}

        for task in dag.tasks:
            source = ScheduledSourceRef(dag.id, schedule.id, task.id)
            action_id = action_ids[(dag.id, schedule.id, task.id)]
            predecessor_ids: tuple[str, ...] = ()
            if task.kind is SemanticTaskKind.TRANSIT:
                logical_core = None
                core_order_index = None
            else:
                runtime_core_id = placements[task.id]
                core = cores[runtime_core_id]
                logical_core = LogicalCoreRef(dag.die_id, core.local_core_id)
                core_order = core_orders[runtime_core_id]
                core_order_index = core_order.index(task.id)
                if core_order_index:
                    predecessor_task_id = core_order[core_order_index - 1]
                    predecessor_ids = (
                        action_ids[(dag.id, schedule.id, predecessor_task_id)],
                    )
            semantic_deps = tuple(
                action_ids[(dag.id, schedule.id, dependency)]
                for dependency in task.deps
            )
            wave_deps = tuple(
                action_ids[
                    (source_dag, schedule_by_dag[source_dag].id, source_task)
                ]
                for source_dag, source_task in wave_task_deps.get(
                    (dag.id, task.id), ()
                )
            )
            actions.append(
                GlobalAction.create(
                    source=source,
                    task_kind=task.kind,
                    origin_ref=task.origin_ref,
                    lowering=regions[task.region_id].lowering,
                    region_id=task.region_id,
                    op_kind=task.op_kind,
                    member_id=task.member_id,
                    flow_id=task.flow_id,
                    chunk_id=task.chunk_id,
                    collective_step=task.collective_step,
                    source_rank=task.source_rank,
                    destination_rank=task.destination_rank,
                    tensor_slice=task.tensor_slice,
                    bytes=task.bytes,
                    dtype=task.dtype,
                    shape=task.shape,
                    read_values=task.read_values,
                    write_values=task.write_values,
                    compute=task.compute,
                    reduction=task.reduction,
                    sync=task.sync,
                    dma=task.dma,
                    logical_core=logical_core,
                    core_order_index=core_order_index,
                    flow=flows.get(task.flow_id),
                    flow_route=routes.get(task.flow_id),
                    runtime_binding=runtimes.get(task.id),
                    buffer_uses=tuple(
                        ActionBufferUse(
                            use.binding_id,
                            use.access,
                            use.role,
                            use.operand_index,
                            use.contribution_rank,
                            use.tensor_slice,
                        )
                        for use in schedule.task_buffer_uses
                        if use.task_id == task.id
                    ),
                    state_uses=tuple(
                        ActionStateUse(
                            use.hbm_binding_ref,
                            use.access,
                        )
                        for use in schedule.task_state_uses
                        if use.task_id == task.id
                    ),
                    deps=_ordered_unique_without_self(
                        semantic_deps + predecessor_ids + wave_deps,
                        action_id,
                    ),
                )
            )

    result = GlobalActionDAG.create(
        producer_pass="global_action_dag",
        source_ir1_id=ir1.id,
        source_state_manifest_id=(
            ir1.persistent_state_manifest.id
            if ir1.persistent_state_manifest is not None else None
        ),
        source_projection_id=projection.id,
        source_schedule_set_id=schedule_set.id,
        scheduled_dags=tuple(scheduled_dags),
        actions=tuple(actions),
    )
    result.validate_against(ir1, projection, schedule_set)
    return result


__all__ = ["build_global_action_dag"]
