"""Build the one-DAG whole-workload MoE projection from an exact overlay."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.swizzle_moe_execution import MoeScaleExecution
from ..schema.swizzle_moe_ir2 import MoeSwizzleIr2Projection
from ..schema.swizzle_moe_plan import MoeSwizzleOverlay
from ..schema.swizzle_moe_state import MoeSwizzleWorkloadStateABI
from ..schema.swizzle_moe_workload import (
    MoeSwizzleWorkloadAction,
    MoeSwizzleWorkloadProjection,
    MoeSwizzleWorkloadTerminal,
)


def project_moe_swizzle_whole_workload(
    overlay: MoeSwizzleOverlay,
    execution: MoeScaleExecution,
    replacement_projection: MoeSwizzleIr2Projection,
    state_abi: MoeSwizzleWorkloadStateABI,
) -> MoeSwizzleWorkloadProjection:
    """Preserve the overlay DAG while attaching typed replacement/state carriers."""

    overlay.validate("project_moe_swizzle_whole_workload.overlay")
    execution.validate("project_moe_swizzle_whole_workload.execution")
    replacement_projection.validate("project_moe_swizzle_whole_workload.replacement_projection")
    state_abi.validate("project_moe_swizzle_whole_workload.state_abi")
    if (
        overlay.source_execution_id != execution.id
        or replacement_projection.source_execution_id != execution.id
        or replacement_projection.source_overlay_id != overlay.id
        or state_abi.source_execution_id != execution.id
    ):
        raise SchemaError("whole-workload source lineage is not exact", path="project_moe_swizzle_whole_workload")

    execution_actions = {item.id: item for item in execution.actions}
    replacement_tasks = {item.id: item for item in replacement_projection.tasks}
    linked = {item.id: item for item in overlay.linked_actions}
    expected_replacement = {item.id for item in overlay.linked_actions if not item.preserved}
    if set(replacement_tasks) != expected_replacement:
        raise SchemaError("replacement projection does not exactly cover fused overlay actions", path="project_moe_swizzle_whole_workload.replacement_projection")
    state_by_action = {
        item.execution_action_ref: item for item in state_abi.action_bindings
    }

    actions = []
    original_to_linked = {}
    for linked_action in overlay.linked_actions:
        for ref in linked_action.source_action_refs:
            if ref in original_to_linked:
                raise SchemaError("source action is materialized more than once", path="project_moe_swizzle_whole_workload.overlay")
            original_to_linked[ref] = linked_action.id
    if set(original_to_linked) != {item.id for item in execution.actions}:
        raise SchemaError("whole-workload source actions are not covered exactly once", path="project_moe_swizzle_whole_workload.overlay")
    for linked_action in overlay.linked_actions:
        if linked_action.preserved:
            source = execution_actions.get(linked_action.source_action_refs[0])
            if source is None:
                raise SchemaError("preserved action lacks execution source", path="project_moe_swizzle_whole_workload.overlay")
            expected_kind = f"preserved.{source.kind.value}"
            state_ref = (
                source.id
                if source.kind.value == "dma_in" and source.id in state_by_action
                else None
            )
            if (
                linked_action.kind != expected_kind
                or (linked_action.rank, linked_action.die_id) != (source.die_id, source.die_id)
                or linked_action.read_value_refs != source.read_values
                or linked_action.write_value_refs != source.write_values
            ):
                raise SchemaError("preserved linked action drifts from execution truth", path="project_moe_swizzle_whole_workload.overlay")
            expected_deps = tuple(
                dict.fromkeys(original_to_linked[ref] for ref in source.deps)
            )
            if linked_action.deps != expected_deps:
                raise SchemaError(
                    "preserved action dependency rewiring is not exact",
                    path="project_moe_swizzle_whole_workload.overlay",
                )
            if (source.kind.value == "dma_in") != (state_ref is not None):
                raise SchemaError("preserved DMA lacks exact typed state binding", path="project_moe_swizzle_whole_workload.state_abi")
            actions.append(MoeSwizzleWorkloadAction(
                linked_action.id, linked_action.rank, linked_action.die_id,
                linked_action.kind, True, linked_action.source_action_refs, None,
                linked_action.deps, source.read_values, source.write_values,
                source.token_index, source.expert_index, source.role,
                source.bytes, source.flops, source.dtype, state_ref,
            ))
        else:
            task = replacement_tasks[linked_action.id]
            if (
                linked_action.source_action_refs != task.original_action_refs
                or (linked_action.rank, linked_action.die_id) != (task.rank, task.die_id)
            ):
                raise SchemaError("replacement linked action drifts from IR2 task", path="project_moe_swizzle_whole_workload.replacement_projection")
            actions.append(MoeSwizzleWorkloadAction(
                linked_action.id, linked_action.rank, linked_action.die_id,
                linked_action.kind, False, linked_action.source_action_refs, task.id,
                linked_action.deps, task.read_value_refs, task.write_value_refs,
                None, task.expert_index, task.work_role,
                task.logical_bytes, task.flops,
                DType.FP16 if task.dtype is None else task.dtype, None,
            ))
    if set(overlay.terminal_value_refs) != {item.value_ref for item in execution.terminals}:
        raise SchemaError("overlay terminal set drifts from execution", path="project_moe_swizzle_whole_workload.overlay")

    terminals = []
    for terminal in execution.terminals:
        producer = original_to_linked.get(terminal.producer_action_ref)
        if producer is None:
            raise SchemaError("terminal source producer has no linked action", path="project_moe_swizzle_whole_workload.terminals")
        terminals.append(MoeSwizzleWorkloadTerminal(
            terminal.kind, terminal.value_ref, producer, terminal.token_index,
            terminal.expert_index, terminal.die_id, terminal.shape,
            terminal.bytes, terminal.dtype,
        ))
    result = MoeSwizzleWorkloadProjection.create(
        source_execution_id=execution.id,
        source_overlay_id=overlay.id,
        replacement_projection_id=replacement_projection.id,
        state_abi_id=state_abi.id,
        actions=tuple(actions),
        terminals=tuple(terminals),
    )
    if tuple(item.id for item in result.actions) != tuple(item.id for item in overlay.linked_actions):
        raise SchemaError("whole-workload action order drifts from overlay", path="project_moe_swizzle_whole_workload.actions")
    return result


__all__ = ["project_moe_swizzle_whole_workload"]
