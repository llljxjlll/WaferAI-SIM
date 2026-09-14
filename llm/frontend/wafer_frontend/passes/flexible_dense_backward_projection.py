"""Build exact Dense backward IR, projection, schedule and global DAG."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.flexible_dense_backward_ir import (
    FlexibleDenseBackwardActionLineage,
    FlexibleDenseBackwardGlobalDAG,
    FlexibleDenseBackwardIR,
    FlexibleDenseBackwardProjection,
    FlexibleDenseBackwardSchedule,
)
from ..schema.flexible_dense_train import (
    FlexibleDenseTrainActionKind,
    FlexibleDenseTrainPlan,
)
from ..schema.serde import canonical_digest


def build_flexible_dense_backward_ir(
    plan: FlexibleDenseTrainPlan,
) -> FlexibleDenseBackwardIR:
    """Project every rank action with exact tape and parameter provenance."""

    plan.validate("plan")
    template_by_state = {
        template.state_ref: template for template in plan.parameter_templates
    }
    tape_origin = {
        binding.backward_node_ref: binding.forward_node_ref
        for binding in plan.tape_bindings
    }
    actions = []
    for action in plan.rank_actions:
        template = (
            None
            if action.state_ref is None
            else template_by_state.get(action.state_ref)
        )
        if action.state_ref is not None and template is None:
            raise SchemaError(
                "action references unknown parameter template",
                path="plan.rank_actions",
            )
        actions.append(
            FlexibleDenseBackwardActionLineage(
                action_id=action.id,
                rank=action.rank,
                index=action.index,
                kind=action.kind,
                state_ref=action.state_ref,
                op_ref=action.op_ref,
                depends_on=action.depends_on,
                forward_consumer_refs=(
                    () if template is None else template.forward_consumer_refs
                ),
                tape_origin_ref=(
                    tape_origin.get(action.op_ref)
                    if action.kind is FlexibleDenseTrainActionKind.BACKWARD
                    else None
                ),
            )
        )
    result = FlexibleDenseBackwardIR.create(
        source_plan_id=plan.id,
        source_plan_digest=canonical_digest(plan),
        mesh_digest=plan.spec.mesh.digest,
        rank_count=plan.spec.mesh.rank_count,
        actions=tuple(actions),
    )
    expected_tape = {
        action.op_ref
        for action in plan.rank_actions
        if action.kind is FlexibleDenseTrainActionKind.BACKWARD
    }
    if set(tape_origin) != expected_tape:
        raise SchemaError(
            "backward tape does not exactly cover rank actions",
            path="plan.tape_bindings",
        )
    return result


def project_flexible_dense_backward(
    source: FlexibleDenseBackwardIR,
    plan: FlexibleDenseTrainPlan,
) -> FlexibleDenseBackwardProjection:
    source.validate("source")
    plan.validate("plan")
    if (
        source.source_plan_id != plan.id
        or source.source_plan_digest != canonical_digest(plan)
    ):
        raise SchemaError("IR does not derive from exact plan", path="source")
    return FlexibleDenseBackwardProjection.create(
        source_ir_id=source.id,
        action_ids=tuple(action.action_id for action in source.actions),
        state_owner_pairs=tuple(
            sorted(
                (template.state_ref, rank)
                for template in plan.parameter_templates
                for rank in template.owner_ranks
            )
        ),
    )


def schedule_flexible_dense_backward(
    source: FlexibleDenseBackwardProjection,
    ir: FlexibleDenseBackwardIR,
) -> FlexibleDenseBackwardSchedule:
    source.validate("source")
    ir.validate("ir")
    if source.source_ir_id != ir.id or source.action_ids != tuple(
        action.action_id for action in ir.actions
    ):
        raise SchemaError("projection action lineage drifted", path="source")
    return FlexibleDenseBackwardSchedule.create(
        source_projection_id=source.id,
        rank_action_ids=tuple(
            tuple(
                action.action_id for action in ir.actions if action.rank == rank
            )
            for rank in range(ir.rank_count)
        ),
    )


def build_flexible_dense_backward_global_dag(
    source: FlexibleDenseBackwardSchedule,
    ir: FlexibleDenseBackwardIR,
) -> FlexibleDenseBackwardGlobalDAG:
    source.validate("source")
    ir.validate("ir")
    action_ids = tuple(action.action_id for action in ir.actions)
    if tuple(item for row in source.rank_action_ids for item in row) != action_ids:
        # IR is rank-major by construction. Refuse silent schedule reordering.
        raise SchemaError("schedule does not exactly flatten to IR", path="source")
    return FlexibleDenseBackwardGlobalDAG.create(
        source_schedule_id=source.id,
        action_ids=action_ids,
        dependency_edges=tuple(
            sorted(
                (dependency, action.action_id)
                for action in ir.actions
                for dependency in action.depends_on
            )
        ),
    )


def build_flexible_dense_backward_lineage(
    plan: FlexibleDenseTrainPlan,
) -> tuple[
    FlexibleDenseBackwardIR,
    FlexibleDenseBackwardProjection,
    FlexibleDenseBackwardSchedule,
    FlexibleDenseBackwardGlobalDAG,
]:
    ir = build_flexible_dense_backward_ir(plan)
    projection = project_flexible_dense_backward(ir, plan)
    schedule = schedule_flexible_dense_backward(projection, ir)
    global_dag = build_flexible_dense_backward_global_dag(schedule, ir)
    return ir, projection, schedule, global_dag


__all__ = [
    "build_flexible_dense_backward_global_dag",
    "build_flexible_dense_backward_ir",
    "build_flexible_dense_backward_lineage",
    "project_flexible_dense_backward",
    "schedule_flexible_dense_backward",
]
