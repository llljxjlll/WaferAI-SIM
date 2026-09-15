"""Build an opt-in versioned MoE P2 action source with pre-dispatch dExpert.

The bounded source has no executable backend: old default P2, shared Dense
model, source identities, weight shapes and physical gate remain unchanged.
Only the official signed schema/validator extension may accept this carrier.
"""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.common import stable_artifact_id
from ..schema.flexible_moe import (
    FLEXIBLE_MOE_PLAN_SCHEMA_VERSION, FlexibleMoeExecutablePlan,
    FlexibleMoeMode, FlexibleMoeSpec, MoeRectAction, MoeRectActionKind,
    MoeRectFlowStage,
)
from .flexible_moe import compile_flexible_moe_baseline

_PRODUCER = "compile_flexible_moe_signed_top1_train_source"


def _resign(action: MoeRectAction, deps: tuple[str, ...]) -> MoeRectAction:
    """Never retain an old action ID if one source predecessor changes."""
    return MoeRectAction.create(
        rank=action.rank, kind=action.kind,
        deps=tuple(dict.fromkeys(deps)),
        assignment_refs=action.assignment_refs,
        flow_ref=action.flow_ref,
        logical_bytes=action.logical_bytes,
        flops=action.flops,
        state_refs=action.state_refs,
    )


def _append_action(rank, kind, deps, *, assignments, bytes_written, flops):
    return MoeRectAction.create(
        rank=rank, kind=kind,
        deps=tuple(dict.fromkeys(deps)),
        assignment_refs=assignments,
        flow_ref=None, logical_bytes=bytes_written,
        flops=flops, state_refs=(),
    )


def compile_moe_signed_top1_train_source_plan(
    spec: FlexibleMoeSpec, *, signed_source,
    _validate_plan: bool = True,
) -> FlexibleMoeExecutablePlan:
    """Re-sign the original typed P2 DAG, preserving its late dX combine.

    The extra SHARED_DCOMBINED_IMPORT source action is a required *named
    external* shared-backbone producer binding. This alone does not prove a
    real LinkedProgramManifest BufferABI consumer or schedule.
    """
    if spec.mode is not FlexibleMoeMode.TRAIN:
        raise SchemaError("signed score backward requires original TRAIN model",
                          path="moe_signed_source.spec")
    if (not hasattr(MoeRectActionKind, "SHARED_DCOMBINED_IMPORT")
            or not hasattr(MoeRectActionKind,
                           "SCORE_WEIGHT_BACKWARD_PRE_DISPATCH")
            or not hasattr(signed_source, "validate_against")):
        raise SchemaError("official opt-in signed-router source schema is absent; old P2 must remain fail closed",
                          path="moe_signed_source.schema")
    signed_source.validate_against(spec)
    original = compile_flexible_moe_baseline(spec)
    original.validate_against(spec)
    old_by_id = {item.id: item for item in original.actions}
    flow_by_id = {item.id: item for item in original.flows}
    rewritten: dict[str, str] = {}
    actions: list[MoeRectAction] = []
    imports: list[MoeRectAction] = []
    early: list[MoeRectAction] = []
    rank0_assignments = tuple(f"assignment.{assignment.token_index}"
                              for assignment in spec.trace.assignments
                              if assignment.source_rank == 0)
    rows = len(rank0_assignments)
    if (rows == 0 or rows != signed_source.route_rows
            or signed_source.route_bytes != 20 * rows):
        raise SchemaError("signed top1 rank0 must preserve all nonzero token routes",
                          path="moe_signed_source.routes")
    for old in original.actions:
        if any(dep not in rewritten for dep in old.deps):
            raise SchemaError("old P2 source actions are not listed topologically",
                              path=f"moe_signed_source.action[{old.id}]")
        deps = tuple(rewritten[item] for item in old.deps)
        if (old.kind is MoeRectActionKind.SEND and old.flow_ref is not None
                and flow_by_id[old.flow_ref].stage is
                    MoeRectFlowStage.BACKWARD_GRADIENT
                and old.rank == 0):
            if len(early) != 1:
                raise SchemaError("remote expert gradient SEND lacks pre-dispatch native dExpert producer",
                                  path="moe_signed_source.BACKWARD_GRADIENT")
            old_combine = next((item for item in old.deps
                               if old_by_id[item].kind is
                                  MoeRectActionKind.WEIGHTED_COMBINE
                               and old_by_id[item].rank == 0), None)
            if old_combine is None:
                raise SchemaError("old P2 BACKWARD_GRADIENT SEND has no real forward combine predecessor",
                                  path="moe_signed_source.BACKWARD_GRADIENT")
            deps = tuple(early[0].id if item == rewritten[old_combine]
                         else item for item in deps)
        if (old.kind in (MoeRectActionKind.EXPERT_DGRAD,
                         MoeRectActionKind.EXPERT_WGRAD)
                and old.rank == 0
                and old.assignment_refs):
            if len(early) != 1:
                raise SchemaError("local expert reverse lacks score-weighted dExpert",
                                  path="moe_signed_source.EXPERT_DGRAD")
            deps = (*deps, early[0].id)
        current = _resign(old, deps)
        rewritten[old.id] = current.id
        actions.append(current)
        if (old.kind is MoeRectActionKind.WEIGHTED_COMBINE
                and old.rank == 0):
            shared_import = _append_action(
                0, MoeRectActionKind.SHARED_DCOMBINED_IMPORT,
                (current.id,), assignments=rank0_assignments,
                bytes_written=2 * rows * spec.hidden_size, flops=0)
            early_action = _append_action(
                0, MoeRectActionKind.SCORE_WEIGHT_BACKWARD_PRE_DISPATCH,
                (current.id, shared_import.id),
                assignments=rank0_assignments,
                bytes_written=2 * rows * (spec.hidden_size +
                                          spec.expert_count),
                flops=3 * rows * spec.hidden_size)
            imports.append(shared_import)
            early.append(early_action)
            actions.extend((shared_import, early_action))
    if len(imports) != 1 or len(early) != 1:
        raise SchemaError("exactly one rank0 shared dCombined import and pre-dispatch score backward required",
                          path="moe_signed_source.actions")
    semantic = dict(
        source_spec_id=spec.id, source_spec_digest=spec.digest,
        mesh_digest=spec.mesh.digest,
        actions=tuple(actions), flows=original.flows,
        state_bindings=original.state_bindings,
        gate_all_reduce=original.gate_all_reduce,
        terminal_action_refs=tuple(rewritten[item]
                                   for item in original.terminal_action_refs),
        symbolic_record_count=original.symbolic_record_count + 8,
        symbolic_file_bytes=original.symbolic_file_bytes + 8 * 48,
        timing_execution=True, functional_execution=False,
    )
    plan = FlexibleMoeExecutablePlan(
        schema_version=FLEXIBLE_MOE_PLAN_SCHEMA_VERSION,
        producer_pass=_PRODUCER,
        id=stable_artifact_id("flexible_moe_plan", semantic,
                              schema_version=FLEXIBLE_MOE_PLAN_SCHEMA_VERSION),
        **semantic,
    )
    # The default validator/backend rejects this signed producer. Once its
    # official opt-in signed-source branch is installed, recompile it from the
    # original spec+source rather than accepting forged action IDs.
    if _validate_plan:
        if not hasattr(plan, "validate_against_signed"):
            raise SchemaError("official signed train-plan validator is not yet installed",
                              path="moe_signed_source.plan")
        plan.validate_against_signed(spec, signed_source)
    return plan


__all__ = ["compile_moe_signed_top1_train_source_plan"]
