"""N5 exact-provenance projection producer wrappers."""

from __future__ import annotations

from ..errors import SchemaError
from ..policies.interfaces import ProjectToIR2
from ..schema.ir2 import IR2ProjectionResult
from ..schema.n4 import (
    InterDiePlanBundle,
    InterDiePlannedProfile,
    Stage4InterDiePlannedIR1,
    TrainInterDiePlannedIR1,
)
from ..schema.n5 import (
    ProjectToIR2Context,
    ProjectedIR2Bundle,
    ProjectedProfileIR2,
    Stage4ProjectedIR2,
    Stage4ProjectToIR2Context,
    TrainProjectedIR2,
    TrainProjectedReplica,
)
from ..schema.stage4_pd import Stage4KvReshardKind, Stage4PdMode
from ..schema.state_transfer import StateTransferLike
from .stage4_segmented_state_transfer import (
    build_stage4_segmented_state_transfers,
)
from .stage4_state_transfer import build_stage4_planned_state_transfers


def project_profile(
    source: InterDiePlannedProfile,
    context: ProjectToIR2Context,
    projector: ProjectToIR2 | None = None,
) -> ProjectedProfileIR2:
    """Project one planned profile without changing plans or provenance."""

    if type(source) is not InterDiePlannedProfile:
        raise SchemaError(
            "must be an InterDiePlannedProfile",
            path="source",
        )
    if type(context) is not ProjectToIR2Context:
        raise SchemaError(
            "must be a ProjectToIR2Context",
            path="project_to_ir2_context",
        )
    source.validate("source")
    context.validate("project_to_ir2_context")
    if projector is None:
        from ..policies.naive_project_to_ir2 import NaiveProjectToIR2

        projector = NaiveProjectToIR2()
    projection = projector.run(
        source.graph,
        source.fusion_plans,
        source.standalone_plans,
        state_transfers=tuple(
            transfer
            for transfer in context.state_transfers
            if transfer.source_ir1_id == source.graph.id
        ),
    )
    if type(projection) is not IR2ProjectionResult:
        raise SchemaError(
            "projector must return an IR2ProjectionResult",
            path="projection",
        )
    result = ProjectedProfileIR2.create(
        source=source,
        context=context,
        projection=projection,
    )
    result.validate_against(source, context)
    return result


def project_bundle(
    source: InterDiePlanBundle,
    context: ProjectToIR2Context,
    projector: ProjectToIR2 | None = None,
) -> ProjectedIR2Bundle:
    """Project every source entry exactly once in canonical tuple order."""

    if type(source) is not InterDiePlanBundle:
        raise SchemaError(
            "must be an InterDiePlanBundle",
            path="source",
        )
    if type(context) is not ProjectToIR2Context:
        raise SchemaError(
            "must be a ProjectToIR2Context",
            path="project_to_ir2_context",
        )
    source.validate("source")
    context.validate("project_to_ir2_context")
    if projector is None:
        from ..policies.naive_project_to_ir2 import NaiveProjectToIR2

        projector = NaiveProjectToIR2()
    entries = tuple(
        project_profile(entry, context, projector)
        for entry in source.entries
    )
    result = ProjectedIR2Bundle.create(
        source=source,
        context=context,
        entries=entries,
    )
    result.validate_against(source, context)
    return result


def project_train_forward(
    source: TrainInterDiePlannedIR1,
    context: ProjectToIR2Context,
    projector: ProjectToIR2 | None = None,
) -> TrainProjectedIR2:
    """Project each disjoint DP replica without merging its task namespace."""

    if type(source) is not TrainInterDiePlannedIR1:
        raise SchemaError("must be a TrainInterDiePlannedIR1", path="source")
    if type(context) is not ProjectToIR2Context:
        raise SchemaError(
            "must be a ProjectToIR2Context",
            path="project_to_ir2_context",
        )
    source.validate("source")
    context.validate("project_to_ir2_context")
    if context.state_transfers:
        raise SchemaError(
            "train forward projection context must not contain state transfers",
            path="project_to_ir2_context.state_transfers",
        )
    if projector is None:
        from ..policies.naive_project_to_ir2 import NaiveProjectToIR2

        projector = NaiveProjectToIR2()
    replicas: list[TrainProjectedReplica] = []
    for index, source_replica in enumerate(source.replicas):
        projection = projector.run(
            source_replica.graph,
            source_replica.fusion_plans,
            source_replica.standalone_plans,
            state_transfers=(),
        )
        if type(projection) is not IR2ProjectionResult:
            raise SchemaError(
                "projector must return an IR2ProjectionResult",
                path=f"projection[{index}]",
            )
        replica = TrainProjectedReplica.create(
            source=source_replica,
            projection=projection,
        )
        replica.validate_against(
            source_replica,
            f"train_projected_replica[{index}]",
        )
        replicas.append(replica)
    result = TrainProjectedIR2.create(
        source=source,
        context=context,
        replicas=tuple(replicas),
    )
    result.validate_against(source, context)
    return result


def build_stage4_project_state_transfers(
    source: Stage4InterDiePlannedIR1,
) -> tuple[StateTransferLike, ...]:
    """Derive the one transfer representation selected by the frozen PD plan."""

    if type(source) is not Stage4InterDiePlannedIR1:
        raise SchemaError(
            "must be a Stage4InterDiePlannedIR1",
            path="source",
        )
    source.validate("source")
    if source.pd_plan.mode is Stage4PdMode.FUSED:
        if source.pd_plan.reshard is not Stage4KvReshardKind.NONE:
            raise SchemaError(
                "fused PD must use no KV reshard",
                path="source.pd_plan.reshard",
            )
        return build_stage4_planned_state_transfers(source)
    if source.pd_plan.reshard is Stage4KvReshardKind.ONE_TO_ONE:
        return build_stage4_planned_state_transfers(source)
    if source.pd_plan.reshard in (
        Stage4KvReshardKind.GATHER,
        Stage4KvReshardKind.SCATTER,
    ):
        return build_stage4_segmented_state_transfers(source)
    raise SchemaError(
        "separated PD requires one_to_one, gather, or scatter KV reshard",
        path="source.pd_plan.reshard",
    )


def project_stage4(
    source: Stage4InterDiePlannedIR1,
    context: Stage4ProjectToIR2Context,
    projector: ProjectToIR2 | None = None,
) -> Stage4ProjectedIR2:
    """Project one Stage 4 carrier with internally derived state transfers."""

    if type(source) is not Stage4InterDiePlannedIR1:
        raise SchemaError(
            "must be a Stage4InterDiePlannedIR1",
            path="source",
        )
    if type(context) is not Stage4ProjectToIR2Context:
        raise SchemaError(
            "must be a Stage4ProjectToIR2Context",
            path="stage4_project_to_ir2_context",
        )
    source.validate("source")
    context.validate("stage4_project_to_ir2_context")
    state_transfers = build_stage4_project_state_transfers(source)
    if projector is None:
        from ..policies.naive_project_to_ir2 import NaiveProjectToIR2

        projector = NaiveProjectToIR2()
    projection = projector.run(
        source.graph,
        source.fusion_plans,
        source.standalone_plans,
        state_transfers=state_transfers,
    )
    if type(projection) is not IR2ProjectionResult:
        raise SchemaError(
            "projector must return an IR2ProjectionResult",
            path="projection",
        )
    result = Stage4ProjectedIR2.create(
        source=source,
        context=context,
        projection=projection,
    )
    result.validate_against(source, context)
    return result


__all__ = [
    "build_stage4_project_state_transfers",
    "project_bundle",
    "project_profile",
    "project_stage4",
    "project_train_forward",
]
