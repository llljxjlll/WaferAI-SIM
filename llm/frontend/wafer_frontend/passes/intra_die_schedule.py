"""N5 exact-provenance intra-die scheduling producer wrappers."""

from __future__ import annotations

from ..errors import SchemaError
from ..policies.interfaces import IntraDiePolicy
from ..schema.ir2 import IntraDieScheduleSet
from ..schema.n5 import (
    IntraDieSchedulingContext,
    ProjectedIR2Bundle,
    ProjectedProfileIR2,
    ScheduledIR2Bundle,
    ScheduledProfileIR2,
    Stage4ProjectedIR2,
    Stage4ScheduledIR2,
    TrainProjectedIR2,
    TrainScheduledIR2,
    TrainScheduledReplica,
)


def schedule_profile(
    source: ProjectedProfileIR2,
    context: IntraDieSchedulingContext,
    policy: IntraDiePolicy | None = None,
) -> ScheduledProfileIR2:
    """Schedule one projected profile without changing its semantic DAGs."""

    if type(source) is not ProjectedProfileIR2:
        raise SchemaError(
            "must be a ProjectedProfileIR2",
            path="source",
        )
    if type(context) is not IntraDieSchedulingContext:
        raise SchemaError(
            "must be an IntraDieSchedulingContext",
            path="intra_die_scheduling_context",
        )
    source.validate("source")
    context.validate("intra_die_scheduling_context")
    if policy is None:
        from ..policies.naive_intra_die import NaiveIntraDiePolicy

        policy = NaiveIntraDiePolicy()
    schedule_set = policy.schedule(source.projection, source.graph)
    if type(schedule_set) is not IntraDieScheduleSet:
        raise SchemaError(
            "policy must return an IntraDieScheduleSet",
            path="schedule_set",
        )
    result = ScheduledProfileIR2.create(
        source=source,
        context=context,
        schedule_set=schedule_set,
    )
    result.validate_against(source, context)
    return result


def schedule_bundle(
    source: ProjectedIR2Bundle,
    context: IntraDieSchedulingContext,
    policy: IntraDiePolicy | None = None,
) -> ScheduledIR2Bundle:
    """Schedule every source entry exactly once in canonical tuple order."""

    if type(source) is not ProjectedIR2Bundle:
        raise SchemaError(
            "must be a ProjectedIR2Bundle",
            path="source",
        )
    if type(context) is not IntraDieSchedulingContext:
        raise SchemaError(
            "must be an IntraDieSchedulingContext",
            path="intra_die_scheduling_context",
        )
    source.validate("source")
    context.validate("intra_die_scheduling_context")
    if policy is None:
        from ..policies.naive_intra_die import NaiveIntraDiePolicy

        policy = NaiveIntraDiePolicy()
    entries = tuple(
        schedule_profile(entry, context, policy)
        for entry in source.entries
    )
    result = ScheduledIR2Bundle.create(
        source=source,
        context=context,
        entries=entries,
    )
    result.validate_against(source, context)
    return result


def schedule_stage4(
    source: Stage4ProjectedIR2,
    context: IntraDieSchedulingContext,
    policy: IntraDiePolicy | None = None,
) -> Stage4ScheduledIR2:
    """Schedule one formal Stage 4 graph without bundle adaptation."""

    if type(source) is not Stage4ProjectedIR2:
        raise SchemaError(
            "must be a Stage4ProjectedIR2",
            path="source",
        )
    if type(context) is not IntraDieSchedulingContext:
        raise SchemaError(
            "must be an IntraDieSchedulingContext",
            path="intra_die_scheduling_context",
        )
    source.validate("source")
    context.validate("intra_die_scheduling_context")
    if policy is None:
        from ..policies.naive_intra_die import NaiveIntraDiePolicy

        policy = NaiveIntraDiePolicy()
    schedule_set = policy.schedule(source.projection, source.graph)
    if type(schedule_set) is not IntraDieScheduleSet:
        raise SchemaError(
            "policy must return an IntraDieScheduleSet",
            path="schedule_set",
        )
    result = Stage4ScheduledIR2.create(
        source=source,
        context=context,
        schedule_set=schedule_set,
    )
    result.validate_against(source, context)
    return result


def schedule_train_forward(
    source: TrainProjectedIR2,
    context: IntraDieSchedulingContext,
    policy: IntraDiePolicy | None = None,
) -> TrainScheduledIR2:
    """Schedule each DP replica independently on its physical cores."""

    if type(source) is not TrainProjectedIR2:
        raise SchemaError("must be a TrainProjectedIR2", path="source")
    if type(context) is not IntraDieSchedulingContext:
        raise SchemaError(
            "must be an IntraDieSchedulingContext",
            path="intra_die_scheduling_context",
        )
    source.validate("source")
    context.validate("intra_die_scheduling_context")
    if policy is None:
        from ..policies.naive_intra_die import NaiveIntraDiePolicy

        policy = NaiveIntraDiePolicy()
    replicas: list[TrainScheduledReplica] = []
    for index, source_replica in enumerate(source.replicas):
        schedule_set = policy.schedule(
            source_replica.projection,
            source_replica.graph,
        )
        if type(schedule_set) is not IntraDieScheduleSet:
            raise SchemaError(
                "policy must return an IntraDieScheduleSet",
                path=f"schedule_set[{index}]",
            )
        replica = TrainScheduledReplica.create(
            source=source_replica,
            schedule_set=schedule_set,
        )
        replica.validate_against(
            source_replica,
            f"train_scheduled_replica[{index}]",
        )
        replicas.append(replica)
    result = TrainScheduledIR2.create(
        source=source,
        context=context,
        replicas=tuple(replicas),
    )
    result.validate_against(source, context)
    return result


__all__ = [
    "schedule_bundle",
    "schedule_profile",
    "schedule_stage4",
    "schedule_train_forward",
]
