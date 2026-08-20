"""N5 exact-provenance global-action producer wrappers."""

from __future__ import annotations

from ..errors import SchemaError
from ..policies.interfaces import GlobalActionDAGBuilder
from ..schema.global_action import GlobalActionDAG
from ..schema.n5 import (
    GlobalActionBundle,
    GlobalActionProfile,
    ScheduledIR2Bundle,
    ScheduledProfileIR2,
    Stage4GlobalAction,
    Stage4ScheduledIR2,
)


def build_global_profile(
    source: ScheduledProfileIR2,
    builder: GlobalActionDAGBuilder | None = None,
) -> GlobalActionProfile:
    """Build the exact one-action-per-task quotient for one profile."""

    if type(source) is not ScheduledProfileIR2:
        raise SchemaError(
            "must be a ScheduledProfileIR2",
            path="source",
        )
    source.validate("source")
    if builder is None:
        from .global_action import build_global_action_dag

        global_dag = build_global_action_dag(
            source.graph,
            source.projection,
            source.schedule_set,
        )
    else:
        global_dag = builder.build(
            source.graph,
            source.projection,
            source.schedule_set,
        )
    if type(global_dag) is not GlobalActionDAG:
        raise SchemaError(
            "builder must return a GlobalActionDAG",
            path="global_dag",
        )
    result = GlobalActionProfile.create(
        source=source,
        global_dag=global_dag,
    )
    result.validate_against(source)
    return result


def build_global_bundle(
    source: ScheduledIR2Bundle,
    builder: GlobalActionDAGBuilder | None = None,
) -> GlobalActionBundle:
    """Build every source entry exactly once in canonical tuple order."""

    if type(source) is not ScheduledIR2Bundle:
        raise SchemaError(
            "must be a ScheduledIR2Bundle",
            path="source",
        )
    source.validate("source")
    entries = tuple(
        build_global_profile(entry, builder)
        for entry in source.entries
    )
    result = GlobalActionBundle.create(source=source, entries=entries)
    result.validate_against(source)
    return result


def build_stage4_global_action(
    source: Stage4ScheduledIR2,
) -> Stage4GlobalAction:
    """Build the exact GlobalAction quotient for one Stage 4 graph."""

    if type(source) is not Stage4ScheduledIR2:
        raise SchemaError(
            "must be a Stage4ScheduledIR2",
            path="source",
        )
    source.validate("source")
    from .global_action import build_global_action_dag

    global_dag = build_global_action_dag(
        source.graph,
        source.projection,
        source.schedule_set,
    )
    result = Stage4GlobalAction.create(
        source=source,
        global_dag=global_dag,
    )
    result.validate_against(source)
    return result


__all__ = [
    "build_global_bundle",
    "build_global_profile",
    "build_stage4_global_action",
]
