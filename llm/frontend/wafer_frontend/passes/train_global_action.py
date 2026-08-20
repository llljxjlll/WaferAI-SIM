"""Build the exact DP-replica forward-train GlobalAction carrier."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.n5 import TrainScheduledIR2
from ..schema.train_global_action import (
    S2LiteTrainGlobalAction,
    TrainGlobalAction,
    TrainGlobalActionReplica,
)
from .global_action import build_global_action_dag


def build_train_global_action(source: TrainScheduledIR2) -> TrainGlobalAction:
    """Quotient every scheduled DP replica exactly once in canonical order."""

    if type(source) is not TrainScheduledIR2:
        raise SchemaError("must be a TrainScheduledIR2", path="source")
    source.validate("source")
    replicas = tuple(
        TrainGlobalActionReplica.create(
            source=replica,
            global_dag=build_global_action_dag(
                replica.projected.graph,
                replica.projected.projection,
                replica.schedule_set,
            ),
        )
        for replica in source.replicas
    )
    result = TrainGlobalAction.create(source=source, replicas=replicas)
    result.validate_against(source)
    return result


def build_s2_lite_train_global_action(
    source: TrainScheduledIR2,
) -> S2LiteTrainGlobalAction:
    """Build the one formal TP1/DP1 LM-head-only training quotient."""

    if type(source) is not TrainScheduledIR2:
        raise SchemaError("must be a TrainScheduledIR2", path="source")
    source.validate("source")
    global_dags = tuple(
        build_global_action_dag(
            replica.projected.graph,
            replica.projected.projection,
            replica.schedule_set,
        )
        for replica in source.replicas
    )
    result = S2LiteTrainGlobalAction.create(
        source=source,
        global_dags=global_dags,
    )
    result.validate_against(source)
    return result


__all__ = [
    "build_s2_lite_train_global_action",
    "build_train_global_action",
]
