"""Pure producers for the isolated S2-Lite DP2 rooted-AR quotient."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.lite_train_dp2 import (
    S2LiteDp2RootedArGlobalAction,
    S2LiteDp2RootedArSource,
)
from ..schema.lite_train_graph import S2LiteLmHeadTrainIR0
from ..schema.n5 import TrainScheduledIR2
from .global_action import build_global_action_dag


def build_s2_lite_dp2_rooted_ar_source(
    base: S2LiteLmHeadTrainIR0,
) -> S2LiteDp2RootedArSource:
    if type(base) is not S2LiteLmHeadTrainIR0:
        raise SchemaError("must be an S2LiteLmHeadTrainIR0", path="base")
    return S2LiteDp2RootedArSource.create(base=base)


def build_s2_lite_dp2_rooted_ar_global_action(
    source: S2LiteDp2RootedArSource,
    scheduled: TrainScheduledIR2,
) -> S2LiteDp2RootedArGlobalAction:
    if type(source) is not S2LiteDp2RootedArSource:
        raise SchemaError("must be a typed DP2 source", path="source")
    if type(scheduled) is not TrainScheduledIR2:
        raise SchemaError("must be a TrainScheduledIR2", path="scheduled")
    source.validate("source")
    scheduled.validate("scheduled")
    local_dags = tuple(
        build_global_action_dag(
            replica.projected.graph,
            replica.projected.projection,
            replica.schedule_set,
        )
        for replica in scheduled.replicas
    )
    return S2LiteDp2RootedArGlobalAction.create(
        source=source,
        scheduled=scheduled,
        local_dags=local_dags,
    )


__all__ = [
    "build_s2_lite_dp2_rooted_ar_global_action",
    "build_s2_lite_dp2_rooted_ar_source",
]
