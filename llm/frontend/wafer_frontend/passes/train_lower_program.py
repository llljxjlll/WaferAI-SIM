"""Lower every forward-train DP replica through the production N6 path."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.train_global_action import TrainGlobalAction
from ..schema.train_n6 import (
    TrainLoweredProgram,
    TrainLoweredReplica,
    train_replica_lowering_context,
)
from .lower_program import _lower_fragments, _resolve_dependencies


def lower_train(source: TrainGlobalAction) -> TrainLoweredProgram:
    """Lower DP replicas independently before the N6.5 unified link step."""

    if type(source) is not TrainGlobalAction:
        raise SchemaError("must be a TrainGlobalAction", path="source")
    source.validate("source")
    dependencies = _resolve_dependencies(None, None, None, None, None)
    replicas = []
    for source_replica in source.replicas:
        context = train_replica_lowering_context(source_replica)
        replicas.append(
            TrainLoweredReplica.create(
                source=source_replica,
                lowering_context=context,
                fragments=_lower_fragments(context, dependencies),
            )
        )
    result = TrainLoweredProgram.create(
        source=source,
        replicas=tuple(replicas),
    )
    result.validate_against(source)
    return result


__all__ = ["lower_train"]
