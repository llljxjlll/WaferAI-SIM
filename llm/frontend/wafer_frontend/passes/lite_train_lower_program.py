"""Production N6 lowering entry for the S2-Lite train carrier."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.lite_train_n6 import (
    S2LiteTrainLoweredProgram,
    s2_lite_train_lowering_context,
)
from ..schema.train_global_action import S2LiteTrainGlobalAction
from .lower_program import _lower_fragments, _resolve_dependencies


def lower_s2_lite_train(
    source: S2LiteTrainGlobalAction,
) -> S2LiteTrainLoweredProgram:
    """Lower the one formal Lite timeline through production lowerers."""

    if type(source) is not S2LiteTrainGlobalAction:
        raise SchemaError(
            "must be an S2LiteTrainGlobalAction", path="source"
        )
    source.validate("source")
    context = s2_lite_train_lowering_context(source)
    dependencies = _resolve_dependencies(None, None, None, None, None)
    result = S2LiteTrainLoweredProgram.create(
        source=source,
        lowering_context=context,
        fragments=_lower_fragments(context, dependencies),
    )
    result.validate_against(source)
    return result


__all__ = ["lower_s2_lite_train"]
