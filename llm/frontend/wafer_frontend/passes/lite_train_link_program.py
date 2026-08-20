"""Production manifest-link entry for S2-Lite training."""

from __future__ import annotations

from ..errors import SchemaError
from ..lowering.linker import NaiveManifestLinker
from ..schema.lite_train_n6 import (
    S2LiteTrainLinkedProgram,
    S2LiteTrainLoweredProgram,
)


def link_s2_lite_train(
    source: S2LiteTrainLoweredProgram,
) -> S2LiteTrainLinkedProgram:
    """Link the one Lite context into one production manifest/timeline."""

    if type(source) is not S2LiteTrainLoweredProgram:
        raise SchemaError(
            "must be an S2LiteTrainLoweredProgram", path="source"
        )
    source.validate("source")
    result = S2LiteTrainLinkedProgram.create(
        source=source,
        manifest=NaiveManifestLinker().link(
            source.lowering_context,
            source.fragments,
        ),
    )
    result.validate_against(source)
    return result


__all__ = ["link_s2_lite_train"]
