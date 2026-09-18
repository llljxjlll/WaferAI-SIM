"""Link all forward-train DP replicas into one executable manifest."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema._validation_session import (
    builder_validation_session, cache_validation_aux,
)
from ..lowering.linker import NaiveManifestLinker
from ..schema.train_n6 import TrainLinkedProgram, TrainLoweredProgram


@builder_validation_session()
def link_train(source: TrainLoweredProgram) -> TrainLinkedProgram:
    """Produce the one canonical ProgramArtifact quotient for Train."""

    if type(source) is not TrainLoweredProgram:
        raise SchemaError(
            "must be a TrainLoweredProgram",
            path="source",
        )
    source.validate("source")
    manifest = NaiveManifestLinker().link_train(source)
    # The canonical linker has just produced this exact object. Let the
    # enclosing builder session reuse that proof instead of linking the same
    # large source again during TrainLinkedProgram.create(). External reads
    # have no such session-local proof and still recompute the quotient.
    cache_validation_aux(source, "strict_train_linked_manifest", manifest)
    result = TrainLinkedProgram.create(source=source, manifest=manifest)
    result.validate_against(source)
    return result


__all__ = ["link_train"]
