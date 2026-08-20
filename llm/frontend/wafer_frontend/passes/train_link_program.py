"""Link all forward-train DP replicas into one executable manifest."""

from __future__ import annotations

from ..errors import SchemaError
from ..lowering.linker import NaiveManifestLinker
from ..schema.train_n6 import TrainLinkedProgram, TrainLoweredProgram


def link_train(source: TrainLoweredProgram) -> TrainLinkedProgram:
    """Produce the one canonical ProgramArtifact quotient for Train."""

    if type(source) is not TrainLoweredProgram:
        raise SchemaError(
            "must be a TrainLoweredProgram",
            path="source",
        )
    source.validate("source")
    result = TrainLinkedProgram.create(
        source=source,
        manifest=NaiveManifestLinker().link_train(source),
    )
    result.validate_against(source)
    return result


__all__ = ["link_train"]
