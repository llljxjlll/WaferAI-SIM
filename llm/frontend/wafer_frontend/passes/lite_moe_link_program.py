"""Public S3-Lite single-manifest link pass."""

from __future__ import annotations

from ..errors import SchemaError
from ..lowering.linker import NaiveManifestLinker
from ..schema.lite_moe_n6 import LiteMoeLinkedProgram, LiteMoeLoweredProgram


def link_lite_moe_n6(source: LiteMoeLoweredProgram) -> LiteMoeLinkedProgram:
    if type(source) is not LiteMoeLoweredProgram:
        raise SchemaError("must be a LiteMoeLoweredProgram", path="source")
    source.validate("source")
    return LiteMoeLinkedProgram.create(
        source=source,
        manifest=NaiveManifestLinker().link_lite_moe(source),
    )


__all__ = ["link_lite_moe_n6"]
