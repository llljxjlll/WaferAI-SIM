"""Public isolated link pass for the S3-Lite MoE backward preview."""

from __future__ import annotations

from ..errors import SchemaError
from ..lowering.lite_moe_backward_linker import link_lite_moe_backward_manifest
from ..schema.lite_moe_backward_n6 import (
    LiteMoeBackwardLinkedProgram,
    LiteMoeBackwardLoweredProgram,
)


def link_lite_moe_backward_program(
    source: LiteMoeBackwardLoweredProgram,
) -> LiteMoeBackwardLinkedProgram:
    if type(source) is not LiteMoeBackwardLoweredProgram:
        raise SchemaError("must be a LiteMoeBackwardLoweredProgram", path="source")
    source.validate("source")
    return LiteMoeBackwardLinkedProgram.create(
        source=source,
        manifest=link_lite_moe_backward_manifest(source),
    )


__all__ = ["link_lite_moe_backward_program"]
