"""Public single-manifest linker entry points for four-die S3-Lite MoE."""

from __future__ import annotations

from ..errors import SchemaError
from ..lowering.lite_moe_dp4_linker import (
    link_lite_moe_dp4_infer_manifest,
    link_lite_moe_dp4_train_forward_manifest,
)
from ..lowering.lite_moe_dp4_backward_linker import (
    link_lite_moe_dp4_backward_manifest,
)
from ..schema.lite_moe_dp4_n6 import (
    LiteMoeDp4BackwardLinkedProgram,
    LiteMoeDp4BackwardLoweredProgram,
    LiteMoeDp4InferLinkedProgram,
    LiteMoeDp4InferLoweredProgram,
    LiteMoeDp4TrainForwardLinkedProgram,
    LiteMoeDp4TrainForwardLoweredProgram,
)


def link_lite_moe_dp4_infer_program(
    source: LiteMoeDp4InferLoweredProgram,
) -> LiteMoeDp4InferLinkedProgram:
    if type(source) is not LiteMoeDp4InferLoweredProgram:
        raise SchemaError("must be a LiteMoeDp4InferLoweredProgram", path="source")
    result = LiteMoeDp4InferLinkedProgram.create(
        source=source,
        manifest=link_lite_moe_dp4_infer_manifest(source),
    )
    result.validate()
    return result


def link_lite_moe_dp4_train_forward_program(
    source: LiteMoeDp4TrainForwardLoweredProgram,
) -> LiteMoeDp4TrainForwardLinkedProgram:
    if type(source) is not LiteMoeDp4TrainForwardLoweredProgram:
        raise SchemaError("must be a LiteMoeDp4TrainForwardLoweredProgram", path="source")
    result = LiteMoeDp4TrainForwardLinkedProgram.create(
        source=source,
        manifest=link_lite_moe_dp4_train_forward_manifest(source),
    )
    result.validate()
    return result


def link_lite_moe_dp4_backward_program(
    source: LiteMoeDp4BackwardLoweredProgram,
) -> LiteMoeDp4BackwardLinkedProgram:
    if type(source) is not LiteMoeDp4BackwardLoweredProgram:
        raise SchemaError("must be a LiteMoeDp4BackwardLoweredProgram", path="source")
    result = LiteMoeDp4BackwardLinkedProgram.create(
        source=source,
        manifest=link_lite_moe_dp4_backward_manifest(source),
    )
    result.validate()
    return result


__all__ = [
    "link_lite_moe_dp4_backward_program",
    "link_lite_moe_dp4_infer_program",
    "link_lite_moe_dp4_train_forward_program",
]
