"""Public N6 lowering entry points for fixed four-die S3-Lite MoE."""

from __future__ import annotations

from ..errors import SchemaError
from ..lowering.lite_moe_dp4 import (
    lower_lite_moe_dp4_infer,
    lower_lite_moe_dp4_train_forward_tapes,
    rebase_lite_moe_dp4_infer_fragments,
)
from ..lowering.lite_moe_dp4_backward import lower_lite_moe_dp4_backward
from ..schema.lite_moe_dp4_backward import LiteMoeDp4Backward
from ..schema.lite_moe_dp4_execution import LiteMoeDp4ExecutionCase
from ..schema.lite_moe_dp4_n6 import (
    LiteMoeDp4InferLoweredProgram,
    LiteMoeDp4BackwardLoweredProgram,
    LiteMoeDp4TrainForwardLoweredProgram,
)
from ..schema.lite_moe_dp4_train_forward import LiteMoeDp4TrainForward
from .lite_moe_dp4_n6 import build_lite_moe_dp4_infer_n6_intent


def lower_lite_moe_dp4_infer_program(
    source: LiteMoeDp4ExecutionCase,
) -> LiteMoeDp4InferLoweredProgram:
    if type(source) is not LiteMoeDp4ExecutionCase:
        raise SchemaError("must be a LiteMoeDp4ExecutionCase", path="source")
    intent = build_lite_moe_dp4_infer_n6_intent(source)
    result = LiteMoeDp4InferLoweredProgram.create(
        source=source,
        intent=intent,
        fragments=lower_lite_moe_dp4_infer(intent, source),
    )
    result.validate()
    return result


def lower_lite_moe_dp4_train_forward_program(
    source: LiteMoeDp4TrainForward,
) -> LiteMoeDp4TrainForwardLoweredProgram:
    if type(source) is not LiteMoeDp4TrainForward:
        raise SchemaError("must be a LiteMoeDp4TrainForward", path="source")
    forward = lower_lite_moe_dp4_infer_program(source.forward)
    tape_abis, tape_fragments = lower_lite_moe_dp4_train_forward_tapes(
        source, forward.intent
    )
    result = LiteMoeDp4TrainForwardLoweredProgram.create(
        source=source,
        forward=forward,
        tape_buffer_abis=tape_abis,
        tape_fragments=tape_fragments,
        fragments=tuple(sorted((
            *rebase_lite_moe_dp4_infer_fragments(forward.fragments, source.id),
            *tape_fragments,
        ), key=lambda item: item.id)),
    )
    result.validate()
    return result


def lower_lite_moe_dp4_backward_program(
    source: LiteMoeDp4Backward,
) -> LiteMoeDp4BackwardLoweredProgram:
    if type(source) is not LiteMoeDp4Backward:
        raise SchemaError("must be a LiteMoeDp4Backward", path="source")
    formal = source.train_forward.forward
    intent = build_lite_moe_dp4_infer_n6_intent(formal)
    result = LiteMoeDp4BackwardLoweredProgram.create(
        source=source,
        intent=intent,
        fragments=lower_lite_moe_dp4_backward(
            source,
            formal.n4,
            formal.projection,
            formal.schedule,
            formal.global_dag,
            intent,
            formal.adapter.spec.trace,
        ),
    )
    result.validate()
    return result


__all__ = [
    "lower_lite_moe_dp4_backward_program",
    "lower_lite_moe_dp4_infer_program",
    "lower_lite_moe_dp4_train_forward_program",
]
