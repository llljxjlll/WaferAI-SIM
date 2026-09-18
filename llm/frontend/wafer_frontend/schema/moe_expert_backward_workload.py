"""Typed expert reverse with recomputation of SwiGLU activations.

The derivative uses actual combine dExpert.  Gate/up projections must be
recomputed from the same forward activation and weights before their FP32
WGRAD and dX operations; no detached or zero-filled tape is legal.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import DType, validate_nonempty


@dataclass(frozen=True, slots=True)
class MoeExpertBackwardWorkload:
    source_forward_op_ref: str
    source_combine_backward_op_ref: str
    source_route_trace_digest: str
    step: int
    layer: int
    expert: int
    token_count: int
    hidden_size: int
    intermediate_size: int
    expert_count: int
    input_dtype: DType = DType.FP16
    gradient_dtype: DType = DType.FP32
    output_dtype: DType = DType.FP16

    def validate(self, path: str = "moe_expert_backward_workload") -> None:
        for name in ("source_forward_op_ref", "source_combine_backward_op_ref",
                     "source_route_trace_digest"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in ("step", "layer", "expert", "token_count",
                     "hidden_size", "intermediate_size", "expert_count"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in ("step", "layer", "expert") else 1):
                raise SchemaError("expert backward has invalid source geometry",
                                  path=f"{path}.{name}")
        if (self.expert_count not in (1, 2)
                or self.expert >= self.expert_count
                or self.input_dtype is not DType.FP16
                or self.gradient_dtype is not DType.FP32
                or self.output_dtype is not DType.FP16):
            raise SchemaError("expert reverse requires EP1/EP2 owner and FP16→FP32",
                              path=path)

    @property
    def recompute_bytes(self) -> int:
        return 4 * self.token_count * self.intermediate_size

    @property
    def fp32_weight_gradient_bytes(self) -> int:
        return 4 * self.hidden_size * self.intermediate_size


__all__ = ["MoeExpertBackwardWorkload"]
