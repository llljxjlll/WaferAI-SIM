"""Physical 0x28 score-weighted top1 MoE combine reverse workload."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import DType


@dataclass(frozen=True, slots=True)
class MoeCombineBackwardWorkload:
    source_forward_op_ref: str
    source_route_trace_digest: str
    step: int
    layer: int
    token_count: int
    hidden_size: int
    expert_count: int
    route_bytes: int
    route_dtype: DType = DType.INT32
    score_dtype: DType = DType.FP16
    expert_dtype: DType = DType.FP16
    upstream_dtype: DType = DType.FP16
    dscore_dtype: DType = DType.FP16
    dexpert_dtype: DType = DType.FP16

    def validate(self, path: str = "moe_combine_backward_workload") -> None:
        if (type(self.source_forward_op_ref) is not str
                or not self.source_forward_op_ref
                or type(self.source_route_trace_digest) is not str
                or len(self.source_route_trace_digest) != 64
                or any(ch not in "0123456789abcdef"
                       for ch in self.source_route_trace_digest)
                or any(type(value) is not int for value in (
                    self.step, self.layer, self.token_count,
                    self.hidden_size, self.expert_count, self.route_bytes))
                or self.step not in (0, 1)
                or self.layer not in (0, 1)
                or min(self.token_count, self.hidden_size,
                       self.expert_count) < 1
                or self.route_bytes != 20 * self.token_count
                or max(self.route_bytes,
                       2 * self.token_count * self.hidden_size,
                       2 * self.token_count * self.expert_count) > 65536
                or self.route_dtype is not DType.INT32
                or any(dtype is not DType.FP16 for dtype in (
                    self.score_dtype, self.expert_dtype,
                    self.upstream_dtype, self.dscore_dtype,
                    self.dexpert_dtype))):
            raise SchemaError(
                "native 0x28 requires source-bound INT32 route and FP16 "
                "score/expert/dCombined/dScore/dExpert extents",
                path=path,
            )


__all__ = ["MoeCombineBackwardWorkload"]
