"""MoE forward IR0 node workload bound to actual per-layer E2E routing.

Each workload carries an E2E operation and frozen trace.  Validation checks
the full H/I/E and token histogram; source graph builders must separately
verify each operation ref/trace digest against the production materialization
and bind genuine norm2 activation→router/dispatch→expert→combine→residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import DType, validate_nonempty


class MoeForwardBlockKind(str, Enum):
    ROUTER = "router"
    ROUTE_FREEZE = "route_freeze"
    DISPATCH = "dispatch"
    EXPERT = "expert_forward"
    COMBINE = "combine"


@dataclass(frozen=True, slots=True)
class MoeFullTrainingBlockWorkload:
    """Top-1 static route with real three-projection SwiGLU per expert."""

    kind: MoeForwardBlockKind
    source_case_ref: str
    source_operation_ref: str
    source_route_trace_ref: str
    source_route_trace_digest: str
    step: int
    layer: int
    expert: int | None
    token_count: int
    hidden_size: int
    intermediate_size: int
    expert_count: int
    expert_histogram: tuple[int, ...]
    frozen_expert_by_token: tuple[int, ...]
    input_dtype: DType = DType.FP16
    route_dtype: DType = DType.INT32
    expert_output_dtype: DType = DType.FP16
    combine_accum_dtype: DType = DType.FP32

    def validate(self, path: str = "moe_full_training_block_workload") -> None:
        if type(self.kind) is not MoeForwardBlockKind:
            raise SchemaError("source MoE forward operation kind is not typed",
                              path=f"{path}.kind")
        for field in ("source_case_ref", "source_operation_ref",
                      "source_route_trace_ref", "source_route_trace_digest"):
            validate_nonempty(getattr(self, field), f"{path}.{field}")
        for field in ("step", "layer", "token_count", "hidden_size",
                      "intermediate_size", "expert_count"):
            value = getattr(self, field)
            if type(value) is not int or value < (0 if field in ("step", "layer") else 1):
                raise SchemaError("MoE source operation has invalid model/step dimensions",
                                  path=f"{path}.{field}")
        if (type(self.expert_histogram) is not tuple
                or len(self.expert_histogram) != self.expert_count
                or any(type(m) is not int or m < 0 for m in self.expert_histogram)
                or sum(self.expert_histogram) != self.token_count
                or type(self.frozen_expert_by_token) is not tuple
                or len(self.frozen_expert_by_token) != self.token_count
                or any(type(expert) is not int or not 0 <= expert < self.expert_count
                       for expert in self.frozen_expert_by_token)
                or self.expert_histogram != tuple(
                    self.frozen_expert_by_token.count(expert)
                    for expert in range(self.expert_count))):
            raise SchemaError("frozen source trace must route every token to exactly one expert",
                              path=f"{path}.frozen_expert_by_token")
        if (self.kind is MoeForwardBlockKind.EXPERT
                and (type(self.expert) is not int
                     or not 0 <= self.expert < self.expert_count)):
            raise SchemaError("expert forward needs its exact EP owner",
                              path=f"{path}.expert")
        if self.kind is not MoeForwardBlockKind.EXPERT and self.expert is not None:
            raise SchemaError("shared MoE operation cannot claim expert owner",
                              path=f"{path}.expert")
        if (self.input_dtype is not DType.FP16
                or self.route_dtype is not DType.INT32
                or self.expert_output_dtype is not DType.FP16
                or self.combine_accum_dtype is not DType.FP32):
            raise SchemaError("FP16 shared/expert and INT32 route dtype contract changed",
                              path=f"{path}.dtype")

    @property
    def owned_token_count(self) -> int:
        return (self.token_count if self.expert is None
                else self.expert_histogram[self.expert])

    @property
    def projected_matmul_flops(self) -> int:
        """True three gate/up/down projections, or source router gate GEMM."""
        if self.kind is MoeForwardBlockKind.ROUTER:
            return 2 * self.token_count * self.hidden_size * self.expert_count
        if self.kind is MoeForwardBlockKind.EXPERT:
            return 6 * self.owned_token_count * self.hidden_size * self.intermediate_size
        return 0

    @property
    def expert_projection_parameter_bytes(self) -> int:
        if self.kind is MoeForwardBlockKind.EXPERT:
            return 2 * (2 * self.hidden_size * self.intermediate_size
                        + self.intermediate_size * self.hidden_size)
        return 0

    @property
    def projected_vector_flops(self) -> int:
        if self.kind is MoeForwardBlockKind.COMBINE:
            return 2 * self.token_count * self.hidden_size
        return 0


__all__ = ["MoeForwardBlockKind", "MoeFullTrainingBlockWorkload"]
