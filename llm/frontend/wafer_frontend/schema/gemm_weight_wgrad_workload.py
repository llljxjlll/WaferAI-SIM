"""Typed rank-local parameter WGRAD tile for a source Dense/MoE GEMM.

The tuple (m,n,k) means dW[m,n] from X[k,m] and upstream dY[k,n].
`k` is the source forward GEMM rank-row count.  A larger weight shard needs
producer-visible tile decomposition and later cross-tile/DP accumulation;
this workload only describes one exact physical tile.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import DType

_MAX_PARAMETER = (1 << 30) - 1
_MAX_SRAM_SPAN = 1 << 16


@dataclass(frozen=True, slots=True)
class GemmWeightWgradWorkload:
    m: int
    n: int
    k: int
    source_forward_op_ref: str
    source_parameter_state_ref: str
    activation_dtype: DType = DType.FP16
    upstream_dtype: DType = DType.FP16
    gradient_dtype: DType = DType.FP32

    def validate(self, path: str = "gemm_weight_wgrad_workload") -> None:
        if (any(type(value) is not int or not 1 <= value <= _MAX_PARAMETER
                for value in (self.m, self.n, self.k))
                or type(self.source_forward_op_ref) is not str
                or not self.source_forward_op_ref
                or type(self.source_parameter_state_ref) is not str
                or not self.source_parameter_state_ref):
            raise SchemaError("WGRAD tile or source forward/parameter refs are invalid",
                              path=path)
        if (self.activation_dtype is not DType.FP16
                or self.upstream_dtype is not DType.FP16
                or self.gradient_dtype is not DType.FP32):
            raise SchemaError("named GEMM WGRAD requires FP16 X/dY to FP32 dW",
                              path=path)
        if (max(self.activation_bytes, self.upstream_bytes,
                self.gradient_bytes) > _MAX_SRAM_SPAN):
            raise SchemaError("one GEMM WGRAD tile must fit typed 16-bit SRAM spans",
                              path=path)

    @property
    def activation_bytes(self) -> int:
        return self.k * self.m * 2

    @property
    def upstream_bytes(self) -> int:
        return self.k * self.n * 2

    @property
    def gradient_bytes(self) -> int:
        return self.m * self.n * 4

    @property
    def fma_ops(self) -> int:
        return self.m * self.n * self.k


__all__ = ["GemmWeightWgradWorkload"]
