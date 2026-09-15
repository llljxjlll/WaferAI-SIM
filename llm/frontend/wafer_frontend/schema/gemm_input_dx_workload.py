"""NEW-only rank-local GEMM input dX source/timing profile.

Forward X[K,M] × W[M,N] → Y[K,N] proves geometry. The derivative reads
FP16 W[M,N] and FP16 dY[K,N] and reserves FP32 dX[K,M]. X is a source/tape
witness, not a fabricated third operand read by the derivative primitive.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import DType

_MAX_PROFILE = (1 << 30) - 1
_MAX_SRAM_SPAN = 1 << 16


@dataclass(frozen=True, slots=True)
class GemmInputDxWorkload:
    k: int
    m: int
    n: int
    source_forward_op_ref: str
    source_parameter_state_ref: str
    activation_dtype: DType = DType.FP16
    weight_dtype: DType = DType.FP16
    upstream_dtype: DType = DType.FP16
    output_dtype: DType = DType.FP32

    def validate(self, path: str = "gemm_input_dx_workload") -> None:
        if (any(type(value) is not int or not 1 <= value <= _MAX_PROFILE
                for value in (self.k, self.m, self.n))
                or any(type(ref) is not str or not ref for ref in (
                    self.source_forward_op_ref, self.source_parameter_state_ref,
                ))):
            raise SchemaError("GEMM dX needs positive tile and real source refs", path=path)
        if (self.activation_dtype is not DType.FP16
                or self.weight_dtype is not DType.FP16
                or self.upstream_dtype is not DType.FP16
                or self.output_dtype is not DType.FP32):
            raise SchemaError("GEMM dX requires source FP16 X/W/dY and FP32 dX", path=path)
        if max(self.weight_bytes, self.upstream_bytes,
               self.output_bytes) > _MAX_SRAM_SPAN:
            raise SchemaError("one GEMM dX tile exceeds 16-bit physical SRAM", path=path)

    @property
    def activation_bytes(self) -> int:
        return 2 * self.k * self.m

    @property
    def weight_bytes(self) -> int:
        return 2 * self.m * self.n

    @property
    def upstream_bytes(self) -> int:
        return 2 * self.k * self.n

    @property
    def output_bytes(self) -> int:
        return 4 * self.k * self.m

    @property
    def fma_ops(self) -> int:
        return self.k * self.m * self.n


__all__ = ["GemmInputDxWorkload"]
