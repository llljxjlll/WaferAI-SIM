"""Typed real TRAIN WGRAD workloads for shared Embedding and RMSNorm gamma.

These represent the new *public* 0x23 and 0x24 record operands.  Old dX-only
internal reverse prims cannot satisfy either parameter-gradient producer.
The source node builder must supply genuine per-row INT32 token IDs from the
forward tape; a synthetic convenient index pattern is never accepted as
runtime evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import DType


_INDEX_TRACE_CAPACITY = 16


@dataclass(frozen=True, slots=True)
class EmbeddingTableWgradWorkload:
    """INT32 index+FP16 dHidden→FP32 table scatter; no dIndex exists."""

    logical_rows: int
    rank_rows: int
    tp_degree: int
    vocab_size: int
    vocab_start: int
    vocab_rows: int
    hidden_size: int
    index_trace: tuple[int, ...]
    index_dtype: DType = DType.INT32
    table_dtype: DType = DType.FP16
    upstream_dtype: DType = DType.FP16
    output_dtype: DType = DType.FP32

    def validate(self, path: str = "embedding_table_wgrad_workload") -> None:
        if (any(type(value) is not int for value in (
                self.logical_rows, self.rank_rows, self.tp_degree,
                self.vocab_size, self.vocab_start, self.vocab_rows,
                self.hidden_size))
                or not 1 <= self.rank_rows <= _INDEX_TRACE_CAPACITY
                or self.logical_rows != self.rank_rows * self.tp_degree
                or self.tp_degree < 1 or self.vocab_size < 1
                or self.vocab_start < 0 or self.vocab_rows < 1
                or self.vocab_start + self.vocab_rows > self.vocab_size
                or self.hidden_size < 1
                or len(self.index_trace) != _INDEX_TRACE_CAPACITY
                or any(type(index) is not int for index in self.index_trace)
                or any(index < 0 or index >= self.vocab_size
                       for index in self.index_trace[:self.rank_rows])
                or any(index != 0 for index in
                       self.index_trace[self.rank_rows:])
                or not any(self.vocab_start <= index <
                           self.vocab_start + self.vocab_rows for index in
                           self.index_trace[:self.rank_rows])):
            raise SchemaError("physical index trace or owned vocabulary tile invalid",
                              path=path)
        if (self.index_dtype is not DType.INT32
                or self.table_dtype is not DType.FP16
                or self.upstream_dtype is not DType.FP16
                or self.output_dtype is not DType.FP32):
            raise SchemaError("table scatter dtype differs from native 0x23",
                              path=path)

    @property
    def input_index_bytes(self) -> int:
        return self.rank_rows * 4

    @property
    def upstream_bytes(self) -> int:
        return self.rank_rows * self.hidden_size * 2

    @property
    def table_tile_bytes(self) -> int:
        return self.vocab_rows * self.hidden_size * 2

    @property
    def physical_gradient_bytes(self) -> int:
        return self.vocab_rows * self.hidden_size * 4


@dataclass(frozen=True, slots=True)
class NormGammaWgradWorkload:
    """FP16 forward activation+upstream→FP32 gamma[hidden]; not dX."""

    logical_rows: int
    rank_rows: int
    tp_degree: int
    hidden_size: int
    mode: int = 0
    input_dtype: DType = DType.FP16
    upstream_dtype: DType = DType.FP16
    output_dtype: DType = DType.FP32

    def validate(self, path: str = "norm_gamma_wgrad_workload") -> None:
        if (any(type(value) is not int for value in (
                self.logical_rows, self.rank_rows, self.tp_degree,
                self.hidden_size, self.mode))
                or self.rank_rows < 1
                or self.logical_rows != self.rank_rows * self.tp_degree
                or self.tp_degree < 1 or self.hidden_size < 1
                or self.mode not in (0, 1)):
            raise SchemaError("norm gamma logical/rank rows and mode invalid",
                              path=path)
        if (self.input_dtype is not DType.FP16
                or self.upstream_dtype is not DType.FP16
                or self.output_dtype is not DType.FP32):
            raise SchemaError("norm gamma native 0x24 requires FP16→FP32",
                              path=path)

    @property
    def forward_bytes(self) -> int:
        return self.rank_rows * self.hidden_size * 2

    @property
    def upstream_bytes(self) -> int:
        return self.forward_bytes

    @property
    def physical_gradient_bytes(self) -> int:
        return self.hidden_size * 4


__all__ = ["EmbeddingTableWgradWorkload", "NormGammaWgradWorkload"]
