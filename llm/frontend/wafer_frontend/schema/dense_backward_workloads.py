"""Typed rank-local workloads for public Dense backbone backward records."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import DType


_MAX_PROFILE = (1 << 30) - 1
_MAX_SRAM_SPAN = 1 << 16


def _positive(values: tuple[int, ...], path: str) -> None:
    if any(type(value) is not int or not 1 <= value <= _MAX_PROFILE
           for value in values):
        raise SchemaError("backward profile requires positive 30-bit integers",
                          path=path)


@dataclass(frozen=True, slots=True)
class RmsNormBackwardWorkload:
    rows: int
    hidden_size: int
    tp_degree: int
    mode: int = 0
    input_dtype: DType = DType.FP16
    upstream_dtype: DType = DType.FP16
    output_dtype: DType = DType.FP16

    def validate(self, path: str = "rmsnorm_backward_workload") -> None:
        _positive((self.rows, self.hidden_size, self.tp_degree), path)
        if (self.mode != 0 or self.input_dtype is not DType.FP16
                or self.upstream_dtype is not DType.FP16
                or self.output_dtype is not DType.FP16
                or self.tensor_bytes > _MAX_SRAM_SPAN):
            raise SchemaError("RMSNorm backward profile/dtype/extent differs",
                              path=path)

    @property
    def tensor_bytes(self) -> int:
        return 2 * self.rows * self.hidden_size


@dataclass(frozen=True, slots=True)
class AttentionBackwardWorkload:
    tokens: int
    rank_heads: int
    rank_kv_heads: int
    head_dim: int
    tp_degree: int
    sequences: int
    pairs: int
    input_dtype: DType = DType.FP16
    upstream_dtype: DType = DType.FP16
    output_dtype: DType = DType.FP16

    def validate(self, path: str = "attention_backward_workload") -> None:
        _positive((self.tokens, self.rank_heads, self.rank_kv_heads,
                   self.head_dim, self.tp_degree, self.sequences, self.pairs),
                  path)
        per = (self.tokens // self.sequences
               if self.tokens % self.sequences == 0 else 0)
        if (not per or self.rank_kv_heads > self.rank_heads
                or self.rank_heads % self.rank_kv_heads
                or self.head_dim % 2
                or self.pairs != self.sequences * per * (per + 1) // 2
                or self.input_dtype is not DType.FP16
                or self.upstream_dtype is not DType.FP16
                or self.output_dtype is not DType.FP16
                or max(self.input_bytes, self.upstream_bytes) > _MAX_SRAM_SPAN):
            raise SchemaError("attention backward causal GQA profile differs",
                              path=path)

    @property
    def input_bytes(self) -> int:
        return 2 * self.tokens * (self.rank_heads + 2 * self.rank_kv_heads) * self.head_dim

    @property
    def upstream_bytes(self) -> int:
        return 2 * self.tokens * self.rank_heads * self.head_dim

    @property
    def output_bytes(self) -> int:
        return self.input_bytes


@dataclass(frozen=True, slots=True)
class RopeBackwardWorkload:
    logical_tokens: int
    rank_tokens: int
    logical_query_heads: int
    logical_kv_heads: int
    rank_query_heads: int
    rank_kv_heads: int
    tp_degree: int
    head_dim: int
    rotary_dim: int
    max_position_embeddings: int
    position_dtype: DType = DType.INT32
    upstream_dtype: DType = DType.FP16
    output_dtype: DType = DType.FP16

    def validate(self, path: str = "rope_backward_workload") -> None:
        _positive((self.logical_tokens, self.rank_tokens,
                   self.logical_query_heads, self.logical_kv_heads,
                   self.rank_query_heads, self.rank_kv_heads, self.tp_degree,
                   self.head_dim, self.rotary_dim,
                   self.max_position_embeddings), path)
        if (self.logical_tokens != self.rank_tokens
                or self.logical_query_heads != self.rank_query_heads * self.tp_degree
                or self.logical_kv_heads != self.rank_kv_heads * self.tp_degree
                or self.rank_query_heads < self.rank_kv_heads
                or self.rank_query_heads % self.rank_kv_heads
                or self.head_dim != self.rotary_dim or self.head_dim % 2
                or self.rank_tokens > self.max_position_embeddings
                or self.position_dtype is not DType.INT32
                or self.upstream_dtype is not DType.FP16
                or self.output_dtype is not DType.FP16
                or max(self.position_bytes, self.packed_bytes) > _MAX_SRAM_SPAN):
            raise SchemaError("RoPE backward position/GQA profile differs", path=path)

    @property
    def position_bytes(self) -> int:
        return 4 * self.rank_tokens

    @property
    def packed_bytes(self) -> int:
        return (2 * self.rank_tokens *
                (self.rank_query_heads + 2 * self.rank_kv_heads) * self.head_dim)

    @property
    def position_trace_tag(self) -> int:
        digest = 14695981039346656037
        for position in range(self.rank_tokens):
            for shift in range(0, 32, 8):
                digest ^= (position >> shift) & 0xFF
                digest = (digest * 1099511628211) & ((1 << 64) - 1)
        return digest & _MAX_PROFILE


@dataclass(frozen=True, slots=True)
class ResidualBackwardWorkload:
    logical_rows: int
    rank_rows: int
    tp_degree: int
    hidden_size: int
    forward_dtype: DType = DType.FP16
    upstream_dtype: DType = DType.FP16
    left_output_dtype: DType = DType.FP16
    right_output_dtype: DType = DType.FP16

    def validate(self, path: str = "residual_backward_workload") -> None:
        _positive((self.logical_rows, self.rank_rows, self.tp_degree,
                   self.hidden_size), path)
        if (self.logical_rows != self.rank_rows * self.tp_degree
                or any(dtype is not DType.FP16 for dtype in (
                    self.forward_dtype, self.upstream_dtype,
                    self.left_output_dtype, self.right_output_dtype))
                or self.tensor_bytes > _MAX_SRAM_SPAN):
            raise SchemaError("residual backward TP profile/dtype differs", path=path)

    @property
    def tensor_bytes(self) -> int:
        return 2 * self.rank_rows * self.hidden_size


@dataclass(frozen=True, slots=True)
class SwiGluBackwardWorkload:
    rows: int
    intermediate_size: int
    input_dtype: DType = DType.FP16
    upstream_dtype: DType = DType.FP16
    output_dtype: DType = DType.FP16

    def validate(self, path: str = "swiglu_backward_workload") -> None:
        _positive((self.rows, self.intermediate_size), path)
        if (self.input_dtype is not DType.FP16
                or self.upstream_dtype is not DType.FP16
                or self.output_dtype is not DType.FP16
                or max(self.input_bytes, self.upstream_bytes) > _MAX_SRAM_SPAN):
            raise SchemaError("SwiGLU backward profile/dtype differs", path=path)

    @property
    def input_bytes(self) -> int:
        return 4 * self.rows * self.intermediate_size

    @property
    def upstream_bytes(self) -> int:
        return 2 * self.rows * self.intermediate_size

    @property
    def output_bytes(self) -> int:
        return self.input_bytes

    @property
    def element_count(self) -> int:
        return self.rows * self.intermediate_size


__all__ = [
    "AttentionBackwardWorkload", "ResidualBackwardWorkload",
    "RmsNormBackwardWorkload", "RopeBackwardWorkload",
    "SwiGluBackwardWorkload",
]
