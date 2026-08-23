"""Exact typed operands for MoE Swizzle standard lowering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from math import prod
from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .swizzle import SwizzleActionKind
from .swizzle_moe_ir2 import (
    MoeSwizzleIr2PacketSlice,
    MoeSwizzleIr2Projection,
)
from .swizzle_plan import SwizzleValueUse


MOE_SWIZZLE_OPERAND_ABI_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_operand_abi/v1alpha1"
)


class MoeSwizzleDteDirection(str, Enum):
    REMOTE_SEND = "remote_send"
    REMOTE_RECV = "remote_recv"
    LOCAL_COPY = "local_copy"


@dataclass(frozen=True, slots=True)
class MoeSwizzleOperandView:
    task_ref: str
    ordinal: int
    use: SwizzleValueUse
    value_ref: str
    origin_ref: str
    slot: int
    shape: tuple[int, ...]
    layout: str
    dtype: DType
    byte_offset: int
    byte_extent: int

    def validate(self, path: str) -> None:
        for name in ("task_ref", "value_ref", "origin_ref", "layout"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        validate_uint64(self.ordinal, f"{path}.ordinal")
        validate_uint64(self.slot, f"{path}.slot")
        validate_uint64(self.byte_offset, f"{path}.byte_offset")
        validate_uint64(self.byte_extent, f"{path}.byte_extent")
        if type(self.use) is not SwizzleValueUse or type(self.dtype) is not DType:
            raise SchemaError("operand use/dtype must be typed", path=path)
        if self.slot not in (0, 1) or not self.shape or self.byte_extent == 0:
            raise SchemaError("operand requires slot 0/1 and nonempty typed extent", path=path)
        dtype_bytes = 2 if self.dtype is DType.FP16 else 4
        if self.byte_extent != prod(self.shape) * dtype_bytes:
            raise SchemaError("operand extent does not equal typed shape", path=f"{path}.byte_extent")


@dataclass(frozen=True, slots=True)
class MoeSwizzleMatmulContract:
    task_ref: str
    m: int
    n: int
    k: int
    dtype: DType
    accumulation_dtype: DType
    flops: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        for name in ("m", "n", "k", "flops"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if min(self.m, self.n, self.k) == 0 or self.flops != 2 * self.m * self.n * self.k:
            raise SchemaError("MATMUL FLOPs do not close m/n/k", path=path)
        if type(self.dtype) is not DType or type(self.accumulation_dtype) is not DType:
            raise SchemaError("MATMUL dtypes must be typed", path=path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleDteContract:
    task_ref: str
    direction: MoeSwizzleDteDirection
    flow_ref: str | None
    packet_ref: str | None
    stage: int | None
    route_ref: str | None
    die_path: tuple[int, ...]
    logical_bytes: int
    payload_bits: int
    assignment_slices: tuple[MoeSwizzleIr2PacketSlice, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        if type(self.direction) is not MoeSwizzleDteDirection:
            raise SchemaError("requires a typed DTE direction", path=f"{path}.direction")
        for name in ("flow_ref", "packet_ref", "route_ref"):
            value = getattr(self, name)
            if value is not None:
                validate_nonempty(value, f"{path}.{name}")
        if self.stage is not None:
            validate_uint64(self.stage, f"{path}.stage")
        validate_uint64(self.logical_bytes, f"{path}.logical_bytes")
        validate_uint64(self.payload_bits, f"{path}.payload_bits")
        if self.logical_bytes == 0 or self.payload_bits != 8 * self.logical_bytes:
            raise SchemaError("DTE payload bits do not close bytes", path=path)
        if self.direction is MoeSwizzleDteDirection.LOCAL_COPY:
            if any(value is not None for value in (self.flow_ref, self.packet_ref, self.stage, self.route_ref)) or self.die_path or self.assignment_slices:
                raise SchemaError("LOCAL_COPY cannot carry remote route provenance", path=path)
        elif self.flow_ref is None or self.packet_ref is None or self.stage is None or self.route_ref is None or len(self.die_path) < 2 or not self.assignment_slices:
            raise SchemaError("remote DTE requires exact flow/packet/stage/route/slices", path=path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleReduceContract:
    task_ref: str
    input_count: int
    element_count: int
    input_dtype: DType
    accumulation_dtype: DType
    output_dtype: DType

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        validate_uint64(self.input_count, f"{path}.input_count")
        validate_uint64(self.element_count, f"{path}.element_count")
        if self.input_count != 2 or self.element_count == 0:
            raise SchemaError("MoE reduce must be nonempty and binary", path=path)
        for name in ("input_dtype", "accumulation_dtype", "output_dtype"):
            if type(getattr(self, name)) is not DType:
                raise SchemaError("reduce dtypes must be typed", path=f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class MoeSwizzleSwiGluContract:
    task_ref: str
    element_count: int
    dtype: DType
    input_bytes: int
    output_bytes: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        for name in ("element_count", "input_bytes", "output_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        dtype_bytes = 2 if self.dtype is DType.FP16 else 4
        if (
            type(self.dtype) is not DType
            or self.element_count == 0
            or self.output_bytes != self.element_count * dtype_bytes
            or self.input_bytes != 2 * self.output_bytes
        ):
            raise SchemaError("SWIGLU bytes do not close typed flat 2N->N operands", path=path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleOperandABI:
    schema_version: str
    producer_pass: str
    id: str
    source_projection_id: str
    operands: tuple[MoeSwizzleOperandView, ...]
    matmuls: tuple[MoeSwizzleMatmulContract, ...]
    dtes: tuple[MoeSwizzleDteContract, ...]
    reductions: tuple[MoeSwizzleReduceContract, ...]
    swiglus: tuple[MoeSwizzleSwiGluContract, ...] = ()

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleOperandABI":
        semantic.setdefault("swiglus", ())
        result = cls(
            MOE_SWIZZLE_OPERAND_ABI_SCHEMA_VERSION,
            "moe_swizzle_operand_abi_builder",
            stable_artifact_id(
                "moe_swizzle_operand_abi",
                semantic,
                schema_version=MOE_SWIZZLE_OPERAND_ABI_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_projection_id", "operands", "matmuls", "dtes",
                "reductions", "swiglus",
            )
        }

    def validate(self, path: str = "moe_swizzle_operand_abi") -> None:
        if self.schema_version != MOE_SWIZZLE_OPERAND_ABI_SCHEMA_VERSION or self.producer_pass != "moe_swizzle_operand_abi_builder":
            raise SchemaError("unsupported operand ABI schema/producer", path=path)
        validate_nonempty(self.source_projection_id, f"{path}.source_projection_id")
        for field_name in ("operands", "matmuls", "dtes", "reductions", "swiglus"):
            values = getattr(self, field_name)
            for index, value in enumerate(values):
                value.validate(f"{path}.{field_name}[{index}]")
        keys = [(item.task_ref, item.ordinal) for item in self.operands]
        if len(keys) != len(set(keys)):
            raise SchemaError("operand ordinals must be unique per task", path=f"{path}.operands")
        for field_name in ("matmuls", "dtes", "reductions", "swiglus"):
            refs = [item.task_ref for item in getattr(self, field_name)]
            if len(refs) != len(set(refs)):
                raise SchemaError("task contracts must be unique", path=f"{path}.{field_name}")
        expected = stable_artifact_id("moe_swizzle_operand_abi", self._semantic(), schema_version=MOE_SWIZZLE_OPERAND_ABI_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError("unstable operand ABI id", path=f"{path}.id")

    def validate_against(self, projection: MoeSwizzleIr2Projection, path: str = "moe_swizzle_operand_abi") -> None:
        self.validate(path)
        expected = build_moe_swizzle_operand_abi(projection)
        if self._semantic() != expected._semantic():
            raise SchemaError("operand ABI is not exact projection quotient", path=path)


def _slot(projection: MoeSwizzleIr2Projection, task: object, value: object) -> int:
    if value.buffer_ref is None:
        return 0
    return next(item.slot for item in task.buffer_uses if item.buffer_ref == value.buffer_ref)


def build_moe_swizzle_operand_abi(projection: MoeSwizzleIr2Projection) -> MoeSwizzleOperandABI:
    projection.validate("projection")
    values = {item.id: item for item in projection.values}
    flows = {item.id: item for item in projection.flows}
    operands = []
    matmuls = []
    dtes = []
    reductions = []
    swiglus = []
    for task in projection.tasks:
        refs = task.read_value_refs + task.write_value_refs
        uses = (SwizzleValueUse.READ,) * len(task.read_value_refs) + (SwizzleValueUse.WRITE,) * len(task.write_value_refs)
        for ordinal, (ref, use) in enumerate(zip(refs, uses)):
            value = values[ref]
            operands.append(MoeSwizzleOperandView(
                task.id, ordinal, use, ref, value.origin_ref,
                _slot(projection, task, value), value.shape, value.layout,
                value.dtype, value.byte_offset, value.size_bytes,
            ))
        if task.kind is SwizzleActionKind.COMP:
            assert task.matmul_m is not None and task.matmul_n is not None and task.matmul_k is not None
            assert task.dtype is not None and task.accumulation_dtype is not None
            matmuls.append(MoeSwizzleMatmulContract(
                task.id, task.matmul_m, task.matmul_n, task.matmul_k,
                task.dtype, task.accumulation_dtype, task.flops,
            ))
        elif task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
            flow = flows[task.flow_ref]
            dtes.append(MoeSwizzleDteContract(
                task.id,
                MoeSwizzleDteDirection.REMOTE_SEND if task.kind is SwizzleActionKind.SEND else MoeSwizzleDteDirection.REMOTE_RECV,
                flow.id, flow.packet_ref, flow.stage, flow.route_ref, flow.die_path,
                flow.logical_bytes, flow.logical_bytes * 8, flow.assignment_slices,
            ))
        elif task.kind is SwizzleActionKind.LOCAL_COPY:
            dtes.append(MoeSwizzleDteContract(
                task.id, MoeSwizzleDteDirection.LOCAL_COPY, None, None, None,
                None, (), task.logical_bytes, task.logical_bytes * 8, (),
            ))
        elif task.kind is SwizzleActionKind.SWIGLU:
            if len(task.read_value_refs) != 1 or len(task.write_value_refs) != 1:
                raise SchemaError("SWIGLU requires one flat input and output", path="projection.tasks")
            input_value = values[task.read_value_refs[0]]
            output_value = values[task.write_value_refs[0]]
            if input_value.dtype is not output_value.dtype:
                raise SchemaError("SWIGLU input/output dtype drifted", path="projection.tasks")
            dtype_bytes = 2 if output_value.dtype is DType.FP16 else 4
            swiglus.append(MoeSwizzleSwiGluContract(
                task.id, output_value.size_bytes // dtype_bytes,
                output_value.dtype, input_value.size_bytes, output_value.size_bytes,
            ))
        elif task.kind is SwizzleActionKind.REDUCE:
            first, accumulator = (values[ref] for ref in task.read_value_refs)
            output = values[task.write_value_refs[0]]
            if first.shape != accumulator.shape or first.shape != output.shape or first.dtype != accumulator.dtype or first.dtype != output.dtype:
                raise SchemaError("binary REDUCE typed operands do not match", path="projection.tasks")
            reductions.append(MoeSwizzleReduceContract(
                task.id, 2, first.size_bytes // (2 if first.dtype is DType.FP16 else 4),
                first.dtype, DType.FP32, output.dtype,
            ))
    result = MoeSwizzleOperandABI.create(
        source_projection_id=projection.id,
        operands=tuple(operands),
        matmuls=tuple(matmuls),
        dtes=tuple(dtes),
        reductions=tuple(reductions),
        swiglus=tuple(swiglus),
    )
    return result


__all__ = [name for name in globals() if name.startswith("MoeSwizzle") or name.startswith("MOE_SWIZZLE") or name == "build_moe_swizzle_operand_abi"]
