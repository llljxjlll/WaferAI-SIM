"""Typed operand contracts and opcode quotient for the UNFUSED comparison."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .artifact_manifest import RecordOpcode
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .ir1 import IR1
from .swizzle import SwizzleActionKind
from .swizzle_unfused import (
    UnfusedComparisonOperand,
    UnfusedComparisonPlan,
    UnfusedComparisonProjection,
)


UNFUSED_COMPARISON_OPERAND_ABI_SCHEMA_VERSION = (
    "wafer_frontend.unfused_comparison_operand_abi/v1alpha1"
)
UNFUSED_COMPARISON_LOWERED_SCHEMA_VERSION = (
    "wafer_frontend.unfused_comparison_lowered/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class UnfusedComparisonMatmulContract:
    task_ref: str
    m: int
    k: int
    n: int
    dtype: DType

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        for name in ("m", "k", "n"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
            if getattr(self, name) == 0:
                raise SchemaError("geometry must be positive", path=f"{path}.{name}")
        if type(self.dtype) is not DType:
            raise SchemaError("requires typed dtype", path=f"{path}.dtype")


@dataclass(frozen=True, slots=True)
class UnfusedComparisonDteContract:
    task_ref: str
    logical_bytes: int
    payload_bits: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        validate_uint64(self.logical_bytes, f"{path}.logical_bytes")
        validate_uint64(self.payload_bits, f"{path}.payload_bits")
        if self.logical_bytes == 0 or self.payload_bits not in (16, 32):
            raise SchemaError("DTE contract requires exact positive typed payload", path=path)


@dataclass(frozen=True, slots=True)
class UnfusedComparisonReduceContract:
    task_ref: str
    input_count: int
    element_count: int
    input_stride_bytes: int
    dtype: DType
    accumulation_dtype: DType
    output_alias_value_ref: str

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        validate_nonempty(self.output_alias_value_ref, f"{path}.output_alias_value_ref")
        for name in ("input_count", "element_count", "input_stride_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
            if getattr(self, name) == 0:
                raise SchemaError("reduce literal must be positive", path=f"{path}.{name}")
        if (
            self.input_count != 2
            or type(self.dtype) is not DType
            or self.accumulation_dtype is not DType.FP32
        ):
            raise SchemaError(
                "V1 requires exact two-input typed FP32 accumulation", path=path
            )


@dataclass(frozen=True, slots=True)
class UnfusedComparisonOperandABI:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    source_plan_ref: str
    source_projection_ref: str
    operands: tuple[UnfusedComparisonOperand, ...]
    matmul_contracts: tuple[UnfusedComparisonMatmulContract, ...]
    dte_contracts: tuple[UnfusedComparisonDteContract, ...]
    reduce_contracts: tuple[UnfusedComparisonReduceContract, ...]

    @classmethod
    def create(cls, **semantic: object) -> "UnfusedComparisonOperandABI":
        result = cls(
            schema_version=UNFUSED_COMPARISON_OPERAND_ABI_SCHEMA_VERSION,
            producer_pass="unfused_comparison_operand_abi_builder",
            id=stable_artifact_id(
                "unfused_comparison_operand_abi", semantic,
                schema_version=UNFUSED_COMPARISON_OPERAND_ABI_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "source_ir1_id", "source_plan_ref", "source_projection_ref",
            "operands", "matmul_contracts", "dte_contracts", "reduce_contracts",
        )}

    def validate(self, path: str = "unfused_comparison_operand_abi") -> None:
        if self.schema_version != UNFUSED_COMPARISON_OPERAND_ABI_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "unfused_comparison_operand_abi_builder":
            raise SchemaError("requires exact producer", path=f"{path}.producer_pass")
        for name in ("source_ir1_id", "source_plan_ref", "source_projection_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for field_name in ("operands", "matmul_contracts", "dte_contracts", "reduce_contracts"):
            for index, item in enumerate(getattr(self, field_name)):
                item.validate(f"{path}.{field_name}[{index}]")
        expected = stable_artifact_id(
            "unfused_comparison_operand_abi", self._semantic_key(),
            schema_version=UNFUSED_COMPARISON_OPERAND_ABI_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_against(
        self, ir1: IR1, plan: UnfusedComparisonPlan,
        projection: UnfusedComparisonProjection,
        path: str = "unfused_comparison_operand_abi",
    ) -> None:
        self.validate(path)
        projection.validate_against(ir1, plan, f"{path}.projection")
        from ..lowering.swizzle_unfused import build_unfused_comparison_operand_abi
        expected = build_unfused_comparison_operand_abi(ir1, plan, projection)
        if self != expected:
            raise SchemaError("operand ABI is not the exact producer result", path=path)


@dataclass(frozen=True, slots=True)
class UnfusedComparisonLoweredTask:
    task_ref: str
    opcodes: tuple[RecordOpcode, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        if not self.opcodes or any(type(item) is not RecordOpcode for item in self.opcodes):
            raise SchemaError("requires nonempty typed opcode quotient", path=f"{path}.opcodes")


@dataclass(frozen=True, slots=True)
class UnfusedComparisonLoweredProgram:
    schema_version: str
    producer_pass: str
    id: str
    source_plan_ref: str
    source_projection_ref: str
    tasks: tuple[UnfusedComparisonLoweredTask, ...]

    @classmethod
    def create(cls, **semantic: object) -> "UnfusedComparisonLoweredProgram":
        result = cls(
            schema_version=UNFUSED_COMPARISON_LOWERED_SCHEMA_VERSION,
            producer_pass="unfused_comparison_opcode_lowering",
            id=stable_artifact_id(
                "unfused_comparison_lowered", semantic,
                schema_version=UNFUSED_COMPARISON_LOWERED_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "source_plan_ref", "source_projection_ref", "tasks",
        )}

    def validate(self, path: str = "unfused_comparison_lowered") -> None:
        if self.schema_version != UNFUSED_COMPARISON_LOWERED_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "unfused_comparison_opcode_lowering":
            raise SchemaError("requires exact producer", path=f"{path}.producer_pass")
        for name in ("source_plan_ref", "source_projection_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        refs = tuple(item.task_ref for item in self.tasks)
        if len(refs) != len(set(refs)):
            raise SchemaError("contains duplicate task quotients", path=f"{path}.tasks")
        for index, item in enumerate(self.tasks):
            item.validate(f"{path}.tasks[{index}]")
        expected = stable_artifact_id(
            "unfused_comparison_lowered", self._semantic_key(),
            schema_version=UNFUSED_COMPARISON_LOWERED_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_against(
        self, plan: UnfusedComparisonPlan,
        projection: UnfusedComparisonProjection,
        path: str = "unfused_comparison_lowered",
    ) -> None:
        self.validate(path)
        from ..lowering.swizzle_unfused import lower_unfused_comparison_opcodes
        expected = lower_unfused_comparison_opcodes(plan, projection)
        if self != expected:
            raise SchemaError("opcode lowering is not exact", path=path)


__all__ = [name for name in globals() if name.startswith("Unfused") or name.startswith("UNFUSED_")]
