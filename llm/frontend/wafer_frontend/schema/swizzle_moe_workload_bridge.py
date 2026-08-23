"""Typed semantic-to-physical value bridge for whole-workload MoE Swizzle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .swizzle_moe_execution import MoeScaleExecutionTerminalKind


MOE_SWIZZLE_WORKLOAD_VALUE_BRIDGE_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_workload_value_bridge/v1alpha1"
)


def _refs(values: tuple[str, ...], path: str, *, nonempty: bool = False) -> None:
    if type(values) is not tuple or (nonempty and not values) or len(values) != len(set(values)):
        raise SchemaError("refs must be an immutable unique tuple", path=path)
    for index, value in enumerate(values):
        validate_nonempty(value, f"{path}[{index}]")


class MoeSwizzleWorkloadValueKind(str, Enum):
    WEIGHT_STAGING = "weight_staging"
    GEMM_OUTPUT = "gemm_output"
    SWIGLU_OUTPUT = "swiglu_output"
    COMBINED_TERMINAL = "combined_terminal"


class MoeSwizzleWorkloadPhysicalUse(str, Enum):
    IR2_READ = "ir2_read"
    IR2_WRITE = "ir2_write"


@dataclass(frozen=True, slots=True)
class MoeSwizzleWorkloadPhysicalValueSlice:
    physical_value_ref: str
    physical_task_ref: str
    anchor_execution_action_ref: str
    use: MoeSwizzleWorkloadPhysicalUse
    semantic_origin: tuple[int, ...]
    physical_byte_offset: int
    size_bytes: int
    shape: tuple[int, ...]
    dtype: DType
    ir2_terminal_ref: str | None = None
    assignment_ref: str | None = None

    def validate(self, path: str) -> None:
        for name in (
            "physical_value_ref", "physical_task_ref", "anchor_execution_action_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.use) is not MoeSwizzleWorkloadPhysicalUse or type(self.dtype) is not DType:
            raise SchemaError("physical use/dtype is not typed", path=path)
        if not self.shape or len(self.semantic_origin) != len(self.shape):
            raise SchemaError("slice shape/origin rank is not exact", path=path)
        for name, values in (
            ("semantic_origin", self.semantic_origin), ("shape", self.shape),
        ):
            for index, value in enumerate(values):
                validate_uint64(value, f"{path}.{name}[{index}]")
                if name == "shape" and value == 0:
                    raise SchemaError("slice extent must be positive", path=f"{path}.{name}[{index}]")
        validate_uint64(self.physical_byte_offset, f"{path}.physical_byte_offset")
        validate_uint64(self.size_bytes, f"{path}.size_bytes")
        dtype_bytes = 2 if self.dtype is DType.FP16 else 4
        if self.size_bytes == 0 or self.size_bytes != prod(self.shape) * dtype_bytes:
            raise SchemaError("slice bytes disagree with shape/dtype", path=path)
        for name in ("ir2_terminal_ref", "assignment_ref"):
            value = getattr(self, name)
            if value is not None:
                validate_nonempty(value, f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class MoeSwizzleWorkloadValueBinding:
    semantic_value_ref: str
    kind: MoeSwizzleWorkloadValueKind
    producer_action_refs: tuple[str, ...]
    consumer_action_refs: tuple[str, ...]
    terminal_kind: MoeScaleExecutionTerminalKind | None
    physical_slices: tuple[MoeSwizzleWorkloadPhysicalValueSlice, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.semantic_value_ref, f"{path}.semantic_value_ref")
        if type(self.kind) is not MoeSwizzleWorkloadValueKind:
            raise SchemaError("value kind is not typed", path=f"{path}.kind")
        _refs(self.producer_action_refs, f"{path}.producer_action_refs")
        _refs(self.consumer_action_refs, f"{path}.consumer_action_refs")
        if self.terminal_kind is not None and type(self.terminal_kind) is not MoeScaleExecutionTerminalKind:
            raise SchemaError("terminal kind is not typed", path=f"{path}.terminal_kind")
        if self.kind is MoeSwizzleWorkloadValueKind.COMBINED_TERMINAL:
            if self.terminal_kind is not MoeScaleExecutionTerminalKind.COMBINED:
                raise SchemaError("combined binding requires combined terminal provenance", path=path)
        elif self.terminal_kind not in (None, MoeScaleExecutionTerminalKind.COMBINED):
            raise SchemaError("only combined IR2 terminals may be bridged", path=path)
        if type(self.physical_slices) is not tuple or not self.physical_slices:
            raise SchemaError("binding requires physical slices", path=f"{path}.physical_slices")
        for index, item in enumerate(self.physical_slices):
            if type(item) is not MoeSwizzleWorkloadPhysicalValueSlice:
                raise SchemaError("requires exact physical slice", path=f"{path}.physical_slices[{index}]")
            item.validate(f"{path}.physical_slices[{index}]")
        keys = tuple(
            (
                item.physical_value_ref, item.physical_task_ref, item.use,
                item.semantic_origin, item.physical_byte_offset, item.shape,
            )
            for item in self.physical_slices
        )
        if len(keys) != len(set(keys)):
            raise SchemaError("binding contains duplicate physical slices", path=f"{path}.physical_slices")


@dataclass(frozen=True, slots=True)
class MoeSwizzleWorkloadValueBridge:
    schema_version: str
    producer_pass: str
    id: str
    source_execution_id: str
    source_overlay_id: str
    source_workload_projection_id: str
    source_replacement_projection_id: str
    bindings: tuple[MoeSwizzleWorkloadValueBinding, ...]
    preserved_terminal_refs: tuple[str, ...]

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleWorkloadValueBridge":
        result = cls(
            MOE_SWIZZLE_WORKLOAD_VALUE_BRIDGE_SCHEMA_VERSION,
            "build_moe_swizzle_workload_value_bridge",
            stable_artifact_id(
                "moe_swizzle_workload_value_bridge",
                semantic,
                schema_version=MOE_SWIZZLE_WORKLOAD_VALUE_BRIDGE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_execution_id", "source_overlay_id",
                "source_workload_projection_id", "source_replacement_projection_id",
                "bindings", "preserved_terminal_refs",
            )
        }

    def validate(self, path: str = "moe_swizzle_workload_value_bridge") -> None:
        if (
            self.schema_version != MOE_SWIZZLE_WORKLOAD_VALUE_BRIDGE_SCHEMA_VERSION
            or self.producer_pass != "build_moe_swizzle_workload_value_bridge"
        ):
            raise SchemaError("unsupported workload value bridge schema/producer", path=path)
        for name in (
            "source_execution_id", "source_overlay_id",
            "source_workload_projection_id", "source_replacement_projection_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.bindings) is not tuple or not self.bindings:
            raise SchemaError("workload value bindings must be nonempty", path=f"{path}.bindings")
        semantic_refs = set()
        for index, item in enumerate(self.bindings):
            if type(item) is not MoeSwizzleWorkloadValueBinding:
                raise SchemaError("requires exact workload value binding", path=f"{path}.bindings[{index}]")
            item.validate(f"{path}.bindings[{index}]")
            if item.semantic_value_ref in semantic_refs:
                raise SchemaError("semantic value is bound more than once", path=f"{path}.bindings")
            semantic_refs.add(item.semantic_value_ref)
        _refs(self.preserved_terminal_refs, f"{path}.preserved_terminal_refs")
        if semantic_refs.intersection(self.preserved_terminal_refs):
            raise SchemaError("physical and preserved terminal sets overlap", path=path)
        expected = stable_artifact_id(
            "moe_swizzle_workload_value_bridge",
            self._semantic(),
            schema_version=MOE_SWIZZLE_WORKLOAD_VALUE_BRIDGE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable workload value bridge id", path=f"{path}.id")


__all__ = [
    name for name in globals()
    if name.startswith("MoeSwizzle") or name.startswith("MOE_SWIZZLE")
]
