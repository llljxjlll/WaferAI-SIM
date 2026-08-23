"""Physical-root and lifecycle quotient for one whole MoE Swizzle workload."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .global_action import LogicalCoreRef
from .ir2 import BufferOwnership


MOE_SWIZZLE_WORKLOAD_ABI_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_workload_abi/v1alpha1"
_FAMILIES = {
    "dispatch_operand", "combine_output", "boundary_input",
    "state_stage.gate", "state_stage.up", "state_stage.down",
    "swiglu_input", "swiglu_output", "terminal_combined", "terminal_tape",
}


@dataclass(frozen=True, slots=True)
class MoeSwizzleWorkloadRootValuePlacement:
    value_ref: str
    storage_unit_ref: str
    root_offset_bytes: int
    extent_bytes: int
    dtype: DType
    layout: str
    lifetime_start: int
    lifetime_end_exclusive: int
    action_refs: tuple[str, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.value_ref, f"{path}.value_ref")
        validate_nonempty(self.storage_unit_ref, f"{path}.storage_unit_ref")
        validate_nonempty(self.layout, f"{path}.layout")
        for name in (
            "root_offset_bytes", "extent_bytes", "lifetime_start",
            "lifetime_end_exclusive",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if type(self.dtype) is not DType or self.extent_bytes == 0:
            raise SchemaError("value placement dtype/extent is invalid", path=path)
        if self.lifetime_start >= self.lifetime_end_exclusive:
            raise SchemaError("value placement lifetime is empty", path=path)
        if (
            type(self.action_refs) is not tuple or not self.action_refs
            or len(self.action_refs) != len(set(self.action_refs))
        ):
            raise SchemaError("value placement action refs must be nonempty/unique", path=f"{path}.action_refs")
        for index, ref in enumerate(self.action_refs):
            validate_nonempty(ref, f"{path}.action_refs[{index}]")


@dataclass(frozen=True, slots=True)
class MoeSwizzleWorkloadStorageRoot:
    logical_core: LogicalCoreRef
    runtime_core_id: int
    family: str
    slot: int
    extent_bytes: int
    ownership: BufferOwnership
    lifetime_start: int
    lifetime_end_exclusive: int
    allocate: bool
    free: bool
    value_placements: tuple[MoeSwizzleWorkloadRootValuePlacement, ...]
    action_refs: tuple[str, ...]

    def validate(self, path: str) -> None:
        self.logical_core.validate(f"{path}.logical_core")
        for name in (
            "runtime_core_id", "slot", "extent_bytes", "lifetime_start",
            "lifetime_end_exclusive",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.family not in _FAMILIES or self.slot not in (0, 1) or self.extent_bytes == 0:
            raise SchemaError("root family/slot/extent is invalid", path=path)
        if type(self.ownership) is not BufferOwnership:
            raise SchemaError("root ownership is not typed", path=f"{path}.ownership")
        if self.lifetime_start >= self.lifetime_end_exclusive:
            raise SchemaError("root lifetime is empty", path=path)
        if type(self.allocate) is not bool or type(self.free) is not bool:
            raise SchemaError("root lifecycle flags are not bool", path=path)
        terminal = self.family.startswith("terminal_")
        external = self.family == "boundary_input"
        if external:
            if self.ownership is not BufferOwnership.BORROWED or self.allocate or self.free:
                raise SchemaError("external boundary root must be borrowed", path=path)
        elif self.ownership is not BufferOwnership.OWNED or not self.allocate or self.free == terminal:
            raise SchemaError("owned transient/terminal lifecycle is invalid", path=path)
        if type(self.value_placements) is not tuple or not self.value_placements:
            raise SchemaError("root requires typed value placements", path=f"{path}.value_placements")
        keys = []
        for index, placement in enumerate(self.value_placements):
            if type(placement) is not MoeSwizzleWorkloadRootValuePlacement:
                raise SchemaError("requires exact root value placement", path=f"{path}.value_placements[{index}]")
            placement.validate(f"{path}.value_placements[{index}]")
            if placement.root_offset_bytes + placement.extent_bytes > self.extent_bytes:
                raise SchemaError("value placement escapes its root", path=f"{path}.value_placements[{index}]")
            if (
                placement.lifetime_start < self.lifetime_start
                or placement.lifetime_end_exclusive > self.lifetime_end_exclusive
                or not set(placement.action_refs).issubset(self.action_refs)
            ):
                raise SchemaError("value placement lifetime/actions escape its root", path=f"{path}.value_placements[{index}]")
            keys.append((placement.root_offset_bytes, placement.value_ref))
        if keys != sorted(keys) or len({item.value_ref for item in self.value_placements}) != len(self.value_placements):
            raise SchemaError("root value placements must be canonical/unique", path=f"{path}.value_placements")
        for index, left in enumerate(self.value_placements):
            for right in self.value_placements[index + 1:]:
                byte_overlap = (
                    left.root_offset_bytes < right.root_offset_bytes + right.extent_bytes
                    and right.root_offset_bytes < left.root_offset_bytes + left.extent_bytes
                )
                lifetime_overlap = (
                    left.lifetime_start < right.lifetime_end_exclusive
                    and right.lifetime_start < left.lifetime_end_exclusive
                )
                if byte_overlap and lifetime_overlap:
                    if left.storage_unit_ref != right.storage_unit_ref:
                        raise SchemaError(
                            "unrelated live values overlap in one physical root: "
                            f"left={(left.value_ref, left.storage_unit_ref, left.root_offset_bytes, left.extent_bytes, left.lifetime_start, left.lifetime_end_exclusive)!r}, "
                            f"right={(right.value_ref, right.storage_unit_ref, right.root_offset_bytes, right.extent_bytes, right.lifetime_start, right.lifetime_end_exclusive)!r}",
                            path=f"{path}.value_placements",
                        )
                    left_end = left.root_offset_bytes + left.extent_bytes
                    right_end = right.root_offset_bytes + right.extent_bytes
                    contained = (
                        left.root_offset_bytes <= right.root_offset_bytes
                        and right_end <= left_end
                    ) or (
                        right.root_offset_bytes <= left.root_offset_bytes
                        and left_end <= right_end
                    )
                    if (
                        not contained or left.dtype is not right.dtype
                        or left.layout != right.layout
                    ):
                        raise SchemaError(
                            "same-storage live aliases require contained spans and matching dtype/layout",
                            path=f"{path}.value_placements",
                        )
        if type(self.action_refs) is not tuple or not self.action_refs or len(self.action_refs) != len(set(self.action_refs)):
            raise SchemaError("root action refs must be nonempty/unique", path=f"{path}.action_refs")
        for index, ref in enumerate(self.action_refs):
            validate_nonempty(ref, f"{path}.action_refs[{index}]")

    @property
    def value_refs(self) -> tuple[str, ...]:
        return tuple(item.value_ref for item in self.value_placements)


@dataclass(frozen=True, slots=True)
class MoeSwizzleWorkloadABI:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    source_workload_projection_id: str
    source_replacement_projection_id: str
    source_state_abi_id: str
    source_value_bridge_id: str
    roots: tuple[MoeSwizzleWorkloadStorageRoot, ...]

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleWorkloadABI":
        result = cls(
            MOE_SWIZZLE_WORKLOAD_ABI_SCHEMA_VERSION,
            "build_moe_swizzle_workload_abi",
            stable_artifact_id(
                "moe_swizzle_workload_abi", semantic,
                schema_version=MOE_SWIZZLE_WORKLOAD_ABI_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_ir1_id", "source_workload_projection_id",
                "source_replacement_projection_id", "source_state_abi_id", "source_value_bridge_id", "roots",
            )
        }

    def validate(self, path: str = "moe_swizzle_workload_abi") -> None:
        if (
            self.schema_version != MOE_SWIZZLE_WORKLOAD_ABI_SCHEMA_VERSION
            or self.producer_pass != "build_moe_swizzle_workload_abi"
        ):
            raise SchemaError("unsupported whole-workload ABI schema/producer", path=path)
        for name in (
            "source_ir1_id", "source_workload_projection_id",
            "source_replacement_projection_id", "source_state_abi_id", "source_value_bridge_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.roots) is not tuple or not self.roots:
            raise SchemaError("whole ABI roots must be nonempty", path=f"{path}.roots")
        keys = []
        for index, root in enumerate(self.roots):
            if type(root) is not MoeSwizzleWorkloadStorageRoot:
                raise SchemaError("requires exact workload root", path=f"{path}.roots[{index}]")
            root.validate(f"{path}.roots[{index}]")
            keys.append((root.runtime_core_id, root.family, root.slot))
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise SchemaError("whole roots must be canonical/unique", path=f"{path}.roots")
        if self.id != stable_artifact_id(
            "moe_swizzle_workload_abi", self._semantic(),
            schema_version=MOE_SWIZZLE_WORKLOAD_ABI_SCHEMA_VERSION,
        ):
            raise SchemaError("unstable whole-workload ABI id", path=f"{path}.id")

    @property
    def alloc_count(self) -> int:
        return sum(item.allocate for item in self.roots)

    @property
    def free_count(self) -> int:
        return sum(item.free for item in self.roots)


__all__ = [name for name in globals() if name.startswith("MoeSwizzle") or name.startswith("MOE_SWIZZLE")]
