"""Typed persistent-weight carrier for the whole-workload MoE Swizzle leaf."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64


MOE_SWIZZLE_STATE_ABI_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_state_abi/v1alpha1"
_ROLES = ("gate", "up", "down")


def parse_moe_expert_weight_tensor_ref(tensor_ref: str) -> tuple[int, str]:
    """Parse the one admitted S3M4 expert-weight identity grammar exactly."""

    validate_nonempty(tensor_ref, "moe_expert_weight_tensor_ref")
    prefix = "S3M4.expert"
    if not tensor_ref.startswith(prefix):
        raise SchemaError("invalid MoE expert-weight tensor prefix", path="moe_expert_weight_tensor_ref")
    suffix = tensor_ref[len(prefix):]
    parts = suffix.split(".")
    if len(parts) != 3 or parts[0] not in ("0", "1", "2", "3") or parts[1] != "weight" or parts[2] not in _ROLES:
        raise SchemaError("invalid MoE expert-weight tensor identity", path="moe_expert_weight_tensor_ref")
    return int(parts[0]), parts[2]


@dataclass(frozen=True, slots=True)
class MoeSwizzleStateTensor:
    expert_index: int
    role: str
    tensor_ref: str
    state_ref: str
    hbm_binding_ref: str
    home_die_id: int
    hbm_address: int
    size_bytes: int
    shape: tuple[int, int]
    dtype: DType
    layout: str
    address_space_base: int
    address_space_size: int
    alignment_bytes: int
    state_access_refs: tuple[str, ...]

    def validate(self, path: str) -> None:
        for name in ("expert_index", "home_die_id", "hbm_address", "size_bytes", "address_space_base", "address_space_size", "alignment_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.expert_index > 3 or self.home_die_id != self.expert_index:
            raise SchemaError("expert weight must reside on its exact home die", path=path)
        if self.role not in _ROLES or parse_moe_expert_weight_tensor_ref(self.tensor_ref) != (self.expert_index, self.role):
            raise SchemaError("expert/role does not match tensor identity", path=path)
        for name in ("state_ref", "hbm_binding_ref", "layout"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if self.dtype is not DType.FP16 or self.layout != "KN_expert_local":
            raise SchemaError("expert weight dtype/layout is not exact", path=path)
        if len(self.shape) != 2 or any(type(item) is not int or item <= 0 for item in self.shape):
            raise SchemaError("expert weight shape must be a positive matrix", path=f"{path}.shape")
        if self.size_bytes != 2 * self.shape[0] * self.shape[1]:
            raise SchemaError("expert weight bytes disagree with shape/dtype", path=f"{path}.size_bytes")
        if self.alignment_bytes == 0 or self.alignment_bytes & (self.alignment_bytes - 1):
            raise SchemaError("HBM alignment must be a positive power of two", path=f"{path}.alignment_bytes")
        if self.hbm_address % self.alignment_bytes:
            raise SchemaError("expert weight HBM address is unaligned", path=f"{path}.hbm_address")
        if (
            self.address_space_size == 0
            or self.hbm_address < self.address_space_base
            or self.hbm_address + self.size_bytes > self.address_space_base + self.address_space_size
        ):
            raise SchemaError("expert weight escapes its home HBM address space", path=f"{path}.hbm_address")
        if type(self.state_access_refs) is not tuple or not self.state_access_refs:
            raise SchemaError("expert weight requires typed READ access coverage", path=f"{path}.state_access_refs")
        if len(self.state_access_refs) != len(set(self.state_access_refs)):
            raise SchemaError("expert weight state accesses are duplicated", path=f"{path}.state_access_refs")
        for index, ref in enumerate(self.state_access_refs):
            validate_nonempty(ref, f"{path}.state_access_refs[{index}]")


@dataclass(frozen=True, slots=True)
class MoeSwizzleStateActionBinding:
    execution_action_ref: str
    token_index: int
    expert_index: int
    role: str
    staging_value_ref: str
    state_ref: str
    hbm_binding_ref: str
    bytes: int

    def validate(self, path: str) -> None:
        for name in ("execution_action_ref", "staging_value_ref", "state_ref", "hbm_binding_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in ("token_index", "expert_index", "bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.expert_index > 3 or self.role not in _ROLES or self.bytes == 0:
            raise SchemaError("state action binding semantics are invalid", path=path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleWorkloadStateABI:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    source_manifest_id: str
    source_execution_id: str
    hidden_size: int
    intermediate_size: int
    tensors: tuple[MoeSwizzleStateTensor, ...]
    action_bindings: tuple[MoeSwizzleStateActionBinding, ...]

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleWorkloadStateABI":
        result = cls(
            MOE_SWIZZLE_STATE_ABI_SCHEMA_VERSION,
            "build_moe_swizzle_workload_state_abi",
            stable_artifact_id(
                "moe_swizzle_workload_state_abi",
                semantic,
                schema_version=MOE_SWIZZLE_STATE_ABI_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_ir1_id", "source_manifest_id", "source_execution_id",
                "hidden_size", "intermediate_size", "tensors", "action_bindings",
            )
        }

    def validate(self, path: str = "moe_swizzle_workload_state_abi") -> None:
        if self.schema_version != MOE_SWIZZLE_STATE_ABI_SCHEMA_VERSION or self.producer_pass != "build_moe_swizzle_workload_state_abi":
            raise SchemaError("unsupported MoE workload StateABI schema/producer", path=path)
        for name in ("source_ir1_id", "source_manifest_id", "source_execution_id"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in ("hidden_size", "intermediate_size"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
            if getattr(self, name) == 0:
                raise SchemaError("model extent must be positive", path=f"{path}.{name}")
        if type(self.tensors) is not tuple or len(self.tensors) != 12:
            raise SchemaError("StateABI requires exactly 12 expert-role tensors", path=f"{path}.tensors")
        tensor_by_key = {}
        state_refs = set()
        binding_refs = set()
        ranges_by_die: dict[int, list[tuple[int, int]]] = {}
        for index, item in enumerate(self.tensors):
            if type(item) is not MoeSwizzleStateTensor:
                raise SchemaError("requires exact MoeSwizzleStateTensor", path=f"{path}.tensors[{index}]")
            item.validate(f"{path}.tensors[{index}]")
            key = (item.expert_index, item.role)
            if key in tensor_by_key or item.state_ref in state_refs or item.hbm_binding_ref in binding_refs:
                raise SchemaError("StateABI tensor identity is duplicated", path=f"{path}.tensors")
            expected_shape = (
                (self.intermediate_size, self.hidden_size)
                if item.role == "down"
                else (self.hidden_size, self.intermediate_size)
            )
            if item.shape != expected_shape:
                raise SchemaError("expert-role weight shape is not exact", path=f"{path}.tensors[{index}].shape")
            for start, end in ranges_by_die.setdefault(item.home_die_id, []):
                if item.hbm_address < end and start < item.hbm_address + item.size_bytes:
                    raise SchemaError("expert weight HBM bindings overlap", path=f"{path}.tensors")
            ranges_by_die[item.home_die_id].append((item.hbm_address, item.hbm_address + item.size_bytes))
            tensor_by_key[key] = item
            state_refs.add(item.state_ref)
            binding_refs.add(item.hbm_binding_ref)
        if set(tensor_by_key) != {(expert, role) for expert in range(4) for role in _ROLES}:
            raise SchemaError("StateABI expert-role closure is not exact", path=f"{path}.tensors")
        if type(self.action_bindings) is not tuple or not self.action_bindings:
            raise SchemaError("StateABI requires execution DMA bindings", path=f"{path}.action_bindings")
        action_refs = set()
        counts = {}
        for index, item in enumerate(self.action_bindings):
            if type(item) is not MoeSwizzleStateActionBinding:
                raise SchemaError("requires exact MoeSwizzleStateActionBinding", path=f"{path}.action_bindings[{index}]")
            item.validate(f"{path}.action_bindings[{index}]")
            if item.execution_action_ref in action_refs:
                raise SchemaError("execution DMA action is bound more than once", path=f"{path}.action_bindings")
            tensor = tensor_by_key.get((item.expert_index, item.role))
            if tensor is None or (item.state_ref, item.hbm_binding_ref, item.bytes) != (tensor.state_ref, tensor.hbm_binding_ref, tensor.size_bytes):
                raise SchemaError("execution DMA binding disagrees with expert-role tensor", path=f"{path}.action_bindings[{index}]")
            action_refs.add(item.execution_action_ref)
            counts[(item.expert_index, item.role)] = counts.get((item.expert_index, item.role), 0) + 1
        for expert in range(4):
            role_counts = {counts.get((expert, role), 0) for role in _ROLES}
            if len(role_counts) != 1 or next(iter(role_counts)) == 0:
                raise SchemaError(
                    "execution DMA expert-role multiplicity is not exact",
                    path=f"{path}.action_bindings",
                )
        expected = stable_artifact_id(
            "moe_swizzle_workload_state_abi",
            self._semantic(),
            schema_version=MOE_SWIZZLE_STATE_ABI_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable workload StateABI id", path=f"{path}.id")


__all__ = [
    "MOE_SWIZZLE_STATE_ABI_SCHEMA_VERSION", "MoeSwizzleStateActionBinding",
    "MoeSwizzleStateTensor", "MoeSwizzleWorkloadStateABI",
    "parse_moe_expert_weight_tensor_ref",
]
