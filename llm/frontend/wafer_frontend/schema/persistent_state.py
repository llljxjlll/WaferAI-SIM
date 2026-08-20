"""Stage-1a logical persistent state and physical HBM backing contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import (
    DType,
    UINT64_MAX,
    stable_artifact_id,
    validate_nonempty,
    validate_uint64,
)


PERSISTENT_STATE_IDENTITY_SCHEMA_VERSION = (
    "wafer_frontend.persistent_state_identity/v1alpha2"
)
PERSISTENT_STATE_DECL_SCHEMA_VERSION = (
    "wafer_frontend.persistent_state_decl/v1alpha2"
)
HBM_ADDRESS_SPACE_SCHEMA_VERSION = "wafer_frontend.hbm_address_space/v1alpha1"
HBM_BINDING_SCHEMA_VERSION = "wafer_frontend.hbm_binding/v1alpha1"
PERSISTENT_STATE_MANIFEST_SCHEMA_VERSION = (
    "wafer_frontend.persistent_state_manifest/v1alpha2"
)
STATE_STAGING_ID_SCHEMA_VERSION = (
    "wafer_frontend.state_staging_identity/v1"
)

_INT32_MAX = (1 << 31) - 1
_DTYPE_BYTES = {DType.FP16: 2, DType.FP32: 4}


class StateKind(str, Enum):
    PARAMETER = "parameter"
    TRAINABLE_PARAMETER = "trainable_parameter"
    KV_KEY = "kv_key"
    KV_VALUE = "kv_value"
    OPTIMIZER_RESERVED = "optimizer_reserved"


class PersistentStateLifetime(str, Enum):
    STEP = "step"
    PERSISTENT = "persistent"


class PersistentStateAccess(str, Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    RESERVED = "reserved"


def canonical_state_staging_value_id(state_access_ref: str) -> str:
    """Return the stable SRAM staging identity for one logical state access."""

    validate_nonempty(state_access_ref, "state_access_ref")
    return stable_artifact_id(
        "state_staging_value",
        {"state_access_ref": state_access_ref},
        schema_version=STATE_STAGING_ID_SCHEMA_VERSION,
    )


def _validate_int32(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value > _INT32_MAX:
        raise SchemaError("must fit signed 32-bit range", path=path)


def _validate_power_of_two(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value == 0 or value & (value - 1):
        raise SchemaError("must be a positive power of two", path=path)


def _checked_end(start: int, size: int, path: str) -> int:
    validate_uint64(start, f"{path}.start")
    validate_uint64(size, f"{path}.size_bytes")
    if size == 0:
        raise SchemaError("must be greater than zero", path=f"{path}.size_bytes")
    if start > UINT64_MAX - size:
        raise SchemaError("half-open byte range overflows uint64", path=path)
    return start + size


def _tensor_bytes(shape: tuple[int, ...], dtype: DType, path: str) -> int:
    if type(shape) is not tuple or not shape:
        raise SchemaError("must be a non-empty immutable tuple", path=f"{path}.shape")
    if type(dtype) is not DType or dtype not in _DTYPE_BYTES:
        raise SchemaError("unsupported persistent-state dtype", path=f"{path}.dtype")
    elements = 1
    for index, extent in enumerate(shape):
        _validate_int32(extent, f"{path}.shape[{index}]")
        if extent == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.shape[{index}]")
        if elements > UINT64_MAX // extent:
            raise SchemaError("tensor element count overflows uint64", path=f"{path}.shape")
        elements *= extent
    element_bytes = _DTYPE_BYTES[dtype]
    if elements > UINT64_MAX // element_bytes:
        raise SchemaError("tensor byte size overflows uint64", path=f"{path}.shape")
    return elements * element_bytes


@dataclass(frozen=True, slots=True)
class PersistentStateIdentity:
    id: str
    kind: StateKind
    instance_ref: str
    mesh_ref: str
    request_ref: str | None
    layer_index: int | None
    tensor_ref: str | None
    shard_index: int
    generation: int

    @classmethod
    def create(
        cls,
        *,
        kind: StateKind,
        instance_ref: str,
        mesh_ref: str,
        request_ref: str | None,
        layer_index: int | None,
        tensor_ref: str | None,
        shard_index: int,
        generation: int,
    ) -> "PersistentStateIdentity":
        key = {
            "kind": kind,
            "instance_ref": instance_ref,
            "mesh_ref": mesh_ref,
            "request_ref": request_ref,
            "layer_index": layer_index,
            "tensor_ref": tensor_ref,
            "shard_index": shard_index,
            "generation": generation,
        }
        result = cls(
            id=stable_artifact_id(
                "persistent_state_identity",
                key,
                schema_version=PERSISTENT_STATE_IDENTITY_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "instance_ref": self.instance_ref,
            "mesh_ref": self.mesh_ref,
            "request_ref": self.request_ref,
            "layer_index": self.layer_index,
            "tensor_ref": self.tensor_ref,
            "shard_index": self.shard_index,
            "generation": self.generation,
        }

    def validate(self, path: str = "persistent_state_identity") -> None:
        if type(self.kind) is not StateKind:
            raise SchemaError("must be a StateKind", path=f"{path}.kind")
        validate_nonempty(self.instance_ref, f"{path}.instance_ref")
        validate_nonempty(self.mesh_ref, f"{path}.mesh_ref")
        _validate_int32(self.shard_index, f"{path}.shard_index")
        _validate_int32(self.generation, f"{path}.generation")
        if self.request_ref is not None:
            validate_nonempty(self.request_ref, f"{path}.request_ref")
        if self.layer_index is not None:
            _validate_int32(self.layer_index, f"{path}.layer_index")
        if self.tensor_ref is not None:
            validate_nonempty(self.tensor_ref, f"{path}.tensor_ref")
        if self.kind in (
            StateKind.PARAMETER,
            StateKind.TRAINABLE_PARAMETER,
            StateKind.OPTIMIZER_RESERVED,
        ):
            if self.tensor_ref is None:
                raise SchemaError(
                    "is required for parameter/optimizer state",
                    path=f"{path}.tensor_ref",
                )
            if self.request_ref is not None or self.layer_index is not None:
                raise SchemaError(
                    "parameter/optimizer state cannot carry request or layer fields",
                    path=path,
                )
        else:
            if self.request_ref is None or self.layer_index is None:
                raise SchemaError("KV state requires request_ref and layer_index", path=path)
            if self.tensor_ref is not None:
                raise SchemaError(
                    "KV kind already identifies the tensor role",
                    path=f"{path}.tensor_ref",
                )
        expected = stable_artifact_id(
            "persistent_state_identity",
            self._semantic_key(),
            schema_version=PERSISTENT_STATE_IDENTITY_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(
                f"unstable artifact id; expected {expected!r}", path=f"{path}.id"
            )


@dataclass(frozen=True, slots=True)
class PersistentStateDecl:
    id: str
    identity: PersistentStateIdentity
    shape: tuple[int, ...]
    dtype: DType
    layout: str
    lifetime: PersistentStateLifetime
    access: PersistentStateAccess
    tensor_bytes: int

    @classmethod
    def create(
        cls,
        *,
        identity: PersistentStateIdentity,
        shape: tuple[int, ...],
        dtype: DType,
        layout: str,
        lifetime: PersistentStateLifetime,
        access: PersistentStateAccess,
    ) -> "PersistentStateDecl":
        key = {
            "identity": identity,
            "shape": shape,
            "dtype": dtype,
            "layout": layout,
            "lifetime": lifetime,
            "access": access,
            "tensor_bytes": _tensor_bytes(shape, dtype, "persistent_state_decl"),
        }
        result = cls(
            id=stable_artifact_id(
                "persistent_state_decl",
                key,
                schema_version=PERSISTENT_STATE_DECL_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "shape": self.shape,
            "dtype": self.dtype,
            "layout": self.layout,
            "lifetime": self.lifetime,
            "access": self.access,
            "tensor_bytes": self.tensor_bytes,
        }

    def validate(self, path: str = "persistent_state_decl") -> None:
        if type(self.identity) is not PersistentStateIdentity:
            raise SchemaError(
                "must be a PersistentStateIdentity", path=f"{path}.identity"
            )
        self.identity.validate(f"{path}.identity")
        expected_bytes = _tensor_bytes(self.shape, self.dtype, path)
        validate_nonempty(self.layout, f"{path}.layout")
        if type(self.lifetime) is not PersistentStateLifetime:
            raise SchemaError(
                "must be a PersistentStateLifetime", path=f"{path}.lifetime"
            )
        if type(self.access) is not PersistentStateAccess:
            raise SchemaError(
                "must be a PersistentStateAccess", path=f"{path}.access"
            )
        validate_uint64(self.tensor_bytes, f"{path}.tensor_bytes")
        if self.tensor_bytes != expected_bytes:
            raise SchemaError(
                "must equal product(shape) * dtype bytes",
                path=f"{path}.tensor_bytes",
            )
        if self.identity.kind is StateKind.PARAMETER:
            if (
                self.lifetime is not PersistentStateLifetime.PERSISTENT
                or self.access is not PersistentStateAccess.READ_ONLY
            ):
                raise SchemaError(
                    "parameter must be PERSISTENT and READ_ONLY", path=path
                )
        elif self.identity.kind is StateKind.TRAINABLE_PARAMETER:
            if (
                self.lifetime is not PersistentStateLifetime.PERSISTENT
                or self.access is not PersistentStateAccess.READ_WRITE
            ):
                raise SchemaError(
                    "trainable parameter must be PERSISTENT and READ_WRITE",
                    path=path,
                )
        elif self.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE):
            if (
                self.lifetime is not PersistentStateLifetime.PERSISTENT
                or self.access is not PersistentStateAccess.READ_WRITE
            ):
                raise SchemaError(
                    "KV state must be PERSISTENT and READ_WRITE", path=path
                )
        elif self.access is not PersistentStateAccess.RESERVED:
            raise SchemaError(
                "optimizer reservation cannot grant DMA access",
                path=f"{path}.access",
            )
        expected = stable_artifact_id(
            "persistent_state_decl",
            self._semantic_key(),
            schema_version=PERSISTENT_STATE_DECL_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(
                f"unstable artifact id; expected {expected!r}", path=f"{path}.id"
            )


@dataclass(frozen=True, slots=True)
class HbmAddressSpace:
    id: str
    die_id: int
    base_address: int
    size_bytes: int
    alignment_bytes: int

    @classmethod
    def create(
        cls,
        *,
        die_id: int,
        base_address: int,
        size_bytes: int,
        alignment_bytes: int,
    ) -> "HbmAddressSpace":
        key = {
            "die_id": die_id,
            "base_address": base_address,
            "size_bytes": size_bytes,
            "alignment_bytes": alignment_bytes,
        }
        result = cls(
            id=stable_artifact_id(
                "hbm_address_space",
                key,
                schema_version=HBM_ADDRESS_SPACE_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "die_id": self.die_id,
            "base_address": self.base_address,
            "size_bytes": self.size_bytes,
            "alignment_bytes": self.alignment_bytes,
        }

    def validate(self, path: str = "hbm_address_space") -> None:
        _validate_int32(self.die_id, f"{path}.die_id")
        _validate_power_of_two(self.alignment_bytes, f"{path}.alignment_bytes")
        _checked_end(self.base_address, self.size_bytes, path)
        if self.base_address % self.alignment_bytes:
            raise SchemaError(
                "base address must satisfy alignment", path=f"{path}.base_address"
            )
        if self.size_bytes % self.alignment_bytes:
            raise SchemaError(
                "address-space size must satisfy alignment", path=f"{path}.size_bytes"
            )
        expected = stable_artifact_id(
            "hbm_address_space",
            self._semantic_key(),
            schema_version=HBM_ADDRESS_SPACE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(
                f"unstable artifact id; expected {expected!r}", path=f"{path}.id"
            )


@dataclass(frozen=True, slots=True)
class HbmBinding:
    id: str
    state_ref: str
    die_id: int
    address: int
    size_bytes: int

    @classmethod
    def create(
        cls,
        *,
        state_ref: str,
        die_id: int,
        address: int,
        size_bytes: int,
    ) -> "HbmBinding":
        key = {
            "state_ref": state_ref,
            "die_id": die_id,
            "address": address,
            "size_bytes": size_bytes,
        }
        result = cls(
            id=stable_artifact_id(
                "hbm_binding",
                key,
                schema_version=HBM_BINDING_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "state_ref": self.state_ref,
            "die_id": self.die_id,
            "address": self.address,
            "size_bytes": self.size_bytes,
        }

    def validate(self, path: str = "hbm_binding") -> None:
        validate_nonempty(self.state_ref, f"{path}.state_ref")
        _validate_int32(self.die_id, f"{path}.die_id")
        _checked_end(self.address, self.size_bytes, path)
        expected = stable_artifact_id(
            "hbm_binding",
            self._semantic_key(),
            schema_version=HBM_BINDING_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(
                f"unstable artifact id; expected {expected!r}", path=f"{path}.id"
            )


@dataclass(frozen=True, slots=True)
class PersistentStateManifest:
    schema_version: str
    producer_pass: str
    id: str
    address_spaces: tuple[HbmAddressSpace, ...]
    declarations: tuple[PersistentStateDecl, ...]
    bindings: tuple[HbmBinding, ...]

    @classmethod
    def create(
        cls,
        *,
        address_spaces: tuple[HbmAddressSpace, ...],
        declarations: tuple[PersistentStateDecl, ...],
        bindings: tuple[HbmBinding, ...],
    ) -> "PersistentStateManifest":
        key = {
            "address_spaces": tuple(
                sorted(address_spaces, key=lambda item: (item.die_id, item.id))
            ),
            "declarations": tuple(
                sorted(declarations, key=lambda item: (item.identity.id, item.id))
            ),
            "bindings": tuple(
                sorted(bindings, key=lambda item: (item.state_ref, item.id))
            ),
        }
        result = cls(
            schema_version=PERSISTENT_STATE_MANIFEST_SCHEMA_VERSION,
            producer_pass="persistent_state_manifest_builder",
            id=stable_artifact_id(
                "persistent_state_manifest",
                key,
                schema_version=PERSISTENT_STATE_MANIFEST_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "address_spaces": self.address_spaces,
            "declarations": self.declarations,
            "bindings": self.bindings,
        }

    def validate(self, path: str = "persistent_state_manifest") -> None:
        if self.schema_version != PERSISTENT_STATE_MANIFEST_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "persistent_state_manifest_builder":
            raise SchemaError(
                "must be 'persistent_state_manifest_builder'",
                path=f"{path}.producer_pass",
            )
        if not self.address_spaces:
            raise SchemaError(
                "must contain at least one HBM address space",
                path=f"{path}.address_spaces",
            )
        if not self.declarations:
            raise SchemaError(
                "must contain at least one state declaration",
                path=f"{path}.declarations",
            )
        if self.address_spaces != tuple(
            sorted(self.address_spaces, key=lambda item: (item.die_id, item.id))
        ):
            raise SchemaError(
                "must use canonical die/id order", path=f"{path}.address_spaces"
            )
        if self.declarations != tuple(
            sorted(self.declarations, key=lambda item: (item.identity.id, item.id))
        ):
            raise SchemaError(
                "must use canonical identity/id order",
                path=f"{path}.declarations",
            )
        if self.bindings != tuple(
            sorted(self.bindings, key=lambda item: (item.state_ref, item.id))
        ):
            raise SchemaError(
                "must use canonical state/id order", path=f"{path}.bindings"
            )

        spaces_by_die: dict[int, HbmAddressSpace] = {}
        space_ranges: list[tuple[int, int]] = []
        for index, space in enumerate(self.address_spaces):
            space.validate(f"{path}.address_spaces[{index}]")
            if space.die_id in spaces_by_die:
                raise SchemaError(
                    "duplicate die HBM address space",
                    path=f"{path}.address_spaces[{index}].die_id",
                )
            start, end = space.base_address, space.base_address + space.size_bytes
            if any(start < old_end and old_start < end for old_start, old_end in space_ranges):
                raise SchemaError(
                    "HBM address spaces must not overlap",
                    path=f"{path}.address_spaces[{index}]",
                )
            spaces_by_die[space.die_id] = space
            space_ranges.append((start, end))

        declarations_by_id: dict[str, PersistentStateDecl] = {}
        identity_ids: set[str] = set()
        for index, declaration in enumerate(self.declarations):
            declaration.validate(f"{path}.declarations[{index}]")
            if declaration.id in declarations_by_id:
                raise SchemaError(
                    "duplicate state declaration id",
                    path=f"{path}.declarations[{index}].id",
                )
            if declaration.identity.id in identity_ids:
                raise SchemaError(
                    "duplicate logical state identity",
                    path=f"{path}.declarations[{index}].identity.id",
                )
            declarations_by_id[declaration.id] = declaration
            identity_ids.add(declaration.identity.id)

        bindings_by_state: dict[str, HbmBinding] = {}
        binding_ranges: dict[int, list[tuple[int, int]]] = {}
        for index, binding in enumerate(self.bindings):
            binding_path = f"{path}.bindings[{index}]"
            binding.validate(binding_path)
            declaration = declarations_by_id.get(binding.state_ref)
            space = spaces_by_die.get(binding.die_id)
            if declaration is None:
                raise SchemaError(
                    "references an unknown state declaration",
                    path=f"{binding_path}.state_ref",
                )
            if space is None:
                raise SchemaError(
                    "references a die without an HBM address space",
                    path=f"{binding_path}.die_id",
                )
            if binding.state_ref in bindings_by_state:
                raise SchemaError(
                    "each state requires exactly one HBM binding",
                    path=f"{binding_path}.state_ref",
                )
            if binding.size_bytes != declaration.tensor_bytes:
                raise SchemaError(
                    "size must equal the state tensor byte size",
                    path=f"{binding_path}.size_bytes",
                )
            if binding.address % space.alignment_bytes:
                raise SchemaError(
                    "address must satisfy the HBM alignment",
                    path=f"{binding_path}.address",
                )
            end = binding.address + binding.size_bytes
            space_end = space.base_address + space.size_bytes
            if binding.address < space.base_address or end > space_end:
                raise SchemaError(
                    "binding must lie wholly inside its home address space",
                    path=binding_path,
                )
            ranges = binding_ranges.setdefault(binding.die_id, [])
            if any(
                binding.address < old_end and old_start < end
                for old_start, old_end in ranges
            ):
                raise SchemaError(
                    "HBM state bindings must not overlap", path=binding_path
                )
            ranges.append((binding.address, end))
            bindings_by_state[binding.state_ref] = binding
        if set(bindings_by_state) != set(declarations_by_id):
            raise SchemaError(
                "bindings must exactly cover all state declarations",
                path=f"{path}.bindings",
            )

        expected = stable_artifact_id(
            "persistent_state_manifest",
            self._semantic_key(),
            schema_version=PERSISTENT_STATE_MANIFEST_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(
                f"unstable artifact id; expected {expected!r}", path=f"{path}.id"
            )
