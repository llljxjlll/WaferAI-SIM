"""Common immutable values shared by versioned frontend artifacts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError


UINT64_MAX = (1 << 64) - 1


class DType(str, Enum):
    FP16 = "fp16"
    FP32 = "fp32"
    INT32 = "int32"


class RoundingMode(str, Enum):
    RNE = "rne"


class ValidationMode(str, Enum):
    TIMING = "timing"
    FUNCTIONAL = "functional"


class MeshAxisName(str, Enum):
    DP = "dp"
    TP = "tp"
    SP = "sp"
    EP = "ep"
    PP = "pp"
    CP = "cp"


def validate_uint64(value: int, path: str) -> None:
    if type(value) is not int or value < 0 or value > UINT64_MAX:
        raise SchemaError(
            f"expected an unsigned 64-bit integer, got {value!r}", path=path
        )


def validate_nonempty(value: str, path: str) -> None:
    if not isinstance(value, str) or not value:
        raise SchemaError("must be a non-empty string", path=path)


def validate_unique_ids(items: tuple[object, ...], path: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for index, item in enumerate(items):
        item_id = getattr(item, "id", None)
        if not isinstance(item_id, str) or not item_id:
            raise SchemaError("must be a non-empty string", path=f"{path}[{index}].id")
        if item_id in result:
            raise SchemaError(f"duplicate id {item_id!r}", path=f"{path}[{index}].id")
        result[item_id] = item
    return result


def validate_dependency_dag(
    items: tuple[object, ...],
    path: str,
) -> dict[str, object]:
    """Validate unique IDs, dependency references, and acyclicity."""

    index = validate_unique_ids(items, path)
    for item_index, item in enumerate(items):
        dependencies = getattr(item, "deps", None)
        if not isinstance(dependencies, tuple):
            raise SchemaError("deps must be an immutable tuple", path=f"{path}[{item_index}].deps")
        if len(set(dependencies)) != len(dependencies):
            raise SchemaError("contains duplicate dependencies", path=f"{path}[{item_index}].deps")
        for dep_index, dependency in enumerate(dependencies):
            validate_nonempty(dependency, f"{path}[{item_index}].deps[{dep_index}]")
            if dependency not in index:
                raise SchemaError(
                    f"dangling dependency {dependency!r}",
                    path=f"{path}[{item_index}].deps[{dep_index}]",
                )

    # Kahn traversal avoids Python's recursion limit for action/task DAGs with
    # thousands of nodes.  Both the initial ready queue and dependent lists use
    # artifact tuple order, so traversal remains deterministic without imposing
    # a new lexical ordering contract on semantically ordered tuples.
    remaining_dependencies = {
        item_id: len(getattr(item, "deps")) for item_id, item in index.items()
    }
    dependents: dict[str, list[str]] = {item_id: [] for item_id in index}
    for item_id, item in index.items():
        for dependency in getattr(item, "deps"):
            dependents[dependency].append(item_id)
    ready = deque(
        item_id
        for item_id in index
        if remaining_dependencies[item_id] == 0
    )
    visited_count = 0
    while ready:
        item_id = ready.popleft()
        visited_count += 1
        for dependent in dependents[item_id]:
            remaining_dependencies[dependent] -= 1
            if remaining_dependencies[dependent] == 0:
                ready.append(dependent)
    if visited_count != len(index):
        raise SchemaError("dependency graph contains a cycle", path=path)
    return index


@dataclass(frozen=True, slots=True)
class Sharding:
    mesh_ref: str
    dim_map: tuple[MeshAxisName | None, ...]
    partial: tuple[MeshAxisName, ...]

    def validate(self, path: str = "sharding") -> None:
        validate_nonempty(self.mesh_ref, f"{path}.mesh_ref")
        mapped_axes = tuple(axis for axis in self.dim_map if axis is not None)
        if len(set(mapped_axes)) != len(mapped_axes):
            raise SchemaError("mesh axis is mapped to multiple dimensions", path=f"{path}.dim_map")
        if len(set(self.partial)) != len(self.partial):
            raise SchemaError("contains duplicate mesh axes", path=f"{path}.partial")
        overlap = set(mapped_axes).intersection(self.partial)
        if overlap:
            raise SchemaError(
                f"axis {sorted(axis.value for axis in overlap)[0]!r} cannot be both sharded and partial",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class TensorValue:
    id: str
    shape: tuple[int, ...]
    dtype: DType
    logical_layout: str
    sharding: Sharding
    producer: str | None
    consumers: tuple[str, ...]
    alias_set: str | None

    def validate(self, path: str = "value") -> None:
        validate_nonempty(self.id, f"{path}.id")
        if type(self.dtype) is not DType:
            raise SchemaError("must be a DType", path=f"{path}.dtype")
        if not self.shape:
            raise SchemaError("tensor rank must be non-zero", path=f"{path}.shape")
        for index, dimension in enumerate(self.shape):
            validate_uint64(dimension, f"{path}.shape[{index}]")
            if dimension == 0:
                raise SchemaError("must be greater than zero", path=f"{path}.shape[{index}]")
        validate_nonempty(self.logical_layout, f"{path}.logical_layout")
        self.sharding.validate(f"{path}.sharding")
        if len(self.sharding.dim_map) != len(self.shape):
            raise SchemaError(
                "dim_map length must equal tensor rank", path=f"{path}.sharding.dim_map"
            )
        if self.producer is not None:
            validate_nonempty(self.producer, f"{path}.producer")
        if len(set(self.consumers)) != len(self.consumers):
            raise SchemaError("contains duplicate node ids", path=f"{path}.consumers")
        for index, consumer in enumerate(self.consumers):
            validate_nonempty(consumer, f"{path}.consumers[{index}]")
        if self.alias_set is not None:
            validate_nonempty(self.alias_set, f"{path}.alias_set")


@dataclass(frozen=True, slots=True)
class ProfileKey:
    """Complete static inference shape; no dimension is inferred from M."""

    prefill_tokens: int
    decode_tokens: int
    num_seqs: int
    context_sum: int
    context_max: int
    kv_pages: int
    expert_load: int | None

    def digest(self) -> str:
        from .serde import canonical_digest

        return canonical_digest(self)

    def stable_id(self) -> str:
        return f"profile_{self.digest()[:16]}"

    def validate(self, path: str = "profile_key") -> None:
        for field_name in (
            "prefill_tokens",
            "decode_tokens",
            "num_seqs",
            "context_sum",
            "context_max",
            "kv_pages",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.expert_load is not None:
            validate_uint64(self.expert_load, f"{path}.expert_load")
        if self.num_seqs == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.num_seqs")
        if self.prefill_tokens + self.decode_tokens == 0:
            raise SchemaError(
                "at least one of prefill_tokens/decode_tokens must be non-zero",
                path=path,
            )
        if self.context_max > self.context_sum:
            raise SchemaError(
                "must not exceed context_sum", path=f"{path}.context_max"
            )


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    schema_version: str
    producer_pass: str
    id: str

    def validate(self, path: str = "metadata") -> None:
        for field_name in ("schema_version", "producer_pass", "id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise SchemaError("must be a non-empty string", path=f"{path}.{field_name}")


def stable_artifact_id(
    kind: str,
    semantic_key: object,
    *,
    schema_version: str,
) -> str:
    """Return a deterministic ID without timestamps, UUIDs, or Python hashes."""

    if not kind or not schema_version:
        raise ValueError("kind and schema_version must be non-empty")
    # Lazy import avoids coupling the common schema definitions to their codec.
    from .serde import canonical_digest

    digest = canonical_digest(
        {
            "kind": kind,
            "schema_version": schema_version,
            "semantic_key": semantic_key,
        }
    )
    return f"{kind}_{digest[:16]}"
