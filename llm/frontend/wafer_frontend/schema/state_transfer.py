"""Typed, exact persistent-state transfers introduced at IR-2 projection."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import (
    AttentionWorkload,
    OpKind,
    StateAccessMode,
    state_access_tensor_view,
)
from .ir1 import CrossGroupRoute, IR1, PairRoute
from .persistent_state import (
    PersistentStateAccess,
    PersistentStateLifetime,
    StateKind,
)


STATE_TRANSFER_CONTRACT_SCHEMA_VERSION = (
    "wafer_frontend.state_transfer_contract/v1alpha1"
)
SLICED_KV_STATE_TRANSFER_CONTRACT_SCHEMA_VERSION = (
    "wafer_frontend.state_transfer_contract/v1alpha2"
)
SEGMENTED_KV_STATE_TRANSFER_CONTRACT_SCHEMA_VERSION = (
    "wafer_frontend.segmented_kv_state_transfer_contract/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class StateTransferContract:
    """One direct, whole-KV transfer between two distinct IR-1 accesses.

    State identity, payload metadata, HBM placement, and route endpoints are
    intentionally not copied into this contract. ``validate_against``
    resolves every such field from the referenced IR-1 artifact.
    """

    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    source_state_access_ref: str
    destination_state_access_ref: str
    pair_route_ref: str

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        source_ir1_id: str,
        source_state_access_ref: str,
        destination_state_access_ref: str,
        pair_route_ref: str,
    ) -> "StateTransferContract":
        semantic_key = {
            "source_ir1_id": source_ir1_id,
            "source_state_access_ref": source_state_access_ref,
            "destination_state_access_ref": destination_state_access_ref,
            "pair_route_ref": pair_route_ref,
        }
        result = cls(
            schema_version=STATE_TRANSFER_CONTRACT_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "state_transfer_contract",
                semantic_key,
                schema_version=STATE_TRANSFER_CONTRACT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_ir1_id": self.source_ir1_id,
            "source_state_access_ref": self.source_state_access_ref,
            "destination_state_access_ref": self.destination_state_access_ref,
            "pair_route_ref": self.pair_route_ref,
        }

    def validate(self, path: str = "state_transfer_contract") -> None:
        if self.schema_version != STATE_TRANSFER_CONTRACT_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        for field_name in (
            "source_ir1_id",
            "source_state_access_ref",
            "destination_state_access_ref",
            "pair_route_ref",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        expected_id = stable_artifact_id(
            "state_transfer_contract",
            self._semantic_key(),
            schema_version=STATE_TRANSFER_CONTRACT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        ir1: IR1,
        path: str = "state_transfer_contract",
    ) -> None:
        self.validate(path)
        if type(ir1) is not IR1:
            raise SchemaError("must be an IR1", path="ir1")
        ir1.validate("ir1")
        if self.source_ir1_id != ir1.id:
            raise SchemaError(
                "references a different IR-1", path=f"{path}.source_ir1_id"
            )
        manifest = ir1.persistent_state_manifest
        if manifest is None:
            raise SchemaError(
                "state transfer requires a persistent-state manifest", path=path
            )

        access_index = {access.id: access for access in ir1.state_accesses}
        source_access = access_index.get(self.source_state_access_ref)
        destination_access = access_index.get(
            self.destination_state_access_ref
        )
        if source_access is None:
            raise SchemaError(
                "references an unknown IR-1 state access",
                path=f"{path}.source_state_access_ref",
            )
        if destination_access is None:
            raise SchemaError(
                "references an unknown IR-1 state access",
                path=f"{path}.destination_state_access_ref",
            )
        if source_access.id == destination_access.id:
            raise SchemaError(
                "source and destination accesses must be distinct", path=path
            )
        if source_access.state_ref == destination_access.state_ref:
            raise SchemaError(
                "source and destination states must be distinct", path=path
            )
        if source_access.mode is not StateAccessMode.READ:
            raise SchemaError(
                "source access must be READ",
                path=f"{path}.source_state_access_ref",
            )
        if destination_access.mode is not StateAccessMode.WRITE:
            raise SchemaError(
                "destination access must be WRITE",
                path=f"{path}.destination_state_access_ref",
            )

        declaration_index = {
            declaration.id: declaration
            for declaration in manifest.declarations
        }
        source_declaration = declaration_index[source_access.state_ref]
        destination_declaration = declaration_index[
            destination_access.state_ref
        ]
        source_identity = source_declaration.identity
        destination_identity = destination_declaration.identity
        kv_kinds = (StateKind.KV_KEY, StateKind.KV_VALUE)
        if (
            source_identity.kind not in kv_kinds
            or destination_identity.kind is not source_identity.kind
            or source_identity.tensor_ref is not None
            or destination_identity.tensor_ref is not None
        ):
            raise SchemaError(
                "v1 requires matching opaque KV_KEY or KV_VALUE state",
                path=path,
            )
        if (
            source_identity.request_ref,
            source_identity.layer_index,
            source_identity.generation,
        ) != (
            destination_identity.request_ref,
            destination_identity.layer_index,
            destination_identity.generation,
        ):
            raise SchemaError(
                "source and destination KV lineage must match",
                path=path,
            )
        if (
            source_declaration.shape,
            source_declaration.dtype,
            source_declaration.layout,
            source_declaration.tensor_bytes,
            source_declaration.lifetime,
        ) != (
            destination_declaration.shape,
            destination_declaration.dtype,
            destination_declaration.layout,
            destination_declaration.tensor_bytes,
            destination_declaration.lifetime,
        ):
            raise SchemaError(
                "source and destination KV payload metadata must match",
                path=path,
            )
        if source_declaration.lifetime is not PersistentStateLifetime.PERSISTENT:
            raise SchemaError("v1 requires persistent KV state", path=path)

        node_index = {node.id: node for node in ir1.nodes}
        if (
            node_index[source_access.node_ref].kind is not OpKind.ATTENTION
            or node_index[destination_access.node_ref].kind
            is not OpKind.ATTENTION
        ):
            raise SchemaError(
                "v1 KV transfer accesses must target ATTENTION nodes", path=path
            )

        binding_index = {
            binding.state_ref: binding for binding in manifest.bindings
        }
        source_binding = binding_index[source_access.state_ref]
        destination_binding = binding_index[destination_access.state_ref]
        if source_binding.die_id == destination_binding.die_id:
            raise SchemaError(
                "source and destination HBM homes must differ", path=path
            )

        route_index: dict[str, PairRoute] = {
            route.id: route
            for group in ir1.groups
            for route in group.embedding.routes
        }
        route = route_index.get(self.pair_route_ref)
        if route is None:
            raise SchemaError(
                "references an unknown IR-1 PairRoute",
                path=f"{path}.pair_route_ref",
            )
        if (
            route.source_rank,
            route.destination_rank,
            route.die_path[0],
            route.die_path[-1],
        ) != (
            source_access.rank,
            destination_access.rank,
            source_binding.die_id,
            destination_binding.die_id,
        ):
            raise SchemaError(
                "PairRoute rank/home endpoints disagree with state accesses",
                path=f"{path}.pair_route_ref",
            )


def _validate_local_slice(
    offset: tuple[int, ...],
    shape: tuple[int, ...],
    *,
    path: str,
) -> None:
    if type(offset) is not tuple or type(shape) is not tuple:
        raise SchemaError("offset and shape must be immutable tuples", path=path)
    if not shape or len(offset) != len(shape):
        raise SchemaError(
            "offset and shape must have equal non-zero rank", path=path
        )
    for field_name, values in (("offset", offset), ("shape", shape)):
        for index, value in enumerate(values):
            validate_uint64(value, f"{path}.{field_name}[{index}]")
            if field_name == "shape" and value == 0:
                raise SchemaError(
                    "must be greater than zero",
                    path=f"{path}.shape[{index}]",
                )


@dataclass(frozen=True, slots=True)
class SlicedKvStateTransferContract:
    """One exact Stage-4 KV-history slice between WRITE accesses."""

    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    source_state_access_ref: str
    destination_state_access_ref: str
    cross_group_route_ref: str
    source_local_offset: tuple[int, ...]
    source_local_shape: tuple[int, ...]
    destination_local_offset: tuple[int, ...]
    destination_local_shape: tuple[int, ...]
    bytes: int

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        source_ir1_id: str,
        source_state_access_ref: str,
        destination_state_access_ref: str,
        cross_group_route_ref: str,
        source_local_offset: tuple[int, ...],
        source_local_shape: tuple[int, ...],
        destination_local_offset: tuple[int, ...],
        destination_local_shape: tuple[int, ...],
        bytes: int,
    ) -> "SlicedKvStateTransferContract":
        semantic_key = {
            "source_ir1_id": source_ir1_id,
            "source_state_access_ref": source_state_access_ref,
            "destination_state_access_ref": destination_state_access_ref,
            "cross_group_route_ref": cross_group_route_ref,
            "source_local_offset": source_local_offset,
            "source_local_shape": source_local_shape,
            "destination_local_offset": destination_local_offset,
            "destination_local_shape": destination_local_shape,
            "bytes": bytes,
        }
        result = cls(
            schema_version=SLICED_KV_STATE_TRANSFER_CONTRACT_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "state_transfer_contract",
                semantic_key,
                schema_version=(
                    SLICED_KV_STATE_TRANSFER_CONTRACT_SCHEMA_VERSION
                ),
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_ir1_id": self.source_ir1_id,
            "source_state_access_ref": self.source_state_access_ref,
            "destination_state_access_ref": (
                self.destination_state_access_ref
            ),
            "cross_group_route_ref": self.cross_group_route_ref,
            "source_local_offset": self.source_local_offset,
            "source_local_shape": self.source_local_shape,
            "destination_local_offset": self.destination_local_offset,
            "destination_local_shape": self.destination_local_shape,
            "bytes": self.bytes,
        }

    def validate(
        self, path: str = "sliced_kv_state_transfer_contract"
    ) -> None:
        if (
            self.schema_version
            != SLICED_KV_STATE_TRANSFER_CONTRACT_SCHEMA_VERSION
        ):
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        for field_name in (
            "source_ir1_id",
            "source_state_access_ref",
            "destination_state_access_ref",
            "cross_group_route_ref",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        _validate_local_slice(
            self.source_local_offset,
            self.source_local_shape,
            path=f"{path}.source_local",
        )
        _validate_local_slice(
            self.destination_local_offset,
            self.destination_local_shape,
            path=f"{path}.destination_local",
        )
        validate_uint64(self.bytes, f"{path}.bytes")
        if self.bytes == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.bytes")
        expected_id = stable_artifact_id(
            "state_transfer_contract",
            self._semantic_key(),
            schema_version=SLICED_KV_STATE_TRANSFER_CONTRACT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        ir1: IR1,
        path: str = "sliced_kv_state_transfer_contract",
    ) -> None:
        self.validate(path)
        if type(ir1) is not IR1:
            raise SchemaError("must be an IR1", path="ir1")
        ir1.validate("ir1")
        if self.source_ir1_id != ir1.id:
            raise SchemaError(
                "references a different IR-1", path=f"{path}.source_ir1_id"
            )
        manifest = ir1.persistent_state_manifest
        if manifest is None:
            raise SchemaError(
                "sliced KV transfer requires persistent-state provenance",
                path=path,
            )

        access_index = {access.id: access for access in ir1.state_accesses}
        source_access = access_index.get(self.source_state_access_ref)
        destination_access = access_index.get(
            self.destination_state_access_ref
        )
        if source_access is None:
            raise SchemaError(
                "references an unknown IR-1 state access",
                path=f"{path}.source_state_access_ref",
            )
        if destination_access is None:
            raise SchemaError(
                "references an unknown IR-1 state access",
                path=f"{path}.destination_state_access_ref",
            )
        if source_access.id == destination_access.id:
            raise SchemaError(
                "source and destination accesses must be distinct", path=path
            )
        if source_access.state_ref == destination_access.state_ref:
            raise SchemaError(
                "source and destination states must be distinct", path=path
            )
        if source_access.mode is not StateAccessMode.WRITE:
            raise SchemaError(
                "source access must be WRITE",
                path=f"{path}.source_state_access_ref",
            )
        if destination_access.mode is not StateAccessMode.WRITE:
            raise SchemaError(
                "destination access must be WRITE",
                path=f"{path}.destination_state_access_ref",
            )

        declaration_index = {
            declaration.id: declaration
            for declaration in manifest.declarations
        }
        source_declaration = declaration_index[source_access.state_ref]
        destination_declaration = declaration_index[
            destination_access.state_ref
        ]
        source_identity = source_declaration.identity
        destination_identity = destination_declaration.identity
        kv_kinds = (StateKind.KV_KEY, StateKind.KV_VALUE)
        if (
            source_identity.kind not in kv_kinds
            or destination_identity.kind is not source_identity.kind
            or source_identity.tensor_ref is not None
            or destination_identity.tensor_ref is not None
        ):
            raise SchemaError(
                "v2 requires matching opaque KV_KEY or KV_VALUE state",
                path=path,
            )
        if (
            source_identity.request_ref,
            source_identity.layer_index,
            source_identity.generation,
        ) != (
            destination_identity.request_ref,
            destination_identity.layer_index,
            destination_identity.generation,
        ):
            raise SchemaError(
                "source and destination KV lineage must match", path=path
            )
        if (
            source_identity.instance_ref
            == destination_identity.instance_ref
        ):
            raise SchemaError(
                "v2 requires distinct source and destination instances",
                path=path,
            )
        if (
            source_declaration.lifetime
            is not PersistentStateLifetime.PERSISTENT
            or destination_declaration.lifetime
            is not PersistentStateLifetime.PERSISTENT
            or source_declaration.access
            is not PersistentStateAccess.READ_WRITE
            or destination_declaration.access
            is not PersistentStateAccess.READ_WRITE
        ):
            raise SchemaError(
                "v2 requires persistent read-write KV state", path=path
            )
        layouts = {"THD_packed", "THD_packed_kv_head_tp"}
        if (
            source_declaration.layout not in layouts
            or destination_declaration.layout not in layouts
            or len(source_declaration.shape) != 3
            or len(destination_declaration.shape) != 3
            or source_declaration.shape[0]
            != destination_declaration.shape[0]
            or source_declaration.shape[2]
            != destination_declaration.shape[2]
            or source_declaration.dtype is not destination_declaration.dtype
        ):
            raise SchemaError(
                "source and destination must use compatible rank-local THD KV domains",
                path=path,
            )

        node_index = {node.id: node for node in ir1.nodes}
        source_node = node_index[source_access.node_ref]
        destination_node = node_index[destination_access.node_ref]
        if (
            source_node.kind is not OpKind.ATTENTION
            or destination_node.kind is not OpKind.ATTENTION
            or type(source_node.workload) is not AttentionWorkload
            or type(destination_node.workload) is not AttentionWorkload
        ):
            raise SchemaError(
                "v2 KV transfer accesses must target ATTENTION nodes", path=path
            )
        source_workload = source_node.workload
        destination_workload = destination_node.workload
        if (
            source_workload.num_kv_heads,
            source_workload.head_dim,
            source_workload.dtype,
        ) != (
            destination_workload.num_kv_heads,
            destination_workload.head_dim,
            destination_workload.dtype,
        ):
            raise SchemaError(
                "source and destination global KV geometry must match", path=path
            )
        if (
            source_workload.dtype is not source_declaration.dtype
            or destination_workload.dtype is not destination_declaration.dtype
            or source_workload.rank_num_kv_heads
            != source_declaration.shape[1]
            or destination_workload.rank_num_kv_heads
            != destination_declaration.shape[1]
            or source_workload.head_dim != source_declaration.shape[2]
            or destination_workload.head_dim
            != destination_declaration.shape[2]
        ):
            raise SchemaError(
                "ATTENTION workload disagrees with rank-local KV state",
                path=path,
            )

        group_index = {group.id: group for group in ir1.groups}
        source_group = group_index[source_node.execution_group_ref]
        destination_group = group_index[
            destination_node.execution_group_ref
        ]
        if (
            len(source_group.logical_shape) != 1
            or len(destination_group.logical_shape) != 1
            or source_group.logical_shape[0]
            * source_workload.rank_num_kv_heads
            != source_workload.num_kv_heads
            or destination_group.logical_shape[0]
            * destination_workload.rank_num_kv_heads
            != destination_workload.num_kv_heads
        ):
            raise SchemaError(
                "KV shards must exactly partition one-dimensional groups",
                path=path,
            )

        route_index: dict[str, CrossGroupRoute] = {
            route.id: route for route in ir1.cross_routes
        }
        route = route_index.get(self.cross_group_route_ref)
        if route is None:
            raise SchemaError(
                "references an unknown IR-1 CrossGroupRoute",
                path=f"{path}.cross_group_route_ref",
            )
        binding_index = {
            binding.state_ref: binding for binding in manifest.bindings
        }
        source_binding = binding_index[source_access.state_ref]
        destination_binding = binding_index[destination_access.state_ref]
        if (
            route.source_group_ref,
            route.source_rank,
            route.destination_group_ref,
            route.destination_rank,
            route.die_path[0],
            route.die_path[-1],
        ) != (
            source_group.id,
            source_access.rank,
            destination_group.id,
            destination_access.rank,
            source_binding.die_id,
            destination_binding.die_id,
        ):
            raise SchemaError(
                "CrossGroupRoute group-local rank/home endpoints disagree with state accesses",
                path=f"{path}.cross_group_route_ref",
            )

        source_write_offset, source_write_shape = state_access_tensor_view(
            source_access,
            source_declaration,
            "write",
            path=f"{path}.source_state_access_ref",
        )
        destination_query_offset, destination_query_shape = (
            state_access_tensor_view(
                destination_access,
                destination_declaration,
                "write",
                path=f"{path}.destination_state_access_ref",
            )
        )
        source_heads = source_workload.rank_num_kv_heads
        destination_heads = destination_workload.rank_num_kv_heads
        head_dim = source_workload.head_dim
        history_tokens = source_write_shape[0]
        if (
            source_write_offset != (0, 0, 0)
            or source_write_shape
            != (history_tokens, source_heads, head_dim)
            or destination_query_offset
            != (history_tokens, 0, 0)
            or destination_query_shape[1:]
            != (destination_heads, head_dim)
        ):
            raise SchemaError(
                "source history and destination query WRITE views must be disjoint, adjacent, and full-head THD slices",
                path=path,
            )

        source_global_start = source_access.rank * source_heads
        destination_global_start = (
            destination_access.rank * destination_heads
        )
        global_start = max(source_global_start, destination_global_start)
        global_end = min(
            source_global_start + source_heads,
            destination_global_start + destination_heads,
        )
        if global_end <= global_start:
            raise SchemaError(
                "route endpoint KV head shards do not intersect", path=path
            )
        head_count = global_end - global_start
        expected_source_offset = (
            0,
            global_start - source_global_start,
            0,
        )
        expected_destination_offset = (
            0,
            global_start - destination_global_start,
            0,
        )
        expected_shape = (history_tokens, head_count, head_dim)
        if (
            self.source_local_offset != expected_source_offset
            or self.destination_local_offset
            != expected_destination_offset
            or self.source_local_shape != expected_shape
            or self.destination_local_shape != expected_shape
        ):
            raise SchemaError(
                "local slices must equal the exact global KV-head intersection",
                path=path,
            )

        element_bytes = {
            DType.FP16: 2,
            DType.FP32: 4,
        }.get(source_declaration.dtype)
        if element_bytes is None:
            raise SchemaError("unsupported KV dtype", path=path)
        expected_bytes = math.prod(expected_shape) * element_bytes
        if self.bytes != expected_bytes:
            raise SchemaError(
                f"bytes must equal exact sliced payload {expected_bytes}",
                path=f"{path}.bytes",
            )


@dataclass(frozen=True, slots=True)
class KvTransferSegment:
    """One pair of equally sized, row-major contiguous THD spans."""

    source_local_offset: tuple[int, ...]
    source_local_shape: tuple[int, ...]
    destination_local_offset: tuple[int, ...]
    destination_local_shape: tuple[int, ...]
    bytes: int

    def validate(self, path: str = "kv_transfer_segment") -> None:
        _validate_local_slice(
            self.source_local_offset,
            self.source_local_shape,
            path=f"{path}.source_local",
        )
        _validate_local_slice(
            self.destination_local_offset,
            self.destination_local_shape,
            path=f"{path}.destination_local",
        )
        if self.source_local_shape != self.destination_local_shape:
            raise SchemaError(
                "source and destination segment shapes must match",
                path=path,
            )
        validate_uint64(self.bytes, f"{path}.bytes")
        if self.bytes == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.bytes")


def _per_token_segments(
    logical_slice: SlicedKvStateTransferContract,
) -> tuple[KvTransferSegment, ...]:
    source_offset = logical_slice.source_local_offset
    destination_offset = logical_slice.destination_local_offset
    shape = logical_slice.source_local_shape
    if (
        len(source_offset) != 3
        or len(destination_offset) != 3
        or len(shape) != 3
        or logical_slice.destination_local_shape != shape
    ):
        raise SchemaError(
            "segmented KV transfer requires equal rank-3 THD logical slices",
            path="logical_slice",
        )
    token_count, head_count, head_dim = shape
    if logical_slice.bytes % token_count:
        raise SchemaError(
            "logical slice bytes must divide evenly across tokens",
            path="logical_slice.bytes",
        )
    segment_bytes = logical_slice.bytes // token_count
    return tuple(
        KvTransferSegment(
            source_local_offset=(
                source_offset[0] + token_index,
                source_offset[1],
                source_offset[2],
            ),
            source_local_shape=(1, head_count, head_dim),
            destination_local_offset=(
                destination_offset[0] + token_index,
                destination_offset[1],
                destination_offset[2],
            ),
            destination_local_shape=(1, head_count, head_dim),
            bytes=segment_bytes,
        )
        for token_index in range(token_count)
    )


@dataclass(frozen=True, slots=True)
class SegmentedKvStateTransferContract:
    """One logical KV slice represented by exact contiguous token segments."""

    schema_version: str
    producer_pass: str
    id: str
    logical_slice: SlicedKvStateTransferContract
    segments: tuple[KvTransferSegment, ...]
    bytes: int

    @classmethod
    def from_logical_slice(
        cls,
        *,
        producer_pass: str,
        logical_slice: SlicedKvStateTransferContract,
    ) -> "SegmentedKvStateTransferContract":
        if type(logical_slice) is not SlicedKvStateTransferContract:
            raise SchemaError("must be a SlicedKvStateTransferContract", path="logical_slice")
        logical_slice.validate("logical_slice")
        return cls.create(
            producer_pass=producer_pass,
            logical_slice=logical_slice,
            segments=_per_token_segments(logical_slice),
            bytes=logical_slice.bytes,
        )

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        logical_slice: SlicedKvStateTransferContract,
        segments: tuple[KvTransferSegment, ...],
        bytes: int,
    ) -> "SegmentedKvStateTransferContract":
        semantic_key = {
            "logical_slice": logical_slice,
            "segments": segments,
            "bytes": bytes,
        }
        result = cls(
            schema_version=SEGMENTED_KV_STATE_TRANSFER_CONTRACT_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "segmented_kv_state_transfer_contract",
                semantic_key,
                schema_version=SEGMENTED_KV_STATE_TRANSFER_CONTRACT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "logical_slice": self.logical_slice,
            "segments": self.segments,
            "bytes": self.bytes,
        }

    def validate(
        self, path: str = "segmented_kv_state_transfer_contract"
    ) -> None:
        if self.schema_version != SEGMENTED_KV_STATE_TRANSFER_CONTRACT_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        if type(self.logical_slice) is not SlicedKvStateTransferContract:
            raise SchemaError(
                "must be a SlicedKvStateTransferContract",
                path=f"{path}.logical_slice",
            )
        self.logical_slice.validate(f"{path}.logical_slice")
        if type(self.segments) is not tuple or not self.segments:
            raise SchemaError(
                "segments must be a non-empty immutable tuple",
                path=f"{path}.segments",
            )
        for index, segment in enumerate(self.segments):
            if type(segment) is not KvTransferSegment:
                raise SchemaError(
                    "must be a KvTransferSegment",
                    path=f"{path}.segments[{index}]",
                )
            segment.validate(f"{path}.segments[{index}]")
        validate_uint64(self.bytes, f"{path}.bytes")
        if self.bytes == 0 or self.bytes != sum(
            segment.bytes for segment in self.segments
        ):
            raise SchemaError(
                "bytes must equal the segment byte sum",
                path=f"{path}.bytes",
            )
        expected_id = stable_artifact_id(
            "segmented_kv_state_transfer_contract",
            self._semantic_key(),
            schema_version=SEGMENTED_KV_STATE_TRANSFER_CONTRACT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        ir1: IR1,
        path: str = "segmented_kv_state_transfer_contract",
    ) -> None:
        self.validate(path)
        self.logical_slice.validate_against(ir1, f"{path}.logical_slice")
        if self.segments != _per_token_segments(self.logical_slice):
            raise SchemaError(
                (
                    "segments must exactly partition the logical slice by "
                    "token without gaps or overlap"
                ),
                path=f"{path}.segments",
            )
        if self.bytes != self.logical_slice.bytes:
            raise SchemaError(
                "bytes must equal the original logical slice",
                path=f"{path}.bytes",
            )

    @property
    def source_ir1_id(self) -> str:
        return self.logical_slice.source_ir1_id

    @property
    def source_state_access_ref(self) -> str:
        return self.logical_slice.source_state_access_ref

    @property
    def destination_state_access_ref(self) -> str:
        return self.logical_slice.destination_state_access_ref

    @property
    def cross_group_route_ref(self) -> str:
        return self.logical_slice.cross_group_route_ref

    @property
    def source_local_offset(self) -> tuple[int, ...]:
        return self.logical_slice.source_local_offset

    @property
    def source_local_shape(self) -> tuple[int, ...]:
        return self.logical_slice.source_local_shape

    @property
    def destination_local_offset(self) -> tuple[int, ...]:
        return self.logical_slice.destination_local_offset

    @property
    def destination_local_shape(self) -> tuple[int, ...]:
        return self.logical_slice.destination_local_shape


StateTransferLike = (
    StateTransferContract
    | SlicedKvStateTransferContract
    | SegmentedKvStateTransferContract
)


__all__ = [
    "SEGMENTED_KV_STATE_TRANSFER_CONTRACT_SCHEMA_VERSION",
    "SLICED_KV_STATE_TRANSFER_CONTRACT_SCHEMA_VERSION",
    "STATE_TRANSFER_CONTRACT_SCHEMA_VERSION",
    "KvTransferSegment",
    "SegmentedKvStateTransferContract",
    "SlicedKvStateTransferContract",
    "StateTransferContract",
    "StateTransferLike",
]
