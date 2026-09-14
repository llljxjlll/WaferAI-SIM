"""Contracts for deterministic blocking HBM residency and offload."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .external_memory import (
    ExternalMemoryFabric,
    ExternalTransferReport,
    ExternalTransferRequest,
)
from .memory_plan import (
    MemoryObjectKind,
    MemoryPlan,
    MemoryStateVersion,
    MemoryTier,
)
from .serde import canonical_digest


OFFLOAD_CHUNK_SCHEMA_VERSION = "wafer_frontend.offload_chunk/v1alpha1"
OFFLOAD_STATE_MAPPING_SCHEMA_VERSION = (
    "wafer_frontend.offload_state_mapping/v1alpha1"
)
OFFLOAD_EVENT_SCHEMA_VERSION = "wafer_frontend.offload_event/v1alpha1"
OFFLOAD_OPERATION_SCHEMA_VERSION = "wafer_frontend.offload_operation/v1alpha1"
OFFLOAD_STATS_SCHEMA_VERSION = "wafer_frontend.offload_stats/v1alpha1"
BLOCKING_OFFLOAD_PLAN_SCHEMA_VERSION = (
    "wafer_frontend.blocking_offload_plan/v1alpha2"
)
BLOCKING_OFFLOAD_EXECUTION_SCHEMA_VERSION = (
    "wafer_frontend.blocking_offload_execution/v1alpha1"
)


class OffloadEventKind(str, Enum):
    PIN = "pin"
    UNPIN = "unpin"
    READ = "read"
    WRITE = "write"


class OffloadResidencyRequirement(str, Enum):
    AUTO_BRING_IN = "auto_bring_in"
    MUST_ALREADY_RESIDENT = "must_already_resident"


class OffloadOperationKind(str, Enum):
    BRING_IN = "bring_in"
    DIRTY_WRITEBACK = "dirty_writeback"
    CLEAN_DISCARD = "clean_discard"
    PIN = "pin"
    UNPIN = "unpin"
    CONSUME_READ = "consume_read"
    CONSUME_WRITE = "consume_write"


def _positive(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value == 0:
        raise SchemaError("must be greater than zero", path=path)


def _stable_id(kind: str, key: object, version: str) -> str:
    return stable_artifact_id(kind, key, schema_version=version)


def _validate_digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


def _require_stable_id(
    actual: str,
    *,
    kind: str,
    key: object,
    version: str,
    path: str,
) -> None:
    expected = _stable_id(kind, key, version)
    if actual != expected:
        raise SchemaError(
            f"unstable artifact id; expected {expected!r}",
            path=f"{path}.id",
        )


@dataclass(frozen=True, slots=True)
class OffloadChunk:
    id: str
    initial_version: MemoryStateVersion
    object_kind: MemoryObjectKind
    size_bytes: int
    alignment_bytes: int
    external_capacity_ref: str
    external_address: int
    connection_ref: str

    @classmethod
    def create(
        cls,
        *,
        initial_version: MemoryStateVersion,
        object_kind: MemoryObjectKind,
        size_bytes: int,
        alignment_bytes: int,
        external_capacity_ref: str,
        external_address: int,
        connection_ref: str,
    ) -> "OffloadChunk":
        key = {
            "initial_version": initial_version,
            "object_kind": object_kind,
            "size_bytes": size_bytes,
            "alignment_bytes": alignment_bytes,
            "external_capacity_ref": external_capacity_ref,
            "external_address": external_address,
            "connection_ref": connection_ref,
        }
        result = cls(
            id=_stable_id(
                "offload_chunk",
                key,
                OFFLOAD_CHUNK_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "initial_version": self.initial_version,
            "object_kind": self.object_kind,
            "size_bytes": self.size_bytes,
            "alignment_bytes": self.alignment_bytes,
            "external_capacity_ref": self.external_capacity_ref,
            "external_address": self.external_address,
            "connection_ref": self.connection_ref,
        }

    def validate(self, path: str = "offload_chunk") -> None:
        self.initial_version.validate(f"{path}.initial_version")
        if self.initial_version.generation != 0:
            raise SchemaError(
                "initial version must be generation zero",
                path=f"{path}.initial_version",
            )
        if type(self.object_kind) is not MemoryObjectKind:
            raise SchemaError(
                "must be a MemoryObjectKind",
                path=f"{path}.object_kind",
            )
        _positive(self.size_bytes, f"{path}.size_bytes")
        _positive(self.alignment_bytes, f"{path}.alignment_bytes")
        if self.alignment_bytes & (self.alignment_bytes - 1):
            raise SchemaError(
                "must be a power of two",
                path=f"{path}.alignment_bytes",
            )
        validate_nonempty(
            self.external_capacity_ref,
            f"{path}.external_capacity_ref",
        )
        validate_uint64(self.external_address, f"{path}.external_address")
        validate_nonempty(self.connection_ref, f"{path}.connection_ref")
        _require_stable_id(
            self.id,
            kind="offload_chunk",
            key=self._key(),
            version=OFFLOAD_CHUNK_SCHEMA_VERSION,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class OffloadStateMapping:
    id: str
    chunk_ref: str
    source_state_version_ref: str
    source_allocation_ref: str

    @classmethod
    def create(
        cls,
        *,
        chunk_ref: str,
        source_state_version_ref: str,
        source_allocation_ref: str,
    ) -> "OffloadStateMapping":
        key = {
            "chunk_ref": chunk_ref,
            "source_state_version_ref": source_state_version_ref,
            "source_allocation_ref": source_allocation_ref,
        }
        result = cls(
            id=_stable_id(
                "offload_state_mapping",
                key,
                OFFLOAD_STATE_MAPPING_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "chunk_ref": self.chunk_ref,
            "source_state_version_ref": self.source_state_version_ref,
            "source_allocation_ref": self.source_allocation_ref,
        }

    def validate(self, path: str = "offload_state_mapping") -> None:
        validate_nonempty(self.chunk_ref, f"{path}.chunk_ref")
        validate_nonempty(
            self.source_state_version_ref,
            f"{path}.source_state_version_ref",
        )
        validate_nonempty(
            self.source_allocation_ref,
            f"{path}.source_allocation_ref",
        )
        _require_stable_id(
            self.id,
            kind="offload_state_mapping",
            key=self._key(),
            version=OFFLOAD_STATE_MAPPING_SCHEMA_VERSION,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class OffloadTraceEvent:
    id: str
    ordinal: int
    kind: OffloadEventKind
    chunk_ref: str
    residency_requirement: OffloadResidencyRequirement
    deps: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        ordinal: int,
        kind: OffloadEventKind,
        chunk_ref: str,
        residency_requirement: OffloadResidencyRequirement = (
            OffloadResidencyRequirement.AUTO_BRING_IN
        ),
        deps: tuple[str, ...] = (),
    ) -> "OffloadTraceEvent":
        key = {
            "ordinal": ordinal,
            "kind": kind,
            "chunk_ref": chunk_ref,
            "residency_requirement": residency_requirement,
            "deps": deps,
        }
        result = cls(
            id=_stable_id(
                "offload_event",
                key,
                OFFLOAD_EVENT_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "kind": self.kind,
            "chunk_ref": self.chunk_ref,
            "residency_requirement": self.residency_requirement,
            "deps": self.deps,
        }

    def validate(self, path: str = "offload_event") -> None:
        validate_uint64(self.ordinal, f"{path}.ordinal")
        if type(self.kind) is not OffloadEventKind:
            raise SchemaError(
                "must be an OffloadEventKind",
                path=f"{path}.kind",
            )
        validate_nonempty(self.chunk_ref, f"{path}.chunk_ref")
        if type(self.residency_requirement) is not OffloadResidencyRequirement:
            raise SchemaError(
                "must be an OffloadResidencyRequirement",
                path=f"{path}.residency_requirement",
            )
        if (
            self.kind in (OffloadEventKind.PIN, OffloadEventKind.UNPIN)
            and self.residency_requirement
            is not OffloadResidencyRequirement.AUTO_BRING_IN
        ):
            raise SchemaError(
                "PIN/UNPIN cannot override residency requirement",
                path=f"{path}.residency_requirement",
            )
        if type(self.deps) is not tuple:
            raise SchemaError(
                "must be an immutable tuple",
                path=f"{path}.deps",
            )
        if len(set(self.deps)) != len(self.deps):
            raise SchemaError("contains duplicates", path=f"{path}.deps")
        for index, dependency in enumerate(self.deps):
            validate_nonempty(dependency, f"{path}.deps[{index}]")
        _require_stable_id(
            self.id,
            kind="offload_event",
            key=self._key(),
            version=OFFLOAD_EVENT_SCHEMA_VERSION,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class OffloadOperation:
    id: str
    sequence: int
    kind: OffloadOperationKind
    chunk_ref: str
    trace_event_ref: str | None
    state_version_ref: str
    hbm_allocation_ref: str | None
    transfer_request_ref: str | None
    depends_on: tuple[str, ...]
    start_cycle: int
    ready_cycle: int
    pin_count_after: int
    dirty_after: bool

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        kind: OffloadOperationKind,
        chunk_ref: str,
        trace_event_ref: str | None,
        state_version_ref: str,
        hbm_allocation_ref: str | None,
        transfer_request_ref: str | None,
        depends_on: tuple[str, ...],
        start_cycle: int,
        ready_cycle: int,
        pin_count_after: int,
        dirty_after: bool,
    ) -> "OffloadOperation":
        key = {
            "sequence": sequence,
            "kind": kind,
            "chunk_ref": chunk_ref,
            "trace_event_ref": trace_event_ref,
            "state_version_ref": state_version_ref,
            "hbm_allocation_ref": hbm_allocation_ref,
            "transfer_request_ref": transfer_request_ref,
            "depends_on": depends_on,
            "start_cycle": start_cycle,
            "ready_cycle": ready_cycle,
            "pin_count_after": pin_count_after,
            "dirty_after": dirty_after,
        }
        result = cls(
            id=_stable_id(
                "offload_operation",
                key,
                OFFLOAD_OPERATION_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "offload_operation") -> None:
        validate_uint64(self.sequence, f"{path}.sequence")
        if type(self.kind) is not OffloadOperationKind:
            raise SchemaError(
                "must be an OffloadOperationKind",
                path=f"{path}.kind",
            )
        validate_nonempty(self.chunk_ref, f"{path}.chunk_ref")
        if self.trace_event_ref is not None:
            validate_nonempty(
                self.trace_event_ref,
                f"{path}.trace_event_ref",
            )
        validate_nonempty(
            self.state_version_ref,
            f"{path}.state_version_ref",
        )
        if self.hbm_allocation_ref is not None:
            validate_nonempty(
                self.hbm_allocation_ref,
                f"{path}.hbm_allocation_ref",
            )
        if self.transfer_request_ref is not None:
            validate_nonempty(
                self.transfer_request_ref,
                f"{path}.transfer_request_ref",
            )
        if type(self.depends_on) is not tuple:
            raise SchemaError(
                "must be an immutable tuple",
                path=f"{path}.depends_on",
            )
        if len(set(self.depends_on)) != len(self.depends_on):
            raise SchemaError(
                "contains duplicates",
                path=f"{path}.depends_on",
            )
        for index, dependency in enumerate(self.depends_on):
            validate_nonempty(
                dependency,
                f"{path}.depends_on[{index}]",
            )
        validate_uint64(self.start_cycle, f"{path}.start_cycle")
        validate_uint64(self.ready_cycle, f"{path}.ready_cycle")
        if self.ready_cycle < self.start_cycle:
            raise SchemaError(
                "must not precede start_cycle",
                path=f"{path}.ready_cycle",
            )
        validate_uint64(self.pin_count_after, f"{path}.pin_count_after")
        if type(self.dirty_after) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.dirty_after")
        is_transfer = self.kind in (
            OffloadOperationKind.BRING_IN,
            OffloadOperationKind.DIRTY_WRITEBACK,
        )
        if is_transfer != (self.transfer_request_ref is not None):
            raise SchemaError(
                "transfer reference must exactly match transfer operation",
                path=f"{path}.transfer_request_ref",
            )
        if is_transfer and self.ready_cycle <= self.start_cycle:
            raise SchemaError(
                "transfer must complete after it starts",
                path=f"{path}.ready_cycle",
            )
        _require_stable_id(
            self.id,
            kind="offload_operation",
            key=self._key(),
            version=OFFLOAD_OPERATION_SCHEMA_VERSION,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class BlockingOffloadStats:
    id: str
    bring_in_count: int
    dirty_writeback_count: int
    clean_discard_count: int
    consume_read_count: int
    consume_write_count: int
    transfer_bytes: int
    blocking_transfer_cycles: int
    hbm_peak_bytes: int
    max_pin_count: int
    final_resident_chunks: int
    final_dirty_chunks: int
    final_pin_count: int

    @classmethod
    def create(cls, **values: int) -> "BlockingOffloadStats":
        key = dict(values)
        result = cls(
            id=_stable_id(
                "offload_stats",
                key,
                OFFLOAD_STATS_SCHEMA_VERSION,
            ),
            **values,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "offload_stats") -> None:
        for name in self.__dataclass_fields__:
            if name != "id":
                validate_uint64(getattr(self, name), f"{path}.{name}")
        _require_stable_id(
            self.id,
            kind="offload_stats",
            key=self._key(),
            version=OFFLOAD_STATS_SCHEMA_VERSION,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class BlockingOffloadPlan:
    schema_version: str
    producer_pass: str
    id: str
    request_digest: str
    logical_graph_digest: str
    source_memory_plan_digest: str
    source_memory_plan: MemoryPlan
    state_mappings: tuple[OffloadStateMapping, ...]
    fabric: ExternalMemoryFabric
    chunks: tuple[OffloadChunk, ...]
    events: tuple[OffloadTraceEvent, ...]
    state_versions: tuple[MemoryStateVersion, ...]
    memory_plan: MemoryPlan
    transfer_requests: tuple[ExternalTransferRequest, ...]
    operations: tuple[OffloadOperation, ...]
    stats: BlockingOffloadStats
    simulator_runtime_integrated: bool

    @classmethod
    def create(
        cls,
        *,
        request_digest: str,
        logical_graph_digest: str,
        source_memory_plan_digest: str,
        source_memory_plan: MemoryPlan,
        state_mappings: tuple[OffloadStateMapping, ...],
        fabric: ExternalMemoryFabric,
        chunks: tuple[OffloadChunk, ...],
        events: tuple[OffloadTraceEvent, ...],
        state_versions: tuple[MemoryStateVersion, ...],
        memory_plan: MemoryPlan,
        transfer_requests: tuple[ExternalTransferRequest, ...],
        operations: tuple[OffloadOperation, ...],
        stats: BlockingOffloadStats,
    ) -> "BlockingOffloadPlan":
        state_mappings = tuple(
            sorted(state_mappings, key=lambda item: item.chunk_ref)
        )
        chunks = tuple(sorted(chunks, key=lambda item: item.id))
        events = tuple(
            sorted(events, key=lambda item: (item.ordinal, item.id))
        )
        state_versions = tuple(
            sorted(
                state_versions,
                key=lambda item: (
                    item.state_ref,
                    item.generation,
                    item.id,
                ),
            )
        )
        transfer_requests = tuple(
            sorted(
                transfer_requests,
                key=lambda item: (item.issue_cycle, item.id),
            )
        )
        operations = tuple(
            sorted(operations, key=lambda item: item.sequence)
        )
        key = {
            "request_digest": request_digest,
            "logical_graph_digest": logical_graph_digest,
            "source_memory_plan_digest": source_memory_plan_digest,
            "source_memory_plan": source_memory_plan,
            "state_mappings": state_mappings,
            "fabric": fabric,
            "chunks": chunks,
            "events": events,
            "state_versions": state_versions,
            "memory_plan": memory_plan,
            "transfer_requests": transfer_requests,
            "operations": operations,
            "stats": stats,
            "simulator_runtime_integrated": False,
        }
        result = cls(
            schema_version=BLOCKING_OFFLOAD_PLAN_SCHEMA_VERSION,
            producer_pass="blocking_offload_planner",
            id=_stable_id(
                "blocking_offload_plan",
                key,
                BLOCKING_OFFLOAD_PLAN_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "request_digest": self.request_digest,
            "logical_graph_digest": self.logical_graph_digest,
            "source_memory_plan_digest": self.source_memory_plan_digest,
            "source_memory_plan": self.source_memory_plan,
            "state_mappings": self.state_mappings,
            "fabric": self.fabric,
            "chunks": self.chunks,
            "events": self.events,
            "state_versions": self.state_versions,
            "memory_plan": self.memory_plan,
            "transfer_requests": self.transfer_requests,
            "operations": self.operations,
            "stats": self.stats,
            "simulator_runtime_integrated": self.simulator_runtime_integrated,
        }

    def validate(self, path: str = "blocking_offload_plan") -> None:
        if self.schema_version != BLOCKING_OFFLOAD_PLAN_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "blocking_offload_planner":
            raise SchemaError(
                "must be 'blocking_offload_planner'",
                path=f"{path}.producer_pass",
            )
        if self.simulator_runtime_integrated is not False:
            raise SchemaError(
                "blocking planner is not integrated with simulator runtime",
                path=f"{path}.simulator_runtime_integrated",
            )
        _validate_digest(self.request_digest, f"{path}.request_digest")
        _validate_digest(
            self.logical_graph_digest,
            f"{path}.logical_graph_digest",
        )
        _validate_digest(
            self.source_memory_plan_digest,
            f"{path}.source_memory_plan_digest",
        )
        self.source_memory_plan.validate(f"{path}.source_memory_plan")
        if self.source_memory_plan_digest != canonical_digest(
            self.source_memory_plan
        ):
            raise SchemaError(
                "does not match embedded source memory plan",
                path=f"{path}.source_memory_plan_digest",
                code="offload_source_digest_mismatch",
            )
        self.fabric.validate(f"{path}.fabric")
        self.memory_plan.validate(f"{path}.memory_plan")
        self.stats.validate(f"{path}.stats")
        for name in (
            "state_mappings",
            "chunks",
            "events",
            "state_versions",
            "transfer_requests",
            "operations",
        ):
            items = getattr(self, name)
            if not items:
                raise SchemaError("must not be empty", path=f"{path}.{name}")
            ids: set[str] = set()
            for index, item in enumerate(items):
                item.validate(f"{path}.{name}[{index}]")
                if item.id in ids:
                    raise SchemaError(
                        "duplicate id",
                        path=f"{path}.{name}[{index}].id",
                    )
                ids.add(item.id)
        if self.state_mappings != tuple(
            sorted(self.state_mappings, key=lambda item: item.chunk_ref)
        ):
            raise SchemaError(
                "must use canonical chunk order",
                path=f"{path}.state_mappings",
            )
        if self.chunks != tuple(sorted(self.chunks, key=lambda item: item.id)):
            raise SchemaError("must use canonical order", path=f"{path}.chunks")
        if self.events != tuple(
            sorted(self.events, key=lambda item: (item.ordinal, item.id))
        ):
            raise SchemaError("must use canonical order", path=f"{path}.events")
        if self.operations != tuple(
            sorted(self.operations, key=lambda item: item.sequence)
        ):
            raise SchemaError(
                "must use sequence order",
                path=f"{path}.operations",
            )
        if tuple(item.sequence for item in self.operations) != tuple(
            range(len(self.operations))
        ):
            raise SchemaError(
                "operation sequence must be contiguous from zero",
                path=f"{path}.operations",
            )
        self._validate_source_state_mappings(path)
        _require_stable_id(
            self.id,
            kind="blocking_offload_plan",
            key=self._key(),
            version=BLOCKING_OFFLOAD_PLAN_SCHEMA_VERSION,
            path=path,
        )

    def _validate_source_state_mappings(self, path: str) -> None:
        chunks = {item.id: item for item in self.chunks}
        source_versions = {
            item.id: item for item in self.source_memory_plan.state_versions
        }
        source_allocations = {
            item.id: item for item in self.source_memory_plan.allocations
        }
        source_requests = {
            item.id: item for item in self.source_memory_plan.requests
        }
        source_capacities = {
            item.id: item for item in self.source_memory_plan.capacities
        }
        external_capacities = {
            item.id: item for item in self.fabric.external_capacities
        }
        mapped_chunks: set[str] = set()
        mapped_versions: set[str] = set()
        mapped_allocations: set[str] = set()
        mapped_state_refs: set[str] = set()
        for index, mapping in enumerate(self.state_mappings):
            mapping_path = f"{path}.state_mappings[{index}]"
            chunk = chunks.get(mapping.chunk_ref)
            if chunk is None:
                raise SchemaError(
                    "references an unknown offload chunk",
                    path=f"{mapping_path}.chunk_ref",
                    code="offload_state_mapping_mismatch",
                )
            source_version = source_versions.get(
                mapping.source_state_version_ref
            )
            if source_version is None:
                raise SchemaError(
                    "references a version outside the source memory plan",
                    path=f"{mapping_path}.source_state_version_ref",
                    code="offload_state_mapping_mismatch",
                )
            source_allocation = source_allocations.get(
                mapping.source_allocation_ref
            )
            if source_allocation is None:
                raise SchemaError(
                    "references an allocation outside the source memory plan",
                    path=f"{mapping_path}.source_allocation_ref",
                    code="offload_state_mapping_mismatch",
                )
            source_request = source_requests[source_allocation.request_ref]
            if (
                mapping.chunk_ref in mapped_chunks
                or mapping.source_state_version_ref in mapped_versions
                or mapping.source_allocation_ref in mapped_allocations
            ):
                raise SchemaError(
                    "chunk, source version, and source allocation must map one-to-one",
                    path=mapping_path,
                    code="offload_state_mapping_mismatch",
                )
            if (
                source_request.state_version_ref != source_version.id
                or chunk.initial_version != source_version
                or source_request.object_kind is not chunk.object_kind
                or source_request.tier is not MemoryTier.EXTERNAL
                or source_request.size_bytes != chunk.size_bytes
                or source_request.alignment_bytes != chunk.alignment_bytes
                or source_allocation.address != chunk.external_address
                or chunk.external_capacity_ref not in external_capacities
                or chunk.external_capacity_ref not in source_capacities
                or source_capacities[chunk.external_capacity_ref]
                != external_capacities[chunk.external_capacity_ref]
                or source_request.location_ref
                != external_capacities[
                    chunk.external_capacity_ref
                ].location_ref
                or source_allocation.address
                < external_capacities[chunk.external_capacity_ref].base_address
                or source_allocation.address + source_allocation.reserved_bytes
                > (
                    external_capacities[chunk.external_capacity_ref].base_address
                    + external_capacities[
                        chunk.external_capacity_ref
                    ].capacity_bytes
                )
            ):
                raise SchemaError(
                    "chunk does not match its source version/allocation",
                    path=mapping_path,
                    code="offload_state_mapping_mismatch",
                )
            mapped_chunks.add(mapping.chunk_ref)
            mapped_versions.add(mapping.source_state_version_ref)
            mapped_allocations.add(mapping.source_allocation_ref)
            mapped_state_refs.add(source_version.state_ref)
        if mapped_chunks != set(chunks):
            raise SchemaError(
                "state mappings must exactly cover offload chunks",
                path=f"{path}.state_mappings",
                code="offload_state_mapping_mismatch",
            )
        if any(
            version.state_ref not in mapped_state_refs
            for version in self.state_versions
        ):
            raise SchemaError(
                "offload state version is not bound to the source workload",
                path=f"{path}.state_versions",
                code="offload_state_mapping_mismatch",
            )


@dataclass(frozen=True, slots=True)
class BlockingOffloadExecution:
    schema_version: str
    producer_pass: str
    id: str
    plan: BlockingOffloadPlan
    transfer_report: ExternalTransferReport
    simulator_runtime_integrated: bool

    @classmethod
    def create(
        cls,
        *,
        plan: BlockingOffloadPlan,
        transfer_report: ExternalTransferReport,
    ) -> "BlockingOffloadExecution":
        key = {
            "plan": plan,
            "transfer_report": transfer_report,
            "simulator_runtime_integrated": False,
        }
        result = cls(
            schema_version=BLOCKING_OFFLOAD_EXECUTION_SCHEMA_VERSION,
            producer_pass="blocking_offload_executor",
            id=_stable_id(
                "blocking_offload_execution",
                key,
                BLOCKING_OFFLOAD_EXECUTION_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "plan": self.plan,
            "transfer_report": self.transfer_report,
            "simulator_runtime_integrated": self.simulator_runtime_integrated,
        }

    def validate(self, path: str = "blocking_offload_execution") -> None:
        if self.schema_version != BLOCKING_OFFLOAD_EXECUTION_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "blocking_offload_executor":
            raise SchemaError(
                "must be 'blocking_offload_executor'",
                path=f"{path}.producer_pass",
            )
        if self.simulator_runtime_integrated is not False:
            raise SchemaError(
                "execution is not integrated with simulator runtime",
                path=f"{path}.simulator_runtime_integrated",
            )
        self.plan.validate(f"{path}.plan")
        self.transfer_report.validate(f"{path}.transfer_report")
        if self.transfer_report.fabric != self.plan.fabric:
            raise SchemaError(
                "transfer report fabric does not match plan",
                path=f"{path}.transfer_report.fabric",
            )
        if self.transfer_report.requests != self.plan.transfer_requests:
            raise SchemaError(
                "transfer report requests do not match plan",
                path=f"{path}.transfer_report.requests",
            )
        _require_stable_id(
            self.id,
            kind="blocking_offload_execution",
            key=self._key(),
            version=BLOCKING_OFFLOAD_EXECUTION_SCHEMA_VERSION,
            path=path,
        )


__all__ = [
    "BLOCKING_OFFLOAD_EXECUTION_SCHEMA_VERSION",
    "BLOCKING_OFFLOAD_PLAN_SCHEMA_VERSION",
    "OFFLOAD_CHUNK_SCHEMA_VERSION",
    "OFFLOAD_EVENT_SCHEMA_VERSION",
    "OFFLOAD_OPERATION_SCHEMA_VERSION",
    "OFFLOAD_STATE_MAPPING_SCHEMA_VERSION",
    "OFFLOAD_STATS_SCHEMA_VERSION",
    "BlockingOffloadExecution",
    "BlockingOffloadPlan",
    "BlockingOffloadStats",
    "OffloadChunk",
    "OffloadEventKind",
    "OffloadOperation",
    "OffloadOperationKind",
    "OffloadResidencyRequirement",
    "OffloadStateMapping",
    "OffloadTraceEvent",
]
