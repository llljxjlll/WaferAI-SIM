"""Versioned contracts for deterministic hierarchical-memory planning.

This module describes SRAM, HBM, and external-memory residency.  The v1
contract deliberately treats external memory as schema-only: no frontend
lowering or simulator transport is implied by an external allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import UINT64_MAX, stable_artifact_id, validate_nonempty, validate_uint64


MEMORY_CAPACITY_SCHEMA_VERSION = "wafer_frontend.memory_capacity/v1alpha1"
MEMORY_STATE_VERSION_SCHEMA_VERSION = "wafer_frontend.memory_state_version/v1alpha1"
MEMORY_ALLOCATION_REQUEST_SCHEMA_VERSION = (
    "wafer_frontend.memory_allocation_request/v1alpha1"
)
MEMORY_ALLOCATION_SCHEMA_VERSION = "wafer_frontend.memory_allocation/v1alpha1"
MEMORY_RESIDENCY_SCHEMA_VERSION = "wafer_frontend.memory_residency/v1alpha1"
MEMORY_PEAK_SCHEMA_VERSION = "wafer_frontend.memory_peak/v1alpha1"
MEMORY_PLAN_SCHEMA_VERSION = "wafer_frontend.memory_plan/v1alpha1"


class MemoryTier(str, Enum):
    SRAM = "sram"
    HBM = "hbm"
    EXTERNAL = "external"


class MemoryObjectKind(str, Enum):
    PARAMETER = "parameter"
    KV = "kv"
    ACTIVATION = "activation"
    GRADIENT = "gradient"
    OPTIMIZER = "optimizer"
    MOE_BUFFER = "moe_buffer"
    COMMUNICATION = "communication"
    WORKSPACE = "workspace"


class ResidencyStatus(str, Enum):
    TRANSFERRING = "transferring"
    CLEAN = "clean"
    DIRTY = "dirty"


class MemoryPlanExecution(str, Enum):
    NO_EXTERNAL_TRANSPORT_REQUIRED = "no_external_transport_required"
    SCHEMA_ONLY_EXTERNAL = "schema_only_external"


def _positive(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value == 0:
        raise SchemaError("must be greater than zero", path=path)


def _power_of_two(value: int, path: str) -> None:
    _positive(value, path)
    if value & (value - 1):
        raise SchemaError("must be a power of two", path=path)


def _range_end(start: int, size: int, path: str) -> int:
    validate_uint64(start, f"{path}.start")
    _positive(size, f"{path}.size_bytes")
    if start > UINT64_MAX - size:
        raise SchemaError("byte range overflows uint64", path=path)
    return start + size


def _stable_id(kind: str, key: object, version: str) -> str:
    return stable_artifact_id(kind, key, schema_version=version)


@dataclass(frozen=True, slots=True)
class MemoryTierCapacity:
    id: str
    tier: MemoryTier
    location_ref: str
    base_address: int
    capacity_bytes: int
    alignment_bytes: int

    @classmethod
    def create(
        cls,
        *,
        tier: MemoryTier,
        location_ref: str,
        base_address: int,
        capacity_bytes: int,
        alignment_bytes: int,
    ) -> "MemoryTierCapacity":
        key = {
            "tier": tier,
            "location_ref": location_ref,
            "base_address": base_address,
            "capacity_bytes": capacity_bytes,
            "alignment_bytes": alignment_bytes,
        }
        result = cls(
            id=_stable_id("memory_capacity", key, MEMORY_CAPACITY_SCHEMA_VERSION),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "location_ref": self.location_ref,
            "base_address": self.base_address,
            "capacity_bytes": self.capacity_bytes,
            "alignment_bytes": self.alignment_bytes,
        }

    def validate(self, path: str = "memory_capacity") -> None:
        if type(self.tier) is not MemoryTier:
            raise SchemaError("must be a MemoryTier", path=f"{path}.tier")
        validate_nonempty(self.location_ref, f"{path}.location_ref")
        _power_of_two(self.alignment_bytes, f"{path}.alignment_bytes")
        _range_end(self.base_address, self.capacity_bytes, path)
        if self.base_address % self.alignment_bytes:
            raise SchemaError("must satisfy alignment", path=f"{path}.base_address")
        if self.capacity_bytes % self.alignment_bytes:
            raise SchemaError("must satisfy alignment", path=f"{path}.capacity_bytes")
        expected = _stable_id("memory_capacity", self._key(), MEMORY_CAPACITY_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MemoryStateVersion:
    id: str
    state_ref: str
    generation: int
    predecessor_ref: str | None
    writable: bool

    @classmethod
    def create(
        cls,
        *,
        state_ref: str,
        generation: int,
        predecessor_ref: str | None,
        writable: bool,
    ) -> "MemoryStateVersion":
        key = {
            "state_ref": state_ref,
            "generation": generation,
            "predecessor_ref": predecessor_ref,
            "writable": writable,
        }
        result = cls(
            id=_stable_id(
                "memory_state_version", key, MEMORY_STATE_VERSION_SCHEMA_VERSION
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "state_ref": self.state_ref,
            "generation": self.generation,
            "predecessor_ref": self.predecessor_ref,
            "writable": self.writable,
        }

    def validate(self, path: str = "memory_state_version") -> None:
        validate_nonempty(self.state_ref, f"{path}.state_ref")
        validate_uint64(self.generation, f"{path}.generation")
        if type(self.writable) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.writable")
        if self.generation == 0:
            if self.predecessor_ref is not None:
                raise SchemaError("generation zero cannot have a predecessor", path=f"{path}.predecessor_ref")
        else:
            if self.predecessor_ref is None:
                raise SchemaError("nonzero generation requires a predecessor", path=f"{path}.predecessor_ref")
            validate_nonempty(self.predecessor_ref, f"{path}.predecessor_ref")
            if not self.writable:
                raise SchemaError("read-only state cannot have a new generation", path=path)
        expected = _stable_id(
            "memory_state_version", self._key(), MEMORY_STATE_VERSION_SCHEMA_VERSION
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MemoryAllocationRequest:
    id: str
    state_version_ref: str
    object_kind: MemoryObjectKind
    tier: MemoryTier
    location_ref: str
    size_bytes: int
    alignment_bytes: int
    lifetime_start: int
    lifetime_end_exclusive: int
    pinned_address: int | None

    @classmethod
    def create(
        cls,
        *,
        state_version_ref: str,
        object_kind: MemoryObjectKind,
        tier: MemoryTier,
        location_ref: str,
        size_bytes: int,
        alignment_bytes: int,
        lifetime_start: int,
        lifetime_end_exclusive: int,
        pinned_address: int | None = None,
    ) -> "MemoryAllocationRequest":
        key = {
            "state_version_ref": state_version_ref,
            "object_kind": object_kind,
            "tier": tier,
            "location_ref": location_ref,
            "size_bytes": size_bytes,
            "alignment_bytes": alignment_bytes,
            "lifetime_start": lifetime_start,
            "lifetime_end_exclusive": lifetime_end_exclusive,
            "pinned_address": pinned_address,
        }
        result = cls(
            id=_stable_id(
                "memory_allocation_request",
                key,
                MEMORY_ALLOCATION_REQUEST_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "state_version_ref": self.state_version_ref,
            "object_kind": self.object_kind,
            "tier": self.tier,
            "location_ref": self.location_ref,
            "size_bytes": self.size_bytes,
            "alignment_bytes": self.alignment_bytes,
            "lifetime_start": self.lifetime_start,
            "lifetime_end_exclusive": self.lifetime_end_exclusive,
            "pinned_address": self.pinned_address,
        }

    def validate(self, path: str = "memory_allocation_request") -> None:
        validate_nonempty(self.state_version_ref, f"{path}.state_version_ref")
        if type(self.object_kind) is not MemoryObjectKind:
            raise SchemaError("must be a MemoryObjectKind", path=f"{path}.object_kind")
        if type(self.tier) is not MemoryTier:
            raise SchemaError("must be a MemoryTier", path=f"{path}.tier")
        validate_nonempty(self.location_ref, f"{path}.location_ref")
        _positive(self.size_bytes, f"{path}.size_bytes")
        _power_of_two(self.alignment_bytes, f"{path}.alignment_bytes")
        validate_uint64(self.lifetime_start, f"{path}.lifetime_start")
        validate_uint64(self.lifetime_end_exclusive, f"{path}.lifetime_end_exclusive")
        if self.lifetime_end_exclusive <= self.lifetime_start:
            raise SchemaError("must be greater than lifetime_start", path=f"{path}.lifetime_end_exclusive")
        if self.pinned_address is not None:
            validate_uint64(self.pinned_address, f"{path}.pinned_address")
            if self.pinned_address % self.alignment_bytes:
                raise SchemaError("must satisfy request alignment", path=f"{path}.pinned_address")
        expected = _stable_id(
            "memory_allocation_request",
            self._key(),
            MEMORY_ALLOCATION_REQUEST_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MemoryAllocation:
    id: str
    request_ref: str
    address: int
    reserved_bytes: int

    @classmethod
    def create(
        cls, *, request_ref: str, address: int, reserved_bytes: int
    ) -> "MemoryAllocation":
        key = {
            "request_ref": request_ref,
            "address": address,
            "reserved_bytes": reserved_bytes,
        }
        result = cls(
            id=_stable_id("memory_allocation", key, MEMORY_ALLOCATION_SCHEMA_VERSION),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "request_ref": self.request_ref,
            "address": self.address,
            "reserved_bytes": self.reserved_bytes,
        }

    def validate(self, path: str = "memory_allocation") -> None:
        validate_nonempty(self.request_ref, f"{path}.request_ref")
        _range_end(self.address, self.reserved_bytes, path)
        expected = _stable_id("memory_allocation", self._key(), MEMORY_ALLOCATION_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MemoryResidency:
    id: str
    state_version_ref: str
    allocation_ref: str
    status: ResidencyStatus
    valid_from: int
    valid_until_exclusive: int

    @classmethod
    def create(
        cls,
        *,
        state_version_ref: str,
        allocation_ref: str,
        status: ResidencyStatus,
        valid_from: int,
        valid_until_exclusive: int,
    ) -> "MemoryResidency":
        key = {
            "state_version_ref": state_version_ref,
            "allocation_ref": allocation_ref,
            "status": status,
            "valid_from": valid_from,
            "valid_until_exclusive": valid_until_exclusive,
        }
        result = cls(
            id=_stable_id("memory_residency", key, MEMORY_RESIDENCY_SCHEMA_VERSION),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "state_version_ref": self.state_version_ref,
            "allocation_ref": self.allocation_ref,
            "status": self.status,
            "valid_from": self.valid_from,
            "valid_until_exclusive": self.valid_until_exclusive,
        }

    def validate(self, path: str = "memory_residency") -> None:
        validate_nonempty(self.state_version_ref, f"{path}.state_version_ref")
        validate_nonempty(self.allocation_ref, f"{path}.allocation_ref")
        if type(self.status) is not ResidencyStatus:
            raise SchemaError("must be a ResidencyStatus", path=f"{path}.status")
        validate_uint64(self.valid_from, f"{path}.valid_from")
        validate_uint64(self.valid_until_exclusive, f"{path}.valid_until_exclusive")
        if self.valid_until_exclusive <= self.valid_from:
            raise SchemaError("must be greater than valid_from", path=f"{path}.valid_until_exclusive")
        expected = _stable_id("memory_residency", self._key(), MEMORY_RESIDENCY_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MemoryPeak:
    id: str
    capacity_ref: str
    at_tick: int
    peak_bytes: int
    active_allocation_refs: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        capacity_ref: str,
        at_tick: int,
        peak_bytes: int,
        active_allocation_refs: tuple[str, ...],
    ) -> "MemoryPeak":
        key = {
            "capacity_ref": capacity_ref,
            "at_tick": at_tick,
            "peak_bytes": peak_bytes,
            "active_allocation_refs": tuple(sorted(active_allocation_refs)),
        }
        result = cls(
            id=_stable_id("memory_peak", key, MEMORY_PEAK_SCHEMA_VERSION),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "capacity_ref": self.capacity_ref,
            "at_tick": self.at_tick,
            "peak_bytes": self.peak_bytes,
            "active_allocation_refs": self.active_allocation_refs,
        }

    def validate(self, path: str = "memory_peak") -> None:
        validate_nonempty(self.capacity_ref, f"{path}.capacity_ref")
        validate_uint64(self.at_tick, f"{path}.at_tick")
        validate_uint64(self.peak_bytes, f"{path}.peak_bytes")
        if self.active_allocation_refs != tuple(sorted(self.active_allocation_refs)):
            raise SchemaError("must use canonical order", path=f"{path}.active_allocation_refs")
        if len(set(self.active_allocation_refs)) != len(self.active_allocation_refs):
            raise SchemaError("contains duplicates", path=f"{path}.active_allocation_refs")
        expected = _stable_id("memory_peak", self._key(), MEMORY_PEAK_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    schema_version: str
    producer_pass: str
    id: str
    execution: MemoryPlanExecution
    capacities: tuple[MemoryTierCapacity, ...]
    state_versions: tuple[MemoryStateVersion, ...]
    requests: tuple[MemoryAllocationRequest, ...]
    allocations: tuple[MemoryAllocation, ...]
    residencies: tuple[MemoryResidency, ...]
    peaks: tuple[MemoryPeak, ...]

    @classmethod
    def create(
        cls,
        *,
        capacities: tuple[MemoryTierCapacity, ...],
        state_versions: tuple[MemoryStateVersion, ...],
        requests: tuple[MemoryAllocationRequest, ...],
        allocations: tuple[MemoryAllocation, ...],
        residencies: tuple[MemoryResidency, ...],
        peaks: tuple[MemoryPeak, ...],
    ) -> "MemoryPlan":
        capacities = tuple(sorted(capacities, key=lambda item: (item.tier.value, item.location_ref, item.id)))
        state_versions = tuple(sorted(state_versions, key=lambda item: (item.state_ref, item.generation, item.id)))
        requests = tuple(sorted(requests, key=lambda item: item.id))
        allocations = tuple(sorted(allocations, key=lambda item: item.request_ref))
        residencies = tuple(
            sorted(
                residencies,
                key=lambda item: (
                    item.allocation_ref,
                    item.valid_from,
                    item.state_version_ref,
                    item.id,
                ),
            )
        )
        peaks = tuple(sorted(peaks, key=lambda item: item.capacity_ref))
        execution = (
            MemoryPlanExecution.SCHEMA_ONLY_EXTERNAL
            if any(item.tier is MemoryTier.EXTERNAL for item in requests)
            else MemoryPlanExecution.NO_EXTERNAL_TRANSPORT_REQUIRED
        )
        key = {
            "execution": execution,
            "capacities": capacities,
            "state_versions": state_versions,
            "requests": requests,
            "allocations": allocations,
            "residencies": residencies,
            "peaks": peaks,
        }
        result = cls(
            schema_version=MEMORY_PLAN_SCHEMA_VERSION,
            producer_pass="hierarchical_memory_planner",
            id=_stable_id("memory_plan", key, MEMORY_PLAN_SCHEMA_VERSION),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "execution": self.execution,
            "capacities": self.capacities,
            "state_versions": self.state_versions,
            "requests": self.requests,
            "allocations": self.allocations,
            "residencies": self.residencies,
            "peaks": self.peaks,
        }

    def validate(self, path: str = "memory_plan") -> None:
        if self.schema_version != MEMORY_PLAN_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "hierarchical_memory_planner":
            raise SchemaError("must be 'hierarchical_memory_planner'", path=f"{path}.producer_pass")
        expected_execution = (
            MemoryPlanExecution.SCHEMA_ONLY_EXTERNAL
            if any(item.tier is MemoryTier.EXTERNAL for item in self.requests)
            else MemoryPlanExecution.NO_EXTERNAL_TRANSPORT_REQUIRED
        )
        if self.execution is not expected_execution:
            raise SchemaError("does not match allocated tiers", path=f"{path}.execution")
        self._validate_canonical_order(path)
        capacities = self._validate_unique(self.capacities, f"{path}.capacities")
        versions = self._validate_unique(self.state_versions, f"{path}.state_versions")
        requests = self._validate_unique(self.requests, f"{path}.requests")
        allocations = self._validate_unique(self.allocations, f"{path}.allocations")
        residencies = self._validate_unique(self.residencies, f"{path}.residencies")
        peaks = self._validate_unique(self.peaks, f"{path}.peaks")
        self._validate_versions(versions, path)
        self._validate_allocations(capacities, versions, requests, allocations, path)
        self._validate_residencies(versions, requests, allocations, residencies, path)
        self._validate_peaks(capacities, requests, allocations, peaks, path)
        expected = _stable_id("memory_plan", self._key(), MEMORY_PLAN_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def _validate_canonical_order(self, path: str) -> None:
        expected = (
            tuple(sorted(self.capacities, key=lambda item: (item.tier.value, item.location_ref, item.id))),
            tuple(sorted(self.state_versions, key=lambda item: (item.state_ref, item.generation, item.id))),
            tuple(sorted(self.requests, key=lambda item: item.id)),
            tuple(sorted(self.allocations, key=lambda item: item.request_ref)),
            tuple(
                sorted(
                    self.residencies,
                    key=lambda item: (
                        item.allocation_ref,
                        item.valid_from,
                        item.state_version_ref,
                        item.id,
                    ),
                )
            ),
            tuple(sorted(self.peaks, key=lambda item: item.capacity_ref)),
        )
        actual = (self.capacities, self.state_versions, self.requests, self.allocations, self.residencies, self.peaks)
        names = ("capacities", "state_versions", "requests", "allocations", "residencies", "peaks")
        for name, actual_items, expected_items in zip(names, actual, expected):
            if actual_items != expected_items:
                raise SchemaError("must use canonical order", path=f"{path}.{name}")

    @staticmethod
    def _validate_unique(items: tuple[object, ...], path: str) -> dict[str, object]:
        result: dict[str, object] = {}
        for index, item in enumerate(items):
            validator = getattr(item, "validate", None)
            if validator is None:
                raise SchemaError("has unsupported item type", path=f"{path}[{index}]")
            validator(f"{path}[{index}]")
            item_id = getattr(item, "id")
            if item_id in result:
                raise SchemaError("duplicate id", path=f"{path}[{index}].id")
            result[item_id] = item
        return result

    @staticmethod
    def _validate_versions(versions: dict[str, object], path: str) -> None:
        by_state_generation: dict[tuple[str, int], MemoryStateVersion] = {}
        for item in versions.values():
            assert isinstance(item, MemoryStateVersion)
            key = (item.state_ref, item.generation)
            if key in by_state_generation:
                raise SchemaError("duplicate state generation", path=f"{path}.state_versions")
            by_state_generation[key] = item
        for item in by_state_generation.values():
            if item.generation == 0:
                continue
            predecessor = versions.get(item.predecessor_ref or "")
            if not isinstance(predecessor, MemoryStateVersion):
                raise SchemaError("references an unknown predecessor", path=f"{path}.state_versions")
            if predecessor.state_ref != item.state_ref or predecessor.generation + 1 != item.generation:
                raise SchemaError("predecessor must be the prior generation of the same state", path=f"{path}.state_versions")

    @staticmethod
    def _validate_allocations(capacities, versions, requests, allocations, path: str) -> None:
        capacity_by_location = {(item.tier, item.location_ref): item for item in capacities.values()}
        if len(capacity_by_location) != len(capacities):
            raise SchemaError("duplicate tier/location capacity", path=f"{path}.capacities")
        allocation_by_request: dict[str, MemoryAllocation] = {}
        ranges: dict[str, list[tuple[int, int, int, int]]] = {}
        for allocation in allocations.values():
            assert isinstance(allocation, MemoryAllocation)
            request = requests.get(allocation.request_ref)
            if not isinstance(request, MemoryAllocationRequest):
                raise SchemaError("references an unknown request", path=f"{path}.allocations")
            if request.state_version_ref not in versions:
                raise SchemaError("request references an unknown state version", path=f"{path}.requests")
            capacity = capacity_by_location.get((request.tier, request.location_ref))
            if capacity is None:
                raise SchemaError("request has no matching capacity", path=f"{path}.requests")
            if allocation.request_ref in allocation_by_request:
                raise SchemaError("request has multiple allocations", path=f"{path}.allocations")
            if allocation.address % max(capacity.alignment_bytes, request.alignment_bytes):
                raise SchemaError("allocation address violates alignment", path=f"{path}.allocations")
            if request.pinned_address is not None and allocation.address != request.pinned_address:
                raise SchemaError("allocation moved a pinned request", path=f"{path}.allocations")
            if allocation.reserved_bytes < request.size_bytes or allocation.reserved_bytes % request.alignment_bytes:
                raise SchemaError("reserved bytes do not cover aligned request", path=f"{path}.allocations")
            end = allocation.address + allocation.reserved_bytes
            if allocation.address < capacity.base_address or end > capacity.base_address + capacity.capacity_bytes:
                raise SchemaError("allocation exceeds capacity", path=f"{path}.allocations", code="memory_capacity_exceeded")
            for old_start, old_end, old_life_start, old_life_end in ranges.setdefault(capacity.id, []):
                spatial = allocation.address < old_end and old_start < end
                temporal = request.lifetime_start < old_life_end and old_life_start < request.lifetime_end_exclusive
                if spatial and temporal:
                    raise SchemaError("overlapping live allocations", path=f"{path}.allocations")
            ranges[capacity.id].append((allocation.address, end, request.lifetime_start, request.lifetime_end_exclusive))
            allocation_by_request[allocation.request_ref] = allocation
        if set(allocation_by_request) != set(requests):
            raise SchemaError("allocations must exactly cover requests", path=f"{path}.allocations")

    @staticmethod
    def _validate_residencies(versions, requests, allocations, residencies, path: str) -> None:
        by_allocation: dict[str, list[MemoryResidency]] = {}
        for residency in residencies.values():
            assert isinstance(residency, MemoryResidency)
            allocation = allocations.get(residency.allocation_ref)
            if not isinstance(allocation, MemoryAllocation):
                raise SchemaError("references an unknown allocation", path=f"{path}.residencies")
            request = requests[allocation.request_ref]
            version = versions.get(residency.state_version_ref)
            request_version = versions.get(request.state_version_ref)
            if (
                not isinstance(version, MemoryStateVersion)
                or not isinstance(request_version, MemoryStateVersion)
                or version.state_ref != request_version.state_ref
            ):
                raise SchemaError(
                    "residency must carry a version of the allocation state",
                    path=f"{path}.residencies",
                )
            if residency.valid_from < request.lifetime_start or residency.valid_until_exclusive > request.lifetime_end_exclusive:
                raise SchemaError("residency exceeds allocation lifetime", path=f"{path}.residencies")
            if residency.status is ResidencyStatus.DIRTY and not version.writable:
                raise SchemaError("read-only state cannot be dirty", path=f"{path}.residencies")
            by_allocation.setdefault(residency.allocation_ref, []).append(
                residency
            )
        if set(by_allocation) != set(allocations):
            raise SchemaError("residencies must exactly cover allocations", path=f"{path}.residencies")
        for allocation_ref, segments in by_allocation.items():
            allocation = allocations[allocation_ref]
            request = requests[allocation.request_ref]
            segments.sort(
                key=lambda item: (
                    item.valid_from,
                    item.valid_until_exclusive,
                    item.id,
                )
            )
            cursor = request.lifetime_start
            prior_version: MemoryStateVersion | None = None
            for segment in segments:
                if segment.valid_from != cursor:
                    raise SchemaError(
                        "residencies must contiguously cover allocation lifetime",
                        path=f"{path}.residencies",
                    )
                version = versions[segment.state_version_ref]
                assert isinstance(version, MemoryStateVersion)
                if prior_version is not None and (
                    version.id != prior_version.id
                    and (
                        version.predecessor_ref != prior_version.id
                        or version.generation != prior_version.generation + 1
                    )
                ):
                    raise SchemaError(
                        "residency version must stay equal or advance one generation",
                        path=f"{path}.residencies",
                    )
                prior_version = version
                cursor = segment.valid_until_exclusive
            if cursor != request.lifetime_end_exclusive:
                raise SchemaError(
                    "residencies must contiguously cover allocation lifetime",
                    path=f"{path}.residencies",
                )

    @staticmethod
    def _validate_peaks(capacities, requests, allocations, peaks, path: str) -> None:
        peak_by_capacity: dict[str, MemoryPeak] = {}
        allocations_by_capacity: dict[str, list[tuple[MemoryAllocation, MemoryAllocationRequest]]] = {item_id: [] for item_id in capacities}
        capacity_by_location = {(item.tier, item.location_ref): item for item in capacities.values()}
        for allocation in allocations.values():
            request = requests[allocation.request_ref]
            capacity = capacity_by_location[(request.tier, request.location_ref)]
            allocations_by_capacity[capacity.id].append((allocation, request))
        for peak in peaks.values():
            if peak.capacity_ref not in capacities:
                raise SchemaError("references an unknown capacity", path=f"{path}.peaks")
            if peak.capacity_ref in peak_by_capacity:
                raise SchemaError("capacity has multiple peak records", path=f"{path}.peaks")
            candidates = allocations_by_capacity[peak.capacity_ref]
            ticks = sorted({request.lifetime_start for _, request in candidates})
            expected_bytes = 0
            expected_tick = 0
            expected_refs: tuple[str, ...] = ()
            for tick in ticks:
                active = tuple(sorted(allocation.id for allocation, request in candidates if request.lifetime_start <= tick < request.lifetime_end_exclusive))
                active_bytes = sum(allocation.reserved_bytes for allocation, request in candidates if request.lifetime_start <= tick < request.lifetime_end_exclusive)
                if active_bytes > expected_bytes:
                    expected_bytes, expected_tick, expected_refs = active_bytes, tick, active
            if (peak.peak_bytes, peak.at_tick, peak.active_allocation_refs) != (expected_bytes, expected_tick, expected_refs):
                raise SchemaError("does not match recomputed lifetime peak", path=f"{path}.peaks")
            capacity = capacities[peak.capacity_ref]
            if peak.peak_bytes > capacity.capacity_bytes:
                raise SchemaError("peak exceeds capacity", path=f"{path}.peaks", code="memory_capacity_exceeded")
            peak_by_capacity[peak.capacity_ref] = peak
        if set(peak_by_capacity) != set(capacities):
            raise SchemaError("peaks must exactly cover capacities", path=f"{path}.peaks")


__all__ = [
    "MEMORY_PLAN_SCHEMA_VERSION",
    "MemoryAllocation",
    "MemoryAllocationRequest",
    "MemoryObjectKind",
    "MemoryPeak",
    "MemoryPlan",
    "MemoryPlanExecution",
    "MemoryResidency",
    "MemoryStateVersion",
    "MemoryTier",
    "MemoryTierCapacity",
    "ResidencyStatus",
]
