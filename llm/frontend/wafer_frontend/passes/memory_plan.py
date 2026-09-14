"""Deterministic allocation and peak accounting for hierarchical memory."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.common import UINT64_MAX
from ..schema.memory_plan import (
    MemoryAllocation,
    MemoryAllocationRequest,
    MemoryObjectKind,
    MemoryPeak,
    MemoryPlan,
    MemoryResidency,
    MemoryStateVersion,
    MemoryTier,
    MemoryTierCapacity,
    ResidencyStatus,
)
from ..schema.persistent_state import (
    PersistentStateAccess,
    PersistentStateManifest,
    StateKind,
)


def _align_up(value: int, alignment: int, *, path: str) -> int:
    remainder = value % alignment
    result = value if remainder == 0 else value + alignment - remainder
    if result > UINT64_MAX:
        raise SchemaError("aligned value overflows uint64", path=path)
    return result


def _lifetimes_overlap(
    left: MemoryAllocationRequest, right: MemoryAllocationRequest
) -> bool:
    return (
        left.lifetime_start < right.lifetime_end_exclusive
        and right.lifetime_start < left.lifetime_end_exclusive
    )


def _request_sort_key(request: MemoryAllocationRequest) -> tuple[object, ...]:
    return (
        request.tier.value,
        request.location_ref,
        0 if request.pinned_address is not None else 1,
        request.lifetime_start,
        request.lifetime_end_exclusive,
        request.id,
    )


def plan_hierarchical_memory(
    *,
    capacities: tuple[MemoryTierCapacity, ...],
    state_versions: tuple[MemoryStateVersion, ...],
    requests: tuple[MemoryAllocationRequest, ...],
    dirty_state_version_refs: tuple[str, ...] = (),
) -> MemoryPlan:
    """Allocate requests with deterministic lifetime-aware first fit.

    External requests are capacity planned but make the resulting plan
    ``schema_only_external``.  This function does not emit transport actions.
    """

    capacity_by_location: dict[tuple[MemoryTier, str], MemoryTierCapacity] = {}
    for index, capacity in enumerate(capacities):
        capacity.validate(f"capacities[{index}]")
        key = (capacity.tier, capacity.location_ref)
        if key in capacity_by_location:
            raise SchemaError("duplicate tier/location capacity", path=f"capacities[{index}]")
        capacity_by_location[key] = capacity

    versions_by_id: dict[str, MemoryStateVersion] = {}
    for index, version in enumerate(state_versions):
        version.validate(f"state_versions[{index}]")
        if version.id in versions_by_id:
            raise SchemaError("duplicate state version", path=f"state_versions[{index}].id")
        versions_by_id[version.id] = version

    dirty_refs = set(dirty_state_version_refs)
    if len(dirty_refs) != len(dirty_state_version_refs):
        raise SchemaError("contains duplicates", path="dirty_state_version_refs")
    unknown_dirty = dirty_refs.difference(versions_by_id)
    if unknown_dirty:
        raise SchemaError("references an unknown state version", path="dirty_state_version_refs")
    if any(not versions_by_id[item].writable for item in dirty_refs):
        raise SchemaError("read-only state cannot be dirty", path="dirty_state_version_refs")

    allocations: list[MemoryAllocation] = []
    allocated_requests: dict[str, MemoryAllocationRequest] = {}
    for index, request in enumerate(sorted(requests, key=_request_sort_key)):
        request.validate(f"requests[{index}]")
        if request.id in allocated_requests:
            raise SchemaError("duplicate allocation request", path=f"requests[{index}].id")
        if request.state_version_ref not in versions_by_id:
            raise SchemaError("references an unknown state version", path=f"requests[{index}].state_version_ref")
        capacity = capacity_by_location.get((request.tier, request.location_ref))
        if capacity is None:
            raise SchemaError("has no matching tier/location capacity", path=f"requests[{index}]")

        alignment = max(capacity.alignment_bytes, request.alignment_bytes)
        reserved_bytes = _align_up(request.size_bytes, request.alignment_bytes, path=f"requests[{index}].size_bytes")
        capacity_end = capacity.base_address + capacity.capacity_bytes
        relevant: list[tuple[int, int]] = []
        for allocation in allocations:
            old_request = allocated_requests[allocation.request_ref]
            if (
                old_request.tier is request.tier
                and old_request.location_ref == request.location_ref
                and _lifetimes_overlap(old_request, request)
            ):
                relevant.append(
                    (allocation.address, allocation.address + allocation.reserved_bytes)
                )
        relevant.sort()

        if request.pinned_address is not None:
            address = request.pinned_address
        else:
            address = _align_up(capacity.base_address, alignment, path=f"requests[{index}]")
            for occupied_start, occupied_end in relevant:
                if address + reserved_bytes <= occupied_start:
                    break
                if address < occupied_end:
                    address = _align_up(occupied_end, alignment, path=f"requests[{index}]")
        end = address + reserved_bytes
        overlaps = any(address < occupied_end and occupied_start < end for occupied_start, occupied_end in relevant)
        if address < capacity.base_address or end > capacity_end or overlaps:
            raise SchemaError(
                (
                    f"{request.tier.value} capacity exceeded at {request.location_ref!r}: "
                    f"request={request.id!r}, required={reserved_bytes}, "
                    f"available={capacity.capacity_bytes}"
                ),
                path=f"requests[{index}]",
                code="memory_capacity_exceeded",
            )
        allocation = MemoryAllocation.create(
            request_ref=request.id,
            address=address,
            reserved_bytes=reserved_bytes,
        )
        allocations.append(allocation)
        allocated_requests[request.id] = request

    allocation_by_request = {item.request_ref: item for item in allocations}
    residencies = tuple(
        MemoryResidency.create(
            state_version_ref=request.state_version_ref,
            allocation_ref=allocation_by_request[request.id].id,
            status=(
                ResidencyStatus.DIRTY
                if request.state_version_ref in dirty_refs
                else ResidencyStatus.CLEAN
            ),
            valid_from=request.lifetime_start,
            valid_until_exclusive=request.lifetime_end_exclusive,
        )
        for request in requests
    )

    peaks: list[MemoryPeak] = []
    for capacity in capacities:
        candidates = tuple(
            (allocation_by_request[request.id], request)
            for request in requests
            if request.tier is capacity.tier
            and request.location_ref == capacity.location_ref
        )
        peak_bytes = 0
        peak_tick = 0
        peak_refs: tuple[str, ...] = ()
        for tick in sorted({request.lifetime_start for _, request in candidates}):
            active = tuple(
                sorted(
                    allocation.id
                    for allocation, request in candidates
                    if request.lifetime_start <= tick < request.lifetime_end_exclusive
                )
            )
            active_bytes = sum(
                allocation.reserved_bytes
                for allocation, request in candidates
                if request.lifetime_start <= tick < request.lifetime_end_exclusive
            )
            if active_bytes > peak_bytes:
                peak_bytes, peak_tick, peak_refs = active_bytes, tick, active
        peaks.append(
            MemoryPeak.create(
                capacity_ref=capacity.id,
                at_tick=peak_tick,
                peak_bytes=peak_bytes,
                active_allocation_refs=peak_refs,
            )
        )

    return MemoryPlan.create(
        capacities=capacities,
        state_versions=state_versions,
        requests=requests,
        allocations=tuple(allocations),
        residencies=residencies,
        peaks=tuple(peaks),
    )


def _persistent_kind(kind: StateKind) -> MemoryObjectKind:
    if kind in (StateKind.PARAMETER, StateKind.TRAINABLE_PARAMETER):
        return MemoryObjectKind.PARAMETER
    if kind in (StateKind.KV_KEY, StateKind.KV_VALUE):
        return MemoryObjectKind.KV
    return MemoryObjectKind.OPTIMIZER


def plan_persistent_hbm(
    manifest: PersistentStateManifest,
    *,
    lifetime_end_exclusive: int,
) -> MemoryPlan:
    """Lift the existing persistent-state manifest into the P2 memory plan."""

    if type(manifest) is not PersistentStateManifest:
        raise SchemaError("must be a PersistentStateManifest", path="manifest")
    manifest.validate("manifest")
    if type(lifetime_end_exclusive) is not int or lifetime_end_exclusive <= 0:
        raise SchemaError("must be a positive integer", path="lifetime_end_exclusive")

    capacities = tuple(
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM,
            location_ref=f"die:{space.die_id}",
            base_address=space.base_address,
            capacity_bytes=space.size_bytes,
            alignment_bytes=space.alignment_bytes,
        )
        for space in manifest.address_spaces
    )
    declaration_by_id = {item.id: item for item in manifest.declarations}
    versions = tuple(
        MemoryStateVersion.create(
            state_ref=declaration.id,
            generation=0,
            predecessor_ref=None,
            writable=declaration.access in (
                PersistentStateAccess.READ_WRITE,
                PersistentStateAccess.RESERVED,
            ),
        )
        for declaration in manifest.declarations
    )
    version_by_state = {item.state_ref: item for item in versions}
    requests = tuple(
        MemoryAllocationRequest.create(
            state_version_ref=version_by_state[binding.state_ref].id,
            object_kind=_persistent_kind(
                declaration_by_id[binding.state_ref].identity.kind
            ),
            tier=MemoryTier.HBM,
            location_ref=f"die:{binding.die_id}",
            size_bytes=binding.size_bytes,
            alignment_bytes=next(
                space.alignment_bytes
                for space in manifest.address_spaces
                if space.die_id == binding.die_id
            ),
            lifetime_start=0,
            lifetime_end_exclusive=lifetime_end_exclusive,
            pinned_address=binding.address,
        )
        for binding in manifest.bindings
    )
    return plan_hierarchical_memory(
        capacities=capacities,
        state_versions=versions,
        requests=requests,
    )


__all__ = ["plan_hierarchical_memory", "plan_persistent_hbm"]
