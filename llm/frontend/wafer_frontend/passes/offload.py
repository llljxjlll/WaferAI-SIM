"""Deterministic next-use blocking offload planning and component execution."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from ..errors import SchemaError
from ..schema.common import UINT64_MAX
from ..schema.external_memory import (
    ExternalMemoryConnection,
    ExternalMemoryFabric,
    ExternalTransferDirection,
    ExternalTransferRequest,
)
from ..schema.memory_plan import (
    MemoryAllocationRequest,
    MemoryPlan,
    MemoryResidency,
    MemoryStateVersion,
    MemoryTier,
    MemoryTierCapacity,
    ResidencyStatus,
)
from ..schema.offload import (
    BlockingOffloadExecution,
    BlockingOffloadPlan,
    BlockingOffloadStats,
    OffloadChunk,
    OffloadEventKind,
    OffloadOperation,
    OffloadOperationKind,
    OffloadResidencyRequirement,
    OffloadStateMapping,
    OffloadTraceEvent,
)
from .external_memory import (
    SparseMemoryImage,
    execute_external_transfers,
)
from .memory_plan import plan_hierarchical_memory


@dataclass(slots=True)
class _ResidencySegment:
    state_version_ref: str
    start_cycle: int
    status: ResidencyStatus
    end_cycle: int | None = None


@dataclass(slots=True)
class _Episode:
    index: int
    chunk_ref: str
    state_version_ref: str
    start_cycle: int
    end_cycle: int | None = None
    segments: list[_ResidencySegment] | None = None


@dataclass(slots=True)
class _Resident:
    episode: _Episode
    state_version: MemoryStateVersion
    dirty: bool = False
    pin_count: int = 0


@dataclass(slots=True)
class _DraftOperation:
    kind: OffloadOperationKind
    chunk_ref: str
    trace_event_ref: str | None
    state_version_ref: str
    episode_index: int
    start_cycle: int
    ready_cycle: int
    pin_count_after: int
    dirty_after: bool


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _connection(
    fabric: ExternalMemoryFabric,
    chunk: OffloadChunk,
) -> ExternalMemoryConnection:
    connection = next(
        (
            item
            for item in fabric.connections
            if item.id == chunk.connection_ref
        ),
        None,
    )
    if connection is None:
        raise SchemaError(
            "chunk references an unknown connection",
            path="chunks",
            code="external_connection_missing",
        )
    link = next(
        item for item in fabric.links if item.id == connection.link_ref
    )
    if link.external_capacity_ref != chunk.external_capacity_ref:
        raise SchemaError(
            "chunk backing does not match connection link",
            path="chunks",
        )
    capacity = next(
        item
        for item in fabric.external_capacities
        if item.id == chunk.external_capacity_ref
    )
    if (
        chunk.external_address < capacity.base_address
        or chunk.external_address + chunk.size_bytes
        > capacity.base_address + capacity.capacity_bytes
    ):
        raise SchemaError(
            "chunk exceeds external capacity",
            path="chunks",
            code="external_memory_address_out_of_range",
        )
    return connection


def _service_cycles(
    fabric: ExternalMemoryFabric,
    chunk: OffloadChunk,
) -> int:
    connection = _connection(fabric, chunk)
    link = next(
        item for item in fabric.links if item.id == connection.link_ref
    )
    cycles = (
        link.latency_cycles
        + (chunk.size_bytes + link.bytes_per_cycle - 1)
        // link.bytes_per_cycle
    )
    if len(connection.route_die_ids) > 1:
        assert connection.route_bytes_per_cycle is not None
        cycles += (
            connection.route_latency_cycles
            + (
                chunk.size_bytes
                + connection.route_bytes_per_cycle
                - 1
            )
            // connection.route_bytes_per_cycle
        )
    return cycles


def _hbm_capacity(
    fabric: ExternalMemoryFabric,
    chunk: OffloadChunk,
) -> MemoryTierCapacity:
    connection = _connection(fabric, chunk)
    return next(
        item
        for item in fabric.hbm_capacities
        if item.id == connection.hbm_capacity_ref
    )


def _validate_inputs(
    fabric: ExternalMemoryFabric,
    chunks: tuple[OffloadChunk, ...],
    events: tuple[OffloadTraceEvent, ...],
) -> tuple[
    dict[str, OffloadChunk],
    tuple[OffloadTraceEvent, ...],
]:
    if type(fabric) is not ExternalMemoryFabric:
        raise SchemaError(
            "must be an ExternalMemoryFabric",
            path="fabric",
        )
    if type(chunks) is not tuple:
        raise SchemaError("must be an immutable tuple", path="chunks")
    if type(events) is not tuple:
        raise SchemaError("must be an immutable tuple", path="events")
    fabric.validate("fabric")
    if not chunks:
        raise SchemaError("must not be empty", path="chunks")
    chunk_by_id: dict[str, OffloadChunk] = {}
    state_refs: set[str] = set()
    external_ranges: dict[str, list[tuple[int, int]]] = {}
    for index, chunk in enumerate(chunks):
        chunk.validate(f"chunks[{index}]")
        if chunk.id in chunk_by_id:
            raise SchemaError("duplicate chunk", path=f"chunks[{index}].id")
        if chunk.initial_version.state_ref in state_refs:
            raise SchemaError(
                "state_ref must identify exactly one chunk",
                path=f"chunks[{index}].initial_version.state_ref",
            )
        _connection(fabric, chunk)
        start = chunk.external_address
        end = start + chunk.size_bytes
        ranges = external_ranges.setdefault(
            chunk.external_capacity_ref,
            [],
        )
        if any(start < old_end and old_start < end for old_start, old_end in ranges):
            raise SchemaError(
                "chunk external backing ranges overlap",
                path=f"chunks[{index}]",
            )
        ranges.append((start, end))
        chunk_by_id[chunk.id] = chunk
        state_refs.add(chunk.initial_version.state_ref)

    ordered = tuple(sorted(events, key=lambda item: (item.ordinal, item.id)))
    if not ordered:
        raise SchemaError("must not be empty", path="events")
    if tuple(item.ordinal for item in ordered) != tuple(range(len(ordered))):
        raise SchemaError(
            "ordinals must be contiguous from zero",
            path="events",
        )
    event_by_id: dict[str, OffloadTraceEvent] = {}
    for index, event in enumerate(ordered):
        event.validate(f"events[{index}]")
        chunk = chunk_by_id.get(event.chunk_ref)
        if chunk is None:
            raise SchemaError(
                "event references an unknown chunk",
                path=f"events[{index}].chunk_ref",
            )
        for dependency in event.deps:
            predecessor = event_by_id.get(dependency)
            if predecessor is None:
                raise SchemaError(
                    "dependency must reference an earlier event",
                    path=f"events[{index}].deps",
                )
        if (
            event.kind is OffloadEventKind.WRITE
            and not chunk.initial_version.writable
        ):
            raise SchemaError(
                "cannot write a read-only chunk",
                path=f"events[{index}]",
            )
        if event.id in event_by_id:
            raise SchemaError("duplicate event", path=f"events[{index}].id")
        event_by_id[event.id] = event
    return chunk_by_id, ordered


def plan_resident_only_memory(
    *,
    fabric: ExternalMemoryFabric,
    chunks: tuple[OffloadChunk, ...],
) -> MemoryPlan:
    """Require every chunk to reside in HBM for the same lifetime."""

    if not chunks:
        raise SchemaError("must not be empty", path="chunks")
    chunk_by_id, _ = _validate_inputs(
        fabric,
        chunks,
        (
            OffloadTraceEvent.create(
                ordinal=0,
                kind=OffloadEventKind.READ,
                chunk_ref=chunks[0].id,
            ),
        ),
    )
    versions = tuple(
        chunk.initial_version for chunk in chunk_by_id.values()
    )
    requests = tuple(
        MemoryAllocationRequest.create(
            state_version_ref=chunk.initial_version.id,
            object_kind=chunk.object_kind,
            tier=MemoryTier.HBM,
            location_ref=_hbm_capacity(
                fabric,
                chunk,
            ).location_ref,
            size_bytes=chunk.size_bytes,
            alignment_bytes=chunk.alignment_bytes,
            lifetime_start=0,
            lifetime_end_exclusive=1,
        )
        for chunk in chunk_by_id.values()
    )
    return plan_hierarchical_memory(
        capacities=fabric.hbm_capacities,
        state_versions=versions,
        requests=requests,
    )


def _next_use(
    events: tuple[OffloadTraceEvent, ...],
    *,
    after_ordinal: int,
    chunk_ref: str,
) -> int | None:
    return next(
        (
            event.ordinal
            for event in events
            if event.ordinal > after_ordinal
            and event.chunk_ref == chunk_ref
            and event.kind
            in (
                OffloadEventKind.PIN,
                OffloadEventKind.READ,
                OffloadEventKind.WRITE,
            )
        ),
        None,
    )


def _eviction_victim(
    *,
    fabric: ExternalMemoryFabric,
    resident: dict[str, _Resident],
    events: tuple[OffloadTraceEvent, ...],
    event_ordinal: int,
    capacity_ref: str,
    chunks: dict[str, OffloadChunk],
) -> str | None:
    candidates = tuple(
        chunk_ref
        for chunk_ref, state in resident.items()
        if state.pin_count == 0
        and _hbm_capacity(fabric, chunks[chunk_ref]).id
        == capacity_ref
    )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda chunk_ref: (
            -(
                _next_use(
                    events,
                    after_ordinal=event_ordinal,
                    chunk_ref=chunk_ref,
                )
                if _next_use(
                    events,
                    after_ordinal=event_ordinal,
                    chunk_ref=chunk_ref,
                )
                is not None
                else UINT64_MAX
            ),
            chunk_ref,
        ),
    )


def plan_offload_blocking(
    *,
    request_digest: str,
    logical_graph_digest: str,
    source_memory_plan_digest: str,
    source_memory_plan: MemoryPlan,
    state_mappings: tuple[OffloadStateMapping, ...],
    fabric: ExternalMemoryFabric,
    chunks: tuple[OffloadChunk, ...],
    events: tuple[OffloadTraceEvent, ...],
    pinned_hbm_addresses: Mapping[str, int] | None = None,
) -> BlockingOffloadPlan:
    """Plan blocking bring-in/consume/evict operations with Belady next-use."""

    chunk_by_id, ordered_events = _validate_inputs(
        fabric,
        chunks,
        events,
    )
    if pinned_hbm_addresses is not None:
        if (not isinstance(pinned_hbm_addresses, Mapping) or
                set(pinned_hbm_addresses) != set(chunk_by_id)):
            raise SchemaError("pinned HBM addresses must exactly cover all chunks",
                              path="pinned_hbm_addresses",
                              code="offload_pinned_hbm_address_mismatch")
        for chunk_ref, address in pinned_hbm_addresses.items():
            chunk = chunk_by_id[chunk_ref]
            capacity = _hbm_capacity(fabric, chunk)
            alignment = max(capacity.alignment_bytes, chunk.alignment_bytes)
            reserved = _align_up(chunk.size_bytes, chunk.alignment_bytes)
            if (type(address) is not int or address < capacity.base_address or
                    address % alignment != 0 or
                    address + reserved > capacity.base_address + capacity.capacity_bytes):
                raise SchemaError("pinned HBM address is outside aligned Die home",
                                  path=f"pinned_hbm_addresses[{chunk_ref!r}]",
                                  code="offload_pinned_hbm_address_mismatch")
    current_versions = {
        chunk.id: chunk.initial_version for chunk in chunks
    }
    versions = list(current_versions.values())
    resident: dict[str, _Resident] = {}
    episodes: list[_Episode] = []
    drafts: list[_DraftOperation] = []
    current_cycle = 0
    sequence = 0
    max_pin_count = 0

    def append_draft(
        kind: OffloadOperationKind,
        chunk: OffloadChunk,
        *,
        trace_event_ref: str | None,
        episode: _Episode,
        start_cycle: int,
        ready_cycle: int,
    ) -> None:
        nonlocal sequence
        state = resident.get(chunk.id)
        drafts.append(
            _DraftOperation(
                kind=kind,
                chunk_ref=chunk.id,
                trace_event_ref=trace_event_ref,
                state_version_ref=current_versions[chunk.id].id,
                episode_index=episode.index,
                start_cycle=start_cycle,
                ready_cycle=ready_cycle,
                pin_count_after=0 if state is None else state.pin_count,
                dirty_after=False if state is None else state.dirty,
            )
        )
        sequence += 1

    def resident_bytes(capacity_ref: str) -> int:
        return sum(
            _align_up(
                chunk_by_id[chunk_ref].size_bytes,
                chunk_by_id[chunk_ref].alignment_bytes,
            )
            for chunk_ref in resident
            if _hbm_capacity(fabric, chunk_by_id[chunk_ref]).id
            == capacity_ref
        )

    def evict(chunk_ref: str) -> None:
        nonlocal current_cycle
        chunk = chunk_by_id[chunk_ref]
        state = resident[chunk_ref]
        if state.pin_count:
            raise SchemaError(
                "cannot evict a pinned chunk",
                path="events",
                code="offload_pinned_capacity",
            )
        start = current_cycle
        if state.dirty:
            current_cycle += _service_cycles(fabric, chunk)
            kind = OffloadOperationKind.DIRTY_WRITEBACK
            state.dirty = False
        else:
            current_cycle += 1
            kind = OffloadOperationKind.CLEAN_DISCARD
        state.episode.end_cycle = current_cycle
        assert state.episode.segments is not None
        state.episode.segments[-1].end_cycle = current_cycle
        append_draft(
            kind,
            chunk,
            trace_event_ref=None,
            episode=state.episode,
            start_cycle=start,
            ready_cycle=current_cycle,
        )
        resident.pop(chunk_ref)

    def bring_in(chunk: OffloadChunk, event: OffloadTraceEvent) -> None:
        nonlocal current_cycle
        capacity = _hbm_capacity(fabric, chunk)
        required = _align_up(chunk.size_bytes, chunk.alignment_bytes)
        if required > capacity.capacity_bytes:
            raise SchemaError(
                "minimum HBM window cannot hold one chunk",
                path=f"events[{event.ordinal}]",
                code="offload_minimum_window_insufficient",
            )
        while resident_bytes(capacity.id) + required > capacity.capacity_bytes:
            victim = _eviction_victim(
                fabric=fabric,
                resident=resident,
                events=ordered_events,
                event_ordinal=event.ordinal,
                capacity_ref=capacity.id,
                chunks=chunk_by_id,
            )
            if victim is None:
                raise SchemaError(
                    "HBM capacity is held by pinned chunks",
                    path=f"events[{event.ordinal}]",
                    code="offload_pinned_capacity",
                )
            evict(victim)
        start = current_cycle
        episode = _Episode(
            index=len(episodes),
            chunk_ref=chunk.id,
            state_version_ref=current_versions[chunk.id].id,
            start_cycle=start,
            segments=[],
        )
        episodes.append(episode)
        state = _Resident(
            episode=episode,
            state_version=current_versions[chunk.id],
        )
        resident[chunk.id] = state
        current_cycle += _service_cycles(fabric, chunk)
        assert episode.segments is not None
        episode.segments.extend(
            (
                _ResidencySegment(
                    state_version_ref=current_versions[chunk.id].id,
                    start_cycle=start,
                    end_cycle=current_cycle,
                    status=ResidencyStatus.TRANSFERRING,
                ),
                _ResidencySegment(
                    state_version_ref=current_versions[chunk.id].id,
                    start_cycle=current_cycle,
                    status=ResidencyStatus.CLEAN,
                ),
            )
        )
        append_draft(
            OffloadOperationKind.BRING_IN,
            chunk,
            trace_event_ref=None,
            episode=episode,
            start_cycle=start,
            ready_cycle=current_cycle,
        )

    for event in ordered_events:
        chunk = chunk_by_id[event.chunk_ref]
        state = resident.get(chunk.id)
        if state is None:
            if (
                event.residency_requirement
                is OffloadResidencyRequirement.MUST_ALREADY_RESIDENT
            ):
                raise SchemaError(
                    "consumer reached before chunk became resident",
                    path=f"events[{event.ordinal}]",
                    code="offload_read_before_resident",
                )
            if event.kind is OffloadEventKind.UNPIN:
                raise SchemaError(
                    "cannot unpin a non-resident chunk",
                    path=f"events[{event.ordinal}]",
                )
            bring_in(chunk, event)
            state = resident[chunk.id]

        event_start = current_cycle
        event_ready = event_start + 1
        if event.kind is OffloadEventKind.PIN:
            state.pin_count += 1
            max_pin_count = max(max_pin_count, state.pin_count)
            operation_kind = OffloadOperationKind.PIN
        elif event.kind is OffloadEventKind.UNPIN:
            if state.pin_count == 0:
                raise SchemaError(
                    "pin reference count underflow",
                    path=f"events[{event.ordinal}]",
                )
            state.pin_count -= 1
            operation_kind = OffloadOperationKind.UNPIN
        elif event.kind is OffloadEventKind.READ:
            operation_kind = OffloadOperationKind.CONSUME_READ
        else:
            previous = current_versions[chunk.id]
            next_version = MemoryStateVersion.create(
                state_ref=previous.state_ref,
                generation=previous.generation + 1,
                predecessor_ref=previous.id,
                writable=True,
            )
            versions.append(next_version)
            current_versions[chunk.id] = next_version
            state.state_version = next_version
            state.dirty = True
            assert state.episode.segments is not None
            state.episode.segments[-1].end_cycle = event_ready
            state.episode.segments.append(
                _ResidencySegment(
                    state_version_ref=next_version.id,
                    start_cycle=event_ready,
                    status=ResidencyStatus.DIRTY,
                )
            )
            operation_kind = OffloadOperationKind.CONSUME_WRITE
        append_draft(
            operation_kind,
            chunk,
            trace_event_ref=event.id,
            episode=state.episode,
            start_cycle=event_start,
            ready_cycle=event_ready,
        )
        current_cycle = event_ready

    leaked = sum(state.pin_count for state in resident.values())
    if leaked:
        raise SchemaError(
            "pin reference count leaked at workload end",
            path="events",
            code="offload_pin_leak",
        )
    for chunk_ref in sorted(tuple(resident)):
        evict(chunk_ref)

    if any(episode.end_cycle is None for episode in episodes):
        raise AssertionError("planner left an open HBM residency episode")
    requests = tuple(
        MemoryAllocationRequest.create(
            state_version_ref=episode.state_version_ref,
            object_kind=chunk_by_id[episode.chunk_ref].object_kind,
            tier=MemoryTier.HBM,
            location_ref=_hbm_capacity(
                fabric,
                chunk_by_id[episode.chunk_ref],
            ).location_ref,
            size_bytes=chunk_by_id[episode.chunk_ref].size_bytes,
            alignment_bytes=chunk_by_id[episode.chunk_ref].alignment_bytes,
            lifetime_start=episode.start_cycle,
            lifetime_end_exclusive=episode.end_cycle or 0,
            pinned_address=(None if pinned_hbm_addresses is None
                            else pinned_hbm_addresses[episode.chunk_ref]),
        )
        for episode in episodes
    )
    base_memory_plan = plan_hierarchical_memory(
        capacities=fabric.hbm_capacities,
        state_versions=tuple(versions),
        requests=requests,
    )
    allocation_by_request = {
        item.request_ref: item for item in base_memory_plan.allocations
    }
    allocation_by_episode = {
        episode.index: allocation_by_request[request.id]
        for episode, request in zip(episodes, requests)
    }
    residencies = tuple(
        MemoryResidency.create(
            state_version_ref=segment.state_version_ref,
            allocation_ref=allocation_by_episode[episode.index].id,
            status=segment.status,
            valid_from=segment.start_cycle,
            valid_until_exclusive=segment.end_cycle or 0,
        )
        for episode in episodes
        for segment in (episode.segments or ())
    )
    memory_plan = MemoryPlan.create(
        capacities=base_memory_plan.capacities,
        state_versions=base_memory_plan.state_versions,
        requests=base_memory_plan.requests,
        allocations=base_memory_plan.allocations,
        residencies=residencies,
        peaks=base_memory_plan.peaks,
    )

    transfer_requests: list[ExternalTransferRequest] = []
    transfer_by_draft: dict[int, ExternalTransferRequest] = {}
    for draft_index, draft in enumerate(drafts):
        if draft.kind not in (
            OffloadOperationKind.BRING_IN,
            OffloadOperationKind.DIRTY_WRITEBACK,
        ):
            continue
        chunk = chunk_by_id[draft.chunk_ref]
        allocation = allocation_by_episode[draft.episode_index]
        transfer = ExternalTransferRequest.create(
            connection_ref=chunk.connection_ref,
            direction=(
                ExternalTransferDirection.EXTERNAL_TO_HBM
                if draft.kind is OffloadOperationKind.BRING_IN
                else ExternalTransferDirection.HBM_TO_EXTERNAL
            ),
            external_address=chunk.external_address,
            hbm_address=allocation.address,
            size_bytes=chunk.size_bytes,
            issue_cycle=draft.start_cycle,
        )
        transfer_requests.append(transfer)
        transfer_by_draft[draft_index] = transfer

    operations: list[OffloadOperation] = []
    previous_ref: str | None = None
    for draft_index, draft in enumerate(drafts):
        allocation = allocation_by_episode[draft.episode_index]
        transfer = transfer_by_draft.get(draft_index)
        operation = OffloadOperation.create(
            sequence=draft_index,
            kind=draft.kind,
            chunk_ref=draft.chunk_ref,
            trace_event_ref=draft.trace_event_ref,
            state_version_ref=draft.state_version_ref,
            hbm_allocation_ref=allocation.id,
            transfer_request_ref=None if transfer is None else transfer.id,
            depends_on=() if previous_ref is None else (previous_ref,),
            start_cycle=draft.start_cycle,
            ready_cycle=draft.ready_cycle,
            pin_count_after=draft.pin_count_after,
            dirty_after=draft.dirty_after,
        )
        operations.append(operation)
        previous_ref = operation.id

    stats = _build_stats(
        operations=tuple(operations),
        chunks=chunk_by_id,
        memory_plan=memory_plan,
        max_pin_count=max_pin_count,
    )
    result = BlockingOffloadPlan.create(
        request_digest=request_digest,
        logical_graph_digest=logical_graph_digest,
        source_memory_plan_digest=source_memory_plan_digest,
        source_memory_plan=source_memory_plan,
        state_mappings=state_mappings,
        fabric=fabric,
        chunks=chunks,
        events=ordered_events,
        state_versions=tuple(versions),
        memory_plan=memory_plan,
        transfer_requests=tuple(transfer_requests),
        operations=tuple(operations),
        stats=stats,
    )
    validate_blocking_offload_plan(
        result,
        request_digest=request_digest,
        logical_graph_digest=logical_graph_digest,
        source_memory_plan_digest=source_memory_plan_digest,
    )
    return result


def _build_stats(
    *,
    operations: tuple[OffloadOperation, ...],
    chunks: dict[str, OffloadChunk],
    memory_plan: MemoryPlan,
    max_pin_count: int,
) -> BlockingOffloadStats:
    transfer_kinds = (
        OffloadOperationKind.BRING_IN,
        OffloadOperationKind.DIRTY_WRITEBACK,
    )
    return BlockingOffloadStats.create(
        bring_in_count=sum(
            item.kind is OffloadOperationKind.BRING_IN
            for item in operations
        ),
        dirty_writeback_count=sum(
            item.kind is OffloadOperationKind.DIRTY_WRITEBACK
            for item in operations
        ),
        clean_discard_count=sum(
            item.kind is OffloadOperationKind.CLEAN_DISCARD
            for item in operations
        ),
        consume_read_count=sum(
            item.kind is OffloadOperationKind.CONSUME_READ
            for item in operations
        ),
        consume_write_count=sum(
            item.kind is OffloadOperationKind.CONSUME_WRITE
            for item in operations
        ),
        transfer_bytes=sum(
            chunks[item.chunk_ref].size_bytes
            for item in operations
            if item.kind in transfer_kinds
        ),
        blocking_transfer_cycles=sum(
            item.ready_cycle - item.start_cycle
            for item in operations
            if item.kind in transfer_kinds
        ),
        hbm_peak_bytes=max(
            (item.peak_bytes for item in memory_plan.peaks),
            default=0,
        ),
        max_pin_count=max_pin_count,
        final_resident_chunks=0,
        final_dirty_chunks=0,
        final_pin_count=0,
    )


def validate_blocking_offload_plan(
    plan: BlockingOffloadPlan,
    *,
    request_digest: str,
    logical_graph_digest: str,
    source_memory_plan_digest: str,
) -> None:
    """Recheck blocking dependencies, residency, versions, and transfers."""

    plan.validate("plan")
    expected_digests = {
        "request_digest": request_digest,
        "logical_graph_digest": logical_graph_digest,
        "source_memory_plan_digest": source_memory_plan_digest,
    }
    for name, expected in expected_digests.items():
        if getattr(plan, name) != expected:
            raise SchemaError(
                "plan is bound to a different workload source",
                path=f"plan.{name}",
                code="offload_workload_binding_mismatch",
            )
    chunks = {item.id: item for item in plan.chunks}
    events = {item.id: item for item in plan.events}
    versions = {item.id: item for item in plan.state_versions}
    authoritative_versions = {
        item.id: item.initial_version.id for item in plan.chunks
    }
    expected_version_ids = set(authoritative_versions.values())
    transfers = {item.id: item for item in plan.transfer_requests}
    allocations = {
        item.id: item for item in plan.memory_plan.allocations
    }
    allocation_requests = {
        item.id: item for item in plan.memory_plan.requests
    }
    residencies_by_allocation: dict[str, list[MemoryResidency]] = {}
    for residency in plan.memory_plan.residencies:
        residencies_by_allocation.setdefault(
            residency.allocation_ref,
            [],
        ).append(residency)

    def resident_version_at(
        allocation_ref: str,
        cycle: int,
    ) -> str | None:
        return next(
            (
                item.state_version_ref
                for item in residencies_by_allocation.get(
                    allocation_ref,
                    (),
                )
                if item.valid_from <= cycle < item.valid_until_exclusive
            ),
            None,
        )

    if plan.memory_plan.state_versions != plan.state_versions:
        raise SchemaError(
            "memory plan state versions do not match offload plan",
            path="plan.memory_plan.state_versions",
        )
    resident: dict[str, tuple[str, str, bool, int, int]] = {}
    used_transfers: set[str] = set()
    seen_events: set[str] = set()
    previous: OffloadOperation | None = None
    for index, operation in enumerate(plan.operations):
        chunk = chunks.get(operation.chunk_ref)
        if chunk is None:
            raise SchemaError(
                "operation references an unknown chunk",
                path=f"plan.operations[{index}].chunk_ref",
            )
        version = versions.get(operation.state_version_ref)
        if version is None:
            raise SchemaError(
                "operation references an unknown state version",
                path=f"plan.operations[{index}].state_version_ref",
            )
        expected_deps = () if previous is None else (previous.id,)
        if operation.depends_on != expected_deps:
            raise SchemaError(
                "blocking operations require exact predecessor dependency",
                path=f"plan.operations[{index}].depends_on",
            )
        if previous is not None and operation.start_cycle < previous.ready_cycle:
            if operation.kind is OffloadOperationKind.BRING_IN:
                raise SchemaError(
                    "HBM allocation reused while transfer is in progress",
                    path=f"plan.operations[{index}]",
                    code="offload_transfer_in_progress_reuse",
                )
            raise SchemaError(
                "operation starts before predecessor completion",
                path=f"plan.operations[{index}]",
            )
        allocation = (
            None
            if operation.hbm_allocation_ref is None
            else allocations.get(operation.hbm_allocation_ref)
        )
        if allocation is None:
            raise SchemaError(
                "operation requires a known HBM allocation",
                path=f"plan.operations[{index}].hbm_allocation_ref",
            )
        allocation_request = allocation_requests[allocation.request_ref]
        state = resident.get(chunk.id)
        if operation.kind is OffloadOperationKind.BRING_IN:
            if state is not None:
                raise SchemaError(
                    "bring-in targets an already resident chunk",
                    path=f"plan.operations[{index}]",
                )
            if version.id != authoritative_versions[chunk.id]:
                raise SchemaError(
                    "bring-in uses a version outside the chunk state chain",
                    path=f"plan.operations[{index}].state_version_ref",
                    code="offload_state_mapping_mismatch",
                )
            transfer = transfers.get(operation.transfer_request_ref or "")
            if (
                transfer is None
                or transfer.direction
                is not ExternalTransferDirection.EXTERNAL_TO_HBM
                or transfer.hbm_address != allocation.address
                or transfer.external_address != chunk.external_address
                or transfer.issue_cycle != operation.start_cycle
            ):
                raise SchemaError(
                    "bring-in transfer does not match chunk/allocation",
                    path=f"plan.operations[{index}]",
                )
            if operation.ready_cycle - operation.start_cycle != _service_cycles(
                plan.fabric,
                chunk,
            ):
                raise SchemaError(
                    "bring-in completion timing is incorrect",
                    path=f"plan.operations[{index}]",
                )
            if (
                resident_version_at(allocation.id, operation.ready_cycle)
                != version.id
            ):
                raise SchemaError(
                    "bring-in completion has no matching state residency",
                    path=f"plan.operations[{index}].state_version_ref",
                )
            used_transfers.add(transfer.id)
            resident[chunk.id] = (
                allocation.id,
                version.id,
                False,
                0,
                operation.ready_cycle,
            )
        elif operation.kind in (
            OffloadOperationKind.CONSUME_READ,
            OffloadOperationKind.CONSUME_WRITE,
            OffloadOperationKind.PIN,
            OffloadOperationKind.UNPIN,
        ):
            if state is None or operation.start_cycle < state[4]:
                raise SchemaError(
                    "consumer reached before chunk became resident",
                    path=f"plan.operations[{index}]",
                    code="offload_read_before_resident",
                )
            allocation_id, prior_version_id, dirty, pins, ready = state
            if allocation_id != allocation.id:
                raise SchemaError(
                    "resident allocation changed without eviction",
                    path=f"plan.operations[{index}]",
                )
            if operation.kind is OffloadOperationKind.CONSUME_WRITE:
                prior_version = versions[prior_version_id]
                if (
                    version.predecessor_ref != prior_version.id
                    or version.generation != prior_version.generation + 1
                ):
                    raise SchemaError(
                        "write does not advance state generation",
                        path=f"plan.operations[{index}]",
                    )
                prior_version_id = version.id
                authoritative_versions[chunk.id] = version.id
                expected_version_ids.add(version.id)
                dirty = True
                version_cycle = operation.ready_cycle
            elif version.id != prior_version_id:
                raise SchemaError(
                    "non-write operation changed state generation",
                    path=f"plan.operations[{index}]",
                )
            else:
                version_cycle = operation.start_cycle
            if (
                resident_version_at(allocation.id, version_cycle)
                != version.id
            ):
                raise SchemaError(
                    "operation state version has no matching HBM residency",
                    path=f"plan.operations[{index}].state_version_ref",
                )
            if operation.kind is OffloadOperationKind.PIN:
                pins += 1
            elif operation.kind is OffloadOperationKind.UNPIN:
                if pins == 0:
                    raise SchemaError(
                        "pin reference count underflow",
                        path=f"plan.operations[{index}]",
                    )
                pins -= 1
            if (
                operation.pin_count_after != pins
                or operation.dirty_after != dirty
            ):
                raise SchemaError(
                    "operation state snapshot is incorrect",
                    path=f"plan.operations[{index}]",
                )
            resident[chunk.id] = (
                allocation_id,
                prior_version_id,
                dirty,
                pins,
                ready,
            )
            if operation.trace_event_ref not in events:
                raise SchemaError(
                    "consume/pin operation lacks trace event",
                    path=f"plan.operations[{index}].trace_event_ref",
                )
            if operation.trace_event_ref in seen_events:
                raise SchemaError(
                    "trace event executed more than once",
                    path=f"plan.operations[{index}].trace_event_ref",
                )
            seen_events.add(operation.trace_event_ref)
            trace_event = events[operation.trace_event_ref]
            expected_kind = {
                OffloadEventKind.PIN: OffloadOperationKind.PIN,
                OffloadEventKind.UNPIN: OffloadOperationKind.UNPIN,
                OffloadEventKind.READ: OffloadOperationKind.CONSUME_READ,
                OffloadEventKind.WRITE: OffloadOperationKind.CONSUME_WRITE,
            }[trace_event.kind]
            if (
                trace_event.chunk_ref != chunk.id
                or operation.kind is not expected_kind
            ):
                raise SchemaError(
                    "operation does not match its trace event",
                    path=f"plan.operations[{index}].trace_event_ref",
                )
            if (
                trace_event.residency_requirement
                is OffloadResidencyRequirement.MUST_ALREADY_RESIDENT
                and previous is not None
                and previous.kind is OffloadOperationKind.BRING_IN
                and previous.chunk_ref == chunk.id
            ):
                raise SchemaError(
                    "consumer required a prior residency",
                    path=f"plan.operations[{index}]",
                    code="offload_read_before_resident",
                )
        else:
            if state is None:
                raise SchemaError(
                    "eviction targets a non-resident chunk",
                    path=f"plan.operations[{index}]",
                )
            allocation_id, current_version_id, dirty, pins, _ = state
            if pins:
                raise SchemaError(
                    "cannot evict a pinned chunk",
                    path=f"plan.operations[{index}]",
                )
            if allocation_id != allocation.id:
                raise SchemaError(
                    "eviction allocation mismatch",
                    path=f"plan.operations[{index}]",
                )
            if operation.kind is OffloadOperationKind.DIRTY_WRITEBACK:
                transfer = transfers.get(
                    operation.transfer_request_ref or ""
                )
                if (
                    not dirty
                    or transfer is None
                    or transfer.direction
                    is not ExternalTransferDirection.HBM_TO_EXTERNAL
                    or transfer.hbm_address != allocation.address
                    or transfer.external_address != chunk.external_address
                    or transfer.issue_cycle != operation.start_cycle
                ):
                    raise SchemaError(
                        "dirty writeback transfer is incorrect",
                        path=f"plan.operations[{index}]",
                    )
                if (
                    operation.ready_cycle - operation.start_cycle
                    != _service_cycles(plan.fabric, chunk)
                ):
                    raise SchemaError(
                        "dirty writeback timing is incorrect",
                        path=f"plan.operations[{index}]",
                    )
                used_transfers.add(transfer.id)
            elif dirty:
                raise SchemaError(
                    "dirty chunk cannot be discarded",
                    path=f"plan.operations[{index}]",
                )
            if (
                resident_version_at(allocation.id, operation.start_cycle)
                != current_version_id
                or operation.state_version_ref != current_version_id
            ):
                raise SchemaError(
                    "eviction state version has no matching HBM residency",
                    path=f"plan.operations[{index}].state_version_ref",
                )
            if operation.pin_count_after != 0 or operation.dirty_after:
                raise SchemaError(
                    "eviction must leave no pin or dirty residency",
                    path=f"plan.operations[{index}]",
                )
            request_end = allocation_request.lifetime_end_exclusive
            if request_end < operation.ready_cycle:
                raise SchemaError(
                    "HBM lifetime ends before eviction completes",
                    path=f"plan.operations[{index}]",
                )
            resident.pop(chunk.id)
        previous = operation
    if seen_events != set(events):
        raise SchemaError(
            "operations must exactly cover trace events",
            path="plan.operations",
        )
    if expected_version_ids != set(versions):
        raise SchemaError(
            "state versions must exactly match mapped initial states and writes",
            path="plan.state_versions",
            code="offload_state_mapping_mismatch",
        )
    if used_transfers != set(transfers):
        raise SchemaError(
            "transfer requests must exactly cover transfer operations",
            path="plan.transfer_requests",
        )
    if resident:
        raise SchemaError(
            "blocking plan did not drain HBM residency",
            path="plan.operations",
        )
    expected_stats = _build_stats(
        operations=plan.operations,
        chunks=chunks,
        memory_plan=plan.memory_plan,
        max_pin_count=max(
            (item.pin_count_after for item in plan.operations),
            default=0,
        ),
    )
    if expected_stats != plan.stats:
        raise SchemaError(
            "offload statistics do not match operations",
            path="plan.stats",
        )


def execute_offload_blocking(
    *,
    plan: BlockingOffloadPlan,
    request_digest: str,
    logical_graph_digest: str,
    source_memory_plan_digest: str,
    external_images: dict[str, SparseMemoryImage],
    hbm_images: dict[str, SparseMemoryImage],
) -> BlockingOffloadExecution:
    """Execute a validated blocking plan in the Python transfer service."""

    validate_blocking_offload_plan(
        plan,
        request_digest=request_digest,
        logical_graph_digest=logical_graph_digest,
        source_memory_plan_digest=source_memory_plan_digest,
    )
    report = execute_external_transfers(
        fabric=plan.fabric,
        requests=plan.transfer_requests,
        external_images=external_images,
        hbm_images=hbm_images,
    )
    return BlockingOffloadExecution.create(
        plan=plan,
        transfer_report=report,
    )


__all__ = [
    "execute_offload_blocking",
    "plan_offload_blocking",
    "plan_resident_only_memory",
    "validate_blocking_offload_plan",
]
