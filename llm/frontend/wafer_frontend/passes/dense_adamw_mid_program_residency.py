"""Derive 83 actual AdamW LSU state lifetimes and tight external subranges."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import RecordOpcode
from ..schema.dense_adamw_linked import DenseAdamwLinkedProgram
from ..schema.persistent_state import StateKind
from ..schema.serde import canonical_digest
from .dense_adamw_offload_preflight import DenseAdamwOffloadWindow


_KINDS = (
    ("parameter", StateKind.TRAINABLE_PARAMETER),
    ("master", StateKind.OPTIMIZER_MASTER),
    ("m", StateKind.OPTIMIZER_MOMENT1),
    ("v", StateKind.OPTIMIZER_MOMENT2),
    ("step", StateKind.OPTIMIZER_STEP),
)


@dataclass(frozen=True, slots=True)
class DenseAdamwExternalStateSpan:
    state_ref: str
    state_abi_id: str
    source_allocation_ref: str
    external_address: int
    hbm_address: int
    size_bytes: int
    kind: StateKind


@dataclass(frozen=True, slots=True)
class DenseAdamwMidProgramEvent:
    step_index: int
    linked_record_index: int
    kind: str
    state_ref: str
    external_address: int
    hbm_address: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DenseAdamwMidProgramResidency:
    source_memory_plan_digest: str
    linked_program_digests: tuple[str, str]
    spans: tuple[DenseAdamwExternalStateSpan, ...]
    events: tuple[DenseAdamwMidProgramEvent, ...]
    peak_active_state_bytes: int
    peak_active_plus_workspace_bytes: int
    hbm_capacity_bytes: int


@dataclass(frozen=True, slots=True)
class DenseAdamwBoundedSlots:
    workspace_end_bytes: int
    state_addresses: tuple[tuple[str, int], ...]
    highest_state_end_bytes: int
    live_state_peak_bytes: int


def assign_dense_adamw_bounded_slots(
    window: DenseAdamwOffloadWindow,
    schedule: DenseAdamwMidProgramResidency,
) -> DenseAdamwBoundedSlots:
    """Time-alias disjoint StateABI slots above signed P3 HBM workspace."""

    source = window.materialization.memory_plan
    requests = {item.id: item for item in source.requests}
    workspace = tuple(
        allocation for allocation in source.allocations
        if requests[allocation.request_ref].tier.value == "hbm"
    )
    if not workspace or schedule.hbm_capacity_bytes != window.resident_hbm_capacity_bytes:
        raise SchemaError("slot workspace/source capacity not signed", path="source")
    workspace_end = max(item.address + item.reserved_bytes for item in workspace)
    if workspace_end != window.hbm_workspace_peak_bytes:
        raise SchemaError("P3 HBM workspace shape differs from production", path="source")
    first_step_events = tuple(item for item in schedule.events if item.step_index == 0)
    second_step_events = tuple(item for item in schedule.events if item.step_index == 1)
    if tuple((e.kind, e.state_ref) for e in first_step_events) != tuple(
        (e.kind, e.state_ref) for e in second_step_events
    ):
        raise SchemaError("paged LSU order changed between step0 and step1", path="schedule")
    intervals = {}
    for event in first_step_events:
        if event.kind == "restore_before_lsu_load":
            if event.state_ref in intervals:
                raise SchemaError("multiple paged restores per state", path="schedule")
            intervals[event.state_ref] = [event.linked_record_index, -1]
        elif event.kind == "writeback_after_lsu_store":
            if event.state_ref not in intervals or intervals[event.state_ref][1] != -1:
                raise SchemaError("paged writeback order invalid", path="schedule")
            intervals[event.state_ref][1] = event.linked_record_index
    if len(intervals) != 83 or any(end <= start for start, end in intervals.values()):
        raise SchemaError("83 full paged state lifetimes are missing", path="schedule")
    spans = {item.state_ref: item for item in schedule.spans}
    assigned: dict[str, int] = {}
    min_slot = (workspace_end + 63) & ~63
    for state_ref in sorted(intervals, key=lambda ref: (intervals[ref][0], ref)):
        start, end = intervals[state_ref]
        size = spans[state_ref].size_bytes
        location = min_slot
        while location + size <= window.resident_hbm_capacity_bytes:
            collision = False
            for other_ref, other_address in assigned.items():
                other_start, other_end = intervals[other_ref]
                if (
                    start < other_end and other_start < end
                    and location < other_address + spans[other_ref].size_bytes
                    and other_address < location + size
                ):
                    collision = True
                    break
            if not collision:
                assigned[state_ref] = location
                break
            location += 64
        if state_ref not in assigned:
            raise SchemaError(
                "signed P3 workspace and two-step StateABI timed slots exceed HBM",
                path="schedule", code="adamw_mid_program_slots_exceeded",
            )
    high = max(assigned[item.state_ref] + item.size_bytes for item in schedule.spans)
    return DenseAdamwBoundedSlots(
        workspace_end_bytes=workspace_end,
        state_addresses=tuple(sorted(assigned.items())),
        highest_state_end_bytes=high,
        live_state_peak_bytes=schedule.peak_active_state_bytes,
    )


def derive_dense_adamw_mid_program_residency(
    window: DenseAdamwOffloadWindow,
    linked: tuple[DenseAdamwLinkedProgram, DenseAdamwLinkedProgram],
) -> DenseAdamwMidProgramResidency:
    """Make exact LSU-gated restore/writeback requests; no runtime hook is implied."""

    if type(linked) is not tuple or len(linked) != 2:
        raise SchemaError("requires two real production AdamW linked steps", path="linked")
    source = window.materialization
    source.validate("source")
    for index, program in enumerate(linked):
        program.validate(f"linked[{index}]")
        if (
            program.step_index != index
            or program.materialization.request.model != source.request.model
            or program.materialization.request.steps != source.request.steps
        ):
            raise SchemaError("offload span lineage changed the real two-step model", path="linked")
    source_requests = {item.id: item for item in source.memory_plan.requests}
    source_versions = {item.id: item for item in source.memory_plan.state_versions}
    inventories = {item.id: item for item in source.state_inventory}
    state_abi = linked[0].manifest.fragments[0].state_abi
    if tuple((a.state_ref, a.address, a.size_bytes, a.id)
             for a in state_abi) != tuple(
                 (a.state_ref, a.address, a.size_bytes, a.id)
                 for a in linked[1].manifest.fragments[0].state_abi
             ):
        raise SchemaError("offload-linked StateABI does not preserve two-step addresses", path="linked")
    source_groups = {}
    for allocation in source.memory_plan.allocations:
        request = source_requests[allocation.request_ref]
        if request.tier.value != "external":
            continue
        version = source_versions[request.state_version_ref]
        inventory = inventories[version.state_ref]
        if inventory.logical_name.startswith("parameter."):
            role = "parameter"
        elif inventory.logical_name.startswith("optimizer.adamw."):
            role = inventory.logical_name.split(".")[2]
        else:
            continue
        if role in source_groups:
            raise SchemaError("source aggregate duplicated", path="source.memory_plan")
        source_groups[role] = (allocation, request)
    if set(source_groups) != {role for role, _kind in _KINDS}:
        raise SchemaError("five source aggregate allocations missing", path="source.memory_plan")
    if window.resident_hbm_capacity_bytes != next(
        item.capacity_bytes for item in source.memory_plan.capacities
        if item.tier.value == "hbm"
    ):
        raise SchemaError("mid-program window differs from signed P3 HBM capacity", path="window")
    spans = []
    for role, kind in _KINDS:
        allocation, request = source_groups[role]
        abis = tuple(sorted(
            (item for item in state_abi if item.kind is kind),
            key=lambda item: item.address,
        ))
        if len(abis) != (15 if role == "parameter" else 17):
            raise SchemaError("83 source ABI spans are incomplete", path=role)
        cursor = allocation.address
        for abi in abis:
            if abi.address + abi.size_bytes > window.resident_hbm_capacity_bytes:
                raise SchemaError("mid-program physical state exceeds real HBM bounds", path=role)
            spans.append(DenseAdamwExternalStateSpan(
                state_ref=abi.state_ref, state_abi_id=abi.id,
                source_allocation_ref=allocation.id,
                external_address=cursor, hbm_address=abi.address,
                size_bytes=abi.size_bytes, kind=kind,
            ))
            cursor += abi.size_bytes
        if cursor != allocation.address + request.size_bytes:
            raise SchemaError("source allocation must equal exact ABI subspan union", path=role)
    if len(spans) != 83 or sum(item.size_bytes for item in spans) != 32100:
        raise SchemaError("83 actual full ABI bytes missing", path="spans")
    by_abi = {item.state_abi_id: item for item in spans}
    by_state = {item.state_ref: item for item in spans}
    events = []
    peak = 0
    for step_index, program in enumerate(linked):
        fragment = program.manifest.fragments[0]
        records = fragment.core_streams[0].records
        active = {}
        seen_load = set()
        seen_store = set()
        for binding in sorted(
            program.manifest.state_operand_bindings,
            key=lambda item: item.fragment_record_index,
        ):
            span = by_abi.get(binding.state_abi_id)
            if span is None or binding.fragment_record_index >= len(records):
                raise SchemaError("LSU state source binding not covered", path="linked")
            index = binding.fragment_record_index
            record = records[index]
            if record.opcode is RecordOpcode.LSU_LOAD:
                if span.state_ref in seen_load or span.state_ref in active:
                    raise SchemaError("duplicate/overlapping state restore", path="linked")
                seen_load.add(span.state_ref)
                active[span.state_ref] = span.size_bytes
                kind = "restore_before_lsu_load"
                peak = max(peak, sum(active.values()))
            elif record.opcode is RecordOpcode.LSU_STORE:
                if span.state_ref not in active or span.state_ref in seen_store:
                    raise SchemaError("writeback without matching state restore", path="linked")
                seen_store.add(span.state_ref)
                kind = "writeback_after_lsu_store"
                active.pop(span.state_ref)
            else:
                raise SchemaError("source state does not bind blocking LSU", path="linked")
            events.append(DenseAdamwMidProgramEvent(
                step_index=step_index, linked_record_index=index,
                kind=kind, state_ref=span.state_ref,
                external_address=span.external_address,
                hbm_address=span.hbm_address, size_bytes=span.size_bytes,
            ))
        if seen_load != set(by_state) or seen_store != set(by_state) or active:
            raise SchemaError("each step must restore and writeback all 83 states", path="linked")
    combined = peak + window.hbm_workspace_peak_bytes
    if combined > window.resident_hbm_capacity_bytes:
        raise SchemaError(
            "mid-program live StateABI plus source workspace exceeds HBM capacity",
            path="window", code="adamw_mid_program_peak_exceeded",
        )
    return DenseAdamwMidProgramResidency(
        source_memory_plan_digest=canonical_digest(source.memory_plan),
        linked_program_digests=(linked[0].digest, linked[1].digest),
        spans=tuple(sorted(spans, key=lambda item: item.state_ref)),
        events=tuple(events), peak_active_state_bytes=peak,
        peak_active_plus_workspace_bytes=combined,
        hbm_capacity_bytes=window.resident_hbm_capacity_bytes,
    )


__all__ = [
    "DenseAdamwExternalStateSpan", "DenseAdamwMidProgramEvent",
    "DenseAdamwMidProgramResidency", "DenseAdamwBoundedSlots",
    "derive_dense_adamw_mid_program_residency", "assign_dense_adamw_bounded_slots",
]
