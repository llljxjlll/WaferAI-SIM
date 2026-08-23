"""Serialize exact whole-workload occupants that reuse one physical root."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from ..errors import SchemaError
from ..schema.swizzle_moe_ir2 import MoeSwizzleIr2Projection
from ..schema.swizzle_moe_placement import MoeWorkloadActionOwner
from ..schema.swizzle_moe_workload import (
    MoeSwizzleWorkloadProjection,
    MoeSwizzleWorkloadStorageSlotAssignment,
)
from ..schema.swizzle_moe_workload_bridge import (
    MoeSwizzleWorkloadPhysicalUse,
    MoeSwizzleWorkloadValueBridge,
    MoeSwizzleWorkloadValueKind,
)


def _topological_order(actions: dict[str, object], deps: dict[str, set[str]]) -> tuple[str, ...]:
    remaining = {ref: set(values) for ref, values in deps.items()}
    ready = sorted(ref for ref, values in remaining.items() if not values)
    result = []
    while ready:
        ref = ready.pop(0)
        result.append(ref)
        for other in sorted(remaining):
            if ref in remaining[other]:
                remaining[other].remove(ref)
                if not remaining[other] and other not in result and other not in ready:
                    ready.append(other)
                    ready.sort()
    if len(result) != len(actions):
        raise SchemaError(
            "storage reuse ordering would make the workload cyclic",
            path="schedule_moe_swizzle_workload_storage_reuse.workload",
        )
    return tuple(result)


def _reaches(source: str, destination: str, deps: dict[str, set[str]]) -> bool:
    pending = [destination]
    seen = set()
    while pending:
        ref = pending.pop()
        if ref == source:
            return True
        if ref in seen:
            continue
        seen.add(ref)
        pending.extend(deps[ref])
    return False


def schedule_moe_swizzle_workload_storage_reuse(
    workload: MoeSwizzleWorkloadProjection,
    replacement: MoeSwizzleIr2Projection,
    value_bridge: MoeSwizzleWorkloadValueBridge,
    placement: tuple[MoeWorkloadActionOwner, ...],
) -> MoeSwizzleWorkloadProjection:
    """Add canonical consumer-to-next-writer edges for fixed storage roots."""

    workload.validate("schedule_moe_swizzle_workload_storage_reuse.workload")
    replacement.validate("schedule_moe_swizzle_workload_storage_reuse.replacement")
    value_bridge.validate("schedule_moe_swizzle_workload_storage_reuse.value_bridge")
    if (
        workload.replacement_projection_id != replacement.id
        or value_bridge.source_workload_projection_id != workload.id
        or value_bridge.source_replacement_projection_id != replacement.id
        or workload.storage_reuse_edges or workload.storage_slot_assignments
    ):
        raise SchemaError(
            "storage scheduler requires exact unscheduled workload/bridge lineage",
            path="schedule_moe_swizzle_workload_storage_reuse",
        )
    actions = {item.id: item for item in workload.actions}
    tasks = {item.id: item for item in replacement.tasks}
    owners = {item.action_ref: item for item in placement}
    if len(owners) != len(placement) or set(owners) != set(actions):
        raise SchemaError(
            "storage scheduler placement coverage is not exact",
            path="schedule_moe_swizzle_workload_storage_reuse.placement",
        )
    original_to_linked = {}
    for action in workload.actions:
        for ref in action.source_action_refs:
            if ref in original_to_linked:
                raise SchemaError(
                    "original action maps to multiple workload actions",
                    path="schedule_moe_swizzle_workload_storage_reuse.workload",
                )
            original_to_linked[ref] = action.id

    occupants = {}
    for binding in value_bridge.bindings:
        if binding.kind is MoeSwizzleWorkloadValueKind.SWIGLU_OUTPUT:
            writes = tuple(
                item for item in binding.physical_slices
                if item.use is MoeSwizzleWorkloadPhysicalUse.IR2_WRITE
            )
            reads = tuple(
                item for item in binding.physical_slices
                if item.use is MoeSwizzleWorkloadPhysicalUse.IR2_READ
            )
            if len(writes) != 1 or not reads:
                raise SchemaError(
                    "SWIGLU occupant lacks one producer and typed consumers",
                    path=f"schedule_moe_swizzle_workload_storage_reuse.bindings[{binding.semantic_value_ref}]",
                )
            writer = writes[0].physical_task_ref
            core = owners[writer].runtime_core_id
            if any(owners[item.physical_task_ref].runtime_core_id != core for item in reads):
                raise SchemaError(
                    "SWIGLU occupant crosses cores without LOCAL_COPY",
                    path=f"schedule_moe_swizzle_workload_storage_reuse.bindings[{binding.semantic_value_ref}]",
                )
            key = (core, "swiglu_output")
            identity = (key, writes[0].physical_value_ref)
            prior = occupants.setdefault(identity, [writer, set()])
            if prior[0] != writer:
                raise SchemaError(
                    "one SWIGLU storage occupant has multiple writers",
                    path="schedule_moe_swizzle_workload_storage_reuse.bindings",
                )
            prior[1].update(item.physical_task_ref for item in reads)
            prior[1].update(
                original_to_linked[ref]
                for ref in binding.producer_action_refs
                + binding.consumer_action_refs
            )

    deps = {ref: set(item.deps) for ref, item in actions.items()}
    base_order = _topological_order(actions, deps)
    order_index = {ref: index for index, ref in enumerate(base_order)}
    edges = []

    def add_edge(predecessor: str, successor: str) -> None:
        if predecessor == successor or predecessor in deps[successor]:
            return
        if _reaches(successor, predecessor, deps):
            raise SchemaError(
                "typed storage reuse edge would create a cycle",
                path="schedule_moe_swizzle_workload_storage_reuse.storage_reuse_edges",
            )
        deps[successor].add(predecessor)
        edges.append((predecessor, successor))

    by_key = defaultdict(list)
    for (key, identity), (writer, consumers) in occupants.items():
        if not consumers:
            raise SchemaError(
                "storage occupant lacks a last consumer",
                path="schedule_moe_swizzle_workload_storage_reuse.bindings",
            )
        ordered_consumers = sorted(consumers, key=lambda ref: (order_index[ref], ref))
        last = ordered_consumers[-1]
        for consumer in ordered_consumers[:-1]:
            add_edge(consumer, last)
        by_key[key].append((writer, last, identity))
    assignments = []
    for key in sorted(by_key):
        ordered_occupants = sorted(
            by_key[key], key=lambda item: (order_index[item[0]], item[2]),
        )
        slot_last = {}
        for writer, last, identity in ordered_occupants:
            selected_slot = None
            for slot in (0, 1):
                prior = slot_last.get(slot)
                if prior is not None and _reaches(writer, prior, deps):
                    continue
                if prior is not None:
                    add_edge(prior, writer)
                selected_slot = slot
                break
            if selected_slot is None:
                raise SchemaError(
                    "typed storage reuse requires more than two physical slots",
                    path="schedule_moe_swizzle_workload_storage_reuse.storage_slot_assignments",
                )
            slot_last[selected_slot] = last
            assignments.append(MoeSwizzleWorkloadStorageSlotAssignment(
                identity, key[0], key[1], selected_slot, writer,
                tuple(sorted(occupants[(key, identity)][1])),
            ))

    _topological_order(actions, deps)
    scheduled_actions = tuple(
        replace(item, deps=tuple(sorted(deps[item.id])))
        for item in workload.actions
    )
    return MoeSwizzleWorkloadProjection.create(
        source_execution_id=workload.source_execution_id,
        source_overlay_id=workload.source_overlay_id,
        replacement_projection_id=workload.replacement_projection_id,
        state_abi_id=workload.state_abi_id,
        actions=scheduled_actions,
        terminals=workload.terminals,
        endpoint_lane_edges=workload.endpoint_lane_edges,
        storage_reuse_edges=tuple(sorted(edges)),
        storage_slot_assignments=tuple(sorted(
            assignments,
            key=lambda item: (
                item.runtime_core_id, item.family, item.slot,
                item.storage_unit_ref,
            ),
        )),
    )


__all__ = ["schedule_moe_swizzle_workload_storage_reuse"]
