"""Deterministically schedule whole-workload endpoint sessions into lanes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import validate_uint64
from ..schema.swizzle_moe_ir2 import MoeSwizzleIr2Projection
from ..schema.swizzle_moe_placement import MoeWorkloadActionOwner
from ..schema.swizzle_moe_workload import MoeSwizzleWorkloadProjection


def _topological_order(workload: MoeSwizzleWorkloadProjection) -> tuple[str, ...]:
    actions = {item.id: item for item in workload.actions}
    remaining = {ref: set(item.deps) for ref, item in actions.items()}
    ready = sorted(ref for ref, deps in remaining.items() if not deps)
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
            "whole workload is cyclic before endpoint scheduling",
            path="schedule_moe_swizzle_workload_endpoints.workload",
        )
    return tuple(result)


def schedule_moe_swizzle_workload_endpoints(
    workload: MoeSwizzleWorkloadProjection,
    replacement: MoeSwizzleIr2Projection,
    placement: tuple[MoeWorkloadActionOwner, ...],
    *,
    capacity_per_core: int,
) -> MoeSwizzleWorkloadProjection:
    """Add canonical lane-reuse edges without changing semantic value edges."""

    if type(workload) is not MoeSwizzleWorkloadProjection:
        raise SchemaError(
            "requires exact whole workload projection",
            path="schedule_moe_swizzle_workload_endpoints.workload",
        )
    if type(replacement) is not MoeSwizzleIr2Projection:
        raise SchemaError(
            "requires exact replacement projection",
            path="schedule_moe_swizzle_workload_endpoints.replacement",
        )
    workload.validate("schedule_moe_swizzle_workload_endpoints.workload")
    replacement.validate("schedule_moe_swizzle_workload_endpoints.replacement")
    validate_uint64(
        capacity_per_core,
        "schedule_moe_swizzle_workload_endpoints.capacity_per_core",
    )
    if capacity_per_core == 0 or workload.endpoint_lane_edges:
        raise SchemaError(
            "endpoint scheduling requires positive capacity and unscheduled input",
            path="schedule_moe_swizzle_workload_endpoints",
        )
    if workload.replacement_projection_id != replacement.id:
        raise SchemaError(
            "workload/replacement lineage is not exact",
            path="schedule_moe_swizzle_workload_endpoints",
        )
    actions = {item.id: item for item in workload.actions}
    owners = {}
    for index, owner in enumerate(placement):
        if type(owner) is not MoeWorkloadActionOwner:
            raise SchemaError(
                "requires exact workload action owners",
                path=f"schedule_moe_swizzle_workload_endpoints.placement[{index}]",
            )
        owner.validate(f"schedule_moe_swizzle_workload_endpoints.placement[{index}]")
        if owner.action_ref in owners:
            raise SchemaError(
                "workload owner is duplicated",
                path="schedule_moe_swizzle_workload_endpoints.placement",
            )
        owners[owner.action_ref] = owner
    if set(owners) != set(actions):
        raise SchemaError(
            "workload owner coverage is not exact",
            path="schedule_moe_swizzle_workload_endpoints.placement",
        )
    endpoint_refs = {
        ref
        for flow in replacement.flows
        for ref in (flow.send_task_ref, flow.recv_task_ref)
    }
    order = _topological_order(workload)
    order_index = {ref: index for index, ref in enumerate(order)}
    by_core = defaultdict(list)
    for ref in endpoint_refs:
        by_core[owners[ref].runtime_core_id].append(ref)

    deps = {ref: list(item.deps) for ref, item in actions.items()}
    lane_edges = []
    for runtime_core_id in sorted(by_core):
        refs = sorted(by_core[runtime_core_id], key=lambda ref: (order_index[ref], ref))
        for index in range(capacity_per_core, len(refs)):
            predecessor = refs[index - capacity_per_core]
            successor = refs[index]
            if predecessor not in deps[successor]:
                deps[successor].append(predecessor)
                lane_edges.append((predecessor, successor))
    scheduled_actions = tuple(
        replace(item, deps=tuple(dict.fromkeys(deps[item.id])))
        for item in workload.actions
    )
    return MoeSwizzleWorkloadProjection.create(
        source_execution_id=workload.source_execution_id,
        source_overlay_id=workload.source_overlay_id,
        replacement_projection_id=workload.replacement_projection_id,
        state_abi_id=workload.state_abi_id,
        actions=scheduled_actions,
        terminals=workload.terminals,
        endpoint_lane_edges=tuple(sorted(lane_edges)),
        storage_reuse_edges=workload.storage_reuse_edges,
        storage_slot_assignments=workload.storage_slot_assignments,
    )


__all__ = ["schedule_moe_swizzle_workload_endpoints"]
