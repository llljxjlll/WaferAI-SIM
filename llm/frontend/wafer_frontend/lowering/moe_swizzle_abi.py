"""Deterministic physical multi-core allocation for typed MoE Swizzle IR2."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.common import stable_artifact_id
from ..schema.global_action import LogicalCoreRef
from ..schema.ir1 import IR1, MemoryInitiator
from ..schema.swizzle import SwizzleActionKind
from ..schema.swizzle_moe import MoeHardwareFacts
from ..schema.swizzle_moe_abi import (
    MOE_SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION,
    MoeSwizzleCoreAddressABI,
    MoeSwizzleStorageRootBinding,
    MoeSwizzleTaskCoreBinding,
    MoeSwizzleTaskRuntimeBinding,
    MoeSwizzleValueAddressBinding,
)
from ..schema.swizzle_moe_ir2 import MoeSwizzleIr2Projection
from ..schema.swizzle_moe_placement import (
    build_moe_swizzle_task_placement,
    build_moe_swizzle_workload_placement,
    measure_moe_swizzle_workload_endpoint_widths,
)
from ..schema.swizzle_moe_workload import MoeSwizzleWorkloadProjection


def _symbol(kind: str, semantic: object) -> str:
    return stable_artifact_id(
        f"moe_swizzle_{kind}",
        semantic,
        schema_version=MOE_SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION,
    )


def _schedule_key(task: object) -> tuple[object, ...]:
    kind_order = {
        SwizzleActionKind.RECV: 0,
        SwizzleActionKind.COMP: 1,
        SwizzleActionKind.SWIGLU: 2,
        SwizzleActionKind.SEND: 3,
        SwizzleActionKind.WAIT: 4,
        SwizzleActionKind.LOCAL_COPY: 5,
        SwizzleActionKind.REDUCE: 6,
    }
    return (
        task.pipeline_index, -1 if task.stage is None else task.stage,
        kind_order[task.kind], task.rank, task.work_role,
        -1 if task.expert_index is None else task.expert_index,
        -1 if task.tile_index is None else task.tile_index,
        -1 if task.n_block is None else task.n_block,
        task.assignment_refs, task.id,
    )


def _topological_tasks(projection: MoeSwizzleIr2Projection) -> tuple[object, ...]:
    by_id = {item.id: item for item in projection.tasks}
    deps = {ref: set(item.deps) for ref, item in by_id.items()}
    ready = sorted((ref for ref, values in deps.items() if not values), key=lambda ref: _schedule_key(by_id[ref]))
    result = []
    while ready:
        ref = ready.pop(0)
        result.append(by_id[ref])
        for other in sorted(deps):
            if ref in deps[other]:
                deps[other].remove(ref)
                if not deps[other] and by_id[other] not in result and other not in ready:
                    ready.append(other)
                    ready.sort(key=lambda item: _schedule_key(by_id[item]))
    if len(result) != len(by_id):
        raise SchemaError("MoE Swizzle task DAG contains a cycle", path="projection.tasks")
    return tuple(result)


def _task_bindings(
    ir1: IR1,
    projection: MoeSwizzleIr2Projection,
    hardware_facts: MoeHardwareFacts,
) -> tuple[MoeSwizzleTaskCoreBinding, ...]:
    ordered = _topological_tasks(projection)
    placement = dict(build_moe_swizzle_task_placement(ir1, projection, hardware_facts))
    per_core_order = defaultdict(int)
    result = []
    for task in ordered:
        owner = placement[task.id]
        logical = owner.logical_core
        order = per_core_order[logical]
        per_core_order[logical] += 1
        result.append(MoeSwizzleTaskCoreBinding(
            task.id, task.rank, logical, owner.runtime_core_id, order,
        ))
    return tuple(result)


def _value_cores(
    projection: MoeSwizzleIr2Projection,
    task_bindings: tuple[MoeSwizzleTaskCoreBinding, ...],
) -> dict[str, tuple[LogicalCoreRef, ...]]:
    tasks = {item.id: item for item in projection.tasks}
    task_core = {item.task_ref: item.logical_core for item in task_bindings}
    alias_consumers = defaultdict(set)
    for alias in projection.values:
        for source_ref in alias.alias_source_refs:
            alias_consumers[source_ref].update(alias.consumer_task_refs)
    result = {}
    for value in projection.values:
        consumer_refs = set(value.consumer_task_refs) | alias_consumers[value.id]
        consumers = {
            task_core[ref]
            for ref in consumer_refs
            if tasks[ref].kind is not SwizzleActionKind.LOCAL_COPY
        }
        copy_consumers = {
            task_core[ref]
            for ref in consumer_refs
            if tasks[ref].kind is SwizzleActionKind.LOCAL_COPY
        }
        consumers.update(copy_consumers)
        if value.producer_task_ref is not None:
            producer_core = task_core[value.producer_task_ref]
            if consumers - {producer_core}:
                raise SchemaError("value crosses cores without an explicit LOCAL_COPY", path="projection.values")
            result[value.id] = (producer_core,)
            continue
        if not consumers:
            raise SchemaError("BORROWED/alias value has no physical first use", path="projection.values")
        if len(consumers) > 1 and not value.replicated:
            raise SchemaError("BORROWED value used by multiple cores must be explicitly replicated or copied", path="projection.values")
        result[value.id] = tuple(sorted(consumers, key=lambda item: (item.die_id, item.local_core_id)))
    return result


def _value_slot(projection: MoeSwizzleIr2Projection, value_ref: str) -> int:
    value = next(item for item in projection.values if item.id == value_ref)
    if value.buffer_ref is None:
        return 0
    related = {value_ref}
    related.update(
        item.id for item in projection.values
        if value_ref in item.alias_source_refs
    )
    slots = {
        use.slot
        for task in projection.tasks
        if related & set(task.read_value_refs + task.write_value_refs)
        for use in task.buffer_uses
        if use.buffer_ref == value.buffer_ref
    }
    if len(slots) != 1:
        raise SchemaError("one physical value must use one exact buffer slot", path="projection.tasks.buffer_uses")
    return next(iter(slots))


def _select_region(ir1: IR1, logical_core: LogicalCoreRef) -> object:
    die = next(item for item in ir1.fabric.dies if item.id == logical_core.die_id)
    core = next(item for item in die.cores if item.local_core_id == logical_core.local_core_id)
    profile = next(item for item in ir1.fabric.sram_profiles if item.id == core.sram_profile_ref)
    eligible = tuple(
        item for item in profile.regions
        if MemoryInitiator.COMPUTE in item.access and MemoryInitiator.DTE in item.access
    )
    if not eligible:
        raise SchemaError("MoE Swizzle core lacks COMPUTE+DTE SRAM region", path="ir1.fabric.sram_profiles")
    return min(eligible, key=lambda item: (0 if item.name == "comm" else 1, item.base_bytes, item.id))


def _allocate_values(
    ir1: IR1,
    projection: MoeSwizzleIr2Projection,
    task_bindings: tuple[MoeSwizzleTaskCoreBinding, ...],
    *, allow_overlap_packing: bool = False,
) -> tuple[tuple[MoeSwizzleStorageRootBinding, ...], tuple[MoeSwizzleValueAddressBinding, ...]]:
    tasks = {item.id: item for item in projection.tasks}
    values = {item.id: item for item in projection.values}
    buffers = {(item.rank, item.buffer_ref): item for item in projection.buffers}
    task_core = {item.task_ref: item.logical_core for item in task_bindings}
    value_cores = _value_cores(projection, task_bindings)
    alias_views = {
        value.id: value.alias_source_refs
        for value in projection.values if value.alias_source_refs
    }
    for ref, source_refs in alias_views.items():
        if any(value_cores[ref] != value_cores[source_ref] for source_ref in source_refs):
            raise SchemaError(
                "alias view crosses cores before LOCAL_COPY",
                path="projection.values",
            )
    global_order = {task.id: index for index, task in enumerate(_topological_tasks(projection))}
    terminal_values = {item.id for item in projection.values if item.terminal_ref is not None or item.terminal_slices}
    alias_consumers = defaultdict(set)
    for alias in projection.values:
        for source_ref in alias.alias_source_refs:
            alias_consumers[source_ref].update(alias.consumer_task_refs)

    lifetimes = {}
    for value in projection.values:
        uses = list(set(value.consumer_task_refs) | alias_consumers[value.id])
        if value.producer_task_ref is not None:
            uses.append(value.producer_task_ref)
        start = min(global_order[ref] for ref in uses)
        end = len(projection.tasks) if value.id in terminal_values else max(global_order[ref] for ref in uses) + 1
        lifetimes[value.id] = (start, end)

    # A binary reduction receives two same-sized lanes and updates the second
    # lane in place.  One dedicated buffer-slot root therefore spans two lanes.
    reduce_offsets = {}
    reduce_roots = set()
    for task in projection.tasks:
        if task.kind is not SwizzleActionKind.REDUCE:
            continue
        first, accumulator = task.read_value_refs
        output = task.write_value_refs[0]
        refs = (first, accumulator, output)
        buffer_refs = {values[ref].buffer_ref for ref in refs}
        sizes = {values[ref].size_bytes for ref in refs}
        slots = {_value_slot(projection, ref) for ref in refs}
        if None in buffer_refs or len(buffer_refs) != 1 or len(sizes) != 1 or len(slots) != 1:
            raise SchemaError("binary REDUCE operands require one exact buffer/slot/extent", path="projection.tasks")
        buffer_ref, slot = next(iter(buffer_refs)), next(iter(slots))
        core = task_core[task.id]
        root_key = (task.rank, core, buffer_ref, slot)
        if root_key in reduce_roots:
            raise SchemaError("one buffer-slot admits one binary REDUCE owner", path="projection.tasks")
        reduce_roots.add(root_key)
        size = next(iter(sizes))
        reduce_offsets[(first, core)] = 0
        reduce_offsets[(accumulator, core)] = size
        reduce_offsets[(output, core)] = size

    members = defaultdict(list)
    for value in projection.values:
        if value.id in alias_views:
            continue
        slot = _value_slot(projection, value.id)
        for core in value_cores[value.id]:
            if value.buffer_ref is not None:
                root_ref = value.buffer_ref
            else:
                category = "borrowed" if value.borrowed else "terminal" if value.terminal_ref is not None or value.terminal_slices else "scratch"
                root_ref = _symbol("packed_family", {
                    "projection": projection.id,
                    "rank": value.rank,
                    "core": core,
                    "category": category,
                    "dtype": value.dtype,
                    "layout": value.layout,
                })
            members[(value.rank, core, root_ref, slot)].append(value.id)

    roots = []
    bindings = []
    cursor_by_core = {}
    for key in sorted(members, key=lambda item: (item[0], item[1].die_id, item[1].local_core_id, item[2], item[3])):
        rank, core, root_ref, slot = key
        region = _select_region(ir1, core)
        cursor = max(cursor_by_core.get(core, region.base_bytes), region.base_bytes)
        address = (cursor + 63) // 64 * 64
        value_refs = sorted(members[key])
        dynamic = all(values[ref].buffer_ref is not None for ref in value_refs)
        packed_offsets = {}
        allocated_offsets = {ref: values[ref].byte_offset for ref in value_refs}
        if dynamic and allow_overlap_packing and key not in reduce_roots:
            # The final whole-workload DAG and typed storage assignments own
            # reuse ordering.  CoreABI retains each IR2 value's canonical
            # buffer-relative offset; relocating individual values here would
            # break multi-source alias views (notably grouped gate/up SwiGLU).
            # The dedicated workload ABI remains the physical lifecycle truth.
            allocated_offsets = {
                ref: values[ref].byte_offset for ref in value_refs
            }
        if dynamic:
            root_span = max(allocated_offsets[ref] + values[ref].size_bytes for ref in value_refs)
        else:
            packed_cursor = 0
            for ref in value_refs:
                packed_cursor = (packed_cursor + 63) // 64 * 64
                packed_offsets[ref] = packed_cursor
                packed_cursor += values[ref].size_bytes
            root_span = packed_cursor
        if key in reduce_roots:
            if any(values[ref].byte_offset for ref in value_refs):
                raise SchemaError("binary REDUCE lanes cannot carry nested byte offsets", path="projection.values")
            root_span = 2 * max(values[ref].size_bytes for ref in value_refs)
        # Without the explicit two-lane reduce layout, values may be typed
        # subranges of one reusable root (grouped gate/up and its flat SWIGLU
        # view are the canonical example).  Reject only a simultaneous byte
        # overlap; differing extents alone do not imply distinct storage.
        if dynamic and key not in reduce_roots and not allow_overlap_packing:
            for index, left in enumerate(value_refs):
                for right in value_refs[index + 1:]:
                    live_overlap = not (
                        lifetimes[left][1] <= lifetimes[right][0]
                        or lifetimes[right][1] <= lifetimes[left][0]
                    )
                    left_begin = values[left].byte_offset
                    right_begin = values[right].byte_offset
                    byte_overlap = not (
                        left_begin + values[left].size_bytes <= right_begin
                        or right_begin + values[right].size_bytes <= left_begin
                    )
                    if live_overlap and byte_overlap:
                        raise SchemaError(
                            f"buffer slot reused before its last reader: {left!r} and {right!r}",
                            path="projection.buffers",
                        )
        if address + root_span > region.base_bytes + region.size_bytes:
            raise SchemaError("MoE Swizzle allocation exceeds exact SRAM region", path="projection.buffers")
        storage_ref = _symbol("storage_root", {
            "projection": projection.id, "rank": rank, "core": core,
            "buffer_ref": root_ref, "slot": slot,
        })
        roots.append(MoeSwizzleStorageRootBinding(
            rank, core, region.id, root_ref, slot, storage_ref, address, root_span,
        ))
        for value_ref in value_refs:
            offset = reduce_offsets.get((value_ref, core), packed_offsets.get(value_ref, allocated_offsets[value_ref]))
            start, end = lifetimes[value_ref]
            bindings.append(MoeSwizzleValueAddressBinding(
                value_ref, rank, core, region.id, slot, storage_ref,
                address + offset, values[value_ref].size_bytes, start, end,
            ))
        cursor_by_core[core] = (address + root_span + 63) // 64 * 64
    by_value_core = {
        (item.value_ref, item.logical_core): item for item in bindings
    }
    for value_ref, source_refs in sorted(alias_views.items()):
        value = values[value_ref]
        start, end = lifetimes[value_ref]
        for core in value_cores[value_ref]:
            parents = tuple(by_value_core[(source_ref, core)] for source_ref in source_refs)
            if len({(item.region_ref, item.slot, item.storage_ref) for item in parents}) != 1:
                raise SchemaError("alias sources do not share one physical root", path="projection.values")
            if value.buffer_ref is None:
                if len(parents) != 1:
                    raise SchemaError("multi-source alias requires an explicit buffer root", path="projection.values")
                address = parents[0].address + value.byte_offset
            else:
                address = min(item.address for item in parents)
            ranges = sorted((item.address, item.address + item.size_bytes) for item in parents)
            cursor = address
            for begin, finish in ranges:
                if finish <= cursor:
                    continue
                if begin != cursor:
                    raise SchemaError("alias sources do not exactly cover its contiguous view", path="projection.values")
                cursor = finish
            if cursor != address + value.size_bytes:
                raise SchemaError("alias view extent differs from its exact sources", path="projection.values")
            parent = parents[0]
            bindings.append(MoeSwizzleValueAddressBinding(
                value_ref, value.rank, core, parent.region_ref, parent.slot,
                parent.storage_ref, address,
                value.size_bytes, start, end,
            ))
    return tuple(roots), tuple(bindings)


def _runtime_bindings(
    projection: MoeSwizzleIr2Projection,
    task_bindings: tuple[MoeSwizzleTaskCoreBinding, ...],
) -> tuple[MoeSwizzleTaskRuntimeBinding, ...]:
    tasks = {item.id: item for item in projection.tasks}
    task_core = {item.task_ref: item.logical_core for item in task_bindings}
    flow_by_task = {
        ref: flow for flow in projection.flows
        for ref in (flow.send_task_ref, flow.recv_task_ref, flow.wait_task_ref)
    }
    def producer_core(value: object) -> object | None:
        if value.producer_task_ref is not None:
            return task_core[value.producer_task_ref]
        if len(value.alias_source_refs) == 1:
            source = next(item for item in projection.values if item.id == value.alias_source_refs[0])
            return producer_core(source)
        return None

    value_producer_core = {
        value.id: producer_core(value) for value in projection.values
    }
    result = []
    for task in projection.tasks:
        if task.kind not in (SwizzleActionKind.SEND, SwizzleActionKind.RECV, SwizzleActionKind.WAIT, SwizzleActionKind.LOCAL_COPY):
            continue
        if task.kind is SwizzleActionKind.LOCAL_COPY:
            result.append(MoeSwizzleTaskRuntimeBinding(
                task.id, None, _symbol("copy_token", {"projection": projection.id, "task": task.id}),
                None, None, value_producer_core.get(task.read_value_refs[0]),
            ))
            continue
        flow = flow_by_task[task.id]
        send_core = task_core[flow.send_task_ref]
        recv_core = task_core[flow.recv_task_ref]
        token = _symbol("dte_token", {"projection": projection.id, "flow": flow.id})
        fsm = _symbol("dte_fsm", {"projection": projection.id, "flow": flow.id})
        result.append(MoeSwizzleTaskRuntimeBinding(
            task.id,
            flow.id,
            token if task.kind in (SwizzleActionKind.RECV, SwizzleActionKind.WAIT) else None,
            fsm if task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV) else None,
            recv_core if task.kind is SwizzleActionKind.SEND else send_core if task.kind is SwizzleActionKind.RECV else None,
            None,
        ))
    return tuple(result)


def allocate_moe_swizzle_core_address_abi(
    ir1: IR1,
    projection: MoeSwizzleIr2Projection,
    *,
    hardware_facts: MoeHardwareFacts | None = None,
    enforce_endpoint_capacity: bool = True,
    workload_projection: MoeSwizzleWorkloadProjection | None = None,
) -> MoeSwizzleCoreAddressABI:
    """Allocate expert/tile work over every real core on each placed die.

    A one-core die is a valid deterministic degradation.  No logical core is
    fabricated, and an ordinary cross-core value edge is rejected unless the
    projection contains an explicit LOCAL_COPY task.
    """

    ir1.validate("ir1")
    projection.validate("projection")
    facts = MoeHardwareFacts.from_fabric(ir1.fabric) if hardware_facts is None else hardware_facts
    facts.validate("hardware_facts")
    if facts != MoeHardwareFacts.from_fabric(ir1.fabric):
        raise SchemaError("MoE hardware facts do not describe the exact IR1 fabric", path="hardware_facts")
    task_bindings = _task_bindings(ir1, projection, facts)
    physical_links = {
        (item.source_die, item.destination_die) for item in facts.route_resources
    }
    legacy_routes = {
        route.id: route
        for group in ir1.groups for route in group.embedding.routes
    }
    for flow in projection.flows:
        edges = tuple(zip(flow.die_path, flow.die_path[1:]))
        if not edges or any(edge not in physical_links for edge in edges):
            raise SchemaError("flow die path is not closed by hardware link incidence", path="projection.flows")
        legacy = legacy_routes.get(flow.route_ref)
        if legacy is not None:
            width = len(flow.die_path)
            if not any(
                tuple(legacy.die_path[index:index + width]) == flow.die_path
                for index in range(len(legacy.die_path) - width + 1)
            ):
                raise SchemaError("flow die path is not an exact PairRoute segment", path="projection.flows")
        elif flow.route_ref != f"moe.scale.route.r{flow.source_rank}.r{flow.destination_rank}":
            raise SchemaError("flow references an unknown PairRoute or canonical scale route", path="projection.flows")

    # Region preflight has only replacement tasks, so its typed stage/pipeline
    # wave is the strongest available admission witness.  Whole-workload
    # lowering instead consumes the final endpoint-lane DAG: preserved and
    # inter-region dependencies are required to close the real antichain.
    tasks = {item.id: item for item in projection.tasks}
    placement = {item.task_ref: item.logical_core for item in task_bindings}
    if type(enforce_endpoint_capacity) is not bool:
        raise SchemaError("endpoint admission flag must be bool", path="enforce_endpoint_capacity")
    whole_mode = workload_projection is not None
    if whole_mode:
        if type(workload_projection) is not MoeSwizzleWorkloadProjection:
            raise SchemaError(
                "whole-mode CoreABI requires exact workload projection",
                path="workload_projection",
            )
        workload_projection.validate("workload_projection")
        if (
            workload_projection.replacement_projection_id != projection.id
            or not workload_projection.storage_slot_assignments
        ):
            raise SchemaError(
                "whole-mode CoreABI requires exact scheduled workload lineage",
                path="workload_projection",
            )
        workload_placement = build_moe_swizzle_workload_placement(
            ir1, workload_projection, projection, facts,
        )
        whole_owners = {item.action_ref: item for item in workload_placement}
        if any(
            whole_owners[ref].logical_core != binding.logical_core
            or whole_owners[ref].runtime_core_id != binding.runtime_core_id
            for ref, binding in ((item.task_ref, item) for item in task_bindings)
        ):
            raise SchemaError(
                "replacement CoreABI placement differs from whole workload",
                path="workload_projection.actions",
            )
        widths = measure_moe_swizzle_workload_endpoint_widths(
            workload_projection, projection, workload_placement,
            capacity_per_core=projection.endpoint_session_capacity,
        )
        if enforce_endpoint_capacity and any(
            item.max_inflight > projection.endpoint_session_capacity
            for item in widths
        ):
            raise SchemaError(
                "whole MoE Swizzle endpoint session wave exceeds capacity",
                path="workload_projection.endpoint_lane_edges",
            )
    else:
        sessions = defaultdict(set)
        for flow in projection.flows:
            for ref in (flow.send_task_ref, flow.recv_task_ref):
                task = tasks[ref]
                sessions[(placement[ref], task.stage, task.pipeline_index)].add(flow.id)
        if enforce_endpoint_capacity and any(
            len(refs) > projection.endpoint_session_capacity for refs in sessions.values()
        ):
            raise SchemaError("MoE Swizzle endpoint session wave exceeds capacity", path="projection.flows")
    roots, values = _allocate_values(
        ir1, projection, task_bindings,
        allow_overlap_packing=whole_mode or not enforce_endpoint_capacity,
    )
    result = MoeSwizzleCoreAddressABI.create(
        source_ir1_id=ir1.id,
        source_projection_id=projection.id,
        task_bindings=task_bindings,
        storage_roots=roots,
        value_bindings=values,
        runtime_bindings=_runtime_bindings(projection, task_bindings),
        source_workload_projection_id=(
            workload_projection.id if workload_projection is not None else None
        ),
    )
    result.validate_against(
        ir1, projection, workload_projection=workload_projection,
    )
    return result


__all__ = ["allocate_moe_swizzle_core_address_abi"]
