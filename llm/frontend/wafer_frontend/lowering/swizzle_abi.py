"""Build explicit W9 core/runtime ABI from caller-owned physical allocations."""

from __future__ import annotations

from math import prod

from ..errors import SchemaError
from ..schema.artifact_manifest import PlanBarrierEventPhase, RecordOpcode
from ..schema.common import DType, MeshAxisName, stable_artifact_id
from ..schema.ir0 import FusionPattern
from ..schema.ir1 import IR1
from ..schema.global_action import LogicalCoreRef
from ..schema.swizzle import SwizzleActionKind
from ..schema.swizzle_abi import (
    SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION,
    SwizzleBarrierEventBinding,
    SwizzleBufferAddressBinding,
    SwizzleBufferSliceBinding,
    SwizzleCoreAddressABI,
    SwizzleRankCoreBinding,
    SwizzleTaskCoreBinding,
    SwizzleTaskRuntimeBinding,
    SwizzleValueAddressBinding,
)
from ..schema.swizzle_ir2 import (
    SwizzleIr2Projection,
    SwizzleIr2ValueOriginKind,
    admits_wang_4rank_packed_layout,
)
from ..schema.swizzle_lowering import validate_swizzle_plan_projection
from ..schema.swizzle_plan import SwizzleFusionPlan
from ..schema.swizzle_operand_abi import build_swizzle_operand_abi


def _symbol(kind: str, semantic: object) -> str:
    return stable_artifact_id(
        f"swizzle_{kind}",
        semantic,
        schema_version=SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION,
    )


def build_swizzle_core_address_abi(
    ir1: IR1,
    plan: SwizzleFusionPlan,
    projection: SwizzleIr2Projection,
    *,
    rank_cores: tuple[SwizzleRankCoreBinding, ...],
    value_bindings: tuple[SwizzleValueAddressBinding, ...],
    buffer_bindings: tuple[SwizzleBufferAddressBinding, ...],
    _producer_pass: str = "swizzle_core_address_abi_builder",
) -> SwizzleCoreAddressABI:
    """Bind caller-provided cores/addresses; no placement or shape inference."""

    ir1.validate("ir1")
    validate_swizzle_plan_projection(plan, projection)
    if tuple(item.rank for item in rank_cores) != tuple(range(len(projection.rank_dags))):
        raise SchemaError(
            "caller must provide one canonical core binding per rank",
            path="rank_cores",
        )
    core_by_rank = {item.rank: item for item in rank_cores}
    task_bindings = tuple(
        SwizzleTaskCoreBinding(
            task_ref=task.id,
            rank=dag.rank,
            logical_core=core_by_rank[dag.rank].logical_core,
            core_order=index,
            runtime_core_id=core_by_rank[dag.rank].runtime_core_id,
        )
        for dag in projection.rank_dags
        for index, task in enumerate(dag.tasks)
    )
    task_index = {task.id: task for dag in projection.rank_dags for task in dag.tasks}
    flow_by_task = {}
    for flow in projection.flows:
        flow_by_task[flow.send_task_ref] = (flow, "send")
        flow_by_task[flow.recv_task_ref] = (flow, "recv")
    token_by_task: dict[str, str] = {}
    for task in task_index.values():
        if task.kind is SwizzleActionKind.RECV:
            flow, role = flow_by_task[task.id]
            token_by_task[task.id] = _symbol(
                "dte_token",
                {"projection": projection.id, "flow": flow.id, "role": role},
            )
        elif task.kind is SwizzleActionKind.LOCAL_COPY:
            token_by_task[task.id] = _symbol(
                "dte_token",
                {"projection": projection.id, "task": task.id},
            )
    for task in task_index.values():
        if task.kind is not SwizzleActionKind.WAIT:
            continue
        recv_deps = tuple(
            dependency
            for dependency in task.deps
            if task_index[dependency].kind is SwizzleActionKind.RECV
        )
        token_by_task[task.id] = (
            token_by_task[recv_deps[0]]
            if len(recv_deps) == 1
            else _symbol("dte_token", {"projection": projection.id, "task": task.id})
        )
    runtime_bindings = tuple(
        SwizzleTaskRuntimeBinding(
            task_ref=task.id,
            flow_ref=(
                flow_by_task[task.id][0].id
                if task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV)
                else None
            ),
            token_symbol_ref=token_by_task.get(task.id),
            fsm_symbol_ref=(
                _symbol(
                    "dte_fsm",
                    {
                        "projection": projection.id,
                        "flow": flow_by_task[task.id][0].id,
                    },
                )
                if task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV)
                else None
            ),
            peer_symbol_ref=(
                _symbol(
                    "peer_core",
                    {
                        "projection": projection.id,
                        "flow": flow_by_task[task.id][0].id,
                        "role": flow_by_task[task.id][1],
                    },
                )
                if task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV)
                else None
            ),
            peer_core=(
                core_by_rank[task.peer_rank].logical_core
                if task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV)
                else None
            ),
        )
        for task in task_index.values()
        if task.kind in (
            SwizzleActionKind.SEND,
            SwizzleActionKind.RECV,
            SwizzleActionKind.WAIT,
            SwizzleActionKind.LOCAL_COPY,
        )
    )

    source_to_task = {task.source_action_ref: task.id for task in task_index.values()}
    barrier_by_ref: dict[str, dict[int, str]] = {}
    for program in plan.rank_programs:
        for action in program.actions:
            barrier = action.sync.barrier
            if barrier is not None:
                barrier_by_ref.setdefault(barrier.id, {})[program.rank] = source_to_task[
                    action.source_action.id
                ]
    events = []
    for barrier_ref, by_rank in sorted(barrier_by_ref.items()):
        leader_rank = min(by_rank)
        leader = by_rank[leader_rank]
        for peer_rank, peer in sorted(by_rank.items()):
            if peer_rank == leader_rank:
                continue
            for owner, source, destination, phase, opcode in (
                (peer, peer, leader, PlanBarrierEventPhase.ARRIVE, RecordOpcode.EVENT_SET),
                (leader, peer, leader, PlanBarrierEventPhase.ARRIVE, RecordOpcode.EVENT_WAIT),
                (leader, leader, peer, PlanBarrierEventPhase.RELEASE, RecordOpcode.EVENT_SET),
                (peer, leader, peer, PlanBarrierEventPhase.RELEASE, RecordOpcode.EVENT_WAIT),
            ):
                semantic = {
                    "projection": projection.id,
                    "barrier": barrier_ref,
                    "source": source,
                    "destination": destination,
                    "phase": phase,
                }
                events.append(
                    SwizzleBarrierEventBinding(
                        barrier_ref=barrier_ref,
                        owner_task_ref=owner,
                        source_task_ref=source,
                        destination_task_ref=destination,
                        phase=phase,
                        opcode=opcode,
                        event_symbol_ref=_symbol("barrier_event", semantic),
                        source_core_symbol_ref=_symbol(
                            "barrier_source_core", semantic
                        ),
                        destination_core_symbol_ref=_symbol(
                            "barrier_destination_core", semantic
                        ),
                        source_core=core_by_rank[task_index[source].rank].logical_core,
                        destination_core=core_by_rank[
                            task_index[destination].rank
                        ].logical_core,
                    )
                )
    result = SwizzleCoreAddressABI.create(
        producer_pass=_producer_pass,
        source_ir1_id=ir1.id,
        source_plan_ref=plan.id,
        source_projection_ref=projection.id,
        task_bindings=task_bindings,
        value_bindings=value_bindings,
        buffer_bindings=buffer_bindings,
        runtime_bindings=runtime_bindings,
        barrier_events=tuple(events),
    )
    result.validate_against(ir1, plan, projection)
    return result


def _dtype_bytes(dtype: DType) -> int:
    if dtype is DType.FP16:
        return 2
    if dtype in (DType.FP32, DType.INT32):
        return 4
    raise SchemaError("unsupported Swizzle ABI dtype", path="operand_abi.operands")


def _allocate_swizzle_address_bindings(
    ir1: IR1,
    plan: SwizzleFusionPlan,
    projection: SwizzleIr2Projection,
) -> tuple[
    tuple[SwizzleRankCoreBinding, ...],
    tuple[SwizzleValueAddressBinding, ...],
    tuple[SwizzleBufferAddressBinding, ...],
]:
    """Return the exact physical allocation without constructing the ABI carrier."""

    operand_abi = build_swizzle_operand_abi(ir1, plan, projection)
    views_by_key = {}
    all_views_by_key = {}
    for view in operand_abi.operands:
        key = (view.value_ref, view.slot)
        all_views_by_key.setdefault(key, []).append(view)
        prior = views_by_key.setdefault(key, view)
        if (
            prior.shape,
            prior.layout,
            prior.dtype,
            prior.byte_offset,
            prior.byte_extent,
        ) != (
            view.shape,
            view.layout,
            view.dtype,
            view.byte_offset,
            view.byte_extent,
        ):
            raise SchemaError(
                "one value-slot must have one exact typed view",
                path="operand_abi.operands",
            )

    rank_cores = []
    for dag in projection.rank_dags:
        die = next(item for item in ir1.fabric.dies if item.id == dag.die_id)
        core = min(die.cores, key=lambda item: item.local_core_id)
        rank_cores.append(SwizzleRankCoreBinding(
            dag.rank,
            LogicalCoreRef(dag.die_id, core.local_core_id),
            core.runtime_core_id,
        ))
    core_by_rank = {item.rank: item for item in rank_cores}
    ir1_values = {item.id: item for item in ir1.values}
    packed_layout = admits_wang_4rank_packed_layout(projection)
    terminal_keys = {
        (value.id, 0)
        for ownership in projection.output_ownership
        for value in projection.rank_dags[ownership.rank].values
        if not value.consumer_task_refs
        and len(value.producer_task_refs) == 1
        and any(
            value.symbolic_ref == boundary_ref
            or value.symbolic_ref.startswith(f"{boundary_ref}::")
            for boundary_ref in ownership.boundary_output_refs
        )
    }
    terminal_by_key = {}
    for ownership in (
        projection.output_ownership if packed_layout else ()
    ):
        dag = projection.rank_dags[ownership.rank]
        tasks = {item.id: item for item in dag.tasks}
        for boundary_ref in ownership.boundary_output_refs:
            terminal_values = tuple(
                item
                for item in dag.values
                if not item.consumer_task_refs
                and len(item.producer_task_refs) == 1
                and (
                    item.symbolic_ref == boundary_ref
                    or item.symbolic_ref.startswith(f"{boundary_ref}::")
                )
            )
            chunks = []
            for value in terminal_values:
                task = tasks[value.producer_task_refs[0]]
                if task.chunk_index is None:
                    raise SchemaError(
                        "terminal output value lacks an exact chunk index",
                        path=f"projection.output_ownership[{ownership.rank}]",
                    )
                view = views_by_key.get((value.id, 0))
                if view is None:
                    raise SchemaError(
                        "terminal output chunk lacks an exact typed view",
                        path=f"projection.output_ownership[{ownership.rank}]",
                    )
                chunks.append((task.chunk_index, value, view))
            chunks.sort(key=lambda item: (item[0], item[1].id))
            if not chunks and projection.pattern is FusionPattern.GEMM_AR:
                continue
            if not chunks or len({item[0] for item in chunks}) != len(chunks):
                raise SchemaError(
                    "terminal output chunks must be nonempty and canonical",
                    path=f"projection.output_ownership[{ownership.rank}]",
                )
            output = ir1_values[boundary_ref]
            expected_bytes = prod(output.shape) * _dtype_bytes(output.dtype)
            if MeshAxisName.TP in output.sharding.dim_map:
                if expected_bytes % len(projection.rank_dags):
                    raise SchemaError(
                        "rank-local output bytes are not TP-divisible",
                        path=f"projection.output_ownership[{ownership.rank}]",
                    )
                expected_bytes //= len(projection.rank_dags)
            if sum(item[2].byte_extent for item in chunks) != expected_bytes:
                raise SchemaError(
                    "terminal chunk extents do not close rank-local output bytes",
                    path=f"projection.output_ownership[{ownership.rank}]",
                )
            storage_ref = _symbol("storage", {
                "projection": projection.id,
                "kind": "terminal_output",
                "rank": ownership.rank,
                "boundary_output_ref": boundary_ref,
            })
            offset = 0
            for _chunk_index, value, view in chunks:
                terminal_by_key[(value.id, 0)] = (
                    storage_ref, offset, boundary_ref,
                )
                offset += view.byte_extent

    buffer_by_rank_ref = {
        (item.rank, item.buffer_ref): item
        for dag in projection.rank_dags
        for item in dag.buffers
    }
    reusable_boundary_groups = set()
    for dag in projection.rank_dags:
        orders = {item.id: index for index, item in enumerate(dag.tasks)}
        candidates = {}
        for value in dag.values:
            if (
                value.producer_task_refs
                or value.origin_kind is not SwizzleIr2ValueOriginKind.BOUNDARY_INPUT
                or not value.symbolic_ref.startswith(f"{value.origin_ref}::")
            ):
                continue
            view = views_by_key.get((value.id, 0))
            uses = all_views_by_key.get((value.id, 0), ())
            if view is None or not uses:
                continue
            candidates.setdefault((dag.rank, value.origin_ref), []).append((
                (
                    view.shape, view.layout, view.dtype,
                    view.byte_offset, view.byte_extent,
                ),
                min(orders[item.task_ref] for item in uses),
                max(orders[item.task_ref] for item in uses) + 1,
            ))
        for key, members in candidates.items():
            if (
                len(members) > 1
                and len({item[0] for item in members}) == 1
                and all(
                    left[2] <= right[1] or right[2] <= left[1]
                    for index, left in enumerate(members)
                    for right in members[index + 1 :]
                )
            ):
                reusable_boundary_groups.add(key)
    specs = {}
    for dag in projection.rank_dags:
        orders = {item.id: index for index, item in enumerate(dag.tasks)}
        for value in dag.values:
            slots = range(
                buffer_by_rank_ref[(dag.rank, value.buffer_ref)].slot_count
                if value.buffer_ref is not None else 1
            )
            for slot in slots:
                key = (value.id, slot)
                views = all_views_by_key.get(key, ())
                view = views_by_key.get(key)
                if value.buffer_ref is not None:
                    size_bytes = buffer_by_rank_ref[
                        (dag.rank, value.buffer_ref)
                    ].size_bytes
                elif view is not None:
                    size_bytes = view.byte_extent
                else:
                    raise SchemaError(
                        "unbuffered value lacks an exact typed view",
                        path=f"projection.rank_dags[{dag.rank}].values",
                    )
                if views:
                    lifetime_start = min(orders[item.task_ref] for item in views)
                    lifetime_end = max(orders[item.task_ref] for item in views) + 1
                else:
                    buffer = buffer_by_rank_ref[(dag.rank, value.buffer_ref)]
                    lifetime_orders = tuple(
                        orders[ref] for ref in buffer.lifetime_task_refs
                    )
                    lifetime_start = min(lifetime_orders)
                    lifetime_end = max(lifetime_orders) + 1
                signature = (
                    view.shape,
                    view.layout,
                    view.dtype,
                    view.byte_offset,
                    view.byte_extent,
                ) if view is not None else ("untyped", value.id, slot)
                terminal = terminal_by_key.get(key)
                if terminal is not None:
                    storage_ref, storage_offset, _boundary_ref = terminal
                    lifetime_end = len(dag.tasks)
                elif value.buffer_ref is not None:
                    storage_ref = _symbol("storage", {
                        "projection": projection.id,
                        "kind": "projected_buffer_slot",
                        "rank": dag.rank,
                        "buffer_ref": value.buffer_ref,
                        "slot": slot,
                    })
                    storage_offset = None
                elif not value.producer_task_refs and value.symbolic_ref == value.origin_ref:
                    storage_ref = _symbol("storage", {
                        "projection": projection.id,
                        "kind": "local_invariant",
                        "rank": dag.rank,
                        "origin_ref": value.origin_ref,
                    })
                    storage_offset = 0
                elif (
                    (dag.rank, value.origin_ref) in reusable_boundary_groups
                    and value.origin_kind
                    is SwizzleIr2ValueOriginKind.BOUNDARY_INPUT
                ):
                    storage_ref = _symbol("storage", {
                        "projection": projection.id,
                        "kind": "local_boundary_chunks",
                        "rank": dag.rank,
                        "origin_ref": value.origin_ref,
                    })
                    storage_offset = 0
                else:
                    storage_ref = _symbol("storage", {
                        "projection": projection.id,
                        "kind": "value",
                        "rank": dag.rank,
                        "value_ref": value.id,
                        "slot": slot,
                    })
                    storage_offset = 0
                if key in terminal_keys:
                    lifetime_end = len(dag.tasks)
                specs[key] = {
                    "rank": dag.rank,
                    "value": value,
                    "slot": slot,
                    "size": size_bytes,
                    "signature": signature,
                    "lifetime_start": lifetime_start,
                    "lifetime_end": lifetime_end,
                    "allocation_lifetime_start": lifetime_start,
                    "allocation_lifetime_end": lifetime_end,
                    "storage_ref": storage_ref,
                    "storage_offset": storage_offset,
                }

    projected_values = {
        item.id: item for dag in projection.rank_dags for item in dag.values
    }
    reduction_special = set()
    reduction_keys = []
    accumulator_keys = set()
    accumulator_predecessor = {}
    accumulator_first_input = {}
    output_accumulator = {}
    for dag in projection.rank_dags:
        for task in dag.tasks:
            if task.kind is not SwizzleActionKind.REDUCE:
                continue
            if len(task.read_value_refs) != 2 or len(task.write_value_refs) != 1:
                raise SchemaError(
                    "physical REDUCE allocation requires two reads and one write",
                    path="projection.rank_dags",
                )
            slot_by_buffer = {use.buffer_ref: use.slot for use in task.buffer_uses}
            keys = tuple(
                (
                    ref,
                    slot_by_buffer.get(projected_values[ref].buffer_ref, 0),
                )
                for ref in task.read_value_refs + task.write_value_refs
            )
            reduction_special.update((keys[0], keys[2]))
            accumulator_keys.add(keys[1])
            if keys[0] in output_accumulator:
                accumulator_predecessor[keys[1]] = output_accumulator[keys[0]]
            else:
                accumulator_first_input[keys[1]] = keys[0]
            output_accumulator[keys[2]] = keys[1]
            reduction_keys.append(keys)
            specs[keys[1]]["allocation_lifetime_start"] = min(
                specs[keys[1]]["allocation_lifetime_start"],
                specs[keys[2]]["lifetime_start"],
            )
            specs[keys[1]]["allocation_lifetime_end"] = max(
                specs[keys[1]]["allocation_lifetime_end"],
                specs[keys[2]]["lifetime_end"],
            )

    accumulator_order = {}
    for key in accumulator_keys:
        chain = []
        current = key
        while current in accumulator_predecessor:
            if current in chain:
                raise SchemaError(
                    "binary REDUCE accumulator chain must be acyclic",
                    path="projection.rank_dags",
                )
            chain.append(current)
            current = accumulator_predecessor[current]
        accumulator_order[key] = (current, len(chain))

    values = {}
    buffers = []
    region_by_rank = {}
    address_by_rank = {}
    for dag in projection.rank_dags:
        logical_core = core_by_rank[dag.rank].logical_core
        die = next(item for item in ir1.fabric.dies if item.id == dag.die_id)
        core = next(
            item for item in die.cores
            if item.local_core_id == logical_core.local_core_id
        )
        profile = next(
            item for item in ir1.fabric.sram_profiles
            if item.id == core.sram_profile_ref
        )
        region = profile.regions[0]
        region_by_rank[dag.rank] = (logical_core, region)
        address = region.base_bytes
        for buffer in dag.buffers:
            address = (address + buffer.size_bytes + 64 + 63) // 64 * 64
            buffer_base = address
            slot_cursor = 0
            slices = []
            for slot in range(buffer.slot_count):
                keys = tuple(
                    (ref, slot) for ref in sorted(buffer.value_refs)
                )
                lanes = []
                accumulator_lanes = {}
                for key in sorted(
                    keys,
                    key=lambda item: (
                        0 if item in accumulator_keys else 1,
                        accumulator_order.get(item, (("", 0), 0)),
                        specs[item]["allocation_lifetime_start"],
                        specs[item]["allocation_lifetime_end"],
                        item[0],
                    ),
                ):
                    spec = specs[key]
                    predecessor = accumulator_predecessor.get(key)
                    if key in accumulator_keys:
                        if predecessor is None:
                            first_spec = specs[accumulator_first_input[key]]
                            lanes.append({
                                "signature": first_spec["signature"],
                                "intervals": [(
                                    first_spec["allocation_lifetime_start"],
                                    first_spec["allocation_lifetime_end"],
                                )],
                            })
                            lane_index = len(lanes)
                        else:
                            lane_index = accumulator_lanes[predecessor] + 1
                        if lane_index != len(lanes):
                            raise SchemaError(
                                "binary REDUCE chain accumulator lanes must "
                                "be allocated as one contiguous block",
                                path="projection.rank_dags",
                            )
                        accumulator_lanes[key] = lane_index
                    else:
                        lane_index = None
                        if packed_layout:
                            for index, lane in enumerate(lanes):
                                if lane["signature"] != spec["signature"]:
                                    continue
                                if all(
                                    end <= spec["allocation_lifetime_start"]
                                    or spec["allocation_lifetime_end"] <= start
                                    for start, end in lane["intervals"]
                                ):
                                    lane_index = index
                                    break
                    if lane_index is None:
                        lane_index = len(lanes)
                    if lane_index == len(lanes):
                        lanes.append({
                            "signature": spec["signature"],
                            "intervals": [],
                        })
                    lanes[lane_index]["intervals"].append((
                        spec["allocation_lifetime_start"],
                        spec["allocation_lifetime_end"],
                    ))
                    spec["storage_offset"] = lane_index * buffer.size_bytes
                slot_span = len(lanes) * buffer.size_bytes
                slot_base = buffer_base + slot_cursor
                for key in keys:
                    spec = specs[key]
                    offset = slot_cursor + spec["storage_offset"]
                    slices.append(SwizzleBufferSliceBinding(
                        key[0], slot, offset, buffer.size_bytes,
                    ))
                    values[key] = SwizzleValueAddressBinding(
                        key[0], slot, dag.rank, logical_core, region.id,
                        buffer_base + offset, buffer.size_bytes, 64,
                        spec["storage_ref"], spec["storage_offset"],
                        spec["lifetime_start"], spec["lifetime_end"],
                    )
                slot_cursor += slot_span
            buffers.append(SwizzleBufferAddressBinding(
                dag.rank, buffer.buffer_ref, logical_core, region.id,
                buffer_base, slot_cursor, 64,
                tuple(sorted(slices, key=lambda item: (item.value_ref, item.slot))),
            ))
            address = (buffer_base + slot_cursor + 63) // 64 * 64
        address_by_rank[dag.rank] = address

    groups = {}
    for key, spec in specs.items():
        if key in values or key in reduction_special:
            continue
        groups.setdefault((spec["rank"], spec["storage_ref"]), []).append(key)
    for (rank, storage_ref), keys in sorted(groups.items()):
        logical_core, region = region_by_rank[rank]
        address = (address_by_rank[rank] + 63) // 64 * 64
        signatures = {specs[key]["signature"] for key in keys}
        if len(keys) > 1 and any(specs[key]["storage_offset"] == 0 for key in keys):
            if len(signatures) != 1 or len({specs[key]["size"] for key in keys}) != 1:
                raise SchemaError(
                    "local invariant storage requires one exact typed view",
                    path=f"projection.rank_dags[{rank}].values",
                )
        span = max(
            specs[key]["storage_offset"] + specs[key]["size"] for key in keys
        )
        for key in keys:
            spec = specs[key]
            values[key] = SwizzleValueAddressBinding(
                key[0], key[1], rank, logical_core, region.id,
                address + spec["storage_offset"], spec["size"], 64,
                storage_ref, spec["storage_offset"],
                spec["lifetime_start"], spec["lifetime_end"],
            )
        address_by_rank[rank] = (address + span + 63) // 64 * 64

    for input_key, accumulator_key, output_key in reduction_keys:
        accumulator = values[accumulator_key]
        input_spec = specs[input_key]
        output_spec = specs[output_key]
        if input_key in values:
            first = values[input_key]
            if (
                first.rank != accumulator.rank
                or first.logical_core != accumulator.logical_core
                or first.region_ref != accumulator.region_ref
                or first.size_bytes != accumulator.size_bytes
                or first.address + first.size_bytes != accumulator.address
            ):
                raise SchemaError(
                    "binary REDUCE chain requires the prior accumulator "
                    "immediately before its next received lane",
                    path="projection.rank_dags",
                )
        else:
            values[input_key] = SwizzleValueAddressBinding(
                input_key[0], input_key[1], accumulator.rank,
                accumulator.logical_core, accumulator.region_ref,
                accumulator.address - accumulator.size_bytes,
                accumulator.size_bytes, 64, accumulator.storage_ref,
                accumulator.storage_offset_bytes - accumulator.size_bytes,
                input_spec["lifetime_start"], input_spec["lifetime_end"],
            )
        values[output_key] = SwizzleValueAddressBinding(
            output_key[0], output_key[1], accumulator.rank,
            accumulator.logical_core, accumulator.region_ref,
            accumulator.address, accumulator.size_bytes, 64,
            accumulator.storage_ref, accumulator.storage_offset_bytes,
            output_spec["lifetime_start"], output_spec["lifetime_end"],
        )

    for rank, (logical_core, region) in region_by_rank.items():
        rank_values = tuple(item for item in values.values() if item.rank == rank)
        if any(
            item.address < region.base_bytes
            or item.address + item.size_bytes > region.base_bytes + region.size_bytes
            for item in rank_values
        ) or address_by_rank[rank] > region.base_bytes + region.size_bytes:
            raise SchemaError(
                "deterministic Swizzle allocation exceeds the named SRAM region",
                path=f"projection.rank_dags[{rank}]",
            )
    return (
        tuple(rank_cores),
        tuple(values[key] for key in sorted(values)),
        tuple(buffers),
    )


def allocate_swizzle_core_address_abi(
    ir1: IR1,
    plan: SwizzleFusionPlan,
    projection: SwizzleIr2Projection,
) -> SwizzleCoreAddressABI:
    """Deterministically allocate V1 Swizzle tasks and typed values in real SRAM.

    V1 assigns each rank to the lowest local core on its placed die and the
    first named SRAM region.  It validates capacity and preserves the explicit
    two-input contiguous LOCAL_REDUCE contract; it never infers tensor extents.
    """

    rank_cores, value_bindings, buffer_bindings = (
        _allocate_swizzle_address_bindings(ir1, plan, projection)
    )
    return build_swizzle_core_address_abi(
        ir1,
        plan,
        projection,
        rank_cores=rank_cores,
        value_bindings=value_bindings,
        buffer_bindings=buffer_bindings,
        _producer_pass="swizzle_core_address_abi_allocator",
    )


__all__ = [
    "allocate_swizzle_core_address_abi",
    "build_swizzle_core_address_abi",
]
