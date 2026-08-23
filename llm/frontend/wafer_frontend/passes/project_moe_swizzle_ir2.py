"""Project only the selected MoE replacement actions into typed IR2."""

from __future__ import annotations

from collections import defaultdict
from math import prod

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.ir0 import GemmWorkload
from ..schema.lite_moe_dp4_execution import LiteMoeDp4ExecutionCase
from ..schema.swizzle import SwizzleActionKind
from ..schema.swizzle_moe_ir2 import (
    MoeSwizzleIr2Buffer,
    MoeSwizzleIr2BufferUse,
    MoeSwizzleIr2Flow,
    MoeSwizzleIr2PacketSlice,
    MoeSwizzleIr2Projection,
    MoeSwizzleIr2Task,
    MoeSwizzleIr2Value,
)
from ..schema.swizzle_moe_plan import MoeSwizzleOverlay


def _unique(refs: object) -> tuple[str, ...]:
    return tuple(dict.fromkeys(refs))


def _buffer_family(item: object) -> str | None:
    if item.buffer_family is not None:
        return item.buffer_family
    if item.work_role in ("moe_dispatch_gemm.transport", "gate", "up", "swiglu"):
        return "dispatch_operand"
    if item.work_role == "moe_gemm_combine.transport":
        return "combine_output"
    return None


def _physical_route(forward: LiteMoeDp4ExecutionCase, source: int, destination: int) -> object:
    matches = {
        route.id: route
        for group in forward.n4.graph.groups
        for route in group.embedding.routes
        if (route.source_rank, route.destination_rank) == (source, destination)
    }
    if len(matches) != 1:
        raise SchemaError("physical source/destination route is not unique", path="project_moe_swizzle_ir2.route")
    return next(iter(matches.values()))


def project_moe_swizzle_ir2(
    overlay: MoeSwizzleOverlay,
    forward: LiteMoeDp4ExecutionCase,
    *,
    endpoint_session_capacity: int,
) -> MoeSwizzleIr2Projection:
    overlay.validate("project_moe_swizzle_ir2.overlay")
    forward.validate("project_moe_swizzle_ir2.forward")
    if overlay.source_forward_execution_id != forward.id:
        raise SchemaError("overlay belongs to another forward execution", path="project_moe_swizzle_ir2")
    replacement = tuple(item for item in overlay.linked_actions if not item.preserved)
    if set(ref for item in replacement for ref in item.source_action_refs) != set(overlay.replaced_action_refs):
        raise SchemaError("replacement projection provenance is not exact", path="project_moe_swizzle_ir2")
    witness = {
        item.id: item
        for program in overlay.replacement_rank_programs
        for item in program.actions
    }
    if set(witness) != {item.id for item in replacement}:
        raise SchemaError("overlay linked/witness replacement sets differ", path="project_moe_swizzle_ir2")
    slots_by_family: dict[str, set[int]] = defaultdict(set)
    for item in witness.values():
        frozen = (item.pipeline_index, item.buffer_slot, item.buffer_family)
        if all(value is None for value in frozen):
            continue
        if any(value is None for value in frozen):
            raise SchemaError("candidate pipeline/slot/family witness is incomplete", path="project_moe_swizzle_ir2.witness")
        if item.buffer_family not in ("dispatch_operand", "combine_output") or item.buffer_slot not in (0, 1):
            raise SchemaError("candidate buffer family/slot is unsupported", path="project_moe_swizzle_ir2.witness")
        slots_by_family[item.buffer_family].add(item.buffer_slot)
    if any(slots not in ({0}, {0, 1}) for slots in slots_by_family.values()):
        raise SchemaError("candidate buffer family slots are not finite and contiguous", path="project_moe_swizzle_ir2.witness")
    source_actions = {item.id: item for item in forward.global_dag.actions}
    source_tasks = {
        item.id: item for die in forward.projection.dies for item in die.tasks
    }
    linked = {item.id: item for item in overlay.linked_actions}

    write_ids = {}
    producers: dict[tuple[int, str], list[str]] = defaultdict(list)
    for action in replacement:
        for origin in action.write_value_refs:
            value_ref = f"moe.ir2.value.r{action.rank}.{action.id}.{origin}"
            write_ids[(action.id, origin)] = value_ref
            producers[(action.rank, origin)].append(action.id)

    def read_id(action: object, origin: str) -> str:
        candidates = producers.get((action.rank, origin), ())
        if len(candidates) == 1:
            return write_ids[(candidates[0], origin)]
        if len(candidates) > 1:
            raise SchemaError("replacement read has ambiguous packed producers", path="project_moe_swizzle_ir2.values")
        return f"moe.ir2.borrowed.r{action.rank}.{origin}"

    task_operands = {}
    value_contracts: dict[str, tuple[tuple[int, ...], DType, str]] = {}
    value_origins = {}
    value_producer = {}
    value_consumers: dict[str, list[str]] = defaultdict(list)
    value_terminal = {}
    value_alias_sources = {}
    value_buffer_family = {}
    value_offsets = {}
    flow_groups: dict[tuple[str, int, int, int], dict[SwizzleActionKind, str]] = defaultdict(dict)

    for action in replacement:
        item = witness[action.id]
        writes = tuple(write_ids[(action.id, origin)] for origin in action.write_value_refs)
        if item.kind is SwizzleActionKind.SWIGLU:
            dependencies = tuple(witness.get(ref) for ref in item.deps)
            by_role = {
                dependency.work_role: dependency
                for dependency in dependencies
                if dependency is not None
            }
            if (
                len(dependencies) != 2
                or set(by_role) != {"gate", "up"}
                or item.packed_value_ref is None
                or len(writes) != 1
                or any(
                    dependency.kind is not SwizzleActionKind.COMP
                    or dependency.assignment_refs != item.assignment_refs
                    or len(linked[dependency.id].write_value_refs) != 1
                    for dependency in dependencies
                )
            ):
                raise SchemaError("grouped SWIGLU producer closure drifted", path="project_moe_swizzle_ir2.swiglu")
            source_refs = tuple(
                write_ids[(by_role[role].id, linked[by_role[role].id].write_value_refs[0])]
                for role in ("gate", "up")
            )
            m_value = len(item.assignment_refs)
            if not m_value or item.logical_bytes % (2 * m_value):
                raise SchemaError("grouped SWIGLU byte contract drifted", path="project_moe_swizzle_ir2.swiglu")
            n_value = item.logical_bytes // (2 * m_value)
            input_ref = f"moe.ir2.value.r{action.rank}.{action.id}.{item.packed_value_ref}.input"
            reads = (input_ref,)
            value_origins[input_ref] = item.packed_value_ref
            value_consumers[input_ref].append(action.id)
            value_alias_sources[input_ref] = source_refs
            value_contracts[input_ref] = ((2 * m_value, n_value), DType.FP16, "token_matrix")
            value_buffer_family[input_ref] = "swiglu_packed"
            value_offsets[input_ref] = 0
        else:
            reads = tuple(read_id(action, origin) for origin in action.read_value_refs)
        task_operands[action.id] = (reads, writes)
        if item.kind is not SwizzleActionKind.SWIGLU:
            for origin, ref in zip(action.read_value_refs, reads, strict=True):
                value_origins.setdefault(ref, origin)
                value_consumers[ref].append(action.id)
        for origin, ref in zip(action.write_value_refs, writes, strict=True):
            value_origins[ref] = origin
            value_producer[ref] = action.id
            if origin in set(overlay.terminal_value_refs):
                value_terminal[ref] = origin
        if item.kind is SwizzleActionKind.COMP:
            if len(item.original_action_refs) != 1:
                raise SchemaError("COMP projection requires one exact source GEMM contract", path="project_moe_swizzle_ir2.comp")
            source = source_actions[item.original_action_refs[0]]
            task = source_tasks[source.task_ref]
            if type(task.workload) is not GemmWorkload:
                raise SchemaError("COMP source lacks typed GEMM workload", path="project_moe_swizzle_ir2.comp")
            m_value, n_value, k_value = task.workload.rank_shape
            contracts = (
                ((m_value, k_value), task.dtype, "token_matrix"),
                ((k_value, n_value), task.dtype, "weight_matrix"),
                ((m_value, n_value), task.dtype, "token_matrix"),
            )
            for ref, contract in zip(reads + writes, contracts, strict=True):
                prior = value_contracts.setdefault(ref, contract)
                if prior != contract:
                    raise SchemaError("value GEMM contracts disagree", path="project_moe_swizzle_ir2.values")
            if item.work_role in ("gate", "up"):
                if _buffer_family(item) != "dispatch_operand":
                    raise SchemaError("Dispatch COMP lacks packed buffer family", path="project_moe_swizzle_ir2.comp")
                value_buffer_family[writes[0]] = "swiglu_packed"
                value_offsets[writes[0]] = (
                    prod(contracts[-1][0]) * 2 if item.work_role == "up" else 0
                )
        elif item.kind is SwizzleActionKind.SWIGLU:
            output_contract = ((m_value, n_value), DType.FP16, "token_matrix")
            prior = value_contracts.setdefault(writes[0], output_contract)
            if prior != output_contract:
                raise SchemaError("SWIGLU output contract disagrees", path="project_moe_swizzle_ir2.values")
        elif item.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
            contract = ((1, item.logical_bytes // 2), DType.FP16, "token_matrix")
            refs = reads if item.kind is SwizzleActionKind.SEND else writes
            if len(refs) != 1 or item.logical_bytes % 2:
                raise SchemaError("transport operand extent is not exact FP16", path="project_moe_swizzle_ir2.transport")
            prior = value_contracts.setdefault(refs[0], contract)
            if prior != contract:
                raise SchemaError("transport value contract disagrees", path="project_moe_swizzle_ir2.values")
            if item.kind is SwizzleActionKind.RECV:
                family = _buffer_family(item)
                if family is None:
                    raise SchemaError("RECV lacks typed buffer family", path="project_moe_swizzle_ir2.buffers")
                value_buffer_family[refs[0]] = family
                if refs[0] in value_terminal:
                    if item.tile_index is None:
                        raise SchemaError("terminal RECV lacks typed tile index", path="project_moe_swizzle_ir2.buffers")
                    value_offsets[refs[0]] = item.tile_index * item.logical_bytes
                else:
                    value_offsets[refs[0]] = 0
            key = (item.packet_ref, item.stage, min(item.rank, item.peer_rank), max(item.rank, item.peer_rank))
            flow_groups[key][item.kind] = action.id
        elif item.kind is SwizzleActionKind.WAIT:
            recv_ref = next(
                (
                    dep for dep in action.deps
                    if dep in witness and witness[dep].kind is SwizzleActionKind.RECV
                ),
                None,
            )
            if recv_ref is None:
                raise SchemaError("WAIT lacks exact replacement RECV", path="project_moe_swizzle_ir2.wait")
            recv = witness[recv_ref]
            key = (item.packet_ref, item.stage, min(recv.rank, recv.peer_rank), max(recv.rank, recv.peer_rank))
            flow_groups[key][item.kind] = action.id

    for ref in set(value_origins):
        if ref not in value_contracts:
            raise SchemaError("replacement value lacks typed shape/dtype contract", path="project_moe_swizzle_ir2.values")

    packet_by_id = {item.id: item for item in overlay.replacement_packetization}
    flow_ref_by_task = {}
    flows = []
    for key in sorted(flow_groups):
        endpoints = flow_groups[key]
        if set(endpoints) != {SwizzleActionKind.SEND, SwizzleActionKind.RECV, SwizzleActionKind.WAIT}:
            raise SchemaError("packet endpoint triple is incomplete", path="project_moe_swizzle_ir2.flows")
        send_ref, recv_ref, wait_ref = (
            endpoints[SwizzleActionKind.SEND],
            endpoints[SwizzleActionKind.RECV],
            endpoints[SwizzleActionKind.WAIT],
        )
        send, recv, wait = witness[send_ref], witness[recv_ref], witness[wait_ref]
        if send.packet_ref != recv.packet_ref or send.packet_ref != wait.packet_ref or send.stage != recv.stage or send.stage != wait.stage:
            raise SchemaError("packet endpoint provenance differs", path="project_moe_swizzle_ir2.flows")
        route = _physical_route(forward, send.rank, recv.rank)
        flow_ref = f"moe.ir2.flow.{send.packet_ref}.s{send.stage}.r{send.rank}.r{recv.rank}"
        for ref in (send_ref, recv_ref, wait_ref):
            flow_ref_by_task[ref] = flow_ref
        assignments = send.assignment_refs
        if assignments != recv.assignment_refs or assignments != wait.assignment_refs:
            raise SchemaError("packet assignment/payload closure differs", path="project_moe_swizzle_ir2.flows")
        packet = packet_by_id.get(send.packet_ref)
        if packet is None:
            if len(assignments) != 1:
                raise SchemaError("multi-assignment flow requires exact packet slices", path="project_moe_swizzle_ir2.flows")
            slices = (MoeSwizzleIr2PacketSlice(assignments[0], 0, 0, send.logical_bytes),)
        else:
            if (
                packet.stage != send.stage
                or packet.source_rank != send.rank
                or packet.destination_rank != recv.rank
                or packet.pivot_rank != send.pivot_rank
                or packet.logical_bytes != send.logical_bytes
                or tuple(item.assignment_ref for item in packet.slices) != assignments
            ):
                raise SchemaError("planner packet witness disagrees with endpoint actions", path="project_moe_swizzle_ir2.flows")
            slices = tuple(
                MoeSwizzleIr2PacketSlice(
                    item.assignment_ref,
                    item.source_offset_bytes,
                    item.destination_offset_bytes,
                    item.bytes,
                )
                for item in packet.slices
            )
        flows.append(
            MoeSwizzleIr2Flow(
                id=flow_ref,
                packet_ref=send.packet_ref,
                stage=send.stage,
                pivot_rank=send.pivot_rank,
                source_rank=send.rank,
                destination_rank=recv.rank,
                source_die_id=send.rank,
                destination_die_id=recv.rank,
                route_ref=route.id,
                die_path=tuple(route.die_path),
                logical_bytes=send.logical_bytes,
                assignment_slices=slices,
                send_task_ref=send_ref,
                recv_task_ref=recv_ref,
                wait_task_ref=wait_ref,
            )
        )

    buffer_values: dict[tuple[int, str], list[str]] = defaultdict(list)
    for ref, family in sorted(value_buffer_family.items()):
        rank = (
            linked[value_producer[ref]].rank
            if ref in value_producer
            else linked[value_consumers[ref][0]].rank
        )
        buffer_values[(rank, family)].append(ref)
    buffer_for_value = {
        ref: f"moe.ir2.buffer.r{rank}.{family}"
        for (rank, family), refs in buffer_values.items()
        for ref in refs
    }
    buffers = tuple(
        MoeSwizzleIr2Buffer(
            rank=rank,
            buffer_ref=f"moe.ir2.buffer.r{rank}.{family}",
            value_refs=tuple(sorted(refs)),
            size_bytes=max(
                value_offsets.get(ref, 0) + prod(value_contracts[ref][0]) * 2
                for ref in refs
            ),
            slot_count=max(slots_by_family.get(family, {0})) + 1,
        )
        for (rank, family), refs in sorted(buffer_values.items())
    )
    tasks = []
    replacement_ids = {item.id for item in replacement}
    for action in replacement:
        item = witness[action.id]
        reads, writes = task_operands[action.id]
        deps = tuple(ref for ref in action.deps if ref in replacement_ids)
        if item.kind is SwizzleActionKind.WAIT:
            deps = (flow_groups[next(key for key, group in flow_groups.items() if group.get(SwizzleActionKind.WAIT) == action.id)][SwizzleActionKind.RECV],)
        source_task = None
        if item.kind is SwizzleActionKind.COMP:
            source_task = source_tasks[source_actions[item.original_action_refs[0]].task_ref]
            m_value, n_value, k_value = source_task.workload.rank_shape
        else:
            m_value = n_value = k_value = None
        frozen = (item.pipeline_index, item.buffer_slot, item.buffer_family)
        if all(value is None for value in frozen):
            pipeline = 0 if item.tile_index is None else item.tile_index
            fixed_slot = None
            buffer_family = None
        elif any(value is None for value in frozen):
            raise SchemaError(
                "candidate pipeline/slot/family witness is incomplete",
                path=f"project_moe_swizzle_ir2.tasks.{item.id}",
            )
        else:
            pipeline = item.pipeline_index
            fixed_slot = item.buffer_slot
            buffer_family = item.buffer_family
            expected_slot = pipeline % 2 if slots_by_family[buffer_family] == {0, 1} else 0
            if fixed_slot != expected_slot:
                raise SchemaError("candidate fixed slot disagrees with pipeline", path=f"project_moe_swizzle_ir2.tasks.{item.id}")
        referenced_buffers = _unique(buffer_for_value[ref] for ref in reads + writes if ref in buffer_for_value)
        use_slot = 0 if fixed_slot is None else fixed_slot
        tasks.append(
            MoeSwizzleIr2Task(
                id=action.id,
                rank=action.rank,
                die_id=action.die_id,
                kind=item.kind,
                work_role=item.work_role,
                deps=deps,
                read_value_refs=reads,
                write_value_refs=writes,
                buffer_uses=tuple(MoeSwizzleIr2BufferUse(ref, use_slot) for ref in referenced_buffers),
                assignment_refs=item.assignment_refs,
                expert_index=item.expert_index,
                tile_index=item.tile_index,
                n_block=item.n_block_index,
                packet_ref=item.packet_ref,
                stage=item.stage,
                pivot_rank=item.pivot_rank,
                original_action_refs=item.original_action_refs,
                pipeline_index=pipeline,
                buffer_slot=fixed_slot,
                buffer_family=buffer_family,
                peer_rank=item.peer_rank,
                flow_ref=flow_ref_by_task.get(action.id),
                route_ref=(
                    next(flow.route_ref for flow in flows if flow.id == flow_ref_by_task[action.id])
                    if action.id in flow_ref_by_task and item.kind is not SwizzleActionKind.WAIT
                    else None
                ),
                logical_bytes=item.logical_bytes,
                flops=item.flops,
                matmul_m=m_value,
                matmul_n=n_value,
                matmul_k=k_value,
                dtype=source_task.dtype if source_task is not None else None,
                accumulation_dtype=DType.FP32 if source_task is not None else None,
            )
        )
    values = tuple(
        MoeSwizzleIr2Value(
            id=ref,
            rank=next(task.rank for task in tasks if task.id == value_producer[ref]) if ref in value_producer else next(task.rank for task in tasks if task.id == value_consumers[ref][0]),
            origin_ref=value_origins[ref],
            shape=value_contracts[ref][0],
            layout=value_contracts[ref][2],
            dtype=value_contracts[ref][1],
            byte_offset=value_offsets.get(ref, 0),
            size_bytes=2 * prod(value_contracts[ref][0]),
            producer_task_ref=value_producer.get(ref),
            consumer_task_refs=tuple(value_consumers.get(ref, ())),
            buffer_ref=buffer_for_value.get(ref),
            terminal_ref=value_terminal.get(ref),
            borrowed=ref not in value_producer and ref not in value_alias_sources,
            replicated=False,
            alias_source_refs=value_alias_sources.get(ref, ()),
        )
        for ref in sorted(value_origins)
    )
    result = MoeSwizzleIr2Projection.create(
        source_execution_id=overlay.source_execution_id,
        source_overlay_id=overlay.id,
        tasks=tuple(tasks),
        values=values,
        buffers=buffers,
        flows=tuple(flows),
        terminal_refs=tuple(sorted(value_terminal.values())),
        endpoint_session_capacity=endpoint_session_capacity,
    )
    validate_moe_swizzle_ir2_against_overlay(result, overlay)
    return result


def validate_moe_swizzle_ir2_against_overlay(
    projection: MoeSwizzleIr2Projection,
    overlay: MoeSwizzleOverlay,
) -> None:
    projection.validate("validate_moe_swizzle_ir2_against_overlay.projection")
    overlay.validate("validate_moe_swizzle_ir2_against_overlay.overlay")
    originals = tuple(ref for item in projection.tasks for ref in item.original_action_refs)
    if (
        projection.source_execution_id != overlay.source_execution_id
        or projection.source_overlay_id != overlay.id
        or len(originals) != len(set(originals))
        or set(originals) != set(overlay.replaced_action_refs)
        or any(ref in set(overlay.preserved_action_refs) for ref in originals)
        or {item.id for item in projection.tasks}
        != {item.id for item in overlay.linked_actions if not item.preserved}
    ):
        raise SchemaError("IR2 is not the exact replacement-only overlay quotient", path="validate_moe_swizzle_ir2_against_overlay")


__all__ = [
    "project_moe_swizzle_ir2",
    "validate_moe_swizzle_ir2_against_overlay",
]
