"""Dedicated S3-Lite projection, scheduling, and global-action producers."""

from __future__ import annotations

import math
from collections import Counter

from ..errors import SchemaError
from ..schema.common import DType, stable_artifact_id
from ..schema.ir0 import GemmWorkload, OpKind, SwiGluWorkload
from ..schema.lite_moe_execution import (
    LITE_MOE_SCHEDULE_SCHEMA_VERSION,
    LiteMoeBufferAccess,
    LiteMoeBufferBinding,
    LiteMoeBufferUse,
    LiteMoeFlow,
    LiteMoeGlobalAction,
    LiteMoeGlobalDag,
    LiteMoePackedSlice,
    LiteMoeProjectedDie,
    LiteMoeProjection,
    LiteMoeScheduled,
    LiteMoeTask,
    LiteMoeTaskKind,
    LiteMoeTaskPlacement,
)
from ..schema.lite_moe_n4 import LiteMoeN4IR1
from ..schema.persistent_state import canonical_state_staging_value_id


def _expert(node_ref: str) -> int:
    marker = ".expert"
    if marker not in node_ref:
        raise SchemaError("compute node lacks expert lineage", path="lite_moe_execution")
    text = node_ref.split(marker, 1)[1].split(".", 1)[0]
    if text not in ("0", "1", "2", "3"):
        raise SchemaError("invalid expert lineage", path="lite_moe_execution")
    return int(text)


def _comp_id(node_ref: str) -> str:
    return f"moe.task.{node_ref}.comp"


def _dma_id(access_ref: str) -> str:
    return f"moe.task.{access_ref}.dma_in"


def _transport_id(binding_ref: str, kind: str) -> str:
    return f"moe.task.{binding_ref}.{kind}"


def _flow_id(binding_ref: str) -> str:
    return f"moe.flow.{binding_ref}"


def _packed_root(node_ref: str) -> str:
    return f"moe.value.{node_ref.rsplit('.', 1)[0]}.gate_up"


def _projection_components(
    source: LiteMoeN4IR1,
) -> tuple[tuple[LiteMoeProjectedDie, ...], tuple[LiteMoeFlow, ...]]:
    graph = source.graph
    if source.fusion_plans or source.standalone_plans:
        raise SchemaError("S3-Lite projection requires empty Dense plans", path="source")
    value_index = {value.id: value for value in graph.values}
    binding_by_node = {binding.node_ref: binding for binding in source.p2p_bindings}
    access_by_node: dict[str, list[object]] = {}
    for access in graph.state_accesses:
        access_by_node.setdefault(access.node_ref, []).append(access)
    manifest = graph.persistent_state_manifest
    if manifest is None:
        raise SchemaError("S3-Lite requires parameter backing", path="source.graph")
    declaration_index = {item.id: item for item in manifest.declarations}
    hbm_index = {item.state_ref: item for item in manifest.bindings}
    group = graph.groups[0]
    die_to_rank = {item.die_id: item.rank for item in group.placements}
    route_index = {
        (item.source_rank, item.destination_rank): item
        for item in group.embedding.routes
    }
    tasks: dict[int, list[LiteMoeTask]] = {0: [], 1: []}
    flows: list[LiteMoeFlow] = []
    terminal_by_node: dict[str, str] = {}

    def producer_dep(value_ref: str) -> tuple[str, ...]:
        producer = value_index[value_ref].producer
        return () if producer is None else (terminal_by_node[producer],)

    for node in graph.nodes:
        if node.kind is OpKind.P2P:
            binding = binding_by_node.get(node.id)
            if binding is None:
                raise SchemaError("P2P node lacks typed MoE binding", path="source.p2p_bindings")
            route = route_index.get(
                (die_to_rank[binding.source_die_id], die_to_rank[binding.destination_die_id])
            )
            if route is None or route.die_path != (
                binding.source_die_id, binding.destination_die_id
            ):
                raise SchemaError("MoE binding has no exact PairRoute", path="source.graph.groups")
            send_id = _transport_id(binding.id, "send")
            recv_id = _transport_id(binding.id, "recv")
            wait_id = _transport_id(binding.id, "wait")
            send = LiteMoeTask(
                send_id, LiteMoeTaskKind.SEND, binding.source_die_id, node.id,
                None, None, None, binding.id, route.id,
                binding.destination_die_id, 32, DType.FP16,
                (node.inputs[0],), (), producer_dep(node.inputs[0]), None, None,
            )
            recv = LiteMoeTask(
                recv_id, LiteMoeTaskKind.RECV, binding.destination_die_id, node.id,
                None, None, None, binding.id, route.id,
                binding.source_die_id, 32, DType.FP16,
                (), (node.outputs[0],), (), None, None,
            )
            wait = LiteMoeTask(
                wait_id, LiteMoeTaskKind.WAIT, binding.destination_die_id, node.id,
                None, None, None, binding.id, route.id,
                binding.source_die_id, 32, DType.FP16,
                (), (), (recv_id,), None, None,
            )
            for item in (send, recv, wait):
                item.validate(); tasks[item.die_id].append(item)
            flow = LiteMoeFlow(
                _flow_id(binding.id), binding.id, route.id,
                binding.source_die_id, binding.destination_die_id,
                node.inputs[0], node.outputs[0], 32, DType.FP16,
                send_id, recv_id, wait_id,
            )
            flow.validate(); flows.append(flow)
            terminal_by_node[node.id] = wait_id
            continue

        die = _expert(node.id) // 2
        deps: list[str] = []
        reads: list[str] = list(node.inputs)
        for value_ref in node.inputs:
            producer = value_index[value_ref].producer
            if producer is not None:
                deps.append(terminal_by_node[producer])
        accesses = access_by_node.get(node.id, [])
        if node.kind is OpKind.GEMM:
            if len(accesses) != 1 or type(node.workload) is not GemmWorkload:
                raise SchemaError("GEMM requires one exact parameter access", path=f"source.graph.nodes[{node.id!r}]")
            access = accesses[0]
            declaration = declaration_index[access.state_ref]
            hbm = hbm_index[access.state_ref]
            staging = canonical_state_staging_value_id(access.id)
            dma = LiteMoeTask(
                _dma_id(access.id), LiteMoeTaskKind.DMA_IN, die, node.id,
                access.id, access.state_ref, hbm.id, None, None, None,
                declaration.tensor_bytes, declaration.dtype,
                (), (staging,), (), None, None,
            )
            dma.validate(); tasks[die].append(dma)
            deps.append(dma.id)
            reads = [node.inputs[0], staging]
            kind = LiteMoeTaskKind.GEMM
            role = node.id.rsplit(".", 1)[-1]
            if role in ("gate", "up"):
                logical_output = node.outputs[0]
                size_bytes = math.prod(value_index[logical_output].shape) * 2
                packed_output = LiteMoePackedSlice(
                    logical_output,
                    _packed_root(node.id),
                    0 if role == "gate" else size_bytes,
                    size_bytes,
                )
                writes = (packed_output.root_value_ref,)
            else:
                packed_output = None
                writes = node.outputs
        elif node.kind is OpKind.ELEMENTWISE:
            if accesses or type(node.workload) is not SwiGluWorkload:
                raise SchemaError("SwiGLU must be pure and typed", path=f"source.graph.nodes[{node.id!r}]")
            # This carrier keeps the two real gate/up buffers. Their combined
            # FP16 bytes equal the workload's canonical packed (M, 2I) view.
            if (
                len(reads) != 2
                or sum(math.prod(value_index[item].shape) for item in node.inputs)
                != math.prod(node.workload.rank_input_shape)
            ):
                raise SchemaError("gate/up buffers do not exactly cover packed SwiGLU view", path=f"source.graph.nodes[{node.id!r}]")
            kind = LiteMoeTaskKind.SWIGLU
            reads = [_packed_root(node.id)]
            writes = node.outputs
            packed_output = None
        else:
            raise SchemaError("unsupported S3-Lite compute kind", path=f"source.graph.nodes[{node.id!r}]")
        comp = LiteMoeTask(
            _comp_id(node.id), kind, die, node.id,
            None, None, None, None, None, None, 0, DType.FP16,
            tuple(reads), writes, tuple(dict.fromkeys(deps)), node.workload,
            packed_output,
        )
        comp.validate(); tasks[die].append(comp)
        terminal_by_node[node.id] = comp.id

    dies = tuple(LiteMoeProjectedDie(die, tuple(tasks[die])) for die in (0, 1))
    return dies, tuple(flows)


def project_lite_moe(source: LiteMoeN4IR1) -> LiteMoeProjection:
    source.validate("source")
    dies, flows = _projection_components(source)
    result = LiteMoeProjection.create(
        source_n4_id=source.id,
        source_ir1_id=source.graph.id,
        planning_context_id=source.planning_context_id,
        dies=dies,
        flows=flows,
    )
    validate_lite_moe_projection(result, source)
    return result


def validate_lite_moe_projection(result: LiteMoeProjection, source: LiteMoeN4IR1) -> None:
    result.validate(); source.validate("source")
    expected_dies, expected_flows = _projection_components(source)
    if (
        result.source_n4_id != source.id
        or result.source_ir1_id != source.graph.id
        or result.planning_context_id != source.planning_context_id
        or result.dies != expected_dies
        or result.flows != expected_flows
    ):
        raise SchemaError("projection does not exactly quotient N4", path="lite_moe_projection")
    counts = Counter(task.kind for die in result.dies for task in die.tasks)
    if counts != Counter({
        LiteMoeTaskKind.DMA_IN: 24,
        LiteMoeTaskKind.GEMM: 24,
        LiteMoeTaskKind.SWIGLU: 8,
        LiteMoeTaskKind.SEND: 8,
        LiteMoeTaskKind.RECV: 8,
        LiteMoeTaskKind.WAIT: 8,
    }) or sum(flow.bytes for flow in result.flows) != 256:
        raise SchemaError("projection count/byte golden mismatch", path="lite_moe_projection")


def _value_sizes(source: LiteMoeN4IR1) -> dict[str, int]:
    result = {item.id: math.prod(item.shape) * 2 for item in source.graph.values}
    manifest = source.graph.persistent_state_manifest
    assert manifest is not None
    declarations = {item.id: item for item in manifest.declarations}
    for access in source.graph.state_accesses:
        result[canonical_state_staging_value_id(access.id)] = declarations[access.state_ref].tensor_bytes
    for node in source.graph.nodes:
        if node.kind is OpKind.ELEMENTWISE and type(node.workload) is SwiGluWorkload:
            result[_packed_root(node.id)] = math.prod(node.workload.rank_input_shape) * 2
    return result


def _schedule_components(
    projection: LiteMoeProjection,
    source: LiteMoeN4IR1,
) -> tuple[tuple[LiteMoeTaskPlacement, ...], tuple[LiteMoeBufferBinding, ...]]:
    sizes = _value_sizes(source)
    profile_index = {item.id: item for item in source.graph.fabric.sram_profiles}
    placements: list[LiteMoeTaskPlacement] = []
    buffers: list[LiteMoeBufferBinding] = []
    for projected_die in projection.dies:
        die = next(item for item in source.graph.fabric.dies if item.id == projected_die.die_id)
        if not die.cores:
            raise SchemaError("S3-Lite requires a schedulable core per die", path="source.graph.fabric")
        core = die.cores[0]
        profile = profile_index[core.sram_profile_ref]
        region = next((item for item in profile.regions if item.name == "comm"), None)
        if region is None:
            raise SchemaError("S3-Lite requires the comm SRAM region", path="source.graph.fabric")
        for ordinal, task in enumerate(projected_die.tasks):
            placements.append(LiteMoeTaskPlacement(task.id, die.id, core.id, ordinal))
        uses: dict[str, list[int]] = {}
        order: list[str] = []
        for ordinal, task in enumerate(projected_die.tasks):
            for value_ref in task.read_values + task.write_values:
                if value_ref not in uses: uses[value_ref] = []; order.append(value_ref)
                uses[value_ref].append(ordinal)
        cursor = region.base_bytes
        for value_ref in order:
            size = sizes[value_ref]
            address = (cursor + 63) // 64 * 64
            if address + size > region.base_bytes + region.size_bytes:
                raise SchemaError("S3-Lite buffers exceed SRAM capacity", path="lite_moe_schedule")
            buffers.append(LiteMoeBufferBinding(
                stable_artifact_id(
                    "s3_lite_static_moe_buffer",
                    {"value_ref": value_ref, "die_id": die.id, "core_ref": core.id, "address": address, "size_bytes": size},
                    schema_version=LITE_MOE_SCHEDULE_SCHEMA_VERSION,
                ),
                value_ref, die.id, core.id, address, size,
                min(uses[value_ref]), max(uses[value_ref]),
            ))
            cursor = address + size
    return tuple(placements), tuple(buffers)


def schedule_lite_moe(projection: LiteMoeProjection, source: LiteMoeN4IR1) -> LiteMoeScheduled:
    validate_lite_moe_projection(projection, source)
    placements, buffers = _schedule_components(projection, source)
    result = LiteMoeScheduled.create(
        source_projection_id=projection.id,
        source_n4_id=source.id,
        placements=placements,
        buffers=buffers,
    )
    validate_lite_moe_schedule(result, projection, source)
    return result


def validate_lite_moe_schedule(result: LiteMoeScheduled, projection: LiteMoeProjection, source: LiteMoeN4IR1) -> None:
    result.validate(); validate_lite_moe_projection(projection, source)
    expected_placements, expected_buffers = _schedule_components(projection, source)
    if (
        result.source_projection_id != projection.id
        or result.source_n4_id != source.id
        or result.placements != expected_placements
        or result.buffers != expected_buffers
    ):
        raise SchemaError("schedule does not exactly bind projection", path="lite_moe_schedule")


def build_lite_moe_global(
    schedule: LiteMoeScheduled,
    projection: LiteMoeProjection,
    source: LiteMoeN4IR1,
) -> LiteMoeGlobalDag:
    validate_lite_moe_schedule(schedule, projection, source)
    task_index = {task.id: task for die in projection.dies for task in die.tasks}
    buffer_index = {(item.die_id, item.value_ref): item for item in schedule.buffers}
    flow_by_task = {
        task_ref: flow.id
        for flow in projection.flows
        for task_ref in (flow.send_task_ref, flow.recv_task_ref, flow.wait_task_ref)
    }
    actions: list[LiteMoeGlobalAction] = []
    for placement in schedule.placements:
        task = task_index[placement.task_ref]
        uses = tuple(
            LiteMoeBufferUse(buffer_index[(task.die_id, value_ref)].id, access)
            for access, values in (
                (LiteMoeBufferAccess.READ, task.read_values),
                (LiteMoeBufferAccess.WRITE, task.write_values),
            )
            for value_ref in values
        )
        actions.append(LiteMoeGlobalAction(
            f"moe.action.{task.id}", task.id, task.kind, task.die_id,
            placement.core_ref, tuple(f"moe.action.{dep}" for dep in task.deps),
            uses, flow_by_task.get(task.id), task.hbm_binding_ref,
        ))
    result = LiteMoeGlobalDag.create(
        source_schedule_id=schedule.id,
        source_projection_id=projection.id,
        source_n4_id=source.id,
        actions=tuple(actions),
    )
    validate_lite_moe_global(result, schedule, projection, source)
    return result


def validate_lite_moe_global(
    result: LiteMoeGlobalDag,
    schedule: LiteMoeScheduled,
    projection: LiteMoeProjection,
    source: LiteMoeN4IR1,
) -> None:
    result.validate(); validate_lite_moe_schedule(schedule, projection, source)
    task_index = {task.id: task for die in projection.dies for task in die.tasks}
    placement_index = {item.task_ref: item for item in schedule.placements}
    buffer_index = {(item.die_id, item.value_ref): item for item in schedule.buffers}
    flow_by_task = {task_ref: flow.id for flow in projection.flows for task_ref in (flow.send_task_ref, flow.recv_task_ref, flow.wait_task_ref)}
    expected: list[LiteMoeGlobalAction] = []
    for placement in schedule.placements:
        task = task_index[placement.task_ref]
        expected.append(LiteMoeGlobalAction(
            f"moe.action.{task.id}", task.id, task.kind, task.die_id,
            placement_index[task.id].core_ref,
            tuple(f"moe.action.{dep}" for dep in task.deps),
            tuple(
                LiteMoeBufferUse(buffer_index[(task.die_id, value_ref)].id, access)
                for access, values in ((LiteMoeBufferAccess.READ, task.read_values), (LiteMoeBufferAccess.WRITE, task.write_values))
                for value_ref in values
            ),
            flow_by_task.get(task.id), task.hbm_binding_ref,
        ))
    if (
        result.source_schedule_id != schedule.id
        or result.source_projection_id != projection.id
        or result.source_n4_id != source.id
        or result.actions != tuple(expected)
    ):
        raise SchemaError("global actions do not exactly quotient schedule", path="lite_moe_global")


__all__ = [
    "build_lite_moe_global", "project_lite_moe", "schedule_lite_moe",
    "validate_lite_moe_global", "validate_lite_moe_projection",
    "validate_lite_moe_schedule",
]
