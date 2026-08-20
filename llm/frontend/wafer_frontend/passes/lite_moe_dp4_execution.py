"""Project, schedule, and globalize fixed EP4 S3-Lite MoE forward."""

from __future__ import annotations

from collections import Counter
import math

from ..errors import SchemaError
from ..schema.common import DType, stable_artifact_id
from ..schema.experiment import ExperimentSpec
from ..schema.ir0 import GemmWorkload, OpKind, SwiGluWorkload
from ..schema.lite_moe_dp4 import LiteMoeDp4IR0Adapter, LiteMoeDp4N4IR1
from ..schema.lite_moe_dp4_execution import (
    LITE_MOE_DP4_SCHEDULE_SCHEMA_VERSION,
    LiteMoeDp4BufferAccess,
    LiteMoeDp4BufferBinding,
    LiteMoeDp4BufferUse,
    LiteMoeDp4ExecutionCase,
    LiteMoeDp4Flow,
    LiteMoeDp4GlobalAction,
    LiteMoeDp4GlobalDag,
    LiteMoeDp4PackedSlice,
    LiteMoeDp4ProjectedDie,
    LiteMoeDp4Projection,
    LiteMoeDp4Scheduled,
    LiteMoeDp4Task,
    LiteMoeDp4TaskKind,
    LiteMoeDp4TaskPlacement,
)
from ..schema.n4 import FusionPartitionContext, InterDiePlanningContext
from ..schema.persistent_state import canonical_state_staging_value_id
from ..schema.placement import PlacementContext
from .lite_moe_dp4 import (
    build_lite_moe_dp4_n4,
    place_lite_moe_dp4_adapter,
    validate_lite_moe_dp4_n4,
    validate_lite_moe_dp4_placement,
)


def _expert(node_ref: str) -> int:
    marker = ".expert"
    text = node_ref.split(marker, 1)[1].split(".", 1)[0] if marker in node_ref else ""
    if text not in ("0", "1", "2", "3"):
        raise SchemaError("node lacks exact expert lineage", path="lite_moe_dp4_execution")
    return int(text)


def _task_id(node_ref: str, suffix: str) -> str:
    return f"moe.dp4.task.{node_ref}.{suffix}"


def _dma_id(access_ref: str) -> str:
    return f"moe.dp4.task.{access_ref}.dma_in"


def _flow_id(binding_ref: str) -> str:
    return f"moe.dp4.flow.{binding_ref}"


def _packed_root(node_ref: str) -> str:
    return f"moe.dp4.value.{node_ref.rsplit('.', 1)[0]}.gate_up"


def _components(
    source: LiteMoeDp4N4IR1,
) -> tuple[tuple[LiteMoeDp4ProjectedDie, ...], tuple[LiteMoeDp4Flow, ...]]:
    source.validate("source")
    graph = source.graph
    values = {item.id: item for item in graph.values}
    binding_by_node = {item.node_ref: item for item in source.p2p_bindings}
    accesses = {}
    for access in graph.state_accesses:
        accesses.setdefault(access.node_ref, []).append(access)
    manifest = graph.persistent_state_manifest
    assert manifest is not None
    declarations = {item.id: item for item in manifest.declarations}
    hbm = {item.state_ref: item for item in manifest.bindings}
    group = graph.groups[0]
    die_to_rank = {item.die_id: item.rank for item in group.placements}
    routes = {
        (item.source_rank, item.destination_rank): item
        for item in group.embedding.routes
    }
    tasks = {die: [] for die in range(4)}
    flows = []
    terminal = {}

    def dep(value_ref: str) -> tuple[str, ...]:
        producer = values[value_ref].producer
        return () if producer is None else (terminal[producer],)

    for node in graph.nodes:
        if node.kind is OpKind.P2P:
            binding = binding_by_node.get(node.id)
            if binding is None:
                raise SchemaError("P2P node lacks DP4 binding", path="source.p2p_bindings")
            route = routes.get(
                (
                    die_to_rank[binding.source_die_id],
                    die_to_rank[binding.destination_die_id],
                )
            )
            if (
                route is None
                or route.die_path[0] != binding.source_die_id
                or route.die_path[-1] != binding.destination_die_id
            ):
                raise SchemaError("binding lacks exact production PairRoute", path="source.graph.groups")
            send_id = _task_id(node.id, "send")
            recv_id = _task_id(node.id, "recv")
            wait_id = _task_id(node.id, "wait")
            send = LiteMoeDp4Task(
                send_id,
                LiteMoeDp4TaskKind.SEND,
                binding.source_die_id,
                node.id,
                None,
                None,
                None,
                binding.id,
                route.id,
                binding.destination_die_id,
                32,
                DType.FP16,
                (node.inputs[0],),
                (),
                dep(node.inputs[0]),
                None,
                None,
            )
            recv = LiteMoeDp4Task(
                recv_id,
                LiteMoeDp4TaskKind.RECV,
                binding.destination_die_id,
                node.id,
                None,
                None,
                None,
                binding.id,
                route.id,
                binding.source_die_id,
                32,
                DType.FP16,
                (),
                (node.outputs[0],),
                (),
                None,
                None,
            )
            wait = LiteMoeDp4Task(
                wait_id,
                LiteMoeDp4TaskKind.WAIT,
                binding.destination_die_id,
                node.id,
                None,
                None,
                None,
                binding.id,
                route.id,
                binding.source_die_id,
                32,
                DType.FP16,
                (),
                (),
                (recv.id,),
                None,
                None,
            )
            for item in (send, recv, wait):
                item.validate()
                tasks[item.die_id].append(item)
            flow = LiteMoeDp4Flow(
                _flow_id(binding.id),
                binding.id,
                route.id,
                binding.token_index,
                binding.expert_index,
                binding.source_die_id,
                binding.destination_die_id,
                node.inputs[0],
                node.outputs[0],
                32,
                DType.FP16,
                send.id,
                recv.id,
                wait.id,
            )
            flow.validate()
            flows.append(flow)
            terminal[node.id] = wait.id
            continue

        die = _expert(node.id)
        node_deps = []
        reads = list(node.inputs)
        for value_ref in node.inputs:
            producer = values[value_ref].producer
            if producer is not None:
                node_deps.append(terminal[producer])
        node_accesses = accesses.get(node.id, [])
        if node.kind is OpKind.GEMM:
            if len(node_accesses) != 1 or type(node.workload) is not GemmWorkload:
                raise SchemaError("GEMM requires one parameter access", path=node.id)
            access = node_accesses[0]
            declaration = declarations[access.state_ref]
            binding = hbm[access.state_ref]
            staging = canonical_state_staging_value_id(access.id)
            dma = LiteMoeDp4Task(
                _dma_id(access.id),
                LiteMoeDp4TaskKind.DMA_IN,
                die,
                node.id,
                access.id,
                access.state_ref,
                binding.id,
                None,
                None,
                None,
                declaration.tensor_bytes,
                declaration.dtype,
                (),
                (staging,),
                (),
                None,
                None,
            )
            dma.validate()
            tasks[die].append(dma)
            node_deps.append(dma.id)
            reads = [node.inputs[0], staging]
            kind = LiteMoeDp4TaskKind.GEMM
            role = node.id.rsplit(".", 1)[-1]
            if role in ("gate", "up"):
                logical = node.outputs[0]
                packed = LiteMoeDp4PackedSlice(
                    logical,
                    _packed_root(node.id),
                    0 if role == "gate" else 64,
                    64,
                )
                writes = (packed.root_value_ref,)
            else:
                packed = None
                writes = node.outputs
        elif node.kind is OpKind.ELEMENTWISE:
            if node_accesses or type(node.workload) is not SwiGluWorkload:
                raise SchemaError("SwiGLU must be pure and typed", path=node.id)
            kind = LiteMoeDp4TaskKind.SWIGLU
            reads = [_packed_root(node.id)]
            writes = node.outputs
            packed = None
        else:
            raise SchemaError("unsupported DP4 compute kind", path=node.id)
        task = LiteMoeDp4Task(
            _task_id(node.id, "comp"),
            kind,
            die,
            node.id,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            DType.FP16,
            tuple(reads),
            writes,
            tuple(dict.fromkeys(node_deps)),
            node.workload,
            packed,
        )
        task.validate()
        tasks[die].append(task)
        terminal[node.id] = task.id
    return (
        tuple(LiteMoeDp4ProjectedDie(die, tuple(tasks[die])) for die in range(4)),
        tuple(flows),
    )


def project_lite_moe_dp4(source: LiteMoeDp4N4IR1) -> LiteMoeDp4Projection:
    dies, flows = _components(source)
    result = LiteMoeDp4Projection.create(
        source_n4_id=source.id,
        source_ir1_id=source.graph.id,
        planning_context_id=source.planning_context_id,
        dies=dies,
        flows=flows,
    )
    validate_lite_moe_dp4_projection(result, source)
    return result


def validate_lite_moe_dp4_projection(
    result: LiteMoeDp4Projection,
    source: LiteMoeDp4N4IR1,
) -> None:
    result.validate()
    dies, flows = _components(source)
    if (
        result.source_n4_id != source.id
        or result.source_ir1_id != source.graph.id
        or result.planning_context_id != source.planning_context_id
        or result.dies != dies
        or result.flows != flows
    ):
        raise SchemaError("projection is not exact N4 quotient", path="projection")
    counts = Counter(task.kind for die in result.dies for task in die.tasks)
    if counts != Counter(
        {
            LiteMoeDp4TaskKind.DMA_IN: 24,
            LiteMoeDp4TaskKind.GEMM: 24,
            LiteMoeDp4TaskKind.SWIGLU: 8,
            LiteMoeDp4TaskKind.SEND: 12,
            LiteMoeDp4TaskKind.RECV: 12,
            LiteMoeDp4TaskKind.WAIT: 12,
        }
    ):
        raise SchemaError("projection task counts changed", path="projection")


def _value_sizes(source: LiteMoeDp4N4IR1) -> dict[str, int]:
    sizes = {item.id: math.prod(item.shape) * 2 for item in source.graph.values}
    manifest = source.graph.persistent_state_manifest
    assert manifest is not None
    declarations = {item.id: item for item in manifest.declarations}
    for access in source.graph.state_accesses:
        sizes[canonical_state_staging_value_id(access.id)] = declarations[
            access.state_ref
        ].tensor_bytes
    for node in source.graph.nodes:
        if node.kind is OpKind.ELEMENTWISE:
            sizes[_packed_root(node.id)] = 128
    return sizes


def _schedule_components(
    projection: LiteMoeDp4Projection,
    source: LiteMoeDp4N4IR1,
) -> tuple[tuple[LiteMoeDp4TaskPlacement, ...], tuple[LiteMoeDp4BufferBinding, ...]]:
    sizes = _value_sizes(source)
    profiles = {item.id: item for item in source.graph.fabric.sram_profiles}
    placements = []
    buffers = []
    for projected in projection.dies:
        die = next(item for item in source.graph.fabric.dies if item.id == projected.die_id)
        core = die.cores[0]
        region = next(
            item for item in profiles[core.sram_profile_ref].regions if item.name == "comm"
        )
        for ordinal, task in enumerate(projected.tasks):
            placements.append(LiteMoeDp4TaskPlacement(task.id, die.id, core.id, ordinal))
        uses = {}
        order = []
        for ordinal, task in enumerate(projected.tasks):
            for value_ref in task.read_values + task.write_values:
                if value_ref not in uses:
                    uses[value_ref] = []
                    order.append(value_ref)
                uses[value_ref].append(ordinal)
        cursor = region.base_bytes
        for value_ref in order:
            size = sizes[value_ref]
            address = (cursor + 63) // 64 * 64
            if address + size > region.base_bytes + region.size_bytes:
                raise SchemaError("DP4 buffers exceed comm SRAM", path="schedule")
            semantic = {
                "value_ref": value_ref,
                "die_id": die.id,
                "core_ref": core.id,
                "address": address,
                "size_bytes": size,
                "first_ordinal": min(uses[value_ref]),
                "last_ordinal": max(uses[value_ref]),
            }
            buffers.append(
                LiteMoeDp4BufferBinding(
                    stable_artifact_id(
                        "s3_lite_moe_dp4_buffer",
                        semantic,
                        schema_version=LITE_MOE_DP4_SCHEDULE_SCHEMA_VERSION,
                    ),
                    **semantic,
                )
            )
            cursor = address + size
    return tuple(placements), tuple(buffers)


def schedule_lite_moe_dp4(
    projection: LiteMoeDp4Projection,
    source: LiteMoeDp4N4IR1,
) -> LiteMoeDp4Scheduled:
    validate_lite_moe_dp4_projection(projection, source)
    placements, buffers = _schedule_components(projection, source)
    result = LiteMoeDp4Scheduled.create(
        source_projection_id=projection.id,
        source_n4_id=source.id,
        placements=placements,
        buffers=buffers,
    )
    validate_lite_moe_dp4_schedule(result, projection, source)
    return result


def validate_lite_moe_dp4_schedule(
    result: LiteMoeDp4Scheduled,
    projection: LiteMoeDp4Projection,
    source: LiteMoeDp4N4IR1,
) -> None:
    result.validate()
    validate_lite_moe_dp4_projection(projection, source)
    expected = _schedule_components(projection, source)
    if (
        result.source_projection_id != projection.id
        or result.source_n4_id != source.id
        or (result.placements, result.buffers) != expected
    ):
        raise SchemaError("schedule is not exact projection quotient", path="schedule")


def _combined_output_refs(source: LiteMoeDp4N4IR1) -> tuple[str, ...]:
    result = []
    for assignment in source.source.source.spec.trace.assignments:
        prefix = f"S3M4.token{assignment.token_index}.expert{assignment.expert_index}.value"
        result.append(
            f"{prefix}.down"
            if assignment.token_index % 4 == assignment.expert_index
            else f"{prefix}.combined"
        )
    return tuple(result)


def _global_components(
    schedule: LiteMoeDp4Scheduled,
    projection: LiteMoeDp4Projection,
) -> tuple[LiteMoeDp4GlobalAction, ...]:
    tasks = {task.id: task for die in projection.dies for task in die.tasks}
    buffers = {(item.die_id, item.value_ref): item for item in schedule.buffers}
    flow_by_task = {
        task_ref: flow.id
        for flow in projection.flows
        for task_ref in (flow.send_task_ref, flow.recv_task_ref, flow.wait_task_ref)
    }
    result = []
    for placement in schedule.placements:
        task = tasks[placement.task_ref]
        uses = tuple(
            LiteMoeDp4BufferUse(buffers[(task.die_id, value_ref)].id, access)
            for access, values in (
                (LiteMoeDp4BufferAccess.READ, task.read_values),
                (LiteMoeDp4BufferAccess.WRITE, task.write_values),
            )
            for value_ref in values
        )
        result.append(
            LiteMoeDp4GlobalAction(
                f"moe.dp4.action.{task.id}",
                task.id,
                task.kind,
                task.die_id,
                placement.core_ref,
                tuple(f"moe.dp4.action.{dep}" for dep in task.deps),
                uses,
                flow_by_task.get(task.id),
                task.hbm_binding_ref,
            )
        )
    return tuple(result)


def build_lite_moe_dp4_global(
    schedule: LiteMoeDp4Scheduled,
    projection: LiteMoeDp4Projection,
    source: LiteMoeDp4N4IR1,
) -> LiteMoeDp4GlobalDag:
    validate_lite_moe_dp4_schedule(schedule, projection, source)
    result = LiteMoeDp4GlobalDag.create(
        source_schedule_id=schedule.id,
        source_projection_id=projection.id,
        source_n4_id=source.id,
        actions=_global_components(schedule, projection),
        combined_output_refs=_combined_output_refs(source),
    )
    validate_lite_moe_dp4_global(result, schedule, projection, source)
    return result


def validate_lite_moe_dp4_global(
    result: LiteMoeDp4GlobalDag,
    schedule: LiteMoeDp4Scheduled,
    projection: LiteMoeDp4Projection,
    source: LiteMoeDp4N4IR1,
) -> None:
    result.validate()
    validate_lite_moe_dp4_schedule(schedule, projection, source)
    if (
        result.source_schedule_id != schedule.id
        or result.source_projection_id != projection.id
        or result.source_n4_id != source.id
        or result.actions != _global_components(schedule, projection)
        or result.combined_output_refs != _combined_output_refs(source)
    ):
        raise SchemaError("global is not exact schedule quotient", path="global")


def build_lite_moe_dp4_execution_case(
    experiment: ExperimentSpec,
    adapter: LiteMoeDp4IR0Adapter,
    placement_context: PlacementContext,
    partition_context: FusionPartitionContext,
    planning_context: InterDiePlanningContext,
) -> LiteMoeDp4ExecutionCase:
    placed = place_lite_moe_dp4_adapter(adapter, experiment, placement_context)
    n4 = build_lite_moe_dp4_n4(placed, partition_context, planning_context)
    projection = project_lite_moe_dp4(n4)
    schedule = schedule_lite_moe_dp4(projection, n4)
    global_dag = build_lite_moe_dp4_global(schedule, projection, n4)
    result = LiteMoeDp4ExecutionCase.create(
        experiment=experiment,
        adapter=adapter,
        placement_context=placement_context,
        partition_context=partition_context,
        planning_context=planning_context,
        placed=placed,
        n4=n4,
        projection=projection,
        schedule=schedule,
        global_dag=global_dag,
    )
    validate_lite_moe_dp4_execution_case(result)
    return result


def validate_lite_moe_dp4_execution_case(result: LiteMoeDp4ExecutionCase) -> None:
    result.validate()
    validate_lite_moe_dp4_placement(result.placed, result.adapter, result.placement_context)
    validate_lite_moe_dp4_n4(
        result.n4,
        result.placed,
        result.partition_context,
        result.planning_context,
    )
    validate_lite_moe_dp4_projection(result.projection, result.n4)
    validate_lite_moe_dp4_schedule(result.schedule, result.projection, result.n4)
    validate_lite_moe_dp4_global(
        result.global_dag,
        result.schedule,
        result.projection,
        result.n4,
    )


__all__ = [
    "build_lite_moe_dp4_execution_case",
    "build_lite_moe_dp4_global",
    "project_lite_moe_dp4",
    "schedule_lite_moe_dp4",
    "validate_lite_moe_dp4_execution_case",
    "validate_lite_moe_dp4_global",
    "validate_lite_moe_dp4_projection",
    "validate_lite_moe_dp4_schedule",
]
