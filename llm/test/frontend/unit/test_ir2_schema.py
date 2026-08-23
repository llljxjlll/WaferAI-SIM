from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    DirectAllGatherPolicy,
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.schema.common import (
    DType,
    MeshAxisName,
    RoundingMode,
    Sharding,
    UINT64_MAX,
)
from llm.frontend.wafer_frontend.schema.action import (
    canonical_compute_operand_roles,
    ComputeContract,
    ComputeOperand,
    ConsumerLayoutBinding,
    FusionActionKind,
    FusionPlan,
    RankProgram,
    ReductionContract,
    StandaloneCollectivePlan,
    SyncContract,
)
from llm.frontend.wafer_frontend.schema.ir0 import (
    CollectiveKind,
    EdgeKind,
    ElementwiseWorkload,
    GraphEdge,
    OpKind,
    PackedQkvLayout,
    ReduceOp,
    RopeQkWorkload,
    SwiGluWorkload,
)
from llm.frontend.wafer_frontend.schema.ir1 import IR1, MemoryInitiator
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferAccess,
    BufferBinding,
    BufferOwnership,
    BufferUseRole,
    CoreOrder,
    FlowRouteBinding,
    FlowRouteRole,
    FusedNodeOrigin,
    IntraDieDAG,
    IntraDieRegion,
    IntraDieSchedule,
    IntraDieValue,
    SwizzleIntraDieValue,
    IR2ProjectionResult,
    LogicalRuntimeBinding,
    OrdinaryNodeOrigin,
    OriginKind,
    PortLeg,
    RegionLowering,
    SemanticFlow,
    SemanticTask,
    SemanticTaskKind,
    StandaloneNodeOrigin,
    StateUseAccess,
    TaskStateUse,
    TaskBufferUse,
    TaskPlacement,
    TensorSlice,
    canonical_semantic_flow_id,
    dense_row_major_view_byte_addend,
)
from llm.frontend.wafer_frontend.schema.swizzle_plan import (
    SwizzleValueOrigin,
    SwizzleValueUse,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
    to_primitive,
)

from _fixtures import valid_ir1
from test_action_schema import (
    bound_standalone_plan,
    valid_ir1 as valid_fusion_ir1,
    valid_plan,
    valid_standalone_plan,
)
from test_naive_inter_die import _partitioned_graph


def valid_dag() -> IntraDieDAG:
    tensor_slice = TensorSlice("v_out", (0, 0), (16, 64))
    ir1 = valid_ir1()
    node = ir1.nodes[0]
    compute = ComputeContract(
        OpKind.ELEMENTWISE,
        ElementwiseWorkload(
            ((16, 64),),
            (16, 64),
            ((16, 64),),
            (16, 64),
            DType.FP16,
        ),
        node.math,
        node.effects,
        node.impl_ref,
        (ComputeOperand("v_in", "input_0"),),
        (ComputeOperand("v_out", "output_0"),),
    )
    comp = SemanticTask(
        id="t_comp", kind=SemanticTaskKind.COMP,
        origin_ref=OrdinaryNodeOrigin(OriginKind.ORDINARY, "p_gemm_0", 0),
        region_id="region_coarse", op_kind=OpKind.ELEMENTWISE, member_id="p_gemm_0",
        flow_id=None, chunk_id=None, collective_step=None, source_rank=None,
        destination_rank=None, tensor_slice=None, bytes=2048, dtype=DType.FP16,
        shape=(16, 64), read_values=("v_in",), write_values=("v_out",),
        compute=compute, reduction=None, sync=None, deps=(),
    )
    send = SemanticTask(
        id="t_send", kind=SemanticTaskKind.SEND,
        origin_ref=FusedNodeOrigin(OriginKind.FUSED, "fp_0", 0, "r0_send0"),
        region_id="region_isa", op_kind=OpKind.COLLECTIVE, member_id="p_rs_0",
        flow_id="flow_0", chunk_id=0, collective_step=0, source_rank=0,
        destination_rank=1, tensor_slice=tensor_slice, bytes=2048,
        dtype=DType.FP16, shape=(16, 64), read_values=("v_out",), write_values=(),
        compute=None, reduction=None, sync=SyncContract("event_t_send", None, None),
        deps=("t_comp",),
    )
    return IntraDieDAG.create(
        producer_pass="project_fixture",
        source_ir1_id=ir1.id,
        die_id=0,
        fusion_plan_ids=("fp_0",),
        standalone_collective_plan_ids=(),
        ordinary_node_ids=("p_gemm_0",),
        tasks=(comp, send),
        values=(
            IntraDieValue("v_in", "fixture_v_in", (16, 64), DType.FP16, "MK", Sharding("mesh_tp", (None, None), ()), None, (), ("t_comp",)),
            IntraDieValue("v_out", "fixture_v_out", (16, 64), DType.FP16, "MN", Sharding("mesh_tp", (None, MeshAxisName.TP), ()), None, ("t_comp",), ("t_send",)),
        ),
        flows=(SemanticFlow("flow_0", "ch_0", "route_0_1", 0, 1, 0, 1, (0, 1), tensor_slice, 2048, DType.FP16, ("t_send",)),),
        regions=(
            IntraDieRegion("region_coarse", None, None, RegionLowering.JSON_COARSE, ("t_comp",)),
            IntraDieRegion("region_isa", "fp_0", None, RegionLowering.ISA_REGION, ("t_send",)),
        ),
    )


def valid_schedule(dag: IntraDieDAG | None = None) -> IntraDieSchedule:
    dag = dag or valid_dag()
    slice_in = TensorSlice("v_in", (0, 0), (16, 64))
    slice_out = TensorSlice("v_out", (0, 0), (16, 64))
    return IntraDieSchedule.create(
        producer_pass="schedule_fixture",
        dag_id=dag.id,
        die_id=0,
        placements=(TaskPlacement("t_comp", 0), TaskPlacement("t_send", 0)),
        buffer_bindings=(
            BufferBinding(
                "b_in", "v_in", slice_in, 0, "sram_main", 0, 2048, 64,
                (0, 1, 2, 3), "storage_in", None, BufferOwnership.BORROWED,
                0, 1, DType.FP16, "MK",
            ),
            BufferBinding(
                "b_out", "v_out", slice_out, 0, "sram_main", 2048, 2048, 64,
                (0, 1, 2, 3), "storage_out", None, BufferOwnership.OWNED,
                0, 2, DType.FP16, "MN",
            ),
        ),
        task_buffer_uses=(
            TaskBufferUse(
                "t_comp", "b_in", BufferAccess.READ,
                BufferUseRole.COMP_INPUT, 0, None, slice_in,
            ),
            TaskBufferUse(
                "t_comp", "b_out", BufferAccess.WRITE,
                BufferUseRole.COMP_OUTPUT, 0, None, slice_out,
            ),
            TaskBufferUse(
                "t_send", "b_out", BufferAccess.READ,
                BufferUseRole.SEND_SOURCE, 0, None, slice_out,
            ),
        ),
        task_state_uses=(),
        flow_routes=(
            FlowRouteBinding(
                "flow_0",
                "route_0_1",
                FlowRouteRole.SOURCE,
                None,
                PortLeg("link_0_1", "east_0"),
                ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
            ),
        ),
        runtime_bindings=(LogicalRuntimeBinding("t_send", "flow_0", "ch_0", "event_t_send", "token_send"),),
        core_orders=(CoreOrder(0, ("t_comp", "t_send")),),
    )


def valid_reduce_case(
    ir1: IR1 | None = None,
) -> tuple[IR1, IntraDieDAG, IntraDieSchedule]:
    ir1 = ir1 or valid_ir1()
    output_slice = TensorSlice("reduce_out", (0,), (16,))
    task = SemanticTask(
        id="t_reduce",
        kind=SemanticTaskKind.REDUCE,
        origin_ref=FusedNodeOrigin(
            OriginKind.FUSED, "fp_reduce", 0, "reduce_action"
        ),
        region_id="region_reduce",
        op_kind=OpKind.COLLECTIVE,
        member_id="p_rs_0",
        flow_id=None,
        chunk_id=0,
        collective_step=0,
        source_rank=None,
        destination_rank=None,
        tensor_slice=output_slice,
        bytes=32,
        dtype=DType.FP16,
        shape=(16,),
        read_values=("reduce_in_0", "reduce_in_1"),
        write_values=("reduce_out",),
        compute=None,
        reduction=ReductionContract(
            ReduceOp.SUM,
            DType.FP16,
            DType.FP32,
            DType.FP16,
            RoundingMode.RNE,
            (0, 1),
        ),
        sync=SyncContract("event_reduce", None, None),
        deps=(),
    )
    sharding = ir1.values[0].sharding
    dag = IntraDieDAG.create(
        producer_pass="reduce_projection_fixture",
        source_ir1_id=ir1.id,
        die_id=0,
        fusion_plan_ids=("fp_reduce",),
        standalone_collective_plan_ids=(),
        ordinary_node_ids=(),
        tasks=(task,),
        values=(
            IntraDieValue(
                "reduce_in_0", "fixture_reduce_in_0", (16,), DType.FP16,
                "flat", sharding, None, (), (task.id,),
            ),
            IntraDieValue(
                "reduce_in_1", "fixture_reduce_in_1", (16,), DType.FP16,
                "flat", sharding, None, (), (task.id,),
            ),
            IntraDieValue(
                "reduce_out", "fixture_reduce_out", (16,), DType.FP16,
                "flat", sharding, None, (task.id,), (),
            ),
        ),
        flows=(),
        regions=(
            IntraDieRegion(
                "region_reduce", "fp_reduce", None,
                RegionLowering.ISA_REGION, (task.id,),
            ),
        ),
    )
    bindings = (
        BufferBinding(
            "b_reduce_in_0", "reduce_in_0",
            TensorSlice("reduce_in_0", (0,), (16,)),
            0, "sram_main", 0, 32, 2, (0,), "storage_reduce_in_0",
            None, BufferOwnership.BORROWED, 0, 1, DType.FP16, "flat",
        ),
        BufferBinding(
            "b_reduce_in_1", "reduce_in_1",
            TensorSlice("reduce_in_1", (0,), (16,)),
            0, "sram_main", 32, 32, 2, (0,), "storage_reduce_in_1",
            None, BufferOwnership.BORROWED, 0, 1, DType.FP16, "flat",
        ),
        BufferBinding(
            "b_reduce_out", "reduce_out",
            TensorSlice("reduce_out", (0,), (16,)),
            0, "sram_main", 64, 32, 2, (1,), "storage_reduce_out",
            None, BufferOwnership.OWNED, 0, 1, DType.FP16, "flat",
        ),
    )
    schedule = IntraDieSchedule.create(
        producer_pass="reduce_schedule_fixture",
        dag_id=dag.id,
        die_id=0,
        placements=(TaskPlacement(task.id, 0),),
        buffer_bindings=bindings,
        task_buffer_uses=(
            TaskBufferUse(
                task.id, bindings[0].id, BufferAccess.READ,
                BufferUseRole.REDUCE_INPUT, 0, 0,
                bindings[0].tensor_slice,
            ),
            TaskBufferUse(
                task.id, bindings[1].id, BufferAccess.READ,
                BufferUseRole.REDUCE_INPUT, 1, 1,
                bindings[1].tensor_slice,
            ),
            TaskBufferUse(
                task.id, bindings[2].id, BufferAccess.WRITE,
                BufferUseRole.REDUCE_OUTPUT, 0, None,
                bindings[2].tensor_slice,
            ),
        ),
        task_state_uses=(),
        flow_routes=(),
        runtime_bindings=(),
        core_orders=(CoreOrder(0, (task.id,)),),
    )
    return ir1, dag, schedule


def recreate_schedule(schedule: IntraDieSchedule, **changes: object) -> IntraDieSchedule:
    fields = schedule._semantic_key()
    fields.update(changes)
    return IntraDieSchedule.create(producer_pass=schedule.producer_pass, **fields)


def recreate_ir1(ir1: IR1, **changes: object) -> IR1:
    fields = ir1._semantic_key()
    fields.update(changes)
    return IR1.create(producer_pass=ir1.producer_pass, **fields)


def bound_plan() -> tuple[IR1, FusionPlan]:
    ir1 = valid_fusion_ir1()
    template = valid_plan()
    fields = template._semantic_key()
    fields.update(
        source_ir1_id=ir1.id,
        profile_key=ir1.profile,
        physical_output_layout=template.logical_output_layout,
        output_permutation=tuple(
            replace(entry, logical_owner_rank=entry.physical_owner_rank)
            for entry in template.output_permutation
        ),
        inverse_permutation=tuple(
            replace(entry, logical_owner_rank=entry.physical_owner_rank)
            for entry in template.inverse_permutation
        ),
    )
    return ir1, FusionPlan.create(producer_pass=template.producer_pass, **fields)


def ordinary_projection() -> tuple[IR1, IR2ProjectionResult]:
    template = valid_ir1()
    source_node = template.nodes[0]
    node = replace(
        source_node,
        kind=OpKind.ELEMENTWISE,
        workload=SwiGluWorkload(
            logical_input_shape=(32, 256),
            logical_output_shape=(32, 128),
            rank_input_shape=(32, 256),
            rank_output_shape=(32, 128),
            dtype=DType.FP16,
        ),
        impl_ref="swiglu",
    )
    fields = template._semantic_key()
    fields.update(
        instances=(replace(template.instances[0], node_ids=(node.id,)),),
        nodes=(node,),
        values=(template.values[0], replace(template.values[1], consumers=())),
        edges=(),
        fusion_candidates=(),
        fused_op_skeletons=(),
    )
    ir1 = IR1.create(producer_pass=template.producer_pass, **fields)
    input_value, output_value = ir1.values
    group = ir1.groups[0]
    dags = []
    for placement in group.placements:
        task_id = f"task_{node.id}_r{placement.rank}"
        region_id = f"region_coarse_r{placement.rank}"
        compute = ComputeContract(
            node.kind,
            node.workload,
            node.math,
            node.effects,
            node.impl_ref,
            (ComputeOperand(node.inputs[0], "gate_up"),),
            (ComputeOperand(node.outputs[0], "swiglu"),),
        )
        task = SemanticTask(
            id=task_id,
            kind=SemanticTaskKind.COMP,
            origin_ref=OrdinaryNodeOrigin(
                OriginKind.ORDINARY, node.id, placement.rank
            ),
            region_id=region_id,
            op_kind=node.kind,
            member_id=node.id,
            flow_id=None,
            chunk_id=None,
            collective_step=None,
            source_rank=None,
            destination_rank=None,
            tensor_slice=None,
            bytes=0,
            dtype=None,
            shape=(),
            read_values=node.inputs,
            write_values=node.outputs,
            compute=compute,
            reduction=None,
            sync=None,
            deps=(),
        )
        dags.append(
            IntraDieDAG.create(
                producer_pass="ordinary_projection_fixture",
                source_ir1_id=ir1.id,
                die_id=placement.die_id,
                fusion_plan_ids=(),
                standalone_collective_plan_ids=(),
                ordinary_node_ids=(node.id,),
                tasks=(task,),
                values=(
                    IntraDieValue(
                        input_value.id,
                        input_value.id,
                        input_value.shape,
                        input_value.dtype,
                        input_value.logical_layout,
                        input_value.sharding,
                        input_value.alias_set,
                        (),
                        (task.id,),
                    ),
                    IntraDieValue(
                        output_value.id,
                        output_value.id,
                        output_value.shape,
                        output_value.dtype,
                        output_value.logical_layout,
                        output_value.sharding,
                        output_value.alias_set,
                        (task.id,),
                        (),
                    ),
                ),
                flows=(),
                regions=(
                    IntraDieRegion(
                        region_id,
                        None,
                        None,
                        RegionLowering.JSON_COARSE,
                        (task.id,),
                    ),
                ),
            )
        )
    return ir1, IR2ProjectionResult.create(
        producer_pass="ordinary_projection_fixture",
        source_ir1_id=ir1.id,
        fusion_plan_ids=(),
        standalone_collective_plan_ids=(),
        dags=tuple(dags),
    )


def _exact_projection(
    ir1: IR1,
    plan: FusionPlan | StandaloneCollectivePlan,
    *,
    standalone: bool,
) -> IR2ProjectionResult:
    group = next(group for group in ir1.groups if group.id == plan.group_ref)
    rank_to_die = {placement.rank: placement.die_id for placement in group.placements}
    route_index = {
        (route.source_rank, route.destination_rank, route.die_path): route
        for route in group.embedding.routes
    }
    chunk_index = {chunk.id: chunk for chunk in plan.chunk_slices}
    ir1_values = {value.id: value for value in ir1.values}
    partial_template = ir1_values["p_v_partial"]
    temp_origins: dict[str, str] = {}
    if not standalone:
        skeleton = next(
            item for item in ir1.fused_op_skeletons if item.id == plan.fused_op_id
        )
        node_index = {node.id: node for node in ir1.nodes}
        partial_value_id = node_index[skeleton.member_node_ids[0]].outputs[0]
        for program in plan.rank_programs:
            for source in program.actions:
                if source.compute is not None and source.compute.tile is not None:
                    for binding in (
                        source.compute.tile.input_slices
                        + source.compute.tile.output_slices
                    ):
                        temp_origins[binding.operand_id] = binding.source_value_id
                if source.kind is FusionActionKind.RECV:
                    for value_id in source.writes:
                        temp_origins[value_id] = partial_value_id
    send_origins = {
        action.logical_channel: (
            StandaloneNodeOrigin(
                OriginKind.STANDALONE_COLLECTIVE,
                plan.id,
                program.rank,
                action.id,
            )
            if standalone
            else FusedNodeOrigin(
                OriginKind.FUSED, plan.id, program.rank, action.id
            )
        )
        for program in plan.rank_programs
        for action in program.actions
        if action.kind is FusionActionKind.SEND
    }
    dags: list[IntraDieDAG] = []
    for program in plan.rank_programs:
        region_id = f"region_{program.rank}"
        task_ids = {action.id: f"task_{action.id}" for action in program.actions}
        tasks: list[SemanticTask] = []
        flows: list[SemanticFlow] = []
        for source in program.actions:
            chunk = chunk_index.get(source.slice_ref or "")
            tensor_slice = (
                TensorSlice(chunk.value_id, chunk.offset, chunk.shape)
                if chunk is not None
                else None
            )
            source_rank = destination_rank = None
            flow_id = None
            if source.kind in (FusionActionKind.SEND, FusionActionKind.RECV):
                source_rank, destination_rank = (
                    (program.rank, source.peer_rank)
                    if source.kind is FusionActionKind.SEND
                    else (source.peer_rank, program.rank)
                )
                route = route_index[(source_rank, destination_rank, source.expected_route)]
                flow_id = canonical_semantic_flow_id(
                    send_origins[source.logical_channel],
                    source.logical_channel,
                )
                flows.append(
                    SemanticFlow(
                        flow_id,
                        source.logical_channel or "",
                        route.id,
                        source_rank,
                        destination_rank,
                        source.expected_route[0],
                        source.expected_route[-1],
                        source.expected_route,
                        tensor_slice,
                        source.bytes,
                        source.dtype,
                        (task_ids[source.id],),
                    )
                )
            tasks.append(
                SemanticTask(
                    id=task_ids[source.id],
                    kind=SemanticTaskKind(source.kind.value),
                    origin_ref=(
                        StandaloneNodeOrigin(
                            OriginKind.STANDALONE_COLLECTIVE,
                            plan.id,
                            program.rank,
                            source.id,
                        )
                        if standalone
                        else FusedNodeOrigin(
                            OriginKind.FUSED, plan.id, program.rank, source.id
                        )
                    ),
                    region_id=region_id,
                    op_kind=(
                        source.compute.op_kind
                        if source.compute is not None
                        else OpKind.COLLECTIVE
                    ),
                    member_id=source.member_id,
                    flow_id=flow_id,
                    chunk_id=source.chunk_id,
                    collective_step=source.collective_step,
                    source_rank=source_rank,
                    destination_rank=destination_rank,
                    tensor_slice=tensor_slice,
                    bytes=source.bytes,
                    dtype=source.dtype,
                    shape=chunk.shape if chunk is not None else (),
                    read_values=source.reads,
                    write_values=source.writes,
                    compute=source.compute,
                    reduction=source.reduction,
                    sync=source.sync,
                    deps=tuple(task_ids[dependency] for dependency in source.deps),
                )
            )
        all_refs = tuple(
            dict.fromkeys(
                value_id
                for task in tasks
                for value_id in task.read_values + task.write_values
            )
        )
        local_values: list[IntraDieValue] = []
        for value_id in all_refs:
            producer_tasks = sorted(
                (task for task in tasks if value_id in task.write_values),
                key=lambda task: (
                    task.tensor_slice.offset if task.tensor_slice is not None else (),
                    task.tensor_slice.shape if task.tensor_slice is not None else (),
                    task.id,
                ),
            )
            producers = tuple(task.id for task in producer_tasks)
            consumers = tuple(task.id for task in tasks if value_id in task.read_values)
            origin = ir1_values.get(temp_origins.get(value_id, value_id))
            if origin is None:
                related = next(
                    task
                    for task in tasks
                    if value_id in task.read_values + task.write_values
                )
                local_values.append(
                    IntraDieValue(
                        value_id,
                        f"synthetic:{value_id}",
                        related.shape,
                        related.dtype or DType.FP16,
                        partial_template.logical_layout,
                        partial_template.sharding,
                        None,
                        producers,
                        consumers,
                    )
                )
            else:
                local_values.append(
                    IntraDieValue(
                        value_id,
                        origin.id,
                        origin.shape,
                        origin.dtype,
                        origin.logical_layout,
                        origin.sharding,
                        origin.alias_set,
                        producers,
                        consumers,
                    )
                )
        dags.append(
            IntraDieDAG.create(
                producer_pass="exact_projection_fixture",
                source_ir1_id=ir1.id,
                die_id=rank_to_die[program.rank],
                fusion_plan_ids=() if standalone else (plan.id,),
                standalone_collective_plan_ids=(plan.id,) if standalone else (),
                ordinary_node_ids=(),
                tasks=tuple(tasks),
                values=tuple(local_values),
                flows=tuple(flows),
                regions=(
                    IntraDieRegion(
                        region_id,
                        None if standalone else plan.id,
                        plan.id if standalone else None,
                        (
                            RegionLowering.STRICT_ACTIONS
                            if standalone
                            else RegionLowering.ISA_REGION
                        ),
                        tuple(task.id for task in tasks),
                    ),
                ),
            )
        )
    result = IR2ProjectionResult.create(
        producer_pass="exact_projection_fixture",
        source_ir1_id=ir1.id,
        fusion_plan_ids=() if standalone else (plan.id,),
        standalone_collective_plan_ids=(plan.id,) if standalone else (),
        dags=tuple(dags),
    )
    return result


def exact_projection() -> tuple[IR1, FusionPlan, IR2ProjectionResult]:
    ir1, plan = bound_plan()
    return ir1, plan, _exact_projection(ir1, plan, standalone=False)


def exact_standalone_projection() -> tuple[
    IR1, StandaloneCollectivePlan, IR2ProjectionResult
]:
    ir1, plan = bound_standalone_plan()
    return ir1, plan, _exact_projection(ir1, plan, standalone=True)


def _ordinary_task(node: object, rank: int, deps: tuple[str, ...]) -> SemanticTask:
    task_id = f"task_{node.id}_r{rank}"
    input_roles, output_roles = canonical_compute_operand_roles(
        node.kind, node.workload, tiled=False
    )
    return SemanticTask(
        id=task_id,
        kind=SemanticTaskKind.COMP,
        origin_ref=OrdinaryNodeOrigin(OriginKind.ORDINARY, node.id, rank),
        region_id=f"region_{node.id}_r{rank}",
        op_kind=node.kind,
        member_id=node.id,
        flow_id=None,
        chunk_id=None,
        collective_step=None,
        source_rank=None,
        destination_rank=None,
        tensor_slice=None,
        bytes=0,
        dtype=None,
        shape=(),
        read_values=node.inputs,
        write_values=node.outputs,
        compute=ComputeContract(
            node.kind,
            node.workload,
            node.math,
            node.effects,
            node.impl_ref,
            tuple(
                ComputeOperand(value_id, role)
                for value_id, role in zip(
                    node.inputs, input_roles, strict=True
                )
            ),
            tuple(
                ComputeOperand(value_id, role)
                for value_id, role in zip(
                    node.outputs, output_roles, strict=True
                )
            ),
        ),
        reduction=None,
        sync=None,
        deps=deps,
    )


def _local_values_for_tasks(
    ir1: IR1,
    tasks: tuple[SemanticTask, ...],
    fusion_plan: FusionPlan,
) -> tuple[IntraDieValue, ...]:
    ir1_values = {value.id: value for value in ir1.values}
    partial_template = ir1_values["p_v_partial"]
    skeleton = next(
        item
        for item in ir1.fused_op_skeletons
        if item.id == fusion_plan.fused_op_id
    )
    node_index = {node.id: node for node in ir1.nodes}
    partial_value_id = node_index[skeleton.member_node_ids[0]].outputs[0]
    temp_origins: dict[str, str] = {}
    for program in fusion_plan.rank_programs:
        for source in program.actions:
            if source.compute is not None and source.compute.tile is not None:
                for binding in (
                    source.compute.tile.input_slices
                    + source.compute.tile.output_slices
                ):
                    temp_origins[binding.operand_id] = binding.source_value_id
            if source.kind is FusionActionKind.RECV:
                for value_id in source.writes:
                    temp_origins[value_id] = partial_value_id
    value_ids = tuple(
        dict.fromkeys(
            value_id
            for task in tasks
            for value_id in task.read_values + task.write_values
        )
    )
    result = []
    for value_id in value_ids:
        writers = sorted(
            (task for task in tasks if value_id in task.write_values),
            key=lambda task: (
                task.tensor_slice.offset if task.tensor_slice is not None else (),
                task.tensor_slice.shape if task.tensor_slice is not None else (),
                task.id,
            ),
        )
        producer_tasks = tuple(task.id for task in writers)
        consumer_tasks = tuple(
            task.id for task in tasks if value_id in task.read_values
        )
        origin = ir1_values.get(temp_origins.get(value_id, value_id))
        if origin is not None:
            result.append(
                IntraDieValue(
                    value_id,
                    origin.id,
                    origin.shape,
                    origin.dtype,
                    origin.logical_layout,
                    origin.sharding,
                    origin.alias_set,
                    producer_tasks,
                    consumer_tasks,
                )
            )
            continue
        related = next(
            task
            for task in tasks
            if value_id in task.read_values + task.write_values
        )
        result.append(
            IntraDieValue(
                value_id,
                f"synthetic:{value_id}",
                related.shape,
                related.dtype or DType.FP16,
                partial_template.logical_layout,
                partial_template.sharding,
                None,
                producer_tasks,
                consumer_tasks,
            )
        )
    return tuple(result)


def dependency_chain() -> tuple[
    IR1,
    FusionPlan,
    StandaloneCollectivePlan,
    IR2ProjectionResult,
]:
    """coarse -> fused -> coarse -> AllGather -> coarse, plus planned CONTROL."""

    base = valid_fusion_ir1()
    gemm, reduce_scatter = base.nodes
    value_in, value_partial, value_out = base.values[:3]
    weight = base.values[3]

    def rope_workload(shape: tuple[int, ...]) -> RopeQkWorkload:
        tokens, width = shape
        head_dim = width // 4
        return RopeQkWorkload(
            profile=base.profile,
            logical_input_shape=shape,
            rank_input_shape=shape,
            logical_output_shape=shape,
            rank_output_shape=shape,
            packed_layout=PackedQkvLayout.Q_K_V,
            num_heads=2,
            num_kv_heads=1,
            rank_num_heads=2,
            rank_num_kv_heads=1,
            head_dim=head_dim,
            rotary_dim=head_dim,
            rope_theta=10_000.0,
            max_position_embeddings=max(tokens, base.profile.context_max),
            dtype=DType.FP16,
        )

    pre = replace(
        gemm,
        id="p_pre",
        origin_node_id="pre",
        inputs=("p_v_pre_in",),
        outputs=(value_in.id,),
        kind=OpKind.ROPE,
        workload=rope_workload(value_in.shape),
        impl_ref="pre_impl",
    )
    middle = replace(
        gemm,
        id="p_middle",
        origin_node_id="middle",
        inputs=(value_out.id,),
        outputs=("p_v_ag_in",),
        kind=OpKind.ROPE,
        workload=rope_workload(value_out.shape),
        impl_ref="middle_impl",
    )
    ag_input = replace(
        value_out,
        id="p_v_ag_in",
        producer=middle.id,
        consumers=("p_ag",),
    )
    ag_output = replace(
        value_out,
        id="p_v_ag_out",
        producer="p_ag",
        consumers=("p_post",),
        logical_layout="MN",
        sharding=Sharding("mesh_tp", (None, None), ()),
    )
    all_gather = replace(
        reduce_scatter,
        id="p_ag",
        origin_node_id="ag",
        inputs=("p_v_ag_in",),
        outputs=("p_v_ag_out",),
        workload=replace(
            reduce_scatter.workload,
            collective=CollectiveKind.ALL_GATHER,
            reduce_op=None,
            reduction_mesh_axes=(),
            scatter_tensor_axis=None,
            gather_tensor_axis=0,
            rank_input_bytes=4096,
            rank_output_bytes=8192,
            input_layout=ag_input.logical_layout,
            output_layout=ag_output.logical_layout,
        ),
    )
    post = replace(
        gemm,
        id="p_post",
        origin_node_id="post",
        inputs=("p_v_ag_out",),
        outputs=("p_v_post_out",),
        kind=OpKind.ROPE,
        workload=rope_workload(value_out.shape),
        impl_ref="post_impl",
    )
    values = (
        replace(
            value_in,
            id="p_v_pre_in",
            producer=None,
            consumers=(pre.id,),
        ),
        replace(value_in, producer=pre.id),
        value_partial,
        replace(value_out, consumers=(middle.id,)),
        ag_input,
        ag_output,
        replace(
            value_out,
            id="p_v_post_out",
            producer=post.id,
            consumers=(),
        ),
        weight,
    )
    ir1_fields = base._semantic_key()
    ir1_fields.update(
        instances=(
            replace(
                base.instances[0],
                node_ids=(
                    pre.id,
                    gemm.id,
                    reduce_scatter.id,
                    middle.id,
                    all_gather.id,
                    post.id,
                ),
            ),
        ),
        nodes=(pre, gemm, reduce_scatter, middle, all_gather, post),
        values=values,
        edges=(
            GraphEdge("edge_00_pre_fusion", EdgeKind.DATA, pre.id, gemm.id, value_in.id),
            GraphEdge("edge_01_fused_internal", EdgeKind.DATA, gemm.id, reduce_scatter.id, value_partial.id),
            GraphEdge("edge_02_fusion_middle", EdgeKind.DATA, reduce_scatter.id, middle.id, value_out.id),
            GraphEdge("edge_03_middle_ag", EdgeKind.DATA, middle.id, all_gather.id, "p_v_ag_in"),
            # Source names the non-terminal GEMM member: CONTROL still means the
            # entire fused unit completes before the entire AG unit enters.
            GraphEdge("edge_04_fused_ag_control", EdgeKind.CONTROL, gemm.id, all_gather.id, None),
            GraphEdge("edge_05_ag_post", EdgeKind.DATA, all_gather.id, post.id, "p_v_ag_out"),
        ),
    )
    ir1 = IR1.create(producer_pass=base.producer_pass, **ir1_fields)

    fusion_template = valid_plan()
    fusion_fields = fusion_template._semantic_key()
    fusion_fields.update(
        source_ir1_id=ir1.id,
        profile_key=ir1.profile,
        physical_output_layout=fusion_template.logical_output_layout,
        output_permutation=tuple(
            replace(entry, logical_owner_rank=entry.physical_owner_rank)
            for entry in fusion_template.output_permutation
        ),
        inverse_permutation=tuple(
            replace(entry, logical_owner_rank=entry.physical_owner_rank)
            for entry in fusion_template.inverse_permutation
        ),
        consumer_layout_bindings=(
            ConsumerLayoutBinding(
                middle.id,
                value_out.id,
                fusion_template.logical_output_layout,
                False,
            ),
        ),
    )
    localized_programs = []
    for program in fusion_fields["rank_programs"]:
        actions = []
        for action in program.actions:
            if action.kind is not FusionActionKind.COMP:
                actions.append(action)
                continue
            assert action.compute is not None and action.compute.tile is not None
            local_inputs = tuple(
                f"{action.id}.operand.{index}"
                for index in range(len(action.compute.inputs))
            )
            actions.append(
                replace(
                    action,
                    reads=local_inputs,
                    compute=replace(
                        action.compute,
                        inputs=tuple(
                            replace(operand, value_id=value_id)
                            for operand, value_id in zip(
                                action.compute.inputs, local_inputs
                            )
                        ),
                        tile=replace(
                            action.compute.tile,
                            input_slices=tuple(
                                replace(binding, operand_id=value_id)
                                for binding, value_id in zip(
                                    action.compute.tile.input_slices,
                                    local_inputs,
                                )
                            ),
                        ),
                    ),
                )
            )
        localized_programs.append(replace(program, actions=tuple(actions)))
    fusion_fields["rank_programs"] = tuple(localized_programs)
    fusion_plan = FusionPlan.create(
        producer_pass=fusion_template.producer_pass,
        **fusion_fields,
    )

    standalone_template = valid_standalone_plan()
    standalone_chunks = tuple(
        replace(chunk, value_id="p_v_ag_out")
        for chunk in standalone_template.chunk_slices
    )
    standalone_programs = []
    for program in standalone_template.rank_programs:
        actions = []
        for source in program.actions:
            reads = source.reads
            writes = source.writes
            if source.kind is FusionActionKind.LOCAL_COPY:
                reads, writes = ("p_v_ag_in",), ("p_v_ag_out",)
            elif source.kind is FusionActionKind.SEND:
                reads = ("p_v_ag_out",)
            elif source.kind is FusionActionKind.RECV:
                writes = ("p_v_ag_out",)
            actions.append(
                replace(
                    source,
                    member_id=all_gather.id,
                    reads=reads,
                    writes=writes,
                )
            )
        standalone_programs.append(RankProgram(program.rank, tuple(actions)))
    standalone_fields = standalone_template._semantic_key()
    standalone_fields.update(
        source_ir1_id=ir1.id,
        op_id=all_gather.id,
        profile_key=ir1.profile,
        chunk_slices=standalone_chunks,
        rank_programs=tuple(standalone_programs),
    )
    standalone_plan = StandaloneCollectivePlan.create(
        producer_pass=standalone_template.producer_pass,
        **standalone_fields,
    )

    fused_projection = _exact_projection(ir1, fusion_plan, standalone=False)
    ag_projection = _exact_projection(ir1, standalone_plan, standalone=True)
    dags = []
    for fused_dag, ag_dag in zip(fused_projection.dags, ag_projection.dags):
        rank = next(
            placement.rank
            for placement in base.groups[0].placements
            if placement.die_id == fused_dag.die_id
        )
        pre_task = _ordinary_task(pre, rank, ())
        fused_tasks = tuple(
            replace(task, deps=task.deps + (pre_task.id,))
            if task.kind is SemanticTaskKind.COMP
            else task
            for task in fused_dag.tasks
        )
        reduce_task = next(
            task for task in fused_tasks if task.kind is SemanticTaskKind.REDUCE
        )
        middle_task = _ordinary_task(middle, rank, (reduce_task.id,))
        ag_region_id = f"region_ag_{rank}"
        ag_tasks = tuple(
            replace(
                task,
                region_id=ag_region_id,
                deps=task.deps + (middle_task.id, reduce_task.id),
            )
            if task.kind is SemanticTaskKind.LOCAL_COPY
            else replace(task, region_id=ag_region_id)
            for task in ag_dag.tasks
        )
        barrier = next(
            task for task in ag_tasks if task.kind is SemanticTaskKind.BARRIER
        )
        post_task = _ordinary_task(post, rank, (barrier.id,))
        tasks = (pre_task,) + fused_tasks + (middle_task,) + ag_tasks + (post_task,)
        regions = (
            IntraDieRegion(pre_task.region_id, None, None, RegionLowering.JSON_COARSE, (pre_task.id,)),
            replace(fused_dag.regions[0], task_ids=tuple(task.id for task in fused_tasks)),
            IntraDieRegion(middle_task.region_id, None, None, RegionLowering.JSON_COARSE, (middle_task.id,)),
            replace(ag_dag.regions[0], id=ag_region_id, task_ids=tuple(task.id for task in ag_tasks)),
            IntraDieRegion(post_task.region_id, None, None, RegionLowering.JSON_COARSE, (post_task.id,)),
        )
        dags.append(
            IntraDieDAG.create(
                producer_pass="dependency_chain_fixture",
                source_ir1_id=ir1.id,
                die_id=fused_dag.die_id,
                fusion_plan_ids=(fusion_plan.id,),
                standalone_collective_plan_ids=(standalone_plan.id,),
                ordinary_node_ids=(pre.id, middle.id, post.id),
                tasks=tasks,
                values=_local_values_for_tasks(ir1, tasks, fusion_plan),
                flows=fused_dag.flows + ag_dag.flows,
                regions=regions,
            )
        )
    return ir1, fusion_plan, standalone_plan, IR2ProjectionResult.create(
        producer_pass="dependency_chain_fixture",
        source_ir1_id=ir1.id,
        fusion_plan_ids=(fusion_plan.id,),
        standalone_collective_plan_ids=(standalone_plan.id,),
        dags=tuple(dags),
    )


def recreate_dag(dag: IntraDieDAG, **changes: object) -> IntraDieDAG:
    fields = dag._semantic_key()
    fields.update(changes)
    return IntraDieDAG.create(producer_pass=dag.producer_pass, **fields)


def recreate_projection(
    projection: IR2ProjectionResult, **changes: object
) -> IR2ProjectionResult:
    fields = projection._semantic_key()
    fields.update(changes)
    return IR2ProjectionResult.create(
        producer_pass=projection.producer_pass, **fields
    )


class IR2SchemaTest(unittest.TestCase):
    def test_swizzle_temporary_value_preserves_exact_origin(self) -> None:
        origin = SwizzleValueOrigin(
            value_ref="tmp.partial",
            use=SwizzleValueUse.WRITE,
            logical_source_ref=None,
            producer_action_ref="action.comp",
            local_member_ref=None,
        )
        value = SwizzleIntraDieValue(
            id="tmp.partial",
            plan_id="swizzle-plan",
            rank=0,
            origins=(origin,),
            producer_tasks=("task.comp",),
            consumer_tasks=(),
        )
        value.validate("value")
        with self.assertRaisesRegex(SchemaError, "must equal carrier id"):
            replace(value, id="forged").validate("value")

    def test_projection_plan_and_dag_tuples_are_canonical(self) -> None:
        graph = _partitioned_graph(tp=2)
        fusion_plans = tuple(
            NaiveInterDiePolicy().plan(graph, skeleton, graph.profile)
            for skeleton in graph.fused_op_skeletons
        )
        fused_members = {
            member_id
            for skeleton in graph.fused_op_skeletons
            for member_id in skeleton.member_node_ids
        }
        standalone_plans = tuple(
            DirectAllGatherPolicy().plan(graph, node, graph.profile)
            for node in graph.nodes
            if node.id not in fused_members
            and node.kind is OpKind.COLLECTIVE
            and getattr(node.workload, "collective", None)
            is CollectiveKind.ALL_GATHER
        )
        projection = NaiveProjectToIR2().run(
            graph, fusion_plans, standalone_plans, state_transfers=()
        )

        reversed_plans = recreate_projection(
            projection,
            fusion_plan_ids=tuple(reversed(projection.fusion_plan_ids)),
        )
        with self.assertRaisesRegex(SchemaError, "input plan tuple"):
            reversed_plans.validate_against(
                graph, fusion_plans, standalone_plans
            )

        reversed_standalone_plans = recreate_projection(
            projection,
            standalone_collective_plan_ids=tuple(
                reversed(projection.standalone_collective_plan_ids)
            ),
        )
        with self.assertRaisesRegex(SchemaError, "input plan tuple"):
            reversed_standalone_plans.validate_against(
                graph, fusion_plans, standalone_plans
            )

        reversed_dags = recreate_projection(
            projection, dags=tuple(reversed(projection.dags))
        )
        with self.assertRaisesRegex(SchemaError, "fabric die order"):
            reversed_dags.validate_against(
                graph, fusion_plans, standalone_plans
            )

        dag = projection.dags[0]
        isa_regions = tuple(
            region
            for region in dag.regions
            if region.lowering is RegionLowering.ISA_REGION
        )
        reversed_isa = iter(reversed(isa_regions))
        local_fusion_dag = recreate_dag(
            dag,
            fusion_plan_ids=tuple(reversed(dag.fusion_plan_ids)),
            regions=tuple(
                next(reversed_isa)
                if region.lowering is RegionLowering.ISA_REGION
                else region
                for region in dag.regions
            ),
        )
        local_fusion = recreate_projection(
            projection, dags=(local_fusion_dag,) + projection.dags[1:]
        )
        with self.assertRaisesRegex(SchemaError, "filtered input plan order"):
            local_fusion.validate_against(
                graph, fusion_plans, standalone_plans
            )

        strict_regions = tuple(
            region
            for region in dag.regions
            if region.lowering is RegionLowering.STRICT_ACTIONS
        )
        reversed_strict = iter(reversed(strict_regions))
        local_standalone_dag = recreate_dag(
            dag,
            standalone_collective_plan_ids=tuple(
                reversed(dag.standalone_collective_plan_ids)
            ),
            regions=tuple(
                next(reversed_strict)
                if region.lowering is RegionLowering.STRICT_ACTIONS
                else region
                for region in dag.regions
            ),
        )
        local_standalone = recreate_projection(
            projection,
            dags=(local_standalone_dag,) + projection.dags[1:],
        )
        with self.assertRaisesRegex(SchemaError, "filtered input plan order"):
            local_standalone.validate_against(
                graph, fusion_plans, standalone_plans
            )

        local_ordinary_dag = recreate_dag(
            dag, ordinary_node_ids=tuple(reversed(dag.ordinary_node_ids))
        )
        local_ordinary = recreate_projection(
            projection, dags=(local_ordinary_dag,) + projection.dags[1:]
        )
        with self.assertRaisesRegex(SchemaError, "filtered IR-1 node order"):
            local_ordinary.validate_against(
                graph, fusion_plans, standalone_plans
            )

    def test_projection_local_tuples_follow_canonical_producer_order(self) -> None:
        ir1, plan, projection = exact_projection()
        dag = projection.dags[0]

        comp_indices = tuple(
            index
            for index, task in enumerate(dag.tasks)
            if task.kind is SemanticTaskKind.COMP and not task.deps
        )
        self.assertGreaterEqual(len(comp_indices), 2)
        left, right = comp_indices[:2]
        swapped_tasks = list(dag.tasks)
        swapped_tasks[left], swapped_tasks[right] = (
            swapped_tasks[right],
            swapped_tasks[left],
        )
        swapped_tasks_tuple = tuple(swapped_tasks)
        swapped_task_dag = recreate_dag(
            dag,
            tasks=swapped_tasks_tuple,
            regions=tuple(
                replace(
                    region,
                    task_ids=tuple(
                        task.id
                        for task in swapped_tasks_tuple
                        if task.region_id == region.id
                    ),
                )
                for region in dag.regions
            ),
        )
        with self.assertRaisesRegex(SchemaError, "canonical unit/rank-program/action"):
            recreate_projection(
                projection,
                dags=(swapped_task_dag,) + projection.dags[1:],
            ).validate_against(ir1, (plan,), ())

        reversed_values_dag = recreate_dag(
            dag, values=tuple(reversed(dag.values))
        )
        with self.assertRaisesRegex(SchemaError, "first-use order"):
            recreate_projection(
                projection,
                dags=(reversed_values_dag,) + projection.dags[1:],
            ).validate_against(ir1, (plan,), ())

        reversed_flows_dag = recreate_dag(
            dag, flows=tuple(reversed(dag.flows))
        )
        with self.assertRaisesRegex(SchemaError, "canonical task order"):
            recreate_projection(
                projection,
                dags=(reversed_flows_dag,) + projection.dags[1:],
            ).validate_against(ir1, (plan,), ())

        graph = _partitioned_graph(tp=2)
        fusion_plans = tuple(
            NaiveInterDiePolicy().plan(graph, skeleton, graph.profile)
            for skeleton in graph.fused_op_skeletons
        )
        fused_members = {
            member_id
            for skeleton in graph.fused_op_skeletons
            for member_id in skeleton.member_node_ids
        }
        standalone_plans = tuple(
            DirectAllGatherPolicy().plan(graph, node, graph.profile)
            for node in graph.nodes
            if node.id not in fused_members
            and node.kind is OpKind.COLLECTIVE
            and getattr(node.workload, "collective", None)
            is CollectiveKind.ALL_GATHER
        )
        complete = NaiveProjectToIR2().run(
            graph, fusion_plans, standalone_plans, state_transfers=()
        )
        complete_dag = complete.dags[0]
        ordinary_region_indices = tuple(
            index
            for index, region in enumerate(complete_dag.regions)
            if region.lowering is RegionLowering.JSON_COARSE
        )
        self.assertGreaterEqual(len(ordinary_region_indices), 2)
        first, second = ordinary_region_indices[:2]
        swapped_regions = list(complete_dag.regions)
        swapped_regions[first], swapped_regions[second] = (
            swapped_regions[second],
            swapped_regions[first],
        )
        reversed_regions_dag = recreate_dag(
            complete_dag, regions=tuple(swapped_regions)
        )
        with self.assertRaisesRegex(SchemaError, "unit/ordinary-rank order"):
            recreate_projection(
                complete,
                dags=(reversed_regions_dag,) + complete.dags[1:],
            ).validate_against(graph, fusion_plans, standalone_plans)

    def test_dag_and_schedule_true_round_trip_and_cross_validation(self) -> None:
        dag = valid_dag()
        schedule = valid_schedule(dag)
        dag.validate()
        schedule.validate_against(dag, valid_ir1())
        for artifact_type, artifact in ((IntraDieDAG, dag), (IntraDieSchedule, schedule)):
            decoded = loads_dataclass(artifact_type, canonical_json(artifact))
            self.assertEqual(decoded, artifact)
            self.assertEqual(canonical_digest(decoded), canonical_digest(artifact))

    def test_dag_rejects_dangling_dependencies_and_cycles(self) -> None:
        dag = valid_dag()
        comp, send = dag.tasks
        for changed in (replace(send, deps=("missing",)), replace(comp, deps=(send.id,))):
            tasks = (changed, send) if changed.id == comp.id else (comp, changed)
            with self.assertRaises(SchemaError):
                replace(dag, tasks=tasks).validate()

    def test_origin_and_value_tables_are_cross_checked(self) -> None:
        dag = valid_dag()
        comp, send = dag.tasks
        with self.assertRaisesRegex(SchemaError, "origin"):
            replace(dag, tasks=(comp, replace(send, origin_ref=replace(send.origin_ref, plan_id="missing")))).validate()
        bad_value = replace(dag.values[1], consumer_tasks=())
        with self.assertRaisesRegex(SchemaError, "consumers"):
            replace(dag, values=(dag.values[0], bad_value)).validate()

    def test_region_lowering_and_origin_must_match_exactly(self) -> None:
        dag = valid_dag()
        coarse, isa = dag.regions
        cases = (
            (
                recreate_dag(
                    dag,
                    regions=(
                        replace(coarse, lowering=RegionLowering.ISA_REGION,
                                fusion_plan_id="fp_0"),
                        isa,
                    ),
                ),
                "ISA region tasks",
            ),
            (
                recreate_dag(
                    dag,
                    regions=(coarse, replace(isa, fusion_plan_id="other")),
                ),
                "dangling fusion plan|same plan",
            ),
            (
                recreate_dag(
                    dag,
                    regions=(coarse, replace(isa, task_ids=())),
                ),
                "task_ids",
            ),
        )
        for changed, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError, message
            ):
                changed.validate()

    def test_tile_source_value_drives_cross_coverage_data_dependency(self) -> None:
        ir1, fusion_plan, standalone_plan, projection = dependency_chain()
        for dag in projection.dags:
            fused_inputs = tuple(
                task
                for task in dag.tasks
                if isinstance(task.origin_ref, FusedNodeOrigin)
                and task.kind is SemanticTaskKind.COMP
            )
            pre = next(
                task
                for task in dag.tasks
                if isinstance(task.origin_ref, OrdinaryNodeOrigin)
                and task.origin_ref.op_id == "p_pre"
            )
            self.assertTrue(fused_inputs)
            self.assertTrue(all(pre.id in task.deps for task in fused_inputs))
            self.assertTrue(
                all("p_v_in" not in task.read_values for task in fused_inputs)
            )
        projection.validate_against(ir1, (fusion_plan,), (standalone_plan,))

    def test_planned_temps_have_exact_ir1_origins_and_global_domains(self) -> None:
        ir1, plan, projection = exact_projection()
        projection.validate_against(ir1, (plan,), ())
        ir1_values = {value.id: value for value in ir1.values}
        skeleton = next(
            item for item in ir1.fused_op_skeletons if item.id == plan.fused_op_id
        )
        gemm = next(
            node for node in ir1.nodes if node.id == skeleton.member_node_ids[0]
        )
        partial = ir1_values[gemm.outputs[0]]
        for dag in projection.dags:
            for value in dag.values:
                if value.id in ir1_values:
                    self.assertEqual(value.origin_value_id, value.id)
                    continue
                origin = ir1_values[value.origin_value_id]
                self.assertEqual(
                    (
                        value.shape,
                        value.dtype,
                        value.logical_layout,
                        value.sharding,
                    ),
                    (
                        origin.shape,
                        origin.dtype,
                        origin.logical_layout,
                        origin.sharding,
                    ),
                )
                writers = tuple(
                    task for task in dag.tasks if value.id in task.write_values
                )
                if writers:
                    self.assertEqual(value.origin_value_id, partial.id)
                    for task in writers:
                        assert task.chunk_id is not None
                        chunk = plan.chunk_slices[task.chunk_id]
                        assert task.tensor_slice is not None
                        self.assertEqual(
                            (task.tensor_slice.offset, task.tensor_slice.shape),
                            (chunk.offset, chunk.shape),
                        )

        dag = projection.dags[0]
        temp_index = next(
            index
            for index, value in enumerate(dag.values)
            if value.id not in ir1_values
        )
        changed_values = list(dag.values)
        changed_values[temp_index] = replace(
            changed_values[temp_index],
            origin_value_id="p_v_out",
        )
        changed_dag = recreate_dag(dag, values=tuple(changed_values))
        changed = recreate_projection(
            projection,
            dags=(changed_dag,) + projection.dags[1:],
        )
        with self.assertRaisesRegex(SchemaError, "planned temp binding"):
            changed.validate_against(ir1, (plan,), ())

    def test_semantic_dag_rejects_physical_binding_fields(self) -> None:
        raw = to_primitive(valid_dag())
        raw["tasks"][0]["core_id"] = 3
        with self.assertRaisesRegex(SchemaError, "core_id"):
            from_data(IntraDieDAG, raw)

    def test_schedule_rejects_final_runtime_ids(self) -> None:
        raw = to_primitive(valid_schedule())
        raw["runtime_bindings"][0]["tag_id"] = 9
        with self.assertRaisesRegex(SchemaError, "tag_id"):
            from_data(IntraDieSchedule, raw)
        for removed_field, value in (
            ("region", "sram"),
            ("offset", 0),
            ("size", 2048),
            ("alignment", 64),
            ("lifetime_end", 1),
            ("producer_event", None),
            ("consumer_events", []),
        ):
            raw = to_primitive(valid_schedule())
            raw["buffer_bindings"][0][removed_field] = value
            with self.subTest(field=removed_field), self.assertRaisesRegex(
                SchemaError, removed_field
            ):
                from_data(IntraDieSchedule, raw)

    def test_task_state_use_is_strict_and_round_trips(self) -> None:
        base = valid_schedule()
        use = TaskStateUse("t_comp", "hbm_binding_0", StateUseAccess.READ)
        schedule = recreate_schedule(base, task_state_uses=(use,))
        schedule.validate()
        self.assertNotEqual(schedule.id, base.id)
        self.assertEqual(
            loads_dataclass(IntraDieSchedule, canonical_json(schedule)),
            schedule,
        )
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(schedule, id=base.id).validate()


        for field in ("task_id", "hbm_binding_ref"):
            with self.subTest(field=field), self.assertRaises(SchemaError):
                replace(use, **{field: ""}).validate("state_use")
        with self.assertRaisesRegex(SchemaError, "StateUseAccess"):
            replace(use, access=BufferAccess.READ).validate("state_use")

    def test_task_state_use_serde_is_fail_closed(self) -> None:
        schedule = recreate_schedule(
            valid_schedule(),
            task_state_uses=(
                TaskStateUse(
                    "t_comp", "hbm_binding_0", StateUseAccess.READ
                ),
            ),
        )
        raw = to_primitive(schedule)
        del raw["task_state_uses"]
        with self.assertRaisesRegex(SchemaError, "task_state_uses"):
            from_data(IntraDieSchedule, raw)

        raw = to_primitive(schedule)
        raw["task_state_uses"][0]["state_ref"] = "forbidden_copy"
        with self.assertRaisesRegex(SchemaError, "state_ref"):
            from_data(IntraDieSchedule, raw)

        raw = to_primitive(schedule)
        raw["task_state_uses"][0]["access"] = "READ"
        with self.assertRaisesRegex(SchemaError, "unknown value"):
            from_data(IntraDieSchedule, raw)

    def test_task_state_uses_are_unique_and_canonical(self) -> None:
        first = TaskStateUse(
            "t_comp", "hbm_binding_0", StateUseAccess.READ
        )
        second = TaskStateUse(
            "t_send", "hbm_binding_1", StateUseAccess.WRITE
        )
        with self.assertRaisesRegex(SchemaError, "canonical"):
            recreate_schedule(
                valid_schedule(), task_state_uses=(second, first)
            ).validate()
        with self.assertRaisesRegex(SchemaError, "duplicate task state"):
            recreate_schedule(
                valid_schedule(),
                task_state_uses=(
                    first,
                    replace(first, hbm_binding_ref="hbm_binding_1"),
                ),
            ).validate()


    def test_schedule_cross_refs_and_dependency_order_are_strict(self) -> None:
        dag = valid_dag()
        schedule = valid_schedule(dag)
        for changed in (
            {"placements": schedule.placements[:1]},
            {"flow_routes": ()},
            {"core_orders": (CoreOrder(0, ("t_send", "t_comp")),)},
        ):
            with self.assertRaises(SchemaError):
                recreate_schedule(schedule, **changed).validate_against(dag, valid_ir1())

    def test_buffer_lifetime_and_use_order_are_strict(self) -> None:
        dag = valid_dag()
        schedule = valid_schedule(dag)
        bad_binding = replace(
            schedule.buffer_bindings[1], lifetime_end_exclusive=0
        )
        bad = recreate_schedule(schedule, buffer_bindings=(schedule.buffer_bindings[0], bad_binding))
        with self.assertRaisesRegex(SchemaError, "lifetime"):
            bad.validate_against(dag, valid_ir1())

    def test_task_buffer_uses_are_exact_operand_indexed_and_canonical(self) -> None:
        ir1 = valid_ir1()
        dag = valid_dag()
        schedule = valid_schedule(dag)
        comp_input, comp_output, send_source = schedule.task_buffer_uses
        cases = (
            (
                schedule.task_buffer_uses[:-1],
                "exactly cover required roles/operands",
            ),
            (
                (
                    comp_input,
                    replace(
                        comp_input,
                        binding_id=comp_output.binding_id,
                        tensor_slice=comp_output.tensor_slice,
                    ),
                    comp_output,
                    send_source,
                ),
                "exactly cover required roles/operands",
            ),
            (
                (
                    comp_input,
                    comp_output,
                    replace(
                        send_source,
                        role=BufferUseRole.RECV_DESTINATION,
                    ),
                ),
                "exactly cover required roles/operands",
            ),
            (
                (
                    comp_input,
                    comp_output,
                    replace(send_source, operand_index=1),
                ),
                "exactly cover required roles/operands",
            ),
            (
                (
                    replace(comp_input, binding_id=comp_output.binding_id),
                    comp_output,
                    send_source,
                ),
                "different value",
            ),
            (
                (
                    replace(comp_input, access=BufferAccess.WRITE),
                    comp_output,
                    send_source,
                ),
                "rank/access/value disagrees",
            ),
            (
                (comp_output, comp_input, send_source),
                "canonical task/role/operand order",
            ),
        )
        for uses, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError, message
            ):
                recreate_schedule(
                    schedule, task_buffer_uses=uses
                ).validate_against(dag, ir1)

        output = schedule.buffer_bindings[1]
        changed_output = replace(
            output,
            tensor_slice=TensorSlice("v_out", (0, 0), (8, 64)),
            size_bytes=1024,
        )
        with self.assertRaisesRegex(
            SchemaError,
            "contained in its root backing",
        ):
            recreate_schedule(
                schedule,
                buffer_bindings=(schedule.buffer_bindings[0], changed_output),
            ).validate_against(dag, ir1)

    def test_dense_root_view_addend_containment_and_contiguity(self) -> None:
        root = TensorSlice("v", (0, 0), (16, 64))
        self.assertEqual(
            dense_row_major_view_byte_addend(
                root, TensorSlice("v", (0, 0), (8, 64)), DType.FP16
            ),
            0,
        )
        self.assertEqual(
            dense_row_major_view_byte_addend(
                root, TensorSlice("v", (8, 0), (8, 64)), DType.FP16
            ),
            1024,
        )
        for view, message in (
            (TensorSlice("other", (0, 0), (8, 64)), "same value"),
            (TensorSlice("v", (16, 0), (1, 64)), "contained"),
            (TensorSlice("v", (0, 16), (16, 32)), "not contiguous"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError, message
            ):
                dense_row_major_view_byte_addend(root, view, DType.FP16)

    def test_schedule_rejects_noncontiguous_or_forged_operand_view(self) -> None:
        ir1 = valid_ir1()
        dag = valid_dag()
        schedule = valid_schedule(dag)
        first, *rest = schedule.task_buffer_uses
        for view, message in (
            (TensorSlice("v_in", (0, 32), (16, 32)), "not contiguous"),
            (TensorSlice("v_in", (16, 0), (1, 64)), "contained"),
            (TensorSlice("v_out", (0, 0), (16, 64)), "different value"),
        ):
            changed = recreate_schedule(
                schedule,
                task_buffer_uses=(replace(first, tensor_slice=view), *rest),
            )
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError, message
            ):
                changed.validate_against(dag, ir1)

    def test_named_region_initiators_are_required_by_buffer_role(self) -> None:
        base = valid_ir1()
        dag = valid_dag()
        schedule = valid_schedule(dag)
        profile = base.fabric.sram_profiles[0]
        region = profile.regions[0]
        for denied, message in (
            (MemoryInitiator.COMPUTE, "compute"),
            (MemoryInitiator.DTE, "dte"),
        ):
            restricted_region = replace(
                region,
                access=tuple(item for item in region.access if item is not denied),
            )
            restricted_profile = replace(
                profile, regions=(restricted_region,)
            )
            restricted_ir1 = recreate_ir1(
                base,
                fabric=replace(
                    base.fabric, sram_profiles=(restricted_profile,)
                ),
            )
            restricted_dag = recreate_dag(
                dag, source_ir1_id=restricted_ir1.id
            )
            with self.subTest(initiator=denied.value), self.assertRaisesRegex(
                SchemaError, f"does not permit {message}"
            ):
                recreate_schedule(
                    schedule, dag_id=restricted_dag.id
                ).validate_against(restricted_dag, restricted_ir1)

    def test_executable_dependencies_must_be_same_core(self) -> None:
        ir1 = valid_ir1()
        dag = valid_dag()
        schedule = valid_schedule(dag)
        moved = recreate_schedule(
            schedule,
            placements=(
                schedule.placements[0],
                replace(schedule.placements[1], core_id=1),
            ),
            core_orders=(
                CoreOrder(0, ("t_comp",)),
                CoreOrder(1, ("t_send",)),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "same core"):
            moved.validate_against(dag, ir1)

    def test_local_reduce_exact_staging_contract(self) -> None:
        ir1, dag, schedule = valid_reduce_case()
        schedule.validate_against(dag, ir1)
        input_zero, input_one, output = schedule.buffer_bindings
        use_zero, use_one, use_output = schedule.task_buffer_uses

        cases = (
            (
                {"task_buffer_uses": (use_zero, use_output)},
                "exactly cover required roles/operands",
            ),
            (
                {
                    "task_buffer_uses": (
                        replace(use_zero, contribution_rank=1),
                        use_one,
                        use_output,
                    )
                },
                "rank/access/value disagrees",
            ),
            (
                {
                    "buffer_bindings": (
                        input_zero,
                        replace(
                            input_one, region_offset_bytes=96, banks=(1,)
                        ),
                        output,
                    )
                },
                "rank-ordered tight-stride",
            ),
            (
                {
                    "buffer_bindings": (
                        input_zero,
                        input_one,
                        replace(
                            output, region_offset_bytes=32, banks=(0,)
                        ),
                    )
                },
                "must not overlap the source span",
            ),
            (
                {
                    "buffer_bindings": (
                        input_zero,
                        input_one,
                        replace(
                            output,
                            region_offset_bytes=65,
                            alignment_bytes=1,
                            banks=(1,),
                        ),
                    )
                },
                "must be 2-byte aligned",
            ),
        )
        for changes, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError, message
            ):
                recreate_schedule(schedule, **changes).validate_against(
                    dag, ir1
                )

    def test_local_reduce_rejects_cross_region_staging(self) -> None:
        base = valid_ir1()
        profile = base.fabric.sram_profiles[0]
        region = profile.regions[0]
        first_size = region.size_bytes // 2
        first = replace(region, size_bytes=first_size)
        second = replace(
            region,
            id="sram_aux",
            name="sram_aux",
            base_bytes=region.base_bytes + first_size,
            size_bytes=region.size_bytes - first_size,
        )
        split_profile = replace(profile, regions=(first, second))
        ir1 = recreate_ir1(
            base,
            fabric=replace(base.fabric, sram_profiles=(split_profile,)),
        )
        ir1, dag, schedule = valid_reduce_case(ir1)
        input_zero, input_one, output = schedule.buffer_bindings
        bank = (
            second.base_bytes // profile.bank_interleave_bytes
        ) % profile.bank_count
        cross_region = replace(
            input_one,
            region_ref=second.id,
            region_offset_bytes=0,
            banks=(bank,),
        )
        with self.assertRaisesRegex(SchemaError, "share one core and named region"):
            recreate_schedule(
                schedule,
                buffer_bindings=(input_zero, cross_region, output),
            ).validate_against(dag, ir1)

    def test_buffer_named_region_slice_size_alignment_and_banks_are_exact(self) -> None:
        ir1 = valid_ir1()
        dag = valid_dag()
        schedule = valid_schedule(dag)
        binding = schedule.buffer_bindings[0]
        bank_one = replace(
            binding,
            tensor_slice=TensorSlice("v_in", (0, 0), (1, 32)),
            region_offset_bytes=64,
            size_bytes=64,
            banks=(1,),
        )
        recreate_schedule(
            schedule,
            buffer_bindings=(bank_one, schedule.buffer_bindings[1]),
            task_buffer_uses=(
                replace(
                    schedule.task_buffer_uses[0],
                    tensor_slice=bank_one.tensor_slice,
                ),
                *schedule.task_buffer_uses[1:],
            ),
        ).validate_against(dag, ir1)
        invalid = (
            (
                replace(
                    binding,
                    tensor_slice=replace(binding.tensor_slice, offset=(1, 0)),
                ),
                "outside its value",
            ),
            (replace(binding, layout="arbitrary_tile"), "logical_layout"),
            (replace(binding, size_bytes=2047), "tight root tensor payload"),
            (replace(binding, region_ref="missing_region"), "unknown named SRAM region"),
            (
                replace(binding, region_offset_bytes=(1 << 20) - 1024),
                "exceeds named SRAM region",
            ),
            (replace(binding, alignment_bytes=3), "power of two"),
            (replace(binding, region_offset_bytes=32), "alignment_bytes"),
            (replace(binding, banks=(0,)), "absolute interleave span"),
        )
        for changed_binding, message in invalid:
            changed = recreate_schedule(
                schedule,
                buffer_bindings=(changed_binding, schedule.buffer_bindings[1]),
            )
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError, message
            ):
                changed.validate_against(dag, ir1)

        huge_value = replace(
            dag.values[0],
            id="v_huge",
            shape=(UINT64_MAX,),
            producer_tasks=(),
            consumer_tasks=(),
        )
        huge_dag = recreate_dag(dag, values=dag.values + (huge_value,))
        huge_binding = replace(
            binding,
            id="b_huge",
            value_id="v_huge",
            tensor_slice=TensorSlice("v_huge", (0,), (UINT64_MAX,)),
            size_bytes=UINT64_MAX,
            storage_id="storage_huge",
        )
        huge_schedule = recreate_schedule(
            schedule,
            dag_id=huge_dag.id,
            buffer_bindings=(huge_binding,),
            task_buffer_uses=(),
        )
        with self.assertRaisesRegex(SchemaError, "overflows uint64"):
            huge_schedule.validate_against(huge_dag, ir1)

    def test_one_root_per_core_value_and_live_overlap_are_rejected(self) -> None:
        ir1 = valid_ir1()
        dag = valid_dag()
        schedule = valid_schedule(dag)
        root = schedule.buffer_bindings[0]
        output = schedule.buffer_bindings[1]
        reused_after_root = replace(
            output,
            id="b_reused",
            region_offset_bytes=root.region_offset_bytes,
            storage_id="storage_reused",
            lifetime_start=1,
            lifetime_end_exclusive=2,
        )
        with self.assertRaisesRegex(SchemaError, "only one root backing"):
            recreate_schedule(
                schedule,
                buffer_bindings=schedule.buffer_bindings
                + (reused_after_root,),
            ).validate_against(dag, ir1)

        overlapping = replace(
            output,
            region_offset_bytes=root.region_offset_bytes,
        )
        invalid = recreate_schedule(
            schedule, buffer_bindings=(root, overlapping)
        )
        with self.assertRaisesRegex(SchemaError, "physical byte/lifetime overlap"):
            invalid.validate_against(dag, ir1)

    def test_alias_must_directly_name_one_canonical_root(self) -> None:
        ir1 = valid_ir1()
        dag = valid_dag()
        aliased_values = tuple(
            replace(value, alias_set="alias_shared") for value in dag.values
        )
        aliased_dag = recreate_dag(dag, values=aliased_values)
        schedule = valid_schedule(aliased_dag)
        root, output = schedule.buffer_bindings
        alias = replace(
            root,
            id="b_alias",
            value_id="v_out",
            tensor_slice=TensorSlice("v_out", (0, 0), (16, 64)),
            storage_id=root.storage_id,
            alias_of=root.id,
            ownership=BufferOwnership.ALIASED,
            lifetime_start=1,
            lifetime_end_exclusive=2,
            layout="MN",
        )
        valid = recreate_schedule(
            schedule,
            buffer_bindings=(root, output, alias),
        )
        valid.validate_against(aliased_dag, ir1)

        bad_span = replace(alias, region_offset_bytes=64)
        with self.assertRaisesRegex(SchemaError, "identical core/region/span/storage"):
            recreate_schedule(
                schedule, buffer_bindings=(root, output, bad_span)
            ).validate_against(aliased_dag, ir1)

        mismatched_values = (
            replace(aliased_values[0], alias_set="root_alias"),
            replace(aliased_values[1], alias_set="other_alias"),
        )
        mismatched_dag = recreate_dag(aliased_dag, values=mismatched_values)
        with self.assertRaisesRegex(SchemaError, "explicit alias_set"):
            recreate_schedule(
                schedule,
                dag_id=mismatched_dag.id,
                buffer_bindings=(root, output, alias),
            ).validate_against(mismatched_dag, ir1)

        chained = replace(alias, id="b_alias_child", alias_of=alias.id)
        with self.assertRaisesRegex(SchemaError, "canonical root"):
            recreate_schedule(
                schedule,
                buffer_bindings=(root, output, alias, chained),
            ).validate_against(aliased_dag, ir1)

        fake_reuse = replace(output, storage_id=root.storage_id)
        with self.assertRaisesRegex(SchemaError, "only be reused by aliases"):
            recreate_schedule(
                schedule, buffer_bindings=(root, fake_reuse)
            ).validate_against(aliased_dag, ir1)

        with self.assertRaisesRegex(SchemaError, "non-ALIASED"):
            replace(root, alias_of=output.id).validate("binding")

    def test_missing_enum_and_uint64_are_rejected(self) -> None:
        raw = to_primitive(valid_schedule())
        del raw["buffer_bindings"][0]["ownership"]
        with self.assertRaisesRegex(SchemaError, "ownership"):
            from_data(IntraDieSchedule, raw)
        raw = to_primitive(valid_schedule())
        raw["placements"][0]["core_id"] = -1
        with self.assertRaises(SchemaError):
            from_data(IntraDieSchedule, raw)

    def test_origin_actions_cross_ref_plan_and_ir1(self) -> None:
        ir1, plan, projection = exact_projection()
        dag = projection.dags[0]
        dag.validate_against(ir1, (plan,), ())
        task = dag.tasks[0]
        bad_task = replace(
            task, origin_ref=replace(task.origin_ref, action_id="missing")
        )
        bad = recreate_dag(dag, tasks=(bad_task,) + dag.tasks[1:])
        with self.assertRaisesRegex(SchemaError, "dangling action"):
            bad.validate_against(ir1, (plan,), ())

    def test_ordinary_projection_is_exact_for_every_execution_group_rank(self) -> None:
        ir1, projection = ordinary_projection()
        projection.validate_against(ir1, (), ())
        self.assertEqual(
            {
                task.origin_ref.rank
                for dag in projection.dags
                for task in dag.tasks
                if isinstance(task.origin_ref, OrdinaryNodeOrigin)
            },
            {0, 1},
        )
        decoded = loads_dataclass(
            IR2ProjectionResult, canonical_json(projection)
        )
        self.assertEqual(decoded, projection)

        removed = recreate_dag(
            projection.dags[1],
            ordinary_node_ids=(),
            tasks=(),
            values=(),
            regions=(),
        )
        missing = recreate_projection(
            projection, dags=(projection.dags[0], removed)
        )
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            missing.validate_against(ir1, (), ())

        wrong_task = replace(
            projection.dags[1].tasks[0],
            origin_ref=replace(
                projection.dags[1].tasks[0].origin_ref, rank=0
            ),
        )
        wrong_rank = recreate_dag(
            projection.dags[1], tasks=(wrong_task,)
        )
        with self.assertRaisesRegex(SchemaError, "wrong die"):
            wrong_rank.validate_against(ir1, (), ())

    def test_ordinary_compute_contract_is_self_contained_and_exact(self) -> None:
        ir1, projection = ordinary_projection()
        dag = projection.dags[0]
        task = dag.tasks[0]
        task.validate("task")
        self.assertEqual(task.compute.impl_ref, ir1.nodes[0].impl_ref)
        for changed_compute in (
            replace(task.compute, impl_ref="different_impl"),
            replace(
                task.compute,
                workload=replace(
                    task.compute.workload,
                    logical_input_shape=(
                        task.compute.workload.logical_input_shape[0] + 1,
                        task.compute.workload.logical_input_shape[1],
                    ),
                    logical_output_shape=(
                        task.compute.workload.logical_output_shape[0] + 1,
                        task.compute.workload.logical_output_shape[1],
                    ),
                    rank_input_shape=(
                        task.compute.workload.rank_input_shape[0] + 1,
                        task.compute.workload.rank_input_shape[1],
                    ),
                    rank_output_shape=(
                        task.compute.workload.rank_output_shape[0] + 1,
                        task.compute.workload.rank_output_shape[1],
                    ),
                ),
            ),
            replace(
                task.compute,
                math=replace(task.compute.math, accumulation_dtype=DType.FP16),
            ),
            replace(
                task.compute,
                effects=replace(task.compute.effects, alias_set="different_alias"),
            ),
        ):
            changed_task = replace(task, compute=changed_compute)
            changed_dag = recreate_dag(dag, tasks=(changed_task,))
            with self.assertRaisesRegex(SchemaError, "disagrees with PhysicalNode"):
                changed_dag.validate_against(ir1, (), ())
        with self.assertRaisesRegex(SchemaError, "COMP requires compute"):
            replace(task, compute=None).validate("task")

    def test_projection_rejects_collective_without_any_plan(self) -> None:
        ir1 = valid_ir1()
        dags = tuple(
            IntraDieDAG.create(
                producer_pass="empty_projection_fixture",
                source_ir1_id=ir1.id,
                die_id=die.id,
                fusion_plan_ids=(),
                standalone_collective_plan_ids=(),
                ordinary_node_ids=(),
                tasks=(),
                values=(),
                flows=(),
                regions=(),
            )
            for die in ir1.fabric.dies
        )
        projection = IR2ProjectionResult.create(
            producer_pass="empty_projection_fixture",
            source_ir1_id=ir1.id,
            fusion_plan_ids=(),
            standalone_collective_plan_ids=(),
            dags=dags,
        )
        with self.assertRaisesRegex(SchemaError, "collective nodes require"):
            projection.validate_against(ir1, (), ())

    def test_semantic_reduce_preserves_full_contract(self) -> None:
        _ir1, _plan, projection = exact_projection()
        task = next(
            task
            for dag in projection.dags
            for task in dag.tasks
            if task.kind is SemanticTaskKind.REDUCE
        )
        task.validate("task")
        decoded = loads_dataclass(SemanticTask, canonical_json(task))
        self.assertEqual(decoded, task)
        self.assertEqual(decoded.reduction.input_ranks, (0, 1))
        for bad in (
            replace(task, reduction=None),
            replace(task, dtype=DType.FP32),
            replace(valid_dag().tasks[0], reduction=task.reduction),
        ):
            with self.assertRaises(SchemaError):
                bad.validate("task")
        raw = to_primitive(task)
        del raw["reduction"]
        with self.assertRaisesRegex(SchemaError, "reduction"):
            from_data(SemanticTask, raw)

    def test_projection_cannot_change_plan_reduction_contract(self) -> None:
        ir1, plan, projection = exact_projection()
        dag = projection.dags[0]
        task_index = next(
            index
            for index, task in enumerate(dag.tasks)
            if task.kind is SemanticTaskKind.REDUCE
        )
        task = dag.tasks[task_index]
        changed_task = replace(
            task,
            reduction=replace(task.reduction, input_ranks=(1, 0)),
        )
        changed_dag = recreate_dag(
            dag,
            tasks=dag.tasks[:task_index]
            + (changed_task,)
            + dag.tasks[task_index + 1 :],
        )
        changed = recreate_projection(
            projection, dags=(changed_dag,) + projection.dags[1:]
        )
        with self.assertRaisesRegex(SchemaError, "fields disagree"):
            changed.validate_against(ir1, (plan,), ())

    def test_projection_result_round_trip_and_exact_action_coverage(self) -> None:
        ir1, plan, projection = exact_projection()
        projection.validate_against(ir1, (plan,), ())
        decoded = loads_dataclass(
            IR2ProjectionResult, canonical_json(projection)
        )
        self.assertEqual(decoded, projection)
        self.assertEqual(canonical_digest(decoded), canonical_digest(projection))

    def test_projection_preflight_rejects_unexecutable_nonidentity_output(self) -> None:
        ir1 = valid_fusion_ir1()
        template = valid_plan()
        cases = {
            "owner": {
                "output_permutation": (
                    replace(template.output_permutation[0], logical_owner_rank=1),
                    replace(template.output_permutation[1], logical_owner_rank=0),
                ),
                "inverse_permutation": (
                    replace(template.inverse_permutation[0], logical_owner_rank=1),
                    replace(template.inverse_permutation[1], logical_owner_rank=0),
                ),
            },
            "layout": {
                "physical_output_layout": "MN_swizzled_tp",
            },
        }
        for name, changes in cases.items():
            with self.subTest(name=name):
                fields = template._semantic_key()
                fields.update(
                    source_ir1_id=ir1.id,
                    profile_key=ir1.profile,
                    **changes,
                )
                plan = FusionPlan.create(
                    producer_pass=template.producer_pass,
                    **fields,
                )
                with self.assertRaisesRegex(
                    SchemaError, "identity|physical output layout"
                ):
                    plan.validate_against(ir1)

    def test_standalone_actions_have_exact_one_time_projection(self) -> None:
        ir1, plan, projection = exact_standalone_projection()
        projection.validate_against(ir1, (), (plan,))
        decoded = loads_dataclass(
            IR2ProjectionResult, canonical_json(projection)
        )
        self.assertEqual(decoded, projection)
        self.assertTrue(
            any(
                task.kind is SemanticTaskKind.LOCAL_COPY
                for dag in projection.dags
                for task in dag.tasks
            )
        )
        dag = projection.dags[0]
        local_index = next(
            index
            for index, task in enumerate(dag.tasks)
            if task.kind is SemanticTaskKind.LOCAL_COPY
        )
        changed_task = replace(dag.tasks[local_index], member_id="wrong_op")
        changed_dag = recreate_dag(
            dag,
            tasks=dag.tasks[:local_index]
            + (changed_task,)
            + dag.tasks[local_index + 1 :],
        )
        changed = recreate_projection(
            projection, dags=(changed_dag,) + projection.dags[1:]
        )
        with self.assertRaisesRegex(SchemaError, "fields disagree"):
            changed.validate_against(ir1, (), (plan,))

    def test_standalone_multi_writer_slices_are_canonical_disjoint_and_full(self) -> None:
        _ir1, _plan, projection = exact_standalone_projection()
        dag = projection.dags[0]
        output_index = next(
            index for index, value in enumerate(dag.values) if value.id == "p_v_out"
        )
        output = dag.values[output_index]
        self.assertEqual(len(output.producer_tasks), 2)

        noncanonical = replace(
            output, producer_tasks=tuple(reversed(output.producer_tasks))
        )
        with self.assertRaisesRegex(SchemaError, "canonically"):
            recreate_dag(
                dag,
                values=dag.values[:output_index]
                + (noncanonical,)
                + dag.values[output_index + 1 :],
            ).validate()

        writers = [
            task for task in dag.tasks if task.id in set(output.producer_tasks)
        ]
        local = next(
            task for task in writers if task.kind is SemanticTaskKind.LOCAL_COPY
        )
        remote = next(task for task in writers if task.kind is SemanticTaskKind.RECV)
        local_index = dag.tasks.index(local)
        for changed_slice, message in (
            (remote.tensor_slice, "overlap"),
            (
                TensorSlice(
                    local.tensor_slice.value_id,
                    local.tensor_slice.offset,
                    (8, local.tensor_slice.shape[1]),
                ),
                "fully cover",
            ),
        ):
            changed_local = replace(local, tensor_slice=changed_slice)
            changed_dag = recreate_dag(
                dag,
                tasks=dag.tasks[:local_index]
                + (changed_local,)
                + dag.tasks[local_index + 1 :],
            )
            with self.assertRaisesRegex(SchemaError, message):
                changed_dag.validate()

    def test_slice_consumer_must_depend_on_its_relevant_producer(self) -> None:
        _ir1, _plan, projection = exact_standalone_projection()
        dag = projection.dags[0]
        send_index = next(
            index
            for index, task in enumerate(dag.tasks)
            if task.kind is SemanticTaskKind.SEND
        )
        changed_send = replace(dag.tasks[send_index], deps=())
        changed = recreate_dag(
            dag,
            tasks=dag.tasks[:send_index]
            + (changed_send,)
            + dag.tasks[send_index + 1 :],
        )
        with self.assertRaisesRegex(SchemaError, "relevant slice producer"):
            changed.validate()

    def test_fused_per_die_writers_may_cover_only_owned_slices(self) -> None:
        _ir1, _plan, projection = exact_projection()
        dag = projection.dags[0]
        reduce = next(
            task for task in dag.tasks if task.kind is SemanticTaskKind.REDUCE
        )
        partial_reduce = replace(
            reduce,
            id="task_partial_owned_reduce",
            origin_ref=replace(reduce.origin_ref, action_id="partial_owned_reduce"),
            chunk_id=1,
            tensor_slice=TensorSlice("p_v_out", (16, 0), (8, 128)),
            bytes=2048,
            shape=(8, 128),
            sync=replace(
                reduce.sync, completion_event="event_partial_owned_reduce"
            ),
        )
        values = []
        for value in dag.values:
            producers = value.producer_tasks
            consumers = value.consumer_tasks
            if value.id == "p_v_out":
                producers = producers + (partial_reduce.id,)
            if value.id in partial_reduce.read_values:
                consumers = consumers + (partial_reduce.id,)
            values.append(
                replace(
                    value,
                    producer_tasks=producers,
                    consumer_tasks=consumers,
                )
            )
        changed = recreate_dag(
            dag,
            tasks=dag.tasks + (partial_reduce,),
            values=tuple(values),
            regions=(
                replace(
                    dag.regions[0],
                    task_ids=dag.regions[0].task_ids + (partial_reduce.id,),
                ),
            ),
        )
        changed.validate()

    def test_projection_rejects_missing_action_and_extra_direct_transit(self) -> None:
        ir1, plan, projection = exact_projection()
        dag = projection.dags[0]
        send_index = next(
            index
            for index, task in enumerate(dag.tasks)
            if task.kind is SemanticTaskKind.SEND
        )
        send = dag.tasks[send_index]
        values = tuple(
            replace(
                value,
                consumer_tasks=tuple(
                    task_id for task_id in value.consumer_tasks if task_id != send.id
                ),
            )
            for value in dag.values
        )
        missing_dag = recreate_dag(
            dag,
            tasks=dag.tasks[:send_index] + dag.tasks[send_index + 1 :],
            values=values,
            flows=tuple(flow for flow in dag.flows if flow.id != send.flow_id),
            regions=(
                replace(
                    dag.regions[0],
                    task_ids=tuple(
                        task_id
                        for task_id in dag.regions[0].task_ids
                        if task_id != send.id
                    ),
                ),
            ),
        )
        missing = recreate_projection(
            projection, dags=(missing_dag,) + projection.dags[1:]
        )
        with self.assertRaisesRegex(SchemaError, "exactly match"):
            missing.validate_against(ir1, (plan,), ())

        flow_index = next(
            index for index, flow in enumerate(dag.flows) if flow.id == send.flow_id
        )
        transit = replace(
            send,
            id="extra_transit",
            kind=SemanticTaskKind.TRANSIT,
            read_values=(),
            write_values=(),
            compute=None,
            reduction=None,
            sync=SyncContract("event_extra_transit", None, None),
            deps=(send.id,),
        )
        changed_flow = replace(
            dag.flows[flow_index],
            task_ids=dag.flows[flow_index].task_ids + (transit.id,),
        )
        transit_dag = recreate_dag(
            dag,
            tasks=dag.tasks + (transit,),
            flows=dag.flows[:flow_index]
            + (changed_flow,)
            + dag.flows[flow_index + 1 :],
            regions=(
                replace(
                    dag.regions[0],
                    task_ids=dag.regions[0].task_ids + (transit.id,),
                ),
            ),
        )
        extra = recreate_projection(
            projection, dags=(transit_dag,) + projection.dags[1:]
        )
        with self.assertRaisesRegex(SchemaError, "TRANSIT"):
            extra.validate_against(ir1, (plan,), ())

    def test_projection_rejects_noncanonical_flow_identity(self) -> None:
        ir1, plan, projection = exact_projection()
        source_flow_id = projection.dags[0].flows[0].id
        changed_dags = []
        for dag in projection.dags:
            changed_dags.append(
                recreate_dag(
                    dag,
                    tasks=tuple(
                        replace(task, flow_id="noncanonical_flow")
                        if task.flow_id == source_flow_id
                        else task
                        for task in dag.tasks
                    ),
                    flows=tuple(
                        replace(flow, id="noncanonical_flow")
                        if flow.id == source_flow_id
                        else flow
                        for flow in dag.flows
                    ),
                )
            )
        changed = recreate_projection(projection, dags=tuple(changed_dags))
        with self.assertRaisesRegex(SchemaError, "canonical flow replica"):
            changed.validate_against(ir1, (plan,), ())

    def test_projection_preserves_compute_sync_and_full_selected_route(self) -> None:
        ir1, plan, projection = exact_projection()
        dag = projection.dags[0]
        comp_index = next(
            index
            for index, task in enumerate(dag.tasks)
            if task.kind is SemanticTaskKind.COMP
        )
        changed_comp = replace(
            dag.tasks[comp_index],
            sync=SyncContract("changed_completion", None, None),
        )
        bad_dag = recreate_dag(
            dag,
            tasks=dag.tasks[:comp_index]
            + (changed_comp,)
            + dag.tasks[comp_index + 1 :],
        )
        with self.assertRaisesRegex(SchemaError, "fields disagree"):
            bad_dag.validate_against(ir1, (plan,), ())

        flow = dag.flows[0]
        changed_flow = replace(flow, die_path=(flow.source_die, 2, flow.destination_die))
        route_dag = recreate_dag(
            dag, flows=(changed_flow,) + dag.flows[1:]
        )
        with self.assertRaisesRegex(SchemaError, "transport identity"):
            route_dag.validate_against(ir1, (plan,), ())

    def test_cross_coverage_chain_has_exact_data_and_control_dependencies(self) -> None:
        ir1, fusion_plan, standalone_plan, projection = dependency_chain()
        projection.validate_against(ir1, (fusion_plan,), (standalone_plan,))
        for dag in projection.dags:
            task_by_member = {
                task.member_id: task
                for task in dag.tasks
                if task.member_id in {"p_pre", "p_middle", "p_post"}
            }
            pre = task_by_member["p_pre"]
            middle = task_by_member["p_middle"]
            post = task_by_member["p_post"]
            fused_comps = tuple(
                task
                for task in dag.tasks
                if isinstance(task.origin_ref, FusedNodeOrigin)
                and task.kind is SemanticTaskKind.COMP
            )
            reduce = next(
                task
                for task in dag.tasks
                if isinstance(task.origin_ref, FusedNodeOrigin)
                and task.kind is SemanticTaskKind.REDUCE
            )
            local = next(
                task
                for task in dag.tasks
                if isinstance(task.origin_ref, StandaloneNodeOrigin)
                and task.kind is SemanticTaskKind.LOCAL_COPY
            )
            barrier = next(
                task
                for task in dag.tasks
                if isinstance(task.origin_ref, StandaloneNodeOrigin)
                and task.kind is SemanticTaskKind.BARRIER
            )
            self.assertTrue(all(task.deps[-1:] == (pre.id,) for task in fused_comps))
            self.assertEqual(middle.deps, (reduce.id,))
            self.assertEqual(local.deps, (middle.id, reduce.id))
            self.assertEqual(post.deps, (barrier.id,))

    def test_cross_coverage_boundaries_reject_missing_or_extra_dependencies(self) -> None:
        ir1, fusion_plan, standalone_plan, projection = dependency_chain()
        dag = projection.dags[0]
        pre = next(task for task in dag.tasks if task.member_id == "p_pre")
        middle = next(task for task in dag.tasks if task.member_id == "p_middle")
        post = next(task for task in dag.tasks if task.member_id == "p_post")
        fused_comp = next(
            task
            for task in dag.tasks
            if isinstance(task.origin_ref, FusedNodeOrigin)
            and task.kind is SemanticTaskKind.COMP
        )
        local = next(
            task
            for task in dag.tasks
            if isinstance(task.origin_ref, StandaloneNodeOrigin)
            and task.kind is SemanticTaskKind.LOCAL_COPY
        )

        mutations = {
            "ordinary_to_fusion_missing": replace(
                fused_comp, deps=tuple(dep for dep in fused_comp.deps if dep != pre.id)
            ),
            "fusion_to_ordinary_extra": replace(
                middle, deps=middle.deps + (pre.id,)
            ),
            "ordinary_to_ag_missing": replace(
                local, deps=tuple(dep for dep in local.deps if dep != middle.id)
            ),
            "ag_to_ordinary_extra": replace(
                post, deps=post.deps + (pre.id,)
            ),
            # The data chain already makes the reduce transitively reachable.
            # Exact direct dependency validation is what detects a missing
            # planned-unit CONTROL edge here.
            "planned_control_missing": replace(
                local,
                deps=tuple(
                    dep
                    for dep in local.deps
                    if not dep.endswith("reduce0")
                ),
            ),
        }
        for name, changed_task in mutations.items():
            with self.subTest(name=name):
                task_index = dag.tasks.index(
                    next(task for task in dag.tasks if task.id == changed_task.id)
                )
                changed_dag = recreate_dag(
                    dag,
                    tasks=dag.tasks[:task_index]
                    + (changed_task,)
                    + dag.tasks[task_index + 1 :],
                )
                with self.assertRaises(SchemaError):
                    changed_dag.validate_against(
                        ir1, (fusion_plan,), (standalone_plan,)
                    )

    def test_cross_coverage_dependency_rejects_wrong_rank_on_local_die(self) -> None:
        ir1, fusion_plan, standalone_plan, projection = dependency_chain()
        dag = projection.dags[0]
        middle_index = next(
            index for index, task in enumerate(dag.tasks) if task.member_id == "p_middle"
        )
        middle = dag.tasks[middle_index]
        wrong = replace(
            middle,
            origin_ref=replace(middle.origin_ref, rank=1),
        )
        changed = recreate_dag(
            dag,
            tasks=dag.tasks[:middle_index] + (wrong,) + dag.tasks[middle_index + 1 :],
        )
        with self.assertRaisesRegex(SchemaError, "wrong die"):
            changed.validate_against(ir1, (fusion_plan,), (standalone_plan,))


if __name__ == "__main__":
    unittest.main()
