"""Deterministic split-K/reduce/double-buffer graph refinement."""

from __future__ import annotations

from dataclasses import replace
import math

from ..errors import SchemaError
from ..schema.action import (
    ComputeContract,
    ComputeOperand,
    ComputeOperandSlice,
    ComputeTileBinding,
    ReductionContract,
    SyncContract,
)
from ..schema.common import DType, RoundingMode, Sharding
from ..schema.intra_die_refine import (
    IntraDieOptimizationOptions,
    SplitKRefineOptions,
)
from ..schema.ir0 import GemmWorkload, OpKind, ReduceOp
from ..schema.intra_die_v2_search import IntraDieV2CandidateKind
from ..schema.ir1 import IR1, MemoryInitiator
from ..schema.ir2 import (
    IR2ProjectionResult,
    IntraDieDAG,
    IntraDieRegion,
    IntraDieValue,
    OrdinaryNodeOrigin,
    RegionLowering,
    SemanticTask,
    SemanticTaskKind,
    StateStagingValue,
    TensorSlice,
    DmaContract,
    dense_row_major_view_byte_addend,
)
from ..schema.split_k_refine import (
    DoubleBufferVersion,
    SplitKInputHandoff,
    split_k_accumulator_value_id,
    SplitKLocalHandoff,
    SplitKRefinedProjection,
    SplitKTaskRewrite,
    split_k_compute_event_id,
    split_k_direct_dma_region_id,
    split_k_direct_dma_task_id,
    split_k_input_flow_id,
    split_k_input_recv_task_id,
    split_k_input_send_task_id,
    split_k_input_wait_task_id,
    split_k_local_flow_id,
    split_k_local_recv_task_id,
    split_k_local_send_task_id,
    split_k_local_wait_task_id,
    split_k_part_task_id,
    split_k_pack_row_task_id,
    split_k_pack_value_id,
    split_k_partial_value_id,
    split_k_ready_event_id,
    split_k_reduce_task_id,
    split_k_reduce_step_task_id,
    split_k_stage_task_id,
    split_k_staged_value_id,
)


def _fail(message: str, path: str) -> None:
    raise SchemaError(message, path=path)


def _element_bytes(dtype: DType, path: str) -> int:
    result = {DType.FP16: 2, DType.FP32: 4}.get(dtype)
    if result is None:
        _fail("split-K supports FP16/FP32 GEMM only", path)
    return result


def _source_origin(
    value: IntraDieValue | StateStagingValue, ir1: IR1
) -> tuple[str, Sharding]:
    if type(value) is IntraDieValue:
        return value.origin_value_id, value.sharding
    manifest = ir1.persistent_state_manifest
    if manifest is None:
        _fail("state-backed split-K input requires a persistent-state manifest", "ir1")
    declaration = next(
        (item for item in manifest.declarations if item.id == value.state_ref),
        None,
    )
    tensor_ref = declaration.identity.tensor_ref if declaration is not None else None
    origin = next((item for item in ir1.values if item.id == tensor_ref), None)
    if origin is None:
        _fail("state-backed split-K input must resolve to one IR-1 tensor", "ir1.values")
    return origin.id, origin.sharding


def _eligible(task: SemanticTask, dag: IntraDieDAG) -> bool:
    output = next(
        (value for value in dag.values if value.id in task.write_values),
        None,
    )
    return (
        task.kind is SemanticTaskKind.COMP
        and isinstance(task.origin_ref, OrdinaryNodeOrigin)
        and task.compute is not None
        and type(task.compute.workload) is GemmWorkload
        and len(task.read_values) == 2
        and len(task.write_values) == 1
        and len(task.compute.inputs) == 2
        and len(task.compute.outputs) == 1
        and output is not None
        and not output.consumer_tasks
    )


def _split_compute(
    source: SemanticTask,
    *,
    part_index: int,
    parts: int,
    input_ids: tuple[str, str],
    input_slices: tuple[TensorSlice, TensorSlice],
    output_id: str,
    output_slice: TensorSlice,
    deps: tuple[str, ...],
) -> SemanticTask:
    assert source.compute is not None
    workload = source.compute.workload
    assert type(workload) is GemmWorkload
    logical_m, logical_n, logical_k = workload.logical_shape
    rank_m, rank_n, rank_k = workload.rank_shape
    if logical_k % parts or rank_k % parts:
        _fail(
            "logical and rank-local K must be divisible by split_k_parts",
            f"tasks[{source.id}].compute.workload",
        )
    logical_k_part = logical_k // parts
    rank_k_part = rank_k // parts
    part_workload = replace(
        workload,
        logical_shape=(logical_m, logical_n, logical_k_part),
        rank_shape=(rank_m, rank_n, rank_k_part),
    )
    compute = ComputeContract(
        op_kind=OpKind.GEMM,
        workload=part_workload,
        math=source.compute.math,
        effects=source.compute.effects,
        impl_ref=source.compute.impl_ref,
        inputs=(
            ComputeOperand(input_ids[0], "lhs"),
            ComputeOperand(input_ids[1], "rhs"),
        ),
        outputs=(ComputeOperand(output_id, "partial"),),
        tile=ComputeTileBinding(
            origin_workload=workload,
            input_slices=tuple(
                ComputeOperandSlice(
                    input_ids[index],
                    source.read_values[index],
                    tensor_slice.offset,
                    tensor_slice.shape,
                )
                for index, tensor_slice in enumerate(input_slices)
            ),
            output_slices=(
                ComputeOperandSlice(
                    output_id,
                    source.write_values[0],
                    output_slice.offset,
                    output_slice.shape,
                ),
            ),
        ),
    )
    return SemanticTask(
        id=split_k_part_task_id(source.id, part_index),
        kind=SemanticTaskKind.COMP,
        origin_ref=source.origin_ref,
        region_id=source.region_id,
        op_kind=OpKind.GEMM,
        member_id=source.member_id,
        flow_id=None,
        chunk_id=part_index,
        collective_step=None,
        source_rank=None,
        destination_rank=None,
        tensor_slice=output_slice,
        bytes=0,
        dtype=workload.dtype,
        shape=output_slice.shape,
        read_values=input_ids,
        write_values=(output_id,),
        compute=compute,
        reduction=None,
        sync=SyncContract(
            split_k_compute_event_id(source.id, part_index), None, None
        ),
        deps=deps,
    )


def _stage_task(
    source: SemanticTask,
    source_value: IntraDieValue | StateStagingValue,
    *,
    part_index: int,
    operand_index: int,
    staged_value_id: str,
    tensor_slice: TensorSlice,
    compute_group_count: int,
    deps: tuple[str, ...],
) -> SemanticTask:
    payload_bytes = math.prod(tensor_slice.shape) * _element_bytes(
        source_value.dtype, f"tasks[{source.id}]"
    )
    return SemanticTask(
        id=split_k_stage_task_id(source.id, part_index, operand_index),
        kind=SemanticTaskKind.LOCAL_COPY,
        origin_ref=source.origin_ref,
        region_id=source.region_id,
        op_kind=OpKind.P2P,
        member_id=source.member_id,
        flow_id=None,
        chunk_id=part_index,
        collective_step=None,
        source_rank=None,
        destination_rank=None,
        tensor_slice=tensor_slice,
        bytes=payload_bytes,
        dtype=source_value.dtype,
        shape=tensor_slice.shape,
        read_values=(source_value.id,),
        write_values=(staged_value_id,),
        compute=None,
        reduction=None,
        sync=SyncContract(
            split_k_ready_event_id(
                source.id, part_index, operand_index, compute_group_count
            ),
            None,
            None,
        ),
        deps=deps,
    )


def _pack_row_task(
    source: SemanticTask,
    source_value: IntraDieValue,
    *,
    part_index: int,
    operand_index: int,
    row_index: int,
    packed_value_id: str,
    tensor_slice: TensorSlice,
    deps: tuple[str, ...],
) -> SemanticTask:
    payload_bytes = math.prod(tensor_slice.shape) * _element_bytes(
        source_value.dtype, f"tasks[{source.id}]"
    )
    return SemanticTask(
        id=split_k_pack_row_task_id(
            source.id, part_index, operand_index, row_index
        ),
        kind=SemanticTaskKind.LOCAL_COPY,
        origin_ref=source.origin_ref, region_id=source.region_id,
        op_kind=OpKind.P2P, member_id=source.member_id, flow_id=None,
        chunk_id=part_index, collective_step=row_index, source_rank=None,
        destination_rank=None, tensor_slice=tensor_slice, bytes=payload_bytes,
        dtype=source_value.dtype, shape=tensor_slice.shape,
        read_values=(source_value.id,), write_values=(packed_value_id,),
        compute=None, reduction=None,
        sync=SyncContract(
            f"event.{source.id}.split_k.part.{part_index}.input."
            f"{operand_index}.pack.row.{row_index}.done",
            None, None,
        ),
        deps=deps,
    )


def _reduce_task(
    source: SemanticTask,
    *,
    task_id: str,
    input_value_ids: tuple[str, str],
    output_value_id: str,
    output_slice: TensorSlice,
    deps: tuple[str, ...],
    step_index: int,
) -> SemanticTask:
    assert source.compute is not None
    workload = source.compute.workload
    assert type(workload) is GemmWorkload
    reduction = ReductionContract(
        reduce_op=ReduceOp.SUM,
        input_dtype=workload.dtype,
        accumulation_dtype=source.compute.math.accumulation_dtype,
        output_dtype=workload.dtype,
        rounding=RoundingMode.RNE,
        input_ranks=(0, 1),
    )
    reduction.validate(f"tasks[{source.id}].split_k.reduce.step[{step_index}]")
    payload_bytes = math.prod(output_slice.shape) * _element_bytes(
        workload.dtype, f"tasks[{source.id}]"
    )
    is_commit = task_id == split_k_reduce_task_id(source.id)
    return SemanticTask(
        id=task_id, kind=SemanticTaskKind.REDUCE, origin_ref=source.origin_ref,
        region_id=source.region_id, op_kind=OpKind.GEMM, member_id=source.member_id,
        flow_id=None, chunk_id=step_index, collective_step=step_index,
        source_rank=None, destination_rank=None, tensor_slice=output_slice,
        bytes=payload_bytes, dtype=workload.dtype, shape=output_slice.shape,
        read_values=input_value_ids, write_values=(output_value_id,),
        compute=None, reduction=reduction,
        sync=(
            source.sync if is_commit and source.sync is not None
            else SyncContract(f"event.{task_id}.done", None, None)
        ),
        deps=deps,
    )


def _direct_dma_task(
    original: SemanticTask, source: SemanticTask,
    *, part_index: int, operand_index: int,
    destination_task_id: str, tensor_slice: TensorSlice, dtype: DType,
) -> tuple[SemanticTask, IntraDieRegion] | None:
    if (
        original.kind is not SemanticTaskKind.DMA_IN
        or original.dma is None
        or original.tensor_slice is None
    ):
        return None
    try:
        byte_addend = dense_row_major_view_byte_addend(
            original.tensor_slice, tensor_slice, dtype,
            path=f"tasks[{source.id}].direct_dma",
        )
    except SchemaError:
        return None
    task_id = split_k_direct_dma_task_id(
        original.id, source.id, part_index, operand_index
    )
    region_id = split_k_direct_dma_region_id(task_id)
    payload_bytes = math.prod(tensor_slice.shape) * _element_bytes(
        dtype, f"tasks[{source.id}]"
    )
    task = replace(
        original, id=task_id, region_id=region_id,
        tensor_slice=tensor_slice, bytes=payload_bytes,
        shape=tensor_slice.shape, deps=original.deps,
        dma=replace(
            original.dma,
            state_offset_bytes=original.dma.state_offset_bytes + byte_addend,
            access_task_refs=(destination_task_id,),
        ),
    )
    task.validate(f"tasks[{task_id}]")
    region = IntraDieRegion(
        region_id, None, None, RegionLowering.STRICT_STATE_IO, (task_id,)
    )
    return task, region


def _local_handoff(
    source: SemanticTask,
    *,
    part_index: int,
    partial_value_id: str,
    output_slice: TensorSlice,
    payload_bytes: int,
    reduce_task_id: str,
    source_compute_group: int,
    source_dependency_task_id: str,
    destination_compute_group: int = 0,
) -> tuple[tuple[SemanticTask, SemanticTask, SemanticTask], SplitKLocalHandoff, IntraDieRegion]:
    assert source.compute is not None and type(source.compute.workload) is GemmWorkload
    payload_dtype = source.compute.workload.dtype
    flow_id = split_k_local_flow_id(source.id, part_index)
    send_id = split_k_local_send_task_id(source.id, part_index)
    recv_id = split_k_local_recv_task_id(source.id, part_index)
    wait_id = split_k_local_wait_task_id(source.id, part_index)
    region_id = f"{source.id}.split_k.part.{part_index}.local_transport_region"
    tensor_slice = TensorSlice(
        partial_value_id, output_slice.offset, output_slice.shape
    )
    common = dict(
        origin_ref=source.origin_ref,
        region_id=region_id,
        op_kind=None,
        member_id=None,
        flow_id=flow_id,
        chunk_id=part_index,
        collective_step=None,
        source_rank=None,
        destination_rank=None,
        compute=None,
        reduction=None,
        sync=None,
    )
    send = SemanticTask(
        id=send_id, kind=SemanticTaskKind.LOCAL_SEND, tensor_slice=tensor_slice,
        bytes=payload_bytes, dtype=payload_dtype, shape=tensor_slice.shape,
        read_values=(), write_values=(), deps=(source_dependency_task_id,),
        **common,
    )
    recv = SemanticTask(
        id=recv_id, kind=SemanticTaskKind.LOCAL_RECV, tensor_slice=tensor_slice,
        bytes=payload_bytes, dtype=payload_dtype, shape=tensor_slice.shape,
        read_values=(), write_values=(), deps=(send_id,), **common,
    )
    wait = SemanticTask(
        id=wait_id, kind=SemanticTaskKind.LOCAL_WAIT, tensor_slice=None,
        bytes=0, dtype=None, shape=(), read_values=(), write_values=(),
        deps=(recv_id,), **common,
    )
    handoff = SplitKLocalHandoff(
        part_index=part_index,
        part_task_id=source_dependency_task_id,
        partial_value_id=partial_value_id,
        reduce_task_id=reduce_task_id,
        flow_id=flow_id, send_task_id=send_id, recv_task_id=recv_id,
        wait_task_id=wait_id, source_compute_group=source_compute_group,
        destination_compute_group=destination_compute_group,
    )
    region = IntraDieRegion(
        region_id, None, None, RegionLowering.JSON_COARSE,
        (send_id, recv_id, wait_id),
    )
    return (send, recv, wait), handoff, region


def _tree_reduction(
    source: SemanticTask, *, partial_ids: tuple[str, ...],
    part_task_ids: tuple[str, ...], output_value_id: str,
    output_slice: TensorSlice, compute_group_count: int, payload_bytes: int,
) -> tuple[
    tuple[SemanticTask, ...], tuple[SplitKLocalHandoff, ...],
    tuple[IntraDieRegion, ...], tuple[str, ...], tuple[str, ...],
]:
    parts = len(partial_ids)
    if compute_group_count != parts:
        _fail(
            "tree reduce requires one physical compute group per part",
            f"tasks[{source.id}].split_k.tree",
        )
    active = {
        index: (partial_ids[index], part_task_ids[index], index)
        for index in range(parts)
    }
    generated: list[SemanticTask] = []
    handoffs: list[SplitKLocalHandoff] = []
    regions: list[IntraDieRegion] = []
    reduction_task_ids: list[str] = []
    accumulator_ids: list[str] = []
    step_index = 1
    stride = 1
    while stride < parts:
        for parent in range(0, parts, stride * 2):
            child = parent + stride
            if child >= parts:
                continue
            parent_value, parent_task, parent_group = active[parent]
            child_value, child_task, child_group = active[child]
            task_id = split_k_reduce_step_task_id(
                source.id, step_index, parts
            )
            is_commit = step_index == parts - 1
            result_value = (
                output_value_id if is_commit
                else split_k_accumulator_value_id(
                    source.id, output_value_id, step_index
                )
            )
            chain, handoff, region = _local_handoff(
                source, part_index=step_index,
                partial_value_id=child_value, output_slice=output_slice,
                payload_bytes=payload_bytes, reduce_task_id=task_id,
                source_compute_group=child_group,
                source_dependency_task_id=child_task,
                destination_compute_group=parent_group,
            )
            reduction = _reduce_task(
                source, task_id=task_id,
                input_value_ids=(parent_value, child_value),
                output_value_id=result_value, output_slice=output_slice,
                deps=(parent_task, handoff.wait_task_id),
                step_index=step_index,
            )
            generated.extend((*chain, reduction))
            handoffs.append(handoff)
            regions.append(region)
            reduction_task_ids.append(task_id)
            if not is_commit:
                accumulator_ids.append(result_value)
            active[parent] = (result_value, task_id, parent_group)
            step_index += 1
        stride *= 2
    if step_index != parts:
        _fail("tree reduce did not close every part", f"tasks[{source.id}]")
    return (
        tuple(generated), tuple(handoffs), tuple(regions),
        tuple(reduction_task_ids), tuple(accumulator_ids),
    )


def _input_handoff(
    source: SemanticTask,
    *,
    part_index: int,
    operand_index: int,
    producer_task_id: str,
    send_dependency_task_id: str,
    destination_task_id: str,
    destination_value_id: str,
    tensor_slice: TensorSlice,
    dtype: DType,
    destination_compute_group: int,
) -> tuple[tuple[SemanticTask, SemanticTask, SemanticTask], SplitKInputHandoff, IntraDieRegion]:
    flow_id = split_k_input_flow_id(source.id, part_index, operand_index)
    send_id = split_k_input_send_task_id(source.id, part_index, operand_index)
    recv_id = split_k_input_recv_task_id(source.id, part_index, operand_index)
    wait_id = split_k_input_wait_task_id(source.id, part_index, operand_index)
    region_id = f"{source.id}.split_k.part.{part_index}.input.{operand_index}.local_transport_region"
    payload_bytes = math.prod(tensor_slice.shape) * _element_bytes(
        dtype, f"tasks[{source.id}]"
    )
    common = dict(
        origin_ref=source.origin_ref, region_id=region_id, op_kind=None,
        member_id=None, flow_id=flow_id, chunk_id=part_index,
        collective_step=None, source_rank=None, destination_rank=None,
        compute=None, reduction=None, sync=None,
    )
    send = SemanticTask(
        id=send_id, kind=SemanticTaskKind.LOCAL_SEND, tensor_slice=tensor_slice,
        bytes=payload_bytes, dtype=dtype, shape=tensor_slice.shape,
        read_values=(), write_values=(), deps=(send_dependency_task_id,), **common,
    )
    recv = SemanticTask(
        id=recv_id, kind=SemanticTaskKind.LOCAL_RECV, tensor_slice=TensorSlice(
            destination_value_id, tensor_slice.offset, tensor_slice.shape
        ),
        bytes=payload_bytes, dtype=dtype, shape=tensor_slice.shape,
        read_values=(), write_values=(), deps=(send_id,), **common,
    )
    wait = SemanticTask(
        id=wait_id, kind=SemanticTaskKind.LOCAL_WAIT, tensor_slice=None,
        bytes=0, dtype=None, shape=(), read_values=(), write_values=(),
        deps=(recv_id,), **common,
    )
    carrier = SplitKInputHandoff(
        part_index=part_index, operand_index=operand_index,
        source_task_id=producer_task_id,
        send_dependency_task_id=send_dependency_task_id,
        destination_task_id=destination_task_id,
        value_id=tensor_slice.value_id, destination_value_id=destination_value_id,
        flow_id=flow_id,
        send_task_id=send_id, recv_task_id=recv_id, wait_task_id=wait_id,
        source_compute_group=0,
        destination_compute_group=destination_compute_group,
    )
    region = IntraDieRegion(
        region_id, None, None, RegionLowering.JSON_COARSE,
        (send_id, recv_id, wait_id),
    )
    return (send, recv, wait), carrier, region


def _compute_group_count(
    dag: IntraDieDAG, options: SplitKRefineOptions, ir1: IR1
) -> int:
    die = next(item for item in ir1.fabric.dies if item.id == dag.die_id)
    profiles = {item.id: item for item in ir1.fabric.sram_profiles}
    common = {MemoryInitiator.COMPUTE}
    if options.enable_double_buffer:
        common.add(MemoryInitiator.LSU)
    source_required = common | {MemoryInitiator.DTE}
    destination_required = common | {MemoryInitiator.NOC_RX}
    def supports(core: object, required: set[MemoryInitiator]) -> bool:
        return any(
            required.issubset(region.access)
            for region in profiles[core.sram_profile_ref].regions
        )
    available = max(
        (
            1 + sum(
                destination.runtime_core_id != source.runtime_core_id
                and supports(destination, destination_required)
                for destination in die.cores
            )
            for source in die.cores
            if supports(source, source_required)
        ),
        default=1,
    )
    return min(
        options.split_k_parts, options.compute_groups_per_die, available
    )


def _refine_dag(
    dag: IntraDieDAG,
    options: SplitKRefineOptions,
    ir1: IR1,
) -> tuple[IntraDieDAG, tuple[SplitKTaskRewrite, ...]]:
    source_values = {
        value.id: value
        for value in (*dag.values, *dag.state_staging_values)
    }
    eligible = tuple(task for task in dag.tasks if _eligible(task, dag))
    generated_by_source: dict[str, tuple[SemanticTask, ...]] = {}
    derived_values: list[IntraDieValue] = []
    rewrites: list[SplitKTaskRewrite] = []
    replacement_dependency: dict[str, str] = {}
    local_regions: list[IntraDieRegion] = []
    direct_dma_by_original: dict[str, list[SemanticTask]] = {}
    replaced_dma_access_refs: dict[str, set[str]] = {}
    direct_dma_regions: list[IntraDieRegion] = []
    compute_group_count = _compute_group_count(dag, options, ir1)

    for source in eligible:
        assert source.compute is not None
        workload = source.compute.workload
        assert type(workload) is GemmWorkload
        if (
            workload.logical_shape[2] % options.split_k_parts
            or workload.rank_shape[2] % options.split_k_parts
        ):
            _fail(
                "eligible GEMM K must divide split_k_parts",
                f"projection.dags[{dag.die_id}].tasks[{source.id}]",
            )
        if options.enable_reduce:
            ReductionContract(
                ReduceOp.SUM,
                workload.dtype,
                source.compute.math.accumulation_dtype,
                workload.dtype,
                RoundingMode.RNE,
                tuple(range(options.split_k_parts)),
            ).validate(f"projection.dags[{dag.die_id}].tasks[{source.id}]")
        output = source_values.get(source.write_values[0])
        inputs = tuple(source_values.get(value_id) for value_id in source.read_values)
        if output is None or any(value is None for value in inputs):
            _fail(
                "split-K source operands must be local IntraDieValue entries",
                f"projection.dags[{dag.die_id}].tasks[{source.id}]",
            )
        assert inputs[0] is not None and inputs[1] is not None
        if type(output) is not IntraDieValue:
            _fail(
                "split-K output must be an ordinary local value",
                f"projection.dags[{dag.die_id}].tasks[{source.id}]",
            )
        from .naive_intra_die import _ordinary_rank_local_view

        output_slice = _ordinary_rank_local_view(
            source, output, ir1
        )
        input_base_views = tuple(
            _ordinary_rank_local_view(source, value, ir1)
            if type(value) is IntraDieValue
            else TensorSlice(
                value.id, (0,) * len(value.shape), value.shape
            )
            for value in inputs
        )
        rank_m, rank_n, rank_k = workload.rank_shape
        rank_k_part = rank_k // options.split_k_parts
        stage_tasks: list[SemanticTask] = []
        input_local_tasks: list[SemanticTask] = []
        input_handoffs: list[SplitKInputHandoff] = []
        parts: list[SemanticTask] = []
        versions: list[DoubleBufferVersion] = []
        direct_dma_task_ids: list[str] = []
        partial_ids = tuple(
            split_k_partial_value_id(source.id, output.id, index)
            for index in range(options.split_k_parts)
        )
        for part_index in range(options.split_k_parts):
            part_input_slices = (
                TensorSlice(
                    inputs[0].id,
                    (
                        input_base_views[0].offset[0],
                        input_base_views[0].offset[1]
                        + part_index * rank_k_part,
                    ),
                    (rank_m, rank_k_part),
                ),
                TensorSlice(
                    inputs[1].id,
                    (
                        input_base_views[1].offset[0]
                        + part_index * rank_k_part,
                        input_base_views[1].offset[1],
                    ),
                    (rank_k_part, rank_n),
                ),
            )
            effective_input_ids = list(source.read_values)
            effective_input_slices = list(part_input_slices)
            pack_ready_by_operand: dict[int, str] = {}
            # A K-column slice of A[M,K] is row-strided when M > 1.
            # Materialize it as exact contiguous rows into one tight local pack.
            lhs = inputs[0]
            lhs_slice = part_input_slices[0]
            lhs_producers = tuple(
                task_id for task_id in lhs.producer_tasks
                if task_id in source.deps
            ) if type(lhs) is IntraDieValue else ()
            needs_lhs_pack = (
                not options.enable_double_buffer
                and type(lhs) is IntraDieValue
                and lhs_slice.shape[0] > 1
                and lhs_slice.shape[1] < input_base_views[0].shape[1]
                and len(lhs_producers) == 1
            )
            if needs_lhs_pack:
                assert type(lhs) is IntraDieValue
                packed_id = split_k_pack_value_id(
                    source.id, lhs.id, part_index, 0
                )
                previous_dep = lhs_producers[0]
                for row_index in range(lhs_slice.shape[0]):
                    row_slice = TensorSlice(
                        lhs.id,
                        (lhs_slice.offset[0] + row_index, lhs_slice.offset[1]),
                        (1, lhs_slice.shape[1]),
                    )
                    pack_deps = (previous_dep,)
                    pack_task = _pack_row_task(
                        source, lhs, part_index=part_index, operand_index=0,
                        row_index=row_index, packed_value_id=packed_id,
                        tensor_slice=row_slice, deps=pack_deps,
                    )
                    stage_tasks.append(pack_task)
                    previous_dep = pack_task.id
                pack_ready_by_operand[0] = previous_dep
                effective_input_ids[0] = packed_id
                effective_input_slices[0] = TensorSlice(
                    packed_id, lhs_slice.offset, lhs_slice.shape
                )
                origin_value_id, origin_sharding = _source_origin(lhs, ir1)
                derived_values.append(IntraDieValue(
                    id=packed_id, origin_value_id=origin_value_id,
                    shape=lhs.shape, dtype=lhs.dtype,
                    logical_layout=lhs.logical_layout,
                    sharding=origin_sharding, alias_set=None,
                    producer_tasks=(), consumer_tasks=(),
                ))
            input_waits: list[str] = []
            direct_dma_by_operand: dict[int, str] = {}
            direct_dma_source_tasks: set[str] = set()
            if options.enable_direct_dma:
                for operand_index, (input_value, tensor_slice) in enumerate(
                    zip(inputs, part_input_slices, strict=True)
                ):
                    if type(input_value) is not StateStagingValue:
                        continue
                    producers = tuple(
                        task_id for task_id in input_value.producer_tasks
                        if task_id in source.deps
                    )
                    if len(producers) != 1:
                        continue
                    original_dma = next(
                        (task for task in dag.tasks if task.id == producers[0]), None
                    )
                    if original_dma is None:
                        continue
                    built = _direct_dma_task(
                        original_dma, source, part_index=part_index,
                        operand_index=operand_index,
                        destination_task_id=split_k_part_task_id(
                            source.id, part_index
                        ),
                        tensor_slice=tensor_slice, dtype=input_value.dtype,
                    )
                    if built is None:
                        continue
                    dma_task, dma_region = built
                    direct_dma_by_original.setdefault(
                        original_dma.id, []
                    ).append(dma_task)
                    replaced_dma_access_refs.setdefault(
                        original_dma.id, set()
                    ).add(source.id)
                    direct_dma_regions.append(dma_region)
                    direct_dma_task_ids.append(dma_task.id)
                    direct_dma_by_operand[operand_index] = dma_task.id
                    direct_dma_source_tasks.add(original_dma.id)
            if (
                options.enable_reduce
                and part_index % compute_group_count != 0
            ):
                for operand_index, (input_value, tensor_slice) in enumerate(
                    zip(inputs, part_input_slices, strict=True)
                ):
                    assert input_value is not None
                    if operand_index in direct_dma_by_operand:
                        continue
                    producers = tuple(
                        task_id for task_id in input_value.producer_tasks
                        if task_id in source.deps
                    )
                    if len(producers) > 1:
                        _fail(
                            "split-K input must have at most one local producer",
                            f"tasks[{source.id}].read_values[{operand_index}]",
                        )
                    if not producers:
                        continue
                    destination_task_id = split_k_part_task_id(
                        source.id, part_index
                    )
                    transport_slice = effective_input_slices[operand_index]
                    transport_value_id = effective_input_ids[operand_index]
                    chain, carrier, region = _input_handoff(
                        source, part_index=part_index,
                        operand_index=operand_index,
                        producer_task_id=producers[0],
                        send_dependency_task_id=pack_ready_by_operand.get(
                            operand_index, producers[0]
                        ),
                        destination_task_id=destination_task_id,
                        destination_value_id=(
                            split_k_staged_value_id(
                                source.id, input_value.id, part_index, operand_index,
                                compute_group_count,
                            )
                            if (
                                options.enable_double_buffer
                                or type(input_value) is StateStagingValue
                            ) else transport_value_id
                        ),
                        tensor_slice=transport_slice, dtype=input_value.dtype,
                        destination_compute_group=(
                            part_index % compute_group_count
                        ),
                    )
                    input_local_tasks.extend(chain)
                    input_handoffs.append(carrier)
                    local_regions.append(region)
                    input_waits.append(carrier.wait_task_id)
            if options.enable_double_buffer:
                direct_receive_operands = tuple(
                    item.operand_index for item in input_handoffs
                    if item.part_index == part_index
                )
                direct_receive_set = set(direct_receive_operands)
                staged_ids = tuple(
                    split_k_staged_value_id(
                        source.id,
                        value.id,
                        part_index,
                        operand_index,
                        compute_group_count,
                    )
                    for operand_index, value in enumerate(inputs)
                    if value is not None
                )
                stage_ids = tuple(
                    split_k_input_wait_task_id(source.id, part_index, operand_index)
                    if operand_index in direct_receive_set
                    else split_k_stage_task_id(source.id, part_index, operand_index)
                    for operand_index in range(2)
                )
                ready_events = tuple(
                    split_k_input_wait_task_id(source.id, part_index, operand_index)
                    if operand_index in direct_receive_set
                    else split_k_ready_event_id(
                        source.id, part_index, operand_index, compute_group_count
                    )
                    for operand_index in range(2)
                )
                versions.append(
                    DoubleBufferVersion(
                        part_index,
                        part_index // compute_group_count,
                        (part_index // compute_group_count) % 2,
                        stage_ids,
                        staged_ids,
                        ready_events,
                        direct_receive_operands,
                        split_k_compute_event_id(source.id, part_index),
                    )
                )
                # Stage only the exact rank-local K tile.  Versions retain logical
                # provenance while the scheduler may map alternating versions onto
                # two physical slots after their reuse dependency has fired.
                stage_slices = part_input_slices
                for operand_index, (input_value, staged_id, tensor_slice) in enumerate(
                    zip(inputs, staged_ids, stage_slices, strict=True)
                ):
                    assert input_value is not None
                    if operand_index in direct_receive_set:
                        origin_value_id, origin_sharding = _source_origin(
                            input_value, ir1
                        )
                        derived_values.append(IntraDieValue(
                            id=staged_id, origin_value_id=origin_value_id,
                            shape=input_value.shape, dtype=input_value.dtype,
                            logical_layout=input_value.logical_layout,
                            sharding=origin_sharding, alias_set=None,
                            producer_tasks=(), consumer_tasks=(),
                        ))
                        continue
                    reuse_dep = (
                        (split_k_part_task_id(source.id, part_index - compute_group_count),)
                        if part_index >= compute_group_count
                        else ()
                    )
                    stage_tasks.append(
                        _stage_task(
                            source,
                            input_value,
                            part_index=part_index,
                            operand_index=operand_index,
                            staged_value_id=staged_id,
                            tensor_slice=tensor_slice,
                            compute_group_count=compute_group_count,
                            deps=tuple(dict.fromkeys(
                                (tuple(input_waits) if part_index % compute_group_count else source.deps)
                                + reuse_dep
                            )),
                        )
                    )
                    origin_value_id, origin_sharding = _source_origin(
                        input_value, ir1
                    )
                    derived_values.append(
                        IntraDieValue(
                            id=staged_id,
                            origin_value_id=origin_value_id,
                            shape=input_value.shape,
                            dtype=input_value.dtype,
                            logical_layout=input_value.logical_layout,
                            sharding=origin_sharding,
                            alias_set=None,
                            producer_tasks=(),
                            consumer_tasks=(),
                        )
                    )
                input_ids = (staged_ids[0], staged_ids[1])
                part_deps = stage_ids
            elif any(type(value) is StateStagingValue for value in inputs):
                exact_input_ids = list(effective_input_ids)
                handed_off_source_tasks = {
                    item.source_task_id for item in input_handoffs
                    if item.part_index == part_index
                }
                handed_off_operand_indices = {
                    item.operand_index for item in input_handoffs
                    if item.part_index == part_index
                }
                exact_stage_ids: list[str] = list(input_waits) + [
                    dependency for dependency in source.deps
                    if dependency not in direct_dma_source_tasks
                    and dependency not in handed_off_source_tasks
                ] + list(direct_dma_by_operand.values()) + [
                    task_id for operand_index, task_id
                    in pack_ready_by_operand.items()
                    if operand_index not in handed_off_operand_indices
                ]
                base_deps = (
                    tuple(input_waits) if part_index % compute_group_count
                    else tuple(dict.fromkeys(
                        tuple(
                            dependency for dependency in source.deps
                            if dependency not in direct_dma_source_tasks
                        ) + tuple(pack_ready_by_operand.values())
                    ))
                )
                for operand_index, (input_value, tensor_slice) in enumerate(
                    zip(inputs, part_input_slices, strict=True)
                ):
                    assert input_value is not None
                    if type(input_value) is not StateStagingValue:
                        continue
                    direct_dma_id = direct_dma_by_operand.get(operand_index)
                    if direct_dma_id is not None:
                        exact_input_ids[operand_index] = input_value.id
                        continue
                    staged_id = split_k_staged_value_id(
                        source.id, input_value.id, part_index, operand_index,
                        compute_group_count,
                    )
                    direct_handoff = next((
                        item for item in input_handoffs
                        if item.part_index == part_index
                        and item.operand_index == operand_index
                    ), None)
                    if direct_handoff is not None:
                        exact_stage_ids.append(direct_handoff.wait_task_id)
                        exact_input_ids[operand_index] = staged_id
                        origin_value_id, origin_sharding = _source_origin(
                            input_value, ir1
                        )
                        derived_values.append(IntraDieValue(
                            id=staged_id, origin_value_id=origin_value_id,
                            shape=input_value.shape, dtype=input_value.dtype,
                            logical_layout=input_value.logical_layout,
                            sharding=origin_sharding, alias_set=None,
                            producer_tasks=(), consumer_tasks=(),
                        ))
                        continue
                    stage = _stage_task(
                        source, input_value, part_index=part_index,
                        operand_index=operand_index, staged_value_id=staged_id,
                        tensor_slice=tensor_slice,
                        compute_group_count=compute_group_count, deps=base_deps,
                    )
                    stage_tasks.append(stage)
                    exact_stage_ids.append(stage.id)
                    exact_input_ids[operand_index] = staged_id
                    origin_value_id, origin_sharding = _source_origin(
                        input_value, ir1
                    )
                    derived_values.append(IntraDieValue(
                        id=staged_id, origin_value_id=origin_value_id,
                        shape=input_value.shape, dtype=input_value.dtype,
                        logical_layout=input_value.logical_layout,
                        sharding=origin_sharding, alias_set=None,
                        producer_tasks=(), consumer_tasks=(),
                    ))
                input_ids = (exact_input_ids[0], exact_input_ids[1])
                part_deps = tuple(dict.fromkeys(exact_stage_ids))
            else:
                input_ids = (effective_input_ids[0], effective_input_ids[1])
                part_deps = (
                    tuple(input_waits)
                    if part_index % compute_group_count
                    else tuple(dict.fromkeys(
                        source.deps + tuple(pack_ready_by_operand.values())
                    ))
                )
            parts.append(
                _split_compute(
                    source,
                    part_index=part_index,
                    parts=options.split_k_parts,
                    input_ids=input_ids,
                    input_slices=(effective_input_slices[0], effective_input_slices[1]),
                    output_id=partial_ids[part_index],
                    output_slice=output_slice,
                    deps=part_deps,
                )
            )
            derived_values.append(
                IntraDieValue(
                    id=partial_ids[part_index],
                    origin_value_id=output.origin_value_id,
                    shape=output.shape,
                    dtype=output.dtype,
                    logical_layout=output.logical_layout,
                    sharding=output.sharding,
                    alias_set=None,
                    producer_tasks=(),
                    consumer_tasks=(),
                )
            )
        part_ids = tuple(task.id for task in parts)
        if options.enable_reduce and options.enable_tree_reduce:
            payload_bytes = math.prod(output_slice.shape) * _element_bytes(
                workload.dtype, f"tasks[{source.id}]"
            )
            (
                reduction_tail, handoffs, tree_regions,
                reduction_task_ids, reduction_accumulator_ids,
            ) = _tree_reduction(
                source, partial_ids=partial_ids, part_task_ids=part_ids,
                output_value_id=output.id, output_slice=output_slice,
                compute_group_count=compute_group_count,
                payload_bytes=payload_bytes,
            )
            local_regions.extend(tree_regions)
            for accumulator_id in reduction_accumulator_ids:
                derived_values.append(IntraDieValue(
                    id=accumulator_id,
                    origin_value_id=output.origin_value_id,
                    shape=output.shape, dtype=output.dtype,
                    logical_layout=output.logical_layout,
                    sharding=output.sharding, alias_set=None,
                    producer_tasks=(), consumer_tasks=(),
                ))
            generated = (
                *input_local_tasks, *stage_tasks, *parts, *reduction_tail,
            )
            reduce_id = reduction_task_ids[-1]
            replacement_dependency[source.id] = reduce_id
            commit_id = None
        elif options.enable_reduce:
            output_local_tasks: list[SemanticTask] = []
            handoffs: list[SplitKLocalHandoff] = []
            ready_by_part = list(part_ids)
            payload_bytes = math.prod(output_slice.shape) * _element_bytes(
                workload.dtype, f"tasks[{source.id}]"
            )
            for part_index, partial_id in enumerate(partial_ids):
                if part_index % compute_group_count == 0:
                    continue
                consumer_reduce_id = split_k_reduce_step_task_id(
                    source.id, part_index, options.split_k_parts
                )
                chain, handoff, region = _local_handoff(
                    source, part_index=part_index, partial_value_id=partial_id,
                    output_slice=output_slice, payload_bytes=payload_bytes,
                    reduce_task_id=consumer_reduce_id,
                    source_compute_group=(part_index % compute_group_count),
                    source_dependency_task_id=split_k_part_task_id(
                        source.id, part_index
                    ),
                )
                output_local_tasks.extend(chain)
                handoffs.append(handoff)
                local_regions.append(region)
                ready_by_part[part_index] = handoff.wait_task_id
            reduction_tasks: list[SemanticTask] = []
            accumulator_ids: list[str] = []
            previous_value = partial_ids[0]
            previous_task_id: str | None = None
            for step_index in range(1, options.split_k_parts):
                task_id = split_k_reduce_step_task_id(
                    source.id, step_index, options.split_k_parts
                )
                is_commit = step_index == options.split_k_parts - 1
                output_value_id = (
                    output.id if is_commit else split_k_accumulator_value_id(
                        source.id, output.id, step_index
                    )
                )
                if options.enable_streaming_reduce:
                    deps = tuple(dict.fromkeys(
                        (
                            (ready_by_part[0],)
                            if step_index == 1
                            else (previous_task_id,)
                        )
                        + (ready_by_part[step_index],)
                    ))
                else:
                    deps = tuple(dict.fromkeys(
                        tuple(ready_by_part)
                        if step_index == 1
                        else (previous_task_id,)
                    ))
                reduction_tasks.append(_reduce_task(
                    source, task_id=task_id,
                    input_value_ids=(previous_value, partial_ids[step_index]),
                    output_value_id=output_value_id, output_slice=output_slice,
                    deps=deps, step_index=step_index,
                ))
                if not is_commit:
                    accumulator_ids.append(output_value_id)
                    derived_values.append(IntraDieValue(
                        id=output_value_id, origin_value_id=output.origin_value_id,
                        shape=output.shape, dtype=output.dtype,
                        logical_layout=output.logical_layout, sharding=output.sharding,
                        alias_set=None, producer_tasks=(), consumer_tasks=(),
                    ))
                previous_value = output_value_id
                previous_task_id = task_id
            final = reduction_tasks[-1]
            replacement_dependency[source.id] = final.id
            if options.enable_streaming_reduce:
                local_tasks_by_part = {
                    handoff.part_index: tuple(
                        task for task in output_local_tasks
                        if task.id in {
                            handoff.send_task_id,
                            handoff.recv_task_id,
                            handoff.wait_task_id,
                        }
                    )
                    for handoff in handoffs
                }
                reduction_tail = tuple(
                    task
                    for step_index, reduction_task in enumerate(
                        reduction_tasks, start=1
                    )
                    for task in (
                        *local_tasks_by_part.get(step_index, ()),
                        reduction_task,
                    )
                )
            else:
                reduction_tail = (
                    *output_local_tasks, *reduction_tasks,
                )
            generated = (
                *input_local_tasks, *stage_tasks, *parts,
                *reduction_tail,
            )
            reduction_task_ids = tuple(task.id for task in reduction_tasks)
            reduction_accumulator_ids = tuple(accumulator_ids)
            reduce_id = final.id
            commit_id = None
        else:
            handoffs = []
            reduction_task_ids = ()
            reduction_accumulator_ids = ()
            commit = replace(
                source,
                deps=tuple(dict.fromkeys(source.deps + part_ids)),
            )
            generated = (*input_local_tasks, *stage_tasks, *parts, commit)
            reduce_id = None
            commit_id = source.id
        generated_by_source[source.id] = tuple(generated)
        rewrites.append(
            SplitKTaskRewrite.create(
                source_dag_id=dag.id,
                source_task_id=source.id,
                source_output_value_id=output.id,
                part_task_ids=part_ids,
                partial_value_ids=partial_ids,
                reduce_task_id=reduce_id,
                reduction_task_ids=reduction_task_ids,
                reduction_accumulator_value_ids=reduction_accumulator_ids,
                semantic_commit_task_id=commit_id,
                temporal_chunks=(options.split_k_parts + compute_group_count - 1) // compute_group_count,
                double_buffer_versions=tuple(versions),
                compute_group_count=compute_group_count,
                part_compute_groups=(
                    tuple(index % compute_group_count for index in range(options.split_k_parts))
                    if options.enable_reduce
                    else (0,) * options.split_k_parts
                ),
                reduce_compute_group=0 if options.enable_reduce else None,
                local_handoffs=tuple(handoffs),
                input_handoffs=tuple(input_handoffs),
                enable_tree_reduce=options.enable_tree_reduce,
                enable_direct_dma=options.enable_direct_dma,
                direct_dma_task_ids=tuple(direct_dma_task_ids),
            )
        )

    expanded: list[SemanticTask] = []
    for task in dag.tasks:
        direct_clones = direct_dma_by_original.get(task.id)
        if direct_clones is not None:
            assert task.dma is not None
            remaining_refs = tuple(
                task_ref for task_ref in task.dma.access_task_refs
                if task_ref not in replaced_dma_access_refs[task.id]
            )
            if remaining_refs:
                expanded.append(replace(
                    task, dma=replace(
                        task.dma, access_task_refs=remaining_refs
                    ),
                ))
            expanded.extend(direct_clones)
            continue
        expanded.extend(generated_by_source.get(task.id, (task,)))
    # A generated split/stage task may itself depend on another source GEMM.
    # Rewrite every dependency after all source replacements are known.
    tasks = tuple(
        replace(
            task,
            deps=tuple(
                dict.fromkeys(
                    replacement_dependency.get(dependency, dependency)
                    for dependency in task.deps
                )
            ),
        )
        for task in expanded
    )
    rewritten_tasks: list[SemanticTask] = []
    for task in tasks:
        if task.dma is None:
            rewritten_tasks.append(task)
            continue
        access_refs: list[str] = []
        for access_ref in task.dma.access_task_refs:
            generated = generated_by_source.get(access_ref)
            if generated is None:
                access_refs.append(
                    replacement_dependency.get(access_ref, access_ref)
                )
                continue
            matching = tuple(
                candidate.id
                for candidate in generated
                if (
                    (
                        task.dma.local_value_ref in (
                            *candidate.read_values, *candidate.write_values
                        )
                        and candidate.id not in {
                            handoff.destination_task_id
                            for rewrite in rewrites
                            for handoff in rewrite.input_handoffs
                            if handoff.value_id == task.dma.local_value_ref
                        }
                    )
                    or candidate.id in {
                        handoff.send_task_id
                        for rewrite in rewrites
                        for handoff in rewrite.input_handoffs
                        if handoff.value_id == task.dma.local_value_ref
                    }
                )
            )
            access_refs.extend(
                matching
                or (replacement_dependency.get(access_ref, access_ref),)
            )
        rewritten_tasks.append(
            replace(
                task,
                dma=replace(
                    task.dma,
                    access_task_refs=tuple(dict.fromkeys(access_refs)),
                ),
            )
        )
    tasks = tuple(rewritten_tasks)
    values_shells = (*dag.values, *derived_values)
    values = tuple(
        replace(
            value,
            producer_tasks=tuple(
                task.id for task in tasks if value.id in task.write_values
            ),
            consumer_tasks=tuple(
                task.id for task in tasks if value.id in task.read_values
            ),
        )
        for value in values_shells
    )
    regions = tuple(
        replace(region, task_ids=region_task_ids)
        for region in dag.regions
        for region_task_ids in (tuple(
            task.id for task in tasks if task.region_id == region.id
        ),)
        if region_task_ids
    ) + tuple(direct_dma_regions) + tuple(local_regions)
    dag_semantic: dict[str, object] = {
        "source_ir1_id": dag.source_ir1_id,
        "die_id": dag.die_id,
        "fusion_plan_ids": dag.fusion_plan_ids,
        "standalone_collective_plan_ids": dag.standalone_collective_plan_ids,
        "ordinary_node_ids": dag.ordinary_node_ids,
        "tasks": tasks,
        "values": values,
        "flows": dag.flows,
        "regions": regions,
        "source_state_manifest_id": dag.source_state_manifest_id,
        "state_access_ids": dag.state_access_ids,
        "state_staging_values": tuple(
            replace(
                value,
                producer_tasks=tuple(
                    task.id for task in tasks
                    if value.id in task.write_values
                ),
                consumer_tasks=tuple(
                    task.id for task in tasks
                    if value.id in task.read_values
                ),
            )
            for value in dag.state_staging_values
        ),
        "state_transfer_ids": dag.state_transfer_ids,
    }
    if dag.swizzle_values:
        dag_semantic["swizzle_values"] = dag.swizzle_values
    refined = IntraDieDAG.create(
        producer_pass="intra_die_refine", **dag_semantic
    )
    refined.validate(f"refined_dag[{dag.die_id}]")
    return refined, tuple(rewrites)

def refine_split_k_projection(
    projection: IR2ProjectionResult,
    options: SplitKRefineOptions | IntraDieOptimizationOptions,
    ir1: IR1,
) -> SplitKRefinedProjection:
    from .intra_die_v2_search import evaluate_intra_die_v2_candidates

    search_decision = evaluate_intra_die_v2_candidates(projection, ir1, options)
    selected = next(
        candidate for candidate in search_decision.candidates
        if candidate.id == search_decision.selected_candidate_ref
    )
    if selected.kind is IntraDieV2CandidateKind.IDENTITY:
        result = SplitKRefinedProjection.create(
            source_projection_id=projection.id, projection=projection, rewrites=(),
            search_decision=search_decision,
        )
        result.validate_against(
            projection, ir1, split_k_parts=1, enable_reduce=False,
            enable_double_buffer=False, enable_streaming_reduce=False,
        )
        return result

    materialization_options = SplitKRefineOptions(
        split_k_parts=selected.split_k_parts,
        enable_reduce=selected.enable_reduce,
        enable_double_buffer=selected.enable_double_buffer,
        enable_streaming_reduce=selected.enable_streaming_reduce,
        enable_tree_reduce=selected.enable_tree_reduce,
        enable_direct_dma=selected.enable_direct_dma,
        compute_groups_per_die=(
            options.compute_groups_per_die
            if type(options) in (SplitKRefineOptions, IntraDieOptimizationOptions)
            else 2
        ),
    )
    refined_dags: list[IntraDieDAG] = []
    rewrites: list[SplitKTaskRewrite] = []
    for dag in projection.dags:
        refined, dag_rewrites = _refine_dag(
            dag, materialization_options, ir1
        )
        refined_dags.append(refined)
        rewrites.extend(dag_rewrites)
    refined_projection = IR2ProjectionResult.create(
        producer_pass="intra_die_refine",
        source_ir1_id=projection.source_ir1_id,
        fusion_plan_ids=projection.fusion_plan_ids,
        standalone_collective_plan_ids=projection.standalone_collective_plan_ids,
        dags=tuple(refined_dags),
        source_state_manifest_id=projection.source_state_manifest_id,
        state_transfers=projection.state_transfers,
    )
    result = SplitKRefinedProjection.create(
        source_projection_id=projection.id, projection=refined_projection,
        rewrites=tuple(rewrites), search_decision=search_decision,
    )
    result.validate_against(
        projection, ir1, split_k_parts=materialization_options.split_k_parts,
        enable_reduce=materialization_options.enable_reduce,
        enable_double_buffer=materialization_options.enable_double_buffer,
        enable_streaming_reduce=materialization_options.enable_streaming_reduce,
        enable_tree_reduce=materialization_options.enable_tree_reduce,
        enable_direct_dma=materialization_options.enable_direct_dma,
    )
    return result


__all__ = ["refine_split_k_projection"]
