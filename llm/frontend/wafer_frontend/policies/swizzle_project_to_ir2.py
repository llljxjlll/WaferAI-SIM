"""Fail-closed common-IR2 projection for the first Swizzle GEMM_RS ABI."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.action import (
    ComputeContract,
    ComputeOperand,
    ComputeOperandSlice,
    ComputeTileBinding,
    FusionActionKind,
    ReductionContract,
    StandaloneCollectivePlan,
    SwizzleBoundActionRef,
)
from ..schema.common import DType, RoundingMode
from ..schema.ir0 import EdgeKind, FusionPattern, GemmPartition, GemmWorkload, OpKind
from ..schema.ir1 import IR1, RankPlacement, TensorValue
from ..schema.ir2 import (
    IR2ProjectionResult,
    IntraDieDAG,
    IntraDieRegion,
    IntraDieValue,
    OrdinaryNodeOrigin,
    OriginKind,
    RegionLowering,
    SemanticFlow,
    SemanticTask,
    SemanticTaskKind,
    StandaloneNodeOrigin,
    SwizzleIntraDieValue,
    SwizzleNodeOrigin,
    TensorSlice,
    canonical_semantic_flow_id,
)
from ..schema.state_transfer import StateTransferLike
from ..schema.swizzle_plan import (
    SwizzleBoundAction,
    SwizzleFusionPlan,
    SwizzleValueOrigin,
)


def _fail(message: str, path: str) -> None:
    raise SchemaError(message, path=path)


def _unsupported(message: str, path: str) -> None:
    raise UnsupportedFeatureError(message, path=path)


def _task_id(action_id: str) -> str:
    return f"task.{action_id}"


def _region_id(plan_id: str, die_id: int) -> str:
    return f"region.fusion.{plan_id}.die.{die_id}"


def _value_id(plan_id: str, rank: int, value_ref: str, ir1_values: set[str]) -> str:
    return value_ref if value_ref in ir1_values else f"swizzle.{plan_id}.rank.{rank}.{value_ref}"


def _bound_value_id(
    plan: SwizzleFusionPlan, rank: int, action: SwizzleBoundAction,
    value_ref: str, ir1_values: set[str], *, write: bool,
) -> str:
    if write and action.fusion_kind is FusionActionKind.LOCAL_COPY:
        outputs = action.source_action.output_refs
        logical_output = plan.decision.problem.collective.output.value_ref
        if (
            outputs != (value_ref,)
            or action.member_ref != plan.decision.problem.collective.node_ref
            or logical_output not in ir1_values
        ):
            _fail(
                "terminal Swizzle LOCAL_COPY lacks one exact IR-1 collective output",
                "fusion_plans.rank_programs.actions",
            )
        return logical_output
    return _value_id(plan.id, rank, value_ref, ir1_values)


def _origin(plan_id: str, rank: int, action_id: str) -> SwizzleNodeOrigin:
    return SwizzleNodeOrigin(
        OriginKind.SWIZZLE_FUSED,
        SwizzleBoundActionRef(plan_id, rank, action_id),
    )


def _element_bytes(dtype: DType, path: str) -> int:
    result = {DType.FP16: 2, DType.FP32: 4}.get(dtype)
    if result is None:
        _unsupported(
            "Swizzle GEMM_RS common IR2 supports FP16/FP32 payloads only",
            path,
        )
    return result


def _chunk_slice(
    action: SwizzleBoundAction,
    values: dict[str, TensorValue],
    *,
    path: str,
) -> tuple[TensorSlice, DType, int]:
    chunk = action.chunk_origin
    if chunk is None:
        _unsupported(
            "Swizzle GEMM_RS common IR2 requires an exact chunk origin for every action",
            f"{path}.chunk_origin",
        )
    source = values.get(chunk.source_value_ref)
    if source is None:
        _fail(
            "chunk origin references a missing IR-1 value",
            f"{path}.chunk_origin.source_value_ref",
        )
    if len(source.shape) != len(chunk.logical_shape) or any(
        offset + extent > source.shape[index]
        for index, (offset, extent) in enumerate(
            zip(chunk.logical_offset, chunk.logical_shape, strict=True)
        )
    ):
        _fail(
            "chunk origin lies outside its IR-1 logical value",
            f"{path}.chunk_origin",
        )
    payload_bytes = math.prod(chunk.logical_shape) * _element_bytes(
        source.dtype, f"{path}.chunk_origin"
    )
    return (
        TensorSlice(
            chunk.source_value_ref,
            chunk.logical_offset,
            chunk.logical_shape,
        ),
        source.dtype,
        payload_bytes,
    )


def _compute_contract(
    action: SwizzleBoundAction,
    placement: RankPlacement,
    *,
    path: str,
) -> ComputeContract:
    source = action.source_action
    origin = action.compute_origin
    chunk = action.chunk_origin
    if origin is None or chunk is None:
        _fail("COMP requires exact compute and chunk origins", path)
    if (
        origin.workload.partition is not GemmPartition.ROW_PARALLEL
        or len(source.input_refs) != 2
        or len(source.output_refs) != 1
        or len(origin.ir1_input_refs) != 2
        or len(origin.ir1_output_refs) != 1
        or len(chunk.logical_shape) != 2
        or chunk.axis != 0
    ):
        _unsupported(
            "Swizzle GEMM_RS common IR2 supports canonical M-chunked row-parallel GEMM only",
            path,
        )

    full_m, full_n, full_k = origin.workload.logical_shape
    rank_m, rank_n, rank_k = origin.workload.rank_shape
    chunk_m, chunk_n = chunk.logical_shape
    chunk_m_offset, chunk_n_offset = chunk.logical_offset
    if (
        rank_m != full_m
        or chunk_n != full_n
        or chunk_n_offset != 0
        or chunk_m_offset + chunk_m > full_m
        or chunk.source_value_ref != origin.ir1_output_refs[0]
    ):
        _fail(
            "COMP chunk does not preserve the exact GEMM output tile",
            f"{path}.chunk_origin",
        )
    if len(placement.logical_coord) != 1:
        _unsupported(
            "Swizzle GEMM_RS common IR2 requires one-dimensional TP placement",
            f"{path}.rank",
        )
    rhs_k_offset = placement.logical_coord[0] * rank_k
    if rhs_k_offset + rank_k > full_k:
        _fail("rank-local RHS slice lies outside logical K", path)

    workload = GemmWorkload(
        logical_shape=(chunk_m, full_n, full_k),
        rank_shape=(chunk_m, rank_n, rank_k),
        partition=origin.workload.partition,
        dtype=origin.workload.dtype,
    )
    expected_flops = 2 * math.prod(workload.rank_shape)
    if source.flops != expected_flops or source.logical_bytes != 0:
        _fail(
            "COMP witness cost disagrees with its exact GEMM tile",
            f"{path}.source_action",
        )
    return ComputeContract(
        op_kind=OpKind.GEMM,
        workload=workload,
        math=origin.math,
        effects=origin.effects,
        impl_ref=origin.impl_ref,
        inputs=(
            ComputeOperand(source.input_refs[0], "lhs"),
            ComputeOperand(source.input_refs[1], "rhs"),
        ),
        outputs=(ComputeOperand(source.output_refs[0], "partial"),),
        tile=ComputeTileBinding(
            origin_workload=origin.workload,
            input_slices=(
                ComputeOperandSlice(
                    source.input_refs[0],
                    origin.ir1_input_refs[0],
                    (chunk_m_offset, rhs_k_offset),
                    (chunk_m, rank_k),
                ),
                ComputeOperandSlice(
                    source.input_refs[1],
                    origin.ir1_input_refs[1],
                    (rhs_k_offset, 0),
                    (rank_k, full_n),
                ),
            ),
            output_slices=(
                ComputeOperandSlice(
                    source.output_refs[0],
                    origin.ir1_output_refs[0],
                    chunk.logical_offset,
                    chunk.logical_shape,
                ),
            ),
        ),
    )


def _reduction_contract(
    action: SwizzleBoundAction,
    action_index: dict[str, tuple[int, SwizzleBoundAction]],
    dtype: DType,
    *,
    path: str,
) -> ReductionContract:
    origin = action.reduction_origin
    source = action.source_action
    if origin is None:
        _fail("REDUCE requires an exact reduction origin", path)
    input_origins = action.value_origins[: len(source.input_refs)]
    input_ranks: list[int] = []
    for index, value_origin in enumerate(input_origins):
        producer_ref = value_origin.producer_action_ref
        if producer_ref is None or producer_ref not in action_index:
            _unsupported(
                "Swizzle GEMM_RS reduction inputs require exact action producers",
                f"{path}.value_origins[{index}]",
            )
        producer_rank, producer = action_index[producer_ref]
        if producer.fusion_kind is FusionActionKind.COMP:
            contribution_rank = producer_rank
        elif producer.fusion_kind is FusionActionKind.RECV:
            peer_rank = producer.source_action.peer_rank
            if peer_rank is None:
                _fail("RECV producer is missing its peer rank", path)
            contribution_rank = peer_rank
        else:
            _unsupported(
                "Swizzle GEMM_RS reduction inputs must come from COMP or RECV",
                f"{path}.value_origins[{index}]",
            )
        input_ranks.append(contribution_rank)
    contract = ReductionContract(
        reduce_op=origin.reduce_op,
        input_dtype=dtype,
        accumulation_dtype=origin.math.accumulation_dtype,
        output_dtype=dtype,
        rounding=RoundingMode.RNE,
        input_ranks=tuple(input_ranks),
    )
    contract.validate(f"{path}.reduction")
    return contract


@dataclass(frozen=True, slots=True)
class _FlowBinding:
    flow_id: str
    logical_channel: str
    pair_route_ref: str
    source_rank: int
    destination_rank: int
    source_die: int
    destination_die: int
    die_path: tuple[int, ...]
    tensor_slice: TensorSlice
    bytes: int
    dtype: DType
    send_action_id: str
    recv_action_id: str


def _flow_bindings(
    ir1: IR1,
    plans: tuple[SwizzleFusionPlan, ...],
    plan_actions: dict[str, dict[str, tuple[int, SwizzleBoundAction]]],
    placements: dict[str, dict[int, RankPlacement]],
    values: dict[str, TensorValue],
) -> tuple[
    tuple[_FlowBinding, ...],
    dict[tuple[str, int, str], _FlowBinding],
]:
    bindings: list[_FlowBinding] = []
    by_endpoint: dict[tuple[str, int, str], _FlowBinding] = {}
    matched_sends: set[tuple[str, str]] = set()
    for plan_index, plan in enumerate(plans):
        action_index = plan_actions[plan.id]
        route_ids = {
            route.id
            for group in ir1.groups
            if group.id == plan.group_ref
            for route in group.embedding.routes
        }
        for program_index, program in enumerate(plan.rank_programs):
            for action_index_in_program, recv in enumerate(program.actions):
                if recv.fusion_kind is not FusionActionKind.RECV:
                    continue
                path = (
                    f"fusion_plans[{plan_index}].rank_programs[{program_index}]"
                    f".actions[{action_index_in_program}]"
                )
                remote_dependencies = tuple(
                    (dependency, action_index[dependency])
                    for dependency in recv.source_action.deps
                    if dependency in action_index
                    and action_index[dependency][0] != program.rank
                )
                if len(remote_dependencies) != 1:
                    _unsupported(
                        "direct Swizzle GEMM_RS RECV requires exactly one remote SEND dependency",
                        f"{path}.source_action.deps",
                    )
                send_ref, (send_rank, send) = remote_dependencies[0]
                recv_source = recv.source_action
                send_source = send.source_action
                if (
                    send.fusion_kind is not FusionActionKind.SEND
                    or send_source.peer_rank != program.rank
                    or recv_source.peer_rank != send_rank
                    or send_source.route_ref is None
                    or send_source.route_ref != recv_source.route_ref
                    or send_source.route_ref not in route_ids
                    or send.expected_route != recv.expected_route
                    or len(send.expected_route) != 2
                    or send.chunk_origin != recv.chunk_origin
                    or send_source.logical_bytes != recv_source.logical_bytes
                ):
                    _fail(
                        "RECV and its remote SEND do not form one exact direct flow",
                        path,
                    )
                send_placement = placements[plan.id][send_rank]
                recv_placement = placements[plan.id][program.rank]
                if send.expected_route != (
                    send_placement.die_id,
                    recv_placement.die_id,
                ):
                    _fail("flow route disagrees with rank placement", path)
                tensor_slice, dtype, payload_bytes = _chunk_slice(
                    send, values, path=path
                )
                if send_source.logical_bytes != payload_bytes:
                    _fail(
                        "transport witness bytes disagree with its exact chunk payload",
                        f"{path}.source_action.logical_bytes",
                    )
                send_key = (plan.id, send_ref)
                if send_key in matched_sends:
                    _fail("SEND is consumed by more than one RECV", path)
                matched_sends.add(send_key)
                channel = (
                    f"swizzle.{plan.id}.send.{send_ref}.recv."
                    f"{recv_source.id}"
                )
                flow_id = canonical_semantic_flow_id(
                    _origin(plan.id, send_rank, send_ref), channel
                )
                binding = _FlowBinding(
                    flow_id=flow_id,
                    logical_channel=channel,
                    pair_route_ref=send_source.route_ref,
                    source_rank=send_rank,
                    destination_rank=program.rank,
                    source_die=send_placement.die_id,
                    destination_die=recv_placement.die_id,
                    die_path=send.expected_route,
                    tensor_slice=tensor_slice,
                    bytes=payload_bytes,
                    dtype=dtype,
                    send_action_id=send_ref,
                    recv_action_id=recv_source.id,
                )
                bindings.append(binding)
                by_endpoint[(plan.id, send_rank, send_ref)] = binding
                by_endpoint[(plan.id, program.rank, recv_source.id)] = binding

        all_sends = {
            (plan.id, action.source_action.id)
            for program in plan.rank_programs
            for action in program.actions
            if action.fusion_kind is FusionActionKind.SEND
        }
        if {item for item in matched_sends if item[0] == plan.id} != all_sends:
            _unsupported(
                "every direct Swizzle GEMM_RS SEND must pair with exactly one RECV",
                f"fusion_plans[{plan_index}].rank_programs",
            )
    return tuple(bindings), by_endpoint


def project_swizzle_gemm_rs_to_ir2(
    ir1: IR1,
    fusion_plans: tuple[SwizzleFusionPlan, ...],
    standalone_plans: tuple[StandaloneCollectivePlan, ...],
    *,
    state_transfers: tuple[StateTransferLike, ...],
) -> IR2ProjectionResult:
    """Project only the proven direct-route GEMM_RS Swizzle subset.

    This function is deliberately not a fallback.  Any unmodelled plan,
    route, value origin, compute tile, or state/standalone composition fails at
    this boundary, and the completed artifact is checked by
    :meth:`IR2ProjectionResult.validate_against` before it is returned.
    """

    if not fusion_plans or any(
        type(plan) is not SwizzleFusionPlan for plan in fusion_plans
    ):
        _fail("requires only SwizzleFusionPlan entries", "fusion_plans")
    if state_transfers:
        _unsupported(
            "Swizzle GEMM_RS common IR2 cannot combine state transfers yet",
            "state_transfers",
        )
    if len({plan.id for plan in fusion_plans}) != len(fusion_plans):
        _fail("contains duplicate Swizzle plan ids", "fusion_plans")
    for index, plan in enumerate(fusion_plans):
        if plan.pattern is not FusionPattern.GEMM_RS:
            _unsupported(
                "Swizzle common IR2 currently supports GEMM_RS only",
                f"fusion_plans[{index}].pattern",
            )
        plan.validate_against(ir1, f"fusion_plans[{index}]")
    for index, plan in enumerate(standalone_plans):
        plan.validate_against(ir1, f"standalone_plans[{index}]")
        if any(
            len(action.expected_route) > 2
            for program in plan.rank_programs for action in program.actions
        ):
            _unsupported(
                "mixed Swizzle/standalone common IR2 supports direct standalone routes only",
                f"standalone_plans[{index}]",
            )

    groups = {group.id: group for group in ir1.groups}
    nodes = {node.id: node for node in ir1.nodes}
    values = {value.id: value for value in ir1.values}
    skeletons = {item.id: item for item in ir1.fused_op_skeletons}
    fusion_by_member = {
        member_id: plan
        for plan in fusion_plans
        for member_id in skeletons[plan.fused_op_id].member_node_ids
    }
    standalone_by_node = {plan.op_id: plan for plan in standalone_plans}
    if (
        len(fusion_by_member)
        != sum(
            len(skeletons[plan.fused_op_id].member_node_ids)
            for plan in fusion_plans
        )
        or len(standalone_by_node) != len(standalone_plans)
        or set(fusion_by_member).intersection(standalone_by_node)
    ):
        _fail("plans must uniquely partition covered nodes", "fusion_plans")
    ordinary_nodes = tuple(
        node
        for node in ir1.nodes
        if node.id not in fusion_by_member
        and node.id not in standalone_by_node
        and node.kind is not OpKind.COLLECTIVE
    )
    placements: dict[str, dict[int, RankPlacement]] = {}
    plan_actions: dict[
        str, dict[str, tuple[int, SwizzleBoundAction]]
    ] = {}
    for plan_index, plan in enumerate(fusion_plans):
        group = groups[plan.group_ref]
        placements[plan.id] = {item.rank: item for item in group.placements}
        if set(placements[plan.id]) != {
            program.rank for program in plan.rank_programs
        }:
            _fail(
                "rank programs must exactly cover group placements",
                f"fusion_plans[{plan_index}].rank_programs",
            )
        action_index: dict[str, tuple[int, SwizzleBoundAction]] = {}
        for program_index, program in enumerate(plan.rank_programs):
            for action_index_in_program, action in enumerate(program.actions):
                action_id = action.source_action.id
                if action_id in action_index:
                    _fail(
                        "contains a duplicate Swizzle action id",
                        f"fusion_plans[{plan_index}].rank_programs[{program_index}]"
                        f".actions[{action_index_in_program}]",
                    )
                action_index[action_id] = (program.rank, action)
                if action.member_ref not in nodes:
                    _fail("action references a missing member", "fusion_plans")
                if len(action.expected_route) > 2:
                    _unsupported(
                        "Swizzle GEMM_RS common IR2 supports direct routes only",
                        f"fusion_plans[{plan_index}].rank_programs[{program_index}]"
                        f".actions[{action_index_in_program}].expected_route",
                    )
        for action_id, (_rank, action) in action_index.items():
            if not set(action.source_action.deps).issubset(action_index):
                _fail(
                    "action dependency is absent from its Swizzle plan",
                    f"fusion_plans[{plan_index}].actions[{action_id}].deps",
                )
        plan_actions[plan.id] = action_index

    bindings, flow_by_endpoint = _flow_bindings(
        ir1, fusion_plans, plan_actions, placements, values
    )
    tasks_by_die: dict[int, list[SemanticTask]] = {
        die.id: [] for die in ir1.fabric.dies
    }
    task_actions_by_die: dict[
        int, list[tuple[str, int, SwizzleBoundAction]]
    ] = {die.id: [] for die in ir1.fabric.dies}
    plan_task_ids_by_die: dict[tuple[str, int], list[str]] = {}
    standalone_task_ids_by_die: dict[tuple[str, int], list[str]] = {}
    standalone_flows_by_die: dict[int, list[SemanticFlow]] = {
        die.id: [] for die in ir1.fabric.dies
    }
    ordinary_ids_by_die: dict[int, list[str]] = {
        die.id: [] for die in ir1.fabric.dies
    }
    ordinary_regions_by_die: dict[int, list[IntraDieRegion]] = {
        die.id: [] for die in ir1.fabric.dies
    }

    for plan_index, plan in enumerate(fusion_plans):
        action_index = plan_actions[plan.id]
        for program_index, program in enumerate(plan.rank_programs):
            placement = placements[plan.id][program.rank]
            die_id = placement.die_id
            region_id = _region_id(plan.id, die_id)
            for action_index_in_program, action in enumerate(program.actions):
                path = (
                    f"fusion_plans[{plan_index}].rank_programs[{program_index}]"
                    f".actions[{action_index_in_program}]"
                )
                source = action.source_action
                kind = SemanticTaskKind(action.fusion_kind.value)
                if action.chunk_origin is None and action.fusion_kind in (
                    FusionActionKind.WAIT,
                    FusionActionKind.BARRIER,
                ):
                    chunk_slice = None
                    dtype = DType.FP16
                    payload_bytes = 0
                else:
                    chunk_slice, dtype, payload_bytes = _chunk_slice(
                        action, values, path=path
                    )
                flow = flow_by_endpoint.get((plan.id, program.rank, source.id))
                if action.fusion_kind in (
                    FusionActionKind.SEND,
                    FusionActionKind.RECV,
                ):
                    if flow is None:
                        _fail("transport action is missing its semantic flow", path)
                    source_rank = flow.source_rank
                    destination_rank = flow.destination_rank
                    task_slice = flow.tensor_slice
                    task_bytes = flow.bytes
                    task_dtype: DType | None = flow.dtype
                    flow_id: str | None = flow.flow_id
                else:
                    if flow is not None:
                        _fail("non-transport action cannot own a semantic flow", path)
                    source_rank = None
                    destination_rank = None
                    flow_id = None
                    if action.fusion_kind in (
                        FusionActionKind.WAIT,
                        FusionActionKind.BARRIER,
                    ):
                        task_slice = None
                        task_bytes = 0
                        task_dtype = None
                    else:
                        task_slice = chunk_slice
                        task_bytes = (
                            payload_bytes
                            if action.fusion_kind
                            in (
                                FusionActionKind.REDUCE,
                                FusionActionKind.LOCAL_COPY,
                            )
                            else 0
                        )
                        task_dtype = dtype

                compute = (
                    _compute_contract(action, placement, path=path)
                    if action.fusion_kind is FusionActionKind.COMP
                    else None
                )
                reduction = (
                    _reduction_contract(
                        action, action_index, dtype, path=path
                    )
                    if action.fusion_kind is FusionActionKind.REDUCE
                    else None
                )
                materialized_reads = tuple(
                    _bound_value_id(
                        plan, program.rank, action, ref, set(values), write=False
                    )
                    for ref in source.input_refs
                )
                materialized_writes = tuple(
                    _bound_value_id(
                        plan, program.rank, action, ref, set(values), write=True
                    )
                    for ref in source.output_refs
                )
                if action.fusion_kind is FusionActionKind.LOCAL_COPY:
                    assert action.chunk_origin is not None
                    logical_output = plan.decision.problem.collective.output.value_ref
                    task_slice = TensorSlice(
                        logical_output, action.chunk_origin.logical_offset,
                        action.chunk_origin.logical_shape,
                    )
                    task_dtype = values[logical_output].dtype
                if compute is not None:
                    tile = compute.tile
                    assert tile is not None
                    compute = replace(
                        compute,
                        inputs=tuple(
                            replace(operand, value_id=materialized_reads[index])
                            for index, operand in enumerate(compute.inputs)
                        ),
                        outputs=tuple(
                            replace(operand, value_id=materialized_writes[index])
                            for index, operand in enumerate(compute.outputs)
                        ),
                        tile=replace(
                            tile,
                            input_slices=tuple(
                                replace(binding, operand_id=materialized_reads[index])
                                for index, binding in enumerate(tile.input_slices)
                            ),
                            output_slices=tuple(
                                replace(binding, operand_id=materialized_writes[index])
                                for index, binding in enumerate(tile.output_slices)
                            ),
                        ),
                    )
                local_dependencies: list[str] = []
                for dependency in source.deps:
                    dependency_rank, dependency_action = action_index[dependency]
                    if dependency_rank == program.rank:
                        local_dependencies.append(
                            _task_id(dependency_action.source_action.id)
                        )
                    elif action.fusion_kind is not FusionActionKind.RECV:
                        _unsupported(
                            "only RECV may carry a cross-rank Swizzle dependency",
                            f"{path}.source_action.deps",
                        )
                task = SemanticTask(
                    id=_task_id(source.id),
                    kind=kind,
                    origin_ref=_origin(plan.id, program.rank, source.id),
                    region_id=region_id,
                    op_kind=nodes[action.member_ref].kind,
                    member_id=action.member_ref,
                    flow_id=flow_id,
                    chunk_id=source.chunk_index,
                    collective_step=None,
                    source_rank=source_rank,
                    destination_rank=destination_rank,
                    tensor_slice=task_slice,
                    bytes=task_bytes,
                    dtype=task_dtype,
                    shape=(
                        action.chunk_origin.logical_shape
                        if action.chunk_origin is not None else ()
                    ),
                    read_values=materialized_reads,
                    write_values=materialized_writes,
                    compute=compute,
                    reduction=reduction,
                    sync=action.sync,
                    deps=tuple(local_dependencies),
                )
                tasks_by_die[die_id].append(task)
                task_actions_by_die[die_id].append(
                    (plan.id, program.rank, action)
                )
                plan_task_ids_by_die.setdefault((plan.id, die_id), []).append(
                    task.id
                )

    # Direct AllGather standalone actions retain their exact published action
    # identity; Swizzle output producers are added only as graph dependencies.
    for plan in standalone_plans:
        group = groups[plan.group_ref]
        rank_die = {item.rank: item.die_id for item in group.placements}
        chunks = {item.id: item for item in plan.chunk_slices}
        routes = {route.id: route for route in group.embedding.routes}
        send_origin = {
            action.logical_channel: StandaloneNodeOrigin(
                OriginKind.STANDALONE_COLLECTIVE, plan.id, program.rank, action.id
            )
            for program in plan.rank_programs for action in program.actions
            if action.kind is FusionActionKind.SEND
        }
        for program in plan.rank_programs:
            die_id = rank_die[program.rank]
            region_id = f"region.standalone.{plan.id}.die.{die_id}"
            local_ids = {action.id: _task_id(action.id) for action in program.actions}
            for action in program.actions:
                chunk = chunks.get(action.slice_ref or "")
                tensor_slice = (
                    TensorSlice(chunk.value_id, chunk.offset, chunk.shape)
                    if chunk is not None else None
                )
                source_rank = destination_rank = None
                flow_id = None
                if action.kind in (FusionActionKind.SEND, FusionActionKind.RECV):
                    assert action.peer_rank is not None and action.logical_channel is not None
                    source_rank, destination_rank = (
                        (program.rank, action.peer_rank)
                        if action.kind is FusionActionKind.SEND
                        else (action.peer_rank, program.rank)
                    )
                    origin = send_origin[action.logical_channel]
                    flow_id = canonical_semantic_flow_id(origin, action.logical_channel)
                    route = routes[action.logical_channel] if action.logical_channel in routes else next(
                        item for item in group.embedding.routes
                        if item.source_rank == source_rank
                        and item.destination_rank == destination_rank
                        and item.die_path == action.expected_route
                    )
                    standalone_flows_by_die[die_id].append(SemanticFlow(
                        id=flow_id, logical_channel=action.logical_channel,
                        pair_route_ref=route.id, source_rank=source_rank,
                        destination_rank=destination_rank,
                        source_die=action.expected_route[0],
                        destination_die=action.expected_route[-1],
                        die_path=action.expected_route,
                        tensor_slice=tensor_slice, bytes=action.bytes,
                        dtype=action.dtype, task_ids=(_task_id(action.id),),
                    ))
                deps = [local_ids[item] for item in action.deps]
                task = SemanticTask(
                    id=_task_id(action.id), kind=SemanticTaskKind(action.kind.value),
                    origin_ref=StandaloneNodeOrigin(
                        OriginKind.STANDALONE_COLLECTIVE, plan.id,
                        program.rank, action.id,
                    ),
                    region_id=region_id, op_kind=OpKind.COLLECTIVE,
                    member_id=action.member_id, flow_id=flow_id,
                    chunk_id=action.chunk_id,
                    collective_step=action.collective_step,
                    source_rank=source_rank, destination_rank=destination_rank,
                    tensor_slice=tensor_slice, bytes=action.bytes,
                    dtype=action.dtype, shape=(chunk.shape if chunk else ()),
                    read_values=action.reads, write_values=action.writes,
                    compute=action.compute, reduction=action.reduction,
                    sync=action.sync, deps=tuple(dict.fromkeys(deps)),
                )
                tasks_by_die[die_id].append(task)
                standalone_task_ids_by_die.setdefault((plan.id, die_id), []).append(task.id)

    # Reuse the legacy ordinary-node carriers verbatim; only the merge and
    # cross-unit edge closure are Swizzle-specific here.  The local import is
    # safe after module initialization and avoids duplicating compute ABI rules.
    from .naive_project_to_ir2 import (
        _ordinary_compute,
        _ordinary_region_id,
        _ordinary_task_id,
    )

    for node in ordinary_nodes:
        group = groups[node.execution_group_ref]
        for placement in group.placements:
            task_id = _ordinary_task_id(node.id, placement.rank)
            region_id = _ordinary_region_id(node.id, placement.rank)
            tasks_by_die[placement.die_id].append(
                SemanticTask(
                    id=task_id, kind=SemanticTaskKind.COMP,
                    origin_ref=OrdinaryNodeOrigin(
                        OriginKind.ORDINARY, node.id, placement.rank
                    ),
                    region_id=region_id, op_kind=node.kind, member_id=node.id,
                    flow_id=None, chunk_id=None, collective_step=None,
                    source_rank=None, destination_rank=None, tensor_slice=None,
                    bytes=0, dtype=None, shape=(), read_values=node.inputs,
                    write_values=node.outputs, compute=_ordinary_compute(node),
                    reduction=None, sync=None, deps=(),
                )
            )
            ordinary_regions_by_die[placement.die_id].append(
                IntraDieRegion(
                    id=region_id, fusion_plan_id=None,
                    standalone_collective_plan_id=None,
                    lowering=RegionLowering.JSON_COARSE, task_ids=(task_id,),
                )
            )
            ordinary_ids_by_die[placement.die_id].append(node.id)

    def coverage(node_id: str) -> tuple[str, str]:
        if node_id in fusion_by_member:
            return ("fusion", fusion_by_member[node_id].id)
        if node_id in standalone_by_node:
            return ("standalone", standalone_by_node[node_id].id)
        if nodes[node_id].kind is OpKind.COLLECTIVE:
            return ("uncovered_collective", node_id)
        return ("ordinary", node_id)

    def belongs(task: SemanticTask, node_id: str, whole: bool) -> bool:
        category, unit_id = coverage(node_id)
        origin = task.origin_ref
        if category == "ordinary":
            return (
                isinstance(origin, OrdinaryNodeOrigin)
                and origin.op_id == unit_id
            )
        if category == "fusion":
            return (
                isinstance(origin, SwizzleNodeOrigin)
                and origin.plan_id == unit_id
                and (whole or task.member_id == node_id)
            )
        return (
            isinstance(origin, StandaloneNodeOrigin)
            and origin.collective_plan_id == unit_id
            and (whole or task.member_id == node_id)
        )

    def reads_source(task: SemanticTask, value_id: str) -> bool:
        if value_id in task.read_values:
            return True
        return (
            task.kind is SemanticTaskKind.COMP
            and task.compute is not None
            and task.compute.tile is not None
            and any(
                binding.source_value_id == value_id
                and binding.operand_id in task.read_values
                for binding in task.compute.tile.input_slices
            )
        )

    # Close every same-die IR-1 edge between independently projected units.
    for die_id, local_tasks in tasks_by_die.items():
        deps_by_task = {task.id: list(task.deps) for task in local_tasks}
        for edge in sorted(ir1.edges, key=lambda item: item.id):
            source_coverage = coverage(edge.source_node)
            destination_coverage = coverage(edge.destination_node)
            if (
                "uncovered_collective"
                in (source_coverage[0], destination_coverage[0])
            ):
                continue
            if source_coverage == destination_coverage:
                continue
            whole = edge.kind is EdgeKind.CONTROL
            entries = tuple(
                task
                for task in local_tasks
                if belongs(task, edge.destination_node, whole)
                and (whole or reads_source(task, edge.value_id))
            )
            source_category = coverage(edge.source_node)[0]
            if whole:
                unit_tasks = tuple(
                    task
                    for task in local_tasks
                    if belongs(task, edge.source_node, True)
                )
                completions = tuple(
                    task
                    for task in unit_tasks
                    if not any(
                        task.id in candidate.deps for candidate in unit_tasks
                    )
                )
            elif source_category == "standalone":
                completions = tuple(
                    task
                    for task in local_tasks
                    if belongs(task, edge.source_node, False)
                    and task.kind is SemanticTaskKind.BARRIER
                )
            else:
                completions = tuple(
                    task
                    for task in local_tasks
                    if belongs(task, edge.source_node, False)
                    and edge.value_id in task.write_values
                )
            if not entries or not completions:
                _fail(
                    "IR-1 edge cannot map to exact local unit endpoints",
                    f"ir1.edges[{edge.id}]",
                )
            for entry in entries:
                deps_by_task[entry.id].extend(
                    completion.id for completion in completions
                )
        tasks_by_die[die_id] = [
            replace(
                task, deps=tuple(dict.fromkeys(deps_by_task[task.id]))
            )
            for task in local_tasks
        ]

    source_order = {node.id: index for index, node in enumerate(ir1.nodes)}
    fusion_anchor = {
        plan.id: source_order[skeletons[plan.fused_op_id].member_node_ids[0]]
        for plan in fusion_plans
    }
    standalone_anchor = {
        plan.id: source_order[plan.op_id] for plan in standalone_plans
    }
    for die_id, local_tasks in tasks_by_die.items():
        original_position = {task.id: index for index, task in enumerate(local_tasks)}

        def unit_anchor(task: SemanticTask) -> int:
            origin = task.origin_ref
            if isinstance(origin, SwizzleNodeOrigin):
                return fusion_anchor[origin.plan_id]
            if isinstance(origin, StandaloneNodeOrigin):
                return standalone_anchor[origin.collective_plan_id]
            assert isinstance(origin, OrdinaryNodeOrigin)
            return source_order[origin.op_id]

        tasks_by_die[die_id] = sorted(
            local_tasks, key=lambda task: (unit_anchor(task), original_position[task.id])
        )

    flows_by_die: dict[int, list[SemanticFlow]] = {
        die.id: [] for die in ir1.fabric.dies
    }
    for binding in bindings:
        for die_id, task_id in (
            (binding.source_die, _task_id(binding.send_action_id)),
            (binding.destination_die, _task_id(binding.recv_action_id)),
        ):
            flows_by_die[die_id].append(
                SemanticFlow(
                    id=binding.flow_id,
                    logical_channel=binding.logical_channel,
                    pair_route_ref=binding.pair_route_ref,
                    source_rank=binding.source_rank,
                    destination_rank=binding.destination_rank,
                    source_die=binding.source_die,
                    destination_die=binding.destination_die,
                    die_path=binding.die_path,
                    tensor_slice=binding.tensor_slice,
                    bytes=binding.bytes,
                    dtype=binding.dtype,
                    task_ids=(task_id,),
                )
            )
    for die_id, standalone_flows in standalone_flows_by_die.items():
        flows_by_die[die_id].extend(standalone_flows)

    ir1_value_ids = set(values)
    dags: list[IntraDieDAG] = []
    for die in ir1.fabric.dies:
        tasks = tuple(tasks_by_die[die.id])
        task_action_rows = task_actions_by_die[die.id]
        referenced_ir1_values = {
            value_id
            for task in tasks
            for value_id in task.read_values + task.write_values
            if value_id in ir1_value_ids
        }
        local_values: list[IntraDieValue] = []
        for value in ir1.values:
            if value.id not in referenced_ir1_values:
                continue
            writers = tuple(
                sorted(
                    (task for task in tasks if value.id in task.write_values),
                    key=lambda task: (
                        task.tensor_slice.offset if task.tensor_slice else (),
                        task.tensor_slice.shape if task.tensor_slice else (),
                        task.id,
                    ),
                )
            )
            local_values.append(
                IntraDieValue(
                    id=value.id,
                    origin_value_id=value.id,
                    shape=value.shape,
                    dtype=value.dtype,
                    logical_layout=value.logical_layout,
                    sharding=value.sharding,
                    alias_set=value.alias_set,
                    producer_tasks=tuple(task.id for task in writers),
                    consumer_tasks=tuple(
                        task.id for task in tasks if value.id in task.read_values
                    ),
                )
            )

        origin_rows: dict[
            tuple[str, int, str], list[SwizzleValueOrigin]
        ] = {}
        temp_order: list[tuple[str, int, str]] = []
        plan_index_by_id = {plan.id: plan for plan in fusion_plans}
        for plan_id, rank, action in task_action_rows:
            for value_origin in action.value_origins:
                materialized_ref = _bound_value_id(
                    plan_index_by_id[plan_id], rank, action,
                    value_origin.value_ref, ir1_value_ids,
                    write=value_origin.value_ref in action.source_action.output_refs,
                )
                if materialized_ref in ir1_value_ids:
                    continue
                key = (plan_id, rank, materialized_ref)
                if key not in origin_rows:
                    origin_rows[key] = []
                    temp_order.append(key)
                materialized_origin = replace(
                    value_origin, value_ref=materialized_ref
                )
                if materialized_origin not in origin_rows[key]:
                    origin_rows[key].append(materialized_origin)
        if len({key[2] for key in temp_order}) != len(temp_order):
            _unsupported(
                "rank-local Swizzle temporary ids must be unique within one die",
                f"dags[die={die.id}].swizzle_values",
            )
        swizzle_values = tuple(
            SwizzleIntraDieValue(
                id=value_id,
                plan_id=plan_id,
                rank=rank,
                origins=tuple(origin_rows[(plan_id, rank, value_id)]),
                producer_tasks=tuple(
                    task.id for task in tasks if value_id in task.write_values
                ),
                consumer_tasks=tuple(
                    task.id for task in tasks if value_id in task.read_values
                ),
            )
            for plan_id, rank, value_id in temp_order
        )
        local_plan_ids = tuple(
            plan.id
            for plan in fusion_plans
            if (plan.id, die.id) in plan_task_ids_by_die
        )
        task_position = {task.id: index for index, task in enumerate(tasks)}
        flows_by_die[die.id] = sorted(
            flows_by_die[die.id],
            key=lambda flow: task_position[flow.task_ids[0]],
        )
        regions = tuple(
            IntraDieRegion(
                id=_region_id(plan_id, die.id),
                fusion_plan_id=plan_id,
                standalone_collective_plan_id=None,
                lowering=RegionLowering.ISA_REGION,
                task_ids=tuple(plan_task_ids_by_die[(plan_id, die.id)]),
            )
            for plan_id in local_plan_ids
        ) + tuple(
            IntraDieRegion(
                id=f"region.standalone.{plan.id}.die.{die.id}",
                fusion_plan_id=None,
                standalone_collective_plan_id=plan.id,
                lowering=RegionLowering.STRICT_ACTIONS,
                task_ids=tuple(standalone_task_ids_by_die[(plan.id, die.id)]),
            )
            for plan in standalone_plans
            if (plan.id, die.id) in standalone_task_ids_by_die
        ) + tuple(ordinary_regions_by_die[die.id])
        regions = tuple(
            sorted(
                regions,
                key=lambda region: min(
                    task_position[task_id] for task_id in region.task_ids
                ),
            )
        )
        dags.append(
            IntraDieDAG.create(
                producer_pass="project_to_ir2",
                source_ir1_id=ir1.id,
                die_id=die.id,
                fusion_plan_ids=local_plan_ids,
                standalone_collective_plan_ids=tuple(
                    plan.id for plan in standalone_plans
                    if (plan.id, die.id) in standalone_task_ids_by_die
                ),
                ordinary_node_ids=tuple(ordinary_ids_by_die[die.id]),
                tasks=tasks,
                values=tuple(local_values),
                flows=tuple(flows_by_die[die.id]),
                regions=regions,
                source_state_manifest_id=(
                    ir1.persistent_state_manifest.id
                    if ir1.persistent_state_manifest is not None
                    else None
                ),
                state_access_ids=(),
                state_staging_values=(),
                state_transfer_ids=(),
                swizzle_values=swizzle_values,
            )
        )

    result = IR2ProjectionResult.create(
        producer_pass="project_to_ir2",
        source_ir1_id=ir1.id,
        fusion_plan_ids=tuple(plan.id for plan in fusion_plans),
        standalone_collective_plan_ids=tuple(plan.id for plan in standalone_plans),
        dags=tuple(dags),
        source_state_manifest_id=(
            ir1.persistent_state_manifest.id
            if ir1.persistent_state_manifest is not None
            else None
        ),
        state_transfers=(),
    )
    result.validate_against(ir1, fusion_plans, standalone_plans)
    return result


__all__ = ["project_swizzle_gemm_rs_to_ir2"]
