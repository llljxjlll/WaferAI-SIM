"""Canonical direct inter-die policies for the Dense naive MVP."""

from __future__ import annotations

import math

from ..errors import SchemaError
from ..schema.action import (
    BarrierContract,
    BarrierScope,
    ChunkDim,
    ChunkSlice,
    CollectiveAlgorithm,
    ComputeContract,
    ComputeOperand,
    ComputeOperandSlice,
    ComputeTileBinding,
    ConsumerLayoutBinding,
    FusionAction,
    FusionActionKind,
    FusionPlan,
    InversePermutationEntry,
    PermutationEntry,
    RankProgram,
    ReductionContract,
    StandaloneCollectivePlan,
    SyncContract,
)
from ..schema.common import DType, MeshAxisName, ProfileKey, RoundingMode, TensorValue
from ..schema.ir0 import (
    CollectiveKind,
    CollectiveWorkload,
    FusionImpl,
    GemmPartition,
    GemmWorkload,
    OpKind,
    ReduceOp,
)
from ..schema.ir1 import FusedOpSkeleton, IR1, PhysicalGroup, PhysicalNode
from ..schema.persistent_state import (
    StateKind,
    canonical_state_staging_value_id,
)


def _fail(message: str, path: str) -> None:
    raise SchemaError(message, path=path)


def _element_bytes(dtype: DType, *, path: str) -> int:
    if dtype is DType.FP16:
        return 2
    if dtype is DType.FP32:
        return 4
    _fail("unsupported tensor dtype", path)
    raise AssertionError("unreachable")


def _validate_inputs(
    ir1: IR1,
    profile: ProfileKey,
    owner_instance_id: str,
) -> None:
    if type(ir1) is not IR1:
        _fail("must be an IR1", "ir1")
    if type(profile) is not ProfileKey:
        _fail("must be a ProfileKey", "profile")
    ir1.validate("ir1")
    profile.validate("profile")
    legacy_profile = not ir1.instance_profiles
    if legacy_profile:
        expected_profile = ir1.profile
    else:
        matches = tuple(
            binding.profile
            for binding in ir1.instance_profiles
            if binding.instance_ref == owner_instance_id
        )
        if len(matches) != 1:
            _fail(
                "owner instance must have exactly one profile binding",
                "profile",
            )
        expected_profile = matches[0]
    if profile != expected_profile:
        _fail(
            (
                "must exactly match IR1 profile"
                if legacy_profile
                else "must exactly match owner instance profile"
            ),
            "profile",
        )


def _canonical_ranks(group: PhysicalGroup, *, path: str) -> tuple[int, ...]:
    ranks = tuple(placement.rank for placement in group.placements)
    expected = tuple(range(len(group.placements)))
    if ranks != expected:
        _fail("group placements must be canonically ordered by rank", path)
    return ranks


def _tp_coordinate(
    group: PhysicalGroup,
    rank: int,
    *,
    path: str,
) -> int:
    if group.axis is not MeshAxisName.TP or len(group.logical_shape) != 1:
        _fail("fused row-GEMM requires a one-dimensional TP group", path)
    placement = next(
        (item for item in group.placements if item.rank == rank),
        None,
    )
    if placement is None:
        _fail("rank has no physical placement", path)
    if len(placement.logical_coord) != 1:
        _fail("TP rank requires one logical coordinate", path)
    return placement.logical_coord[0]


def _rhs_operand_id(
    ir1: IR1,
    gemm: PhysicalNode,
    rank: int,
    *,
    region_id: str,
) -> str:
    if ir1.persistent_state_manifest is None:
        return f"temp.{region_id}.rank.{rank}.b"
    declarations = {
        item.id: item for item in ir1.persistent_state_manifest.declarations
    }
    matching = tuple(
        access
        for access in ir1.state_accesses
        if access.node_ref == gemm.id
        and access.rank == rank
        and declarations[access.state_ref].identity.kind is StateKind.PARAMETER
        and declarations[access.state_ref].identity.tensor_ref == gemm.inputs[1]
    )
    if len(matching) != 1:
        _fail(
            "stateful fused GEMM requires exactly one rank-local RHS parameter access",
            "fused_op.state_accesses",
        )
    return canonical_state_staging_value_id(matching[0].id)


def _route_index(
    group: PhysicalGroup,
    *,
    path: str,
) -> dict[tuple[int, int], tuple[int, ...]]:
    result: dict[tuple[int, int], tuple[int, ...]] = {}
    for route in group.embedding.routes:
        key = (route.source_rank, route.destination_rank)
        if key in result:
            _fail("group contains duplicate ordered-pair routes", path)
        result[key] = route.die_path
    ranks = _canonical_ranks(group, path=path)
    expected = {(source, destination) for source in ranks for destination in ranks if source != destination}
    if set(result) != expected:
        _fail("group must contain exactly one route for every ordered rank pair", path)
    return result


def _chunk_slices(
    *,
    region_id: str,
    value: TensorValue,
    rank_count: int,
    axis: int,
    path: str,
) -> tuple[ChunkSlice, ...]:
    if axis not in (0, 1):
        _fail("canonical action schema supports only M/N chunk axes", path)
    if len(value.shape) <= axis or value.shape[axis] % rank_count:
        _fail("chunk axis must be evenly divisible by group size", path)
    extent = value.shape[axis] // rank_count
    element_bytes = _element_bytes(value.dtype, path=f"{path}.dtype")
    result = []
    for chunk_id in range(rank_count):
        offset = tuple(
            chunk_id * extent if index == axis else 0
            for index in range(len(value.shape))
        )
        shape = tuple(
            extent if index == axis else dimension
            for index, dimension in enumerate(value.shape)
        )
        result.append(
            ChunkSlice(
                id=f"{region_id}.chunk.{chunk_id}",
                chunk_id=chunk_id,
                value_id=value.id,
                offset=offset,
                shape=shape,
                bytes=math.prod(shape) * element_bytes,
                owner_rank=chunk_id,
            )
        )
    return tuple(result)


def _sync(action_id: str) -> SyncContract:
    return SyncContract(
        completion_event=f"event.{action_id}",
        wait_event=None,
        barrier=None,
    )


def _fusion_action(
    action_id: str,
    kind: FusionActionKind,
    *,
    member_id: str,
    chunk_id: int | None,
    collective_step: int | None,
    peer_rank: int | None = None,
    expected_route: tuple[int, ...] = (),
    slice_ref: str | None = None,
    bytes: int = 0,
    dtype: DType | None = None,
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
    logical_channel: str | None = None,
    compute: ComputeContract | None = None,
    reduction: ReductionContract | None = None,
    sync: SyncContract | None = None,
    deps: tuple[str, ...] = (),
) -> FusionAction:
    return FusionAction(
        id=action_id,
        kind=kind,
        member_id=member_id,
        chunk_id=chunk_id,
        collective_step=collective_step,
        peer_rank=peer_rank,
        expected_route=expected_route,
        slice_ref=slice_ref,
        bytes=bytes,
        dtype=dtype,
        reads=reads,
        writes=writes,
        logical_channel=logical_channel,
        compute=compute,
        reduction=reduction,
        sync=sync or _sync(action_id),
        deps=deps,
    )


def _compute_contract(
    *,
    region_id: str,
    rank: int,
    rhs_operand_id: str,
    rhs_k_offset: int,
    chunk: ChunkSlice,
    gemm: PhysicalNode,
) -> tuple[ComputeContract, tuple[str, str], str]:
    if type(gemm.workload) is not GemmWorkload:
        _fail("fused compute member must carry GemmWorkload", "fused_op")
    full_m, full_n, full_k = gemm.workload.logical_shape
    rank_m, rank_n, rank_k = gemm.workload.rank_shape
    if len(gemm.inputs) != 2 or len(gemm.outputs) != 1:
        _fail("chunk-local GEMM requires two inputs and one output", "fused_op")
    if chunk.shape[0] > full_m or rank_m != full_m:
        _fail("GEMM workload is incompatible with canonical M chunks", "fused_op")

    prefix = f"temp.{region_id}.chunk.{chunk.chunk_id}.rank.{rank}"
    input_ids = (
        f"{prefix}.a",
        rhs_operand_id,
    )
    output_id = f"{prefix}.partial"
    workload = GemmWorkload(
        logical_shape=(chunk.shape[0], full_n, full_k),
        rank_shape=(chunk.shape[0], rank_n, rank_k),
        partition=gemm.workload.partition,
        dtype=gemm.workload.dtype,
    )
    compute = ComputeContract(
        op_kind=gemm.kind,
        workload=workload,
        math=gemm.math,
        effects=gemm.effects,
        impl_ref=gemm.impl_ref,
        inputs=(
            ComputeOperand(input_ids[0], "lhs"),
            ComputeOperand(input_ids[1], "rhs"),
        ),
        outputs=(ComputeOperand(output_id, "partial"),),
        tile=ComputeTileBinding(
            origin_workload=gemm.workload,
            input_slices=(
                ComputeOperandSlice(
                    input_ids[0],
                    gemm.inputs[0],
                    (chunk.offset[0], 0),
                    (chunk.shape[0], full_k),
                ),
                ComputeOperandSlice(
                    input_ids[1],
                    gemm.inputs[1],
                    (rhs_k_offset, 0),
                    (rank_k, full_n),
                ),
            ),
            output_slices=(
                ComputeOperandSlice(
                    output_id,
                    gemm.outputs[0],
                    chunk.offset,
                    chunk.shape,
                ),
            ),
        ),
    )
    return compute, input_ids, output_id


class NaiveInterDiePolicy:
    """Plan canonical DIRECT row-GEMM plus SUM-ReduceScatter actions."""

    def plan(
        self,
        ir1: IR1,
        fused_op: FusedOpSkeleton,
        profile: ProfileKey,
    ) -> FusionPlan:
        if type(fused_op) is not FusedOpSkeleton:
            _fail("must be a FusedOpSkeleton", "fused_op")
        _validate_inputs(ir1, profile, fused_op.instance_id)
        bound = next(
            (item for item in ir1.fused_op_skeletons if item.id == fused_op.id),
            None,
        )
        if bound != fused_op:
            _fail("must exactly reference a skeleton in IR1", "fused_op")
        if fused_op.impl is not FusionImpl.NONE:
            _fail("fusion partition must remain unplanned", "fused_op.impl")
        if len(fused_op.member_node_ids) != 2:
            _fail("DIRECT fused planning requires exactly two members", "fused_op")

        nodes = {node.id: node for node in ir1.nodes}
        values = {value.id: value for value in ir1.values}
        gemm = nodes[fused_op.member_node_ids[0]]
        reduce_scatter = nodes[fused_op.member_node_ids[1]]
        if (
            gemm.kind is OpKind.GEMM
            and type(gemm.workload) is GemmWorkload
            and gemm.workload.partition
            is GemmPartition.SEQUENCE_PARALLEL_REPLICATED_WEIGHT
        ):
            _fail(
                "sequence-parallel replicated-weight GEMM cannot enter inter-die fusion planning",
                "fused_op",
            )
        if (
            gemm.kind is not OpKind.GEMM
            or type(gemm.workload) is not GemmWorkload
            or gemm.workload.partition is not GemmPartition.ROW_PARALLEL
        ):
            _fail("first fused member must be a row-parallel GEMM", "fused_op")
        if (
            reduce_scatter.kind is not OpKind.COLLECTIVE
            or type(reduce_scatter.workload) is not CollectiveWorkload
            or reduce_scatter.workload.collective is not CollectiveKind.REDUCE_SCATTER
            or reduce_scatter.workload.reduce_op is not ReduceOp.SUM
        ):
            _fail("second fused member must be SUM ReduceScatter", "fused_op")
        if (
            gemm.instance_id != reduce_scatter.instance_id
            or gemm.stage != reduce_scatter.stage
            or gemm.phase is not reduce_scatter.phase
            or gemm.execution_group_ref != reduce_scatter.execution_group_ref
        ):
            _fail("fused members must share execution scope", "fused_op")
        if (
            fused_op.boundary_inputs != gemm.inputs
            or fused_op.boundary_outputs != reduce_scatter.outputs
            or len(fused_op.boundary_outputs) != 1
        ):
            _fail("fused boundaries must exactly match GEMM and ReduceScatter", "fused_op")

        group = next(
            (
                item
                for item in ir1.groups
                if item.id == gemm.execution_group_ref
            ),
            None,
        )
        if group is None:
            _fail("fused execution group is missing", "fused_op")
        ranks = _canonical_ranks(group, path="fused_op.group")
        routes = _route_index(group, path="fused_op.group.embedding.routes")
        output = values[fused_op.boundary_outputs[0]]
        chunks = _chunk_slices(
            region_id=fused_op.id,
            value=output,
            rank_count=len(ranks),
            axis=0,
            path="fused_op.boundary_outputs",
        )
        if reduce_scatter.workload.scatter_tensor_axis != 0:
            _fail("naive fused ReduceScatter supports only M scatter", "fused_op")

        actions: dict[int, list[FusionAction]] = {rank: [] for rank in ranks}
        rhs_operand_ids = {
            rank: _rhs_operand_id(ir1, gemm, rank, region_id=fused_op.id)
            for rank in ranks
        }
        rhs_k_offsets = {
            rank: _tp_coordinate(
                group, rank, path="fused_op.group.placements"
            )
            * gemm.workload.rank_shape[2]
            for rank in ranks
        }
        for chunk in chunks:
            owner = chunk.owner_rank
            comps: dict[int, FusionAction] = {}
            contribution_ids: dict[int, str] = {}
            for rank in ranks:
                compute, reads, contribution_id = _compute_contract(
                    region_id=fused_op.id,
                    rank=rank,
                    rhs_operand_id=rhs_operand_ids[rank],
                    rhs_k_offset=rhs_k_offsets[rank],
                    chunk=chunk,
                    gemm=gemm,
                )
                action_id = (
                    f"action.{fused_op.id}.chunk.{chunk.chunk_id}.rank.{rank}.comp"
                )
                comp = _fusion_action(
                    action_id,
                    FusionActionKind.COMP,
                    member_id=gemm.id,
                    chunk_id=chunk.chunk_id,
                    collective_step=None,
                    slice_ref=chunk.id,
                    bytes=chunk.bytes,
                    dtype=output.dtype,
                    reads=reads,
                    writes=(contribution_id,),
                    compute=compute,
                )
                comps[rank] = comp
                contribution_ids[rank] = contribution_id
                actions[rank].append(comp)

            remote_ids: dict[int, str] = {}
            waits: dict[int, FusionAction] = {}
            for rank in ranks:
                if rank == owner:
                    continue
                channel = (
                    f"channel.{fused_op.id}.chunk.{chunk.chunk_id}"
                    f".rank.{rank}.to.rank.{owner}"
                )
                send_id = (
                    f"action.{fused_op.id}.chunk.{chunk.chunk_id}"
                    f".rank.{rank}.send.to.{owner}"
                )
                recv_id = (
                    f"action.{fused_op.id}.chunk.{chunk.chunk_id}"
                    f".rank.{owner}.recv.from.{rank}"
                )
                remote_id = (
                    f"temp.{fused_op.id}.chunk.{chunk.chunk_id}"
                    f".rank.{owner}.recv.from.{rank}"
                )
                send = _fusion_action(
                    send_id,
                    FusionActionKind.SEND,
                    member_id=reduce_scatter.id,
                    chunk_id=chunk.chunk_id,
                    collective_step=0,
                    peer_rank=owner,
                    expected_route=routes[(rank, owner)],
                    slice_ref=chunk.id,
                    bytes=chunk.bytes,
                    dtype=output.dtype,
                    reads=(contribution_ids[rank],),
                    logical_channel=channel,
                    deps=(comps[rank].id,),
                )
                recv = _fusion_action(
                    recv_id,
                    FusionActionKind.RECV,
                    member_id=reduce_scatter.id,
                    chunk_id=chunk.chunk_id,
                    collective_step=0,
                    peer_rank=rank,
                    expected_route=routes[(rank, owner)],
                    slice_ref=chunk.id,
                    bytes=chunk.bytes,
                    dtype=output.dtype,
                    writes=(remote_id,),
                    logical_channel=channel,
                )
                wait_id = (
                    f"action.{fused_op.id}.chunk.{chunk.chunk_id}"
                    f".rank.{owner}.wait.from.{rank}"
                )
                wait = _fusion_action(
                    wait_id,
                    FusionActionKind.WAIT,
                    member_id=reduce_scatter.id,
                    chunk_id=chunk.chunk_id,
                    collective_step=0,
                    sync=SyncContract(
                        completion_event=f"event.{wait_id}",
                        wait_event=recv.sync.completion_event,
                        barrier=None,
                    ),
                    deps=(recv.id,),
                )
                actions[rank].append(send)
                remote_ids[rank] = remote_id
                waits[rank] = wait
                actions[owner].extend((recv, wait))

            reduce_id = (
                f"action.{fused_op.id}.chunk.{chunk.chunk_id}"
                f".rank.{owner}.reduce"
            )
            reduce = _fusion_action(
                reduce_id,
                FusionActionKind.REDUCE,
                member_id=reduce_scatter.id,
                chunk_id=chunk.chunk_id,
                collective_step=1,
                slice_ref=chunk.id,
                bytes=chunk.bytes,
                dtype=output.dtype,
                reads=tuple(
                    contribution_ids[rank] if rank == owner else remote_ids[rank]
                    for rank in ranks
                ),
                writes=(output.id,),
                reduction=ReductionContract(
                    reduce_op=ReduceOp.SUM,
                    input_dtype=reduce_scatter.workload.dtype,
                    accumulation_dtype=reduce_scatter.math.accumulation_dtype,
                    output_dtype=output.dtype,
                    rounding=RoundingMode.RNE,
                    input_ranks=ranks,
                ),
                deps=tuple(
                    comps[rank].id if rank == owner else waits[rank].id
                    for rank in ranks
                ),
            )
            actions[owner].append(reduce)

        bindings = tuple(
            ConsumerLayoutBinding(
                consumer_node_id=consumer,
                value_id=value_id,
                accepted_layout=reduce_scatter.workload.output_layout,
                applies_inverse_permutation=False,
            )
            for value_id in fused_op.boundary_outputs
            for consumer in values[value_id].consumers
            if consumer not in fused_op.member_node_ids
        )
        plan = FusionPlan.create(
            producer_pass="inter_die_plan",
            source_ir1_id=ir1.id,
            fused_op_id=fused_op.id,
            group_ref=group.id,
            impl=FusionImpl.NAIVE,
            profile_key=profile,
            collective_algorithm=CollectiveAlgorithm.DIRECT,
            chunk_dim=ChunkDim.M,
            chunk_count=len(chunks),
            chunk_slices=chunks,
            rank_programs=tuple(
                RankProgram(rank=rank, actions=tuple(actions[rank])) for rank in ranks
            ),
            input_layout=reduce_scatter.workload.input_layout,
            logical_output_layout=reduce_scatter.workload.output_layout,
            physical_output_layout=reduce_scatter.workload.output_layout,
            output_permutation=tuple(
                PermutationEntry(chunk_id, chunk_id, chunk_id)
                for chunk_id in ranks
            ),
            inverse_permutation=tuple(
                InversePermutationEntry(chunk_id, chunk_id, chunk_id)
                for chunk_id in ranks
            ),
            consumer_layout_bindings=bindings,
        )
        plan.validate_against(ir1)
        return plan


class DirectAllGatherPolicy:
    """Plan canonical owner fan-out DIRECT AllGather actions."""

    def plan(
        self,
        ir1: IR1,
        collective_op: PhysicalNode,
        profile: ProfileKey,
    ) -> StandaloneCollectivePlan:
        if type(collective_op) is not PhysicalNode:
            _fail("must be a PhysicalNode", "collective_op")
        _validate_inputs(ir1, profile, collective_op.instance_id)
        bound = next(
            (item for item in ir1.nodes if item.id == collective_op.id),
            None,
        )
        if bound != collective_op:
            _fail("must exactly reference a node in IR1", "collective_op")
        if (
            collective_op.kind is not OpKind.COLLECTIVE
            or type(collective_op.workload) is not CollectiveWorkload
            or collective_op.workload.collective is not CollectiveKind.ALL_GATHER
        ):
            _fail("DIRECT standalone planning supports only AllGather", "collective_op")
        if len(collective_op.inputs) != 1 or len(collective_op.outputs) != 1:
            _fail("AllGather requires one input and one output", "collective_op")

        group = next(
            (
                item
                for item in ir1.groups
                if item.id == collective_op.execution_group_ref
            ),
            None,
        )
        if group is None:
            _fail("collective execution group is missing", "collective_op")
        ranks = _canonical_ranks(group, path="collective_op.group")
        routes = _route_index(group, path="collective_op.group.embedding.routes")
        values = {value.id: value for value in ir1.values}
        output = values[collective_op.outputs[0]]
        gather_axis = collective_op.workload.gather_tensor_axis
        if gather_axis not in (0, 1):
            _fail("DIRECT AllGather supports only canonical M/N gather axes", "collective_op")
        chunk_dim = ChunkDim.M if gather_axis == 0 else ChunkDim.N
        chunks = _chunk_slices(
            region_id=collective_op.id,
            value=output,
            rank_count=len(ranks),
            axis=gather_axis,
            path="collective_op.outputs",
        )

        actions: dict[int, list[FusionAction]] = {rank: [] for rank in ranks}
        placed: dict[tuple[int, int], FusionAction] = {}
        for chunk in chunks:
            owner = chunk.owner_rank
            local_id = (
                f"action.{collective_op.id}.chunk.{chunk.chunk_id}"
                f".rank.{owner}.local_copy"
            )
            local = _fusion_action(
                local_id,
                FusionActionKind.LOCAL_COPY,
                member_id=collective_op.id,
                chunk_id=chunk.chunk_id,
                collective_step=0,
                slice_ref=chunk.id,
                bytes=chunk.bytes,
                dtype=output.dtype,
                reads=collective_op.inputs,
                writes=collective_op.outputs,
            )
            actions[owner].append(local)
            placed[(owner, chunk.chunk_id)] = local
            for rank in ranks:
                if rank == owner:
                    continue
                channel = (
                    f"channel.{collective_op.id}.chunk.{chunk.chunk_id}"
                    f".rank.{owner}.to.rank.{rank}"
                )
                send_id = (
                    f"action.{collective_op.id}.chunk.{chunk.chunk_id}"
                    f".rank.{owner}.send.to.{rank}"
                )
                recv_id = (
                    f"action.{collective_op.id}.chunk.{chunk.chunk_id}"
                    f".rank.{rank}.recv.from.{owner}"
                )
                send = _fusion_action(
                    send_id,
                    FusionActionKind.SEND,
                    member_id=collective_op.id,
                    chunk_id=chunk.chunk_id,
                    collective_step=0,
                    peer_rank=rank,
                    expected_route=routes[(owner, rank)],
                    slice_ref=chunk.id,
                    bytes=chunk.bytes,
                    dtype=output.dtype,
                    reads=collective_op.outputs,
                    logical_channel=channel,
                    deps=(local.id,),
                )
                recv = _fusion_action(
                    recv_id,
                    FusionActionKind.RECV,
                    member_id=collective_op.id,
                    chunk_id=chunk.chunk_id,
                    collective_step=0,
                    peer_rank=owner,
                    expected_route=routes[(owner, rank)],
                    slice_ref=chunk.id,
                    bytes=chunk.bytes,
                    dtype=output.dtype,
                    writes=collective_op.outputs,
                    logical_channel=channel,
                )
                actions[owner].append(send)
                actions[rank].append(recv)
                placed[(rank, chunk.chunk_id)] = recv

        barrier = BarrierContract(
            id=f"barrier.{collective_op.id}.plan",
            participant_ranks=ranks,
            arrival_count=len(ranks),
            scope=BarrierScope.PLAN,
        )
        for rank in ranks:
            barrier_id = f"action.{collective_op.id}.rank.{rank}.barrier.plan"
            actions[rank].append(
                _fusion_action(
                    barrier_id,
                    FusionActionKind.BARRIER,
                    member_id=collective_op.id,
                    chunk_id=None,
                    collective_step=None,
                    sync=SyncContract(
                        completion_event=f"event.{barrier_id}",
                        wait_event=None,
                        barrier=barrier,
                    ),
                    deps=tuple(
                        placed[(rank, chunk.chunk_id)].id for chunk in chunks
                    ),
                )
            )

        plan = StandaloneCollectivePlan.create(
            producer_pass="inter_die_plan",
            source_ir1_id=ir1.id,
            op_id=collective_op.id,
            algorithm=CollectiveAlgorithm.DIRECT,
            group_ref=group.id,
            profile_key=profile,
            chunk_dim=chunk_dim,
            chunk_slices=chunks,
            rank_programs=tuple(
                RankProgram(rank=rank, actions=tuple(actions[rank])) for rank in ranks
            ),
        )
        plan.validate_against(ir1)
        return plan


__all__ = ["DirectAllGatherPolicy", "NaiveInterDiePolicy"]
