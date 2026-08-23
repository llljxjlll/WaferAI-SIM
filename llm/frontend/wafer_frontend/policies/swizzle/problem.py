"""Production construction of a typed :class:`SwizzleProblem` from IR-1."""

from __future__ import annotations


from ...errors import SchemaError
from ...schema.common import MeshAxisName, TensorValue
from ...schema.ir0 import (
    CollectiveKind,
    CollectiveWorkload,
    FusionPattern,
    GemmWorkload,
    GemmPartition,
    OpKind,
)
from ...schema.ir1 import FusedOpSkeleton, IR1, PhysicalNode
from ...schema.swizzle import (
    SwizzleCollectiveDescriptor,
    SwizzleCollectivePosition,
    SwizzleConstraints,
    SwizzleGemmDescriptor,
    SwizzleGroupView,
    SwizzleHardwareProfile,
    SwizzleProblem,
    SwizzleRankPlacement,
    SwizzleRouteView,
    SwizzleTensorAxisRole,
    SwizzleTensorView,
)
from .topology import SwizzleTopologyView, build_topology_view


def _fail(message: str, path: str) -> None:
    raise SchemaError(message, path=path)


def _tensor_view(
    value: TensorValue,
    *,
    shape: tuple[int, ...],
    roles: tuple[SwizzleTensorAxisRole, ...],
) -> SwizzleTensorView:
    if len(shape) != len(roles):
        _fail("shape and role ranks differ", "swizzle_problem.tensor")
    sharding = value.sharding
    if len(sharding.dim_map) == len(shape):
        dim_map = sharding.dim_map
    else:
        # The descriptor can intentionally express a rank-local collective
        # shard while TensorValue retains its logical global shape.
        dim_map = (None,) * len(shape)
    return SwizzleTensorView(
        value_ref=value.id,
        shape=shape,
        layout=value.logical_layout,
        axis_roles=roles,
        sharding_dim_map=dim_map,
        partial_mesh_axes=sharding.partial,
    )


def _gemm_descriptor(
    gemm: PhysicalNode,
    values: dict[str, TensorValue],
    boundary_input_refs: tuple[str, ...],
) -> SwizzleGemmDescriptor:
    if gemm.kind is not OpKind.GEMM or type(gemm.workload) is not GemmWorkload:
        _fail("fused GEMM member must carry GemmWorkload", "fused_op.members")
    if len(gemm.inputs) not in (1, 2) or len(gemm.outputs) != 1:
        _fail("Swizzle GEMM requires one/two inputs and one output", "fused_op.members")
    try:
        lhs_value = values[gemm.inputs[0]]
        rhs_value = values[gemm.inputs[1]] if len(gemm.inputs) == 2 else None
        output_value = values[gemm.outputs[0]]
    except KeyError as error:
        _fail(f"GEMM references missing value {error.args[0]!r}", "ir1.values")
        raise AssertionError("unreachable")
    m, n, k = gemm.workload.logical_shape
    lhs_roles = (
        SwizzleTensorAxisRole.FREE_LHS,
        SwizzleTensorAxisRole.CONTRACT,
    )
    rhs_roles = (
        SwizzleTensorAxisRole.CONTRACT,
        SwizzleTensorAxisRole.FREE_RHS,
    )
    output_roles = (
        SwizzleTensorAxisRole.FREE_LHS,
        SwizzleTensorAxisRole.FREE_RHS,
    )
    if rhs_value is not None:
        rhs_view = _tensor_view(
            rhs_value,
            shape=(k, n),
            roles=rhs_roles,
        )
        local_operand_refs: tuple[str, ...] = ()
    else:
        rhs_ref = f"{gemm.id}::implicit_rhs"
        if gemm.workload.partition is GemmPartition.ROW_PARALLEL:
            rhs_dim_map = (MeshAxisName.TP, None)
        elif gemm.workload.partition is GemmPartition.COLUMN_PARALLEL:
            rhs_dim_map = (None, MeshAxisName.TP)
        else:
            rhs_dim_map = (None, None)
        rhs_view = SwizzleTensorView(
            value_ref=rhs_ref,
            shape=(k, n),
            layout="KN_weight_local",
            axis_roles=rhs_roles,
            sharding_dim_map=rhs_dim_map,
            partial_mesh_axes=(),
        )
        local_operand_refs = (rhs_ref,)
    return SwizzleGemmDescriptor(
        node_ref=gemm.id,
        partition=gemm.workload.partition,
        m=m,
        n=n,
        k=k,
        batch_shape=(),
        lhs=_tensor_view(lhs_value, shape=(m, k), roles=lhs_roles),
        rhs=rhs_view,
        boundary_input_refs=boundary_input_refs,
        local_operand_refs=local_operand_refs,
        output=_tensor_view(output_value, shape=(m, n), roles=output_roles),
        dtype=gemm.workload.dtype,
        accumulation_dtype=gemm.math.accumulation_dtype,
        flops=2 * m * n * k,
    )


def _collective_roles(
    pattern: FusionPattern,
    descriptor: SwizzleGemmDescriptor,
    collective_output_ref: str,
) -> tuple[SwizzleTensorAxisRole, ...]:
    if pattern is not FusionPattern.AG_GEMM:
        return descriptor.output.axis_roles
    if descriptor.lhs.value_ref == collective_output_ref:
        return descriptor.lhs.axis_roles
    if descriptor.rhs.value_ref == collective_output_ref:
        return descriptor.rhs.axis_roles
    _fail(
        "AllGather output must feed exactly one GEMM operand",
        "fused_op.boundary",
    )
    raise AssertionError("unreachable")


def _collective_descriptor(
    collective: PhysicalNode,
    pattern: FusionPattern,
    gemm: SwizzleGemmDescriptor,
    values: dict[str, TensorValue],
    ranks: tuple[int, ...],
) -> SwizzleCollectiveDescriptor:
    if collective.kind is not OpKind.COLLECTIVE or type(collective.workload) is not CollectiveWorkload:
        _fail("fused collective member must carry CollectiveWorkload", "fused_op.members")
    if len(collective.inputs) != 1 or len(collective.outputs) != 1:
        _fail("Swizzle collective requires one input and one output", "fused_op.members")
    workload = collective.workload
    try:
        input_value = values[collective.inputs[0]]
        output_value = values[collective.outputs[0]]
    except KeyError as error:
        _fail(f"collective references missing value {error.args[0]!r}", "ir1.values")
        raise AssertionError("unreachable")
    roles = _collective_roles(pattern, gemm, output_value.id)
    participants = len(ranks)
    if workload.participant_count != participants:
        _fail("collective participant count disagrees with group", "fused_op.group")

    if workload.collective is CollectiveKind.ALL_GATHER:
        axis = workload.gather_tensor_axis
        if axis is None or axis >= len(gemm.lhs.shape):
            _fail("AllGather axis is outside the GEMM operand", "collective.gather_tensor_axis")
        output_shape = (
            gemm.lhs.shape
            if gemm.lhs.value_ref == output_value.id
            else gemm.rhs.shape
        )
        input_shape = list(output_shape)
        if input_shape[axis] % participants:
            _fail("AllGather operand extent is not divisible by participants", "collective.gather_tensor_axis")
        input_shape[axis] //= participants
        position = SwizzleCollectivePosition.BEFORE_GEMM
    elif workload.collective is CollectiveKind.REDUCE_SCATTER:
        axis = workload.scatter_tensor_axis
        if axis is None or axis >= len(gemm.output.shape):
            _fail("ReduceScatter axis is outside GEMM output", "collective.scatter_tensor_axis")
        input_shape = list(gemm.output.shape)
        output_shape_list = list(input_shape)
        if output_shape_list[axis] % participants:
            _fail("ReduceScatter extent is not divisible by participants", "collective.scatter_tensor_axis")
        output_shape_list[axis] //= participants
        output_shape = tuple(output_shape_list)
        position = SwizzleCollectivePosition.AFTER_GEMM
    elif workload.collective is CollectiveKind.ALL_REDUCE:
        input_shape = list(gemm.output.shape)
        output_shape = gemm.output.shape
        position = SwizzleCollectivePosition.AFTER_GEMM
    else:
        _fail("unsupported Swizzle collective", "collective.kind")
        raise AssertionError("unreachable")

    return SwizzleCollectiveDescriptor(
        node_ref=collective.id,
        kind=workload.collective,
        reduce_op=workload.reduce_op,
        position=position,
        mesh_axes=workload.mesh_axes,
        participant_ranks=ranks,
        gather_tensor_axis=workload.gather_tensor_axis,
        scatter_tensor_axis=workload.scatter_tensor_axis,
        logical_bytes=workload.logical_tensor_bytes,
        rank_input_bytes=workload.rank_input_bytes,
        rank_output_bytes=workload.rank_output_bytes,
        input=_tensor_view(input_value, shape=tuple(input_shape), roles=roles),
        output=_tensor_view(output_value, shape=tuple(output_shape), roles=roles),
    )


def _typed_group(topology: SwizzleTopologyView) -> SwizzleGroupView:
    logical_shape = (
        topology.physical_shape
        if topology.is_complete_rectangle
        else (topology.rank_count, 1)
    )
    result = SwizzleGroupView(
        group_ref=topology.group_ref,
        logical_shape=logical_shape,
        placements=tuple(
            SwizzleRankPlacement(
                rank=item.rank,
                x=item.physical_coord[0],
                y=item.physical_coord[1],
            )
            for item in topology.ranks
        ),
        routes=tuple(
            SwizzleRouteView(
                id=item.id,
                source_rank=item.source_rank,
                destination_rank=item.destination_rank,
                die_path=item.die_path,
                resource_ids=item.resource_ids,
            )
            for item in topology.routes
        ),
    )
    result.validate("swizzle_problem.group")
    return result


def build_swizzle_problem(
    ir1: IR1,
    fused_op: FusedOpSkeleton,
    hardware_profile: SwizzleHardwareProfile,
    constraints: SwizzleConstraints,
) -> SwizzleProblem:
    """Close IR-1 provenance and build one deterministic planner problem."""

    if type(ir1) is not IR1:
        _fail("must be an IR1", "ir1")
    if type(fused_op) is not FusedOpSkeleton:
        _fail("must be a FusedOpSkeleton", "fused_op")
    ir1.validate("ir1")
    hardware_profile.validate("hardware_profile")
    constraints.validate("constraints")
    bound = next((item for item in ir1.fused_op_skeletons if item.id == fused_op.id), None)
    if bound != fused_op:
        _fail("must exactly reference a skeleton in IR1", "fused_op")
    if len(fused_op.member_node_ids) != 2:
        _fail("Swizzle V1 requires exactly two members", "fused_op.member_node_ids")
    pattern = fused_op.semantic_contract.pattern
    nodes = {node.id: node for node in ir1.nodes}
    try:
        first, second = (nodes[ref] for ref in fused_op.member_node_ids)
    except KeyError as error:
        _fail(f"fused member {error.args[0]!r} is missing", "fused_op.member_node_ids")
        raise AssertionError("unreachable")
    if pattern is FusionPattern.AG_GEMM:
        collective_node, gemm_node = first, second
    else:
        gemm_node, collective_node = first, second
    if len({gemm_node.execution_group_ref, collective_node.execution_group_ref}) != 1:
        _fail("fused members must share one execution group", "fused_op.member_node_ids")
    if gemm_node.instance_id != fused_op.instance_id or collective_node.instance_id != fused_op.instance_id:
        _fail("fused members must belong to the skeleton instance", "fused_op.instance_id")
    group = next(
        (item for item in ir1.groups if item.id == gemm_node.execution_group_ref),
        None,
    )
    if group is None:
        _fail("fused execution group is missing", "fused_op.group")
    topology = build_topology_view(group, ir1.fabric)
    values = {value.id: value for value in ir1.values}
    gemm = _gemm_descriptor(gemm_node, values, fused_op.boundary_inputs)
    ranks = tuple(item.rank for item in topology.ranks)
    collective = _collective_descriptor(
        collective_node,
        pattern,
        gemm,
        values,
        ranks,
    )
    result = SwizzleProblem.create(
        source_ir1_id=ir1.id,
        fused_op_id=fused_op.id,
        pattern=pattern,
        gemm=gemm,
        collective=collective,
        group=_typed_group(topology),
        hardware_profile=hardware_profile,
        constraints=constraints,
    )
    result.validate("swizzle_problem")
    return result


__all__ = ["build_swizzle_problem"]
