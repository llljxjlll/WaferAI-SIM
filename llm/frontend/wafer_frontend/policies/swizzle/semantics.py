"""Pure, fail-closed semantic analysis for supported Swizzle patterns."""

from __future__ import annotations

from ...errors import SchemaError
from ...schema.ir0 import CollectiveKind, FusionPattern, GemmPartition, ReduceOp
from ...schema.swizzle import (
    SwizzleCollectiveDescriptor,
    SwizzleCollectivePosition,
    SwizzleGemmDescriptor,
    SwizzleOperand,
    SwizzleSemanticWitness,
    SwizzleTensorAxis,
    SwizzleTensorAxisRole,
    SwizzleTensorView,
    SwizzleUpdateKind,
)


def _require(condition: bool, message: str, path: str) -> None:
    if not condition:
        raise SchemaError(message, path=path)


def _axis(view: SwizzleTensorView, index: int, *, path: str) -> SwizzleTensorAxis:
    _require(index < len(view.shape), "tensor axis is out of range", path)
    role = view.axis_roles[index]
    return SwizzleTensorAxis(
        tensor_ref=view.value_ref,
        index=index,
        name=f"{role.value}_{index}",
        extent=view.shape[index],
        role=role,
    )


def _common(
    gemm: SwizzleGemmDescriptor,
    collective: SwizzleCollectiveDescriptor,
    *,
    path: str,
) -> None:
    gemm.validate(f"{path}.gemm")
    collective.validate(f"{path}.collective")
    _require(gemm.node_ref != collective.node_ref, "members must be distinct", path)


def _require_boundary(
    observed_inputs: tuple[str, ...],
    expected_inputs: tuple[str, ...],
    observed_outputs: tuple[str, ...],
    expected_outputs: tuple[str, ...],
    *,
    path: str,
) -> None:
    _require(observed_inputs == expected_inputs, "boundary input order is not exact", f"{path}.boundary_input_refs")
    _require(observed_outputs == expected_outputs, "boundary output order is not exact", f"{path}.boundary_output_refs")


def _require_ag_sharding(
    collective: SwizzleCollectiveDescriptor,
    axis: int,
    *,
    path: str,
) -> None:
    source = collective.input
    gathered = collective.output
    _require(len(collective.mesh_axes) == 1, "v1 AG requires one mesh axis", f"{path}.mesh_axes")
    mesh_axis = collective.mesh_axes[0]
    _require(source.sharding_dim_map[axis] is mesh_axis, "AG input must be sharded on gather axis", f"{path}.input.sharding_dim_map")
    _require(gathered.sharding_dim_map[axis] is None, "AG output gather axis must be replicated", f"{path}.output.sharding_dim_map")
    for index, (before, after) in enumerate(zip(source.sharding_dim_map, gathered.sharding_dim_map, strict=True)):
        if index != axis:
            _require(before is after, "AG may change only gather-axis sharding", f"{path}.output.sharding_dim_map[{index}]")
    _require(source.partial_mesh_axes == gathered.partial_mesh_axes, "AG must preserve partial axes", f"{path}.output.partial_mesh_axes")


def _require_rs_sharding(
    collective: SwizzleCollectiveDescriptor,
    axis: int,
    *,
    path: str,
) -> None:
    partial = collective.input
    shard = collective.output
    _require(len(collective.mesh_axes) == 1, "v1 RS requires one mesh axis", f"{path}.mesh_axes")
    mesh_axis = collective.mesh_axes[0]
    _require(mesh_axis in partial.partial_mesh_axes, "RS input must be partial on reduction mesh", f"{path}.input.partial_mesh_axes")
    _require(mesh_axis not in shard.partial_mesh_axes, "RS output must clear the reduced partial axis", f"{path}.output.partial_mesh_axes")
    _require(partial.sharding_dim_map[axis] is None, "RS input scatter axis must be unsharded", f"{path}.input.sharding_dim_map")
    _require(shard.sharding_dim_map[axis] is mesh_axis, "RS output must shard the scatter axis", f"{path}.output.sharding_dim_map")
    for index, (before, after) in enumerate(zip(partial.sharding_dim_map, shard.sharding_dim_map, strict=True)):
        if index != axis:
            _require(before is after, "RS may change only scatter-axis sharding", f"{path}.output.sharding_dim_map[{index}]")


def _require_ar_sharding(
    collective: SwizzleCollectiveDescriptor,
    *,
    path: str,
) -> None:
    _require(
        set(collective.mesh_axes).issubset(collective.input.partial_mesh_axes),
        "AR input must be partial on every reduction mesh axis",
        f"{path}.input.partial_mesh_axes",
    )
    _require(
        not set(collective.mesh_axes).intersection(collective.output.partial_mesh_axes),
        "AR output must clear every reduced partial axis",
        f"{path}.output.partial_mesh_axes",
    )
    _require(
        collective.input.sharding_dim_map == collective.output.sharding_dim_map,
        "AR must preserve dimension sharding",
        f"{path}.output.sharding_dim_map",
    )


def analyze_ag_gemm(
    gemm: SwizzleGemmDescriptor,
    collective: SwizzleCollectiveDescriptor,
    *,
    boundary_input_refs: tuple[str, ...],
    boundary_output_refs: tuple[str, ...],
    path: str = "ag_gemm",
) -> SwizzleSemanticWitness:
    """Prove ``AllGather -> GEMM`` and return its exact decomposition role."""

    _common(gemm, collective, path=path)
    _require(collective.kind is CollectiveKind.ALL_GATHER, "requires AllGather", f"{path}.collective.kind")
    _require(collective.position is SwizzleCollectivePosition.BEFORE_GEMM, "AllGather must precede GEMM", f"{path}.collective.position")
    _require(collective.reduce_op is None, "AllGather cannot reduce", f"{path}.collective.reduce_op")
    axis_index = collective.gather_tensor_axis
    assert axis_index is not None  # descriptor validation proves this

    matches = tuple(
        (operand, view)
        for operand, view in (
            (SwizzleOperand.LHS, gemm.lhs),
            (SwizzleOperand.RHS, gemm.rhs),
        )
        if view == collective.output
    )
    _require(len(matches) == 1, "gathered value must feed exactly one GEMM operand", path)
    operand, gathered_view = matches[0]
    _require(collective.input.axis_roles == collective.output.axis_roles, "AG axis roles must be preserved", f"{path}.collective.output.axis_roles")
    _require_ag_sharding(collective, axis_index, path=f"{path}.collective")
    split_axis = _axis(gathered_view, axis_index, path=f"{path}.collective.gather_tensor_axis")
    _require(
        split_axis.role in (
            SwizzleTensorAxisRole.BATCH,
            SwizzleTensorAxisRole.FREE_LHS,
            SwizzleTensorAxisRole.FREE_RHS,
            SwizzleTensorAxisRole.CONTRACT,
        ),
        "gather axis has no GEMM dimension role",
        f"{path}.collective.gather_tensor_axis",
    )
    local_operands = set(gemm.local_operand_refs)
    expected_inputs = (
        (collective.input.value_ref,)
        + (() if gemm.rhs.value_ref in local_operands else (gemm.rhs.value_ref,))
        if operand is SwizzleOperand.LHS
        else (() if gemm.lhs.value_ref in local_operands else (gemm.lhs.value_ref,))
        + (collective.input.value_ref,)
    )
    _require(
        gemm.boundary_input_refs == expected_inputs,
        "typed GEMM boundary inputs do not close AG+GEMM",
        f"{path}.gemm.boundary_input_refs",
    )
    _require_boundary(
        boundary_input_refs,
        gemm.boundary_input_refs,
        boundary_output_refs,
        (gemm.output.value_ref,),
        path=path,
    )
    update = (
        SwizzleUpdateKind.PARTIAL_ACCUMULATION
        if split_axis.role is SwizzleTensorAxisRole.CONTRACT
        else SwizzleUpdateKind.OUTPUT_SLICE
    )
    result = SwizzleSemanticWitness(
        pattern=FusionPattern.AG_GEMM,
        member_refs=(collective.node_ref, gemm.node_ref),
        boundary_input_refs=boundary_input_refs,
        boundary_output_refs=boundary_output_refs,
        intermediate_value_ref=collective.output.value_ref,
        gemm_operand=operand,
        split_axis=split_axis,
        update_kind=update,
        gather_axis=axis_index,
        reduction_axis=None,
        has_reduction_phase=False,
        has_replication_phase=False,
        input_layout_closed=True,
        output_layout_closed=True,
        sharding_transition_closed=True,
    )
    result.validate(path)
    return result


def analyze_gemm_rs(
    gemm: SwizzleGemmDescriptor,
    collective: SwizzleCollectiveDescriptor,
    *,
    boundary_input_refs: tuple[str, ...],
    boundary_output_refs: tuple[str, ...],
    path: str = "gemm_rs",
) -> SwizzleSemanticWitness:
    """Prove ``row GEMM -> SUM ReduceScatter``."""

    _common(gemm, collective, path=path)
    _require(gemm.partition is GemmPartition.ROW_PARALLEL, "requires row-parallel GEMM", f"{path}.gemm.partition")
    _require(collective.kind is CollectiveKind.REDUCE_SCATTER, "requires ReduceScatter", f"{path}.collective.kind")
    _require(collective.position is SwizzleCollectivePosition.AFTER_GEMM, "ReduceScatter must follow GEMM", f"{path}.collective.position")
    _require(collective.reduce_op is ReduceOp.SUM, "requires SUM", f"{path}.collective.reduce_op")
    _require(collective.input == gemm.output, "GEMM partial output must be the RS input", f"{path}.collective.input")
    axis_index = collective.scatter_tensor_axis
    assert axis_index is not None
    _require(collective.input.axis_roles == collective.output.axis_roles, "RS axis roles must be preserved", f"{path}.collective.output.axis_roles")
    _require_rs_sharding(collective, axis_index, path=f"{path}.collective")
    expected_inputs = tuple(
        ref
        for ref in (gemm.lhs.value_ref, gemm.rhs.value_ref)
        if ref not in set(gemm.local_operand_refs)
    )
    _require(gemm.boundary_input_refs == expected_inputs, "typed GEMM boundary inputs do not close GEMM+RS", f"{path}.gemm.boundary_input_refs")
    _require_boundary(
        boundary_input_refs,
        gemm.boundary_input_refs,
        boundary_output_refs,
        (collective.output.value_ref,),
        path=path,
    )
    result = SwizzleSemanticWitness(
        pattern=FusionPattern.GEMM_RS,
        member_refs=(gemm.node_ref, collective.node_ref),
        boundary_input_refs=boundary_input_refs,
        boundary_output_refs=boundary_output_refs,
        intermediate_value_ref=gemm.output.value_ref,
        gemm_operand=SwizzleOperand.OUTPUT,
        split_axis=_axis(collective.input, axis_index, path=f"{path}.collective.scatter_tensor_axis"),
        update_kind=SwizzleUpdateKind.PARTIAL_ACCUMULATION,
        gather_axis=None,
        reduction_axis=2,
        has_reduction_phase=True,
        has_replication_phase=False,
        input_layout_closed=True,
        output_layout_closed=True,
        sharding_transition_closed=True,
    )
    result.validate(path)
    return result


def analyze_gemm_ar(
    gemm: SwizzleGemmDescriptor,
    collective: SwizzleCollectiveDescriptor,
    *,
    boundary_input_refs: tuple[str, ...],
    boundary_output_refs: tuple[str, ...],
    path: str = "gemm_ar",
) -> SwizzleSemanticWitness:
    """Prove ``row GEMM -> SUM AllReduce`` with explicit RS+AG phases."""

    _common(gemm, collective, path=path)
    _require(gemm.partition is GemmPartition.ROW_PARALLEL, "requires row-parallel GEMM", f"{path}.gemm.partition")
    _require(collective.kind is CollectiveKind.ALL_REDUCE, "requires AllReduce", f"{path}.collective.kind")
    _require(collective.position is SwizzleCollectivePosition.AFTER_GEMM, "AllReduce must follow GEMM", f"{path}.collective.position")
    _require(collective.reduce_op is ReduceOp.SUM, "requires SUM", f"{path}.collective.reduce_op")
    _require(collective.input == gemm.output, "GEMM partial output must be the AR input", f"{path}.collective.input")
    _require(collective.input.axis_roles == collective.output.axis_roles, "AR axis roles must be preserved", f"{path}.collective.output.axis_roles")
    _require_ar_sharding(collective, path=f"{path}.collective")
    expected_inputs = tuple(
        ref
        for ref in (gemm.lhs.value_ref, gemm.rhs.value_ref)
        if ref not in set(gemm.local_operand_refs)
    )
    _require(gemm.boundary_input_refs == expected_inputs, "typed GEMM boundary inputs do not close GEMM+AR", f"{path}.gemm.boundary_input_refs")
    _require_boundary(
        boundary_input_refs,
        gemm.boundary_input_refs,
        boundary_output_refs,
        (collective.output.value_ref,),
        path=path,
    )
    # AR has no boundary scatter axis.  K is the semantic reduction dimension;
    # use the output M axis as the deterministic pipeline split witness.
    split_axis = _axis(gemm.output, len(gemm.batch_shape), path=f"{path}.gemm.output")
    result = SwizzleSemanticWitness(
        pattern=FusionPattern.GEMM_AR,
        member_refs=(gemm.node_ref, collective.node_ref),
        boundary_input_refs=boundary_input_refs,
        boundary_output_refs=boundary_output_refs,
        intermediate_value_ref=gemm.output.value_ref,
        gemm_operand=SwizzleOperand.OUTPUT,
        split_axis=split_axis,
        update_kind=SwizzleUpdateKind.REDUCE_THEN_REPLICATE,
        gather_axis=None,
        reduction_axis=2,
        has_reduction_phase=True,
        has_replication_phase=True,
        input_layout_closed=True,
        output_layout_closed=True,
        sharding_transition_closed=True,
    )
    result.validate(path)
    return result


def analyze_semantics(
    pattern: FusionPattern,
    gemm: SwizzleGemmDescriptor,
    collective: SwizzleCollectiveDescriptor,
    *,
    boundary_input_refs: tuple[str, ...],
    boundary_output_refs: tuple[str, ...],
    path: str = "swizzle_semantics",
) -> SwizzleSemanticWitness:
    """Dispatch to the only three pattern validators admitted by v1."""

    if pattern is FusionPattern.AG_GEMM:
        return analyze_ag_gemm(
            gemm,
            collective,
            boundary_input_refs=boundary_input_refs,
            boundary_output_refs=boundary_output_refs,
            path=path,
        )
    if pattern is FusionPattern.GEMM_RS:
        return analyze_gemm_rs(
            gemm,
            collective,
            boundary_input_refs=boundary_input_refs,
            boundary_output_refs=boundary_output_refs,
            path=path,
        )
    if pattern is FusionPattern.GEMM_AR:
        return analyze_gemm_ar(
            gemm,
            collective,
            boundary_input_refs=boundary_input_refs,
            boundary_output_refs=boundary_output_refs,
            path=path,
        )
    raise SchemaError("unsupported Swizzle fusion pattern", path=f"{path}.pattern")


def validate_semantic_witness(
    witness: SwizzleSemanticWitness,
    gemm: SwizzleGemmDescriptor,
    collective: SwizzleCollectiveDescriptor,
    *,
    path: str = "swizzle_semantic_witness",
) -> None:
    """Rebuild a witness and reject any semantic/provenance tampering."""

    witness.validate(path)
    expected = analyze_semantics(
        witness.pattern,
        gemm,
        collective,
        boundary_input_refs=witness.boundary_input_refs,
        boundary_output_refs=witness.boundary_output_refs,
        path=path,
    )
    if witness != expected:
        raise SchemaError("semantic witness does not match descriptors", path=path)


__all__ = [
    "analyze_ag_gemm",
    "analyze_gemm_ar",
    "analyze_gemm_rs",
    "analyze_semantics",
    "validate_semantic_witness",
]
