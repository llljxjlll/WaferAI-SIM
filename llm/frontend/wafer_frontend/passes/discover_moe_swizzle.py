"""Discover MoE fusion regions from generalized production execution truth."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.ir0 import FusionPattern
from ..schema.swizzle import SwizzleRouteView, SwizzleTensorAxisRole
from ..schema.swizzle_moe import (
    MoeExpertGemmView,
    MoeFusionRegion,
    MoePersonalizedTrafficView,
    MoeSemanticWitness,
    MoeTokenAssignmentView,
)
from ..schema.swizzle_moe_execution import (
    MoeScaleExecution,
    MoeScaleExecutionActionKind,
    MoeScaleExecutionFlowRole,
)
from ..schema.swizzle_moe_scale import MoeSwizzleScaleOracle, MoeSwizzleScaleSpec


def _xy_path(source: int, destination: int, columns: int) -> tuple[int, ...]:
    source_row, source_column = divmod(source, columns)
    destination_row, destination_column = divmod(destination, columns)
    path = [source]
    if source_column != destination_column:
        path.append(source_row * columns + destination_column)
    if source_row != destination_row:
        path.append(destination_row * columns + destination_column)
    return tuple(path)


def _route_id(source: int, destination: int) -> str:
    return f"moe.scale.route.r{source}.r{destination}"


def _routes(spec: MoeSwizzleScaleSpec) -> tuple[SwizzleRouteView, ...]:
    ranks = range(spec.mesh_rows * spec.mesh_columns)
    result = []
    for source in ranks:
        for destination in ranks:
            if source == destination:
                continue
            path = _xy_path(source, destination, spec.mesh_columns)
            result.append(
                SwizzleRouteView(
                    id=_route_id(source, destination),
                    source_rank=source,
                    destination_rank=destination,
                    die_path=path,
                    resource_ids=tuple(
                        f"moe.scale.link.{left}.{right}"
                        for left, right in zip(path, path[1:])
                    ),
                )
            )
    return tuple(result)


def _boundaries(
    execution: MoeScaleExecution,
    member_refs: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    members = set(member_refs)
    producers: dict[str, str] = {}
    consumers: dict[str, list[str]] = {}
    for action in execution.actions:
        for value_ref in action.write_values:
            producers[value_ref] = action.id
        for value_ref in action.read_values:
            consumers.setdefault(value_ref, []).append(action.id)
    terminal_values = {item.value_ref for item in execution.terminals}
    inputs = []
    outputs = []
    actions = {item.id: item for item in execution.actions}
    for member_ref in member_refs:
        action = actions[member_ref]
        for value_ref in action.read_values:
            if producers.get(value_ref) not in members and value_ref not in inputs:
                inputs.append(value_ref)
        for value_ref in action.write_values:
            external = any(ref not in members for ref in consumers.get(value_ref, ()))
            if (external or value_ref in terminal_values) and value_ref not in outputs:
                outputs.append(value_ref)
    return tuple(inputs), tuple(outputs)


def _assignments(
    spec: MoeSwizzleScaleSpec,
    execution: MoeScaleExecution,
    pattern: FusionPattern,
) -> tuple[MoeTokenAssignmentView, ...]:
    flow_index = {(item.token_index, item.role): item for item in execution.flows}
    actions_by_token_role = {
        (item.token_index, item.role): item for item in execution.actions
    }
    result = []
    contributor_ordinals = [0] * spec.expert_count
    token_bytes = spec.hidden_size * 2
    for item in spec.trace.assignments:
        token = item.token_index
        expert = item.expert_index
        source = spec.token_source_die_ids[token]
        home = spec.expert_home_die_ids[expert]
        dispatch = flow_index.get((token, MoeScaleExecutionFlowRole.DISPATCH))
        combine = flow_index.get((token, MoeScaleExecutionFlowRole.COMBINE))
        local = source == home
        if local != (dispatch is None and combine is None):
            raise SchemaError(
                "trace locality disagrees with execution flows",
                path=f"discover_moe_swizzle.assignments[{token}]",
            )
        gate = actions_by_token_role[(token, "gate")]
        down = actions_by_token_role[(token, "down")]
        swiglu = actions_by_token_role[(token, "swiglu")]
        contributor_ordinal = contributor_ordinals[expert]
        contributor_ordinals[expert] += 1
        result.append(
            MoeTokenAssignmentView.create(
                token_index=token,
                source_rank=source,
                expert_index=expert,
                expert_rank=home,
                contributor_ordinal=contributor_ordinal,
                gate_weight_ref=None,
                dispatch_flow_ref=None if dispatch is None else dispatch.flow_ref,
                combine_flow_ref=None if combine is None else combine.flow_ref,
                dispatch_route_ref=None if dispatch is None else _route_id(source, home),
                combine_route_ref=None if combine is None else _route_id(home, source),
                payload_value_ref=(
                    gate.read_values[0]
                    if pattern is FusionPattern.MOE_DISPATCH_GEMM
                    else down.write_values[0]
                ),
                payload_bytes=token_bytes,
                swiglu_action_ref=(
                    swiglu.id
                    if pattern is FusionPattern.MOE_DISPATCH_GEMM else None
                ),
            )
        )
    return tuple(result)


def _expert_gemms(
    spec: MoeSwizzleScaleSpec,
    execution: MoeScaleExecution,
    pattern: FusionPattern,
) -> tuple[MoeExpertGemmView, ...]:
    roles = (
        ("gate", "up")
        if pattern is FusionPattern.MOE_DISPATCH_GEMM
        else ("down",)
    )
    result = []
    for expert in range(spec.expert_count):
        members = tuple(
            item
            for item in execution.actions
            if item.kind is MoeScaleExecutionActionKind.GEMM
            and item.expert_index == expert
            and item.role in roles
        )
        count = spec.trace.expert_histogram[expert]
        if len(members) != count * len(roles):
            raise SchemaError(
                "expert GEMM group cardinality differs from trace",
                path=f"discover_moe_swizzle.expert_gemms[{expert}]",
            )
        result.append(
            MoeExpertGemmView.create(
                expert_index=expert,
                rank=spec.expert_home_die_ids[expert],
                member_refs=tuple(item.id for item in members),
                m_tokens=count,
                n=(
                    spec.intermediate_size
                    if pattern is FusionPattern.MOE_DISPATCH_GEMM
                    else spec.hidden_size
                ),
                k=(
                    spec.hidden_size
                    if pattern is FusionPattern.MOE_DISPATCH_GEMM
                    else spec.intermediate_size
                ),
                dtype=spec.dtype,
                accumulation_dtype=DType.FP32,
                flops=sum(item.flops for item in members),
            )
        )
    return tuple(result)


def _region(
    spec: MoeSwizzleScaleSpec,
    oracle: MoeSwizzleScaleOracle,
    execution: MoeScaleExecution,
    pattern: FusionPattern,
) -> MoeFusionRegion:
    flow_role = (
        MoeScaleExecutionFlowRole.DISPATCH
        if pattern is FusionPattern.MOE_DISPATCH_GEMM
        else MoeScaleExecutionFlowRole.COMBINE
    )
    gemm_roles = (
        ("gate", "up")
        if pattern is FusionPattern.MOE_DISPATCH_GEMM
        else ("down",)
    )
    flow_action_refs = {
        ref
        for flow in execution.flows
        if flow.role is flow_role
        for ref in (flow.send_action_ref, flow.recv_action_ref, flow.wait_action_ref)
    }
    member_refs = tuple(
        item.id
        for item in execution.actions
        if item.id in flow_action_refs
        or (
            item.kind is MoeScaleExecutionActionKind.GEMM
            and item.role in gemm_roles
        )
        or (
            pattern is FusionPattern.MOE_DISPATCH_GEMM
            and item.kind is MoeScaleExecutionActionKind.SWIGLU
        )
    )
    assignments = _assignments(spec, execution, pattern)
    gemms = _expert_gemms(spec, execution, pattern)
    boundaries = _boundaries(execution, member_refs)
    expected_boundary_counts = (
        (3 * spec.tokens, spec.tokens)
        if pattern is FusionPattern.MOE_DISPATCH_GEMM
        else (2 * spec.tokens, spec.tokens)
    )
    if tuple(map(len, boundaries)) != expected_boundary_counts:
        raise SchemaError("region boundary action/value closure changed", path="discover_moe_swizzle.boundary")
    token_bytes = spec.hidden_size * 2
    boundary_bytes = (
        spec.tokens * spec.intermediate_size * 2
        if pattern is FusionPattern.MOE_DISPATCH_GEMM
        else oracle.combined_terminal_bytes
    )
    traffic = MoePersonalizedTrafficView(
        assignments=assignments,
        expert_gemms=gemms,
        pair_routes=_routes(spec),
        top_k=spec.top_k,
        capacity_tokens_per_expert=spec.capacity_per_expert,
        trace_digest=oracle.trace_digest,
        logical_payload_bytes=(
            oracle.dispatch_logical_bytes
            if pattern is FusionPattern.MOE_DISPATCH_GEMM
            else oracle.combine_logical_bytes
        ),
        expert_gemm_flops=sum(item.flops for item in gemms),
        region_boundary_output_bytes=boundary_bytes,
    )
    witness = MoeSemanticWitness(
        pattern=pattern,
        traffic=traffic,
        split_axis=(
            SwizzleTensorAxisRole.FREE_LHS
            if pattern is FusionPattern.MOE_DISPATCH_GEMM
            else SwizzleTensorAxisRole.FREE_RHS
        ),
        gate_weight_applied=False,
        reduce_dtype=None,
        boundary_closed=True,
        route_reversal_closed=all(
            item.source_rank == item.expert_rank
            or (
                item.dispatch_route_ref == _route_id(item.source_rank, item.expert_rank)
                and item.combine_route_ref == _route_id(item.expert_rank, item.source_rank)
            )
            for item in assignments
        ),
    )
    return MoeFusionRegion.create(
        source_spec_id=spec.id,
        source_oracle_id=oracle.id,
        source_execution_id=execution.id,
        pattern=pattern,
        member_refs=member_refs,
        boundary_input_refs=boundaries[0],
        boundary_output_refs=boundaries[1],
        assignment_refs=tuple(item.id for item in assignments),
        trace_digest=oracle.trace_digest,
        semantic_witness=witness,
    )


def discover_moe_swizzle_regions(
    spec: MoeSwizzleScaleSpec,
    oracle: MoeSwizzleScaleOracle,
    execution: MoeScaleExecution,
) -> tuple[MoeFusionRegion, ...]:
    if (
        type(spec) is not MoeSwizzleScaleSpec
        or type(oracle) is not MoeSwizzleScaleOracle
        or type(execution) is not MoeScaleExecution
    ):
        raise SchemaError("requires exact scale spec/oracle/execution", path="discover_moe_swizzle")
    execution.validate_against(spec, oracle, "discover_moe_swizzle.execution")
    if not execution.capacity.admitted:
        raise SchemaError("execution capacity is not admitted", path="discover_moe_swizzle.capacity")
    regions = tuple(
        _region(spec, oracle, execution, pattern)
        for pattern in (
            FusionPattern.MOE_DISPATCH_GEMM,
            FusionPattern.MOE_GEMM_COMBINE,
        )
    )
    members = tuple(ref for region in regions for ref in region.member_refs)
    if len(members) != len(set(members)):
        raise SchemaError("MoE regions overlap", path="discover_moe_swizzle.regions")
    return regions


__all__ = ["discover_moe_swizzle_regions"]
