"""Build personalized MoE problems from generalized execution truth."""

from __future__ import annotations

from ...errors import SchemaError
from ...schema.ir0 import FusionPattern
from ...schema.swizzle import (
    SwizzleAlgorithm,
    SwizzleGroupView,
    SwizzleRankPlacement,
)
from ...schema.swizzle_moe import (
    MoeFusionRegion,
    MoeHardwareFacts,
    MoeEndpointSessionContract,
    MoeSwizzleProblem,
    MoeTopologyWitness,
    MoeTrafficScenario,
    MoeTrafficScenarioKind,
    MoeTrafficQuantileMethod,
)
from ...schema.swizzle_moe_execution import MoeScaleExecution
from ...schema.swizzle_moe_scale import MoeSwizzleScaleOracle, MoeSwizzleScaleSpec


def _topology(
    region: MoeFusionRegion,
    spec: MoeSwizzleScaleSpec,
) -> MoeTopologyWitness:
    rows, columns = spec.mesh_rows, spec.mesh_columns
    ranks = tuple(range(rows * columns))
    group = SwizzleGroupView(
        group_ref=f"moe.scale.group.{spec.id}",
        logical_shape=(rows, columns),
        placements=tuple(
            SwizzleRankPlacement(
                rank=rank,
                x=rank % columns,
                y=rank // columns,
            )
            for rank in ranks
        ),
        routes=region.semantic_witness.traffic.pair_routes,
    )
    row_orders = tuple(
        tuple(rank for rank in ranks if rank // columns == row)
        for row in range(rows)
    )
    column_orders = tuple(
        tuple(rank for rank in ranks if rank % columns == column)
        for column in range(columns)
    )
    pivots = tuple(
        (
            source,
            destination,
            (source // columns) * columns + destination % columns,
        )
        for source in ranks
        for destination in ranks
        if source != destination
    )
    routes = {
        (item.source_rank, item.destination_rank): item for item in group.routes
    }
    complete = (
        rows > 1
        and columns > 1
        and len(routes) == len(ranks) * (len(ranks) - 1)
        and all(
            pivot in routes[(source, destination)].die_path
            for source, destination, pivot in pivots
        )
    )
    result = MoeTopologyWitness(
        group=group,
        row_orders=row_orders,
        column_orders=column_orders,
        pivot_by_pair=pivots,
        complete_rectangle=complete,
    )
    result.validate("moe_problem.topology")
    return result


def build_moe_swizzle_problem(
    region: MoeFusionRegion,
    spec: MoeSwizzleScaleSpec,
    oracle: MoeSwizzleScaleOracle,
    execution: MoeScaleExecution,
    *,
    hardware_facts: MoeHardwareFacts,
    endpoint_session_contract: MoeEndpointSessionContract,
    max_candidates: int = 32,
) -> MoeSwizzleProblem:
    if (
        type(region) is not MoeFusionRegion
        or type(spec) is not MoeSwizzleScaleSpec
        or type(oracle) is not MoeSwizzleScaleOracle
        or type(execution) is not MoeScaleExecution
    ):
        raise SchemaError("requires typed region/spec/oracle/execution", path="build_moe_swizzle_problem")
    region.validate("build_moe_swizzle_problem.region")
    execution.validate_against(spec, oracle, "build_moe_swizzle_problem.execution")
    if (
        region.source_spec_id != spec.id
        or region.source_oracle_id != oracle.id
        or region.source_execution_id != execution.id
    ):
        raise SchemaError("region source provenance mismatch", path="build_moe_swizzle_problem")
    traffic = region.semantic_witness.traffic
    topology = _topology(region, spec)
    actual_matrix = tuple(
        tuple(sum(
            item.source_rank == source and item.expert_index == expert
            for item in traffic.assignments
        ) for expert in range(spec.expert_count))
        for source in range(spec.mesh_rows * spec.mesh_columns)
    )
    actual = MoeTrafficScenario(
        kind=MoeTrafficScenarioKind.ACTUAL,
        expert_token_counts=oracle.expert_token_counts,
        source_expert_token_counts=actual_matrix,
        logical_payload_bytes=traffic.logical_payload_bytes,
        expert_gemm_flops=traffic.expert_gemm_flops,
        region_boundary_output_bytes=traffic.region_boundary_output_bytes,
        executable_binding=True,
        quantile_method=None,
    )
    capacity_counts = (spec.capacity_per_expert,) * spec.expert_count
    capacity_assignments = sum(capacity_counts)
    capacity_matrix_rows = [[0] * spec.expert_count for _ in range(spec.mesh_rows * spec.mesh_columns)]
    for expert, count in enumerate(capacity_counts):
        home = spec.expert_home_die_ids[expert]
        farthest = max(
            range(spec.mesh_rows * spec.mesh_columns),
            key=lambda source: ((abs(source % spec.mesh_columns - home % spec.mesh_columns) + abs(source // spec.mesh_columns - home // spec.mesh_columns)), -source),
        )
        capacity_matrix_rows[farthest][expert] = count
    capacity_matrix = tuple(tuple(row) for row in capacity_matrix_rows)
    token_bytes = spec.hidden_size * 2
    if region.pattern is FusionPattern.MOE_DISPATCH_GEMM:
        per_token_flops = 4 * spec.hidden_size * spec.intermediate_size
        per_token_boundary = 2 * spec.intermediate_size * 2
    elif region.pattern is FusionPattern.MOE_GEMM_COMBINE:
        per_token_flops = 2 * spec.hidden_size * spec.intermediate_size
        per_token_boundary = token_bytes
    else:
        raise SchemaError("Dense pattern entered MoE problem", path="build_moe_swizzle_problem.region.pattern")
    capacity = MoeTrafficScenario(
        kind=MoeTrafficScenarioKind.CAPACITY,
        expert_token_counts=capacity_counts,
        source_expert_token_counts=capacity_matrix,
        logical_payload_bytes=capacity_assignments * token_bytes,
        expert_gemm_flops=capacity_assignments * per_token_flops,
        region_boundary_output_bytes=capacity_assignments * per_token_boundary,
        executable_binding=False,
        quantile_method=None,
    )
    p95_counts = oracle.expert_token_counts
    p95_assignments = sum(p95_counts)
    p95 = MoeTrafficScenario(
        kind=MoeTrafficScenarioKind.P95,
        expert_token_counts=p95_counts,
        source_expert_token_counts=actual_matrix,
        logical_payload_bytes=actual.logical_payload_bytes,
        expert_gemm_flops=actual.expert_gemm_flops,
        region_boundary_output_bytes=actual.region_boundary_output_bytes,
        executable_binding=False,
        quantile_method=MoeTrafficQuantileMethod.DETERMINISTIC_SINGLE_TRACE,
    )
    algorithms = (
        SwizzleAlgorithm.UNFUSED,
        SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A,
    ) + (
        (SwizzleAlgorithm.COMET_MESH_PERSONALIZED_A2A,)
        if topology.complete_rectangle
        else ()
    )
    return MoeSwizzleProblem.create(
        source_execution_id=execution.id,
        scale_name=spec.name,
        scale_role=spec.role,
        region=region,
        topology=topology,
        traffic_scenarios=tuple(
            sorted((actual, capacity, p95), key=lambda item: item.kind.value)
        ),
        allowed_algorithms=algorithms,
        hardware_facts=hardware_facts,
        endpoint_session_contract=endpoint_session_contract,
        max_candidates=max_candidates,
    )


__all__ = ["build_moe_swizzle_problem"]
