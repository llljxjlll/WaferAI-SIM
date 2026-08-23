"""Build the self-contained W10/W11 three-case comparison plan."""

from __future__ import annotations

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.project_swizzle_plan import (
    project_swizzle_plan,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize import (
    force_swizzle_deployment,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize_ir1 import (
    materialize_swizzle_plan,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest
from llm.frontend.wafer_frontend.schema.swizzle_evidence import (
    SwizzleComparisonCasePlan,
    SwizzleComparisonSuitePlan,
)

from swizzle_cases import SwizzleIntegrationCase, build_swizzle_integration_cases


def _case_plan(case: SwizzleIntegrationCase) -> SwizzleComparisonCasePlan:
    case.validate()
    deployment = force_swizzle_deployment(case.decision).deployment_selection
    plan = materialize_swizzle_plan(
        case.partitioned_graph,
        case.decision,
        case.partitioned_graph.profile,
        deployment_selection=deployment,
    )
    projected = project_swizzle_plan(case.partitioned_graph, plan)
    split_axis = plan.candidate.split_axis
    if split_axis is None:
        raise SchemaError(
            "production Swizzle candidate lost split axis",
            path=f"swizzle_comparison.{case.name}.candidate.split_axis",
        )
    policy_configuration_digest = canonical_digest(
        {
            "naive_policy_ref": "INTER_DIE/naive",
            "swizzle_policy_ref": "INTER_DIE/swizzle_topo",
            "hardware_profile": case.decision.problem.hardware_profile,
            "constraints": case.decision.problem.constraints,
        }
    )
    result = SwizzleComparisonCasePlan.create(
        case_id=case.name,
        pattern=case.pattern,
        synthetic_placement=case.synthetic_placement,
        source_ir0_id=case.source_graph.id,
        source_ir0_digest=canonical_digest(case.source_graph),
        placed_ir1_id=case.placed_graph.id,
        placed_ir1_digest=canonical_digest(case.placed_graph),
        partitioned_ir1_id=case.partitioned_graph.id,
        partitioned_ir1_digest=canonical_digest(case.partitioned_graph),
        fusion_candidate_refs=tuple(
            candidate.id for candidate in case.source_graph.fusion_candidates
        ),
        naive_fusion_refs=case.naive_fusion_refs,
        swizzle_fusion_refs=case.swizzle_fusion_refs,
        skeleton_ref=case.skeleton_ref,
        naive_policy_ref="INTER_DIE/naive",
        swizzle_policy_ref="INTER_DIE/swizzle_topo",
        policy_configuration_digest=policy_configuration_digest,
        decision_ref=case.decision.id,
        decision_digest=canonical_digest(case.decision),
        deployment_selection_ref=deployment.id,
        candidate_ref=plan.candidate.id,
        swizzle_plan_ref=plan.id,
        swizzle_plan_digest=canonical_digest(plan),
        projection_ref=projected.id,
        projection_digest=canonical_digest(projected),
        algorithm=plan.algorithm,
        split_axis=split_axis,
        chunk_count=plan.candidate.chunk_count,
        unroll_degree=plan.candidate.unroll_degree,
        baseline_cost=case.decision.baseline.cost,
        selected_cost=plan.candidate.cost,
    )
    result.validate()
    return result


def build_swizzle_comparison_suite() -> SwizzleComparisonSuitePlan:
    """Build canonical AG/RS/AR plans from production discovery and planning."""

    result = SwizzleComparisonSuitePlan.create(
        tuple(_case_plan(case) for case in build_swizzle_integration_cases())
    )
    result.validate()
    return result


__all__ = ["build_swizzle_comparison_suite"]
