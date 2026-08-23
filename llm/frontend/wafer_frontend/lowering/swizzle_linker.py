"""Deterministic Python linker for the independent W9 Swizzle carrier."""

from __future__ import annotations

from ..schema.swizzle_ir2 import SwizzleIr2Projection
from ..schema.swizzle_lowering import (
    SwizzleFinalizerContract,
    SwizzleLinkedManifest,
    SwizzleLoweredProgram,
    expected_swizzle_manifest_digests,
)
from ..schema.swizzle_plan import SwizzleFusionPlan


def link_swizzle_manifest(
    plan: SwizzleFusionPlan,
    projection: SwizzleIr2Projection,
    lowered: SwizzleLoweredProgram,
) -> SwizzleLinkedManifest:
    """Link exact provenance and opcode streams while keeping finalization gated."""

    lowered.validate_against(plan, projection)
    result = SwizzleLinkedManifest.create(
        producer_pass="swizzle_manifest_linker",
        source_plan_ref=plan.id,
        source_projection_ref=projection.id,
        source_lowered_ref=lowered.id,
        source_ir1_id=plan.source_ir1_id,
        source_decision_ref=plan.decision.id,
        source_candidate_ref=plan.candidate.id,
        pattern=plan.pattern,
        algorithm=plan.algorithm,
        input_digests=expected_swizzle_manifest_digests(
            plan,
            projection,
            lowered,
        ),
        rank_streams=lowered.rank_streams,
        finalizer_contract=(
            SwizzleFinalizerContract.REQUIRES_EXACT_CORE_ADDRESS_ABI_V1
        ),
        timing_execution=True,
        functional_execution=False,
        finalizer_gate_reason=(
            "Swizzle projection has no intra-Die core schedule, address allocation, "
            "or runtime DTE/event bindings required by CommandFragment v1alpha13"
        ),
    )
    result.validate_against(plan, projection, lowered)
    return result


__all__ = ["link_swizzle_manifest"]
